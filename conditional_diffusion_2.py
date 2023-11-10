import os
import copy
import torch
import random
from torch import nn
from tqdm import tqdm
from PIL import Image
from p_tqdm import p_map
import lightning.pytorch as pl
import torch.nn.functional as F
from dataclasses import dataclass
import torchvision.transforms as T
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import OneCycleLR
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, DDIMScheduler
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint


transform = T.ToPILImage()
wandb_logger = WandbLogger(project="Diffusion-cat-dog-bird", log_model=True, name='128-vae-DPMSolverMulti-continued')


@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 256
    eval_batch_size = train_batch_size  # how many images to sample during evaluation
    num_epochs = 300
    gradient_accumulation_steps = 4
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    num_workers = 22
    save_model_epochs = 30
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'cat-dogs-birds'  # the model namy locally and on the HF Hub
    #accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=gradient_accumulation_steps)

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


class DiffusionModel(pl.LightningModule):

    def __init__(self, config, train_loader_len):
        super().__init__()
        self.config = config

        self.train_loader_len = train_loader_len

        self.last_image = ""
        self.label_projection = nn.Embedding(3, 64)

        # Initialize UNet model
        #self.unet = UNet_conditional(c_in=4, c_out=4, time_dim=256, num_classes=3, remove_deep_conv=False)
        # self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", 
        #                                                   #revision="fp16", torch_dtype=torch.float16,
        #                                                   subfolder="unet")

        self.unet = UNet2DConditionModel(
                    sample_size=self.config.image_size//8,  # the target image resolution
                    in_channels=4,  # the number of input channels, 3 for RGB images
                    out_channels=4,  # the number of output channels
                    layers_per_block=2,  # how many ResNet layers to use per UNet block
                    block_out_channels=(128, 256, 256, 512),  # the number of output channels for each UNet block , 256, 256, 512, 512
                    time_embedding_dim=256,
                    encoder_hid_dim=64,
                    transformer_layers_per_block=2,
                    down_block_types=(
                        "CrossAttnDownBlock2D",  # a regular ResNet downsampling block
                        "CrossAttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                        "CrossAttnDownBlock2D",
                        "DownBlock2D",
                    ),
                    up_block_types=(
                        "UpBlock2D",  # a regular ResNet upsampling block
                        "CrossAttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                        "CrossAttnUpBlock2D",
                        "CrossAttnUpBlock2D"
                    ),
                    mid_block_type=(
                        "UNetMidBlock2DCrossAttn"
                    )
                )

        # Initialize VAE
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4",
                                                  revision="fp16", torch_dtype=torch.float16,
                                                  subfolder="vae").requires_grad_(False)

        # Initialize noise scheduler
        #self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
        self.noise_scheduler.set_timesteps(1000)

        # Initialize EMA
        self.ema_decay = 0.995  # You should define this in your config
        self.ema_unet = copy.deepcopy(self.unet)
        for param in self.ema_unet.parameters():
            param.requires_grad_(False)

    def update_ema(self):
        with torch.no_grad():
            model_params = dict(self.unet.named_parameters())
            ema_params = dict(self.ema_unet.named_parameters())
            for name in model_params:
                model_param = model_params[name]
                ema_param = ema_params[name]
                ema_param.copy_(ema_param * self.ema_decay + (1.0 - self.ema_decay) * model_param)

    def forward(self, x, timesteps, hidden_embed):
        # Forward pass through UNet
        #noise_pred = self.unet(x, timesteps, hidden_embed)
        noise_pred = self.unet(x, timesteps, encoder_hidden_states=hidden_embed.unsqueeze(1))["sample"]
        return noise_pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.config.learning_rate)
        scheduler = OneCycleLR(
                        optimizer,
                        max_lr=self.config.learning_rate,
                        pct_start=2/self.trainer.max_epochs,
                        epochs=self.trainer.max_epochs,
                        steps_per_epoch=self.train_loader_len,
                        anneal_strategy='cos',
                        div_factor=100,
                        final_div_factor=100,
                        #three_phase=True
                )
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
        #return(optimizer)

    def process_batch(self, batch):

        clean_images, targets = batch

        # Encode the clean images to obtain latent representations
        #with torch.no_grad():
        clean_latent = self.vae.encode(clean_images.to(torch.float16)).latent_dist.sample() * 0.18215
            #clean_latent = ((clean_latent * 2) - 1) 

        # Sample noise to add to the images
        noise = torch.randn(clean_latent.shape, device=self.device)

        # Sample a random timestep for each image
        bs = clean_images.size(0)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bs,), device=self.device).long()

        # Add noise to the clean latent representations (forward diffusion)
        noisy_images = self.noise_scheduler.add_noise(clean_latent, noise, timesteps)

        # Label Projection
        target_embed = self.label_projection(targets)

        # Predict the noise residual
        noise_pred = self(noisy_images, timesteps, target_embed)  # Calls the forward method

        # Compute the loss
        loss = F.mse_loss(noise_pred, noise)

        self.last_image = (clean_images[-1], targets[-1])

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if self.current_epoch > 20:
            self.update_ema()

        return loss

    # def validation_step(self, batch, batch_idx):
    #     val_loss = self.process_batch(batch)
    #     self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
    #     return val_loss
    
    def latents_to_pil(self, latents):
        # batch of latents -> list of images
        latents = (1 / 0.18215) * latents
        #with torch.no_grad():
        image = self.vae.decode(latents.to(torch.float16)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images][0]
        return pil_images

    def generate_and_save_images(self, current_epoch):
        generator = torch.manual_seed(42)
        noise = torch.randn(1, 4, self.config.image_size//8, self.config.image_size//8, generator=generator).cuda()
        label = self.label_projection(torch.tensor(random.randint(0, 2)).cuda())
        noise = noise * self.noise_scheduler.init_noise_sigma

        original_unet = self.unet
        self.unet = self.ema_unet

        for i, t in tqdm(enumerate(self.noise_scheduler.timesteps), total=len(self.noise_scheduler.timesteps)):
            noise = self.noise_scheduler.scale_model_input(noise, t)

            with torch.no_grad():
                #noise_pred = self.unet(noise, torch.tensor(t).to(noise.device), torch.tensor([random.randint(0, 2)]).to(noise.device))
                noise_pred = self.unet(noise, t, encoder_hidden_states=label.unsqueeze(0).unsqueeze(0))["sample"]

            noise = self.noise_scheduler.step(noise_pred, t.long(), noise).prev_sample

        decoded_noise = self.latents_to_pil(noise)
        self.unet = original_unet
        
        # Save the generated images to a unique folder for the current epoch
        save_dir = f"generated_images/epoch_{current_epoch}"
        os.makedirs(save_dir, exist_ok=True)
        
        decoded_noise.save(os.path.join(save_dir, f"image.png"))
        wandb_logger.log_image(key=f"generated_epoch_{current_epoch}", images=[decoded_noise])
        #self.logger.log_image(f"generated_epoch_{current_epoch}", [decoded_noise,]) 

    def apply_transform(self, image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return(pil_images)
    
    def generate_visualize(self, current_epoch):
        clean_images, targets = self.last_image
        clean_images = clean_images.unsqueeze(0)

        with torch.no_grad():
            clean_latent = self.vae.encode(clean_images.to(torch.float16)).latent_dist.sample()
            image = self.vae.decode(clean_latent.to(torch.float16)).sample
            
        save_dir = f"coded_images/epoch_{current_epoch}"
        save_dir_orig = f"original_images/epoch_{current_epoch}"

        clean_images = self.apply_transform(clean_images)[0]
        image = self.apply_transform(image)[0]

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir_orig, exist_ok=True)

        image.save(os.path.join(save_dir, f"image.png"))
        clean_images.save(os.path.join(save_dir_orig, f"image.png"))
            
    def on_train_epoch_end(self):
        self.generate_and_save_images(self.current_epoch)
        #self.generate_visualize(self.current_epoch)


config = TrainingConfig()

train_transform = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

test_transform = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )


def main():

    trainer = pl.Trainer(
        precision='16-mixed' if config.mixed_precision=='fp16' else 32,  # Set precision
        accelerator='auto',
        devices='auto',
        strategy='auto',
        max_epochs=config.num_epochs,
        logger=[TensorBoardLogger("logs/", name="stable-diffusion"), wandb_logger],
        callbacks=[LearningRateMonitor(logging_interval="step"), ModelCheckpoint(monitor="train_loss_epoch", mode="min")],
        accumulate_grad_batches=config.gradient_accumulation_steps
        #limit_train_batches=0.3, 
    )

    torch.set_float32_matmul_precision('high')
    train_dataset = datasets.ImageFolder(root='high-resolution-catdogbird-image-dataset-13000', transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers)
    diffusion_model = DiffusionModel.load_from_checkpoint("/root/epoch=93-step=1316.ckpt", config=config, train_loader_len=len(train_loader) )

    trainer.fit(diffusion_model, train_loader)

if __name__ == "__main__":
    main()