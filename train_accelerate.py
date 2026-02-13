import argparse
import itertools
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, AutoTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)
from diffusers.optimization import get_scheduler
import bitsandbytes as bnb
from accelerate import Accelerator
from src.dreambooth.configs import DreamBoothConfig
from src.dreambooth.dataset import DreamBoothDataset

def main():
    parser = argparse.ArgumentParser(description="DreamBooth Training")
    parser.add_argument("--config", type=str, default="./configs/config_dogbackpack.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="output_model/output_model_accelerate", help="Directory to save the model and checkpoints")
    args = parser.parse_args()

    config = DreamBoothConfig(args.config)
    
    pretrained_model_name_or_path = config.model['pretrained_model_name_or_path']
    learning_rate = float(config.training['learning_rate'])
    use_8bit_adam = config.training['use_8bit_adam']
    mixed_precision = config.training['mixed_precision']
    train_text_encoder = config.training.get('train_text_encoder', False)
    gradient_accumulation_steps = config.training.get('gradient_accumulation_steps', 1)

    # Accelerator
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=args.output_dir
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))
        accelerator.print(f"Setup & Models (Config: {args.config}, Output: {args.output_dir})")
        accelerator.print(f"Accelerator Initialized. Device: {accelerator.device}, Mixed Precision: {mixed_precision}")

    # Load Models (Tokenizer, Scheduler, TextEncoder, VAE, UNet)
    accelerator.print(f"Loading models from {pretrained_model_name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")

    # Frozen VAE
    vae.requires_grad_(False)
    
    # Text Encoder
    if not train_text_encoder:
        text_encoder.requires_grad_(False)
    else:
        text_encoder.requires_grad_(True)
        text_encoder.train()
    
    unet.requires_grad_(True)
    
    # Gradient Checkpointing
    if config.training['gradient_checkpointing']:
        unet.enable_gradient_checkpointing()
        if train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    weight_dtype = torch.float16
    
    vae.to(accelerator.device, dtype=weight_dtype)
    
    if not train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    accelerator.print(f"Frozen models moved to {accelerator.device} and cast to {weight_dtype}")

    # Optimizer
    if use_8bit_adam:
        try:
            optimizer_class = bnb.optim.AdamW8bit
            accelerator.print("Using 8-bit AdamW optimizer.")
        except ImportError:
            accelerator.print("bitsandbytes not found. Using standard AdamW.")
            optimizer_class = torch.optim.AdamW
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = unet.parameters()
    if train_text_encoder:
        params_to_optimize = itertools.chain(unet.parameters(), text_encoder.parameters())

    optimizer = optimizer_class(
        params_to_optimize,
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    
    # LR Scheduler
    lr_scheduler_type = config.training.get('lr_scheduler', 'constant')
    accelerator.print(f"Using LR Scheduler: {lr_scheduler_type}")

    max_train_steps = config.training['max_train_steps']

    lr_scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.training.get('lr_warmup_steps', 0) * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )
    
    accelerator.print("Optimizer & Scheduler setup complete.")

    # Data Loading
    accelerator.print("Data Loading")

    train_dataset = DreamBoothDataset(
        instance_data_root=config.dataset['instance_data_dir'],
        instance_prompt=config.dataset['instance_prompt'],
        class_data_root=config.dataset['class_data_dir'],
        class_prompt=config.dataset['class_prompt'],
        tokenizer=tokenizer,
        size=config.dataset['resolution'],
        center_crop=config.dataset['center_crop'],
    )
    
    def collate_fn(examples):
        pixel_values = [example["instance_images"] for example in examples]
        input_ids = [example["instance_prompt_ids"] for example in examples]

        if config.dataset['class_data_dir']:
            pixel_values += [example["class_images"] for example in examples]
            input_ids += [example["class_prompt_ids"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack(input_ids)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
        }

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training['train_batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1,
    )
    
    # Prepare with Accelerator
    if train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    accelerator.print(f"Data loader created. Batch size: {config.training['train_batch_size']}")
    
    # Calculate Epochs
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if num_update_steps_per_epoch == 0:
        num_update_steps_per_epoch = 1
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    
    accelerator.print(f"Num epochs: {num_train_epochs}")
    accelerator.print(f"Num train steps: {max_train_steps}")
    accelerator.print("Data Loading complete.")

    # Training Loop
    accelerator.print("Training Loop")
    
    global_step = 0
    progress_bar = tqdm(range(max_train_steps), desc="Training", disable=not accelerator.is_local_main_process)
    
    for epoch in range(num_train_epochs):
        unet.train()
        if train_text_encoder:
            text_encoder.train()
            
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet): # Handles gradient accumulation
                # Get batch (Accelerate handles device placement automatically for Dataloader)
                pixel_values = batch["pixel_values"]
                input_ids = batch["input_ids"]

                # Cast pixel_values to float16 for VAE encode
                latents = vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device
                )
                timesteps = timesteps.long()
                
                # Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Text Embeddings
                encoder_hidden_states = text_encoder(input_ids)[0]
                
                if not train_text_encoder:
                    encoder_hidden_states = encoder_hidden_states.to(dtype=weight_dtype)

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample.to(dtype=weight_dtype)

                # Loss Backward 시 float32가 안전함.
                if model_pred.dtype != torch.float32:
                    model_pred = model_pred.float()

                
                # Loss
                if config.dataset['class_data_dir']:
                    noise = noise.float()
                    
                    model_pred_instance, model_pred_class = torch.chunk(model_pred, 2, dim=0)
                    noise_instance, noise_class = torch.chunk(noise, 2, dim=0)
                    loss_instance = F.mse_loss(model_pred_instance, noise_instance, reduction="mean")
                    loss_prior = F.mse_loss(model_pred_class, noise_class, reduction="mean")
                    loss = loss_instance + config.prior['prior_loss_weight'] * loss_prior
                else:
                    # Target noise도 float32로 변환
                    loss = F.mse_loss(model_pred, noise.float(), reduction="mean")
                
                # Backward
                accelerator.backward(loss)
                
                # Clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Post-step (only when gradients synced)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Checkpointing
                if global_step % config.training.get('checkpointing_steps', 500) == 0:
                    if accelerator.is_main_process:
                        save_path = Path(args.output_dir) / f"checkpoint-{global_step}"
                        save_path.mkdir(parents=True, exist_ok=True)
                        
                        # Unwrap and save
                        unwrapped_unet = accelerator.unwrap_model(unet)
                        unwrapped_text_encoder = accelerator.unwrap_model(text_encoder) 
                        # Note: If text_encoder was frozen, 'unwrap_model' just returns it (which is good).
                        
                        pipeline = StableDiffusionPipeline.from_pretrained(
                            pretrained_model_name_or_path,
                            unet=unwrapped_unet,
                            text_encoder=unwrapped_text_encoder,
                            vae=vae,
                            torch_dtype=torch.float16,
                        )
                        pipeline.save_pretrained(save_path)
                        accelerator.print(f" Saved checkpoint to {save_path}")

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            if global_step >= max_train_steps:
                break
        
        if global_step >= max_train_steps:
            break

    accelerator.print("Training finished.")
    accelerator.wait_for_everyone()
    
    # Save Final Model
    if accelerator.is_main_process:
        accelerator.print("Saving model pipeline...")
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)

        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            unet=unwrapped_unet,
            text_encoder=unwrapped_text_encoder,
            vae=vae,
            torch_dtype=torch.float16,
        )
        
        save_path = Path(args.output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        pipeline.save_pretrained(save_path)
        accelerator.print(f"Model saved to {save_path}\n")
    
    accelerator.end_training()

if __name__ == "__main__":
    main()
