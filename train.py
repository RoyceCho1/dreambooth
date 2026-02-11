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
from src.dreambooth.configs import DreamBoothConfig
from src.dreambooth.dataset import DreamBoothDataset

def main():
    print("DreamBooth Training Phase 1: Setup & Models")
    
    # Config 로드
    config = DreamBoothConfig("./configs/config_backpack.yaml")
    
    # 설정값 변수 할당
    pretrained_model_name_or_path = config.model['pretrained_model_name_or_path']
    learning_rate = float(config.training['learning_rate'])
    use_8bit_adam = config.training['use_8bit_adam']
    mixed_precision = config.training['mixed_precision']
    
    # Device 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16 if mixed_precision == "fp16" else torch.float32
    print(f"Device: {device}, Precision: {mixed_precision}")

    # Load Models (Tokenizer, Scheduler, TextEncoder, VAE, UNet)
    print(f"Loading models from {pretrained_model_name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")

    # VAE와 Text Encoder는 학습하지 않음 (Frozen)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # UNet은 학습 (Trainable)
    unet.requires_grad_(True)
    
    # Gradient Checkpointing (VRAM 절약)
    if config.training['gradient_checkpointing']:
        unet.enable_gradient_checkpointing()

    # Frozen 모델들은 메모리 절약을 위해 fp16으로 변환하여 GPU에 올림
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    
    # 시드 고정 (Reproducibility)
    seed = config.training.get('seed') # config.yaml에 있는 값 사용
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Seed set to {seed}")
    else:
        print("No seed found in config, using random seed.")
    
    # UNet은 학습 중 32-bit 연산이 필요할 수 있으므로 기본적으로 float32로 유지 
    # (Mixed Precision 사용 시 Accelerator가 알아서 처리하지만, 여기서는 수동 관리)
    unet.to(device, dtype=torch.float32) 

    print("Model loaded and frozen components moved to device.")

    # Optimizer Setup
    # 8-bit Adam을 사용한다
    if use_8bit_adam:
        try:
            optimizer_class = bnb.optim.AdamW8bit
            print("Using 8-bit AdamW optimizer.")
        except ImportError:
            print("bitsandbytes not found. Using standard AdamW.")
            optimizer_class = torch.optim.AdamW
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer에는 학습할 UNet의 파라미터만 전달
    optimizer = optimizer_class(
        unet.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    
    # LR Scheduler 추가
    # 지원되는 스케줄러: "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    lr_scheduler_type = config.training.get('lr_scheduler', 'constant')
    print(f"Using LR Scheduler: {lr_scheduler_type}")

    lr_scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.training.get('lr_warmup_steps', 0),
        num_training_steps=config.training['max_train_steps'],
    )
    
    print("Optimizer & Scheduler setup complete.")
    print("Phase 1 complete.")
    # ------------------------------------------------------------------
    # Phase 2: Data Loading
    print("DreamBooth Training Phase 2: Data Loading")

    # 1. Dataset 생성
    train_dataset = DreamBoothDataset(
        instance_data_root=config.dataset['instance_data_dir'],
        instance_prompt=config.dataset['instance_prompt'],
        class_data_root=config.dataset['class_data_dir'],
        class_prompt=config.dataset['class_prompt'],
        tokenizer=tokenizer,
        size=config.dataset['resolution'],
        center_crop=config.dataset['center_crop'],
    )
    
    # 2. Collate Function 정의
    # 데이터셋의 개별 아이템들을 모아서 하나의 배치로 만드는 함수
    def collate_fn(examples):
        pixel_values = [example["instance_images"] for example in examples]
        input_ids = [example["instance_prompt_ids"] for example in examples]

        # Prior Preservation이 켜져 있다면, Class Image도 배치에 추가
        # 구조: [Instance_1, ..., Instance_N, Class_1, ..., Class_N]
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

    # 3. DataLoader 생성
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training['train_batch_size'],
        shuffle=True, # 학습 시 데이터 순서를 섞음
        collate_fn=collate_fn,
        num_workers=1, # 데이터 로딩 병렬 처리 개수 (시스템 사양에 따라 조절)
    )
    
    print(f"Data loader created. Batch size: {config.training['train_batch_size']}")
    print("Phase 2 complete.")

    # ------------------------------------------------------------------
    # Phase 3: Training Loop
    print("DreamBooth Training Phase 3: Training Loop")
    
    # Training Setup
    num_update_steps_per_epoch = len(train_dataloader) # 200 = (max(class_num, instance_num))
    max_train_steps = config.training['max_train_steps'] # 1000
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch) # 5
    
    print(f"Num epochs: {num_train_epochs}")
    print(f"Num train steps: {max_train_steps}")
    print(f"Steps per epoch: {num_update_steps_per_epoch}")

    # Main Loop
    global_step = 0
    progress_bar = tqdm(range(max_train_steps), desc="Training")
    
    for epoch in range(num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            input_ids = batch["input_ids"].to(device)

            # Images -> Latents (VAE Encoding)
            # VAE는 Frozen 상태이므로 gradient 계산 불필요
            # pixel_values: [Batch*2, 3, 512, 512] -> latents: [Batch*2, 4, 64, 64]
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215 # Scaling factor(Stable Diffusion)
            # float32 for UNet input(결국 UNet은 fl32로 연산함)
            latents = latents.to(dtype=torch.float32)

            # noise 샘플링
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # random timestep 샘플링
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
            ) # (0,1000) 사이의 랜덤 정수를 bsz(batch size)만큼 생성
            timesteps = timesteps.long()

            # scheduler 기반으로 latnet에 noise 추가
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Text Embeddings
            # input_ids: [Batch*2, 77] -> encoder_hidden_states: [Batch*2, 77, 768]
            encoder_hidden_states = text_encoder(input_ids)[0]
            # float32 for UNet input(결국 UNet은 fl32로 연산함)
            encoder_hidden_states = encoder_hidden_states.to(dtype=torch.float32)

            # U-Net으로 noise 예측
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Loss 계산 (Prior Preservation)
            if config.dataset['class_data_dir']:
                # 청크로 쪼개기: [Instance, Class]
                # model_pred도 [Instance, Class] 순서로 나옴
                model_pred_instance, model_pred_class = torch.chunk(model_pred, 2, dim=0)
                noise_instance, noise_class = torch.chunk(noise, 2, dim=0)
                
                # Instance Loss
                loss_instance = F.mse_loss(model_pred_instance, noise_instance, reduction="mean")
                
                # Prior Loss
                loss_prior = F.mse_loss(model_pred_class, noise_class, reduction="mean")
                
                # Total Loss = Instance + lambda * Prior
                loss = loss_instance + config.prior['prior_loss_weight'] * loss_prior
            else:
                loss = F.mse_loss(model_pred, noise, reduction="mean")

            # Gradient Accumulation
            # 배치 사이즈를 늘리는 효과를 냄
            gradient_accumulation_steps = config.training.get('gradient_accumulation_steps', 1)
            loss = loss / gradient_accumulation_steps
            
            # Backpropagation
            loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step() # 스케줄러 업데이트
                optimizer.zero_grad()
                
                # Update Progress (Step 단위로 업데이트)
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps}) # 원래 loss 값 표시
                global_step += 1

                # Checkpointing
                if global_step % config.training.get('checkpointing_steps', 500) == 0:
                    save_path = Path(f"output_model_backpack/checkpoint-{global_step}")
                    # 만약 폴더가 없으면 생성
                    save_path.mkdir(parents=True, exist_ok=True)
                    
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        pretrained_model_name_or_path,
                        unet=unet,
                        text_encoder=text_encoder,
                        vae=vae,
                        torch_dtype=torch.float16,
                    )
                    pipeline.save_pretrained(save_path)
                    print(f" Saved checkpoint to {save_path}")

            if global_step >= max_train_steps:
                break
        
        if global_step >= max_train_steps:
            break

    print("Training finished.")

    # 학습된 UNet과 나머지 Frozen 모델들을 합쳐서 파이프라인으로 저장
    print("Saving model pipeline...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        torch_dtype=torch.float16,
    )
    
    save_path = Path("output_model_backpack")
    pipeline.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
