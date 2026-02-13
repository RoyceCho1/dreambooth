import argparse
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline
from tqdm.auto import tqdm
from src.dreambooth.configs import DreamBoothConfig

def generate_class_images(
    model_id: str,
    class_prompt: str,
    class_data_dir: str,
    num_class_images: int,
    batch_size: int = 1,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    # Prior Preservation Loss를 위한 class image 생성
    
    class_data_dir = Path(class_data_dir)
    if not class_data_dir.exists():
        class_data_dir.mkdir(parents=True, exist_ok=True)
    
    cur_class_images = len(list(class_data_dir.iterdir()))
    
    if cur_class_images >= num_class_images:
        print(f"Found {cur_class_images} class images. No need to generate more.")
        return

    print(f"Generating {num_class_images - cur_class_images} class images for prior preservation.")

    # Pre-trained 모델을 이용해서 class image 생성
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None, # 속도 향상을 위해서
        requires_safety_checker=False
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    num_new_images = num_class_images - cur_class_images
    
    # batch 단위로 class image 생성
    for i in tqdm(range(0, num_new_images, batch_size), desc="Generating class images"):
        
        # batch size 결정
        this_batch_size = min(batch_size, num_new_images - i)

        # class image 생성
        images = pipe(
            prompt=[class_prompt] * this_batch_size,
            num_inference_steps=50,
            guidance_scale=7.5 # CFG scale(프롬프트를 얼마나 엄격하게 따를지)
        ).images

        for idx, image in enumerate(images):
            image_filename = class_data_dir / f"class_{cur_class_images + i + idx}.jpg"
            image.save(image_filename)

    del pipe
    torch.cuda.empty_cache()
    print("Class image generation complete.")

def main():
    parser = argparse.ArgumentParser(description="Generate Class Images")
    parser.add_argument("--config", type=str, default="./config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = DreamBoothConfig(args.config)
    
    model_id = config.model['pretrained_model_name_or_path']
    class_prompt = config.dataset['class_prompt']
    class_data_dir = config.dataset['class_data_dir']
    
    # Prior Preservation 설정 확인
    if not config.prior['enabled']:
        print("Prior preservation not enabled in config. Skipping generation.")
        return

    num_class_images = config.prior['num_class_images']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Starting class image generation...")
    print(f"- Model: {model_id}")
    print(f"- Prompt: '{class_prompt}'")
    print(f"- Target Count: {num_class_images}")
    print(f"- Output Dir: {class_data_dir}")
    
    generate_class_images(
        model_id=model_id,
        class_prompt=class_prompt,
        class_data_dir=class_data_dir,
        num_class_images=num_class_images,
        batch_size=config.training['train_batch_size'],
        device=device,
        dtype=dtype
    )

if __name__ == "__main__":
    main()
