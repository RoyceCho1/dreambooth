import argparse
import os
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def main():
    parser = argparse.ArgumentParser(description="DreamBooth Inference")
    parser.add_argument("--model_path", type=str, default="./output_model/output_model_dogbackpack_textencoder", help="Path to the trained model")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="low quality, worst quality, bad anatomy, deformed, blurry", help="Negative prompt")
    parser.add_argument("--output_dir", type=str, default="./generated_images/dogbackpack_textencoder/coffee", help="Output directory")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale (CFG)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to generate")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print(f"Loading model from {args.model_path}")
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    # 학습용 스케줄러(DDPM) 대신, 추론 속도와 품질이 좋은 DPM-Solver++로 교체
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.to(device)
    
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generation Loop
    print(f"Generating {args.num_images} images with prompt: '{args.prompt}'")
    
    for i in range(args.num_images):
        # seed 설정
        seed = args.seed if args.seed is not None else torch.randint(0, 1000000, (1,)).item()
        generator = torch.Generator(device).manual_seed(seed)
        
        image = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]
        
        # Image 저장
        filename = f"{seed}_{i}.png"
        save_path = save_dir / filename
        image.save(save_path)
        print(f"Saved: {save_path} (Seed: {seed})")
        
    print("Inference finished.")

if __name__ == "__main__":
    main()
