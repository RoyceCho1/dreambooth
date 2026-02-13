import argparse
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from transformers import ViTModel, ViTImageProcessor
from diffusers import StableDiffusionPipeline
from src.dreambooth.configs import DreamBoothConfig

def main():
    parser = argparse.ArgumentParser(description="DreamBooth Evaluation Script")
    parser.add_argument("--config", type=str, default="./config/config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="./output_model/output_model_dog", help="Path to the trained model directory")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing generated images to evaluate (skips generation if provided)")
    parser.add_argument("--num_images", type=int, default=4, help="Number of images to generate for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    try:
        config = DreamBoothConfig(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        return

    print(f"Using device: {args.device}")

    prompt = config.dataset['instance_prompt']
    generated_images = []

    if args.image_dir:
        # Load existing images from directory
        print(f"Loading generated images from {args.image_dir}...")
        image_dir_path = Path(args.image_dir)
        if not image_dir_path.exists():
            print(f"Error: Image directory {args.image_dir} not found.")
            return
            
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            image_files.extend(list(image_dir_path.glob(ext)))
            
        if not image_files:
            print(f"Error: No images found in {args.image_dir}.")
            return
            
        print(f"Found {len(image_files)} images.")
        generated_images = [Image.open(p).convert("RGB") for p in image_files]                
    else:
        # Generate images using the model
        print(f"Loading trained pipeline from {args.output_dir}")
        try:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.output_dir,
                torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
                safety_checker=None
            ).to(args.device)
            pipeline.set_progress_bar_config(disable=True)
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            return

        print(f"Generating {args.num_images} images with prompt: '{prompt}'")
        
        for _ in tqdm(range(args.num_images), desc="Generating Images"):
            with torch.autocast(args.device):
                image = pipeline(prompt, num_inference_steps=50).images[0]
                generated_images.append(image)
    
    # DINO (Subject Fidelity)
    dino_model_id = "facebook/dino-vits16"
    dino_processor = ViTImageProcessor.from_pretrained(dino_model_id)
    dino_model = ViTModel.from_pretrained(dino_model_id).to(args.device)
    dino_model.eval()
    
    instance_data_dir = Path(config.dataset['instance_data_dir'])
    if not instance_data_dir.exists():
        print(f"Warning: Instance data directory {instance_data_dir} not found. Skipping DINO Score.")
        dino_score = None
    else:
        # Load Real Images
        real_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            real_images.extend(list(instance_data_dir.glob(ext)))
        
        if not real_images:
            print("Warning: No images found in instance data directory. Skipping DINO Score.")
            dino_score = None
        else:
            real_pil_images = [Image.open(p).convert("RGB") for p in real_images]
            
            def get_dino_features(images):
                inputs = dino_processor(images=images, return_tensors="pt").to(args.device)
                with torch.no_grad():
                    outputs = dino_model(**inputs)
                return outputs.last_hidden_state[:, 0, :]

            real_features = get_dino_features(real_pil_images)
            gen_features = get_dino_features(generated_images)

            # Normalize features
            real_features = F.normalize(real_features, p=2, dim=-1)
            gen_features = F.normalize(gen_features, p=2, dim=-1)

            # Calculate Average Cosine Similarity
            # For each generated image, find average similarity to all real images (average pairwise or average max).
            # Average Pairwise Cosine Similarity (Generated <-> Real)
            # shape: [Num_Gen, Hidden] @ [Num_Real, Hidden].T = [Num_Gen, Num_Real]
            similarity_matrix = torch.mm(gen_features, real_features.t())
            dino_score = similarity_matrix.mean().item()



    # Report
    print("\n" + "="*40)
    print(f" Evaluation Report")
    print("="*40)
    print(f"Model: {args.output_dir}")
    print(f"Prompt: {prompt}")
    print("-" * 40)
    if dino_score is not None:
        print(f"DINO Score (Subject Fidelity):  {dino_score:.4f}")
    else:
        print(f"DINO Score (Subject Fidelity):  N/A")
    print("="*40)

if __name__ == "__main__":
    main()
