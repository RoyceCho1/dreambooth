import argparse
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor, ViTModel, ViTImageProcessor
from diffusers import StableDiffusionPipeline
from src.dreambooth.configs import DreamBoothConfig

def main():
    parser = argparse.ArgumentParser(description="DreamBooth Evaluation Script")
    parser.add_argument("--config", type=str, default="./configs/config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="./output_model/output_model_dog", help="Path to the trained model directory")
    parser.add_argument("--num_images", type=int, default=4, help="Number of images to generate for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    try:
        config = DreamBoothConfig(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        return

    print(f"Using device: {args.device}")

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

    prompt = config.dataset['instance_prompt']
    print(f"Generating {args.num_images} images with prompt: '{prompt}'")
    
    generated_images = []
    for _ in tqdm(range(args.num_images), desc="Generating Images"):
        with torch.autocast(args.device):
            image = pipeline(prompt, num_inference_steps=50).images[0]
            generated_images.append(image)
    
    # DINO (Subject Fidelity)
    dino_model_id = "facebook/dino-vits16"
    dino_processor = ViTImageProcessor.from_pretrained(dino_model_id)
    dino_model = ViTModel.from_pretrained(dino_model_id).to(args.device)
    dino_model.eval()

    # CLIP (Prompt Fidelity / Image-Text Similarity)
    clip_model_id = "openai/clip-vit-large-patch14"
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
    clip_model = CLIPModel.from_pretrained(clip_model_id).to(args.device)
    clip_model.eval()
    
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
            # For each generated image, find average similarity to all real images
            # or average max similarity? DreamBooth paper usually implies average pairwise or average max.
            # Let's do Average Pairwise Cosine Similarity (Generated <-> Real)
            
            # shape: [Num_Gen, Hidden] @ [Num_Real, Hidden].T = [Num_Gen, Num_Real]
            similarity_matrix = torch.mm(gen_features, real_features.t())
            dino_score = similarity_matrix.mean().item()

    # 5. Calculate CLIP-T Score (Text-Image Similarity)
    print("Calculating CLIP-T Score (Prompt Fidelity)...")
    
    inputs_text = clip_processor(text=[prompt], return_tensors="pt", padding=True).to(args.device)
    inputs_images = clip_processor(images=generated_images, return_tensors="pt", padding=True).to(args.device)
    
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs_text)
        image_features = clip_model.get_image_features(**inputs_images)

    # Normalize
    text_features = F.normalize(text_features, p=2, dim=-1)
    image_features = F.normalize(image_features, p=2, dim=-1)

    # Cosine Similarity
    # [1, Hidden] * [Num_Gen, Hidden]
    clip_scores = torch.mm(image_features, text_features.t())
    clip_t_score = clip_scores.mean().item()

    # 6. Report
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
    print(f"CLIP-T Score (Prompt Fidelity): {clip_t_score:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
