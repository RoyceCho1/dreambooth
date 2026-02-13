import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor
from diffusers import StableDiffusionPipeline
from src.dreambooth.configs import DreamBoothConfig

def main():
    parser = argparse.ArgumentParser(description="DreamBooth CLIP-T Evaluation")
    parser.add_argument("--config", type=str, default="./config/config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="./output_model/output_model_dog", help="Path to the trained model directory")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing generated images to evaluate (skips generation if provided)")
    parser.add_argument("--num_images", type=int, default=4, help="Number of images to generate for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", type=str, default=None, help="The prompt used to generate the images (overrides config instance_prompt)")
    args = parser.parse_args()

    # Load Config
    try:
        config = DreamBoothConfig(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        return

    print(f"Using device: {args.device}")

    # Determine prompt
    if args.prompt:
        prompt = args.prompt
    else:
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

    # CLIP (Prompt Fidelity / Image-Text Similarity)
    print("Loading CLIP model...")
    clip_model_id = "openai/clip-vit-large-patch14"
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
    clip_model = CLIPModel.from_pretrained(clip_model_id).to(args.device)
    clip_model.eval()

    # Calculate CLIP-T Score (Text-Image Similarity)
    print("Calculating CLIP-T Score (Prompt Fidelity)...")
    
    inputs_text = clip_processor(text=[prompt], return_tensors="pt", padding=True).to(args.device)
    # Process images in batches if necessary, here we do all at once
    inputs_images = clip_processor(images=generated_images, return_tensors="pt", padding=True).to(args.device)
    
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs_text)
        image_features = clip_model.get_image_features(**inputs_images)

    # Normalize
    text_features = F.normalize(text_features, p=2, dim=-1)
    image_features = F.normalize(image_features, p=2, dim=-1)

    # Cosine Similarity
    # [Num_Gen, Hidden] * [1, Hidden].T = [Num_Gen, 1]
    clip_scores = torch.mm(image_features, text_features.t())
    clip_t_score = clip_scores.mean().item()

    # Report
    print("\n" + "="*40)
    print(f" CLIP-T Evaluation Report")
    print("="*40)
    if not args.image_dir:
        print(f"Model: {args.output_dir}")
    else:
        print(f"Image Dir: {args.image_dir}")
    print(f"Prompt: {prompt}")
    print("-" * 40)
    print(f"CLIP-T Score (Prompt Fidelity): {clip_t_score:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
