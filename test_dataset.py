from pathlib import Path
from transformers import CLIPTokenizer
from configs import DreamBoothConfig
from dataset import DreamBoothDataset
import torch

def test_dataset():
    """
    dataset.py 구현 검증 스크립트.
    1. 데이터 로드 확인
    2. Length Mismatch (Cycling) 동작 확인
    3. 전처리 (Tensor Shape, Tokenizer output) 확인
    """
    
    # 1. Config & Tokenizer 준비
    config = DreamBoothConfig("./config.yaml")
    model_id = config.model['pretrained_model_name_or_path']
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    # 2. Dataset 생성
    print(f"Creating dataset from {config.dataset['instance_data_dir']}...")
    try:
        dataset = DreamBoothDataset(
            instance_data_root=config.dataset['instance_data_dir'],
            instance_prompt=config.dataset['instance_prompt'],
            class_data_root=config.dataset['class_data_dir'],
            class_prompt=config.dataset['class_prompt'],
            tokenizer=tokenizer,
            size=config.dataset['resolution'],
            center_crop=config.dataset['center_crop'],
        )
    except FileNotFoundError:
        print("Error: Instance data directory not found. Please create 'data/instance_images' and add a dummy image.")
        return

    # 3. 검증: 전체 길이
    print(f"\n[Verification 1] Dataset Length")
    print(f"- Total Length: {len(dataset)}")
    print(f"- Num Instance Images: {dataset.num_instance_images}")
    if config.dataset['class_data_dir']:
        print(f"- Num Class Images: {dataset.num_class_images}")
    
    # 길이가 더 긴 쪽(Class)에 맞춰졌는지 확인
    expected_len = max(dataset.num_instance_images, dataset.num_class_images)
    if len(dataset) == expected_len:
        print("✅ Length Check Passed: Dataset length matches max(instance, class).")
    else:
        print(f"❌ Length Check Failed: Expected {expected_len}, got {len(dataset)}.")

    # 4. 검증: 데이터 형태 (Shape)
    print(f"\n[Verification 2] Data Shape & Type")
    sample = dataset[0]
    
    # Image Tensor Check
    img_shape = sample["instance_images"].shape
    print(f"- Instance Image Shape: {img_shape}")
    if img_shape == (3, 512, 512):
        print("✅ Image Shape Check Passed: (3, 512, 512).")
    else:
        print(f"❌ Image Shape Check Failed: Expected (3, 512, 512), got {img_shape}.")
        
    # Tensor Value Range Check (Normalization)
    img_min, img_max = sample["instance_images"].min(), sample["instance_images"].max()
    print(f"- Image Value Range: [{img_min:.2f}, {img_max:.2f}]")
    if -1.0 <= img_min and img_max <= 1.0:
        print("✅ Normalization Check Passed: Values are within [-1, 1].")
    else:
        print("❌ Normalization Check Failed: Values are out of range.")

    # Tokenizer Check
    token_shape = sample["instance_prompt_ids"].shape
    print(f"- Token Shape: {token_shape}")
    if token_shape == (77,):
        print("✅ Tokenizer Check Passed: Fixed length 77.")
    else:
        print(f"❌ Tokenizer Check Failed: Expected (77,), got {token_shape}.")

    # 5. 검증: Cycling Logic (인스턴스 이미지 반복 확인)
    print(f"\n[Verification 3] Cycling Logic")
    # 인스턴스 이미지가 5장이면, index 0과 index 5는 같은 이미지여야 함.
    idx1 = 0
    idx2 = dataset.num_instance_images  # Cycling point
    
    if idx2 < len(dataset):
        img1 = dataset[idx1]["instance_images"]
        img2 = dataset[idx2]["instance_images"]
        
        # 텐서 값이 같은지 확인(이때 config에서 center_crop이 True로 설정해야된다)
        if torch.equal(img1, img2):
            print(f"✅ Cycling Check Passed: Index {idx1} and {idx2} return the same instance image.")
        else:
            print(f"❌ Cycling Check Failed: Logic might be wrong.")
    else:
        print("⚠️ Skipping Cycling Check: Not enough data length to cycle.")

if __name__ == "__main__":
    test_dataset()
