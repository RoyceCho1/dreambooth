from pathlib import Path
from typing import Optional, Dict, Any
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class DreamBoothDataset(Dataset):
    """
    DreamBooth 학습을 위한 Dataset 클래스.
    
    기능:
    1. Instance/Class 이미지 로딩 (Length Mismatch 해결)
    2. 이미지 전처리 (Resize -> Crop -> ToTensor -> Normalize)
    3. 텍스트 토크나이징 (padding='max_length'로 배치 처리가능하도록 고정 길이 반환)
    """
    def __init__(
        self,
        instance_data_root: str,
        instance_prompt: str,
        tokenizer,
        class_data_root: Optional[str] = None,
        class_prompt: Optional[str] = None,
        size: int = 512,
        center_crop: bool = False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        # 1. Instance Data
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance data root doesn't exist.")
            
        self.instance_images_path = list(self.instance_data_root.iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        # 2. Class Data
        self.class_data_root = None
        if class_data_root:
            self.class_data_root = Path(class_data_root)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            
            # 길이 불일치(Mismatch) 해결: 더 긴 쪽(보통 Class Data)에 맞춤
            self._length = max(self.num_instance_images, self.num_class_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        # 3. Image Transforms
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = {}

        # 1. Instance Data 처리
        # Cycling: 인덱스가 범위를 넘어가면 처음으로 돌아감
        instance_image_path = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        
        example["instance_images"] = self.image_transforms(instance_image)
        
        # Tokenizing: padding="max_length" & truncation=True 필수!
        # 배치 처리를 위해 모든 텍스트의 토큰 길이를 77(CLIP max)로 고정해야 함.
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="max_length",
            truncation=True,    # 토큰이 max_length를 초과하면 잘라냄
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0] # [1, 77] -> [77] (Squeeze batch dim)

        # 2. Class Data 처리 (Optional)
        if self.class_data_root:
            class_image_path = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(class_image_path)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
                
            example["class_images"] = self.image_transforms(class_image)
            
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]

        return example
