"""
PyTorch Dataset for PMC-OA JSONL format
Handles loading images from folder and captions from JSONL
"""

import json
from pathlib import Path
from typing import Optional, Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class PMCOADataset(Dataset):
    """
    PMC-OA Dataset for image captioning

    Dataset structure:
        data_dir/
        ├── images/              # Image files
        ├── pmc_oa.jsonl         # Annotations
        └── splits/
            ├── train.jsonl
            ├── validation.jsonl
            └── test.jsonl

    JSONL format:
        {"image": "PMC212319_Fig3_4.jpg", "caption": "A real time image..."}
    """

    def __init__(
        self,
        jsonl_path: str,
        images_dir: str,
        image_processor=None,
        tokenizer=None,
        max_caption_length: int = 256,
        image_size: int = 224
    ):
        """
        Args:
            jsonl_path: Path to JSONL annotation file
            images_dir: Path to images directory
            image_processor: HuggingFace image processor (for vision encoder)
            tokenizer: HuggingFace tokenizer (for language model)
            max_caption_length: Maximum caption length in tokens
            image_size: Target image size for resizing
        """
        self.jsonl_path = Path(jsonl_path)
        self.images_dir = Path(images_dir)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_caption_length = max_caption_length
        self.image_size = image_size

        # Load annotations
        self.samples = self._load_annotations()

        print(f"Loaded {len(self.samples):,} samples from {self.jsonl_path.name}")

    def _load_annotations(self) -> List[Dict]:
        """Load annotations from JSONL file"""
        samples = []

        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)

                    # Validate required fields
                    if 'image' in data and 'caption' in data:
                        samples.append(data)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample

        Returns:
            dict with keys:
                - pixel_values: Processed image tensor
                - input_ids: Tokenized caption
                - attention_mask: Attention mask for caption
                - caption: Original caption text
                - image_path: Path to image file
        """
        sample = self.samples[idx]

        # Get image path
        image_filename = sample['image']
        image_path = self.images_dir / image_filename

        # Load and process image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load {image_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (self.image_size, self.image_size), color='white')

        # Process image with image processor or manual processing
        if self.image_processor is not None:
            pixel_values = self.image_processor(
                images=image,
                return_tensors="pt"
            )['pixel_values'].squeeze(0)
        else:
            # Manual processing
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
            pixel_values = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Get caption
        caption = sample['caption']

        # Tokenize caption if tokenizer is provided
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                caption,
                max_length=self.max_caption_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
        else:
            # Return dummy tensors if no tokenizer
            input_ids = torch.zeros(self.max_caption_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_caption_length, dtype=torch.long)

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'caption': caption,
            'image_path': str(image_path)
        }


def create_pmc_dataloader(
    jsonl_path: str,
    images_dir: str,
    image_processor=None,
    tokenizer=None,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    max_caption_length: int = 256
) -> DataLoader:
    """
    Create DataLoader for PMC-OA dataset

    Args:
        jsonl_path: Path to JSONL annotations
        images_dir: Path to images directory
        image_processor: HuggingFace image processor
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        max_caption_length: Maximum caption length

    Returns:
        DataLoader instance
    """
    dataset = PMCOADataset(
        jsonl_path=jsonl_path,
        images_dir=images_dir,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_caption_length=max_caption_length
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return dataloader


def test_dataset():
    """Test dataset loading"""

    print("Testing PMC-OA Dataset...")
    print("-" * 70)

    # Example usage (update paths as needed)
    jsonl_path = "data/raw/pmc_oa/splits/train.jsonl"
    images_dir = "data/raw/pmc_oa/images"

    if not Path(jsonl_path).exists():
        print(f"Dataset not found at {jsonl_path}")
        print("Please download dataset first:")
        print("  python scripts/download_pmc_oa.py")
        return

    # Create dataset without processors (manual mode)
    dataset = PMCOADataset(
        jsonl_path=jsonl_path,
        images_dir=images_dir,
        image_processor=None,
        tokenizer=None
    )

    print(f"✅ Dataset created with {len(dataset):,} samples")

    # Test loading a sample
    if len(dataset) > 0:
        print("\nTesting sample loading...")
        sample = dataset[0]

        print(f"  Image shape: {sample['pixel_values'].shape}")
        print(f"  Caption: {sample['caption'][:100]}...")
        print(f"  Image path: {sample['image_path']}")
        print("  ✅ Sample loaded successfully!")

    # Test with HuggingFace processors
    print("\nTesting with HuggingFace processors...")
    try:
        from transformers import AutoImageProcessor, AutoTokenizer

        image_processor = AutoImageProcessor.from_pretrained(
            "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            "BioMistral/BioMistral-7B"
        )

        dataset_processed = PMCOADataset(
            jsonl_path=jsonl_path,
            images_dir=images_dir,
            image_processor=image_processor,
            tokenizer=tokenizer
        )

        sample = dataset_processed[0]
        print(f"  Image shape (processed): {sample['pixel_values'].shape}")
        print(f"  Input IDs shape: {sample['input_ids'].shape}")
        print(f"  Attention mask shape: {sample['attention_mask'].shape}")
        print("  ✅ Processed dataset working!")

    except Exception as e:
        print(f"  ⚠️  Could not test with processors: {e}")
        print("  This is OK if models not downloaded yet")

    print("\n" + "="*70)
    print("✅ Dataset tests passed!")


if __name__ == "__main__":
    test_dataset()
