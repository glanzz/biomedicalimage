"""
PyTorch Dataset classes for PMC-OA biomedical image captioning

Supports loading from:
- Raw HuggingFace datasets (with on-the-fly preprocessing)
- Preprocessed NPZ files (faster, recommended for training)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from PIL import Image
import numpy as np
from typing import Optional, Dict, Callable, Union
import os
import logging

logger = logging.getLogger(__name__)


class PMCImageCaptionDataset(Dataset):
    """
    Dataset for PMC-OA image-caption pairs.

    Supports two modes:
    1. Raw mode: Load from HuggingFace datasets with on-the-fly preprocessing
    2. Preprocessed mode: Load from preprocessed NPZ files (faster)

    Args:
        dataset_path: Path to dataset (HF dataset or NPZ file directory)
        split: Dataset split ('train', 'valid', 'test')
        tokenizer: HuggingFace tokenizer for caption encoding
        image_processor: Image processor (for raw mode only)
        max_length: Maximum caption length in tokens
        use_preprocessed: Whether to load from NPZ files
        subset_size: Use only first N samples (for debugging)
    """

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        tokenizer = None,
        image_processor = None,
        max_length: int = 256,
        use_preprocessed: bool = False,
        subset_size: Optional[int] = None
    ):
        self.dataset_path = dataset_path
        self.split = split
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.use_preprocessed = use_preprocessed

        logger.info(f"Loading dataset: {dataset_path} ({split})")
        logger.info(f"Mode: {'Preprocessed' if use_preprocessed else 'Raw'}")

        if use_preprocessed:
            self._load_preprocessed()
        else:
            self._load_raw()

        # Apply subset if specified
        if subset_size is not None:
            self._apply_subset(subset_size)

        logger.info(f"Dataset loaded: {len(self)} samples")

    def _load_raw(self):
        """Load raw HuggingFace dataset"""
        try:
            self.dataset = load_from_disk(self.dataset_path)[self.split]
            self.mode = 'raw'
            logger.info(f"✓ Loaded raw dataset with {len(self.dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to load raw dataset: {e}")
            raise

    def _load_preprocessed(self):
        """Load preprocessed NPZ files"""
        npz_path = os.path.join(
            self.dataset_path,
            f"{self.split}_processed_parallel.npz"
        )

        if not os.path.exists(npz_path):
            logger.warning(f"Preprocessed file not found: {npz_path}")
            logger.warning("Falling back to raw mode...")
            self.use_preprocessed = False
            self._load_raw()
            return

        try:
            data = np.load(npz_path, allow_pickle=True)
            self.images = data['images']  # [N, 224, 224, 3]
            self.captions = data['captions']  # [N]
            self.image_ids = data['image_ids']  # [N]
            self.mode = 'preprocessed'

            logger.info(f"✓ Loaded preprocessed data with {len(self.images)} samples")

        except Exception as e:
            logger.error(f"Failed to load preprocessed data: {e}")
            raise

    def _apply_subset(self, subset_size: int):
        """Limit dataset to subset_size samples"""
        if self.mode == 'raw':
            original_size = len(self.dataset)
            if subset_size < original_size:
                self.dataset = self.dataset.select(range(subset_size))
                logger.info(f"Applied subset: {original_size} → {subset_size} samples")
        else:
            original_size = len(self.images)
            if subset_size < original_size:
                self.images = self.images[:subset_size]
                self.captions = self.captions[:subset_size]
                self.image_ids = self.image_ids[:subset_size]
                logger.info(f"Applied subset: {original_size} → {subset_size} samples")

    def __len__(self) -> int:
        """Get dataset size"""
        if self.mode == 'raw':
            return len(self.dataset)
        else:
            return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
                - pixel_values: Preprocessed image [3, 224, 224]
                - input_ids: Tokenized caption [seq_len]
                - attention_mask: Attention mask [seq_len]
                - labels: Target labels for loss [seq_len]
                - caption: Original caption text (for evaluation)
        """
        if self.mode == 'raw':
            return self._get_raw_item(idx)
        else:
            return self._get_preprocessed_item(idx)

    def _get_raw_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from raw dataset"""
        sample = self.dataset[idx]

        # Process image
        image = sample['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Preprocess image
        if self.image_processor is not None:
            pixel_values = self.image_processor(
                images=image,
                return_tensors="pt"
            )['pixel_values'].squeeze(0)
        else:
            # Fallback: simple resize and normalize
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            pixel_values = transform(image)

        # Process caption
        caption = sample['caption']
        tokenized = self._tokenize_caption(caption)

        return {
            'pixel_values': pixel_values,
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids'].clone(),  # For language modeling
            'caption': caption
        }

    def _get_preprocessed_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from preprocessed data"""
        # Load preprocessed image (already normalized)
        image_array = self.images[idx]  # [224, 224, 3]

        # Convert to torch tensor and permute to [3, 224, 224]
        pixel_values = torch.from_numpy(image_array).permute(2, 0, 1).float()

        # Get caption
        caption = str(self.captions[idx])

        # Tokenize caption
        tokenized = self._tokenize_caption(caption)

        return {
            'pixel_values': pixel_values,
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids'].clone(),
            'caption': caption
        }

    def _tokenize_caption(self, caption: str) -> Dict[str, torch.Tensor]:
        """Tokenize caption text"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not provided")

        tokenized = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0)
        }


def create_dataloader(
    dataset_path: str,
    split: str,
    tokenizer,
    image_processor=None,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    use_preprocessed: bool = False,
    subset_size: Optional[int] = None,
    max_length: int = 256,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create DataLoader for training/evaluation.

    Args:
        dataset_path: Path to dataset
        split: Dataset split
        tokenizer: Tokenizer for captions
        image_processor: Image processor (for raw mode)
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        use_preprocessed: Use preprocessed NPZ files
        subset_size: Limit to N samples
        max_length: Maximum caption length
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    dataset = PMCImageCaptionDataset(
        dataset_path=dataset_path,
        split=split,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=max_length,
        use_preprocessed=use_preprocessed,
        subset_size=subset_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == 'train')  # Drop last incomplete batch for training
    )

    logger.info(f"DataLoader created: {len(dataloader)} batches")

    return dataloader


# Unit tests
if __name__ == "__main__":
    import sys

    print("="*70)
    print("TESTING PMC Dataset and DataLoader")
    print("="*70)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    try:
        from transformers import AutoTokenizer, AutoProcessor

        print("\n[Test 1] Loading tokenizer and processor...")
        tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B")
        tokenizer.pad_token = tokenizer.eos_token

        processor = AutoProcessor.from_pretrained(
            "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        print("✓ Tokenizer and processor loaded")

        # Test with raw dataset (if exists)
        print("\n[Test 2] Testing raw dataset loading...")
        raw_dataset_path = "data/raw/pmc_oa_10k"

        if os.path.exists(raw_dataset_path):
            dataset = PMCImageCaptionDataset(
                dataset_path=raw_dataset_path,
                split="train",
                tokenizer=tokenizer,
                image_processor=processor,
                use_preprocessed=False,
                subset_size=100
            )

            print(f"  Dataset size: {len(dataset)}")

            # Test single sample
            sample = dataset[0]
            print(f"  Pixel values shape: {sample['pixel_values'].shape}")
            print(f"  Input IDs shape: {sample['input_ids'].shape}")
            print(f"  Caption: {sample['caption'][:50]}...")

            assert sample['pixel_values'].shape == (3, 224, 224), "Image shape mismatch"
            assert len(sample['input_ids'].shape) == 1, "Input IDs should be 1D"

            print("✓ Raw dataset works")
        else:
            print(f"  ⚠️  Raw dataset not found at {raw_dataset_path}")
            print("     Skipping raw dataset test")

        # Test with preprocessed data (if exists)
        print("\n[Test 3] Testing preprocessed dataset loading...")
        preprocessed_path = "data/processed/pmc_oa_10k"

        if os.path.exists(preprocessed_path):
            dataset_prep = PMCImageCaptionDataset(
                dataset_path=preprocessed_path,
                split="train",
                tokenizer=tokenizer,
                use_preprocessed=True,
                subset_size=100
            )

            print(f"  Dataset size: {len(dataset_prep)}")

            sample = dataset_prep[0]
            print(f"  Pixel values shape: {sample['pixel_values'].shape}")
            print(f"  Input IDs shape: {sample['input_ids'].shape}")

            print("✓ Preprocessed dataset works")
        else:
            print(f"  ⚠️  Preprocessed data not found at {preprocessed_path}")
            print("     Skipping preprocessed dataset test")

        # Test DataLoader
        print("\n[Test 4] Testing DataLoader...")

        # Use whichever dataset is available
        if os.path.exists(raw_dataset_path):
            test_path = raw_dataset_path
            use_prep = False
        elif os.path.exists(preprocessed_path):
            test_path = preprocessed_path
            use_prep = True
        else:
            print("  ⚠️  No dataset available for DataLoader test")
            print("     Run Phase 2 to download data")
            print("\n✅ Architecture tests passed (data required for full tests)")
            sys.exit(0)

        dataloader = create_dataloader(
            dataset_path=test_path,
            split="train",
            tokenizer=tokenizer,
            image_processor=processor if not use_prep else None,
            batch_size=4,
            num_workers=2,
            use_preprocessed=use_prep,
            subset_size=20
        )

        print(f"  DataLoader batches: {len(dataloader)}")

        # Test batch
        batch = next(iter(dataloader))
        print(f"  Batch pixel values: {batch['pixel_values'].shape}")
        print(f"  Batch input IDs: {batch['input_ids'].shape}")
        print(f"  Batch attention mask: {batch['attention_mask'].shape}")
        print(f"  Batch labels: {batch['labels'].shape}")

        assert batch['pixel_values'].shape[0] == 4, "Batch size mismatch"
        assert batch['pixel_values'].shape[1:] == (3, 224, 224), "Image dims mismatch"

        print("✓ DataLoader works")

        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✅")
        print("="*70)
        print("\nDataset and DataLoader are ready for training.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
