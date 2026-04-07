#!/usr/bin/env python3
"""
Download PMC-OA dataset from official source
Dataset structure:
  - images.zip: All images
  - pmc_oa.jsonl: Main dataset annotations
  - pmc_oa_beta.jsonl: Beta version annotations

Each JSONL row format:
{
    "image": "PMC212319_Fig3_4.jpg",
    "caption": "A. Real time image of the translocation..."
}
"""

import os
import sys
import json
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import argparse

# PMC-OA dataset URLs (update these with actual URLs)
DATASET_URLS = {
    "images": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/images.zip",
    "pmc_oa": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/pmc_oa.jsonl",
    "pmc_oa_beta": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/pmc_oa_beta.jsonl",
}

def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """Download file with progress bar"""

    print(f"Downloading: {url}")
    print(f"Saving to: {output_path}")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"✅ Downloaded: {output_path.name} ({total_size / (1024**3):.2f} GB)")
        return True

    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False


def unzip_images(zip_path: Path, extract_to: Path):
    """Unzip images with progress bar"""

    print(f"\nExtracting images from {zip_path.name}...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()

            with tqdm(total=len(members), desc="Extracting images") as pbar:
                for member in members:
                    zip_ref.extract(member, extract_to)
                    pbar.update(1)

        print(f"✅ Extracted {len(members):,} files to {extract_to}")

        # Optionally remove zip file to save space
        # zip_path.unlink()

        return True

    except Exception as e:
        print(f"❌ Failed to extract {zip_path}: {e}")
        return False


def load_and_validate_jsonl(jsonl_path: Path, limit: int = 5):
    """Load and validate JSONL file"""

    print(f"\nValidating {jsonl_path.name}...")

    try:
        count = 0
        samples = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    count += 1

                    if count <= limit:
                        samples.append(data)

        print(f"✅ Found {count:,} samples in {jsonl_path.name}")

        if samples:
            print("\nSample entries:")
            for i, sample in enumerate(samples[:3], 1):
                print(f"\n  Sample {i}:")
                print(f"    Image: {sample.get('image', 'N/A')}")
                caption = sample.get('caption', 'N/A')
                print(f"    Caption: {caption[:100]}..." if len(caption) > 100 else f"    Caption: {caption}")

        return count

    except Exception as e:
        print(f"❌ Failed to validate {jsonl_path}: {e}")
        return 0


def create_dataset_splits(jsonl_path: Path, output_dir: Path,
                         train_ratio: float = 0.85,
                         val_ratio: float = 0.10,
                         test_ratio: float = 0.05):
    """Create train/val/test splits from JSONL file"""

    print(f"\nCreating dataset splits...")
    print(f"  Train: {train_ratio*100:.0f}%")
    print(f"  Val: {val_ratio*100:.0f}%")
    print(f"  Test: {test_ratio*100:.0f}%")

    try:
        # Load all samples
        samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        total = len(samples)

        # Shuffle samples for random split
        import random
        random.seed(42)
        random.shuffle(samples)

        # Calculate split indices
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        splits = {
            'train': samples[:train_end],
            'validation': samples[train_end:val_end],
            'test': samples[val_end:]
        }

        # Save splits
        output_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in splits.items():
            split_path = output_dir / f"{split_name}.jsonl"

            with open(split_path, 'w', encoding='utf-8') as f:
                for sample in split_data:
                    f.write(json.dumps(sample) + '\n')

            print(f"  ✅ {split_name}: {len(split_data):,} samples → {split_path.name}")

        return True

    except Exception as e:
        print(f"❌ Failed to create splits: {e}")
        return False


def download_pmc_oa(output_dir: str = "data/raw/pmc_oa",
                    download_images: bool = True,
                    create_splits: bool = True):
    """Main function to download PMC-OA dataset"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("PMC-OA Dataset Download")
    print("="*70)
    print(f"Output directory: {output_path.absolute()}")
    print()

    success = True

    # Download JSONL files
    print("[1/4] Downloading annotation files...")
    print("-" * 70)

    jsonl_files = {}
    for name in ["pmc_oa", "pmc_oa_beta"]:
        if name in DATASET_URLS:
            output_file = output_path / f"{name}.jsonl"

            if output_file.exists():
                print(f"⏭️  Skipping {name}.jsonl (already exists)")
                jsonl_files[name] = output_file
            else:
                if download_file(DATASET_URLS[name], output_file):
                    jsonl_files[name] = output_file
                else:
                    success = False

    # Download and extract images
    if download_images and "images" in DATASET_URLS:
        print("\n[2/4] Downloading images...")
        print("-" * 70)

        images_zip = output_path / "images.zip"
        images_dir = output_path / "images"

        if images_dir.exists() and any(images_dir.iterdir()):
            print(f"⏭️  Skipping images download (directory already exists)")
        else:
            if download_file(DATASET_URLS["images"], images_zip):
                print("\n[3/4] Extracting images...")
                print("-" * 70)
                if not unzip_images(images_zip, output_path):
                    success = False
            else:
                success = False
                print("⚠️  Warning: Failed to download images")
    else:
        print("\n⏭️  Skipping images download (use --download-images to enable)")

    # Validate JSONL files
    print("\n[4/4] Validating dataset...")
    print("-" * 70)

    for name, jsonl_path in jsonl_files.items():
        num_samples = load_and_validate_jsonl(jsonl_path)

        if num_samples == 0:
            success = False

    # Create train/val/test splits
    if create_splits and "pmc_oa" in jsonl_files:
        splits_dir = output_path / "splits"
        create_dataset_splits(jsonl_files["pmc_oa"], splits_dir)

    # Print summary
    print("\n" + "="*70)
    if success:
        print("✅ Download Complete!")
    else:
        print("⚠️  Download completed with some errors")
    print("="*70)

    print(f"\nDataset location: {output_path.absolute()}")
    print("\nDirectory structure:")
    print("  pmc_oa/")
    print("  ├── images/              # All images")
    print("  ├── pmc_oa.jsonl         # Main annotations")
    print("  ├── pmc_oa_beta.jsonl    # Beta annotations")
    print("  └── splits/              # Train/val/test splits")
    print("      ├── train.jsonl")
    print("      ├── validation.jsonl")
    print("      └── test.jsonl")

    print("\nNext steps:")
    print("  1. Explore data: python scripts/eda.py --data_dir data/raw/pmc_oa")
    print("  2. Preprocess: python scripts/benchmark_preprocessing.py")
    print("  3. Train model: python -m src.training.single_gpu_trainer")

    return success


def main():
    parser = argparse.ArgumentParser(description="Download PMC-OA dataset")
    parser.add_argument("--output_dir", default="data/raw/pmc_oa",
                       help="Output directory for dataset")
    parser.add_argument("--no-images", dest="download_images", action="store_false",
                       help="Skip downloading images (only download annotations)")
    parser.add_argument("--no-splits", dest="create_splits", action="store_false",
                       help="Skip creating train/val/test splits")

    args = parser.parse_args()

    success = download_pmc_oa(
        output_dir=args.output_dir,
        download_images=args.download_images,
        create_splits=args.create_splits
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
