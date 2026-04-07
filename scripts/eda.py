#!/usr/bin/env python3
"""
Exploratory Data Analysis for PMC-OA dataset (JSONL format)
Analyzes image properties, caption statistics, and modality distribution

Works with the new PMC-OA structure:
  data/raw/pmc_oa/
  ├── images/
  ├── pmc_oa.jsonl
  └── splits/
      ├── train.jsonl
      ├── validation.jsonl
      └── test.jsonl
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import argparse
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_jsonl(jsonl_path: Path, max_samples: int = None):
    """Load samples from JSONL file"""
    samples = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break

            if line.strip():
                data = json.loads(line)
                samples.append(data)

    return samples


def analyze_captions(samples, output_dir: Path):
    """Analyze caption statistics"""

    print("\n[2/5] Analyzing captions...")
    print("-" * 70)

    caption_lengths = []
    word_counts = []

    # Import nltk word tokenizer
    try:
        from nltk.tokenize import word_tokenize
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("  Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        use_nltk = True
    except ImportError:
        print("  Warning: NLTK not available, using simple split for word count")
        word_tokenize = lambda x: x.split()
        use_nltk = False

    for sample in tqdm(samples, desc="  Processing captions"):
        caption = sample.get('caption', '')

        if caption:
            caption_lengths.append(len(caption))
            try:
                word_counts.append(len(word_tokenize(caption)))
            except:
                word_counts.append(len(caption.split()))

    # Calculate statistics
    stats = {
        'Caption Length (chars)': {
            'mean': np.mean(caption_lengths),
            'std': np.std(caption_lengths),
            'min': np.min(caption_lengths),
            'max': np.max(caption_lengths),
            'median': np.median(caption_lengths),
            '95th percentile': np.percentile(caption_lengths, 95)
        },
        'Word Count': {
            'mean': np.mean(word_counts),
            'std': np.std(word_counts),
            'min': np.min(word_counts),
            'max': np.max(word_counts),
            'median': np.median(word_counts),
            '95th percentile': np.percentile(word_counts, 95)
        }
    }

    print("\n  Caption Statistics:")
    print("  " + "-" * 66)
    for metric, values in stats.items():
        print(f"\n  {metric}:")
        for k, v in values.items():
            print(f"    {k:20s}: {v:.2f}")

    # Plot caption distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(caption_lengths, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Caption Length (characters)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Caption Lengths')
    axes[0].axvline(np.mean(caption_lengths), color='r', linestyle='--',
                     label=f'Mean: {np.mean(caption_lengths):.0f}', linewidth=2)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(word_counts, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1].set_xlabel('Word Count')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Word Counts')
    axes[1].axvline(np.mean(word_counts), color='r', linestyle='--',
                     label=f'Mean: {np.mean(word_counts):.0f}', linewidth=2)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "caption_distributions.png", dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Saved: {output_dir}/caption_distributions.png")
    plt.close()

    return stats, caption_lengths, word_counts


def analyze_images(samples, images_dir: Path, output_dir: Path, max_images: int = 5000):
    """Analyze image statistics"""

    print("\n[3/5] Analyzing images...")
    print("-" * 70)

    image_widths = []
    image_heights = []
    image_aspects = []
    image_modes = []
    missing_images = []

    for i, sample in enumerate(tqdm(samples[:max_images], desc="  Processing images")):
        if i >= max_images:
            break

        image_filename = sample.get('image', '')
        if not image_filename:
            continue

        image_path = images_dir / image_filename

        try:
            if image_path.exists():
                img = Image.open(image_path)
                image_widths.append(img.width)
                image_heights.append(img.height)
                image_aspects.append(img.width / img.height)
                image_modes.append(img.mode)
                img.close()
            else:
                missing_images.append(image_filename)
        except Exception as e:
            missing_images.append(f"{image_filename} (error: {str(e)[:30]})")
            continue

    if missing_images:
        print(f"\n  ⚠️  Warning: {len(missing_images)} images not found or failed to load")
        if len(missing_images) <= 10:
            for img in missing_images[:10]:
                print(f"      - {img}")

    if not image_widths:
        print("  ⚠️  No images found! Check images directory:")
        print(f"      {images_dir}")
        return None

    print("\n  Image Statistics:")
    print("  " + "-" * 66)
    print(f"    Images analyzed:  {len(image_widths):,}")
    print(f"    Width:            {np.mean(image_widths):.0f} ± {np.std(image_widths):.0f} pixels")
    print(f"    Height:           {np.mean(image_heights):.0f} ± {np.std(image_heights):.0f} pixels")
    print(f"    Aspect Ratio:     {np.mean(image_aspects):.2f} ± {np.std(image_aspects):.2f}")

    print(f"\n  Image Modes:")
    mode_counts = Counter(image_modes)
    for mode, count in mode_counts.most_common():
        print(f"    {mode:8s}: {count:,} ({count/len(image_modes)*100:.1f}%)")

    # Plot image dimensions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(image_widths, image_heights, alpha=0.3, s=1)
    axes[0].set_xlabel('Width (pixels)')
    axes[0].set_ylabel('Height (pixels)')
    axes[0].set_title('Image Dimensions Scatter Plot')
    axes[0].set_xlim(0, min(2000, max(image_widths)))
    axes[0].set_ylim(0, min(2000, max(image_heights)))
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(image_widths, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    axes[1].set_xlabel('Width (pixels)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Width Distribution')
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(image_heights, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[2].set_xlabel('Height (pixels)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Height Distribution')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "image_dimensions.png", dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Saved: {output_dir}/image_dimensions.png")
    plt.close()

    return {
        'width_mean': float(np.mean(image_widths)),
        'width_std': float(np.std(image_widths)),
        'height_mean': float(np.mean(image_heights)),
        'height_std': float(np.std(image_heights)),
        'aspect_ratio_mean': float(np.mean(image_aspects)),
        'aspect_ratio_std': float(np.std(image_aspects)),
        'modes': dict(Counter(image_modes)),
        'num_analyzed': len(image_widths),
        'num_missing': len(missing_images)
    }


def analyze_dataset(
    data_dir: str = "data/raw/pmc_oa",
    output_dir: str = "outputs/eda",
    max_caption_samples: int = 10000,
    max_image_samples: int = 5000
):
    """Perform comprehensive EDA on PMC-OA dataset (JSONL format)"""

    print("="*70)
    print("PMC-OA Dataset - Exploratory Data Analysis")
    print("="*70)

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if data directory exists
    if not data_path.exists():
        print(f"\n❌ Error: Data directory not found: {data_path}")
        print("\nPlease download the dataset first:")
        print("  python3 scripts/download_pmc_oa.py")
        sys.exit(1)

    # Determine which JSONL files to analyze
    splits_dir = data_path / "splits"
    main_jsonl = data_path / "pmc_oa.jsonl"

    jsonl_files = {}

    if splits_dir.exists():
        # Use split files if available
        print(f"\n[1/5] Loading dataset from splits...")
        print("-" * 70)

        for split_name in ['train', 'validation', 'test']:
            split_path = splits_dir / f"{split_name}.jsonl"
            if split_path.exists():
                samples = load_jsonl(split_path)
                jsonl_files[split_name] = samples
                print(f"  {split_name:12s}: {len(samples):,} samples")
            else:
                print(f"  {split_name:12s}: Not found (skipping)")

        # Use train split for detailed analysis
        analysis_samples = jsonl_files.get('train', [])

    elif main_jsonl.exists():
        # Use main JSONL file
        print(f"\n[1/5] Loading dataset from {main_jsonl.name}...")
        print("-" * 70)

        analysis_samples = load_jsonl(main_jsonl, max_samples=max_caption_samples)
        jsonl_files['all'] = analysis_samples
        print(f"  Total samples: {len(analysis_samples):,}")

    else:
        print(f"\n❌ Error: No JSONL files found in {data_path}")
        print("\nExpected structure:")
        print("  data/raw/pmc_oa/")
        print("  ├── images/")
        print("  ├── pmc_oa.jsonl")
        print("  └── splits/")
        sys.exit(1)

    if not analysis_samples:
        print("\n❌ Error: No samples to analyze!")
        sys.exit(1)

    # Print split summary
    if len(jsonl_files) > 1:
        print("\n  Split Summary:")
        print("  " + "-" * 66)
        total = sum(len(samples) for samples in jsonl_files.values())
        for split_name, samples in jsonl_files.items():
            pct = len(samples) / total * 100
            print(f"    {split_name:12s}: {len(samples):>8,} ({pct:>5.1f}%)")
        print("  " + "-" * 66)
        print(f"    {'Total':12s}: {total:>8,} (100.0%)")

    # Limit samples for analysis
    if len(analysis_samples) > max_caption_samples:
        print(f"\n  Note: Limiting caption analysis to {max_caption_samples:,} samples")
        caption_samples = analysis_samples[:max_caption_samples]
    else:
        caption_samples = analysis_samples

    # Analyze captions
    caption_stats, caption_lengths, word_counts = analyze_captions(caption_samples, output_path)

    # Analyze images
    images_dir = data_path / "images"

    if not images_dir.exists():
        print(f"\n  ⚠️  Warning: Images directory not found: {images_dir}")
        print(f"  Skipping image analysis. Run with --no-images option or download images.")
        image_stats = None
    else:
        image_stats = analyze_images(caption_samples, images_dir, output_path, max_image_samples)

    # Analyze common medical terms
    print("\n[4/5] Analyzing medical terminology...")
    print("-" * 70)

    medical_terms = {
        'imaging': ['ct', 'mri', 'x-ray', 'ultrasound', 'scan', 'radiograph', 'microscopy'],
        'anatomy': ['cell', 'tissue', 'organ', 'brain', 'heart', 'lung', 'liver', 'kidney'],
        'pathology': ['cancer', 'tumor', 'disease', 'infection', 'inflammation', 'lesion'],
        'colors': ['staining', 'fluorescence', 'contrast', 'intensity']
    }

    term_counts = {}
    for category, terms in medical_terms.items():
        term_counts[category] = {}
        for term in terms:
            count = sum(1 for s in caption_samples if term in s.get('caption', '').lower())
            if count > 0:
                term_counts[category][term] = count

    print("\n  Common Medical Terms:")
    for category, terms in term_counts.items():
        if terms:
            print(f"\n  {category.title()}:")
            for term, count in sorted(terms.items(), key=lambda x: x[1], reverse=True)[:5]:
                pct = count / len(caption_samples) * 100
                print(f"    {term:15s}: {count:>6,} ({pct:>5.1f}%)")

    # Save comprehensive statistics
    print("\n[5/5] Saving results...")
    print("-" * 70)

    metadata = {
        'dataset_info': {
            'name': 'PMC-OA',
            'source': 'axiong/pmc_oa',
            'description': 'Biomedical image-caption pairs from PubMed Central',
            'format': 'JSONL + images directory'
        },
        'splits': {name: len(samples) for name, samples in jsonl_files.items()},
        'total_samples': sum(len(samples) for samples in jsonl_files.values()),
        'caption_stats': {
            'length_mean': float(np.mean(caption_lengths)),
            'length_std': float(np.std(caption_lengths)),
            'length_min': int(np.min(caption_lengths)),
            'length_max': int(np.max(caption_lengths)),
            'length_median': float(np.median(caption_lengths)),
            'words_mean': float(np.mean(word_counts)),
            'words_std': float(np.std(word_counts)),
            'words_min': int(np.min(word_counts)),
            'words_max': int(np.max(word_counts)),
            'words_median': float(np.median(word_counts))
        },
        'image_stats': image_stats if image_stats else 'Not analyzed',
        'medical_terms': term_counts,
        'sample_captions': [s.get('caption', '') for s in caption_samples[:10]]
    }

    # Save to output directory
    with open(output_path / "dataset_statistics.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved: {output_path}/dataset_statistics.json")

    # Save to data metadata folder
    metadata_dir = Path("data/metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_dir / "dataset_stats.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved: {metadata_dir}/dataset_stats.json")

    print("\n" + "="*70)
    print("✅ EDA Complete!")
    print("="*70)
    print(f"\nResults saved to: {output_path}/")
    print("  - caption_distributions.png")
    if image_stats:
        print("  - image_dimensions.png")
    print("  - dataset_statistics.json")

    print("\nNext steps:")
    print("  1. Preprocess data: python scripts/benchmark_preprocessing.py")
    print("  2. Train model: python -m src.training.single_gpu_trainer")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Perform EDA on PMC-OA dataset (JSONL format)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze full dataset
  python scripts/eda.py --data_dir data/raw/pmc_oa

  # Analyze with custom sample limits
  python scripts/eda.py --max_caption_samples 5000 --max_image_samples 2000

  # Specify output directory
  python scripts/eda.py --output_dir outputs/my_eda
        """
    )

    parser.add_argument('--data_dir', type=str, default='data/raw/pmc_oa',
                       help='Path to PMC-OA dataset directory (default: data/raw/pmc_oa)')
    parser.add_argument('--output_dir', type=str, default='outputs/eda',
                       help='Directory to save EDA results (default: outputs/eda)')
    parser.add_argument('--max_caption_samples', type=int, default=20000,
                       help='Maximum samples to analyze for captions (default: 10000)')
    parser.add_argument('--max_image_samples', type=int, default=20000,
                       help='Maximum images to analyze (default: 20000)')

    args = parser.parse_args()

    analyze_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_caption_samples=args.max_caption_samples,
        max_image_samples=args.max_image_samples
    )

