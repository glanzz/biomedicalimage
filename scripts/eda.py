"""
Exploratory Data Analysis for PMC-OA dataset
Analyzes image properties, caption statistics, and modality distribution
"""

import os
import json
import numpy as np
import pandas as pd
from datasets import load_from_disk
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

def analyze_dataset(data_dir="data/raw/pmc_oa_dataset", output_dir="outputs/eda", max_samples=10000):
    """Perform comprehensive EDA on PMC-OA dataset"""

    print("="*60)
    print("PMC-OA Dataset - Exploratory Data Analysis")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    print("\n[1/5] Loading dataset...")
    try:
        dataset = load_from_disk(data_dir)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Please ensure the dataset exists at: {data_dir}")
        print("Run: ./scripts/download_data.sh to download the dataset first")
        sys.exit(1)

    # Analyze splits
    print("\n[2/5] Dataset Splits:")
    print("-" * 40)
    split_info = {}
    for split in ['train', 'valid', 'test']:
        split_size = len(dataset[split])
        split_info[split] = split_size
        print(f"  {split:10s}: {split_size:,} samples")
    print("-" * 40)
    total = sum(split_info.values())
    print(f"  {'Total':10s}: {total:,} samples")

    # Sample statistics on train set
    train_data = dataset['train']

    # --- Caption Analysis ---
    print(f"\n[3/5] Analyzing captions (sampling {min(max_samples, len(train_data)):,} samples)...")
    caption_lengths = []
    word_counts = []

    # Import nltk word tokenizer
    try:
        from nltk.tokenize import word_tokenize
        import nltk
        # Try to download punkt if not available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
    except ImportError:
        print("Warning: NLTK not available, using simple split for word count")
        word_tokenize = lambda x: x.split()

    for i in tqdm(range(min(max_samples, len(train_data))), desc="Processing captions"):
        caption = train_data[i]['caption']
        caption_lengths.append(len(caption))
        try:
            word_counts.append(len(word_tokenize(caption)))
        except:
            word_counts.append(len(caption.split()))

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

    print("\nCaption Statistics:")
    print("-" * 40)
    for metric, values in stats.items():
        print(f"\n{metric}:")
        for k, v in values.items():
            print(f"  {k:20s}: {v:.2f}")

    # Plot caption distributions
    print("\n[4/5] Generating visualizations...")
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
    plt.savefig(f"{output_dir}/caption_distributions.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/caption_distributions.png")
    plt.close()

    # --- Image Analysis ---
    print(f"\n[5/5] Analyzing images (sampling {min(5000, len(train_data))} samples)...")
    image_widths = []
    image_heights = []
    image_aspects = []
    image_modes = []

    for i in tqdm(range(min(5000, len(train_data))), desc="Processing images"):
        try:
            img = train_data[i]['image']
            if img is not None:
                image_widths.append(img.width)
                image_heights.append(img.height)
                image_aspects.append(img.width / img.height)
                image_modes.append(img.mode)
        except Exception as e:
            continue

    print("\nImage Statistics:")
    print("-" * 40)
    print(f"  Width:        {np.mean(image_widths):.0f} ± {np.std(image_widths):.0f} pixels")
    print(f"  Height:       {np.mean(image_heights):.0f} ± {np.std(image_heights):.0f} pixels")
    print(f"  Aspect Ratio: {np.mean(image_aspects):.2f} ± {np.std(image_aspects):.2f}")
    print(f"\nImage Modes:")
    mode_counts = Counter(image_modes)
    for mode, count in mode_counts.most_common():
        print(f"  {mode:8s}: {count:,} ({count/len(image_modes)*100:.1f}%)")

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
    plt.savefig(f"{output_dir}/image_dimensions.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/image_dimensions.png")
    plt.close()

    # Save statistics to JSON
    metadata = {
        'dataset_info': {
            'name': 'PMC-OA',
            'source': 'axiong/pmc_oa (HuggingFace)',
            'description': 'Biomedical image-caption pairs from PubMed Central'
        },
        'splits': split_info,
        'total_samples': total,
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
        'image_stats': {
            'width_mean': float(np.mean(image_widths)),
            'width_std': float(np.std(image_widths)),
            'height_mean': float(np.mean(image_heights)),
            'height_std': float(np.std(image_heights)),
            'aspect_ratio_mean': float(np.mean(image_aspects)),
            'aspect_ratio_std': float(np.std(image_aspects)),
            'modes': dict(Counter(image_modes))
        },
        'sample_captions': [train_data[i]['caption'] for i in range(min(10, len(train_data)))]
    }

    with open(f"{output_dir}/dataset_statistics.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved: {output_dir}/dataset_statistics.json")

    # Save metadata to data folder
    os.makedirs("data/metadata", exist_ok=True)
    with open("data/metadata/dataset_stats.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved: data/metadata/dataset_stats.json")

    print("\n" + "="*60)
    print("EDA Complete!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}/")
    print("  - caption_distributions.png")
    print("  - image_dimensions.png")
    print("  - dataset_statistics.json")
    print("\nNext steps:")
    print("  1. Create subsets: python scripts/create_subsets.py")
    print("  2. Proceed to Phase 3: Parallel preprocessing")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform EDA on PMC-OA dataset')
    parser.add_argument('--data_dir', type=str, default='data/raw/pmc_oa_dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='outputs/eda',
                        help='Directory to save EDA results')
    parser.add_argument('--max_caption_samples', type=int, default=10000,
                        help='Maximum samples to analyze for captions')

    args = parser.parse_args()

    analyze_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_samples=args.max_caption_samples
    )
