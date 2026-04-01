#!/bin/bash

# Download PMC-OA dataset from HuggingFace
# Dataset: https://huggingface.co/datasets/axiong/pmc_oa

set -e

DATA_DIR="data/raw"
CACHE_DIR="$HOME/.cache/huggingface"

echo "=================================================="
echo "PMC-OA Dataset Download"
echo "=================================================="
echo "Dataset: axiong/pmc_oa"
echo "Size: ~50GB+ (1.6M image-caption pairs)"
echo "This may take several hours depending on network speed"
echo "=================================================="

# Create directories if they don't exist
mkdir -p "$DATA_DIR"
mkdir -p "$CACHE_DIR"

# Download dataset using Python
python << EOF
import os
from datasets import load_dataset
import sys

# Set cache directory
os.environ['HF_HOME'] = '$CACHE_DIR'

print("\n[1/3] Loading dataset from HuggingFace...")
print("This will download the dataset to cache and then save to disk...")

try:
    # Download dataset
    dataset = load_dataset(
        "axiong/pmc_oa",
        cache_dir="$CACHE_DIR",
        num_proc=8  # Parallel download
    )

    print("\n[2/3] Dataset loaded successfully!")
    print(f"  Train samples: {len(dataset['train']):,}")
    print(f"  Validation samples: {len(dataset['valid']):,}")
    print(f"  Test samples: {len(dataset['test']):,}")

    # Save to disk in Arrow format for faster loading
    print("\n[3/3] Saving dataset to disk...")
    dataset.save_to_disk("$DATA_DIR/pmc_oa_dataset")

    print("\n" + "="*50)
    print("Dataset downloaded successfully!")
    print("="*50)
    print(f"Location: $DATA_DIR/pmc_oa_dataset")
    print(f"Total samples: {len(dataset['train']) + len(dataset['valid']) + len(dataset['test']):,}")
    print("\nNext steps:")
    print("  1. Run EDA: python scripts/eda.py")
    print("  2. Create subsets: python scripts/create_subsets.py")

except Exception as e:
    print(f"\nError downloading dataset: {e}", file=sys.stderr)
    print("\nTroubleshooting:", file=sys.stderr)
    print("  1. Check internet connection", file=sys.stderr)
    print("  2. Ensure you have sufficient disk space (>100GB recommended)", file=sys.stderr)
    print("  3. Try running: huggingface-cli login (if dataset requires authentication)", file=sys.stderr)
    sys.exit(1)

EOF

echo ""
echo "Download script completed!"
