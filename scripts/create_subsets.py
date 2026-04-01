"""
Create smaller subsets of PMC-OA dataset for faster development iteration

Generates:
  - 10k subset: For quick testing and debugging
  - 100k subset: For medium-scale experiments
"""

import os
import argparse
from datasets import load_from_disk, DatasetDict

def create_subsets(
    full_dataset_path="data/raw/pmc_oa_dataset",
    output_dir="data/raw"
):
    """Create development subsets of the dataset"""

    print("="*60)
    print("Creating Development Subsets")
    print("="*60)

    print(f"\n[1/3] Loading full dataset from: {full_dataset_path}")
    try:
        dataset = load_from_disk(full_dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Please ensure the dataset exists at: {full_dataset_path}")
        print("Run: ./scripts/download_data.sh to download the dataset first")
        return

    print(f"  ✓ Loaded dataset")
    print(f"    - Train: {len(dataset['train']):,} samples")
    print(f"    - Valid: {len(dataset['valid']):,} samples")
    print(f"    - Test:  {len(dataset['test']):,} samples")

    # Create 10k subset for quick testing
    print(f"\n[2/3] Creating 10k subset...")
    subset_10k = {
        'train': dataset['train'].select(range(min(10000, len(dataset['train'])))),
        'valid': dataset['valid'].select(range(min(1000, len(dataset['valid'])))),
        'test': dataset['test'].select(range(min(1000, len(dataset['test']))))
    }

    subset_10k = DatasetDict(subset_10k)
    output_path_10k = os.path.join(output_dir, "pmc_oa_10k")
    subset_10k.save_to_disk(output_path_10k)
    print(f"  ✓ Created 10k subset:")
    print(f"    - Train: {len(subset_10k['train']):,} samples")
    print(f"    - Valid: {len(subset_10k['valid']):,} samples")
    print(f"    - Test:  {len(subset_10k['test']):,} samples")
    print(f"    - Location: {output_path_10k}")

    # Create 100k subset for medium-scale testing
    print(f"\n[3/3] Creating 100k subset...")
    subset_100k = {
        'train': dataset['train'].select(range(min(100000, len(dataset['train'])))),
        'valid': dataset['valid'].select(range(min(5000, len(dataset['valid'])))),
        'test': dataset['test'].select(range(min(5000, len(dataset['test']))))
    }

    subset_100k = DatasetDict(subset_100k)
    output_path_100k = os.path.join(output_dir, "pmc_oa_100k")
    subset_100k.save_to_disk(output_path_100k)
    print(f"  ✓ Created 100k subset:")
    print(f"    - Train: {len(subset_100k['train']):,} samples")
    print(f"    - Valid: {len(subset_100k['valid']):,} samples")
    print(f"    - Test:  {len(subset_100k['test']):,} samples")
    print(f"    - Location: {output_path_100k}")

    print("\n" + "="*60)
    print("Subset Creation Complete!")
    print("="*60)
    print("\nCreated subsets:")
    print(f"  1. 10k subset:  {output_path_10k}")
    print(f"     - Use for: Quick testing, debugging")
    print(f"  2. 100k subset: {output_path_100k}")
    print(f"     - Use for: Medium-scale experiments")
    print(f"  3. Full dataset: {full_dataset_path}")
    print(f"     - Use for: Final training")
    print("\nUsage in configs:")
    print('  - Set dataset_path to one of the above paths')
    print('  - Example: dataset_path: "data/raw/pmc_oa_10k"')
    print("\nNext steps:")
    print("  1. Proceed to Phase 3: Parallel preprocessing")
    print("  2. Or run initial tests with 10k subset")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create dataset subsets')
    parser.add_argument(
        '--full_dataset_path',
        type=str,
        default='data/raw/pmc_oa_dataset',
        help='Path to full dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/raw',
        help='Directory to save subsets'
    )

    args = parser.parse_args()

    create_subsets(
        full_dataset_path=args.full_dataset_path,
        output_dir=args.output_dir
    )
