"""
Create smaller subsets of PMC-OA dataset for faster development iteration

Works with the new JSONL + images format:
  - Input: pmc_oa.jsonl + images/ directory
  - Output: Subset JSONL files + shared images directory

Generates:
  - 10k subset: For quick testing and debugging (10k train, 1k val, 1k test)
  - 100k subset: For medium-scale experiments (100k train, 5k val, 5k test)
"""

import argparse
from pathlib import Path
from tqdm import tqdm


def create_subsets(
    full_dataset_path="data/raw/pmc_oa",
    output_dir="data/raw"
):
    """
    Create development subsets of the dataset

    Args:
        full_dataset_path: Path to full dataset directory containing splits/ and images/
        output_dir: Directory to save subsets
    """

    print("="*70)
    print("Creating Development Subsets (JSONL Format)")
    print("="*70)

    dataset_path = Path(full_dataset_path)
    splits_dir = dataset_path / "splits"
    images_dir = dataset_path / "images"

    # Verify input structure
    if not splits_dir.exists():
        print(f"❌ Error: Splits directory not found: {splits_dir}")
        print("\nPlease run first:")
        print("  python scripts/download_pmc_oa.py --no-images")
        return False

    if not images_dir.exists():
        print(f"⚠️  Warning: Images directory not found: {images_dir}")
        print("  Subsets will be created, but images must be added later")

    # Check split files
    split_files = {
        'train': splits_dir / 'train.jsonl',
        'validation': splits_dir / 'validation.jsonl',
        'test': splits_dir / 'test.jsonl'
    }

    for split_name, split_path in split_files.items():
        if not split_path.exists():
            print(f"❌ Error: {split_name} split not found: {split_path}")
            return False

    # Count samples in each split
    print(f"\n📊 Counting samples in full dataset...")
    split_counts = {}
    for split_name, split_path in split_files.items():
        with open(split_path, 'r', encoding='utf-8') as f:
            count = sum(1 for line in f if line.strip())
            split_counts[split_name] = count

    print(f"  ✅ Full dataset:")
    print(f"    - Train: {split_counts['train']:,} samples")
    print(f"    - Validation: {split_counts['validation']:,} samples")
    print(f"    - Test: {split_counts['test']:,} samples")

    # Create subsets
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    subsets = [
        {
            'name': '10k',
            'sizes': {'train': 10000, 'validation': 1000, 'test': 1000},
            'description': 'Quick testing and debugging'
        },
        {
            'name': '100k',
            'sizes': {'train': 100000, 'validation': 5000, 'test': 5000},
            'description': 'Medium-scale experiments'
        }
    ]

    created_subsets = []

    for subset in subsets:
        subset_name = subset['name']
        subset_sizes = subset['sizes']

        print(f"\n{'='*70}")
        print(f"Creating {subset_name} subset")
        print(f"{'='*70}")

        # Create output directory
        subset_output_dir = output_path / f"pmc_oa_{subset_name}"
        subset_splits_dir = subset_output_dir / "splits"
        subset_splits_dir.mkdir(parents=True, exist_ok=True)

        # Create symbolic link or note about images
        if images_dir.exists():
            # Create symlink to original images (saves disk space)
            subset_images_link = subset_output_dir / "images"
            if subset_images_link.exists():
                subset_images_link.unlink()

            try:
                subset_images_link.symlink_to(images_dir.absolute())
                print(f"  ✅ Created symlink to images directory")
            except:
                # If symlinks don't work, create a note file
                with open(subset_output_dir / "IMAGES_LOCATION.txt", 'w') as f:
                    f.write(f"Images are located at: {images_dir.absolute()}\n")
                    f.write(f"Use this path when loading the dataset.\n")
                print(f"  ℹ️  Images location saved to IMAGES_LOCATION.txt")

        # Create subset JSONL files
        actual_counts = {}
        for split_name, split_path in split_files.items():
            target_size = min(subset_sizes[split_name], split_counts[split_name])
            output_split_path = subset_splits_dir / f"{split_name}.jsonl"

            print(f"\n  Creating {split_name} split ({target_size:,} samples)...")

            with open(split_path, 'r', encoding='utf-8') as f_in:
                with open(output_split_path, 'w', encoding='utf-8') as f_out:
                    count = 0
                    for line in tqdm(f_in, total=target_size, desc=f"  {split_name}"):
                        if count >= target_size:
                            break
                        if line.strip():
                            f_out.write(line)
                            count += 1

            actual_counts[split_name] = count
            print(f"    ✅ Created: {output_split_path.name} ({count:,} samples)")

        # Create README
        readme_path = subset_output_dir / "README.txt"
        with open(readme_path, 'w') as f:
            f.write(f"PMC-OA {subset_name.upper()} Subset\n")
            f.write("="*50 + "\n\n")
            f.write(f"Description: {subset['description']}\n\n")
            f.write("Dataset Structure:\n")
            f.write(f"  - Train: {actual_counts['train']:,} samples\n")
            f.write(f"  - Validation: {actual_counts['validation']:,} samples\n")
            f.write(f"  - Test: {actual_counts['test']:,} samples\n\n")
            f.write("Usage:\n")
            f.write(f"  from src.data.optimized_dataset import create_optimized_dataloader\n\n")
            f.write(f"  train_loader = create_optimized_dataloader(\n")
            f.write(f"      jsonl_path='{subset_splits_dir}/train.jsonl',\n")
            f.write(f"      images_dir='{images_dir}',\n")
            f.write(f"      batch_size=8,\n")
            f.write(f"      num_workers=4\n")
            f.write(f"  )\n")

        created_subsets.append({
            'name': subset_name,
            'path': subset_output_dir,
            'counts': actual_counts,
            'description': subset['description']
        })

        print(f"\n  ✅ {subset_name} subset created at: {subset_output_dir}")

    # Print summary
    print("\n" + "="*70)
    print("✅ Subset Creation Complete!")
    print("="*70)

    print("\nCreated subsets:")
    for i, subset_info in enumerate(created_subsets, 1):
        print(f"\n  {i}. {subset_info['name'].upper()} Subset: {subset_info['path']}")
        print(f"     Description: {subset_info['description']}")
        print(f"     Train: {subset_info['counts']['train']:,} samples")
        print(f"     Validation: {subset_info['counts']['validation']:,} samples")
        print(f"     Test: {subset_info['counts']['test']:,} samples")

    print(f"\n  Full Dataset: {dataset_path}")
    print(f"     Train: {split_counts['train']:,} samples")
    print(f"     Validation: {split_counts['validation']:,} samples")
    print(f"     Test: {split_counts['test']:,} samples")

    # Usage examples
    print("\n" + "="*70)
    print("Usage Examples")
    print("="*70)

    for subset_info in created_subsets:
        print(f"\n{subset_info['name'].upper()} Subset:")
        print(f"  python scripts/train_with_optimized_dataset.py \\")
        print(f"      --data_dir {subset_info['path']} \\")
        print(f"      --batch_size 8 --num_workers 4")

    print("\nFull Dataset:")
    print(f"  python scripts/train_with_optimized_dataset.py \\")
    print(f"      --data_dir {dataset_path} \\")
    print(f"      --batch_size 8 --num_workers 4")

    print("\n" + "="*70)
    print("Next Steps")
    print("="*70)
    print("  1. Test with 10k subset: python scripts/train_with_optimized_dataset.py --data_dir data/raw/pmc_oa_10k --test_only")
    print("  2. Train on 10k subset for quick validation")
    print("  3. Scale up to 100k subset for medium experiments")
    print("  4. Train on full dataset for final model")
    print("="*70)

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create dataset subsets')
    parser.add_argument(
        '--full_dataset_path',
        type=str,
        default='data/raw/pmc_oa',
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
