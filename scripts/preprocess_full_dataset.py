"""
Preprocess the full PMC-OA dataset using parallel processing
Supports processing full dataset or subsets with configurable worker count
"""

import os
import sys
import json
import argparse
import time
from datasets import load_from_disk

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import ParallelImagePreprocessor


def preprocess_dataset(
    dataset_path="data/raw/pmc_oa",
    output_dir="data/processed",
    n_workers=32,
    partition_size=100,
    splits=None
):
    """
    Preprocess dataset splits using parallel processing

    Args:
        dataset_path: Path to input dataset
        output_dir: Directory to save preprocessed data
        n_workers: Number of parallel workers
        partition_size: Samples per partition
        splits: List of splits to process (default: ['train', 'valid', 'test'])
    """

    if splits is None:
        splits = ['train', 'valid', 'test']

    print("="*70)
    print("PARALLEL DATASET PREPROCESSING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Output: {output_dir}")
    print(f"  Workers: {n_workers}")
    print(f"  Partition size: {partition_size}")
    print(f"  Splits: {', '.join(splits)}")

    # Load dataset
    print(f"\n[1/3] Loading dataset...")
    try:
        dataset = load_from_disk(dataset_path)
        print(f"  ✓ Dataset loaded")
        for split in splits:
            print(f"    - {split}: {len(dataset[split]):,} samples")
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        print(f"  Please ensure dataset exists at: {dataset_path}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize preprocessor
    print(f"\n[2/3] Initializing parallel preprocessor...")
    preprocessor = ParallelImagePreprocessor(
        n_workers=n_workers,
        threads_per_worker=2,
        image_size=224
    )

    # Process each split
    print(f"\n[3/3] Processing splits...")
    all_metrics = {}
    overall_start = time.time()

    for split in splits:
        print("\n" + "-"*70)
        print(f"Processing {split.upper()} split ({len(dataset[split]):,} samples)")
        print("-"*70)

        split_start = time.time()

        # Prepare samples
        samples = [
            {
                'image': s['image'],
                'caption': s['caption'],
                'image_id': i
            }
            for i, s in enumerate(dataset[split])
        ]

        # Process
        try:
            metrics = preprocessor.preprocess_batch_parallel(
                samples,
                output_dir,
                split,
                partition_size=partition_size
            )

            split_time = time.time() - split_start

            # Add additional metrics
            metrics['total_time'] = split_time
            metrics['samples_total'] = len(dataset[split])

            all_metrics[split] = metrics

            print(f"\n✓ {split.upper()} split complete:")
            print(f"  Samples processed: {metrics['num_samples']:,}")
            print(f"  Time: {split_time:.2f}s ({split_time/60:.1f} min)")
            print(f"  Throughput: {metrics['throughput']:.2f} images/sec")
            print(f"  Output: {metrics['output_file']}")

        except Exception as e:
            print(f"\n✗ Error processing {split} split: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Close cluster
    preprocessor.close_cluster()

    # Overall statistics
    overall_time = time.time() - overall_start
    total_samples = sum(m['num_samples'] for m in all_metrics.values())
    overall_throughput = total_samples / overall_time if overall_time > 0 else 0

    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"\nOverall Statistics:")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Total time: {overall_time:.2f}s ({overall_time/60:.1f} min, {overall_time/3600:.2f} hr)")
    print(f"  Average throughput: {overall_throughput:.2f} images/sec")

    print(f"\nPer-split Statistics:")
    print(f"  {'Split':<10} {'Samples':<12} {'Time':<15} {'Throughput'}")
    print(f"  {'-'*10} {'-'*12} {'-'*15} {'-'*15}")

    for split, metrics in all_metrics.items():
        print(f"  {split:<10} {metrics['num_samples']:<12,} "
              f"{metrics['total_time']:<15.2f}s {metrics['throughput']:<15.2f} img/s")

    # Save metadata
    metadata = {
        'dataset_path': dataset_path,
        'output_dir': output_dir,
        'n_workers': n_workers,
        'partition_size': partition_size,
        'overall_time': overall_time,
        'total_samples': total_samples,
        'overall_throughput': overall_throughput,
        'splits': all_metrics
    }

    metadata_file = os.path.join(output_dir, 'preprocessing_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved: {metadata_file}")
    print(f"\nPreprocessed data location: {output_dir}/")
    print("  Files:")
    for split in splits:
        if split in all_metrics:
            print(f"    - {split}_processed_parallel.npz")

    print("\n✅ All preprocessing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess full dataset with parallel processing')

    parser.add_argument(
        '--dataset_path',
        type=str,
        default='data/raw/pmc_oa',
        help='Path to input dataset (default: full dataset)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='Directory to save preprocessed data'
    )
    parser.add_argument(
        '--n_workers',
        type=int,
        default=32,
        help='Number of parallel workers (default: 32)'
    )
    parser.add_argument(
        '--partition_size',
        type=int,
        default=100,
        help='Samples per partition (default: 100)'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'valid', 'test'],
        help='Splits to process (default: train valid test)'
    )

    args = parser.parse_args()

    # Recommendation based on dataset
    if 'pmc_oa_10k' in args.dataset_path:
        print("\n💡 Tip: Using 10k subset - consider fewer workers (e.g., --n_workers 4)")
    elif 'pmc_oa_100k' in args.dataset_path:
        print("\n💡 Tip: Using 100k subset - good for testing with 8-16 workers")
    else:
        print("\n💡 Tip: Using full dataset - maximize workers for best performance")

    print()
    preprocess_dataset(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        n_workers=args.n_workers,
        partition_size=args.partition_size,
        splits=args.splits
    )
