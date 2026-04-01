"""
Benchmark preprocessing performance: Sequential vs Parallel
Tests different numbers of workers: 1, 4, 8, 16
Generates comprehensive performance visualizations
"""

import os
import sys
import json
import time
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import ImagePreprocessor, ParallelImagePreprocessor


def benchmark_preprocessing(
    dataset_path="data/raw/pmc_oa_10k",
    output_dir="outputs/preprocessing_benchmark",
    max_samples=5000,
    worker_counts=[4, 8, 16]
):
    """Run preprocessing benchmarks"""

    print("="*70)
    print("PREPROCESSING BENCHMARK")
    print("="*70)

    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print(f"\n[1/4] Loading dataset from: {dataset_path}")
    try:
        dataset = load_from_disk(dataset_path)
        print(f"  ✓ Dataset loaded")
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        print(f"  Please ensure dataset exists at: {dataset_path}")
        print(f"  Run: python scripts/create_subsets.py")
        return

    # Prepare samples
    num_samples = min(max_samples, len(dataset['train']))
    print(f"\n[2/4] Preparing {num_samples} samples for benchmark...")

    samples = dataset['train'].select(range(num_samples))
    samples = [{'image': s['image'], 'caption': s['caption'], 'image_id': i}
               for i, s in enumerate(samples)]

    print(f"  ✓ Samples prepared")

    results = []

    # 1. Sequential baseline
    print("\n[3/4] Running benchmarks...")
    print("\n" + "-"*70)
    print("SEQUENTIAL BASELINE (1 worker)")
    print("-"*70)

    preprocessor = ImagePreprocessor()
    try:
        metrics = preprocessor.preprocess_batch_sequential(
            samples, f"{output_dir}/sequential", "train"
        )
        metrics['method'] = 'Sequential'
        metrics['workers'] = 1
        results.append(metrics)
        print(f"✓ Time: {metrics['elapsed_time']:.2f}s")
        print(f"  Throughput: {metrics['throughput']:.2f} images/sec")
    except Exception as e:
        print(f"✗ Sequential benchmark failed: {e}")
        return

    # 2. Parallel with different worker counts
    for n_workers in worker_counts:
        print("\n" + "-"*70)
        print(f"PARALLEL PROCESSING ({n_workers} workers)")
        print("-"*70)

        parallel_preprocessor = ParallelImagePreprocessor(
            n_workers=n_workers,
            threads_per_worker=2
        )

        try:
            metrics = parallel_preprocessor.preprocess_batch_parallel(
                samples, f"{output_dir}/parallel_{n_workers}", "train"
            )
            parallel_preprocessor.close_cluster()

            metrics['method'] = f'Parallel-{n_workers}w'
            metrics['workers'] = n_workers
            results.append(metrics)

            print(f"✓ Time: {metrics['elapsed_time']:.2f}s")
            print(f"  Throughput: {metrics['throughput']:.2f} images/sec")

        except Exception as e:
            print(f"✗ Parallel benchmark ({n_workers} workers) failed: {e}")
            parallel_preprocessor.close_cluster()
            continue

    # Calculate speedups
    baseline_time = results[0]['elapsed_time']
    baseline_throughput = results[0]['throughput']

    for r in results:
        r['speedup'] = baseline_time / r['elapsed_time']
        r['efficiency'] = (r['speedup'] / r['workers']) * 100  # As percentage
        r['throughput_improvement'] = (r['throughput'] / baseline_throughput - 1) * 100

    # Save results
    print("\n[4/4] Saving results...")
    results_file = f"{output_dir}/benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Results saved: {results_file}")

    # Visualize results
    plot_benchmark_results(results, output_dir)

    # Print summary
    print_summary(results)

    return results


def plot_benchmark_results(results, output_dir):
    """Generate comprehensive benchmark visualization plots"""

    import pandas as pd
    df = pd.DataFrame(results)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Throughput comparison
    ax = axes[0, 0]
    colors = ['steelblue'] + ['forestgreen'] * (len(df) - 1)
    bars = ax.bar(df['method'], df['throughput'], color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Throughput (images/sec)', fontsize=11, fontweight='bold')
    ax.set_title('Preprocessing Throughput', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)

    # 2. Speedup vs workers
    ax = axes[0, 1]
    ax.plot(df['workers'], df['speedup'], marker='o', linewidth=2.5,
            markersize=10, color='forestgreen', label='Actual')
    ax.plot(df['workers'], df['workers'], '--', label='Ideal Linear',
            color='red', linewidth=2)
    ax.set_xlabel('Number of Workers', fontsize=11, fontweight='bold')
    ax.set_ylabel('Speedup', fontsize=11, fontweight='bold')
    ax.set_title('Speedup vs Number of Workers', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add value labels
    for i, (w, s) in enumerate(zip(df['workers'], df['speedup'])):
        ax.text(w, s + 0.3, f'{s:.2f}x', ha='center', fontsize=9)

    # 3. Scaling efficiency
    ax = axes[0, 2]
    ax.plot(df['workers'], df['efficiency'], marker='s',
            linewidth=2.5, markersize=10, color='green')
    ax.set_xlabel('Number of Workers', fontsize=11, fontweight='bold')
    ax.set_ylabel('Efficiency (%)', fontsize=11, fontweight='bold')
    ax.set_title('Parallel Scaling Efficiency', fontsize=12, fontweight='bold')
    ax.axhline(100, linestyle='--', color='red', label='Ideal (100%)', linewidth=2)
    ax.axhline(70, linestyle=':', color='orange', label='Target (70%)', linewidth=1.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add value labels
    for i, (w, e) in enumerate(zip(df['workers'], df['efficiency'])):
        ax.text(w, e + 2, f'{e:.1f}%', ha='center', fontsize=9)

    # 4. Processing time
    ax = axes[1, 0]
    colors = ['coral'] + ['lightcoral'] * (len(df) - 1)
    bars = ax.bar(df['method'], df['elapsed_time'], color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('Total Processing Time', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom', fontsize=9)

    # 5. Throughput improvement
    ax = axes[1, 1]
    improvement = df['throughput_improvement'].tolist()
    colors = ['gray'] + ['dodgerblue'] * (len(improvement) - 1)
    bars = ax.bar(df['method'], improvement, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Throughput Improvement (%)', fontsize=11, fontweight='bold')
    ax.set_title('Throughput Improvement over Sequential', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

    # 6. Summary table
    ax = axes[1, 2]
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row['method'],
            f"{row['workers']}",
            f"{row['throughput']:.1f}",
            f"{row['speedup']:.2f}x",
            f"{row['efficiency']:.1f}%"
        ])

    table = ax.table(cellText=table_data,
                     colLabels=['Method', 'Workers', 'Throughput\n(img/s)', 'Speedup', 'Efficiency'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style cells
    for i in range(1, len(table_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)

    plt.suptitle('Parallel Preprocessing Performance Benchmark',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_file = f"{output_dir}/preprocessing_benchmark.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Plot saved: {output_file}")
    plt.close()


def print_summary(results):
    """Print benchmark summary"""

    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)

    baseline = results[0]
    print(f"\nBaseline (Sequential):")
    print(f"  Time: {baseline['elapsed_time']:.2f}s")
    print(f"  Throughput: {baseline['throughput']:.2f} images/sec")

    if len(results) > 1:
        best = max(results[1:], key=lambda x: x['speedup'])
        print(f"\nBest Parallel Performance ({best['workers']} workers):")
        print(f"  Time: {best['elapsed_time']:.2f}s")
        print(f"  Throughput: {best['throughput']:.2f} images/sec")
        print(f"  Speedup: {best['speedup']:.2f}x")
        print(f"  Efficiency: {best['efficiency']:.1f}%")

        print(f"\nAll Results:")
        print(f"  {'Method':<20} {'Workers':<10} {'Time (s)':<12} {'Throughput':<15} {'Speedup':<12} {'Efficiency'}")
        print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*15} {'-'*12} {'-'*10}")

        for r in results:
            print(f"  {r['method']:<20} {r['workers']:<10} {r['elapsed_time']:<12.2f} "
                  f"{r['throughput']:<15.2f} {r['speedup']:<12.2f}x {r['efficiency']:<10.1f}%")

    print("\n" + "="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark preprocessing performance')
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='data/raw/pmc_oa_10k',
        help='Path to dataset (default: 10k subset)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/preprocessing_benchmark',
        help='Directory to save results'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=5000,
        help='Maximum number of samples to process'
    )
    parser.add_argument(
        '--workers',
        type=int,
        nargs='+',
        default=[4, 8, 16],
        help='Worker counts to test (e.g., --workers 4 8 16 32)'
    )

    args = parser.parse_args()

    results = benchmark_preprocessing(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        worker_counts=args.workers
    )

    if results:
        print("\n✅ Benchmark complete!")
        print(f"\nResults saved to: {args.output_dir}/")
        print("  - benchmark_results.json")
        print("  - preprocessing_benchmark.png")
