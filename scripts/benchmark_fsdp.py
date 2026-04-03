"""
FSDP Scaling Benchmark Script

Benchmarks FSDP performance across different GPU configurations (1, 2, 4 GPUs).
Measures:
- Training time
- Memory usage per GPU
- Speedup and scaling efficiency
- Memory reduction vs DDP
- Throughput

Usage:
    python3 scripts/benchmark_fsdp.py
    python3 scripts/benchmark_fsdp.py --dataset_path data/raw/pmc_oa_10k --gpu_configs 1 2
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def run_fsdp_experiment(
    num_gpus: int,
    dataset_path: str,
    output_base_dir: str,
    num_epochs: int = 1,
    batch_size: int = 8
):
    """Run FSDP training experiment with specified number of GPUs

    Args:
        num_gpus: Number of GPUs to use
        dataset_path: Path to dataset
        output_base_dir: Base output directory
        num_epochs: Number of epochs to train
        batch_size: Batch size per GPU

    Returns:
        Dictionary with performance metrics
    """

    output_dir = os.path.join(output_base_dir, f"fsdp_{num_gpus}gpu")

    print(f"\n{'='*80}")
    print(f"RUNNING FSDP EXPERIMENT - {num_gpus} GPU(s)")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Batch size per GPU: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"{'='*80}\n")

    start_time = time.time()

    # Build command
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "--nnodes=1",
        "--node_rank=0",
        "--master_addr=localhost",
        "--master_port=12355",
        "src/training/fsdp_trainer.py",
        "--dataset_path", dataset_path,
        "--output_dir", output_dir,
        "--num_gpus", str(num_gpus),
        "--batch_size", str(batch_size),
        "--num_epochs", str(num_epochs),
        "--learning_rate", "2e-4",
        "--gradient_accumulation_steps", "2",
        "--no_wandb"  # Disable WandB for benchmarking
    ]

    # Run experiment
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        if result.stdout:
            print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"❌ Error running FSDP experiment with {num_gpus} GPU(s)")
        print(f"Error output: {e.stderr}")
        raise

    elapsed_time = time.time() - start_time

    # Load performance metrics
    metrics_file = os.path.join(output_dir, f"fsdp_performance_{num_gpus}gpu.json")

    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    metrics['num_gpus'] = num_gpus
    metrics['elapsed_time'] = elapsed_time
    metrics['method'] = 'FSDP'

    print(f"\n✓ Experiment completed in {elapsed_time:.2f}s")
    print(f"  Peak memory per GPU: {metrics.get('peak_memory_allocated_GB', 0):.2f} GB")
    if 'avg_throughput_samples_per_sec' in metrics:
        print(f"  Throughput: {metrics['avg_throughput_samples_per_sec']:.2f} samples/sec")

    return metrics


def analyze_fsdp_scaling(
    results: list,
    output_dir: str
):
    """Analyze FSDP scaling performance

    Args:
        results: List of benchmark results
        output_dir: Output directory for plots
    """

    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results)

    # Calculate speedup and efficiency
    baseline_time = df[df['num_gpus'] == 1]['elapsed_time'].values[0]
    df['speedup'] = baseline_time / df['elapsed_time']
    df['efficiency'] = (df['speedup'] / df['num_gpus']) * 100
    df['ideal_speedup'] = df['num_gpus']

    print(f"\n{'='*80}")
    print("ANALYZING FSDP SCALING PERFORMANCE")
    print(f"{'='*80}\n")

    # Print scaling table
    print("Scaling Performance:")
    print("-" * 80)
    print(f"{'GPUs':<8} {'Time (s)':<12} {'Speedup':<12} {'Efficiency':<14} {'Ideal':<10}")
    print("-" * 80)

    for _, row in df.iterrows():
        print(f"{row['num_gpus']:<8} {row['elapsed_time']:<12.2f} "
              f"{row['speedup']:<12.2f}x {row['efficiency']:<14.1f}% "
              f"{row['ideal_speedup']:<10.1f}x")

    print("-" * 80)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Training time comparison
    axes[0, 0].bar(df['num_gpus'], df['elapsed_time'], color='coral', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Number of GPUs', fontsize=12)
    axes[0, 0].set_ylabel('Training Time (seconds)', fontsize=12)
    axes[0, 0].set_title('FSDP Training Time vs GPUs', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(df['num_gpus'])
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (gpus, time_val) in enumerate(zip(df['num_gpus'], df['elapsed_time'])):
        axes[0, 0].text(gpus, time_val, f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')

    # 2. Speedup comparison
    axes[0, 1].plot(df['num_gpus'], df['speedup'], marker='o', linewidth=2, markersize=10,
                    label='FSDP Actual', color='coral')
    axes[0, 1].plot(df['num_gpus'], df['ideal_speedup'], linestyle='--', linewidth=2,
                    label='Ideal Linear', color='red')
    axes[0, 1].set_xlabel('Number of GPUs', fontsize=12)
    axes[0, 1].set_ylabel('Speedup', fontsize=12)
    axes[0, 1].set_title('FSDP Speedup vs Ideal', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(df['num_gpus'])

    # Add speedup annotations
    for gpus, speedup in zip(df['num_gpus'], df['speedup']):
        axes[0, 1].annotate(f'{speedup:.2f}x', xy=(gpus, speedup),
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', fontsize=9, fontweight='bold')

    # 3. Scaling efficiency
    axes[1, 0].plot(df['num_gpus'], df['efficiency'], marker='s', linewidth=2,
                    markersize=10, color='green')
    axes[1, 0].axhline(100, linestyle='--', color='red', linewidth=1.5, label='Ideal (100%)')
    axes[1, 0].axhline(80, linestyle=':', color='orange', linewidth=1.5, label='Good (80%)')
    axes[1, 0].set_xlabel('Number of GPUs', fontsize=12)
    axes[1, 0].set_ylabel('Scaling Efficiency (%)', fontsize=12)
    axes[1, 0].set_title('FSDP Scaling Efficiency', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(df['num_gpus'])
    axes[1, 0].set_ylim([0, 110])

    # Add efficiency annotations
    for gpus, eff in zip(df['num_gpus'], df['efficiency']):
        axes[1, 0].annotate(f'{eff:.1f}%', xy=(gpus, eff),
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', fontsize=9, fontweight='bold')

    # 4. Memory usage per GPU
    if 'peak_memory_allocated_GB' in df.columns:
        axes[1, 1].bar(df['num_gpus'], df['peak_memory_allocated_GB'],
                      color='purple', edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Number of GPUs', fontsize=12)
        axes[1, 1].set_ylabel('Peak Memory per GPU (GB)', fontsize=12)
        axes[1, 1].set_title('FSDP Memory Usage per GPU', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(df['num_gpus'])
        axes[1, 1].grid(axis='y', alpha=0.3)

        # Add value labels
        for gpus, mem in zip(df['num_gpus'], df['peak_memory_allocated_GB']):
            axes[1, 1].text(gpus, mem, f'{mem:.2f}GB', ha='center', va='bottom', fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'Memory data not available',
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'fsdp_scaling_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Scaling plot saved: {plot_file}")

    # Save results to CSV
    csv_file = os.path.join(output_dir, 'fsdp_scaling_results.csv')
    df.to_csv(csv_file, index=False)
    print(f"✓ Results saved: {csv_file}")

    # Print summary
    print(f"\n{'='*80}")
    print("FSDP SCALING SUMMARY")
    print(f"{'='*80}\n")

    print(f"Baseline (1 GPU): {baseline_time:.2f}s\n")

    for _, row in df[df['num_gpus'] > 1].iterrows():
        gpus = int(row['num_gpus'])
        speedup = row['speedup']
        efficiency = row['efficiency']
        print(f"{gpus} GPUs: {speedup:.2f}x speedup, {efficiency:.1f}% efficiency")

    # Quality assessment
    max_efficiency = df[df['num_gpus'] > 1]['efficiency'].max()
    if max_efficiency >= 90:
        quality = "Excellent"
    elif max_efficiency >= 80:
        quality = "Good"
    elif max_efficiency >= 70:
        quality = "Fair"
    else:
        quality = "Needs improvement"

    max_gpus = df['num_gpus'].max()
    final_efficiency = df[df['num_gpus'] == max_gpus]['efficiency'].values[0]

    print(f"\nScaling Quality: {quality} ({final_efficiency:.1f}% efficiency at {max_gpus} GPUs)")
    print(f"{'='*80}\n")


def main():
    """Main benchmarking function"""

    parser = argparse.ArgumentParser(description='Benchmark FSDP scaling')
    parser.add_argument('--dataset_path', type=str, default='data/raw/pmc_oa_100k',
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/benchmarks/fsdp',
                       help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=1,
                       help='Number of epochs for benchmark')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size per GPU')
    parser.add_argument('--gpu_configs', type=int, nargs='+', default=[1, 2, 4],
                       help='GPU configurations to test')

    args = parser.parse_args()

    # Validate dataset
    if not os.path.exists(args.dataset_path):
        print(f"❌ Error: Dataset not found at {args.dataset_path}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("FSDP SCALING BENCHMARK")
    print("="*80)
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"GPU configurations: {args.gpu_configs}")
    print("="*80)

    # Run experiments
    results = []

    for num_gpus in sorted(args.gpu_configs):
        try:
            metrics = run_fsdp_experiment(
                num_gpus=num_gpus,
                dataset_path=args.dataset_path,
                output_base_dir=args.output_dir,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size
            )
            results.append(metrics)

        except Exception as e:
            print(f"❌ Failed to run experiment with {num_gpus} GPU(s): {e}")
            continue

    if not results:
        print("❌ No experiments completed successfully")
        sys.exit(1)

    # Analyze results
    analyze_fsdp_scaling(results, args.output_dir)

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
