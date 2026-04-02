"""
Benchmark DDP scaling across 1, 2, 4 GPUs
Measures speedup, scaling efficiency, and communication overhead

This script runs DDP training experiments with different numbers of GPUs
and analyzes the scaling performance. It generates detailed metrics and
visualizations to help understand DDP efficiency.
"""

import json
import subprocess
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import os
import sys

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def run_ddp_experiment(num_gpus, dataset_path, output_dir, num_epochs=1, batch_size=4):
    """Run DDP training experiment

    Args:
        num_gpus: Number of GPUs to use
        dataset_path: Path to dataset
        output_dir: Output directory
        num_epochs: Number of epochs (default: 1 for benchmarking)
        batch_size: Batch size per GPU

    Returns:
        Dictionary of performance metrics
    """

    print(f"\n{'='*80}")
    print(f"RUNNING DDP EXPERIMENT - {num_gpus} GPU(s)")
    print(f"{'='*80}\n")

    experiment_dir = f"{output_dir}/ddp_{num_gpus}gpu"
    os.makedirs(experiment_dir, exist_ok=True)

    start_time = time.time()

    # Build command
    if num_gpus == 1:
        # Use single-GPU trainer for baseline
        cmd = [
            "python3", "-m", "src.training.single_gpu_trainer",
            "--dataset_path", dataset_path,
            "--output_dir", experiment_dir,
            "--batch_size", str(batch_size),
            "--num_epochs", str(num_epochs),
            "--bf16",
            "--no_wandb",  # Disable WandB for benchmarking
            "--logging_steps", "50",
            "--eval_steps", "1000",  # Less frequent eval for benchmarking
            "--save_steps", "10000"   # Don't save during benchmark
        ]
    else:
        # Use DDP trainer
        cmd = [
            "python3", "-m", "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            "--nnodes=1",
            "--node_rank=0",
            "--master_addr=localhost",
            f"--master_port={12350 + num_gpus}",  # Different port for each experiment
            "src/training/ddp_trainer.py",
            "--dataset_path", dataset_path,
            "--output_dir", experiment_dir,
            "--num_gpus", str(num_gpus),
            "--batch_size", str(batch_size),
            "--num_epochs", str(num_epochs),
            "--bf16",
            "--no_wandb",
            "--gradient_accumulation_steps", "4"
        ]

    print(f"Command: {' '.join(cmd)}")
    print("")

    # Run training
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        print("STDOUT:")
        print(result.stdout[-1000:])  # Last 1000 chars

        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr[-500:])  # Last 500 chars

    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        print(f"STDOUT: {e.stdout[-1000:]}")
        print(f"STDERR: {e.stderr[-500:]}")
        return None

    elapsed_time = time.time() - start_time

    # Load performance metrics
    if num_gpus == 1:
        # Single-GPU trainer doesn't save detailed metrics, create basic ones
        metrics = {
            'num_gpus': 1,
            'elapsed_time': elapsed_time,
            'batch_size_per_gpu': batch_size,
            'total_batch_size': batch_size
        }
    else:
        metrics_file = f"{experiment_dir}/ddp_performance_{num_gpus}gpu.json"
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                metrics = json.load(f)
            metrics['elapsed_time'] = elapsed_time
        else:
            print(f"Warning: Metrics file not found: {metrics_file}")
            metrics = {
                'num_gpus': num_gpus,
                'elapsed_time': elapsed_time,
                'batch_size_per_gpu': batch_size,
                'total_batch_size': batch_size * num_gpus
            }

    print(f"\n✓ Experiment completed in {elapsed_time:.2f}s")

    return metrics


def analyze_scaling(results, output_dir):
    """Analyze DDP scaling performance

    Calculates speedup, efficiency, and generates visualizations.

    Args:
        results: List of performance metrics dictionaries
        output_dir: Directory to save analysis results
    """

    print(f"\n{'='*80}")
    print("ANALYZING SCALING PERFORMANCE")
    print(f"{'='*80}\n")

    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('num_gpus')

    # Calculate speedup and efficiency
    baseline_time = df[df['num_gpus'] == 1]['elapsed_time'].values[0]

    df['speedup'] = baseline_time / df['elapsed_time']
    df['efficiency'] = (df['speedup'] / df['num_gpus']) * 100
    df['ideal_speedup'] = df['num_gpus']

    # Print results table
    print("\nScaling Performance:")
    print("-" * 80)
    print(f"{'GPUs':<6} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<12} {'Ideal':<10}")
    print("-" * 80)

    for _, row in df.iterrows():
        print(f"{row['num_gpus']:<6} {row['elapsed_time']:<12.2f} "
              f"{row['speedup']:<10.2f}x {row['efficiency']:<12.1f}% "
              f"{row['ideal_speedup']:<10.1f}x")

    print("-" * 80)

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Training time
    axes[0, 0].bar(df['num_gpus'], df['elapsed_time'], color='steelblue', edgecolor='black', width=0.6)
    axes[0, 0].set_xlabel('Number of GPUs', fontsize=12)
    axes[0, 0].set_ylabel('Training Time (seconds)', fontsize=12)
    axes[0, 0].set_title('DDP Training Time vs GPUs', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(df['num_gpus'])
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (gpus, t) in enumerate(zip(df['num_gpus'], df['elapsed_time'])):
        axes[0, 0].text(gpus, t + max(df['elapsed_time']) * 0.02, f'{t:.1f}s',
                       ha='center', va='bottom', fontsize=10)

    # 2. Speedup
    axes[0, 1].plot(df['num_gpus'], df['speedup'], marker='o', linewidth=2.5,
                   markersize=10, label='Actual Speedup', color='green')
    axes[0, 1].plot(df['num_gpus'], df['ideal_speedup'], '--', linewidth=2,
                   label='Ideal Linear Speedup', color='red')
    axes[0, 1].set_xlabel('Number of GPUs', fontsize=12)
    axes[0, 1].set_ylabel('Speedup', fontsize=12)
    axes[0, 1].set_title('DDP Speedup Analysis', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(df['num_gpus'])

    # Add speedup values
    for gpus, speedup in zip(df['num_gpus'], df['speedup']):
        axes[0, 1].annotate(f'{speedup:.2f}x', (gpus, speedup),
                           textcoords="offset points", xytext=(0,10),
                           ha='center', fontsize=9)

    # 3. Scaling efficiency
    axes[1, 0].plot(df['num_gpus'], df['efficiency'], marker='s',
                   linewidth=2.5, markersize=10, color='purple')
    axes[1, 0].axhline(100, linestyle='--', color='red', linewidth=2,
                      label='Ideal (100%)', alpha=0.7)
    axes[1, 0].axhline(80, linestyle=':', color='orange', linewidth=1.5,
                      label='Good (80%)', alpha=0.7)
    axes[1, 0].set_xlabel('Number of GPUs', fontsize=12)
    axes[1, 0].set_ylabel('Scaling Efficiency (%)', fontsize=12)
    axes[1, 0].set_title('DDP Scaling Efficiency', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(df['num_gpus'])
    axes[1, 0].set_ylim(0, 110)

    # Add efficiency values
    for gpus, eff in zip(df['num_gpus'], df['efficiency']):
        axes[1, 0].annotate(f'{eff:.1f}%', (gpus, eff),
                           textcoords="offset points", xytext=(0,5),
                           ha='center', fontsize=9)

    # 4. Time breakdown (if available)
    time_components = ['avg_data_loading', 'avg_forward', 'avg_backward', 'avg_optimizer_step']
    has_breakdown = all(comp in df.columns for comp in time_components)

    if has_breakdown:
        # Stack components
        bottom = np.zeros(len(df))
        colors = ['#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD']
        labels = ['Data Loading', 'Forward Pass', 'Backward Pass', 'Optimizer Step']

        for comp, color, label in zip(time_components, colors, labels):
            axes[1, 1].bar(df['num_gpus'], df[comp], bottom=bottom,
                          label=label, color=color, edgecolor='black', width=0.6)
            bottom += df[comp].values

        axes[1, 1].set_xlabel('Number of GPUs', fontsize=12)
        axes[1, 1].set_ylabel('Time per Step (seconds)', fontsize=12)
        axes[1, 1].set_title('Time Breakdown per Training Step', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].set_xticks(df['num_gpus'])
    else:
        # Show throughput instead
        if 'elapsed_time' in df.columns:
            throughput = df['total_batch_size'] / (df['elapsed_time'] / df.get('num_batches', 100))
            axes[1, 1].bar(df['num_gpus'], throughput, color='coral', edgecolor='black', width=0.6)
            axes[1, 1].set_xlabel('Number of GPUs', fontsize=12)
            axes[1, 1].set_ylabel('Throughput (samples/sec)', fontsize=12)
            axes[1, 1].set_title('Training Throughput', fontsize=14, fontweight='bold')
            axes[1, 1].set_xticks(df['num_gpus'])
            axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_file = f"{output_dir}/ddp_scaling_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Scaling plot saved: {plot_file}")

    # Save results to CSV
    csv_file = f"{output_dir}/ddp_scaling_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"✓ Results saved: {csv_file}")

    # Print summary
    print(f"\n{'='*80}")
    print("DDP SCALING SUMMARY")
    print(f"{'='*80}")
    print(f"\nBaseline (1 GPU): {baseline_time:.2f}s")
    for _, row in df[df['num_gpus'] > 1].iterrows():
        print(f"{int(row['num_gpus'])} GPUs: {row['speedup']:.2f}x speedup, "
              f"{row['efficiency']:.1f}% efficiency")

    # Scaling quality assessment
    max_efficiency = df[df['num_gpus'] == df['num_gpus'].max()]['efficiency'].values[0]
    if max_efficiency >= 85:
        quality = "Excellent"
    elif max_efficiency >= 70:
        quality = "Good"
    elif max_efficiency >= 50:
        quality = "Fair"
    else:
        quality = "Poor"

    print(f"\nScaling Quality: {quality} ({max_efficiency:.1f}% efficiency at {df['num_gpus'].max()} GPUs)")
    print(f"{'='*80}\n")

    return df


def main():
    """Main benchmarking function"""

    parser = argparse.ArgumentParser(description="Benchmark DDP scaling")
    parser.add_argument("--dataset_path", type=str, default="data/raw/pmc_oa_10k",
                       help="Path to dataset (use small dataset for faster benchmarking)")
    parser.add_argument("--output_dir", type=str, default="outputs/benchmarks/ddp",
                       help="Output directory for results")
    parser.add_argument("--num_epochs", type=int, default=1,
                       help="Number of epochs (1 for quick benchmark)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size per GPU")
    parser.add_argument("--gpu_configs", type=int, nargs='+', default=[1, 2, 4],
                       help="GPU configurations to test (e.g., 1 2 4)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("DDP SCALING BENCHMARK")
    print("="*80)
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"GPU configs: {args.gpu_configs}")
    print("="*80)

    results = []

    # Run experiments
    for num_gpus in args.gpu_configs:
        metrics = run_ddp_experiment(
            num_gpus=num_gpus,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size
        )

        if metrics:
            results.append(metrics)
        else:
            print(f"Warning: Experiment with {num_gpus} GPUs failed")

    if len(results) < 2:
        print("Error: Need at least 2 successful experiments for scaling analysis")
        sys.exit(1)

    # Analyze results
    analyze_scaling(results, args.output_dir)

    print("\n✅ Benchmark complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
