"""
DDP vs FSDP Comparison Analysis

Compares performance between Distributed Data Parallel (DDP) and
Fully Sharded Data Parallel (FSDP) training strategies.

Analyzes:
- Training speed and throughput
- Memory usage per GPU
- Scaling efficiency
- When to use each method

Usage:
    python3 scripts/compare_ddp_fsdp.py
    python3 scripts/compare_ddp_fsdp.py --output_dir outputs/analysis
"""

import os
import sys
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


def load_metrics(method: str, num_gpus: int, base_dir: str = "outputs/benchmarks"):
    """Load performance metrics from benchmark results

    Args:
        method: 'ddp' or 'fsdp'
        num_gpus: Number of GPUs
        base_dir: Base directory for benchmarks

    Returns:
        Dictionary with performance metrics
    """

    metrics_file = os.path.join(
        base_dir,
        method,
        f"{method}_{num_gpus}gpu",
        f"{method}_performance_{num_gpus}gpu.json"
    )

    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Metrics not found: {metrics_file}")

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    metrics['method'] = method.upper()
    metrics['num_gpus'] = num_gpus

    return metrics


def create_comparison_plots(ddp_df: pd.DataFrame, fsdp_df: pd.DataFrame, output_dir: str):
    """Create comprehensive comparison visualizations

    Args:
        ddp_df: DataFrame with DDP results
        fsdp_df: DataFrame with FSDP results
        output_dir: Output directory for plots
    """

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    # Extract GPU counts
    gpu_counts = sorted(ddp_df['num_gpus'].unique())
    x_pos = np.arange(len(gpu_counts))
    width = 0.35

    # 1. Training Time Comparison
    ddp_times = [ddp_df[ddp_df['num_gpus'] == g]['total_time'].values[0] for g in gpu_counts]
    fsdp_times = [fsdp_df[fsdp_df['num_gpus'] == g]['total_time'].values[0] for g in gpu_counts]

    axes[0, 0].bar(x_pos - width/2, ddp_times, width, label='DDP', color='steelblue', edgecolor='black')
    axes[0, 0].bar(x_pos + width/2, fsdp_times, width, label='FSDP', color='coral', edgecolor='black')
    axes[0, 0].set_xlabel('Number of GPUs', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Training Time: DDP vs FSDP', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(gpu_counts)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (ddp_t, fsdp_t) in enumerate(zip(ddp_times, fsdp_times)):
        axes[0, 0].text(i - width/2, ddp_t, f'{ddp_t:.1f}s', ha='center', va='bottom', fontsize=9)
        axes[0, 0].text(i + width/2, fsdp_t, f'{fsdp_t:.1f}s', ha='center', va='bottom', fontsize=9)

    # 2. Memory Usage Comparison
    if 'peak_memory_allocated_GB' in ddp_df.columns and 'peak_memory_allocated_GB' in fsdp_df.columns:
        ddp_mem = [ddp_df[ddp_df['num_gpus'] == g]['peak_memory_allocated_GB'].values[0] for g in gpu_counts]
        fsdp_mem = [fsdp_df[fsdp_df['num_gpus'] == g]['peak_memory_allocated_GB'].values[0] for g in gpu_counts]

        axes[0, 1].bar(x_pos - width/2, ddp_mem, width, label='DDP', color='steelblue', edgecolor='black')
        axes[0, 1].bar(x_pos + width/2, fsdp_mem, width, label='FSDP', color='coral', edgecolor='black')
        axes[0, 1].set_xlabel('Number of GPUs', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Peak Memory per GPU (GB)', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Memory Usage: DDP vs FSDP', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(gpu_counts)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(axis='y', alpha=0.3)

        # Add value labels and savings
        for i, (ddp_m, fsdp_m) in enumerate(zip(ddp_mem, fsdp_mem)):
            axes[0, 1].text(i - width/2, ddp_m, f'{ddp_m:.2f}GB', ha='center', va='bottom', fontsize=9)
            axes[0, 1].text(i + width/2, fsdp_m, f'{fsdp_m:.2f}GB', ha='center', va='bottom', fontsize=9)
            savings = ((ddp_m - fsdp_m) / ddp_m) * 100
            axes[0, 1].text(i, max(ddp_m, fsdp_m) * 1.1, f'-{savings:.1f}%',
                          ha='center', va='bottom', fontsize=10, fontweight='bold', color='green')

    # 3. Speedup Comparison
    ddp_baseline = ddp_df[ddp_df['num_gpus'] == 1]['total_time'].values[0]
    fsdp_baseline = fsdp_df[fsdp_df['num_gpus'] == 1]['total_time'].values[0]

    ddp_speedup = [ddp_baseline / ddp_df[ddp_df['num_gpus'] == g]['total_time'].values[0] for g in gpu_counts]
    fsdp_speedup = [fsdp_baseline / fsdp_df[fsdp_df['num_gpus'] == g]['total_time'].values[0] for g in gpu_counts]

    axes[1, 0].plot(gpu_counts, ddp_speedup, marker='o', linewidth=2.5, markersize=10,
                    label='DDP', color='steelblue')
    axes[1, 0].plot(gpu_counts, fsdp_speedup, marker='s', linewidth=2.5, markersize=10,
                    label='FSDP', color='coral')
    axes[1, 0].plot(gpu_counts, gpu_counts, '--', linewidth=2, label='Ideal Linear', color='red')
    axes[1, 0].set_xlabel('Number of GPUs', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Speedup', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Speedup Comparison: DDP vs FSDP', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(gpu_counts)

    # Add speedup annotations
    for i, (ddp_s, fsdp_s) in enumerate(zip(ddp_speedup, fsdp_speedup)):
        axes[1, 0].annotate(f'{ddp_s:.2f}x', xy=(gpu_counts[i], ddp_s),
                          xytext=(0, 10), textcoords='offset points',
                          ha='center', fontsize=9, color='steelblue', fontweight='bold')
        axes[1, 0].annotate(f'{fsdp_s:.2f}x', xy=(gpu_counts[i], fsdp_s),
                          xytext=(0, -15), textcoords='offset points',
                          ha='center', fontsize=9, color='coral', fontweight='bold')

    # 4. Scaling Efficiency Comparison
    ddp_efficiency = [(ddp_baseline / ddp_df[ddp_df['num_gpus'] == g]['total_time'].values[0] / g) * 100
                      for g in gpu_counts]
    fsdp_efficiency = [(fsdp_baseline / fsdp_df[fsdp_df['num_gpus'] == g]['total_time'].values[0] / g) * 100
                       for g in gpu_counts]

    axes[1, 1].plot(gpu_counts, ddp_efficiency, marker='o', linewidth=2.5, markersize=10,
                    label='DDP', color='steelblue')
    axes[1, 1].plot(gpu_counts, fsdp_efficiency, marker='s', linewidth=2.5, markersize=10,
                    label='FSDP', color='coral')
    axes[1, 1].axhline(100, linestyle='--', color='red', linewidth=1.5, label='Ideal (100%)')
    axes[1, 1].axhline(80, linestyle=':', color='orange', linewidth=1.5, label='Good (80%)')
    axes[1, 1].set_xlabel('Number of GPUs', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Scaling Efficiency (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Scaling Efficiency: DDP vs FSDP', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(gpu_counts)
    axes[1, 1].set_ylim([0, 110])

    # 5. Time Breakdown (4 GPUs)
    if 4 in gpu_counts:
        ddp_4gpu = ddp_df[ddp_df['num_gpus'] == 4].iloc[0]
        fsdp_4gpu = fsdp_df[fsdp_df['num_gpus'] == 4].iloc[0]

        components = ['avg_forward', 'avg_backward', 'avg_optimizer_step']
        component_labels = ['Forward', 'Backward', 'Optimizer']

        ddp_times = [ddp_4gpu.get(c, 0) for c in components]
        fsdp_times = [fsdp_4gpu.get(c, 0) for c in components]

        x_comp = np.arange(len(components))
        axes[2, 0].bar(x_comp - width/2, ddp_times, width, label='DDP', color='steelblue', edgecolor='black')
        axes[2, 0].bar(x_comp + width/2, fsdp_times, width, label='FSDP', color='coral', edgecolor='black')
        axes[2, 0].set_xlabel('Component', fontsize=12, fontweight='bold')
        axes[2, 0].set_ylabel('Time per Step (seconds)', fontsize=12, fontweight='bold')
        axes[2, 0].set_title('Time Breakdown (4 GPUs)', fontsize=14, fontweight='bold')
        axes[2, 0].set_xticks(x_comp)
        axes[2, 0].set_xticklabels(component_labels)
        axes[2, 0].legend(fontsize=11)
        axes[2, 0].grid(axis='y', alpha=0.3)

    # 6. Memory Savings Summary
    if 'peak_memory_allocated_GB' in ddp_df.columns and 'peak_memory_allocated_GB' in fsdp_df.columns:
        memory_savings = []
        for g in gpu_counts:
            ddp_m = ddp_df[ddp_df['num_gpus'] == g]['peak_memory_allocated_GB'].values[0]
            fsdp_m = fsdp_df[fsdp_df['num_gpus'] == g]['peak_memory_allocated_GB'].values[0]
            savings = ((ddp_m - fsdp_m) / ddp_m) * 100
            memory_savings.append(savings)

        axes[2, 1].bar(x_pos, memory_savings, color='green', edgecolor='black', alpha=0.7)
        axes[2, 1].set_xlabel('Number of GPUs', fontsize=12, fontweight='bold')
        axes[2, 1].set_ylabel('Memory Savings (%)', fontsize=12, fontweight='bold')
        axes[2, 1].set_title('FSDP Memory Savings vs DDP', fontsize=14, fontweight='bold')
        axes[2, 1].set_xticks(x_pos)
        axes[2, 1].set_xticklabels(gpu_counts)
        axes[2, 1].grid(axis='y', alpha=0.3)

        # Add value labels
        for i, saving in enumerate(memory_savings):
            axes[2, 1].text(i, saving, f'{saving:.1f}%', ha='center', va='bottom',
                          fontsize=10, fontweight='bold')

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'ddp_vs_fsdp_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved: {plot_file}")


def print_comparison_summary(ddp_df: pd.DataFrame, fsdp_df: pd.DataFrame):
    """Print detailed comparison summary

    Args:
        ddp_df: DataFrame with DDP results
        fsdp_df: DataFrame with FSDP results
    """

    print("\n" + "="*80)
    print("DDP vs FSDP COMPARISON SUMMARY")
    print("="*80)

    gpu_counts = sorted(ddp_df['num_gpus'].unique())

    # Training time comparison
    print("\n1. Training Time (seconds):")
    print("-" * 80)
    print(f"{'GPUs':<8} {'DDP':<15} {'FSDP':<15} {'Winner':<15} {'Difference':<15}")
    print("-" * 80)

    for g in gpu_counts:
        ddp_time = ddp_df[ddp_df['num_gpus'] == g]['total_time'].values[0]
        fsdp_time = fsdp_df[fsdp_df['num_gpus'] == g]['total_time'].values[0]
        winner = "DDP" if ddp_time < fsdp_time else "FSDP"
        diff = abs(ddp_time - fsdp_time)
        diff_pct = (diff / max(ddp_time, fsdp_time)) * 100

        print(f"{g:<8} {ddp_time:<15.2f} {fsdp_time:<15.2f} {winner:<15} {diff_pct:<15.1f}%")

    # Memory comparison
    if 'peak_memory_allocated_GB' in ddp_df.columns and 'peak_memory_allocated_GB' in fsdp_df.columns:
        print("\n2. Peak Memory per GPU (GB):")
        print("-" * 80)
        print(f"{'GPUs':<8} {'DDP':<15} {'FSDP':<15} {'FSDP Savings':<15}")
        print("-" * 80)

        for g in gpu_counts:
            ddp_mem = ddp_df[ddp_df['num_gpus'] == g]['peak_memory_allocated_GB'].values[0]
            fsdp_mem = fsdp_df[fsdp_df['num_gpus'] == g]['peak_memory_allocated_GB'].values[0]
            savings = ((ddp_mem - fsdp_mem) / ddp_mem) * 100

            print(f"{g:<8} {ddp_mem:<15.2f} {fsdp_mem:<15.2f} {savings:<15.1f}%")

    # Speedup comparison
    ddp_baseline = ddp_df[ddp_df['num_gpus'] == 1]['total_time'].values[0]
    fsdp_baseline = fsdp_df[fsdp_df['num_gpus'] == 1]['total_time'].values[0]

    print("\n3. Speedup vs Single GPU:")
    print("-" * 80)
    print(f"{'GPUs':<8} {'DDP Speedup':<15} {'FSDP Speedup':<15} {'Better':<15}")
    print("-" * 80)

    for g in gpu_counts:
        ddp_time = ddp_df[ddp_df['num_gpus'] == g]['total_time'].values[0]
        fsdp_time = fsdp_df[fsdp_df['num_gpus'] == g]['total_time'].values[0]
        ddp_speedup = ddp_baseline / ddp_time
        fsdp_speedup = fsdp_baseline / fsdp_time
        better = "DDP" if ddp_speedup > fsdp_speedup else "FSDP"

        print(f"{g:<8} {ddp_speedup:<15.2f}x {fsdp_speedup:<15.2f}x {better:<15}")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    print("\nUse DDP when:")
    print("  • Speed is critical (DDP is typically slightly faster)")
    print("  • Model fits comfortably in GPU memory")
    print("  • Each GPU has sufficient memory for full model copy")
    print("  • Communication overhead is acceptable")

    print("\nUse FSDP when:")
    print("  • Memory is constrained (40-60% memory savings)")
    print("  • Training larger models or batch sizes")
    print("  • Scaling to many GPUs (8+)")
    print("  • Need CPU offloading capability")
    print("  • Trading slight speed for memory efficiency")

    # Calculate best approach for 4 GPUs
    if 4 in gpu_counts:
        ddp_4 = ddp_df[ddp_df['num_gpus'] == 4].iloc[0]
        fsdp_4 = fsdp_df[fsdp_df['num_gpus'] == 4].iloc[0]

        print("\n" + "="*80)
        print("FOR THIS PROJECT (4 GPUs):")
        print("="*80)

        ddp_time = ddp_4['total_time']
        fsdp_time = fsdp_4['total_time']

        if 'peak_memory_allocated_GB' in ddp_4 and 'peak_memory_allocated_GB' in fsdp_4:
            ddp_mem = ddp_4['peak_memory_allocated_GB']
            fsdp_mem = fsdp_4['peak_memory_allocated_GB']
            mem_savings = ((ddp_mem - fsdp_mem) / ddp_mem) * 100

            print(f"\nDDP:  {ddp_time:.2f}s, {ddp_mem:.2f} GB/GPU")
            print(f"FSDP: {fsdp_time:.2f}s, {fsdp_mem:.2f} GB/GPU ({mem_savings:.1f}% memory savings)")

            if fsdp_time < ddp_time * 1.1 and mem_savings > 30:
                print(f"\n✓ RECOMMENDED: FSDP (similar speed, {mem_savings:.1f}% memory savings)")
            elif ddp_time < fsdp_time * 0.9:
                print(f"\n✓ RECOMMENDED: DDP ({((fsdp_time - ddp_time) / fsdp_time * 100):.1f}% faster)")
            else:
                print("\n✓ RECOMMENDED: Either approach works well for this use case")

    print("\n" + "="*80)


def main():
    """Main comparison function"""

    parser = argparse.ArgumentParser(description='Compare DDP and FSDP performance')
    parser.add_argument('--benchmarks_dir', type=str, default='outputs/benchmarks',
                       help='Base directory containing benchmark results')
    parser.add_argument('--output_dir', type=str, default='outputs/analysis',
                       help='Output directory for analysis')
    parser.add_argument('--gpu_configs', type=int, nargs='+', default=[1, 2, 4],
                       help='GPU configurations to compare')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("DDP vs FSDP COMPARISON")
    print("="*80)
    print(f"Benchmarks directory: {args.benchmarks_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"GPU configurations: {args.gpu_configs}")
    print("="*80)

    # Load metrics
    ddp_results = []
    fsdp_results = []

    for num_gpus in args.gpu_configs:
        try:
            ddp_metrics = load_metrics('ddp', num_gpus, args.benchmarks_dir)
            ddp_results.append(ddp_metrics)
            print(f"✓ Loaded DDP {num_gpus}-GPU metrics")
        except FileNotFoundError as e:
            print(f"⚠ Warning: {e}")

        try:
            fsdp_metrics = load_metrics('fsdp', num_gpus, args.benchmarks_dir)
            fsdp_results.append(fsdp_metrics)
            print(f"✓ Loaded FSDP {num_gpus}-GPU metrics")
        except FileNotFoundError as e:
            print(f"⚠ Warning: {e}")

    if not ddp_results or not fsdp_results:
        print("\n❌ Error: Insufficient data for comparison")
        print("   Please run benchmarks first:")
        print("     python3 scripts/benchmark_ddp.py")
        print("     python3 scripts/benchmark_fsdp.py")
        sys.exit(1)

    # Create DataFrames
    ddp_df = pd.DataFrame(ddp_results)
    fsdp_df = pd.DataFrame(fsdp_results)

    # Create comparison plots
    create_comparison_plots(ddp_df, fsdp_df, args.output_dir)

    # Print summary
    print_comparison_summary(ddp_df, fsdp_df)

    # Save comparison data
    comparison_file = os.path.join(args.output_dir, 'ddp_vs_fsdp_comparison.csv')
    combined_df = pd.concat([ddp_df, fsdp_df])
    combined_df.to_csv(comparison_file, index=False)
    print(f"\n✓ Comparison data saved: {comparison_file}")

    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()
