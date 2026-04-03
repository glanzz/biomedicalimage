"""
Final Results Analysis Script

Analyzes all experimental results across all phases and creates
comprehensive final report with visualizations.

Aggregates:
- Preprocessing benchmarks
- DDP training results
- FSDP training results
- CPU baseline results
- Model evaluation metrics

Usage:
    python3 scripts/analyze_results.py
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)


def load_all_results(base_dir="outputs"):
    """Load all experimental results from outputs directory

    Args:
        base_dir: Base outputs directory

    Returns:
        Dictionary with all results
    """
    print("="*80)
    print("LOADING EXPERIMENTAL RESULTS")
    print("="*80)

    results = {
        'preprocessing': None,
        'ddp': {},
        'fsdp': {},
        'cpu_baseline': None,
        'evaluation': {}
    }

    # Preprocessing results
    prep_file = f"{base_dir}/preprocessing_benchmark/benchmark_results.json"
    if os.path.exists(prep_file):
        with open(prep_file) as f:
            results['preprocessing'] = json.load(f)
        print(f"✓ Loaded preprocessing results")
    else:
        print(f"⚠ Preprocessing results not found: {prep_file}")

    # DDP results
    for num_gpus in [1, 2, 4]:
        ddp_file = f"{base_dir}/benchmarks/ddp/ddp_{num_gpus}gpu/ddp_performance_{num_gpus}gpu.json"
        if os.path.exists(ddp_file):
            with open(ddp_file) as f:
                results['ddp'][num_gpus] = json.load(f)
            print(f"✓ Loaded DDP {num_gpus}-GPU results")

    # FSDP results
    for num_gpus in [1, 2, 4]:
        fsdp_file = f"{base_dir}/benchmarks/fsdp/fsdp_{num_gpus}gpu/fsdp_performance_{num_gpus}gpu.json"
        if os.path.exists(fsdp_file):
            with open(fsdp_file) as f:
                results['fsdp'][num_gpus] = json.load(f)
            print(f"✓ Loaded FSDP {num_gpus}-GPU results")

    # CPU Baseline
    cpu_file = f"{base_dir}/baseline/cpu_baseline_results.json"
    if os.path.exists(cpu_file):
        with open(cpu_file) as f:
            results['cpu_baseline'] = json.load(f)
        print(f"✓ Loaded CPU baseline results")

    # Evaluation results
    for config in ['single_gpu', 'ddp_4gpu', 'fsdp_4gpu']:
        eval_file = f"{base_dir}/evaluation/{config}/evaluation_metrics.json"
        if os.path.exists(eval_file):
            with open(eval_file) as f:
                results['evaluation'][config] = json.load(f)
            print(f"✓ Loaded evaluation results for {config}")

    print("="*80)
    return results


def create_final_report(results, output_dir="outputs/final_report"):
    """Create comprehensive final report with visualizations

    Args:
        results: Dictionary of all results
        output_dir: Output directory for report
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("CREATING FINAL REPORT")
    print("="*80)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    # 1. GPU Training Speedup
    ax1 = fig.add_subplot(gs[0, :2])

    gpu_counts = [1, 2, 4]
    has_data = False

    if results['ddp']:
        ddp_gpus = sorted([k for k in results['ddp'].keys() if k in gpu_counts])
        if ddp_gpus and 'total_time' in results['ddp'][ddp_gpus[0]]:
            ddp_times = [results['ddp'][n]['total_time'] for n in ddp_gpus]
            ddp_speedup = [ddp_times[0] / t for t in ddp_times]
            ax1.plot(ddp_gpus, ddp_speedup, marker='o', linewidth=2.5, markersize=10,
                    label='DDP', color='steelblue')
            has_data = True

    if results['fsdp']:
        fsdp_gpus = sorted([k for k in results['fsdp'].keys() if k in gpu_counts])
        if fsdp_gpus and 'total_time' in results['fsdp'][fsdp_gpus[0]]:
            fsdp_times = [results['fsdp'][n]['total_time'] for n in fsdp_gpus]
            fsdp_speedup = [fsdp_times[0] / t for t in fsdp_times]
            ax1.plot(fsdp_gpus, fsdp_speedup, marker='s', linewidth=2.5, markersize=10,
                    label='FSDP', color='coral')
            has_data = True

    if has_data:
        ax1.plot(gpu_counts, gpu_counts, '--', linewidth=2, label='Ideal Linear', color='red')
        ax1.set_xlabel('Number of GPUs', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Speedup', fontsize=12, fontweight='bold')
        ax1.set_title('GPU Training Speedup', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(gpu_counts)

    # 2. Memory Usage Comparison
    ax2 = fig.add_subplot(gs[0, 2])

    mem_data = {}
    if results['ddp'] and 4 in results['ddp'] and 'peak_memory_allocated_GB' in results['ddp'][4]:
        mem_data['DDP\n4-GPU'] = results['ddp'][4]['peak_memory_allocated_GB']
    if results['fsdp'] and 4 in results['fsdp'] and 'peak_memory_allocated_GB' in results['fsdp'][4]:
        mem_data['FSDP\n4-GPU'] = results['fsdp'][4]['peak_memory_allocated_GB']

    if mem_data:
        bars = ax2.bar(mem_data.keys(), mem_data.values(),
                      color=['steelblue', 'coral'], edgecolor='black', alpha=0.7)
        ax2.set_ylabel('Memory (GB)', fontsize=12, fontweight='bold')
        ax2.set_title('Peak Memory Usage', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, mem_data.values()):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}GB', ha='center', va='bottom', fontweight='bold')

    # 3. Scaling Efficiency
    ax3 = fig.add_subplot(gs[1, 0])

    if results['ddp'] and ddp_gpus:
        ddp_eff = [(ddp_speedup[i] / ddp_gpus[i]) * 100 for i in range(len(ddp_speedup))]
        ax3.plot(ddp_gpus, ddp_eff, marker='o', linewidth=2.5, markersize=10,
                label='DDP', color='steelblue')

    if results['fsdp'] and fsdp_gpus:
        fsdp_eff = [(fsdp_speedup[i] / fsdp_gpus[i]) * 100 for i in range(len(fsdp_speedup))]
        ax3.plot(fsdp_gpus, fsdp_eff, marker='s', linewidth=2.5, markersize=10,
                label='FSDP', color='coral')

    ax3.axhline(100, linestyle='--', color='red', linewidth=1.5, label='Ideal')
    ax3.axhline(80, linestyle=':', color='orange', linewidth=1.5, label='Good (80%)')
    ax3.set_xlabel('Number of GPUs', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Efficiency (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Scaling Efficiency', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 110])

    # 4. CPU Baseline Performance
    ax4 = fig.add_subplot(gs[1, 1])

    if results['cpu_baseline']:
        cpu_data = {
            'Sequential': results['cpu_baseline']['sequential']['throughput'],
            'Parallel': results['cpu_baseline']['parallel_train']['throughput']
        }
        bars = ax4.bar(cpu_data.keys(), cpu_data.values(),
                      color='green', edgecolor='black', alpha=0.7)
        ax4.set_ylabel('Throughput (images/sec)', fontsize=12, fontweight='bold')
        ax4.set_title('CPU Baseline Throughput', fontsize=14, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, cpu_data.values()):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

        # Add speedup annotation
        speedup = results['cpu_baseline']['speedup']
        ax4.text(0.5, max(cpu_data.values()) * 0.7,
                f'Speedup: {speedup:.1f}x',
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 5. Model Quality Metrics
    ax5 = fig.add_subplot(gs[1, 2])

    if results['evaluation']:
        # Use first available evaluation
        eval_key = list(results['evaluation'].keys())[0]
        eval_metrics = results['evaluation'][eval_key]

        metric_names = []
        metric_values = []

        for metric in ['BLEU-4', 'ROUGE-L-F', 'CIDEr']:
            if metric in eval_metrics:
                metric_names.append(metric)
                metric_values.append(eval_metrics[metric])

        if metric_names:
            bars = ax5.bar(metric_names, metric_values,
                          color='purple', edgecolor='black', alpha=0.7)
            ax5.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax5.set_title('Model Quality Metrics', fontsize=14, fontweight='bold')
            ax5.set_ylim([0, 1])
            ax5.grid(axis='y', alpha=0.3)

            # Add value labels
            for bar, val in zip(bars, metric_values):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    # 6. Summary Table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('tight')
    ax6.axis('off')

    summary_data = []

    # DDP summary
    if results['ddp'] and 4 in results['ddp']:
        ddp_4 = results['ddp'][4]
        summary_data.append([
            'DDP 4-GPU',
            f"{ddp_speedup[-1] if 'ddp_speedup' in locals() else 'N/A'}x",
            f"{ddp_4.get('peak_memory_allocated_GB', 0):.1f} GB",
            f"{(ddp_speedup[-1] / 4 * 100) if 'ddp_speedup' in locals() else 0:.1f}%",
            f"{ddp_4.get('total_time', 0)/60:.1f} min"
        ])

    # FSDP summary
    if results['fsdp'] and 4 in results['fsdp']:
        fsdp_4 = results['fsdp'][4]
        summary_data.append([
            'FSDP 4-GPU',
            f"{fsdp_speedup[-1] if 'fsdp_speedup' in locals() else 'N/A'}x",
            f"{fsdp_4.get('peak_memory_allocated_GB', 0):.1f} GB",
            f"{(fsdp_speedup[-1] / 4 * 100) if 'fsdp_speedup' in locals() else 0:.1f}%",
            f"{fsdp_4.get('total_time', 0)/60:.1f} min"
        ])

    # CPU Baseline summary
    if results['cpu_baseline']:
        cpu = results['cpu_baseline']
        summary_data.append([
            'CPU Baseline',
            f"{cpu.get('speedup', 0):.1f}x",
            f"{cpu['parallel_train'].get('throughput', 0):.1f} img/s",
            f"{cpu.get('efficiency', 0):.1f}%",
            f"{cpu.get('training', {}).get('training_time', 0)/60:.1f} min"
        ])

    if summary_data:
        table = ax6.table(
            cellText=summary_data,
            colLabels=['Method', 'Speedup', 'Memory/Throughput', 'Efficiency', 'Time'],
            cellLoc='center',
            loc='center',
            colWidths=[0.2, 0.15, 0.25, 0.15, 0.15]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)

        # Style header
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')

        ax6.set_title('Performance Summary', fontsize=16, fontweight='bold', pad=20)

    # Save figure
    plt.savefig(f"{output_dir}/final_results_summary.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved final report: {output_dir}/final_results_summary.png")
    plt.close()

    # Save numerical summary
    with open(f"{output_dir}/numerical_summary.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved numerical summary: {output_dir}/numerical_summary.json")

    # Print text summary
    print_text_summary(results)


def print_text_summary(results):
    """Print text summary of all results

    Args:
        results: Dictionary of all results
    """
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    # DDP Results
    if results['ddp']:
        print("\nDDP Training:")
        print("-"*80)
        for num_gpus in sorted(results['ddp'].keys()):
            data = results['ddp'][num_gpus]
            print(f"  {num_gpus} GPU(s):")
            print(f"    Time: {data.get('total_time', 0):.2f}s ({data.get('total_time', 0)/60:.1f}m)")
            print(f"    Memory: {data.get('peak_memory_allocated_GB', 0):.2f} GB/GPU")

    # FSDP Results
    if results['fsdp']:
        print("\nFSDP Training:")
        print("-"*80)
        for num_gpus in sorted(results['fsdp'].keys()):
            data = results['fsdp'][num_gpus]
            print(f"  {num_gpus} GPU(s):")
            print(f"    Time: {data.get('total_time', 0):.2f}s ({data.get('total_time', 0)/60:.1f}m)")
            print(f"    Memory: {data.get('peak_memory_allocated_GB', 0):.2f} GB/GPU")

    # CPU Baseline
    if results['cpu_baseline']:
        print("\nCPU Baseline:")
        print("-"*80)
        cpu = results['cpu_baseline']
        print(f"  Sequential throughput: {cpu['sequential']['throughput']:.2f} img/s")
        print(f"  Parallel throughput: {cpu['parallel_train']['throughput']:.2f} img/s")
        print(f"  Speedup: {cpu['speedup']:.2f}x")
        print(f"  Efficiency: {cpu['efficiency']:.1f}%")
        if 'evaluation' in cpu:
            print(f"  Accuracy: {cpu['evaluation']['accuracy']:.4f}")

    # Model Evaluation
    if results['evaluation']:
        print("\nModel Quality:")
        print("-"*80)
        for config, metrics in results['evaluation'].items():
            print(f"  {config}:")
            for metric in ['BLEU-4', 'ROUGE-L-F', 'CIDEr']:
                if metric in metrics:
                    print(f"    {metric}: {metrics[metric]:.4f}")

    print("\n" + "="*80)


def main():
    """Main analysis function"""

    print("="*80)
    print("FINAL RESULTS ANALYSIS")
    print("="*80)

    # Load all results
    results = load_all_results()

    # Create final report
    create_final_report(results)

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
