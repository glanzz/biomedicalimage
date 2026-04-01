"""
Visualization utilities for plotting training metrics and results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional
import os


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_training_curves(
    metrics: Dict[str, List[float]],
    output_path: str,
    title: str = "Training Metrics"
):
    """
    Plot training curves (loss, accuracy, etc.)

    Args:
        metrics: Dictionary of metric name -> list of values
        output_path: Path to save plot
        title: Plot title
    """

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))

    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        ax.plot(values, linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name.capitalize()} over Training')
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot: {output_path}")


def plot_speedup_curves(
    num_workers: List[int],
    speedups: List[float],
    output_path: str,
    title: str = "Parallel Speedup",
    efficiency: Optional[List[float]] = None
):
    """
    Plot speedup curves for parallel performance

    Args:
        num_workers: List of worker counts
        speedups: List of speedup values
        output_path: Path to save plot
        title: Plot title
        efficiency: Optional list of efficiency values
    """

    if efficiency is None:
        efficiency = [s / w * 100 for s, w in zip(speedups, num_workers)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Speedup plot
    axes[0].plot(num_workers, speedups, marker='o', linewidth=2,
                 markersize=8, label='Actual')
    axes[0].plot(num_workers, num_workers, '--', linewidth=2,
                 color='red', label='Ideal Linear')
    axes[0].set_xlabel('Number of Workers')
    axes[0].set_ylabel('Speedup')
    axes[0].set_title('Speedup vs Number of Workers')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Efficiency plot
    axes[1].plot(num_workers, efficiency, marker='s', linewidth=2,
                 markersize=8, color='green')
    axes[1].axhline(100, linestyle='--', color='red', label='Ideal (100%)')
    axes[1].set_xlabel('Number of Workers')
    axes[1].set_ylabel('Efficiency (%)')
    axes[1].set_title('Parallel Scaling Efficiency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved speedup plot: {output_path}")


def plot_comparison_bars(
    categories: List[str],
    values_dict: Dict[str, List[float]],
    output_path: str,
    ylabel: str = "Value",
    title: str = "Comparison"
):
    """
    Plot grouped bar chart for comparison

    Args:
        categories: Category names (x-axis)
        values_dict: Dictionary of series name -> values
        output_path: Path to save plot
        ylabel: Y-axis label
        title: Plot title
    """

    x = np.arange(len(categories))
    width = 0.8 / len(values_dict)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, values) in enumerate(values_dict.items()):
        offset = (i - len(values_dict) / 2) * width + width / 2
        ax.bar(x + offset, values, width, label=name)

    ax.set_xlabel('Configuration')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison plot: {output_path}")


def plot_metrics_comparison(
    metrics: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Metrics Comparison"
):
    """
    Plot comparison of multiple metrics across configurations

    Args:
        metrics: Nested dict of config -> metric -> value
        output_path: Path to save plot
        title: Plot title
    """

    configs = list(metrics.keys())
    metric_names = list(list(metrics.values())[0].keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(configs))
    width = 0.8 / len(metric_names)

    for i, metric in enumerate(metric_names):
        values = [metrics[cfg][metric] for cfg in configs]
        offset = (i - len(metric_names) / 2) * width + width / 2
        ax.bar(x + offset, values, width, label=metric)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved metrics comparison: {output_path}")


# Example usage
if __name__ == "__main__":
    # Test training curves
    plot_training_curves(
        metrics={
            'loss': [2.0, 1.5, 1.2, 1.0, 0.9, 0.8],
            'accuracy': [0.3, 0.5, 0.6, 0.7, 0.75, 0.8]
        },
        output_path="outputs/plots/test_training.png",
        title="Test Training Curves"
    )

    # Test speedup curves
    plot_speedup_curves(
        num_workers=[1, 2, 4, 8],
        speedups=[1.0, 1.8, 3.2, 5.5],
        output_path="outputs/plots/test_speedup.png"
    )

    # Test comparison bars
    plot_comparison_bars(
        categories=['Config 1', 'Config 2', 'Config 3'],
        values_dict={
            'DDP': [10, 15, 20],
            'FSDP': [12, 17, 22]
        },
        output_path="outputs/plots/test_comparison.png",
        ylabel="Memory (GB)",
        title="Memory Usage Comparison"
    )

    print("Visualization tests complete!")
