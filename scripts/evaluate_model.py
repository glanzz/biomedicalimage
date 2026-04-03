"""
Comprehensive Model Evaluation Script

Evaluates trained VLM model on test set and generates detailed report.
Computes BLEU, ROUGE-L, and CIDEr metrics.

Usage:
    python3 scripts/evaluate_model.py \
        --checkpoint_path outputs/checkpoints/ddp_4gpu/checkpoint-10000 \
        --dataset_path data/raw/pmc_oa_10k \
        --output_dir outputs/evaluation
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.vlm import BiomedVLM
from src.data.dataset import create_dataloader
from src.evaluation.metrics import CaptionMetrics, evaluate_model, print_sample_predictions

# Set style
sns.set_style("whitegrid")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def comprehensive_evaluation(
    checkpoint_path: str,
    dataset_path: str = "data/raw/pmc_oa_10k",
    output_dir: str = "outputs/evaluation",
    max_samples: int = 1000,
    batch_size: int = 8,
    num_beams: int = 3,
    device: str = "cuda"
):
    """Run comprehensive model evaluation

    Args:
        checkpoint_path: Path to model checkpoint directory
        dataset_path: Path to dataset
        output_dir: Output directory for results
        max_samples: Maximum samples to evaluate
        batch_size: Batch size for inference
        num_beams: Number of beams for generation
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Dictionary of evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Max samples: {max_samples}")
    print(f"Device: {device}")
    print("="*80)

    # Check device availability
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    # Load model
    logger.info("Loading model...")
    try:
        model = BiomedVLM(freeze_vision_encoder=True, use_lora=True)

        checkpoint_file = os.path.join(checkpoint_path, 'pytorch_model.bin')
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

        checkpoint = torch.load(checkpoint_file, map_location=device)

        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        model = model.to(device)

        logger.info("✓ Model loaded successfully")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Creating fresh model for testing (no checkpoint loaded)")
        model = BiomedVLM(freeze_vision_encoder=True, use_lora=True)
        model.eval()
        model = model.to(device)

    # Create dataloader
    logger.info("Creating dataloader...")
    try:
        test_loader = create_dataloader(
            dataset_path=dataset_path,
            split="test",
            tokenizer=model.language_model.tokenizer,
            image_processor=model.vision_encoder.processor,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False
        )
        logger.info(f"✓ Dataloader created: {len(test_loader)} batches")
    except Exception as e:
        logger.error(f"Error creating dataloader: {e}")
        raise

    # Evaluate model
    logger.info("Evaluating model...")
    try:
        metrics, references, predictions = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            max_samples=max_samples,
            num_beams=num_beams
        )
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

    # Print metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    for metric, value in sorted(metrics.items()):
        print(f"{metric:.<40} {value:.4f}")
    print("="*80)

    # Save metrics
    metrics_file = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"✓ Metrics saved: {metrics_file}")

    # Save sample predictions
    samples = []
    for i in range(min(20, len(references))):
        samples.append({
            'id': i,
            'reference': references[i],
            'prediction': predictions[i]
        })

    samples_file = os.path.join(output_dir, 'sample_predictions.json')
    with open(samples_file, 'w') as f:
        json.dump(samples, f, indent=2)
    logger.info(f"✓ Samples saved: {samples_file}")

    # Print sample predictions
    print_sample_predictions(references, predictions, num_samples=5)

    # Create visualization
    try:
        visualize_metrics(metrics, output_dir)
    except Exception as e:
        logger.warning(f"Could not create visualization: {e}")

    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print("="*80)

    return metrics


def visualize_metrics(metrics: dict, output_dir: str):
    """Create visualization of evaluation metrics

    Args:
        metrics: Dictionary of metric scores
        output_dir: Output directory
    """
    logger.info("Creating visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Extract BLEU scores
    bleu_scores = {k: v for k, v in metrics.items() if 'BLEU' in k}

    if bleu_scores:
        axes[0].bar(bleu_scores.keys(), bleu_scores.values(),
                    color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].set_title('BLEU Scores', fontsize=14, fontweight='bold')
        axes[0].set_ylim(0, 1)
        axes[0].grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (k, v) in enumerate(bleu_scores.items()):
            axes[0].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    # Other metrics
    other_metrics = {}
    if 'ROUGE-L-F' in metrics:
        other_metrics['ROUGE-L'] = metrics['ROUGE-L-F']
    if 'CIDEr' in metrics:
        other_metrics['CIDEr'] = metrics['CIDEr'] / 10  # Scale for visualization

    if other_metrics:
        axes[1].bar(other_metrics.keys(), other_metrics.values(),
                    color='coral', edgecolor='black', alpha=0.7)
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].set_title('ROUGE-L and CIDEr (scaled /10)', fontsize=14, fontweight='bold')
        axes[1].set_ylim(0, 1)
        axes[1].grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (k, v) in enumerate(other_metrics.items()):
            label = f'{v:.3f}' if 'ROUGE' in k else f'{v*10:.2f}'
            axes[1].text(i, v, label, ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    plot_file = os.path.join(output_dir, 'evaluation_metrics.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Visualization saved: {plot_file}")
    plt.close()


def main():
    """Main evaluation function"""

    parser = argparse.ArgumentParser(description='Evaluate trained VLM model')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint directory')
    parser.add_argument('--dataset_path', type=str, default='data/raw/pmc_oa_10k',
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                       help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for inference')
    parser.add_argument('--num_beams', type=int, default=3,
                       help='Number of beams for generation')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')

    args = parser.parse_args()

    try:
        metrics = comprehensive_evaluation(
            checkpoint_path=args.checkpoint_path,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            num_beams=args.num_beams,
            device=args.device
        )

        # Exit with success
        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
