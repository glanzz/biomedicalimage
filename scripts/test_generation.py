"""
Test caption generation on sample images

This script loads a trained model checkpoint and generates captions for
random samples from the test set. It's useful for:
- Qualitative evaluation of model outputs
- Debugging generation parameters
- Comparing model checkpoints
- Demonstrating model capabilities
"""

import torch
from PIL import Image
import random
import argparse
import os
import sys
import logging
from typing import List, Dict, Optional
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.vlm import BiomedVLM
from datasets import load_from_disk
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda") -> BiomedVLM:
    """Load model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint directory or checkpoint file
        device: Device to load model on

    Returns:
        Loaded BiomedVLM model
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Initialize model
    model = BiomedVLM(freeze_vision_encoder=True, use_lora=True)

    # Load checkpoint
    if os.path.isdir(checkpoint_path):
        checkpoint_file = os.path.join(checkpoint_path, 'pytorch_model.bin')
    else:
        checkpoint_file = checkpoint_path

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location='cpu')

    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded from step {checkpoint.get('step', 'unknown')}")
        logger.info(f"Checkpoint metrics: {checkpoint.get('metrics', {})}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)

    logger.info("✓ Model loaded successfully")
    return model


def test_generation(
    checkpoint_path: str,
    dataset_path: str = "data/raw/pmc_oa_10k",
    num_samples: int = 5,
    split: str = "test",
    max_new_tokens: int = 256,
    num_beams: int = 3,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda",
    save_outputs: bool = False,
    output_dir: Optional[str] = None
):
    """Test caption generation on random samples

    Args:
        checkpoint_path: Path to model checkpoint
        dataset_path: Path to dataset
        num_samples: Number of samples to test
        split: Dataset split to use ('test', 'valid', 'train')
        max_new_tokens: Maximum number of tokens to generate
        num_beams: Number of beams for beam search
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling threshold
        device: Device to run on
        save_outputs: Whether to save outputs to file
        output_dir: Directory to save outputs (if save_outputs=True)
    """

    # Load model
    model = load_model_from_checkpoint(checkpoint_path, device)

    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    try:
        dataset = load_from_disk(dataset_path)[split]
        logger.info(f"✓ Loaded {len(dataset)} samples from {split} split")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Sample random indices
    if num_samples > len(dataset):
        logger.warning(f"Requested {num_samples} samples but only {len(dataset)} available")
        num_samples = len(dataset)

    indices = random.sample(range(len(dataset)), num_samples)
    logger.info(f"Testing on {num_samples} random samples")

    # Generation parameters
    logger.info("\nGeneration parameters:")
    logger.info(f"  Max tokens: {max_new_tokens}")
    logger.info(f"  Num beams: {num_beams}")
    logger.info(f"  Temperature: {temperature}")
    logger.info(f"  Top-p: {top_p}")
    logger.info("")

    results = []

    # Generate captions for each sample
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample['image']
        ground_truth = sample['caption']

        # Preprocess image
        pixel_values = model.vision_encoder.processor(
            images=image,
            return_tensors="pt"
        )['pixel_values'].to(device)

        # Generate caption
        logger.info(f"\n{'='*80}")
        logger.info(f"Sample {i+1}/{num_samples} (Dataset index: {idx})")
        logger.info(f"{'='*80}")

        with torch.no_grad():
            generated = model.generate_caption(
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0
            )

        generated_caption = generated[0] if isinstance(generated, list) else generated

        logger.info(f"Ground Truth: {ground_truth}")
        logger.info(f"Generated:    {generated_caption}")

        results.append({
            'index': idx,
            'ground_truth': ground_truth,
            'generated': generated_caption
        })

    # Save outputs if requested
    if save_outputs:
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(checkpoint_path),
                'generation_outputs'
            )

        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(
            output_dir,
            f'generated_captions_{split}_{num_samples}samples.json'
        )

        with open(output_file, 'w') as f:
            json.dump({
                'checkpoint': checkpoint_path,
                'dataset': dataset_path,
                'split': split,
                'num_samples': num_samples,
                'generation_params': {
                    'max_new_tokens': max_new_tokens,
                    'num_beams': num_beams,
                    'temperature': temperature,
                    'top_p': top_p
                },
                'results': results
            }, f, indent=2)

        logger.info(f"\n✓ Outputs saved to {output_file}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("GENERATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Tested {num_samples} samples from {split} split")
    logger.info(f"Checkpoint: {checkpoint_path}")
    if save_outputs:
        logger.info(f"Results saved to: {output_file}")
    logger.info("="*80)


def compare_checkpoints(
    checkpoint_paths: List[str],
    dataset_path: str,
    num_samples: int = 3,
    device: str = "cuda"
):
    """Compare generation quality across multiple checkpoints

    Useful for tracking model improvement during training or comparing
    different training configurations.

    Args:
        checkpoint_paths: List of checkpoint paths to compare
        dataset_path: Path to dataset
        num_samples: Number of samples to test
        device: Device to run on
    """

    logger.info("="*80)
    logger.info("CHECKPOINT COMPARISON")
    logger.info("="*80)
    logger.info(f"Comparing {len(checkpoint_paths)} checkpoints")
    logger.info("")

    # Load dataset
    dataset = load_from_disk(dataset_path)['test']

    # Sample fixed indices for fair comparison
    indices = random.sample(range(len(dataset)), num_samples)

    for checkpoint_path in checkpoint_paths:
        logger.info(f"\n{'='*80}")
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"{'='*80}")

        model = load_model_from_checkpoint(checkpoint_path, device)

        for idx in indices:
            sample = dataset[idx]
            image = sample['image']

            pixel_values = model.vision_encoder.processor(
                images=image,
                return_tensors="pt"
            )['pixel_values'].to(device)

            with torch.no_grad():
                generated = model.generate_caption(
                    pixel_values=pixel_values,
                    max_new_tokens=128,
                    num_beams=3
                )

            logger.info(f"\nSample {idx}:")
            logger.info(f"  Generated: {generated[0]}")

        # Clean up
        del model
        torch.cuda.empty_cache()


def main():
    """Main function"""

    parser = argparse.ArgumentParser(
        description="Test caption generation on biomedical images"
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/raw/pmc_oa_10k",
        help="Path to dataset"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to test"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Dataset split to use"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=3,
        help="Number of beams for beam search"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0 = greedy)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on"
    )
    parser.add_argument(
        "--save_outputs",
        action="store_true",
        help="Save generated captions to file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save outputs"
    )

    args = parser.parse_args()

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Run generation test
    test_generation(
        checkpoint_path=args.checkpoint_path,
        dataset_path=args.dataset_path,
        num_samples=args.num_samples,
        split=args.split,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        save_outputs=args.save_outputs,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
