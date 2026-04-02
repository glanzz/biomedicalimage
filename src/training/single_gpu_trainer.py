"""
Single-GPU training script for baseline

This module implements a complete training pipeline for single-GPU training
of the biomedical vision-language model. It includes:
- Mixed precision training (bf16/fp16)
- Gradient accumulation
- Learning rate scheduling with warmup
- Checkpointing and model saving
- WandB logging and progress tracking
- Evaluation on validation set
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
from typing import Optional
import time
import json
import os
import sys
import logging
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.vlm import BiomedVLM
from src.data.dataset import create_dataloader
from src.utils.training_utils import (
    TrainingConfig,
    create_optimizer,
    create_scheduler,
    CheckpointManager,
    compute_num_training_steps
)

logger = logging.getLogger(__name__)


class SingleGPUTrainer:
    """Trainer for single-GPU training

    Implements a complete training loop with all modern best practices:
    - Mixed precision training for faster training and lower memory
    - Gradient accumulation for larger effective batch sizes
    - Learning rate warmup and cosine decay
    - Gradient clipping for training stability
    - Periodic evaluation on validation set
    - Checkpointing with automatic cleanup
    - Integration with Weights & Biases for experiment tracking

    Args:
        model: BiomedVLM model to train
        config: Training configuration
        train_dataloader: DataLoader for training data
        eval_dataloader: DataLoader for validation data
        device: Device to train on ('cuda' or 'cpu')
        use_wandb: Whether to use Weights & Biases for logging
    """

    def __init__(
        self,
        model: BiomedVLM,
        config: TrainingConfig,
        train_dataloader,
        eval_dataloader,
        device: str = "cuda",
        use_wandb: bool = True
    ):
        self.model = model.to(device)
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device

        # Optimizer and scheduler
        num_training_steps = compute_num_training_steps(
            dataset_size=len(train_dataloader.dataset),
            batch_size=config.batch_size,
            num_epochs=config.num_epochs,
            gradient_accumulation_steps=config.gradient_accumulation_steps
        )

        self.optimizer = create_optimizer(model, config)
        self.scheduler = create_scheduler(self.optimizer, num_training_steps, config)

        # Mixed precision training
        self.scaler = GradScaler() if config.fp16 else None
        self.use_bf16 = config.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        # Checkpointing
        self.checkpoint_manager = CheckpointManager(config.output_dir)

        # Logging
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project="biomedical-image-captioning",
                name="single-gpu-baseline",
                config=config.to_dict()
            )
            logger.info("✓ Weights & Biases initialized")

        self.global_step = 0
        self.best_eval_loss = float('inf')

        logger.info("="*70)
        logger.info("SINGLE-GPU TRAINER INITIALIZED")
        logger.info("="*70)
        logger.info(f"Device: {device}")
        logger.info(f"Mixed precision: {'bf16' if self.use_bf16 else 'fp16' if config.fp16 else 'fp32'}")
        logger.info(f"Gradient accumulation: {config.gradient_accumulation_steps} steps")
        logger.info(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
        logger.info(f"Total training steps: {num_training_steps}")
        logger.info("="*70)

    def train(self):
        """Main training loop

        Executes the complete training process:
        1. Iterates over epochs
        2. Trains on training set with gradient accumulation
        3. Periodically evaluates on validation set
        4. Saves checkpoints when validation loss improves
        5. Logs metrics to console and WandB
        """

        logger.info("\n" + "="*70)
        logger.info("STARTING TRAINING")
        logger.info("="*70)
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"Training samples: {len(self.train_dataloader.dataset)}")
        logger.info(f"Validation samples: {len(self.eval_dataloader.dataset)}")
        logger.info("="*70 + "\n")

        training_start_time = time.time()

        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            self.train_epoch(epoch)
            epoch_time = time.time() - epoch_start_time

            logger.info(f"\nEpoch {epoch+1} completed in {epoch_time:.2f}s")

        total_time = time.time() - training_start_time

        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        logger.info(f"Best validation loss: {self.best_eval_loss:.4f}")
        logger.info("="*70)

        if self.use_wandb:
            wandb.finish()

    def train_epoch(self, epoch: int):
        """Train for one epoch

        Args:
            epoch: Current epoch number (0-indexed)
        """

        self.model.train()
        epoch_loss = 0
        step = 0

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
            total=len(self.train_dataloader)
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass with mixed precision
            if self.use_bf16:
                with autocast(dtype=torch.bfloat16):
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / self.config.gradient_accumulation_steps
            elif self.scaler is not None:  # fp16
                with autocast():
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / self.config.gradient_accumulation_steps
            else:  # fp32
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item()

            # Gradient accumulation - perform optimizer step after N batches
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                step += 1

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    actual_loss = loss.item() * self.config.gradient_accumulation_steps

                    logs = {
                        'loss': actual_loss,
                        'learning_rate': lr,
                        'epoch': epoch,
                        'step': self.global_step
                    }

                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f'{actual_loss:.4f}',
                        'lr': f'{lr:.2e}'
                    })

                    # Log to WandB
                    if self.use_wandb:
                        wandb.log(logs, step=self.global_step)

                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate()

                    logger.info(f"\nStep {self.global_step} - Evaluation metrics:")
                    logger.info(f"  Loss: {eval_metrics['eval_loss']:.4f}")
                    logger.info(f"  Perplexity: {eval_metrics['perplexity']:.2f}")

                    if self.use_wandb:
                        wandb.log(eval_metrics, step=self.global_step)

                    # Save if best model
                    if eval_metrics['eval_loss'] < self.best_eval_loss:
                        self.best_eval_loss = eval_metrics['eval_loss']
                        logger.info(f"  ✓ New best model! Saving checkpoint...")
                        self.checkpoint_manager.save_checkpoint(
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            epoch,
                            self.global_step,
                            eval_metrics
                        )

                # Regular checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.checkpoint_manager.save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        self.global_step,
                        {'train_loss': loss.item() * self.config.gradient_accumulation_steps}
                    )

        avg_epoch_loss = epoch_loss / len(self.train_dataloader)
        logger.info(f"\nEpoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}")

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set

        Returns:
            Dictionary containing evaluation metrics:
            - eval_loss: Average cross-entropy loss
            - perplexity: exp(loss) - measures prediction uncertainty
        """

        self.model.eval()
        total_loss = 0
        num_batches = 0

        eval_progress = tqdm(self.eval_dataloader, desc="Evaluating", leave=False)

        for batch in eval_progress:
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass with mixed precision
            if self.use_bf16:
                with autocast(dtype=torch.bfloat16):
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
            elif self.scaler is not None:
                with autocast():
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
            else:
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

            total_loss += outputs.loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        self.model.train()

        return {
            'eval_loss': avg_loss,
            'perplexity': perplexity
        }


def main():
    """Main training function

    Parses command-line arguments, sets up logging, creates model and data loaders,
    and launches the training process.
    """

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse arguments
    parser = argparse.ArgumentParser(description="Single-GPU training for biomedical VLM")
    parser.add_argument("--dataset_path", type=str, default="data/raw/pmc_oa_100k",
                       help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints/single_gpu",
                       help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--bf16", action="store_true", default=True,
                       help="Use bfloat16 mixed precision")
    parser.add_argument("--fp16", action="store_true", default=False,
                       help="Use float16 mixed precision")
    parser.add_argument("--logging_steps", type=int, default=100,
                       help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    parser.add_argument("--use_preprocessed", action="store_true", default=False,
                       help="Use preprocessed dataset")

    args = parser.parse_args()

    # Create configuration
    config = TrainingConfig(
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        bf16=args.bf16,
        fp16=args.fp16,
        use_wandb=not args.no_wandb
    )

    # Save configuration
    os.makedirs(args.output_dir, exist_ok=True)
    config.save(os.path.join(args.output_dir, 'training_config.json'))

    # Check for GPU
    if not torch.cuda.is_available():
        logger.warning("CUDA not available! Training on CPU will be very slow.")
        device = "cpu"
    else:
        device = "cuda"
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create model
    logger.info("\n" + "="*70)
    logger.info("INITIALIZING MODEL")
    logger.info("="*70)

    model = BiomedVLM(
        freeze_vision_encoder=True,
        use_lora=True
    )

    # Get tokenizer and image processor from model
    tokenizer = model.tokenizer
    image_processor = model.vision_encoder.processor

    # Create dataloaders
    logger.info("\n" + "="*70)
    logger.info("CREATING DATALOADERS")
    logger.info("="*70)

    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Preprocessed: {args.use_preprocessed}")

    train_loader = create_dataloader(
        dataset_path=args.dataset_path,
        split="train",
        tokenizer=tokenizer,
        image_processor=image_processor,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        use_preprocessed=args.use_preprocessed
    )

    eval_loader = create_dataloader(
        dataset_path=args.dataset_path,
        split="valid",
        tokenizer=tokenizer,
        image_processor=image_processor,
        batch_size=args.batch_size * 2,  # Larger batch for eval
        num_workers=4,
        shuffle=False,
        use_preprocessed=args.use_preprocessed
    )

    logger.info(f"✓ Training batches: {len(train_loader)}")
    logger.info(f"✓ Validation batches: {len(eval_loader)}")

    # Create trainer
    trainer = SingleGPUTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        device=device,
        use_wandb=not args.no_wandb
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
