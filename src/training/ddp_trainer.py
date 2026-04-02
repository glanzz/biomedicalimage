"""
Distributed Data Parallel (DDP) trainer
Replicates model on each GPU and synchronizes gradients

This module implements multi-GPU training using PyTorch's Distributed Data Parallel (DDP).
Key features:
- Model replication across GPUs
- Automatic gradient synchronization
- Distributed data loading with DistributedSampler
- Performance tracking and benchmarking
- NCCL backend for efficient GPU communication
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
import time
import json
import sys
import logging
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.vlm import BiomedVLM
from src.data.dataset import PMCImageCaptionDataset
from src.utils.training_utils import (
    TrainingConfig,
    create_optimizer,
    create_scheduler,
    CheckpointManager,
    compute_num_training_steps
)

logger = logging.getLogger(__name__)


def setup_distributed(rank, world_size):
    """Initialize distributed training

    Sets up process group for distributed training with NCCL backend.
    Each process (GPU) gets a unique rank.

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes (GPUs)
    """

    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

    # Initialize process group with NCCL backend (optimized for GPUs)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # Set default GPU for this process
    torch.cuda.set_device(rank)

    if rank == 0:
        logger.info(f"Distributed training initialized with {world_size} GPUs")
        logger.info(f"Master: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")

    logger.info(f"[Rank {rank}] Process initialized on cuda:{rank}")


def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()


class DDPTrainer:
    """Distributed Data Parallel trainer

    Implements multi-GPU training with DDP. Features:
    - Model replication on each GPU
    - Automatic gradient synchronization via NCCL
    - Distributed data sampling (each GPU sees different data)
    - Performance tracking for scaling analysis
    - Rank 0 handles checkpointing and logging

    Args:
        rank: Process rank
        world_size: Total number of GPUs
        model: BiomedVLM model
        config: Training configuration
        dataset_path: Path to dataset
        tokenizer: Tokenizer for captions
        image_processor: Image processor
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        model: BiomedVLM,
        config: TrainingConfig,
        dataset_path: str,
        tokenizer,
        image_processor
    ):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.device = f"cuda:{rank}"

        # Move model to device
        self.model = model.to(self.device)

        # Wrap with DDP - synchronizes gradients across GPUs
        self.model = DDP(
            self.model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False  # Set to True if you have unused parameters
        )

        if rank == 0:
            logger.info("Model wrapped with DistributedDataParallel")

        # Create dataloaders with DistributedSampler
        # Each GPU gets a different subset of the data
        self.train_loader = self._create_distributed_dataloader(
            dataset_path, "train", tokenizer, image_processor,
            batch_size=config.batch_size, shuffle=True
        )

        self.eval_loader = self._create_distributed_dataloader(
            dataset_path, "valid", tokenizer, image_processor,
            batch_size=config.eval_batch_size, shuffle=False
        )

        # Optimizer and scheduler
        num_training_steps = compute_num_training_steps(
            dataset_size=len(self.train_loader.dataset),
            batch_size=config.batch_size,
            num_epochs=config.num_epochs,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            world_size=world_size
        )

        self.optimizer = create_optimizer(self.model, config)
        self.scheduler = create_scheduler(self.optimizer, num_training_steps, config)

        # Mixed precision
        self.scaler = GradScaler() if config.fp16 else None
        self.use_bf16 = config.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        # Checkpointing (only on rank 0)
        if rank == 0:
            self.checkpoint_manager = CheckpointManager(config.output_dir)

        # Logging (only on rank 0)
        if rank == 0 and config.use_wandb:
            wandb.init(
                project="biomedical-image-captioning",
                name=f"ddp-{world_size}gpu",
                config=config.to_dict()
            )
            logger.info("Weights & Biases initialized")

        self.global_step = 0
        self.best_eval_loss = float('inf')

        # Performance tracking for benchmarking
        self.timings = {
            'data_loading': [],
            'forward': [],
            'backward': [],
            'optimizer_step': [],
            'communication': []  # DDP gradient sync overhead
        }

        if rank == 0:
            logger.info("="*70)
            logger.info(f"DDP TRAINER INITIALIZED - {world_size} GPUs")
            logger.info("="*70)
            logger.info(f"Batch size per GPU: {config.batch_size}")
            logger.info(f"Total batch size: {config.batch_size * world_size}")
            logger.info(f"Gradient accumulation: {config.gradient_accumulation_steps}")
            logger.info(f"Effective batch size: {config.batch_size * world_size * config.gradient_accumulation_steps}")
            logger.info(f"Training samples per GPU: {len(self.train_loader.dataset) // world_size}")
            logger.info("="*70)

    def _create_distributed_dataloader(
        self,
        dataset_path,
        split,
        tokenizer,
        image_processor,
        batch_size,
        shuffle
    ):
        """Create dataloader with DistributedSampler

        DistributedSampler ensures each GPU sees a different subset of data
        and no samples are duplicated across GPUs.
        """

        dataset = PMCImageCaptionDataset(
            dataset_path=dataset_path,
            split=split,
            tokenizer=tokenizer,
            image_processor=image_processor
        )

        # DistributedSampler divides dataset among GPUs
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            drop_last=False
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

        if self.rank == 0:
            logger.info(f"{split.capitalize()} dataset: {len(dataset)} samples total, "
                       f"{len(dataset) // self.world_size} per GPU")

        return dataloader

    def train(self):
        """Main training loop"""

        if self.rank == 0:
            logger.info("\n" + "="*70)
            logger.info("STARTING DDP TRAINING")
            logger.info("="*70)

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            # IMPORTANT: Set epoch for DistributedSampler
            # This ensures different shuffle order for each epoch
            self.train_loader.sampler.set_epoch(epoch)

            epoch_start = time.time()
            self.train_epoch(epoch)
            epoch_time = time.time() - epoch_start

            if self.rank == 0:
                logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")

        total_time = time.time() - start_time

        if self.rank == 0:
            logger.info(f"\n{'='*70}")
            logger.info(f"TRAINING COMPLETE!")
            logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
            logger.info(f"Best validation loss: {self.best_eval_loss:.4f}")
            logger.info(f"{'='*70}")

            # Save performance metrics for benchmarking
            self._save_performance_metrics(total_time)

            if self.config.use_wandb:
                wandb.finish()

    def train_epoch(self, epoch: int):
        """Train one epoch"""

        self.model.train()
        epoch_loss = 0

        # Only rank 0 shows progress bar
        if self.rank == 0:
            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
                total=len(self.train_loader)
            )
        else:
            progress_bar = self.train_loader

        for batch_idx, batch in enumerate(progress_bar):
            # Track data loading time
            data_start = time.time()

            pixel_values = batch['pixel_values'].to(self.device, non_blocking=True)
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)

            self.timings['data_loading'].append(time.time() - data_start)

            # Forward pass with mixed precision
            forward_start = time.time()

            if self.use_bf16:
                with autocast(dtype=torch.bfloat16):
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / self.config.gradient_accumulation_steps
            elif self.scaler is not None:
                with autocast():
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / self.config.gradient_accumulation_steps
            else:
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / self.config.gradient_accumulation_steps

            self.timings['forward'].append(time.time() - forward_start)

            # Backward pass (DDP synchronizes gradients here)
            backward_start = time.time()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            self.timings['backward'].append(time.time() - backward_start)

            epoch_loss += loss.item()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Optimizer step
                opt_start = time.time()

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

                self.timings['optimizer_step'].append(time.time() - opt_start)

                self.global_step += 1

                # Logging (rank 0 only)
                if self.rank == 0 and self.global_step % self.config.logging_steps == 0:
                    self._log_metrics(loss, epoch, progress_bar)

                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate()

                    if self.rank == 0:
                        logger.info(f"\nStep {self.global_step} - Eval Loss: {eval_metrics['eval_loss']:.4f}, "
                                   f"Perplexity: {eval_metrics['perplexity']:.2f}")

                        if self.config.use_wandb:
                            wandb.log(eval_metrics, step=self.global_step)

                        # Save best model
                        if eval_metrics['eval_loss'] < self.best_eval_loss:
                            self.best_eval_loss = eval_metrics['eval_loss']
                            logger.info("  ✓ New best model! Saving checkpoint...")
                            self._save_checkpoint(epoch, eval_metrics)

                # Regular checkpointing
                if self.rank == 0 and self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint(epoch, {'train_loss': loss.item() * self.config.gradient_accumulation_steps})

        # Synchronize all GPUs at end of epoch
        dist.barrier()

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set

        Computes loss on validation set and averages across all GPUs.
        """

        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in self.eval_loader:
            pixel_values = batch['pixel_values'].to(self.device, non_blocking=True)
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)

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

        # Average across GPUs using all_reduce
        avg_loss = torch.tensor(total_loss / num_batches).to(self.device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)

        self.model.train()

        return {
            'eval_loss': avg_loss.item(),
            'perplexity': torch.exp(avg_loss).item()
        }

    def _log_metrics(self, loss, epoch, progress_bar=None):
        """Log training metrics (rank 0 only)"""

        lr = self.scheduler.get_last_lr()[0]

        # Calculate throughput (samples/sec)
        recent_timings = 10
        if len(self.timings['forward']) >= recent_timings:
            avg_time_per_batch = (
                sum(self.timings['forward'][-recent_timings:]) / recent_timings +
                sum(self.timings['backward'][-recent_timings:]) / recent_timings +
                sum(self.timings['optimizer_step'][-recent_timings:]) / recent_timings
            )
            throughput = (self.config.batch_size * self.world_size) / avg_time_per_batch
        else:
            throughput = 0

        actual_loss = loss.item() * self.config.gradient_accumulation_steps

        logs = {
            'loss': actual_loss,
            'learning_rate': lr,
            'epoch': epoch,
            'step': self.global_step,
            'throughput': throughput,
            'num_gpus': self.world_size
        }

        # Update progress bar
        if progress_bar is not None and isinstance(progress_bar, tqdm):
            progress_bar.set_postfix({
                'loss': f'{actual_loss:.4f}',
                'lr': f'{lr:.2e}',
                'tput': f'{throughput:.1f} s/s'
            })

        # Log to WandB
        if self.config.use_wandb:
            wandb.log(logs, step=self.global_step)

    def _save_checkpoint(self, epoch, metrics):
        """Save checkpoint (rank 0 only)"""

        if self.rank == 0:
            # Unwrap DDP model before saving
            self.checkpoint_manager.save_checkpoint(
                self.model.module,
                self.optimizer,
                self.scheduler,
                epoch,
                self.global_step,
                metrics
            )

    def _save_performance_metrics(self, total_time):
        """Save performance metrics for benchmarking"""

        if self.rank != 0:
            return

        metrics = {
            'total_time': total_time,
            'world_size': self.world_size,
            'batch_size_per_gpu': self.config.batch_size,
            'total_batch_size': self.config.batch_size * self.world_size,
            'avg_data_loading': sum(self.timings['data_loading']) / len(self.timings['data_loading']),
            'avg_forward': sum(self.timings['forward']) / len(self.timings['forward']),
            'avg_backward': sum(self.timings['backward']) / len(self.timings['backward']),
            'avg_optimizer_step': sum(self.timings['optimizer_step']) / len(self.timings['optimizer_step']),
            'peak_memory_GB': torch.cuda.max_memory_allocated(self.device) / 1e9
        }

        output_file = os.path.join(self.config.output_dir, f'ddp_performance_{self.world_size}gpu.json')
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"\nPerformance metrics saved to {output_file}")


def main_worker(rank, world_size, config, dataset_path, use_preprocessed=False):
    """Worker function for each GPU

    This function is spawned for each GPU process.
    Each process has a unique rank (0 to world_size-1).
    """

    # Setup logging for this process
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize distributed training
    setup_distributed(rank, world_size)

    # Create model (each process creates its own copy)
    if rank == 0:
        logger.info("Creating model...")

    model = BiomedVLM(freeze_vision_encoder=True, use_lora=True)

    # Get tokenizer and image processor
    tokenizer = model.tokenizer
    image_processor = model.vision_encoder.processor

    # Create trainer
    trainer = DDPTrainer(
        rank=rank,
        world_size=world_size,
        model=model,
        config=config,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        image_processor=image_processor
    )

    # Train
    trainer.train()

    # Cleanup
    cleanup_distributed()


def main():
    """Main function to launch DDP training"""

    parser = argparse.ArgumentParser(description="DDP training for biomedical VLM")
    parser.add_argument("--dataset_path", type=str, default="data/raw/pmc_oa_100k",
                       help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints/ddp",
                       help="Output directory for checkpoints")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count(),
                       help="Number of GPUs to use")
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
    parser.add_argument("--use_preprocessed", action="store_true", default=False,
                       help="Use preprocessed dataset")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable Weights & Biases logging")

    args = parser.parse_args()

    # Configuration
    config = TrainingConfig(
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
        bf16=args.bf16,
        fp16=args.fp16,
        use_wandb=not args.no_wandb
    )

    # Save configuration
    os.makedirs(args.output_dir, exist_ok=True)
    config.save(os.path.join(args.output_dir, 'training_config.json'))

    world_size = args.num_gpus

    if world_size < 2:
        print("Error: DDP requires at least 2 GPUs. Use single_gpu_trainer.py for 1 GPU.")
        sys.exit(1)

    print("="*70)
    print(f"LAUNCHING DDP TRAINING WITH {world_size} GPUs")
    print("="*70)
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Total batch size: {args.batch_size * world_size}")
    print(f"Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
    print("="*70)

    # Launch processes using torch.multiprocessing.spawn
    # This creates world_size processes, each running main_worker
    torch.multiprocessing.spawn(
        main_worker,
        args=(world_size, config, args.dataset_path, args.use_preprocessed),
        nprocs=world_size,
        join=True
    )

    print("\n" + "="*70)
    print("DDP TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
