"""
Fully Sharded Data Parallel (FSDP) trainer
Shards model parameters, gradients, and optimizer states across GPUs for memory efficiency

This module implements multi-GPU training using PyTorch's FSDP.
Key features:
- Parameter sharding across GPUs (reduces memory per GPU)
- Gradient and optimizer state sharding
- Mixed precision training with bfloat16
- Automatic model wrapping with size-based policy
- Lower memory footprint enables larger batch sizes
- Performance tracking and memory profiling

Memory Comparison vs DDP:
- DDP: Each GPU holds full model copy (~8GB per GPU)
- FSDP: Model sharded across GPUs (~2-4GB per GPU)
- Enables 2-4x larger batch sizes on same hardware
"""

import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
    CPUOffload
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap
)
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from tqdm import tqdm
import wandb
import time
import json
import sys
import logging
import argparse
from functools import partial

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.vlm import BiomedVLM
from src.data.dataset import PMCImageCaptionDataset
from src.utils.training_utils import (
    TrainingConfig,
    create_optimizer,
    create_scheduler,
    compute_num_training_steps
)

logger = logging.getLogger(__name__)


def setup_distributed(rank, world_size):
    """Initialize distributed training for FSDP

    Sets up process group for FSDP with NCCL backend.

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes (GPUs)
    """

    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

    # Initialize process group with NCCL backend
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # Set default GPU for this process
    torch.cuda.set_device(rank)

    if rank == 0:
        logger.info(f"FSDP initialized with {world_size} GPUs")
        logger.info(f"Master: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")

    logger.info(f"[Rank {rank}] FSDP process initialized on cuda:{rank}")


def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()


class FSDPTrainer:
    """Fully Sharded Data Parallel trainer

    Implements multi-GPU training with FSDP for memory efficiency.
    Features:
    - Full parameter sharding across GPUs
    - Gradient and optimizer state sharding
    - Mixed precision (bfloat16) for A100 GPUs
    - Automatic model wrapping
    - Memory-efficient checkpointing
    - Performance profiling

    Memory Advantages:
    - DDP: O(N) memory per GPU (N = model size)
    - FSDP: O(N/G) memory per GPU (G = num GPUs)
    - Enables 2-4x larger batch sizes

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

        # FSDP Mixed Precision Configuration
        # Use bfloat16 for parameters and computations (A100 optimized)
        # Use float32 for gradient reduction (better numerical stability)
        # Note: config.bf16 is always True for FSDP (required for A100 GPUs)
        if config.bf16:
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
                cast_forward_inputs=True
            )
        else:
            # Fallback to fp16 if bf16 not supported
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float16,
                cast_forward_inputs=True
            )

        # Auto-wrap policy: automatically shard layers with >100M parameters
        # This balances communication overhead vs memory savings
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=100_000_000  # 100M parameters
        )

        if rank == 0:
            logger.info("Configuring FSDP model...")
            logger.info(f"  Sharding Strategy: FULL_SHARD")
            logger.info(f"  Mixed Precision: bfloat16")
            logger.info(f"  Auto-wrap threshold: 100M params")

        # Move model to device first
        self.model = model.to(self.device)

        # Wrap model with FSDP
        self.model = FSDP(
            self.model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision_policy,
            auto_wrap_policy=auto_wrap_policy,
            device_id=rank,
            limit_all_gathers=True,  # Reduce memory spikes during forward pass
            use_orig_params=True,     # Required for LoRA fine-tuning
            sync_module_states=True   # Ensure all ranks start with same params
        )

        if rank == 0:
            logger.info("Model wrapped with FSDP successfully")
            # Log approximate memory per GPU
            param_count = sum(p.numel() for p in self.model.parameters())
            param_memory_gb = (param_count * 2) / 1e9  # bfloat16 = 2 bytes
            sharded_memory_gb = param_memory_gb / world_size
            logger.info(f"  Total params: {param_count:,}")
            logger.info(f"  Full model memory: {param_memory_gb:.2f} GB")
            logger.info(f"  Sharded memory per GPU: {sharded_memory_gb:.2f} GB")

        # Create distributed dataloaders
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
            len(self.train_loader),
            config.num_epochs,
            config.gradient_accumulation_steps
        )

        self.optimizer = create_optimizer(self.model, config)
        self.scheduler = create_scheduler(self.optimizer, num_training_steps, config)

        if rank == 0:
            logger.info(f"Optimizer: AdamW with LR={config.learning_rate}")
            logger.info(f"Scheduler: Cosine with warmup")
            logger.info(f"Training steps: {num_training_steps:,}")

        # Logging (only rank 0)
        if rank == 0 and config.use_wandb:
            wandb.init(
                project="biomedical-image-captioning",
                name=f"fsdp-{world_size}gpu",
                config={
                    **vars(config),
                    'method': 'FSDP',
                    'sharding_strategy': 'FULL_SHARD',
                    'mixed_precision': 'bfloat16'
                }
            )
            logger.info("WandB logging initialized")

        self.global_step = 0
        self.best_eval_loss = float('inf')

        # Performance tracking
        self.timings = {
            'data_loading': [],
            'forward': [],
            'backward': [],
            'optimizer_step': [],
            'communication': []
        }

        # Memory tracking
        self.memory_stats = {
            'allocated': [],
            'reserved': [],
            'max_allocated': []
        }

    def _create_distributed_dataloader(
        self,
        dataset_path: str,
        split: str,
        tokenizer,
        image_processor,
        batch_size: int,
        shuffle: bool
    ):
        """Create dataloader with DistributedSampler

        Args:
            dataset_path: Path to dataset
            split: Dataset split ('train', 'valid', 'test')
            tokenizer: Text tokenizer
            image_processor: Image processor
            batch_size: Batch size per GPU
            shuffle: Whether to shuffle data

        Returns:
            DataLoader with distributed sampling
        """

        dataset = PMCImageCaptionDataset(
            dataset_path=dataset_path,
            split=split,
            tokenizer=tokenizer,
            image_processor=image_processor
        )

        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            drop_last=True  # Ensure all batches have same size
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2
        )

        if self.rank == 0:
            logger.info(f"{split} dataloader: {len(dataset):,} samples, {len(dataloader)} batches")

        return dataloader

    def train(self):
        """Main training loop

        Trains model for specified epochs with:
        - Distributed data parallel training
        - Mixed precision (bfloat16)
        - Gradient accumulation
        - Periodic evaluation and checkpointing
        - Performance profiling
        """

        if self.rank == 0:
            logger.info("="*80)
            logger.info(f"FSDP Training - {self.world_size} GPUs")
            logger.info("="*80)
            logger.info(f"Epochs: {self.config.num_epochs}")
            logger.info(f"Batch size per GPU: {self.config.batch_size}")
            logger.info(f"Global batch size: {self.config.batch_size * self.world_size}")
            logger.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
            logger.info(f"Effective batch size: {self.config.batch_size * self.world_size * self.config.gradient_accumulation_steps}")
            logger.info("="*80)

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            # Set epoch for distributed sampler (ensures different shuffle each epoch)
            self.train_loader.sampler.set_epoch(epoch)

            epoch_start = time.time()
            self.train_epoch(epoch)
            epoch_time = time.time() - epoch_start

            if self.rank == 0:
                logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")

        total_time = time.time() - start_time

        if self.rank == 0:
            logger.info(f"\nTotal training time: {total_time:.2f}s ({total_time/60:.2f}m)")

            # Save performance metrics
            self._save_performance_metrics(total_time)

            if self.config.use_wandb:
                wandb.finish()

    def train_epoch(self, epoch: int):
        """Train one epoch

        Args:
            epoch: Current epoch number
        """

        self.model.train()
        epoch_loss = 0
        num_batches = 0

        # Only show progress bar on rank 0
        if self.rank == 0:
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        else:
            progress_bar = self.train_loader

        for batch_idx, batch in enumerate(progress_bar):
            # Track data loading time
            data_start = time.time()

            # Move batch to device
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.timings['data_loading'].append(time.time() - data_start)

            # Forward pass with mixed precision
            forward_start = time.time()

            # FSDP handles mixed precision internally
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss / self.config.gradient_accumulation_steps

            self.timings['forward'].append(time.time() - forward_start)

            # Backward pass
            backward_start = time.time()
            loss.backward()
            self.timings['backward'].append(time.time() - backward_start)

            epoch_loss += loss.item()
            num_batches += 1

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                opt_start = time.time()

                # Gradient clipping (FSDP provides clip_grad_norm_)
                self.model.clip_grad_norm_(self.config.max_grad_norm)

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.timings['optimizer_step'].append(time.time() - opt_start)

                self.global_step += 1

                # Memory tracking (rank 0 only)
                if self.rank == 0:
                    self._track_memory()

                # Logging (rank 0 only)
                if self.rank == 0 and self.global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    self._log_metrics(avg_loss * self.config.gradient_accumulation_steps, epoch)

                    # Update progress bar
                    if isinstance(progress_bar, tqdm):
                        progress_bar.set_postfix({
                            'loss': f"{avg_loss * self.config.gradient_accumulation_steps:.4f}",
                            'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                            'mem': f"{torch.cuda.memory_allocated(self.device)/1e9:.2f}GB"
                        })

                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate()

                    if self.rank == 0:
                        logger.info(f"Step {self.global_step} - Eval Loss: {eval_metrics['eval_loss']:.4f}, Perplexity: {eval_metrics['perplexity']:.2f}")

                        if self.config.use_wandb:
                            wandb.log(eval_metrics, step=self.global_step)

                        # Save best model
                        if eval_metrics['eval_loss'] < self.best_eval_loss:
                            self.best_eval_loss = eval_metrics['eval_loss']
                            logger.info(f"New best model! Eval loss: {self.best_eval_loss:.4f}")
                            self._save_checkpoint(epoch, eval_metrics)

                # Regular checkpointing
                if self.rank == 0 and self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint(epoch, {'train_loss': loss.item() * self.config.gradient_accumulation_steps})

        # End of epoch summary
        if self.rank == 0:
            avg_epoch_loss = epoch_loss / num_batches * self.config.gradient_accumulation_steps
            logger.info(f"Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}")

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set

        Returns:
            Dictionary with evaluation metrics
        """

        self.model.eval()
        total_loss = 0
        num_batches = 0

        eval_bar = tqdm(self.eval_loader, desc="Evaluating") if self.rank == 0 else self.eval_loader

        for batch in eval_bar:
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass (FSDP handles mixed precision)
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()
            num_batches += 1

        # Average loss across all batches
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Reduce across all GPUs
        avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)

        self.model.train()

        return {
            'eval_loss': avg_loss_tensor.item(),
            'perplexity': torch.exp(avg_loss_tensor).item()
        }

    def _log_metrics(self, loss: float, epoch: int):
        """Log training metrics (rank 0 only)

        Args:
            loss: Current training loss
            epoch: Current epoch
        """

        lr = self.scheduler.get_last_lr()[0]

        # Calculate throughput (samples per second)
        if len(self.timings['forward']) >= 10:
            recent_forward = sum(self.timings['forward'][-10:]) / 10
            recent_backward = sum(self.timings['backward'][-10:]) / 10
            recent_opt = sum(self.timings['optimizer_step'][-10:]) / 10
            step_time = recent_forward + recent_backward + recent_opt
            throughput = (self.config.batch_size * self.world_size) / step_time if step_time > 0 else 0
        else:
            throughput = 0

        # Memory statistics
        memory_allocated = torch.cuda.memory_allocated(self.device) / 1e9
        memory_reserved = torch.cuda.memory_reserved(self.device) / 1e9
        max_memory = torch.cuda.max_memory_allocated(self.device) / 1e9

        logs = {
            'train_loss': loss,
            'learning_rate': lr,
            'epoch': epoch,
            'step': self.global_step,
            'throughput_samples_per_sec': throughput,
            'num_gpus': self.world_size,
            'memory_allocated_GB': memory_allocated,
            'memory_reserved_GB': memory_reserved,
            'max_memory_GB': max_memory
        }

        if self.config.use_wandb:
            wandb.log(logs, step=self.global_step)

    def _track_memory(self):
        """Track GPU memory usage"""

        self.memory_stats['allocated'].append(
            torch.cuda.memory_allocated(self.device) / 1e9
        )
        self.memory_stats['reserved'].append(
            torch.cuda.memory_reserved(self.device) / 1e9
        )
        self.memory_stats['max_allocated'].append(
            torch.cuda.max_memory_allocated(self.device) / 1e9
        )

    def _save_checkpoint(self, epoch: int, metrics: dict):
        """Save FSDP checkpoint (rank 0 only)

        FSDP checkpointing collects full state dict on rank 0 and saves it.

        Args:
            epoch: Current epoch
            metrics: Training metrics to save
        """

        if self.rank == 0:
            logger.info(f"Saving checkpoint at step {self.global_step}...")

            # Configure state dict collection (gather on rank 0, offload to CPU)
            save_policy = FullStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True
            )

            # Collect full state dict
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                save_policy
            ):
                model_state_dict = self.model.state_dict()

            # Create checkpoint directory
            checkpoint_dir = os.path.join(
                self.config.output_dir,
                f"checkpoint-{self.global_step}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'step': self.global_step,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'metrics': metrics,
                'config': vars(self.config)
            }

            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, 'pytorch_model.bin')
            )

            logger.info(f"Checkpoint saved: {checkpoint_dir}")

            # Keep only last N checkpoints
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoints to save disk space

        Args:
            keep_last: Number of recent checkpoints to keep
        """

        import glob
        import shutil

        checkpoints = sorted(glob.glob(os.path.join(self.config.output_dir, 'checkpoint-*')))

        if len(checkpoints) > keep_last:
            for old_ckpt in checkpoints[:-keep_last]:
                shutil.rmtree(old_ckpt)
                logger.info(f"Removed old checkpoint: {old_ckpt}")

    def _save_performance_metrics(self, total_time: float):
        """Save detailed performance metrics

        Args:
            total_time: Total training time in seconds
        """

        # Calculate averages
        avg_metrics = {
            'total_time': total_time,
            'total_time_minutes': total_time / 60,
            'world_size': self.world_size,
            'num_epochs': self.config.num_epochs,
            'batch_size_per_gpu': self.config.batch_size,
            'global_batch_size': self.config.batch_size * self.world_size,
        }

        # Timing statistics
        for key in ['data_loading', 'forward', 'backward', 'optimizer_step']:
            if self.timings[key]:
                avg_metrics[f'avg_{key}'] = sum(self.timings[key]) / len(self.timings[key])
                avg_metrics[f'total_{key}'] = sum(self.timings[key])

        # Memory statistics
        if self.memory_stats['allocated']:
            avg_metrics['avg_memory_allocated_GB'] = sum(self.memory_stats['allocated']) / len(self.memory_stats['allocated'])
            avg_metrics['peak_memory_allocated_GB'] = max(self.memory_stats['max_allocated'])
            avg_metrics['avg_memory_reserved_GB'] = sum(self.memory_stats['reserved']) / len(self.memory_stats['reserved'])

        # Throughput
        total_step_time = avg_metrics.get('avg_forward', 0) + avg_metrics.get('avg_backward', 0) + avg_metrics.get('avg_optimizer_step', 0)
        if total_step_time > 0:
            avg_metrics['avg_throughput_samples_per_sec'] = (self.config.batch_size * self.world_size) / total_step_time

        # Save metrics
        output_file = os.path.join(
            self.config.output_dir,
            f'fsdp_performance_{self.world_size}gpu.json'
        )

        with open(output_file, 'w') as f:
            json.dump(avg_metrics, f, indent=2)

        logger.info(f"Performance metrics saved: {output_file}")

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("FSDP PERFORMANCE SUMMARY")
        logger.info("="*80)
        logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
        logger.info(f"GPUs: {self.world_size}")
        logger.info(f"Peak memory per GPU: {avg_metrics.get('peak_memory_allocated_GB', 0):.2f} GB")
        if 'avg_throughput_samples_per_sec' in avg_metrics:
            logger.info(f"Throughput: {avg_metrics['avg_throughput_samples_per_sec']:.2f} samples/sec")
        logger.info("="*80)


def main_worker(rank: int, world_size: int, config: TrainingConfig, dataset_path: str):
    """Worker function for each GPU process

    Args:
        rank: Process rank
        world_size: Total number of processes
        config: Training configuration
        dataset_path: Path to dataset
    """

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize distributed training
    setup_distributed(rank, world_size)

    try:
        # Create model
        if rank == 0:
            logger.info("Initializing BiomedVLM model...")

        model = BiomedVLM(
            freeze_vision_encoder=True,
            use_lora=True
        )

        if rank == 0:
            logger.info("Model initialized")

        # Get tokenizer and image processor
        tokenizer = model.language_model.tokenizer
        image_processor = model.vision_encoder.processor

        # Create FSDP trainer
        trainer = FSDPTrainer(
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

    except Exception as e:
        logger.error(f"Error in training: {e}")
        raise
    finally:
        # Cleanup
        cleanup_distributed()


def main():
    """Main function to launch FSDP training"""

    parser = argparse.ArgumentParser(description='FSDP Training for BiomedVLM')
    parser.add_argument('--dataset_path', type=str, default='data/raw/pmc_oa_100k',
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/checkpoints/fsdp',
                       help='Output directory for checkpoints')
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(),
                       help='Number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size per GPU (FSDP allows larger batches)')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                       help='Gradient accumulation steps')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable WandB logging')

    args = parser.parse_args()

    # Validate GPU count
    if args.num_gpus > torch.cuda.device_count():
        raise ValueError(f"Requested {args.num_gpus} GPUs but only {torch.cuda.device_count()} available")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create training configuration
    config = TrainingConfig(
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size * 2,  # Larger eval batch with FSDP
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
        bf16=True,  # Use bfloat16 for A100
        use_wandb=not args.no_wandb,
        save_steps=1000,
        eval_steps=500,
        logging_steps=100
    )

    world_size = args.num_gpus

    print("="*80)
    print("FSDP Training Configuration")
    print("="*80)
    print(f"GPUs: {world_size}")
    print(f"Batch size per GPU: {config.batch_size}")
    print(f"Global batch size: {config.batch_size * world_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * world_size * config.gradient_accumulation_steps}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print("="*80)

    # Launch distributed training
    torch.multiprocessing.spawn(
        main_worker,
        args=(world_size, config, args.dataset_path),
        nprocs=world_size,
        join=True
    )

    print("\n✅ FSDP Training completed!")


if __name__ == "__main__":
    main()
