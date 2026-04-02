"""
Training utilities: optimizer, scheduler, checkpointing

This module provides utilities for training vision-language models:
- TrainingConfig: Configuration dataclass for training parameters
- create_optimizer: Creates AdamW optimizer with weight decay
- create_scheduler: Creates learning rate scheduler with warmup
- CheckpointManager: Manages model checkpointing and loading
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import os
from typing import Dict, Optional
import logging
import json

logger = logging.getLogger(__name__)


class TrainingConfig:
    """Training configuration

    Encapsulates all training hyperparameters and settings for easy management
    and serialization.

    Args:
        learning_rate: Initial learning rate for AdamW optimizer
        weight_decay: Weight decay coefficient for regularization
        adam_beta1: Beta1 parameter for AdamW
        adam_beta2: Beta2 parameter for AdamW
        adam_epsilon: Epsilon parameter for numerical stability
        num_epochs: Number of training epochs
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
        warmup_ratio: Ratio of total steps for warmup
        fp16: Use mixed precision training with float16
        bf16: Use mixed precision training with bfloat16
        save_steps: Save checkpoint every N steps
        eval_steps: Run evaluation every N steps
        logging_steps: Log metrics every N steps
        output_dir: Directory to save checkpoints and logs
        batch_size: Training batch size per GPU
        eval_batch_size: Evaluation batch size per GPU
        use_wandb: Whether to use Weights & Biases for logging
    """

    def __init__(self, **kwargs):
        # Optimizer parameters
        self.learning_rate = kwargs.get('learning_rate', 2e-4)
        self.weight_decay = kwargs.get('weight_decay', 0.01)
        self.adam_beta1 = kwargs.get('adam_beta1', 0.9)
        self.adam_beta2 = kwargs.get('adam_beta2', 0.999)
        self.adam_epsilon = kwargs.get('adam_epsilon', 1e-8)

        # Training parameters
        self.num_epochs = kwargs.get('num_epochs', 3)
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 4)
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        self.warmup_ratio = kwargs.get('warmup_ratio', 0.03)

        # Mixed precision
        self.fp16 = kwargs.get('fp16', False)
        self.bf16 = kwargs.get('bf16', True)

        # Checkpointing and logging
        self.save_steps = kwargs.get('save_steps', 1000)
        self.eval_steps = kwargs.get('eval_steps', 500)
        self.logging_steps = kwargs.get('logging_steps', 100)
        self.output_dir = kwargs.get('output_dir', 'outputs/checkpoints')

        # Batch sizes
        self.batch_size = kwargs.get('batch_size', 4)
        self.eval_batch_size = kwargs.get('eval_batch_size', 8)

        # Logging
        self.use_wandb = kwargs.get('use_wandb', True)

    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items()}

    def save(self, path: str):
        """Save config to JSON file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Config saved to {path}")

    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Load config from dictionary"""
        return cls(**config_dict)

    @classmethod
    def load(cls, path: str):
        """Load config from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def create_optimizer(model, config: TrainingConfig):
    """Create AdamW optimizer with weight decay

    Separates parameters into two groups:
    1. Parameters with weight decay (weights)
    2. Parameters without weight decay (biases, layer norms)

    This is a common practice in transformer training to avoid
    regularizing biases and normalization parameters.

    Args:
        model: PyTorch model to optimize
        config: Training configuration

    Returns:
        AdamW optimizer configured with parameter groups
    """

    # Separate parameters for weight decay
    # Don't apply weight decay to biases and layer normalization parameters
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon
    )

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Optimizer created with {trainable_params:,} trainable parameters")

    return optimizer


def create_scheduler(optimizer, num_training_steps, config: TrainingConfig):
    """Create learning rate scheduler with warmup

    Uses cosine schedule with linear warmup:
    1. Linear warmup from 0 to learning_rate over warmup_steps
    2. Cosine decay from learning_rate to 0 over remaining steps

    This is the standard schedule used in transformer training.

    Args:
        optimizer: PyTorch optimizer
        num_training_steps: Total number of training steps
        config: Training configuration

    Returns:
        Learning rate scheduler
    """

    num_warmup_steps = int(num_training_steps * config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    logger.info(f"Scheduler created:")
    logger.info(f"  Total steps: {num_training_steps}")
    logger.info(f"  Warmup steps: {num_warmup_steps}")
    logger.info(f"  Warmup ratio: {config.warmup_ratio:.1%}")

    return scheduler


class CheckpointManager:
    """Manage model checkpointing

    Handles saving and loading model checkpoints with automatic
    cleanup of old checkpoints to save disk space.

    Features:
    - Saves model, optimizer, scheduler states
    - Saves training metrics
    - Maintains only the N most recent checkpoints
    - Supports loading from checkpoints

    Args:
        output_dir: Directory to save checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
    """

    def __init__(self, output_dir: str, max_checkpoints: int = 3):
        self.output_dir = output_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"CheckpointManager initialized:")
        logger.info(f"  Output dir: {output_dir}")
        logger.info(f"  Max checkpoints: {max_checkpoints}")

    def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        step: int,
        metrics: Dict
    ):
        """Save checkpoint

        Saves a complete training checkpoint including:
        - Model state dict
        - Optimizer state dict
        - Scheduler state dict
        - Training metrics (loss, accuracy, etc.)
        - Epoch and step information

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch number
            step: Current global step
            metrics: Dictionary of metrics to save
        """

        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model state
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics
        }

        torch.save(checkpoint, os.path.join(checkpoint_dir, 'pytorch_model.bin'))

        # Save config if available
        if hasattr(model, 'config'):
            model.config.save_pretrained(checkpoint_dir)

        # Track checkpoints
        self.checkpoints.append((step, checkpoint_dir))

        # Remove old checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            _, old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                import shutil
                shutil.rmtree(old_checkpoint)
                logger.info(f"Removed old checkpoint: {old_checkpoint}")

        logger.info(f"✓ Checkpoint saved: {checkpoint_dir}")
        logger.info(f"  Epoch: {epoch}, Step: {step}")
        logger.info(f"  Metrics: {metrics}")

    def load_checkpoint(self, checkpoint_path: str, model, optimizer=None, scheduler=None):
        """Load checkpoint

        Loads a saved checkpoint and restores model, optimizer, and scheduler states.

        Args:
            checkpoint_path: Path to checkpoint directory
            model: PyTorch model to load state into
            optimizer: (Optional) PyTorch optimizer to load state into
            scheduler: (Optional) Learning rate scheduler to load state into

        Returns:
            Tuple of (epoch, step, metrics) from the checkpoint
        """

        checkpoint_file = os.path.join(checkpoint_path, 'pytorch_model.bin')

        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_file, map_location='cpu')

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        step = checkpoint.get('step', 0)
        metrics = checkpoint.get('metrics', {})

        logger.info(f"✓ Checkpoint loaded:")
        logger.info(f"  Epoch: {epoch}, Step: {step}")
        logger.info(f"  Metrics: {metrics}")

        return epoch, step, metrics

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the latest checkpoint"""
        if not self.checkpoints:
            return None
        return self.checkpoints[-1][1]


def compute_num_training_steps(
    dataset_size: int,
    batch_size: int,
    num_epochs: int,
    gradient_accumulation_steps: int = 1,
    world_size: int = 1
) -> int:
    """Compute total number of training steps

    Args:
        dataset_size: Number of samples in dataset
        batch_size: Batch size per GPU
        num_epochs: Number of training epochs
        gradient_accumulation_steps: Number of gradient accumulation steps
        world_size: Number of GPUs/processes

    Returns:
        Total number of optimizer steps
    """
    steps_per_epoch = dataset_size // (batch_size * world_size)
    total_steps = steps_per_epoch * num_epochs // gradient_accumulation_steps
    return total_steps


# Unit test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("="*70)
    print("TESTING TRAINING UTILITIES")
    print("="*70)

    # Test TrainingConfig
    print("\n[1/4] Testing TrainingConfig...")
    config = TrainingConfig(
        learning_rate=1e-4,
        num_epochs=5,
        batch_size=8
    )
    print(f"✓ Config created: {config.to_dict()}")

    # Test optimizer creation
    print("\n[2/4] Testing optimizer creation...")
    dummy_model = nn.Linear(100, 10)
    optimizer = create_optimizer(dummy_model, config)
    print(f"✓ Optimizer created: {type(optimizer).__name__}")

    # Test scheduler creation
    print("\n[3/4] Testing scheduler creation...")
    scheduler = create_scheduler(optimizer, num_training_steps=1000, config=config)
    print(f"✓ Scheduler created: {type(scheduler).__name__}")

    # Test checkpoint manager
    print("\n[4/4] Testing CheckpointManager...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_manager = CheckpointManager(tmpdir, max_checkpoints=2)

        # Save checkpoint
        checkpoint_manager.save_checkpoint(
            model=dummy_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            step=100,
            metrics={'loss': 0.5}
        )

        # Load checkpoint
        epoch, step, metrics = checkpoint_manager.load_checkpoint(
            checkpoint_path=os.path.join(tmpdir, 'checkpoint-100'),
            model=dummy_model,
            optimizer=optimizer,
            scheduler=scheduler
        )

        print(f"✓ Checkpoint loaded: epoch={epoch}, step={step}, metrics={metrics}")

    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
