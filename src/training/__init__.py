"""
Training modules
- single_gpu_trainer: Single-GPU baseline trainer
- ddp_trainer: Distributed Data Parallel trainer
- fsdp_trainer: Fully Sharded Data Parallel trainer (Phase 7)
- cpu_trainer: CPU baseline trainer (optional)
"""

from .single_gpu_trainer import SingleGPUTrainer
from .ddp_trainer import DDPTrainer, setup_distributed, cleanup_distributed

__all__ = ['SingleGPUTrainer', 'DDPTrainer', 'setup_distributed', 'cleanup_distributed']
