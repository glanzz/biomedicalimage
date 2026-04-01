"""
Distributed training utilities
Helpers for DDP and FSDP setup
"""

import os
import torch
import torch.distributed as dist
from typing import Optional


def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """
    Initialize distributed training

    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Communication backend (nccl for GPU, gloo for CPU)
    """

    # Set environment variables if not already set
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'

    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # Set device
    torch.cuda.set_device(rank)

    print(f"[Rank {rank}] Distributed training initialized")
    print(f"[Rank {rank}] World size: {world_size}")
    print(f"[Rank {rank}] Backend: {backend}")


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get current process rank"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if current process is main process (rank 0)"""
    return get_rank() == 0


def barrier():
    """Synchronize all processes"""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """
    Reduce dictionary values across all processes

    Args:
        input_dict: Dictionary with tensor values
        average: Whether to average values (True) or sum (False)

    Returns:
        Reduced dictionary
    """
    world_size = get_world_size()

    if world_size < 2:
        return input_dict

    with torch.no_grad():
        names = []
        values = []

        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])

        values = torch.stack(values, dim=0)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)

        if average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}

    return reduced_dict


def print_once(*args, **kwargs):
    """Print only on main process"""
    if is_main_process():
        print(*args, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    print("Distributed utilities module")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
