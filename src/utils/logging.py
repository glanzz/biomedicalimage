"""
Logging utilities for training and evaluation
Supports both console and file logging with WandB integration
"""

import logging
import os
from typing import Optional
import yaml


def setup_logger(
    name: str,
    log_dir: Optional[str] = "outputs/logs",
    level: str = "INFO"
) -> logging.Logger:
    """
    Set up logger with console and file handlers

    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger
    """

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # Remove existing handlers
    logger.handlers = []

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"{name}.log")
        )
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class MetricsLogger:
    """Logger for training metrics with WandB support"""

    def __init__(self, use_wandb: bool = True, project: str = "biomedical-image-captioning"):
        self.use_wandb = use_wandb
        self.metrics = []

        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed, disabling WandB logging")
                self.use_wandb = False

    def log(self, metrics: dict, step: int):
        """Log metrics to console and WandB"""
        self.metrics.append({'step': step, **metrics})

        # Print to console
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Step {step}: {metric_str}")

        # Log to WandB
        if self.use_wandb:
            self.wandb.log(metrics, step=step)

    def save(self, filepath: str):
        """Save metrics to JSON file"""
        import json
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)


# Example usage
if __name__ == "__main__":
    # Test logger
    logger = setup_logger("test_logger", log_dir="outputs/logs", level="INFO")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test metrics logger
    metrics_logger = MetricsLogger(use_wandb=False)
    for step in range(10):
        metrics_logger.log({
            'loss': 1.0 / (step + 1),
            'accuracy': step * 0.1
        }, step)

    metrics_logger.save("outputs/logs/test_metrics.json")
    print("Logging test complete!")
