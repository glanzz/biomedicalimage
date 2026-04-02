#!/bin/bash

# Single-GPU training script for biomedical vision-language model
# This script trains the model on a single GPU with optimized settings

# Exit on error
set -e

echo "=============================================="
echo "Single-GPU Training Script"
echo "=============================================="
echo ""

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null
then
    echo "Warning: nvidia-smi not found. GPU may not be available."
    echo "Training will proceed but may fall back to CPU."
else
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# Configuration
DATASET_PATH="${DATASET_PATH:-data/raw/pmc_oa_100k}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/checkpoints/single_gpu}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

echo "Training Configuration:"
echo "  Dataset: $DATASET_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Effective batch size: $((BATCH_SIZE * GRAD_ACCUM))"
echo ""

# Set GPU device (use first GPU by default)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
echo "Starting training..."
echo ""

python -m src.training.single_gpu_trainer \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --bf16 \
    --logging_steps 100 \
    --eval_steps 500 \
    --save_steps 1000

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo "Checkpoints saved to: $OUTPUT_DIR"
