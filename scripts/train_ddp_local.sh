#!/bin/bash

# Local DDP training script (for machines with multiple GPUs)
# This script can be used to test DDP training locally without SLURM

echo "=============================================="
echo "Local DDP Training Script"
echo "=============================================="
echo ""

# Check for GPUs
if ! command -v nvidia-smi &> /dev/null
then
    echo "Error: nvidia-smi not found. GPU may not be available."
    exit 1
fi

# Count available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Available GPUs: $NUM_GPUS"

if [ "$NUM_GPUS" -lt 2 ]; then
    echo "Error: DDP requires at least 2 GPUs. Found: $NUM_GPUS"
    echo "Use scripts/train_single_gpu.sh for single GPU training"
    exit 1
fi

# Configuration
DATASET_PATH="${DATASET_PATH:-data/raw/pmc_oa_100k}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/checkpoints/ddp_local}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
GPUS_TO_USE="${GPUS_TO_USE:-$NUM_GPUS}"

# Limit to available GPUs
if [ "$GPUS_TO_USE" -gt "$NUM_GPUS" ]; then
    echo "Warning: Requested $GPUS_TO_USE GPUs but only $NUM_GPUS available"
    GPUS_TO_USE=$NUM_GPUS
fi

echo ""
echo "Training Configuration:"
echo "  Dataset: $DATASET_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  GPUs: $GPUS_TO_USE"
echo "  Total batch size: $((BATCH_SIZE * GPUS_TO_USE))"
echo "  Epochs: $NUM_EPOCHS"
echo ""

# Show GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -n "$GPUS_TO_USE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=12355

echo "Starting DDP training with $GPUS_TO_USE GPUs..."
echo ""

# Launch training with torchrun
torchrun \
    --nproc_per_node="$GPUS_TO_USE" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/training/ddp_trainer.py \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_gpus "$GPUS_TO_USE" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --bf16

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "Performance metrics: $OUTPUT_DIR/ddp_performance_${GPUS_TO_USE}gpu.json"
echo ""
