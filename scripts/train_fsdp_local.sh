#!/bin/bash
# FSDP Local Training Script
# For testing FSDP on local multi-GPU machines without SLURM
#
# Usage:
#   bash scripts/train_fsdp_local.sh                    # Auto-detect GPUs
#   GPUS_TO_USE=2 bash scripts/train_fsdp_local.sh     # Use 2 GPUs
#   DATASET_PATH=data/raw/pmc_oa_10k bash scripts/train_fsdp_local.sh  # Custom dataset

set -e  # Exit on error

# Configuration
DATASET_PATH=${DATASET_PATH:-"data/raw/pmc_oa_100k"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/checkpoints/fsdp_local"}
BATCH_SIZE=${BATCH_SIZE:-8}  # FSDP allows larger batches
NUM_EPOCHS=${NUM_EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-2}

# Auto-detect number of GPUs
if [ -z "$GPUS_TO_USE" ]; then
    GPUS_TO_USE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Auto-detected $GPUS_TO_USE GPUs"
fi

# Validate GPU count
if [ "$GPUS_TO_USE" -lt 1 ]; then
    echo "❌ Error: No GPUs detected!"
    exit 1
fi

# Set distributed training environment
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=$GPUS_TO_USE

echo "=================================================="
echo "FSDP Local Training Configuration"
echo "=================================================="
echo "GPUs: $GPUS_TO_USE"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Global batch size: $((BATCH_SIZE * GPUS_TO_USE))"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "Effective batch size: $((BATCH_SIZE * GPUS_TO_USE * GRADIENT_ACCUMULATION))"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "=================================================="

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ Error: Dataset not found at $DATASET_PATH"
    echo "   Please download the dataset first or specify correct path"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "outputs/logs"

# Print GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo "=================================================="

# Start time
START_TIME=$(date +%s)
echo "Start time: $(date)"
echo "=================================================="

# Run FSDP training
echo "Launching FSDP training with $GPUS_TO_USE GPUs..."
echo ""

torchrun \
    --nproc_per_node=$GPUS_TO_USE \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/training/fsdp_trainer.py \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_gpus $GPUS_TO_USE \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    2>&1 | tee "$OUTPUT_DIR/training.log"

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

# End time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ FSDP Training completed successfully!"
else
    echo "❌ FSDP Training failed with exit code $EXIT_CODE"
fi
echo "End time: $(date)"
echo "Duration: $((DURATION / 60)) minutes $((DURATION % 60)) seconds"
echo "=================================================="

# Print final GPU memory usage
echo "Final GPU Memory Usage:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "=================================================="

# Display performance metrics if available
PERF_FILE="$OUTPUT_DIR/fsdp_performance_${GPUS_TO_USE}gpu.json"
if [ -f "$PERF_FILE" ]; then
    echo "Performance Metrics:"
    echo ""
    cat "$PERF_FILE"
    echo ""
    echo "=================================================="
fi

# Display saved checkpoints
echo "Saved Checkpoints:"
ls -lh "$OUTPUT_DIR"/checkpoint-* 2>/dev/null || echo "No checkpoints found"
echo "=================================================="

exit $EXIT_CODE
