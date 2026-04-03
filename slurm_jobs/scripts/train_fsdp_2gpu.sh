#!/bin/bash
#SBATCH --job-name=fsdp-2gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_jobs/logs/%j_fsdp_2gpu.out
#SBATCH --error=slurm_jobs/logs/%j_fsdp_2gpu.err

# FSDP 2-GPU Training Script
# Memory advantage: ~50% less memory per GPU vs DDP
# Enables 2x larger batch sizes

echo "=================================================="
echo "FSDP 2-GPU Training"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 2 x A100"
echo "Memory: 80G"
echo "=================================================="

# Load modules
module load anaconda3/2023.03
module load cuda/11.8

# Activate environment
source ~/parallelml_final/venv/bin/activate

# Set environment variables for distributed training
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export WORLD_SIZE=2

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World Size: $WORLD_SIZE"
echo "=================================================="

# Print GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "=================================================="

# Start time
START_TIME=$(date +%s)
echo "Start time: $(date)"

# Run FSDP training with torchrun
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/training/fsdp_trainer.py \
    --dataset_path data/raw/pmc_oa_100k \
    --output_dir outputs/checkpoints/fsdp_2gpu \
    --num_gpus 2 \
    --batch_size 8 \
    --num_epochs 3 \
    --learning_rate 2e-4 \
    --gradient_accumulation_steps 2

# End time and duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "=================================================="
echo "Training completed!"
echo "End time: $(date)"
echo "Duration: $((DURATION / 60)) minutes"
echo "=================================================="

# Print final GPU memory usage
echo "Final GPU Memory:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
echo "=================================================="
