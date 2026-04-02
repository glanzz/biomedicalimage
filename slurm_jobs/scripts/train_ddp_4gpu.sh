#!/bin/bash
#SBATCH --job-name=ddp-4gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_jobs/logs/%j_ddp_4gpu.out
#SBATCH --error=slurm_jobs/logs/%j_ddp_4gpu.err

# DDP Training Script for 4 GPUs

echo "=============================================="
echo "DDP Training - 4 GPUs"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""

# Load modules
module load anaconda3/2023.03
module load cuda/11.8

# Activate environment
source ~/parallelml_final/venv/bin/activate

# Set environment variables
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo ""

# Create log directory
mkdir -p slurm_jobs/logs

# Run training
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/training/ddp_trainer.py \
    --dataset_path data/raw/pmc_oa_100k \
    --output_dir outputs/checkpoints/ddp_4gpu \
    --num_gpus 4 \
    --batch_size 4 \
    --num_epochs 3 \
    --bf16

echo ""
echo "Training complete! Checkpoints: outputs/checkpoints/ddp_4gpu"
