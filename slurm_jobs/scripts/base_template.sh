#!/bin/bash
#SBATCH --job-name=biomedvlm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_jobs/logs/%j.out
#SBATCH --error=slurm_jobs/logs/%j.err

# Load modules
module load anaconda3/2023.03
module load cuda/11.8

# Activate environment
source ~/parallelml_final/venv/bin/activate

# Set environment variables
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

# Print debug info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Run command (to be specified by specific job scripts)
