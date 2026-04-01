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

module load anaconda3/2023.03
module load cuda/11.8

source ~/parallelml_final/venv/bin/activate

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: 4"

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/training/ddp_trainer.py \
    --dataset_path data/raw/pmc_oa_100k \
    --output_dir outputs/checkpoints/ddp_4gpu \
    --num_gpus 4 \
    --batch_size 4
