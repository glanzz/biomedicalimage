#!/bin/bash
#SBATCH --job-name=biomedvlm-1gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_jobs/logs/%j_single_gpu.out
#SBATCH --error=slurm_jobs/logs/%j_single_gpu.err

# Load modules
module load anaconda3/2023.03
module load cuda/11.8

# Activate environment
source ~/parallelml_final/venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Single GPU Training"

# Run training
python -m src.training.single_gpu_trainer \
    --dataset_path data/raw/pmc_oa_100k \
    --output_dir outputs/checkpoints/single_gpu \
    --batch_size 4 \
    --num_epochs 3 \
    --learning_rate 2e-4 \
    --gradient_accumulation_steps 4 \
    --bf16 \
    --logging_steps 100 \
    --eval_steps 500 \
    --save_steps 1000
