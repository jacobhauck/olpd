#!/bin/bash
#SBATCH --job_name=arch_test
#SBATCH --output=log_%a.out
#SBATCH --array=1-6
#SBATCH --gpus=1
#SBATCH --mem=5000
#SBATCH --time=500

mlx pd2d arch_test/mfear$SLURM_ARRAY_TASK_ID