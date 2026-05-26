#!/bin/bash
#SBATCH --job-name=gamma_test
#SBATCH --output=log_%a.out
#SBATCH --array=1-5
#SBATCH --gpus=1
#SBATCH --mem=5000
#SBATCH --time=530

mlx pd2d mfear gamma$SLURM_ARRAY_TASK_ID
