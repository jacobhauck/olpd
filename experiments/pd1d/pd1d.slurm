#!/bin/bash
#SBATCH --job-name=pd1d
#SBATCH --output=experiments/pd1d/pd1d.out
#SBATCH --error=experiments/pd1d/pd1d.err
#SBATCH --gpus=1
#SBATCH --time=40
#SBATCH --mem=3000

python -m mlx.run pd1d mfear 
