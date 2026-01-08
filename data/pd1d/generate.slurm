#!/bin/bash
#SBATCH --job-name=gen-pd1d
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --out=generate.out

module load matlab/2024a
matlab -nosplash -r "OLPDGenerate1; exit;"
