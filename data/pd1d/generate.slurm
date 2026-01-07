#!/bin/bash
#SBATCH --job-name=gen-pd1d
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --out=data/pd1d/gen.out

module load matlab/2024a
matlab -nodesktotp -nosplash -r "OLPDGenerate1; exit;"
