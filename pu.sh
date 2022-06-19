#!/bin/sh
#SBATCH --job-name bash
#SBATCH --partition gpu
#SBATCH --output="pu.out"
#SBATCH --error="pu.err"

python ./DASPOT.py 
