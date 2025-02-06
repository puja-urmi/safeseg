#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8000M
#SBATCH --time=60:00

# Load the required modules
module load scipy-stack
source /home/psaha03/scratch/safeseg/env/bin/activate

python resize.py