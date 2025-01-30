#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100l:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32000M
#SBATCH --time=120:00
#SBATCH --output=training_%J.log   

# Load the required modules

source /home/psaha03/scratch/safeseg/env/bin/activate

python segresnet.py