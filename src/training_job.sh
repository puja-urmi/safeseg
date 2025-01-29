#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32000M
#SBATCH --time=120:00
#SBATCH --output=training_%J.log   

# Load the required modules
module load python/3.11
python3 -m venv /home/psaha03/scratch/safeseg/env
source /home/psaha03/scratch/safeseg/env/bin/activate

# Display GPU information
nvidia-smi

# Execute the Python script
pip install -r requirements.txt
mkdir ./workspace_brats
python3 -u -m nvflare.private.fed.app.simulator.simulator './configs/brats_central' -w './workspace_brats/brats_central' -n 1 -t 1 -gpu 0