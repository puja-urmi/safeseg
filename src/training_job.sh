#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100:4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32000M
#SBATCH --time=120:00
#SBATCH --output=brats_fedavg%_J.log   

# Load the required modules
module load python/3.11
python3 -m venv /home/psaha03/scratch/safeseg/env
source /home/psaha03/scratch/safeseg/env/bin/activate

# Display GPU information
nvidia-smi

# Execute the Python script
pip install -r requirements.txt
mkdir /home/psaha03/scratch/workspace_brats_fedavg
nvflare simulator './configs/brats_fedavg' -w '/home/psaha03/scratch/workspace_brats_fedavg/brats_fedavg' -n 4 -t 4 -gpu 0,1,2,3