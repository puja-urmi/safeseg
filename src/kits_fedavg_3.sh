#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100:3
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32000M
#SBATCH --time=4-00:00:00
#SBATCH --output=kits_fedavg_3_%J.log   

# Load the required modules
module load python/3.11
python3 -m venv /home/psaha03/scratch/safeseg/env
source /home/psaha03/scratch/safeseg/env/bin/activate

# Display GPU information
nvidia-smi

# Execute the Python script
pip install -r requirements.txt
mkdir /home/psaha03/scratch/workspace_kits_fedavg_3
nvflare simulator './configs' -w '/home/psaha03/scratch/workspace_kits_fedavg_3' -n 2 -t 2 -gpu 0,1