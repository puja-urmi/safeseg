#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:2
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32000M
#SBATCH --time=3-00:00:00
#SBATCH --output=kits_fedavg_4_%J.log   

# Load the required modules
module load python/3.11
python3 -m venv /home/psaha03/scratch/safeseg/env
source /home/psaha03/scratch/safeseg/env/bin/activate

# Display GPU information
nvidia-smi

# Execute the Python script
pip install -r requirements.txt
mkdir /home/psaha03/scratch/workspace_kits_fedavg_4
nvflare simulator './configs' -w '/home/psaha03/scratch/workspace_kits_fedavg_4' -n 4 -t 4 -gpu 0,0,1,1