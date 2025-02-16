#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32000M
#SBATCH --time=60:00
#SBATCH --output=kits_central_test_%J.log   

# Load the required modules
module load python/3.11
python3 -m venv /home/psaha03/scratch/safeseg/env
source /home/psaha03/scratch/safeseg/env/bin/activate


workspace_path="/home/psaha03/scratch/outputs/kits/central/0-235"
dataset_path="/home/psaha03/scratch/dataset_kits23/dataset"
datalist_path="/home/psaha03/scratch/dataset_kits23/datalist"

echo "Testing Centralized"
python3 kits_3d_test_only.py --model_path "${workspace_path}/server/simulate_job/app_server/best_FL_global_model.pt" --dataset_base_dir ${dataset_path} --datalist_json_path "${datalist_path}/site-test.json"
