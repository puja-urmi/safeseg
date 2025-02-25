#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32000M
#SBATCH --time=0-02:00:00
#SBATCH --output=brats_central_test_%J.log   



workspace_path="/home/psaha03/scratch/workspace_brats_central"
dataset_path="/home/psaha03/scratch/dataset_brats24/dataset"
datalist_path="/home/psaha03/scratch/dataset_brats24/datalist"

echo "Centralized"
python3 brats_3d_test_only.py --model_path "${workspace_path}/server/simulate_job/app_server/best_FL_global_model.pt" --dataset_base_dir ${dataset_path} --datalist_json_path "${datalist_path}/site-test.json"
