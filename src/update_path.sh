for alg in kits_central kits_fedavg kits_fedavg_dp
do
  sed -i "s|DATASET_ROOT|/home/psaha03/scratch/dataset_kits23/dataset|g" configs/${alg}/app/config/config_train.json
  sed -i "s|DATALIST_ROOT|/home/psaha03/scratch/dataset_kits23/datalist|g" configs/${alg}/app/config/config_train.json
done