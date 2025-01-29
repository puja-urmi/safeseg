for alg in brats_central brats_fedavg brats_fedavg_dp
do
  sed -i "s|DATASET_ROOT|${PWD}/dataset_brats/dataset|g" configs/${alg}/app/config/config_train.json
  sed -i "s|DATALIST_ROOT|${PWD}/dataset_brats/datalist|g" configs/${alg}/app/config/config_train.json
done