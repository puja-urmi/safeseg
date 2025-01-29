for alg in brats_central brats_fedavg brats_fedavg_dp
do
  sed -i "s|DATASET_ROOT|/home/psaha03/scratch/dataset_brats24/dataset|g" configs/${alg}/app/config/config_train.json
  sed -i "s|DATALIST_ROOT|/home/psaha03/scratch/dataset_brats24/datalist|g" configs/${alg}/app/config/config_train.json
done