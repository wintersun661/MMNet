## description
codes for MMNet and its further adaptation

train:

python train.py --lr 0.0005 --backbone_name resnet101 --ckp_path ckp_name_u_specify_it --gpu 0

backbone_name: resnet101, resnet50, resnext-101, fcn-resnet101
gpu: number specified

test:
python test.py --alpha 0.05 --ckp_path ckp_path_ckp_saved --ckp_type ckp_file_name

