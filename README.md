# MMNet
This repo is the official implementation of ICCV 2021 paper ["Multi-scale Matching Networks for Semantic Correspondence."](https://arxiv.org/pdf/2108.00211.pdf).


## Pre-requisite
```bash
conda create -n mmnet python==3.8.0
conda activate mmnet
conda install torch==1.8.1 torchvision==0.9.1
pip install matplotlib scikit-image pandas
```
for installation of gluoncvth (fcn-resnet101):
```bash
git clone https://github.com/StacyYang/gluoncv-torch.git
cd gluoncv-torch
python setup.py install
```


## Reproduction
### for training
python train.py --seed 0 --lr 0.0005

### for test
Trained models are available on [[google drive](https://drive.google.com/drive/folders/13rBUJLxWbwgOHihWCZvcLnyDBN_guQFq?usp=sharing)].


pascal with fcn-resnet101 backbone(PCK@0.05:81.6%):
```bash
python test.py --alpha 0.05 --backbone fcn-resnet101 --ckp_name path\to\ckp_pascal_fcnres101.pth --resize 224,320
```

spair with fcn-resnet101 backbone(PCK@0.1):
```bash
python test.py --alpha 0.05 --benchmark spair --backbone fcn-resnet101 --ckp_name path\to\ckp_spair_fcnres101.pth --resize 224,320
```

## Bibtex
If you use this code for your research, please consider citing:
````Bibtex
@article{zhao2021multi,
  title={Multi-scale Matching Networks for Semantic Correspondence},
  author={Zhao, Dongyang and Song, Ziyang and Ji, Zhenghao and Zhao, Gangming and Ge, Weifeng and Yu, Yizhou},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
````
