# FMN

The original anonymous code submission for FMN.
## Dependencies and Installation
* python=3.6(Recommend to use Anaconda or Miniconda)
* PyTorch>=1.3
* NVIDIA GPU + CUDA
1. Clone repo
```
git clone https://github.com/ECCV2022/FMN.git
```
2. Install dependent packages
```
pip install -r requirements.txt
```
Note that FMN is only tested in Ubuntu, and may be not suitable for Window
## Running example
### Preparing datasets


We evaluate our system in several datasets, including `CUB-200-2011, CIFAR100, miniImageNet`. Please download CUB-200-2011, CIFAR100 and miniImageNet and place them under 'data/CUB_200_2011', 'data/cifar-100-python' and 'data/miniimagenet' respectively. (Note: For CIFAR100, you could simply run the corresponding script and it will be downloaded in the right place automatically)


### Training on miniimagenet

```
python train.py -project base -dataset mini_imagenet -epochs_base 100 -epochs_new 30 -gamma 0.1 -decay 0.0005 -gpu 0 -temperature 16 -basetrain False -shot 5 -lr_base 0.1 -schedule Step -step 40  -sh 0 
```

