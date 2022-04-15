from base.Network import MYNET
import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

class MiniImageNet(Dataset):

    def __init__(self, root='./data', train=True,
                 transform=None,
                 index_path=None, index=None, base_sess=None):
        if train:
            setname = 'train'
        else:
            setname = 'test'
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.IMAGE_PATH = os.path.join(root, 'miniimagenet/images')
        self.SPLIT_PATH = os.path.join(root, 'miniimagenet/split')

        id2label={}
        id2label_path = os.path.join(root, 'miniimagenet/id2label.txt')
        lines = [x.strip() for x in open(id2label_path, 'r').readlines()][:]
        for l in lines:
            id, label = l.split(',')
            id2label[id] = label

        csv_path = osp.join(self.SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.data = []
        self.targets = []
        self.data2label = {}

        for l in lines:
            name, id = l.split(',')
            path = osp.join(self.IMAGE_PATH, name)
            self.data.append(path)
            self.targets.append(int(id2label[id]))
            self.data2label[path] = int(id2label[id])

            # ttt = osp.join(self.IMAGE_PATH, 'n0761348000001221.jpg')
            # print(self.data2label[ttt])

        if train:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)

        else:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)


    def SelectfromTxt(self, data2label, index_path):
        index=[]
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
        for line in lines:
            index.append(line.split('/')[3])
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.IMAGE_PATH, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        print(index)
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets

def show_statistics(x, mode='integral'):
    model = torch.load('../../checkpoint/mini_imagenet/base/ft_cos-ft_cos-data_init-start_0/Epo_100-Lr_0.1000-Step_40-Gam_0.10-Bs_128-Mom_0.90-T_16.00-ftLR_0.100-ftEpoch_150/session0_max_acc.pth')
    if mode == 'integral':
        output = model(x).shape
        return output

if __name__ == '__main__':
    txt_path = "../../data/index_list/mini_imagenet/session_1.txt"
    # class_index = open(txt_path).read().splitlines()
    base_class = 100
    class_index = np.arange(base_class)
    dataroot = '../../data'
    batch_size_base = 400
    index=np.arange(0,60)
    trainset = MiniImageNet(root=dataroot, train=True, transform=None, index=index, base_sess=True)

    # cls = np.unique(trainset.targets)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=False, num_workers=8,
                                              pin_memory=True)
    print(len(trainloader.dataset.data))

    # txt_path = "../../data/index_list/cifar100/session_2.txt"
    # # class_index = open(txt_path).read().splitlines()
    # class_index = np.arange(base_class)
    # trainset = CIFAR100(root=dataroot, train=True, download=True, transform=None, index=class_index,
    #                     base_sess=True)
    # cls = np.unique(trainset.targets)
    # trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
    #                                           pin_memory=True)
