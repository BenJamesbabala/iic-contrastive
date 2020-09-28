import torch
from collections import namedtuple
from torchvision.datasets import CIFAR10
import torchvision.transforms as T 
from archs import *
from data import *
from data.transforms import *
from data.make_data import *

config = {
    "output_k_A": 70,
    "output_k_B": 10,
    "output_k_C": 128,
    "gt_K": 10, #grond truth classes
    "input_sz": 32,
    "rand_crop_sz": 20,
    "fluid_warp": False,
    "crop_orig": False,
    "cutout": False,
    "cutout_p": False,
    "cutout_max_box": False,
    "demean": False,
    "per_img_demean": False,
    "simclr_bs": 512,
    "iic_dataloader_bs": 132, # iic_bs / num_dataloaders
    "iic_bs": 660,
    "batchnorm_track" : True,
    "include_rgb" : False,
    "in_channels" : 1,
    "num_sub_heads" : 5,
    "num_dataloaders": 2,
    "dataset": 'CIFAR10',
    "dataset_root": './tmp/',
    "train_partitions_head_A": [True,False],
    "train_partitions_head_B": [True,False],
    "mapping_assignment_partitions": [True,False],
    "mapping_test_partitions": [True,False],
    "eval_mode": 'orig', #orig for original, hung for hungarian
    "mode": 'IID',
    "test_code": True,
    "num_epochs": 200,
    "device": 'cuda',
    "warmup_epochs": 10,
    "lr_iic": 0.1,
    "lr_simclr": 0.1,
    "num_epochs_head": 1
}
config = namedtuple('Config',config.keys())(*config.values())

def output_sizes():
    t = T.Compose([T.Resize(config.input_sz), T.ToTensor()])
    train_s = CIFAR10('tmp/', train=True, transform=t, download=True)
    train_s = torch.utils.data.DataLoader(train_s, batch_size=config.batch_sz, shuffle=True)
    img, lbl = next(iter(train_s))
    print('Image batch: ', img.size())
    print('Label batch: ', lbl.size())
    net = ClusterNetMulHead(config)
    y1 = net.forward(img,head='A')
    y2 = net.forward(img,head='B')
    y3 = net.forward(img,head='C')
    print(len(y1), ' elems of size ', y1[0].size())
    print(len(y2), ' elems of size ', y2[0].size())
    print(len(y3), ' elems of size ', y3[0].size())

def greyscale_sizes():
    t = T.Compose([custom_greyscale_to_tensor(include_rgb=False)])
    train_s = CIFAR10('tmp/', train=True, transform=t, download=True)
    train_s = torch.utils.data.DataLoader(train_s, batch_size=config.iic_bs, shuffle=True)
    img, lbl = next(iter(train_s))
    print('Image batch: ', img.size())
    print('Label batch: ', lbl.size())
    net = ClusterNetMulHead(config)
    y1 = net.forward(img,head='A')
    y2 = net.forward(img,head='B')
    y3 = net.forward(img,head='C')
    print(len(y1), ' elems of size ', y1[0].size())
    print(len(y2), ' elems of size ', y2[0].size())
    print(len(y3), ' elems of size ', y3[0].size())

def data():
    d_a, d_b, d_simclr, d_map, d_test = cluster_twohead_create_dataloaders(config)

data()