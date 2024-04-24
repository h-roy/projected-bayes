import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from src.data.mnist import MNIST, get_mnist, get_rotated_mnist
from src.data.fmnist import FashionMNIST, get_fmnist, get_rotated_fmnist
from src.data.cifar10 import CIFAR10, get_cifar10, get_cifar10_corrupted
from src.data.cifar100 import CIFAR100, get_cifar100
from src.data.svhn import get_svhn
from src.data import get_dataloaders

train_set, val_set, test_set = get_dataloaders("CIFAR-10")
for batch in train_set:
    print(batch['image'].shape)
    print(batch['label'].shape)
    print(np.mean(batch['image'], axis=(0, 1, 2)))
    print(np.std(batch['image'], axis=(0, 1, 2)))
    break