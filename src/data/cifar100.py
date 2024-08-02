import torch
import torchvision
from torchvision import transforms as T
import torch.nn.functional as F
from PIL import Image
import numpy as np
from src.data.utils import get_loader, get_subset_data, image_to_numpy, ToChannelsLast, numpy_collate_fn
from torchvision.datasets import CIFAR100
from src.data.utils import set_seed
import torch.utils.data as data
from typing import Literal 

def get_cifar100(
        train_batch_size = 128,
        val_batch_size = 128,
        purp: Literal["train", "sample"] = "train",
        transform = None,
        seed = 0,
        download: bool = True,
        data_path="/dtu/p1/hroy/data",
        n_samples_per_class: int = None,
    ):
    n_classes = 100
    train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=download)
    means = (train_dataset.data / 255.0).mean(axis=(0,1,2))
    std = (train_dataset.data / 255.0).std(axis=(0,1,2))
    normalize = image_to_numpy(means, std)
    test_transform = normalize
    # For training, we add some augmentation. Networks are too powerful and would overfit.
    if purp == "train":
        train_transform = T.Compose([T.RandomHorizontalFlip(),
                                    T.RandomResizedCrop((32,32), scale=(0.8,1.0), ratio=(0.9,1.1)),
                                    normalize
                                    ])
    elif purp == "sample":
        train_transform = test_transform
    if transform is not None:
        train_transform = T.Compose([train_transform, transform])
        test_transform = T.Compose([test_transform, transform])

    train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True, transform=train_transform, download=download)
    val_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True, transform=test_transform, download=download)
    set_seed(seed)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    set_seed(seed)
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])
    test_set = torchvision.datasets.CIFAR100(root=data_path, train=False, transform=test_transform, download=download)
    if n_samples_per_class is not None:
        set_seed(seed)
        n_data = n_samples_per_class * 10
        train_set, _ = torch.utils.data.random_split(train_dataset, [n_data, len(train_dataset)-n_data])
        val_set, _  = torch.utils.data.random_split(val_dataset, [n_data, len(val_dataset)-n_data])
        test_set, _  = torch.utils.data.random_split(test_set, [n_data, len(test_set)-n_data])

    train_set.dataset.targets = torch.nn.functional.one_hot(torch.tensor(train_set.dataset.targets), n_classes).numpy()
    val_set.dataset.targets = torch.nn.functional.one_hot(torch.tensor(val_set.dataset.targets), n_classes).numpy()
    test_set.targets = torch.nn.functional.one_hot(torch.tensor(test_set.targets), n_classes).numpy()
    if train_batch_size < 0:
        train_batch_size = len(train_set)
    if purp == "train":
        train_loader = data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4, collate_fn=numpy_collate_fn)
    elif purp == "sample":
        train_loader = data.DataLoader(train_set, batch_size=train_batch_size, drop_last=True, pin_memory=True, num_workers=4, collate_fn=numpy_collate_fn, sampler = data.sampler.SequentialSampler(train_set))
    val_loader = data.DataLoader(val_set, batch_size=val_batch_size, shuffle=False, drop_last=False, num_workers=4, collate_fn=numpy_collate_fn)
    test_loader = data.DataLoader(test_set, batch_size=val_batch_size, shuffle=False, drop_last=False, num_workers=4, collate_fn=numpy_collate_fn)
    return train_loader, val_loader, test_loader

