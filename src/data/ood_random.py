import os
from pathlib import Path
from typing import Literal
import jax
import torch
import torchvision
from src.data.imagenet import Imagenet_testset
from src.data.mnist import MNIST
from src.data.svhn import SVHN
from src.data.utils import get_loader, get_subset_data, RotationTransform, image_to_numpy, numpy_collate_fn, set_seed
from src.data.fmnist import FashionMNIST
import torch.utils.data as data
import torch.utils.data as data
from torchvision import transforms as T
import torch.nn.functional as F
from torchvision import datasets
from PIL import Image
import numpy as np






class Random_OOD(torch.utils.data.Dataset):
    def __init__(
        self,
        data_shape: tuple,
        num_classes: int,
        size: int,
        seed: int = 0,
    ):
        set_seed(seed)
        self.dataset = torch.randn(size=(size, *data_shape))
        self.targets = torch.randint(low=0, high=num_classes, size=(size,))
        self.targets = torch.nn.functional.one_hot(torch.tensor(self.targets), num_classes).numpy()

    def __getitem__(self, index):
        img, target = self.dataset[index], self.targets[index]
        return img, target

    def __len__(self):
        return len(self.dataset)
    
def get_ood_rand_dataset(
        batch_size: int,
        data_shape: tuple,
        num_classes: int,
        size: int,
        seed: int = 0,

):
    dataset = Random_OOD(data_shape, num_classes, size, seed)
    if batch_size < 1:
        batch_size = len(dataset)
    data_loader = data.DataLoader(dataset, batch_size=batch_size,drop_last=True, pin_memory=True, num_workers=4, collate_fn=numpy_collate_fn, sampler = data.sampler.SequentialSampler(dataset))
    return data_loader


class Random_perturbation(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_name,
            variance,
            n_samples_per_class,
            seed: int = 0,
    ):
        set_seed(seed)
        self.dataset_name = dataset_name
        if dataset_name == "MNIST":
            dataclass = MNIST(train=True,n_samples_per_class=n_samples_per_class,classes=list(range(10)),seed=seed,download=True, data_path='/dtu/p1/hroy/data')
            self.dataset = dataclass.data + torch.randn_like(dataclass.data)*variance
            self.targets = dataclass.targets
        elif dataset_name == "FMNIST":
            dataclass = FashionMNIST(train=True,n_samples_per_class=n_samples_per_class,classes=list(range(10)),seed=seed,download=True, data_path='/dtu/p1/hroy/data')
            self.dataset = dataclass.data + torch.randn_like(dataclass.data)*variance
            self.targets = dataclass.targets
        elif dataset_name == "CIFAR-10":
            dataclass = torchvision.datasets.CIFAR10(root='/dtu/p1/hroy/data', train=True, download=True)
            means = (dataclass.data / 255.0).mean(axis=(0,1,2))
            std = (dataclass.data / 255.0).std(axis=(0,1,2))
            normalize = image_to_numpy(means, std)
            transform = normalize
            train_dataset = torchvision.datasets.CIFAR10(root='/dtu/p1/hroy/data', train=True, transform=transform, download=True)
            if n_samples_per_class is not None:
                n_data = n_samples_per_class * 10
                train_set, _ = torch.utils.data.random_split(train_dataset, [n_data, len(train_dataset)-n_data])
            train_set.dataset.targets = torch.nn.functional.one_hot(torch.tensor(train_set.dataset.targets), 10).numpy()
            self.dataset = torch.Tensor(train_set.dataset.data) + torch.randn_like(torch.Tensor(train_set.dataset.data))*variance
            self.dataset = self.dataset.numpy()
            self.targets = train_set.dataset.targets

        elif dataset_name == "SVHN":
            dataclass = SVHN(root='/dtu/p1/hroy/data', split='train', download=True)
            means = (dataclass.data / 255.0).mean(axis=(0,1,2))
            std = (dataclass.data / 255.0).std(axis=(0,1,2))
            normalize = image_to_numpy(means, std)
            transform = normalize
            train_dataset = SVHN(root='/dtu/p1/hroy/data', split='train', transform=transform, download=True)
            if n_samples_per_class is not None:
                n_data = n_samples_per_class * 10
                train_set, _ = torch.utils.data.random_split(train_dataset, [n_data, len(train_dataset)-n_data])
            train_set.dataset.targets = torch.nn.functional.one_hot(torch.tensor(train_set.dataset.targets), 10).numpy()
            self.dataset = train_set.dataset.data + torch.randn_like(torch.Tensor(train_set.dataset.data))*variance
            self.targets = train_set.dataset.targets

    def __getitem__(self, index):
        img, target = self.dataset[index], self.targets[index]
        return img, target

    def __len__(self):
        return len(self.dataset)
    
def get_ood_perturbation_dataset(
        dataset_name,
        variance,
        n_samples_per_class,
        batch_size,
        seed: int = 0,
):
    if dataset_name != "ImageNet":
        dataset = Random_perturbation(dataset_name, variance, n_samples_per_class, seed)
        if batch_size < 1:
            batch_size = len(dataset)
        data_loader = data.DataLoader(dataset, batch_size=batch_size,drop_last=True, pin_memory=True, num_workers=4, collate_fn=numpy_collate_fn, sampler = data.sampler.SequentialSampler(dataset))
        return data_loader
    else:
        set_seed(seed)
        n_classes = 1000
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        def random_perturbation(image):
            return image + np.random.normal(loc=0., scale=np.sqrt(variance), size=image.shape)

        normalize = image_to_numpy(mean, std)
        transform = T.Compose([T.Resize(256), T.CenterCrop(224), normalize, random_perturbation])
        def target_transform(y):
            return F.one_hot(torch.tensor(y), n_classes).numpy()

        dataset = Imagenet_testset(
        root_dir="/dtu/imagenet/ILSVRC/Data/CLS-LOC/val/", label_file="/dtu/p1/hroy/projected-bayes/src/data/val_label.txt", 
        transform=transform, test_transform=target_transform
        )
        if n_samples_per_class is not None:
            n_data = int(n_samples_per_class * n_classes)
            dataset, _ = torch.utils.data.random_split(
                dataset, [n_data, len(dataset) - n_data]
            )
        return data.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            pin_memory=True,
            collate_fn=numpy_collate_fn,
        )

