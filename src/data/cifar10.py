from pathlib import Path
import torch
import torchvision
from torchvision import transforms as T
import torch.nn.functional as F
from PIL import Image
import numpy as np
from src.data.utils import get_loader, get_subset_data, get_mean_and_std, image_to_numpy, numpy_collate_fn
from torchvision.datasets import CIFAR10
from src.data.utils import set_seed, ToChannelsLast
import torch.utils.data as data
from typing import Literal 


def get_cifar10(
        train_batch_size = 128,
        val_batch_size = 128,
        purp: Literal["train", "sample"] = "train",
        transform = None,
        seed = 0,
        download: bool = True,
        data_path="/dtu/p1/hroy/data",
        n_samples_per_class: int = None,
    ):
    n_classes = 10
    train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=download)
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

    train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=train_transform, download=download)
    val_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=test_transform, download=download)
    set_seed(seed)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    set_seed(seed)
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])
    test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, transform=test_transform, download=download)
    if n_samples_per_class is not None:
        set_seed(seed)
        n_data = n_samples_per_class * 10
        train_set, _ = torch.utils.data.random_split(train_dataset, [n_data, len(train_dataset)-n_data])
        val_set, _  = torch.utils.data.random_split(val_dataset, [n_data, len(val_dataset)-n_data])
        test_set, _  = torch.utils.data.random_split(test_set, [n_data, len(test_set)-n_data])
        test_set.dataset.targets = torch.nn.functional.one_hot(torch.tensor(test_set.dataset.targets), n_classes).numpy()

        # test_set.targets = torch.nn.functional.one_hot(torch.tensor(test_set.dataset.targets), n_classes).numpy()
        

    train_set.dataset.targets = torch.nn.functional.one_hot(torch.tensor(train_set.dataset.targets), n_classes).numpy()
    val_set.dataset.targets = torch.nn.functional.one_hot(torch.tensor(val_set.dataset.targets), n_classes).numpy()
    if n_samples_per_class is None:
        test_set.targets = torch.nn.functional.one_hot(torch.tensor(test_set.targets), n_classes).numpy()
    if train_batch_size < 0:
        train_batch_size = len(train_set)
    if purp == "train":
        train_loader = data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4, collate_fn=numpy_collate_fn)
    elif purp == "sample":
        train_loader = data.DataLoader(train_set, batch_size=train_batch_size, drop_last=True, pin_memory=True, num_workers=4, collate_fn=numpy_collate_fn, sampler = data.sampler.SequentialSampler(train_set))
    val_loader = data.DataLoader(val_set, batch_size=val_batch_size, shuffle=False, drop_last=True, num_workers=4, collate_fn=numpy_collate_fn)
    test_loader = data.DataLoader(test_set, batch_size=val_batch_size, shuffle=False, drop_last=True, num_workers=4, collate_fn=numpy_collate_fn)
    return train_loader, val_loader, test_loader

corruption_types = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "saturate",
    "shot_noise",
    "snow",
    "spatter",
    "speckle_noise",
    "zoom_blur",
]



class CorruptedCIFAR10(CIFAR10):
    def __init__(
        self,
        corr_type,
        severity_level: int = 5,
        transform = None,
        n_samples_per_class: int = None,
        classes: list = list(range(10)),
        seed: int = 0,
        download: bool = False,
        data_path: str = "/dtu/p1/hroy/data",
    ):
        self.transform = transform


        if download:
            raise ValueError("Please download dataset manually from https://www.tensorflow.org/datasets/catalog/cifar10_corrupted")
        self.data = np.load(f"{data_path}/CIFAR-10-C/{corr_type}.npy")
        self.targets = np.load(f"{data_path}/CIFAR-10-C/labels.npy").astype(np.int64)
        self.data = self.data[(severity_level-1) * 10000 : (severity_level) * 10000]
        self.targets = self.targets[(severity_level-1) * 10000 : (severity_level) * 10000]
        self.data = torch.from_numpy(self.data).float()
        self.targets = torch.from_numpy(self.targets)

        mean, std = [x*255 for x in (0.4914, 0.4822, 0.4465)], [x*255 for x in (0.2470, 0.2435, 0.2616)]
        self.data = torchvision.transforms.functional.normalize(
            self.data.permute(0, 3, 1, 2),
            mean,
            std
        ).permute(0, 2, 3, 1).numpy()
        self.targets = torch.nn.functional.one_hot(torch.tensor(self.targets), len(classes)).numpy()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target
    
    def __len__(self):
        return len(self.data)




def get_cifar10_corrupted(
        corr_type,
        severity_level: int = 5,
        batch_size = 128,
        shuffle = False,
        seed = 0,
        download: bool = False,
        data_path="/dtu/p1/hroy/data",
    ):

    dataset = CorruptedCIFAR10(
        corr_type,
        severity_level = severity_level,
        data_path = data_path,
    )
    test_loader = get_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        seed=seed
    )
    return None, None, test_loader
