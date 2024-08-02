import os
from typing import Literal
from PIL import Image

import torch
import torchvision
from src.data.utils import image_to_numpy, numpy_collate_fn
from src.helper import set_seed
import torch.utils.data as data
from torchvision import transforms as T
import torch.nn.functional as F

from torchvision import datasets
import jax
import jax.numpy as jnp



def ImageNet1k_loaders(batch_size: int = 128,
                       purp: Literal["train", "sample"] = "train",
                       seed: int = 0, 
                       n_samples_per_class=None):
    set_seed(seed)
    n_classes = 1000
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = image_to_numpy(mean, std)
    if purp == "train":
        train_transform = T.Compose(
            [T.RandomResizedCrop(224), T.RandomHorizontalFlip(), normalize]
        )
    elif purp == "sample":
        train_transform = T.Compose(
            [T.CenterCrop(224), normalize]
        )

    # test_transform = T.Compose([T.Resize(256), T.CenterCrop(224), normalize])
    def target_transform(y):
        return F.one_hot(torch.tensor(y), n_classes).numpy()

    # target_transform = lambda y: F.one_hot(torch.tensor(y), n_classes).numpy()
    train_path = "/dtu/imagenet/ILSVRC/Data/CLS-LOC/train/"
    train_dataset = datasets.ImageFolder(
        train_path, transform=train_transform, target_transform=target_transform
    )
    if n_samples_per_class is not None:
        set_seed(seed)
        n_data = int(n_samples_per_class * n_classes)
        train_dataset, _ = torch.utils.data.random_split(
            train_dataset, [n_data, len(train_dataset) - n_data]
        )

    if purp == "train":
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4, collate_fn=numpy_collate_fn)
    elif purp == "sample":
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True, pin_memory=True, collate_fn=numpy_collate_fn, sampler = data.sampler.SequentialSampler(train_dataset))

    return train_loader


def split_train_test(inputs, targets, *, train=0.9):
    num_data = len(inputs)

    # Select subsets
    num_train = int(train * num_data)
    train_set = inputs[:num_train], targets[:num_train]
    test_set = inputs[num_train:], targets[num_train:]
    return train_set, test_set


def split_train_test_shuffle(key, /, inputs, targets, *, train=0.9):
    num_data = len(inputs)

    # Shuffle
    p = jax.random.permutation(key, jnp.arange(num_data))
    inputs, targets = inputs[p], targets[p]

    # Select subsets
    num_train = int(train * num_data)
    train_set = inputs[:num_train], targets[:num_train]
    test_set = inputs[num_train:], targets[num_train:]
    return train_set, test_set


class Imagenet_testset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir="/dtu/imagenet/ILSVRC/Data/CLS-LOC/val/",
        label_file="/dtu/p1/hroy/projected-bayes/src/data/val_label.txt",
        transform=None,
        test_transform=None,
    ):
        self.root_dir = root_dir
        self.label_file = label_file
        self.transform = transform
        self.test_transform = test_transform
        self.size = 0
        self.files_list = []

        if not os.path.isfile(self.label_file):
            print(self.label_file + "does not exist!")
        file = open(self.label_file)
        for f in file:
            self.files_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.root_dir + self.files_list[idx].split(" ")[0]
        if not os.path.isfile(image_path):
            print(image_path + "does not exist!")
            return None
        # image = io.imread(image_path)   # use skitimage
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        label = int(self.files_list[idx].split(" ")[1])
        if self.transform:
            image = self.transform(image)
        if self.test_transform:
            label = self.test_transform(label)
        return (image, label)


def get_imagenet_test_loader(batch_size=128, seed=0, n_samples_per_class=None):
    set_seed(seed)
    n_classes = 1000
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = image_to_numpy(mean, std)
    test_transform = T.Compose([T.Resize(256), T.CenterCrop(224), normalize])

    def target_transform(y):
        return F.one_hot(torch.tensor(y), n_classes).numpy()

    test_set = Imagenet_testset(
        transform=test_transform, test_transform=target_transform
    )

    if n_samples_per_class is not None:
        set_seed(seed)
        n_data = int(n_samples_per_class * n_classes)
        test_set, _ = torch.utils.data.random_split(
            test_set, [n_data, len(test_set) - n_data]
        )

    return data.DataLoader(
        test_set,
        batch_size=batch_size,
        drop_last=True,
        pin_memory=True,
        collate_fn=numpy_collate_fn,
    )


def get_places365(
    batch_size=128,
    seed=0,
    download: bool = False,
    data_path="/dtu/p1/hroy/data",
    n_samples_per_class=None,
):
    n_classes = 1000
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = image_to_numpy(mean, std)
    transform = normalize

    def target_transform(y):
        return F.one_hot(torch.tensor(y), n_classes).numpy()

    # For training, we add some augmentation. Networks are too powerful and would overfit.

    dataset = torchvision.datasets.Places365(
        root=data_path,
        split="val",
        transform=transform,
        target_transform=target_transform,
        small=True,
        download=download,
    )
    if n_samples_per_class is not None:
        set_seed(seed)
        n_data = int(n_samples_per_class * 10)
        dataset, _ = torch.utils.data.random_split(
            dataset, [n_data, len(dataset) - n_data]
        )
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=numpy_collate_fn,
    )
    return loader
