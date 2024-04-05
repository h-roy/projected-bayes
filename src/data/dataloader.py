from .torch_datasets import MNIST, FashionMNIST, CIFAR10, ImageNette
from .utils import get_mean_and_std, numpy_collate_fn
from typing import Literal
from torch.utils import data
import torch

DATASETS = {"MNIST": MNIST, "FMNIST": FashionMNIST, "CIFAR10": CIFAR10, "ImageNette": ImageNette}


def get_dataloaders(
    dataset: Literal["MNIST", "FMNIST", "CIFAR10", "ImageNette"],
    bs: int,
    data_path: str,
    seed: int,
    val_frac: float = 0.1,
    n_samples: int = None,
    cls: int = None,
):
    train_stats = None

    if not dataset in ["MNIST", "FMNIST"]:
        train_stats = get_mean_and_std(
            data_train=DATASETS[dataset](path_root=data_path, set_purp="train", n_samples=n_samples, cls=cls),
            val_frac=val_frac,
            seed=seed,
        )
        print(f"Normalizing with mean = {train_stats['mean']} and  std = {train_stats['std']} ")

    data_train = DATASETS[dataset](
        path_root=data_path, set_purp="train", n_samples=n_samples, cls=cls, normalizing_stats=train_stats
    )
    data_val = DATASETS[dataset](
        path_root=data_path, set_purp="val", n_samples=n_samples, cls=cls, normalizing_stats=train_stats
    )
    data_test = DATASETS[dataset](
        path_root=data_path, set_purp="test", n_samples=None, cls=cls, normalizing_stats=train_stats
    )

    len_val = int(len(data_train) * val_frac)
    len_train = len(data_train) - len_val

    data_train, _ = data.random_split(data_train, [len_train, len_val], generator=torch.Generator().manual_seed(seed))
    _, data_val = data.random_split(data_val, [len_train, len_val], generator=torch.Generator().manual_seed(seed))

    train_loader = data.DataLoader(
        data_train,
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        collate_fn=numpy_collate_fn,
        num_workers=8,
        persistent_workers=True,
    )

    val_loader = data.DataLoader(
        data_val,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        collate_fn=numpy_collate_fn,
        num_workers=4,
        persistent_workers=True,
    )

    test_loader = data.DataLoader(
        data_test,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        collate_fn=numpy_collate_fn,
        num_workers=4,
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader