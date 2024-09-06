from pathlib import Path
from typing import Literal
import torch
import torchvision
from src.data.utils import get_loader, get_subset_data, RotationTransform, numpy_collate_fn
from src.helper import set_seed
import torch.utils.data as data


class FashionMNIST(torch.utils.data.Dataset):
    def __init__(
        self,
        train: bool = True,
        transform = None,
        n_samples_per_class: int = None,
        classes: list = list(range(10)),
        seed: int = 0,
        download: bool = True,
        data_path: str = "/dtu/p1/hroy/data",
    ):
        self.transform = transform
        self.path = Path(data_path)
        self.dataset = torchvision.datasets.FashionMNIST(root=self.path, train=train, download=download)

        if len(classes)>=10 and n_samples_per_class is None:
            self.data, self.targets = self.dataset.data, self.dataset.targets
        else:
            self.data, self.targets = get_subset_data(self.dataset.data, self.dataset.targets, classes, n_samples_per_class=n_samples_per_class, seed=seed)
        
        self.data = (self.data.float().unsqueeze(-1) / 255.0).numpy()
        self.targets = torch.nn.functional.one_hot(torch.tensor(self.targets), len(classes)).numpy()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            # torch wants channel dimension before height and width
            img = torch.from_numpy(img).permute(2, 0, 1)
            img = self.transform(img)
            img = img.permute(1, 2, 0).numpy()
        return img, target

    def __len__(self):
        return len(self.data)


def get_fmnist(
        train_batch_size = 128,
        val_batch_size = 128,
        purp: Literal["train", "sample"] = "train",
        n_samples_per_class: int = None,
        classes: list = list(range(10)),
        seed = 0,
        download: bool = True,
        data_path="/dtu/p1/hroy/data",
    ):
    train_dataset = FashionMNIST(train=True,n_samples_per_class=n_samples_per_class,classes=classes,seed=seed,download=download, data_path=data_path)
    val_dataset = FashionMNIST(train=True,n_samples_per_class=n_samples_per_class,classes=classes,seed=seed,download=download, data_path=data_path)
    test_set = FashionMNIST(train=False,n_samples_per_class=None,classes=classes,seed=seed,download=download, data_path=data_path)
    set_seed(seed)
    train_set, _ = torch.utils.data.random_split(train_dataset, [55000, 5000])
    set_seed(seed)
    _, val_set = torch.utils.data.random_split(val_dataset, [55000, 5000])
    if train_batch_size < 0:
        train_batch_size = len(train_set)
    if purp == "train":
        train_loader = data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4, collate_fn=numpy_collate_fn)
    elif purp == "sample":
        train_loader = data.DataLoader(train_set, batch_size=train_batch_size, drop_last=True, pin_memory=True, num_workers=4, collate_fn=numpy_collate_fn, sampler = data.sampler.SequentialSampler(train_set))
    val_loader = data.DataLoader(val_set, batch_size=val_batch_size, shuffle=False, drop_last=True, num_workers=4, collate_fn=numpy_collate_fn)
    test_loader = data.DataLoader(test_set, batch_size=val_batch_size, shuffle=False, drop_last=True, num_workers=4, collate_fn=numpy_collate_fn)
    return train_loader, val_loader, test_loader



def get_rotated_fmnist(
        angle: float = 0, 
        batch_size = 128,
        shuffle = False,
        n_samples_per_class: int = None,
        classes: list = list(range(10)),
        seed = 0,
        download: bool = True,
        data_path="/dtu/p1/hroy/data",
    ):
    rotation = torchvision.transforms.Compose([RotationTransform(angle)])
    dataset = FashionMNIST(
        train=True,
        transform=rotation,
        n_samples_per_class=n_samples_per_class,
        classes=classes,
        seed=seed,
        download=download, 
        data_path=data_path, 
    )
    dataset_test = FashionMNIST(
        train=False,
        transform=rotation,
        n_samples_per_class=None,
        classes=classes,
        seed=seed,
        download=download, 
        data_path=data_path, 
    )
    train_loader, valid_loader = get_loader(
        dataset,
        split_train_val_ratio = 0.9,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        seed=seed
    )
    test_loader = get_loader(
        dataset_test,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        seed=seed
    )
    return train_loader, valid_loader, test_loader
