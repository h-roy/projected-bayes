from pathlib import Path
import torch
import torchvision
from src.data.utils import get_loader, get_subset_data, RotationTransform

class EMNIST(torch.utils.data.Dataset):
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
        self.dataset = torchvision.datasets.EMNIST(root=self.path, split='digits', download=download)

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


def get_emnist(
        batch_size = 128,
        shuffle = False,
        n_samples_per_class: int = None,
        classes: list = list(range(10)),
        seed = 0,
        download: bool = True,
        data_path="/dtu/p1/hroy/data",
    ):
    dataset = EMNIST(
        train=True,
        n_samples_per_class=n_samples_per_class,
        classes=classes,
        seed=seed,
        download=download, 
        data_path=data_path, 
    )
    dataset_test = EMNIST(
        train=False,
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


def get_rotated_emnist(
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
    dataset = EMNIST(
        train=True,
        transform=rotation,
        n_samples_per_class=n_samples_per_class,
        classes=classes,
        seed=seed,
        download=download, 
        data_path=data_path, 
    )
    dataset_test = EMNIST(
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
