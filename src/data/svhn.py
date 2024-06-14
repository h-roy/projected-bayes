from pathlib import Path
import torch
import torchvision
from src.data.utils import ToChannelsLast, set_seed, numpy_collate_fn, image_to_numpy
from torchvision import transforms as T
import torch.utils.data as data
from typing import Literal
from PIL import Image

class SVHN(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        transform = None,
        seed: int = 0,
        download: bool = True,
        root: str = "/dtu/p1/hroy/data",
    ):
        n_classes = 10
        set_seed(seed)
        self.transform = transform
        self.path = Path(root)
        self.dataset = torchvision.datasets.SVHN(root=self.path, split=split, download=download)
        self.targets = torch.nn.functional.one_hot(torch.tensor(self.dataset.labels), n_classes).numpy()
        self.data = self.dataset.data

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.transpose(1,2,0))
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

def get_svhn(
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
        train_dataset = SVHN(root=data_path, split='train', download=download)
        means = (train_dataset.data / 255.0).mean(axis=(0,2,3))
        std = (train_dataset.data / 255.0).std(axis=(0,2,3))
        normalize = image_to_numpy(means, std)
        test_transform = normalize
        # For training, we add some augmentation. Networks are too powerful and would overfit.
        if purp == "train":
            train_transform = T.Compose([
                                        T.RandomHorizontalFlip(),
                                        T.RandomResizedCrop((32,32), scale=(0.8,1.0), ratio=(0.9,1.1)),
                                        normalize
                                        ])
        elif purp == "sample":
            train_transform = test_transform
        if transform is not None:
            train_transform = T.Compose([train_transform, transform])
            test_transform = T.Compose([test_transform, transform])

        train_dataset = SVHN(root=data_path, split='train', transform=train_transform, download=download)
        val_dataset = SVHN(root=data_path, split='train', transform=test_transform, download=download)
        set_seed(seed)
        train_set, _ = torch.utils.data.random_split(train_dataset, [70000, 3257])
        set_seed(seed)
        _, val_set = torch.utils.data.random_split(val_dataset, [70000, 3257])
        test_set = SVHN(root=data_path, split='test', transform=test_transform, download=download)
        if n_samples_per_class is not None:
            set_seed(seed)
            n_data = n_samples_per_class * 10
            train_set, _ = torch.utils.data.random_split(train_dataset, [n_data, len(train_dataset)-n_data])
            val_set, _  = torch.utils.data.random_split(val_dataset, [n_data, len(val_dataset)-n_data])
            test_set, _  = torch.utils.data.random_split(test_set, [n_data, len(test_set)-n_data])

        # train_set.dataset.labels = torch.nn.functional.one_hot(torch.tensor(train_set.dataset.labels), n_classes).numpy()
        # val_set.dataset.labels = torch.nn.functional.one_hot(torch.tensor(val_set.dataset.labels), n_classes).numpy()
        # test_set.labels = torch.nn.functional.one_hot(torch.tensor(test_set.labels), n_classes).numpy()
        if purp == "train":
            train_loader = data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4, collate_fn=numpy_collate_fn)
        elif purp == "sample":
            train_loader = data.DataLoader(train_set, batch_size=train_batch_size, drop_last=True, pin_memory=True, num_workers=4, collate_fn=numpy_collate_fn, sampler = data.sampler.SequentialSampler(train_set))
        val_loader = data.DataLoader(val_set, batch_size=val_batch_size, shuffle=False, drop_last=False, num_workers=4, collate_fn=numpy_collate_fn)
        test_loader = data.DataLoader(test_set, batch_size=val_batch_size, shuffle=False, drop_last=False, num_workers=4, collate_fn=numpy_collate_fn)
        return train_loader, val_loader, test_loader

