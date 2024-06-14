import torch

from src.data.sinusoidal import Sinusoidal, get_sinusoidal
from src.data.mnist import MNIST, get_mnist, get_rotated_mnist
from src.data.emnist import EMNIST, get_emnist, get_rotated_emnist
from src.data.kmnist import KMNIST, get_kmnist, get_rotated_kmnist
from src.data.fmnist import FashionMNIST, get_fmnist, get_rotated_fmnist
from src.data.cifar10 import CIFAR10, get_cifar10, get_cifar10_corrupted
from src.data.cifar100 import CIFAR100, get_cifar100
from src.data.svhn import get_svhn

def get_dataloaders(
        dataset_name,
        n_samples = None,
        train_batch_size: int = 128,
        val_batch_size: int = 128,
        shuffle = True,
        seed: int = 0,
        download: bool = True,
        data_path: str = "/dtu/p1/hroy/data",
        transform = None,
        purp = "train",             # for SVHN, CIFAR-10 and CIFAR-100
        angle: float = 0,           # for rotated datasets
        corr_type: str = "fog",     # for corrupted datasets
        severity_level: int = 5,    # for corrupted datasets
    ):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if dataset_name.startswith("MNIST-R"):
        angle = int(dataset_name.removeprefix("MNIST-R"))
        dataset_name = "MNIST-R"
    elif dataset_name.startswith("FMNIST-R"):
        angle = int(dataset_name.removeprefix("FMNIST-R"))
        dataset_name = "FMNIST-R"
    elif dataset_name.startswith("CIFAR-10-C"):
        severity_level, corr_type = dataset_name.removeprefix("CIFAR-10-C").split('-')
        severity_level = int(severity_level)
        dataset_name = "CIFAR-10-C"
        
    if dataset_name == "Sinusoidal":
        train_loader, valid_loader, test_loader = get_sinusoidal(
            batch_size = val_batch_size, 
            shuffle = shuffle,
            n_samples = n_samples,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "MNIST":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_mnist(
            train_batch_size = train_batch_size,
            val_batch_size = val_batch_size, 
            purp = purp,
            n_samples_per_class = int(n_samples/10) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "MNIST-R":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_rotated_mnist(
            angle = angle,
            batch_size = val_batch_size, 
            shuffle = shuffle,
            n_samples_per_class = int(n_samples/10) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "FMNIST":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_fmnist(
            train_batch_size = train_batch_size,
            val_batch_size = val_batch_size,
            shuffle = shuffle,
            n_samples_per_class = int(n_samples/10) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "FMNIST-R":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_rotated_fmnist(
            angle = angle,
            batch_size = val_batch_size, 
            shuffle = shuffle,
            n_samples_per_class = int(n_samples/10) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "EMNIST":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_emnist(
            batch_size = val_batch_size, 
            shuffle = shuffle,
            n_samples_per_class = int(n_samples/10) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "KMNIST":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_kmnist(
            batch_size = val_batch_size, 
            shuffle = shuffle,
            n_samples_per_class = int(n_samples/10) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "CIFAR-10":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_cifar10(
            train_batch_size = train_batch_size,
            val_batch_size = val_batch_size, 
            purp = purp,
            transform= transform,
            seed = seed,
            download = download, 
            data_path = data_path,
            n_samples_per_class= int(n_samples/10) if n_samples is not None else None
        )
    elif dataset_name == "CIFAR-10-C":
        classes = list(range(10))
        # train and valid are None
        train_loader, valid_loader, test_loader = get_cifar10_corrupted(
            corr_type = corr_type,
            severity_level = severity_level,
            batch_size = val_batch_size, 
            shuffle = shuffle,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "CIFAR-100":
        classes = list(range(100))
        train_loader, valid_loader, test_loader = get_cifar100(
            train_batch_size = train_batch_size,
            val_batch_size = val_batch_size,
            purp = purp,
            transform= transform,
            seed = seed,
            download = download, 
            data_path = data_path,
            n_samples_per_class= int(n_samples/10) if n_samples is not None else None
        )
    elif dataset_name == "SVHN":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_svhn(
            train_batch_size = train_batch_size,
            val_batch_size = val_batch_size,
            purp = purp,
            transform= transform,
            seed = seed,
            download = download, 
            data_path = data_path,
            n_samples_per_class= int(n_samples/10) if n_samples is not None else None
        )
    elif dataset_name == "CIFAR-10-OOD":
        ood_test = {}
        classes = list(range(10))

        _, cifar10_valid_loader, cifar10_test_loader = get_cifar10(
            train_batch_size = train_batch_size,
            val_batch_size = val_batch_size, 
            purp = purp,
            transform= transform,
            seed = seed,
            download = download, 
            data_path = data_path
        )
        ood_test["CIFAR-10-val"] = cifar10_valid_loader
        ood_test["CIFAR-10-test"] = cifar10_test_loader
        _, svhn_valid_loader, svhn_test_loader = get_svhn(
            train_batch_size = train_batch_size,
            val_batch_size = val_batch_size, 
            purp = purp,
            transform= transform,
            seed = seed,
            download = download, 
            data_path = data_path
        )
        ood_test["SVHN-val"] = svhn_valid_loader
        ood_test["SVHN-test"] = svhn_test_loader
        _, cifar_100_valid_loader, cifar100_test_loader = get_cifar100(
            train_batch_size = train_batch_size,
            val_batch_size = val_batch_size, 
            purp = purp,
            transform= transform,
            seed = seed,
            download = download, 
            data_path = data_path
        )
        ood_test["CIFAR-100-val"] = cifar_100_valid_loader
        ood_test["CIFAR-100-test"] = cifar100_test_loader
        return ood_test

    else:
        raise ValueError(f"Dataset {dataset_name} is not implemented")
    
    return train_loader, valid_loader, test_loader