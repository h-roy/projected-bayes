import torch
import torchvision
from src.data.sinusoidal import Sinusoidal, get_sinusoidal
from src.data.mnist import MNIST, get_mnist, get_rotated_mnist, get_mnist_ood
from src.data.emnist import EMNIST, get_emnist, get_rotated_emnist
from src.data.kmnist import KMNIST, get_kmnist, get_rotated_kmnist
from src.data.fmnist import FashionMNIST, get_fmnist, get_rotated_fmnist
from src.data.cifar10 import CIFAR10, get_cifar10, get_cifar10_corrupted
from src.data.cifar100 import CIFAR100, get_cifar100
from src.data.svhn import get_svhn, SVHN
from src.data.imagenet import ImageNet1k_loaders,get_imagenet_val_loader, get_imagenet_test_loader, get_places365
from src.data.utils import image_to_numpy, numpy_collate_fn, set_seed

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
    if dataset_name.startswith("R-MNIST"):
        angle = int(dataset_name.removeprefix("R-MNIST"))
        dataset_name = "R-MNIST"
    elif dataset_name.startswith("R-FMNIST"):
        angle = int(dataset_name.removeprefix("R-FMNIST"))
        dataset_name = "R-FMNIST"
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
    elif dataset_name == "R-MNIST":
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
            purp = purp,
            n_samples_per_class = int(n_samples/10) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path
        )
    elif dataset_name == "R-FMNIST":
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
            data_path = data_path,
            n_samples_per_class= int(n_samples/10) if n_samples is not None else None
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
    elif dataset_name == "ImageNet":
        train_loader = ImageNet1k_loaders(batch_size=train_batch_size, purp = purp, seed=seed, n_samples_per_class=int(n_samples/1000) if n_samples is not None else None)
        valid_loader = get_imagenet_val_loader(batch_size=val_batch_size, seed=seed, n_samples_per_class=int(n_samples/1000) if n_samples is not None else None)
        test_loader = get_imagenet_test_loader(batch_size=val_batch_size, seed=seed, n_samples_per_class=int(n_samples/1000) if n_samples is not None else None)
    else:
        raise ValueError(f"Dataset {dataset_name} is not implemented")
    
    return train_loader, valid_loader, test_loader

def get_ood_datasets(experiment, 
                     ood_batch_size, 
                     n_samples, 
                     val=False, 
                     seed=0,
                     corruption="fog"):
    ood_dict = {}
    if experiment == "R-MNIST":
        ids = [0, 15, 30, 60, 90, 120, 150, 180]
        for id in ids:
            _, val_loader, test_loader = get_dataloaders(experiment+f"{id}", n_samples=n_samples, val_batch_size=ood_batch_size, seed=seed, purp="train", angle=id)
            ood_dict[id] = val_loader if val else test_loader
    elif experiment == "R-FMNIST":
        ids = [0, 15, 30, 60, 90, 120, 150, 180]
        for id in ids:
            _, val_loader, test_loader = get_dataloaders(experiment+f"{id}", n_samples=n_samples, val_batch_size=ood_batch_size, seed=seed, purp="train", angle=id)
            ood_dict[id] = val_loader if val else test_loader
    elif experiment == "CIFAR-10-C":
        severity = [1, 2, 3, 4, 5]
        c = corruption
        for s in severity:
            _, _, test_loader = get_dataloaders(experiment+f"{s}-{c}", n_samples=n_samples, val_batch_size=ood_batch_size, seed=seed, purp="train")
            ood_dict[f"{s}-{c}"] = test_loader

    elif experiment in ["MNIST-OOD", "FMNIST-OOD"]:
        ids = ["MNIST", "FMNIST", "KMNIST", "EMNIST"]
        num_samples_per_class = n_samples // 10
        for id in ids:
            _, val_loader, test_loader = get_mnist_ood(id, ood_batch_size, n_samples_per_class=num_samples_per_class, seed=seed)
            ood_dict[id] = val_loader if val else test_loader
    # elif experiment in ["CIFAR-10-OOD", "SVHN-OOD", "CIFAR-100-OOD"]:
    #     ood_datasets_dict = get_dataloaders("CIFAR-10-OOD", val_batch_size=ood_batch_size, n_samples=n_samples, seed=seed)
    #     ids = ["CIFAR-10", "SVHN", "CIFAR-100"]
    #     for id in ids:
    #         ood_dict[id] = ood_datasets_dict[id + ('-val' if val else '-test')]
    elif experiment == "CIFAR-10-OOD":
        train_dataset = torchvision.datasets.CIFAR10(root='/dtu/p1/hroy/data', train=True, download=True)
        means = (train_dataset.data / 255.0).mean(axis=(0,1,2))
        std = (train_dataset.data / 255.0).std(axis=(0,1,2))
        normalize = image_to_numpy(means, std)
        test_set = torchvision.datasets.CIFAR10(root='/dtu/p1/hroy/data', train=False, download=True, transform=normalize)
        test_set.targets = torch.nn.functional.one_hot(torch.tensor(test_set.targets), 10).numpy()
        svhn_test_set = SVHN(root='/dtu/p1/hroy/data', split='test', download=True, transform=normalize)
        cifar100_test_set = torchvision.datasets.CIFAR100(root='/dtu/p1/hroy/data', train=False, download=True, transform=normalize)
        cifar100_test_set.targets = torch.nn.functional.one_hot(torch.tensor(cifar100_test_set.targets), 100).numpy()
        if n_samples is not None:
            set_seed(seed)
            test_set, _ = torch.utils.data.random_split(test_set, [n_samples, len(test_set)-n_samples])
            svhn_test_set, _  = torch.utils.data.random_split(svhn_test_set, [n_samples, len(svhn_test_set)-n_samples])
            cifar100_test_set, _  = torch.utils.data.random_split(cifar100_test_set, [n_samples, len(cifar100_test_set)-n_samples])
        ood_dict["CIFAR-10"] = torch.utils.data.DataLoader(test_set, batch_size=ood_batch_size, shuffle=True, drop_last=True, num_workers=4, collate_fn=numpy_collate_fn)
        ood_dict["SVHN"] = torch.utils.data.DataLoader(svhn_test_set, batch_size=ood_batch_size, shuffle=True, drop_last=True, num_workers=4, collate_fn=numpy_collate_fn)
        ood_dict["CIFAR-100"] = torch.utils.data.DataLoader(cifar100_test_set, batch_size=ood_batch_size, shuffle=True, drop_last=True, num_workers=4, collate_fn=numpy_collate_fn)
        return ood_dict
    elif experiment == "SVHN-OOD":
        train_dataset = torchvision.datasets.SVHN(root='/dtu/p1/hroy/data', split='train', download=True)
        means = (train_dataset.data / 255.0).mean(axis=(0,2,3))
        std = (train_dataset.data / 255.0).std(axis=(0,2,3))
        normalize = image_to_numpy(means, std)
        test_set = SVHN(root='/dtu/p1/hroy/data', split='test', download=True, transform=normalize)
        cifar10_test_set = torchvision.datasets.CIFAR10(root='/dtu/p1/hroy/data', train=False, download=True, transform=normalize)
        cifar10_test_set.targets = torch.nn.functional.one_hot(torch.tensor(cifar10_test_set.targets), 10).numpy()
        cifar100_test_set = torchvision.datasets.CIFAR100(root='/dtu/p1/hroy/data', train=False, download=True, transform=normalize)
        cifar100_test_set.targets = torch.nn.functional.one_hot(torch.tensor(cifar100_test_set.targets), 100).numpy()
        if n_samples is not None:
            set_seed(seed)
            test_set, _ = torch.utils.data.random_split(test_set, [n_samples, len(test_set)-n_samples])
            cifar10_test_set, _  = torch.utils.data.random_split(cifar10_test_set, [n_samples, len(cifar10_test_set)-n_samples])
            cifar100_test_set, _  = torch.utils.data.random_split(cifar100_test_set, [n_samples, len(cifar100_test_set)-n_samples])
        ood_dict["SVHN"] = torch.utils.data.DataLoader(test_set, batch_size=ood_batch_size, shuffle=True, drop_last=True, num_workers=4, collate_fn=numpy_collate_fn)
        ood_dict["CIFAR-10"] = torch.utils.data.DataLoader(cifar10_test_set, batch_size=ood_batch_size, shuffle=True, drop_last=True, num_workers=4, collate_fn=numpy_collate_fn)
        ood_dict["CIFAR-100"] = torch.utils.data.DataLoader(cifar100_test_set, batch_size=ood_batch_size, shuffle=True, drop_last=True, num_workers=4, collate_fn=numpy_collate_fn)
        return ood_dict
    elif experiment == "ImageNet-OOD":
        ood_dict["ImageNet"] = get_imagenet_val_loader(batch_size=ood_batch_size, seed=seed, n_samples_per_class=int(n_samples/1000) if n_samples is not None else None)
        ood_dict["Places365"] = get_places365(batch_size=ood_batch_size, seed=seed, n_samples_per_class=int(n_samples/1000) if n_samples is not None else None)

    return ood_dict
