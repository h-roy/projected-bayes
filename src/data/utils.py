import torch
import numpy as np
from torchvision.transforms.functional import normalize
from torch.utils import data

def n_classes(dataset_name):
    if dataset_name == "CIFAR-100":
        return 100
    else:
        return 10

def select_num_samples(dataset, n_samples, cls_to_idx, seed=0):
    np.random.seed(seed)
    idxs = []
    for key,_ in cls_to_idx.items():
        indices = np.where(dataset.targets == key)[0]
        if n_samples>len(indices):
            raise ValueError(f"Class {key} has only {len(indices)} data, you are asking for {n_samples}.")
        idxs.append(np.random.choice(indices, n_samples, replace=False))
    idxs = np.concatenate(idxs)
    dataset.data = dataset.data[idxs]
    dataset.targets = dataset.targets[idxs]
    return dataset

def select_classes(dataset, classes):
    idxs = []
    for i in classes:
        indices = np.where(dataset.targets == i)[0]
        idxs.append(indices)
    idxs = np.concatenate(idxs).astype(int)
    dataset.data = dataset.data[idxs]
    dataset.targets = dataset.targets[idxs]
    return dataset

def numpy_collate_fn(batch):
    data, target = zip(*batch)
    data = np.stack(data)
    target = np.stack(target)
    return {"image": data, "label": target}


def get_mean_and_std(data_train, val_frac, seed):
    len_val = int(len(data_train) * val_frac)
    len_train = len(data_train) - len_val

    data_train, _ = data.random_split(data_train, [len_train, len_val], generator=torch.Generator().manual_seed(seed))
    _data = data_train.dataset.data[data_train.indices].transpose(0, 3, 1, 2) / 255.0
    mean_train = _data.mean(axis=(0, 2, 3))
    std_train = _data.std(axis=(0, 2, 3))

    return {"mean": tuple(mean_train.tolist()), "std": tuple(std_train.tolist())}


def image_to_numpy(img):
    img = np.array(img, dtype=np.float32).transpose(1, 2, 0)
    return img


def channel_normalization(tensor, mean, std):
    tensor = torch.from_numpy(tensor).float().transpose(1, 3)
    tensor = normalize(tensor, mean, std)
    return tensor