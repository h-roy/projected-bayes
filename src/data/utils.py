import jax
import torch
import torchvision
import numpy as np
from PIL import Image
import random
import jax.numpy as jnp
import math

class RotationTransform:
    """Rotate the given angle."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return torchvision.transforms.functional.rotate(x, self.angle)
    
class ToChannelsLast:
    def __call__(self, x):
        if x.ndim == 3:
            x = x.permute((1,2,0))
        elif x.ndim !=3:
            raise RuntimeError
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ToImage:
    def __call__(self, x):
        if x.ndim == 3:
            x = x.permute((1,2,0))
        elif x.ndim !=3:
            raise RuntimeError
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'

class DatafeedImage(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train, transform=None):
        self.x_train = x_train
        self.y_train = y_train
        self.transform = transform

    def __getitem__(self, index):
        img = self.x_train[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        label = self.y_train[index]
        label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=10)

        return img, label

    def __len__(self):
        return len(self.x_train)


def get_subset_data(data, targets, classes, n_samples_per_class=None, seed=0):
    np.random.seed(seed)
    targets = np.array(targets)
    idxs = []
    for target in classes:
        indices = np.where(targets == target)[0]
        if n_samples_per_class is None:
            # take all elements of that class
            idxs.append(indices)
        else:
            # subset only "n_samples_per_class" elements per class
            if n_samples_per_class>len(indices):
                raise ValueError(f"Class {target} has only {len(indices)} data, you are asking for {n_samples_per_class}.")
            idxs.append(np.random.choice(indices, n_samples_per_class, replace=False))
    idxs = np.concatenate(idxs).astype(int)
    targets = targets[idxs]

    clas_to_index = { c : i for i, c in enumerate(classes)}
    targets = np.array([clas_to_index[clas.item()] for clas in targets])
    data = data[idxs]
    return data, targets



def get_loader(
        dataset,
        batch_size = 128,
        split_train_val_ratio: float = 1.0,
        shuffle: bool = False,
        drop_last: bool = True,
        seed = 0
    ):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if split_train_val_ratio == 1.0:
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=4, 
            pin_memory=False, 
            drop_last=drop_last,
            collate_fn=numpy_collate_fn,
        )
    else:
        train_size = int(split_train_val_ratio * len(dataset))
        valid_size = len(dataset) - train_size
        dataset_train, dataset_valid = torch.utils.data.random_split(
            dataset, (train_size, valid_size), generator=torch.Generator().manual_seed(0)
        )
        return (
            torch.utils.data.DataLoader(
                dataset_train, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                num_workers=4, 
                pin_memory=False, 
                drop_last=drop_last,
                collate_fn=numpy_collate_fn,
            ),
            torch.utils.data.DataLoader(
                dataset_valid, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                num_workers=4, 
                pin_memory=False, 
                drop_last=drop_last,
                collate_fn=numpy_collate_fn,
            ),
        )
    
def get_output_dim(dataset_name):
    if dataset_name in ["Sinusoidal", "UCI"]:
        return 1 
    elif dataset_name == "CelebA":
        return 6 #40
    elif dataset_name == "CIFAR-100":
        return 100
    else:
        return 10
    
def image_to_numpy(mean, std):
    def normalize(img):
        img = np.array(img, dtype=np.float32)
        img = (img / 255. - mean) / std
        return img
    return normalize


# For visualization, we might want to map JAX or numpy tensors back to PyTorch
def jax_to_torch(imgs):
    imgs = jax.device_get(imgs)
    imgs = torch.from_numpy(imgs.astype(np.float32))
    imgs = imgs.permute(0, 3, 1, 2)
    return imgs

def numpy_collate_fn(batch):
    data, target = zip(*batch)
    data = np.stack(data)
    target = np.stack(target)
    return {"image": data, "label": target}

def get_mean_and_std(data_train, val_frac, seed):
    len_val = int(len(data_train) * val_frac)
    len_train = len(data_train) - len_val

    data_train, _ = torch.utils.data.random_split(data_train, [len_train, len_val], generator=torch.Generator().manual_seed(seed))
    _data = data_train.dataset.data[data_train.indices].transpose(0, 3, 1, 2) / 255.0
    mean_train = _data.mean(axis=(0, 2, 3))
    std_train = _data.std(axis=(0, 2, 3))

    return {"mean": tuple(mean_train.tolist()), "std": tuple(std_train.tolist())}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_image(ndarray, fp, nrow=8, padding=2, pad_value=0.0, format_img=None):
  """Make a grid of images and Save it into an image file.

  Args:
    ndarray (array_like): 4D mini-batch images of shape (B x H x W x C)
    fp:  A filename(string) or file object
    nrow (int, optional): Number of images displayed in each row of the grid.
      The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
    padding (int, optional): amount of padding. Default: ``2``.
    pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    format_img(Optional):  If omitted, the format to use is determined from the
      filename extension. If a file object was used instead of a filename,
      this parameter should always be used.
  """

  if not (
      isinstance(ndarray, jnp.ndarray)
      or (
          isinstance(ndarray, list)
          and all(isinstance(t, jnp.ndarray) for t in ndarray)
      )
  ):
    raise TypeError(f'array_like of tensors expected, got {type(ndarray)}')

  ndarray = jnp.asarray(ndarray)

  if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
    ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

  # make the mini-batch of images into a grid
  nmaps = ndarray.shape[0]
  xmaps = min(nrow, nmaps)
  ymaps = int(math.ceil(float(nmaps) / xmaps))
  height, width = (
      int(ndarray.shape[1] + padding),
      int(ndarray.shape[2] + padding),
  )
  num_channels = ndarray.shape[3]
  grid = jnp.full(
      (height * ymaps + padding, width * xmaps + padding, num_channels),
      pad_value,
  ).astype(jnp.float32)
  k = 0
  for y in range(ymaps):
    for x in range(xmaps):
      if k >= nmaps:
        break
      grid = grid.at[
          y * height + padding : (y + 1) * height,
          x * width + padding : (x + 1) * width,
      ].set(ndarray[k])
      k = k + 1

  # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
  ndarr = np.array(jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8))
  im = Image.fromarray(ndarr.copy())
  im.save(fp, format=format_img)