from src.data.sinusoidal import Sinusoidal, get_sinusoidal
from src.data.mnist import MNIST, get_mnist, get_rotated_mnist
from src.data.emnist import EMNIST, get_emnist, get_rotated_emnist
from src.data.kmnist import KMNIST, get_kmnist, get_rotated_kmnist
from src.data.fmnist import FashionMNIST, get_fmnist, get_rotated_fmnist
from src.data.cifar10 import get_cifar10, get_cifar10_corrupted
from src.data.cifar100 import get_cifar100
from src.data.svhn import get_svhn

from src.data.all_datasets import get_dataloaders
from src.data.utils import get_output_dim, numpy_collate_fn, get_mean_and_std