# from src.data.mnist import MNIST
# from src.data.fmnist import FashionMNIST
from src.data.svhn import SVHN
# from src.data.cifar10 import CIFAR10
from src.data.cifar100 import CIFAR100
from .torch_datasets import MNIST, CIFAR10, FashionMNIST, ImageNette
from .dataloader import get_dataloaders
from .utils import get_mean_and_std, numpy_collate_fn, n_classes
