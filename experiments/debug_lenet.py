import jax
from src.data.all_datasets import get_dataloaders
from src.models import LeNet, ResNet_small,ResNetBlock_small
from flax import linen as nn
import jax.numpy as jnp

train_loader, val_loader, test_loader = get_dataloaders(
    dataset_name="CIFAR-10",
    batch_size=128,
    seed=0,
)
class ConvNet(nn.Module):
    output_dim: int = 10

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=4, kernel_size=(3, 3), strides=(2, 2), padding=1)(x)
        x = nn.tanh(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=4, kernel_size=(3, 3), strides=(2, 2), padding=1)(x)
        x = nn.tanh(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        return nn.Dense(features=self.output_dim)(x)

model_hparams = {
            "output_dim": 10,
            "activation": "tanh"
        }
init_rng = jax.random.PRNGKey(0)
# model = ConvNet(output_dim=10)
model = LeNet(**model_hparams)
# hparams = {
#     "num_classes": 10,
#     "c_hidden": (16, 32, 64),
#     "num_blocks": (3, 3, 3),
#     "act_fn": nn.relu,
#     "block_class": ResNetBlock_small
# }
# print(hparams)
# model = ResNet_small(**hparams)
batch = next(iter(train_loader))
imgs, labels = batch['image'], batch['label']

variables = model.init(init_rng, imgs)
