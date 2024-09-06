## Standard libraries
import argparse
import json
import logging
import os
import pickle
import numpy as np
from PIL import Image
from typing import Any, Literal
from collections import defaultdict
import time
import math
## Imports for plotting
import matplotlib.pyplot as plt
import matplotlib

from src.data.all_datasets import get_dataloaders
from src.data.utils import get_output_dim, save_image
from src.helper import load_obj, set_seed
from src.sampling.sample_utils import vectorize_nn
from src.training import TRAINERS, get_model_hyperparams, get_optimizer_hyperparams
from src.training.train_utils import train
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()
import wandb
from src.models import vae 
from src.models.vae import Encoder, Decoder, reparameterize

## Progress bar
from tqdm.auto import tqdm
from src.training.classification_trainer import Classification_Trainer


## JAX
import jax
import jax.numpy as jnp
from jax import random
# Seeding for random operations
main_rng = random.PRNGKey(42)

## Flax (NN in JAX)
from flax import linen as nn
from flax.training import train_state

## PyTorch
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from src.data import get_cifar10
from src.models import MODELS_DICT, ResNetBlock_small, ResNet_small
import optax

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["MNIST", "FMNIST"], default="MNIST")
parser.add_argument("--data_path", type=str, default="/dtu/p1/hroy/data", help="root of dataset")
parser.add_argument("--n_epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for the network")
parser.add_argument("--sgd_momentum", type=float, default=0.9, help="momentum to use in case we are using SGD")
parser.add_argument("--lr_scheduler", action="store_true", required=False, default=False)
parser.add_argument("--scale_by_block_norm", action="store_true", required=False, default=False)
parser.add_argument("--clip_delta", type=float, default=1.0)
parser.add_argument("--optimizer", type=str, choices=["SGD", "adam", "rmsprop", "adamw"], default="SGD")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--latents", default=20, type=int)
parser.add_argument("--train_samples", default=None, type=int, help="Number of training samples per class.")


def process_optimizer_args(args):
    optimizer_hparams = {"lr": args["lr"]}
    if args["optimizer"] == "SGD":
        optimizer_hparams["momentum"] = args["sgd_momentum"]
    elif args["optimizer"] == "adamw":
        optimizer_hparams["weight_decay"] = 0.01

    return optimizer_hparams

@jax.vmap
def kl_divergence(mean, logvar):
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

def mse_recon_loss(recon_imgs, imgs):
    loss = ((recon_imgs - imgs) ** 2).mean(axis=0).sum()  # Mean over batch, sum over pixels
    return loss

def compute_metrics(recon_x, x, mean, logvar):
  mse_loss = mse_recon_loss(recon_x, x).mean()
  kld_loss = kl_divergence(mean, logvar).mean()
  return {'mse': mse_loss, 'kld': kld_loss, 'loss': mse_loss + kld_loss}

def train_step(state, batch, z_rng, model_fn):
  def loss_fn(params):
    encoder_params, decoder_params = params
    recon_x, (mean, logvar) = model_fn(encoder_params, decoder_params, batch, z_rng)    
    mse_loss = mse_recon_loss(recon_x, batch).mean()
    kld_loss = kl_divergence(mean, logvar).mean()
    loss = mse_loss + kld_loss
    return loss

  grads = jax.grad(loss_fn)(state.params)
  return state.apply_gradients(grads=grads)

def eval_f(params, images, z, z_rng, model_fn, decode_fn):
  def eval_model(model_fn):
    encoder_params, decoder_params = params
    recon_images, (mean, logvar) = model_fn(encoder_params, decoder_params, images, z_rng)
    comparison = jnp.concatenate([
        images[:8].reshape(-1, 28, 28, 1),
        recon_images[:8].reshape(-1, 28, 28, 1),
    ])
    generate_images = decode_fn(decoder_params, z)
    generate_images = generate_images.reshape(-1, 28, 28, 1)
    metrics = compute_metrics(recon_images, images, mean, logvar)
    return metrics, comparison, generate_images
  return eval_model(model_fn)

def main(args: dict):
    set_seed(args["seed"])
    print("Device:", jax.devices()[0])
    dataset = args["dataset"]
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=dataset,
        train_batch_size=args["batch_size"],
        val_batch_size=args["batch_size"],
        data_path=args["data_path"],
        seed=args["seed"],
        n_samples=args["train_samples"],
    )
    rng = random.key(args["seed"])
    rng, key = random.split(rng)
    batch = next((iter(train_loader)))
    img = jnp.asarray(batch['image'], float)
    encoder = vae.Encoder(c_hid=20, latents=args["latents"])
    decoder = vae.Decoder(c_out=1, c_hid=20, latents=args["latents"])
    encoder_params = encoder.init(key, img)
    z = encoder.apply(encoder_params, img)
    decoder_params = decoder.init(key, z[0])

    # model_fn = lambda pe, pd, x, rng: decoder.apply(pd, reparameterize(encoder.apply(pe, x), rng))
    model_fn = lambda pe, pd, x, rng: (decoder.apply(pd, reparameterize(encoder.apply(pe, x), rng)), encoder.apply(pe, x))
    state = train_state.TrainState.create(
      apply_fn=model_fn,
      params=(encoder_params, decoder_params),
      tx=optax.adam(args["lr"]),
    )
    test_imgs = next(iter(test_loader))['image']
    test_imgs = jnp.asarray(test_imgs, float)#.reshape((test_imgs.shape[0], -1))
    z_key, eval_rng = random.split(rng)
    z = jax.random.normal(z_key, (64, args["latents"]))
    os.makedirs(f"./results/VAE/{dataset}/", exist_ok=True)
    for epoch in range(args["n_epochs"]):
        for batch in train_loader:
            rng, key = random.split(rng)
            img = jnp.asarray(batch['image'], float)#.reshape((batch['image'].shape[0], -1))
            state = train_step(state, img, key, model_fn)
            metrics, comparison, sample = eval_f(
                state.params, test_imgs, z, eval_rng, model_fn, decoder.apply
            )
            save_image(
                comparison, f'./results/VAE//{dataset}/reconstruction_{epoch}.png', nrow=8
            )
            save_image(sample, f'./results/VAE//{dataset}/sample_{epoch}.png', nrow=8)
            print(
                'eval epoch: {}, loss: {:.4f}, mse: {:.4f}, KLD: {:.4f}'.format(
                    epoch + 1, metrics['loss'], metrics['mse'], metrics['kld']
                )
            )
    save_dir = f"./checkpoints/VAE/{dataset}/"
    os.makedirs(save_dir, exist_ok=True)
    state_dict =  {"params": state.params, "rng": rng}
    pickle.dump(state_dict, open(f"{save_dir}vae_params.pickle", "wb"))


if __name__ == "__main__":
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)

