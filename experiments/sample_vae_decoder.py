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
from src.data.utils import save_image
from src.helper import load_obj, set_seed
from src.sampling import sample_loss_gen_projections_dataloader
from src.sampling.sample_utils import vectorize_nn, linearize_model_fn
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



## To run JAX on TPU in Google Colab, uncomment the two lines below
# import jax.tools.colab_tpu
# jax.tools.colab_tpu.setup_tpu()

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
parser.add_argument("--num_iterations", type=int, default=1000)
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--macro_batch_size", type=int, default=50000)
parser.add_argument("--sample_batch_size", type=int, default=256)
parser.add_argument("--clip_delta", type=float, default=1.0)
parser.add_argument("--optimizer", type=str, choices=["SGD", "adam", "rmsprop", "adamw"], default="SGD")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--latents", default=20, type=int)
parser.add_argument("--train_samples", default=None, type=int, help="Number of training samples per class.")
parser.add_argument("--prior_precision", default=0.01, type=float, help="scaling distance.")


def kl_divergence_perdatapoint(mean, logvar):
  return -0.5 * (1 + logvar - jnp.square(mean) - jnp.exp(logvar)).sum(axis=-1)

def mse_recon_loss_perdatapoint(recon_imgs, imgs):
    loss = ((recon_imgs - imgs) ** 2).sum(axis=(-1, -2, -3))  # Mean over batch, sum over pixels
    return loss

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
        train_batch_size=args["macro_batch_size"],
        val_batch_size=args["sample_batch_size"],
        data_path=args["data_path"],
        seed=args["seed"],
        n_samples=args["train_samples"],
        purp='sample'
    )

    rng = random.key(args["seed"])
    rng, key = random.split(rng)
    batch = next((iter(train_loader)))
    img = jnp.asarray(batch['image'], float)
    # img = img.reshape((img.shape[0], -1))


    state_dict = pickle.load(open(f"./checkpoints/VAE/{dataset}/vae_params.pickle", "rb"))
    encoder_params, decoder_params = state_dict['params']
    rng = state_dict['rng']
    encoder = vae.Encoder(c_hid=20, latents=args["latents"])
    decoder = vae.Decoder(c_out=1, c_hid=20, latents=args["latents"])
    # encoder_params = {"params": params['params']['encoder']}
    # decoder_params = {"params": params['params']['decoder']}
    model_fn = lambda p, x: decoder.apply(p, x)
    params_vec, unflatten, model_fn_vec = vectorize_nn(model_fn, decoder_params)
    linearized_decoder = linearize_model_fn(model_fn, decoder_params)
    n_iterations = args["num_iterations"]
    n_samples = args["num_samples"]
    n_params = len(params_vec)
    print("Number of parameters:", n_params)

    sample_key = jax.random.PRNGKey(args["seed"])
    alpha = 0.1
    eps = jax.random.normal(sample_key, (n_samples, n_params))
    model_fn = lambda pe, pd, x, rng: (decoder.apply(pd, reparameterize(encoder.apply(pe, x), rng)), encoder.apply(pe, x))
    def encoder_loss_model_fn(decoder_params_vec, imgs, encoder_params, rng):
        z = reparameterize(encoder.apply(encoder_params, imgs), rng)
        recon_imgs = model_fn_vec(decoder_params_vec, z)
        loss = mse_recon_loss_perdatapoint(recon_imgs, imgs)
        return loss
    loss_model_fn = lambda params_vec, x: encoder_loss_model_fn(params_vec, x, encoder_params, rng)
    posterior_samples, metrics = sample_loss_gen_projections_dataloader(
                                                      loss_model_fn,
                                                      params_vec,
                                                      eps,
                                                      train_loader,
                                                      args["sample_batch_size"],
                                                      args["seed"],
                                                      n_iterations,
                                                      img,
                                                      unflatten,
                                                      acceleration=False
                                                      )
    
    test_imgs = next(iter(test_loader))['image']
    test_imgs = jnp.asarray(test_imgs, float)#.reshape((test_imgs.shape[0], -1))
    z_key, eval_rng = random.split(rng)
    z = jax.random.normal(z_key, (64, args["latents"]))
    prior_precision = args["prior_precision"]
    posterior_samples = jax.vmap(lambda sample: jax.tree_map(lambda x, y: (x - y) * prior_precision + y, sample, decoder_params))(posterior_samples)
    std_recon = []
    std_gen = []
    # eval_model_fn = lambda pe, pd, x, rng: (decoder.apply(pd, reparameterize(encoder.apply(pe, x), rng)), encoder.apply(pe, x))
    eval_model_fn = lambda pe, pd, x, rng: (linearized_decoder(pd, reparameterize(encoder.apply(pe, x), rng)), encoder.apply(pe, x))
    for i in range(n_samples):
        param_sample = jax.tree_map(lambda x: x[i], posterior_samples)
        metrics, comparison, sample = eval_f(
                (encoder_params, param_sample), test_imgs, z, eval_rng, eval_model_fn, linearized_decoder #decoder.apply
            )
        save_image(
            comparison, f'./results/VAE/{dataset}/reconstruction_sample_{i}.pdf', nrow=8
        )
        save_image(sample, f'./results/VAE/{dataset}/generation_sample_{i}.pdf', nrow=8)
        print(
            'eval sample: {}, loss: {:.4f}, mse: {:.4f}, KLD: {:.4f}'.format(
                i + 1, metrics['loss'], metrics['mse'], metrics['kld']
            )
        )
        std_recon.append(comparison[10])
        std_gen.append(sample[10])
    std_recon = (jnp.stack(std_recon)).std(axis=0)
    std_gen = (jnp.stack(std_gen)).std(axis=0)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].imshow(comparison[2])
    ax[0, 1].imshow(sample[10])
    ax[1, 0].imshow(std_recon)
    ax[1, 1].imshow(std_gen)
    fig.savefig(f'./results/VAE/{dataset}/reconstruction_std.pdf')
    fixed_z = jax.random.normal(z_key, (1, args["latents"]))
    map_sample = decoder.apply(decoder_params, fixed_z)
    fig, ax = plt.subplots(ncols=n_samples + 1, figsize=(40, 10))
    ax[0].imshow(map_sample[0])
    for i in range(n_samples):
        param_sample = jax.tree_map(lambda x: x[i], posterior_samples)
        sample = linearized_decoder(param_sample, fixed_z)
        ax[i+1].imshow(sample[0])
    fig.savefig(f'./results/VAE/{dataset}/generation_single_sample.pdf')

    save_dir = f"./checkpoints/VAE/{dataset}/posterior_samples/"
    os.makedirs(save_dir, exist_ok=True)
    state_dict =  {"params": posterior_samples, "rng": rng}
    pickle.dump(state_dict, open(f"{save_dir}vae_sample_params.pickle", "wb"))

if __name__ == "__main__":
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)

