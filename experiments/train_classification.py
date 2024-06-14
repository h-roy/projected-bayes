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

## Imports for plotting
import matplotlib.pyplot as plt
import matplotlib

from src.data.all_datasets import get_dataloaders
from src.data.utils import get_output_dim
from src.helper import load_obj, set_seed
from src.training import TRAINERS, get_model_hyperparams, get_optimizer_hyperparams
from src.training.train_utils import train
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()
import wandb

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
parser.add_argument("--dataset", type=str, choices=["MNIST", "FMNIST", "CIFAR-10", "SVHN", "CIFAR-100", "ImageNette"], default="CIFAR-10")
parser.add_argument("--data_path", type=str, default="/dtu/p1/hroy/data", help="root of dataset")
parser.add_argument(
    "--model",
    type=str,
    choices=["LeNet", "MLP", "ResNet_small", "ResNet18", "DenseNet", "GoogleNet", "VisionTransformer"],
    default="ResNet_small",
)
parser.add_argument(
    "--CHECKPOINT_PATH",
    type=str,
    default="/dtu/p1/hroy/projected-bayes/checkpoints/",
    help="Path for checkpointing and logging",
)
parser.add_argument("--trainer", type=str, choices=["Classification", "VIT", "lenet"], default="Classification")
parser.add_argument("--loss_type", type=str, choices=["cross_entropy", "sq"], default="cross_entropy")
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for the network")
parser.add_argument("--sgd_momentum", type=float, default=0.9, help="momentum to use in case we are using SGD")
parser.add_argument("--lr_scheduler", action="store_true", required=False, default=False)
parser.add_argument("--scale_by_block_norm", action="store_true", required=False, default=False)
parser.add_argument("--clip_delta", type=float, default=1.0)
parser.add_argument("--optimizer", type=str, choices=["SGD", "adam", "rmsprop", "adamw"], default="SGD")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--run_name", default="resnet", help="Fix the save file name.")
parser.add_argument("--early_stopping", default=-1, type=int)
parser.add_argument("--wandb_project", type=str, default="large_scale_laplace-part3")
parser.add_argument("--train_samples", default=None, type=int, help="Number of training samples per class.")


def process_optimizer_args(args):
    optimizer_hparams = {"lr": args["lr"]}
    if args["optimizer"] == "SGD":
        optimizer_hparams["momentum"] = args["sgd_momentum"]
    elif args["optimizer"] == "adamw":
        optimizer_hparams["weight_decay"] = 0.01

    return optimizer_hparams

def main(args: dict):
    wandb_logger = wandb.init(project=args["wandb_project"], name=args["run_name"], entity="dmiai-mh", config=args)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        handlers=[logging.FileHandler(f"{wandb_logger.dir}/pythonlog.txt"), logging.StreamHandler()],
    )

    set_seed(args["seed"])

    if args["model"] == "VisionTransformer" and args["trainer"] != "VIT":
        print("VisionTransformer model can only be trained with VIT trainer")
        assert args["trainer"] == "VIT"

    n_classes = get_output_dim(args["dataset"])

    print("Device:", jax.devices()[0])

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=args["dataset"],
        train_batch_size=args["batch_size"],
        val_batch_size=args["batch_size"],
        data_path=args["data_path"],
        seed=args["seed"],
        n_samples=args["train_samples"],
    )

    model_hparams = get_model_hyperparams(n_classes, args["model"])
    optimizer_name, optimizer_hparams = get_optimizer_hyperparams(args["model"])
    print("Model Hyperparameters", model_hparams)
    print("Optimizer Hyperparameters", optimizer_hparams)

    batch = next(iter(train_loader))
    imgs, labels = batch['image'], batch['label']
    print("Batch mean", imgs.mean(axis=(0,1,2)))
    print("Batch std", imgs.std(axis=(0,1,2)))   

    def train_classifier(TrainerModule, train_loader, val_loader, test_loader, *args, num_epochs=200, **kwargs):
        # Create a trainer module with specified hyperparameters
        trainer = TrainerModule(*args, **kwargs)
        trainer.train_model(train_loader, val_loader, num_epochs=num_epochs)
        # Test trained model
        val_acc = trainer.eval_model(val_loader)
        test_acc = trainer.eval_model(test_loader)
        return trainer, {'val': val_acc, 'test': test_acc}

    batch = next(iter(train_loader))
    imgs, labels = batch['image'], batch['label']

    resnet_trainer, resnet_results = train_classifier(
                                                    TrainerModule=TRAINERS[args["trainer"]],
                                                    model_name=args["model"],
                                                    model_class=load_obj(MODELS_DICT[args["model"]]),
                                                    model_hparams=model_hparams,
                                                    optimizer_name=optimizer_name,
                                                    # optimizer_hparams=process_optimizer_args(args),
                                                    optimizer_hparams=optimizer_hparams,
                                                    exmp_imgs=jax.device_put(
                                                                    imgs),
                                                    num_epochs=200,
                                                    train_loader=train_loader,
                                                    val_loader=val_loader,
                                                    test_loader=test_loader,
                                                    wandb_logger=wandb_logger,
                                                    args_dict=args,
                                                    seed = args["seed"])

    print(resnet_results)

if __name__ == "__main__":
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)


