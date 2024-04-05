import argparse

from jax import config

config.update("jax_debug_nans", True)

import wandb

from src.helper import set_seed, load_obj
from src.data import get_dataloaders
from src.training import train, TRAINERS
from src.models import MODELS_DICT
import jax
from jax import random
from src.models import ResNetBlock
from flax import linen as nn

import logging

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["MNIST", "FMNIST", "CIFAR10", "ImageNette"], default="CIFAR10")
parser.add_argument("--data_path", type=str, default="/dtu/p1/hroy/data", help="root of dataset")
parser.add_argument(
    "--model",
    type=str,
    choices=["LeNet", "MLP", "ResNet", "DenseNet", "GoogleNet", "VisionTransformer"],
    default="ResNet",
)
parser.add_argument(
    "--CHECKPOINT_PATH",
    type=str,
    default="/dtu/p1/hroy/projected-bayes/checkpoints/",
    help="Path for checkpointing and logging",
)
parser.add_argument("--trainer", type=str, choices=["Classification", "VIT"], default="Classification")
parser.add_argument("--loss_type", type=str, choices=["cross_entropy", "sq"], default="cross_entropy")
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for the network")
parser.add_argument("--sgd_momentum", default=None, help="momentum to use in case we are using SGD")
parser.add_argument("--lr_scheduler", action="store_true", required=False, default=False)
parser.add_argument("--scale_by_block_norm", action="store_true", required=False, default=False)
parser.add_argument("--clip_delta", type=float, default=-1.0)
parser.add_argument("--optimizer", type=str, choices=["SGD", "adam", "rmsprop", "adamw"], default="SGD")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--train_samples", default=None, type=int, help="Number of training samples per class.")
parser.add_argument("--run_name", default="resnet", help="Fix the save file name.")
parser.add_argument("--early_stopping", default=-1, type=int)
parser.add_argument("--wandb_project", type=str, default="large_scale_laplace-part3")

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

    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    n_classes = 10

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=args["dataset"],
        bs=args["batch_size"],
        data_path=args["data_path"],
        seed=args["seed"],
        n_samples=args["train_samples"],
        cls=classes,
    )
    #Resnet
    # hparams = {
    #         "num_classes" : n_classes,
    #         "c_hidden" :(16, 32, 64),
    #         "num_blocks" : (3, 3, 3),
    #         "act_fn" : nn.relu,
    #         "block_class" : ResNetBlock #PreActResNetBlock #

    # }
    #Densenet
    # hparams = {
    #         "num_classes": 10,
    #         "act_fn": nn.relu
    #         }
    #VisionTransformer
    hparams = {
        "embed_dim": 256,
        "hidden_dim": 512,
        "num_heads": 8,
        "num_layers": 6,
        "patch_size": 4,
        "num_channels": 3,
        "num_patches": 64,
        "num_classes": 10,
        "dropout_prob": 0.1,
    }

    
    trainer, results = train(
        args_dict=args,
        model_name=args["model"],
        model_class=load_obj(MODELS_DICT[args["model"]]),
        model_hparams=(
            hparams
            # {"n_classes": n_classes, "hidden_dim": 64, "num_layers": 3}
            # if args["model"] == "MLP"
            # else {
            #     "n_classes": n_classes,
            # }
        ),
        optimizer_name=args["optimizer"],
        optimizer_hparams=process_optimizer_args(args),
        exmp_imgs=jax.device_put(next(iter(train_loader))["image"]),
        trainermodule=TRAINERS[args["trainer"]],
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=args["n_epochs"],
        wandb_logger=wandb_logger,
    )
    wandb.finish()
    print(results)


if __name__ == "__main__":
    args = parser.parse_args()
    args_dict = vars(args)

    main(args_dict)
