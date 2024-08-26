import os
import time
import argparse
import jax
import matplotlib.pyplot as plt
import datetime
import tree_math as tm
from flax import linen as nn
from jax import nn as jnn
from jax import numpy as jnp
import pickle
from jax_models import load_model

from src.data.emnist import get_emnist
from src.data.mnist import get_mnist
from src.models import MODELS_DICT, ResNetBlock_small, ResNet_small
from src.sampling.projection_loss_sampling import sample_loss_projections_dataloader
from src.training import get_model_hyperparams, get_imagenet_model_fn

from src.losses import cross_entropy_loss, accuracy_preds, cross_entropy_loss_per_datapoint, nll
from src.helper import calculate_exact_ggn, load_obj, tree_random_normal_like, compute_num_params
from src.sampling import sample_projections_dataloader, vectorize_nn
from src.helper import set_seed
from src.data import get_cifar10, get_dataloaders, get_output_dim
import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)

import wandb
from torch.utils import data

import matplotlib.pyplot as plt
from src.data import get_cifar10, numpy_collate_fn
# from jax import config
# config.update("jax_disable_jit", True)

# Samples from daigonal GGN might be better because -> normal distributions 
# basically orthogonal to the plane and project down to the MAP param

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    choices=["swin-small-224", "swin-tiny-224", 
             "van-base", "van-tiny", "van-small", "van-large",
             "pvit-b0", "pvit-b1", "pvit-b2", "pvit-b2-linear", "pvit-b3", "pvit-b4", "pvit-b5",
             "convnext-small", "convnext-tiny", "convnext-large-224-1k", "convnext-base-224-1k",
             "cait-s24-224", "cait-xxs36-224"],
    default="swin-small-224",
)
parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/CIFAR10/ResNet_small_CIFAR-10_0_params", help="path of model")
parser.add_argument("--run_name", default="Projection_Sampling_CIFAR10", help="Fix the save file name.")
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--num_iterations", type=int, default=200)
parser.add_argument("--sample_seed",  type=int, default=0)
parser.add_argument("--macro_batch_size",  type=int, default=1000)
parser.add_argument("--sample_batch_size",  type=int, default=16)
parser.add_argument("--posthoc_precision",  type=float, default=1.0)
parser.add_argument("--acceleration", action="store_true", required=False, default=False)


if __name__ == "__main__":
    now = datetime.datetime.now()
    now_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    args = parser.parse_args()
    set_seed(args.sample_seed)


    ###############
    ### dataset ###
    ###############
    output_dim = 1000
    seed = args.sample_seed
    train_loader, val_loader, test_loader = get_dataloaders(
            dataset_name="ImageNet",
            train_batch_size=args.macro_batch_size,
            val_batch_size=args.sample_batch_size,
            data_path='/dtu/p1/hroy/data',
            seed=seed,
            purp='sample',
        ) 

    batch = next(iter(val_loader))
    img, label = batch["image"], batch["label"]
    x_val, y_val = jnp.asarray(img, dtype=float), jnp.asarray(label, dtype=float)

    
    #############
    ### model ###
    #############
    rng = jax.random.PRNGKey(seed)
    model_rng, split_rng = jax.random.split(rng)

    if args.model[:3] == "van":
        model, params, batch_stats = load_model(args.model, attach_head=True, num_classes=1000, dropout=0.0, pretrained=True)
    else:
        model, params = load_model(args.model, attach_head=True, num_classes=1000, dropout=0.0, pretrained=True)
        batch_stats = None
    model_fn = get_imagenet_model_fn(args.model, model, model_rng, batch_stats=batch_stats)
    loss_fn = cross_entropy_loss_per_datapoint
    model_name = args.model
    params_vec, unflatten, model_fn_vec = vectorize_nn(model_fn, params)
    wandb_project = "large_scale_laplace-part3"
    wandb_logger = wandb.init(project=wandb_project, name=args.run_name, entity="dmiai-mh", config=args)

    start_time = time.time()


    # Sample Priors or Load Checkpoints
    n_iterations = args.num_iterations
    n_samples = args.num_samples
    n_params = len(params_vec)
    print("Number of parameters:", n_params)


    alpha = args.posthoc_precision
    sample_key, split_rng = jax.random.split(split_rng)
    eps = jax.random.normal(sample_key, (n_samples, n_params))
    #Sample projections
    posterior_samples, metrics = sample_loss_projections_dataloader(
                                                      model_fn_vec,
                                                      loss_fn,
                                                      params_vec,
                                                      eps,
                                                      alpha,
                                                      train_loader,
                                                      args.sample_batch_size,
                                                      seed,
                                                      n_iterations,
                                                      x_val,
                                                      y_val,
                                                      n_params,
                                                      unflatten,
                                                      True,
                                                      args.acceleration
                                                      )

    print(f"Projection Sampling (for a {n_params} parameter model with {n_iterations} steps, {n_samples} samples) took {time.time()-start_time:.5f} seconds")
    metrics["time"] = time.time()-start_time
    metrics["num_params"] = n_params
    metrics["num_samples"] = n_samples
    metrics["num_iterations"] = n_iterations
    posterior_dict = {
        "posterior_samples": posterior_samples,
    }
    os.makedirs(f"./checkpoints/loss_kernel_samples/ImageNet/{args.model}", exist_ok=True)
    if args.run_name is not None:
        save_name = f"{args.run_name}_sample_seed_{args.sample_seed}"
    else:
        save_name = f"started_{now_string}"


    save_path = f"./checkpoints/loss_kernel_samples/ImageNet/{args.model}/{save_name}"
    pickle.dump(posterior_dict, open(f"{save_path}_params.pickle", "wb"))
    pickle.dump(metrics, open(f"{save_path}_metrics.pickle", "wb"))
