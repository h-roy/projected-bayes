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
from src.data.emnist import get_emnist
from src.data.mnist import get_mnist
from src.models import MODELS_DICT, ResNetBlock_small, ResNet_small
from src.training import get_model_hyperparams, get_model_apply_fn

from src.losses import cross_entropy_loss, accuracy_preds, nll
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
parser.add_argument("--dataset", type=str, choices=["MNIST", "FMNIST", "CIFAR-10", "SVHN", "CIFAR-100", "ImageNet"])
parser.add_argument(
    "--model",
    type=str,
    choices=["LeNet", "MLP", "ResNet_small", "ResNet18", "DenseNet", "GoogleNet", "VisionTransformer"],
)
parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/CIFAR10/ResNet_small_CIFAR-10_0_params", help="path of model")
parser.add_argument("--run_name", default="Projection_Sampling_CIFAR10", help="Fix the save file name.")
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--num_iterations", type=int, default=200)
parser.add_argument("--sample_seed",  type=int, default=0)
parser.add_argument("--macro_batch_size",  type=int, default=-1)
parser.add_argument("--sample_batch_size",  type=int, default=32)
parser.add_argument("--posthoc_precision",  type=float, default=1.0)
parser.add_argument("--data_sharding", action="store_true", required=False, default=False)
parser.add_argument("--num_gpus", type=int, default=1)


if __name__ == "__main__":
    now = datetime.datetime.now()
    now_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    args = parser.parse_args()
    set_seed(args.sample_seed)


    ###############
    ### dataset ###
    ###############
    dataset = args.dataset
    output_dim = get_output_dim(dataset)
    seed = args.sample_seed
    train_loader, val_loader, test_loader = get_dataloaders(
            dataset_name=dataset,
            train_batch_size=args.macro_batch_size,
            val_batch_size=args.sample_batch_size,
            data_path='/dtu/p1/hroy/data',
            seed=seed,
            purp='sample'
        ) 

    x_val = next(iter(val_loader))['image']

    
    #############
    ### model ###
    #############
    
    param_dict = pickle.load(open(f"{args.checkpoint_path}.pickle", "rb"))
    params = param_dict['params']

    batch_stats = param_dict['batch_stats'] if 'batch_stats' in param_dict else None
    rng = param_dict['rng'] if 'rng' in param_dict else None

    model_name = args.model
    model_class = load_obj(MODELS_DICT[model_name])
    model_hparams = get_model_hyperparams(output_dim, model_name)

    model = model_class(**model_hparams)    
    model_fn = get_model_apply_fn(model_name, model.apply, batch_stats=batch_stats, rng=rng)
    params_vec, unflatten, model_fn_vec = vectorize_nn(model_fn, params)
    wandb_project = "large_scale_laplace-part3"
    wandb_logger = wandb.init(project=wandb_project, name=args.run_name, entity="dmiai-mh", config=args)

    start_time = time.time()


    # Sample Priors or Load Checkpoints
    n_iterations = args.num_iterations
    n_samples = args.num_samples
    n_params = len(params_vec)
    print("Number of parameters:", n_params)

    sample_key = jax.random.PRNGKey(seed)

    alpha = args.posthoc_precision

    eps = jax.random.normal(sample_key, (n_samples, n_params))

    #Sample projections
    data_sharding = args.data_sharding
    num_gpus = args.num_gpus
    posterior_samples, metrics = sample_projections_dataloader(
                                                      model_fn_vec,
                                                      params_vec,
                                                      eps,
                                                      alpha,
                                                      train_loader,
                                                      args.sample_batch_size,
                                                      seed,
                                                      output_dim,
                                                      n_iterations,
                                                      x_val,
                                                      n_params,
                                                      unflatten,
                                                      True,
                                                      data_sharding,
                                                      num_gpus
                                                      )

    print(f"Projection Sampling (for a {n_params} parameter model with {n_iterations} steps, {n_samples} samples) took {time.time()-start_time:.5f} seconds")
    metrics["time"] = time.time()-start_time
    metrics["num_params"] = n_params
    metrics["num_samples"] = n_samples
    metrics["num_iterations"] = n_iterations
    posterior_dict = {
        "posterior_samples": posterior_samples,
    }
    os.makedirs(f"./checkpoints/posterior_samples/{args.dataset}/{args.model}", exist_ok=True)
    if args.run_name is not None:
        save_name = f"{args.run_name}_sample_seed_{args.sample_seed}"
    else:
        save_name = f"started_{now_string}"


    save_path = f"./checkpoints/posterior_samples/{args.dataset}/{args.model}/{save_name}"
    pickle.dump(posterior_dict, open(f"{save_path}_params.pickle", "wb"))
    pickle.dump(metrics, open(f"{save_path}_metrics.pickle", "wb"))
