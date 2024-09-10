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
from src.helper import calculate_exact_ggn, load_obj, tree_random_normal_like, get_ggn_vector_product, ggn_vector_product_fast
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
    '--xla_gpu_strict_conv_algorithm_picker=false'
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
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--sample_seed",  type=int, default=0)
parser.add_argument("--macro_batch_size",  type=int, default=-1)
parser.add_argument("--sample_batch_size",  type=int, default=32)
parser.add_argument("--posthoc_precision",  type=float, default=1.0)
parser.add_argument("--vmap_dim", type=int, default=5)

if __name__ == "__main__":
    model = "ResNet_small"
    dataset = "ImageNet"
    checkpoint_path = "./checkpoints/CIFAR10/ResNet_small_CIFAR-10_0_params"

    now = datetime.datetime.now()
    now_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    args = parser.parse_args()
    set_seed(args.sample_seed)


    ###############
    ### dataset ###
    ###############
    output_dim = get_output_dim(dataset)
    seed = args.sample_seed
    #############
    ### model ###
    #############
    
    param_dict = pickle.load(open(f"{checkpoint_path}.pickle", "rb"))
    params = param_dict['params']

    batch_stats = param_dict['batch_stats'] if 'batch_stats' in param_dict else None
    rng = param_dict['rng'] if 'rng' in param_dict else None

    model_name = model
    model_class = load_obj(MODELS_DICT[model_name])
    model_hparams = get_model_hyperparams(output_dim, model_name)

    model = model_class(**model_hparams)    
    model_fn = get_model_apply_fn(model_name, model.apply, batch_stats=batch_stats, rng=rng)
    params_vec, unflatten, model_fn_vec = vectorize_nn(model_fn, params)

    start_time = time.time()


    # Sample Priors or Load Checkpoints
    n_samples = args.num_samples
    n_params = len(params_vec)
    print("Number of parameters:", n_params)

    sample_key = jax.random.PRNGKey(seed)
    alpha = args.posthoc_precision
    eps = jax.random.normal(sample_key, (n_params,))

    # train_loader, val_loader, test_loader = get_dataloaders(
    #     dataset_name=dataset,
    #     train_batch_size=128,
    #     val_batch_size=args.sample_batch_size,
    #     data_path='/dtu/p1/hroy/data',
    #     seed=seed,
    #     purp='sample'
    # ) 

    # start_time = time.time()
    # ggn_vp_1 = get_ggn_vector_product(params, model, None, train_loader, "classification", True, batch_stats)
    # ggn_vp_out = ggn_vp_1(eps)
    # print("Time taken for a single GGN Vector Product:", time.time() - start_time)

    sample_key = jax.random.PRNGKey(seed)
    alpha = args.posthoc_precision
    eps = jax.random.normal(sample_key, (1, n_params,))

    train_loader, val_loader, test_loader = get_dataloaders(
            dataset_name=dataset,
            train_batch_size=100000,
            val_batch_size=args.sample_batch_size,
            data_path='/dtu/p1/hroy/data',
            seed=seed,
            purp='sample'
        ) 

    start_time = time.time()
    ggn_vp_2_out = ggn_vector_product_fast(eps, model_fn_vec, params_vec, train_loader, 128, vmap_dim=1, likelihood_type="classification")
    print("Time taken for a single GGN Vector Product:", time.time() - start_time)
    breakpoint()





