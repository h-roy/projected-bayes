import time
import argparse
import jax
import matplotlib.pyplot as plt
import datetime
import tree_math as tm
from flax import linen as nn
from jax import nn as jnn
from jax import numpy as jnp
import json
from jax import random, jit
import pickle
from src.models import LeNet, MLP, ResNet, ResNetBlock
from src.losses import cross_entropy_loss, accuracy_preds, nll
from src.helper import calculate_exact_ggn, tree_random_normal_like, compute_num_params
from src.sampling.predictive_samplers import sample_predictive, sample_hessian_predictive
from src.sampling.projection_sampling import sample_projections
from jax import flatten_util
import matplotlib.pyplot as plt
from src.data.datasets import get_rotated_mnist_loaders
from src.data import MNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN, n_classes
from src.ood_functions.evaluate import evaluate
from src.ood_functions.metrics import compute_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/MNIST/LeNet/OOD_MNIST_seed420", help="path of model")
parser.add_argument("--run_name", default=None, help="Fix the save file name.")
parser.add_argument("--diffusion_steps", type=int, default=20)
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--lanczos_iters", type=int, default=1000)
parser.add_argument("--sample_seed",  type=int, default=0)
parser.add_argument("--posthoc_precision",  type=float, default=1.0)


if __name__ == "__main__":
    now = datetime.datetime.now()
    now_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    args = parser.parse_args()

    param_dict = pickle.load(open("./checkpoints/CIFAR-10/ResNet/epoch200_seed0_params.pickle", "rb"))
    params = param_dict['params']
    batch_stats = param_dict['batch_stats']

     ###############
    ### dataset ###
    n_samples_per_class = None 
    cls=list(range(n_classes("CIFAR-10")))
    dataset = CIFAR10(path_root='/dtu/p1/hroy/data', train=True, n_samples_per_class=n_samples_per_class, download=True, cls=cls, seed=0)
    dataset_size = len(dataset)

    #############
    ### model ###
    output_dim = 10
    model = ResNet(
        num_classes = output_dim,
        c_hidden =(16, 32, 64),
        num_blocks = (3, 3, 3),
        act_fn = nn.relu,
        block_class = ResNetBlock #PreActResNetBlock #
    )
    x_train = jnp.array([data[0] for data in dataset])
    y_train = jnp.array([data[1] for data in dataset])
    n_iterations = 15
    n_samples = args.num_samples
    rank = args.lanczos_iters
    n_params = compute_num_params(params)
    alpha = args.posthoc_precision
    sample_key = jax.random.PRNGKey(args.sample_seed)

    start_time = time.time()
    x_train = x_train
    n_batches = x_train.shape[0] // 20
    x_train_batched = x_train[:n_batches * 20].reshape((n_batches, -1) + x_train.shape[1:])
    model_fn = lambda p, x: model.apply({'params': p, 'batch_stats': batch_stats}, x, train=False, mutable=False)
    posterior_samples = sample_projections(model_fn,
                                           params,
                                           x_train_batched,
                                           n_batches,
                                           sample_key,
                                           alpha,
                                           output_dim,
                                           n_samples,
                                           n_iterations,
                                           "regression")  
    breakpoint()                                      
    print(f"Lanczos diffusion (for a {n_params} parameter model with {n_iterations - 1} steps, {n_samples} samples and {rank} iterations) took {time.time()-start_time:.5f} seconds")
    
    posterior_dict = {
        "posterior_samples": posterior_samples,
    }
    if args.run_name is not None:
        save_name = f"{args.run_name}_seed{args.sample_seed}_type_{args.posterior_type}"
    else:
        save_name = f"started_{now_string}"


    save_path = f"./checkpoints/CIFAR-10/proj_posterior_samples_{save_name}"
    pickle.dump(posterior_dict, open(f"{save_path}_params.pickle", "wb"))
