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
from src.models import LeNet, MLP, ResNet_small, ResNetBlock_small, VisionTransformer
from src.losses import cross_entropy_loss, accuracy_preds, nll
from src.helper import calculate_exact_ggn, tree_random_normal_like, compute_num_params
from src.sampling import sample_projections, vectorize_nn
from src.helper import set_seed
import wandb
from torch.utils import data

import matplotlib.pyplot as plt
from src.data import MNIST, FashionMNIST, numpy_collate_fn
# from jax import config
# config.update("jax_disable_jit", True)

# Samples from daigonal GGN might be better because -> normal distributions 
# basically orthogonal to the plane and project down to the MAP param

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/MNIST/LeNet_MNIST_0_params", help="path of model")
parser.add_argument("--run_name", default="Projection_Sampling_MNIST", help="Fix the save file name.")
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--num_iterations", type=int, default=200)
parser.add_argument("--sample_seed",  type=int, default=0)
parser.add_argument("--sample_batch_size",  type=int, default=32)
parser.add_argument("--posthoc_precision",  type=float, default=1.0)


if __name__ == "__main__":
    now = datetime.datetime.now()
    now_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    args = parser.parse_args()
    set_seed(args.sample_seed)

    param_dict = pickle.load(open(f"{args.checkpoint_path}.pickle", "rb"))
    params = param_dict['params']

    ###############
    ### dataset ###
    ###############

    n_samples_per_class = None 
    classes = list(range(10))
    seed = args.sample_seed
    dataset = MNIST(
        train=True,
        n_samples_per_class=n_samples_per_class,
        classes=classes,
        seed=seed,
        download=True, 
    )
    x_train = jnp.array([data[0] for data in dataset])
    y_train = jnp.array([data[1] for data in dataset])

    N = len(dataset)
    bs = args.sample_batch_size
    labels = jnp.where(y_train==1.)[1]
    idx = jnp.argsort(labels)
    x_train = x_train[idx,:]
    y_train = y_train[idx,:]
    n_batches = x_train.shape[0] // bs
    x_train_batched = x_train[:n_batches * bs].reshape((n_batches, -1) + x_train.shape[1:])

    ############
    ### model ##
    ############

    output_dim = 10
    hparams = {
        "output_dim": output_dim,
        "activation": "tanh"
    }
    
    model = LeNet(**hparams)
    params_vec, unflatten, model_fn = vectorize_nn(model.apply, params)
    
    wandb_project = "large_scale_laplace-part3"
    wandb_logger = wandb.init(project=wandb_project, name=args.run_name, entity="dmiai-mh", config=args)

    start_time = time.time()

    x_val = x_train_batched[0]

    # Sample Priors or Load Checkpoints
    n_iterations = args.num_iterations
    n_samples = args.num_samples
    n_params = compute_num_params(params)
    print("Number of parameters:", n_params)

    sample_key = jax.random.PRNGKey(seed)

    alpha = args.posthoc_precision

    eps = jax.random.normal(sample_key, (n_samples, n_params))

    #Sample projections

    posterior_samples = sample_projections(model_fn,
                                           params_vec,
                                           eps,
                                           alpha,
                                           x_train_batched,
                                           output_dim,
                                           n_iterations,
                                           x_val,
                                           n_params,
                                           unflatten,
    )

    print(f"Projection Sampling (for a {n_params} parameter model with {n_iterations} steps, {n_samples} samples) took {time.time()-start_time:.5f} seconds")
    posterior_dict = {
        "posterior_samples": posterior_samples,
    }
    os.makedirs("./checkpoints/posterior_MNIST/", exist_ok=True)
    if args.run_name is not None:
        save_name = f"{args.run_name}_seed{args.sample_seed}"
    else:
        save_name = f"started_{now_string}"


    save_path = f"./checkpoints/posterior_MNIST/{save_name}"
    pickle.dump(posterior_dict, open(f"{save_path}_params.pickle", "wb"))
