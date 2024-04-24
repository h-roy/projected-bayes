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
from src.models import LeNet, MLP, ResNet_small, ResNetBlock_small, VisionTransformer
from src.losses import cross_entropy_loss, accuracy_preds, nll
from src.helper import calculate_exact_ggn, tree_random_normal_like, compute_num_params
from src.sampling import sample_predictive, sample_hessian_predictive, sample_projections, sample_projections_dataloader, kernel_check, sample_accuracy
from src.helper import set_seed
import wandb
from torch.utils import data

import matplotlib.pyplot as plt
from src.data import MNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN, numpy_collate_fn
from src.ood_functions.evaluate import evaluate
from src.ood_functions.metrics import compute_metrics
# from jax import config
# config.update("jax_disable_jit", True)


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/MNIST/LeNet/OOD_MNIST_seed420", help="path of model")
parser.add_argument("--run_name", default="Proj Sampling", help="Fix the save file name.")
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--sample_seed",  type=int, default=0)
parser.add_argument("--posthoc_precision",  type=float, default=1.0)


if __name__ == "__main__":
    now = datetime.datetime.now()
    now_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    args = parser.parse_args()
    set_seed(args.sample_seed)

    

    param_dict = pickle.load(open("./checkpoints/CIFAR-10/ResNet/epoch200_seed0_params.pickle", "rb"))
    params = param_dict['params']
    batch_stats = param_dict['batch_stats']

     ###############
    ### dataset ###
    n_samples_per_class = None 
    dataset = CIFAR10(seed=args.sample_seed, n_samples_per_class=100)
    dataset_size = len(dataset)
    wandb_project = "large_scale_laplace-part3"
    wandb_logger = wandb.init(project=wandb_project, name=args.run_name, entity="dmiai-mh", config=args)
    wandb.config.update({"dataset_size": dataset_size})

    #############
    ### model ###
    output_dim = 10
    # model = ResNet(
    #     num_classes = output_dim,
    #     c_hidden =(16, 32, 64),
    #     num_blocks = (3, 3, 3),
    #     act_fn = nn.relu,
    #     block_class = ResNetBlock #PreActResNetBlock #
    # )
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
    model = VisionTransformer(**hparams)
    param_dict = pickle.load(open("./checkpoints/ViT2024-03-30-23-30-29/VisionTransformer_CIFAR10_42_params.pickle", "rb"))
    params = param_dict['params']
    model_fn = lambda p, x: model.apply({'params': p},
                                        x,
                                        train=False,
                                        rngs={'dropout': param_dict['rng']})

    ##############
    x_train = jnp.array([data[0] for data in dataset])
    y_train = jnp.array([data[1] for data in dataset])

    start_time = time.time()
    N = 1000
    # bs = 30
    bs = 30
    x_train = x_train[:N]
    y_train = y_train[:N]
    labels = jnp.where(y_train==1.)[1]
    idx = jnp.argsort(labels)
    x_train = x_train[idx,:]
    y_train = y_train[idx,:]
    n_batches = x_train.shape[0] // bs
    x_train_batched = x_train[:n_batches * bs].reshape((n_batches, -1) + x_train.shape[1:])
    model_fn = lambda p, x: model.apply({'params': p, 'batch_stats': batch_stats}, x, train=False, mutable=False)
    x_val = x_train_batched[0]

    # Sample Priors or Load Checkpoints
    n_iterations = 5
    n_samples = args.num_samples
    n_params = compute_num_params(params)

    sample_key = jax.random.PRNGKey(args.sample_seed)
    alpha = args.posthoc_precision
    eps = tree_random_normal_like(sample_key, params, n_samples)
    prior_samples = jax.tree_map(lambda x: 1/jnp.sqrt(alpha) * x, eps)

    #Sample projections
    # posterior_samples = sample_projections(model_fn,
    #                                        params,
    #                                        eps,
    #                                        x_train_batched,
    #                                        output_dim,
    #                                        n_iterations,
    #                                        x_val,
    # )
    dataloader_hparams = {
        "dataset":dataset,
        "batch_size":bs,
        # "shuffle":False,
        "drop_last":True,
        "collate_fn":numpy_collate_fn,
        "sampler":data.sampler.SequentialSampler(dataset)
    }
    train_loader = data.DataLoader(**dataloader_hparams)

    # train_loader = data.DataLoader(
    #     dataset,
    #     batch_size=bs,
    #     # shuffle=False,
    #     drop_last=True,
    #     collate_fn=numpy_collate_fn,
    #     sampler = data.sampler.SequentialSampler(dataset)
    # )
    posterior_samples = sample_projections_dataloader(
                                        model_fn,
                                        params,
                                        eps,
                                        train_loader,
                                        args.sample_seed,
                                        output_dim,
                                        n_iterations,
                                        x_val)


    print(f"Projection Sampling (for a {n_params} parameter model with {n_iterations} steps, {n_samples} samples) took {time.time()-start_time:.5f} seconds")
    pred_posterior = sample_predictive(posterior_samples, params, model_fn, x_train[:N], True, "Pytree")
    map_preds = model_fn(params, x_train[:N])
    print("Kernel check:", kernel_check(posterior_samples, model_fn, params, x_train[:N]))
    print("Distance from Map:", jax.vmap(lambda x,y : jnp.linalg.norm(x - y)/jnp.linalg.norm(y), in_axes=(0,None))(pred_posterior, map_preds))
    print("MAP accuracy:", accuracy_preds(map_preds, y_train[:N])/N * 100)
    print("Predictive accuracy:", sample_accuracy(pred_posterior, y_train[:N]))
    # breakpoint()
    posterior_dict = {
        "posterior_samples": posterior_samples,
    }
    if args.run_name is not None:
        save_name = f"{args.run_name}_seed{args.sample_seed}"
    else:
        save_name = f"started_{now_string}"


    save_path = f"./checkpoints/CIFAR-10/proj_posterior_samples_{save_name}"
    pickle.dump(posterior_dict, open(f"{save_path}_params.pickle", "wb"))
