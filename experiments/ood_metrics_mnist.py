import pickle
import os
import argparse
import torch
from jax import random
import json
import datetime
from src.data.all_datasets import get_dataloaders
from src.data.cifar10 import get_cifar10
from src.data.utils import get_output_dim
from src.losses import sse_loss
from src.helper import calculate_exact_ggn, tree_random_normal_like, compute_num_params
from src.sampling.predictive_samplers import sample_predictive, sample_hessian_predictive
from jax import numpy as jnp
import jax
from jax import flatten_util
import matplotlib.pyplot as plt
from src.models import LeNet
from src.data import get_rotated_mnist, get_mnist_ood
from src.ood_functions.evaluate import evaluate, evaluate_map
from src.ood_functions.metrics import compute_metrics
from src.data import MNIST
from collections import defaultdict
import tree_math as tm
from src.ood_functions.metrics import get_auroc
from src.models import MODELS_DICT
from src.helper import load_obj
from src.training.configs import get_model_apply_fn, get_model_hyperparams

# passed as an arg
 # "FMNIST", "CIFAR-10", "SVHN", "CIFAR-100", "ImageNet" and "Imagenette"
# Get OOD datasets corresponding to the dataset
# THe only difficult step is to try the ood datasets and see if they work fine!
# Pass parameter and posterior checkpoint
# Load them
# Evaluate them on id test set
# Evaluate them on OOD test set
# create a dictionary of metrics and save them. create a plot of od vs id metrics and save them
# Ground truth comparison can be done in a notebook!
# Need to modify to include baselines and multiple seeds

# Parse these args
dataset = "MNIST"
# dataset = "CIFAR-10"
checkpoint_path = "./checkpoints/MNIST/LeNet_MNIST_0_params"
posterior_path = "./checkpoints/posterior_MNIST/Projection_Sampling_MNIST_seed0_params"
param_dict = pickle.load(open(f"{checkpoint_path}.pickle", "rb"))['params']
batch_size = 128
num_samples_per_class = 500

val = True
# Load checkpoints
posterior_params_dict = {}
posterior_params_dict["Projection"] = pickle.load(open(f"{posterior_path}.pickle", "rb"))['posterior_samples']
posterior_params_dict["MAP"] = param_dict
posterior_params_dict["Projection"] = jax.vmap(lambda sample: jax.tree_map(lambda x, y: (x - y) * 25 + y, sample, posterior_params_dict["MAP"]))(posterior_params_dict["Projection"])

batch_stats = param_dict['batch_stats'] if 'batch_stats' in param_dict else None
rng = param_dict['rng'] if 'rng' in param_dict else None


# Initialize Model
model_name = "LeNet"
output_dim = get_output_dim(dataset)
n_samples = output_dim * num_samples_per_class
model_class = load_obj(MODELS_DICT[model_name])
model_hparams = get_model_hyperparams(output_dim, model_name)
model = model_class(**model_hparams)    

model_fn = get_model_apply_fn(model_name, model.apply, batch_stats=batch_stats, rng=rng)

# Get OOD Datasets
#ood_mnist works for both id and ood numbers for both MNIST and FMNIST(except no EMNIST)
posterior_types = ["MAP", "Projection"]
metrics_dict = {}
var_dict_dict = {}

for posterior_type in posterior_types:
    metrics_dict[posterior_type] = []
    eval_args = {}
    if posterior_type in ["MAP", "Projection"]:
        eval_args[posterior_type] = {}
        eval_args[posterior_type]["linearised_laplace"] = True
        eval_args[posterior_type]["posterior_sample_type"] = "Pytree"
        eval_args[posterior_type]["likelihood"] = "classification"

    ood_dataset_dict = {}
    if dataset in ["MNIST", "FMNIST"]:
        ood_datasets = ["MNIST", "FMNIST", "KMNIST"]
        for i, id in enumerate(ood_datasets):
            _, val_loader, test_loader = get_mnist_ood(id, batch_size, n_samples_per_class=num_samples_per_class)
            if val:
                loader = val_loader
            else:
                loader = test_loader
            if posterior_type != "MAP":
                some_metrics, all_y_prob, all_y_true, all_y_var = evaluate(loader, posterior_params_dict[posterior_type], posterior_params_dict["MAP"], model_fn, eval_args[posterior_type])
                if id == dataset:
                    all_y_prob_in = all_y_prob
                    all_y_var_in = all_y_var
                more_metrics = compute_metrics(
                    i, id, all_y_prob, loader, all_y_prob_in, all_y_var, benchmark="MNIST-OOD"
                    )
                metrics_dict[posterior_type].append({**some_metrics, **more_metrics})
            else:
                some_metrics, all_y_prob, all_y_true, all_y_var = evaluate_map(loader, posterior_params_dict["MAP"], model_fn, eval_args["MAP"])
                if id == dataset:
                    all_y_prob_in = all_y_prob
                    all_y_var_in = all_y_var
                more_metrics = compute_metrics(
                    i, id, all_y_prob, loader, all_y_prob_in, all_y_var, benchmark="MNIST-OOD"
                    )
                metrics_dict[posterior_type].append({**some_metrics, **more_metrics})

    elif dataset in ["CIFAR-10", "SVHN", "CIFAR-100"]:
        ood_datasets_dict = get_dataloaders("CIFAR-10-OOD", n_samples=1000)
        ood_datasets = ["CIFAR-10", "SVHN", "CIFAR-100"]
        for i, id in enumerate(ood_datasets):
            if val:
                loader = ood_datasets_dict[id +'-val']
            else:
                loader = ood_datasets_dict[id + '-test']

        breakpoint()
        
breakpoint()
eval_args = {}
eval_args["linearised_laplace"] = False
eval_args["posterior_sample_type"] = "Pytree"
eval_args["likelihood"] = "classification"

ids = [0, 15, 30, 60, 90, 120, 150, 180]#, 210, 240, 270, 300, 330, 345, 360]
n_datapoint=500
ood_batch_size = 50
metrics_map = []
for i, id in enumerate(ids):
    # params = params_dict['params']
    train_loader, _, _ = get_rotated_mnist(id, data_path="/dtu/p1/hroy/data", download=True, batch_size=ood_batch_size, n_samples_per_class=500)
    some_metrics, all_y_prob, all_y_true, all_y_var = evaluate_map(train_loader, params, model_fn, eval_args)
    if i == 0:
        all_y_prob_in = all_y_prob
    more_metrics = compute_metrics(
            i, id, all_y_prob, train_loader, all_y_prob_in, all_y_var, benchmark="R-MNIST"
        )
    metrics_map.append({**some_metrics, **more_metrics})
print("MAP AUROC:", get_auroc(all_y_prob_in, all_y_prob))

eval_args = {}
eval_args["linearised_laplace"] = False
eval_args["posterior_sample_type"] = "Pytree"
eval_args["likelihood"] = "classification"

ids = [0, 15, 30, 60, 90, 120, 150, 180]#, 210, 240, 270, 300, 330, 345, 360]
n_datapoint=500
ood_batch_size = 50
metrics_lr = []
for i, id in enumerate(ids):
    train_loader, _, _ = get_rotated_mnist(id, data_path="/dtu/p1/hroy/data", download=True, batch_size=ood_batch_size, n_samples_per_class=500)
    some_metrics, all_y_prob, all_y_true, all_y_var = evaluate(train_loader, proj_samples, params, model_fn, eval_args)
    if i == 0:
        all_y_prob_in = all_y_prob
        all_y_var_in = all_y_var
    more_metrics = compute_metrics(
            i, id, all_y_prob, train_loader, all_y_prob_in, all_y_var, benchmark="R-MNIST"
        )
    metrics_lr.append({**some_metrics, **more_metrics})
print("AUROC:", get_auroc(all_y_var_in, all_y_var))
breakpoint()
    

