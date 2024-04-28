import pickle
import os
import argparse
import torch
from jax import random
import json
import datetime
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

train_loader, _ = get_mnist_ood("EMNIST", data_path="/dtu/p1/hroy/data", download=True, batch_size=500, n_samples_per_class=500)
params = pickle.load(open("./checkpoints/MNIST/LeNet_MNIST_0_params.pickle", "rb"))['params']
proj_samples = pickle.load(open("./checkpoints/posterior_MNIST/Projection Sampling MNIST_seed0_params.pickle", "rb"))['posterior_samples']

model = LeNet(output_dim=10, activation="tanh")
model_fn = model.apply

print("Mean distance:", jax.vmap(lambda x: (tm.Vector(x) - tm.Vector(params)) @ (tm.Vector(x) - tm.Vector(params)))(proj_samples).mean())
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
    

