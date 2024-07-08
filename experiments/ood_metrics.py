import pickle
import os
import argparse
import torch
from jax import random
import json
import datetime
from src.data.all_datasets import get_dataloaders, get_ood_datasets
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
from sklearn.metrics import roc_auc_score
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["MNIST", "FMNIST", "CIFAR-10", "SVHN", "CIFAR-100", "ImageNet"])
parser.add_argument(
    "--model",
    type=str,
    choices=["LeNet", "MLP", "ResNet_small", "ResNet18", "DenseNet", "GoogleNet", "VisionTransformer"],
)
parser.add_argument("--posterior_type", type=str, choices=["MAP", "Projection"])
parser.add_argument("--experiment", type=str, choices=["MNIST-OOD", "R-MNIST", "FMNIST-OOD", "R-FMNIST", "CIFAR-10-OOD", "CIFAR-10-C", "SVHN-OOD", "CIFAR-100-OOD", "ImageNet-OOD"])
parser.add_argument("--parameter_path", type=str, default="./checkpoints/MNIST/LeNet_MNIST_0_params")
parser.add_argument("--posterior_path", type=str, default="./checkpoints/posterior_samples/MNIST/LeNet/mnist_samples_seed_0_params")
parser.add_argument("--ood_batch_size", type=int, default=128)
parser.add_argument("--num_samples_per_class", type=int, default=500)
parser.add_argument("--val", action="store_true", required=False, default=False)
parser.add_argument("--prior_precision", type=float, default=None)
args = parser.parse_args()

dataset = args.dataset 
model_name = args.model 
posterior_type = args.posterior_type 
experiment = args.experiment 

parameter_path = args.parameter_path 
posterior_path = args.posterior_path 

def auroc(scores_id, scores_ood):
    labels = np.zeros(len(scores_id) + len(scores_ood), dtype="int32")
    labels[len(scores_id) :] = 1
    scores = np.concatenate([scores_id, scores_ood])
    return roc_auc_score(labels, scores)


param_dict = pickle.load(open(f"{parameter_path}.pickle", "rb"))
params = param_dict['params']
batch_stats = param_dict['batch_stats'] if 'batch_stats' in param_dict else None
rng = param_dict['rng'] if 'rng' in param_dict else None

posterior_dict = pickle.load(open(f"{posterior_path}.pickle", "rb"))['posterior_samples']
prior_precision = args.prior_precision
if prior_precision is not None:
    posterior_dict = jax.vmap(lambda sample: jax.tree_map(lambda x, y: (x - y) * prior_precision + y, sample, params))(posterior_dict)
ood_batch_size = args.ood_batch_size 
num_samples_per_class = args.num_samples_per_class 
val = args.val 

# Initialize Model
output_dim = get_output_dim(dataset)
n_samples = output_dim * num_samples_per_class
model_class = load_obj(MODELS_DICT[model_name])
model_hparams = get_model_hyperparams(output_dim, model_name)
model = model_class(**model_hparams)    

model_fn = get_model_apply_fn(model_name, model.apply, batch_stats=batch_stats, rng=rng)

ood_datasets_dict = get_ood_datasets(experiment, ood_batch_size=ood_batch_size, n_samples=n_samples)
ids = list(ood_datasets_dict.keys())
if experiment[-3:] == 'OOD':
    idx = ids.index(dataset)
    ids[0], ids[idx] = ids[idx], ids[0]

evaluate_fn = evaluate_map if posterior_type == "MAP" else evaluate

eval_args = {}
eval_args["posterior_sample_type"] = "Pytree"
eval_args["likelihood"] = "classification"

if posterior_type in ["Projection"]:
    eval_args["linearised_laplace"] = True
elif posterior_type in ["MAP"]:
    eval_args["linearised_laplace"] = False

results_dict = defaultdict(dict)
results_dict[posterior_type] = defaultdict(dict)

for i, id in enumerate(ids):
    data_loader = ood_datasets_dict[id]
    if posterior_type == "MAP":
        some_metrics, all_y_prob, all_y_true, all_y_var = evaluate_map(data_loader, params, model_fn, eval_args)
    else:
        some_metrics, all_y_prob, all_y_true, all_y_var = evaluate(data_loader, posterior_dict, params, model_fn, eval_args)
    if i == 0:
        all_y_prob_in = all_y_prob
        id_score = all_y_var.max(axis=-1) if posterior_type != "MAP" else None # Can also be mean
    more_metrics = compute_metrics(
            i, id, all_y_prob, data_loader, all_y_prob_in, all_y_var, benchmark=experiment
        )
    if i != 0:
        ood_score = all_y_var.max(axis=-1) if posterior_type != "MAP" else None
        more_metrics['score_auroc'] = auroc(id_score, ood_score) if posterior_type != "MAP" else None
    results_dict[posterior_type][id] = {**some_metrics, **more_metrics}
# python experiments/ood_metrics.py --dataset MNIST --model LeNet --posterior_type Projection --experiment MNIST-OOD --parameter_path ./checkpoints/MNIST/LeNet_MNIST_0_params --posterior_path ./checkpoints/posterior_samples/MNIST/LeNet/mnist_samples_seed_0_params --ood_batch_size 128 --num_samples_per_class 500
os.makedirs(f"./results/{args.dataset}/{args.model}/{args.experiment}", exist_ok=True)

save_path = f"./results/{args.dataset}/{args.model}/{args.experiment}/{posterior_type}"
pickle.dump(posterior_dict, open(f"{save_path}_metrics.pickle", "wb"))
