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
from src.sampling import ood_projections
from jax import numpy as jnp
import jax
from jax import flatten_util
import matplotlib.pyplot as plt
from src.models import LeNet
from src.data import get_rotated_mnist, get_mnist_ood
from src.ood_functions.evaluate import evaluate, evaluate_map, evaluate_ensembles
from src.ood_functions.metrics import compute_metrics
from src.data import MNIST
from collections import defaultdict
import tree_math as tm
from src.ood_functions.metrics import get_auroc
from src.models import MODELS_DICT
from src.helper import load_obj
from src.sampling.sample_utils import vectorize_nn
from src.training.configs import get_model_apply_fn, get_model_hyperparams
from sklearn.metrics import roc_auc_score
import numpy as np
from src.sampling import linearize_model_fn
parser = argparse.ArgumentParser()
parser.add_argument("--corruption", type=str, choices=["fog", "zoom_blur", "gaussian_blur", "snow", "brightness", "contrast"], default="fog")
parser.add_argument("--posterior_type", type=str, choices=["MAP", "Projection", "Loss-Kernel", "Exact_Diag", "Hutchinson_Diag", "SWAG", "Subnetwork"])
parser.add_argument("--experiment", type=str, choices=["MNIST-OOD", "R-MNIST", "FMNIST-OOD", "R-FMNIST", "CIFAR-10-OOD", "CIFAR-10-C", "SVHN-OOD", "CIFAR-100-OOD", "ImageNet-OOD"])
parser.add_argument("--parameter_path", type=str, default="./checkpoints/MNIST/LeNet_MNIST_0_params")
parser.add_argument("--ood_batch_size", type=int, default=128)
parser.add_argument("--num_samples_per_class", type=int, default=500)

args = parser.parse_args()

def auroc(scores_id, scores_ood):
    labels = np.zeros(len(scores_id) + len(scores_ood), dtype="int32")
    labels[len(scores_id) :] = 1
    scores = np.concatenate([scores_id, scores_ood])
    return roc_auc_score(labels, scores)

ood_batch_size = args.ood_batch_size
num_samples_per_class = args.num_samples_per_class
# MNIST
# model_name = "LeNet"
# output_dim = get_output_dim("MNIST")
# n_samples = output_dim * num_samples_per_class
# model_class = load_obj(MODELS_DICT[model_name])
# model_hparams = get_model_hyperparams(output_dim, model_name)
# model = model_class(**model_hparams)    

# param_1_path = "./checkpoints/MNIST/LeNet_MNIST_0_params"
# param_1_dict =  pickle.load(open(f"{param_1_path}.pickle", "rb"))
# param_1 = param_1_dict['params']
# batch_stats_1 = param_1_dict['batch_stats'] if 'batch_stats' in param_1_dict else None
# rng_1 = param_1_dict['rng'] if 'rng' in param_1_dict else None
# model_fn_1 = get_model_apply_fn(model_name, model.apply, batch_stats=batch_stats_1, rng=rng_1)


# param_2_path = "./checkpoints/MNIST/LeNet_MNIST_1_params"
# param_2_dict =  pickle.load(open(f"{param_2_path}.pickle", "rb"))
# param_2 = param_2_dict['params']
# batch_stats_2 = param_2_dict['batch_stats'] if 'batch_stats' in param_2_dict else None
# rng_2 = param_2_dict['rng'] if 'rng' in param_2_dict else None
# model_fn_2 = get_model_apply_fn(model_name, model.apply, batch_stats=batch_stats_2, rng=rng_2)

# param_3_path = "./checkpoints/MNIST/LeNet_MNIST_2_params"
# param_3_dict =  pickle.load(open(f"{param_3_path}.pickle", "rb"))
# param_3 = param_3_dict['params']
# batch_stats_3 = param_3_dict['batch_stats'] if 'batch_stats' in param_3_dict else None
# rng_3 = param_3_dict['rng'] if 'rng' in param_3_dict else None
# model_fn_3 = get_model_apply_fn(model_name, model.apply, batch_stats=batch_stats_3, rng=rng_3)

# r_mnist = get_ood_datasets("R-MNIST", ood_batch_size=ood_batch_size, n_samples=n_samples, corruption=args.corruption, seed=10)
# ids = list(r_mnist.keys())

# eval_args = {}
# eval_args["posterior_sample_type"] = "Pytree"
# eval_args["likelihood"] = "classification"
# eval_args["linearised_laplace"] = False

# results_dict_rmnist = defaultdict(dict)

# for i, id in enumerate(ids):
#     data_loader = r_mnist[id]
#     some_metrics, all_y_prob, all_y_true, all_y_var, all_y_sample_probs = evaluate_ensembles(data_loader, param_1, param_2, param_3, model_fn_1, model_fn_2, model_fn_3, eval_args)
#     if i == 0:
#         all_y_prob_in = all_y_prob
#         all_y_sample_probs_in = all_y_sample_probs
#         id_score = all_y_var.max(axis=-1)
#     more_metrics = compute_metrics(
#             i, id, all_y_prob, data_loader, all_y_prob_in, all_y_var, benchmark="R-MNIST"
#         )
#     if i != 0:
#         all_y_sample_probs_out = all_y_sample_probs
#         ood_score = all_y_var.max(axis=-1)
#         more_metrics['score_auroc'] = auroc(id_score, ood_score)
#     if i > 0:
#         break

#     results_dict_rmnist[id] = {**some_metrics, **more_metrics}
# save_path = "./results/MNIST/LeNet/R-MNIST/ensembles"
# pickle.dump(results_dict_rmnist, open(f"{save_path}_metrics.pickle", "wb"))


# mnist_ood = get_ood_datasets("MNIST-OOD", ood_batch_size=ood_batch_size, n_samples=n_samples, corruption=args.corruption, seed=10)
# ids = list(mnist_ood.keys())
# idx = ids.index("MNIST")
# ids[0], ids[idx] = ids[idx], ids[0]

# eval_args = {}
# eval_args["posterior_sample_type"] = "Pytree"
# eval_args["likelihood"] = "classification"
# eval_args["linearised_laplace"] = False

# results_dict_mnist_ood = defaultdict(dict)

# for i, id in enumerate(ids):
#     data_loader = mnist_ood[id]
#     some_metrics, all_y_prob, all_y_true, all_y_var, all_y_sample_probs = evaluate_ensembles(data_loader, param_1, param_2, param_3, model_fn_1, model_fn_2, model_fn_3, eval_args)
#     if i == 0:
#         all_y_prob_in = all_y_prob
#         all_y_sample_probs_in = all_y_sample_probs
#         id_score = all_y_var.max(axis=-1)
#     more_metrics = compute_metrics(
#             i, id, all_y_prob, data_loader, all_y_prob_in, all_y_var, benchmark="MNIST-OOD"
#         )
#     if i != 0:
#         all_y_sample_probs_out = all_y_sample_probs
#         ood_score = all_y_var.max(axis=-1)
#         more_metrics['score_auroc'] = auroc(id_score, ood_score)

#     results_dict_mnist_ood[id] = {**some_metrics, **more_metrics}
# save_path = "./results/MNIST/LeNet/MNIST-OOD/ensembles"
# pickle.dump(results_dict_mnist_ood, open(f"{save_path}_metrics.pickle", "wb"))

# #FMNIST
# param_1_path = "./checkpoints/FMNIST/LeNet_FMNIST_0_params"
# param_1_dict =  pickle.load(open(f"{param_1_path}.pickle", "rb"))
# param_1 = param_1_dict['params']
# batch_stats_1 = param_1_dict['batch_stats'] if 'batch_stats' in param_1_dict else None
# rng_1 = param_1_dict['rng'] if 'rng' in param_1_dict else None
# model_fn_1 = get_model_apply_fn(model_name, model.apply, batch_stats=batch_stats_1, rng=rng_1)


# param_2_path = "./checkpoints/FMNIST/LeNet_FMNIST_1_params"
# param_2_dict =  pickle.load(open(f"{param_2_path}.pickle", "rb"))
# param_2 = param_2_dict['params']
# batch_stats_2 = param_2_dict['batch_stats'] if 'batch_stats' in param_2_dict else None
# rng_2 = param_2_dict['rng'] if 'rng' in param_2_dict else None
# model_fn_2 = get_model_apply_fn(model_name, model.apply, batch_stats=batch_stats_2, rng=rng_2)

# param_3_path = "./checkpoints/FMNIST/LeNet_FMNIST_2_params"
# param_3_dict =  pickle.load(open(f"{param_3_path}.pickle", "rb"))
# param_3 = param_3_dict['params']
# batch_stats_3 = param_3_dict['batch_stats'] if 'batch_stats' in param_3_dict else None
# rng_3 = param_3_dict['rng'] if 'rng' in param_3_dict else None
# model_fn_3 = get_model_apply_fn(model_name, model.apply, batch_stats=batch_stats_3, rng=rng_3)


# r_fmnist = get_ood_datasets("R-FMNIST", ood_batch_size=ood_batch_size, n_samples=n_samples, corruption=args.corruption, seed=10)
# ids = list(r_fmnist.keys())

# eval_args = {}
# eval_args["posterior_sample_type"] = "Pytree"
# eval_args["likelihood"] = "classification"
# eval_args["linearised_laplace"] = False

# results_dict_rfmnist = defaultdict(dict)

# for i, id in enumerate(ids):
#     data_loader = r_fmnist[id]
#     some_metrics, all_y_prob, all_y_true, all_y_var, all_y_sample_probs = evaluate_ensembles(data_loader, param_1, param_2, param_3, model_fn_1, model_fn_2, model_fn_3, eval_args)
#     if i == 0:
#         all_y_prob_in = all_y_prob
#         all_y_sample_probs_in = all_y_sample_probs
#         id_score = all_y_var.max(axis=-1)
#     more_metrics = compute_metrics(
#             i, id, all_y_prob, data_loader, all_y_prob_in, all_y_var, benchmark="R-FMNIST"
#         )
#     if i != 0:
#         all_y_sample_probs_out = all_y_sample_probs
#         ood_score = all_y_var.max(axis=-1)
#         more_metrics['score_auroc'] = auroc(id_score, ood_score)
#     if i > 0:
#         break
#     results_dict_rfmnist[id] = {**some_metrics, **more_metrics}
# save_path = "./results/FMNIST/LeNet/R-FMNIST/ensembles"
# pickle.dump(results_dict_rfmnist, open(f"{save_path}_metrics.pickle", "wb"))


# fmnist_ood = get_ood_datasets("FMNIST-OOD", ood_batch_size=ood_batch_size, n_samples=n_samples, corruption=args.corruption, seed=10)
# ids = list(fmnist_ood.keys())
# idx = ids.index("FMNIST")
# ids[0], ids[idx] = ids[idx], ids[0]

# eval_args = {}
# eval_args["posterior_sample_type"] = "Pytree"
# eval_args["likelihood"] = "classification"
# eval_args["linearised_laplace"] = False

# results_dict_fmnist_ood = defaultdict(dict)

# for i, id in enumerate(ids):
#     data_loader = fmnist_ood[id]
#     some_metrics, all_y_prob, all_y_true, all_y_var, all_y_sample_probs = evaluate_ensembles(data_loader, param_1, param_2, param_3, model_fn_1, model_fn_2, model_fn_3, eval_args)
#     if i == 0:
#         all_y_prob_in = all_y_prob
#         all_y_sample_probs_in = all_y_sample_probs
#         id_score = all_y_var.max(axis=-1)
#     more_metrics = compute_metrics(
#             i, id, all_y_prob, data_loader, all_y_prob_in, all_y_var, benchmark="FMNIST-OOD"
#         )
#     if i != 0:
#         all_y_sample_probs_out = all_y_sample_probs
#         ood_score = all_y_var.max(axis=-1)
#         more_metrics['score_auroc'] = auroc(id_score, ood_score)

#     results_dict_fmnist_ood[id] = {**some_metrics, **more_metrics}
# save_path = "./results/FMNIST/LeNet/FMNIST-OOD/ensembles"
# pickle.dump(results_dict_fmnist_ood, open(f"{save_path}_metrics.pickle", "wb"))

#CIFAR10

# model_name = "ResNet_small"
# output_dim = get_output_dim("CIFAR-10")
# n_samples = output_dim * num_samples_per_class
# model_class = load_obj(MODELS_DICT[model_name])
# model_hparams = get_model_hyperparams(output_dim, model_name)
# model = model_class(**model_hparams)    

# param_1_path = "./checkpoints/CIFAR10/ResNet_small_CIFAR-10_0_params"
# param_1_dict =  pickle.load(open(f"{param_1_path}.pickle", "rb"))
# param_1 = param_1_dict['params']
# batch_stats_1 = param_1_dict['batch_stats'] if 'batch_stats' in param_1_dict else None
# rng_1 = param_1_dict['rng'] if 'rng' in param_1_dict else None
# model_fn_1 = get_model_apply_fn(model_name, model.apply, batch_stats=batch_stats_1, rng=rng_1)


# param_2_path = "./checkpoints/CIFAR10/ResNet_small_CIFAR-10_1_params"
# param_2_dict =  pickle.load(open(f"{param_2_path}.pickle", "rb"))
# param_2 = param_2_dict['params']
# batch_stats_2 = param_2_dict['batch_stats'] if 'batch_stats' in param_2_dict else None
# rng_2 = param_2_dict['rng'] if 'rng' in param_2_dict else None
# model_fn_2 = get_model_apply_fn(model_name, model.apply, batch_stats=batch_stats_2, rng=rng_2)

# param_3_path = "./checkpoints/CIFAR10/ResNet_small_CIFAR-10_2_params"
# param_3_dict =  pickle.load(open(f"{param_3_path}.pickle", "rb"))
# param_3 = param_3_dict['params']
# batch_stats_3 = param_3_dict['batch_stats'] if 'batch_stats' in param_3_dict else None
# rng_3 = param_3_dict['rng'] if 'rng' in param_3_dict else None
# model_fn_3 = get_model_apply_fn(model_name, model.apply, batch_stats=batch_stats_3, rng=rng_3)

# cifar_c = get_ood_datasets("CIFAR-10-C", ood_batch_size=ood_batch_size, n_samples=n_samples, corruption=args.corruption, seed=10)
# ids = list(cifar_c.keys())
# eval_args = {}
# eval_args["posterior_sample_type"] = "Pytree"
# eval_args["likelihood"] = "classification"
# eval_args["linearised_laplace"] = False

# results_dict_cifar10 = defaultdict(dict)

# for i, id in enumerate(ids):
#     data_loader = cifar_c[id]
#     some_metrics, all_y_prob, all_y_true, all_y_var, all_y_sample_probs = evaluate_ensembles(data_loader, param_1, param_2, param_3, model_fn_1, model_fn_2, model_fn_3, eval_args)
#     if i == 0:
#         all_y_prob_in = all_y_prob
#         all_y_sample_probs_in = all_y_sample_probs
#         id_score = all_y_var.max(axis=-1)
#     more_metrics = compute_metrics(
#             i, id, all_y_prob, data_loader, all_y_prob_in, all_y_var, benchmark="CIFAR-10-C"
#         )
#     if i != 0:
#         all_y_sample_probs_out = all_y_sample_probs
#         ood_score = all_y_var.max(axis=-1)
#         more_metrics['score_auroc'] = auroc(id_score, ood_score)
#     if i > 0:
#         break

#     results_dict_cifar10[id] = {**some_metrics, **more_metrics}
# save_path = "./results/CIFAR-10/ResNet_small/CIFAR-10-C/ensembles"
# pickle.dump(results_dict_cifar10, open(f"{save_path}_metrics.pickle", "wb"))


# cifar_ood = get_ood_datasets("CIFAR-10-OOD", ood_batch_size=ood_batch_size, n_samples=n_samples, corruption=args.corruption, seed=10)
# ids = list(cifar_ood.keys())
# idx = ids.index("CIFAR-10")
# ids[0], ids[idx] = ids[idx], ids[0]

# eval_args = {}
# eval_args["posterior_sample_type"] = "Pytree"
# eval_args["likelihood"] = "classification"
# eval_args["linearised_laplace"] = False

# results_dict_cifar_ood = defaultdict(dict)

# for i, id in enumerate(ids):
#     data_loader = cifar_ood[id]
#     some_metrics, all_y_prob, all_y_true, all_y_var, all_y_sample_probs = evaluate_ensembles(data_loader, param_1, param_2, param_3, model_fn_1, model_fn_2, model_fn_3, eval_args)
#     if i == 0:
#         all_y_prob_in = all_y_prob
#         all_y_sample_probs_in = all_y_sample_probs
#         id_score = all_y_var.max(axis=-1)
#     more_metrics = compute_metrics(
#             i, id, all_y_prob, data_loader, all_y_prob_in, all_y_var, benchmark="CIFAR-10-OOD"
#         )
#     if i != 0:
#         all_y_sample_probs_out = all_y_sample_probs
#         ood_score = all_y_var.max(axis=-1)
#         more_metrics['score_auroc'] = auroc(id_score, ood_score)

#     results_dict_cifar_ood[id] = {**some_metrics, **more_metrics}
# save_path = "./results/CIFAR-10/ResNet_small/CIFAR-10-OOD/ensembles"
# pickle.dump(results_dict_cifar_ood, open(f"{save_path}_metrics.pickle", "wb"))


#SVHN

model_name = "ResNet_small"
output_dim = get_output_dim("SVHN")
n_samples = output_dim * num_samples_per_class
model_class = load_obj(MODELS_DICT[model_name])
model_hparams = get_model_hyperparams(output_dim, model_name)
model = model_class(**model_hparams)    

param_1_path = "./checkpoints/SVHN/ResNet_small_SVHN_0_params"
param_1_dict =  pickle.load(open(f"{param_1_path}.pickle", "rb"))
param_1 = param_1_dict['params']
batch_stats_1 = param_1_dict['batch_stats'] if 'batch_stats' in param_1_dict else None
rng_1 = param_1_dict['rng'] if 'rng' in param_1_dict else None
model_fn_1 = get_model_apply_fn(model_name, model.apply, batch_stats=batch_stats_1, rng=rng_1)


param_2_path = "./checkpoints/SVHN/ResNet_small_SVHN_1_params"
param_2_dict =  pickle.load(open(f"{param_2_path}.pickle", "rb"))
param_2 = param_2_dict['params']
batch_stats_2 = param_2_dict['batch_stats'] if 'batch_stats' in param_2_dict else None
rng_2 = param_2_dict['rng'] if 'rng' in param_2_dict else None
model_fn_2 = get_model_apply_fn(model_name, model.apply, batch_stats=batch_stats_2, rng=rng_2)

param_3_path = "./checkpoints/SVHN/ResNet_small_SVHN_2_params"
param_3_dict =  pickle.load(open(f"{param_3_path}.pickle", "rb"))
param_3 = param_3_dict['params']
batch_stats_3 = param_3_dict['batch_stats'] if 'batch_stats' in param_3_dict else None
rng_3 = param_3_dict['rng'] if 'rng' in param_3_dict else None
model_fn_3 = get_model_apply_fn(model_name, model.apply, batch_stats=batch_stats_3, rng=rng_3)

svhn_ood = get_ood_datasets("SVHN-OOD", ood_batch_size=ood_batch_size, n_samples=n_samples, corruption=args.corruption, seed=10)
ids = list(svhn_ood.keys())
eval_args = {}
eval_args["posterior_sample_type"] = "Pytree"
eval_args["likelihood"] = "classification"
eval_args["linearised_laplace"] = False

results_dict_svhn = defaultdict(dict)

for i, id in enumerate(ids):
    data_loader = svhn_ood[id]
    some_metrics, all_y_prob, all_y_true, all_y_var, all_y_sample_probs = evaluate_ensembles(data_loader, param_1, param_2, param_3, model_fn_1, model_fn_2, model_fn_3, eval_args)
    if i == 0:
        all_y_prob_in = all_y_prob
        all_y_sample_probs_in = all_y_sample_probs
        id_score = all_y_var.max(axis=-1)
    more_metrics = compute_metrics(
            i, id, all_y_prob, data_loader, all_y_prob_in, all_y_var, benchmark="SVHN-OOD"
        )
    if i != 0:
        all_y_sample_probs_out = all_y_sample_probs
        ood_score = all_y_var.max(axis=-1)
        more_metrics['score_auroc'] = auroc(id_score, ood_score)

    results_dict_svhn[id] = {**some_metrics, **more_metrics}
save_path = "./results/SVHN/ResNet_small/SVHN-OOD/ensembles"
pickle.dump(results_dict_svhn, open(f"{save_path}_metrics.pickle", "wb"))


breakpoint()