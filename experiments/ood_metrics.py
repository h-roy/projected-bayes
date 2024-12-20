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
from src.ood_functions.evaluate import evaluate, evaluate_map
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
parser.add_argument("--dataset", type=str, choices=["MNIST", "FMNIST", "CIFAR-10", "SVHN", "CIFAR-100", "ImageNet"])
parser.add_argument(
    "--model",
    type=str,
    choices=["LeNet", "MLP", "ResNet_small", "ResNet18", "DenseNet", "GoogleNet", "VisionTransformer"],
)
parser.add_argument("--corruption", type=str, choices=["fog", "zoom_blur", "gaussian_blur", "snow", "brightness", "contrast"], default="fog")
parser.add_argument("--posterior_type", type=str, choices=["MAP", "Projection", "Loss-Kernel", "Exact_Diag", "Hutchinson_Diag", "SWAG", "Subnetwork"])
parser.add_argument("--experiment", type=str, choices=["MNIST-OOD", "R-MNIST", "FMNIST-OOD", "R-FMNIST", "CIFAR-10-OOD", "CIFAR-10-C", "SVHN-OOD", "CIFAR-100-OOD", "ImageNet-OOD"])
parser.add_argument("--parameter_path", type=str, default="./checkpoints/MNIST/LeNet_MNIST_0_params")
parser.add_argument("--posterior_path", type=str, default="./checkpoints/posterior_samples/MNIST/LeNet/mnist_samples_seed_0_params")
parser.add_argument("--ood_batch_size", type=int, default=128)
parser.add_argument("--num_samples_per_class", type=int, default=500)
parser.add_argument("--val", action="store_true", required=False, default=False)
parser.add_argument("--linearised_laplace", type=str, required=False, default=None)
parser.add_argument("--prior_precision", type=float, default=None)
parser.add_argument("--run_name", type=str, default="1")
parser.add_argument("--refinement", action="store_true", required=False, default=False)

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
ood_datasets_dict = get_ood_datasets(experiment, ood_batch_size=ood_batch_size, n_samples=n_samples, corruption=args.corruption, seed=10)
ids = list(ood_datasets_dict.keys())
if experiment[-3:] == 'OOD':
    idx = ids.index(dataset)
    ids[0], ids[idx] = ids[idx], ids[0]

evaluate_fn = evaluate_map if posterior_type == "MAP" else evaluate
eval_args = {}
eval_args["posterior_sample_type"] = "Pytree"
eval_args["likelihood"] = "classification"
if args.linearised_laplace is None:
    if posterior_type in ["Projection", "Loss-Kernel"]:
        eval_args["linearised_laplace"] = True
    elif posterior_type in ["MAP", "Exact_Diag", "Hutchinson_Diag", "SWAG", "Subnetwork"]:
        eval_args["linearised_laplace"] = False
else:
    eval_args["linearised_laplace"] = False
results_dict = defaultdict(dict)
results_dict[posterior_type] = defaultdict(dict)

id_batch = next(iter(ood_datasets_dict[ids[0]]))
x_id, y_id = id_batch['image'], id_batch['label'][:10]

ood_batch = next(iter(ood_datasets_dict[ids[1]]))
x_ood, y_ood = ood_batch['image'], ood_batch['label'][:10]
params_vec, unflatten = jax.flatten_util.ravel_pytree(params)

# cenetered_posterior = jax.vmap(lambda sample: jax.tree_map(lambda x,y: x - y, sample, params))(posterior_dict)
# lmbd_id = lambda p: model_fn(p, x_id)
# jvp_id_fn = lambda p: jax.jvp(lmbd_id, (params,), (p,))[1]
# lmbd_ood = lambda p: model_fn(p, x_ood)
# jvp_ood_fn = lambda p: jax.jvp(lmbd_ood, (params,), (p,))[1]

# jvp_ids = jax.vmap(jvp_id_fn)(cenetered_posterior)
# jvp_oods = jax.vmap(jvp_ood_fn)(cenetered_posterior)

# print("ID jacobian norm:", jnp.linalg.norm(jvp_ids))
# print("OOD jacobian norm:", jnp.linalg.norm(jvp_oods))
# print(jax.vmap(jnp.linalg.norm)(x_id - x_ood))
# min_norm = 9999
# max_norm = 0.
# cross_dist = []
# for i,x in enumerate(x_id):
#     for j,y in enumerate(x_id):
#         if i != j and (jnp.linalg.norm(x - y) < min_norm).all():
#             min_norm = jnp.linalg.norm(x - y)
#         if i != j and (jnp.linalg.norm(x - y) > max_norm).all():
#             max_norm = jnp.linalg.norm(x - y)
# breakpoint()
# Cross distances are closer than the OOD distances. But max id distances > min ood distances so image projection will project into some other image

# print("Jacobian Norms ID:",jax.vmap(lambda p: jnp.linalg.norm(jvp_id_fn(p)))(cenetered_posterior))
# print("Jacobian Norms OOD:", jax.vmap( lambda p: jnp.linalg.norm(jvp_ood_fn(p)))(cenetered_posterior))
# # breakpoint()
if args.refinement:
    ood_n_samples = 5000
    ood_batch_size = 128 # Works best
    # ood_batch_size = 1000 # Doesn't work yet(batch_size = data size is correct thing to do)
    refinement_dict = get_ood_datasets(experiment, ood_batch_size=ood_batch_size, n_samples=ood_n_samples, corruption=args.corruption, val=True)
    test_loader = refinement_dict[ids[0]]
    ood_loader = refinement_dict[ids[1]]
    params_vec, unflatten, model_fn_vec = vectorize_nn(model_fn, params)
    refinement_batch_size = 32
    seed = 0
    posterior_dict = ood_projections(model_fn_vec, params, posterior_dict, test_loader, ood_loader,
                                    refinement_batch_size, seed, output_dim, unflatten)
    # ood_loader = ood_datasets_dict[ids[2]]
    # posterior_dict_2 = ood_projections(model_fn_vec, params, posterior_dict, test_loader, ood_loader,
    #                             refinement_batch_size, seed, output_dim, unflatten)

# cenetered_posterior = jax.vmap(lambda sample: jax.tree_map(lambda x,y: x - y, sample, params))(posterior_dict)
# lmbd_id = lambda p: model_fn(p, x_id)
# jvp_id_fn = lambda p: jax.jvp(lmbd_id, (params,), (p,))[1]
# lmbd_ood = lambda p: model_fn(p, x_ood)
# jvp_ood_fn = lambda p: jax.jvp(lmbd_ood, (params,), (p,))[1]

# jvp_ids = jax.vmap(jvp_id_fn)(cenetered_posterior)
# jvp_oods = jax.vmap(jvp_ood_fn)(cenetered_posterior)

# print("ID jacobian norm:", jnp.linalg.norm(jvp_ids))
# print("OOD jacobian norm:", jnp.linalg.norm(jvp_oods))

# print("Jacobian Norms ID:",jax.vmap(lambda p: jnp.linalg.norm(jvp_id_fn(p)))(cenetered_posterior))
# print("Jacobian Norms OOD:", jax.vmap( lambda p: jnp.linalg.norm(jvp_ood_fn(p)))(cenetered_posterior))
# breakpoint()
for i, id in enumerate(ids):
    data_loader = ood_datasets_dict[id]
    if posterior_type == "MAP":
        some_metrics, all_y_prob, all_y_true, all_y_var, all_y_sample_probs = evaluate_map(data_loader, params, model_fn, eval_args)
    else:
        some_metrics, all_y_prob, all_y_true, all_y_var, all_y_sample_probs = evaluate(data_loader, posterior_dict, params, model_fn, eval_args)
    if i == 0:
        all_y_prob_in = all_y_prob
        all_y_sample_probs_in = all_y_sample_probs
        id_score_ = 1 - all_y_sample_probs_in.max(axis=(0,-1)) if posterior_type != "MAP" else None
        # id_score = 1 - all_y_prob_in.max(axis=-1) + all_y_var.max(axis=-1) if posterior_type != "MAP" else None # Can also be mean
        id_score = all_y_var.max(axis=-1) if posterior_type != "MAP" else None # Can also be mean
    more_metrics = compute_metrics(
            i, id, all_y_prob, data_loader, all_y_prob_in, all_y_var, benchmark=experiment
        )
    if i != 0:
        all_y_sample_probs_out = all_y_sample_probs
        ood_score_ = 1 - all_y_sample_probs_out.max(axis=(0,-1))  if posterior_type != "MAP" else None
        # ood_score = 1 - all_y_prob.max(axis=-1) + all_y_var.max(axis=-1) if posterior_type != "MAP" else None
        ood_score = all_y_var.max(axis=-1) if posterior_type != "MAP" else None
        more_metrics['score_auroc'] = auroc(id_score, ood_score) if posterior_type != "MAP" else None
        if posterior_type == 'Projection':
            more_metrics['auroc'] = auroc(id_score_, ood_score_)

    results_dict[posterior_type][id] = {**some_metrics, **more_metrics}
# python experiments/ood_metrics.py --dataset MNIST --model LeNet --posterior_type Projection --experiment MNIST-OOD --parameter_path ./checkpoints/MNIST/LeNet_MNIST_0_params --posterior_path ./checkpoints/posterior_samples/MNIST/LeNet/mnist_samples_seed_0_params --ood_batch_size 128 --num_samples_per_class 500
# python experiments/ood_metrics.py --dataset FMNIST --model LeNet --posterior_type Projection --experiment FMNIST-OOD --parameter_path ./checkpoints/FMNIST/LeNet_FMNIST_0_params --posterior_path ./checkpoints/posterior_samples/FMNIST/LeNet/fmnist_samples_seed_0_params --ood_batch_size 128 --num_samples_per_class 500
# python experiments/ood_metrics.py --dataset CIFAR-10 --model ResNet_small --posterior_type Projection --experiment CIFAR-10-OOD --parameter_path ./checkpoints/CIFAR10/ResNet_small_CIFAR-10_0_params --posterior_path ./checkpoints/posterior_samples/CIFAR-10/ResNet_small/cifar_samples_1_seed_0_params --ood_batch_size 128 --num_samples_per_class 500
# python experiments/ood_metrics.py --dataset MNIST --model LeNet --posterior_type Projection --experiment MNIST-OOD --parameter_path ./checkpoints/MNIST/LeNet_MNIST_0_params --posterior_path ./checkpoints/loss_kernel_samples/MNIST/LeNet/mnist_samples_seed_0_params --ood_batch_size 128 --num_samples_per_class 500
# python experiments/ood_metrics.py --dataset FMNIST --model LeNet --posterior_type Loss-Kernel --experiment FMNIST-OOD --parameter_path ./checkpoints/FMNIST/LeNet_FMNIST_0_params --posterior_path ./checkpoints/loss_kernel_samples/FMNIST/LeNet/fmnist_samples_seed_0_params --ood_batch_size 128 --num_samples_per_class 500
# python experiments/ood_metrics.py --dataset FMNIST --model LeNet --posterior_type Loss-Kernel --experiment R-FMNIST --parameter_path ./checkpoints/FMNIST/LeNet_FMNIST_0_params --posterior_path ./checkpoints/loss_kernel_samples/FMNIST/LeNet/fmnist_samples_seed_0_params --ood_batch_size 128 --num_samples_per_class 500
# python experiments/ood_metrics.py --dataset CIFAR-10 --model ResNet_small --posterior_type Projection --experiment CIFAR-10-OOD --parameter_path ./checkpoints/CIFAR10/ResNet_small_CIFAR-10_0_params --posterior_path ./checkpoints/posterior_samples/CIFAR-10/ResNet_small/cifar_samples_0_sample_seed_0_params --ood_batch_size 128 --num_samples_per_class 500

os.makedirs(f"./results/{args.dataset}/{args.model}/{args.experiment}", exist_ok=True)
if args.experiment == "CIFAR-10-C":
    save_path = f"./results/{args.dataset}/{args.model}/{args.experiment}/{posterior_type}_prec_{prior_precision}_{args.run_name}_{args.corruption}" if prior_precision is not None else f"./results/{args.dataset}/{args.model}/{args.experiment}/{posterior_type}_{args.run_name}_{args.corruption}"
else:
    save_path = f"./results/{args.dataset}/{args.model}/{args.experiment}/{posterior_type}_prec_{prior_precision}_{args.run_name}" if prior_precision is not None else f"./results/{args.dataset}/{args.model}/{args.experiment}/{posterior_type}_{args.run_name}"
pickle.dump(results_dict, open(f"{save_path}_metrics.pickle", "wb"))
