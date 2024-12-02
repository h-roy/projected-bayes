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
from src.losses import cross_entropy_loss_per_datapoint, sse_loss
from src.helper import calculate_exact_ggn, tree_random_normal_like, compute_num_params
from jax_models import load_model
from src.sampling.ood_projection import ood_loss_projections
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
from src.training.configs import get_imagenet_model_fn, get_model_apply_fn, get_model_hyperparams
from sklearn.metrics import roc_auc_score
import numpy as np
from src.sampling import linearize_model_fn
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    choices=["swin-small-224", "swin-tiny-224", 
             "van-base", "van-tiny", "van-small", "van-large",
             "pvit-b0", "pvit-b1", "pvit-b2", "pvit-b2-linear", "pvit-b3", "pvit-b4", "pvit-b5",
             "convnext-small", "convnext-tiny", "convnext-large-224-1k", "convnext-base-224-1k",
             "cait-s24-224", "cait-xxs36-224"],
    default="swin-small-224",
)
parser.add_argument("--posterior_type", type=str, choices=["MAP", "Loss-Kernel"])
parser.add_argument("--posterior_path", type=str, default="./checkpoints/loss_kernel_samples/ImageNet/swin-tiny-224/swin_tiny_imagenet_checkpoint_sample_seed_0_params")
parser.add_argument("--ood_batch_size", type=int, default=128)
parser.add_argument("--num_samples_per_class", type=int, default=10)
parser.add_argument("--val", action="store_true", required=False, default=False)
parser.add_argument("--linearised_laplace", type=str, required=False, default=None)
parser.add_argument("--prior_precision", type=float, default=0.005)
parser.add_argument("--run_name", type=str, default="0")
parser.add_argument("--refinement", action="store_true", required=False, default=False)

args = parser.parse_args()

dataset = "ImageNet"
model_name = args.model 
posterior_type = args.posterior_type 
experiment = "ImageNet-OOD"

posterior_path = args.posterior_path 

def auroc(scores_id, scores_ood):
    labels = np.zeros(len(scores_id) + len(scores_ood), dtype="int32")
    labels[len(scores_id) :] = 1
    scores = np.concatenate([scores_id, scores_ood])
    return roc_auc_score(labels, scores)

model_rng = random.PRNGKey(0)
# model_rng = pickle.load(open(f"{posterior_path}.pickle", "rb"))['rng']

if args.model[:3] == "van":
    model, params, batch_stats = load_model(args.model, attach_head=True, num_classes=1000, dropout=0.0, pretrained=True)
else:
    model, params = load_model(args.model, attach_head=True, num_classes=1000, dropout=0.0, pretrained=True)
    batch_stats = None
model_fn = get_imagenet_model_fn(args.model, model, model_rng, batch_stats=batch_stats)
posterior_dict = pickle.load(open(f"{posterior_path}.pickle", "rb"))['posterior_samples']
prior_precision = args.prior_precision
if prior_precision is not None:
    posterior_dict = jax.vmap(lambda sample: jax.tree_map(lambda x, y: (x - y) * prior_precision + y, sample, params))(posterior_dict)
ood_batch_size = args.ood_batch_size 
num_samples_per_class = args.num_samples_per_class 
val = args.val 


# Initialize Model
output_dim = 1000
n_samples = output_dim * num_samples_per_class

ood_datasets_dict = get_ood_datasets(experiment, ood_batch_size=ood_batch_size, n_samples=n_samples, seed=50)
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

# eval_args["linearised_laplace"] = False

results_dict = defaultdict(dict)
results_dict[posterior_type] = defaultdict(dict)

id_batch = next(iter(ood_datasets_dict[ids[0]]))
x_id, y_id = id_batch['image'], id_batch['label'][:10]

ood_batch = next(iter(ood_datasets_dict[ids[1]]))
x_ood, y_ood = ood_batch['image'], ood_batch['label'][:10]
params_vec, unflatten = jax.flatten_util.ravel_pytree(params)

if args.refinement:
    ood_n_samples = 1000
    ood_batch_size = 128 # Works best
    # ood_batch_size = 1000 # Doesn't work yet(batch_size = data size is correct thing to do)
    refinement_dict = get_ood_datasets(experiment, ood_batch_size=ood_batch_size, n_samples=ood_n_samples, val=True)
    test_loader = refinement_dict[ids[0]]
    ood_loader = refinement_dict[ids[1]]
    params_vec, unflatten, model_fn_vec = vectorize_nn(model_fn, params)
    refinement_batch_size = 8
    seed = 5
    loss_fn = cross_entropy_loss_per_datapoint
    posterior_dict = ood_loss_projections(model_fn_vec, loss_fn, params, posterior_dict, test_loader, ood_loader,
                                    refinement_batch_size, seed, unflatten)
    # ood_loader = ood_datasets_dict[ids[2]]
    # posterior_dict_2 = ood_projections(model_fn_vec, params, posterior_dict, test_loader, ood_loader,
    #                             refinement_batch_size, seed, output_dim, unflatten)

model_fn = jax.jit(model_fn)
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
os.makedirs(f"./results/{dataset}/{args.model}/{experiment}", exist_ok=True)
save_path = f"./results/{dataset}/{args.model}/{experiment}/{posterior_type}_prec_{prior_precision}_{args.run_name}" if prior_precision is not None else f"./results/{args.dataset}/{args.model}/{args.experiment}/{posterior_type}_{args.run_name}"
pickle.dump(results_dict, open(f"{save_path}_metrics.pickle", "wb"))
