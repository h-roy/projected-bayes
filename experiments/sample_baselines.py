import os
import time
import argparse
import jax
import datetime
from flax import linen as nn
from jax import numpy as jnp
import pickle
from src.data.all_datasets import get_dataloaders
from src.losses import cross_entropy_loss, accuracy_preds, nll
from src.helper import compute_num_params, load_obj
from src.baselines import exact_diagonal_laplace, hutchinson_diagonal_laplace, last_layer_lapalce, swag_score_fun
import matplotlib.pyplot as plt
from src.data.utils import get_output_dim, numpy_collate_fn
from torch.utils import data

from src.models import MODELS_DICT
from src.training.configs import get_model_apply_fn, get_model_hyperparams

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["MNIST", "FMNIST", "CIFAR-10", "SVHN", "CIFAR-100", "ImageNet"])
parser.add_argument(
    "--model",
    type=str,
    choices=["LeNet", "MLP", "ResNet_small", "ResNet18", "DenseNet", "GoogleNet", "VisionTransformer"],
)

parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/CIFAR10/ResNet_small_CIFAR-10_0_params", help="path of model")
parser.add_argument("--method", type=str, choices=["Subnetwork", "Hutchinson_Diag", "Exact_Diag", "SWAG"], default="last_layer_laplace", help="Method to use for sampling")
parser.add_argument("--run_name", default=None, help="Fix the save file name.")
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--sample_seed",  type=int, default=0)
parser.add_argument("--posthoc_precision",  type=float, default=1.0)
parser.add_argument("--batch_size",  type=int, default=100)

parser.add_argument("--num_ll_params",  type=int, default=1000)
parser.add_argument("--hutchinson_samples",  type=int, default=100)
parser.add_argument("--hutchinson_levels",  type=int, default=3)
parser.add_argument("--gvp_batch_size",  type=int, default=50)
parser.add_argument("--likelihood",  type=str, choices=["classification", "regression"], default="classification")



if __name__ == "__main__":
    now = datetime.datetime.now()
    now_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    args = parser.parse_args()

    ###############
    ### dataset ###
    ###############
    dataset = args.dataset
    output_dim = get_output_dim(dataset)
    seed = args.sample_seed
    train_loader, val_loader, test_loader = get_dataloaders(
            dataset_name=dataset,
            train_batch_size=args.batch_size,
            val_batch_size=args.batch_size,
            data_path='/dtu/p1/hroy/data',
            seed=seed,
            purp='train'
        )
    
    #############
    ### model ###
    #############
    
    param_dict = pickle.load(open(f"{args.checkpoint_path}.pickle", "rb"))
    params = param_dict['params']

    batch_stats = param_dict['batch_stats'] if 'batch_stats' in param_dict else None
    rng = param_dict['rng'] if 'rng' in param_dict else None

    model_name = args.model
    model_class = load_obj(MODELS_DICT[model_name])
    model_hparams = get_model_hyperparams(output_dim, model_name)

    model = model_class(**model_hparams)    
    model_fn = get_model_apply_fn(model_name, model.apply, batch_stats=batch_stats, rng=rng)

    ############

    n_samples = args.num_samples
    n_params = compute_num_params(params)
    alpha = args.posthoc_precision
    batch_size = args.batch_size
    sample_key = jax.random.PRNGKey(args.sample_seed)
    start_time = time.perf_counter()
    num_ll_params = args.num_ll_params
    likelihood = args.likelihood
    method = args.method


    hutchinson_samples = args.hutchinson_samples
    num_levels = args.hutchinson_levels
    gvp_batch_size = args.gvp_batch_size
    if method == "Exact_Diag":
        posterior_samples, metrics = exact_diagonal_laplace(model_fn,
                                                            params,
                                                            n_samples,
                                                            alpha,
                                                            train_loader,
                                                            sample_key,
                                                            output_dim,
                                                            likelihood,)
    elif method == "Hutchinson_Diag":
        posterior_samples, metrics = hutchinson_diagonal_laplace(model_fn, 
                                                        params, 
                                                        n_samples,
                                                        alpha,
                                                        gvp_batch_size,
                                                        train_loader,
                                                        sample_key,
                                                        num_levels,
                                                        hutchinson_samples,
                                                        likelihood,
                                                        "parallel")
    elif method == "SWAG":
        posterior_samples = swag_score_fun(model, param_dict, sample_key, n_samples, train_loader, likelihood, max_num_models=3, diag_only=False)
        metrics = {"time": time.perf_counter() - start_time,}
    elif method == "Subnetwork":
        posterior_samples, metrics = last_layer_lapalce(
                                        model_fn,
                                        params,
                                        alpha,
                                        sample_key,
                                        n_samples,
                                        train_loader,
                                        num_ll_params,
                                        "classification",
                                        )
    print(f"{method} for a {n_params} parameter model {n_samples} samples took {time.time()-start_time:.5f} seconds")   
    posterior_dict = {
        "posterior_samples": posterior_samples,
    }
    os.makedirs(f"./checkpoints/baseline_samples/{args.method}/{args.dataset}/{args.model}", exist_ok=True)
    if args.run_name is not None:
        save_name = f"{args.run_name}_seed_{args.sample_seed}_prec_{args.posthoc_precision}"
    else:
        save_name = f"started_{now_string}"

    save_path = f"./checkpoints/baseline_samples/{args.method}/{args.dataset}/{args.model}/{save_name}"
    pickle.dump(posterior_dict, open(f"{save_path}_params.pickle", "wb"))
    pickle.dump(metrics, open(f"{save_path}_metrics.pickle", "wb"))
