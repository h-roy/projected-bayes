from functools import partial
import time
from typing import Callable, Literal, Optional

import jax
import tree_math as tm
from jax import numpy as jnp
from src.helper import tree_random_normal_like, get_gvp_fun
import jax
import jax.numpy as jnp
from jax import flatten_util
import optax
from tqdm import tqdm
import time
from src.laplace import exact_diagonal, hutchinson_diagonal, last_layer_ggn

def exact_diagonal_laplace( 
    model_fn: Callable,
    params,
    n_samples,
    alpha: float,
    train_loader,
    key,
    output_dims: int,
    likelihood: Literal["classification", "regression"] = "classification",
):
    # model_fn takes in pytrees and aprams are also pytrees
    variables, unflatten = jax.flatten_util.ravel_pytree(params)
    key_list = jax.random.split(key, n_samples)
    diag = 0
    start_time = time.perf_counter()
    for i, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        x_batch = jnp.asarray(batch['image'])
        diag += jax.flatten_util.ravel_pytree(exact_diagonal(model_fn, params, output_dims, x_batch, likelihood))[0]
    def sample(key):
        eps = jax.random.normal(key, shape=(len(variables),))
        sample = 1/jnp.sqrt(diag + alpha) * eps
        return sample + variables
    posterior_samples = jax.vmap(lambda k: unflatten(sample(k)))(key_list)
    metrics = {"time": time.perf_counter() - start_time,}
    return posterior_samples, metrics


def hutchinson_diagonal_laplace( 
    model_fn: Callable,
    params,
    n_samples,
    alpha: float,
    gvp_batch_size: int,
    train_loader,
    key,
    num_levels: int = 5,
    hutch_samples: int = 200,
    likelihood: Literal["classification", "regression"] = "classification",
    computation_type: Literal["serial", "parallel"] = "parallel",
):
    # model_fn takes in pytrees and aprams are also pytrees
    variables, unflatten = jax.flatten_util.ravel_pytree(params)
    
    diag = 0
    start_time = time.perf_counter()
    for i, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        subkey, key = jax.random.split(key)
        img = jnp.asarray(batch['image'])
        data_array = jnp.asarray(img).reshape((-1, gvp_batch_size) +  img.shape[1:])
        diag += jax.flatten_util.ravel_pytree(hutchinson_diagonal(model_fn, params, gvp_batch_size, hutch_samples, subkey, data_array, likelihood, num_levels=num_levels,computation_type=computation_type))[0]
    def sample(key):
        eps = jax.random.normal(key, shape=(len(variables),))
        sample = 1/jnp.sqrt(diag + alpha) * eps
        return sample + variables
    key_list = jax.random.split(key, n_samples)
    posterior_samples = jax.vmap(sample)(key_list)
    posterior_samples = jax.vmap(unflatten)(posterior_samples)
    metrics = {"time": time.perf_counter() - start_time,}
    return posterior_samples, metrics

def last_layer_lapalce(
        model_fn,
        params,
        alpha,
        key,
        num_samples,
        train_loader,
        num_ll_params = None,
        likelihood: Literal["classification", "regression"] = "classification",
):
    # ggn_ll = last_layer_ggn(model_fn, params, x_batch, likelihood)
    ggn_ll = 0
    start_time = time.perf_counter()
    for batch in tqdm(train_loader):
        img, label = batch['image'], batch['label']
        img = jnp.asarray(img)
        ggn_ll += last_layer_ggn(model_fn, params, img, likelihood, num_ll_params)
    # model_fn takes in pytrees and aprams are also pytrees
    prec = ggn_ll + alpha * jnp.eye(ggn_ll.shape[0])
    leafs, _ = jax.tree_util.tree_flatten(params)
    if num_ll_params is None:
        N_llla = len(leafs[-1].flatten()) #+ len(leafs[-2])
    else:
        N_llla = num_ll_params
    params_vec, unflatten = jax.flatten_util.ravel_pytree(params)
    map_ll = params_vec.at[-N_llla:].get()
    map_fl = params_vec.at[:-N_llla].get()
    def sample(key):
        eps = jax.random.normal(key, shape=(ggn_ll.shape[0],))
        sample = jnp.linalg.cholesky(jnp.linalg.inv(prec)) @ eps
        return sample + map_ll
    key_list = jax.random.split(key, num_samples)
    posterior_ll_samples = jax.vmap(sample)(key_list)
    posterior_samples = jax.vmap(lambda x: unflatten(jnp.concatenate([map_fl, x])))(posterior_ll_samples)
    metrics = {"time": time.perf_counter() - start_time,}
    return posterior_samples, metrics

