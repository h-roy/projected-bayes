from functools import partial
import time
from typing import Callable, Optional, Literal
import jax
import tree_math as tm
from jax import numpy as jnp
from src.sampling import precompute_inv, precompute_inv_batch, kernel_proj_vp, kernel_proj_vp_batch
from jax.tree_util import Partial as jaxPartial
from tqdm import tqdm
from src.laplace.diagonal import hutchinson_diagonal, exact_diagonal
from src.sampling.sample_utils import kernel_check
from src.helper import set_seed

from jax import config
config.update("jax_debug_nans", True)

def sample_exact_diagonal( 
    model_fn: Callable,
    params,
    n_samples,
    alpha: float,
    train_loader,
    seed,
    output_dims: int,
    likelihood: Literal["classification", "regression"] = "classification",
):
    # model_fn takes in pytrees and aprams are also pytrees
    set_seed(seed)
    key = jax.random.PRNGKey(seed)
    variables, _ = jax.flatten_util.ravel_pytree(params)
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
    posterior_samples = jax.vmap(sample)(key_list)
    metrics = {"time": time.perf_counter() - start_time,}
    return posterior_samples, metrics


def sample_hutchinson( 
    model_fn: Callable,
    params,
    n_samples,
    alpha: float,
    gvp_batch_size: int,
    train_loader,
    seed,
    num_levels: int = 5,
    hutch_samples: int = 200,
    likelihood: Literal["classification", "regression"] = "classification",
    computation_type: Literal["serial", "parallel"] = "parallel",
):
    # model_fn takes in pytrees and aprams are also pytrees
    set_seed(seed)
    key = jax.random.PRNGKey(seed)
    variables, _ = jax.flatten_util.ravel_pytree(params)
    
    diag = 0
    start_time = time.perf_counter()
    for i, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        subkey, key = jax.random.split(key)
        img = jnp.asarray(batch['image'])
        data_array = jnp.asarray(img).reshape((-1, gvp_batch_size) +  img.shape[1:])
        diag += jax.flatten_util.ravel_pytree(hutchinson_diagonal(model_fn, params, gvp_batch_size, hutch_samples, subkey, data_array, likelihood, num_levels=num_levels,computation_type=computation_type))[0]
    def sample(key):
        eps = jax.random.normal(key, shape=(len(variables),))
        sample = 1/(diag + alpha) * eps
        return sample + variables
    key_list = jax.random.split(key, n_samples)
    posterior_samples = jax.vmap(sample)(key_list)
    metrics = {"time": time.perf_counter() - start_time,}
    return posterior_samples, metrics
