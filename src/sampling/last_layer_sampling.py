import time
from typing import Callable, Literal
import jax
from jax import numpy as jnp
from src.laplace.last_layer.extract_last_layer import last_layer_ggn
from tqdm import tqdm
from src.helper import set_seed

from jax import config
config.update("jax_debug_nans", True)

def sample_last_layer( 
    model_fn: Callable,
    params,
    n_samples,
    alpha: float,
    train_loader,
    seed,
    likelihood: Literal["classification", "regression"] = "classification",
):
    # model_fn takes in pytrees and aprams are also pytrees
    set_seed(seed)
    key = jax.random.PRNGKey(seed)
    ggn_ll = 0
    start_time = time.perf_counter()
    for batch in train_loader:
        img, label = batch['image'], batch['label']
        img = jnp.asarray(img)
        ggn_ll += last_layer_ggn(model_fn, params, img, likelihood)
    prec = ggn_ll + alpha * jnp.eye(ggn_ll.shape[0])
    leafs, _ = jax.tree_util.tree_flatten(params)
    N_llla = len(leafs[-1]) + len(leafs[-2])
    params_vec, _ = jax.flatten_util.ravel_pytree(params)
    map_ll = params_vec[-N_llla:]
    def sample(key):
        eps = jax.random.normal(key, shape=(ggn_ll.shape[0],))
        sample = jnp.linalg.cholesky(jnp.linalg.inv(prec)) @ eps
        return sample + map_ll
    key_list = jax.random.split(key, n_samples)
    posterior_samples = jax.vmap(sample)(key_list)
    metrics = {"time": time.perf_counter() - start_time,}
    return posterior_samples, metrics


