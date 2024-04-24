from typing import Callable, Optional, Literal
import jax
import tree_math as tm
from jax import numpy as jnp
from src.helper import tree_random_normal_like
from src.sampling import precompute_inv, precompute_inv_batch, kernel_proj_vp, kernel_proj_vp_batch
from jax.tree_util import Partial as jaxPartial
from tqdm import tqdm
from torch.utils import data
from src.sampling.sample_utils import kernel_check
from src.helper import set_seed

from jax import config
config.update("jax_debug_nans", True)


def sample_projections( 
    model_fn: Callable,
    params,
    eps,
    x_train_batched,
    output_dim: int,
    n_iterations: int,
    x_val: jnp.ndarray,
):
    # eps = tree_random_normal_like(key, params, n_posterior_samples)
    # prior_samples = jax.tree_map(lambda x: 1/jnp.sqrt(alpha) * x, eps)
    prior_samples = eps
    batched_eigvecs, batched_inv_eigvals = precompute_inv(model_fn, params, x_train_batched, output_dim, "scan")
    proj_vp_fn = lambda v : kernel_proj_vp(vec=v, model_fn=model_fn, params=params, x_train_batched=x_train_batched, 
                                           batched_eigvecs=batched_eigvecs, batched_inv_eigvals=batched_inv_eigvals, 
                                           output_dim=output_dim, n_iterations=n_iterations, x_val=x_val)
    projected_samples = jax.vmap(proj_vp_fn)(prior_samples)
    posterior_samples = jax.tree_map(lambda x, y: x + y, projected_samples, params)
    return posterior_samples

def sample_projections_dataloader( 
    model_fn: Callable,
    params,
    eps,
    train_loader,
    seed,
    output_dim: int,
    n_iterations: int,
    x_val: jnp.ndarray,
):
    set_seed(seed)
    # key = jax.random.PRNGKey(seed)
    # eps = tree_random_normal_like(key, params, n_posterior_samples)
    # projected_samples = jax.tree_map(lambda x: 1/jnp.sqrt(alpha) * x, eps)
    # train_loader = data.DataLoader(**data_loader_hparams)
    projected_samples = eps
    batched_eigvecs = []
    batched_inv_eigvals = []
    for i, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        x_batch = batch['image']
        eigvecs, inv_eigvals = precompute_inv_batch(model_fn, params, x_batch, output_dim)
        batched_eigvecs.append(eigvecs)
        batched_inv_eigvals.append(inv_eigvals)
    batched_eigvecs = jnp.stack(batched_eigvecs)
    batched_inv_eigvals = jnp.stack(batched_inv_eigvals)
    for iter_num in tqdm(range(n_iterations)):
        # train_loader = data.DataLoader(**data_loader_hparams)
        for j, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            x_batch = batch['image']
            # eigvecs, inv_eigvals = precompute_inv_batch(model_fn, params, x_batch, output_dim)
            eigvecs, inv_eigvals = batched_eigvecs[j], batched_inv_eigvals[j]
            proj_vp_fn = lambda v : kernel_proj_vp_batch(vec=v, model_fn=model_fn, params=params, x_batch=x_batch, eigvecs=eigvecs, inv_eigvals=inv_eigvals, output_dim=output_dim)
            projected_samples = jax.vmap(proj_vp_fn)(projected_samples)
        proj_norm = tm.Vector(projected_samples) @ tm.Vector(projected_samples)
        kernel_norm = jnp.mean(jnp.array(kernel_check(jax.tree_map(lambda x, y: x + y, projected_samples, params), model_fn, params, x_val)))
        jax.debug.print("Iteration: {iter} Proj Norm: {proj_norm} Kernel Check: {kernel_norm}", iter=iter_num, proj_norm=proj_norm, kernel_norm=kernel_norm)

    posterior_samples = jax.tree_map(lambda x, y: x + y, projected_samples, params)
    return posterior_samples
