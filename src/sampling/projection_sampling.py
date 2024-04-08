from typing import Callable, Optional, Literal
import jax
import tree_math as tm
from jax import numpy as jnp
from src.helper import tree_random_normal_like
from src.sampling import precompute_inv, precompute_inv_batch, kernel_proj_vp, kernel_proj_vp_batch
from jax.tree_util import Partial as jaxPartial
from tqdm import tqdm

from jax import config
config.update("jax_debug_nans", True)


def sample_projections( 
    model_fn: Callable,
    params,
    x_train_batched,
    key,
    alpha: float,
    output_dim: int,
    n_posterior_samples: int,
    n_iterations: int,
):
    eps = tree_random_normal_like(key, params, n_posterior_samples)
    prior_samples = jax.tree_map(lambda x: 1/jnp.sqrt(alpha) * x, eps)
    batched_eigvecs, batched_inv_eigvals = precompute_inv(model_fn, params, x_train_batched, output_dim, "scan")
    proj_vp_fn = lambda v : kernel_proj_vp(vec=v, model_fn=model_fn, params=params, x_train_batched=x_train_batched, batched_eigvecs=batched_eigvecs, batched_inv_eigvals=batched_inv_eigvals, output_dim=output_dim, n_iterations=n_iterations)
    projected_samples = jax.vmap(proj_vp_fn)(prior_samples)
    posterior_samples = jax.tree_map(lambda x, y: x + y, projected_samples, params)
    return posterior_samples

def sample_projections_dataloader( 
    model_fn: Callable,
    params,
    train_loader,
    key,
    alpha: float,
    output_dim: int,
    n_posterior_samples: int,
    n_iterations: int,
):
    eps = tree_random_normal_like(key, params, n_posterior_samples)
    projected_samples = jax.tree_map(lambda x: 1/jnp.sqrt(alpha) * x, eps)
    # batched_eigvecs = []
    # batched_inv_eigvals = []
    # for i, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
    #     x_batch = batch['image']
    #     if i == 0:
    #         jax.debug.print("Norm of first image: {x}",x=jnp.linalg.norm(x_batch))
    #     eigvecs, inv_eigvals = precompute_inv_batch(model_fn, params, x_batch, output_dim)
    #     batched_eigvecs.append(eigvecs)
    #     batched_inv_eigvals.append(inv_eigvals)
    # batched_eigvecs = jnp.stack(batched_eigvecs)
    # batched_inv_eigvals = jnp.stack(batched_inv_eigvals)
    for _ in tqdm(range(n_iterations)):
        for j, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            x_batch = batch['image']
            if j == 0:
                jax.debug.print("Norm of first image: {x}",x=jnp.linalg.norm(x_batch))
            eigvecs, inv_eigvals = precompute_inv_batch(model_fn, params, x_batch, output_dim)
            # eigvecs, inv_eigvals = batched_eigvecs[j], batched_inv_eigvals[j]
            proj_vp_fn = lambda v : kernel_proj_vp_batch(vec=v, model_fn=model_fn, params=params, x_batch=x_batch, eigvecs=eigvecs, inv_eigvals=inv_eigvals, output_dim=output_dim)
            projected_samples = jax.vmap(proj_vp_fn)(projected_samples)
            # jax.debug.print("Projected samples norm: {x}", x = tm.Vector(projected_samples)@tm.Vector(projected_samples))
    posterior_samples = jax.tree_map(lambda x, y: x + y, projected_samples, params)
    return posterior_samples
