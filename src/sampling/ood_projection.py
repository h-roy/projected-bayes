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
from jax.experimental import mesh_utils


from jax import config
config.update("jax_debug_nans", True)

def ood_projections( 
    model_fn: Callable,
    params,
    posterior_samples,
    test_loader,
    ood_loader,
    sample_batch_size: int,
    seed,
    output_dims: int,
    unflatten_fn: Callable,
):
    set_seed(seed)
    params_vec, _ = jax.flatten_util.ravel_pytree(params)
    eps = jax.vmap(lambda sample: jax.tree_map(lambda x,y: x - y, sample, params))(posterior_samples)
    eps, _ = jax.flatten_util.ravel_pytree(eps)
    n_samples = len(eps) // len(params_vec)
    eps = jnp.asarray(eps).reshape((n_samples, -1))
    projected_samples = eps
    x_val_ood = jnp.asarray(next(iter(ood_loader))['image'])
    for i, batch in enumerate(tqdm(ood_loader, desc="Training", leave=False)):
        x_data = jnp.asarray(batch['image'])
        N = x_data.shape[0]
        n_batches = N // sample_batch_size
        x_train_batched = x_data[:n_batches * sample_batch_size].reshape((n_batches, -1) + x_data.shape[1:])
        batched_eigvecs, batched_inv_eigvals = precompute_inv(model_fn, params_vec, x_train_batched, output_dims, "scan")
        proj_vp_fn = lambda v : (v - kernel_proj_vp(vec=v, model_fn=model_fn, params=params_vec, x_train_batched=x_train_batched, 
                                        batched_eigvecs=batched_eigvecs, batched_inv_eigvals=batched_inv_eigvals, 
                                        output_dims=output_dims, n_iterations=5, x_val=x_val_ood, acceleration=True))
        projected_samples = jax.vmap(proj_vp_fn)(projected_samples)
        del x_train_batched, x_data, batched_eigvecs, batched_inv_eigvals, proj_vp_fn
    projected_samples = jax.tree_map(lambda x: 200 * x, projected_samples)

    x_val = jnp.asarray(next(iter(test_loader))['image'])
    for i, batch in enumerate(tqdm(test_loader, desc="Training", leave=False)):
        x_data = jnp.asarray(batch['image'])
        N = x_data.shape[0]
        n_batches = N // sample_batch_size
        x_train_batched = x_data[:n_batches * sample_batch_size].reshape((n_batches, -1) + x_data.shape[1:])
        batched_eigvecs, batched_inv_eigvals = precompute_inv(model_fn, params_vec, x_train_batched, output_dims, "scan")
        proj_vp_fn = lambda v : kernel_proj_vp(vec=v, model_fn=model_fn, params=params_vec, x_train_batched=x_train_batched, 
                                           batched_eigvecs=batched_eigvecs, batched_inv_eigvals=batched_inv_eigvals, 
                                           output_dims=output_dims, n_iterations=5, x_val=x_val, acceleration=True)
        projected_samples = jax.vmap(proj_vp_fn)(projected_samples)
        del x_train_batched, x_data, batched_eigvecs, batched_inv_eigvals, proj_vp_fn

    posterior_samples = jax.vmap(lambda single_sample: unflatten_fn(params_vec + single_sample))(projected_samples)
    return posterior_samples
