from typing import Callable, Optional, Literal
import jax
import tree_math as tm
from jax import numpy as jnp
from src.helper import tree_random_normal_like
from src.sampling import loss_kernel_proj_vp, precompute_loss_inv
from jax.tree_util import Partial as jaxPartial
from tqdm import tqdm
from torch.utils import data
from src.sampling.sample_utils import kernel_check
from src.helper import set_seed
from jax.experimental import mesh_utils


from jax import config
config.update("jax_debug_nans", True)

def sample_loss_projections_dataloader( 
    model_fn: Callable,
    loss_fn: Callable,
    params_vec,
    eps,
    alpha: float,
    train_loader,
    sample_batch_size: int,
    seed,
    n_iterations: int,
    x_val: jnp.ndarray,
    y_val: jnp.ndarray,
    n_params,
    unflatten_fn: Callable,
    use_optimal_alpha: bool = False,
):
    set_seed(seed)
    projected_samples = eps
    for i, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        x_data = jnp.asarray(batch['image'], dtype=float)
        y_data = jnp.asarray(batch['label'], dtype=float)
        N = x_data.shape[0]
        n_batches = N // sample_batch_size
        x_train_batched = x_data[:n_batches * sample_batch_size].reshape((n_batches, -1) + x_data.shape[1:])
        y_train_batched = y_data[:n_batches * sample_batch_size].reshape((n_batches, -1) + y_data.shape[1:])
        batched_eigvecs, batched_inv_eigvals = precompute_loss_inv(model_fn, loss_fn, params_vec, x_train_batched, y_train_batched)
        proj_vp_fn = lambda v : loss_kernel_proj_vp(vec=v, model_fn=model_fn, loss_fn=loss_fn, params=params_vec, 
                                                    x_train_batched=x_train_batched, y_train_batched=y_train_batched, batched_eigvecs=batched_eigvecs, batched_inv_eigvals=batched_inv_eigvals, 
                                                    n_iterations=n_iterations, x_val=x_val, y_val=y_val)
        projected_samples = jax.vmap(proj_vp_fn)(projected_samples)
        del x_train_batched, x_data, batched_eigvecs, batched_inv_eigvals, proj_vp_fn
    trace_proj = (jax.vmap(lambda e, x: jnp.dot(e, x), in_axes=(0,0))(eps, projected_samples)).mean()
    if use_optimal_alpha:
        print("Initial alpha:", alpha)
        alpha = jnp.dot(params_vec, params_vec) / (n_params - trace_proj)
        print("Optimal alpha:", alpha) 
    posterior_samples = jax.vmap(lambda single_sample: unflatten_fn(params_vec + 1/jnp.sqrt(alpha) * single_sample))(projected_samples)
    metrics = {"kernel_dim": trace_proj, "alpha": alpha}
    return posterior_samples, metrics

