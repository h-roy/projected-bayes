from typing import Callable, Optional, Literal
import jax
import tree_math as tm
from jax import numpy as jnp
from src.sampling import precompute_inv, precompute_loss_inv, kernel_proj_vp, loss_kernel_proj_vp
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
    rng = jax.random.PRNGKey(seed)
    params_vec, _ = jax.flatten_util.ravel_pytree(params)
    eps = jax.vmap(lambda sample: jax.tree_map(lambda x,y: x - y, sample, params))(posterior_samples)
    eps, _ = jax.flatten_util.ravel_pytree(eps)
    n_samples = len(eps) // len(params_vec)
    eps = jnp.asarray(eps).reshape((n_samples, -1))
    projected_samples = eps
    x_val_ood = jnp.asarray(next(iter(test_loader))['image'])
    # for i, batch in enumerate(tqdm(test_loader, desc="Training", leave=False)):
    #     new_rng, rng = jax.random.split(rng)
    #     x_data = jnp.asarray(batch['image'])
    #     N = x_data.shape[0]
    #     n_batches = N // sample_batch_size
    #     x_train_batched = x_data[:n_batches * sample_batch_size].reshape((n_batches, -1) + x_data.shape[1:])
    #     rand_dir = jax.random.normal(new_rng, x_train_batched.shape)
    #     rand_dir = jax.vmap(lambda x: x / jnp.linalg.norm(x))(rand_dir)
    #     std = jax.random.uniform(new_rng, minval=100., maxval=200.)
    #     x_train_batched = x_train_batched + std * rand_dir
    #     # Information to kill from x_train_batched -> Zeros or rand like x_train batched then add mean and std are removed. Finally replace with test loader

    #     batched_eigvecs, batched_inv_eigvals = precompute_inv(model_fn, params_vec, x_train_batched, output_dims, "scan")
    #     proj_vp_fn = lambda v : (v - kernel_proj_vp(vec=v, model_fn=model_fn, params=params_vec, x_train_batched=x_train_batched, 
    #                                     batched_eigvecs=batched_eigvecs, batched_inv_eigvals=batched_inv_eigvals, 
    #                                     output_dims=output_dims, n_iterations=50, x_val=x_val_ood, acceleration=True))
    #     projected_samples = jax.vmap(proj_vp_fn)(projected_samples)
    #     del x_train_batched, x_data, batched_eigvecs, batched_inv_eigvals, proj_vp_fn

    for i, batch in enumerate(tqdm(ood_loader, desc="Training", leave=False)):
        new_rng, rng = jax.random.split(rng)
        x_data = jnp.asarray(batch['image'])
        N = x_data.shape[0]
        n_batches = N // sample_batch_size
        x_train_batched = x_data[:n_batches * sample_batch_size].reshape((n_batches, -1) + x_data.shape[1:])
        # mean = jnp.mean(x_train_batched, axis=(1, 2, 3))
        # std = jnp.std(x_train_batched, axis=(1, 2, 3))

        # x_train_batched = jax.random.normal(jax.random.PRNGKey(0), x_train_batched.shape) * std[:, None, None, None] + mean[:, None, None, None]   #Does not work!
        # x_train_batched = (jnp.zeros_like(x_train_batched) + mean[:, None, None, None]) #/ std[:, None, None, None] # Also works but worse, so does normalised data
        # x_train_batched = jax.random.normal(new_rng, x_train_batched.shape) * std[:, None, None, None] + mean[:, None, None, None]
        # Information to kill from x_train_batched -> Zeros or rand like x_train batched then add mean and std are removed. Finally replace with test loader

        batched_eigvecs, batched_inv_eigvals = precompute_inv(model_fn, params_vec, x_train_batched, output_dims, "scan")
        proj_vp_fn = lambda v : (v - kernel_proj_vp(vec=v, model_fn=model_fn, params=params_vec, x_train_batched=x_train_batched, 
                                        batched_eigvecs=batched_eigvecs, batched_inv_eigvals=batched_inv_eigvals, 
                                        output_dims=output_dims, n_iterations=50, x_val=x_val_ood, acceleration=True))
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



def ood_loss_projections( 
    model_fn: Callable,
    loss_fn: Callable,
    params,
    posterior_samples,
    test_loader,
    ood_loader,
    sample_batch_size: int,
    seed,
    unflatten_fn: Callable,
):
    set_seed(seed)
    rng = jax.random.PRNGKey(seed)
    params_vec, _ = jax.flatten_util.ravel_pytree(params)
    eps = jax.vmap(lambda sample: jax.tree_map(lambda x,y: x - y, sample, params))(posterior_samples)
    eps, _ = jax.flatten_util.ravel_pytree(eps)
    n_samples = len(eps) // len(params_vec)
    eps = jnp.asarray(eps).reshape((n_samples, -1))
    projected_samples = eps
    x_val_ood = jnp.asarray(next(iter(test_loader))['image'])
    y_val_ood = jnp.asarray(next(iter(test_loader))['label'])

    for i, batch in enumerate(tqdm(ood_loader, desc="Training", leave=False)):
        new_rng, rng = jax.random.split(rng)
        x_data, y_data = jnp.asarray(batch['image']), jnp.asarray(batch['label'])
        N = x_data.shape[0]
        n_batches = N // sample_batch_size
        x_train_batched = x_data[:n_batches * sample_batch_size].reshape((n_batches, -1) + x_data.shape[1:])
        y_train_batched = y_data[:n_batches * sample_batch_size].reshape((n_batches, -1) + y_data.shape[1:])
        batched_eigvecs, batched_inv_eigvals = precompute_loss_inv(model_fn, loss_fn, params_vec, x_train_batched, y_train_batched)
        proj_vp_fn = lambda v : (v - loss_kernel_proj_vp(vec=v, model_fn=model_fn, loss_fn=loss_fn, params=params_vec,
                                                        x_train_batched=x_train_batched, y_train_batched=y_train_batched, 
                                                        batched_eigvecs=batched_eigvecs, batched_inv_eigvals=batched_inv_eigvals, 
                                                        n_iterations=50, x_val=x_val_ood, y_val=y_val_ood, acceleration=True))
        projected_samples = jax.vmap(proj_vp_fn)(projected_samples)
        del x_train_batched, x_data, batched_eigvecs, batched_inv_eigvals, proj_vp_fn
    # projected_samples = jax.tree_map(lambda x: 200 * x, projected_samples)

    x_val = jnp.asarray(next(iter(test_loader))['image'])
    for i, batch in enumerate(tqdm(test_loader, desc="Training", leave=False)):
        x_data = jnp.asarray(batch['image'])
        N = x_data.shape[0]
        n_batches = N // sample_batch_size
        x_train_batched = x_data[:n_batches * sample_batch_size].reshape((n_batches, -1) + x_data.shape[1:])
        batched_eigvecs, batched_inv_eigvals = precompute_loss_inv(model_fn, loss_fn, params_vec, x_train_batched, y_train_batched)
        proj_vp_fn = lambda v : loss_kernel_proj_vp(vec=v, model_fn=model_fn, loss_fn=loss_fn, params=params_vec, 
                                                        x_train_batched=x_train_batched, y_train_batched=y_train_batched, 
                                                        batched_eigvecs=batched_eigvecs, batched_inv_eigvals=batched_inv_eigvals, 
                                                        n_iterations=5, x_val=x_val_ood, y_val=y_val_ood, acceleration=True)
        projected_samples = jax.vmap(proj_vp_fn)(projected_samples)
        del x_train_batched, x_data, batched_eigvecs, batched_inv_eigvals, proj_vp_fn

    posterior_samples = jax.vmap(lambda single_sample: unflatten_fn(params_vec + single_sample))(projected_samples)
    return posterior_samples
