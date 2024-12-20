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


def sample_projections( 
    model_fn: Callable,
    params_vec,
    eps,
    alpha: float,
    x_train_batched,
    output_dims: int,
    n_iterations: int,
    x_val: jnp.ndarray,
    n_params,
    unflatten_fn: Callable,
    use_optimal_alpha: bool = False,
):
    # eps = tree_random_normal_like(key, params, n_posterior_samples)
    # prior_samples = jax.tree_map(lambda x: 1/jnp.sqrt(alpha) * x, eps)

    # Eps is a Standard Random Normal Pytree
    prior_samples = eps
    batched_eigvecs, batched_inv_eigvals = precompute_inv(model_fn, params_vec, x_train_batched, output_dims, "scan")
    proj_vp_fn = lambda v : kernel_proj_vp(vec=v, model_fn=model_fn, params=params_vec, x_train_batched=x_train_batched, 
                                           batched_eigvecs=batched_eigvecs, batched_inv_eigvals=batched_inv_eigvals, 
                                           output_dims=output_dims, n_iterations=n_iterations, x_val=x_val)
    projected_samples = jax.vmap(proj_vp_fn)(prior_samples)
    trace_proj = (jax.vmap(lambda e, x: jnp.dot(e, x), in_axes=(0,0))(eps, projected_samples)).mean()
    if use_optimal_alpha:
        print("Initial alpha:", alpha)
        alpha = jnp.dot(params_vec, params_vec) / (n_params - trace_proj)
        print("Optimal alpha:", alpha) 
    posterior_samples = jax.vmap(lambda single_sample: unflatten_fn(params_vec + 1/jnp.sqrt(alpha) * single_sample))(projected_samples)
    metrics = {"kernel_dim": trace_proj, "alpha": alpha}
    return posterior_samples, metrics

def sample_projections_dataloader( 
    model_fn: Callable,
    params_vec,
    eps,
    alpha: float,
    train_loader,
    sample_batch_size: int,
    seed,
    output_dims: int,
    n_iterations: int,
    x_val: jnp.ndarray,
    n_params,
    unflatten_fn: Callable,
    vmap_dim: int = 5,
    use_optimal_alpha: bool = False,
    data_sharding: bool = False,
    num_gpus: int = 1,
    acceleration: bool = False
):
    set_seed(seed)
    projected_samples = eps
    for i, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        x_data = jnp.asarray(batch['image'])

        N = x_data.shape[0]
        n_batches = N // sample_batch_size
        x_train_batched = x_data[:n_batches * sample_batch_size].reshape((n_batches, -1) + x_data.shape[1:])
        if data_sharding:
            P = jax.sharding.PartitionSpec
            devices = mesh_utils.create_device_mesh((num_gpus,))
            mesh = jax.sharding.Mesh(devices, ('x',))
            sharding = jax.sharding.NamedSharding(mesh, P('x',))
            # Sketchy
            num_batches = x_train_batched.shape[0]//num_gpus
            x_train_batched = x_train_batched[:num_batches * num_gpus]
            x_train_batched = jax.device_put(x_train_batched, sharding)
            params_vec = jax.device_put(params_vec, sharding)

        batched_eigvecs, batched_inv_eigvals = precompute_inv(model_fn, params_vec, x_train_batched, output_dims, "scan")

        proj_vp_fn = lambda v : kernel_proj_vp(vec=v, model_fn=model_fn, params=params_vec, x_train_batched=x_train_batched, 
                                           batched_eigvecs=batched_eigvecs, batched_inv_eigvals=batched_inv_eigvals, 
                                           output_dims=output_dims, n_iterations=n_iterations, x_val=x_val, acceleration=acceleration)
        # projected_samples = jax.vmap(proj_vp_fn)(projected_samples)
        projected_samples = projected_samples.reshape((-1, vmap_dim) + projected_samples.shape[1:])
        projected_samples = jax.lax.map(lambda p: jax.vmap(proj_vp_fn)(p), projected_samples)
        projected_samples = projected_samples.reshape((-1,) + projected_samples.shape[2:])
        del x_train_batched, x_data, batched_eigvecs, batched_inv_eigvals, proj_vp_fn
    trace_proj = (jax.vmap(lambda e, x: jnp.dot(e, x), in_axes=(0,0))(eps, projected_samples)).mean()
    if use_optimal_alpha:
        print("Initial alpha:", alpha)
        alpha = jnp.dot(params_vec, params_vec) / (n_params - trace_proj)
        print("Optimal alpha:", alpha) 
    posterior_samples = jax.vmap(lambda single_sample: unflatten_fn(params_vec + 1/jnp.sqrt(alpha) * single_sample))(projected_samples)
    metrics = {"kernel_dim": trace_proj, "alpha": alpha}
    return posterior_samples, metrics

    # set_seed(seed)
    # projected_samples = eps
    # batched_eigvecs = []
    # batched_inv_eigvals = []
    # for i, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
    #     x_batch = batch['image']
    #     eigvecs, inv_eigvals = precompute_inv_batch(model_fn, params_vec, x_batch, output_dims)
    #     batched_eigvecs.append(eigvecs)
    #     batched_inv_eigvals.append(inv_eigvals)
    # batched_eigvecs = jnp.stack(batched_eigvecs)
    # batched_inv_eigvals = jnp.stack(batched_inv_eigvals)
    # for iter_num in tqdm(range(n_iterations)):
    #     # train_loader = data.DataLoader(**data_loader_hparams)
    #     for j, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
    #         x_batch = jnp.asarray(batch['image'])
    #         # eigvecs, inv_eigvals = precompute_inv_batch(model_fn, params_vec, x_batch, output_dims)
    #         eigvecs, inv_eigvals = batched_eigvecs[j], batched_inv_eigvals[j]
    #         proj_vp_fn = lambda v : kernel_proj_vp_batch(vec=v, model_fn=model_fn, params=params_vec, x_batch=x_batch, eigvecs=eigvecs, inv_eigvals=inv_eigvals, output_dims=output_dims)
    #         projected_samples = jax.vmap(proj_vp_fn)(projected_samples)
    #         proj_norm = jnp.mean(jnp.asarray(jax.vmap(lambda p: jnp.dot(p, p))(projected_samples)))
    #         kernel_norm = jnp.mean(jnp.array(kernel_check(projected_samples + params_vec, model_fn, params_vec, x_val)))
    #     jax.debug.print("Iteration: {iter} Proj Norm: {proj_norm} Kernel Check: {kernel_norm}", iter=iter_num, proj_norm=proj_norm, kernel_norm=kernel_norm)

    # trace_proj = (jax.vmap(lambda e, x: jnp.dot(e, x), in_axes=(0,0))(eps, projected_samples)).mean()
    # if use_optimal_alpha:
    #     print("Initial alpha:", alpha)
    #     alpha = jnp.dot(params_vec, params_vec) / (n_params - trace_proj)
    #     print("Optimal alpha:", alpha) 
    # posterior_samples = jax.vmap(lambda single_sample: unflatten_fn(params_vec + 1/jnp.sqrt(alpha) * single_sample))(projected_samples)
    # metrics = {"kernel_dim": trace_proj, "alpha": alpha}
    # return posterior_samples, metrics



def sample_projections_ood_dataloader( 
    model_fn: Callable,
    params_vec,
    eps,
    alpha: float,
    train_loader,
    ood_loader,
    sample_batch_size: int,
    seed,
    output_dims: int,
    n_iterations: int,
    n_ood_iterations: int,
    x_val: jnp.ndarray,
    n_params,
    unflatten_fn: Callable,
    vmap_dim: int = 5,
    use_optimal_alpha: bool = False,
    data_sharding: bool = False,
    num_gpus: int = 1,
    acceleration: bool = False
):
    set_seed(seed)
    projected_samples = eps
    for _ in range(2):
        for i, batch in enumerate(tqdm(ood_loader, desc="Training", leave=False)):
            x_data = jnp.asarray(batch['image'])
            N = x_data.shape[0]
            n_batches = N // sample_batch_size
            x_train_batched = x_data[:n_batches * sample_batch_size].reshape((n_batches, -1) + x_data.shape[1:])
            x_val_ood = x_train_batched[0]
            batched_eigvecs, batched_inv_eigvals = precompute_inv(model_fn, params_vec, x_train_batched, output_dims, "scan")
            proj_vp_fn = lambda v : (v - kernel_proj_vp(vec=v, model_fn=model_fn, params=params_vec, x_train_batched=x_train_batched, 
                                            batched_eigvecs=batched_eigvecs, batched_inv_eigvals=batched_inv_eigvals, 
                                            output_dims=output_dims, n_iterations=n_ood_iterations, x_val=x_val_ood, acceleration=True))
            projected_samples = jax.vmap(proj_vp_fn)(projected_samples)
            del x_train_batched, x_data, batched_eigvecs, batched_inv_eigvals, proj_vp_fn
        # projected_samples = jax.tree_map(lambda x: 200 * x, projected_samples)

        for i, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            x_data = jnp.asarray(batch['image'])

            N = x_data.shape[0]
            n_batches = N // sample_batch_size
            x_train_batched = x_data[:n_batches * sample_batch_size].reshape((n_batches, -1) + x_data.shape[1:])
            if data_sharding:
                P = jax.sharding.PartitionSpec
                devices = mesh_utils.create_device_mesh((num_gpus,))
                mesh = jax.sharding.Mesh(devices, ('x',))
                sharding = jax.sharding.NamedSharding(mesh, P('x',))
                # Sketchy
                num_batches = x_train_batched.shape[0]//num_gpus
                x_train_batched = x_train_batched[:num_batches * num_gpus]
                x_train_batched = jax.device_put(x_train_batched, sharding)
                params_vec = jax.device_put(params_vec, sharding)

            batched_eigvecs, batched_inv_eigvals = precompute_inv(model_fn, params_vec, x_train_batched, output_dims, "scan")

            proj_vp_fn = lambda v : kernel_proj_vp(vec=v, model_fn=model_fn, params=params_vec, x_train_batched=x_train_batched, 
                                            batched_eigvecs=batched_eigvecs, batched_inv_eigvals=batched_inv_eigvals, 
                                            output_dims=output_dims, n_iterations=n_iterations, x_val=x_val, acceleration=acceleration)
            projected_samples = jax.vmap(proj_vp_fn)(projected_samples)
            # projected_samples = projected_samples.reshape((-1, vmap_dim) + projected_samples.shape[1:])
            # projected_samples = jax.lax.map(lambda p: jax.vmap(proj_vp_fn)(p), projected_samples)
            # projected_samples = projected_samples.reshape((-1,) + projected_samples.shape[2:])
            del x_train_batched, x_data, batched_eigvecs, batched_inv_eigvals, proj_vp_fn
    trace_proj = (jax.vmap(lambda e, x: jnp.dot(e, x), in_axes=(0,0))(eps, projected_samples)).mean()
    if use_optimal_alpha:
        print("Initial alpha:", alpha)
        alpha = jnp.dot(params_vec, params_vec) / (n_params - trace_proj)
        print("Optimal alpha:", alpha) 
    posterior_samples = jax.vmap(lambda single_sample: unflatten_fn(params_vec + 1/jnp.sqrt(alpha) * single_sample))(projected_samples)
    metrics = {"kernel_dim": trace_proj, "alpha": alpha}
    return posterior_samples, metrics
