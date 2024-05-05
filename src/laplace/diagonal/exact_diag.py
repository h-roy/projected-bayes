from functools import partial
from typing import Literal
import jax
import tree_math as tm
from jax import hessian, numpy as jnp
from src.helper import tree_random_normal_like
# from jax.config import config
# config.update('jax_disable_jit', True)
from src.losses import cross_entropy_loss

@partial(jax.jit, static_argnames=("model_fn", "output_dim", "likelihood"))
def exact_diagonal(model_fn,
                   params,
                   output_dim,
                   x_train_batch, 
                   likelihood: Literal["classification", "regression"] = "classification"
                   ):
    """"
    This function computes the exact diagonal of the GGN matrix.
    """
    n_data_pts = x_train_batch.shape[0]
    output_dim_vec = jnp.arange(output_dim)
    diag_init = jax.tree_map(lambda x: jnp.zeros_like(x), params)
    
    if likelihood == "regression":
        def body_fn(n, res):
            def single_dim_grad(carry, output_dim):
                model_single_dim = lambda p: (model_fn(p, x_train_batch[n])[output_dim])[0]
                new_grad = jax.grad(model_single_dim)(params)
                out = jax.tree_map(lambda x, y: x + y**2, carry, new_grad)
                return out, None
            scan_init = jax.tree_map(lambda x: jnp.zeros_like(x), params)
            grad, _ = jax.lax.scan(single_dim_grad, scan_init, output_dim_vec)
            return jax.tree_map(lambda x, y: x + y, res, grad)
        diag = jax.lax.fori_loop(0, n_data_pts, body_fn, diag_init)
        return diag
    elif likelihood == "classification":
        output_dim_vec = jnp.arange(output_dim)
        grid = jnp.meshgrid(output_dim_vec, output_dim_vec)
        coord_list = [entry.ravel() for entry in grid]
        output_cross_vec = jnp.vstack(coord_list).T
        def body_fn(n, res):
            preds_i = model_fn(params, x_train_batch[n][None, ...])
            preds_i = jax.nn.softmax(preds_i, axis=1)
            preds_i = jax.lax.stop_gradient(preds_i)
            D = jax.vmap(jnp.diag)(preds_i)
            H = jnp.einsum("bo, bi->boi", preds_i, preds_i)
            H = D - H

            def single_dim_grad(carry, output_dims):
                o_1, o_2 = output_dims
                model_single_dim_1 = lambda p: (model_fn(p, x_train_batch[n][None, ...])[0, o_1])
                model_single_dim_2 = lambda p: (model_fn(p, x_train_batch[n][None, ...])[0, o_2])
                new_grad_1 = jax.grad(model_single_dim_1)(params)
                new_grad_2 = jax.grad(model_single_dim_2)(params)
                h = H.at[0, o_1, o_2].get()
                prod_grad = (tm.Vector(new_grad_1) * tm.Vector(new_grad_2)).tree
                out = jax.tree_map(lambda x, y: x + h * y, carry, prod_grad)
                return out, None
            scan_init = jax.tree_map(lambda x: jnp.zeros_like(x), params)
            grad, _ = jax.lax.scan(single_dim_grad, scan_init, output_cross_vec)
            del H
            return jax.tree_map(lambda x, y: x + y, res, grad)
        diag = jax.lax.fori_loop(0, n_data_pts, body_fn, diag_init)
        return diag