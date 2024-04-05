from typing import Literal
import jax
import tree_math as tm
from jax import hessian, numpy as jnp
from src.helper import tree_random_normal_like
from jax import config

def exact_diagonal(model_fn,
                   params,
                   output_dim,
                   x_train,
                   n_data_pts, 
                   likelihood: Literal["classification", "regression"] = "classification"
                   ):
    """"
    This function computes the exact diagonal of the GGN matrix.
    """
    output_dim_vec = jnp.arange(output_dim)
    diag_init = jax.tree_map(lambda x: jnp.zeros_like(x), params)
    def body_fn(n, res):
        def single_dim_grad(output_dim):
            model_single_dim = lambda p: (model_fn(p, x_train[n])[output_dim])[0]
            return jax.grad(model_single_dim)(params)
        all_grads = jax.vmap(single_dim_grad)(output_dim_vec)
        grad = jax.tree_map(lambda x: (x**2).sum(axis=0), all_grads)
        return jax.tree_map(lambda x, y: x + y, res, grad)
    diag = jax.lax.fori_loop(0, n_data_pts, body_fn, diag_init)
    return diag