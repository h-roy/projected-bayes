from typing import Literal
import jax
import tree_math as tm
from jax import hessian, numpy as jnp
from src.helper import tree_random_normal_like, get_gvp_fun
from jax import config

def hutchinson_diagonal(model_fn,
                        params,
                        gvp_batch_size,
                        n_samples,
                        key,
                        x_train,
                        likelihood: Literal["classification", "regression"] = "classification",
                        num_levels=5,
                        computation_type: Literal["serial", "parallel"] = "serial"
                        ):
    """"
    This function computes the diagonal of the GGN matrix using Hutchinson's method.
    """
    gvp_fn = get_gvp_fun(params, model_fn, x_train, gvp_batch_size, likelihood, "running", "tree")
    diag_init = jax.tree_map(lambda x: jnp.zeros_like(x), params)
    if computation_type == "serial":
        def diag_estimate_fn(key, control_variate):
            diag_init_ = jax.tree_map(lambda x: jnp.zeros_like(x), params)
            key_list = jax.random.split(key, n_samples)
            def single_eps_diag(n, res):
                diag = res
                key = key_list[n]
                eps = tree_random_normal_like(key, control_variate)
                c_v = tm.Vector(control_variate) * tm.Vector(eps)
                gvp = gvp_fn(eps)
                new_diag = (tm.Vector(eps) * (tm.Vector(gvp) - c_v) + tm.Vector(control_variate)).tree
                return jax.tree_map(lambda x, y: x + y, new_diag, diag)
            diag = jax.lax.fori_loop(0, n_samples, single_eps_diag, diag_init_)
            return jax.tree_map(lambda x: x/n_samples, diag)
    elif computation_type == "parallel":
        def diag_estimate_fn(key, control_variate):
            key_list = jax.random.split(key, n_samples)
            @jax.vmap
            def single_eps_diag(key):
                eps = tree_random_normal_like(key, control_variate)
                c_v = tm.Vector(control_variate) * tm.Vector(eps)
                gvp = gvp_fn(eps)
                new_diag = (tm.Vector(eps) * (tm.Vector(gvp) - c_v) + tm.Vector(control_variate)).tree
                return new_diag
            diag = single_eps_diag(key_list)
            return jax.tree_map(lambda x: x.mean(axis=0), diag)
    
    def body_fun(n, res):
        diag, key = res
        key, subkey = jax.random.split(key)
        diag_update = diag_estimate_fn(subkey, diag)
        diag = ((tm.Vector(diag) * n + tm.Vector(diag_update)) / (n + 1)).tree
        return (diag, key)
    diag, _ = jax.lax.fori_loop(0, num_levels, body_fun, (diag_init, key))
    return diag    