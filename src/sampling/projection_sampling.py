from typing import Callable, Optional, Literal
import jax
import tree_math as tm
from jax import numpy as jnp

from src.helper import compute_num_params, get_gvp_fun, tree_random_normal_like
from jax import config
config.update("jax_debug_nans", True)

from functools import partial

def proj_vp(
        model_fn: Callable,
        params,
        x_train_batched,
        n_batches: int,
        output_dim: int,
        n_iterations: int,
        loss_type: Literal["regression", "classification"] = "classification",):
    
    def orth_proj_vp(v, x):
        lmbd = lambda p: model_fn(p, x)
        out = lmbd(params)
        out = jax.nn.softmax(out, axis=1)
        if loss_type == "regression":
            H_sqrt = jax.vmap(lambda _: jnp.eye(output_dim))(out)
        elif loss_type == "classification":
            D_sqrt = jax.vmap(jnp.diag)(out**0.5)
            # H_sqrt = jnp.einsum('bo, bii, bi->boi', v, D_sqrt_inv, v)
            H_sqrt = jnp.einsum("bo, bi->boi", out, out**0.5)
            H_sqrt = D_sqrt - H_sqrt

        def kvp(v_):
            # v_ = jnp.einsum("bij, bj -> bi", H_sqrt, v_)
            _, jtv_fn = jax.vjp(lmbd, params)
            Jtv = jtv_fn(v_.reshape((x.shape[0],output_dim)))[0]
            _, JJtv = jax.jvp(lmbd, (params,), (Jtv,))
            # JJtv = jnp.einsum("bij, bj -> bi", H_sqrt, JJtv)
            return JJtv       

        _, Jv = jax.jvp(lmbd, (params,), (v,)) 
        JJt = jax.jacfwd(kvp)(Jv)
        JJt = JJt.reshape(x.shape[0] * output_dim, x.shape[0] * output_dim)

        
        # Jv = jnp.einsum("bij, bj -> bi", H_sqrt, Jv) 

        # JJt_inv_Jv = jnp.linalg.lstsq(JJt, Jv.reshape(-1))[0]
        # jax.debug.print("Residual lstsq: {x}", x=jnp.linalg.norm(JJt @ JJt_inv_Jv - Jv.reshape(-1)))

        eigvals, eigvecs = jnp.linalg.eigh(JJt)
        idx = eigvals < 1e-7
        inv_eigvals = jnp.where(idx, 1., eigvals)
        inv_eigvals = 1/inv_eigvals
        inv_eigvals = jnp.where(idx, 0., inv_eigvals)

        JJt_inv_Jv =eigvecs.T @ Jv.reshape(-1)
        JJt_inv_Jv = eigvecs @ (inv_eigvals * JJt_inv_Jv)
        # jax.debug.print("H_sqrt: {x}", x=H_sqrt)
        jax.debug.print("Residual eigen: {x}", x=jnp.linalg.norm(JJt @ JJt_inv_Jv - Jv.reshape(-1)))

        _, jtv_fn = jax.vjp(lmbd, params)
        JJt_inv_Jv = JJt_inv_Jv.reshape((x.shape[0],output_dim))
        # JJt_inv_Jv = jnp.einsum("bij, bj -> bi", H_sqrt, JJt_inv_Jv) 
        Jt_JJt_inv_Jv = jtv_fn(JJt_inv_Jv)[0]
        return (tm.Vector(v) - tm.Vector(Jt_JJt_inv_Jv)).tree
    
    def proj_through_data(iter, v):
        def body_fun(n, res):
            return orth_proj_vp(res, x_train_batched[n])
        v_ = jax.lax.fori_loop(0, n_batches - 1, body_fun, v)
        return v_
    @jax.jit
    def proj_prior(v):
        v_ = jax.lax.fori_loop(0, n_iterations - 1, proj_through_data, v)
        return v_
    return proj_prior

@partial(jax.jit, static_argnames=("loss_type", "n_posterior_samples", "output_dim", "model_fn"))
def sample_projections( 
    model_fn: Callable,
    params,
    x_train_batched,
    n_batches: int,
    key,
    alpha: float,
    output_dim: int,
    n_posterior_samples: int,
    n_iterations: int,
    loss_type: Literal["regression", "classification"] = "classification",
):
    eps = tree_random_normal_like(key, params, n_posterior_samples)
    prior_samples = jax.tree_map(lambda x: 1/jnp.sqrt(alpha) * x, eps)
    proj_vp_fn = proj_vp(model_fn, params, x_train_batched, n_batches, output_dim, n_iterations, loss_type)
    projected_samples = jax.vmap(proj_vp_fn)(prior_samples)
    posterior_samples = jax.tree_map(lambda x, y: x + y, projected_samples, params)
    return posterior_samples