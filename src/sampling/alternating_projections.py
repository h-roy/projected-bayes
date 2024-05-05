from typing import Callable, Optional, Literal
import jax
from jax import numpy as jnp
from functools import partial
from src.sampling.sample_utils import kernel_check
from jax import config
config.update("jax_debug_nans", True)
import logging

@partial(jax.jit, static_argnames=("model_fn", "output_dims", "n_iterations"))
def kernel_proj_vp(
        vec,
        model_fn: Callable,
        params,
        x_train_batched: jnp.ndarray,
        batched_eigvecs: jnp.ndarray, 
        batched_inv_eigvals: jnp.ndarray,
        output_dims: int,
        n_iterations: int,
        x_val: jnp.ndarray,
        ):
        
    def orth_proj_vp(v, x, eigvecs, inv_eigvals):
        lmbd = lambda p: model_fn(p, x)
        _, Jv = jax.jvp(lmbd, (params,), (v,)) 
        JJt_inv_Jv = eigvecs.T @ Jv.reshape(-1)
        JJt_inv_Jv = eigvecs @ (inv_eigvals * JJt_inv_Jv)
        _, jtv_fn = jax.vjp(lmbd, params)
        JJt_inv_Jv = JJt_inv_Jv.reshape((x.shape[0],output_dims))
        Jt_JJt_inv_Jv = jtv_fn(JJt_inv_Jv)[0]
        return v - Jt_JJt_inv_Jv
    def proj_through_data(iter, v):
        
        def body_fun(carry, batch):
            x, eigvecs, inv_eigvals = batch
            pv = carry
            out = orth_proj_vp(pv, x, eigvecs, inv_eigvals)
            return out, None
        init_carry = v
        Qv, _ = jax.lax.scan(body_fun, init_carry, (x_train_batched, batched_eigvecs, batched_inv_eigvals)) #memory error?
        proj_norm = Qv @ Qv
        _, jvp = jax.jvp(lambda p: model_fn(p, x_val), (params,), (Qv,))
        kernel_norm = jnp.linalg.norm(jvp)
        jax.debug.print("Iteration: {iter} Proj Norm: {proj_norm} Kernel Check: {kernel_norm}", iter=iter, proj_norm=proj_norm, kernel_norm=kernel_norm)
        return Qv
    Pv = jax.lax.fori_loop(0, n_iterations, proj_through_data, vec)
    return Pv

@partial(jax.jit, static_argnames=("output_dims", "model_fn"))
def kernel_proj_vp_batch(
        vec,
        model_fn: Callable,
        params,
        x_batch: jnp.ndarray,
        eigvecs: jnp.ndarray, 
        inv_eigvals: jnp.ndarray,
        output_dims: int,
        ):
    
    lmbd = lambda p: model_fn(p, x_batch)
    _, Jv = jax.jvp(lmbd, (params,), (vec,)) 
    JJt_inv_Jv = eigvecs.T @ Jv.reshape(-1)
    JJt_inv_Jv = eigvecs @ (inv_eigvals * JJt_inv_Jv)
    _, jtv_fn = jax.vjp(lmbd, params)
    JJt_inv_Jv = JJt_inv_Jv.reshape((x_batch.shape[0],output_dims))
    Jt_JJt_inv_Jv = jtv_fn(JJt_inv_Jv)[0]
    return vec - Jt_JJt_inv_Jv
    


