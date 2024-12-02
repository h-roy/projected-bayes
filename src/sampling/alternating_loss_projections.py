from typing import Callable, Optional, Literal
import jax
from jax import numpy as jnp
from functools import partial
from src.sampling.sample_utils import kernel_check
from jax import config
config.update("jax_debug_nans", True)
import logging

@partial(jax.jit, static_argnames=("model_fn", "loss_fn", "n_iterations", "acceleration"))
def loss_kernel_proj_vp(
        vec,
        model_fn: Callable,
        loss_fn: Callable,
        params,
        x_train_batched: jnp.ndarray,
        y_train_batched: jnp.ndarray,
        batched_eigvecs: jnp.ndarray, 
        batched_inv_eigvals: jnp.ndarray,
        n_iterations: int,
        x_val: jnp.ndarray,
        y_val: jnp.ndarray,
        acceleration: bool = False
        ):
        
    def orth_proj_vp(v, x, y, eigvecs, inv_eigvals):
        lmbd = lambda p: loss_fn(model_fn(p, x), y)
        _, Jv = jax.jvp(lmbd, (params,), (v,)) 
        JJt_inv_Jv = eigvecs.T @ Jv.reshape(-1)
        JJt_inv_Jv = eigvecs @ (inv_eigvals * JJt_inv_Jv)
        _, jtv_fn = jax.vjp(lmbd, params)
        JJt_inv_Jv = JJt_inv_Jv.reshape((x.shape[0],))
        Jt_JJt_inv_Jv = jtv_fn(JJt_inv_Jv)[0]
        return v - Jt_JJt_inv_Jv
    def proj_through_data(iter, v):
        def body_fun(carry, batch):
            x, y, eigvecs, inv_eigvals = batch
            pv = carry
            out = orth_proj_vp(pv, x, y, eigvecs, inv_eigvals)
            return out, None
        init_carry = v
        Qv, _ = jax.lax.scan(body_fun, init_carry, (x_train_batched, y_train_batched, batched_eigvecs, batched_inv_eigvals)) #memory error?
        if acceleration:
            t_k = v @ (v - Qv)/ ((v - Qv) @ (v - Qv))
            x_k = t_k * Qv + (1 - t_k) * v
            proj_norm = x_k @ x_k #Qv @ Qv
            _, jvp = jax.jvp(lambda p: model_fn(p, x_val), (params,), (x_k,))
            kernel_norm = jnp.linalg.norm(jvp)
            jax.debug.print("Iteration: {iter} Proj Norm: {proj_norm} Kernel Check: {kernel_norm}", iter=iter, proj_norm=proj_norm, kernel_norm=kernel_norm/proj_norm)
            return x_k
        else:
            proj_norm = Qv @ Qv
            _, jvp = jax.jvp(lambda p: model_fn(p, x_val), (params,), (Qv,))
            kernel_norm = jnp.linalg.norm(jvp)
            jax.debug.print("Iteration: {iter} Proj Norm: {proj_norm} Kernel Check: {kernel_norm}", iter=iter, proj_norm=proj_norm, kernel_norm=kernel_norm/proj_norm)
            return Qv
    Pv = jax.lax.fori_loop(0, n_iterations, proj_through_data, vec)

    return Pv


@partial(jax.jit, static_argnames=("loss_model_fn", "n_iterations", "acceleration"))
def loss_kernel_gen_proj_vp(
        vec,
        loss_model_fn: Callable,
        params,
        x_train_batched: jnp.ndarray,
        batched_eigvecs: jnp.ndarray, 
        batched_inv_eigvals: jnp.ndarray,
        n_iterations: int,
        x_val: jnp.ndarray,
        acceleration: bool = False
        ):
        
    def orth_proj_vp(v, x, eigvecs, inv_eigvals):
        lmbd = lambda p: loss_model_fn(p, x)
        _, Jv = jax.jvp(lmbd, (params,), (v,)) 
        JJt_inv_Jv = eigvecs.T @ Jv.reshape(-1)
        JJt_inv_Jv = eigvecs @ (inv_eigvals * JJt_inv_Jv)
        _, jtv_fn = jax.vjp(lmbd, params)
        JJt_inv_Jv = JJt_inv_Jv.reshape((x.shape[0],))
        Jt_JJt_inv_Jv = jtv_fn(JJt_inv_Jv)[0]
        
        # _, jvp_fn = jax.linearize(lmbd, params)
        # vjp_fn = jax.linear_transpose(lmbd, params)
        # Jv = jvp_fn(v)
        # JJt_inv_Jv = eigvecs.T @ Jv.reshape(-1)
        # JJt_inv_Jv = eigvecs @ (inv_eigvals * JJt_inv_Jv)

        # # ...
        # Jt_JJt_inv_Jv = vjp_fn(JJt_inv_Jv)[0]
        return v - Jt_JJt_inv_Jv
    def proj_through_data(iter, v):
        def body_fun(carry, batch):
            x, eigvecs, inv_eigvals = batch
            pv = carry
            out = orth_proj_vp(pv, x, eigvecs, inv_eigvals)
            return out, None
        init_carry = v
        Qv, _ = jax.lax.scan(body_fun, init_carry, (x_train_batched, batched_eigvecs, batched_inv_eigvals)) #memory error?
        if acceleration:
            t_k = v @ (v - Qv)/ ((v - Qv) @ (v - Qv))
            x_k = t_k * Qv + (1 - t_k) * v
            proj_norm = x_k @ x_k #Qv @ Qv
            _, jvp = jax.jvp(lambda p: loss_model_fn(p, x_val), (params,), (x_k,))
            kernel_norm = jnp.linalg.norm(jvp)
            jax.debug.print("Iteration: {iter} Proj Norm: {proj_norm} Kernel Check: {kernel_norm}", iter=iter, proj_norm=proj_norm, kernel_norm=kernel_norm/proj_norm)
            return x_k
        else:
            proj_norm = Qv @ Qv
            _, jvp = jax.jvp(lambda p: loss_model_fn(p, x_val), (params,), (Qv,))
            kernel_norm = jnp.linalg.norm(jvp)
            jax.debug.print("Iteration: {iter} Proj Norm: {proj_norm} Kernel Check: {kernel_norm}", iter=iter, proj_norm=proj_norm, kernel_norm=kernel_norm/proj_norm)
            return Qv
    Pv = jax.lax.fori_loop(0, n_iterations, proj_through_data, vec)

    return Pv

