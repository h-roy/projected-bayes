from typing import Callable, Optional, Literal
import jax
import tree_math as tm
from jax import numpy as jnp
from functools import partial
from src.sampling.sample_utils import kernel_check
from jax import config
config.update("jax_debug_nans", True)
import logging

@partial(jax.jit, static_argnames=("model_fn", "output_dim", "n_iterations"))
def kernel_proj_vp(
        vec,
        model_fn: Callable,
        params,
        x_train_batched: jnp.ndarray,
        batched_eigvecs: jnp.ndarray, 
        batched_inv_eigvals: jnp.ndarray,
        output_dim: int,
        n_iterations: int,
        x_val: jnp.ndarray,
        ):
        
    def orth_proj_vp(v, x, eigvecs, inv_eigvals):
        lmbd = lambda p: model_fn(p, x)
        _, Jv = jax.jvp(lmbd, (params,), (v,)) 
        JJt_inv_Jv = eigvecs.T @ Jv.reshape(-1)
        JJt_inv_Jv = eigvecs @ (inv_eigvals * JJt_inv_Jv)
        _, jtv_fn = jax.vjp(lmbd, params)
        JJt_inv_Jv = JJt_inv_Jv.reshape((x.shape[0],output_dim))
        Jt_JJt_inv_Jv = jtv_fn(JJt_inv_Jv)[0]
        return (tm.Vector(v) - tm.Vector(Jt_JJt_inv_Jv)).tree
    def proj_through_data(iter, v):
        
        def body_fun(carry, batch):
            x, eigvecs, inv_eigvals = batch
            pv = carry
            out = orth_proj_vp(pv, x, eigvecs, inv_eigvals)
            return out, None
        init_carry = v
        Pv_k, _ = jax.lax.scan(body_fun, init_carry, (x_train_batched, batched_eigvecs, batched_inv_eigvals)) #memory error?
        I_Pv_k = tm.Vector(v) - tm.Vector(Pv_k)
        t = tm.Vector(v) @ I_Pv_k / (I_Pv_k @ I_Pv_k)
        Qv = tm.Vector(v) - t * I_Pv_k
        proj_norm = Qv @ Qv
        _, jvp = jax.jvp(lambda p: model_fn(p, x_val), (params,), (Qv.tree,))
        kernel_norm = jnp.linalg.norm(jvp)
        jax.debug.print("Iteration: {iter} Proj Norm: {proj_norm} Kernel Check: {kernel_norm}", iter=iter, proj_norm=proj_norm, kernel_norm=kernel_norm)
        return Qv.tree
    Pv = jax.lax.fori_loop(0, n_iterations, proj_through_data, vec)
    I_Pv = tm.Vector(vec) - tm.Vector(Pv)
    t = tm.Vector(vec) @ I_Pv / (I_Pv @ I_Pv)
    v_ = tm.Vector(vec) - t * I_Pv
    return v_.tree

    # @jax.jit
    # def proj_prior(v):
    #     Pv = jax.lax.fori_loop(0, n_iterations, proj_through_data, v)
    #     I_Pv = tm.Vector(v) - tm.Vector(Pv)
    #     t = tm.Vector(v) @ I_Pv / (I_Pv @ I_Pv)
    #     v_ = tm.Vector(v) - t * I_Pv
    #     return v_.tree
    # return proj_prior(vec)
    # return jaxPartial(proj_prior)
    # return proj_prior #Dont Jit

@partial(jax.jit, static_argnames=("output_dim", "model_fn"))
def kernel_proj_vp_batch(
        vec,
        model_fn: Callable,
        params,
        x_batch: jnp.ndarray,
        eigvecs: jnp.ndarray, 
        inv_eigvals: jnp.ndarray,
        output_dim: int,
        ):
    
    lmbd = lambda p: model_fn(p, x_batch)
    _, Jv = jax.jvp(lmbd, (params,), (vec,)) 
    JJt_inv_Jv = eigvecs.T @ Jv.reshape(-1)
    JJt_inv_Jv = eigvecs @ (inv_eigvals * JJt_inv_Jv)
    _, jtv_fn = jax.vjp(lmbd, params)
    JJt_inv_Jv = JJt_inv_Jv.reshape((x_batch.shape[0],output_dim))
    Jt_JJt_inv_Jv = jtv_fn(JJt_inv_Jv)[0]
    return (tm.Vector(vec) - tm.Vector(Jt_JJt_inv_Jv)).tree
    


