from typing import Callable, Literal
import jax
from jax import numpy as jnp
from jax import config
config.update("jax_debug_nans", True)

from functools import partial

def kernel_vp(fn, v, batch_size, output_dims, params):
    _, jtv_fn = jax.vjp(fn, params)
    Jtv = jtv_fn(v.reshape((batch_size, output_dims)))[0]
    _, JJtv = jax.jvp(fn, (params,), (Jtv,))
    return JJtv

@partial(jax.jit, static_argnames=( "output_dims", "type", "model_fn"))
def precompute_inv(
        model_fn: Callable,
        params,
        x_train_batched,
        output_dims: int,
        type: Literal["scan", "map", "vmap"] = "scan"
):
    if type == "scan":
        def body_fn(carry, batch):
            x = batch
            lmbd = lambda p: model_fn(p, x)
            kvp = lambda w: kernel_vp(lmbd, w, x.shape[0], output_dims=output_dims, params=params)
            batch_size = x.shape[0]
            JJt = jax.jacfwd(kvp, argnums=0)(jnp.ones((x.shape[0], output_dims)))
            JJt = JJt.reshape(batch_size * output_dims, batch_size * output_dims)
            eigvals, eigvecs = jnp.linalg.eigh(JJt)
            idx = eigvals < 1e-3
            inv_eigvals = jnp.where(idx, 1., eigvals)
            inv_eigvals = 1/inv_eigvals
            inv_eigvals = jnp.where(idx, 0., inv_eigvals)
            del lmbd, kvp, JJt
            return None, (inv_eigvals, eigvecs)
        init_spec = None
        _, (inv_eigvals, eigvecs) = jax.lax.scan(body_fn, init_spec, x_train_batched, unroll=1)
        return eigvecs, inv_eigvals
    elif type == "map":
        def body_fn(x):
            lmbd = lambda p: model_fn(p, x)
            kvp = lambda w: kernel_vp(lmbd, w, x.shape[0], output_dims=output_dims, params=params)   
            JJt = jax.jacfwd(kvp)(jnp.ones((x.shape[0], output_dims)))
            JJt = JJt.reshape(x.shape[0] * output_dims, x.shape[0] * output_dims)
            eigvals, eigvecs = jnp.linalg.eigh(JJt)
            idx = eigvals < 1e-7
            inv_eigvals = jnp.where(idx, 1., eigvals)
            inv_eigvals = 1/inv_eigvals
            inv_eigvals = jnp.where(idx, 0., inv_eigvals)
            return eigvecs, inv_eigvals
        eigvecs, inv_eigvals = jax.lax.map(body_fn, x_train_batched)
        return eigvecs, inv_eigvals
    elif type == "vmap":
        def body_fn(x):
            lmbd = lambda p: model_fn(p, x)
            kvp = lambda w: kernel_vp(lmbd, w, x.shape[0], output_dims=output_dims, params=params)  
            JJt = jax.jacfwd(kvp)(jnp.ones((x.shape[0], output_dims)))
            JJt = JJt.reshape(x.shape[0] * output_dims, x.shape[0] * output_dims)
            eigvals, eigvecs = jnp.linalg.eigh(JJt)
            idx = eigvals < 1e-7
            inv_eigvals = jnp.where(idx, 1., eigvals)
            inv_eigvals = 1/inv_eigvals
            inv_eigvals = jnp.where(idx, 0., inv_eigvals)
            return eigvecs, inv_eigvals
        eigvecs, inv_eigvals = jax.vmap(body_fn)(x_train_batched)
        return eigvecs, inv_eigvals

@partial(jax.jit, static_argnames=( "output_dims", "model_fn"))
def precompute_inv_batch(
        model_fn: Callable,
        params,
        x_batch,
        output_dims: int,
):
        lmbd = lambda p: model_fn(p, x_batch)
        batch_size = x_batch.shape[0]
        J = jax.jacrev(lmbd)(params)
        J = J.reshape(batch_size * output_dims, -1)
        JJt = J @ J.T
        # kvp = lambda w: kernel_vp(lmbd, w, batch_size, output_dims=output_dims, params=params)
        # JJt = jax.jacfwd(kvp, argnums=0)(jnp.ones((batch_size, output_dims)))
        # JJt = JJt.reshape(batch_size * output_dims, batch_size * output_dims)
        eigvals, eigvecs = jnp.linalg.eigh(JJt)
        idx = eigvals < 1e-7
        inv_eigvals = jnp.where(idx, 1., eigvals)
        inv_eigvals = 1/inv_eigvals
        inv_eigvals = jnp.where(idx, 0., inv_eigvals)
        return eigvecs, inv_eigvals
