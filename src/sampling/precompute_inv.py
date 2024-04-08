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

@partial(jax.jit, static_argnames=( "output_dim", "type", "model_fn"))
def precompute_inv(
        model_fn: Callable,
        params,
        x_train_batched,
        output_dim: int,
        type: Literal["scan", "map", "vmap"] = "scan"
):
    if type == "scan":
        def body_fn(carry, batch):
            x = batch
            lmbd = lambda p: model_fn(p, x)
            # def kvp(v):
            #     _, jtv_fn = jax.vjp(lmbd, params)
            #     Jtv = jtv_fn(v.reshape((x.shape[0], output_dim)))[0]
            #     _, JJtv = jax.jvp(lmbd, (params,), (Jtv,))
            #     return JJtv 
            kvp = lambda w: kernel_vp(lmbd, w, x.shape[0], output_dims=output_dim, params=params)
            batch_size = x.shape[0]
            # v0 = jnp.ones((x.shape[0] * output_dim))
            # jacfun = lambda v: jax.jvp(kvp, (v0,), (v,))[1]
            # JJt = jax.lax.map(jacfun, jnp.eye(batch_size * output_dim))
            JJt = jax.jacfwd(kvp, argnums=0)(jnp.ones((x.shape[0], output_dim)))
            JJt = JJt.reshape(batch_size * output_dim, batch_size * output_dim)
            eigvals, eigvecs = jnp.linalg.eigh(JJt)
            idx = eigvals < 1e-7
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
            # def kvp(v_):
            #     _, jtv_fn = jax.vjp(lmbd, params)
            #     Jtv = jtv_fn(v_.reshape((x.shape[0], output_dim)))[0]
            #     _, JJtv = jax.jvp(lmbd, (params,), (Jtv,))
            #     return JJtv    
            kvp = lambda w: kernel_vp(lmbd, w, x.shape[0], output_dims=output_dim, params=params)   
            JJt = jax.jacfwd(kvp)(jnp.ones((x.shape[0], output_dim)))
            JJt = JJt.reshape(x.shape[0] * output_dim, x.shape[0] * output_dim)
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
            # def kvp(v_):
            #     _, jtv_fn = jax.vjp(lmbd, params)
            #     Jtv = jtv_fn(v_.reshape((x.shape[0], output_dim)))[0]
            #     _, JJtv = jax.jvp(lmbd, (params,), (Jtv,))
            #     return JJtv
            kvp = lambda w: kernel_vp(lmbd, w, x.shape[0], output_dims=output_dim, params=params)  
            JJt = jax.jacfwd(kvp)(jnp.ones((x.shape[0], output_dim)))
            JJt = JJt.reshape(x.shape[0] * output_dim, x.shape[0] * output_dim)
            eigvals, eigvecs = jnp.linalg.eigh(JJt)
            idx = eigvals < 1e-7
            inv_eigvals = jnp.where(idx, 1., eigvals)
            inv_eigvals = 1/inv_eigvals
            inv_eigvals = jnp.where(idx, 0., inv_eigvals)
            return eigvecs, inv_eigvals
        eigvecs, inv_eigvals = jax.vmap(body_fn)(x_train_batched)
        return eigvecs, inv_eigvals

@partial(jax.jit, static_argnames=( "output_dim", "model_fn"))
def precompute_inv_batch(
        model_fn: Callable,
        params,
        x_batch,
        output_dim: int,
):
        lmbd = lambda p: model_fn(p, x_batch)
        # def kvp(v):
        #     _, jtv_fn = jax.vjp(lmbd, params)
        #     Jtv = jtv_fn(v.reshape((x.shape[0], output_dim)))[0]
        #     _, JJtv = jax.jvp(lmbd, (params,), (Jtv,))
        #     return JJtv 
        batch_size = x_batch.shape[0]
        kvp = lambda w: kernel_vp(lmbd, w, batch_size, output_dims=output_dim, params=params)
        JJt = jax.jacfwd(kvp, argnums=0)(jnp.ones((batch_size, output_dim)))
        JJt = JJt.reshape(batch_size * output_dim, batch_size * output_dim)
        eigvals, eigvecs = jnp.linalg.eigh(JJt)
        idx = eigvals < 1e-7
        inv_eigvals = jnp.where(idx, 1., eigvals)
        inv_eigvals = 1/inv_eigvals
        inv_eigvals = jnp.where(idx, 0., inv_eigvals)
        return eigvecs, inv_eigvals
