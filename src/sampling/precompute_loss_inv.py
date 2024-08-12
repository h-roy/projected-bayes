from typing import Callable, Literal
import jax
from jax import numpy as jnp
from jax import config
config.update("jax_debug_nans", True)

from functools import partial

def loss_kernel_vp(fn, v, params):
    _, jtv_fn = jax.vjp(fn, params)
    Jtv = jtv_fn(v.reshape((-1,)))[0]
    _, JJtv = jax.jvp(fn, (params,), (Jtv,))
    return JJtv

@partial(jax.jit, static_argnames=( "loss_fn", "model_fn"))
def precompute_loss_inv(
        model_fn: Callable,
        loss_fn: Callable,
        params,
        x_train_batched: jnp.ndarray,
        y_train_batched: jnp.ndarray,
):
    def body_fn(carry, batch):
        x, y = batch
        lmbd = lambda p: loss_fn(model_fn(p, x), y)
        kvp = lambda w: loss_kernel_vp(lmbd, w, params=params)
        batch_size = x.shape[0]
        JJt = jax.jacfwd(kvp, argnums=0)(jnp.ones((batch_size,)))
        JJt = JJt.reshape(batch_size, batch_size)
        eigvals, eigvecs = jnp.linalg.eigh(JJt)
        idx = eigvals < 1e-3
        inv_eigvals = jnp.where(idx, 1., eigvals)
        inv_eigvals = 1/inv_eigvals
        inv_eigvals = jnp.where(idx, 0., inv_eigvals)
        del lmbd, kvp, JJt
        return None, (inv_eigvals, eigvecs)
    init_spec = None
    _, (inv_eigvals, eigvecs) = jax.lax.scan(body_fn, init_spec, (x_train_batched, y_train_batched), unroll=1)
    return eigvecs, inv_eigvals

