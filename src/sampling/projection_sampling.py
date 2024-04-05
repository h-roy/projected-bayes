from typing import Callable, Optional, Literal
import jax
import tree_math as tm
from jax import numpy as jnp
import time
from src.helper import compute_num_params, get_gvp_fun, tree_random_normal_like
from jax import config
config.update("jax_debug_nans", True)

from functools import partial

def kernel_vp(fn, v, batch_size, output_dims, params):
    _, jtv_fn = jax.vjp(fn, params)
    Jtv = jtv_fn(v.reshape((batch_size, output_dims)))[0]
    _, JJtv = jax.jvp(fn, (params,), (Jtv,))
    return JJtv



def precompute_inv(
        model_fn: Callable,
        params,
        x_train_batched,
        output_dim: int,
        n_batches: int,
):
    def body_fn(carry, batch):
        x = batch
        lmbd = lambda p: model_fn(p, x)
        def kvp(v):
            _, jtv_fn = jax.vjp(lmbd, params)
            Jtv = jtv_fn(v.reshape((x.shape[0], output_dim)))[0]
            _, JJtv = jax.jvp(lmbd, (params,), (Jtv,))
            return JJtv 
        # kvp = lambda w: kernel_vp(lmbd, w, x.shape[0], output_dims=output_dims, params=params)
        batch_size = x.shape[0]
        JJt = jax.jacfwd(kvp, argnums=0)(jnp.ones((x.shape[0], output_dim)))
        JJt = JJt.reshape(batch_size * output_dim, batch_size * output_dim)
        eigvals, eigvecs = jnp.linalg.eigh(JJt)
        idx = eigvals < 1e-7
        inv_eigvals = jnp.where(idx, 1., eigvals)
        inv_eigvals = 1/inv_eigvals
        inv_eigvals = jnp.where(idx, 0., inv_eigvals)
        return None, (inv_eigvals, eigvecs)

    init_spec = None #(jnp.empty((batch_size * output_dims,)), jnp.empty((batch_size * output_dims, batch_size * output_dims)))
    _, (inv_eigvals, eigvecs) = jax.lax.scan(body_fn, init_spec, x_train_batched, unroll=1)
    return inv_eigvals, eigvecs
    
def ker_proj_vp(
        model_fn: Callable,
        params,
        x_train_batched,
        batched_eigvecs, 
        batched_inv_eigvals,
        n_batches: int,
        output_dim: int,
        n_iterations: int,
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
        v_, _ = jax.lax.scan(body_fun, init_carry, (x_train_batched, batched_eigvecs, batched_inv_eigvals)) #memory error?
        return v_
    @jax.jit
    def proj_prior(v):
        Pv = jax.lax.fori_loop(0, n_iterations, proj_through_data, v)
        I_Pv = tm.Vector(v) - tm.Vector(Pv)
        t = tm.Vector(v) @ I_Pv / (I_Pv @ I_Pv)
        v_ = tm.Vector(v) - t * I_Pv
        return v_.tree
    return proj_prior


# def precompute_inv(
#         model_fn: Callable,
#         params,
#         x_train_batched,
#         output_dim: int,
#         n_batches: int,
#         type: Literal["running", "parallel"] = "running"
# ):
#     if type == "running":
#         def body_fn(n, res):
#             x = x_train_batched[n]
#             carry_eigvecs, carry_inv_eigvals = res
#             lmbd = lambda p: model_fn(p, x)
#             # def kvp(v_):
#             #     _, jtv_fn = jax.vjp(lmbd, params)
#             #     Jtv = jtv_fn(v_.reshape((x.shape[0], output_dim)))[0]
#             #     _, JJtv = jax.jvp(lmbd, (params,), (Jtv,))
#             #     return JJtv    
#             lmbd = lambda p: model_fn(p, x)
#             kvp = lambda w: kernel_vp(lmbd, w, x.shape[0], output_dims=output_dim, params=params) 
#             JJt = jax.jacfwd(kvp)(jnp.ones((x.shape[0], output_dim))) # memory issue?
#             JJt = JJt.reshape(x.shape[0] * output_dim, x.shape[0] * output_dim)
#             eigvals, eigvecs = jnp.linalg.eigh(JJt)
#             idx = eigvals < 1e-7
#             inv_eigvals = jnp.where(idx, 1., eigvals)
#             inv_eigvals = 1/inv_eigvals
#             inv_eigvals = jnp.where(idx, 0., inv_eigvals)
#             carry_eigvecs = carry_eigvecs.at[n].set(eigvecs)
#             carry_inv_eigvals = carry_inv_eigvals.at[n].set(inv_eigvals)
#             return (carry_eigvecs, carry_inv_eigvals)
#         batch_size = x_train_batched.shape[1]
#         init_eigvecs, init_inv_eigvals = jnp.zeros((n_batches, batch_size * output_dim, batch_size * output_dim)), jnp.zeros((n_batches, batch_size * output_dim,))
#         eigvecs, inv_eigvals = jax.lax.fori_loop(0, n_batches, body_fn, (init_eigvecs, init_inv_eigvals)) # Scan over the data
#         return eigvecs, inv_eigvals
#     elif type == "parallel":
#         print("--> Here")
#         def body_fn(x):
            
#             # def kvp(v_):
#             #     lmbd = lambda p: model_fn(p, x)
#             #     _, jtv_fn = jax.vjp(lmbd, params)
#             #     Jtv = jtv_fn(v_.reshape((x.shape[0], output_dim)))[0]
#             #     _, JJtv = jax.jvp(lmbd, (params,), (Jtv,))
#             #     return JJtv 
#             lmbd = lambda p: model_fn(p, x)
#             kvp = lambda w: kernel_vp(lmbd, w, x.shape[0], output_dims=output_dim, params=params)   
#             JJt = jax.jacfwd(kvp)(jnp.ones((x.shape[0], output_dim)))
#             JJt = JJt.reshape(x.shape[0] * output_dim, x.shape[0] * output_dim)
#             eigvals, eigvecs = jnp.linalg.eigh(JJt)
#             idx = eigvals < 1e-7
#             inv_eigvals = jnp.where(idx, 1., eigvals)
#             inv_eigvals = 1/inv_eigvals
#             inv_eigvals = jnp.where(idx, 0., inv_eigvals)
#             return eigvecs, inv_eigvals
#         eigvecs, inv_eigvals = jax.lax.map(body_fn, x_train_batched)
#         return eigvecs, inv_eigvals

@partial(jax.jit, static_argnames=("n_posterior_samples", "output_dim", "n_batches", "n_iterations", "model_fn"))
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
):
    eps = tree_random_normal_like(key, params, n_posterior_samples)
    prior_samples = jax.tree_map(lambda x: 1/jnp.sqrt(alpha) * x, eps)
    # batched_eigvecs, batched_inv_eigvals = precompute_inv(model_fn, params, x_train_batched, output_dim, n_batches, "running")
    # batch_size = x_train_batched.shape[1]
    batched_eigvecs, batched_inv_eigvals = precompute_inv(model_fn, params, x_train_batched, output_dim, n_batches)
    # batched_eigvecs, batched_inv_eigvals = precompute_inv(model_fn, params, x_train_batched, output_dim, n_batches, "running")
    proj_vp_fn = ker_proj_vp(model_fn, params, x_train_batched, batched_eigvecs, batched_inv_eigvals, n_batches, output_dim, n_iterations)
    # im_proj_vp = lambda v: (tm.Vector(v) - tm.Vector(proj_vp_fn(v))).tree
    # proj_vp_fn = random_proj_vp(model_fn, params, x_train_batched, n_batches, output_dim, n_iterations, loss_type)
    projected_samples = jax.vmap(proj_vp_fn)(prior_samples)
    posterior_samples = jax.tree_map(lambda x, y: x + y, projected_samples, params)
    return posterior_samples




def averaged_proj_vp(
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
        # JJt_inv_Jv = jax.lax.custom_linear_solve(lambda v: kvp(v.reshape(x.shape[0], output_dim)).reshape(x.shape[0] * output_dim), Jv.reshape(-1))
        jax.debug.print("Residual eigen: {x}", x=jnp.linalg.norm(JJt @ JJt_inv_Jv - Jv.reshape(-1))/jnp.linalg.norm(Jv.reshape(-1)))

        _, jtv_fn = jax.vjp(lmbd, params)
        JJt_inv_Jv = JJt_inv_Jv.reshape((x.shape[0],output_dim))
        # JJt_inv_Jv = jnp.einsum("bij, bj -> bi", H_sqrt, JJt_inv_Jv) 
        Jt_JJt_inv_Jv = jtv_fn(JJt_inv_Jv)[0]
        return (tm.Vector(v) - tm.Vector(Jt_JJt_inv_Jv)).tree
    
    def proj_through_data(iter, v):
        def body_fun(n, res):
            return (tm.Vector(res) + tm.Vector(orth_proj_vp(v, x_train_batched[n]))).tree
        v_ = jax.lax.fori_loop(0, n_batches, body_fun, v)
        v_ = tm.Vector(v_) / n_batches
        return v_.tree
    @jax.jit
    def proj_prior(v):
        Pv = jax.lax.fori_loop(0, n_iterations, proj_through_data, v)
        return Pv
    return proj_prior


def random_proj_vp(
        model_fn: Callable,
        params,
        x_train_batched,
        n_batches: int,
        output_dim: int,
        n_iterations: int,
        # key: jax.random.KeyArray,
        loss_type: Literal["regression", "classification"] = "classification",):
    
    def orth_proj_vp(v, x):
        # lmbd = lambda p: model_fn(p, x)[jnp.arange(x.shape[0]), idx]
        # lmbd = lambda p: jnp.max(jax.nn.hard_tanh(model_fn(p, x)), axis=1)
        # lmbd = lambda p: jnp.stack([jnp.max(model_fn(p, x), axis=1), jnp.min(model_fn(p, x), axis=1)], axis=1)
        # out = lmbd(params)
        # out = jax.nn.softmax(out, axis=1)
        lmbd = lambda p: model_fn(p, x)

        def kvp(v_):
            # v_ = jnp.einsum("bij, bj -> bi", H_sqrt, v_)
            _, jtv_fn = jax.vjp(lmbd, params)
            Jtv = jtv_fn(v_.reshape((x.shape[0], output_dim)))[0]
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
        idx2 = eigvals < 1e-7
        inv_eigvals = jnp.where(idx2, 1., eigvals)
        inv_eigvals = 1/inv_eigvals
        inv_eigvals = jnp.where(idx2, 0., inv_eigvals)

        JJt_inv_Jv =eigvecs.T @ Jv.reshape(-1)
        JJt_inv_Jv = eigvecs @ (inv_eigvals * JJt_inv_Jv)
        # jax.debug.print("H_sqrt: {x}", x=H_sqrt)
        # JJt_inv_Jv = jax.lax.custom_linear_solve(lambda v: kvp(v.reshape(x.shape[0], output_dim)).reshape(x.shape[0] * output_dim), Jv.reshape(-1))
        jax.debug.print("Residual eigen: {x}", x=jnp.linalg.norm(JJt @ JJt_inv_Jv - Jv.reshape(-1))/jnp.linalg.norm(Jv.reshape(-1)))

        _, jtv_fn = jax.vjp(lmbd, params)
        JJt_inv_Jv = JJt_inv_Jv.reshape((x.shape[0], output_dim))
        # JJt_inv_Jv = jnp.einsum("bij, bj -> bi", H_sqrt, JJt_inv_Jv) 
        Jt_JJt_inv_Jv = jtv_fn(JJt_inv_Jv)[0]
        return (tm.Vector(v) - tm.Vector(Jt_JJt_inv_Jv)).tree
    
    def proj_through_data(iter, v):
        def body_fun(n, res):
            # use_key, key = jax.random.split(key)
            # idx = jax.random.randint(use_key, (n_batches,), 0, output_dim)[0]
            return orth_proj_vp(res, x_train_batched[n])
        v_ = jax.lax.fori_loop(0, n_batches, body_fun, v)
        return v_
    @jax.jit
    def proj_prior(v):
        Pv = jax.lax.fori_loop(0, n_iterations, proj_through_data, v)
        I_Pv = tm.Vector(v) - tm.Vector(Pv)
        t = tm.Vector(v) @ I_Pv / (I_Pv @ I_Pv)
        v_ = tm.Vector(v) - t * I_Pv
        return v_.tree
    return proj_prior

