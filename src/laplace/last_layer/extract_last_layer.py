from functools import partial
from typing import Literal
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnames=("likelihood", "model_fn", "num_ll_params"))
def last_layer_ggn(
    model_fn,
    params,
    x_batch,
    likelihood: Literal["classification", "regression"] = "classification",
    num_ll_params = None,
):
    if num_ll_params is None:
        leafs, _ = jax.tree_util.tree_flatten(params)
        N_llla = len(leafs[-1].flatten()) #+ len(leafs[-2])
    else:
        N_llla = num_ll_params
    params_vec, unflatten_fn = jax.flatten_util.ravel_pytree(params)

    def model_apply_vec(params_vectorized, x):
        return model_fn(unflatten_fn(params_vectorized), x)

    def last_layer_model_fn(last_params_vec, first_params, x):
        first_params = jax.lax.stop_gradient(first_params)
        vectorized_params = jnp.concatenate([first_params, last_params_vec])
        return model_apply_vec(vectorized_params, x)

    params_ll = params_vec.at[-N_llla:].get()
    params_fl = params_vec.at[:-N_llla].get()

    J_ll = jax.jacfwd(last_layer_model_fn, argnums=0)(
        params_ll, params_fl, x_batch
    )
    # B , O , N_llla
    if likelihood == "regression":
        J_ll = J_ll.reshape(-1, N_llla)
        return J_ll.T @ J_ll

    elif likelihood == "classification":
        pred = model_fn(params, x_batch)  # B, O
        pred = jax.nn.softmax(pred, axis=1)
        pred = jax.lax.stop_gradient(pred)
        D = jax.vmap(jnp.diag)(pred)
        H = jnp.einsum("bo, bi->boi", pred, pred)
        H = D - H  # B, O, O
        return jnp.einsum("mob, boo, bon->mn", J_ll.T, H, J_ll)
    else:
        raise ValueError("Invalid likelihood")







