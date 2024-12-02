import jax
import jax.numpy as jnp

from src.losses import accuracy_preds


def kernel_check(posterior, model_fn, params, x_test):
    def check_if_kernel(single_posterior):
        centered_posterior = single_posterior - params
        lmbd = lambda p: model_fn(p, x_test)
        _, Jv = jax.jvp(lmbd, (params,), (centered_posterior,))
        return jnp.linalg.norm(Jv)
    return jax.vmap(check_if_kernel)(posterior)

def sample_accuracy(pred_posterior, y_test):
    return jax.vmap(accuracy_preds, in_axes=(0,None))(pred_posterior, y_test)/y_test.shape[0] * 100

def vectorize_nn(model_fn, params):
    """Vectorize the Neural Network
    Inputs:
    parameters: Pytree of parameters
    model_fn: A function that takes in pytree parameters and data

    Outputs:
    params_vec: Vectorized parameters
    unflatten_fn: Unflatten function
    model_apply_vec: A function that takes in vectorized parameters and data
    """
    params_vec, unflatten_fn = jax.flatten_util.ravel_pytree(params)
    def model_apply_vec(params_vectorized, x):
        return model_fn(unflatten_fn(params_vectorized), x)
    return params_vec, unflatten_fn, model_apply_vec

def linearize_model_fn(model_fn, linearization_params):
    def linearized_model_fn(params, x):
        centered_params = jax.tree_map(lambda x,y: x - y, params, linearization_params)
        map_pred = model_fn(linearization_params, x)
        model_fn_p = lambda p: model_fn(p, x)
        jvp = jax.jvp(model_fn_p, (linearization_params,), (centered_params,))[1]
        return map_pred + jvp
    return jax.jit(linearized_model_fn)
