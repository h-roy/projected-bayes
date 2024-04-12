import jax
import jax.numpy as jnp

from src.losses import accuracy_preds


def kernel_check(posterior, model_fn, params, x_test):
    def check_if_kernel(single_posterior):
        centered_posterior = jax.tree_map(lambda x, y: x - y, single_posterior, params)
        lmbd = lambda p: model_fn(p, x_test)
        _, Jv = jax.jvp(lmbd, (params,), (centered_posterior,))
        return jnp.linalg.norm(Jv)
    return jax.vmap(check_if_kernel)(posterior)

def sample_accuracy(pred_posterior, y_test):
    return jax.vmap(accuracy_preds, in_axes=(0,None))(pred_posterior, y_test)/y_test.shape[0] * 100