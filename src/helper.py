from functools import partial
import jax
from jax import numpy as jnp
from typing import Optional
from typing import Callable, Literal, Optional
from functools import partial

from tqdm import tqdm
from src.losses import cross_entropy_loss, gaussian_log_lik_loss
import flax
import torch
import importlib
from typing import Any
import os
import random
import numpy as np

def ggn_vector_product_fast(
            vec: jnp.ndarray,
            model_fn: Callable,
            params_vec,
            train_loader,
            prod_batch_size: int,
            vmap_dim: int,
            likelihood_type: str = "regression",

):
    """
    vec: Array of vectors to be multiplied with the GGN.
    model_fn: Function that takes in vectorized parameters and data and returns the model output.
    params_vec: Vectorized parameters.
    alpha: Prior Precision.
    train_loader: DataLoader for the training data with very large batch size.
    prod_batch_size: Micro Batch size for the product.
    likelihood_type: Type of likelihood. Either "regression" or "classification".
    """
    # Can also use associative scan [Test later]
    # Linearize + Lienar transpose could be a bit faster
    assert vec.shape[0] % vmap_dim == 0
    out = jnp.zeros_like(vec)
    for i, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        x_data = jnp.asarray(batch['image'], dtype=float)
        N = x_data.shape[0]
        # assert N % prod_batch_size == 0
        n_batches = N // prod_batch_size
        x_train_batched = x_data[:n_batches * prod_batch_size].reshape((n_batches, -1) + x_data.shape[1:])
        gvp_fn = lambda v: ggn_vector_product(v, model_fn, params_vec, x_train_batched, likelihood_type)
        vec_t = vec.reshape((-1, vmap_dim) + vec.shape[1:])
        vec_ = jax.lax.map(lambda p: jax.vmap(gvp_fn)(p), vec_t)
        out += vec_.reshape(vec.shape)
    return out

@partial(jax.jit, static_argnames=("model_fn", "likelihood_type", "sum_type"))
def ggn_vector_product(
            vec: jnp.ndarray,
            model_fn: Callable,
            params_vec: jnp.ndarray,
            x_train_batched: jnp.ndarray,
            likelihood_type: str = "regression",
            sum_type: Literal["running", "parallel"] = "running",

):
    def gvp(vec):
        if sum_type == "running":
            def body_fn(carry, batch):
                x = batch
                model_on_data = lambda p: model_fn(p, x)
                _, J = jax.jvp(model_on_data, (params_vec,), (vec,))
                pred, model_on_data_vjp = jax.vjp(model_on_data, params_vec)
                if likelihood_type == "regression":
                    HJ = J
                elif likelihood_type == "classification":
                    pred = jax.nn.softmax(pred, axis=1)
                    pred = jax.lax.stop_gradient(pred)
                    D = jax.vmap(jnp.diag)(pred)
                    H = jnp.einsum("bo, bi->boi", pred, pred)
                    H = D - H
                    HJ = jnp.einsum("boi, bi->bo", H, J)
                else:
                    raise ValueError(f"Likelihood {likelihood_type} not supported. Use either 'regression' or 'classification'.")
                JtHJ = model_on_data_vjp(HJ)[0]
                return JtHJ, None
            init_carry = jnp.zeros_like(vec)
            return jax.lax.scan(body_fn, init_carry, x_train_batched)[0]
        elif sum_type == "running":
            def body_fn(x):
                model_on_data = lambda p: model_fn(p, x)
                _, J = jax.jvp(model_on_data, (params_vec,), (vec,))
                pred, model_on_data_vjp = jax.vjp(model_on_data, params_vec)
                if likelihood_type == "regression":
                    HJ = J
                elif likelihood_type == "classification":
                    pred = jax.nn.softmax(pred, axis=1)
                    pred = jax.lax.stop_gradient(pred)
                    D = jax.vmap(jnp.diag)(pred)
                    H = jnp.einsum("bo, bi->boi", pred, pred)
                    H = D - H
                    HJ = jnp.einsum("boi, bi->bo", H, J)
                else:
                    raise ValueError(f"Likelihood {likelihood_type} not supported. Use either 'regression' or 'classification'.")
                JtHJ = model_on_data_vjp(HJ)[0]
                return JtHJ
            ggn_vp = jax.vmap(body_fn)(x_train_batched)
            return jnp.sum(ggn_vp, axis=0)
    return gvp(vec)

def get_ggn_tree_product(
        params,
        model: flax.linen.Module,
        data_array: jax.Array = None,
        data_loader: torch.utils.data.DataLoader = None,
        likelihood_type: str = "regression",
        is_resnet: bool = False,
        batch_stats = None
    ):
    """
    takes as input a parameters pytree, a model and a dataset.
    returns a function v -> GGN * v, where v is a pytree "vector".
    Dataset can be given either ad an array or as a dataloader.
    """
    if data_array is not None:
        @jax.jit
        def ggn_tree_product(tree):
            if is_resnet:
                model_on_data = lambda p: model.apply({'params': p, 'batch_stats': batch_stats}, data_array, train=False, mutable=False)
            else:
                model_on_data = lambda p: model.apply(p, data_array)
            _, J_tree = jax.jvp(model_on_data, (params,), (tree,))
            pred, model_on_data_vjp = jax.vjp(model_on_data, params)
            if likelihood_type == "regression":
                HJ_tree = J_tree
            elif likelihood_type == "classification":
                pred = jax.nn.softmax(pred, axis=1)
                pred = jax.lax.stop_gradient(pred)
                D = jax.vmap(jnp.diag)(pred)
                H = jnp.einsum("bo, bi->boi", pred, pred)
                H = D - H
                HJ_tree = jnp.einsum("boi, bi->bo", H, J_tree)
            else:
                raise ValueError(f"Likelihood {likelihood_type} not supported. Use either 'regression' or 'classification'.")
            JtHJ_tree = model_on_data_vjp(HJ_tree)[0]
            return JtHJ_tree
    else:
        assert data_loader is not None
        @jax.jit
        def ggn_tree_product_single_batch(tree, data_array):
            if is_resnet:
                model_on_data = lambda p: model.apply({'params': p, 'batch_stats': batch_stats}, data_array, train=False, mutable=False)
            else:
                model_on_data = lambda p: model.apply(p, data_array)
            _, J_tree = jax.jvp(model_on_data, (params,), (tree,))
            pred, model_on_data_vjp = jax.vjp(model_on_data, params)
            if likelihood_type == "regression":
                HJ_tree = J_tree
            elif likelihood_type == "classification":
                pred = jax.nn.softmax(pred, axis=1)
                pred = jax.lax.stop_gradient(pred)
                D = jax.vmap(jnp.diag)(pred)
                H = jnp.einsum("bo, bi->boi", pred, pred)
                H = D - H
                HJ_tree = jnp.einsum("boi, bi->bo", H, J_tree)
            else:
                raise ValueError(f"Likelihood {likelihood_type} not supported. Use either 'regression' or 'classification'.")
            JtHJ_tree = model_on_data_vjp(HJ_tree)[0]
            return JtHJ_tree
        #@jax.jit
        def ggn_tree_product(tree):
            result = jax.tree_util.tree_map(lambda x : x*0, tree)
            for batch in data_loader:
                data_array = jnp.asarray(batch['image'], dtype=float)
                JtHJ_tree = ggn_tree_product_single_batch(tree, data_array)
                result = jax.tree_util.tree_map(lambda a, b: a+b, JtHJ_tree, result)
            return result
    return ggn_tree_product

def get_ggn_vector_product(
        params,
        model: flax.linen.Module,
        data_array: jax.Array = None,
        data_loader: torch.utils.data.DataLoader = None,
        likelihood_type: str = "regression",
        is_resnet: bool = False,
        batch_stats = None
    ):
    """
    takes as input a parameters pytree, a model and a dataset.
    returns a function v -> GGN * v, where v is a jnp.array vector.
    Dataset can be given either ad an array or as a dataloader.
    """
    ggn_tree_product = get_ggn_tree_product(
        params, 
        model, 
        data_array,
        data_loader,
        likelihood_type=likelihood_type,
        is_resnet=is_resnet,
        batch_stats=batch_stats)
    devectorize_fun = jax.flatten_util.ravel_pytree(params)[1]
    def ggn_vector_product(v):
        tree = devectorize_fun(v)
        ggn_tree = ggn_tree_product(tree)
        ggn_v = jax.flatten_util.ravel_pytree(ggn_tree)[0]
        return jnp.array(ggn_v)
    
    if data_array is not None:
        return jax.jit(ggn_vector_product)
    else:
        return ggn_vector_product
    
# @partial(jax.jit, static_argnames=("model", "likelihood_type"))

def get_gvp_fun(params,
                model_fn,
                data_array: jax.Array,
                batch_size = -1,
                likelihood_type: str = "regression",
                sum_type: Literal["running", "parallel"] = "running",
                v_in_type: Literal["vector", "tree"] = "vector"
  ) -> Callable:
  if sum_type == "running":
    def gvp(eps):
        def scan_fun(carry, batch):
            x_ = batch
            if batch.shape[0]>1:
                model_on_data = lambda p: model_fn(p,x_)
            else:
                model_on_data = lambda p: model_fn(p,x_[None,:])
            # Linearise and use transpose
            _, J_tree = jax.jvp(model_on_data, (params,), (eps,))
            pred, model_on_data_vjp = jax.vjp(model_on_data, params)
            if likelihood_type == "regression":
                HJ_tree = J_tree
            elif likelihood_type == "classification":
                pred = jax.nn.softmax(pred, axis=1)
                pred = jax.lax.stop_gradient(pred)
                D = jax.vmap(jnp.diag)(pred)
                H = jnp.einsum("bo, bi->boi", pred, pred)
                H = D - H
                HJ_tree = jnp.einsum("boi, bi->bo", H, J_tree)
            else:
                raise ValueError(f"Likelihood {likelihood_type} not supported. Use either 'regression' or 'classification'.")
            JtHJ_tree = model_on_data_vjp(HJ_tree)[0]
            return jax.tree_map(lambda c, v: c + v, carry, JtHJ_tree), None
        init_value = jax.tree_map(lambda x: jnp.zeros_like(x), params)
        return jax.lax.scan(scan_fun, init_value, data_array)[0]
    _, unravel_func_p = jax.flatten_util.ravel_pytree(params)
    def matvec(v_like_params):
        p_unravelled = unravel_func_p(v_like_params)
        ggn_vp = gvp(p_unravelled)
        f_eval, _ = jax.flatten_util.ravel_pytree(ggn_vp)
        return f_eval
    if v_in_type == "vector":
        return matvec
    elif v_in_type == "tree":
        return gvp
    # return jax.jit(matvec)
  elif sum_type == "parallel":
    def gvp(eps):
        @jax.vmap
        def body_fn(batch):  
            x_ = batch
            if batch_size>0:
                model_on_data = lambda p: model_fn(p,x_)
            else:
                model_on_data = lambda p: model_fn(p,x_[None,:])
            # Linearise and use transpose
            _, J_tree = jax.jvp(model_on_data, (params,), (eps,))
            pred, model_on_data_vjp = jax.vjp(model_on_data, params)
            if likelihood_type == "regression":
                HJ_tree = J_tree
            elif likelihood_type == "classification":
                pred = jax.nn.softmax(pred, axis=1)
                pred = jax.lax.stop_gradient(pred)
                D = jax.vmap(jnp.diag)(pred)
                H = jnp.einsum("bo, bi->boi", pred, pred)
                H = D - H
                HJ_tree = jnp.einsum("boi, bi->bo", H, J_tree)
            else:
                raise ValueError(f"Likelihood {likelihood_type} not supported. Use either 'regression' or 'classification'.")
            JtHJ_tree = model_on_data_vjp(HJ_tree)[0]
            return JtHJ_tree
        return jax.tree_map(lambda x: x.sum(axis=0), body_fn(data_array))
    _, unravel_func_p = jax.flatten_util.ravel_pytree(params)
    def matvec(v_like_params):
        p_unravelled = unravel_func_p(v_like_params)
        ggn_vp = gvp(p_unravelled)
        f_eval, _ = jax.flatten_util.ravel_pytree(ggn_vp)
        return f_eval
    # return jax.jit(matvec)
    if v_in_type == "vector":
        return matvec
    elif v_in_type == "tree":
        return gvp
  
def compute_num_params(pytree):
    return sum(x.size if hasattr(x, "size") else 0 for x in jax.tree_util.tree_leaves(pytree))

def calculate_exact_ggn(loss_fn, model_fn, params, X, y, n_params):
    def body_fun(carry, a_tuple):
        x, y = a_tuple
        my_model_fn = partial(model_fn, x=x)  # model_fn wrt parameters
        my_loss_fn = partial(loss_fn, y=y)  # loss_fn wrt model output
        pred = my_model_fn(params)
        jacobian = jax.jacfwd(my_model_fn)(params)
        jacobian = jax.tree_map(lambda x: jnp.reshape(x, (x.shape[0], -1)), jacobian)
        jacobian = jnp.concatenate(jax.tree_util.tree_flatten(jacobian)[0], axis=-1)
        loss_hessian = jax.hessian(my_loss_fn)(pred)
        ggn = jacobian.T @ loss_hessian @ jacobian
        return jax.tree_map(lambda a, b: a + b, carry, ggn), None

    init_value = jnp.zeros((n_params, n_params))  # jacobian.T @ loss_hessian @ jacobian
    return jax.lax.scan(body_fun, init_value, (X, y))[0]

def random_split_like_tree(rng_key, target=None, treedef=None):
    # https://github.com/google/jax/discussions/9508
    if treedef is None:
        treedef = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def tree_random_normal_like(rng_key, target, n_samples: Optional[int] = None):
    # https://github.com/google/jax/discussions/9508
    keys_tree = random_split_like_tree(rng_key, target)
    if n_samples is None:
        return jax.tree_util.tree_map(
            lambda l, k: jax.random.normal(k, l.shape, l.dtype),
            target,
            keys_tree,
        )
    else:
        return jax.tree_util.tree_map(
            lambda l, k: jax.random.normal(k, (n_samples,) + l.shape, l.dtype),
            target,
            keys_tree,
        )


def tree_random_uniform_like(rng_key, target, n_samples: Optional[int] = None, minval: int = 0, maxval: int = 1):
    keys_tree = random_split_like_tree(rng_key, target)
    if n_samples is None:
        return jax.tree_util.tree_map(
            lambda l, k: jax.random.uniform(k, l.shape, l.dtype, minval, maxval),
            target,
            keys_tree,
        )
    else:
        return jax.tree_util.tree_map(
            lambda l, k: jax.random.uniform(k, (n_samples,) + l.shape, l.dtype, minval, maxval),
            target,
            keys_tree,
        )

def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def set_seed(seed: int = 666, precision: int = 10) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_printoptions(precision=precision)
