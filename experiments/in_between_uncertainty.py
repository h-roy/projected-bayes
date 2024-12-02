import argparse
from functools import partial
import pickle
import time
from typing import Callable, Literal
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from src.baselines.laplace_baselines import last_layer_lapalce
from src.laplace.last_layer.extract_last_layer import last_layer_ggn
from src.models.fc import FC_NN
import tree_math as tm
from src.helper import compute_num_params, tree_random_normal_like
from src.losses import sse_loss
from src.sampling.predictive_samplers import sample_predictive
from src.sampling import sample_projections, sample_loss_projections

from src.sampling import precompute_inv, kernel_proj_vp

def f(x):
    return jnp.cos(4 * x + 0.8)

def sse_loss_per_datapoint(preds, y):
    preds = preds.squeeze()
    y = y.squeeze()
    residual = preds - y
    return (residual**2)

# Projection Sampling
param_dict = pickle.load(open("./checkpoints/syntetic_regression.pickle", "rb"))
params = param_dict['params']
params_vec, unflatten = jax.flatten_util.ravel_pytree(params)
alpha = param_dict['alpha']

rho = param_dict['rho']
X_train, Y_train, X_val, Y_val, model, D = param_dict["train_stats"]['x_train'],param_dict["train_stats"]['y_train'],param_dict["train_stats"]['x_val'],param_dict["train_stats"]['y_val'],param_dict["train_stats"]['model'], param_dict["train_stats"]['n_params']
X_train_batched = X_train.reshape(-1, 100, 1)
Y_train_batched = Y_train.reshape(-1, 100, 1)
output_dim = 1
model_fn = model.apply
n_samples = 200
key = jax.random.PRNGKey(0)
eps = tree_random_normal_like(key, params, n_samples)
n_iterations = 1
x_track = X_train_batched[0]
y_track = Y_train_batched[0]

def model_vec_apply(params_vec, x):
    return model.apply(unflatten(params_vec), x)
eps = jax.random.normal(key, (n_samples, D))
alpha = 0.5

#Proj Samples
proj_samples, metrics = sample_projections(model_vec_apply,
                             params_vec,
                             eps,
                             alpha,
                             X_train_batched,
                             output_dim,
                             n_iterations,
                             x_track,
                             D,
                             unflatten,

    )
linearised_lapalce = True
proj_predictive = sample_predictive(proj_samples, params, model_fn, X_val, linearised_lapalce, "Pytree")
proj_posterior_predictive_mean = jnp.mean(proj_predictive, axis=0).squeeze()
proj_posterior_predictive_std = jnp.std(proj_predictive, axis=0).squeeze()

#Loss Kernel Samples
loss_proj_samples, metrics = sample_loss_projections(
                            model_vec_apply,
                            sse_loss_per_datapoint,
                            params_vec,
                            eps,
                            alpha,
                            X_train_batched,
                            Y_train_batched,
                            n_iterations,
                            x_track,
                            y_track,
                            D,
                            unflatten,
                            False,
                            False
    )

linearised_lapalce = True
loss_proj_predictive = sample_predictive(loss_proj_samples, params, model_fn, X_val, linearised_lapalce, "Pytree")
loss_proj_posterior_predictive_mean = jnp.mean(loss_proj_predictive, axis=0).squeeze()
loss_proj_posterior_predictive_std = jnp.std(loss_proj_predictive, axis=0).squeeze()

# Full Linearised Samples

J = jax.jacfwd(model_vec_apply, argnums=0)(params_vec, X_train)
J = J.squeeze()
ggn = J.T @ J
full_samples = jax.vmap(lambda x: unflatten(params_vec + jnp.linalg.cholesky(jnp.linalg.inv(ggn + alpha * jnp.eye(D))) @ x))(eps)

linearised_predictive = sample_predictive(full_samples, params, model_fn, X_val, True, "Pytree")
linearised_posterior_predictive_mean = jnp.mean(linearised_predictive, axis=0).squeeze()
linearised_posterior_predictive_std = jnp.std(linearised_predictive, axis=0).squeeze()

sampled_predictive = sample_predictive(full_samples, params, model_fn, X_val, False, "Pytree")
sampled_posterior_predictive_mean = jnp.mean(sampled_predictive, axis=0).squeeze()
sampled_posterior_predictive_std = jnp.std(sampled_predictive, axis=0).squeeze()

# Diagonal GGN samples
diag_samples = jax.vmap(lambda x: unflatten(params_vec + 1/jnp.sqrt(jnp.diag(ggn) + alpha) * x))(eps)
diag_predictive = sample_predictive(diag_samples, params, model_fn, X_val, False, "Pytree")
diag_posterior_predictive_mean = jnp.mean(diag_predictive, axis=0).squeeze()
diag_posterior_predictive_std = jnp.std(diag_predictive, axis=0).squeeze()

# Last-layer GGN samples
ggn_ll = last_layer_ggn(model_fn, params, X_train, "regression", None)
prec = ggn_ll + alpha * jnp.eye(ggn_ll.shape[0])
N_llla = ggn_ll.shape[0]
map_fl = params_vec.at[:-N_llla].get()
map_ll = params_vec.at[-N_llla:].get()
def sample(key):
    eps = jax.random.normal(key, shape=(ggn_ll.shape[0],))
    sample = jnp.linalg.cholesky(jnp.linalg.inv(prec)) @ eps
    return sample + map_ll
key_list = jax.random.split(key, n_samples)
posterior_ll_samples = jax.vmap(sample)(key_list)
last_layer_samples = jax.vmap(lambda x: unflatten(jnp.concatenate([map_fl, x])))(posterior_ll_samples)
last_layer_predictive = sample_predictive(last_layer_samples, params, model_fn, X_val, False, "Pytree")
last_layer_posterior_predictive_mean = jnp.mean(last_layer_predictive, axis=0).squeeze()
last_layer_posterior_predictive_std = jnp.std(last_layer_predictive, axis=0).squeeze()


# Plotting
X_train, Y_train, X_val, Y_val = X_train.squeeze(), Y_train.squeeze(), X_val.squeeze(), Y_val.squeeze()

fig, ax = plt.subplots(ncols=5, figsize=(50, 10))

# line = ax[0].plot(X_val, proj_posterior_predictive_mean, label="Projection Posterior", marker="None")
# ax[0].fill_between(
#     X_val, proj_posterior_predictive_mean - proj_posterior_predictive_std, proj_posterior_predictive_mean + proj_posterior_predictive_std, alpha=0.1, color=line[0].get_color()
# )
# ax[0].plot(X_train, Y_train, "o", label="Training Data")
# ax[0].plot(X_val, Y_val, label="Ground Truth")
# ax[0].plot(X_val, model.apply(params, X_val), label="MAP")
# ax[0].set_title("Kernel Projection Posterior")
# ax[0].legend()

line = ax[0].plot(X_val, loss_proj_posterior_predictive_mean, label="Projection Posterior", marker="None")
ax[0].fill_between(
    X_val, loss_proj_posterior_predictive_mean - loss_proj_posterior_predictive_std, loss_proj_posterior_predictive_mean + loss_proj_posterior_predictive_std, alpha=0.1, color=line[0].get_color()
)
ax[0].plot(X_train, Y_train, "o", label="Training Data")
ax[0].plot(X_val, Y_val, label="Ground Truth")
ax[0].plot(X_val, model.apply(params, X_val), label="MAP")
ax[0].set_title("Loss Kernel Projection Posterior")
ax[0].legend()


line = ax[1].plot(X_val, last_layer_posterior_predictive_mean, label="Projection Posterior", marker="None")
ax[1].fill_between(
    X_val, last_layer_posterior_predictive_mean - last_layer_posterior_predictive_std, last_layer_posterior_predictive_mean + last_layer_posterior_predictive_std, alpha=0.1, color=line[0].get_color()
)
ax[1].plot(X_train, Y_train, "o", label="Training Data")
ax[1].plot(X_val, Y_val, label="Ground Truth")
ax[1].plot(X_val, model.apply(params, X_val), label="MAP")
ax[1].set_title("Last-Layer Posterior")
ax[1].legend()



line = ax[2].plot(X_val, linearised_posterior_predictive_mean, label="Linearised Laplace", marker="None")
ax[2].fill_between(
    X_val, linearised_posterior_predictive_mean - linearised_posterior_predictive_std, linearised_posterior_predictive_mean + linearised_posterior_predictive_std, alpha=0.1, color=line[0].get_color()
)
ax[2].plot(X_train, Y_train, "o", label="Training Data")
ax[2].plot(X_val, Y_val, label="Ground Truth")
ax[2].plot(X_val, model.apply(params, X_val), label="MAP")
ax[2].set_title("Linearised Laplace")
ax[2].legend()

line = ax[3].plot(X_val, sampled_posterior_predictive_mean, label="Sampled Laplace", marker="None")
ax[3].fill_between(
    X_val, sampled_posterior_predictive_mean - sampled_posterior_predictive_std, sampled_posterior_predictive_mean + sampled_posterior_predictive_std, alpha=0.1, color=line[0].get_color()
)
ax[3].plot(X_train, Y_train, "o", label="Training Data")
ax[3].plot(X_val, Y_val, label="Ground Truth")
ax[3].plot(X_val, model.apply(params, X_val), label="MAP")
ax[3].set_title("Sampled Laplace")
ax[3].legend()

line = ax[4].plot(X_val, diag_posterior_predictive_mean, label="Diagonal Laplace", marker="None")
ax[4].fill_between(
    X_val, diag_posterior_predictive_mean - diag_posterior_predictive_std, diag_posterior_predictive_mean + diag_posterior_predictive_std, alpha=0.1, color=line[0].get_color()
)
ax[4].plot(X_train, Y_train, "o", label="Training Data")
ax[4].plot(X_val, Y_val, label="Ground Truth")
ax[4].plot(X_val, model.apply(params, X_val), label="MAP")
ax[4].set_title("Diagonal Laplace")
ax[4].legend()

plt.savefig("figures/in_between_uncertainty.pdf")

