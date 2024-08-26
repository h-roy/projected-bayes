"""Classify a Gaussian mixture model."""
import argparse
import os
import pickle

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import optax
from src.models.fc import FC_NN
from src.losses import cross_entropy_loss, accuracy_preds, cross_entropy_loss_per_datapoint
from tueplots import bundles
config_plt = bundles.neurips2023()
config_plt['text.usetex'] = False
plt.rcParams.update(config_plt)


# A bunch of hyperparameters
seed = 1
num_data_in = 100
num_data_out = 100  # OOD
train_num_epochs = 100
train_batch_size = num_data_in
train_lrate = 1e-2
train_print_frequency = 10
calibrate_num_epochs = 100
calibrate_batch_size = num_data_in
calibrate_lrate = 1e-1
calibrate_print_frequency = 10
calibrate_log_alpha_min = 1e-3
evaluate_num_samples = 100
plot_num_linspace = 250
plot_xmin, plot_xmax = -7, 7
plot_figsize = (8, 3)

# Create data
key = jax.random.PRNGKey(seed)
key, key_1, key_2 = jax.random.split(key, num=3)
m = 2.0
mu_1, mu_2 = jnp.array((-m, m)), jnp.array((m, -m))
x_1 = 0.6 * jax.random.normal(key_1, (num_data_in, 2)) + mu_1[None, :]
y_1 = jnp.asarray(num_data_in * [[1, 0]])
x_2 = 0.6 * jax.random.normal(key_2, (num_data_in, 2)) + mu_2[None, :]
y_2 = jnp.asarray(num_data_in * [[0, 1]])
x_train = jnp.concatenate([x_1, x_2], axis=0)
y_train = jnp.concatenate([y_1, y_2], axis=0)

# Create model
hidden = 16
num_layers = 2
model = FC_NN(2, 16, 2)
model_apply = model.apply
key, subkey = jax.random.split(key, num=2)
variables_dict = model.init(subkey, x_train)
variables, unflatten = jax.flatten_util.ravel_pytree(variables_dict)

# Train the model

optimizer = optax.adam(train_lrate)
optimizer_state = optimizer.init(variables)


def loss_p(v, x, y):
    logits = model_apply(unflatten(v), x)
    return cross_entropy_loss(preds=logits, y=y)


loss_value_and_grad = jax.jit(jax.value_and_grad(loss_p, argnums=0))

for epoch in range(train_num_epochs):
    # Subsample data
    key, subkey = jax.random.split(key)
    idx = jax.random.choice(
        subkey, x_train.shape[0], (train_batch_size,), replace=False
    )

    # Apply an optimizer-step
    loss, grad = loss_value_and_grad(variables, x_train[idx], y_train[idx])
    updates, optimizer_state = optimizer.update(grad, optimizer_state)
    variables = optax.apply_updates(variables, updates)

    # Look at intermediate results
    if epoch % train_print_frequency == 0:
        y_pred = model_apply(unflatten(variables), x_train[idx])
        y_probs = jax.nn.softmax(y_pred, axis=-1)
        acc = accuracy_preds(preds=y_probs, batch_y=y_train[idx])
        print(f"Epoch {epoch}, loss {loss:.3f}, accuracy {acc:.3f}")

print()

# Calibrate the linearised Laplace

# Linearise the model around the calibrated alpha
model_fn = lambda v, x: model_apply(unflatten(v), x)

def model_linear(sample, v, x):
    """Evaluate the model after linearising around the optimal parameters."""
    fx = model_fn(v, x)
    _, jvp = jax.jvp(lambda p: model_fn(p, x), (v,), (sample - v,))
    return fx + jvp

# Samples from laplace Diffusion, sampled laplace and linearised laplace
J = jax.jacfwd(model_fn)(variables, x_train)
loss_fn = lambda v, x: cross_entropy_loss_per_datapoint(model_fn(v, x), y_train)
J_loss = jax.jacfwd(loss_fn)(variables, x_train)

pred = model_fn(variables, x_train)
pred = jax.nn.softmax(pred, axis=1)
pred = jax.lax.stop_gradient(pred)
GGN = 0
GGN_loss = 0
for i in range(J.shape[0]):
    # H = jax.hessian(cross_entropy_loss)(pred[i], y_train[i])
    # GGN += J[i].T @ H @ J[i]
    GGN += J[i].T @ J[i]

GGN_loss = J_loss.T @ J_loss

alpha = 1.0
Cov_sqrt = jnp.linalg.cholesky(jnp.linalg.inv(GGN + alpha * jnp.eye(GGN.shape[0])))

key, subkey = jax.random.split(key)
num_samples = 20
eps = jax.random.normal(subkey, (num_samples, len(variables)))

eigvals, eigvecs = jnp.linalg.eigh(GGN)
threshold = 1e-7
# idx = eigvals < threshold
idx_ = eigvals >= threshold

diag_ker = 1/jnp.sqrt(eigvals + alpha)
diag_ker = jnp.where(idx_, 0., diag_ker)
Cov_ker = eigvecs @ (diag_ker * eigvecs.T)
ker_samples = jax.vmap(lambda e: variables + (eigvecs @ (diag_ker * e)))(eps)


eigvals_loss, eigvecs_loss = jnp.linalg.eigh(GGN_loss)
threshold = 1e-7
# idx = eigvals < threshold
idx_ = eigvals_loss >= threshold

diag_ker_loss = 1/jnp.sqrt(eigvals_loss + alpha)
diag_ker_loss = jnp.where(idx_, 0., diag_ker_loss)
Cov_ker = eigvecs @ (diag_ker_loss * eigvecs.T)
ker_loss_samples = jax.vmap(lambda e: variables + (eigvecs_loss @ (diag_ker_loss * e)))(eps)


# Predict (in-distribution)
linearized_logits_fn = lambda sample_pytree, x: jax.vmap(lambda s: model_linear(s, variables, x))(sample_pytree)
sampled_logits_fn = lambda sample_pytree, x: jax.vmap(lambda s: model_fn(s, x))(sample_pytree)


# Create plotting grid
x_1d = jnp.linspace(plot_xmin, plot_xmax, num=plot_num_linspace)
x_plot_x, x_plot_y = jnp.meshgrid(x_1d, x_1d)
x_plot = jnp.stack((x_plot_x, x_plot_y)).reshape((2, -1)).T

# Compute marginal standard deviations for plotting inputs
linearized_ker_logits = linearized_logits_fn(ker_samples, x_plot)
sampled_ker_logits = sampled_logits_fn(ker_samples, x_plot)

linearized_ker_probs = jax.nn.softmax(linearized_ker_logits, axis=-1)
sampled_ker_probs = jax.nn.softmax(sampled_ker_logits, axis=-1)

stdvs_ker_linearized = jnp.sum(jnp.std(linearized_ker_probs, axis=0), axis=-1)
stdvs_ker_sampled = jnp.sum(jnp.std(sampled_ker_probs, axis=0), axis=-1)

stdev_ker_linearized_plot = stdvs_ker_linearized.T.reshape((plot_num_linspace, plot_num_linspace))
stdev_ker_sampled_plot = stdvs_ker_sampled.T.reshape((plot_num_linspace, plot_num_linspace))


linearized_ker_loss_logits = linearized_logits_fn(ker_loss_samples, x_plot)
sampled_ker_loss_logits = sampled_logits_fn(ker_loss_samples, x_plot)

linearized_ker_loss_probs = jax.nn.softmax(linearized_ker_loss_logits, axis=-1)
sampled_ker_loss_probs = jax.nn.softmax(sampled_ker_loss_logits, axis=-1)

stdvs_ker_loss_linearized = jnp.sum(jnp.std(linearized_ker_loss_logits, axis=0), axis=-1)
stdvs_ker_loss_sampled = jnp.sum(jnp.std(sampled_ker_loss_probs, axis=0), axis=-1)

stdev_ker_loss_linearized_plot = stdvs_ker_loss_linearized.T.reshape((plot_num_linspace, plot_num_linspace))
stdev_ker_loss_sampled_plot = stdvs_ker_loss_sampled.T.reshape((plot_num_linspace, plot_num_linspace))


# Compute labels for plotting inputs
logits_plot = model_apply(unflatten(variables), x_plot)
labels_plot = jax.nn.log_softmax(logits_plot).argmax(axis=-1)
labels_plot = labels_plot.T.reshape((plot_num_linspace, plot_num_linspace))
# Choose a plotting style
style_data = {
    "in": {
        "color": "black",
        "zorder": 1,
        "linestyle": "None",
        "marker": "o",
        "markeredgecolor": "grey",
        "alpha": 0.75,
    },
    "out": {
        "color": "white",
        "zorder": 1,
        "linestyle": "None",
        "marker": "P",
        "markeredgecolor": "black",
        "alpha": 0.75,
    },
}
style_contour = {
    "uq": {"cmap": "viridis", "zorder": 0}, #"vmin": 0, "vmax": 1},
    # "bdry": {"vmin": 0, "vmax": 1, "cmap": "seismic", "zorder": 0, "alpha": 0.5},
}


# Plot the results
# layout = [["uq_sam_ker"],["uq_lin_ker"]] #"bdry"]]

layout = [ ["uq_sam_ker","uq_lin_ker",], ["uq_loss_sam_ker","uq_loss_lin_ker",]] #"bdry"]]
_fig, axes = plt.subplot_mosaic(layout, sharex=True, sharey=True, figsize=plot_figsize, constrained_layout=True)

# axes["bdry"].set_title("Decision boundary")
# axes["bdry"].contourf(x_plot_x, x_plot_y, labels_plot, 3, **style_contour["bdry"])
# axes["bdry"].plot(x_train[:, 0], x_train[:, 1], **style_data["in"])



axes["uq_sam_ker"].set_title("Sampled Kernel Distribution")
axes["uq_sam_ker"].plot(x_train[:, 0], x_train[:, 1], **style_data["in"])
cbar = axes["uq_sam_ker"].contourf(x_plot_x, x_plot_y, stdev_ker_sampled_plot, **style_contour["uq"])

axes["uq_lin_ker"].set_title("Linearized Kernel uncertainty")
axes["uq_lin_ker"].plot(x_train[:, 0], x_train[:, 1], **style_data["in"])
cbar = axes["uq_lin_ker"].contourf(x_plot_x, x_plot_y, stdev_ker_linearized_plot, **style_contour["uq"])


axes["uq_loss_sam_ker"].set_title("Sampled Loss Kernel Distribution")
axes["uq_loss_sam_ker"].plot(x_train[:, 0], x_train[:, 1], **style_data["in"])
cbar = axes["uq_loss_sam_ker"].contourf(x_plot_x, x_plot_y, stdev_ker_loss_sampled_plot, **style_contour["uq"])

axes["uq_loss_lin_ker"].set_title("Linearized Loss Kernel uncertainty")
axes["uq_loss_lin_ker"].plot(x_train[:, 0], x_train[:, 1], **style_data["in"])
cbar = axes["uq_loss_lin_ker"].contourf(x_plot_x, x_plot_y, stdev_ker_loss_linearized_plot, **style_contour["uq"])

plt.colorbar(cbar)
# Save the plot to a file
plt.savefig("./figures/classify_gaussian_mixture.pdf")
