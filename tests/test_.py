import pickle
import os
import argparse
import torch
from jax import random
import json
import datetime
from src.laplace.last_layer.extract_last_layer import last_layer_ggn
from src.losses import sse_loss, cross_entropy_loss
from src.helper import calculate_exact_ggn, tree_random_normal_like
from src.sampling.predictive_samplers import sample_predictive, sample_hessian_predictive
from jax import numpy as jnp
import jax
from jax import flatten_util
import matplotlib.pyplot as plt
import tree_math as tm
from src.laplace.diagonal import hutchinson_diagonal, exact_diagonal

import optax
from src.models import ConvNet
from src.data import get_mnist
from src.losses import cross_entropy_loss

def accuracy(v, x, y):
    logits = model_fn(v, x)
    return jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(y, axis=-1))

model = ConvNet(10)
batch_size = 1
train_loader, val_loader, _ = get_mnist(batch_size, n_samples_per_class=5)
val_img, val_label = next(iter(val_loader))['image'], next(iter(val_loader))['label']
params = model.init(random.PRNGKey(0), next(iter(train_loader))['image'])
variables, unflatten = jax.flatten_util.ravel_pytree(params)
# model_fn = lambda vec, x: model.apply(unflatten(vec), x)
model_fn = model.apply
def loss_fn(v, x, y):
    logits = model_fn(v, x)
    return cross_entropy_loss(logits, y)

value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

train_lrate = 1e-2
optimizer = optax.adam(train_lrate)
# optimizer_state = optimizer.init(variables)
optimizer_state = optimizer.init(params)
n_epochs = 10

for epoch in range(n_epochs):
    for batch in train_loader:
        img, label = batch['image'], batch['label']
        loss, grad = value_and_grad_fn(params, img, label)
        updates, optimizer_state = optimizer.update(grad, optimizer_state)
        params = optax.apply_updates(params, updates)
    
    acc = accuracy(params, val_img, val_label)
    print(f"Epoch {epoch}, loss {loss:.3f}, accuracy {acc:.3f}")

#Exact GGN:
variables, unflatten = jax.flatten_util.ravel_pytree(params)
model_vec_fn = lambda vec, x: model.apply(unflatten(vec), x)

n_params = len(variables)
b = batch_size
o = 10
GGN = 0
for batch in train_loader:
    img, label = batch['image'], batch['label']
    preds = model_vec_fn(variables, img)
    J = jax.jacfwd(model_vec_fn, argnums=0)(variables, img)
    H = jax.hessian(cross_entropy_loss, argnums=0)(preds, label)
    J = J.reshape(b * o, n_params)
    H = H.reshape(b * o, b * o)
    GGN += J.T @ H @ J

leafs, _ = jax.tree_util.tree_flatten(params)
N_llla = len(leafs[-1]) + len(leafs[-2])
ggn_ll = GGN[-N_llla:, -N_llla:]
ggn_ll_2 = 0
for batch in train_loader:
    img, label = batch['image'], batch['label']
    img = jnp.asarray(img)
    ggn_ll_2 += last_layer_ggn(model.apply, params, img, "classification")

print(jnp.allclose(ggn_ll, ggn_ll_2))
breakpoint()