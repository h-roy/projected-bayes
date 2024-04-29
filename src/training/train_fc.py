import argparse
import pickle
import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from src.models.fc import FC_NN
import tree_math as tm
from src.helper import compute_num_params
from src.losses import sse_loss

def f(x):
    return jnp.cos(4 * x + 0.8)

def build_inbetween_dataset(N=100, noise_var=0.01, key=jax.random.PRNGKey(0)):
    key_1, key_2, key_3 = jax.random.split(key, 3)
    X_1 = jax.random.uniform(key_1, shape=(N//2, 1), minval=-1, maxval=-0.7)
    X_2 = jax.random.uniform(key_2, shape=(N//2, 1), minval=0.5, maxval=1)
    X = jnp.concatenate([X_1, X_2], axis=0)
    cosx = jnp.cos(4 * X + 0.8)
    randn = jax.random.normal(key_3, shape=(N, 1))*jnp.sqrt(noise_var)
    Y = cosx + randn
    return X, Y

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--noise_var", type=int, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(args)

    N = args.N
    noise_var = args.noise_var
    seed = args.seed
    model_key, data_key = jax.random.split(jax.random.PRNGKey(seed))
    rho = 1 / noise_var
    alpha = 1.

    X_train, Y_train = build_inbetween_dataset(N=N, noise_var=noise_var, key=data_key)
    X_val = jnp.linspace(-2, 2, 100).reshape(-1, 1)
    Y_val = f(X_val)

    model = FC_NN(out_dims=1, hidden_dim=10, num_layers=2)
    batch_size = 10
    n_batches = N // batch_size
    params = model.init(model_key, X_train[:batch_size])
    D = compute_num_params(params)

    log_alpha = jnp.log(alpha)
    log_rho = jnp.log(rho)

    def map_loss(params, x, y):
        B = x.shape[0]
        O = y.shape[-1]
        out = model.apply(params, x)
        vparams = tm.Vector(params)
        log_likelihood = (
            -N * O / 2 * jnp.log(2 * jnp.pi)
            + N * O / 2 * log_rho
            - (N / B) * 0.5 * rho * jnp.sum(jax.vmap(sse_loss)(out, y))  # Sum over the observations
        )
        log_prior = -D / 2 * jnp.log(2 * jnp.pi) + D / 2 * log_alpha - 0.5 * alpha * vparams @ vparams
        loss = log_likelihood + log_prior
        return -loss, (log_likelihood, log_prior)

    lr = 1e-3
    n_epochs = 2000
    optim = optax.adam(lr)
    opt_state = optim.init(params)

    def make_step(params, opt_state, x, y):
        grad_fn = jax.value_and_grad(map_loss, argnums=0, has_aux=True)
        loss, grads = grad_fn(params, x, y)
        param_updates, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, param_updates)
        return loss, params, opt_state

    jit_make_step = jax.jit(make_step)

    losses = []
    log_likelihoods = []
    log_priors = []
    mean_loss = []
    mse_preds = []

    print("Starting training...")
    for epoch in range(n_epochs):
        start_time = time.time()
        shuffle_key, split_key = jax.random.split(data_key)
        batch_indices_shuffled = jax.random.permutation(shuffle_key, X_train.shape[0])
        for i in range(n_batches):
            train_key, split_key = jax.random.split(split_key)
            x_batch = X_train[batch_indices_shuffled[i * batch_size : (i + 1) * batch_size]]
            y_batch = Y_train[batch_indices_shuffled[i * batch_size : (i + 1) * batch_size]]
            loss, params, opt_state = jit_make_step(
                params, opt_state, x_batch, y_batch
            )
            loss, (log_likelihood, log_prior) = loss
            losses.append(loss)
            log_likelihoods.append(log_likelihood.item())
            log_priors.append(log_prior.item())
        epoch_time = time.time() - start_time
        log_likelihood_epoch_loss = jnp.mean(jnp.array(log_likelihoods[-batch_size:]))
        epoch_loss = jnp.mean(jnp.array(losses[-batch_size:]))
        epoch_prior = jnp.mean(jnp.array(log_priors[-batch_size:]))
        print(
            f"epoch={epoch}, log likelihood loss={log_likelihood_epoch_loss:.2f}, loss ={epoch_loss:.2f}, prior loss={epoch_prior:.2f}, time={epoch_time:.3f}s"
        )

    # Plot the results
    
    plt.plot(X_train, Y_train, 'o')
    plt.plot(X_val, Y_val, label="True function")
    plt.plot(X_val, model.apply(params, X_val), label="Predictions")
    plt.legend()
    plt.savefig("figures/trained_inbetween_data.pdf")


    # Save Learned parameters
    train_stats_dict = {}
    train_stats_dict['x_train'] = X_train
    train_stats_dict['y_train'] = Y_train
    train_stats_dict['x_val'] = X_val
    train_stats_dict['y_val'] = Y_val
    train_stats_dict['model'] = model
    train_stats_dict['n_params'] = D

    with open(f"./checkpoints/syntetic_regression.pickle", "wb") as file:
        pickle.dump(
            {"args": args, "params": params, "alpha": alpha, "rho": rho, "train_stats": train_stats_dict}, file
        )
