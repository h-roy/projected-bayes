import numpy as onp
import jax.numpy as jnp
import jax
from src.losses import nll
from src.sampling.predictive_samplers import sample_predictive

def evaluate(test_loader, posterior_samples, params, model_fn, eval_args):
    all_y_prob = []
    all_y_log_prob = []
    all_y_true = []
    all_y_var = []
    all_y_sample_probs = []
    for batch in test_loader:
        x_test, y_test = batch['image'], batch['label']
        x_test = jnp.array(x_test)
        y_test = jnp.array(y_test)
        if len(y_test.shape) == 1:
            y_test = jax.nn.one_hot(y_test, num_classes=10)

        predictive_samples = sample_predictive(
            posterior_samples=posterior_samples,
            params=params,
            model_fn=model_fn,
            x_test=x_test,
            linearised_laplace=eval_args["linearised_laplace"],
            posterior_sample_type=eval_args["posterior_sample_type"],
        )

        predictive_samples_mean = jnp.mean(predictive_samples, axis=0)
        # if eval_args["likelihood"] == "regression":
        # predictive_samples_std = jnp.std(predictive_samples, axis=0)
        # all_y_var.append(predictive_samples_std**2)


        y_prob = jnp.mean(jax.nn.softmax(predictive_samples, axis=-1), axis=0)
        y_log_prob = jnp.mean(jax.nn.log_softmax(predictive_samples, axis=-1), axis=0)

        # y_prob = jax.nn.softmax(predictive_samples_mean, axis=-1)
        # y_log_prob = jax.nn.log_softmax(predictive_samples_mean, axis=-1)


        predictive_samples_std = jnp.std(jax.nn.softmax(predictive_samples, axis=-1), axis=0)
        all_y_var.append(predictive_samples_std**2)
        all_y_sample_probs.append(jax.nn.softmax(predictive_samples, axis=-1))


        # import pdb; pdb.set_trace()
        all_y_prob.append(y_prob)
        all_y_log_prob.append(y_log_prob)
        all_y_true.append(y_test)

    all_y_prob = jnp.concatenate(all_y_prob, axis=0)
    all_y_log_prob = jnp.concatenate(all_y_log_prob, axis=0)
    all_y_true = jnp.concatenate(all_y_true, axis=0)
    if all_y_true.shape[-1] != all_y_log_prob.shape[-1]:
        all_y_true = all_y_true[... , :all_y_log_prob.shape[-1]]

    # compute some metrics: mean confidence, accuracy and negative log-likelihood
    metrics = {}
    if eval_args["likelihood"] == "classification":
        all_y_var = jnp.concatenate(all_y_var, axis=0)
        all_y_sample_probs = jnp.concatenate(all_y_sample_probs, axis=1)
        metrics["conf"] = (jnp.max(all_y_prob, axis=1)).mean().item()
        metrics["nll"] = (-jnp.mean(jnp.sum(all_y_log_prob * all_y_true, axis=-1), axis=-1)).mean()
        metrics["acc"] =  (jnp.argmax(all_y_prob, axis=1)==jnp.argmax(all_y_true, axis=1)).mean().item()
    elif eval_args["likelihood"] == "regression":
        sigma_noise = 1  # TODO: define sigma_noise!
        all_y_var = jnp.concatenate(all_y_var, axis=0) + sigma_noise**2
        all_y_sample_probs = None
        metrics["nll"] = nll(all_y_prob, all_y_true, all_y_var)
    else:
        raise ValueError("Unknown likelihood.")

    # cast to numpy
    all_y_prob = onp.array(all_y_prob)
    

    return metrics, all_y_prob, all_y_true, all_y_var, all_y_sample_probs



def evaluate_map(test_loader, params, model_fn, eval_args):
    all_y_prob = []
    all_y_log_prob = []
    all_y_true = []
    all_y_var = []
    for batch in test_loader:
        x_test, y_test = batch['image'], batch['label']
        x_test = jnp.array(x_test)
        y_test = jnp.array(y_test)
        if len(y_test.shape) == 1:
            y_test = jax.nn.one_hot(y_test, num_classes=10)

        predictive_samples = model_fn(params, x_test)
        # predictive_samples_mean = jnp.mean(predictive_samples, axis=0)
        if eval_args["likelihood"] == "regression":
            predictive_samples_std = jnp.std(predictive_samples, axis=0)
            all_y_var.append(predictive_samples_std**2)

        y_prob = jax.nn.softmax(predictive_samples, axis=-1)
        y_log_prob = jax.nn.log_softmax(predictive_samples, axis=-1)

        # import pdb; pdb.set_trace()
        all_y_prob.append(y_prob)
        all_y_log_prob.append(y_log_prob)
        all_y_true.append(y_test)

    all_y_prob = jnp.concatenate(all_y_prob, axis=0)
    all_y_log_prob = jnp.concatenate(all_y_log_prob, axis=0)
    all_y_true = jnp.concatenate(all_y_true, axis=0)
    # compute some metrics: mean confidence, accuracy and negative log-likelihood
    if all_y_true.shape[-1] != all_y_log_prob.shape[-1]:
        all_y_true = all_y_true[... , :all_y_log_prob.shape[-1]]
        
    metrics = {}
    if eval_args["likelihood"] == "classification":
        all_y_var = None
        all_y_sample_probs = None
        metrics["conf"] = (jnp.max(all_y_prob, axis=1)).mean().item()
        metrics["nll"] = (-jnp.mean(jnp.sum(all_y_log_prob * all_y_true, axis=-1), axis=-1)).mean()
        metrics["acc"] =  (jnp.argmax(all_y_prob, axis=1)==jnp.argmax(all_y_true, axis=1)).mean().item()
    elif eval_args["likelihood"] == "regression":
        sigma_noise = 1  # TODO: define sigma_noise!
        all_y_sample_probs = None
        all_y_var = jnp.concatenate(all_y_var, axis=0) + sigma_noise**2
        metrics["nll"] = nll(all_y_prob, all_y_true, all_y_var)
    else:
        raise ValueError("Unknown likelihood.")

    # cast to numpy
    all_y_prob = onp.array(all_y_prob)
    

    return metrics, all_y_prob, all_y_true, all_y_var, all_y_sample_probs

def evaluate_samples(test_loader, posterior_samples, params, model_fn, eval_args):
    all_y_prob = []
    all_y_log_prob = []
    all_y_true = []
    all_y_var = []
    for x_test, y_test in test_loader:
        x_test = jnp.array(x_test.numpy())
        y_test = jnp.array(y_test.numpy())
        if len(y_test.shape) == 1:
            y_test = jax.nn.one_hot(y_test, num_classes=10)

        predictive_samples = sample_predictive(
            posterior_samples=posterior_samples,
            params=params,
            model_fn=model_fn,
            x_test=x_test,
            linearised_laplace=eval_args["linearised_laplace"],
            posterior_sample_type=eval_args["posterior_sample_type"],
        )
        # predictive_samples_mean = jnp.mean(predictive_samples, axis=0)
        if eval_args["likelihood"] == "regression":
            predictive_samples_std = jnp.std(predictive_samples, axis=0)
            all_y_var.append(predictive_samples_std**2)

        y_prob = jax.nn.softmax(predictive_samples, axis=-1)
        y_log_prob = jax.nn.log_softmax(predictive_samples, axis=-1)

        # import pdb; pdb.set_trace()
        all_y_prob.append(y_prob)
        all_y_log_prob.append(y_log_prob)
        all_y_true.append(y_test)
    all_y_prob = jnp.concatenate(all_y_prob, axis=1)
    all_y_log_prob = jnp.concatenate(all_y_log_prob, axis=1)
    all_y_true = jnp.concatenate(all_y_true, axis=0)


    # compute some metrics: mean confidence, accuracy and negative log-likelihood
    metrics = {}
    if eval_args["likelihood"] == "classification":
        all_y_var = None
        metrics["conf"] = list(map(lambda x: (jnp.max(x, axis=-1)).mean().item(), all_y_prob))
        metrics["nll"] = list(map(lambda x: (-jnp.sum(jnp.sum(x * all_y_true, axis=-1), axis=-1)).mean(), all_y_log_prob))
        metrics["acc"] =  list(map(lambda x: (jnp.argmax(x, axis=1)==jnp.argmax(all_y_true, axis=1)).mean().item(), all_y_prob))
    elif eval_args["likelihood"] == "regression":
        sigma_noise = 1  # TODO: define sigma_noise!
        all_y_var = jnp.concatenate(all_y_var, axis=0) + sigma_noise**2
        metrics["nll"] = nll(all_y_prob, all_y_true, all_y_var)
    else:
        raise ValueError("Unknown likelihood.")

    # cast to numpy
    all_y_prob = onp.array(all_y_prob)
    

    return metrics, all_y_prob, all_y_true, all_y_var


def evaluate_ensembles(test_loader, params_1, params_2, params_3, model_fn_1, model_fn_2, model_fn_3, eval_args):
    all_y_prob = []
    all_y_log_prob = []
    all_y_true = []
    all_y_var = []
    for batch in test_loader:
        x_test, y_test = batch['image'], batch['label']
        x_test = jnp.array(x_test)
        y_test = jnp.array(y_test)
        if len(y_test.shape) == 1:
            y_test = jax.nn.one_hot(y_test, num_classes=10)

        predictive_samples_1 = model_fn_1(params_1, x_test)
        predictive_samples_2 = model_fn_2(params_2, x_test)
        predictive_samples_3 = model_fn_3(params_3, x_test)

        predictive_samples = 1/3 * (predictive_samples_1 + predictive_samples_2 + predictive_samples_3)
        concat_samples = jnp.stack([predictive_samples_1, predictive_samples_2, predictive_samples_3])
        predictive_samples_std = jnp.std(jax.nn.softmax(concat_samples, axis=-1), axis=0)
        all_y_var.append(predictive_samples_std**2)

        # predictive_samples_mean = jnp.mean(predictive_samples, axis=0)
        if eval_args["likelihood"] == "regression":
            predictive_samples_std = jnp.std(predictive_samples, axis=0)
            all_y_var.append(predictive_samples_std**2)

        y_prob = jax.nn.softmax(predictive_samples, axis=-1)
        y_log_prob = jax.nn.log_softmax(predictive_samples, axis=-1)

        # import pdb; pdb.set_trace()
        all_y_prob.append(y_prob)
        all_y_log_prob.append(y_log_prob)
        all_y_true.append(y_test)

    all_y_prob = jnp.concatenate(all_y_prob, axis=0)
    all_y_log_prob = jnp.concatenate(all_y_log_prob, axis=0)
    all_y_true = jnp.concatenate(all_y_true, axis=0)
    # compute some metrics: mean confidence, accuracy and negative log-likelihood
    if all_y_true.shape[-1] != all_y_log_prob.shape[-1]:
        all_y_true = all_y_true[... , :all_y_log_prob.shape[-1]]
        
    metrics = {}
    if eval_args["likelihood"] == "classification":
        all_y_sample_probs = None
        all_y_var = jnp.concatenate(all_y_var, axis=0)
        metrics["conf"] = (jnp.max(all_y_prob, axis=1)).mean().item()
        metrics["nll"] = (-jnp.mean(jnp.sum(all_y_log_prob * all_y_true, axis=-1), axis=-1)).mean()
        metrics["acc"] =  (jnp.argmax(all_y_prob, axis=1)==jnp.argmax(all_y_true, axis=1)).mean().item()
    elif eval_args["likelihood"] == "regression":
        sigma_noise = 1  # TODO: define sigma_noise!
        all_y_sample_probs = None
        all_y_var = jnp.concatenate(all_y_var, axis=0) + sigma_noise**2
        metrics["nll"] = nll(all_y_prob, all_y_true, all_y_var)
    else:
        raise ValueError("Unknown likelihood.")

    # cast to numpy
    all_y_prob = onp.array(all_y_prob)
    

    return metrics, all_y_prob, all_y_true, all_y_var, all_y_sample_probs
