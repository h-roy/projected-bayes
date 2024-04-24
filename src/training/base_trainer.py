import os
import datetime
from collections import defaultdict
from typing import Any
import logging

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import checkpoints
from jax.tree_util import tree_flatten, tree_unflatten
import jax.random as random
from tqdm import tqdm
import tree_math as tm

from flax.training.train_state import TrainState
from src.helper import compute_num_params
from optax import softmax_cross_entropy

import pickle
import json



class TrainState(TrainState):
    # A simple extension of TrainState to also include batch statistics
    batch_stats: Any


class TrainerModule:
    def __init__(
        self,
        model_name: str,
        model_class: nn.Module,
        model_hparams: dict,
        optimizer_name: str,
        optimizer_hparams: dict,
        exmp_imgs: Any,
        args_dict: dict,
        wandb_logger,
        seed=42,
    ):
        """
        Module for summarizing all training functionalities for classification on CIFAR10.

        Inputs:
            model_name - String of the class name, used for logging and saving
            model_class - Class implementing the neural network
            model_hparams - Hyperparameters of the model, used as input to model constructor
            optimizer_name - String of the optimizer name, supporting ['sgd', 'adam', 'adamw']
            optimizer_hparams - Hyperparameters of the optimizer, including learning rate as 'lr'
            exmp_imgs - Example imgs, used as input to initialize the model
            seed - Seed to use in the model initialization
        """

        super().__init__()
        self.model_name = model_name
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.seed = seed
        self.args = args_dict
        self.early_stopping = self.args["early_stopping"]
        self.rng = random.PRNGKey(self.args["seed"])

        # Create empty model. Note: no parameters yet
        self.model = self.model_class(**self.model_hparams)
        # now = datetime.datetime.now()
        # now_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        folder = "".join(x for x in self.args["run_name"] if x.isalnum()) #+ now_string
        # Prepare logging
        self.log_dir = os.path.join(self.args["CHECKPOINT_PATH"], folder)
        self.logger = wandb_logger
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model(exmp_imgs)

    def init_model(self, exmp_imgs):
        # Initialize model
        init_rng = jax.random.PRNGKey(self.seed)
        variables = self.model.init(init_rng, exmp_imgs)

        self.init_params = variables["params"]
        self.init_batch_stats = variables["batch_stats"] if "batch_stats" in variables.keys() else None
        self.state = None
        self.num_params = compute_num_params(self.init_params)#compute_num_params(self.init_params[0])
        logging.info(f"Number of trainable parameters network: {self.num_params}")
        jax.debug.print("Number of trainable parameters network: {num_params}", num_params=self.num_params)

    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        self.n_steps_per_epoch = num_steps_per_epoch
        # Initialize learning rate schedule and optimizer
        if self.optimizer_name.lower() == "adam":
            opt_class = optax.adam
        elif self.optimizer_name.lower() == "adamw":
            opt_class = optax.adamw
        elif self.optimizer_name.lower() == "sgd":
            opt_class = optax.sgd
        elif self.optimizer_name.lower() == "rmsprop":
            opt_class = optax.rmsprop
        else:
            assert False, f'Unknown optimizer "{self.optimizer_name.lower()}"'

        if self.args["clip_delta"] < 0:
            logging.info("No gradient clipping")
            transf = []
        else:
            logging.info(f'Gradient clipping to have max global norm of: {self.args["clip_delta"]}')
            transf = [optax.clip_by_global_norm(self.args["clip_delta"])]

        if opt_class == optax.sgd and 'weight_decay' in self.optimizer_hparams:  # wd is integrated in adamw
            transf.append(optax.add_decayed_weights(self.optimizer_hparams.pop('weight_decay')))
        
        lr = self.optimizer_hparams.pop("lr")

        if self.args["lr_scheduler"]:
            logging.info("LR scheduling")
            lr_schedule = optax.piecewise_constant_schedule(
                init_value=lr,
                boundaries_and_scales={
                    int(num_steps_per_epoch * num_epochs * 0.6): 0.5,
                    int(num_steps_per_epoch * num_epochs * 0.85): 0.5,
                },
            )

            transf.append(opt_class(lr_schedule, **self.optimizer_hparams))
        else:
            logging.info("No LR scheduling")
            transf.append(opt_class(lr, **self.optimizer_hparams))

        if self.args["scale_by_block_norm"]:
            logging.info("Scaling by parameter block norm")
            transf.append(optax.scale_by_param_block_norm())

        optimizer = optax.chain(*transf)
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.init_params if self.state is None else self.state.params,
            batch_stats=(
                self.init_batch_stats if self.state is None else self.state.batch_stats
            ),  # something here should add a batch stats None to FC and LeNet (dirty dirt)
            tx=optimizer,
        )
    def train_model(self, train_loader, val_loader, num_epochs=200, logger=None):
        # Train model for defined number of epochs
        # We first need to create optimizer and the scheduler for the given number of epochs
        self.init_optimizer(num_epochs, len(train_loader))

        # Track best eval accuracy
        best_eval = 0.0
        best_eval_epoch = 0
        for epoch_idx in tqdm(range(1, num_epochs + 1)):
            self.rng, key1, key2 = random.split(self.rng, 3)

            self.train_epoch(train_loader, epoch=epoch_idx)
            if epoch_idx % 2 == 0:
                eval_acc = self.eval_model(val_loader, epoch_idx)
                print(f"Eval acc: {eval_acc}")
                logging.info(f"Epoch {epoch_idx} eval acc: {eval_acc}")

                if eval_acc >= best_eval:
                    best_eval = eval_acc
                    best_eval_epoch = epoch_idx
                    self.save_model(step=epoch_idx)
                if (self.early_stopping > 0) and (epoch_idx - best_eval_epoch > self.early_stopping):
                    logging.info(
                        f"Eval accuracy has not improved in {self.early_stopping} epochs. \n Training stopped at best eval accuracy {best_eval} at epoch {best_eval_epoch}"
                    )

                    break

    def train_epoch(self, train_loader, epoch):
        # Train model for one epoch, and log avg loss and accuracy
        _, random_key = random.split(self.rng)
        N_total = len(train_loader) * self.args["batch_size"]
        acc = defaultdict(list)
        for i, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            self.state, self.rng, loss, metrics_dict = self.train_step(self.state, batch, rng=random_key)

            self.logger.log({"train_" + "loss" + "_batch": loss}, step=i + self.n_steps_per_epoch * epoch)
            for dict_key, dict_val in metrics_dict.items():
                self.logger.log({"train_" + dict_key + "_batch": dict_val}, step=i + self.n_steps_per_epoch * epoch)
            acc['accuracy'].append(metrics_dict['accuracy'])
        print("train Accuracy:", np.stack(jax.device_get(acc['accuracy'])).mean())

    def eval_model(self, data_loader, epoch=None, eval_type=None):
        # Test model on all images of a data loader and return avg loss
        rng, _ = random.split(self.rng)

        N_total = len(data_loader) * self.args["batch_size"]

        metrics = defaultdict(list)
        correct_class, count = 0, 0
        for batch in data_loader:
            metrics_dict, _, self.rng = self.eval_step(self.state, batch, rng=rng)
            correct_class += metrics_dict["accuracy"] * batch["image"].shape[0]
            count += batch["image"].shape[0]

            for dict_key in metrics_dict.keys():
                metrics[dict_key].append(metrics_dict[dict_key])

        eval_acc = (correct_class / count).item()
        (
            self.logger.log({"eval_" + "accuracy": eval_acc}, step=self.n_steps_per_epoch * (epoch + 1))
            if epoch is not None
            else self.logger.log(
                {f"{eval_type}_" + "accuracy": eval_acc}, step=self.n_steps_per_epoch * (self.args["n_epochs"] + 1)
            )
        )

        for dict_key in metrics:
            if dict_key == "acc":
                continue
            avg_val = np.stack(jax.device_get(metrics[dict_key])).mean()
            (
                self.logger.log(
                    {"eval_" + dict_key + "_batch_avg": avg_val}, step=self.n_steps_per_epoch * (epoch + 1)
                )
                if epoch is not None
                else self.logger.log(
                    {f"{eval_type}_" + dict_key + "_batch_avg": avg_val},
                    step=self.n_steps_per_epoch * (self.args["n_epochs"] + 1),
                )
            )

        return eval_acc

    def save_model(self, step=0):
        os.makedirs(self.log_dir, exist_ok=True)
        self.save_path = os.path.join(self.log_dir, f"{self.model_name}_{self.args['dataset']}_{self.seed}")
        state_dict = {"params": self.state.params, "batch_stats": self.state.batch_stats, "rng": self.rng}
        pickle.dump(state_dict, open(f"{self.save_path}_params.pickle", "wb"))
        with open(f"{self.save_path}_args.json", "w") as f:
            json.dump(self.args, f)
        
    def load_model(self, pretrained=False):
        state_dict = pickle.load(open(f"{self.save_path}_params.pickle", "rb"))

        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=state_dict["params"],
            batch_stats=state_dict["batch_stats"],
            tx=self.state.tx if self.state else optax.sgd(1e-6),  # Default optimizer
        )
        self.rng = state_dict["rng"]


    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this
        return os.path.isfile(os.path.join(self.args["CHECKPOINT_PATH"], f"{self.model_name}.ckpt"))