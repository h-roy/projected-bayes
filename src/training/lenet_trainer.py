from collections import defaultdict
import jax
import jax.numpy as jnp
from tqdm import tqdm
import tree_math as tm
from optax import softmax_cross_entropy
from .base_trainer import TrainerModule
import jax.random as random
import logging
from src.helper import compute_num_params

class LeNet_Trainer(TrainerModule):

    def init_model(self, exmp_imgs):
        # Initialize model
        self.rng, init_rng, dropout_init_rng = random.split(self.rng, 3)
        self.init_params = self.model.init(init_rng,
                                           exmp_imgs)        
        self.init_batch_stats = None
        self.state = None
        self.num_params = compute_num_params(self.init_params)#compute_num_params(self.init_params[0])
        logging.info(f"Number of trainable parameters network: {self.num_params}")
        jax.debug.print("Number of trainable parameters network: {num_params}", num_params=self.num_params)

    def create_functions(self):
            # Function to calculate the classification loss and accuracy for a model
            def calculate_loss(params, batch, train, **kwargs):
                rng = kwargs.get('rng', self.rng)
                imgs, labels = batch['image'], batch['label']
                logits = self.model.apply(params,
                                        imgs)
                model_state = None
                loss = softmax_cross_entropy(logits, labels).mean()
                acc = (logits.argmax(axis=-1) == labels.argmax(axis=-1)).mean()
                metrics = {'accuracy': acc}
                return loss, (metrics, model_state, rng)
            # Training function
            def train_step(state, batch, **kwargs):
                rng = kwargs.get('rng', self.rng)
                loss_fn = lambda params: calculate_loss(params, batch, train=True, rng=rng)
                # Get loss, gradients for loss, and other outputs of loss function
                ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
                loss, metrics, _, rng = ret[0], *ret[1]
                # Update parameters and batch statistics
                state = state.apply_gradients(grads=grads)
                return state, rng, loss, metrics
            # Eval function
            def eval_step(state, batch, **kwargs):
                rng = kwargs.get('rng', self.rng)
                # Return the accuracy for a single batch
                _, (metrics, new_model_state, rng) = calculate_loss(state.params, batch, train=False, rng=rng)
                return metrics, new_model_state, rng
            # jit for efficiency
            self.train_step = jax.jit(train_step)
            self.eval_step = jax.jit(eval_step)

