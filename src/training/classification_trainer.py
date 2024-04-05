import jax
import jax.numpy as jnp
import tree_math as tm
from optax import softmax_cross_entropy
from .base_trainer import TrainerModule

class Classification_Trainer(TrainerModule):
    def create_functions(self):
        # Function to calculate the classification loss and accuracy for a model
        def calculate_loss(params, batch, train, **kwargs):
            batch_stats = kwargs.get('batch_stats', None)
            rng = kwargs.get('rng', self.rng)
            imgs, labels = batch['image'], batch['label']
            # Run model. During training, we need to update the BatchNorm statistics.
            outs = self.model.apply({'params': params, 'batch_stats': batch_stats}, 
                                    imgs,
                                    train=train,
                                    mutable=['batch_stats'] if train else False)
            logits, new_model_state = outs if train else (outs, None)
            loss = softmax_cross_entropy(logits, labels).mean()

            acc = (logits.argmax(axis=-1) == labels.argmax(axis=-1)).mean()
            metrics = {'accuracy': acc}
            return loss, (metrics, new_model_state, rng)
        # Training function
        def train_step(state, batch, **kwargs):
            rng = kwargs.get('rng', self.rng)
            loss_fn = lambda params: calculate_loss(params, batch, train=True, batch_stats=state.batch_stats)
            # Get loss, gradients for loss, and other outputs of loss function
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)  
            loss, metrics, new_model_state, rng = ret[0], *ret[1]
            # Update parameters and batch statistics
            state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
            return state, rng, loss, metrics
        # Eval function
        def eval_step(state, batch, **kwargs):
            # Return the accuracy for a single batch
            _, (metrics, new_model_state, rng) = calculate_loss(state.params, batch, train=False, batch_stats=state.batch_stats)
            return metrics, new_model_state, rng
        # jit for efficiency
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)