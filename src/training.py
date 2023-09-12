import jax
import jax.numpy as jnp
from jax import random
from flax.training import train_state
from flax import struct
import optax
from clu import metrics

@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')

class TrainState(train_state.TrainState):
    metrics: Metrics

def create_train_state(module, subkey, learning_rate,):
    """Creates an initial `TrainState`."""
    params = module.init(subkey, jnp.ones([1, 50, 100]))['params']
    tx = optax.adamw(learning_rate,)
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        metrics=Metrics.empty()
    )

@jax.jit
def train_step(state, batch, subkey):
    """Train for a single step."""
    def loss_fn(params):
        output, rates = state.apply_fn({'params': params}, batch[0], init_key=subkey)
        loss_SET = optax.squared_error(output[:, -5:, :], batch[1]).mean()
        loss_rates = jnp.float32(0.0001) * optax.squared_error(rates).mean() * jnp.float32(0.01)
        return loss_SET + loss_rates
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state

@jax.jit
def compute_metrics(state, batch, subkey):
    pass # TODO: Implement