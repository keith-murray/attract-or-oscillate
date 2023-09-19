import jax
import jax.numpy as jnp
from jax import random
from flax.training import train_state
from flax import struct
from flax import serialization
import optax
from clu import metrics
import pandas as pd

@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')
    accuracy: metrics.Average.from_output('accuracy')

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
def compute_custom_accuracy(output, label):
    condition_positive = output > jnp.float32(0.5)
    condition_negative = output < jnp.float32(-0.5)

    output_clipped = jnp.where(condition_positive, jnp.float32(1.0), jnp.where(condition_negative, jnp.float32(-1.0), jnp.float32(0.0)))
    matched = output_clipped == label
    accuracy = jnp.mean(matched)

    return accuracy

@jax.jit
def compute_metrics(state, batch, subkey):
    output, rates = state.apply_fn({'params': state.params}, batch[0], init_key=subkey)
    loss_SET = optax.squared_error(output[:, -5:, :], batch[1]).mean()
    loss_rates = jnp.float32(0.0001) * optax.squared_error(rates).mean() * jnp.float32(0.01)
    loss = loss_SET + loss_rates
    accuracy = compute_custom_accuracy(output[:,-1,-1], batch[1][:,-1,-1])
    metric_updates = state.metrics.single_from_model_output(loss=loss, accuracy=accuracy)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state

class MetricsHistory:
    def __init__(self, metrics):
        self.history = {metric: [] for metric in metrics}
        
    def append(self, metric, value):
        self.history[metric].append(value)

    def fill_none(self, metric, epochs):
        self.history[metric] = [None for _ in range(epochs)]
        
    def save_to_csv(self, save_loc):
        df = pd.DataFrame(self.history)
        df.to_csv(save_loc, index=False)

    def load_from_csv(self, save_loc):
        loaded_df = pd.read_csv(save_loc)
        self.history = loaded_df.to_dict(orient='list')
    
    def print_latest_metrics(self):
        for metric, values in self.history.items():
            latest_value = values[-1] if values else "N/A"
            print(f"{metric}: {latest_value}")
        print("\n")

class ModelParameters:
    def __init__(self, state):
        self.params = {'params': state.params}
    
    def serialize(self, save_loc):
        bytes_output = serialization.to_bytes(self.params)
        with open(save_loc, 'wb') as f:
            f.write(bytes_output)
    
    def deserialize(self, save_loc):
        with open(save_loc, 'rb') as f:
            bytes_output = f.read()
        self.params = serialization.from_bytes(self.params, bytes_output)

def compute_metrics_and_update_history(subkey, state, batch, metric_prefix, metrics_history):
    """
    Compute metrics for a given dataset batch and update the metrics history.

    Parameters:
        state: The model state.
        batch: The dataset batch.
        subkey: The random subkey.
        metric_prefix: A string prefix to use for the metrics (e.g., 'train', 'test').
        metrics_history: An instance of MetricsHistory.
    """
    if batch is not None:
        new_state = compute_metrics(state, batch, subkey)
        for metric, value in new_state.metrics.compute().items():
            metrics_history.append(f'{metric_prefix}_{metric}', value)

def train_model(key, state, train_ds, test_ds, grok_ds, corrupt_ds, epochs,):
    """
    Trains a model using provided datasets and records various performance metrics.

    Parameters:
        key (random.PRNGKey): The JAX random key for stochastic operations.
        state (object): The initial state of the model, including parameters.
        train_ds (Dataset): The dataset for training the model.
        test_ds (Dataset): The dataset for testing the model.
        grok_ds (Dataset): The dataset for additional evaluation of the model (grok).
        corrupt_ds (Dataset): The dataset for evaluating the model's robustness (corrupt).
        epochs (int): The number of training epochs.

    Returns:
        tuple: 
    """
    metrics_history = MetricsHistory([
        'train_loss', 'train_accuracy',
        'test_loss', 'test_accuracy',
        'grok_loss', 'grok_accuracy',
        'corrupt_loss', 'corrupt_accuracy',
    ])
    
    test_batch = list(test_ds.as_numpy_iterator())[0]
    if grok_ds is not None:
        grok_batch = list(grok_ds.as_numpy_iterator())[0]
    else:
        grok_batch = None
        metrics_history.fill_none('grok_loss', epochs)
        metrics_history.fill_none('grok_accuracy', epochs)
    if corrupt_ds is not None:
        corrupt_batch = list(corrupt_ds.as_numpy_iterator())[0]
    else:
        corrupt_batch = None
        metrics_history.fill_none('corrupt_loss', epochs)
        metrics_history.fill_none('corrupt_accuracy', epochs)

    for epoch in range(epochs):

        for step, batch in enumerate(train_ds.as_numpy_iterator()):
            key, subkey = random.split(key)
            state = train_step(state, batch, subkey,)
            state = compute_metrics(state, batch, subkey,)

        for metric, value in state.metrics.compute().items():
            metrics_history[f'train_{metric}'].append(value)
        state = state.replace(metrics=state.metrics.empty())

        key, subkey = random.split(key)
        compute_metrics_and_update_history(subkey, state, test_batch, 'test', metrics_history)
        
        key, subkey = random.split(key)
        compute_metrics_and_update_history(subkey, state, grok_batch, 'grok', metrics_history)

        key, subkey = random.split(key)
        compute_metrics_and_update_history(subkey, state, corrupt_batch, 'corrupt', metrics_history)

        if (epoch+1) % 50 == 0:
            print(f'Metrics after epoch {epoch+1}:')
            print_latest_metrics(metrics_history)
    
    model_params = ModelParameters(state)
    
    return model_params, metrics_history