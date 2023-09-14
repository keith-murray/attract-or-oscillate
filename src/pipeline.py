import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from src.task import SETDataset
from src.model import EulerCTRNNCell
from src.training import create_train_state, train_model, serialize_parameters, save_metrics_to_csv, deserialize_parameters, load_metrics_from_csv
from src.analysis import generate_summary_plot
import os

key = random.PRNGKey(0) # Want to change this param: could be any int

key, subkey = random.split(key)
set_dataset = SETDataset(subkey, 15, 5, 5, 32)
set_dataset.grok_SET(2) # Want to change this param: 0 - 4
set_dataset.corrupt_SET(3) # Want to change this param: 0 - 4
set_dataset.print_training_testing()
training_tf_dataset, testing_tf_dataset, grok_tf_dataset, corrupt_tf_dataset = set_dataset.tf_datasets()

features = 100
alpha = jnp.float32(0.1) # Want to change this param: 0.1 or 1.0
noise = jnp.float32(0.1)

ctrnn = nn.RNN(EulerCTRNNCell(features=features, alpha=alpha, noise=noise,))

lr = 0.001
epochs = 200

key, subkey = random.split(key)
state = create_train_state(ctrnn, subkey, lr,)

key, subkey = random.split(key)
trained_state, metrics_history = train_model(
    subkey, 
    state, 
    training_tf_dataset, 
    testing_tf_dataset, 
    grok_tf_dataset, 
    corrupt_tf_dataset, 
    epochs,
)

params = {'params': trained_state.params}

save_loc = '../results/script_examples' # Want this to be a param

key, subkey = random.split(key)
generate_summary_plot(
    subkey, 
    ctrnn, 
    params, 
    metrics_history, 
    training_tf_dataset, 
    testing_tf_dataset, 
    os.path.join(save_loc, 'summary_plot.jpg')
)

serialize_parameters(params, os.path.join(save_loc, 'params.bin'))

save_metrics_to_csv(metrics_history, os.path.join(save_loc, 'metrics_history.csv'))