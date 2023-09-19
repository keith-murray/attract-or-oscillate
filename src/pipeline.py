import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from src.task import SETDataset
from src.model import EulerCTRNNCell
from src.training import create_train_state, train_model
from src.analysis import generate_summary_plot

import json
import os
import sys

task_id = sys.argv[1] # Here is the $LLSUB_RANK slurm argument
experiment_folder = sys.argv[2] # Here is a non-slurm argument, this folder is the same across the entire job
task_folder = os.path.join(experiment_folder, f"task_{task_id}")

json_path = os.path.join(task_folder, "params.json")
with open(json_path, 'r') as f:
    json_params = json.load(f)

seed = json_params['seed']
alpha = json_params['alpha']
grok = json_params['grok']
corrupt = json_params['corrupt']

key = random.PRNGKey(seed) # json_param seed

key, subkey = random.split(key)
set_dataset = SETDataset(subkey, 15, 5, 108)
set_dataset.grok_SET(grok) # json_param grok
set_dataset.corrupt_SET(corrupt) # json_param corrupt
set_dataset.print_training_testing()
training_tf_dataset, testing_tf_dataset, grok_tf_dataset, corrupt_tf_dataset = set_dataset.tf_datasets()

features = 100
alpha = jnp.float32(alpha) # json_param alpha
noise = jnp.float32(0.1)

ctrnn = nn.RNN(EulerCTRNNCell(features=features, alpha=alpha, noise=noise,))

lr = 0.0001
epochs = 2500

key, subkey = random.split(key)
state = create_train_state(ctrnn, subkey, lr,)

key, subkey = random.split(key)
model_params, metrics_history = train_model(
    subkey, 
    state, 
    training_tf_dataset, 
    testing_tf_dataset, 
    grok_tf_dataset, 
    corrupt_tf_dataset, 
    epochs,
)

key, subkey = random.split(key)
generate_summary_plot(
    subkey, 
    ctrnn, 
    model_params.params, 
    metrics_history.history, 
    training_tf_dataset, 
    testing_tf_dataset, 
    os.path.join(task_folder, 'summary_plot.jpg')
)

model_params.serialize(os.path.join(task_folder, 'params.bin'))

metrics_history.save_to_csv(os.path.join(task_folder, 'metrics_history.csv'))