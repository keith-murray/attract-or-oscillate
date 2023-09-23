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

seed = json_params.get('seed', 0)
alpha = json_params.get('alpha', 0.1)
grok = json_params.get('grok', 0)
grok_label = json_params.get('grok_label', None)
corrupt = json_params.get('corrupt', 0)
min_samples = json_params.get('min_samples', 30)
test_samples = json_params.get('test_samples', 5)
batch_size = json_params.get('batch_size', 108)
lr = json_params.get('lr', 0.001)
norm_clip = json_params.get('norm_clip', 1.0)
epochs = json_params.get('epochs', 500)

key = random.PRNGKey(seed)

key, subkey = random.split(key)
set_dataset = SETDataset(subkey, min_samples, test_samples, batch_size)
set_dataset.grok_SET(grok, grok_label)
set_dataset.corrupt_SET(corrupt)
set_dataset.print_training_testing()
training_tf_dataset, testing_tf_dataset, grok_tf_dataset, corrupt_tf_dataset = set_dataset.tf_datasets()

features = 100
alpha = jnp.float32(alpha) # json_param alpha
noise = jnp.float32(0.1)

ctrnn = nn.RNN(EulerCTRNNCell(features=features, alpha=alpha, noise=noise,))

key, subkey = random.split(key)
state = create_train_state(ctrnn, subkey, lr, norm_clip,)

key, subkey = random.split(key)
results = train_model(
    subkey, 
    state, 
    training_tf_dataset, 
    testing_tf_dataset, 
    grok_tf_dataset, 
    corrupt_tf_dataset, 
    epochs,
)

results["final_params"].serialize(os.path.join(task_folder, 'final_params.bin'))
results["min_test_loss_params"].serialize(os.path.join(task_folder, 'test_params.bin'))
results["min_grok_loss_params"].serialize(os.path.join(task_folder, 'grok_params.bin'))
results["min_corrupt_loss_params"].serialize(os.path.join(task_folder, 'corrupt_params.bin'))

results["metrics_history"].save_to_csv(os.path.join(task_folder, 'metrics_history.csv'))

key, subkey = random.split(key)
generate_summary_plot(
    subkey, 
    ctrnn, 
    results["min_test_loss_params"].params, 
    results["metrics_history"].history, 
    training_tf_dataset, 
    testing_tf_dataset, 
    os.path.join(task_folder, 'summary_plot.jpg')
)