import jax
import jax.numpy as jnp
from jax import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def retrieve_outputs_and_rates(key, model, params, dataset):
    """
    Retrieves outputs, rates, and labels from a dataset using a specified model.

    Parameters:
        key (jax.random.PRNGKey): A random number generator key.
        model: The model to use for prediction.
        params: Model parameters.
        dataset: The dataset over which the function iterates to get outputs and rates.

    Returns:
        outputs_tensor (jnp.ndarray): Tensor containing the model's output values.
        rates_tensor (jnp.ndarray): Tensor containing the rate values.
        labels_tensor (jnp.ndarray): Tensor containing the label values.
    """
    outputs = []
    rates = []
    labels = []

    for step, batch in enumerate(dataset):
        key, subkey = random.split(key)
        output, rate = model.apply(params, batch[0], init_key=subkey)

        outputs.append(output[:, :, -1])
        rates.append(rate)
        labels.append(batch[1][:, -1, -1])

    outputs_tensor = jnp.stack(outputs, axis=0)
    rates_tensor = jnp.stack(rates, axis=0)
    labels_tensor = jnp.stack(labels, axis=0)

    return outputs_tensor, rates_tensor, labels_tensor

def create_metrics_plot(plot_axis, metrics_history, metric_type):
    """
    Create a plot of metrics on a given axis.

    Parameters:
        plot_axis (matplotlib.axes.Axes): The axis on which to plot the data.
        metrics_history (dict): A dictionary containing historical metric data.
        metric_type (str): The type of metric to plot ('accuracy' or 'loss').
    """
    for key, values in metrics_history.items():
        if metric_type.lower() in key.lower():
            plot_axis.plot(values, label=key)

    plot_axis.set_title(f'{metric_type} Metrics Over Time')
    plot_axis.set_xlabel('Epochs')
    plot_axis.set_ylabel(metric_type)
    plot_axis.legend()

def plot_trials_with_labels(plot_axis, outputs_tensor, labels_tensor):
    """
    Plots output for each trial with colors indicating the labels.

    Parameters:
        plot_axis: The axis on which to plot.
        outputs_tensor (numpy.ndarray or jnp.ndarray): Shape should be (trials, time).
        labels_tensor (numpy.ndarray or jnp.ndarray): Shape should be (trials,).
    """
    if hasattr(outputs_tensor, 'block_until_ready'):
        outputs_tensor = np.array(outputs_tensor)
    if hasattr(labels_tensor, 'block_until_ready'):
        labels_tensor = np.array(labels_tensor)

    for i, (output, label) in enumerate(zip(outputs_tensor, labels_tensor)):
        color = 'g' if label == 1.0 else 'r'
        plot_axis.plot(output, color=color,)
        
    plot_axis.set_xlabel('Time')
    plot_axis.set_ylabel('Output')
    plot_axis.set_title('Outputs for Each Trial')

    handles, labels = plot_axis.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plot_axis.legend(by_label.values(), by_label.keys())

def plot_2D_PCA(plot_axis, training_rates, testing_rates, testing_labels):
    pass