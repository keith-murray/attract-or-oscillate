import jax
import jax.numpy as jnp
from jax import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def retrieve_outputs_and_rates(key, model, params, dataset,):
    """
    Retrieves outputs, rates, and labels from a dataset using a specified model.

    Parameters:
        key (jax.random.PRNGKey): A random number generator key.
        model: The model to use for prediction.
        params: Model parameters.
        dataset: The dataset over which the function iterates to get outputs and rates.

    Returns:
        outputs_tensor (numpy.ndarray): Tensor containing the model's output values.
        rates_tensor (numpy.ndarray): Tensor containing the rate values.
        labels_tensor (numpy.ndarray): Tensor containing the label values.
    """
    outputs = []
    rates = []
    labels = []

    for step, batch in enumerate(dataset.as_numpy_iterator()):
        key, subkey = random.split(key)
        output, rate = model.apply(params, batch[0], init_key=subkey)

        outputs.append(output[:, :, -1])
        rates.append(rate)
        labels.append(batch[1][:, -1, -1])

    outputs_tensor = np.array(jnp.concatenate(outputs, axis=0))
    rates_tensor = np.array(jnp.concatenate(rates, axis=0))
    labels_tensor = np.array(jnp.concatenate(labels, axis=0))

    return outputs_tensor, rates_tensor, labels_tensor

def create_metrics_plot(plot_axis, metrics_history, metric_type,):
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

    plot_axis.set_title(f'{metric_type} Metrics Over Epochs')
    plot_axis.set_xlabel('Epochs')
    plot_axis.set_ylabel(metric_type)
    plot_axis.legend()

def plot_trials_with_labels(plot_axis, outputs_tensor, labels_tensor,):
    """
    Plots output for each trial with colors indicating the labels.

    Parameters:
        plot_axis: The axis on which to plot.
        outputs_tensor (numpy.ndarray): Shape should be (trials, time).
        labels_tensor (numpy.ndarray): Shape should be (trials,).
    """
    for i, (output, label) in enumerate(zip(outputs_tensor, labels_tensor)):
        color = 'g' if label == 1.0 else 'r'
        plot_axis.plot(output, color=color,)
        
    plot_axis.set_xlabel('Time')
    plot_axis.set_ylabel('Output')
    plot_axis.set_title('Outputs for Each Trial')

def plot_2D_PCA(plot_axis, training_rates, testing_rates, testing_labels):
    """
    Plots 2D PCA trajectories with endpoints labeled according to the labels.

    Parameters:
        plot_axis (matplotlib.axes.Axes): The axis on which to plot.
        training_rates (numpy.ndarray): Array of shape (batch, time, features) for training data.
        testing_rates (numpy.ndarray): Array of shape (batch, time, features) for testing data.
        testing_labels (numpy.ndarray): Array of shape (batch,), labels for the testing data.

    """
    # Reshape training and testing data to fit PCA
    n_training_samples, time_steps, n_features = training_rates.shape
    training_data = training_rates.reshape(-1, n_features)
    testing_data = testing_rates.reshape(-1, n_features)

    # Fit PCA using training data
    pca = PCA(n_components=3)
    pca.fit(training_data)

    # Transform both training and testing data
    transformed_testing = pca.transform(testing_data)
    transformed_testing = transformed_testing.reshape(-1, time_steps, 3)

    # Create a new colormap (copper scale)
    cmap = plt.cm.copper
    
    # Define start and end times
    start_time = 0.00  # in seconds
    end_time = 0.50  # in seconds
    
    # Plotting
    for i, (trajectory, label) in enumerate(zip(transformed_testing, testing_labels)):
        n_points = trajectory.shape[0]
        time_values = np.linspace(start_time, end_time, n_points)
        colors = cmap((time_values - start_time) / (end_time - start_time))
        
        # Plot the PCA trajectory with a color gradient
        for i in range(n_points - 1):
            plot_axis.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], color=colors[i])
        
        # Color the endpoint based on label
        endpoint_color = 'g' if label == 1.0 else 'r'
        plot_axis.scatter(trajectory[-1, 0], trajectory[-1, 1], color=endpoint_color)
        
    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    plot_axis.set_title(f'2D PCA of Testing Rates\nExplained Variance: {explained_var:.2f}%')
    plot_axis.set_xlabel('Principal Component 1')
    plot_axis.set_ylabel('Principal Component 2')
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=start_time, vmax=end_time))
    sm.set_array([])
    cbar = plot_axis.figure.colorbar(sm, ax=plot_axis, orientation='vertical', pad=0)
    cbar.set_label('Time (s)')

def generate_summary_plot(key, model, params, metrics_history, training_dataset, testing_dataset, save_loc):
    """
    Generates a summary plot with four panels.
    
    Parameters:
        key (jax.random.PRNGKey): A random number generator key.
        model: The model to use for prediction.
        params: Model parameters.
        metrics_history (dict): A dictionary containing historical metric data.
        training_dataset: The dataset for training.
        testing_dataset: The dataset for testing.
        save_loc (str): The save location and name for the figure.
    """
    training_outputs, training_rates, training_labels = retrieve_outputs_and_rates(key, model, params, training_dataset)
    testing_outputs, testing_rates, testing_labels = retrieve_outputs_and_rates(key, model, params, testing_dataset)

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    # Top-left: Plot losses for all metrics
    create_metrics_plot(axs[0, 0], metrics_history, 'loss')
    
    # Top-right: Plot accuracy for all metrics
    create_metrics_plot(axs[0, 1], metrics_history, 'accuracy')
    
    # Bottom-left: Plot the outputs of the testing trial
    plot_trials_with_labels(axs[1, 0], testing_outputs, testing_labels)
    
    # Bottom-right: Plot the 2D PCA
    plot_2D_PCA(axs[1, 1], training_rates, testing_rates, testing_labels)
    
    plt.tight_layout()
    plt.savefig(save_loc)
    plt.show()