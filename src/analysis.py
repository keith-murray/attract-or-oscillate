import jax
import jax.numpy as jnp
from jax import random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
            plot_axis.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], color=colors[i], zorder=1)

        # Color the endpoint based on label
        endpoint_color = 'g' if label == 1.0 else 'r'
        plot_axis.scatter(trajectory[-1, 0], trajectory[-1, 1], color=endpoint_color, zorder=2)
        
    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    plot_axis.set_title(f'2D PCA of Testing Rates\nExplained Variance: {explained_var:.2f}%')
    plot_axis.set_xlabel('Principal Component 1')
    plot_axis.set_ylabel('Principal Component 2')
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=start_time, vmax=end_time))
    sm.set_array([])
    cbar = plot_axis.figure.colorbar(sm, ax=plot_axis, orientation='vertical', pad=0)
    cbar.set_label('Time (s)')

def plot_2D_PCA_oscillatory(plot_axis, training_rates, testing_rates, null_rates, testing_labels):
    """
    Plots 2D PCA trajectories with endpoints labeled according to the labels.

    Parameters:
        plot_axis (matplotlib.axes.Axes): The axis on which to plot.
        training_rates (numpy.ndarray): Array of shape (batch, time, features) for training data.
        testing_rates (numpy.ndarray): Array of shape (batch, time, features) for testing data.
        null_rates (numpy.ndarray): Array of shape (batch, time, features) for null data.
        testing_labels (numpy.ndarray): Array of shape (batch,), labels for the testing data.

    """
    # Reshape training and testing data to fit PCA
    n_training_samples, time_steps, n_features = training_rates.shape
    training_data = training_rates.reshape(-1, n_features)
    testing_data = testing_rates.reshape(-1, n_features)
    null_data = null_rates.reshape(-1, n_features)

    # Fit PCA using training data
    pca = PCA(n_components=3)
    pca.fit(training_data)

    # Transform both training and testing data
    transformed_testing = pca.transform(testing_data)
    transformed_testing = transformed_testing.reshape(-1, time_steps, 3)

    transformed_null = pca.transform(null_data)
    transformed_null = transformed_null.reshape(-1, time_steps, 3)

    # Create a new colormap (copper scale)
    cmap = plt.cm.copper
    
    # Define start and end times
    start_time = 0.00  # in seconds
    end_time = 0.50  # in seconds
    
    # Plotting
    n_points = transformed_null.shape[1]
    time_values = np.linspace(start_time, end_time, n_points)
    colors = cmap((time_values - start_time) / (end_time - start_time))
    
    # Plot the PCA trajectory with a color gradient
    for i in range(n_points - 1):
        plot_axis.plot(transformed_null[0,i:i+2, 0], transformed_null[0,i:i+2, 1], color=colors[i], zorder=1)
    
    for i, (trajectory, label) in enumerate(zip(transformed_testing, testing_labels)):
        # Color the endpoint based on label
        endpoint_color = 'g' if label == 1.0 else 'r'
        plot_axis.scatter(trajectory[-1, 0], trajectory[-1, 1], color=endpoint_color, zorder=2)
        
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
    key, subkey = random.split(key)
    training_outputs, training_rates, training_labels = retrieve_outputs_and_rates(
        subkey, 
        model, 
        params, 
        training_dataset
    )
    key, subkey = random.split(key)
    testing_outputs, testing_rates, testing_labels = retrieve_outputs_and_rates(
        subkey, 
        model, 
        params, 
        testing_dataset
    )

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

def color_set_axis(ax, colors, SET_input, time):
    zero_indices = jnp.where(SET_input == 0)[0]
    pairs = []
    start_idx = zero_indices[0]
    for i in range(1, len(zero_indices)):
        if SET_input[i] != 0 and SET_input[i] != SET_input[i-1]:
            pairs.append((time[i-1], time[i+1]))
    
    for i, (start, end) in enumerate(pairs):
        ax.axvspan(start, end, facecolor=colors[i], alpha=0.5)

def generate_dynamics_plot(key, model, params, training_dataset, testing_dataset, dataset, dynamics_type, save_loc):
    """
    Generates a plot that elegantly displays the dynamics of the model.
    """
    key, subkey = random.split(key)
    training_outputs, training_rates, training_labels = retrieve_outputs_and_rates(
        subkey, 
        model, 
        params, 
        training_dataset
    )
    key, subkey = random.split(key)
    testing_outputs, testing_rates, testing_labels = retrieve_outputs_and_rates(
        subkey, 
        model, 
        params, 
        testing_dataset
    )

    gpr = dataset.create_trial('GPR', 1)
    key, subkey = random.split(key)
    output_gpr, rate_gpr = model.apply(params, jnp.expand_dims(gpr[0], 0), init_key=subkey)
    
    prp = dataset.create_trial('PGP', 2)
    key, subkey = random.split(key)
    output_prp, rate_prp = model.apply(params, jnp.expand_dims(prp[0], 0), init_key=subkey)
    
    key, subkey = random.split(key)
    output_null, rate_null = model.apply(params, jnp.zeros((1,50,100)), init_key=subkey)
    
    graphing_inputs = [gpr[0][:,0], prp[0][:,0]]
    graphing_sets = [["#5CD629", "#662BF0", "#DA3A32"], ["#662BF0", "#5CD629","#662BF0", ],]
    graphing_outputs = [output_gpr, output_prp]

    font_size = 12
    time = jnp.linspace(0, 0.5, 50)
    
    fig = plt.figure(figsize=(12, 4))
    
    # Define the grid
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], )
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[1, 0])
    ax2 = plt.subplot(gs[:, 1])  # Span both rows for the 2nd column
    ax3 = plt.subplot(gs[:, 2])  # Span both rows for the 3rd column
    
    # Plotting for the first two subplots in the first column
    for i, ax in enumerate([ax0, ax1]):
        ax.plot(time, graphing_outputs[i][0, :, 0], label=f'Row {i}', color="#F0B72B")
        color_set_axis(ax, graphing_sets[i], graphing_inputs[i], time)
        ax.set_ylabel('RNN output', fontsize=font_size)
        ax.set_ylim([-1.85, 1.85])
        ax.set_yticks([-1, 0, 1])
        if i == 0:  # For the first subplot, remove x-axis ticks and labels
            ax.set_xticklabels([])
            ax.set_xlabel('')
            ax.set_xticks([])
    
    ax1.set_xlabel('Time (s)', fontsize=font_size)  # x-label only for the bottom plot in the first column
    
    if dynamics_type == "attractive":
        ax3.set_xticklabels([])
        ax3.set_xlabel('PCA 1', fontsize=font_size)
        ax3.set_xticks([])
        ax3.set_yticklabels([])
        ax3.set_ylabel('PCA 2', fontsize=font_size)
        ax3.set_yticks([])
    else:
        ax3.set_xticklabels([])
        ax3.set_xlabel('Phase angle state space', fontsize=font_size)
        ax3.set_xticks([])
        ax3.set_yticklabels([])
        ax3.set_ylabel('', fontsize=font_size)
        ax3.set_yticks([])
    
    if dynamics_type == "attractive":
        plot_2D_PCA(ax2, training_rates, testing_rates, testing_labels)
    else:
        plot_2D_PCA_oscillatory(ax2, training_rates, testing_rates, rate_null, testing_labels)
    
    ax2.set_title('', fontsize=font_size)
    ax2.set_xticklabels([])
    ax2.set_xlabel('PCA 1', fontsize=font_size)
    ax2.set_xticks([])
    ax2.set_yticklabels([])
    ax2.set_ylabel('PCA 2', fontsize=font_size)
    ax2.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(save_loc)
    plt.show()