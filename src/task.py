import jax.numpy as jnp
from jax import random
from itertools import product

def generate_encoded_colors(key):
    """
    Generate encoded color vectors for 'G', 'P', and 'R'.
    
    Parameters:
        key (random.PRNGKey): The random key for JAX's random functions.
        
    Returns:
        dict: A dictionary containing the color codes 'G', 'P', and 'R' and their associated random vectors.
    """
    color_dict = {}
    for color, new_key in zip(['G', 'P', 'R'], random.split(key, 3)):
        color_dict[color] = random.normal(new_key, (100,))
    return color_dict

def create_color_dict():
    """
    Create a dictionary of all color combinations.
    
    Returns:
        dict: A dictionary with keys as color combinations (e.g. 'GGG', 'GRR', etc.) and values as empty lists.
    """
    all_combinations = product(['G', 'P', 'R'], repeat=3)
    training_dict = {"".join(comb): [] for comb in all_combinations}
    return training_dict

def create_trial(key, color_dict, SET_combination):
    """
    Create a trial with random "ball" positions and associated colors.
    
    Parameters:
        key (random.PRNGKey): The random key for JAX's random functions.
        color_dict (dict): Dictionary containing encoded color vectors.
        SET_combination (str): A string containing a sequence of colors like 'GGP', 'RPG', etc.
        
    Returns:
        tuple: A tuple containing:
            - input_array (jnp.ndarray): JAX array of shape (50, 100) representing the trial.
            - output_array (jnp.ndarray): JAX array of shape (1), value is either -1 or 1 based on unique colors.
    """
    # Initialize a zero JAX array with shape (50, 100)
    input_array = jnp.zeros((50, 100))
    
    # Select random "ball" positions, ensuring minimum index distance of 5 between each "ball"
    while True:
        key, subkey = random.split(key)
        ball_indices = sorted(random.randint(subkey, shape=(3,), minval=3, maxval=40))
        if all(ball_indices[i] - ball_indices[i-1] >= 5 for i in range(1, len(ball_indices))):
            break
            
    # Place the color vectors based on the random "ball" positions
    for i, ball_idx in enumerate(ball_indices):
        color = SET_combination[i]
        color_vector = color_dict[color]
        input_array = input_array.at[ball_idx].add(color_vector)
        input_array = input_array.at[ball_idx + 1].add(color_vector)
        
    # Determine the output_array based on the uniqueness of colors
    unique_colors = len(set(list(SET_combination)))
    output_array = jnp.array([-1.0]) if unique_colors == 2 else jnp.array([1.0])
    
    return (input_array, output_array)


def test_create_trial():
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    color_dict = generate_encoded_colors(subkey)
    
    for SET_combination in ['GGG', 'GGP', 'GPP', 'PPP', 'PPR', 'PRR', 'RRR']:
        key, subkey = random.split(key)
        input_array, output_array = create_trial(subkey, color_dict, SET_combination)
        
        # Test if the distance between "ball" indices is at least 5
        ball_indices = jnp.where(input_array.sum(axis=1) != 0)[0]
        ball_indices = sorted(list(ball_indices))
        ball_indices = ball_indices[::2]
        
        assert all(ball_indices[i] - ball_indices[i-1] >= 5 for i in range(1, len(ball_indices))), f"Failed for {SET_combination}"

        # Test if output_array is correct
        unique_colors = len(set(list(SET_combination)))
        expected_output_array = jnp.array([-1.0]) if unique_colors == 2 else jnp.array([1.0])
        assert jnp.all(output_array == expected_output_array), f"Failed for {SET_combination}"
        
if __name__ == "__main__":
    test_create_trial()