import jax.numpy as jnp
from jax import random
from itertools import product

class SETDataset:
    def __init__(self, key, accepted_trials, rejected_trials):
        """
        Initialize the SETDataset class.
        
        Parameters:
            key (random.PRNGKey): The initial random key.
            accepted_trials (int): The number of accepted trials.
            rejected_trials (int): The number of rejected trials.
        """
        self.accepted_trials = accepted_trials
        self.rejected_trials = rejected_trials
        self.key = key
        
        self.encoded_colors = self.generate_encoded_colors()
        
        self.training_dict = self.create_data_dict()
        self.fill_data_dict(self.training_dict, self.accepted_trials, self.rejected_trials,)
        print("\nTRAINING DATASET\n")
        self.print_data_dict(self.training_dict,)

        print("\n----------")
        
        self.testing_dict = self.create_data_dict()
        self.fill_data_dict(self.testing_dict, 1, 1,)
        print("\nTESTING DATASET\n")
        self.print_data_dict(self.testing_dict,)

    def generate_subkey(self,):
        """
        Generate a new subkey for random operations.
        
        Returns:
            random.PRNGKey: The new subkey.
        """
        key, subkey = random.split(self.key)
        self.key = key
        return subkey

    def generate_encoded_colors(self,):
        """
        Generate encoded color vectors for 'G', 'P', and 'R'.
        
        Returns:
            dict: A dictionary containing the color codes 'G', 'P', and 'R' and their associated random vectors.
        """
        encoded_colors = {}
        subkey = self.generate_subkey()
        for color, new_key in zip(['G', 'P', 'R'], random.split(subkey, 3)):
            encoded_colors[color] = random.normal(new_key, (100,))
        return encoded_colors

    def create_data_dict(self,):
        """
        Create a dictionary of all color combinations.
        
        Returns:
            dict: A dictionary with keys as color combinations (e.g. 'GGG', 'GRR', etc.) and values as empty lists.
        """
        all_combinations = product(['G', 'P', 'R'], repeat=3)
        training_dict = {"".join(comb): [] for comb in all_combinations}
        return training_dict

    def create_trial(self, SET_combination, unique_colors,):
        """
        Create a trial with random "ball" positions and associated colors.
        
        Parameters:
            SET_combination (str): A string containing a sequence of colors like 'GGP', 'RPG', etc.
            unique_colors (int): The number of unique colors in the combination.
            
        Returns:
            tuple: A tuple containing:
                - input_array (jnp.ndarray): JAX array of shape (50, 100) representing the trial.
                - output_array (jnp.ndarray): JAX array of shape (1), value is either -1 or 1 based on unique colors.
        """
        # Initialize a zero JAX array with shape (50, 100)
        input_array = jnp.zeros((50, 100))
        
        # Select random "ball" positions, ensuring minimum index distance of 5 between each "ball"
        while True:
            subkey = self.generate_subkey()
            ball_indices = sorted(random.randint(subkey, shape=(3,), minval=3, maxval=40))
            if all(ball_indices[i] - ball_indices[i-1] >= 5 for i in range(1, len(ball_indices))):
                break
                
        # Place the color vectors based on the random "ball" positions
        for i, ball_idx in enumerate(ball_indices):
            color = SET_combination[i]
            color_vector = self.encoded_colors[color]
            input_array = input_array.at[ball_idx].add(color_vector)
            input_array = input_array.at[ball_idx + 1].add(color_vector)
            
        # Determine the output_array based on the uniqueness of colors
        output_array = jnp.array([-1.0]) if unique_colors == 2 else jnp.array([1.0])
        
        return (input_array, output_array)

    def fill_data_dict(self, data_dict, accepted_trials, rejected_trials,):
        """
        Fill the data dictionary with trials based on unique colors.
        
        Parameters:
            data_dict (dict): The dictionary to be filled.
            accepted_trials (int): The number of accepted trials.
            rejected_trials (int): The number of rejected trials.
        """
        for SET_combination in data_dict:
            unique_colors = len(set(list(SET_combination)))

            if unique_colors == 2:
                for trial in range(rejected_trials):
                    data_dict[SET_combination].append(self.create_trial(SET_combination, unique_colors))

            else:
                for trial in range(accepted_trials):
                    data_dict[SET_combination].append(self.create_trial(SET_combination, unique_colors))

    def check_and_return_label(self, data_dict, SET_combination,):
        """
        Check if all labels for a given SET combination are the same and return that label.
        
        Parameters:
            data_dict (dict): The data dictionary containing trials.
            SET_combination (str): The SET combination to be checked.
            
        Returns:
            int: The label of the SET combination (-1 or 1).
        """
        SET_trials = data_dict[SET_combination]
        SET_labels = [trial[1].item() for trial in SET_trials]
    
        # Check if all labels are the same
        first_label = SET_labels[0]
        assert all(label == first_label for label in SET_labels), "Labels are not the same for all trials."
    
        return first_label

    def print_data_dict(self, data_dict):
        """
        Print two grids. One grid has all the positive label SET_combinations, 
        and the other grid has all the negative label SET_combinations.

        Parameters:
            data_dict (dict): The data dictionary containing trials.
        """
        print("Positive Label Grid:")
        print("SET_combinations | Number of Trials | Label")
        for comb in data_dict:
            label = self.check_and_return_label(data_dict, comb)
            if label == 1:
                print(f"{comb} | {len(data_dict[comb])} | {label}")
                
        print("\nNegative Label Grid:")
        print("SET_combinations | Number of Trials | Label")
        for comb in data_dict:
            label = self.check_and_return_label(data_dict, comb)
            if label == -1:
                print(f"{comb} | {len(data_dict[comb])} | {label}")
    
    
if __name__ == "__main__":
    pass