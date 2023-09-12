import jax
import jax.numpy as jnp
from jax import random
from itertools import product
import tensorflow as tf

class SETDataset:
    def __init__(self, key, min_training_trials, testing_trials, validate_trials, train_batch_size,):
        """
        Initialize the SETDataset class.
        
        Parameters:
            key (random.PRNGKey): The initial random key.
            min_training_trials (int): The minimum number of training trials per SET.
            testing_trials (int): The number of testing trials.
            validate_trials (int): The number of validate trials.
            train_batch_size (int): The size of training batches
        """
        self.key = key
        self.min_training_trials = min_training_trials
        self.testing_trials = testing_trials
        self.validate_trials = validate_trials
        self.train_batch_size = train_batch_size
        
        self.encoded_colors = self.generate_encoded_colors()
        self.instantiate_SET_dict()
        
        self.training_dict = self.create_data_dict()
        self.fill_data_dict(self.training_dict, self.min_training_trials,)
        self.balance_data_dict(self.training_dict, self.min_training_trials,)
        
        self.testing_dict = self.create_data_dict()
        self.fill_data_dict(self.testing_dict, self.testing_trials,)
        
        self.grok_dict = {}
        self.corrupt_dict = {}
        
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

    def instantiate_SET_dict(self,):
        """
        Creates and fills the SET_dict with the standard SET labels.
        """
        self.SET_dict = self.create_data_dict()
        
        for SET_combination in self.SET_dict:
            unique_colors = len(set(list(SET_combination)))

            if unique_colors == 2:
                self.SET_dict[SET_combination] = (-1, "")
            else:
                self.SET_dict[SET_combination] = (1, "")

    def print_SET_dict(self,):
        """
        Print the contents of SET_dict with SET combinations in the first column,
        label in the second column, and status in the last column.
        """
        print("Contents SET_dict:")
        print("SET_combination | Label | Status")
        for SET, (label, status) in self.SET_dict.items():
            print(f"{SET} | {label} | {status}")
    
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

    def fill_data_dict(self, data_dict, trials,):
        """
        Fill the data dictionary with trials based on unique colors.
        
        Parameters:
            data_dict (dict): The dictionary to be filled.
            trials (int): The number of trials per SET.
        """
        for SET_combination in data_dict:
            unique_colors = len(set(list(SET_combination)))
            data_dict[SET_combination] = [self.create_trial(SET_combination, unique_colors) for _ in range(trials)]

    def balance_data_dict(self, data_dict, min_trials,):
        """
        Balances the number of accepting and rejecting labels in the data_dict.

        Parameters:
            data_dict (dict): The data dictionary containing trials.
            min_trials (int): The minimum number of trials per SET.
        """
        accepting_SETs = []
        rejecting_SETs = []
        
        for SET in data_dict:
            label = self.check_and_return_label(data_dict, SET)

            if label == 1:
                accepting_SETs.append(SET)
            elif label == -1:
                rejecting_SETs.append(SET)
                
        if len(accepting_SETs) > len(rejecting_SETs):
            over_represented_SETs = accepting_SETs
            over_represented_type = 1
            under_represented_SETs = rejecting_SETs
            under_represented_type = 2
        else:
            over_represented_SETs = rejecting_SETs
            over_represented_type = 2
            under_represented_SETs = accepting_SETs
            under_represented_type = 1

        under_represented_trials = round((len(over_represented_SETs) * min_trials) / len(under_represented_SETs))

        for SET in under_represented_SETs:
            data_dict[SET] = [self.create_trial(SET, under_represented_type) for _ in range(under_represented_trials)]

        for SET in over_represented_SETs:
            if len(data_dict[SET]) != min_trials:
                data_dict[SET] = [self.create_trial(SET, over_represented_type) for _ in range(min_trials)]
        
    def grok_specified_SET(self, SET,):
        """
        Remove specified SET from the training_dict.

        Parameters:
            SET (str): Specific SET to grok.
        """
        if SET in self.training_dict:
            unique_colors = len(set(list(SET)))
            
            self.SET_dict[SET] = (self.SET_dict[SET][0], 'Grokked')

            del self.training_dict[SET]
            
            self.grok_dict[SET] = [self.create_trial(SET, unique_colors) for _ in range(self.validate_trials)]

    def corrupt_specified_SET(self, SET,):
        """
        Inverse the label for specified SET from the training_dict and testing_dict.
    
        Parameters:
            SET (str): Specific SET to corrupt.
        """
        if SET in self.training_dict:
            unique_colors = len(set(list(SET)))
            
            self.SET_dict[SET] = (-self.SET_dict[SET][0], 'Corrupted')
            
            self.training_dict[SET] = [(x[0], -x[1]) for x in self.training_dict[SET]]
            self.testing_dict[SET] = [(x[0], -x[1]) for x in self.testing_dict[SET]]
            
            self.corrupt_dict[SET] = [self.create_trial(SET, unique_colors) for _ in range(self.validate_trials)]
            self.corrupt_dict[SET] = [(x[0], -x[1]) for x in self.corrupt_dict[SET]]

    def grok_SET(self, num_SETs,):
        """
        Remove randomly selected SETs from the training_dict if they are not corrupted.

        Parameters:
            num_SETs (int): The number of SETs to grok.
        """
        available_sets = [SET for SET, value in self.SET_dict.items() if value[1] == '']
        selected_indices = random.choice(self.generate_subkey(), jnp.arange(len(available_sets)), shape=(num_SETs,), replace=False)
        grokked_sets = [available_sets[i] for i in selected_indices]

        for SET in grokked_sets:
            self.grok_specified_SET(SET)
        
        self.balance_data_dict(self.training_dict, self.min_training_trials,)
    
    def corrupt_SET(self, num_SETs,):
        """
        Inverse the label for randomly selected SETs from the training_dict and testing_dict
        if they are not grokked.
    
        Parameters:
            num_SETs (int): The number of SETs to corrupt.
        """
        available_sets = [SET for SET, value in self.SET_dict.items() if value[1] == '']
        selected_indices = random.choice(self.generate_subkey(), jnp.arange(len(available_sets)), shape=(num_SETs,), replace=False)
        corrupted_sets = [available_sets[i] for i in selected_indices]
    
        for SET in corrupted_sets:
            self.corrupt_specified_SET(SET)

        self.balance_data_dict(self.training_dict, self.min_training_trials,)
    
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
        SET_dict_label = self.SET_dict[SET_combination][0]
        assert all(label == SET_dict_label for label in SET_labels), "Labels are not the same for all trials."
    
        return SET_dict_label
    
    def print_data_dict(self, data_dict):
        """
        Print two grids. One grid has all the positive label SET_combinations, 
        and the other grid has all the negative label SET_combinations.
        Within each grid, the status ('Grokked' or 'Corrupted') of each SET is displayed.
    
        Parameters:
            data_dict (dict): The data dictionary containing trials.
        """
        print("Accepting Grid:")
        print("SET_combinations | Number of Trials | Status")
        for comb in data_dict:
            label = self.check_and_return_label(data_dict, comb)
            if label == 1:
                status = self.SET_dict.get(comb, ('', ''))[1]
                print(f"{comb} | {len(data_dict[comb])} | {status}")
                    
        print("\nRejecting Grid:")
        print("SET_combinations | Number of Trials | Status")
        for comb in data_dict:
            label = self.check_and_return_label(data_dict, comb)
            if label == -1:
                status = self.SET_dict.get(comb, ('', ''))[1]
                print(f"{comb} | {len(data_dict[comb])} | {status}")

    def print_training_testing(self,):
        """
        Print all relevant information pertaining to training and testing dicts.
        """
        print("\nTRAINING DATA\n")
        self.print_data_dict(self.training_dict)
        print("\n----------")
        print("\nTESTING DATA\n")
        self.print_data_dict(self.testing_dict)
        print("\n----------")
        print("\nGROK DATA\n")
        self.print_data_dict(self.grok_dict)
        print("\n----------")
        print("\nCORRUPT DATA\n")
        self.print_data_dict(self.corrupt_dict)
        print("\n")
    
    def generate_jax_tensor(self, data_dict):
        """
        Create features and labels tensors for the tensorflow dataset.
        
        Parameters:
            data_dict (dict): The data dictionary containing trials.
        
        Returns:
            tuple: A tuple containing:
                - features_tensor (jnp.ndarray): A tensor containing all the feature arrays stacked along axis 0. 
                  Shape is (total_trials, 50, 100).
                - labels_tensor (jnp.ndarray): A tensor containing all the label arrays stacked along axis 0.
                  Shape is (total_trials, 5, 1).
        """
        # Initialize lists to hold feature and label arrays
        features_list = []
        labels_list = []
        
        # Iterate through each SET_combination in the data_dict
        for SET_combination, trials in data_dict.items():
            for input_array, output_array in trials:
                features_list.append(input_array)
                
                # Expand the label from shape (1) to (5, 1)
                expanded_label = jnp.tile(output_array, (5, 1))
                labels_list.append(expanded_label)
        
        # Stack the features and labels to create the final tensors
        features_tensor = jnp.stack(features_list, axis=0)
        labels_tensor = jnp.stack(labels_list, axis=0)
        
        return features_tensor, labels_tensor
    
    def generate_tf_dataset(self, data_dict, batch_size):
        """
        Create a TensorFlow Dataset object from the provided data_dict and batch size.
        
        This method first generates JAX tensors for the features and labels using the 
        `generate_jax_tensor` method. These tensors are then converted into a TensorFlow 
        Dataset, which is shuffled and batched based on the provided batch size.
        
        Parameters:
            data_dict (dict): The data dictionary containing trials for each SET_combination.
            batch_size (int): The number of samples to include in each batch.
            
        Returns:
            tf.data.Dataset: A shuffled and batched TensorFlow Dataset object ready for training or evaluation.
        """
        features_tensor, labels_tensor = self.generate_jax_tensor(data_dict)
        dataset = tf.data.Dataset.from_tensor_slices((features_tensor, labels_tensor))
        subkey = self.generate_subkey()
        dataset = dataset.shuffle(dataset.cardinality(), reshuffle_each_iteration=True, seed=subkey[0].item())
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(2)

        return dataset

    def tf_datasets(self,):
        """
        Create both training, testing, grok, and corrupt TensorFlow Datasets.
        """
        training_tf_dataset = self.generate_tf_dataset(self.training_dict, self.train_batch_size)
        test_batch_size = sum([len(trials) for _, trials in self.testing_dict.items()])
        testing_tf_dataset = self.generate_tf_dataset(self.testing_dict, test_batch_size)
        
        try:
            grok_batch_size = sum([len(trials) for _, trials in self.grok_dict.items()])
            grok_tf_dataset = self.generate_tf_dataset(self.grok_dict, grok_batch_size)
        except NameError:
            grok_tf_dataset = None

        try:
            corrupt_batch_size = sum([len(trials) for _, trials in self.corrupt_dict.items()])
            corrupt_tf_dataset = self.generate_tf_dataset(self.corrupt_dict, corrupt_batch_size)
        except NameError:
            corrupt_tf_dataset = None
        
        return training_tf_dataset, testing_tf_dataset, grok_tf_dataset, corrupt_tf_dataset