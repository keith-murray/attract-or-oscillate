import jax.numpy as jnp
from jax import random
from itertools import product

def generate_encoded_colors(key):
    color_dict = {}
    for color, new_key in zip(['G', 'P', 'R'], random.split(key, 3)):
        color_dict[color] = random.normal(new_key, (1, 100))
    return color_dict

def create_color_dict():
    all_combinations = product(['G', 'P', 'R'], repeat=3)
    training_dict = {"".join(comb): [] for comb in all_combinations}
    return training_dict