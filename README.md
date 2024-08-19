<div align="center">
<img src="https://github.com/keith-murray/attract-or-oscillate/blob/main/results/dynamical_mechanisms.png" alt="Dynamical mechanisms figure" width="450">
</div>

# To attract or to oscillate: Validating dynamics with behavior
This repository contains the code for [my MEng thesis](https://hdl.handle.net/1721.1/153709), _To attract or to oscillate: Validating dynamics with behavior_. All code is written in Python, and recurrent neural network (RNN) models are written and trained in JAX, Flax, and Optax.

## Synopsis
RNNs can learn two distinct dynamical systems to compute modular arithmetic, specifically $a + b + c \equiv 0\ (\text{mod } 3)$. One dynamical system is characterized by a lattice of fixed-point attractors, termed the attractive mechanism, and the other is characterized by a limit cycle, termed the oscillatory mechanism. These two systems exhibit unique psychometric curves when computing modular arithmetic for ambiguous stimuli and unique generalization curves when trained on partial datasets. Fundamentally, they differ in how they encode stimuli: the attractive mechanism encodes stimuli as vectors, while the oscillatory mechanism encodes stimuli as angles.

## Repository organization
In the `results/` and `scripts/` folders, there are two subfolders:
- `development/` - Jupyter notebooks used for the development of the `src/` python package
- `experiments/` - Jupyter notebooks used to train models and generate the results discussed in the thesis

## A technical note
[JAX](https://jax.readthedocs.io/en/latest/quickstart.html) is an incredibly powerful Python package. In the context of training RNNs, JAX's [`scan` function](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan) is significantly faster than using a `for` loop. I was able to train [16,128 RNNs](https://openreview.net/forum?id=ql3u5ITQ5C) on the [MIT SuperCloud HPC](https://doi.org/10.1109/HPEC.2018.8547629) in about 60 hours. Given that training and analyzing RNNs has become [routine in computational neuroscience](https://doi.org/10.1016/j.conb.2017.06.003), I'm betting that the field will shift to using JAX, [Flax](https://flax.readthedocs.io), and [Optax](https://optax.readthedocs.io/en/latest/).
