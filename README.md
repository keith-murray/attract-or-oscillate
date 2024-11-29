# Grokking in recurrent networks with attractive and oscillatory dynamics
This repository contains code for the paper, "[Grokking in recurrent networks with attractive and oscillatory dynamics](https://openreview.net/forum?id=ql3u5ITQ5C)". All code is written in Python and recurrent neural networks (RNNs) are implemented and trained in JAX, Flax, and Optax.

<div align="center">
<img src="https://github.com/keith-murray/attract-or-oscillate/blob/main/results/dynamical_mechanisms.png" alt="Dynamical mechanisms figure" width="450">
</div>

## Synopsis
RNNs can learn two distinct dynamical systems to compute modular arithmetic, specifically $a + b + c \equiv 0\ (\text{mod } 3)$. One dynamical system is characterized by a lattice of fixed-point attractors, termed the attractive mechanism, and the other is characterized by a limit cycle, termed the oscillatory mechanism. In a series of _grokking_ experiments, the attractive mechanism generalizes more frequently in training regimes with few withheld data points while the oscillatory mechanism generalizes more frequently in training regimes with many withheld data points.

## Repository organization
In the `results/` and `scripts/` folders, there are two subfolders:
- `development/` - Jupyter notebooks used for the development of the `src/` python package
- `experiments/` - Jupyter notebooks used to train models and execute _grokking_ experiments

## A technical note
[JAX](https://jax.readthedocs.io/en/latest/quickstart.html) is an incredibly powerful deep learning framework. In the context of training RNNs, JAX's [`scan` function](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan) is significantly faster than using a `for` loop. With JAX's `scan` function, I was able to train 16,128 RNNs on the [MIT SuperCloud HPC](https://doi.org/10.1109/HPEC.2018.8547629) in about 60 hours.

Checkout [`keith-murray/ctrnn-jax`](https://github.com/keith-murray/ctrnn-jax) for a sleek JAX implementation of continuous-time recurrent neural networks (CT-RNNs) based on the work in this repo.
