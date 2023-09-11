from functools import partial
from typing import Any, Optional
import jax.numpy as jnp
from jax import random
from flax import linen as nn  # Linen API
from flax.linen.module import compact

class EulerCTRNNCell(nn.RNNCellBase):
    """
    A continuous-time recurrent neural network (CTRNN) cell
    that is discritized with the Euler method.

    Parameters:
        features (int): The number of output features.
        alpha (jnp.float32): The ratio of dt to tau.
        noise (jnp.float32): The noise multiplier.
    """
    features: int
    alpha: jnp.float32
    noise: jnp.float32
    kernel_init: initializers.Initializer = nn.initializers.glorot_normal()
    bias_init: initializers.Initializer = nn.initializers.zeros_init()
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32
    carry_init: initializers.Initializer = nn.initializers.zeros_init()

    @compact
    def __call__(self, carry, inputs,):
        """
        A call to a EulerCTRNN cell.

        Parameters:
            carry (tuple):
            inputs:

        Returns:

        """
        h, key = carry
        key, subkey = random.split(key)
        hidden_features = h.shape[-1]

        dense_h = partial(
            nn.Dense,
            features=hidden_features,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        dense_i = partial(
            nn.Dense,
            features=hidden_features,
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        noise_shape = h.shape
        noise = random.normal(subkey, noise_shape)

        new_h = (float32(1.0) - self.alpha) * h + self.alpha * (dense_h(name='recurrent_kernel')(nn.activation.tanh(h)) + dense_i(name='input_kernel')(inputs) + self.noise * noise)
        
        return (new_h, key), nn.activation.tanh(new_h)

    def initialize_carry(self, key, input_shape,):
        """
        Initialize the EulerCTRNN cell carry.

        Parameters:
            key (random.PRNGKey):
            input_shape (tuple(int)):

        Returns:

        """
        batch_dims = input_shape[:1]
        mem_shape = batch_dims + (self.features,)
        key, subkey = random.split(key)
        h = self.carry_init(subkey, mem_shape, self.param_dtype,)

        return (h, key)

    @property
    def num_feature_axes(self,):
        return 1

class EulerCTRNN(nn.Module):
    features: int
    alpha: jnp.float32
    noise: jnp.float32
    kernel_init: initializers.Initializer = nn.initializers.glorot_normal()
    bias_init: initializers.Initializer = nn.initializers.zeros_init()
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32

    @compact
    def __call__(self, x,):
        x, key = x

        dense_o = partial(
            nn.Dense,
            features=1,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        ctrnn = nn.RNN(EulerCTRNNCell(
            features=self.features,
            alpha=self.alpha,
            noise=self.noise,
        ))

        init_carry = ctrnn(key, x.shape)
        rates = ctrnn.apply(init_carry, x)
        z = dense_o(name='output_kernel')(rates)

        return z, rates