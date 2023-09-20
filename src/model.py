from functools import partial
from typing import Any, Optional
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

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
    one: jnp.float32 = jnp.float32(1.0)
    kernel_init: nn.initializers.Initializer = nn.initializers.glorot_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32
    carry_init: nn.initializers.Initializer = nn.initializers.ones_init()

    @nn.compact
    def __call__(self, carry, inputs,):
        """
        Compute the next state and output of the EulerCTRNN cell given the current state and input.

        Parameters:
            carry (tuple): A tuple containing the current hidden state (jnp.ndarray) and a JAX random key.
            inputs (jnp.ndarray): The input tensor for the current time step.

        Returns:
            tuple: A tuple containing two elements:
                - A new carry tuple containing the updated hidden state (jnp.ndarray) and a new JAX random key.
                - A tuple containing the output tensor (jnp.ndarray) and the rate tensor (jnp.ndarray).
        """
        h, key = carry
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
        dense_o = partial(
            nn.Dense,
            features=1,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        noise_shape = h.shape
        key, subkey = random.split(key)
        noise = random.normal(subkey, noise_shape)

        new_h = (self.one - self.alpha) * h + self.alpha * (dense_h(name='recurrent_kernel')(nn.activation.tanh(h)) + dense_i(name='input_kernel')(inputs) + self.noise * noise)

        rates = nn.activation.tanh(new_h)
        z = dense_o(name='output_kernel')(rates)
        
        return (new_h, key), (z, rates)

    def initialize_carry(self, key, input_shape,):
        """
        Initialize the state (carry) of the EulerCTRNN cell.

        Parameters:
            key (random.PRNGKey): A JAX random key for initialization.
            input_shape (tuple[int]): The shape of an input sample, not including the time dimension.

        Returns:
            tuple: A carry tuple containing the initial hidden state (jnp.ndarray) and the JAX random key.
        """
        batch_dims = input_shape[:1]
        mem_shape = batch_dims + (self.features,)
        key, subkey = random.split(key)
        h = self.carry_init(subkey, mem_shape, self.param_dtype,)
        
        return h, key

    @property
    def num_feature_axes(self,):
        return 1