from typing import Union, Callable

import jax.numpy as jnp
from chex import Array, ArrayNumpy, dataclass, ArrayTree

Array = Union[ArrayNumpy, Array, ArrayTree]


@dataclass
class SamplerState:
    x: Array


@dataclass
class CoupledSamplerState:
    state_1: SamplerState
    state_2: SamplerState
    flags: Union[bool, Array] = False

    @property
    def is_coupled(self):
        return jnp.all(self.flags)
