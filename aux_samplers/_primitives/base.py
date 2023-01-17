from typing import Union

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
    flags: Array

    @property
    def is_coupled(self):
        return jnp.all(self.flags)
