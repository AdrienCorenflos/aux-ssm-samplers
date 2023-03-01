from typing import Union

from chex import Array, ArrayNumpy, dataclass, ArrayTree

Array = Union[ArrayNumpy, Array, ArrayTree]


@dataclass
class SamplerState:
    x: Array
