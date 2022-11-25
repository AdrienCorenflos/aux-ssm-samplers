from chex import dataclass
from jaxtyping import Array, Float


@dataclass(frozen=True)
class LGSSM:
    # initial state
    m0: Float[Array, "dim_x"]
    P0: Float[Array, "dim_x dim_x"]

    # dynamics
    Fs: Float[Array, "b-1 dim_x dim_x"]
    Qs: Float[Array, "b-1 dim_x dim_x"]
    bs: Float[Array, "b-1 dim_x"]

    # emission
    Hs: Float[Array, "b dim_y dim_x"]
    Rs: Float[Array, "b dim_y dim_y"]
    cs: Float[Array, "b dim_y"]

    """
    NamedTuple encapsulating the parameters of the LGSSM.

    Attributes
    ----------
    m0 : Array
        The initial state mean.
    P0 : Array
        The initial state covariance.
    Fs : Array
        The transition matrices.
    Qs : Array
        The transition covariance matrices.
    bs : Array
        The transition offsets.
    Hs : Array
        The observation matrices.
    Rs : Array
        The observation noise covariance matrices.
    cs : Array
        The observation offsets.
    """