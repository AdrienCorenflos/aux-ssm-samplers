"""
Implements the auxiliary particle Gibbs algorithm with independent proposals, i.e.,
the version of Finke and Thiery (2021) but in the auxiliary paradigm.
"""
from typing import Optional

from .common import AuxiliaryM0, AuxiliaryG0, AuxiliaryMt, AuxiliaryGt
from .generic import get_kernel as get_base_kernel
from .._primitives.csmc.base import Distribution, UnivariatePotential, Dynamics, Potential


def get_kernel(M0: Distribution, G0: UnivariatePotential, Mt: Dynamics, Gt: Potential, N: int,
               backward: bool = False, Pt: Optional[Dynamics] = None, parallel: bool = False):
    """
    Get a local auxiliary kernel with separable proposals.

    Parameters:
    -----------
    M0:
        Initial distribution.
    G0:
        Initial potential.
    Mt:
        Dynamics of the model.
    Gt:
        Potential of the model.
    N: int
        Total number of particles to use in the cSMC sampler.
    backward: bool
        Whether to perform backward sampling or not. If True, the dynamics must implement a valid logpdf method.
    Pt:
        Dynamics of the true model. If None, it is assumed to be the same as Mt.

    Returns:
    --------
    kernel: Callable
        cSMC kernel.
    init: Callable
        Function to initialize the state of the sampler given a trajectory.
    """
    # This function uses the classes defined below
    M0_factory = lambda u, scale: AuxiliaryM0(u=u, sqrt_half_delta=scale)
    G0_factory = lambda u, scale: AuxiliaryG0(M0=M0, G0=G0)
    Mt_factory = lambda u, scale: AuxiliaryMt(params=(u, scale))
    Gt_factory = lambda u, scale: AuxiliaryGt(Mt=Mt, Gt=Gt)
    if not parallel:
        return get_base_kernel(M0_factory, G0_factory, Mt_factory, Gt_factory, N, backward, Pt)
    else:
        raise NotImplementedError("Parallel version not implemented yet.")
