#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from psydac.linalg.stencil import StencilVectorSpace
from psydac.linalg.block   import BlockVectorSpace
from psydac.polar .dense   import DenseVectorSpace
from psydac.polar .c1_cart import C1_Cart

__all__ = ('new_c1_vector_space',)

#==============================================================================
def new_c1_vector_space(V, radial_dim=0, angle_dim=1):
    """
    Create a new product space from a given stencil vector space.

    Parameters
    ----------
    V : StencilVectorSpace
        Space of the coefficients of a tensor-product finite-element space
        built on a mapping with a polar singularity (O-point).

    radial_dim : int
        Index of the dimension that corresponds to the 'radial' direction.

    angle_dim : int
        Index of the dimension that corresponds to the 'angle' direction.

    Returns
    -------
    P : BlockVectorSpace
        Space of the coefficients of a new finite-element space which has
        C^1 continuity at the O-point.

    """
    assert isinstance(V, StencilVectorSpace)
    assert isinstance(radial_dim, int)
    assert isinstance( angle_dim, int)
    assert 0 <= radial_dim < V.ndim
    assert 0 <=  angle_dim < V.ndim
    assert V.ndim >= 2
    assert V.periods[radial_dim] == False
    assert V.periods[ angle_dim] == True

    c1_cart = C1_Cart(V.cart, radial_dim)
    S = StencilVectorSpace(cart=c1_cart, dtype=V.dtype)

    if V.parallel:
        D = DenseVectorSpace(3, cart=V.cart, radial_dim=radial_dim, angle_dim=angle_dim)
    else:
        D = DenseVectorSpace(3)

    P = BlockVectorSpace(D, S)

    return P
