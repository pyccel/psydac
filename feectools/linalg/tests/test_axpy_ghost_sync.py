from feectools.ddm.cart import DomainDecomposition, CartDecomposition
from feectools.linalg.stencil import StencilVectorSpace


def _space():
    dd = DomainDecomposition([8, 8], [True, True])
    cart = CartDecomposition(dd, [8, 8], [[0], [0]], [[7], [7]], [2, 2], [1, 1])
    return StencilVectorSpace(cart)


def test_axpy_ghost_sync():
    """y += a*x must mark y out of sync if x is, and must not modify the flag of x."""
    V = _space()
    x, y = V.zeros(), V.zeros()

    x.ghost_regions_in_sync = False
    y.ghost_regions_in_sync = True
    y.mul_iadd(2.0, x)
    assert not y.ghost_regions_in_sync
    assert not x.ghost_regions_in_sync

    x.ghost_regions_in_sync = True
    y.ghost_regions_in_sync = False
    y.mul_iadd(2.0, x)
    assert not y.ghost_regions_in_sync
    assert x.ghost_regions_in_sync

    x.ghost_regions_in_sync = True
    y.ghost_regions_in_sync = True
    y.mul_iadd(2.0, x)
    assert y.ghost_regions_in_sync
