import numpy as np

from sympde.topology import Mapping

from psydac.linalg.basic import LinearOperator
from psydac.linalg.block import BlockVectorSpace

__all__ = (
    'SquareTorus',
    'Annulus',
    'SinMapping1D',
    'check_linop_equality_using_rng'
)

class SquareTorus(Mapping):
    _expressions = {'x': 'x1 * cos(x2)',
                    'y': 'x1 * sin(x2)',
                    'z': 'x3'}
    _ldim = 3
    _pdim = 3


class Annulus(Mapping):
    _expressions = {'x': 'x1 * cos(x2)',
                    'y': 'x1 * sin(x2)'}
    _ldim = 2
    _pdim = 2


class SinMapping1D(Mapping):
    _expressions = {'x': 'sin((pi/2)*x1)'}
    _ldim = 1
    _pdim = 1


def check_linop_equality_using_rng(A, B, tol=1e-15):
    """
    A simple tool to check with almost certainty that two linear operators are
    identical, by applying them repeatedly to random vectors with entries drawn
    uniformly in [0, 1).

    Parameters
    ----------
    A : LinearOperator
        First linear operator to compare.
    B : LinearOperator
        Second linear operator to compare.
    tol : float, optional
        Tolerance for the comparison on the relative 2-norm of the error.
        The default is 1e-15.

    Raises
    ------
    AssertionError
        If the two operators are incompatible, or found to be different. This
        behavior is intended for using this function in unit tests with Pytest.
    
    """

    assert isinstance(A, LinearOperator)
    assert isinstance(B, LinearOperator)
    assert A.domain is B.domain
    assert A.codomain is B.codomain

    rng = np.random.default_rng(42)

    x   = A.domain.zeros()
    y1  = A.codomain.zeros()
    y2  = A.codomain.zeros()

    n   = 10

    for _ in range(n):

        x *= 0.

        if isinstance(A.domain, BlockVectorSpace):
            for block in x.blocks:
                rng.random(size=block._data.shape, dtype="float64", out=block._data)
        else:
            rng.random(size=x._data.shape, dtype="float64", out=x._data)

        A.dot(x, out=y1)
        B.dot(x, out=y2)

        diff            = y1 - y2
        scaled_err_sqr  = diff.inner(diff) / diff.space.dimension**2
        
        assert scaled_err_sqr < tol**2
