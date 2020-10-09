import numpy as np
import pytest

from psydac.core.bsplines    import make_knots
from psydac.fem.basic        import FemField
from psydac.fem.tensor       import SplineSpace, TensorFemSpace
from psydac.fem.vector       import ProductFemSpace
from psydac.feec.derivatives import Grad, Curl, Div

#==============================================================================
@pytest.mark.parametrize('domain', [(0, 1), (-2, 3)])
@pytest.mark.parametrize('ncells', [11, 37])
@pytest.mark.parametrize('degree', [2, 3, 4, 5])
@pytest.mark.parametrize('periodic', [True, False])

def test_Grad_1d(domain, ncells, degree, periodic):

    breaks = np.linspace(*domain, num=ncells+1)
    knots  = make_knots(breaks, degree, periodic)

    # H1 (0-forms)
    N  = SplineSpace(degree=degree, knots=knots, periodic=periodic, basis='B')
    V0 = TensorFemSpace(N)

    # L2 (1-forms)
    V1 = V0.reduce_degree(axes=[0], basis='M')

    # Linear operator: gradient
    grad = Grad(V0, V1)

    # Create random field in V0
    u0 = FemField(V0)

    s, = V0.spaces[0].vector_space.starts
    e, = V0.spaces[0].vector_space.ends

    u0.coeffs[s:e+1] = np.random.random(e-s+1)

    # Compute gradient (=derivative) of u0
    u1 = grad(u0)

    # Create evaluation grid, and check if ∂/∂x u0(x) == u1(x)
    xgrid = np.linspace(*N.domain, num=11)
    y_du0_dx = np.array([u0.gradient(x)[0] for x in xgrid])
    y_u1 = np.array([u1(x) for x in xgrid])

    assert np.allclose(y_du0_dx, y_u1, rtol=1e-14, atol=1e-14)

#==============================================================================
if __name__ == '__main__':

    test_gradient_matrix_1d(domain=[0, 1], ncells=12, degree=3, periodic=False)
    test_gradient_matrix_1d(domain=[0, 1], ncells=12, degree=3, periodic=True)

