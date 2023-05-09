import numpy as np
import pytest

from psydac.core.bsplines          import make_knots
from psydac.fem.basic              import FemField
from psydac.fem.splines            import SplineSpace
from psydac.fem.tensor             import TensorFemSpace
from psydac.feec.global_projectors import Projector_H1, Projector_L2
from psydac.ddm.cart               import DomainDecomposition

#==============================================================================
@pytest.mark.parametrize('domain', [(0, 2*np.pi)])
@pytest.mark.parametrize('ncells', [500])
@pytest.mark.parametrize('degree', [1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize('periodic', [False, True])

def test_H1_projector_1d(domain, ncells, degree, periodic):

    breaks = np.linspace(*domain, num=ncells+1)
    knots  = make_knots(breaks, degree, periodic)

    domain_decomposition = DomainDecomposition([ncells], [periodic])

    # H1 space (0-forms)
    N  = SplineSpace(degree=degree, knots=knots, periodic=periodic, basis='B')
    V0 = TensorFemSpace(domain_decomposition, N)

    # Projector onto H1 space (1D interpolation)
    P0 = Projector_H1(V0)

    # Function to project
    f  = lambda xi1 : np.sin( xi1 + 0.5 )

    # Compute the projection
    u0 = P0(f)

    # Create evaluation grid, and check if  u0(x) == f(x)
    xgrid = np.linspace(*N.domain, num=101)
    vals_u0 = np.array([u0(x) for x in xgrid])
    vals_f  = np.array([f(x)  for x in xgrid])

    # Test if max-norm of error is <= TOL
    maxnorm_error = abs(vals_u0 - vals_f).max()
    print(ncells, maxnorm_error)
#    assert maxnorm_error <= 1e-14

#==============================================================================
@pytest.mark.parametrize('domain', [(0, 2*np.pi)])
@pytest.mark.parametrize('ncells', [100, 200, 300])
@pytest.mark.parametrize('degree', [2])
@pytest.mark.parametrize('periodic', [False])
@pytest.mark.parametrize('nquads', [100, 120, 140, 160])

def test_L2_projector_1d(domain, ncells, degree, periodic, nquads):

    breaks = np.linspace(*domain, num=ncells+1)
    knots  = make_knots(breaks, degree, periodic)

    domain_decomposition = DomainDecomposition([ncells], [periodic])

    # H1 space (0-forms)
    N  = SplineSpace(degree=degree, knots=knots, periodic=periodic, basis='B')
    V0 = TensorFemSpace(domain_decomposition, N)

    # L2 space (1-forms)
    V1 = V0.reduce_degree(axes=[0], basis='M')

    # Projector onto L2 space (1D histopolation)
    P1 = Projector_L2(V1, nquads=[nquads])

    # Function to project
    f  = lambda xi1 : np.sin( xi1 + 0.5 )

    # Compute the projection
    u1 = P1(f)

    # Create evaluation grid, and check if  u1(x) == f(x)
    xgrid = np.linspace(*N.domain, num=11)
    vals_u1 = np.array([u1(x) for x in xgrid])
    vals_f  = np.array([f(x)  for x in xgrid])

    # Test if max-norm of error is <= TOL
    maxnorm_error = abs(vals_u1 - vals_f).max()
    print(ncells, maxnorm_error)
#    assert maxnorm_error <= 1e-14

#==============================================================================
if __name__ == '__main__':

    domain   = (0, 2*np.pi)
    degree   = 3
    periodic = True
    ncells   = [10, 20, 40, 80, 160, 320, 640]

    for nc in ncells:
        test_H1_projector_1d(domain, nc, degree, periodic)

    nquads = degree
    for nc in ncells:
        test_L2_projector_1d(domain, nc, degree, periodic, nquads)
