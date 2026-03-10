#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np
import pytest

from psydac.core.bsplines                       import make_knots
from psydac.fem.basic                           import FemField
from psydac.fem.splines                         import SplineSpace
from psydac.fem.tensor                          import TensorFemSpace
from psydac.feec.global_geometric_projectors    import GlobalGeometricProjectorH1
from psydac.feec.global_geometric_projectors    import GlobalGeometricProjectorL2

from psydac.ddm.cart               import DomainDecomposition
from sympde.topology               import Square, Cube
from psydac.api.discretization     import discretize
from sympde.topology               import element_of, Derham


#==============================================================================
@pytest.mark.parametrize('domain', [(0, 2*np.pi)])
@pytest.mark.parametrize('ncells', [500])
@pytest.mark.parametrize('degree', [1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('multiplicity', [1, 2])

def test_H1_projector_1d(domain, ncells, degree, periodic, multiplicity, verbose=False):
    if verbose:
        print('\n1d_h1')

    #change mulitplicity if higher than degree to avoid problems (case p<m doesn't work)
    multiplicity = min(multiplicity,degree)
    breaks = np.linspace(*domain, num=ncells+1)
    knots  = make_knots(breaks, degree, periodic, multiplicity=multiplicity)

    domain_decomposition = DomainDecomposition([ncells], [periodic])

    # H1 space (0-forms)
    N  = SplineSpace(degree=degree, knots=knots, periodic=periodic, basis='B')
    V0 = TensorFemSpace(domain_decomposition, N)

    # Projector onto H1 space (1D interpolation)
    P0 = GlobalGeometricProjectorH1(V0)

    # Function to project
    f  = lambda xi1 : np.sin( xi1 + 0.5 )

    # Compute the projection
    u0 = P0(f)

    # Create evaluation grid, and check if  u0(x) == f(x)
    xgrid = np.linspace(*N.domain, num=101)
    vals_u0 = np.array([u0(x) for x in xgrid])
    vals_f  = np.array([f(x)  for x in xgrid])

    # Test max-norm of error is converging with right order
    maxnorm_error = abs(vals_u0 - vals_f).max()
    error_estim = ncells**(-degree-1)
    if verbose:
        print(ncells, maxnorm_error / error_estim)
    assert maxnorm_error <= 5 * error_estim

#==============================================================================
@pytest.mark.parametrize('domain', [(0, 2*np.pi)])
@pytest.mark.parametrize('ncells', [100, 200, 300])
@pytest.mark.parametrize('degree', [2])
@pytest.mark.parametrize('periodic', [True, False])
@pytest.mark.parametrize('nquads', [100, 120, 140, 160])
@pytest.mark.parametrize('multiplicity', [1, 2])

def test_L2_projector_1d(domain, ncells, degree, periodic, nquads, multiplicity, verbose=False):
    if verbose:
        print('\n1d_l2')

    #change mulitplicity if higher than degree to avoid problems (case p<m doesn't work)
    multiplicity = min(multiplicity,degree)
    breaks = np.linspace(*domain, num=ncells+1)
    knots  = make_knots(breaks, degree, periodic, multiplicity=multiplicity)

    domain_decomposition = DomainDecomposition([ncells], [periodic])

    # H1 space (0-forms)
    #change multiplicity if higher than degree to avoid problems (case p<m doesn't work)
    multiplicity = min(multiplicity, degree)
    # H1 space (0-forms)
    N  = SplineSpace(degree=degree, knots=knots, periodic=periodic, basis='B')
    V0 = TensorFemSpace(domain_decomposition, N)

    # L2 space (1-forms)
    V1 = V0.reduce_degree(axes=[0], basis='M')

    # Projector onto L2 space (1D histopolation)
    P1 = GlobalGeometricProjectorL2(V1, nquads=[nquads])

    # Function to project
    f  = lambda xi1 : np.sin( xi1 + 0.5 )

    # Compute the projection
    u1 = P1(f)

    # Create evaluation grid, and check if  u1(x) == f(x)
    xgrid = np.linspace(*N.domain, num=11)
    vals_u1 = np.array([u1(x) for x in xgrid])
    vals_f  = np.array([f(x)  for x in xgrid])

    # Test max-norm of error is converging with right order
    maxnorm_error = abs(vals_u1 - vals_f).max()
    error_estim = ncells**(-degree-1)
    if verbose:
        print(ncells, maxnorm_error / error_estim)
    assert maxnorm_error <= 10 * error_estim
    
#==============================================================================
@pytest.mark.parametrize('ncells', [[57,64]])
@pytest.mark.parametrize('degree', [[2,2], [2,3], [3,3]])
@pytest.mark.parametrize('periodic', [[False, False], [True, True]])
@pytest.mark.parametrize('multiplicity', [(1, 1), (2, 2)])

def test_derham_projector_2d_hdiv(ncells, degree, periodic, multiplicity, verbose=False):
    if verbose:
        print('\n2d_hdiv')

    domain = Square('Omega', bounds1 = (0,2*np.pi), bounds2 = (0,2*np.pi))
    domain_h = discretize(domain, ncells=ncells, periodic=periodic)
    
    # Reduce the multiplicity if higher than the degree (otherwise splines are discontinuous)
    multiplicity = [min(m, p) for p, m in zip(degree, multiplicity)]

    derham   = Derham(domain, ["H1", "Hdiv", "L2"])
    derham_h   = discretize(derham, domain_h, degree=degree, get_H1vec_space = True, multiplicity=multiplicity)
    P0, P1, P2, PX = derham_h.projectors(nquads=[2*p+1 for p in degree])

    # Function to project
    f1  = lambda xi1, xi2 : np.sin( xi1 + 0.5 ) * np.cos( xi2 + 0.3 )
    f2  = lambda xi1, xi2 : np.cos( xi1 + 0.5 ) * np.sin( xi2 - 0.2 )

    # Compute the projection
    u0 = P0(f1)
    u2 = P2(f1)
    u1 = P1((f1,f2))
    ux = PX((f1,f2))

    # Create evaluation grid, and check if  u0(x) == f(x)
    xgrid = np.linspace(0, 2*np.pi, num=51)
    vals_u0   = np.array([[u0(x, y) for x in xgrid] for y in xgrid])
    vals_u1_1 = np.array([[u1(x, y)[0] for x in xgrid] for y in xgrid])
    vals_u2   = np.array([[u2(x, y) for x in xgrid] for y in xgrid])
    vals_ux_1 = np.array([[ux(x, y)[0] for x in xgrid] for y in xgrid])
    vals_f    = np.array([[f1(x, y) for x in xgrid] for y in xgrid])

    # Test max-norm of error is converging with right order
    nc_min = min(ncells)
    deg_min = min(degree)
    error_estim = nc_min**(-deg_min-1)
    error_estim_low = nc_min**(-deg_min)  # deg-1 for certain components

    maxnorm_error = abs(vals_u0 - vals_f).max()
    if verbose:
        print(ncells, maxnorm_error / error_estim)
    assert maxnorm_error <= 10 * error_estim

    maxnorm_error = abs(vals_u1_1 - vals_f).max()
    if verbose:
        print(ncells, maxnorm_error / error_estim_low)
    assert maxnorm_error <= 10 * error_estim_low

    maxnorm_error = abs(vals_u2 - vals_f).max()
    if verbose:
        print(ncells, maxnorm_error / error_estim_low)
    assert maxnorm_error <= 10 * error_estim_low

    maxnorm_error = abs(vals_ux_1 - vals_f).max()
    if verbose:
        print(ncells, maxnorm_error / error_estim)
    assert maxnorm_error <= 10 * error_estim

#==============================================================================
@pytest.mark.parametrize('ncells', [[50,66]])
@pytest.mark.parametrize('degree', [[2,2], [2,3], [3,3]])
@pytest.mark.parametrize('periodic', [[False, False], [True, True]])
@pytest.mark.parametrize('multiplicity', [(1, 1), (2, 2)])

def test_derham_projector_2d_hdiv_2(ncells, degree, periodic, multiplicity, verbose=False):
    if verbose:
        print('\n2d_hdiv_2')

    domain = Square('Omega', bounds1 = (0,1), bounds2 = (0,1))
    domain_h = discretize(domain, ncells=ncells, periodic=periodic)
    
    # Reduce the multiplicity if higher than the degree (otherwise splines are discontinuous)
    multiplicity = [min(m, p) for p, m in zip(degree, multiplicity)]
    
    derham   = Derham(domain, ["H1", "Hdiv", "L2"])
    derham_h   = discretize(derham, domain_h, degree=degree, get_H1vec_space = True, multiplicity=multiplicity)
    P0, P1, P2, PX = derham_h.projectors()

    # Function to project
    f1  = lambda xi1, xi2 : 20 * xi1**2*(xi1-1.)**2 
    f2  = lambda xi1, xi2 : 100 * xi2**2*(xi2-1.)**2

    # Compute the projection
    u0 = P0(f1)
    u2 = P2(f1)
    u1 = P1((f1,f2))
    ux = PX((f1,f2))

    # Create evaluation grid, and check if  u0(x) == f(x)
    xgrid = np.linspace(0, 1, num=51)
    vals_u0   = np.array([[u0(x, y) for x in xgrid] for y in xgrid])
    vals_u1_2 = np.array([[u1(x, y)[1] for x in xgrid] for y in xgrid])
    vals_u2   = np.array([[u2(x, y) for x in xgrid] for y in xgrid])
    vals_ux_2 = np.array([[ux(x, y)[1] for x in xgrid] for y in xgrid])
    vals_f1   = np.array([[f1(x, y) for x in xgrid] for y in xgrid])
    vals_f2   = np.array([[f2(x, y) for x in xgrid] for y in xgrid])

    # Test max-norm of error is converging with right order
    nc_min = min(ncells)
    deg_min = min(degree)
    error_estim = nc_min**(-deg_min-1)
    error_estim_low = nc_min**(-deg_min) # deg-1 for certain components

    maxnorm_error = abs(vals_u0 - vals_f1).max()
    if verbose:
        print(ncells, maxnorm_error / error_estim)
    assert maxnorm_error <= 10 * error_estim

    maxnorm_error = abs(vals_u1_2 - vals_f2).max()
    if verbose:
        print(ncells, maxnorm_error / error_estim_low)
    assert maxnorm_error <= 10 * error_estim_low

    maxnorm_error = abs(vals_u2 - vals_f1).max()
    if verbose:
        print(ncells, maxnorm_error / error_estim_low)
    assert maxnorm_error <= 10 * error_estim_low

    maxnorm_error = abs(vals_ux_2 - vals_f2).max()
    if verbose:
        print(ncells, maxnorm_error / error_estim)
    assert maxnorm_error <= 10 * error_estim
    
#==============================================================================
@pytest.mark.parametrize('ncells', [[62,58]])
@pytest.mark.parametrize('degree', [[2,2], [2,3], [3,3]])
@pytest.mark.parametrize('periodic', [[False, False], [True, False] ,[True, True]])
@pytest.mark.parametrize('multiplicity', [[1,1],[2,2]])

def test_derham_projector_2d_hcurl(ncells, degree, periodic, multiplicity, verbose=False):
    if verbose:
        print('\n2d_hcurl')

    domain = Square('Omega', bounds1 = (0,2*np.pi), bounds2 = (0,2*np.pi))
    domain_h = discretize(domain, ncells=ncells, periodic=periodic)
    
    # Reduce the multiplicity if higher than the degree (otherwise splines are discontinuous)
    multiplicity = [min(m, p) for p, m in zip (degree, multiplicity)]
    
    derham   = Derham(domain, ["H1", "Hcurl", "L2"])
    derham_h   = discretize(derham, domain_h, degree=degree, get_H1vec_space = True, multiplicity=multiplicity)
    P0, P1, P2, PX = derham_h.projectors()

    # Function to project
    f1  = lambda xi1, xi2 : np.sin( xi1 + 0.5 ) * np.cos( xi2 + 0.3 )
    f2  = lambda xi1, xi2 : np.cos( xi1 + 0.5 ) * np.sin( xi2 - 0.2 )

    # Compute the projection
    u0 = P0(f1)
    u2 = P2(f1)
    u1 = P1((f1,f2))
    ux = PX((f1,f2))

    # Create evaluation grid, and check if  u0(x) == f(x)
    xgrid = np.linspace(0, 2*np.pi, num=51)
    vals_u0   = np.array([[u0(x, y) for x in xgrid] for y in xgrid])
    vals_u1_1 = np.array([[u1(x, y)[0] for x in xgrid] for y in xgrid])
    vals_u2   = np.array([[u2(x, y) for x in xgrid] for y in xgrid])
    vals_ux_1 = np.array([[ux(x, y)[0] for x in xgrid] for y in xgrid])
    vals_f    = np.array([[f1(x, y) for x in xgrid] for y in xgrid])

    # Test max-norm of error is converging with right order
    nc_min = min(ncells)
    deg_min = min(degree)
    error_estim = nc_min**(-deg_min-1)
    error_estim_low = nc_min**(-deg_min) # deg-1 for certain components

    maxnorm_error = abs(vals_u0 - vals_f).max()
    if verbose:
        print(ncells, maxnorm_error / error_estim)
    assert maxnorm_error <= 10 * error_estim

    maxnorm_error = abs(vals_u1_1 - vals_f).max()
    if verbose:
        print(ncells, maxnorm_error / error_estim_low)
    assert maxnorm_error <= 10 * error_estim_low

    maxnorm_error = abs(vals_u2 - vals_f).max()
    if verbose:
        print(ncells, maxnorm_error / error_estim_low)
    assert maxnorm_error <= 10 * error_estim_low

    maxnorm_error = abs(vals_ux_1 - vals_f).max()
    if verbose:
        print(ncells, maxnorm_error / error_estim)
    assert maxnorm_error <= 10 * error_estim
    
#==============================================================================
@pytest.mark.parametrize('ncells', [[10,9,12]])
@pytest.mark.parametrize('degree', [[2,2,2], [2,3,2], [3,3,3]])
@pytest.mark.parametrize('periodic', [[False, False, False], [True, True, True]])
@pytest.mark.parametrize('multiplicity', [[1,1,1], [1,2,2], [2,2,2]])

def test_derham_projector_3d(ncells, degree, periodic, multiplicity, verbose=False):

    domain = Cube('Omega', bounds1 = (0,2*np.pi), bounds2 = (0,2*np.pi), bounds3 = (0,2*np.pi))
    domain_h = discretize(domain, ncells=ncells, periodic=periodic)
    
    derham   = Derham(domain)
    #change multiplicity if higher than degree to avoid problems (case p<m doesn't work)
    multiplicity = [min(m, p) for p, m in zip(degree, multiplicity)]

    derham_h   = discretize(derham, domain_h, degree=degree, get_H1vec_space = True, multiplicity = multiplicity)
    P0, P1, P2, P3, PX = derham_h.projectors()

    # Function to project
    f1 = lambda xi1, xi2, xi3 : np.sin( xi1 + 0.51 ) * np.cos( xi2 + 0.32 ) * np.sin( xi3 - 0.43)
    f2 = lambda xi1, xi2, xi3 : np.cos( xi1 + 0.27 ) * np.sin( xi2 - 0.29 ) * np.cos( xi3 + 0.67)
    f3 = lambda xi1, xi2, xi3 : np.cos( xi1 + 0.72 ) * np.sin( xi2 - 0.73 ) * np.cos( xi3 - 0.14)

    # Compute the projection
    u0 = P0(f1)
    u3 = P3(f1)
    u1 = P1((f1,f2,f3))
    u2 = P2((f1,f2,f3))
    ux = PX((f1,f2,f3))

    # Create evaluation grid, and check if  u0(x) == f(x)
    xgrid = np.linspace(0, 2*np.pi, num=37)
    vals_u0   = np.array([[[u0(x, y, z) for x in xgrid] for y in xgrid] for z in xgrid])
    vals_u1_1 = np.array([[[u1(x, y, z)[0] for x in xgrid] for y in xgrid] for z in xgrid])
    vals_u2_1 = np.array([[[u2(x, y, z)[0] for x in xgrid] for y in xgrid] for z in xgrid])
    vals_ux_1 = np.array([[[ux(x, y, z)[0] for x in xgrid] for y in xgrid] for z in xgrid])
    vals_u3   = np.array([[[u3(x, y, z) for x in xgrid] for y in xgrid] for z in xgrid])
    vals_f    = np.array([[[f1(x, y, z) for x in xgrid] for y in xgrid] for z in xgrid])

    # Test max-norm of error is converging with right order
    nc_min = min(ncells)
    deg_min = min(degree)
    error_estim = nc_min**(-deg_min-1)
    error_estim_low = nc_min**(-deg_min) # deg-1 for certain components

    maxnorm_error = abs(vals_u0 - vals_f).max()
    if verbose:
        print(ncells, maxnorm_error / error_estim)
    assert maxnorm_error <= 15 * error_estim

    maxnorm_error = abs(vals_u1_1 - vals_f).max()
    if verbose:
        print(ncells, maxnorm_error / error_estim_low)
    assert maxnorm_error <= 10 * error_estim_low

    maxnorm_error = abs(vals_u2_1 - vals_f).max()
    if verbose:
        print(ncells, maxnorm_error / error_estim_low)
    assert maxnorm_error <= 10 * error_estim_low

    maxnorm_error = abs(vals_u3 - vals_f).max()
    if verbose:
        print(ncells, maxnorm_error / error_estim_low)
    assert maxnorm_error <= 10 * error_estim_low

    maxnorm_error = abs(vals_ux_1 - vals_f).max()
    if verbose:
        print(ncells, maxnorm_error / error_estim)
    assert maxnorm_error <= 15 * error_estim

#==============================================================================
def manual_convergence_tests(dim):
    """
    compute several projections and check that the errors converge at least with expected order
    """
    domain   = (0, 2*np.pi)    
    periodic = True

    if dim == 1:
        degree = 3
        ncells = [10, 20, 40, 80, 160, 320, 640]
        for nc in ncells:
            test_H1_projector_1d(domain, nc, degree, periodic, multiplicity=1, verbose=True)
            test_H1_projector_1d(domain, nc, degree, periodic, multiplicity=2, verbose=True)
            test_L2_projector_1d(domain, nc, degree, periodic, nquads=degree, multiplicity=1, verbose=True)
    
    elif dim == 2:
        degree = [3, 3]
        ncells = [10, 20, 40, 80, 160]
        for nc in ncells:
            test_derham_projector_2d_hcurl ([nc, nc], degree, [periodic, periodic], multiplicity=[1,1], verbose=True)
            test_derham_projector_2d_hdiv  ([nc, nc], degree, [periodic, periodic], multiplicity=[1,1], verbose=True)
            test_derham_projector_2d_hdiv  ([nc, nc], degree, [periodic, periodic], multiplicity=[2,2], verbose=True)
            test_derham_projector_2d_hdiv_2([nc, nc], degree, [periodic, periodic], multiplicity=[1,1], verbose=True)
    
    elif dim == 3:
        print("\n3d")
        degrees  = [[2,2,2], [2,3,2], [3,3,3]]
        mult = [[1,1,1], [1,2,2], [2,2,2]]
        ncells = [10, 20, 40]
        for deg in degrees:
            for m in mult:
                print(f"degree       = {deg}")
                print(f"multiplicity = {m}")
                for nc in ncells:
                    test_derham_projector_3d([nc, nc, nc], deg, [periodic, periodic, periodic], multiplicity=m, verbose=True)
                    print()

#==============================================================================
if __name__ == '__main__':

    for dim in [1, 2, 3]:
        manual_convergence_tests(dim)
