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

def test_H1_projector_1d(domain, ncells, degree, periodic, multiplicity):
    
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

    # Test if max-norm of error is <= TOL
    maxnorm_error = abs(vals_u0 - vals_f).max()
    print(ncells, maxnorm_error)
    assert maxnorm_error <= 1e-9

#==============================================================================
@pytest.mark.parametrize('domain', [(0, 2*np.pi)])
@pytest.mark.parametrize('ncells', [100, 200, 300])
@pytest.mark.parametrize('degree', [2])
@pytest.mark.parametrize('periodic', [True, False])
@pytest.mark.parametrize('nquads', [100, 120, 140, 160])
@pytest.mark.parametrize('multiplicity', [1, 2])

def test_L2_projector_1d(domain, ncells, degree, periodic, nquads, multiplicity):

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

    # Test if max-norm of error is <= TOL
    maxnorm_error = abs(vals_u1 - vals_f).max()
    print(ncells, maxnorm_error)
    assert maxnorm_error <= 1e-3
    
#==============================================================================
@pytest.mark.parametrize('ncells', [[200,200]])
@pytest.mark.parametrize('degree', [[2,2], [2,3], [3,3]])
@pytest.mark.parametrize('periodic', [[False, False], [True, True]])
@pytest.mark.parametrize('multiplicity', [(1, 1), (2, 2)])

def test_derham_projector_2d_hdiv(ncells, degree, periodic, multiplicity):

    domain = Square('Omega', bounds1 = (0,2*np.pi), bounds2 = (0,2*np.pi))
    domain_h = discretize(domain, ncells=ncells, periodic=periodic)
    #change multiplicity if higher than degree to avoid problems (case p<m doesn't work)
    multiplicity = [min(m, p) for p, m in zip(degree, multiplicity)]

    derham   = Derham(domain, ["H1", "Hdiv", "L2"])
    derham_h   = discretize(derham, domain_h, degree=degree, get_H1vec_space = True, multiplicity=multiplicity)
    P0, P1, P2, PX = derham_h.projectors(nquads=[2*p+1 for p in degree])

    # Projector onto H1 space (1D interpolation)

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

    # Test if max-norm of error is <= TOL
    maxnorm_error = abs(vals_u0 - vals_f).max()
    print(ncells, maxnorm_error)
    assert maxnorm_error <= 1e-3
    maxnorm_error = abs(vals_u1_1 - vals_f).max()
    print(ncells, maxnorm_error)
    assert maxnorm_error <= 1e-3
    maxnorm_error = abs(vals_u2 - vals_f).max()
    print(ncells, maxnorm_error)
    assert maxnorm_error <= 1e-3
    maxnorm_error = abs(vals_ux_1 - vals_f).max()
    print(ncells, maxnorm_error)
    assert maxnorm_error <= 1e-3
    
#==============================================================================
@pytest.mark.parametrize('ncells', [[200,200]])
@pytest.mark.parametrize('degree', [[2,2], [2,3], [3,3]])
@pytest.mark.parametrize('periodic', [[False, False], [True, True]])
@pytest.mark.parametrize('multiplicity', [(1, 1), (2, 2)])

def test_derham_projector_2d_hdiv_2(ncells, degree, periodic, multiplicity):

    domain = Square('Omega', bounds1 = (0,1), bounds2 = (0,1))
    domain_h = discretize(domain, ncells=ncells, periodic=periodic)
    
    multiplicity = [min(m, p) for p, m in zip(degree, multiplicity)]
    derham   = Derham(domain, ["H1", "Hdiv", "L2"])
    derham_h   = discretize(derham, domain_h, degree=degree, get_H1vec_space = True, multiplicity=multiplicity)
    P0, P1, P2, PX = derham_h.projectors()

    # Projector onto H1 space (1D interpolation)

    # Function to project
    f1  = lambda xi1, xi2 : xi1**2*(xi1-1.)**2 
    #function C0 restricted to [0,1] with periodic BC (0 at x1=0 and x1=1)
    f2  = lambda xi1, xi2 : xi2**2*(xi2-1.)**2

    # Compute the projection
    u0 = P0(f1)
    u2 = P2(f1)
    u1 = P1((f1,f2))
    ux = PX((f1,f2))

    # Create evaluation grid, and check if  u0(x) == f(x)
    xgrid = np.linspace(0, 1, num=51)
    vals_u0   = np.array([[u0(x, y) for x in xgrid] for y in xgrid])
    vals_u1_1 = np.array([[u1(x, y)[0] for x in xgrid] for y in xgrid])
    vals_u2   = np.array([[u2(x, y) for x in xgrid] for y in xgrid])
    vals_ux_1 = np.array([[ux(x, y)[0] for x in xgrid] for y in xgrid])
    vals_f    = np.array([[f1(x, y) for x in xgrid] for y in xgrid])

    # Test if max-norm of error is <= TOL
    maxnorm_error = abs(vals_u0 - vals_f).max()
    print(ncells, maxnorm_error)
    assert maxnorm_error <= 1e-3
    maxnorm_error = abs(vals_u1_1 - vals_f).max()
    print(ncells, maxnorm_error)
    assert maxnorm_error <= 1e-3
    maxnorm_error = abs(vals_u2 - vals_f).max()
    print(ncells, maxnorm_error)
    assert maxnorm_error <= 1e-3
    maxnorm_error = abs(vals_ux_1 - vals_f).max()
    print(ncells, maxnorm_error)
    assert maxnorm_error <= 1e-3
    
#==============================================================================
@pytest.mark.parametrize('ncells', [[200,200]])
@pytest.mark.parametrize('degree', [[2,2], [2,3], [3,3]])
@pytest.mark.parametrize('periodic', [[False, False], [True, False] ,[True, True]])
@pytest.mark.parametrize('multiplicity', [[1,1],[2,2]])

def test_derham_projector_2d_hcurl(ncells, degree, periodic, multiplicity):

    domain = Square('Omega', bounds1 = (0,2*np.pi), bounds2 = (0,2*np.pi))
    domain_h = discretize(domain, ncells=ncells, periodic=periodic)
    
    multiplicity = [min(m,p) for p, m in zip (degree, multiplicity)]
    derham   = Derham(domain, ["H1", "Hdiv", "L2"])
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

    # Test if max-norm of error is <= TOL
    maxnorm_error = abs(vals_u0 - vals_f).max()
    print(ncells, maxnorm_error)
    assert maxnorm_error <= 1e-3
    maxnorm_error = abs(vals_u1_1 - vals_f).max()
    print(ncells, maxnorm_error)
    assert maxnorm_error <= 1e-3
    maxnorm_error = abs(vals_u2 - vals_f).max()
    print(ncells, maxnorm_error)
    assert maxnorm_error <= 1e-3
    maxnorm_error = abs(vals_ux_1 - vals_f).max()
    print(ncells, maxnorm_error)
    assert maxnorm_error <= 1e-3
    
#==============================================================================
@pytest.mark.parametrize('ncells', [[20,20,20]])
@pytest.mark.parametrize('degree', [[2,2,2], [2,3,2], [3,3,3]])
@pytest.mark.parametrize('periodic', [[False, False, False], [True, True, True]])
@pytest.mark.parametrize('multiplicity', [[1,1,1], [1,2,2], [2,2,2]])

def test_derham_projector_3d(ncells, degree, periodic, multiplicity):

    domain = Cube('Omega', bounds1 = (0,2*np.pi), bounds2 = (0,2*np.pi), bounds3 = (0,2*np.pi))
    domain_h = discretize(domain, ncells=ncells, periodic=periodic)
    
    derham   = Derham(domain)
    #change multiplicity if higher than degree to avoid problems (case p<m doesn't work)
    multiplicity = [min(m, p) for p, m in zip(degree, multiplicity)]

    derham_h   = discretize(derham, domain_h, degree=degree, get_H1vec_space = True, multiplicity = multiplicity)
    P0, P1, P2, P3, PX = derham_h.projectors()

    # Function to project
    f1  = lambda xi1, xi2, xi3 : np.sin( xi1 + 0.5 ) * np.cos( xi2 + 0.3 ) * np.sin( 2 * xi3 )
    f2  = lambda xi1, xi2, xi3 : np.cos( xi1 + 0.5 ) * np.sin( xi2 - 0.2 ) * np.cos( xi3 )
    f3  = lambda xi1, xi2, xi3 : np.cos( xi1 + 0.7 ) * np.sin( 2*xi2 - 0.2 ) * np.cos( xi3 )

    # Compute the projection
    u0 = P0(f1)
    u3 = P3(f1)
    u1 = P1((f1,f2,f3))
    u2 = P2((f1,f2,f3))
    ux = PX((f1,f2,f3))

    # Create evaluation grid, and check if  u0(x) == f(x)
    xgrid = np.linspace(0, 2*np.pi, num=21)
    vals_u0   = np.array([[[u0(x, y, z) for x in xgrid] for y in xgrid] for z in xgrid])
    vals_u1_1 = np.array([[[u1(x, y, z)[0] for x in xgrid] for y in xgrid] for z in xgrid])
    vals_u2_1 = np.array([[[u2(x, y, z)[0] for x in xgrid] for y in xgrid] for z in xgrid])
    vals_ux_1 = np.array([[[ux(x, y, z)[0] for x in xgrid] for y in xgrid] for z in xgrid])
    vals_u3   = np.array([[[u3(x, y, z) for x in xgrid] for y in xgrid] for z in xgrid])
    vals_f    = np.array([[[f1(x, y, z) for x in xgrid] for y in xgrid] for z in xgrid])

    # Test if max-norm of error is <= TOL
    maxnorm_error = abs(vals_u0 - vals_f).max()
    print(ncells, maxnorm_error)
    assert maxnorm_error <= 0.01

    maxnorm_error = abs(vals_u1_1 - vals_f).max()
    print(ncells, maxnorm_error)
    assert maxnorm_error <= 0.01

    maxnorm_error = abs(vals_u2_1 - vals_f).max()
    print(ncells, maxnorm_error)
    assert maxnorm_error <= 0.05

    maxnorm_error = abs(vals_u3 - vals_f).max()
    print(ncells, maxnorm_error)
    assert maxnorm_error <= 0.05

    maxnorm_error = abs(vals_ux_1 - vals_f).max()
    print(ncells, maxnorm_error)
    assert maxnorm_error <= 0.02

#==============================================================================
if __name__ == '__main__':

    domain   = (0, 2*np.pi)
    degree   = 3
    periodic = True
    ncells   = [10, 20, 40, 80, 160, 320, 640]
    
    for nc in ncells:
        test_derham_projector_2d_hdiv([nc, nc], [degree, degree], [periodic, periodic], [2, 2])
    
    for nc in ncells:
        test_H1_projector_1d(domain, nc, degree, periodic, multiplicity = 2)

    nquads = degree
    for nc in ncells:
        test_L2_projector_1d(domain, nc, degree, periodic, nquads)
        
    for nc in ncells:
        test_derham_projector_2d_hdiv_2([nc, nc], [degree, degree], [periodic, periodic])
        test_derham_projector_2d_hdiv([nc, nc], [degree, degree], [periodic, periodic], 2)
        
    for nc in ncells :
        test_derham_projector_2d_hcurl([nc, nc], [degree, degree], [periodic, periodic])

    for nc in ncells[:3] :
        test_derham_projector_3d([nc, nc, nc], [degree, degree, degree], [periodic, periodic, periodic])

