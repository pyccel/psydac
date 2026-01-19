#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import time

from mpi4py import MPI
import numpy as np
import pytest

from psydac.core.bsplines import make_knots
from psydac.fem.basic     import FemField
from psydac.fem.splines   import SplineSpace
from psydac.fem.tensor    import TensorFemSpace
from psydac.ddm.cart      import DomainDecomposition

from psydac.fem.tests.utilities              import horner, random_grid
from psydac.fem.tests.splines_error_bounds   import spline_1d_error_bound
from psydac.fem.tests.analytical_profiles_1d import (AnalyticalProfile1D_Cos, AnalyticalProfile1D_Poly)
#===============================================================================
@pytest.mark.parametrize( "ncells", [1,5,10,23] )
@pytest.mark.parametrize( "degree", range(1,11) )
def test_SplineInterpolation1D_exact( ncells, degree ):

    domain   = [-1.0, 1.0]
    periodic = False

    poly_coeffs = np.random.random_sample( degree+1 ) # 0 <= c < 1
    poly_coeffs = 1.0 - poly_coeffs                   # 0 < c <= 1
    f = lambda x : horner( x, *poly_coeffs )

    grid  = random_grid( domain, ncells, 0.5 )
    space = SplineSpace( degree=degree, grid=grid, periodic=periodic )
    field = FemField( space )

    xg = space.greville
    ug = f( xg )

    space.compute_interpolant( ug, field )

    xt  = np.linspace( *domain, num=100 )
    err = np.array( [field( x ) - f( x ) for x in xt] )

    max_norm_err = np.max( abs( err ) )
    assert max_norm_err < 1.0e-13

#===============================================================================
def args_SplineInterpolation1D_cosine():
    for ncells in [5,10,23]:
        for periodic in [True, False]:
            pmax = min(ncells,9) if periodic else 9
            for degree in range(1,pmax+1):
                yield (ncells, degree, periodic)


@pytest.mark.parametrize( "ncells,degree,periodic",
    args_SplineInterpolation1D_cosine() )
def test_SplineInterpolation1D_cosine( ncells, degree, periodic ):

    f = AnalyticalProfile1D_Cos()

    grid, dx = np.linspace( *f.domain, num=ncells+1, retstep=True )
    space = SplineSpace( degree=degree, grid=grid, periodic=periodic )
    field = FemField( space )

    xg = space.greville
    ug = f.eval( xg )

    space.compute_interpolant( ug, field )
    xt  = np.linspace( *f.domain, num=100 )
    err = np.array( [field( x ) - f.eval( x ) for x in xt] )

    max_norm_err = np.max( abs( err ) )
    err_bound    = spline_1d_error_bound( f, dx, degree )

    assert max_norm_err < err_bound

#===============================================================================
@pytest.mark.mpi
@pytest.mark.parametrize( "nc1", [7,10,23] )
@pytest.mark.parametrize( "nc2", [7,10,23] )
@pytest.mark.parametrize( "deg1", range(1,5) )
@pytest.mark.parametrize( "deg2", range(1,5) )
def test_SplineInterpolation2D_parallel_exact( nc1, nc2, deg1, deg2 ):

    # Communicator, size, rank
    mpi_comm = MPI.COMM_WORLD
    mpi_size = mpi_comm.Get_size()
    mpi_rank = mpi_comm.Get_rank()

    domain1   = [-1.0, 0.8]
    periodic1 = False

    domain2   = [-0.9, 1.0]
    periodic2 = False

    # Random coefficients of 1D polynomial (identical on all processes!)
    poly_coeffs = np.random.random_sample( min(deg1,deg2)+1 ) # 0 <= c < 1
    poly_coeffs = 1.0 - poly_coeffs                           # 0 < c <= 1
    mpi_comm.Bcast( poly_coeffs, root=0 )

    # 2D exact solution: 1D polynomial of linear combination z=x1-x2/2
    f = lambda x1,x2 : horner( x1-0.5*x2, *poly_coeffs )

    # Random 1D grids (identical on all processes!)
    grid1 = random_grid( domain1, nc1, 0.1 )
    grid2 = random_grid( domain2, nc2, 0.1 )
    mpi_comm.Bcast( grid1, root=0 )
    mpi_comm.Bcast( grid2, root=0 )

    # 1D spline spaces along x1 and x2
    space1 = SplineSpace( degree=deg1, grid=grid1, periodic=periodic1 )
    space2 = SplineSpace( degree=deg2, grid=grid2, periodic=periodic2 )

    domain_decomposition = DomainDecomposition([nc1, nc2], [periodic1, periodic2], comm=mpi_comm)
    # Tensor-product 2D spline space, distributed, and field
    tensor_space = TensorFemSpace( domain_decomposition, space1, space2 )
    tensor_field = FemField( tensor_space )

    # Coordinates of Greville points (global)
    x1g = space1.greville
    x2g = space2.greville

    # Interpolation data on Greville points (distributed)
    V     = tensor_space.coeff_space
    ug    = V.zeros()
    s1,s2 = V.starts
    e1,e2 = V.ends
    n1,n2 = V.npts
    for i1 in range( s1, e1+1 ):
        for i2 in range( s2, e2+1 ):
            ug[i1,i2] = f( x1g[i1], x2g[i2] )

    ug.update_ghost_regions()

    # Compute 2D spline interpolant
    tensor_space.compute_interpolant( ug, tensor_field )

    #------------
    # DIAGNOSTICS
    #------------

    # Verify that solution is exact at Greville points
    err = V.zeros()
    for i1 in range( s1, e1+1 ):
        for i2 in range( s2, e2+1 ):
            err[i1,i2] = ug[i1,i2] - tensor_field( x1g[i1], x2g[i2] )
    interp_error = abs( err[:,:] ).max()

    # Compute L2 norm of error
    integrand = lambda x1,x2: (f(x1,x2)-tensor_field(x1,x2))**2
    l2_error  = np.sqrt( tensor_space.integral( integrand ) )

    # Print some information to terminal
    for i in range( mpi_size ):
        if i == mpi_rank:
            print( '--------------------------------------------------' )
            print( ' RANK = {}'.format( mpi_rank ) )
            print( '--------------------------------------------------' )
            print( '> Degree        :: [{:2d},{:2d}]'.format( deg1, deg2 ) )
            print( '> Ncells        :: [{:2d},{:2d}]'.format(  nc1,  nc2 ) )
            print( '> Nbasis        :: [{:2d},{:2d}]'.format(   n1,   n2 ) )
            print( '> Starts        :: [{:2d},{:2d}]'.format(   s1,   s2 ) )
            print( '> Ends          :: [{:2d},{:2d}]'.format(   e1,   e2 ) )
            print( '> Interp. error :: {:.2e}'.format( interp_error ) )
            print( '> L2 error      :: {:.2e}'.format( l2_error ) )
            print( '', flush=True )
            time.sleep( 0.1 )
        mpi_comm.Barrier()

    # Verify that error is only caused by finite precision arithmetic
    assert interp_error < 1.0e-13
    assert     l2_error < 1.0e-13

#===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================
if __name__ == '__main__':
    test_SplineInterpolation2D_parallel_exact( 10, 16, 3, 5 )
