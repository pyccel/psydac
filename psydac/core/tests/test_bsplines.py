#coding: utf-8

import pytest
import numpy as np

from psydac.core.bsplines import ( find_span,
        basis_funs,
        basis_funs_1st_der,
        basis_funs_all_ders,
        collocation_matrix )

#==============================================================================
@pytest.mark.parametrize( 'lims', ([0,1], [-2,3]) )
@pytest.mark.parametrize( 'nc', (10, 18, 33) )
@pytest.mark.parametrize( 'p' , (1,2,3,7,10) )

def test_find_span( lims, nc, p, eps=1e-12 ):

    grid  = np.linspace( *lims, num=nc+1 )
    knots = np.r_[ [grid[0]]*p, grid, [grid[-1]]*p ]

    for i,xi in enumerate( grid ):
        assert find_span( knots, p, x=xi-eps ) == p + max( 0,  i-1 )
        assert find_span( knots, p, x=xi     ) == p + min( i, nc-1 )
        assert find_span( knots, p, x=xi+eps ) == p + min( i, nc-1 )

#==============================================================================
@pytest.mark.parametrize( 'lims', ([0,1], [-2,3]) )
@pytest.mark.parametrize( 'nc', (10, 18, 33) )
@pytest.mark.parametrize( 'p' , (1,2,3,7,10) )

def test_basis_funs( lims, nc, p, tol=1e-14 ):

    grid  = np.linspace( *lims, num=nc+1 )
    knots = np.r_[ [grid[0]]*p, grid, [grid[-1]]*p ]

    xx = np.linspace( *lims, num=101 )
    for x in xx:
        span  =  find_span( knots, p, x )
        basis = basis_funs( knots, p, x, span )
        assert len( basis ) == p+1
        assert np.all( basis >= 0 )
        assert abs( sum( basis ) - 1.0 ) < tol

#==============================================================================
@pytest.mark.parametrize( 'lims', ([0,1], [-2,3]) )
@pytest.mark.parametrize( 'nc', (10, 18, 33) )
@pytest.mark.parametrize( 'p' , (1,2,3,7,10) )

def test_basis_funs_1st_der( lims, nc, p, tol=1e-14 ):

    grid  = np.linspace( *lims, num=nc+1 )
    knots = np.r_[ [grid[0]]*p, grid, [grid[-1]]*p ]

    xx = np.linspace( *lims, num=101 )
    for x in xx:
        span = find_span( knots, p, x )
        ders = basis_funs_1st_der( knots, p, x, span )
        assert len( ders ) == p+1
        assert abs( sum( ders ) ) < tol

#==============================================================================
@pytest.mark.parametrize( 'lims', ([0,1], [-2,3]) )
@pytest.mark.parametrize( 'nc', (10, 18, 33) )
@pytest.mark.parametrize( 'p' , (1,2,3,7,10) )

def test_basis_funs_all_ders( lims, nc, p, tol=1e-14 ):

    # Maximum derivative required
    n = p+2

    grid, dx = np.linspace( *lims, num=nc+1, retstep=True )
    knots = np.r_[ [grid[0]]*p, grid, [grid[-1]]*p ]

    xx = np.linspace( *lims, num=101 )
    for x in xx:
        span = find_span( knots, p, x )
        ders = basis_funs_all_ders( knots, p, x, span, n )

        # Test output array
        assert ders.shape == (1+n,1+p)
        assert ders.dtype == np.dtype( float )

        # Test 0th derivative
        der0 = basis_funs( knots, p, x, span )
        assert np.allclose( ders[0,:], der0, rtol=1e-15, atol=1e-15 )
        assert np.all( ders[0,:] >= 0.0 )

        # Test 1st derivative
        der1 = basis_funs_1st_der( knots, p, x, span )
        assert np.allclose( ders[1,:], der1, rtol=1e-15, atol=1e-15/dx )

        # Test 2nd to n-th derivatives
        for i in range(2,n+1):
            assert abs( ders[i,:].sum() ) <= tol * abs( ders[i,:] ).max()

        # Test that all derivatives of degree > p are zero
        assert np.all( ders[p+1:,:] == 0.0 )

#==============================================================================
@pytest.mark.parametrize( 'lims', ([0,1], [-2,3]) )
@pytest.mark.parametrize( 'nc', (10, 18, 33) )
@pytest.mark.parametrize( 'p' , (1,2,3,7,10) )

# TODO: construct knots from grid
# TODO: evaluate on Greville points
# TODO: improve checks
def test_collocation_matrix_non_periodic( lims, nc, p, tol=1e-14 ):

    grid, dx = np.linspace( *lims, num=nc+1, retstep=True )
    knots = np.r_[ [grid[0]]*p, grid, [grid[-1]]*p ]

    mat = collocation_matrix( knots, p, grid, periodic=False )

    for row in mat:
        assert all( row >= 0.0 )
        assert len( row.nonzero()[0] ) in [1,p,p+1]
        assert abs( sum( row ) - 1.0 ) < tol

#==============================================================================
@pytest.mark.parametrize( 'lims', ([0,1], [-2,3]) )
@pytest.mark.parametrize( 'nc', (10, 18, 33) )
@pytest.mark.parametrize( 'p' , (1,2,3,7,8) )

# TODO: construct knots from grid
# TODO: evaluate on Greville points
# TODO: improve checks
def test_collocation_matrix_periodic( lims, nc, p, tol=1e-14 ):

    grid, dx = np.linspace( *lims, num=nc+1, retstep=True )
    period = lims[1]-lims[0]
    knots  = np.r_[ grid[-p-1:-1]-period, grid, grid[1:1+p]+period ]

    mat = collocation_matrix( knots, p, grid[:-1], periodic=True )

    for row in mat:
        assert all( row >= 0.0 )
        assert len( row.nonzero()[0] ) in [p,p+1]
        assert abs( sum( row ) - 1.0 ) < tol

#==============================================================================
# SCRIPT FUNCTIONALITY: PLOT BASIS FUNCTIONS
#==============================================================================
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Domain limits, number of cells and spline degree
    lims = [0, 1]
    nc   = 4
    p    = 3

    # Repeat internal knot with index 'k' for 'm' times
    k = 5
    m = 2

    # Grid (breakpoints) and clamped knot sequence
    grid  = np.linspace( *lims, num=nc+1 )
    knots = np.r_[ [grid[0]]*p, grid, [grid[-1]]*p ]

    # Insert repeated internal knot
    knots = list( knots )
    knots = knots[:k] + [knots[k]]*m + knots[k+1:]
    knots = np.array( knots )

    # Evaluation grid
    xx = np.linspace( *lims, num=501 )

    # Compute values of each basis function on evaluation grid
    yy = np.zeros( (len(xx),len(knots)-p-1) )
    zz = np.zeros( (len(xx),len(knots)-p-1) )
    for i,x in enumerate( xx ):
        span = find_span( knots, p, x )
        yy[i,span-p:span+1] = basis_funs        ( knots, p, x, span )
        zz[i,span-p:span+1] = basis_funs_1st_der( knots, p, x, span )

    # Create figure
    fig, axes = plt.subplots( 2, 1, sharex=True )

    # Plot basis functions
    axes[0].plot( xx, yy )
    axes[0].set_title( "Basis functions y=B(x)" )
    axes[0].set_xlabel( 'x' )
    axes[0].set_ylabel( 'y', rotation='horizontal' )

    # Plot first derivative of basis functions
    axes[1].plot( xx, zz )
    axes[1].set_title( "First derivative z=B'(x)" )
    axes[1].set_xlabel( 'x' )
    axes[1].set_ylabel( 'z', rotation='horizontal' )

    # Plot knot sequence and add grid
    values, counts = np.unique( knots, return_counts=True )
    y = np.concatenate( [np.linspace(0,1,c,endpoint=True) for c in counts] )
    for ax in axes:
        ax.plot( knots, y, 'ko', mew=1.0, mfc='None' )
        ax.grid()

    # Show figure and keep it open if necessary
    fig.tight_layout()
    fig.show()

    import __main__ as main
    if hasattr( main, '__file__' ):
        try:
           __IPYTHON__
        except NameError:
            plt.show()
