#coding: utf-8

import pytest
import numpy as np

from spl.bsplines.bsplines_non_uniform import (find_span, basis_funs,
        basis_funs_1st_der)

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
