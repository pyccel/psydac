#coding: utf-8

import numpy as np

from spl.bsplines.bsplines_non_uniform import (find_span, basis_funs)

#==============================================================================
def test_find_span():

    nc  = 10
    p   = 3
    eps = 1e-12

    grid  = np.linspace( 0, 1, nc+1 )
    knots = np.r_[ [grid[0]]*p, grid, [grid[-1]]*p ]

    for i,xi in enumerate( grid ):
        assert find_span( knots, p, x=xi-eps ) == p + max( 0,  i-1 )
        assert find_span( knots, p, x=xi     ) == p + min( i, nc-1 )
        assert find_span( knots, p, x=xi+eps ) == p + min( i, nc-1 )

#==============================================================================
def test_basis_funs():

    nc  = 10
    p   = 3
    tol = 1e-14

    grid  = np.linspace( 0, 1, nc+1 )
    knots = np.r_[ [grid[0]]*p, grid, [grid[-1]]*p ]

    xx = np.linspace( 0, 1, 101 )
    for x in xx:
        span  =  find_span( knots, p, x )
        basis = basis_funs( knots, p, x, span )
        assert len( basis ) == p+1
        assert np.all( basis >= 0 )
        assert abs( sum( basis ) - 1.0 ) < tol

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
    m = 3

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
    for i,x in enumerate( xx ):
        span  =  find_span( knots, p, x )
        basis = basis_funs( knots, p, x, span )
        yy[i,span-p:span+1] = basis

    # Create figure and plot basis functions
    fig, ax = plt.subplots( 1, 1 )
    ax.plot( xx, yy )

    # Plot knot sequence
    values, counts = np.unique( knots, return_counts=True )
    y = np.concatenate( [np.linspace(0,1,c,endpoint=True) for c in counts] )
    ax.plot( knots, y, 'kx' )

    # Show figure and keep it open if necessary
    ax.grid()
    fig.show()

    import __main__ as main
    if hasattr( main, '__file__' ):
        try:
           __IPYTHON__
        except NameError:
            plt.show()
