import numpy as np

from spl.bsplines.bsplines_non_uniform import (find_span, basis_funs)

#==============================================================================
def test_find_span():

    nc  = 10
    p   = 3
    eps = 1e-12

    grid  = np.linspace( 0, 1, nc+1, dtype=float )
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
