#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import pytest
import numpy as np

from psydac.core.bsplines import ( find_span,
        basis_funs,
        basis_funs_1st_der,
        basis_funs_all_ders,
        make_knots,
        elevate_knots,
        greville,
        collocation_matrix,
        histopolation_matrix,
        cell_index)

from psydac.fem.tests.utilities import random_grid

# TODO: add unit tests for
#  - make_knots
#  - elevate_knots
#  - greville

#==============================================================================
@pytest.mark.parametrize( 'lims', ([0,1], [-2,3]) )
@pytest.mark.parametrize( 'nc', (10, 18, 33) )
@pytest.mark.parametrize( 'p' , (0,1,2,3,7,10) )

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
@pytest.mark.parametrize( 'p' , (0,1,2,3,7,10) )

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
@pytest.mark.parametrize( 'p' , (0,1,2,3,7,10) )

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
@pytest.mark.parametrize( 'p' , (0,1,2,3,7,10) )

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
@pytest.mark.parametrize( 'p' , (0,1,2,3,7,8) )
@pytest.mark.parametrize( 'periodic' , (True, False) )

# TODO: improve checks
def test_collocation_matrix(lims, nc, p, periodic, tol=1e-13):

    breaks = random_grid(domain=lims, ncells=nc, random_fraction=0.3)
    knots  = make_knots(breaks, p, periodic)
    xgrid  = greville(knots, p, periodic)
    mat    = collocation_matrix(knots, p, periodic, normalization='B', xgrid=xgrid)

    acceptable_nonzeros_in_row = [p, p+1] if periodic else [1, p, p+1]

    for row in mat:
        assert all( row >= 0.0 )
        assert abs( sum( row ) - 1.0 ) < tol
        assert (abs(row) > tol).sum() in acceptable_nonzeros_in_row

#==============================================================================
@pytest.mark.parametrize( 'lims', ([0,1], [-2,3]) )
@pytest.mark.parametrize( 'nc', (10, 18, 33) )
@pytest.mark.parametrize( 'p' , (0,1,2,3,4,5,6) )
@pytest.mark.parametrize( 'periodic' , (True, False) )

# TODO: improve checks
def test_histopolation_matrix(lims, nc, p, periodic, tol=1e-13):

    breaks = random_grid(domain=lims, ncells=nc, random_fraction=0.3)
    knots  = make_knots(breaks, p, periodic)
    xgrid  = greville(elevate_knots(knots, p, periodic), p+1, periodic)
    mat    = histopolation_matrix(knots, p, periodic, normalization='M', xgrid=xgrid)

    for col in mat.T:
        assert all( col >= 0.0 )
        assert abs( sum( col ) - 1.0 ) < tol
#        assert (abs(col) > tol).sum() <= 2*p + 1

#==============================================================================
@pytest.mark.parametrize("i_grid, expected", [([0.05, 0.15, 0.21, 0.05, 0.55],[0, 1, 2, 0, 5]),
                                              ([0.1, 0.1, 0.0, 0.4, 0.4, 0.9, 0.9], [0, 1, 0, 3, 4, 8, 9]),
                                              ([0., 0.1, 0.1, 1], [0, 0, 1, 9])])
def test_cell_index(i_grid, expected):
    breaks = np.array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    out = cell_index(breaks, np.asarray(i_grid))
    assert np.array_equal(expected, out)

#==============================================================================
# SCRIPT FUNCTIONALITY: PLOT BASIS FUNCTIONS
#==============================================================================
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    np.set_printoptions(linewidth=130)

    # Domain limits, number of cells and spline degree
    lims = [0, 1]
    nc   = 4
    p    = 3

    # Repeat internal knot with index 'k' for 'm' times
    k = 5
    m = 2

    # Grid (breakpoints) and clamped knot sequence
    grid  = np.linspace( *lims, num=nc+1 )
    grid[1:-1] += 0.1*np.random.random_sample(nc-1) - 0.05  # Perturb internal breakpoints
    knots = np.r_[ [grid[0]]*p, grid, [grid[-1]]*p ]

    # Insert repeated internal knot
    knots = list( knots )
    knots = knots[:k] + [knots[k]]*m + knots[k+1:]
    knots = np.array( knots )

    # Number of basis functions
    nb = len(knots)-p-1

    # Evaluation grid
    xx = np.linspace( *lims, num=501 )

    # Compute values of each basis function on evaluation grid
    yy = np.zeros( (len(xx), nb) )
    zz = np.zeros( (len(xx), nb) )
    for i,x in enumerate( xx ):
        span = find_span( knots, p, x )
        yy[i,span-p:span+1] = basis_funs        ( knots, p, x, span )
        zz[i,span-p:span+1] = basis_funs_1st_der( knots, p, x, span )

    # Check partition of unity on evaluation grid
    unity = yy.sum(axis=1)
    print("\nPartition of unity on evaluation grid:")
    print(unity)

    # ...
    # Integrals of each B-spline over domain (theoretical values)
    #
    #   \int B(i) dx = length(support(B)) / (p + 1) = (T[i + p + 1] - T[i]) / (p + 1)
    #
    integrals_theory = np.array([(knots[i+p+1] - knots[i]) / (p+1) for i in range(nb)])

    # Integrals of each B-spline over domain (Gaussian quadrature)
    from psydac.utilities.quadratures import gauss_legendre
    from psydac.core.bsplines import quadrature_grid, basis_ders_on_quad_grid
    from psydac.core.bsplines import elements_spans

    u, w = gauss_legendre(p + 1)
    quad_x, quad_w = quadrature_grid(grid, u, w)
    quad_basis = basis_ders_on_quad_grid(knots, p, quad_x, nders=0, normalization='B')
    integrals  = np.zeros(nb)
    for ie, span in enumerate(elements_spans(knots, p)):
        integrals[span-p:span+1] += np.dot(quad_basis[ie, :, 0, :], quad_w[ie, :])

    # Compare theory results with computed integrals
    print("\nIntegrals of basis functions over domain:")
    print("Theory  :", integrals_theory)
    print("Computed:", integrals)
    # ...

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
