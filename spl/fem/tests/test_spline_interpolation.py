# coding: utf-8
# Copyright 2018 Yaman Güçlü

import pytest
import numpy as np

from spl.fem.splines   import SplineSpace
from spl.fem.basic     import FemField
from spl.core.bsplines import make_knots

from spl.fem.tests.utilities              import horner, random_grid
from spl.fem.tests.splines_error_bounds   import spline_1d_error_bound
from spl.fem.tests.analytical_profiles_1d import (AnalyticalProfile1D_Cos,
                                                  AnalyticalProfile1D_Poly)

#===============================================================================
@pytest.mark.serial
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
    field = FemField( space, 'f' )

    xg = space.greville
    ug = f( xg )

    space.compute_interpolant( ug, field )

    xt  = np.linspace( *domain, num=100 )
    err = np.array( [field( x ) - f( x ) for x in xt] )

    max_norm_err = np.max( abs( err ) )
    assert max_norm_err < 1.0e-14

#===============================================================================
def args_SplineInterpolation1D_cosine():
    for ncells in [5,10,23]:
        for periodic in [True, False]:
            pmax = min(ncells,9) if periodic else 9
            for degree in range(1,pmax+1):
                yield (ncells, degree, periodic)

@pytest.mark.serial
@pytest.mark.parametrize( "ncells,degree,periodic",
    args_SplineInterpolation1D_cosine() )

def test_SplineInterpolation1D_cosine( ncells, degree, periodic ):

    f = AnalyticalProfile1D_Cos()

    grid  = random_grid( f.domain, ncells, 0.5 )
    space = SplineSpace( degree=degree, grid=grid, periodic=periodic )
    field = FemField( space, 'f' )

    xg = space.greville
    ug = f.eval( xg )

    space.compute_interpolant( ug, field )

    xt  = np.linspace( *f.domain, num=100 )
    err = np.array( [field( x ) - f.eval( x ) for x in xt] )

    max_norm_err = np.max( abs( err ) )
    err_bound    = spline_1d_error_bound( f, np.diff( grid ).max(), degree )

    assert max_norm_err < err_bound

# #===============================================================================
# @pytest.mark.parametrize( "nc1", [1,5,10,23] )
# @pytest.mark.parametrize( "nc2", [1,5,10,23] )
# @pytest.mark.parametrize( "deg1", range(1,5) )
# @pytest.mark.parametrize( "deg2", range(1,5) )
# 
# def test_SplineInterpolation2D_exact( nc1, nc2, deg1, deg2 ):
# 
#     domain1   = [-1.0, 0.8]
#     periodic1 = False
# 
#     domain2   = [-0.9, 1.0]
#     periodic2 = False
# 
#     degree = min( deg1, deg2 )
# 
#     poly_coeffs = np.random.random_sample( degree+1 ) # 0 <= c < 1
#     poly_coeffs = 1.0 - poly_coeffs                   # 0 < c <= 1
#     f = lambda x1,x2 : horner( x1-0.5*x2, *poly_coeffs )
# 
#     # Along x1
#     breaks1 = random_grid( domain1, nc1, 0.0 )
#     knots1  = make_knots( breaks1, deg1, periodic1 )
#     basis1  = BSplines( knots1, deg1, periodic1 )
# 
#     # Along x2
#     breaks2 = random_grid( domain2, nc2, 0.0 )
#     knots2  = make_knots( breaks2, deg2, periodic2 )
#     basis2  = BSplines( knots2, deg2, periodic2 )
# 
#     # 2D spline and interpolator on tensor-product space
#     spline  = Spline2D( basis1, basis2 )
#     interp  = SplineInterpolator2D( basis1, basis2 )
# 
#     x1g = basis1.greville
#     x2g = basis2.greville
#     ug  = f( *np.meshgrid( x1g, x2g, indexing='ij' ) )
# 
#     interp.compute_interpolant( ug, spline )
# 
#     x1t = np.linspace( *domain1, num=100 )
#     x2t = np.linspace( *domain2, num=100 )
#     err = spline.eval( x1t, x2t ) - f( *np.meshgrid( x1t, x2t, indexing='ij' ) )
# 
#     max_norm_err = np.max( abs( err ) )
#     assert max_norm_err < 2.0e-14
