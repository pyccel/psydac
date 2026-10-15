#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#

# This file is the Python translation of a Selalib Fortran module:
# 'selalib/src/splines/tests/m_splines_error_bounds.F90'

__all__ = ('spline_1d_error_bound', 'spline_1d_error_bound_on_deriv',
           'spline_2d_error_bound', 'spline_2d_error_bounds_on_grad')

#===============================================================================

k = (     1 /          2.0,
          1 /          8.0,
          1 /         24.0,
          5 /        384.0,
          1 /        240.0,
         61 /      46080.0,
         17 /      40320.0,
        277 /    2064384.0,
         31 /     725760.0,
      50521 / 3715891200.0 )

def tihomirov_error_bound( h, deg, norm_f ):
    """
    Error bound in max norm for spline interpolation of periodic functions from:

    V M Tihomirov 1969 Math. USSR Sb. 9 275
    https://doi.org/10.1070/SM1969v009n02ABEH002052 (page 286, bottom)

    Yu. S. Volkov and Yu. N. Subbotin
    https://doi.org/10.1134/S0081543815020236 (equation 14)

    Also applicable to first derivative by passing deg-1 instead of deg
    Volkov & Subbotin 2015, eq. 15

    Parameters
    ----------
    h : float
        Cell width

    deg : int
        Degree of spline S

    norm_f : float
        Max of function f(x) (or its derivative) over domain

    Result
    ------
    norm_e : float
        Max of error $E(x):=f(x)-S(x)$ over domain

    """
    norm_e = k[deg] * h**deg * norm_f

    return norm_e

#===============================================================================
def spline_1d_error_bound( profile_1d, dx, deg ):
    """
    Compute error bound for spline approximation of 1D analytical profile.

    Parameters
    ----------
    profile_1d : 1D analytical profile
        Must provide 'max_norm( n )' method to compute max norm of n-th
        derivative of profile over domain.

    dx : float
        Grid spacing.

    deg : int
        Spline degree.

    Result
    ------
    max_error : float
        Error bound: max-norm of error over domain should be smaller than this.

    """
    max_norm  = profile_1d.max_norm( deg+1 )
    max_error = tihomirov_error_bound( dx, deg, max_norm )
    return max_error

#===============================================================================
def spline_1d_error_bound_on_deriv( profile_1d, dx, deg ):
    """ Compute error bound on first derivative, for spline approximation of 1D
        analytical profile. Signature is identical to 'spline_1d_error_bound'.
    """
    max_norm  = profile_1d.max_norm( deg+1 )
    max_error = tihomirov_error_bound( dx, deg-1, max_norm )
    return max_error

#===============================================================================
def spline_2d_error_bound( profile_2d, dx1, dx2, deg1, deg2 ):
    """
    Compute error bound for spline approximation of 2D analytical profile.

    Parameters
    ----------
    profile_2d : 2D analytical profile
        Must provide 'max_norm( n1,n2 )' method to compute max norm of its
        mixed derivative of degree (n1,n2) over domain.

    dx1 : float
        Grid spacing along 1st dimension.

    dx2 : float
        Grid spacing along 2nd dimension.

    deg1 : int
        Spline degree along 1st dimension.

    deg2 : int
        Spline degree along 2nd dimension.

    Result
    ------
    max_error : float
        Error bound: max-norm of error over domain should be smaller than this.

    """
    # Max norm of highest partial derivatives in x1 and x2 of analytical profile
    max_norm1 = profile_2d.max_norm( deg1+1, 0      )
    max_norm2 = profile_2d.max_norm( 0     , deg2+1 )

    # Error bound on function value
    max_error = f_tihomirov_error_bound( dx1, deg1, max_norm1 ) \
              + f_tihomirov_error_bound( dx2, deg2, max_norm2 )

    # Empirical correction: for linear interpolation increase estimate by 5%
    if (deg1 == 1 or deg2 == 1):
        max_error = 1.05 * max_error

    return max_error

#===============================================================================
def spline_2d_error_bounds_on_grad( profile_2d, dx1, dx2, deg1, deg2 ):
    """
    Compute error bound on gradient, for spline approximation of 2D
    analytical profile. Signature is identical to 'spline_2d_error_bound'.

    """
    # Max norm of highest partial derivatives in x1 and x2 of analytical profile
    max_norm1 = profile_2d.max_norm( deg1+1, 0      )
    max_norm2 = profile_2d.max_norm( 0     , deg2+1 )

    # Error bound on x1-derivative
    max_error1 = f_tihomirov_error_bound( dx1, deg1-1, max_norm1 ) \
               + f_tihomirov_error_bound( dx2, deg2  , max_norm2 )

    # Error bound on x2-derivative
    max_error2 = f_tihomirov_error_bound( dx1, deg1  , max_norm1 ) \
               + f_tihomirov_error_bound( dx2, deg2-1, max_norm2 )

    return (max_error1, max_error2)
