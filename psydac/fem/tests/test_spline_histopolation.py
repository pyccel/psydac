#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import pytest
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from psydac.fem.basic   import FemField
from psydac.fem.splines import SplineSpace
#from psydac.fem.tensor  import TensorFemSpace

from psydac.fem.tests.utilities              import horner, random_grid
from psydac.fem.tests.splines_error_bounds   import spline_1d_error_bound
from psydac.fem.tests.analytical_profiles_1d import AnalyticalProfile1D_Cos

#==============================================================================
def histopolate_polynomial(basis, ncells, degree):

    domain   = (-1.0, 1.0)
    periodic = False

    # Polynomial to be approximated
    poly_coeffs = np.random.random_sample( degree+1 ) # 0 <= c < 1
    poly_coeffs = 1.0 - poly_coeffs                   # 0 < c <= 1
    f = lambda x : horner( x, *poly_coeffs )

    # Define spline space and field
    grid = random_grid( domain, ncells, 0.5 )
    Vh = SplineSpace( degree=degree, grid=grid, periodic=periodic, basis=basis )
    fh = FemField( Vh )

    # Compute histopolant
    xg = Vh.ext_greville
    Ig = np.array([quad(f, xg[i], xg[i+1])[0] for i in range(len(xg)-1)])
    Vh.compute_histopolant(Ig, fh)

    return domain, f, fh

#==============================================================================
@pytest.mark.parametrize('basis', ['B', 'M'])
@pytest.mark.parametrize('ncells', [10, 20, 33])
@pytest.mark.parametrize('degree', [2, 5, 7])
def test_histopolation_exact(basis, ncells, degree, num_pts=100, tol=1e-11):

    domain, f, fh = histopolate_polynomial(basis, ncells, degree)

    # Compare to exact solution
    x  = np.linspace(*domain, num=num_pts)
    y  = f(x)
    yh = np.array([fh(xi) for xi in x])

    assert np.allclose(yh, y, rtol=tol, atol=tol)

#==============================================================================
@pytest.mark.parametrize('basis', ['B', 'M'])
@pytest.mark.parametrize('ncells', [10, 20, 33])
@pytest.mark.parametrize('degree', [2, 5, 7])
@pytest.mark.parametrize('periodic', [True, False])
def test_histopolation_cosine(basis, ncells, degree, periodic, num_pts=100):

    # Function to be approximated
    # TODO: write function and domain explicitly
    f = AnalyticalProfile1D_Cos()

    # Define spline space and field
    grid, dx = np.linspace(*f.domain, num=ncells+1, retstep=True)
    Vh = SplineSpace(degree=degree, grid=grid, periodic=periodic)
    fh = FemField(Vh)

    # Compute histopolant
    xg = Vh.histopolation_grid
    Ig = np.array([quad(f.eval, xl, xr)[0] for xl, xr in zip(xg[:-1], xg[1:])])
    Vh.compute_histopolant(Ig, fh)

    # Compare to exact solution
    x  = np.linspace(*f.domain, num=num_pts)
    y  = f.eval(x)
    yh = np.array([fh(xi) for xi in x])

    max_norm_err = np.max(abs(y - yh))
    err_bound    = spline_1d_error_bound(f, dx, degree)

    assert max_norm_err < err_bound

#==============================================================================
# Diagnostics
#==============================================================================
def compare_and_plot(domain, f, fh, num_pts=100):

    x  = np.linspace(*domain, num=num_pts)
    y  = f(x)
    yh = np.array([fh(xi) for xi in x])

    max_norm_err = np.max(abs(yh - y))
    print("Maximum error on evaluation grid: {}".format(max_norm_err))

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y, label='f(x)')
    ax.plot(x, yh, '.', label='f1_h(x)')
    ax.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.legend()
    fig.show()

#==============================================================================
# Diagnostics
#==============================================================================
if __name__ == '__main__':

    domain, f, fh = histopolate_polynomial(basis='B', ncells=10, degree=7)
    compare_and_plot(domain, f, fh)

    domain, f, fh = histopolate_polynomial(basis='M', ncells=10, degree=7)
    compare_and_plot(domain, f, fh)

    import __main__ as main
    if hasattr( main, '__file__' ):
        try:
           __IPYTHON__
        except NameError:
            plt.show()
