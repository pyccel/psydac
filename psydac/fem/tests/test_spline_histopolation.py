import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from psydac.core.bsplines import make_knots
from psydac.fem.basic     import FemField
from psydac.fem.splines   import SplineSpace
from psydac.fem.tensor    import TensorFemSpace

from psydac.fem.tests.utilities              import horner, random_grid
from psydac.fem.tests.splines_error_bounds   import spline_1d_error_bound
from psydac.fem.tests.analytical_profiles_1d import (AnalyticalProfile1D_Cos,
                                                  AnalyticalProfile1D_Poly)

#==============================================================================

ncells = 10
degree = 7

#==============================================================================

domain   = (-1.0, 1.0)
periodic = False

poly_coeffs = np.random.random_sample( degree+1 ) # 0 <= c < 1
poly_coeffs = 1.0 - poly_coeffs                   # 0 < c <= 1
f1 = lambda x : horner( x, *poly_coeffs )


grid = random_grid( domain, ncells, 0.5 )
V1_h = SplineSpace( degree=degree, grid=grid, periodic=periodic, basis='M' )
f1_h = FemField( V1_h )

xg = V1_h.ext_greville
c1 = np.array([quad(f1, xg[i], xg[i+1])[0] for i in range(len(xg)-1)])
V1_h.compute_interpolant( c1, f1_h )

#==============================================================================
# Diagnostics
#==============================================================================
xt  = np.linspace( *domain, num=100 )
err = np.array( [f1_h( x ) - f1( x ) for x in xt] )

max_norm_err = np.max( abs( err ) )
print("Maximum error on evaluation grid: {}".format(max_norm_err))

fig, ax = plt.subplots(1, 1)
ax.plot(xt, f1(xt), label='f1(x)')
ax.plot(xt, [f1_h(x) for x in xt], '.', label='f1_h(x)')
ax.grid(True)
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.legend()
fig.show()

import __main__ as main
if hasattr( main, '__file__' ):
    try:
       __IPYTHON__
    except NameError:
        plt.show()
