# coding: utf-8
# Copyright 2018 Yaman Güçlü

import cunumpy as xp

#===============================================================================
def horner( x, *poly_coeffs ):
    """ Use Horner's Scheme to evaluate a polynomial
        of coefficients *poly_coeffs at location x.
    """
    # Spline metadata (notably Greville abscissas) is intentionally host-side.
    # Convert it at the numerical API boundary so CuPy coefficients and NumPy
    # coordinates can be combined just as they can under NumPy.
    x = xp.asarray(x)
    p = 0
    for c in poly_coeffs[::-1]:
        p = p*x + c
    return p

#===============================================================================
def random_grid( domain, ncells, random_fraction ):
    """ Create random grid over 1D domain with given number of cells.
    """
    # Create uniform grid on [0,1]
    x = xp.linspace( 0.0, 1.0, ncells+1 )

    # Apply random displacement to all points, then sort grid
    x += (xp.random.random_sample( ncells+1 )-0.5) * (random_fraction/ncells)
    x.sort()

    # Apply linear transformation y=m*x+q to match domain limits
    xa, xb = x[0], x[-1]
    ya, yb = domain
    m = (   yb-ya   )/(xb-xa)
    q = (xb*ya-xa*yb)/(xb-xa)
    y = m*x + q

    # Avoid possible round-off
    y[0], y[-1] = domain

    return y

#===============================================================================
def falling_factorial( x, n ):
  """ Calculate falling factorial of x.
      [https://en.wikipedia.org/wiki/Falling_and_rising_factorials]
  """
  c = 1
  for k in range(n):
      c *= x-k
  return c
