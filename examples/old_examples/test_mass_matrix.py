#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
Post-processing script for example `poisson_2d_mapping.py`.

Use the mass matrix M to compute the L^2 projection f2 of a callable function
f1 onto the tensor-product spline space V with mapping. Note that both f1 and
f2 are defined over the logical domain Omega.

Afterwards check the L^2 norm of the error between f1 and f2.

USAGE
=====

$ ipython

In [1]: run poisson_2d_mapping.py -t target -n 10 20 -s -c

In [2]: run -i visualize_matrices.py

In [3]: run -i test_mass_matrix.py

"""

from psydac.linalg.solvers import inverse

globals().update(namespace)

#===============================================================================

f1 = lambda s, t: 1.0 + s**3

rhs = assemble_rhs(V, mapping, f1)

M_inv = inverse(M, 'cg', tol=1e-10, maxiter=100, verbose=True)
sol   = M_inv @ rhs
info  = M_inv.get_info()

for key, value in info.items():
    print(f'{key:8s} :: {value}')

f2 = FemField(V, coeffs=sol)

# Compute L2 norm of error
sqrt_g    = lambda *x: np.sqrt(mapping.metric_det(*x))
integrand = lambda *x: (f1(*x) - f2(*x))**2 * sqrt_g(*x)
l2_error  = np.sqrt(V.integral(integrand))

print(f'L2 error :: {l2_error:.2e}')
