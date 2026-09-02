#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
Unit tests for `TensorFemSpace.integral()`, the natural directly-testable
consumer of `TensorFemSpace.get_assembly_grids()` (which builds one
`FemAssemblyGrid` per direction). These tests do not go through
`discretize()`/assembly at all; they only check that composite Gauss-Legendre
quadrature over the space's mesh of elements integrates polynomials exactly.
"""
import numpy as np
import pytest

from psydac.ddm.cart    import DomainDecomposition
from psydac.fem.splines import SplineSpace
from psydac.fem.tensor  import TensorFemSpace

#==============================================================================
def make_tensor_fem_space(ncells, degree, periodic):
    breaks = [np.linspace(0., 1., n + 1) for n in ncells]
    spaces = [SplineSpace(degree=p, grid=b, periodic=per)
              for p, b, per in zip(degree, breaks, periodic)]
    domain_decomposition = DomainDecomposition(list(ncells), list(periodic))
    return TensorFemSpace(domain_decomposition, *spaces)

#==============================================================================
@pytest.mark.parametrize('ncells', [(3,), (3, 4), (2, 3, 2)])
@pytest.mark.parametrize('nq',     [2, 3])
def test_integral_exact_for_polynomials(ncells, nq):

    ldim   = len(ncells)
    degree = (2,) * ldim
    nquads = (nq,) * ldim

    V = make_tensor_fem_space(ncells, degree, periodic=(False,) * ldim)

    # Gauss-Legendre with `nq` points per direction integrates a polynomial
    # of degree <= 2*nq - 1 exactly. Use degree `2*nq - 1` in each variable.
    power = 2 * nq - 1
    f = lambda *x: np.prod([xi**power for xi in x])

    approx  = V.integral(f, nquads=nquads)
    exact   = (1.0 / (power + 1)) ** ldim
    assert approx == pytest.approx(exact)

#==============================================================================
@pytest.mark.parametrize('ncells', [(3,), (2, 3)])
def test_integral_default_nquads_constant_function(ncells):

    ldim   = len(ncells)
    degree = (2,) * ldim

    V = make_tensor_fem_space(ncells, degree, periodic=(False,) * ldim)

    # No `nquads` given: defaults to `degree + 1` along each direction, which
    # is always enough to integrate the constant function f = 1 exactly.
    f = lambda *x: 1.0
    assert V.integral(f) == pytest.approx(1.0)

#==============================================================================
@pytest.mark.parametrize('ncells', [(4,), (3, 4)])
@pytest.mark.parametrize('nq',     [2, 3])
def test_integral_exact_for_polynomials_periodic(ncells, nq):

    ldim   = len(ncells)
    degree = (2,) * ldim
    nquads = (nq,) * ldim

    V = make_tensor_fem_space(ncells, degree, periodic=(True,) * ldim)

    power = 2 * nq - 1
    f = lambda *x: np.prod([xi**power for xi in x])

    approx = V.integral(f, nquads=nquads)
    exact  = (1.0 / (power + 1)) ** ldim
    assert approx == pytest.approx(exact)
