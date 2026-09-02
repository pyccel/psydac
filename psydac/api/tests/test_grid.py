#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
Unit tests for the "precomputation" stage of the assembly pipeline: the
`QuadratureGrid` and `BasisValues` classes in `psydac.api.grid`, which are
built directly from a `TensorFemSpace` (i.e. without going through the full
`discretize()` + `assemble()` machinery).
"""
import numpy as np
import pytest

from psydac.ddm.cart      import DomainDecomposition
from psydac.fem.splines   import SplineSpace
from psydac.fem.tensor    import TensorFemSpace
from psydac.api.grid      import QuadratureGrid, BasisValues

#==============================================================================
def make_1d_space(ncells, degree, periodic=False):
    breaks = np.linspace(0., 1., ncells + 1)
    V1d    = SplineSpace(degree=degree, grid=breaks, periodic=periodic)
    domain_decomposition = DomainDecomposition([ncells], [periodic])
    return TensorFemSpace(domain_decomposition, V1d)

#==============================================================================
def make_2d_space(ncells, degree, periodic=(False, False)):
    breaks = [np.linspace(0., 1., n + 1) for n in ncells]
    spaces = [SplineSpace(degree=p, grid=b, periodic=per)
              for p, b, per in zip(degree, breaks, periodic)]
    domain_decomposition = DomainDecomposition(list(ncells), list(periodic))
    return TensorFemSpace(domain_decomposition, *spaces)

#==============================================================================
@pytest.mark.parametrize('ncells', [4, 7])
@pytest.mark.parametrize('nq',     [2, 3, 4])
def test_quadrature_grid_1d_shapes_and_exact_integration(ncells, nq):

    degree = 2
    V = make_1d_space(ncells, degree)

    grid = QuadratureGrid(V, nquads=[nq])

    assert grid.n_elements == [ncells]
    assert grid.nquads     == [nq]
    assert grid.points [0].shape == (ncells, nq)
    assert grid.weights[0].shape == (ncells, nq)

    # A Gauss-Legendre rule with `nq` points integrates polynomials of degree
    # <= 2*nq - 1 exactly; check this by integrating x**2 (degree 2) over
    # [0, 1], whose exact value is 1/3.
    x, w = grid.points[0], grid.weights[0]
    approx_integral = np.sum(x**2 * w)
    assert approx_integral == pytest.approx(1.0 / 3.0)

    # Quadrature weights on each element must sum to the element length.
    element_lengths = np.sum(w, axis=1)
    assert element_lengths == pytest.approx(1.0 / ncells)

#==============================================================================
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('ext',  [-1, 1])
def test_quadrature_grid_boundary_reduces_to_single_point(axis, ext):

    V = make_2d_space((4, 5), (2, 3))

    grid = QuadratureGrid(V, axis=axis, ext=ext, nquads=[3, 3])

    # The direction normal to the boundary is reduced to a single point...
    assert grid.points [axis].shape == (1, 1)
    assert grid.weights[axis].shape == (1, 1)
    assert grid.weights[axis][0, 0] == 1.0

    expected_boundary = 0.0 if ext == -1 else 1.0
    assert grid.points[axis][0, 0] == pytest.approx(expected_boundary)

    # ...while the tangential direction is untouched.
    tangential_axis = 1 - axis
    ncells_tangential = V.ncells[tangential_axis]
    assert grid.points[tangential_axis].shape == (ncells_tangential, 3)

#==============================================================================
@pytest.mark.parametrize('ncells', [(3, 4)])
@pytest.mark.parametrize('degree', [(2, 3)])
def test_basis_values_shape_and_partition_of_unity(ncells, degree):

    nq     = [p + 1 for p in degree]
    nderiv = 1
    V      = make_2d_space(ncells, degree)
    grid   = QuadratureGrid(V, nquads=nq)

    # Use trial=True so that basis values are *not* pre-multiplied by the
    # quadrature weights, allowing a direct partition-of-unity check.
    basis_values = BasisValues(V, nderiv=nderiv, nquads=nq, trial=True, grid=grid)

    for axis, (n, p, q) in enumerate(zip(ncells, degree, nq)):
        basis_axis = basis_values.basis[0][axis]
        assert basis_axis.shape == (n, p + 1, nderiv + 1, q)

        # B-splines (non-rational, 'B' normalization) form a partition of
        # unity: the values (0-th derivative) of all p+1 non-zero local
        # basis functions sum to 1 at every quadrature point.
        values = basis_axis[:, :, 0, :]
        assert np.sum(values, axis=1) == pytest.approx(np.ones((n, q)))

#==============================================================================
def test_basis_values_test_space_is_weighted_by_quadrature_weights():

    ncells, degree = 4, 2
    nq     = [degree + 1]
    V      = make_1d_space(ncells, degree)
    grid   = QuadratureGrid(V, nquads=nq)

    test_values  = BasisValues(V, nderiv=1, nquads=nq, trial=False, grid=grid)
    trial_values = BasisValues(V, nderiv=1, nquads=nq, trial=True,  grid=grid)

    weights  = grid.weights[0]
    expected = trial_values.basis[0][0] * weights[:, None, None, :]

    assert test_values.basis[0][0] == pytest.approx(expected)
