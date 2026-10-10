#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from itertools import product

import numpy as np
import pytest
from mpi4py import MPI

from psydac.cad.cad     import elevate, refine, translate
from psydac.cad.gallery import annulus, circle, quart_circle
from psydac.cad.tests.test_gallery import make_nurbs_mapping
from psydac.ddm.cart    import DomainDecomposition
from psydac.fem.splines import SplineSpace
from psydac.fem.tensor  import TensorFemSpace
from psydac.mapping.discrete import NurbsMapping, SplineMapping
from psydac.mapping.discrete_gallery import discrete_mapping

def make_curve(*, nurbs: bool = False, pdim: int = 1) -> SplineMapping:
    """Create a nonlinear spline or NURBS curve in `pdim` dimensions."""
    W = SplineSpace(degree=2, knots=[0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0])
    V = TensorFemSpace(DomainDecomposition([W.ncells], [False]), W)
    points = np.array([[0.0,  0.0,  1.0],
                       [0.2,  0.5,  0.8],
                       [0.9,  0.7,  0.3],
                       [1.5,  0.4, -0.2],
                       [2.0, -0.1,  0.0]])[:, :pdim]
    if nurbs:
        weights = np.array([1.0, 0.5, 2.0, 0.7, 1.0])
        return NurbsMapping.from_control_points_weights(V, points, weights)
    return SplineMapping.from_control_points(V, points)

def make_surface(*, pdim: int = 2) -> NurbsMapping:
    """Create a NURBS surface in `pdim` dimensions from the quarter annulus,
    lifted to z = x * y if pdim = 3."""
    degrees, knots, points, weights = quart_circle()
    if pdim == 3:
        z = points[..., 0] * points[..., 1]
        points = np.concatenate([points, z[..., None]], axis=-1)
    return make_nurbs_mapping(degrees, knots, points, weights)

def make_volume() -> SplineMapping:
    """Create a nonlinear spline volume in 3D."""
    return discrete_mapping('collela', ncells=[2, 2, 2], degree=[2, 2, 2])

mappings_1d = pytest.mark.parametrize('make_mapping', [
    make_curve,
    lambda: make_curve(nurbs=True),
    lambda: make_curve(pdim=2),
    lambda: make_curve(nurbs=True, pdim=3),
], ids=['spline', 'nurbs', 'spline_curve_2d', 'nurbs_curve_3d'])

# A spline mapping, NURBS mappings whose weights vary along axis 0
# (quart_circle), axis 1 (annulus), or both (circle), and a NURBS surface
mappings_2d = pytest.mark.parametrize('make_mapping', [
    lambda: discrete_mapping('collela', ncells=[4, 4], degree=[2, 2]),
    lambda: make_nurbs_mapping(*quart_circle()),
    lambda: make_nurbs_mapping(*annulus()),
    lambda: make_nurbs_mapping(*circle()),
    lambda: make_surface(pdim=3),
], ids=['spline', 'quart_circle', 'annulus', 'circle', 'surface_3d'])

#==============================================================================
def assert_same_geometry(F, G) -> None:
    """Check that two mappings agree on a uniform grid of the logical domain of G."""
    eta = list(product(np.linspace(0.0, 1.0, 9), repeat=G.ldim))
    assert np.allclose([F(*e) for e in eta], [G(*e) for e in eta], rtol=0, atol=1e-14)

def check_translate(F) -> None:
    """Check that translation moves the geometry by a fixed displacement."""
    displ = np.array([1.0, -2.0, 0.5])[:F.pdim]
    G = translate(F, displ)

    assert type(G) is type(F)
    assert_same_geometry(lambda *eta: np.asarray(F(*eta)) + displ, G)

def check_elevate(F, axis: int) -> None:
    """Check that degree elevation along an axis preserves the geometry."""
    G = elevate(F, axis=axis, times=1)

    expected_degree = list(F.space.degree)
    expected_degree[axis] += 1
    assert type(G) is type(F)
    assert list(G.space.degree) == expected_degree
    assert_same_geometry(F, G)

def check_refine(F, axis: int, values: tuple[float, ...] = (0.1, 0.35, 0.6, 0.9)) -> None:
    """Check that knot insertion along an axis preserves the geometry."""
    # The default values avoid the double knots of annulus, which already have multiplicity p
    G = refine(F, axis=axis, values=values)

    expected_breaks = np.union1d(F.space.spaces[axis].breaks, values)
    assert type(G) is type(F)
    assert np.allclose(G.space.spaces[axis].breaks, expected_breaks, rtol=0, atol=1e-15)
    assert_same_geometry(F, G)

#==============================================================================
@mappings_1d
def test_translate_1d(make_mapping) -> None:
    check_translate(make_mapping())

@mappings_2d
def test_translate_2d(make_mapping) -> None:
    check_translate(make_mapping())

#==============================================================================
@mappings_1d
def test_elevate_1d(make_mapping) -> None:
    check_elevate(make_mapping(), axis=0)

@mappings_2d
@pytest.mark.parametrize('axis', [0, 1])
def test_elevate_2d(axis: int, make_mapping) -> None:
    check_elevate(make_mapping(), axis)

@pytest.mark.parametrize('axis', [0, 1, 2])
def test_elevate_3d(axis: int) -> None:
    check_elevate(make_volume(), axis)

#==============================================================================
@mappings_1d
def test_refine_1d(make_mapping) -> None:
    check_refine(make_mapping(), axis=0)

# Values which only increase the multiplicity of a knot create no new cells
@pytest.mark.parametrize('values', [(0.3,), (0.5, 0.5)], ids=['existing_knot', 'repeated_value'])
def test_refine_multiplicity(values: tuple[float, ...]) -> None:
    check_refine(make_curve(nurbs=True), axis=0, values=values)

@mappings_2d
@pytest.mark.parametrize('axis', [0, 1])
def test_refine_2d(axis: int, make_mapping) -> None:
    check_refine(make_mapping(), axis)

@pytest.mark.parametrize('axis', [0, 1, 2])
def test_refine_3d(axis: int) -> None:
    check_refine(make_volume(), axis)

#==============================================================================
def make_mapping_parallel(comm: MPI.Comm, mpi_dims_mask: list[bool] | None = None):
    return discrete_mapping('collela', ncells=[8, 8], degree=[2, 2], comm=comm,
                            mpi_dims_mask=mpi_dims_mask)

def assert_same_local_coeffs(F, G) -> None:
    """Check that the local data (owned and ghost coefficients) of a distributed
    mapping G agrees with the corresponding part of a serial mapping F."""
    for f, g in zip(F.fields, G.fields):
        V = g.coeffs.space
        everything = (slice(None),) * V.ndim
        local_part = tuple(slice(s - m*p, e + m*p + 1)
                           for s, e, p, m in zip(V.starts, V.ends, V.pads, V.shifts))
        assert np.allclose(g.coeffs[everything], f.coeffs[local_part],
                           rtol=0, atol=1e-14)

#==============================================================================
@pytest.mark.mpi
@pytest.mark.parametrize('axis', [0, 1])
def test_elevate_parallel(axis: int) -> None:

    F = make_mapping_parallel(MPI.COMM_WORLD)
    F_serial = make_mapping_parallel(MPI.COMM_SELF)

    G = elevate(F, axis=axis, times=1)
    G_serial = elevate(F_serial, axis=axis, times=1)

    assert_same_local_coeffs(G_serial, G)

#==============================================================================
# The shifted values move the subdomain boundaries by more than p cells
@pytest.mark.mpi
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('values', [
    [0.1, 0.35, 0.6, 0.9],
    [0.01, 0.04, 0.07, 0.1, 0.13, 0.16, 0.19, 0.22],
], ids=['spread', 'shifted'])
@pytest.mark.parametrize('mpi_dims_mask', [None, [False, True], [True, False]])
def test_refine_parallel(mpi_dims_mask: list[bool] | None, values: list, axis: int) -> None:

    F = make_mapping_parallel(MPI.COMM_WORLD, mpi_dims_mask)
    F_serial = make_mapping_parallel(MPI.COMM_SELF)

    G = refine(F, axis=axis, values=values)
    G_serial = refine(F_serial, axis=axis, values=values)

    assert_same_local_coeffs(G_serial, G)

    # The directions excluded by the mask are not decomposed (see #622)
    ddm = G.space.domain_decomposition
    assert ddm.mpi_dims_mask == F.space.domain_decomposition.mpi_dims_mask
    if mpi_dims_mask is not None:
        assert all(n == 1 for n, use_dim in zip(ddm.nprocs, mpi_dims_mask) if not use_dim)
