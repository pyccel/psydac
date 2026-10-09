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
from psydac.mapping.discrete_gallery import discrete_mapping

# A spline mapping, and NURBS mappings whose weights vary along axis 0
# (quart_circle), axis 1 (annulus), or both (circle)
mappings_2d = pytest.mark.parametrize('make_mapping', [
    lambda: discrete_mapping('collela', ncells=[4, 4], degree=[2, 2]),
    lambda: make_nurbs_mapping(*quart_circle()),
    lambda: make_nurbs_mapping(*annulus()),
    lambda: make_nurbs_mapping(*circle()),
], ids=['spline', 'quart_circle', 'annulus', 'circle'])

def make_mapping_3d():
    return discrete_mapping('collela', ncells=[2, 2, 2], degree=[2, 2, 2])

#==============================================================================
def assert_same_geometry(F, G) -> None:
    """Check that two mappings agree on a uniform grid of the logical domain of G."""
    eta = list(product(np.linspace(0.0, 1.0, 9), repeat=G.ldim))
    assert np.allclose([F(*e) for e in eta], [G(*e) for e in eta], rtol=0, atol=1e-14)

def check_elevate(F, axis: int) -> None:
    """Check that degree elevation along an axis preserves the geometry."""
    G = elevate(F, axis=axis, times=1)

    expected_degree = list(F.space.degree)
    expected_degree[axis] += 1
    assert type(G) is type(F)
    assert list(G.space.degree) == expected_degree
    assert_same_geometry(F, G)

def check_refine(F, axis: int) -> None:
    """Check that knot insertion along an axis preserves the geometry."""
    # Avoid the double knots of annulus, which already have multiplicity p
    values = [0.1, 0.35, 0.6, 0.9]
    G = refine(F, axis=axis, values=values)

    expected_breaks = np.union1d(F.space.spaces[axis].breaks, values)
    assert type(G) is type(F)
    assert np.allclose(G.space.spaces[axis].breaks, expected_breaks, rtol=0, atol=1e-15)
    assert_same_geometry(F, G)

#==============================================================================
@mappings_2d
def test_translate_2d(make_mapping) -> None:

    displ = np.array([1.0, -2.0])
    F = make_mapping()
    G = translate(F, displ)

    assert type(G) is type(F)
    assert_same_geometry(lambda *eta: np.asarray(F(*eta)) + displ, G)

#==============================================================================
@mappings_2d
@pytest.mark.parametrize('axis', [0, 1])
def test_elevate_2d(axis: int, make_mapping) -> None:
    check_elevate(make_mapping(), axis)

@pytest.mark.parametrize('axis', [0, 1, 2])
def test_elevate_3d(axis: int) -> None:
    check_elevate(make_mapping_3d(), axis)

#==============================================================================
@mappings_2d
@pytest.mark.parametrize('axis', [0, 1])
def test_refine_2d(axis: int, make_mapping) -> None:
    check_refine(make_mapping(), axis)

@pytest.mark.parametrize('axis', [0, 1, 2])
def test_refine_3d(axis: int) -> None:
    check_refine(make_mapping_3d(), axis)

#==============================================================================
def make_mapping_parallel(comm: MPI.Comm):
    return discrete_mapping('collela', ncells=[8, 8], degree=[2, 2], comm=comm)

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
def test_refine_parallel(values: list, axis: int) -> None:

    F = make_mapping_parallel(MPI.COMM_WORLD)
    F_serial = make_mapping_parallel(MPI.COMM_SELF)

    G = refine(F, axis=axis, values=values)
    G_serial = refine(F_serial, axis=axis, values=values)

    assert_same_local_coeffs(G_serial, G)
