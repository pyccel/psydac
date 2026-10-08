#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np
import pytest

from psydac.cad.cad     import elevate, refine
from psydac.cad.gallery import quart_circle
from psydac.cad.tests.test_gallery import make_nurbs_mapping

#==============================================================================
def assert_same_geometry(F, G) -> None:
    """Check that two 2D mappings agree on a uniform grid of the unit square."""
    t = np.linspace(0.0, 1.0, 9)
    x_F = [F(e1, e2) for e1 in t for e2 in t]
    x_G = [G(e1, e2) for e1 in t for e2 in t]
    assert np.allclose(x_F, x_G, rtol=0, atol=1e-14)

#==============================================================================
# The weights of the quarter annulus vary along axis 0 only
@pytest.mark.parametrize('axis', [0, 1])
def test_elevate(axis: int) -> None:

    F = make_nurbs_mapping(*quart_circle(rmin=0.5, rmax=1.0))
    G = elevate(F, axis=axis, times=1)

    expected_degree = list(F.space.degree)
    expected_degree[axis] += 1
    assert list(G.space.degree) == expected_degree
    assert_same_geometry(F, G)

#==============================================================================
@pytest.mark.parametrize('axis', [0, 1])
def test_refine(axis: int) -> None:

    values = [0.3, 0.6]
    F = make_nurbs_mapping(*quart_circle(rmin=0.5, rmax=1.0))
    G = refine(F, axis=axis, values=values)

    expected_breaks = np.union1d(F.space.spaces[axis].breaks, values)
    assert np.allclose(G.space.spaces[axis].breaks, expected_breaks, rtol=0, atol=1e-15)
    assert_same_geometry(F, G)
