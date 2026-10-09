#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np
import pytest

from psydac.cad.cad     import elevate, refine, translate
from psydac.cad.gallery import annulus, circle, quart_circle
from psydac.cad.tests.test_gallery import make_nurbs_mapping
from psydac.mapping.discrete_gallery import discrete_mapping

# The weights vary along axis 0 (quart_circle), axis 1 (annulus), or both (circle)
gallery_functions = pytest.mark.parametrize('gallery_function', [quart_circle, annulus, circle],
                                            ids=['quart_circle', 'annulus', 'circle'])

#==============================================================================
def assert_same_geometry(F, G) -> None:
    """Check that two 2D mappings agree on a uniform grid of the unit square."""
    t = np.linspace(0.0, 1.0, 9)
    x_F = [F(e1, e2) for e1 in t for e2 in t]
    x_G = [G(e1, e2) for e1 in t for e2 in t]
    assert np.allclose(x_F, x_G, rtol=0, atol=1e-14)

#==============================================================================
@pytest.mark.parametrize('make_mapping', [
    lambda: discrete_mapping('collela', ncells=[4, 4], degree=[2, 2]),
    lambda: make_nurbs_mapping(*quart_circle()),
    lambda: make_nurbs_mapping(*annulus()),
    lambda: make_nurbs_mapping(*circle()),
], ids=['spline', 'quart_circle', 'annulus', 'circle'])
def test_translate_2d(make_mapping) -> None:

    displ = np.array([1.0, -2.0])
    F = make_mapping()
    G = translate(F, displ)

    assert type(G) is type(F)
    assert_same_geometry(lambda *eta: np.asarray(F(*eta)) + displ, G)

#==============================================================================
@gallery_functions
@pytest.mark.parametrize('axis', [0, 1])
def test_elevate(axis: int, gallery_function) -> None:

    F = make_nurbs_mapping(*gallery_function())
    G = elevate(F, axis=axis, times=1)

    expected_degree = list(F.space.degree)
    expected_degree[axis] += 1
    assert list(G.space.degree) == expected_degree
    assert_same_geometry(F, G)

#==============================================================================
@gallery_functions
@pytest.mark.parametrize('axis', [0, 1])
def test_refine(axis: int, gallery_function) -> None:

    # Avoid the double knots of annulus, which already have multiplicity p
    values = [0.1, 0.35, 0.6, 0.9]
    F = make_nurbs_mapping(*gallery_function())
    G = refine(F, axis=axis, values=values)

    expected_breaks = np.union1d(F.space.spaces[axis].breaks, values)
    assert np.allclose(G.space.spaces[axis].breaks, expected_breaks, rtol=0, atol=1e-15)
    assert_same_geometry(F, G)
