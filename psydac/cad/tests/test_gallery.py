#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np
import pytest

from psydac.cad.gallery      import annulus, circle, quart_circle
from psydac.ddm.cart         import DomainDecomposition
from psydac.fem.splines      import SplineSpace
from psydac.fem.tensor       import TensorFemSpace
from psydac.mapping.discrete import NurbsMapping

#==============================================================================
def make_nurbs_mapping(degrees, knots, points, weights) -> NurbsMapping:
    """Create a NURBS mapping from the output of a gallery function."""
    spaces = [SplineSpace(knots=k, degree=p) for k, p in zip(knots, degrees)]
    ncells = [len(V.breaks) - 1 for V in spaces]
    domain_decomposition = DomainDecomposition(ncells, [False] * len(spaces))
    space = TensorFemSpace(domain_decomposition, *spaces)
    return NurbsMapping.from_control_points_weights(space, points, weights)

#==============================================================================
# Each circular edge of the logical square is given as (axis, value, radius)
@pytest.mark.parametrize('center', [None, (1.0, -2.0)])
@pytest.mark.parametrize(('gallery_function', 'kwargs', 'circular_edges'), [
    (quart_circle, dict(rmin=0.5, rmax=1.0), [(1, 0.0, 0.5), (1, 1.0, 1.0)]),
    (annulus,      dict(rmin=0.5, rmax=1.0), [(0, 0.0, 0.5), (0, 1.0, 1.0)]),
    (circle,       dict(radius=2.0),         [(0, 0.0, 2.0), (0, 1.0, 2.0), (1, 0.0, 2.0), (1, 1.0, 2.0)]),
], ids=['quart_circle', 'annulus', 'circle'])
def test_gallery_circular_edges(gallery_function, kwargs: dict, circular_edges: list, center) -> None:

    F = make_nurbs_mapping(*gallery_function(**kwargs, center=center))
    x0 = np.zeros(2) if center is None else np.array(center)

    t = np.linspace(0.0, 1.0, 17)
    for axis, value, radius in circular_edges:
        eta = [(value, ti) if axis == 0 else (ti, value) for ti in t]
        r = [np.linalg.norm(F(*e) - x0) for e in eta]
        assert np.allclose(r, radius, rtol=0, atol=1e-14)
