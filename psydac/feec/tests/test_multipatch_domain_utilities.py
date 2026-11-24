#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#

import pytest
from sympde.api import CzarnyMapping, PolarMapping, TargetMapping
from sympy import Rational, simplify

from psydac.feec.multipatch_domain_utilities import (
    TransposedCzarnyMapping,
    TransposedPolarMapping,
    TransposedTargetMapping,
)


@pytest.mark.parametrize(
    ("Transposed", "Original", "params"),
    [
        (TransposedPolarMapping, PolarMapping, dict(c1=0.5, c2=-1, rmin=0.2, rmax=1.5)),
        (TransposedTargetMapping, TargetMapping, dict(c1=0.5, c2=-1, k=0.3, D=0.2)),
        # SymPDE takes minutes to build CzarnyMapping with float parameters
        (TransposedCzarnyMapping, CzarnyMapping, dict(c2=-1, eps=Rational(1, 5), b=Rational(7, 5))),
    ],
)
def test_transposed_mapping(Transposed, Original, params):
    F = Transposed("F", **params)
    G = Original("G", **params)
    assert F.ldim == F.pdim == 2

    x1, x2 = G.logical_coordinates
    swap = {x1: x2, x2: x1}
    for f, g in zip(F.expressions, G.expressions):
        assert simplify(f - g.subs(swap, simultaneous=True)) == 0

    # Swapping the logical coordinates reverses the orientation
    det_F = F.jacobian_expr.det()
    det_G = G.jacobian_expr.det().subs(swap, simultaneous=True)
    assert simplify(det_F + det_G) == 0
