#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np

from psydac.core.bsplines_kernels import cell_index_p


def test_cell_index_p():
    breaks = np.array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    breaks = np.ascontiguousarray(breaks, dtype=float)
    out = np.zeros_like(breaks, dtype=np.int64)
    tol = 1e-15

    # limit case: code should decide wether point is in or out, not fall in infinite loop
    i_grid = breaks.copy()
    i_grid[:] += tol
    status = cell_index_p(breaks, i_grid, tol, out)
    assert status in [0,-1]

    # usual cases: points inside or outside domain
    for offset, expected_status in [(0, 0), (tol/2, 0), (2*tol, -1)]:
        i_grid = breaks.copy()
        i_grid[:] += offset
        status = cell_index_p(breaks, i_grid, tol, out)
        assert status == expected_status

    # checking that the values match those of searchsorted (-1) for arbitrary grid points
    i_grid = np.array([0.14320482, 0.86569833, 0.77775327, 0.00895956, 0.074629  ,
       0.45682646, 0.5384352 , 0.20915311, 0.73121977, 0.01057414,
       0.33756086, 0.17839759, 0.14023414, 0.09846206, 0.79970392,
       0.65330406, 0.82716552, 0.24185731, 0.24054685, 0.72466651,
       0.69125033, 0.3136558 , 0.64794089, 0.47975527, 0.99802844,
       0.64402598, 0.41263526, 0.28178414, 0.57274384, 0.73218562])
    out = np.zeros_like(i_grid, dtype=np.int64)
    status = cell_index_p(breaks, i_grid, tol, out)
    assert status == 0
    nps = np.searchsorted(breaks, i_grid)-1
    assert np.allclose(out, nps)

