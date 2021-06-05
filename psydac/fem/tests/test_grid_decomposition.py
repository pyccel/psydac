import pytest

import numpy as np

from psydac.fem.splines import SplineSpace
from psydac.fem.grid import FemAssemblyGrid

@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('degree', [2, 3])
@pytest.mark.parametrize('pad', [3])
@pytest.mark.parametrize('localsizes', [[100], [10,80,10], [2]*50, [1,4,9,16,25,36,9]])
def test_grid_decomposition(periodic, degree, pad, localsizes, gridcnt=100):
    # this test shall verify that the FemAssemblyGrid is broken into the correct parts
    splinespace = SplineSpace(degree, grid=np.linspace(0, 1, gridcnt+1), periodic=periodic, pads=pad)

    n = splinespace.ncells

    start = 0
    total = []
    for size in localsizes:
        end = start + size - 1
        grid = FemAssemblyGrid(splinespace, start, end)

        if periodic:
            offset = pad
        else:
            offset = min(pad, start)

        realstart = start - offset

        # check correctness of local indices
        assert len(grid.indices) == size + offset
        assert np.array_equal(grid.indices, [(i+n)%n for i in range(realstart, end+1)])

        # some more assertions
        assert np.array_equal(grid.indices, (grid.spans - degree + n) % n)
        assert grid.num_elements == end+1 - realstart

        total += [grid.indices[i] for i in range(grid.local_element_start, grid.local_element_end+1)]

        start = end + 1
    
    # check that each spline is sent to exactly one quadrature element
    if periodic:
        elem_num = gridcnt
    else:
        elem_num = gridcnt - degree + 1
    print(total)
    assert np.array_equal(sorted(total), [i for i in range(elem_num)])

if __name__ == '__main__':
    test_grid_decomposition(True, 3, 2, [10, 80, 10])
