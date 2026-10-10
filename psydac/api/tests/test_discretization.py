#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np
import pytest

from psydac.api.discretization import get_max_degree, get_max_degree_of_one_space
from psydac.ddm.cart import DomainDecomposition
from psydac.fem.splines import SplineSpace
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import VectorFemSpace

#==============================================================================
def make_tensor_space_2d(degree: tuple[int, int],
                         domain_decomposition: DomainDecomposition) -> TensorFemSpace:
    """
    Create a 2D tensor-product spline space on a uniform grid.

    Parameters
    ----------
    degree : tuple[int, int]
        The spline degree along each direction.

    domain_decomposition : DomainDecomposition
        The decomposition of the 2D grid, which also gives its number of cells.

    Returns
    -------
    TensorFemSpace
        The spline space.
    """
    spaces = [SplineSpace(degree=p, grid=np.linspace(0.0, 1.0, n + 1))
              for p, n in zip(degree, domain_decomposition.ncells)]
    return TensorFemSpace(domain_decomposition, *spaces)

#==============================================================================
def test_get_max_degree_of_one_space() -> None:

    domain_decomposition = DomainDecomposition([3, 5], [False, False])
    V1 = make_tensor_space_2d((2, 3), domain_decomposition)
    V2 = make_tensor_space_2d((3, 1), domain_decomposition)

    assert list(get_max_degree_of_one_space(V1)) == [2, 3]
    assert list(get_max_degree_of_one_space(VectorFemSpace(V1, V2))) == [3, 3]

    with pytest.raises(TypeError, match='SplineSpace'):
        get_max_degree_of_one_space(V1.spaces[0])

#==============================================================================
def test_get_max_degree() -> None:

    domain_decomposition = DomainDecomposition([3, 5], [False, False])
    V1 = make_tensor_space_2d((2, 3), domain_decomposition)
    V2 = make_tensor_space_2d((3, 1), domain_decomposition)
    V3 = make_tensor_space_2d((1, 4), domain_decomposition)

    assert list(get_max_degree(V1, V2)) == [3, 3]
    assert list(get_max_degree(VectorFemSpace(V1, V2), V3)) == [3, 4]
