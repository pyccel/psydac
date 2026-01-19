#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import pytest
from numpy import linspace

from psydac.fem.basic   import FemField
from psydac.fem.splines import SplineSpace
from psydac.fem.tensor  import TensorFemSpace
from psydac.fem.vector  import VectorFemSpace
from psydac.ddm.cart    import DomainDecomposition
from psydac.linalg.block import BlockVector, BlockVectorSpace


def test_vector_space_2d():
    p = 2
    grid_1 = linspace(0., 1., 3)
    grid_2 = linspace(0., 1., 5)

    # ... first component
    V1 = SplineSpace(p-1, grid=grid_1)
    V2 = SplineSpace(p, grid=grid_2)

    domain_decomposition = DomainDecomposition([V1.ncells, V2.ncells], [False, False])
    Vx = TensorFemSpace(domain_decomposition, V1, V2)
    # ...

    # ... second component
    V1 = SplineSpace(p, grid=grid_1)
    V2 = SplineSpace(p-1, grid=grid_2)

    Vy = TensorFemSpace(domain_decomposition, V1, V2)
    # ...

    V = VectorFemSpace(Vx, Vy)
    F = FemField(V)

    # Check properties of V from abstract interface
    assert V.ldim == 2
    assert V.periodic == (False, False)
    assert V.mapping is None
    assert isinstance(V.coeff_space, BlockVectorSpace)
    assert not V.is_multipatch
    assert V.is_vector_valued
    assert V.symbolic_space is None
    assert V.patch_spaces == (V,)
    assert V.component_spaces == (Vx, Vy)
    with pytest.raises(NotImplementedError):
        getattr(V, 'axis_spaces')
    assert V.is_multipatch == False
    assert V.is_vector_valued == True

    # Check other properties of V
    assert V.nbasis == Vx.nbasis + Vy.nbasis
    assert V.degree == [Vx.degree, Vy.degree]
    assert V.multiplicity == [Vx.multiplicity, Vy.multiplicity]
    assert V.pads == [Vx.pads, Vy.pads]
    assert V.ncells == Vx.ncells == Vy.ncells
    assert V.spaces == (Vx, Vy)

    # Check properties of F
    assert F.space == V
    assert isinstance(F.coeffs, BlockVector)
    assert len(F.fields) == 2
    assert isinstance(F.fields[0], FemField)
    assert isinstance(F.fields[1], FemField)
    assert F.fields[0].space == Vx
    assert F.fields[1].space == Vy
    assert F.patch_fields == (F,)
    assert F.component_fields == F.fields


def test_vector_space_3d():
    p = 2
    grid_1 = linspace(0., 1., 3)
    grid_2 = linspace(0., 1., 5)
    grid_3 = linspace(0., 1., 7)

    # ... first component
    V1 = SplineSpace(p-1, grid=grid_1)
    V2 = SplineSpace(p, grid=grid_2)
    V3 = SplineSpace(p, grid=grid_3)

    domain_decomposition = DomainDecomposition([V1.ncells, V2.ncells, V3.ncells], [False, False, False])

    Vx = TensorFemSpace(domain_decomposition, V1, V2, V3)
    # ...

    # ... second component
    V1 = SplineSpace(p, grid=grid_1)
    V2 = SplineSpace(p-1, grid=grid_2)
    V3 = SplineSpace(p, grid=grid_3)

    Vy = TensorFemSpace(domain_decomposition, V1, V2, V3)
    # ...

    # ... third component
    V1 = SplineSpace(p, grid=grid_1)
    V2 = SplineSpace(p, grid=grid_2)
    V3 = SplineSpace(p-1, grid=grid_3)

    Vz = TensorFemSpace(domain_decomposition, V1, V2, V3)
    # ...

    V = VectorFemSpace(Vx, Vy, Vz)
    F = FemField(V)

    # Check properties of V from abstract interface
    assert V.ldim == 3
    assert V.periodic == (False, False, False)
    assert V.mapping is None
    assert isinstance(V.coeff_space, BlockVectorSpace)
    assert not V.is_multipatch
    assert V.is_vector_valued
    assert V.symbolic_space is None
    assert V.patch_spaces == (V,)
    assert V.component_spaces == (Vx, Vy, Vz)
    with pytest.raises(NotImplementedError):
        getattr(V, 'axis_spaces')
    assert V.is_multipatch == False
    assert V.is_vector_valued == True

    # Check other properties of V
    assert V.nbasis == Vx.nbasis + Vy.nbasis + Vz.nbasis
    assert V.degree == [Vx.degree, Vy.degree, Vz.degree]
    assert V.multiplicity == [Vx.multiplicity, Vy.multiplicity, Vz.multiplicity]
    assert V.pads == [Vx.pads, Vy.pads, Vz.pads]
    assert V.ncells == Vx.ncells == Vy.ncells == Vz.ncells
    assert V.spaces == (Vx, Vy, Vz)

    # Check properties of F
    assert F.space == V
    assert isinstance(F.coeffs, BlockVector)
    assert len(F.fields) == 3
    assert isinstance(F.fields[0], FemField)
    assert isinstance(F.fields[1], FemField)
    assert isinstance(F.fields[2], FemField)
    assert F.fields[0].space == Vx
    assert F.fields[1].space == Vy
    assert F.fields[2].space == Vz
    assert F.patch_fields == (F,)
    assert F.component_fields == F.fields

###############################################
if __name__ == '__main__':

    print('>>> test_vector_space_2d')
    test_vector_space_2d()

    print ('>>> test_vector_space_3d')
    test_vector_space_3d()
