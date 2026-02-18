#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from numpy import linspace

from psydac.fem.basic   import FemField
from psydac.fem.splines import SplineSpace
from psydac.fem.tensor  import TensorFemSpace
from psydac.fem.vector  import VectorFemSpace
from psydac.ddm.cart    import DomainDecomposition


def test_1d_1():
    print ('>>> test_1d_1')

    knots = [0., 0., 0., 1., 1., 1.]
    p = 2
    V = SplineSpace(p, knots=knots)
    print (V)
    F = FemField(V)

def test_1d_2():
    print ('>>> test_1d_2')

    p = 2
    grid = linspace(0., 1., 5)
    V = SplineSpace(p, grid=grid)
    print (V)
    F = FemField(V)

def test_1d_3():
    print ('>>> test_1d_3')

    p = 2
    grid = linspace(0., 1., 5)
    V1 = SplineSpace(p, grid=grid)
    V2 = SplineSpace(p+1, grid=grid)

    domain_decomposition = DomainDecomposition([V1.ncells, V2.ncells], [False, False])
    V = TensorFemSpace(domain_decomposition, V1, V2)
    print (V)

def test_2d_1():
    print ('>>> test_2d_1')

    knots_1 = [0., 0., 0., 1., 1., 1.]
    knots_2 = [0., 0., 0., 0.5, 1., 1., 1.]
    p_1 = 2
    p_2 = 2
    V1 = SplineSpace(p_1, knots=knots_1)
    V2 = SplineSpace(p_2, knots=knots_2)

    domain_decomposition = DomainDecomposition([V1.ncells, V2.ncells], [False, False])
    V = TensorFemSpace(domain_decomposition, V1, V2)
    print (V)
    F = FemField(V)

def test_2d_2():
    print ('>>> test_2d_2')

    p_1 = 2
    p_2 = 2
    grid_1 = linspace(0., 1., 3)
    grid_2 = linspace(0., 1., 5)
    V1 = SplineSpace(p_1, grid=grid_1)
    V2 = SplineSpace(p_2, grid=grid_2)

    domain_decomposition = DomainDecomposition([V1.ncells, V2.ncells], [False, False])
    V = TensorFemSpace(domain_decomposition, V1, V2)
    print (V)
    F = FemField(V)

def test_2d_3():
    print ('>>> test_2d_3')

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
    print (V)

def test_3d_1():
    print ('>>> test_3d_1')

    knots_1 = [0., 0., 0., 1., 1., 1.]
    knots_2 = [0., 0., 0., 0.5, 1., 1., 1.]
    knots_3 = [0., 0., 0.5, 1., 1.]
    p_1 = 2
    p_2 = 2
    p_3 = 1
    V1 = SplineSpace(p_1, knots=knots_1)
    V2 = SplineSpace(p_2, knots=knots_2)
    V3 = SplineSpace(p_3, knots=knots_3)

    domain_decomposition = DomainDecomposition([V1.ncells, V2.ncells, V3.ncells], [False, False, False])
    V = TensorFemSpace(domain_decomposition, V1, V2, V3)
    print (V)
    F = FemField(V)

def test_3d_2():
    print ('>>> test_3d_2')

    p_1 = 2
    p_2 = 2
    p_3 = 1
    grid_1 = linspace(0., 1., 3)
    grid_2 = linspace(0., 1., 5)
    grid_3 = linspace(0., 1., 7)
    V1 = SplineSpace(p_1, grid=grid_1)
    V2 = SplineSpace(p_2, grid=grid_2)
    V3 = SplineSpace(p_3, grid=grid_3)

    domain_decomposition = DomainDecomposition([V1.ncells, V2.ncells, V3.ncells], [False, False, False])
    V = TensorFemSpace(domain_decomposition, V1, V2, V3)
    print (V)
    F = FemField(V)

def test_3d_3():
    print ('>>> test_3d_3')

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
    print (V)


###############################################
if __name__ == '__main__':

    test_1d_1()
    test_1d_2()
    test_1d_3()

    test_2d_1()
    test_2d_2()
    test_2d_3()

    test_3d_1()
    test_3d_2()
    test_3d_3()
