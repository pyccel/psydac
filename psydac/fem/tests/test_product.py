# -*- coding: UTF-8 -*-

from psydac.fem.basic   import FemField
from psydac.fem.splines import SplineSpace
from psydac.fem.tensor  import TensorFemSpace
from psydac.fem.vector  import ProductFemSpace, VectorFemField

from numpy import linspace

def test_product_space_2d():
    print ('>>> test_product_space_2d')

    p = 2
    grid_1 = linspace(0., 1., 3)
    grid_2 = linspace(0., 1., 5)

    # ... first component
    V1 = SplineSpace(p-1, grid=grid_1)
    V2 = SplineSpace(p, grid=grid_2)

    Vx = TensorFemSpace(V1, V2)
    # ...

    # ... second component
    V1 = SplineSpace(p, grid=grid_1)
    V2 = SplineSpace(p-1, grid=grid_2)

    Vy = TensorFemSpace(V1, V2)
    # ...

    V = ProductFemSpace(Vx, Vy)
    F = VectorFemField(V)

def test_product_space_3d():
    print ('>>> test_product_space_3d')

    p = 2
    grid_1 = linspace(0., 1., 3)
    grid_2 = linspace(0., 1., 5)
    grid_3 = linspace(0., 1., 7)

    # ... first component
    V1 = SplineSpace(p-1, grid=grid_1)
    V2 = SplineSpace(p, grid=grid_2)
    V3 = SplineSpace(p, grid=grid_3)

    Vx = TensorFemSpace(V1, V2, V3)
    # ...

    # ... second component
    V1 = SplineSpace(p, grid=grid_1)
    V2 = SplineSpace(p-1, grid=grid_2)
    V3 = SplineSpace(p, grid=grid_3)

    Vy = TensorFemSpace(V1, V2, V3)
    # ...

    # ... third component
    V1 = SplineSpace(p, grid=grid_1)
    V2 = SplineSpace(p, grid=grid_2)
    V3 = SplineSpace(p-1, grid=grid_3)

    Vz = TensorFemSpace(V1, V2, V3)
    # ...

    V = ProductFemSpace(Vx, Vy, Vz)
    F = VectorFemField(V)


###############################################
if __name__ == '__main__':

    test_product_space_2d()
    test_product_space_3d()
