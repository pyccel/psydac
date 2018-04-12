# -*- coding: UTF-8 -*-

from spl.fem.splines import SplineSpace
from spl.fem.splines import Spline
from spl.fem.tensor  import TensorSpace


def test_1d():
    knots = [0., 0., 0., 1., 1., 1.]
    p = 2
    V = SplineSpace(knots, p)
    print (V)
    F = Spline(V)
#    print(F.coeffs)

def test_2d():
    knots_1 = [0., 0., 0., 1., 1., 1.]
    knots_2 = [0., 0., 0., 0.5, 1., 1., 1.]
    p_1 = 2
    p_2 = 2
    V1 = SplineSpace(knots_1, p_1)
    V2 = SplineSpace(knots_2, p_2)

    V = TensorSpace(V1, V2)
    print (V)
    F = Spline(V)
#    print(F.coeffs)

def test_3d():
    knots_1 = [0., 0., 0., 1., 1., 1.]
    knots_2 = [0., 0., 0., 0.5, 1., 1., 1.]
    knots_3 = [0., 0., 0.5, 1., 1.]
    p_1 = 2
    p_2 = 2
    p_3 = 1
    V1 = SplineSpace(knots_1, p_1)
    V2 = SplineSpace(knots_2, p_2)
    V3 = SplineSpace(knots_3, p_3)

    V = TensorSpace(V1, V2, V3)
    print (V)
    F = Spline(V)
#    print(F.coeffs)


###############################################
if __name__ == '__main__':
    test_1d()
    test_2d()
    test_3d()
