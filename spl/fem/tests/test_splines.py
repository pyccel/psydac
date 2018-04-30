# -*- coding: UTF-8 -*-

from spl.fem.splines import SplineSpace
from spl.fem.splines import Spline
from spl.fem.tensor  import TensorSpace

from numpy import linspace


def test_1d_1():
    knots = [0., 0., 0., 1., 1., 1.]
    p = 2
    V = SplineSpace(p, knots=knots)
    print (V)
    F = Spline(V)

def test_1d_2():
    p = 2
    grid = linspace(0., 1., 5)
    V = SplineSpace(p, grid=grid)
    print (V)
    F = Spline(V)

def test_2d():
    knots_1 = [0., 0., 0., 1., 1., 1.]
    knots_2 = [0., 0., 0., 0.5, 1., 1., 1.]
    p_1 = 2
    p_2 = 2
    V1 = SplineSpace(p_1, knots=knots_1)
    V2 = SplineSpace(p_2, knots=knots_2)

    V = TensorSpace(V1, V2)
    print (V)
    F = Spline(V)

def test_3d():
    knots_1 = [0., 0., 0., 1., 1., 1.]
    knots_2 = [0., 0., 0., 0.5, 1., 1., 1.]
    knots_3 = [0., 0., 0.5, 1., 1.]
    p_1 = 2
    p_2 = 2
    p_3 = 1
    V1 = SplineSpace(p_1, knots=knots_1)
    V2 = SplineSpace(p_2, knots=knots_2)
    V3 = SplineSpace(p_3, knots=knots_3)

    V = TensorSpace(V1, V2, V3)
    print (V)
    F = Spline(V)


###############################################
if __name__ == '__main__':
    test_1d_1()
    test_1d_2()
    test_2d()
    test_3d()
