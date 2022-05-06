# coding: utf-8
#
# Copyright 2018 Ahmed Ratnani

import numpy as np

from psydac.core.cad import ( point_on_bspline_curve,
                              insert_knot_bspline_curve,
                              point_on_bspline_surface,
                            )

# ==========================================================
def test_point_on_bspline_curve_1():
    """Example 2.3, The NURBS Book, page 82"""
    pu = 2
    Tu = [0., 0., 0., 1., 2., 3., 4., 4., 5., 5., 5.]
    Tu = np.array(Tu)
    nu = len(Tu) - pu - 1

    d = 3
    P = np.random.random((nu, d))
    out = np.zeros(d)

    u = 5./2.
    expected = 1./8. * P[2] + 6./8. * P[3] + 1./8. * P[4]
    point_on_bspline_curve(pu, Tu, P, u, out)

    assert(np.allclose(out-expected, 1.e-13))

# ==========================================================
def test_insert_knot_bspline_curve_1():
    """Example 5.1, The NURBS Book, page 144"""
    pu = 3
    Tu = [0., 0., 0., 0., 1., 2., 3., 4., 5., 5., 5., 5.]
    Tu = np.array(Tu)
    nu = len(Tu) - pu - 1

    d = 3
    P = np.random.random((nu, d))

    u = 5./2. ; times = 1
    Tu_out = np.zeros(nu+1+pu+1)
    P_out = np.zeros((nu+1, d))

    Tu_expected = [0., 0., 0., 0., 1., 2., 5./2., 3., 4., 5., 5., 5., 5.]
    Tu_expected = np.array(Tu_expected)
    P_expected = np.zeros((nu+1, d))
    P_expected[0:3,:] = P[0:3,:]
    P_expected[3,:] = 5./6.*P[3,:] + 1./6.*P[2,:]
    P_expected[4,:] = 1./2.*P[4,:] + 1./2.*P[3,:]
    P_expected[5,:] = 1./6.*P[5,:] + 5./6.*P[4,:]
    P_expected[6:,:] = P[5:,:]

    insert_knot_bspline_curve(pu, Tu, P, u, times, Tu_out, P_out)

    assert(np.allclose(Tu_out-Tu_expected, 1.e-13))
    assert(np.allclose(P_out-P_expected, 1.e-13))

# ==========================================================
def test_insert_knot_bspline_curve_2():
    """Example 5.2, The NURBS Book, page 144"""
    pu = 3
    Tu = [0., 0., 0., 0., 1., 2., 3., 4., 5., 5., 5., 5.]
    Tu = np.array(Tu)
    nu = len(Tu) - pu - 1

    d = 3
    P = np.random.random((nu, d))

    u = 2. ; times = 1
    Tu_out = np.zeros(nu+1+pu+1)
    P_out = np.zeros((nu+1, d))

    Tu_expected = [0., 0., 0., 0., 1., 2., 2., 3., 4., 5., 5., 5., 5.]
    Tu_expected = np.array(Tu_expected)
    P_expected = np.zeros((nu+1, d))
    P_expected[0:3,:] = P[0:3,:]
    P_expected[3,:] = 2./3.*P[3,:] + 1./3.*P[2,:]
    P_expected[4,:] = 1./3.*P[4,:] + 2./3.*P[3,:]
    P_expected[5,:] = P[4,:]
    P_expected[6:,:] = P[5:,:]

    insert_knot_bspline_curve(pu, Tu, P, u, times, Tu_out, P_out)

    assert(np.allclose(Tu_out-Tu_expected, 1.e-13))
    assert(np.allclose(P_out-P_expected, 1.e-13))

# ==========================================================
def test_point_on_bspline_surface_1():
    pu = 2
    pv = 2
    Tu = [0., 0., 0., 2./5., 3./5., 1., 1., 1.]
    Tv = [0., 0., 0., 1./5., 1./2., 4./5., 1., 1., 1.]
    Tu = np.array(Tu)
    Tv = np.array(Tv)
    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1

    d = 3
    P = np.random.random((nu, nv, d))
    out = np.zeros(d)

    u = 1./5. ; v = 3./5.
#    expected =
#    point_on_bspline_surface( pu, pv, Tu, Tv, P, u, v, out )
#    assert(np.allclose(out-expected, 1.e-13))


############################################################
if __name__ == '__main__':
    print('')

    test_point_on_bspline_curve_1()
    test_point_on_bspline_surface_1()
    test_insert_knot_bspline_curve_1()
    test_insert_knot_bspline_curve_2()
