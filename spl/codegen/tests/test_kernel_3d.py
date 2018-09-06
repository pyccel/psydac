# coding: utf-8

# TODO: - remove empty lines at the end of the kernel

import os

from sympy import Symbol
from sympy.core.containers import Tuple
from sympy import symbols
from sympy import IndexedBase
from sympy import Matrix
from sympy import Function
from sympy import pi, cos, sin

from sympde.core import dx, dy, dz
from sympde.core import Constant
from sympde.core import Field
from sympde.core import grad, dot, inner, cross, rot, curl, div
from sympde.core import FunctionSpace
from sympde.core import TestFunction
from sympde.core import VectorTestFunction
from sympde.core import BilinearForm, LinearForm, FunctionForm
from sympde.core import Mapping

from spl.codegen.ast import Kernel
from spl.codegen.printing import pycode

sanitize = lambda txt: os.linesep.join([s for s in txt.splitlines() if s.strip()])

# ...............................................
#              expected kernels
# ...............................................
expected_bilinear_3d_scalar_1 = """
def kernel(test_p1, test_p2, test_p3, trial_p1, trial_p2, trial_p3, test_bs1, test_bs2, test_bs3, trial_bs1, trial_bs2, trial_bs3, u1, u2, u3, w1, w2, w3, mat_00):
    k1 = len(u1)
    k2 = len(u2)
    k3 = len(u3)
    mat_00[ : ,  : ,  : ,  : ,  : ,  : ] = 0.0
    for il1 in range(0, test_p1, 1):
        for il2 in range(0, test_p2, 1):
            for il3 in range(0, test_p3, 1):
                for jl1 in range(0, trial_p1, 1):
                    for jl2 in range(0, trial_p2, 1):
                        for jl3 in range(0, trial_p3, 1):
                            v_00 = 0.0
                            for g1 in range(0, k1, 1):
                                for g2 in range(0, k2, 1):
                                    for g3 in range(0, k3, 1):
                                        x = u1[g1]
                                        y = u2[g2]
                                        z = u3[g3]
                                        u_x = trial_bs1[jl1, 1, g1]*trial_bs2[jl2, 0, g2]*trial_bs3[jl3, 0, g3]
                                        v_x = test_bs1[il1, 1, g1]*test_bs2[il2, 0, g2]*test_bs3[il3, 0, g3]
                                        u_y = trial_bs1[jl1, 0, g1]*trial_bs2[jl2, 1, g2]*trial_bs3[jl3, 0, g3]
                                        v_y = test_bs1[il1, 0, g1]*test_bs2[il2, 1, g2]*test_bs3[il3, 0, g3]
                                        u_z = trial_bs1[jl1, 0, g1]*trial_bs2[jl2, 0, g2]*trial_bs3[jl3, 1, g3]
                                        v_z = test_bs1[il1, 0, g1]*test_bs2[il2, 0, g2]*test_bs3[il3, 1, g3]
                                        wvol = w1[g1]*w2[g2]*w3[g3]
                                        v_00 += wvol*(u_x*v_x + u_y*v_y + u_z*v_z)
                            mat_00[il1, il2, il3, -il1 + jl1 + trial_p1, -il2 + jl2 + trial_p2, -il3 + jl3 + trial_p3] = v_00
"""
expected_bilinear_3d_scalar_1 = sanitize(expected_bilinear_3d_scalar_1)

expected_bilinear_3d_scalar_2 = """
def kernel(test_p1, test_p2, test_p3, trial_p1, trial_p2, trial_p3, test_bs1, test_bs2, test_bs3, trial_bs1, trial_bs2, trial_bs3, u1, u2, u3, w1, w2, w3, mat_00, c):
    k1 = len(u1)
    k2 = len(u2)
    k3 = len(u3)
    mat_00[ : ,  : ,  : ,  : ,  : ,  : ] = 0.0
    for il1 in range(0, test_p1, 1):
        for il2 in range(0, test_p2, 1):
            for il3 in range(0, test_p3, 1):
                for jl1 in range(0, trial_p1, 1):
                    for jl2 in range(0, trial_p2, 1):
                        for jl3 in range(0, trial_p3, 1):
                            v_00 = 0.0
                            for g1 in range(0, k1, 1):
                                for g2 in range(0, k2, 1):
                                    for g3 in range(0, k3, 1):
                                        x = u1[g1]
                                        y = u2[g2]
                                        z = u3[g3]
                                        u_x = trial_bs1[jl1, 1, g1]*trial_bs2[jl2, 0, g2]*trial_bs3[jl3, 0, g3]
                                        v_x = test_bs1[il1, 1, g1]*test_bs2[il2, 0, g2]*test_bs3[il3, 0, g3]
                                        u_y = trial_bs1[jl1, 0, g1]*trial_bs2[jl2, 1, g2]*trial_bs3[jl3, 0, g3]
                                        v_y = test_bs1[il1, 0, g1]*test_bs2[il2, 1, g2]*test_bs3[il3, 0, g3]
                                        u_z = trial_bs1[jl1, 0, g1]*trial_bs2[jl2, 0, g2]*trial_bs3[jl3, 1, g3]
                                        v_z = test_bs1[il1, 0, g1]*test_bs2[il2, 0, g2]*test_bs3[il3, 1, g3]
                                        u = trial_bs1[jl1, 0, g1]*trial_bs2[jl2, 0, g2]*trial_bs3[jl3, 0, g3]
                                        v = test_bs1[il1, 0, g1]*test_bs2[il2, 0, g2]*test_bs3[il3, 0, g3]
                                        wvol = w1[g1]*w2[g2]*w3[g3]
                                        v_00 += wvol*(c*u*v + u_x*v_x + u_y*v_y + u_z*v_z)
                            mat_00[il1, il2, il3, -il1 + jl1 + trial_p1, -il2 + jl2 + trial_p2, -il3 + jl3 + trial_p3] = v_00
"""
expected_bilinear_3d_scalar_2 = sanitize(expected_bilinear_3d_scalar_2)

# ...............................................


def test_kernel_bilinear_3d_scalar_1():
    print('============ test_kernel_bilinear_3d_scalar_1 =============')

    U = FunctionSpace('U', ldim=3)
    V = FunctionSpace('V', ldim=3)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))

    a = BilinearForm((v,u), expr)

    kernel = Kernel(a, name='kernel')
    code = pycode(kernel)
    code = sanitize(code)

#    print('-----------')
#    print(code)
#    print('-----------')

    assert(str(code) == expected_bilinear_3d_scalar_1)

def test_kernel_bilinear_3d_scalar_2():
    print('============ test_kernel_bilinear_3d_scalar_2 =============')

    U = FunctionSpace('U', ldim=3)
    V = FunctionSpace('V', ldim=3)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    c = Constant('c', real=True, label='mass stabilization')

    expr = dot(grad(v), grad(u)) + c*v*u

    a = BilinearForm((v,u), expr)

    kernel = Kernel(a, name='kernel')
    code = pycode(kernel)
    code = sanitize(code)

#    print('-----------')
#    print(code)
#    print('-----------')

    assert(str(code) == expected_bilinear_3d_scalar_2)

#................................
if __name__ == '__main__':

    test_kernel_bilinear_3d_scalar_1()
    test_kernel_bilinear_3d_scalar_2()
