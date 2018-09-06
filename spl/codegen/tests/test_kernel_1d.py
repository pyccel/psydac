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
from sympde.printing.pycode import pycode

from spl.codegen.ast import Kernel

sanitize = lambda txt: os.linesep.join([s for s in txt.splitlines() if s.strip()])

# ...............................................
#              expected kernels
# ...............................................
expected_bilinear_1d_scalar_1 = """
def kernel(test_p1, trial_p1, k1, test_bs0, trial_bs0, u0, w0, mat):
    mat[ : ,  : ] = 0.0
    for il1 in range(0, test_p1, 1):
        for jl1 in range(0, trial_p1, 1):
            v = 0.0
            for g1 in range(0, k1, 1):
                x = u0[g1]
                u_x = trial_bs0[jl1, 1, g1]
                v_x = test_bs0[il1, 1, g1]
                wvol = w0[g1]
                v += wvol*u_x*v_x
            mat[il1, -il1 + jl1 + trial_p1] = v
"""
expected_bilinear_1d_scalar_1 = sanitize(expected_bilinear_1d_scalar_1)

expected_bilinear_1d_scalar_2 = """
def kernel(test_p1, trial_p1, k1, test_bs0, trial_bs0, u0, w0, mat, c):
    mat[ : ,  : ] = 0.0
    for il1 in range(0, test_p1, 1):
        for jl1 in range(0, trial_p1, 1):
            v = 0.0
            for g1 in range(0, k1, 1):
                x = u0[g1]
                u_x = trial_bs0[jl1, 1, g1]
                v_x = test_bs0[il1, 1, g1]
                u = trial_bs0[jl1, 0, g1]
                v = test_bs0[il1, 0, g1]
                wvol = w0[g1]
                v += wvol*(c*u*v + u_x*v_x)
            mat[il1, -il1 + jl1 + trial_p1] = v
"""
expected_bilinear_1d_scalar_2 = sanitize(expected_bilinear_1d_scalar_2)

# ...............................................


def test_kernel_bilinear_1d_scalar_1():
    print('============ test_kernel_bilinear_1d_scalar_1 =============')

    U = FunctionSpace('U', ldim=1)
    V = FunctionSpace('V', ldim=1)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))

    a = BilinearForm((v,u), expr)

    kernel = Kernel(a, name='kernel')
    code = pycode(kernel.expr)
    code = sanitize(code)

    assert(str(code) == expected_bilinear_1d_scalar_1)


def test_kernel_bilinear_1d_scalar_2():
    print('============ test_kernel_bilinear_1d_scalar_2 =============')

    U = FunctionSpace('U', ldim=1)
    V = FunctionSpace('V', ldim=1)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    c = Constant('c', real=True, label='mass stabilization')

    expr = dot(grad(v), grad(u)) + c*v*u

    a = BilinearForm((v,u), expr)

    kernel = Kernel(a, name='kernel')
    code = pycode(kernel.expr)
    code = sanitize(code)

#    print('----------')
#    print(code)
#    print('----------')
#    print(expected_bilinear_1d_scalar_2)

    assert(str(code) == expected_bilinear_1d_scalar_2)


# TODO not working yet
#def test_kernel_bilinear_1d_scalar_3():
#    print('============ test_kernel_bilinear_1d_scalar_3 =============')
#
#    U = FunctionSpace('U', ldim=1)
#    V = FunctionSpace('V', ldim=1)
#
#    v = TestFunction(V, name='v')
#    u = TestFunction(U, name='u')
#
#    F = Field('F', space=V)
#
#    expr = dot(grad(v), grad(u)) + F*v*u
#
#    a = BilinearForm((v,u), expr)
#
#    kernel = Kernel(a, name='kernel')
#    code = pycode(kernel.expr)
#    code = sanitize(code)
#
##    print('----------')
##    print(code)
##    print('----------')
#
##    assert(str(code) == expected_bilinear_1d_scalar_3)

#................................
if __name__ == '__main__':

    test_kernel_bilinear_1d_scalar_1()
    test_kernel_bilinear_1d_scalar_2()
##    test_kernel_bilinear_1d_scalar_3()
