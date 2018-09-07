# coding: utf-8

# TODO: - remove empty lines at the end of the interface

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

from spl.api.codegen.ast import Interface
from spl.api.codegen.printing import pycode

sanitize = lambda txt: os.linesep.join([s for s in txt.splitlines() if s.strip()])

# ...............................................
#              expected interface
# ...............................................
# ...............................................


def test_interface_bilinear_1d_scalar_1():
    print('============ test_interface_bilinear_1d_scalar_1 =============')

    U = FunctionSpace('U', ldim=1)
    V = FunctionSpace('V', ldim=1)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))

    a = BilinearForm((v,u), expr)

    interface = Interface(a, name='interface')
    code = pycode(interface)
    print(code)

def test_interface_bilinear_1d_scalar_2():
    print('============ test_interface_bilinear_1d_scalar_2 =============')

    U = FunctionSpace('U', ldim=1)
    V = FunctionSpace('V', ldim=1)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    c = Constant('c', real=True, label='mass stabilization')

    expr = dot(grad(v), grad(u)) + c*v*u

    a = BilinearForm((v,u), expr)

    interface = Interface(a, name='interface')
    code = pycode(interface)
    print(code)

def test_interface_bilinear_1d_scalar_3():
    print('============ test_interface_bilinear_1d_scalar_3 =============')

    U = FunctionSpace('U', ldim=1)
    V = FunctionSpace('V', ldim=1)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    F = Field('F', space=V)

    expr = dot(grad(v), grad(u)) + F*v*u

    a = BilinearForm((v,u), expr)

    interface = Interface(a, name='interface')
    code = pycode(interface)
    print(code)

def test_interface_bilinear_1d_scalar_4():
    print('============ test_interface_bilinear_1d_scalar_4 =============')

    U = FunctionSpace('U', ldim=1)
    V = FunctionSpace('V', ldim=1)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    F = Field('F', space=V)
    G = Field('G', space=V)

    expr = dot(grad(G*v), grad(u)) + F*v*u

    a = BilinearForm((v,u), expr)

    interface = Interface(a, name='interface')
    code = pycode(interface)
    print(code)

def test_interface_bilinear_1d_block_1():
    print('============ test_interface_bilinear_1d_block_1 =============')
    # 1d wave problem

    U = FunctionSpace('U', ldim=1)
    V = FunctionSpace('V', ldim=1)

    # trial functions
    u = TestFunction(U, name='u')
    f = TestFunction(V, name='f')

    # test functions
    v = TestFunction(U, name='v')
    w = TestFunction(V, name='w')

    rho = Constant('rho', real=True, label='mass density')
    dt = Constant('dt', real=True, label='time step')

    mass = BilinearForm((v,u), v*u)
    adv  = BilinearForm((v,u), dx(v)*u)

    expr = rho*mass(v,u) + dt*adv(v, f) + dt*adv(w,u) + mass(w,f)
    a = BilinearForm(((v,w), (u,f)), expr)

    interface = Interface(a, name='interface')
    code = pycode(interface)
    print(code)

#................................
if __name__ == '__main__':

#    # ... scalar case
#    test_interface_bilinear_1d_scalar_1()
#    test_interface_bilinear_1d_scalar_2()
#    test_interface_bilinear_1d_scalar_3()
#    test_interface_bilinear_1d_scalar_4()
#    # ...

    # ... block case
    test_interface_bilinear_1d_block_1()
    # ...
