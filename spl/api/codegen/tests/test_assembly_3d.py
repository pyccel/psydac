# coding: utf-8

# TODO: - remove empty lines at the end of the assembly

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

from spl.api.codegen.ast import Assembly
from spl.api.codegen.printing import pycode

sanitize = lambda txt: os.linesep.join([s for s in txt.splitlines() if s.strip()])

# ...............................................
#              expected assembly
# ...............................................
# ...............................................


def test_assembly_bilinear_3d_scalar_1():
    print('============ test_assembly_bilinear_3d_scalar_1 =============')

    U = FunctionSpace('U', ldim=3)
    V = FunctionSpace('V', ldim=3)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))

    a = BilinearForm((v,u), expr)

    assembly = Assembly(a, name='assembly')
    code = pycode(assembly)
    print(code)

def test_assembly_bilinear_3d_scalar_2():
    print('============ test_assembly_bilinear_3d_scalar_2 =============')

    U = FunctionSpace('U', ldim=3)
    V = FunctionSpace('V', ldim=3)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    c = Constant('c', real=True, label='mass stabilization')

    expr = dot(grad(v), grad(u)) + c*v*u

    a = BilinearForm((v,u), expr)

    assembly = Assembly(a, name='assembly')
    code = pycode(assembly)
    print(code)

def test_assembly_bilinear_3d_scalar_3():
    print('============ test_assembly_bilinear_3d_scalar_3 =============')

    U = FunctionSpace('U', ldim=3)
    V = FunctionSpace('V', ldim=3)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    F = Field('F', space=V)

    expr = dot(grad(v), grad(u)) + F*v*u

    a = BilinearForm((v,u), expr)

    assembly = Assembly(a, name='assembly')
    code = pycode(assembly)
    print(code)

def test_assembly_bilinear_3d_scalar_4():
    print('============ test_assembly_bilinear_3d_scalar_4 =============')

    U = FunctionSpace('U', ldim=3)
    V = FunctionSpace('V', ldim=3)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    F = Field('F', space=V)
    G = Field('G', space=V)

    expr = dot(grad(G*v), grad(u)) + F*v*u

    a = BilinearForm((v,u), expr)

    assembly = Assembly(a, name='assembly')
    code = pycode(assembly)
    print(code)

def test_assembly_bilinear_3d_block_1():
    print('============ test_assembly_bilinear_3d_block_1 =============')

    U = FunctionSpace('U', ldim=3, is_block=True, shape=3)
    V = FunctionSpace('V', ldim=3, is_block=True, shape=3)

    v = VectorTestFunction(V, name='v')
    u = VectorTestFunction(U, name='u')

    expr = div(v) * div(u) + dot(curl(v), curl(u))

    a = BilinearForm((v,u), expr)

    assembly = Assembly(a, name='assembly')
    code = pycode(assembly)
    print(code)

#................................
if __name__ == '__main__':

#    # ... scalar case
#    test_assembly_bilinear_3d_scalar_1()
#    test_assembly_bilinear_3d_scalar_2()
#    test_assembly_bilinear_3d_scalar_3()
#    test_assembly_bilinear_3d_scalar_4()
#    # ...

    # ... block case
    test_assembly_bilinear_3d_block_1()
    # ...
