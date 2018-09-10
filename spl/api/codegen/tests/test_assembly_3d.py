# coding: utf-8

import os

from sympy import Symbol
from sympy.core.containers import Tuple
from sympy import symbols
from sympy import IndexedBase
from sympy import Matrix
from sympy import Function
from sympy import pi, cos, sin
from sympy import S

from sympde.core import dx, dy, dz
from sympde.core import Constant
from sympde.core import Field
from sympde.core import grad, dot, inner, cross, rot, curl, div
from sympde.core import FunctionSpace
from sympde.core import TestFunction
from sympde.core import VectorTestFunction
from sympde.core import BilinearForm, LinearForm, Integral
from sympde.core import Mapping

from spl.api.codegen.ast import Assembly
from spl.api.codegen.printing import pycode

sanitize = lambda txt: os.linesep.join([s for s in txt.splitlines() if s.strip()])

DEBUG = False
DIM = 3

def test_assembly_bilinear_3d_scalar_1(mapping=False):
    print('============ test_assembly_bilinear_3d_scalar_1 =============')

    if mapping: mapping = Mapping('M', rdim=DIM)

    U = FunctionSpace('U', ldim=DIM)
    V = FunctionSpace('V', ldim=DIM)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    assembly = Assembly(a, name='assembly')
    code = pycode(assembly)
    if DEBUG: print(code)

def test_assembly_bilinear_3d_scalar_2(mapping=False):
    print('============ test_assembly_bilinear_3d_scalar_2 =============')

    if mapping: mapping = Mapping('M', rdim=DIM)

    U = FunctionSpace('U', ldim=DIM)
    V = FunctionSpace('V', ldim=DIM)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    c = Constant('c', real=True, label='mass stabilization')

    expr = dot(grad(v), grad(u)) + c*v*u
    a = BilinearForm((v,u), expr, mapping=mapping)

    assembly = Assembly(a, name='assembly')
    code = pycode(assembly)
    if DEBUG: print(code)

def test_assembly_bilinear_3d_scalar_3(mapping=False):
    print('============ test_assembly_bilinear_3d_scalar_3 =============')

    if mapping: mapping = Mapping('M', rdim=DIM)

    U = FunctionSpace('U', ldim=DIM)
    V = FunctionSpace('V', ldim=DIM)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    F = Field('F', space=V)

    expr = dot(grad(v), grad(u)) + F*v*u
    a = BilinearForm((v,u), expr, mapping=mapping)

    assembly = Assembly(a, name='assembly')
    code = pycode(assembly)
    if DEBUG: print(code)

def test_assembly_bilinear_3d_scalar_4(mapping=False):
    print('============ test_assembly_bilinear_3d_scalar_4 =============')

    if mapping: mapping = Mapping('M', rdim=DIM)

    U = FunctionSpace('U', ldim=DIM)
    V = FunctionSpace('V', ldim=DIM)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    F = Field('F', space=V)
    G = Field('G', space=V)

    expr = dot(grad(G*v), grad(u)) + F*v*u
    a = BilinearForm((v,u), expr, mapping=mapping)

    assembly = Assembly(a, name='assembly')
    code = pycode(assembly)
    if DEBUG: print(code)

def test_assembly_bilinear_3d_block_1(mapping=False):
    print('============ test_assembly_bilinear_3d_block_1 =============')

    if mapping: mapping = Mapping('M', rdim=DIM)

    U = FunctionSpace('U', ldim=DIM, is_block=True, shape=DIM)
    V = FunctionSpace('V', ldim=DIM, is_block=True, shape=DIM)

    v = VectorTestFunction(V, name='v')
    u = VectorTestFunction(U, name='u')

    expr = div(v) * div(u) + dot(curl(v), curl(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    assembly = Assembly(a, name='assembly')
    code = pycode(assembly)
    if DEBUG: print(code)

def test_assembly_linear_3d_scalar_1(mapping=False):
    print('============ test_assembly_linear_3d_scalar_1 =============')

    if mapping: mapping = Mapping('M', rdim=DIM)

    V = FunctionSpace('V', ldim=DIM)
    x,y,z = V.coordinates

    v = TestFunction(V, name='v')

    expr = cos(2*pi*x)*cos(4*pi*y)*cos(4*pi*z)*v
    a = LinearForm(v, expr, mapping=mapping)

    assembly = Assembly(a, name='assembly')
    code = pycode(assembly)
    if DEBUG: print(code)

def test_assembly_linear_3d_scalar_2(mapping=False):
    print('============ test_assembly_linear_3d_scalar_2 =============')

    if mapping: mapping = Mapping('M', rdim=DIM)

    V = FunctionSpace('V', ldim=DIM)
    x,y,z = V.coordinates

    v = TestFunction(V, name='v')

    c = Constant('c', real=True, label='mass stabilization')

    expr = c*cos(2*pi*x)*cos(4*pi*y)*cos(4*pi*z)*v
    a = LinearForm(v, expr, mapping=mapping)

    assembly = Assembly(a, name='assembly')
    code = pycode(assembly)
    if DEBUG: print(code)

def test_assembly_linear_3d_scalar_3(mapping=False):
    print('============ test_assembly_linear_3d_scalar_3 =============')

    if mapping: mapping = Mapping('M', rdim=DIM)

    V = FunctionSpace('V', ldim=DIM)
    x,y,z = V.coordinates

    v = TestFunction(V, name='v')

    F = Field('F', space=V)

    expr = F*v
    a = LinearForm(v, expr, mapping=mapping)

    assembly = Assembly(a, name='assembly')
    code = pycode(assembly)
    if DEBUG: print(code)

def test_assembly_linear_3d_scalar_4(mapping=False):
    print('============ test_assembly_linear_3d_scalar_4 =============')

    if mapping: mapping = Mapping('M', rdim=DIM)

    V = FunctionSpace('V', ldim=DIM)
    x,y,z = V.coordinates

    v = TestFunction(V, name='v')

    F = Field('F', space=V)

    expr = dx(F)*v
    a = LinearForm(v, expr, mapping=mapping)

    assembly = Assembly(a, name='assembly')
    code = pycode(assembly)
    if DEBUG: print(code)

def test_assembly_function_3d_scalar_1(mapping=False):
    print('============ test_assembly_function_3d_scalar_1 =============')

    if mapping: mapping = Mapping('M', rdim=DIM)

    V = FunctionSpace('V', ldim=DIM)
    x,y,z = V.coordinates

    expr = S.One
    a = Integral(expr, space=V, mapping=mapping)

    assembly = Assembly(a, name='assembly')
    code = pycode(assembly)
    if DEBUG: print(code)

def test_assembly_function_3d_scalar_2(mapping=False):
    print('============ test_assembly_function_3d_scalar_2 =============')

    if mapping: mapping = Mapping('M', rdim=DIM)

    V = FunctionSpace('V', ldim=DIM)
    x,y,z = V.coordinates

    F = Field('F', space=V)

    expr = F-cos(2*pi*x)*cos(3*pi*y)*cos(4*pi*z)
    a = Integral(expr, space=V, mapping=mapping)

    assembly = Assembly(a, name='assembly')
    code = pycode(assembly)
    if DEBUG: print(code)

def test_assembly_function_3d_scalar_3(mapping=False):
    print('============ test_assembly_function_3d_scalar_3 =============')

    if mapping: mapping = Mapping('M', rdim=DIM)

    V = FunctionSpace('V', ldim=DIM)
    x,y,z = V.coordinates

    F = Field('F', space=V)

    error = F-cos(2*pi*x)*cos(3*pi*y)*cos(4*pi*z)
    expr = dot(grad(error), grad(error))
    a = Integral(expr, space=V, mapping=mapping)

    assembly = Assembly(a, name='assembly')
    code = pycode(assembly)
    if DEBUG: print(code)

#................................
if __name__ == '__main__':

    # .................................
    # without mapping
    test_assembly_bilinear_3d_scalar_1(mapping=False)
    test_assembly_bilinear_3d_scalar_2(mapping=False)
    test_assembly_bilinear_3d_scalar_3(mapping=False)
    test_assembly_bilinear_3d_scalar_4(mapping=False)
    test_assembly_bilinear_3d_block_1(mapping=False)

    # with mapping
    test_assembly_bilinear_3d_scalar_1(mapping=True)
    test_assembly_bilinear_3d_scalar_2(mapping=True)
    test_assembly_bilinear_3d_scalar_3(mapping=True)
    test_assembly_bilinear_3d_scalar_4(mapping=True)
    test_assembly_bilinear_3d_block_1(mapping=True)
    # .................................

    # .................................
    # without mapping
    test_assembly_linear_3d_scalar_1(mapping=False)
    test_assembly_linear_3d_scalar_2(mapping=False)
    test_assembly_linear_3d_scalar_3(mapping=False)
    test_assembly_linear_3d_scalar_4(mapping=False)

    # with mapping
    test_assembly_linear_3d_scalar_1(mapping=True)
    test_assembly_linear_3d_scalar_2(mapping=True)
    test_assembly_linear_3d_scalar_3(mapping=True)
    test_assembly_linear_3d_scalar_4(mapping=True)
   # .................................

    # .................................
    # without mapping
    test_assembly_function_3d_scalar_1(mapping=False)
    test_assembly_function_3d_scalar_2(mapping=False)
    test_assembly_function_3d_scalar_3(mapping=False)

    # with mapping
    test_assembly_function_3d_scalar_1(mapping=True)
    test_assembly_function_3d_scalar_2(mapping=True)
    test_assembly_function_3d_scalar_3(mapping=True)
    # .................................
