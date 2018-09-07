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

from spl.codegen.ast import Interface
from spl.codegen.printing import pycode

sanitize = lambda txt: os.linesep.join([s for s in txt.splitlines() if s.strip()])

# ...............................................
#              expected interface
# ...............................................
expected_bilinear_2d_scalar_1 = """
"""
expected_bilinear_2d_scalar_1 = sanitize(expected_bilinear_2d_scalar_1)

# ...............................................


def test_interface_bilinear_2d_scalar_1():
    print('============ test_interface_bilinear_2d_scalar_1 =============')

    U = FunctionSpace('U', ldim=2)
    V = FunctionSpace('V', ldim=2)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))

    a = BilinearForm((v,u), expr)

    interface = Interface(a, name='interface')
    code = pycode(interface)
    code = sanitize(code)

    from spl.codegen.ast import Kernel
    from spl.codegen.ast import Assembly

#    print('***********')
#    print(sanitize(pycode(interface.assembly.kernel)))
#    print(sanitize(pycode(interface.assembly)))
#    print('***********')

    print('-----------')
    print(code)
    print('-----------')

#    assert(str(code) == expected_bilinear_2d_scalar_1)

def test_interface_bilinear_2d_scalar_2():
    print('============ test_interface_bilinear_2d_scalar_2 =============')

    U = FunctionSpace('U', ldim=2)
    V = FunctionSpace('V', ldim=2)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    c = Constant('c', real=True, label='mass stabilization')

    expr = dot(grad(v), grad(u)) + c*v*u

    a = BilinearForm((v,u), expr)

    interface = Interface(a, name='interface')
    code = pycode(interface)
    code = sanitize(code)

    from spl.codegen.ast import Kernel
    from spl.codegen.ast import Assembly

#    print('***********')
#    print(sanitize(pycode(interface.assembly.kernel)))
#    print(sanitize(pycode(interface.assembly)))
#    print('***********')

    print('-----------')
    print(code)
    print('-----------')

#    assert(str(code) == expected_bilinear_2d_scalar_2)

def test_interface_bilinear_2d_scalar_3():
    print('============ test_interface_bilinear_2d_scalar_3 =============')

    U = FunctionSpace('U', ldim=2)
    V = FunctionSpace('V', ldim=2)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    F = Field('F', space=V)

    expr = dot(grad(v), grad(u)) + F*v*u

    a = BilinearForm((v,u), expr)

    interface = Interface(a, name='interface')
    code = pycode(interface)
    code = sanitize(code)

    from spl.codegen.ast import Kernel
    from spl.codegen.ast import Assembly

#    print('***********')
#    print(sanitize(pycode(interface.assembly.kernel)))
#    print(sanitize(pycode(interface.assembly)))
#    print('***********')

    print('-----------')
    print(code)
    print('-----------')

#    assert(str(code) == expected_bilinear_2d_scalar_3)

#................................
if __name__ == '__main__':

#    test_interface_bilinear_2d_scalar_1()
#    test_interface_bilinear_2d_scalar_2()
    test_interface_bilinear_2d_scalar_3()
