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
from sympde.core import Domain
from sympde.core import Field
from sympde.core import VectorField
from sympde.core import grad, dot, inner, cross, rot, curl, div
from sympde.core import FunctionSpace
from sympde.core import TestFunction
from sympde.core import VectorFunctionSpace
from sympde.core import VectorTestFunction
from sympde.core import BilinearForm, LinearForm, Integral
from sympde.core import Mapping

from spl.api.codegen.ast import EvalField
from spl.api.codegen.ast import EvalVectorField
from spl.api.codegen.printing import pycode

domain = Domain('\Omega', dim=2)

# ...............................................
#              expected kernels
# ...............................................
# ...............................................


def test_eval_field_2d_1():
    print('============ test_eval_field_2d_1 =============')

    V = FunctionSpace('V', domain)

    F = Field('F', space=V)

    eval_field = EvalField(V, [F], name='eval_field')
    code = pycode(eval_field)
    print(code)

def test_eval_field_2d_2():
    print('============ test_eval_field_2d_2 =============')

    V = FunctionSpace('V', domain)

    F = Field('F', space=V)

    eval_field = EvalField(V, [F, dx(F), dy(F), dx(dx(F))], name='eval_field')
    code = pycode(eval_field)
    print(code)

def test_eval_field_2d_3():
    print('============ test_eval_field_2d_3 =============')

    V = VectorFunctionSpace('V', domain)

    F = VectorField(V, name='F')

    eval_field = EvalVectorField(V, [F[0], dx(F[0]), dy(F[1]), dx(dx(F[1]))], name='eval_field')
    code = pycode(eval_field)
    print(code)

#................................
if __name__ == '__main__':

#    test_eval_field_2d_1()
#    test_eval_field_2d_2()
    test_eval_field_2d_3()
