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

from spl.codegen.ast import EvalField
from spl.codegen.printing import pycode

sanitize = lambda txt: os.linesep.join([s for s in txt.splitlines() if s.strip()])

# ...............................................
#              expected kernels
# ...............................................
expected_2d_1 = """
"""
expected_2d_1 = sanitize(expected_2d_1)
# ...............................................


def test_field_2d_1():
    print('============ test_field_2d_1 =============')

    V = FunctionSpace('V', ldim=2)

    F = Field('F', space=V)

#    eval_field = EvalField(V, [F, dx(F), dy(F)], name='eval_field')
    eval_field = EvalField(V, [F], name='eval_field')
    code = pycode(eval_field)
    code = sanitize(code)

    print('-----------')
    print(code)
    print('-----------')

#    assert(str(code) == expected_2d_1)

#................................
if __name__ == '__main__':

    test_field_2d_1()
