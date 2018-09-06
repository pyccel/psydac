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
from sympde.printing.pycode import pycode

from spl.codegen.ast import Assembly

sanitize = lambda txt: os.linesep.join([s for s in txt.splitlines() if s.strip()])

# ...............................................
#              expected assembly
# ...............................................
expected_bilinear_2d_scalar_1 = """
"""
expected_bilinear_2d_scalar_1 = sanitize(expected_bilinear_2d_scalar_1)

# ...............................................


def test_assembly_bilinear_2d_scalar_1():
    print('============ test_assembly_bilinear_2d_scalar_1 =============')

    U = FunctionSpace('U', ldim=2)
    V = FunctionSpace('V', ldim=2)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))

    a = BilinearForm((v,u), expr)

    assembly = Assembly(a, name='assembly')
    code = pycode(assembly.expr)
    code = sanitize(code)

    from spl.codegen.ast import Kernel

    print('***********')
    print(sanitize(pycode(Kernel(a, name='assembly').expr)))
    print('***********')

    print('-----------')
    print(code)
    print('-----------')

#    assert(str(code) == expected_bilinear_2d_scalar_1)

#................................
if __name__ == '__main__':

    test_assembly_bilinear_2d_scalar_1()
