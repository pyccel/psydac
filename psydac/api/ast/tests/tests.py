# -*- coding: UTF-8 -*-

from sympy import symbols

# TODO remove
from pyccel.codegen.printing.pycode import pycode

from sympde.calculus import grad, dot
from sympde.topology import ScalarFunctionSpace
from sympde.topology import elements_of
from sympde.topology import Square
from sympde.topology import Mapping

from psydac.api.new_ast.nodes import Grid
from psydac.api.new_ast.nodes import Element
from psydac.api.new_ast.nodes import GlobalTensorQuadrature
from psydac.api.new_ast.nodes import LocalTensorQuadrature
from psydac.api.new_ast.nodes import TensorQuadrature
from psydac.api.new_ast.nodes import GlobalTensorQuadratureTestBasis
from psydac.api.new_ast.nodes import LocalTensorQuadratureTestBasis
from psydac.api.new_ast.nodes import TensorQuadratureTestBasis
from psydac.api.new_ast.nodes import TensorTestBasis
from psydac.api.new_ast.nodes import GlobalTensorQuadratureTrialBasis
from psydac.api.new_ast.nodes import LocalTensorQuadratureTrialBasis
from psydac.api.new_ast.nodes import TensorQuadratureTrialBasis
from psydac.api.new_ast.nodes import TensorTrialBasis
from psydac.api.new_ast.nodes import GlobalSpan
from psydac.api.new_ast.nodes import Span
#from psydac.api.new_ast.nodes import index_dof
#from psydac.api.new_ast.nodes import ComputePhysical
#from psydac.api.new_ast.nodes import ComputeLogical
from psydac.api.new_ast.nodes import ComputePhysicalBasis
from psydac.api.new_ast.nodes import MatrixLocalBasis
from psydac.api.new_ast.nodes import CoefficientBasis
from psydac.api.new_ast.nodes import StencilMatrixLocalBasis
from psydac.api.new_ast.nodes import StencilVectorLocalBasis
from psydac.api.new_ast.nodes import StencilMatrixGlobalBasis
from psydac.api.new_ast.nodes import StencilVectorGlobalBasis

from psydac.api.new_ast.parser import parse
from sympy import Tuple

# ... abstract model
domain = Square()
M      = Mapping('M', domain.dim)

V      = ScalarFunctionSpace('V', domain)
u,v    = elements_of(V, names='u,v')
expr   = dot(grad(v), grad(u))
# ...

# ...
grid    = Grid()
element = Element()
g_quad  = GlobalTensorQuadrature()
l_quad  = LocalTensorQuadrature()
quad    = TensorQuadrature()

g_basis   = GlobalTensorQuadratureTrialBasis(u)
l_basis   = LocalTensorQuadratureTrialBasis(u)
a_basis   = TensorQuadratureTrialBasis(u)
basis     = TensorTrialBasis(u)
g_basis_v = GlobalTensorQuadratureTestBasis(v)
l_basis_v = LocalTensorQuadratureTestBasis(v)
a_basis_v = TensorQuadratureTestBasis(v)
basis_v   = TensorTestBasis(v)

g_span  = GlobalSpan(u)
span    = Span(u)

coeff   = CoefficientBasis(u)
l_coeff = MatrixLocalBasis(u)


pads    = symbols('p1, p2, p3')[:domain.dim]
l_mat   = StencilMatrixLocalBasis(pads)
l_vec   = StencilVectorLocalBasis(pads)
g_mat   = StencilMatrixGlobalBasis(pads)
g_vec   = StencilVectorGlobalBasis(pads)

x,y = symbols('x, y')
# ...


def test_loop_local_dof_quad_2d_2():
    # ...
    args   = [dx(u),dx(v),dy(u),dy(v)]
    stmts  = [ComputePhysicalBasis(i) for i in args]
    # ...

    # ...

    # ...

    # ...
    #stmts = [loop]
    #loop  = Loop(l_basis, index_dof, stmts)
    # ...

    stmt = parse(Tuple(*stmts), settings={'dim': domain.dim, 'nderiv': 2, 'mapping':M})
    print()
    print(*(pycode(i) for i in stmt), sep='\n')
    print()

test_loop_local_dof_quad_2d_2()
