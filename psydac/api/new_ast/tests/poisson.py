# -*- coding: UTF-8 -*-

from sympy import Symbol
from sympy import Mul
from sympy import symbols
from sympy import cos

from pyccel.ast import Assign
from pyccel.ast import AugAssign
# TODO remove
from pyccel.codegen.printing.pycode import pycode

from sympde.calculus import grad, dot
from sympde.topology import (dx, dy, dz)
from sympde.topology import (dx1, dx2, dx3)
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of, elements_of
from sympde.topology import Square
from sympde.topology import Mapping
from sympde.expr     import integral
from sympde.expr     import LinearForm
from sympde.expr     import BilinearForm

from psydac.api.new_ast.nodes import Grid
from psydac.api.new_ast.nodes import Element
from psydac.api.new_ast.nodes import TensorIterator
from psydac.api.new_ast.nodes import TensorGenerator
from psydac.api.new_ast.nodes import ProductIterator
from psydac.api.new_ast.nodes import ProductGenerator
from psydac.api.new_ast.nodes import Loop
from psydac.api.new_ast.nodes import GlobalTensorQuadrature
from psydac.api.new_ast.nodes import LocalTensorQuadrature
from psydac.api.new_ast.nodes import TensorQuadrature
from psydac.api.new_ast.nodes import MatrixQuadrature
from psydac.api.new_ast.nodes import GlobalTensorQuadratureBasis
from psydac.api.new_ast.nodes import LocalTensorQuadratureBasis
from psydac.api.new_ast.nodes import TensorQuadratureBasis
from psydac.api.new_ast.nodes import TensorBasis
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
from psydac.api.new_ast.nodes import BasisAtom
from psydac.api.new_ast.nodes import PhysicalBasisValue
from psydac.api.new_ast.nodes import LogicalBasisValue
from psydac.api.new_ast.nodes import index_element
from psydac.api.new_ast.nodes import index_quad
from psydac.api.new_ast.nodes import index_dof, index_dof_test, index_dof_trial
#from psydac.api.new_ast.nodes import ComputePhysical
#from psydac.api.new_ast.nodes import ComputeLogical
from psydac.api.new_ast.nodes import ComputePhysicalBasis
from psydac.api.new_ast.nodes import ComputeLogicalBasis
from psydac.api.new_ast.nodes import Reduce
from psydac.api.new_ast.nodes import Reduction
from psydac.api.new_ast.nodes import construct_logical_expressions
from psydac.api.new_ast.nodes import GeometryAtom
from psydac.api.new_ast.nodes import GeometryExpressions
from psydac.api.new_ast.nodes import PhysicalGeometryValue
from psydac.api.new_ast.nodes import LogicalGeometryValue
from psydac.api.new_ast.nodes import AtomicNode
from psydac.api.new_ast.nodes import MatrixLocalBasis
from psydac.api.new_ast.nodes import CoefficientBasis
from psydac.api.new_ast.nodes import StencilMatrixLocalBasis
from psydac.api.new_ast.nodes import StencilVectorLocalBasis
from psydac.api.new_ast.nodes import StencilMatrixGlobalBasis
from psydac.api.new_ast.nodes import StencilVectorGlobalBasis
from psydac.api.new_ast.nodes import ElementOf
from psydac.api.new_ast.nodes import WeightedVolumeQuadrature
from psydac.api.new_ast.nodes import ComputeKernelExpr
from psydac.api.new_ast.nodes import AST
from psydac.api.new_ast.nodes import Block
from psydac.api.new_ast.nodes import Reset

from psydac.api.new_ast.parser import parse

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


a = BilinearForm((u,v), integral(domain, dot(grad(u), grad(v))))
ast_b = AST(a)
b = LinearForm(v, integral(domain, v*cos(x)))
ast_l = AST(b)

stmt_b = parse(ast_b.expr, settings={'dim': ast_b.dim, 'nderiv': ast_b.nderiv})
stmt_l = parse(ast_l.expr, settings={'dim': ast_l.dim, 'nderiv': ast_l.nderiv})
print('============================================BilinearForm=========================================')
print()
print(pycode(stmt_b))
print()
print('============================================LinearForm===========================================')
print()
print(pycode(stmt_l))

