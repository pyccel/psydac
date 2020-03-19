# -*- coding: UTF-8 -*-

from sympy import Symbol
from sympy import Mul
from sympy import symbols
from sympy import cos

from pyccel.ast import AugAssign
# TODO remove
from pyccel.codegen.printing.pycode import pycode

from sympde.calculus import grad, dot
from sympde.topology import ScalarFunctionSpace
from sympde.topology import elements_of
from sympde.topology import Square
from sympde.topology import Mapping
from sympde.expr     import integral
from sympde.expr     import LinearForm
from sympde.expr     import BilinearForm

from psydac.api.new_ast.nodes import Grid
from psydac.api.new_ast.nodes import Element
from psydac.api.new_ast.nodes import ProductGenerator
from psydac.api.new_ast.nodes import Loop
from psydac.api.new_ast.nodes import GlobalTensorQuadrature
from psydac.api.new_ast.nodes import LocalTensorQuadrature
from psydac.api.new_ast.nodes import TensorQuadrature
from psydac.api.new_ast.nodes import MatrixQuadrature
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
from psydac.api.new_ast.nodes import AtomicNode
from psydac.api.new_ast.nodes import MatrixLocalBasis
from psydac.api.new_ast.nodes import CoefficientBasis
from psydac.api.new_ast.nodes import StencilMatrixLocalBasis
from psydac.api.new_ast.nodes import StencilVectorLocalBasis
from psydac.api.new_ast.nodes import StencilMatrixGlobalBasis
from psydac.api.new_ast.nodes import StencilVectorGlobalBasis
from psydac.api.new_ast.nodes import ElementOf
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

#==============================================================================
def test_basis_atom_2d_1():
    expr = dx(u)
    lhs  = BasisAtom(expr)
    rhs  = PhysicalBasisValue(expr)

    settings = {'dim': domain.dim, 'nderiv': 1}
    _parse = lambda expr: parse(expr, settings=settings)

    u_x  = Symbol('u_x')
    u_x1 = Symbol('u_x1')

    assert(lhs.atom == u)
    assert(_parse(lhs) == u_x)
    assert(_parse(rhs) == u_x1)

#==============================================================================
def test_basis_atom_2d_2():
    expr = dy(dx(u))
    lhs  = BasisAtom(expr)
    rhs  = PhysicalBasisValue(expr)

    settings = {'dim': domain.dim, 'nderiv': 1}
    _parse = lambda expr: parse(expr, settings=settings)

    u_xy   = Symbol('u_xy')
    u_x1x2 = Symbol('u_x1x2')

    assert(lhs.atom == u)
    assert(_parse(lhs) == u_xy)
    assert(_parse(rhs) == u_x1x2)

#==============================================================================
def test_geometry_atom_2d_1():
    expr = M[0]
    lhs  = GeometryAtom(expr)

    settings = {'dim': domain.dim, 'nderiv': 1, 'mapping': M}
    _parse = lambda expr: parse(expr, settings=settings)

    x = Symbol('x')

    assert(_parse(lhs) == x)
    # TODO add assert on parse rhs

#==============================================================================
def test_loop_local_quad_2d_1():
    stmts = []
    loop  = Loop(l_quad, index_quad, stmts)

    stmt = parse(loop, settings={'dim': domain.dim})
    print(pycode(stmt))
    print()

#==============================================================================
def test_loop_local_dof_quad_2d_1():
    # ...
    stmts = []
    loop  = Loop((l_quad, a_basis), index_quad, stmts)
    # ...

    # ...
    stmts = [loop]
    loop  = Loop(l_basis, index_dof, stmts)
    # ...

    # TODO bug when nderiv=0
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 1})
    print()
    print(pycode(stmt))
    print()

#==============================================================================
def test_loop_local_dof_quad_2d_2():
    # ...
    args   = [dx(u), dx(dy(u)), dy(dy(u)), dx(u) + dy(u)]
    stmts  = [ComputePhysicalBasis(i) for i in args]
    # ...

    # ...
    loop  = Loop((l_quad, a_basis), index_quad, stmts)
    # ...

    # ...
    stmts = [loop]
    loop  = Loop(l_basis, index_dof, stmts)
    # ...

    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 3})
    print()
    print(pycode(stmt))
    print()

#==============================================================================
def test_loop_local_dof_quad_2d_3():
    # ...
    stmts  = [dx1(u)]
    stmts  = [ComputeLogicalBasis(i) for i in stmts]
    # ...

    # ...
    loop  = Loop((l_quad, a_basis), index_quad, stmts)
    # ...

    # ...
    stmts = [loop]
    loop  = Loop(l_basis, index_dof, stmts)
    # ...

    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 3})
    print()
    print(pycode(stmt))
    print()

#==============================================================================
def test_loop_local_dof_quad_2d_4():
    # ...
    stmts = []

    expressions  = [dx1(u), dx2(u)]
    stmts += [ComputeLogicalBasis(i) for i in expressions]

    expressions  = [dx(u)]
    stmts += [ComputePhysicalBasis(i) for i in expressions]
    # ...

    # ...
    loop  = Loop((l_quad, a_basis), index_quad, stmts)
    # ...

    # ...
    stmts = [loop]
    loop  = Loop(l_basis, index_dof, stmts)
    # ...

    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 3})
    print()
    print(pycode(stmt))
    print()

#==============================================================================
def test_loop_global_local_quad_2d_1():
    # ...
    stmts = []
    loop  = Loop(l_quad, index_quad, stmts)
    # ...

    # ...
    stmts = [loop]
    loop  = Loop(g_quad, index_element, stmts)
    # ...

    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 2})
    print(pycode(stmt))
    print()

#==============================================================================
def test_global_span_2d_1():
    # ...
    stmts = []
    loop  = Loop(g_span, index_element, stmts)
    # ...

    # TODO do we need nderiv here?
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 2})
    print(pycode(stmt))
    print()

#==============================================================================
def test_global_quad_span_2d_1():
    # ...
    stmts = []
    loop  = Loop((g_quad, g_span), index_element, stmts)
    # ...

    # TODO do we need nderiv here?
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 2})
    print(pycode(stmt))
    print()

#==============================================================================
def test_global_quad_basis_span_2d_1():
    # ...
    stmts = []
    loop  = Loop((g_quad, g_basis, g_span), index_element, stmts)
    # ...

    # TODO do we need nderiv here?
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 2})
    print(pycode(stmt))
    print()

#==============================================================================
def test_global_quad_basis_span_2d_2():
    # ...
    nderiv = 2
    stmts = construct_logical_expressions(u, nderiv)

    expressions = [dx(u), dx(dy(u)), dy(dy(u))]
    stmts  += [ComputePhysicalBasis(i) for i in expressions]
    # ...

    # ...
    stmts  += [Reduction('+', ComputePhysicalBasis(dx(u)*dx(v)))]
    # ...

    # ...
    loop  = Loop((l_quad, a_basis), index_quad, stmts)
    # ...

    # ...
    stmts = [loop]
    loop  = Loop(l_basis, index_dof, stmts)
    # ...

    # ...
    stmts = [loop]
    loop  = Loop((g_quad, g_basis, g_span), index_element, stmts)
    # ...

    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': nderiv})
    print(pycode(stmt))
    print()

#==============================================================================
def test_loop_local_quad_geometry_2d_1():
    # ...
    nderiv = 1
    stmts = []
    loop  = Loop((l_quad, GeometryExpressions(M, nderiv)), index_quad, stmts)
    # ...

    settings = {'dim': domain.dim, 'nderiv': nderiv, 'mapping': M}
    _parse = lambda expr: parse(expr, settings=settings)

    stmt = _parse(loop)
    print()
    print(pycode(stmt))

    print()

#==============================================================================
def test_eval_field_2d_1():
    # ...
    args = [dx1(u), dx2(u)]

    # TODO improve
    stmts = [AugAssign(ProductGenerator(MatrixQuadrature(i), index_quad),
                                        '+', Mul(coeff,AtomicNode(i)))
             for i in args]
    # ...

    # ...
    nderiv = 1
    body = construct_logical_expressions(u, nderiv)
    # ...

    # ...
    stmts = body + stmts
    loop  = Loop((l_quad, a_basis), index_quad, stmts)
    # ...

    # ...
    stmts = [loop]
    loop  = Loop((l_basis, l_coeff), index_dof, stmts)
    # ...

    # TODO do we need nderiv here?
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': nderiv})
    print(pycode(stmt))
    print()

#==============================================================================
def test_global_quad_basis_span_2d_vector_1():
    # ...
    nderiv = 2
    stmts = construct_logical_expressions(v, nderiv)

    expressions = [dx(v), dx(dy(v)), dy(dy(v))]
    stmts  += [ComputePhysicalBasis(i) for i in expressions]
    # ...

    # ...
    loop  = Loop((l_quad, a_basis), index_quad, stmts)
    # ...

    # ...
    loop = Reduce('+', ComputeKernelExpr(dx(v)*cos(x+y)), ElementOf(l_vec), loop)
    # ...

    # ... loop over tests
    stmts = [loop]
    loop  = Loop(l_basis_v, index_dof_test, stmts)
    # ...

    # ...
    body  = (Reset(l_vec), loop)
    stmts = Block(body)
    # ...

    # ...
    loop  = Loop((g_quad, g_basis_v, g_span), index_element, stmts)
    # ...

    # ...
    body = (Reset(g_vec), Reduce('+', l_vec, g_vec, loop))
    stmt = Block(body)
    # ...

    stmt = parse(stmt, settings={'dim': domain.dim, 'nderiv': nderiv})
    print(pycode(stmt))
    print()

#==============================================================================
def test_global_quad_basis_span_2d_vector_2():
    # ...
    nderiv = 1
    stmts = construct_logical_expressions(v, nderiv)

#    expressions = [dx(v), v]  # TODO Wrong result
    expressions = [dx(v), dy(v)]
    stmts  += [ComputePhysicalBasis(i) for i in expressions]
    # ...

    # ... case with mapping <> identity
    loop  = Loop((l_quad, a_basis, GeometryExpressions(M, nderiv)), index_quad, stmts)
    # ...

    # ...
    loop = Reduce('+', ComputeKernelExpr(dx(v)*cos(x+y)), ElementOf(l_vec), loop)
    # ...

    # ... loop over tests
    stmts = [loop]
    loop  = Loop(l_basis_v, index_dof_test, stmts)
    # ...

    # ...
    body  = (Reset(l_vec), loop)
    stmts = Block(body)
    # ...

    # ...
    loop  = Loop((g_quad, g_basis_v, g_span), index_element, stmts)
    # ...

    # ...
    body = (Reset(g_vec), Reduce('+', l_vec, g_vec, loop))
    stmt = Block(body)
    # ...

    stmt = parse(stmt, settings={'dim': domain.dim, 'nderiv': nderiv, 'mapping': M})
    print(pycode(stmt))
    print()

#==============================================================================
def test_global_quad_basis_span_2d_matrix_1():
    # ...
    nderiv = 2
    stmts = construct_logical_expressions(u, nderiv)

    expressions = [dx(u), dx(dy(u)), dy(dy(u))]
    stmts  += [ComputePhysicalBasis(i) for i in expressions]
    # ...

    # ...
    loop  = Loop((l_quad, a_basis), index_quad, stmts)
    # ...

    # ...
    loop = Reduce('+', ComputeKernelExpr(dx(u)*dx(v)), ElementOf(l_mat), loop)
    # ...

    # ... loop over trials
    stmts = [loop]
    loop  = Loop(l_basis, index_dof_trial, stmts)
    # ...

    # ... loop over tests
    stmts = [loop]
    loop  = Loop(l_basis_v, index_dof_test, stmts)
    # ...

    # ...
    body  = (Reset(l_mat), loop)
    stmts = Block(body)
    # ...

    # ...
    loop  = Loop((g_quad, g_basis, g_basis_v, g_span), index_element, stmts)
    # ...

    # ...
    body = (Reset(g_mat), Reduce('+', l_mat, g_mat, loop))
    stmt = Block(body)
    # ...

    stmt = parse(stmt, settings={'dim': domain.dim, 'nderiv': nderiv})
    print(pycode(stmt))
    print()

#==============================================================================
def test_global_quad_basis_span_2d_matrix_2():
    # ...
    nderiv = 1
    stmts = construct_logical_expressions(u, nderiv)

    expressions = [dx(v), dy(v), dx(u), dy(u)]
    stmts  += [ComputePhysicalBasis(i) for i in expressions]
    # ...

    # ...
    loop  = Loop((l_quad, a_basis, GeometryExpressions(M, nderiv)), index_quad, stmts)
    # ...

    # ...
    loop = Reduce('+', ComputeKernelExpr(dx(u)*dx(v)), ElementOf(l_mat), loop)
    # ...

    # ... loop over trials
    stmts = [loop]
    loop  = Loop(l_basis, index_dof_trial, stmts)
    # ...

    # ... loop over tests
    stmts = [loop]
    loop  = Loop(l_basis_v, index_dof_test, stmts)
    # ...

    # ...
    body  = (Reset(l_mat), loop)
    stmts = Block(body)
    # ...

    # ...
    loop  = Loop((g_quad, g_basis, g_basis_v, g_span), index_element, stmts)
    # ...

    # ...
    body = (Reset(g_mat), Reduce('+', l_mat, g_mat, loop))
    stmt = Block(body)
    # ...

    stmt = parse(stmt, settings={'dim': domain.dim, 'nderiv': nderiv, 'mapping': M})
    print(pycode(stmt))
    print()

#==============================================================================
def test_assembly_linear_form_2d_1():
    b = LinearForm(v, integral(domain, v*cos(x)))
    ast = AST(b)

    stmt = parse(ast.expr, settings={'dim': ast.dim, 'nderiv': ast.nderiv})
    print(pycode(stmt))
    print()

#==============================================================================
def test_assembly_bilinear_form_2d_1():
    a = BilinearForm((u,v), integral(domain, dot(grad(u), grad(v))))
    ast = AST(a)

    stmt = parse(ast.expr, settings={'dim': ast.dim, 'nderiv': ast.nderiv})
    print(pycode(stmt))
    print()



#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()


#==============================================================================
#test_assembly_linear_form_2d_1()
#test_assembly_bilinear_form_2d_1()
#test_global_quad_basis_span_2d_vector_1()
#test_global_quad_basis_span_2d_vector_2()
#test_global_quad_basis_span_2d_matrix_1()
#test_global_quad_basis_span_2d_matrix_2()
#import sys; sys.exit(0)
# tests with assert
test_basis_atom_2d_1()
test_basis_atom_2d_2()
test_geometry_atom_2d_1()

# tests without assert
test_loop_local_quad_2d_1()
test_loop_local_dof_quad_2d_1()
test_loop_local_dof_quad_2d_2()
test_loop_local_dof_quad_2d_3()
test_loop_local_dof_quad_2d_4()
test_loop_global_local_quad_2d_1()
test_global_span_2d_1()
test_global_quad_span_2d_1()
test_global_quad_basis_span_2d_1()
test_global_quad_basis_span_2d_2()
test_eval_field_2d_1()

test_global_quad_basis_span_2d_vector_1()
test_global_quad_basis_span_2d_vector_2()
test_global_quad_basis_span_2d_matrix_1()
test_global_quad_basis_span_2d_matrix_2()
test_loop_local_quad_geometry_2d_1()
test_assembly_linear_form_2d_1()
test_assembly_bilinear_form_2d_1()

