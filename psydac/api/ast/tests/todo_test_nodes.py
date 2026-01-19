#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from sympy import Symbol
from sympy import Mul
from sympy import symbols
from sympy import cos

from psydac.pyccel.ast.core import AugAssign
# TODO remove
from psydac.pyccel.codegen.printing.pycode import pycode

from sympde.calculus import grad, dot
from sympde.topology import dx, dy
from sympde.topology import dx1, dx2
from sympde.topology import ScalarFunctionSpace
from sympde.topology import elements_of
from sympde.topology import Square
from sympde.topology import Mapping, IdentityMapping
from sympde.expr     import integral
from sympde.expr     import LinearForm
from sympde.expr     import BilinearForm

from psydac.api.ast.nodes import Grid
from psydac.api.ast.nodes import Element
from psydac.api.ast.nodes import ProductGenerator
from psydac.api.ast.nodes import Loop
from psydac.api.ast.nodes import GlobalTensorQuadratureGrid
from psydac.api.ast.nodes import LocalTensorQuadratureGrid
from psydac.api.ast.nodes import TensorQuadrature
from psydac.api.ast.nodes import MatrixQuadrature
from psydac.api.ast.nodes import GlobalTensorQuadratureTestBasis
from psydac.api.ast.nodes import LocalTensorQuadratureTestBasis
from psydac.api.ast.nodes import TensorQuadratureTestBasis
from psydac.api.ast.nodes import TensorTestBasis
from psydac.api.ast.nodes import GlobalTensorQuadratureTrialBasis
from psydac.api.ast.nodes import LocalTensorQuadratureTrialBasis
from psydac.api.ast.nodes import TensorQuadratureTrialBasis
from psydac.api.ast.nodes import TensorTrialBasis
from psydac.api.ast.nodes import GlobalSpanArray
from psydac.api.ast.nodes import Span
from psydac.api.ast.nodes import BasisAtom
from psydac.api.ast.nodes import PhysicalBasisValue
from psydac.api.ast.nodes import index_element
from psydac.api.ast.nodes import index_quad
from psydac.api.ast.nodes import index_dof, index_dof_test, index_dof_trial
#from psydac.api.ast.nodes import ComputePhysical
#from psydac.api.ast.nodes import ComputeLogical
from psydac.api.ast.nodes import ComputePhysicalBasis
from psydac.api.ast.nodes import ComputeLogicalBasis
from psydac.api.ast.nodes import Reduce
from psydac.api.ast.nodes import Reduction
from psydac.api.ast.nodes import construct_logical_expressions
from psydac.api.ast.nodes import GeometryAtom
from psydac.api.ast.nodes import GeometryExpressions
from psydac.api.ast.nodes import AtomicNode
from psydac.api.ast.nodes import MatrixLocalBasis
from psydac.api.ast.nodes import CoefficientBasis
from psydac.api.ast.nodes import StencilMatrixLocalBasis
from psydac.api.ast.nodes import StencilVectorLocalBasis
from psydac.api.ast.nodes import StencilMatrixGlobalBasis
from psydac.api.ast.nodes import StencilVectorGlobalBasis
from psydac.api.ast.nodes import ElementOf
from psydac.api.ast.nodes import ComputeKernelExpr
from psydac.api.ast.nodes import Reset

from psydac.api.ast.fem    import Block
from psydac.api.ast.fem    import AST
from psydac.api.ast.parser import parse

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
g_quad  = GlobalTensorQuadratureGrid()
l_quad  = LocalTensorQuadratureGrid()
quad    = TensorQuadrature()

g_basis   = GlobalTensorQuadratureTrialBasis(u)
l_basis   = LocalTensorQuadratureTrialBasis(u)
a_basis   = TensorQuadratureTrialBasis(u)
basis     = TensorTrialBasis(u)
g_basis_v = GlobalTensorQuadratureTestBasis(v)
l_basis_v = LocalTensorQuadratureTestBasis(v)
a_basis_v = TensorQuadratureTestBasis(v)
basis_v   = TensorTestBasis(v)

g_span  = GlobalSpanArray(u)
span    = Span(u)

coeff   = CoefficientBasis(u)
l_coeff = MatrixLocalBasis(u)

pads    = symbols('p1, p2, p3')[:domain.dim]
l_mat   = StencilMatrixLocalBasis(u, v, pads)
l_vec   = StencilVectorLocalBasis(v, pads)
g_mat   = StencilMatrixGlobalBasis(u, v, pads)
g_vec   = StencilVectorGlobalBasis(v, pads)

x,y = symbols('x, y')
# ...

#==============================================================================
def basis_atom_2d_1():
    expr = dx(u)
    lhs  = BasisAtom(expr)
    rhs  = PhysicalBasisValue(expr)

    settings = {'dim': domain.dim, 'nderiv': 1, 'mapping': IdentityMapping('M', 2)}
    _parse = lambda expr: parse(expr, settings=settings)

    u_x  = Symbol('u_x')
    u_x1 = Symbol('u_x1')

    assert(lhs.atom == u)
    assert(_parse(lhs) == u_x)
    assert(_parse(rhs) == u_x1)

#==============================================================================
def basis_atom_2d_2():
    expr = dy(dx(u))
    lhs  = BasisAtom(expr)
    rhs  = PhysicalBasisValue(expr)

    settings = {'dim': domain.dim, 'nderiv': 1, 'mapping': IdentityMapping('M', 2)}
    _parse = lambda expr: parse(expr, settings=settings)

    u_xy   = Symbol('u_xy')
    u_x1x2 = Symbol('u_x1x2')

    assert(lhs.atom == u)
    assert(_parse(lhs) == u_xy)
    assert(_parse(rhs) == u_x1x2)

#==============================================================================
def geometry_atom_2d_1():
    expr = M[0]
    lhs  = GeometryAtom(expr)

    settings = {'dim': domain.dim, 'nderiv': 1, 'mapping': IdentityMapping('M', 2)}
    _parse = lambda expr: parse(expr, settings=settings)

    x = Symbol('x')

    assert(_parse(lhs) == x)
    # TODO add assert on parse rhs

#==============================================================================
def loop_local_quad_2d_1():
    stmts = []
    loop  = Loop(l_quad, index_quad, stmts)

    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 1 , 'mapping': M})
    print(pycode(stmt))
    print()

#==============================================================================
def loop_local_dof_quad_2d_1():
    # ...
    stmts = []
    loop  = Loop((l_quad, a_basis), index_quad, stmts)
    # ...

    # ...
    stmts = [loop]
    loop  = Loop(l_basis, index_dof, stmts)
    # ...

    # TODO bug when nderiv=0
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 1, 'mapping': M})
    print()
    print(pycode(stmt))
    print()

#==============================================================================
def loop_local_dof_quad_2d_2():
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
def loop_local_dof_quad_2d_3():
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
def loop_local_dof_quad_2d_4():
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
def loop_global_local_quad_2d_1():
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
def global_span_2d_1():
    # ...
    stmts = []
    loop  = Loop(g_span, index_element, stmts)
    # ...

    # TODO do we need nderiv here?
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 2})
    print(pycode(stmt))
    print()

#==============================================================================
def global_quad_span_2d_1():
    # ...
    stmts = []
    loop  = Loop((g_quad, g_span), index_element, stmts)
    # ...

    # TODO do we need nderiv here?
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 2})
    print(pycode(stmt))
    print()

#==============================================================================
def global_quad_basis_span_2d_1():
    # ...
    stmts = []
    loop  = Loop((g_quad, g_basis, g_span), index_element, stmts)
    # ...

    # TODO do we need nderiv here?
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 2})
    print(pycode(stmt))
    print()

#==============================================================================
def global_quad_basis_span_2d_2():
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
def loop_local_quad_geometry_2d_1():
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
def eval_field_2d_1():
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
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': nderiv, 'mapping': M})
    print(pycode(stmt))
    print()

#==============================================================================
def global_quad_basis_span_2d_vector_1():
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

    stmt = parse(stmt, settings={'dim': domain.dim, 'nderiv': nderiv, 'mapping': M})
    print(pycode(stmt))
    print()

#==============================================================================
def global_quad_basis_span_2d_vector_2():
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
def global_quad_basis_span_2d_matrix_1():
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

    stmt = parse(stmt, settings={'dim': domain.dim, 'nderiv': nderiv, 'mapping': M})
    print(pycode(stmt))
    print()

#==============================================================================
def global_quad_basis_span_2d_matrix_2():
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
def assembly_linear_form_2d_1():
    b = LinearForm(v, integral(domain, v*cos(x)))
    ast = AST(b, TerminalExpr(b, domain)[0], v.space, nquads=(3, 3))

    stmt = parse(ast.expr, settings={'dim': ast.dim, 'nderiv': ast.nderiv, 'mapping': M})
    print(pycode(stmt))
    print()

#==============================================================================
def assembly_bilinear_form_2d_1():
    a = BilinearForm((u,v), integral(domain, dot(grad(u), grad(v))))
    ast = AST(a, TerminalExpr(a, domain)[0], [u.space, v.space], nquads=(3, 3))

    stmt = parse(ast.expr, settings={'dim': ast.dim, 'nderiv': ast.nderiv, 'mapping': M})
    print(pycode(stmt))
    print()



#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()


#==============================================================================
#assembly_linear_form_2d_1()
#assembly_bilinear_form_2d_1()
#global_quad_basis_span_2d_vector_1()
#global_quad_basis_span_2d_vector_2()
#global_quad_basis_span_2d_matrix_1()
#global_quad_basis_span_2d_matrix_2()
#import sys; sys.exit(0)
# tests with assert
#basis_atom_2d_1()
#basis_atom_2d_2()
#geometry_atom_2d_1()

# tests without assert
#loop_local_quad_2d_1()
#loop_local_dof_quad_2d_1()
#loop_local_dof_quad_2d_2()
#loop_local_dof_quad_2d_3()
#loop_local_dof_quad_2d_4()
#loop_global_local_quad_2d_1()
#global_span_2d_1()
#global_quad_span_2d_1()
#global_quad_basis_span_2d_1()
#global_quad_basis_span_2d_2()
#eval_field_2d_1()

#global_quad_basis_span_2d_vector_1()
#global_quad_basis_span_2d_vector_2()
#global_quad_basis_span_2d_matrix_1()
#global_quad_basis_span_2d_matrix_2()
#loop_local_quad_geometry_2d_1()
#assembly_linear_form_2d_1()
#assembly_bilinear_form_2d_1()

