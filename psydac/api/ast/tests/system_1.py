# -*- coding: UTF-8 -*-

from sympy import pi, sin, Tuple, Matrix

from sympde.calculus import grad, dot, inner
from sympde.topology import VectorFunctionSpace
from sympde.topology import element_of
from sympde.topology import Square
from sympde.topology import Mapping, IdentityMapping#, PolarMapping
from sympde.expr     import integral
from sympde.expr     import LinearForm
from sympde.expr     import BilinearForm
from sympde.expr     import Norm

from psydac.api.new_ast.fem  import AST
from psydac.api.new_ast.parser import parse

from pyccel.codegen.printing.pycode import pycode

# ...
# ... abstract model

domain = Square()
M      = Mapping('M', domain.dim)
V = VectorFunctionSpace('W', domain)

x,y = domain.coordinates

f = Tuple(2*pi**2*sin(pi*x)*sin(pi*y),
          2*pi**2*sin(pi*x)*sin(pi*y))

Fe = Tuple(sin(pi*x)*sin(pi*y), sin(pi*x)*sin(pi*y))

F = element_of(V, name='F')

u,v = [element_of(V, name=i) for i in ['u', 'v']]

int_0 = lambda expr: integral(domain , expr)

b = BilinearForm((u,v), int_0(inner(grad(v), grad(u))))
l = LinearForm(v, int_0(dot(f, v)))

error = Matrix([F[0]-Fe[0], F[1]-Fe[1]])
l2norm_F = Norm(error, domain, kind='l2')
h1norm_F = Norm(error, domain, kind='h1')


print('============================================Norm===========================================')
ast_norm = AST(h1norm_F,V, M)
stmt_n = parse(ast_norm.expr, settings={'dim': ast_norm.dim, 'nderiv': ast_norm.nderiv, 'mapping':M})
print(pycode(stmt_n))

print('============================================LinearForm===========================================')
ast_l    = AST(l, V, M)
stmt_l = parse(ast_l.expr, settings={'dim': ast_l.dim, 'nderiv': ast_l.nderiv, 'mapping':M})
print(pycode(stmt_l))

print('============================================BilinearForm=========================================')
ast_b    = AST(b, [V,V], M)
stmt_b = parse(ast_b.expr, settings={'dim': ast_b.dim, 'nderiv': ast_b.nderiv, 'mapping':M})
print(pycode(stmt_b))
