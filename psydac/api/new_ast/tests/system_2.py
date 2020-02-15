# -*- coding: UTF-8 -*-


from sympy import symbols
from sympy import pi, cos, sin, Tuple, Matrix

from sympde.topology import (dx, dy, dz)
from sympde.topology import (dx1, dx2, dx3)
from sympde.calculus import grad, dot, inner,div
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import element_of, elements_of
from sympde.topology import Square
from sympde.topology import Mapping, IdentityMapping, PolarMapping
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
V1     = VectorFunctionSpace('V1', domain, kind='H1')
V2     = ScalarFunctionSpace('V2', domain, kind='L2')

x,y    = domain.coordinates

F      = element_of(V1, name='F')

u,v    = [element_of(V1, name=i) for i in ['u', 'v']]
p,q    = [element_of(V2, name=i) for i in ['p', 'q']]

int_0  = lambda expr: integral(domain , expr)
f1     = cos(x)*sin(y)
f2     = sin(2*x)*sin(2*y)

b  = BilinearForm(((u,p),(v,q)), int_0(inner(grad(u),grad(v)) + div(u)*q - p*div(v)) )
l  = LinearForm((v,q), int_0(f1*v[0]+f2*v[1]+q))

print('============================================BilinearForm=========================================')
ast_b    = AST(b, [V1*V2,V1*V2], M)
stmt_b = parse(ast_b.expr, settings={'dim': ast_b.dim, 'nderiv': ast_b.nderiv, 'mapping':M})
print(pycode(stmt_b))
raise
print('============================================LinearForm===========================================')
ast_l    = AST(l, V1*V2, M)
stmt_l = parse(ast_l.expr, settings={'dim': ast_l.dim, 'nderiv': ast_l.nderiv, 'mapping':M})
print(pycode(stmt_l))
