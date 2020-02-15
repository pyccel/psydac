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
V = VectorFunctionSpace('W', domain,kind='hdiv')

x,y = domain.coordinates

f = Tuple(2*pi**2*sin(pi*x)*sin(pi*y),
          2*pi**2*sin(pi*x)*sin(pi*y))

Fe = Tuple(sin(pi*x)*sin(pi*y), sin(pi*x)*sin(pi*y))

F = element_of(V, name='F')

u,v = [element_of(V, name=i) for i in ['u', 'v']]

int_0 = lambda expr: integral(domain , expr)

b = BilinearForm((v,u), int_0(div(v)*div(u)))
print('============================================BilinearForm=========================================')
ast_b    = AST(b, [V,V], M)
stmt_b = parse(ast_b.expr, settings={'dim': ast_b.dim, 'nderiv': ast_b.nderiv, 'mapping':M})
print(pycode(stmt_b))
