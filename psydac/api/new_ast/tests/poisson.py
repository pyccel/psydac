# -*- coding: UTF-8 -*-


from sympy import symbols
from sympy import cos,sin

from sympde.calculus import grad, dot
from sympde.topology import (dx, dy, dz)
from sympde.topology import (dx1, dx2, dx3)
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of, elements_of
from sympde.topology import Square
from sympde.topology import Mapping, IdentityMapping
from sympde.expr     import integral
from sympde.expr     import LinearForm
from sympde.expr     import BilinearForm
from sympde.expr     import Norm

from psydac.api.new_ast.nodes  import AST
from psydac.api.new_ast.parser import parse

from pyccel.codegen.printing.pycode import pycode

# ...
domain = Square()
M      = IdentityMapping('M', domain.dim)
V      = ScalarFunctionSpace('V', domain)
u,v    = elements_of(V, names='u,v')

x,y    = symbols('x, y')

a     = BilinearForm((u,v), integral(domain, dot(grad(u), grad(v))))
b     = LinearForm(v, integral(domain, v*cos(x)))

error  = u - cos(x)*sin(x)
l2norm = Norm(error, domain, kind='l2')
h1norm = Norm(error, domain, kind='h1')

ast_b    = AST(a)
ast_l    = AST(b)
ast_norm = AST(h1norm) 

print('============================================BilinearForm=========================================')
print()
stmt_b = parse(ast_b.expr, settings={'dim': ast_b.dim, 'nderiv': ast_b.nderiv, 'mapping':M})
print(pycode(stmt_b))
print()
print('============================================LinearForm===========================================')
stmt_l = parse(ast_l.expr, settings={'dim': ast_l.dim, 'nderiv': ast_l.nderiv, 'mapping':M})
print()
print(pycode(stmt_l))
print('============================================Norm===========================================')
stmt_n = parse(ast_norm.expr, settings={'dim': ast_l.dim, 'nderiv': ast_l.nderiv, 'mapping':M})
print()
print(pycode(stmt_n))


def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()
