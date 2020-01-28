# -*- coding: UTF-8 -*-


from sympy import symbols
from sympy import cos

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

ast_b = AST(a)
ast_l = AST(b)

print('============================================BilinearForm=========================================')
print()
stmt_b = parse(ast_b.expr, settings={'dim': ast_b.dim, 'nderiv': ast_b.nderiv, 'mapping':M})
print(pycode(stmt_b))
print()
print('============================================LinearForm===========================================')
stmt_l = parse(ast_l.expr, settings={'dim': ast_l.dim, 'nderiv': ast_l.nderiv, 'mapping':M})
print()
print(pycode(stmt_l))


def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()
