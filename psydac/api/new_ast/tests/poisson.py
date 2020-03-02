# -*- coding: UTF-8 -*-


from sympy import symbols
from sympy import cos,sin

from sympde.calculus import grad, dot
from sympde.topology import (dx, dy, dz)
from sympde.topology import (dx1, dx2, dx3)
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of, elements_of
from sympde.topology import Square
from sympde.topology import Mapping, IdentityMapping, PolarMapping
from sympde.expr     import integral
from sympde.expr     import LinearForm
from sympde.expr     import BilinearForm
from sympde.expr     import Norm

from psydac.api.new_ast.fem  import AST
from psydac.api.new_ast.parser import parse
from psydac.api.discretization import discretize

from pyccel.codegen.printing.pycode import pycode

# ...
domain = Square()
M      = Mapping('M', domain.dim)
V      = ScalarFunctionSpace('V', domain)
u,v    = elements_of(V, names='u,v')

x,y    = symbols('x, y')

b     = BilinearForm((u,v), integral(domain, dot(grad(u), grad(v))))
l     = LinearForm(v, integral(domain, v*cos(x)))

# Create computational domain from topological domain
domain_h = discretize(domain, ncells=[3,3])

# Discrete spaces
Vh = discretize(V, domain_h, degree=[2,2])


error  = u - cos(x)*sin(x)
l2norm = Norm(error, domain, kind='l2')
h1norm = Norm(error, domain, kind='h1')

# Discretize forms

b_h      = discretize(b, domain_h, [Vh, Vh])
l_h      = discretize(l, domain_h, Vh)

l2norm_h = discretize(l2norm, domain_h, Vh)
h1norm_h = discretize(h1norm, domain_h, Vh)
