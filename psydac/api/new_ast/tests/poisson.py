# -*- coding: UTF-8 -*-


from sympy import symbols
from sympy import cos,sin,pi

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
from sympde.expr import find, EssentialBC
from pyccel.codegen.printing.pycode import pycode
from sympde.topology import Domain
from psydac.fem.basic          import FemField
import os
# ...
try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']

except:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..','..')
    mesh_dir = os.path.join(base_dir, 'mesh')
    filename = os.path.join(mesh_dir, 'identity_2d_2.h5')

domain = Domain.from_file(filename)
V      = ScalarFunctionSpace('V', domain)
u,v    = elements_of(V, names='u,v')

x,y    = symbols('x, y')

bc    = [EssentialBC(u, 0, domain.boundary)]
b     = BilinearForm((u,v), integral(domain, dot(grad(u), grad(v))))
l     = LinearForm(v, integral(domain, v*2*pi**2*sin(pi*x)*sin(pi*y)))
equation = find(u, forall=v, lhs=b(u, v), rhs=l(v), bc=bc)
# Create computational domain from topological domain
domain_h = discretize(domain, filename=filename)

# Discrete spaces
Vh = discretize(V, domain_h)

error  = u - sin(pi*x)*sin(pi*y)
l2norm = Norm(error, domain, kind='l2')
h1norm = Norm(error, domain, kind='h1')

# Discretize forms

equation_h = discretize(equation, domain_h, [Vh, Vh])
x  = equation_h.solve()
uh = FemField(Vh, x)
l2norm_h = discretize(l2norm, domain_h, Vh)
l2_error = l2norm_h.assemble(u=uh)

h1norm_h = discretize(h1norm, domain_h, Vh)
h1_error = h1norm_h.assemble(u=uh)


