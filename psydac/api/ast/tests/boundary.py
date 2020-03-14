# -*- coding: UTF-8 -*-


from sympy import symbols
from sympy import cos,sin,pi,Tuple

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
from sympde.topology import Boundary, NormalVector

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
    filename = os.path.join(mesh_dir, 'identity_2d.h5')

kappa = 10**15
domain  = Square()
mapping = IdentityMapping('M',2)
nn      = NormalVector('nn')
V       = ScalarFunctionSpace('V', domain)
u,v     = elements_of(V, names='u,v')

x,y     = symbols('x, y')
B       = domain.get_boundary(axis=0,ext=1)
int_1   = lambda expr: integral(B, expr)
int_0   = lambda expr: integral(domain, expr)
g       = Tuple(x**2, y**2)

b       = BilinearForm((u,v), int_1(v*dot(grad(u), nn)) + int_0(u*v))
l       = LinearForm(v, int_1(-x*y*(1-y)*dot(grad(v),nn) + kappa*x*y*(1-y)*v))

# Create computational domain from topological domain
domain_h = discretize(domain, ncells=[2**2,2**2])

# Discrete spaces
Vh = discretize(V, domain_h, degree=[1,1], mapping=mapping)

# Discretize forms
l_h = discretize(l, domain_h,  Vh)
b_h = discretize(b, domain_h, [Vh, Vh])
d   = l_h.assemble()
M   = b_h.assemble()
print(d.toarray())
