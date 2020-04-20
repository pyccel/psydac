# -*- coding: UTF-8 -*-


from sympy import symbols
from sympy import sin,pi

from sympde.calculus import grad, dot

from sympde.topology import ScalarFunctionSpace
from sympde.topology import elements_of
from sympde.topology import Square
from sympde.topology import IdentityMapping#,Mapping PolarMapping
from sympde.expr     import integral
from sympde.expr     import LinearForm
from sympde.expr     import BilinearForm
from sympde.expr     import Norm

from psydac.api.discretization import discretize
from sympde.expr import find, EssentialBC
from psydac.fem.basic          import FemField
import os
from psydac.api.essential_bc         import apply_essential_bc
# ...
try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']

except KeyError:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..','..')
    mesh_dir = os.path.join(base_dir, 'mesh')
    filename = os.path.join(mesh_dir, 'identity_2d.h5')

domain  = Square()
mapping = IdentityMapping('M',2, c1= 1., c2= 3., rmin = 1., rmax=2.)
V       = ScalarFunctionSpace('V', domain)
u,v     = elements_of(V, names='u,v')

x,y      = symbols('x, y')
bc       = [EssentialBC(u, 0, domain.boundary)]
b        = BilinearForm((u,v), integral(domain, dot(grad(u), grad(v))))
l        = LinearForm(v, integral(domain, v*2*pi**2*sin(pi*x)*sin(pi*y)))
equation = find(u, forall=v, lhs=b(u, v), rhs=l(v), bc=bc)
# Create computational domain from topological domain
domain_h = discretize(domain, ncells=[2**1,2**1])

# Discrete spaces
Vh = discretize(V, domain_h, degree=[1,1], mapping=mapping)

error  = u - sin(pi*x)*sin(pi*y)
l2norm = Norm(error, domain, kind='l2')
h1norm = Norm(error, domain, kind='h1')

# Discretize forms

equation_h = discretize(equation, domain_h, [Vh, Vh])
equation_h.lhs._set_func('dependencies_mq3duq68','assembly')

#x  = equation_h.solve()
M = equation_h.rhs.assemble()
for i in equation_h.bc:
    apply_essential_bc(equation_h.test_space, i, M)

uh = FemField(Vh, x)

l2_error = l2norm_h.assemble(u=uh)


h1_error = h1norm_h.assemble(u=uh)
print(l2_error, h1_error)

