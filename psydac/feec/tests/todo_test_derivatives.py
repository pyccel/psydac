import pytest
import numpy as np
from sympy import pi, cos, sin
from sympy.abc import x, y
from sympy.utilities.lambdify import implemented_function
from mpi4py import MPI

from sympde.calculus import grad, dot
from sympde.calculus import laplace
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of
from sympde.topology import NormalVector
from sympde.topology import Square
from sympde.topology import Union
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm
from sympde.expr     import find, EssentialBC

from psydac.linalg.stencil     import StencilVector
from psydac.linalg.block       import BlockVector,ProductSpace
from psydac.fem.basic          import FemField
from psydac.fem.splines        import SplineSpace
from psydac.fem.tensor         import TensorFemSpace
from psydac.fem.vector         import ProductFemSpace
from psydac.feec.utilities     import Interpolation, interpolation_matrices
from psydac.feec.derivatives   import Grad, Curl, Div
from psydac.api.discretization import discretize

#===========================================================================================
#+++++++++++++++++++++++++++++++
# 1. Abstract model
#+++++++++++++++++++++++++++++++
domain = Square()
x,y    = domain.coordinates
f      = 2*x

V  = ScalarFunctionSpace('V', domain)
u  = element_of(V, name='u')
v  = element_of(V, name='v')

# Bilinear form a: V x V --> R
a = BilinearForm((u, v), integral(domain, u*v))

# Linear form l: V --> R
l = LinearForm(v, integral(domain, f * v))
# Variational model
equation = find(u, forall=v, lhs=a(u, v), rhs=l(v))


#+++++++++++++++++++++++++++++++
# 2. Discretization
#+++++++++++++++++++++++++++++++

# Create computational domain from topological domain
domain_h = discretize(domain, ncells=[1,1])

# Discrete spaces
Vh = discretize(V, domain_h, degree=[2,2])

# Discretize equation using Dirichlet bc
equation_h = discretize(equation, domain_h, [Vh, Vh])

#+++++++++++++++++++++++++++++++
# 3. Solution
#+++++++++++++++++++++++++++++++

# Solve linear system
x  = equation_h.solve()
x  = FemField( Vh, x )
#======================================================================================

H1    = Vh
Hcurl = ProductFemSpace(H1.reduce_degree([0]), H1.reduce_degree([1]))

grad = Grad(H1, Hcurl)
G    = grad._matrix.tosparse()

print(grad(x.coeffs).toarray())

#print(G.dot(x.coeffs.toarray()))

#raise SystemExit()
## ... H1
#f = lambda x,y: x*(1.-x)*y*(1.-y)
#F = Int('H1', f)

## ... Hcurl
#g0 = lambda x,y: (1.-2.*x)*y*(1.-y)
#g1 = lambda x,y: x*(1.-x)*(1.-2.*y)
#G = Int('Hcurl', [g0, g1])

#F =  M0.solve(F)
#G =  M1.solve(G)

#grad = Grad(H1, Hcurl.vector_space)
#curl = Curl(H1, Hcurl.vector_space, L2.vector_space)

#x    = BlockVector(ProductSpace(H1.vector_space))
#y    = BlockVector(Hcurl.vector_space)

#x[0]._data = F._data
#x = grad(x).toarray()
#y = np.concatenate(tuple(g.toarray() for g in G))
#print(np.abs(x-y).max())

