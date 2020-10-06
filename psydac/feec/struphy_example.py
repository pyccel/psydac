
from sympy import pi, cos, sin, symbols
from sympy.utilities.lambdify import implemented_function
import pytest
import numpy as np
from sympde.calculus import grad, dot
from sympde.calculus import laplace
from sympde.topology import ScalarFunctionSpace
from sympde.topology import elements_of
from sympde.topology import NormalVector
from sympde.topology import Cube, Derham, Line
from sympde.topology import Union
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm
from sympde.expr     import find, EssentialBC

from psydac.fem.basic          import FemField
from psydac.api.discretization import discretize

from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL, PSYDAC_BACKEND_NUMBA

np.set_printoptions(precision=3, linewidth=180)
# Continuous world: SymPDE

domain  = Line('C')
#domain = Mapping(domain)
V  = ScalarFunctionSpace('M', domain)

u0, v0 = elements_of(V, names='u0, v0')


a0 = BilinearForm((u0, v0), integral(domain, u0*v0))

#==============================================================================
# Discrete objects: Psydac
domain_h = discretize(domain, ncells=(2,))
Vh = discretize(V, domain_h, degree=(2, ), periodic=(True,))

a0_h = discretize(a0, domain_h, (Vh, Vh), backend=PSYDAC_BACKEND_NUMBA)

# StencilMatrix objects
M0 = a0_h.assemble()
print(M0[0,0])
print(M0._data[2:-2])
print(M0.toarray())
