from mpi4py import MPI
from sympy import pi, cos, sin, symbols
from sympy.utilities.lambdify import implemented_function
import pytest

from sympde.topology import Mapping
from sympde.calculus import grad, dot
from sympde.calculus import laplace
from sympde.topology import ScalarFunctionSpace
from sympde.topology import elements_of
from sympde.topology import NormalVector
from sympde.topology import Cube, Derham
from sympde.topology import Union
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm
from sympde.expr     import find, EssentialBC

from psydac.fem.basic          import FemField
from psydac.api.discretization import discretize

from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL, PSYDAC_BACKEND_NUMBA

# Continuous world: SymPDE

class CollelaMapping3D(Mapping):
    """
    Represents a Collela 3D Mapping object.

    Examples

    """
    _expressions = {'x': 'k1*(x1 + eps*sin(2.*pi*x1)*sin(2.*pi*x2))',
                    'y': 'k2*(x2 + eps*sin(2.*pi*x1)*sin(2.*pi*x2))',
                    'z': 'k3*x3'}

    _ldim        = 3
    _pdim        = 3

M       = CollelaMapping3D('M', k1=1, k2=1, k3=1, eps=0.1)

domain  = Cube('C', bounds1=(0, 1), bounds2=(0, 1), bounds3=(0, 1))
domain  = M(domain)
derham  = Derham(domain)

u0, v0 = elements_of(derham.V0, names='u0, v0')
u1, v1 = elements_of(derham.V1, names='u1, v1')
u2, v2 = elements_of(derham.V2, names='u2, v2')
u3, v3 = elements_of(derham.V3, names='u3, v3')

a0 = BilinearForm((u0, v0), integral(domain, u0*v0))
a1 = BilinearForm((u1, v1), integral(domain, dot(u1, v1)))
a2 = BilinearForm((u2, v2), integral(domain, dot(u2, v2)))
a3 = BilinearForm((u3, v3), integral(domain, u3*v3))

#==============================================================================
# Discrete objects: Psydac
domain_h = discretize(domain, ncells=(6, 6, 6))
derham_h = discretize(derham, domain_h, degree=(2, 2, 2), periodic=(True, True, True))

a0_h = discretize(a0, domain_h, (derham_h.V0, derham_h.V0), backend=PSYDAC_BACKEND_NUMBA)
a1_h = discretize(a1, domain_h, (derham_h.V1, derham_h.V1))
a2_h = discretize(a2, domain_h, (derham_h.V2, derham_h.V2), backend=PSYDAC_BACKEND_NUMBA)
a3_h = discretize(a3, domain_h, (derham_h.V3, derham_h.V3), backend=PSYDAC_BACKEND_NUMBA)

# StencilMatrix objects
M0 = a0_h.assemble().tosparse()
M1 = a1_h.assemble().tosparse()
M2 = a2_h.assemble().tosparse()
M3 = a3_h.assemble().tosparse()

