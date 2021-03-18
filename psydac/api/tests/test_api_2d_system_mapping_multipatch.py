import pytest
import numpy as np
from mpi4py import MPI
from sympy  import pi, sin, cos, Tuple, Matrix

from sympde.calculus      import grad, dot, curl, cross
from sympde.calculus      import minus, plus
from sympde.topology      import VectorFunctionSpace
from sympde.topology      import elements_of
from sympde.topology      import NormalVector
from sympde.topology      import Square
from sympde.topology      import IdentityMapping, PolarMapping
from sympde.expr.expr     import LinearForm, BilinearForm
from sympde.expr.expr     import integral
from sympde.expr.expr     import Norm
from sympde.expr.equation import find, EssentialBC

from psydac.api.discretization       import discretize
from psydac.fem.basic                import FemField
from psydac.linalg.iterative_solvers import pcg

#==============================================================================

def run_maxwell_2d(uex, f, alpha, domain, ncells, degree, comm=None, return_sol=False):

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    V  = VectorFunctionSpace('V', domain, kind='hcurl')

    u, v, F  = elements_of(V, names='u, v, F')
    nn  = NormalVector('nn')

    error   = Matrix([F[0]-uex[0],F[1]-uex[1]])

    kappa        = 10**8*degree[0]**2*ncells[0]
    penalization = 10**7

    I        = domain.interfaces
    boundary = domain.boundary

    # Bilinear form a: V x V --> R
    eps = -1

    expr_I  = -(0.5*curl(plus(u))*cross(minus(v),nn)       - 0.5*curl(minus(u))*cross(plus(v),nn))
    expr_I += eps*(0.5*curl(plus(v))*cross(minus(u),nn)    - 0.5*curl(minus(v))*cross(plus(u),nn))
    expr_I += -kappa*cross(plus(u),nn) *cross(minus(v),nn) - kappa*cross(plus(v),nn) * cross(minus(u),nn)
    expr_I += -(0.5*curl(minus(u))*cross(minus(v),nn)     + 0.5*curl(plus(u))*cross(plus(v),nn))
    expr_I += eps*(0.5*curl(minus(v))*cross(minus(u),nn)  + 0.5*curl(plus(v))*cross(plus(u),nn))
    expr_I += kappa*cross(minus(u),nn)*cross(minus(v),nn) + kappa*cross(plus(u),nn) *cross(plus(v),nn)

    expr   = curl(u)*curl(v) + alpha*dot(u,v)
    expr_b = penalization * cross(u, nn) * cross(v, nn)

    a = BilinearForm((u,v),  integral(domain, expr) + integral(I, expr_I) + integral(boundary, expr_b))

    expr   = dot(f,v)
    expr_b = penalization * cross(uex, nn) * cross(v, nn)

    l = LinearForm(v, integral(domain, expr) + integral(boundary, expr_b))

    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v))

    l2norm = Norm(error, domain, kind='l2')
    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    domain_h = discretize(domain, ncells=ncells, comm=comm)
    Vh       = discretize(V, domain_h, degree=degree)

    equation_h = discretize(equation, domain_h, [Vh, Vh])
    l2norm_h   = discretize(l2norm, domain_h, Vh)

    equation_h.assemble()
    
    A = equation_h.linear_system.lhs
    b = equation_h.linear_system.rhs
    
    x, info = pcg(A, b, pc='jacobi', tol=1e-8)

    uh = FemField( Vh, x )

    l2_error = l2norm_h.assemble(F=uh)

    return l2_error

#------------------------------------------------------------------------------

def test_maxwell_2d_2_patch_dirichlet_0():

    bounds1   = (0.5, 1.)
    bounds2_A = (0, np.pi/2)
    bounds2_B = (np.pi/2, np.pi)

    A = Square('A',bounds1=bounds1, bounds2=bounds2_A)
    B = Square('B',bounds1=bounds1, bounds2=bounds2_B)

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    D1     = mapping_1(A)
    D2     = mapping_2(B)

    domain = D1.join(D2, name = 'domain',
                bnd_minus = D1.get_boundary(axis=1, ext=1),
                bnd_plus  = D2.get_boundary(axis=1, ext=-1))

    x,y    = domain.coordinates
    alpha    = 1.
    uex      = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
    f        = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y), 
                     alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))

    l2_error = run_maxwell_2d(uex, f, alpha, domain, ncells=[2**3, 2**3], degree=[2,2])
    expected_l2_error = 0.006038532417958093

    assert ( abs(l2_error - expected_l2_error) < 1e-7)


#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()


