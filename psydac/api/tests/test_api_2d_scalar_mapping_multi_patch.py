import pytest

from sympy.core.containers import Tuple
from sympy import Matrix
from sympy import Function
from sympy import pi, cos, sin, exp

from sympde.core import Constant
from sympde.calculus import grad, dot, inner, rot, div
from sympde.calculus import laplace, bracket, convect
from sympde.calculus import jump, avg, Dn, minus, plus
#from sympde.topology import dx
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of, elements_of
from sympde.topology import InteriorDomain, Union
from sympde.topology import Boundary, NormalVector
from sympde.topology import Domain
from sympde.topology import trace_1
from sympde.topology import Square
from sympde.topology import ElementDomain
from sympde.topology import Area
from sympde.topology import IdentityMapping, PolarMapping

from sympde.expr.expr import LinearExpr
from sympde.expr.expr import LinearForm, BilinearForm
from sympde.expr.expr import integral
from sympde.expr.expr import Functional, Norm
from sympde.expr.expr import linearize
from sympde.expr.evaluation import TerminalExpr
from psydac.api.discretization import discretize
from sympde.expr     import find, EssentialBC
from psydac.fem.vector                  import VectorFemField
from psydac.fem.basic                   import FemField
import numpy as np

from mpi4py import MPI

#==============================================================================

def run_poisson_2d(solution, f, domain, ncells, degree, comm=None):

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    V   = ScalarFunctionSpace('V', domain, kind=None)

    u, v = elements_of(V, names='u, v')
    nn   = NormalVector('nn')

    bc   = EssentialBC(u, solution, domain.boundary)

    error  = u - solution

    I = domain.interfaces

    kappa  = 10**3

   # expr_I =(
   #         - dot(grad(plus(u)),nn)*minus(v)  + dot(grad(minus(v)),nn)*plus(u) - kappa*plus(u)*minus(v)
   #         + dot(grad(minus(u)),nn)*plus(v)  - dot(grad(plus(v)),nn)*minus(u) - kappa*plus(v)*minus(u)
   #         - dot(grad(plus(v)),nn)*plus(u)   + kappa*plus(u)*plus(v)
   #         - dot(grad(minus(v)),nn)*minus(u) + kappa*minus(u)*minus(v))

    expr_I =- 0.5*dot(grad(plus(u)),nn)*minus(v)  + 0.5*dot(grad(minus(v)),nn)*plus(u)  - kappa*plus(u)*minus(v)\
            + 0.5*dot(grad(minus(u)),nn)*plus(v)  - 0.5*dot(grad(plus(v)),nn)*minus(u)  - kappa*plus(v)*minus(u)\
            - 0.5*dot(grad(minus(v)),nn)*minus(u) - 0.5*dot(grad(minus(u)),nn)*minus(v) + kappa*minus(u)*minus(v)\
            - 0.5*dot(grad(plus(v)),nn)*plus(u)   - 0.5*dot(grad(plus(u)),nn)*plus(v)   + kappa*plus(u)*plus(v)

    expr   = dot(grad(u),grad(v))

    a = BilinearForm((u,v),  integral(domain, expr) + integral(I, expr_I))
    l = LinearForm(v, integral(domain, f*v))

    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v), bc=bc)

    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    domain_h = discretize(domain, ncells=ncells, comm=comm)
    Vh       = discretize(V, domain_h, degree=degree)

    equation_h = discretize(equation, domain_h, [Vh, Vh])
    l2norm_h = discretize(l2norm, domain_h, Vh)
    h1norm_h = discretize(h1norm, domain_h, Vh)

    x  = equation_h.solve()

    uh = VectorFemField( Vh )

    for i in range(len(uh.coeffs[:])):
        uh.coeffs[i][:,:] = x[i][:,:]

    l2_error = l2norm_h.assemble(u=uh)
    h1_error = h1norm_h.assemble(u=uh)

    return l2_error, h1_error

#------------------------------------------------------------------------------

def test_poisson_2d_2_patch_dirichlet_0():

    A = Square('A',bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi))

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    D1     = mapping_1(A)
    D2     = mapping_2(B)

    domain = D1.join(D2, name = 'domain',
                bnd_minus = D1.get_boundary(axis=1, ext=1),
                bnd_plus  = D2.get_boundary(axis=1, ext=-1))

    x,y = domain.coordinates
    solution = x**2 + y**2
    f        = -4

    l2_error, h1_error = run_poisson_2d(solution, f, domain, ncells=[2**2, 2**2], degree=[2,2])

    expected_l2_error = 6.223948817460227e-09
    expected_h1_error = 8.184613465986152e-09


    assert ( abs(l2_error - expected_l2_error) < 1e-7)
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

#------------------------------------------------------------------------------

def test_poisson_2d_2_patch_dirichlet_1():

    A = Square('A',bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi))

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    D1     = mapping_1(A)
    D2     = mapping_2(B)

    domain = D1.join(D2, name = 'domain',
                bnd_minus = D1.get_boundary(axis=1, ext=1),
                bnd_plus  = D2.get_boundary(axis=1, ext=-1))

    x,y = domain.coordinates
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    l2_error, h1_error = run_poisson_2d(solution, f, domain, ncells=[2**2,2**2], degree=[2,2])

    expected_l2_error = 0.008813239980488852
    expected_h1_error = 0.12252946949143205

    assert ( abs(l2_error - expected_l2_error) < 1e-7 )
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

def test_poisson_2d_2_patch_dirichlet_2():

    A = Square('A',bounds1=(0.5, 1.), bounds2=(-1., 0.))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
#    C = Square('C',bounds1=(-1., 0.), bounds2=(0.5, 1.))

    mapping_1 = IdentityMapping('M1', 2)
    mapping_2 = PolarMapping   ('M2', 2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    #mapping_3 = IdentityMapping('M3', 2)

    D1     = mapping_1(A)
    D2     = mapping_2(B)
    #D3     = mapping_3(C)

    D1D2        = D1.join(D2, name = 'AB',
                bnd_minus = D1.get_boundary(axis=1, ext=1),
                bnd_plus  = D2.get_boundary(axis=1, ext=-1))

    #D1D2D3       = D1D2.join(D3, name = 'ABC',
    #            bnd_minus = D2.get_boundary(axis=1, ext=1),
    #            bnd_plus  = D3.get_boundary(axis=0, ext=1))

    x,y       = D1D2.coordinates
    solution  = x**2 + y**2
    f         = -4

    l2_error, h1_error = run_poisson_2d(solution, f, D1D2, ncells=[2**2,2**2], degree=[2,2])

    expected_l2_error = 3.044249692897913e-09
    expected_h1_error = 1.3784679832274999e-08

    assert ( abs(l2_error - expected_l2_error) < 1e-7)
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

###############################################################################
#            PARALLEL TESTS
###############################################################################

#==============================================================================
@pytest.mark.parallel
def test_poisson_2d_2_patch_dirichlet_parallel_0():

    A = Square('A',bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi))

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    D1     = mapping_1(A)
    D2     = mapping_2(B)

    domain = D1.join(D2, name = 'domain',
                bnd_minus = D1.get_boundary(axis=1, ext=1),
                bnd_plus  = D2.get_boundary(axis=1, ext=-1))

    x,y = domain.coordinates
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    l2_error, h1_error = run_poisson_2d(solution, f, domain, ncells=[2**2,2**2], degree=[2,2],
                                        comm=MPI.COMM_WORLD)

    expected_l2_error = 0.008813239980488852
    expected_h1_error = 0.12252946949143205

    assert ( abs(l2_error - expected_l2_error) < 1e-7 )
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()

