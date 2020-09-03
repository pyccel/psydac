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

def run_poisson_2d(solution, f, domain, mapping, ncells, degree, comm=None):

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    V    = ScalarFunctionSpace('V', domain, kind=None)

    u, v = elements_of(V, names='u, v')

    bc   = EssentialBC(u, solution, domain.boundary)

    error  = u - solution

    a = BilinearForm((u,v),  integral(domain, dot(grad(u),grad(v))))
    l = LinearForm(v, integral(domain, f*v))

    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v), bc=bc)

    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++
    
    domain_h = discretize(domain, ncells=ncells)
    Vh       = discretize(V, domain_h, mapping=mapping, degree=degree)  
    
    equation_h = discretize(equation, domain_h, [Vh, Vh])

    l2norm_h = discretize(l2norm, domain_h, Vh)
    h1norm_h = discretize(h1norm, domain_h, Vh)

    x  = equation_h.solve()
    
    uh = FemField( Vh , x)

    l2_error = l2norm_h.assemble(u=uh)
    h1_error = h1norm_h.assemble(u=uh)

    return l2_error, h1_error

#------------------------------------------------------------------------------

def test_poisson_2d_analytical_mapping_0():

    domain  = Square('A',bounds1=(0., 1.), bounds2=(0, np.pi))
    mapping = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    mapped_domain = mapping(domain)

    x,y = mapped_domain.coordinates
    solution = x**2 + y**2
    f        = -4

    l2_error, h1_error = run_poisson_2d(solution, f, mapped_domain, mapping, ncells=[2**2,2**2], degree=[2,2])

    expected_l2_error = 1.0930839536997034e-09
    expected_h1_error = 1.390398663745195e-08


    assert ( abs(l2_error - expected_l2_error) < 1e-7 )
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )


#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()

