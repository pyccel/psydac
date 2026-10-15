#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np
from mpi4py import MPI

from sympde.calculus      import grad, dot
from sympde.topology      import ScalarFunctionSpace
from sympde.topology      import elements_of
from sympde.topology      import Square
from sympde.topology      import PolarMapping
from sympde.expr.expr     import LinearForm, BilinearForm
from sympde.expr.expr     import integral
from sympde.expr.expr     import Norm, SemiNorm
from sympde.expr.equation import find, EssentialBC

from psydac.api.discretization import discretize

#==============================================================================
def run_poisson_2d(solution, f, domain, ncells, degree, comm=None):

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

    l2norm =     Norm(error, domain, kind='l2')
    h1norm = SemiNorm(error, domain, kind='h1')

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    domain_h = discretize(domain, ncells=ncells)
    Vh       = discretize(V, domain_h, degree=degree)

    equation_h = discretize(equation, domain_h, [Vh, Vh])

    l2norm_h = discretize(l2norm, domain_h, Vh)
    h1norm_h = discretize(h1norm, domain_h, Vh)

    uh = equation_h.solve()

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

    l2_error, h1_error = run_poisson_2d(solution, f, mapped_domain, ncells=[2**2,2**2], degree=[2,2])

    expected_l2_error = 1.0930839536997034e-09
    expected_h1_error = 1.390398663745195e-08

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
