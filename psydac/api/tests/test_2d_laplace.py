#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from mpi4py import MPI
from sympy import pi, cos, sin, symbols
from sympy.utilities.lambdify import implemented_function
import pytest

from sympde.calculus import grad, dot
from sympde.calculus import laplace
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of
from sympde.topology import NormalVector
from sympde.topology import Square
from sympde.topology import Union
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm, SemiNorm
from sympde.expr     import find, EssentialBC

from psydac.api.discretization import discretize

x,y,z = symbols('x1, x2, x3', real=True)
#==============================================================================
def get_boundaries(*args):

    if not args:
        return ()
    else:
        assert all(1 <= a <= 4 for a in args)
        assert len(set(args)) == len(args)

    boundaries = {1: {'axis': 0, 'ext': -1},
                  2: {'axis': 0, 'ext':  1},
                  3: {'axis': 1, 'ext': -1},
                  4: {'axis': 1, 'ext':  1}}

    return tuple(boundaries[i] for i in args)

#==============================================================================
def run_laplace_2d(solution, f, dir_zero_boundary, dir_nonzero_boundary,
        ncells, degree, backend=None, comm=None):

    assert isinstance(dir_zero_boundary   , (list, tuple))
    assert isinstance(dir_nonzero_boundary, (list, tuple))

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++
    domain = Square()

    B_dirichlet_0 = Union(*[domain.get_boundary(**kw) for kw in dir_zero_boundary])
    B_dirichlet_i = Union(*[domain.get_boundary(**kw) for kw in dir_nonzero_boundary])
    B_dirichlet   = Union(B_dirichlet_0, B_dirichlet_i)
    B_neumann = domain.boundary.complement(B_dirichlet)

    V  = ScalarFunctionSpace('V', domain)
    u  = element_of(V, name='u')
    v  = element_of(V, name='v')
    nn = NormalVector('nn')

    # Bilinear form a: V x V --> R
    a = BilinearForm((u, v), integral(domain, dot(grad(u), grad(v)) + u * v))

    # Linear form l: V --> R
    l0 = LinearForm(v, integral(domain, f * v))
    if B_neumann:
        l1 = LinearForm(v, integral(B_neumann, v * dot(grad(solution), nn)))
        l  = LinearForm(v, l0(v) + l1(v))
    else:
        l = l0

    # Dirichlet boundary conditions
    bc = []
    if B_dirichlet_0:  bc += [EssentialBC(u,        0, B_dirichlet_0)]
    if B_dirichlet_i:  bc += [EssentialBC(u, solution, B_dirichlet_i)]

    # Variational model
    equation = find(u, forall=v, lhs=a(u, v), rhs=l(v), bc=bc)

    # Error norms
    error  = u - solution
    l2norm =     Norm(error, domain, kind='l2')
    h1norm = SemiNorm(error, domain, kind='h1')

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    # Create computational domain from topological domain
    domain_h = discretize(domain, ncells=ncells, comm=comm)

    # Discrete spaces
    Vh = discretize(V, domain_h, degree=degree)

    # Discretize equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh], backend=backend)

    # Discretize error norms
    l2norm_h = discretize(l2norm, domain_h, Vh, backend=backend)
    h1norm_h = discretize(h1norm, domain_h, Vh, backend=backend)

    #+++++++++++++++++++++++++++++++
    # 3. Solution
    #+++++++++++++++++++++++++++++++

    # Solve linear system
    uh = equation_h.solve()

    # Compute error norms
    l2_error = l2norm_h.assemble(u=uh)
    h1_error = h1norm_h.assemble(u=uh)

    return l2_error, h1_error


###############################################################################
#            SERIAL TESTS
###############################################################################

def test_laplace_2d_neu0_1234():

    solution = cos(pi*x)*cos(pi*y)
    f        = (2.*pi**2 + 1.)*solution

    dir_zero_boundary    = get_boundaries()
    dir_nonzero_boundary = get_boundaries()

    l2_error, h1_error = run_laplace_2d(solution, f, dir_zero_boundary,
            dir_nonzero_boundary, ncells=[2**3, 2**3], degree=[2, 2])

    expected_l2_error =  0.0002172846538950129
    expected_h1_error =  0.012984852988125026

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()
