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
def run_biharmonic_2d_dir(solution, f, dir_zero_boundary, ncells, degree, backend=None, comm=None):

    assert isinstance(dir_zero_boundary, (list, tuple))

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++
    domain = Square()

    B_dirichlet_0 = Union(*[domain.get_boundary(**kw) for kw in dir_zero_boundary])
    B_dirichlet_i = domain.boundary.complement(B_dirichlet_0)

    V  = ScalarFunctionSpace('V', domain)
    u  = element_of(V, name='u')
    v  = element_of(V, name='v')
    nn = NormalVector('nn')

    # Bilinear form a: V x V --> R
    a = BilinearForm((u, v), integral(domain, laplace(u) * laplace(v)))

    # Linear form l: V --> R
    l = LinearForm(v, integral(domain, f * v))

    # Essential boundary conditions
    dn = lambda a: dot(grad(a), nn)
    bc = []
    if B_dirichlet_0:
        bc += [EssentialBC(   u , 0, B_dirichlet_0)]
        bc += [EssentialBC(dn(u), 0, B_dirichlet_0)]
    if B_dirichlet_i:
        bc += [EssentialBC(   u ,    solution , B_dirichlet_i)]
        bc += [EssentialBC(dn(u), dn(solution), B_dirichlet_i)]

    # Variational model
    equation = find(u, forall=v, lhs=a(u, v), rhs=l(v), bc=bc)

    # Error norms
    error  = u - solution
    l2norm =     Norm(error, domain, kind='l2')
    h1norm = SemiNorm(error, domain, kind='h1')
    h2norm = SemiNorm(error, domain, kind='h2')

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
    h2norm_h = discretize(h2norm, domain_h, Vh, backend=backend)

    #+++++++++++++++++++++++++++++++
    # 3. Solution
    #+++++++++++++++++++++++++++++++

    # Solve linear system
    uh = equation_h.solve()

    # Compute error norms
    l2_error = l2norm_h.assemble(u=uh)
    h1_error = h1norm_h.assemble(u=uh)
    h2_error = h2norm_h.assemble(u=uh)

    return l2_error, h1_error, h2_error


###############################################################################
#            SERIAL TESTS
###############################################################################

def test_biharmonic_2d_dir0_1234():

    solution = sin(pi * x)**2 * sin(pi * y)**2
    f        = laplace(laplace(solution))

    dir_zero_boundary = get_boundaries(1, 2, 3, 4)

    l2_error, h1_error, h2_error = run_biharmonic_2d_dir(solution, f,
            dir_zero_boundary, ncells=[2**3, 2**3], degree=[3, 3])

    expected_l2_error = 0.00019981371108040476
    expected_h1_error = 0.0063205179028178295
    expected_h2_error = 0.2123929568623994

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    assert( abs(h2_error - expected_h2_error) < 1.e-7)

#------------------------------------------------------------------------------
@pytest.mark.xfail
def test_biharmonic_2d_dir0_123_diri_4():

    solution = sin(pi * x)**2 * sin(0.5*pi * y)**2
    f        = laplace(laplace(solution))

    dir_zero_boundary = get_boundaries(1, 2, 3)

    l2_error, h1_error, h2_error = run_biharmonic_2d_dir(solution, f,
            dir_zero_boundary, ncells=[2**3, 2**3], degree=[3, 3])

    print()
    print(l2_error)
    print(h1_error)
    print(h2_error)
    print()

    assert False

#------------------------------------------------------------------------------
@pytest.mark.xfail
def test_biharmonic_2d_dir0_13_diri_24():

    solution = sin(3*pi/2 * x)**2 * sin(3*pi/2 * y)**2
    f        = laplace(laplace(solution))

    dir_zero_boundary = get_boundaries(1, 3)

    l2_error, h1_error, h2_error = run_biharmonic_2d_dir(solution, f,
            dir_zero_boundary, ncells=[2**3, 2**3], degree=[3, 3])

    print()
    print(l2_error)
    print(h1_error)
    print(h2_error)
    print()

    assert False

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()
