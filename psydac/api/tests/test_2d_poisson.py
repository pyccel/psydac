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
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of
from sympde.topology import NormalVector
from sympde.topology import Square
from sympde.topology import Union
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm, SemiNorm
from sympde.expr     import find, EssentialBC

from psydac.api.discretization import discretize
from psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL

x,y,z = symbols('x1, x2, x3', real=True)

# backend to activate multi threading
PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP           = PSYDAC_BACKEND_GPYCCEL.copy()
PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP['openmp'] = True
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
def run_poisson_2d(solution, f, dir_zero_boundary, dir_nonzero_boundary,
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
    a = BilinearForm((u, v), integral(domain, dot(grad(u), grad(v))))

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
@pytest.mark.parametrize('backend',  [None, PSYDAC_BACKEND_GPYCCEL, PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP])
def test_poisson_2d_dir0_1234(backend):

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)

    dir_zero_boundary    = get_boundaries(1, 2, 3, 4)
    dir_nonzero_boundary = get_boundaries()

    l2_error, h1_error = run_poisson_2d(solution, f, dir_zero_boundary,
            dir_nonzero_boundary, ncells=[2**3, 2**3], degree=[2, 2],
            comm=MPI.COMM_WORLD, backend=backend)

    expected_l2_error =  0.00021808678604760232
    expected_h1_error =  0.013023570720360362

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_dir0_234_neu0_1():

    solution = cos(0.5*pi*x)*sin(pi*y)
    f        = (5./4.)*pi**2*solution

    dir_zero_boundary    = get_boundaries(2, 3, 4)
    dir_nonzero_boundary = get_boundaries()

    l2_error, h1_error = run_poisson_2d(solution, f, dir_zero_boundary,
            dir_nonzero_boundary, ncells=[2**3, 2**3], degree=[2, 2])

    expected_l2_error =  0.00015546057796452772
    expected_h1_error =  0.00926930278452745

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_dir0_134_neu0_2():

    solution = sin(0.5*pi*x)*sin(pi*y)
    f        = (5./4.)*pi**2*solution

    dir_zero_boundary    = get_boundaries(1, 3, 4)
    dir_nonzero_boundary = get_boundaries()

    l2_error, h1_error = run_poisson_2d(solution, f, dir_zero_boundary,
            dir_nonzero_boundary, ncells=[2**3, 2**3], degree=[2, 2])

    expected_l2_error =  0.0001554605779481901
    expected_h1_error =  0.009269302784527256

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_dir0_124_neu0_3():

    solution = sin(pi*x)*cos(0.5*pi*y)
    f        = (5./4.)*pi**2*solution

    dir_zero_boundary    = get_boundaries(1, 2, 4)
    dir_nonzero_boundary = get_boundaries()

    l2_error, h1_error = run_poisson_2d(solution, f, dir_zero_boundary,
            dir_nonzero_boundary, ncells=[2**3, 2**3], degree=[2, 2])

    expected_l2_error =  0.0001554605779681901
    expected_h1_error =  0.009269302784528678

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_dir0_123_neu0_4():

    solution = sin(pi*x)*sin(0.5*pi*y)
    f        = (5./4.)*pi**2*solution

    dir_zero_boundary    = get_boundaries(1, 2, 3)
    dir_nonzero_boundary = get_boundaries()

    l2_error, h1_error = run_poisson_2d(solution, f, dir_zero_boundary,
            dir_nonzero_boundary, ncells=[2**3, 2**3], degree=[2, 2])

    expected_l2_error =  0.00015546057796339546
    expected_h1_error =  0.009269302784526841

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_dir0_24_neu0_13():

    solution = cos(0.5*pi*x)*cos(0.5*pi*y)
    f        = (1./2.)*pi**2*solution

    dir_zero_boundary    = get_boundaries(2, 4)
    dir_nonzero_boundary = get_boundaries()

    l2_error, h1_error = run_poisson_2d(solution, f, dir_zero_boundary,
            dir_nonzero_boundary, ncells=[2**3, 2**3], degree=[2, 2])

    expected_l2_error =  2.6119892736036942e-05
    expected_h1_error =  0.0016032430287934746

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_dir0_13_neu0_24():

    solution = sin(0.5*pi*x)*sin(0.5*pi*y)
    f        = (1./2.)*pi**2*solution

    dir_zero_boundary    = get_boundaries(1, 3)
    dir_nonzero_boundary = get_boundaries()

    l2_error, h1_error = run_poisson_2d(solution, f, dir_zero_boundary,
            dir_nonzero_boundary, ncells=[2**3, 2**3], degree=[2, 2])

    expected_l2_error =  2.611989253883369e-05
    expected_h1_error =  0.0016032430287973409

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_dir0_4_neu0_123():

    solution = cos(pi*x)*cos(0.5*pi*y)
    f        = 5./4.*pi**2*solution

    dir_zero_boundary    = get_boundaries(4)
    dir_nonzero_boundary = get_boundaries()

    l2_error, h1_error = run_poisson_2d(solution, f, dir_zero_boundary,
            dir_nonzero_boundary, ncells=[2**3, 2**3], degree=[2, 2])

    expected_l2_error =  0.00015494478505412876
    expected_h1_error =  0.009242166414700994

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_dir0_234_neui_1():

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    dir_zero_boundary    = get_boundaries(2, 3, 4)
    dir_nonzero_boundary = get_boundaries()

    l2_error, h1_error = run_poisson_2d(solution, f, dir_zero_boundary,
            dir_nonzero_boundary, ncells=[2**3, 2**3], degree=[2, 2])

    expected_l2_error = 0.00021786960672322118
    expected_h1_error = 0.01302350067761091

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_dir0_134_neui_2():

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    dir_zero_boundary    = get_boundaries(1, 3, 4)
    dir_nonzero_boundary = get_boundaries()

    l2_error, h1_error = run_poisson_2d(solution, f, dir_zero_boundary,
            dir_nonzero_boundary, ncells=[2**3, 2**3], degree=[2, 2])

    expected_l2_error = 0.00021786960672322118
    expected_h1_error = 0.01302350067761091

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_dir0_124_neui_3():

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    dir_zero_boundary    = get_boundaries(1, 2, 4)
    dir_nonzero_boundary = get_boundaries()

    l2_error, h1_error = run_poisson_2d(solution, f, dir_zero_boundary,
            dir_nonzero_boundary, ncells=[2**3, 2**3], degree=[2, 2])

    expected_l2_error = 0.00021786960672322118
    expected_h1_error = 0.01302350067761091

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_dir0_123_neui_4():

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    dir_zero_boundary    = get_boundaries(1, 2, 3)
    dir_nonzero_boundary = get_boundaries()

    l2_error, h1_error = run_poisson_2d(solution, f, dir_zero_boundary,
            dir_nonzero_boundary, ncells=[2**3, 2**3], degree=[2, 2])

    expected_l2_error = 0.00021786960672322118
    expected_h1_error = 0.01302350067761091

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_dir0_123_diri_4():

    solution = sin(pi * x) * sin(0.5*pi * y)
    f        = 5/4*pi**2 * solution

    dir_zero_boundary    = get_boundaries(1, 2, 3)
    dir_nonzero_boundary = get_boundaries(4)

    l2_error, h1_error = run_poisson_2d(solution, f, dir_zero_boundary,
            dir_nonzero_boundary, ncells=[2**3, 2**3], degree=[2, 2])

    expected_l2_error = 0.00015292215711784052
    expected_h1_error = 0.009293161646614652

    assert abs(l2_error - expected_l2_error) < 1.e-7
    assert abs(h1_error - expected_h1_error) < 1.e-7

#------------------------------------------------------------------------------
def test_poisson_2d_dir0_13_diri_24():

    solution = sin(3*pi/2 * x) * sin(3*pi/2 * y)
    f        = 9/2*pi**2 * solution

    dir_zero_boundary    = get_boundaries(1, 3)
    dir_nonzero_boundary = get_boundaries(2, 4)

    l2_error, h1_error = run_poisson_2d(solution, f, dir_zero_boundary,
            dir_nonzero_boundary, ncells=[2**3, 2**3], degree=[2, 2])

    expected_l2_error = 0.0007786454571731944
    expected_h1_error = 0.0449669071240554

    assert abs(l2_error - expected_l2_error) < 1.e-7
    assert abs(h1_error - expected_h1_error) < 1.e-7

#------------------------------------------------------------------------------
@pytest.mark.xfail
def test_poisson_2d_dir0_1234_user_function():

    solution = sin(pi*x)*sin(pi*y)

    # ...
    # User provides right-hand side in the form of a callable Python function:
    def f(x, y):
        from numpy import pi, sin
        return 2*pi**2*sin(pi*x)*sin(pi*y)

    # Python function is converted to Sympy's "implemented function" and then
    # called with symbolic arguments (x, y):
    f = implemented_function('f', f)(x, y)
    # ...

    dir_zero_boundary    = get_boundaries(1, 2, 3, 4)
    dir_nonzero_boundary = get_boundaries()

    l2_error, h1_error = run_poisson_2d(solution, f, dir_zero_boundary,
            dir_nonzero_boundary, ncells=[2**3, 2**3], degree=[2, 2])

    expected_l2_error =  0.00021808678604760232
    expected_h1_error =  0.013023570720360362

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)


###############################################################################
#            PARALLEL TESTS
###############################################################################

@pytest.mark.mpi
@pytest.mark.parametrize('backend',  [None, PSYDAC_BACKEND_GPYCCEL, PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP])
def test_poisson_2d_dir0_1234_parallel(backend):

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)

    dir_zero_boundary    = get_boundaries(1, 2, 3, 4)
    dir_nonzero_boundary = get_boundaries()

    l2_error, h1_error = run_poisson_2d(solution, f, dir_zero_boundary,
            dir_nonzero_boundary, ncells=[2**4, 2**4], degree=[2, 2],
            comm=MPI.COMM_WORLD, backend=backend)

    expected_l2_error =  2.6130834310750093e-05
    expected_h1_error =  0.0032076762540615743

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
