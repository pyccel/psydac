#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from mpi4py import MPI
from sympy import pi, cos, sin
import pytest

from sympde.calculus import grad, dot
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of
from sympde.topology import NormalVector
from sympde.topology import Cube
from sympde.topology import Union
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm, SemiNorm
from sympde.expr     import find, EssentialBC

from psydac.api.discretization import discretize

#==============================================================================
def run_poisson_3d_dir(solution, f, ncells, degree, comm=None):

    # ... abstract model
    domain = Cube()

    V = ScalarFunctionSpace('V', domain)

    x,y,z = domain.coordinates

    v = element_of(V, name='v')
    u = element_of(V, name='u')

    int_0 = lambda expr: integral(domain , expr)

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), int_0(expr))

    expr = f*v
    l = LinearForm(v, int_0(expr))

    error  = u - solution
    l2norm =     Norm(error, domain, kind='l2')
    h1norm = SemiNorm(error, domain, kind='h1')

    bc = EssentialBC(u, 0, domain.boundary)
    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v), bc=bc)
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells, comm=comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    # ...

    # ... dsicretize the equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh])
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, domain_h, Vh)
    h1norm_h = discretize(h1norm, domain_h, Vh)
    # ...

    # ... solve the discrete equation
    uh = equation_h.solve()
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(u = uh)
    h1_error = h1norm_h.assemble(u = uh)
    # ...

    return l2_error, h1_error

#==============================================================================
def run_poisson_3d_dirneu(solution, f, boundary, ncells, degree, comm=None):

    assert( isinstance(boundary, (list, tuple)) )

    # ... abstract model
    domain = Cube()

    V = ScalarFunctionSpace('V', domain)

    B_neumann = [domain.get_boundary(**kw) for kw in boundary]
    if len(B_neumann) == 1:
        B_neumann = B_neumann[0]

    else:
        B_neumann = Union(*B_neumann)

    x,y,z = domain.coordinates

    v = element_of(V, name='v')
    u = element_of(V, name='u')

    nn = NormalVector('nn')

    int_0 = lambda expr: integral(domain , expr)
    int_1 = lambda expr: integral(B_neumann , expr)

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), int_0(expr))

    expr = f*v
    l0 = LinearForm(v, int_0(expr))

    expr = v*dot(grad(solution), nn)
    l_B_neumann = LinearForm(v, int_1(expr))

    expr = l0(v) + l_B_neumann(v)
    l = LinearForm(v, expr)

    error  = u - solution
    l2norm =     Norm(error, domain, kind='l2')
    h1norm = SemiNorm(error, domain, kind='h1')

    B_dirichlet = domain.boundary.complement(B_neumann)
    bc = EssentialBC(u, 0, B_dirichlet)

    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v), bc=bc)
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells, comm=comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    # ...

    # ... dsicretize the equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh])
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, domain_h, Vh)
    h1norm_h = discretize(h1norm, domain_h, Vh)
    # ...

    # ... solve the discrete equation
    uh = equation_h.solve()
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(u = uh)
    h1_error = h1norm_h.assemble(u = uh)
    # ...

    return l2_error, h1_error


###############################################################################
#            SERIAL TESTS
###############################################################################

#==============================================================================
def test_api_poisson_3d_dir_1():

    from sympy import symbols
    x1, x2, x3 = symbols('x1, x2, x3', real=True)

    solution = sin(pi*x1)*sin(pi*x2)*sin(pi*x3)
    f        = 3*pi**2*sin(pi*x1)*sin(pi*x2)*sin(pi*x3)

    l2_error, h1_error = run_poisson_3d_dir(solution, f,
                                            ncells=[2**2,2**2,2**2], degree=[2,2,2])

    expected_l2_error =  0.0017546148822053188
    expected_h1_error =  0.048189500102744275

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

##==============================================================================
#def test_api_poisson_3d_dirneu_1():
#
#    from sympy.abc import x,y,z
#
#    solution = sin(0.5*pi*x)*sin(pi*y)*sin(pi*z)
#    f        = (9./4.)*pi**2*solution
#
#    l2_error, h1_error = run_poisson_3d_dirneu(solution, f, [{'axis': 0, 'ext': -1}],
#                                               ncells=[2**2,2**2,2**2], degree=[2,2,2])
#
#    expected_l2_error =  0.0014388350122198999
#    expected_h1_error =  0.03929404299152041
#
#    assert( abs(l2_error - expected_l2_error) < 1.e-7)
#    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_3d_dirneu_2():

    from sympy import symbols
    x1, x2, x3 = symbols('x1, x2, x3', real=True)

    solution = sin(0.5*pi*x1)*sin(pi*x2)*sin(pi*x3)
    f        = (9./4.)*pi**2*solution

    l2_error, h1_error = run_poisson_3d_dirneu(solution, f, [{'axis': 0, 'ext': 1}],
                                               ncells=[2**2,2**2,2**2], degree=[2,2,2])

    expected_l2_error =  0.0014388350122198999
    expected_h1_error =  0.03929404299152041

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_3d_dirneu_13():

    from sympy import symbols
    x1, x2, x3 = symbols('x1, x2, x3', real=True)

    solution = cos(0.5*pi*x1)*cos(0.5*pi*x2)*sin(pi*x3)
    f        = (3./2.)*pi**2*solution

    l2_error, h1_error = run_poisson_3d_dirneu(solution, f,
                                               [{'axis': 0, 'ext': -1},
                                                {'axis': 1, 'ext': -1}],
                                               ncells=[2**2,2**2,2**2], degree=[2,2,2])

    expected_l2_error =  0.0010275451113309177
    expected_h1_error =  0.027938446826372313

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_3d_dirneu_24():

    from sympy import symbols
    x1, x2, x3 = symbols('x1, x2, x3', real=True)

    solution = sin(0.5*pi*x1)*sin(0.5*pi*x2)*sin(pi*x3)
    f        = (3./2.)*pi**2*solution

    l2_error, h1_error = run_poisson_3d_dirneu(solution, f,
                                               [{'axis': 0, 'ext': 1},
                                                {'axis': 1, 'ext': 1}],
                                               ncells=[2**2,2**2,2**2], degree=[2,2,2])

    expected_l2_error =  0.0010275451113330345
    expected_h1_error =  0.027938446826372445

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

##==============================================================================
## TODO DEBUG, not working since merge with devel
#def test_api_poisson_3d_dirneu_123():
#
#    from sympy.abc import x,y,z
#
#    solution = cos(0.25*pi*x)*cos(0.5*pi*y)*sin(pi*z)
#    f        = (21./16.)*pi**2*solution
#
#    l2_error, h1_error = run_poisson_3d_dirneu(solution, f,
#                                               [{'axis': 0, 'ext': -1},
#                                                {'axis': 0, 'ext': 1},
#                                                {'axis': 1, 'ext': -1}],
#                                               ncells=[2**2,2**2,2**2], degree=[2,2,2])
#
#    expected_l2_error =  0.0013124098938818625
#    expected_h1_error =  0.035441679549891296
#
#    assert( abs(l2_error - expected_l2_error) < 1.e-7)
#    assert( abs(h1_error - expected_h1_error) < 1.e-7)

##==============================================================================
## TODO DEBUG, not working since merge with devel
#def test_api_poisson_3d_dirneu_1235():
#
#    from sympy.abc import x,y,z
#
#    solution = cos(0.25*pi*x)*cos(0.5*pi*y)*cos(0.5*pi*z)
#    f        = (9./16.)*pi**2*solution
#
#    l2_error, h1_error = run_poisson_3d_dirneu(solution, f,
#                                               [{'axis': 0, 'ext': -1},
#                                                {'axis': 0, 'ext': 1},
#                                                {'axis': 1, 'ext': -1},
#                                                {'axis': 2, 'ext': -1}],
#                                               ncells=[2**2,2**2,2**2], degree=[2,2,2])
#
#    expected_l2_error =  0.00019677816055503394
#    expected_h1_error =  0.0058786142515821265
#
#    assert( abs(l2_error - expected_l2_error) < 1.e-7)
#    assert( abs(h1_error - expected_h1_error) < 1.e-7)

###############################################################################
#            PARALLEL TESTS
###############################################################################

#==============================================================================
@pytest.mark.mpi
def test_api_poisson_3d_dir_1_parallel():

    from sympy import symbols
    x1, x2, x3 = symbols('x1, x2, x3', real=True)

    solution = sin(pi*x1)*sin(pi*x2)*sin(pi*x3)
    f        = 3*pi**2*sin(pi*x1)*sin(pi*x2)*sin(pi*x3)

    l2_error, h1_error = run_poisson_3d_dir(solution, f,
                                            ncells=[2**2,2**2,2**2], degree=[2,2,2],
                                            comm=MPI.COMM_WORLD)

    expected_l2_error =  0.0017546148822053188
    expected_h1_error =  0.048189500102744275

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
