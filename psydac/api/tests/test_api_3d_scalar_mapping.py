#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os
from pathlib import Path

from mpi4py import MPI
from sympy import pi, cos, sin, symbols
import pytest

from sympde.calculus import grad, dot
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import element_of
from sympde.topology import NormalVector
from sympde.topology import Union
from sympde.topology import Domain
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm, SemiNorm
from sympde.expr     import find, EssentialBC

from psydac.api.discretization import discretize

# Get the mesh directory
import psydac.cad.mesh as mesh_mod
mesh_dir = Path(mesh_mod.__file__).parent

#==============================================================================
def run_poisson_3d_dir(filename, solution, f, comm=None):

    # ... abstract model
    domain = Domain.from_file(filename)

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
    domain_h = discretize(domain, filename=filename, comm=comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h)
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
def run_poisson_3d_dirneu(filename, solution, f, boundary, comm=None):

    assert( isinstance(boundary, (list, tuple)) )

    # ... abstract model
    domain = Domain.from_file(filename)

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
    domain_h = discretize(domain, filename=filename, comm=comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h)
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
def run_laplace_3d_neu(filename, solution, f, comm=None):

    # ... abstract model
    domain = Domain.from_file(filename)

    V = ScalarFunctionSpace('V', domain)

    B_neumann = domain.boundary

    x,y,z = domain.coordinates

    F = element_of(V, name='F')

    v = element_of(V, name='v')
    u = element_of(V, name='u')

    nn = NormalVector('nn')

    int_0 = lambda expr: integral(domain , expr)
    int_1 = lambda expr: integral(B_neumann , expr)

    expr = dot(grad(v), grad(u)) + v*u
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

    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v))
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, filename=filename, comm=comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h)
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
def test_api_poisson_3d_dir_collela():

    filename = os.path.join(mesh_dir, 'collela_3d.h5')

    x,y,z = symbols('x,y,z', real=True)

    solution = sin(pi*x)*sin(pi*y)*sin(pi*z)
    f        = 3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)

    l2_error, h1_error = run_poisson_3d_dir(filename, solution, f)

    expected_l2_error =  0.15687494944868827
    expected_h1_error =  1.518006054794389

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)


#==============================================================================
def test_api_poisson_3d_dirneu_identity_2():
    filename = os.path.join(mesh_dir, 'identity_3d.h5')

    x,y,z = symbols('x,y,z', real=True)

    solution = sin(0.5*pi*x)*sin(pi*y)*sin(pi*z)
    f        = (9./4.)*pi**2*solution

    l2_error, h1_error = run_poisson_3d_dirneu(filename, solution, f, [{'axis': 0, 'ext': 1}])

    expected_l2_error =  0.001438835012218704
    expected_h1_error =  0.03929404299152016

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)


#==============================================================================
def test_api_poisson_3d_dirneu_identity_13():
    filename = os.path.join(mesh_dir, 'identity_3d.h5')

    x,y,z = symbols('x,y,z', real=True)

    solution = cos(0.5*pi*x)*cos(0.5*pi*y)*sin(pi*z)
    f        = (3./2.)*pi**2*solution

    l2_error, h1_error = run_poisson_3d_dirneu(filename, solution, f,
                                               [{'axis': 0, 'ext': -1},
                                                {'axis': 1, 'ext': -1}])

    expected_l2_error =  0.0010275451113313282
    expected_h1_error =  0.027938446826372126

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_3d_dirneu_identity_24():
    filename = os.path.join(mesh_dir, 'identity_3d.h5')

    x,y,z = symbols('x,y,z', real=True)

    solution = sin(0.5*pi*x)*sin(0.5*pi*y)*sin(pi*z)
    f        = (3./2.)*pi**2*solution

    l2_error, h1_error = run_poisson_3d_dirneu(filename, solution, f,
                                               [{'axis': 0, 'ext': 1},
                                                {'axis': 1, 'ext': 1}])

    expected_l2_error =  0.001027545111330973
    expected_h1_error =  0.027938446826371813

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

##==============================================================================
## TODO DEBUG, not working since merge with devel
#def test_api_poisson_3d_dirneu_identity_123():
#    filename = os.path.join(mesh_dir, 'identity_3d.h5')
#
#    from sympy.abc import x,y,z
#
#    solution = cos(0.25*pi*x)*cos(0.5*pi*y)*sin(pi*z)
#    f        = (21./16.)*pi**2*solution
#
#    l2_error, h1_error = run_poisson_3d_dirneu(filename, solution, f,
#                                               [{'axis': 0, 'ext': -1},
#                                                {'axis': 0, 'ext': 1},
#                                                {'axis': 1, 'ext': -1}])
#
#    expected_l2_error =  0.0013124098938804697
#    expected_h1_error =  0.035441679549890456
#
#    assert( abs(l2_error - expected_l2_error) < 1.e-7)
#    assert( abs(h1_error - expected_h1_error) < 1.e-7)

##==============================================================================
## TODO DEBUG, not working since merge with devel
#def test_api_poisson_3d_dirneu_identity_1235():
#    filename = os.path.join(mesh_dir, 'identity_3d.h5')
#
#    from sympy.abc import x,y,z
#
#    solution = cos(0.25*pi*x)*cos(0.5*pi*y)*cos(0.5*pi*z)
#    f        = (9./16.)*pi**2*solution
#
#    l2_error, h1_error = run_poisson_3d_dirneu(filename, solution, f,
#                                               [{'axis': 0, 'ext': -1},
#                                                {'axis': 0, 'ext': 1},
#                                                {'axis': 1, 'ext': -1},
#                                                {'axis': 2, 'ext': -1}])
#
#    expected_l2_error =  0.00019677816039781896
#    expected_h1_error =  0.0058786142515790405
#
#    assert( abs(l2_error - expected_l2_error) < 1.e-7)
#    assert( abs(h1_error - expected_h1_error) < 1.e-7)


#==============================================================================
def test_api_poisson_3d_dirneu_collela_2():
    filename = os.path.join(mesh_dir, 'collela_3d.h5')

    x,y,z = symbols('x,y,z', real=True)

    solution = sin(0.25*pi*(x+1.))*sin(pi*y)*sin(pi*z)
    f        = (33./16.)*pi**2*solution

    l2_error, h1_error = run_poisson_3d_dirneu(filename, solution, f,
                                               [{'axis': 0, 'ext': 1}])

    expected_l2_error =  0.06091240085930318
    expected_h1_error =  0.6380043932563333

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)


##==============================================================================
## TODO DEBUG, not working since merge with devel
#def test_api_poisson_3d_dirneu_collela_13():
#    filename = os.path.join(mesh_dir, 'collela_3d.h5')
#
#    from sympy.abc import x,y,z
#
#    solution = sin(0.25*pi*(1.-x))*sin(0.25*pi*(1.-y))*sin(pi*z)
#    f        = (9./8.)*pi**2*solution
#
#    l2_error, h1_error = run_poisson_3d_dirneu(filename, solution, f,
#                                               [{'axis': 0, 'ext': -1},
#                                                {'axis': 1, 'ext': -1}])
#
#    expected_l2_error =  0.03786854933218588
#    expected_h1_error =  0.38437667047918933
#
#    assert( abs(l2_error - expected_l2_error) < 1.e-7)
#    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_3d_dirneu_collela_24():
    filename = os.path.join(mesh_dir, 'collela_3d.h5')

    x,y,z = symbols('x,y,z', real=True)

    solution = sin(0.25*pi*(x+1.))*sin(0.25*pi*(y+1.))*sin(pi*z)
    f        = (9./8.)*pi**2*solution

    l2_error, h1_error = run_poisson_3d_dirneu(filename, solution, f,
                                               [{'axis': 0, 'ext': 1},
                                                {'axis': 1, 'ext': 1}])

    expected_l2_error =  0.03793880183960465
    expected_h1_error =  0.38439642303250143

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

##==============================================================================
## TODO DEBUG, not working since merge with devel
#def test_api_poisson_3d_dirneu_collela_123():
#    filename = os.path.join(mesh_dir, 'collela_3d.h5')
#
#    from sympy.abc import x,y,z
#
#    solution = cos(pi*x)*sin(0.25*pi*(1.-y))*sin(pi*z)
#    f        = (33./16.)*pi**2*solution
#
#    l2_error, h1_error = run_poisson_3d_dirneu(filename, solution, f,
#                                               [{'axis': 0, 'ext': -1},
#                                                {'axis': 0, 'ext': 1},
#                                                {'axis': 1, 'ext': -1}])
#
#    expected_l2_error =  0.11963989196330076
#    expected_h1_error =  1.1267766354124575
#
#    assert( abs(l2_error - expected_l2_error) < 1.e-7)
#    assert( abs(h1_error - expected_h1_error) < 1.e-7)
#
##==============================================================================
## TODO DEBUG, not working since merge with devel
#def test_api_poisson_3d_dirneu_collela_1235():
#    filename = os.path.join(mesh_dir, 'collela_3d.h5')
#
#    from sympy.abc import x,y,z
#
#    solution = cos(pi*x)*sin(0.25*pi*(1.-y))*sin(0.25*pi*(1.-z))
#    f        = (9./8.)*pi**2*solution
#
#    l2_error, h1_error = run_poisson_3d_dirneu(filename, solution, f,
#                                               [{'axis': 0, 'ext': -1},
#                                                {'axis': 0, 'ext': 1},
#                                                {'axis': 1, 'ext': -1},
#                                                {'axis': 2, 'ext': -1}])
#
#    expected_l2_error =  0.13208728319093133
#    expected_h1_error =  0.9964934429086868

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_laplace_3d_neu_identity():
    filename = os.path.join(mesh_dir, 'identity_3d.h5')

    x,y,z = symbols('x,y,z', real=True)

    solution = cos(pi*x)*cos(pi*y)*cos(pi*z)
    f        = (3.*pi**2 + 1.)*solution

    l2_error, h1_error = run_laplace_3d_neu(filename, solution, f)

    expected_l2_error =  0.0016975430150953524
    expected_h1_error =  0.047009063231215

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

##==============================================================================
## TODO DEBUG, not working since merge with devel
#def test_api_laplace_3d_neu_collela():
#    filename = os.path.join(mesh_dir, 'collela_3d.h5')
#
#    from sympy.abc import x,y,z
#
#    solution = cos(pi*x)*cos(pi*y)*cos(pi*z)
#    f        = (3.*pi**2 + 1.)*solution
#
#    l2_error, h1_error = run_laplace_3d_neu(filename, solution, f)
#
#    expected_l2_error =  0.1768000505351402
#    expected_h1_error =  1.7036022067226382
#
#    assert( abs(l2_error - expected_l2_error) < 1.e-7)
#    assert( abs(h1_error - expected_h1_error) < 1.e-7)

###############################################################################
#            PARALLEL TESTS
###############################################################################

#==============================================================================
@pytest.mark.mpi
def test_api_poisson_3d_dir_collela():

    filename = os.path.join(mesh_dir, 'collela_3d.h5')

    x,y,z = symbols('x,y,z', real=True)

    solution = sin(pi*x)*sin(pi*y)*sin(pi*z)
    f        = 3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)

    l2_error, h1_error = run_poisson_3d_dir(filename, solution, f,
                                            comm=MPI.COMM_WORLD)

    expected_l2_error =  0.15687494944868827
    expected_h1_error =  1.518006054794389

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
