# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin
from sympy import S
from sympy import Tuple
from sympy import Matrix

from sympde.core import Constant
from sympde.calculus import grad, dot, inner, cross, rot, curl, div
from sympde.calculus import laplace, hessian
from sympde.topology import (dx, dy, dz)
from sympde.topology import FunctionSpace, VectorFunctionSpace
from sympde.topology import ScalarField, VectorField
from sympde.topology import ProductSpace
from sympde.topology import ScalarTestFunction
from sympde.topology import VectorTestFunction
from sympde.topology import Unknown
from sympde.topology import InteriorDomain, Union
from sympde.topology import Boundary, NormalVector, TangentVector
from sympde.topology import Domain, Line, Square, Cube
from sympde.topology import Trace, trace_0, trace_1
from sympde.topology import Union
from sympde.topology import Mapping
from sympde.expr import BilinearForm, LinearForm
from sympde.expr import Norm
from sympde.expr import find, EssentialBC

from spl.fem.basic   import FemField
from spl.fem.vector  import ProductFemSpace, VectorFemField
from spl.api.discretization import discretize

from spl.mapping.discrete import SplineMapping

from numpy import linspace, zeros, allclose
from mpi4py import MPI
import pytest

import os

# ... get the mesh directory
try:
    mesh_dir = os.environ['SPL_MESH_DIR']

except:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')
# ...

#==============================================================================
def run_poisson_2d_dir(filename, solution, f, comm=None):

    # ... abstract model
    domain = Domain.from_file(filename)

    V = FunctionSpace('V', domain)

    x,y = domain.coordinates

    F = ScalarField(V, name='F')

    v = ScalarTestFunction(V, name='v')
    u = ScalarTestFunction(V, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    expr = f*v
    l = LinearForm(v, expr)

    error = F - solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

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
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, x )
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)
    # ...

    return l2_error, h1_error

#==============================================================================
def run_poisson_2d_dirneu(filename, solution, f, boundary, comm=None):

    assert( isinstance(boundary, (list, tuple)) )

    # ... abstract model
    domain = Domain.from_file(filename)

    V = FunctionSpace('V', domain)

    B_neumann = [domain.get_boundary(i) for i in boundary]
    if len(B_neumann) == 1:
        B_neumann = B_neumann[0]

    else:
        B_neumann = Union(*B_neumann)

    x,y = domain.coordinates

    F = ScalarField(V, name='F')

    v = ScalarTestFunction(V, name='v')
    u = ScalarTestFunction(V, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    expr = f*v
    l0 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B_neumann)
    l_B_neumann = LinearForm(v, expr)

    expr = l0(v) + l_B_neumann(v)
    l = LinearForm(v, expr)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

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
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, x )
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)
    # ...

    return l2_error, h1_error

#==============================================================================
def run_laplace_2d_neu(filename, solution, f, comm=None):

    # ... abstract model
    domain = Domain.from_file(filename)

    V = FunctionSpace('V', domain)

    B_neumann = domain.boundary

    x,y = domain.coordinates

    F = ScalarField(V, name='F')

    v = ScalarTestFunction(V, name='v')
    u = ScalarTestFunction(V, name='u')

    expr = dot(grad(v), grad(u)) + v*u
    a = BilinearForm((v,u), expr)

    expr = f*v
    l0 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B_neumann)
    l_B_neumann = LinearForm(v, expr)

    expr = l0(v) + l_B_neumann(v)
    l = LinearForm(v, expr)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

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
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, x )
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)
    # ...

    return l2_error, h1_error

###############################################################################
#            SERIAL TESTS
###############################################################################

#==============================================================================
def test_api_poisson_2d_dir_identity():
    filename = os.path.join(mesh_dir, 'identity_2d.h5')

    from sympy.abc import x,y

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)

    l2_error, h1_error = run_poisson_2d_dir(filename, solution, f)

    expected_l2_error =  0.00021808678604159413
    expected_h1_error =  0.013023570720357957

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dir_collela():
    filename = os.path.join(mesh_dir, 'collela_2d.h5')

    from sympy.abc import x,y

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)

    l2_error, h1_error = run_poisson_2d_dir(filename, solution, f)

    expected_l2_error =  0.03032933682661518
    expected_h1_error =  0.41225081526853247

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
# TODO h1 norm not working => regression due to the last changes in sympde
def test_api_poisson_2d_dir_quart_circle():
    filename = os.path.join(mesh_dir, 'quart_circle.h5')

    from sympy.abc import x,y

    c = pi / (1. - 0.5**2)
    r2 = 1. - x**2 - y**2
    solution = x*y*sin(c * r2)
    f = 4.*c**2*x*y*(x**2 + y**2)*sin(c * r2) + 12.*c*x*y*cos(c * r2)

    l2_error, h1_error = run_poisson_2d_dir(filename, solution, f)

    expected_l2_error =  0.00010289930281268989
    expected_h1_error =  0.009473407914765117

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
#    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_identity_1():
    filename = os.path.join(mesh_dir, 'identity_2d.h5')

    from sympy.abc import x,y

    solution = cos(0.5*pi*x)*sin(pi*y)
    f        = (5./4.)*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(filename, solution, f, ['Gamma_1'])

    expected_l2_error =  0.00015546057795986509
    expected_h1_error =  0.009269302784527035

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_identity_2():
    filename = os.path.join(mesh_dir, 'identity_2d.h5')

    from sympy.abc import x,y

    solution = sin(0.5*pi*x)*sin(pi*y)
    f        = (5./4.)*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(filename, solution, f, ['Gamma_2'])

    expected_l2_error =  0.00015546057795095866
    expected_h1_error =  0.009269302784528054

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_identity_3():
    filename = os.path.join(mesh_dir, 'identity_2d.h5')

    from sympy.abc import x,y

    solution = sin(pi*x)*cos(0.5*pi*y)
    f        = (5./4.)*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(filename, solution, f, ['Gamma_3'])

    expected_l2_error =  0.00015546057796188848
    expected_h1_error =  0.009269302784527448

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_identity_4():
    filename = os.path.join(mesh_dir, 'identity_2d.h5')

    from sympy.abc import x,y

    solution = sin(pi*x)*sin(0.5*pi*y)
    f        = (5./4.)*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(filename, solution, f, ['Gamma_4'])

    expected_l2_error =  0.00015546057795073548
    expected_h1_error =  0.009269302784522822

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_identity_13():
    filename = os.path.join(mesh_dir, 'identity_2d.h5')

    from sympy.abc import x,y

    solution = cos(0.5*pi*x)*cos(0.5*pi*y)
    f        = (1./2.)*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(filename, solution, f,
                                               ['Gamma_1', 'Gamma_3'])

    expected_l2_error =  2.6119892693464717e-05
    expected_h1_error =  0.0016032430287989195

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_identity_123():
    filename = os.path.join(mesh_dir, 'identity_2d.h5')

    from sympy.abc import x,y

    solution = cos(pi*x)*cos(0.5*pi*y)
    f        = 5./4.*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(filename, solution, f,
                                               ['Gamma_1', 'Gamma_2', 'Gamma_3'])

    expected_l2_error =  0.00015492540684276186
    expected_h1_error =  0.009242166615517364

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)


##==============================================================================
# TODO DEBUG, not working since merge with devel
#def test_api_poisson_2d_dirneu_collela_1():
#    filename = os.path.join(mesh_dir, 'collela_2d.h5')
#
#    from sympy.abc import x,y
#
#    solution = cos(0.25*pi*x)*sin(pi*y)
#    f        = (17./16.)*pi**2*solution
#
#    l2_error, h1_error = run_poisson_2d_dirneu(filename, solution, f, ['Gamma_1'])
#
#    expected_l2_error =  0.013540717397796734
#    expected_h1_error =  0.19789463571596025
#
#    assert( abs(l2_error - expected_l2_error) < 1.e-7)
#    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_collela_2():
    filename = os.path.join(mesh_dir, 'collela_2d.h5')

    from sympy.abc import x,y

    solution = sin(0.25*pi*(x+1.))*sin(pi*y)
    f        = (17./16.)*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(filename, solution, f, ['Gamma_2'])

    expected_l2_error =  0.012890849094111699
    expected_h1_error =  0.19553563279728328

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

##==============================================================================
## TODO DEBUG, not working since merge with devel
#def test_api_poisson_2d_dirneu_collela_3():
#    filename = os.path.join(mesh_dir, 'collela_2d.h5')
#
#    from sympy.abc import x,y
#
#    solution = cos(0.25*pi*y)*sin(pi*x)
#    f        = (17./16.)*pi**2*solution
#
#    l2_error, h1_error = run_poisson_2d_dirneu(filename, solution, f, ['Gamma_3'])
#
#    expected_l2_error =  0.013540717397817427
#    expected_h1_error =  0.19789463571595994
#
#    assert( abs(l2_error - expected_l2_error) < 1.e-7)
#    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_collela_4():
    filename = os.path.join(mesh_dir, 'collela_2d.h5')

    from sympy.abc import x,y

    solution = sin(0.25*pi*(y+1.))*sin(pi*x)
    f        = (17./16.)*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(filename, solution, f, ['Gamma_4'])

    expected_l2_error =  0.012890849094111942
    expected_h1_error =  0.19553563279728325

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_laplace_2d_neu_identity():
    filename = os.path.join(mesh_dir, 'identity_2d.h5')

    from sympy.abc import x,y

    solution = cos(pi*x)*cos(pi*y)
    f        = (2.*pi**2 + 1.)*solution

    l2_error, h1_error = run_laplace_2d_neu(filename, solution, f)

    expected_l2_error =  0.00021728465388208586
    expected_h1_error =  0.012984852988123631

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

##==============================================================================
## TODO DEBUG, not working since merge with devel
#def test_api_laplace_2d_neu_collela():
#    filename = os.path.join(mesh_dir, 'collela_2d.h5')
#
#    from sympy.abc import x,y
#
#    solution = cos(pi*x)*cos(pi*y)
#    f        = (2.*pi**2 + 1.)*solution
#
#    l2_error, h1_error = run_laplace_2d_neu(filename, solution, f)
#
#    expected_l2_error =  0.029603335241478155
#    expected_h1_error =  0.4067746760978581
#
#    assert( abs(l2_error - expected_l2_error) < 1.e-7)
#    assert( abs(h1_error - expected_h1_error) < 1.e-7)

###############################################################################
#            PARALLEL TESTS
###############################################################################

#==============================================================================
@pytest.mark.parallel
def test_api_poisson_2d_dir_identity_parallel():
    filename = os.path.join(mesh_dir, 'identity_2d.h5')

    from sympy.abc import x,y

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)

    l2_error, h1_error = run_poisson_2d_dir(filename, solution, f,
                                            comm=MPI.COMM_WORLD)

    expected_l2_error =  0.00021808678604159413
    expected_h1_error =  0.013023570720357957

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)


#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()
