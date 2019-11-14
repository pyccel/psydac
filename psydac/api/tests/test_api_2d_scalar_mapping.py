# -*- coding: UTF-8 -*-
#
# A note on the mappings used in these tests:
#
#   - 'identity_2d.h5' is the identity mapping on the unit square [0, 1] X [0, 1]
#
#   - 'collela_2d.h5' is a NURBS mapping from the unit square [0, 1]^2 to the
#      larger square [-1, 1]^2, with deformations going as sin(pi x) * sin(pi y)
#
#   - 'quarter_annulus.h5' is a NURBS transformation from the unit square [0, 1]^2
#      to the quarter annulus in the lower-left quadrant of the Cartesian place
#      (hence both x and y are negative), with r_min = 0.5 and r_max = 0.5

from mpi4py import MPI
from sympy import pi, cos, sin
import pytest
import os

from sympde.calculus import grad, dot
from sympde.calculus import laplace
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of
from sympde.topology import NormalVector
from sympde.topology import Domain
from sympde.topology import Union
from sympde.expr import BilinearForm, LinearForm, integral
from sympde.expr import Norm
from sympde.expr import find, EssentialBC

from psydac.fem.basic          import FemField
from psydac.api.discretization import discretize

# ... get the mesh directory
try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']

except:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')
# ...

#==============================================================================
def run_poisson_2d_dir(filename, solution, f, dir_boundary=None, comm=None):

    # ... abstract model
    domain = Domain.from_file(filename)

    V = ScalarFunctionSpace('V', domain)

    if dir_boundary is None:
        B_dirichlet = domain.boundary
    elif len(dir_boundary) == 1:
        B_dirichlet = domain.get_boundary(**dir_boundary[0])
    else:
        B_dirichlet = Union(*[domain.get_boundary(**kw) for kw in dir_boundary])

    x,y = domain.coordinates

    F = element_of(V, name='F')

    v = element_of(V, name='v')
    u = element_of(V, name='u')

    int_0 = lambda expr: integral(domain , expr)

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), int_0(expr))

    expr = f*v
    l = LinearForm(v, int_0(expr))

    error = F - solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

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
def run_poisson_2d_dirneu(filename, solution, f, boundary, comm=None):

    assert( isinstance(boundary, (list, tuple)) )

    # ... abstract model
    domain = Domain.from_file(filename)

    V = ScalarFunctionSpace('V', domain)

    B_neumann = [domain.get_boundary(**kw) for kw in boundary]
    if len(B_neumann) == 1:
        B_neumann = B_neumann[0]

    else:
        B_neumann = Union(*B_neumann)

    x,y = domain.coordinates

    int_0 = lambda expr: integral(domain , expr)
    int_1 = lambda expr: integral(B_neumann , expr)

    F = element_of(V, name='F')

    v = element_of(V, name='v')
    u = element_of(V, name='u')

    nn = NormalVector('nn')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), int_0(expr))

    expr = f*v
    l0 = LinearForm(v, int_0(expr))

    expr = v*dot(grad(solution), nn)
    l_B_neumann = LinearForm(v, int_1(expr))

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

    V = ScalarFunctionSpace('V', domain)

    B_neumann = domain.boundary

    x,y = domain.coordinates

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

#==============================================================================
def run_biharmonic_2d_dir(filename, solution, f, comm=None):

    # ... abstract model
    domain = Domain.from_file(filename)

    V = ScalarFunctionSpace('V', domain)

    F = element_of(V, name='F')

    v = element_of(V, name='v')
    u = element_of(V, name='u')

    int_0 = lambda expr: integral(domain , expr)

    expr = laplace(v) * laplace(u)
    a = BilinearForm((v,u),int_0(expr))

    expr = f*v
    l = LinearForm(v, int_0(expr))

    error = F - solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')
    h2norm = Norm(error, domain, kind='h2')

    nn = NormalVector('nn')
    bc  = [EssentialBC(u, 0, domain.boundary)]
    bc += [EssentialBC(dot(grad(u), nn), 0, domain.boundary)]
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
    h2norm_h = discretize(h2norm, domain_h, Vh)
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
    h2_error = h2norm_h.assemble(F=phi)
    # ...

    return l2_error, h1_error, h2_error

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
def test_api_poisson_2d_dir_quarter_annulus():
    filename = os.path.join(mesh_dir, 'quarter_annulus.h5')

    from sympy.abc import x,y

    c = pi / (1. - 0.5**2)
    r2 = 1. - x**2 - y**2
    solution = x*y*sin(c * r2)
    f = 4.*c**2*x*y*(x**2 + y**2)*sin(c * r2) + 12.*c*x*y*cos(c * r2)

    l2_error, h1_error = run_poisson_2d_dir(filename, solution, f)

    expected_l2_error =  0.00010289930281268989
    expected_h1_error =  0.009473407914765117

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dir_circle():
    filename = os.path.join(mesh_dir, 'circle.h5')

    from sympy.abc import x,y

    solution = (1 - (x**2 + y**2)) * cos(2*pi*x) * cos(2*pi*y)
    f        = -laplace(solution)

    dir_boundary = [{'axis': 0, 'ext': 1}]
    l2_error, h1_error = run_poisson_2d_dir(filename, solution, f, dir_boundary)

    expected_l2_error = 0.0015245737751297718
    expected_h1_error = 0.06653900724243668

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_identity_1():
    filename = os.path.join(mesh_dir, 'identity_2d.h5')

    from sympy.abc import x,y

    solution = cos(0.5*pi*x)*sin(pi*y)
    f        = (5./4.)*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(filename, solution, f,
                                               [{'axis': 0, 'ext': -1}])

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

    l2_error, h1_error = run_poisson_2d_dirneu(filename, solution, f,
                                               [{'axis': 0, 'ext': 1}])

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

    l2_error, h1_error = run_poisson_2d_dirneu(filename, solution, f,
                                               [{'axis': 1, 'ext': -1}])

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

    l2_error, h1_error = run_poisson_2d_dirneu(filename, solution, f,
                                               [{'axis': 1, 'ext': 1}])

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
                                               [{'axis': 0, 'ext': -1},
                                                {'axis': 1, 'ext': -1}])

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
                                               [{'axis': 0, 'ext': -1},
                                                {'axis': 0, 'ext': 1},
                                                {'axis': 1, 'ext': -1}])

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
#    l2_error, h1_error = run_poisson_2d_dirneu(filename, solution, f, [{'axis': 0, 'ext': -1}])
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

    l2_error, h1_error = run_poisson_2d_dirneu(filename, solution, f, [{'axis': 0, 'ext': 1}])

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
#    l2_error, h1_error = run_poisson_2d_dirneu(filename, solution, f, [{'axis': 1, 'ext': -1}])
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

    l2_error, h1_error = run_poisson_2d_dirneu(filename, solution, f, [{'axis': 1, 'ext': 1}])

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

#==============================================================================
def test_api_biharmonic_2d_dir_identity():
    filename = os.path.join(mesh_dir, 'identity_2d.h5')

    from sympy.abc import x,y

    solution = (sin(pi*x)*sin(pi*y))**2
    f        = laplace(laplace(solution))

    l2_error, h1_error, h2_error = run_biharmonic_2d_dir(filename, solution, f)

    expected_l2_error = 0.015086415626060034
    expected_h1_error = 0.08773346232941553
    expected_h2_error = 1.9368842415954024

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    assert( abs(h2_error - expected_h2_error) < 1.e-7)

#==============================================================================
def test_api_biharmonic_2d_dir_collela():
    filename = os.path.join(mesh_dir, 'collela_2d.h5')

    from sympy.abc import x,y

    solution = (cos(pi*x/2)*cos(pi*y/2))**2
    f        = laplace(laplace(solution))

    l2_error, h1_error, h2_error = run_biharmonic_2d_dir(filename, solution, f)

    expected_l2_error = 0.10977627980052021
    expected_h1_error = 0.32254511059711766
    expected_h2_error = 1.87205519824758

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    assert( abs(h2_error - expected_h2_error) < 1.e-7)

#==============================================================================
def test_api_biharmonic_2d_dir_quarter_annulus():
    filename = os.path.join(mesh_dir, 'quarter_annulus.h5')

    from sympy.abc import x,y

    r_in  = 0.5
    r_out = 1
    kappa = 1 / 0.00643911127175763
    solution = kappa * (x * y * (x**2 + y**2 - r_in**2) * (x**2 + y**2 - r_out**2))**2
    f        = laplace(laplace(solution))

    l2_error, h1_error, h2_error = run_biharmonic_2d_dir(filename, solution, f)

    expected_l2_error = 0.016730298635551484
    expected_h1_error = 0.21243295522291714
    expected_h2_error = 7.572921831391894

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    assert( abs(h2_error - expected_h2_error) < 1.e-7)

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
