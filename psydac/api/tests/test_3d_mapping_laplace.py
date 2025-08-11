import os

from sympy import pi, cos, symbols

from sympde.calculus import grad, dot
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of
from sympde.topology import NormalVector
from sympde.topology import Domain
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm, SemiNorm
from sympde.expr     import find

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

#==============================================================================
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

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()