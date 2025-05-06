# -*- coding: UTF-8 -*-

import os

from sympy import Tuple, Matrix, symbols
from sympy import pi, sin

from sympde.calculus import grad, inner
from sympde.topology import VectorFunctionSpace
from sympde.topology import elements_of
from sympde.topology import Domain
from sympde.expr import BilinearForm, LinearForm, integral
from sympde.expr import Norm, SemiNorm
from sympde.expr import find, EssentialBC

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
def run_vector_poisson_3d_dir(filename, solution, f):

    # ... abstract model
    domain = Domain.from_file(filename)

    V = VectorFunctionSpace('V', domain)
    u, v = elements_of(V, names='u, v')

    int_0 = lambda expr: integral(domain, expr)

    a = BilinearForm((u, v), int_0(inner(grad(u), grad(v))))

    l = LinearForm(v, int_0(inner(f, v)))

    error  = Matrix([u[0]-solution[0], u[1]-solution[1], u[2]-solution[2]])
    l2norm =     Norm(error, domain, kind='l2')
    h1norm = SemiNorm(error, domain, kind='h1')

    bc = EssentialBC(u, 0, domain.boundary)
    equation = find(u, forall=v, lhs=a(u, v), rhs=l(v), bc=bc)
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, filename=filename)
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
def test_api_vector_poisson_3d_dir_identity():
    filename = os.path.join(mesh_dir, 'identity_3d.h5')

    x,y,z = symbols('x,y,z', real=True)

    u1 = sin(pi*x)*sin(pi*y)*sin(pi*z)
    u2 = sin(pi*x)*sin(pi*y)*sin(pi*z)
    u3 = sin(pi*x)*sin(pi*y)*sin(pi*z)
    solution = Tuple(u1, u2, u3)

    f1 = 3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)
    f2 = 3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)
    f3 = 3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)
    f = Tuple(f1, f2, f3)

    l2_error, h1_error = run_vector_poisson_3d_dir(filename, solution, f)

    expected_l2_error =  0.0030390821236941324
    expected_h1_error =  0.0834666625692994

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_vector_poisson_3d_dir_collela():
    filename = os.path.join(mesh_dir, 'collela_3d.h5')

    x,y,z = symbols('x,y,z', real=True)

    u1 = sin(pi*x)*sin(pi*y)*sin(pi*z)
    u2 = sin(pi*x)*sin(pi*y)*sin(pi*z)
    u3 = sin(pi*x)*sin(pi*y)*sin(pi*z)
    solution = Tuple(u1, u2, u3)

    f1 = 3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)
    f2 = 3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)
    f3 = 3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)
    f = Tuple(f1, f2, f3)

    l2_error, h1_error = run_vector_poisson_3d_dir(filename, solution, f)

    expected_l2_error =  0.2717153828799274
    expected_h1_error =  2.6292636131010663

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
