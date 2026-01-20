#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os
from pathlib import Path

from sympy import Tuple, Matrix, symbols
from sympy import pi, sin

from sympde.calculus import grad, dot, inner
from sympde.topology import VectorFunctionSpace
from sympde.topology import element_of
from sympde.topology import Domain
from sympde.topology import Union
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm, SemiNorm
from sympde.expr     import find, EssentialBC

from psydac.api.discretization import discretize

# Get the mesh directory
import psydac.cad.mesh as mesh_mod
mesh_dir = Path(mesh_mod.__file__).parent

x,y = symbols('x,y', real=True)

#==============================================================================
def run_vector_poisson_2d_dir(filename, solution, f):

    # ... abstract model
    domain = Domain.from_file(filename)

    V = VectorFunctionSpace('V', domain)

    x,y = domain.coordinates

    v = element_of(V, name='v')
    u = element_of(V, name='u')

    int_0 = lambda expr: integral(domain , expr)

    expr = inner(grad(v), grad(u))
    a = BilinearForm((v,u), int_0(expr))

    expr = dot(f, v)
    l = LinearForm(v, int_0(expr))

    error  = Matrix([u[0]-solution[0], u[1]-solution[1]])
    l2norm =     Norm(error, domain, kind='l2')
    h1norm = SemiNorm(error, domain, kind='h1')

    bc = EssentialBC(u, 0, domain.boundary)
    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v), bc=bc)
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
def test_api_vector_poisson_2d_dir_identity():
    filename = os.path.join(mesh_dir, 'identity_2d.h5')

    u1 = sin(pi*x)*sin(pi*y)
    u2 = sin(pi*x)*sin(pi*y)
    solution = Tuple(u1, u2)

    f1 = 2*pi**2*sin(pi*x)*sin(pi*y)
    f2 = 2*pi**2*sin(pi*x)*sin(pi*y)
    f = Tuple(f1, f2)

    l2_error, h1_error = run_vector_poisson_2d_dir(filename, solution, f)

    expected_l2_error =  0.0003084212905795541
    expected_h1_error =  0.01841811034325851

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_vector_poisson_2d_dir_collela():
    filename = os.path.join(mesh_dir, 'collela_2d.h5')

    u1 = sin(pi*x)*sin(pi*y)
    u2 = sin(pi*x)*sin(pi*y)
    solution = Tuple(u1, u2)

    f1 = 2*pi**2*sin(pi*x)*sin(pi*y)
    f2 = 2*pi**2*sin(pi*x)*sin(pi*y)
    f = Tuple(f1, f2)

    l2_error, h1_error = run_vector_poisson_2d_dir(filename, solution, f)
    print(l2_error, h1_error)

    expected_l2_error =  0.04289215947798109
    expected_h1_error =  0.583010694052124

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
