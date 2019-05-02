# -*- coding: UTF-8 -*-

from sympy import Tuple, Matrix
from sympy import pi, cos, sin

from sympde.core import Constant
from sympde.calculus import grad, dot, inner, cross, rot, curl, div
from sympde.calculus import laplace, hessian
from sympde.topology import (dx, dy, dz)
from sympde.topology import FunctionSpace, VectorFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import element_of_space
from sympde.topology import Boundary, NormalVector, TangentVector
from sympde.topology import Domain, Line, Square, Cube
from sympde.topology import Trace, trace_0, trace_1
from sympde.topology import Union
from sympde.expr import BilinearForm, LinearForm
from sympde.expr import Norm
from sympde.expr import find, EssentialBC

from psydac.fem.vector  import VectorFemField
from psydac.api.discretization import discretize

from numpy import linspace, zeros, allclose

import os

# ... get the mesh directory
try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']

except:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')
# ...

#==============================================================================
def run_vector_poisson_2d_dir(filename, solution, f):

    # ... abstract model
    domain = Domain.from_file(filename)

    V = VectorFunctionSpace('V', domain)

    x,y = domain.coordinates

    F = element_of_space(V, name='F')

    v = element_of_space(V, name='v')
    u = element_of_space(V, name='u')

    expr = inner(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    expr = dot(f, v)
    l = LinearForm(v, expr)

    error = Matrix([F[0]-solution[0], F[1]-solution[1]])
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

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
    x = equation_h.solve()
    # ...

    # ...
    phi = VectorFemField( Vh, x )
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)
    # ...

    return l2_error, h1_error

#==============================================================================
def test_api_vector_poisson_2d_dir_identity():
    filename = os.path.join(mesh_dir, 'identity_2d.h5')

    from sympy.abc import x,y

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

    from sympy.abc import x,y

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
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()
