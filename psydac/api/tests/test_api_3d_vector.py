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


#==============================================================================
def run_vector_poisson_3d_dir(solution, f, ncells, degree):

    # ... abstract model
    domain = Cube()

    V = VectorFunctionSpace('V', domain)

    x,y,z = domain.coordinates

    F = element_of_space(V, name='F')

    v = element_of_space(V, name='v')
    u = element_of_space(V, name='u')

    expr = inner(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    expr = dot(f, v)
    l = LinearForm(v, expr)

    error = Matrix([F[0]-solution[0], F[1]-solution[1], F[2]-solution[2]])
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    bc = EssentialBC(u, 0, domain.boundary)
    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v), bc=bc)
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells)
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
def test_api_vector_poisson_3d_dir_1():

    from sympy.abc import x,y,z

    u1 = sin(pi*x)*sin(pi*y)*sin(pi*z)
    u2 = sin(pi*x)*sin(pi*y)*sin(pi*z)
    u3 = sin(pi*x)*sin(pi*y)*sin(pi*z)
    solution = Tuple(u1, u2, u3)

    f1 = 3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)
    f2 = 3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)
    f3 = 3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)
    f = Tuple(f1, f2, f3)

    l2_error, h1_error = run_vector_poisson_3d_dir(solution, f,
                                                   ncells=[2**2,2**2,2**2], degree=[2,2,2])

    expected_l2_error =  0.0030390821236931788
    expected_h1_error =  0.08346666256929804

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
