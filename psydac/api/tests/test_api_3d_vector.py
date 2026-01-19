#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from sympy import Tuple, Matrix
from sympy import pi, sin

from sympde.calculus import grad, dot, inner
from sympde.topology import VectorFunctionSpace
from sympde.topology import element_of
from sympde.topology import Cube
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm, SemiNorm
from sympde.expr     import find, EssentialBC

from psydac.api.discretization import discretize

#==============================================================================
def run_vector_poisson_3d_dir(solution, f, ncells, degree):

    # ... abstract model
    domain = Cube()

    V = VectorFunctionSpace('V', domain)

    x,y,z = domain.coordinates

    v = element_of(V, name='v')
    u = element_of(V, name='u')

    int_0 = lambda expr: integral(domain , expr)

    expr = inner(grad(v), grad(u))
    a = BilinearForm((v,u), int_0(expr))

    expr = dot(f, v)
    l = LinearForm(v, int_0(expr))

    error  = Matrix([u[0]-solution[0], u[1]-solution[1], u[2]-solution[2]])
    l2norm =     Norm(error, domain, kind='l2')
    h1norm = SemiNorm(error, domain, kind='h1')

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
    uh = equation_h.solve()
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(u = uh)
    h1_error = h1norm_h.assemble(u = uh)
    # ...

    return l2_error, h1_error

#==============================================================================
def test_api_vector_poisson_3d_dir_1():

    from sympy import symbols
    x1, x2, x3 = symbols('x1, x2, x3', real=True)

    u1 = sin(pi*x1)*sin(pi*x2)*sin(pi*x3)
    u2 = sin(pi*x1)*sin(pi*x2)*sin(pi*x3)
    u3 = sin(pi*x1)*sin(pi*x2)*sin(pi*x3)
    solution = Tuple(u1, u2, u3)

    f1 = 3*pi**2*sin(pi*x1)*sin(pi*x2)*sin(pi*x3)
    f2 = 3*pi**2*sin(pi*x1)*sin(pi*x2)*sin(pi*x3)
    f3 = 3*pi**2*sin(pi*x1)*sin(pi*x2)*sin(pi*x3)
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
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()
