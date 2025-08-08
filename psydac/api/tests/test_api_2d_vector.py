# -*- coding: UTF-8 -*-
import time
from collections import namedtuple

from sympy import Tuple, Matrix
from sympy import pi, sin

from sympde.calculus import grad, dot, inner
from sympde.topology import VectorFunctionSpace
from sympde.topology import element_of
from sympde.topology import Square
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm, SemiNorm
from sympde.expr     import find, EssentialBC

from psydac.api.discretization import discretize
from psydac.api.essential_bc   import apply_essential_bc
from psydac.linalg.solvers     import inverse
from psydac.fem.basic          import FemField

#==============================================================================
def run_vector_poisson_2d_dir(solution, f, *, ncells, degree,
                              backend = None, profile = False):

    # Dictionary for the timings, which will remain empty if profile is False
    timing = {}

    # ... abstract model
    domain = Square()

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
    domain_h = discretize(domain, ncells=ncells)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    # ...

    # ... discretize the equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh], backend=backend)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, domain_h, Vh, backend=backend)
    h1norm_h = discretize(h1norm, domain_h, Vh, backend=backend)
    # ...

    # ... solve the discrete equation
    if profile:
        tb = time.time()
        A  = equation_h.lhs.assemble()
        te = time.time()
        timing['matrix'] = te - tb

        tb = time.time()
        b  = equation_h.rhs.assemble()
        te = time.time()
        timing['rhs'] = te - tb

        # Apply essential BCs to A and b
        apply_essential_bc(A, *equation_h.bc)
        apply_essential_bc(b, *equation_h.bc)

        # Solve linear system
        A_inv = inverse(A, solver='cg')
        x = A_inv.dot(b)

        # Store result in a new FEM field
        uh = FemField(Vh, coeffs=x)

    else:
        uh = equation_h.solve()
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(u = uh)
    h1_error = h1norm_h.assemble(u = uh)
    # ...

    return l2_error, h1_error, timing

#==============================================================================
def test_api_vector_poisson_2d_dir_1(backend=None, profile=False):

    from sympy import symbols
    x1, x2 = symbols('x1, x2', real=True)

    u1 = sin(pi*x1)*sin(pi*x2)
    u2 = sin(pi*x1)*sin(pi*x2)
    solution = Tuple(u1, u2)

    f1 = 2*pi**2*sin(pi*x1)*sin(pi*x2)
    f2 = 2*pi**2*sin(pi*x1)*sin(pi*x2)
    f = Tuple(f1, f2)

#    l2_error, h1_error = run_vector_poisson_2d_dir(solution, f,
#                                            ncells=[2**3,2**3], degree=[2,2])

    l2_error, h1_error, timing = run_vector_poisson_2d_dir(solution, f,
                                        ncells=[2**3,2**3], degree=[2,2],
                                        backend = backend,
                                        profile = profile)

    expected_l2_error =  0.00030842129060800865
    expected_h1_error =  0.018418110343256442

    assert abs(l2_error - expected_l2_error) < 1.e-7
    assert abs(h1_error - expected_h1_error) < 1.e-7

    return timing

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()
