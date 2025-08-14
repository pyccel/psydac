# -*- coding: UTF-8 -*-
import time

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
                              comm=None, backend=None, timing=None):

    # The dictionary for the timings is modified in-place
    profile = (timing is not None)
    if profile:
        assert isinstance(timing, dict)

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
    domain_h = discretize(domain, ncells=ncells, comm=comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    # ...

    # ... discretize the equation
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
        tb = time.time()
        A_inv = inverse(A, solver='cg')
        x = A_inv.dot(b)
        te = time.time()
        timing['solution'] = te - tb

        # Store result in a new FEM field
        uh = FemField(Vh, coeffs=x)

    else:
        uh = equation_h.solve()
    # ...

    # ... compute norms
    if profile: tb = time.time()
    l2_error = l2norm_h.assemble(u=uh)
    if profile: te = time.time()
    if profile: timing['L2 error'] = te - tb

    if profile: tb = time.time()
    h1_error = h1norm_h.assemble(u=uh)
    if profile: te = time.time()
    if profile: timing['H1 error'] = te - tb
    # ...

    return l2_error, h1_error

#==============================================================================
def test_vector_poisson_2d_dir0(comm=None, backend=None, timing=None):

    from sympy import symbols
    x1, x2 = symbols('x1, x2', real=True)

    u1 = sin(pi*x1) * sin(pi*x2)
    u2 = sin(pi*x1) * sin(pi*x2)
    solution = Tuple(u1, u2)

    f1 = 2*pi**2 * sin(pi*x1) * sin(pi*x2)
    f2 = 2*pi**2 * sin(pi*x1) * sin(pi*x2)
    f = Tuple(f1, f2)

    l2_error, h1_error = run_vector_poisson_2d_dir(solution, f,
                                    ncells=[2**3, 2**3], degree=[2, 2],
                                    comm=comm, backend=backend, timing=timing)

    expected_l2_error =  0.00030842129060800865
    expected_h1_error =  0.018418110343256442

    assert abs(l2_error - expected_l2_error) < 1.e-7
    assert abs(h1_error - expected_h1_error) < 1.e-7

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()

#==============================================================================
# SCRIPT USAGE
#==============================================================================

if __name__ == "__main__":

    from mpi4py import MPI
    from psydac.api.settings import PSYDAC_BACKENDS

    params = dict(
        comm = MPI.COMM_WORLD,
        backend = PSYDAC_BACKENDS['pyccel-gcc'],
    )

    functions_to_run = (
        test_vector_poisson_2d_dir0,
    )

    print(f"> Input parameters:")
    for k, v in params.items():
        if isinstance(v, dict):
            print(f">   {k} = ")
            for kk, vv in v.items():
                print(f">     {kk} = {vv}")
        else:
            print(f">   {k} = {v}")
    print()

    print("> Functions to run:")
    for f in functions_to_run:
        print(f">   {f.__name__}")

    for f in functions_to_run:
        print()
        print(f"> Running function: {f.__name__}... ", end='')
        timing = {}
        f(**params, timing=timing)
        print(f"PASSED!")
        print(f"> Timings:")
        for k, v in timing.items():
            print(f">   {k} = {v}")
    print()
