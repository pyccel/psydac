import os
import time

from sympy import pi, cos, symbols

from sympde.calculus import grad, dot
from sympde.topology import ScalarFunctionSpace
from sympde.topology import elements_of
from sympde.topology import NormalVector
from sympde.topology import Domain
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm, SemiNorm
from sympde.expr     import find

from psydac.api.discretization import discretize
from psydac.linalg.solvers     import inverse
from psydac.fem.basic          import FemField

# ... get the mesh directory
try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']
except KeyError:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')
# ...

#==============================================================================
def run_laplace_3d_neu(filename, solution, f, *,
                       backend=None, timing=None, comm=None):

    # The dictionary for the timings is modified in-place
    profile = (timing is not None)
    if profile:
        assert isinstance(timing, dict)

    # ... abstract model
    domain = Domain.from_file(filename)

    V = ScalarFunctionSpace('V', domain)

    u, v = elements_of(V, names='u, v')

    B_neumann = domain.boundary
    nn = NormalVector('nn')

    int_0 = lambda expr: integral(domain, expr)
    int_1 = lambda expr: integral(B_neumann, expr)

    expr = dot(grad(v), grad(u)) + v * u
    a = BilinearForm((v,u), int_0(expr))

    expr = f * v
    l0 = LinearForm(v, int_0(expr))

    expr = v * dot(grad(solution), nn)
    l_B_neumann = LinearForm(v, int_1(expr))

    expr = l0(v) + l_B_neumann(v)
    l = LinearForm(v, expr)

    error  = u - solution
    l2norm =     Norm(error, domain, kind='l2')
    h1norm = SemiNorm(error, domain, kind='h1')

    equation = find(u, forall=v, lhs=a(u, v), rhs=l(v))
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, filename=filename, comm=comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h)
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
        # Assemble matrix
        tb = time.time()
        A  = equation_h.lhs.assemble()
        te = time.time()
        timing['matrix'] = te - tb

        # Assemble right-hand-side vector
        tb = time.time()
        b  = equation_h.rhs.assemble()
        te = time.time()
        timing['rhs'] = te - tb

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

###############################################################################
#            SERIAL TESTS
###############################################################################

def test_3d_laplace_neu_identity(comm=None, backend=None, timing=None):

    filename = os.path.join(mesh_dir, 'identity_3d.h5')

    x, y, z = symbols('x,y,z', real=True)

    solution = cos(pi*x) * cos(pi*y) * cos(pi*z)
    f        = (3.*pi**2 + 1.) * solution

    l2_error, h1_error = run_laplace_3d_neu(filename, solution, f,
                                     backend=backend, timing=timing, comm=comm)

    expected_l2_error =  0.0016975430150953524
    expected_h1_error =  0.047009063231215

    assert abs(l2_error - expected_l2_error) < 1.e-7
    assert abs(h1_error - expected_h1_error) < 1.e-7

#==============================================================================
def test_3d_laplace_neu_collela(comm=None, backend=None, timing=None):

    filename = os.path.join(mesh_dir, 'collela_3d.h5')

    x, y, z = symbols('x, y, z', real=True)

    solution = cos(pi*x) * cos(pi*y) * cos(pi*z)
    f        = (3.*pi**2 + 1.) * solution

    l2_error, h1_error = run_laplace_3d_neu(filename, solution, f,
                                   backend=backend, timing=timing, comm=comm)

    expected_l2_error =  0.1768000505351402
    expected_h1_error =  1.7036022067226382

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
        test_3d_laplace_neu_identity,
        test_3d_laplace_neu_collela,
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
