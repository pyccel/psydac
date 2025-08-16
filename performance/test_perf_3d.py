# -*- coding: UTF-8 -*-

#from sympy import pi, cos, sin
#from sympy import S
#
#from sympde.core     import Constant
#from sympde.calculus import grad, dot, inner, cross, rot, curl, div
#
#from sympde.topology import dx, dy, dz
#from sympde.topology import ScalarField
#from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
#from sympde.topology import element_of
#from sympde.topology import Domain
#from sympde.topology import Boundary, trace_0, trace_1
#from sympde.expr     import BilinearForm, LinearForm
#from sympde.expr     import Norm
#from sympde.expr     import find, EssentialBC
#from sympde.topology import Domain, Line, Square, Cube
#
#from psydac.fem.basic   import FemField
#from psydac.fem.splines import SplineSpace
#from psydac.fem.tensor  import TensorFemSpace
#from psydac.api.discretization import discretize
#from psydac.api.settings import PSYDAC_BACKEND_PYTHON, PSYDAC_BACKEND_GPYCCEL

#from numpy import linspace, zeros
#
#import time
#from tabulate import tabulate
#from collections import namedtuple
#
#Timing = namedtuple('Timing', ['kind', 'python', 'pyccel'])
#
#domain = Cube()

#==============================================================================
#def print_timing(ls):
#    # ...
#    table   = []
#    headers = ['Assembly time', 'Python', 'Pyccel', 'Speedup']
#
#    for timing in ls:
#        speedup = timing.python / timing.pyccel
#        line   = [timing.kind, timing.python, timing.pyccel, speedup]
#        table.append(line)
#
#    print(tabulate(table, headers=headers, tablefmt='latex'))
#    # ...
#
#==============================================================================
#def test_api_stokes_3d():
#    print('============ test_api_stokes_3d =============')
#
#    # ... abstract model
#    U = VectorFunctionSpace('V', domain)
#    V = ScalarFunctionSpace('W', domain)
#
#    W = U*V
#
#    v = element_of(U, 'v')
#    u = element_of(U, 'u')
#    p = element_of(V, 'p')
#    q = element_of(V, 'q')
#
#    A = BilinearForm((v,u), inner(grad(v), grad(u)))
#    B = BilinearForm((v,p), div(v)*p)
#    a = BilinearForm(((v,q),(u,p)), A(v,u) - B(v,p) + B(u,q))
#    # ...
#
#    domain_h = discretize(domain, ncells=(2**3, 2**3, 2**3))
#    # ...
#
#    # ... discrete spaces
#    Vh = discretize(W, domain_h, degree=(3, 3, 3))
#    # ...
#
#    # ...
#    ah = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_GPYCCEL)
#    tb = time.time()
#    M_f90 = ah.assemble()
#    te = time.time()
#    print('> [pyccel] elapsed time (matrix) = ', te-tb)
#    t_f90 = te-tb
#
#    ah = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_PYTHON)
#    tb = time.time()
#    M_py = ah.assemble()
#    te = time.time()
#    print('> [python] elapsed time (matrix) = ', te-tb)
#    t_py = te-tb
#
#    matrix_timing = Timing('matrix', t_py, t_f90)
#    # ...
#
#    # ...
#    print_timing([matrix_timing])
##    print_timing([matrix_timing, rhs_timing, l2norm_timing])
#    # ...

#==============================================================================
import pytest
from psydac.api.settings import PSYDAC_BACKENDS
from utilities import print_timings_table

# IMPORT FUNCTIONS TO BE PROFILED
from psydac.api.tests.test_3d_poisson import test_3d_poisson_dir_1
from psydac.api.tests.test_3d_mapping_laplace import test_3d_laplace_neu_collela

test_functions = (
    test_3d_poisson_dir_1,
    test_3d_laplace_neu_collela,
)

#==============================================================================
@pytest.mark.parametrize('func', test_functions)
def test_compare_python_with_pyccel_gcc(func):

    print()
    print(f'Benchmarking: {func.__name__}')

    # Simple Python run
    python_timing = {}
    func(backend=None, timing=python_timing)

    # Use Pyccel with Fortran code generation, serial
    backend = PSYDAC_BACKENDS['pyccel-gcc']
    pyccel_timing = {}
    func(backend=backend, timing=pyccel_timing)

    print_timings_table(python_timing, pyccel_timing)
    print('NOTE: Pyccel = Fortran language, GFortran compiler, no OpenMP')
    print()

#==============================================================================
def test_compare_psydac_with_petsc_poisson():
    import time
    from sympy import pi, cos, sin, symbols

    from sympde.calculus import grad, dot
    from sympde.topology import ScalarFunctionSpace
    from sympde.topology import element_of
    from sympde.topology import NormalVector
    from sympde.topology import Cube
    from sympde.topology import Union
    from sympde.expr     import BilinearForm, LinearForm, integral
    from sympde.expr     import Norm, SemiNorm
    from sympde.expr     import find, EssentialBC

    from psydac.api.discretization import discretize
    from psydac.api.essential_bc   import apply_essential_bc
    from psydac.linalg.solvers     import inverse
    from psydac.fem.basic          import FemField

    from petsc4py import PETSc

    print()
    print(f'Benchmarking: Poisson 3D')

    x1, x2, x3 = symbols('x1, x2, x3', real=True)
    solution = sin(pi*x1)*sin(pi*x2)*sin(pi*x3)
    f        = 3*pi**2*sin(pi*x1)*sin(pi*x2)*sin(pi*x3)
    N = 20
    d = 3
    ncells = [N, N, N]
    degree = [d, d, d]
    backend = PSYDAC_BACKENDS['pyccel-gcc']
    backend['language'] = 'fortran'
    backend['openmp'] = True
    timing = {}
    ITER = {'solution': 10, 'matmul': 100}

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++
    domain = Cube()

    V = ScalarFunctionSpace('V', domain)
    u = element_of(V, name='u')
    v = element_of(V, name='v')

    int_0 = lambda expr: integral(domain, expr)

    expr = dot(grad(u), grad(v))
    a = BilinearForm((u, v), int_0(expr))

    expr = f*v
    l = LinearForm(v, int_0(expr))

    error  = u - solution
    l2norm =     Norm(error, domain, kind='l2')
    h1norm = SemiNorm(error, domain, kind='h1')

    bc = EssentialBC(u, 0, domain.boundary)
    equation = find(u, forall=v, lhs=a(u, v), rhs=l(v), bc=bc)

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    # Create computational domain from topological domain
    domain_h = discretize(domain, ncells=ncells)

    # Discrete spaces
    Vh = discretize(V, domain_h, degree=degree)

    # Discretize equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh], backend=backend)

    # Discretize error norms
    l2norm_h = discretize(l2norm, domain_h, Vh, backend=backend)
    h1norm_h = discretize(h1norm, domain_h, Vh, backend=backend)

    #+++++++++++++++++++++++++++++++
    # 3. Solution
    #+++++++++++++++++++++++++++++++

    ksp = PETSc.KSP()
    ksp.create(domain_h.comm)
    tol = 1e-11

    opts = PETSc.Options()
    opts["ksp_type"] = "minres"
    opts["pc_type"] = "none"
    opts["ksp_rtol"] = tol
    opts["ksp_atol"] = tol
    ksp.setFromOptions()

    # Assemble matrix corresponding to discrete bilinear form
    tb = time.time()
    A  = equation_h.lhs.assemble()
    te = time.time()
    timing['matrix'] = te - tb

    # Assemble vector corresponding to discrete linear form
    tb = time.time()
    b  = equation_h.rhs.assemble()
    te = time.time()
    timing['rhs'] = te - tb

    # Apply essential BCs to A and b
    apply_essential_bc(A, *equation_h.bc)
    apply_essential_bc(b, *equation_h.bc)

    A_petsc = A.topetsc()
    b_petsc = b.topetsc()

    # Solve linear system
    tb = time.time()
    for _ in range(ITER['solution']):
        A_inv = inverse(A, solver='cg')
        x = A_inv.dot(b)
    te = time.time()
    timing['solution'] = (te - tb) / ITER['solution']

    # Solve linear system
    tb = time.time()
    for _ in range(ITER['solution']):
        ksp.setOperators(A_petsc)
        x_petsc = b_petsc.duplicate()
        ksp.solve(b_petsc, x_petsc)
    te = time.time()
    timing['solution_petsc'] = (te - tb) / ITER['solution']

    print('Solution with petsc4py: success = {}'.format(ksp.is_converged))

    # Store result in a new FEM field
    uh = FemField(Vh, coeffs=x)

    # Compute error norms
    tb = time.time()
    l2_error = l2norm_h.assemble(u=uh)
    te = time.time()
    timing['L2 error'] = te - tb

    tb = time.time()
    h1_error = h1norm_h.assemble(u=uh)
    te = time.time()
    timing['H1 error'] = te - tb

    print("Errors : ", l2_error, h1_error)

    # Time matrix-vector product
    r = b.space.zeros()
    r_petsc = r.topetsc()

    tb = time.time()
    for _ in range(ITER['matmul']):
        A.dot(x, out=r)
    te = time.time()
    timing['matmul'] = (te - tb) / ITER['matmul']

    tb = time.time()
    for _ in range(ITER['matmul']):
        A_petsc.mult(x_petsc, r_petsc)
    te = time.time()
    timing['matmul_petsc'] = (te - tb) / ITER['matmul']

    print()
    print("Comparing PSYDAC with PETSc. Time to solve matrix equation:")
    print("   PSYDAC  | PETSc")
    print(f" {timing['solution']:.3e} | {timing['solution_petsc']:.3e}")
    print("Comparing PSYDAC with PETSc. Time for matrix-vector product:")
    print("   PSYDAC  | PETSc")
    print(f" {timing['matmul']:.3e} | {timing['matmul_petsc']:.3e}")

#==============================================================================
# SCRIPT USAGE
#==============================================================================
if __name__ == '__main__':

    for func in test_functions:
        test_compare_python_with_pyccel_gcc(func)

    test_compare_psydac_with_petsc_poisson()

#    test_api_stokes_3d()
