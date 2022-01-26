# -*- coding: UTF-8 -*-

import os
import pytest

from mpi4py import MPI
from sympy import pi, sin, cos, symbols

from sympde.calculus import grad, dot
from sympde.calculus import laplace
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of
from sympde.topology import NormalVector
from sympde.topology import Square, Domain
from sympde.topology import Union
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm
from sympde.expr     import find, EssentialBC

from psydac.api.discretization import discretize
from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL

#comm = MPI.COMM_WORLD
PSYDAC_BACKEND_GPYCCEL = PSYDAC_BACKEND_GPYCCEL.copy()
PSYDAC_BACKEND_GPYCCEL['openmp'] = True
os.environ['OMP_NUM_THREADS']    = "2"

#==============================================================================
try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']

except:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')
# ...

#==============================================================================
def get_boundaries(*args):

    if not args:
        return ()
    else:
        assert all(1 <= a <= 4 for a in args)
        assert len(set(args)) == len(args)

    boundaries = {1: {'axis': 0, 'ext': -1},
                  2: {'axis': 0, 'ext':  1},
                  3: {'axis': 1, 'ext': -1},
                  4: {'axis': 1, 'ext':  1}}

    return tuple(boundaries[i] for i in args)

x,y,z = symbols('x, y, z')
#==============================================================================
def run_poisson_2d_with_mapping(filename, solution, f, dir_zero_boundary,
        dir_nonzero_boundary, neumann_boundary, comm=MPI.COMM_WORLD):

    assert isinstance(   dir_zero_boundary, (list, tuple))
    assert isinstance(dir_nonzero_boundary, (list, tuple))
    assert isinstance(    neumann_boundary, (list, tuple))

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++
    domain = Domain.from_file(filename)

    B_dirichlet_0 = Union(*[domain.get_boundary(**kw) for kw in dir_zero_boundary])
    B_dirichlet_i = Union(*[domain.get_boundary(**kw) for kw in dir_nonzero_boundary])
    B_neumann     = Union(*[domain.get_boundary(**kw) for kw in neumann_boundary])

    V  = ScalarFunctionSpace('V', domain)
    u  = element_of(V, name='u')
    v  = element_of(V, name='v')
    nn = NormalVector('nn')

    # Bilinear form a: V x V --> R
    a = BilinearForm((u, v), integral(domain, dot(grad(u),grad(v))))

    # Linear form l: V --> R
    l0 = LinearForm(v, integral(domain, f * v))
    if B_neumann:
        l1 = LinearForm(v, integral(B_neumann, v * dot(grad(solution), nn)))
        l  = LinearForm(v, l0(v) + l1(v))
    else:
        l = l0

    # Dirichlet boundary conditions
    bc = []
    if B_dirichlet_0:  bc += [EssentialBC(u,        0, B_dirichlet_0)]
    if B_dirichlet_i:  bc += [EssentialBC(u, solution, B_dirichlet_i)]

    # Variational model
    equation = find(u, forall=v, lhs=a(u, v), rhs=l(v), bc=bc)

    # Error norms
    error  = u - solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    # Create computational domain from topological domain
    domain_h = discretize(domain, filename=filename, comm=comm)

    # Discrete spaces
    Vh = discretize(V, domain_h)

    # Discretize equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_GPYCCEL)

    # Discretize error norms
    l2norm_h = discretize(l2norm, domain_h, Vh, backend=PSYDAC_BACKEND_GPYCCEL)
    h1norm_h = discretize(h1norm, domain_h, Vh, backend=PSYDAC_BACKEND_GPYCCEL)

    #+++++++++++++++++++++++++++++++
    # 3. Solution
    #+++++++++++++++++++++++++++++++

    # Solve linear system
    uh = equation_h.solve()


    # Compute error norms
    l2_error = l2norm_h.assemble(u=uh)
    h1_error = h1norm_h.assemble(u=uh)
    
    return l2_error, h1_error

#==============================================================================
def run_poisson_2d(solution, f, dir_zero_boundary, dir_nonzero_boundary,
        ncells, degree):

    assert isinstance(dir_zero_boundary   , (list, tuple))
    assert isinstance(dir_nonzero_boundary, (list, tuple))

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++
    domain = Square()

    B_dirichlet_0 = Union(*[domain.get_boundary(**kw) for kw in dir_zero_boundary])
    B_dirichlet_i = Union(*[domain.get_boundary(**kw) for kw in dir_nonzero_boundary])
    B_dirichlet   = Union(B_dirichlet_0, B_dirichlet_i)
    B_neumann = domain.boundary.complement(B_dirichlet)

    V  = ScalarFunctionSpace('V', domain)
    u  = element_of(V, name='u')
    v  = element_of(V, name='v')
    nn = NormalVector('nn')

    # Bilinear form a: V x V --> R
    a = BilinearForm((u, v), integral(domain, dot(grad(u), grad(v))))

    # Linear form l: V --> R
    l0 = LinearForm(v, integral(domain, f * v))
    if B_neumann:
        l1 = LinearForm(v, integral(B_neumann, v * dot(grad(solution), nn)))
        l  = LinearForm(v, l0(v) + l1(v))
    else:
        l = l0

    # Dirichlet boundary conditions
    bc = []
    if B_dirichlet_0:  bc += [EssentialBC(u,        0, B_dirichlet_0)]
    if B_dirichlet_i:  bc += [EssentialBC(u, solution, B_dirichlet_i)]

    # Variational model
    equation = find(u, forall=v, lhs=a(u, v), rhs=l(v), bc=bc)

    # Error norms
    error  = u - solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    # Create computational domain from topological domain
    domain_h = discretize(domain, ncells=ncells, comm=comm)

    # Discrete spaces
    Vh = discretize(V, domain_h, degree=degree)

    # Discretize equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_GPYCCEL)

    # Discretize error norms
    l2norm_h = discretize(l2norm, domain_h, Vh, backend=PSYDAC_BACKEND_GPYCCEL)
    h1norm_h = discretize(h1norm, domain_h, Vh, backend=PSYDAC_BACKEND_GPYCCEL)

    #+++++++++++++++++++++++++++++++
    # 3. Solution
    #+++++++++++++++++++++++++++++++

    # Solve linear system
    uh = equation_h.solve()

    # Compute error norms
    l2_error = l2norm_h.assemble(u=uh)
    h1_error = h1norm_h.assemble(u=uh)

    return l2_error, h1_error

###############################################################################
#            SERIAL TESTS
###############################################################################
def test_poisson_2d_dir0_1234():

    x,y,z = symbols('x1, x2, x3')

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)

    dir_zero_boundary    = get_boundaries(1, 2, 3, 4)
    dir_nonzero_boundary = get_boundaries()

    l2_error, h1_error = run_poisson_2d(solution, f, dir_zero_boundary,
            dir_nonzero_boundary, ncells=[2**3, 2**3], degree=[2, 2])

    expected_l2_error =  0.00021808678604760232
    expected_h1_error =  0.013023570720360362

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_collela_dir0_123_neui_4():

    filename = os.path.join(mesh_dir, 'collela_2d.h5')
    solution = cos(pi/2 * x) * sin(pi/3 * (1 + y))
    f        = (13/36)*pi**2 * solution

    dir_zero_boundary    = get_boundaries(1, 2, 3)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries(4)

    l2_error, h1_error = run_poisson_2d_with_mapping(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error = 0.0026061796931066174
    expected_h1_error = 0.04400143055955377

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_collela_dir0_13_diri_24():

    filename = os.path.join(mesh_dir, 'collela_2d.h5')
    solution = sin(pi/3 * (1 + x)) * sin(pi/3 * (1 + y))
    f        = (2/9)*pi**2 * solution

    dir_zero_boundary    = get_boundaries(1, 3)
    dir_nonzero_boundary = get_boundaries(2, 4)
    neumann_boundary     = get_boundaries()

    l2_error, h1_error = run_poisson_2d_with_mapping(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error = 0.0012801077606328381
    expected_h1_error = 0.02314405549486328

    assert abs(l2_error - expected_l2_error) < 1.e-7
    assert abs(h1_error - expected_h1_error) < 1.e-7

#------------------------------------------------------------------------------
def test_poisson_2d_quarter_annulus_dir0_12_diri_34():

    filename = os.path.join(mesh_dir, 'quarter_annulus.h5')
    solution = sin(pi * x) * sin(pi * y)
    f        = 2*pi**2 * solution

    dir_zero_boundary    = get_boundaries(1, 2)
    dir_nonzero_boundary = get_boundaries(3, 4)
    neumann_boundary     = get_boundaries()

    l2_error, h1_error = run_poisson_2d_with_mapping(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error = 0.0005982761090480573
    expected_h1_error = 0.021271053089631443

    assert abs(l2_error - expected_l2_error) < 1.e-7
    assert abs(h1_error - expected_h1_error) < 1.e-7

#------------------------------------------------------------------------------
def test_poisson_2d_quarter_annulus_diri_34_neui_12():

    filename = os.path.join(mesh_dir, 'quarter_annulus.h5')
    solution = sin(pi*x + pi/4) * sin(pi*y + pi/4)
    f        = 2*pi**2 * solution

    dir_zero_boundary    = get_boundaries()
    dir_nonzero_boundary = get_boundaries(3, 4)
    neumann_boundary     = get_boundaries(1, 2)

    l2_error, h1_error = run_poisson_2d_with_mapping(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error = 0.0006527836834289991
    expected_h1_error = 0.025919435390680808

    assert abs(l2_error - expected_l2_error) < 1.e-7
    assert abs(h1_error - expected_h1_error) < 1.e-7

#------------------------------------------------------------------------------
def test_poisson_2d_circle_dir0():

    filename = os.path.join(mesh_dir, 'circle.h5')
    solution = (1 - (x**2 + y**2)) * cos(2*pi*x) * cos(2*pi*y)
    f        = -laplace(solution)

    dir_zero_boundary    = get_boundaries(2) # only boundary is at r = r_max
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries()

    l2_error, h1_error = run_poisson_2d_with_mapping(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error = 0.0015245737751297718
    expected_h1_error = 0.06653900724243668

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_pipe_dir_1234():

    filename = os.path.join(mesh_dir, 'pipe.h5')
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)

    dir_zero_boundary    = get_boundaries()
    dir_nonzero_boundary = get_boundaries(1, 2, 3, 4)
    neumann_boundary     = get_boundaries()

    l2_error, h1_error = run_poisson_2d_with_mapping(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error =  0.0008629074796755705
    expected_h1_error =  0.038151393401512884
 

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

###############################################################################
#            PARALLEL TESTS
###############################################################################
@pytest.mark.parallel
def test_poisson_2d_dir0_1234_parallel():

    x,y,z = symbols('x1, x2, x3')

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)

    dir_zero_boundary    = get_boundaries(1, 2, 3, 4)
    dir_nonzero_boundary = get_boundaries()

    l2_error, h1_error = run_poisson_2d(solution, f, dir_zero_boundary,
            dir_nonzero_boundary, ncells=[2**4, 2**4], degree=[2, 2])

    expected_l2_error =  2.6130834310749216e-05
    expected_h1_error =  0.00320767625406208

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

def delete_env():
    del os.environ['OMP_NUM_THREADS']
