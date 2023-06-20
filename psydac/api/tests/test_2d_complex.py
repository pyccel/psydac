# -*- coding: UTF-8 -*-

from mpi4py import MPI
from sympy import pi, cos, sin, symbols, conjugate, exp
from sympy.utilities.lambdify import implemented_function
import pytest

from sympde.calculus import grad, dot
from sympde.calculus      import minus, plus
from sympde.calculus import laplace
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of
from sympde.topology import NormalVector
from sympde.topology import Union
from sympde.topology import Domain, Square
from sympde.topology      import IdentityMapping, AffineMapping
from sympde.expr     import BilinearForm, LinearForm, integral, SesquilinearForm
from sympde.expr     import Norm
from sympde.expr     import find, EssentialBC

from psydac.api.discretization import discretize
from psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL
import os

# ... get the mesh directory
try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']

except:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')


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

#==============================================================================
def run_biharmonic_2d_dir(solution, f, dir_zero_boundary, ncells=None, degree=None, backend=None, comm=None, filename=None):

    assert isinstance(dir_zero_boundary, (list, tuple))

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++
    if filename:
        domain = Domain.from_file(filename)
    else:
        domain = Square()

    B_dirichlet_0 = Union(*[domain.get_boundary(**kw) for kw in dir_zero_boundary])

    V  = ScalarFunctionSpace('V', domain, codomain_complex=True)
    u  = element_of(V, name='u')
    v  = element_of(V, name='v')
    nn = NormalVector('nn')

    # Bilinear form a: V x V --> R
    a = SesquilinearForm((u, v), integral(domain, dot(laplace(u), laplace(v))))

    # Linear form l: V --> R
    l = LinearForm(v, integral(domain, f * v))

    # Essential boundary conditions
    dn = lambda a: dot(grad(a), nn)
    bc = []
    bc += [EssentialBC(   u , 0, B_dirichlet_0)]
    bc += [EssentialBC(dn(u), 0, B_dirichlet_0)]

    # Variational model
    equation = find(u, forall=v, lhs=a(u, v), rhs=l(v), bc=bc)

    # Error norms
    error  = u - solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')
    h2norm = Norm(error, domain, kind='h2')

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    # Create computational domain from topological domain
    if filename:
        domain_h = discretize(domain, filename=filename, comm=comm)
    else:
        domain_h = discretize(domain, ncells=ncells, comm=comm)

    # Discrete spaces
    if filename:
        Vh = discretize(V, domain_h)
    else:
        Vh = discretize(V, domain_h, degree=degree)

    # Discretize equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh], backend=backend)

    # Discretize error norms
    l2norm_h = discretize(l2norm, domain_h, Vh, backend=backend)
    h1norm_h = discretize(h1norm, domain_h, Vh, backend=backend)
    h2norm_h = discretize(h2norm, domain_h, Vh, backend=backend)

    #+++++++++++++++++++++++++++++++
    # 3. Solution
    #+++++++++++++++++++++++++++++++

    # Solve linear system
    equation_h.set_solver('bicg', tol=1e-9)
    uh = equation_h.solve()

    # Compute error norms
    l2_error = l2norm_h.assemble(u=uh)
    h1_error = h1norm_h.assemble(u=uh)
    h2_error = h2norm_h.assemble(u=uh)

    return l2_error, h1_error, h2_error

#==============================================================================
def run_poisson_2d(solution, f, domain, ncells=None, degree=None, filename=None, backend=None):

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    V   = ScalarFunctionSpace('V', domain, kind=None, codomain_complex=True)

    u = element_of(V, name='u')
    v = element_of(V, name='v')

    nn   = NormalVector('nn')

    bc   = EssentialBC(u, 0, domain.boundary)

    error  = u - solution

    I = domain.interfaces

    kappa  = 10**3

    #expr_I =- dot(grad(plus(u)),nn)*minus(v)  + dot(grad(minus(v)),nn)*plus(u) - kappa*plus(u)*minus(v)\
    #        + dot(grad(minus(u)),nn)*plus(v)  - dot(grad(plus(v)),nn)*minus(u) - kappa*plus(v)*minus(u)\
    #        - dot(grad(plus(v)),nn)*plus(u)   + kappa*plus(u)*plus(v)\
    #        - dot(grad(minus(v)),nn)*minus(u) + kappa*minus(u)*minus(v)

    expr_I =- 0.5*dot(grad(plus(u)),nn)*minus(v)  + 0.5*dot(grad(minus(v)),nn)*plus(u)  - kappa*plus(u)*minus(v)\
            + 0.5*dot(grad(minus(u)),nn)*plus(v)  - 0.5*dot(grad(plus(v)),nn)*minus(u)  - kappa*plus(v)*minus(u)\
            - 0.5*dot(grad(minus(v)),nn)*minus(u) - 0.5*dot(grad(minus(u)),nn)*minus(v) + kappa*minus(u)*minus(v)\
            + 0.5*dot(grad(plus(v)),nn)*plus(u)   + 0.5*dot(grad(plus(u)),nn)*plus(v)   + kappa*plus(u)*plus(v)

    expr   = dot(grad(u),grad(v))

    a = SesquilinearForm((u,v), integral(domain, expr) + integral(I, expr_I))
    l = LinearForm(v, integral(domain, dot(f,v)))

    equation = find(u, forall=v, lhs=1j*a(u,v), rhs=1j*l(v), bc=bc)

    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    if filename is None:
        domain_h = discretize(domain, ncells=ncells)
        Vh       = discretize(V, domain_h, degree=degree)
    else:
        domain_h = discretize(domain, filename=filename)
        Vh       = discretize(V, domain_h)

    equation_h = discretize(equation, domain_h, [Vh, Vh], backend=backend)

    l2norm_h = discretize(l2norm, domain_h, Vh, backend=backend)
    h1norm_h = discretize(h1norm, domain_h, Vh, backend=backend)

    equation_h.set_solver('bicg')
    uh = equation_h.solve()

    l2_error = l2norm_h.assemble(u=uh)
    h1_error = h1norm_h.assemble(u=uh)

    return l2_error, h1_error

###############################################################################
#            SERIAL TESTS
###############################################################################
def test_complex_biharmonic_2d():
    x, y, z = symbols('x1, x2, x3')
    solution = (sin(pi * x)**2 * sin(pi * y)**2 + 1j * sin(2*pi * x)**2 * sin(2*pi * y)**2) * exp(pi * 1j * (x**2+y**2))
    f        = laplace(laplace(solution))

    dir_zero_boundary = get_boundaries(1, 2, 3, 4)

    l2_error, h1_error, h2_error = run_biharmonic_2d_dir(solution, f,
            dir_zero_boundary, ncells=[2**3, 2**3], degree=[3, 3])

    expected_l2_error = 0.0027365784556742626
    expected_h1_error = 0.07976499145119309
    expected_h2_error = 1.701552032688161

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    assert( abs(h2_error - expected_h2_error) < 1.e-7)

def test_complex_biharmonic_2d_mapping():

    x, y, z = symbols('x, y, z')
    filename = os.path.join(mesh_dir, 'collela_2d.h5')

    factor=2.5
    solution = factor * (cos(1) + sin(1) * 1j) * (cos(pi*x/2)*cos(pi*y/2))**2
    f        = laplace(laplace(solution))

    dir_zero_boundary    = get_boundaries(1, 2, 3, 4)

    l2_error, h1_error, h2_error = run_biharmonic_2d_dir(solution, f,
            dir_zero_boundary, filename=filename)

    expected_l2_error = 0.10977627980052021
    expected_h1_error = 0.32254511059711766
    expected_h2_error = 1.87205519824758

    assert( abs(l2_error/factor - expected_l2_error) < 1.e-7)
    assert( abs(h1_error/factor - expected_h1_error) < 1.e-7)
    assert( abs(h2_error/factor - expected_h2_error) < 1.e-7)

def test_complex_poisson_2d_multipatch():
    A = Square('A',bounds1=(0, 0.5), bounds2=(0, 1))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(0, 1))

    domain = A.join(B, name = 'domain',
                bnd_minus = A.get_boundary(axis=0, ext=1),
                bnd_plus  = B.get_boundary(axis=0, ext=-1))

    x,y = domain.coordinates

    solution = (cos(1) + sin(1) * 1j) * x*y*(1-y)*(1-x)
    f        = (-2*x*(x - 1) -2*y*(y - 1))*(cos(1) + sin(1) * 1j)

    l2_error, h1_error = run_poisson_2d(solution, f, domain, ncells=[2**2,2**2], degree=[2,2])

    expected_l2_error = 2.176726763610992e-09
    expected_h1_error = 2.9725703533101877e-09

    assert ( abs(l2_error - expected_l2_error) < 1e-7 )
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

def test_complex_poisson_2d_multipatch_mapping ():

    filename = os.path.join(mesh_dir, 'multipatch/square.h5')
    domain   = Domain.from_file(filename)

    x,y = domain.coordinates
    solution = (cos(1) + sin(1) * 1j) * x*y*(1-y)*(1-x)
    f        = (-2*x*(x - 1) -2*y*(y - 1))*(cos(1) + sin(1) * 1j)

    l2_error, h1_error = run_poisson_2d(solution, f, domain, filename=filename, backend=PSYDAC_BACKEND_GPYCCEL)

    expected_l2_error = 2.176726763610992e-09
    expected_h1_error = 2.9725703533101877e-09

    assert ( abs(l2_error - expected_l2_error) < 1e-7 )
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )
