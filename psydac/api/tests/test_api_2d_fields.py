# -*- coding: UTF-8 -*-
#
# A note on the mappings used in these tests:
#
#   - 'identity_2d.h5' is the identity mapping on the unit square [0, 1] X [0, 1]
#
#   - 'collela_2d.h5' is a NURBS mapping from the unit square [0, 1]^2 to the
#      larger square [-1, 1]^2, with deformations going as sin(pi x) * sin(pi y)
#
#   - 'quarter_annulus.h5' is a NURBS transformation from the unit square [0, 1]^2
#      to the quarter annulus in the lower-left quadrant of the Cartesian plane
#      (hence both x and y are negative), with r_min = 0.5 and r_max = 1
#
#      Please note that the logical coordinates (x1, x2) correspond to the polar
#      coordinates (r, theta), but with reversed order: hence x1=theta and x2=r

from mpi4py import MPI
from sympy import pi, cos, sin
from sympy.abc import x, y
import pytest
import os

from sympde.calculus import grad, dot
from sympde.calculus import laplace
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of
from sympde.topology import NormalVector
from sympde.topology import Domain
from sympde.topology import Union
from sympde.expr import BilinearForm, LinearForm, integral
from sympde.expr import Norm
from sympde.expr import find, EssentialBC

from psydac.fem.basic          import FemField
from psydac.api.discretization import discretize

# ... get the mesh directory
try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']

except:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')
# ...

#+++++++++++++++++++++++++++++++
# 1. Abstract model
#+++++++++++++++++++++++++++++++
def run_poisson_2d(filename, solution, f):
    domain = Domain.from_file(filename)

    B_dirichlet_0 = domain.boundary

    V  = ScalarFunctionSpace('V', domain)
    u  = element_of(V, name='u')
    v  = element_of(V, name='v')
    F  = element_of(V, name='F')

    # Bilinear form a: V x V --> R
    a = BilinearForm((u, v), integral(domain, dot(grad(u), grad(v))))

    # Linear form l: V --> R
    l = LinearForm(v, integral(domain, f * v))

    # Dirichlet boundary conditions
    bc = [EssentialBC(u,  0, B_dirichlet_0)]


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
    domain_h = discretize(domain, filename=filename)

    # Discrete spaces
    Vh = discretize(V, domain_h)

    # Discretize equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh])

    # Discretize error norms
    l2norm_h = discretize(l2norm, domain_h, Vh)
    h1norm_h = discretize(h1norm, domain_h, Vh)

    #+++++++++++++++++++++++++++++++
    # 3. Solution
    #+++++++++++++++++++++++++++++++

    # Solve linear system
    x  = equation_h.solve()
    uh = FemField( Vh, x )

    #+++++++++++++++++++++++++++++++
    l1   = LinearForm( v, integral(domain, F*v))
    l2   = LinearForm( v, integral(domain, solution*v))
    l1_h = discretize(l1, domain_h,  Vh)
    l2_h = discretize(l2, domain_h,  Vh)

    a1   = BilinearForm( (u,v), integral(domain, F*u*v))
    a2   = BilinearForm( (u,v), integral(domain, solution*u*v))
    a1_h = discretize(a1, domain_h,  [Vh, Vh])
    a2_h = discretize(a2, domain_h,  [Vh, Vh])
              
    x1 = l1_h.assemble(F=uh)
    x2 = l2_h.assemble()

    A1 = a1_h.assemble(F=uh)
    A2 = a2_h.assemble()

    error_1 = abs((x1-x2).toarray()).max()
    error_2 = abs((A1-A2).toarray()).max()
    return error_1, error_2

###############################################################################
#            SERIAL TESTS
###############################################################################

def test_poisson_2d_identity_1_dir0_1234():

    filename = os.path.join(mesh_dir, 'identity_2d.h5')
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)

    error_1, error_2 = run_poisson_2d(filename, solution, f)

    expected_error_1 =  1.2902405843379702e-06
    expected_error_2 =  5.691117428555293e-07

    assert( abs(error_1 - expected_error_1) < 1.e-7)
    assert( abs(error_2 - expected_error_2) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_identity_2_dir0_1234():

    filename = os.path.join(mesh_dir, 'identity_2d.h5')
    solution = x*y*(x-1)*(y-1)
    f        = -(solution.diff(x,2) + solution.diff(y,2))

    error_1, error_2 = run_poisson_2d(filename, solution, f)

    expected_error_1 =  8.784051502667978e-14
    expected_error_2 =  4.066666166140098e-14

    assert( abs(error_1 - expected_error_1) < 1.e-14)
    assert( abs(error_2 - expected_error_2) < 1.e-14)
#------------------------------------------------------------------------------
def test_poisson_2d_collela_dir0_1234():

    filename = os.path.join(mesh_dir, 'collela_2d.h5')
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)

    error_1, error_2 = run_poisson_2d(filename, solution, f)

    expected_error_1 =  0.0007343500640612094
    expected_error_2 =  0.00022901839284597547

    assert( abs(error_1 - expected_error_1) < 1.e-7)
    assert( abs(error_2 - expected_error_2) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_quarter_annulus_dir0_1234():

    filename = os.path.join(mesh_dir, 'quarter_annulus.h5')
    c        = pi / (1. - 0.5**2)
    r2       = 1. - x**2 - y**2
    solution = x*y*sin(c * r2)
    f        = 4.*c**2*x*y*(x**2 + y**2)*sin(c * r2) + 12.*c*x*y*cos(c * r2)

    error_1, error_2 = run_poisson_2d(filename, solution, f)

    expected_error_1 =  7.954918451356864e-07
    expected_error_2 =  3.3779301725050655e-07

    assert( abs(error_1 - expected_error_1) < 1.e-7)
    assert( abs(error_2 - expected_error_2) < 1.e-7)

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()

