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

def run_field_test(filename, f):

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    domain = Domain.from_file(filename)

    V  = ScalarFunctionSpace('V', domain)
    u  = element_of(V, name='u')
    v  = element_of(V, name='v')
    F  = element_of(V, name='F')

    # Bilinear form a: V x V --> R
    a = BilinearForm((u, v), integral(domain, u * v))

    # Linear form l: V --> R
    l = LinearForm(v, integral(domain, f * v))

    # Variational model
    equation = find(u, forall=v, lhs=a(u, v), rhs=l(v))

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    # Create computational domain from topological domain
    domain_h = discretize(domain, filename=filename)

    # Discrete spaces
    Vh = discretize(V, domain_h)

    # Discretize equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh])

    #+++++++++++++++++++++++++++++++
    # 3. Solution
    #+++++++++++++++++++++++++++++++

    # Solve linear system
    # uh is the L2-projection of the analytical field "f"

    x  = equation_h.solve()
    uh = FemField( Vh, x )

    #+++++++++++++++++++++++++++++++
    l1   = LinearForm( v, integral(domain, F*v))
    l2   = LinearForm( v, integral(domain, f*v))
    l1_h = discretize(l1, domain_h,  Vh)
    l2_h = discretize(l2, domain_h,  Vh)

    a1   = BilinearForm( (u,v), integral(domain, F*u*v))
    a2   = BilinearForm( (u,v), integral(domain, f*u*v))
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
    f        = sin(pi*x)*sin(pi*y)

    error_1, error_2 = run_field_test(filename, f)

    expected_error_1 =  4.77987181085604e-12
    expected_error_2 =  1.196388887893425e-07

    assert( abs(error_1 - expected_error_1) < 1.e-7)
    assert( abs(error_2 - expected_error_2) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_identity_2_dir0_1234():

    filename = os.path.join(mesh_dir, 'identity_2d.h5')
    f        = x*y*(x-1)*(y-1)

    error_1, error_2 = run_field_test(filename, f)

    expected_error_1 =  5.428295909559039e-11
    expected_error_2 =  2.9890068935570224e-11

    assert( abs(error_1 - expected_error_1) < 1.e-10)
    assert( abs(error_2 - expected_error_2) < 1.e-10)
#------------------------------------------------------------------------------
def test_poisson_2d_collela_dir0_1234():

    filename = os.path.join(mesh_dir, 'collela_2d.h5')
    f        = sin(pi*x)*sin(pi*y)

    error_1, error_2 = run_field_test(filename, f)

    expected_error_1 =  1.9180860719170134e-10
    expected_error_2 =  0.00010748308338081464

    assert( abs(error_1 - expected_error_1) < 1.e-7)
    assert( abs(error_2 - expected_error_2) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_quarter_annulus_dir0_1234():

    filename = os.path.join(mesh_dir, 'quarter_annulus.h5')
    c        = pi / (1. - 0.5**2)
    r2       = 1. - x**2 - y**2
    f        = x*y*sin(c * r2)

    error_1, error_2 = run_field_test(filename, f)

    expected_error_1 =  1.1146377538410329e-10
    expected_error_2 =  9.18920469410037e-08

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

