#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#

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

import os
from pathlib import Path

from mpi4py import MPI
from sympy import pi, cos, sin, symbols
import pytest
import numpy as np

from sympde.calculus import grad, dot
from sympde.calculus import laplace
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of
from sympde.topology import NormalVector
from sympde.topology import Domain,Square
from sympde.topology import Union
from sympde.expr import BilinearForm, LinearForm, integral
from sympde.expr import Norm, SemiNorm
from sympde.expr import find, EssentialBC

from psydac.api.discretization import discretize

# Get the mesh directory
import psydac.cad.mesh as mesh_mod
mesh_dir = Path(mesh_mod.__file__).parent

x,y = symbols('x,y', real=True)

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
def run_laplace_2d(filename, solution, f, dir_zero_boundary,
        dir_nonzero_boundary, neumann_boundary, backend=None, comm=None):

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
    a = BilinearForm((u, v), integral(domain, dot(grad(u), grad(v)) + u * v))

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
    l2norm =     Norm(error, domain, kind='l2')
    h1norm = SemiNorm(error, domain, kind='h1')

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    # Create computational domain from topological domain
    domain_h = discretize(domain, filename=filename, comm=comm)

    # Discrete spaces
    Vh = discretize(V, domain_h)

    # Discretize equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh], backend=backend)

    # Discretize error norms
    l2norm_h = discretize(l2norm, domain_h, Vh, backend=backend)
    h1norm_h = discretize(h1norm, domain_h, Vh, backend=backend)

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

def test_laplace_2d_identity_neu0_1234():

    filename = os.path.join(mesh_dir, 'identity_2d.h5')
    solution = cos(pi*x)*cos(pi*y)
    f        = (2.*pi**2 + 1.)*solution

    dir_zero_boundary    = get_boundaries()
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries(1, 2, 3, 4)

    l2_error, h1_error = run_laplace_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error =  0.00021728465388208586
    expected_h1_error =  0.012984852988123631

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_laplace_2d_collela_neu0_1234():

    filename = os.path.join(mesh_dir, 'collela_2d.h5')
    solution = cos(pi*x)*cos(pi*y)
    f        = (2.*pi**2 + 1.)*solution

    dir_zero_boundary    = get_boundaries()
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries(1, 2, 3, 4)

    l2_error, h1_error = run_laplace_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error =  0.029603335241478155
    expected_h1_error =  0.4067746760978581

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
