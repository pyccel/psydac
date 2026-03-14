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
from sympde.topology import Domain
from sympde.topology import Union
from sympde.expr import BilinearForm, LinearForm, integral
from sympde.expr import Norm, SemiNorm
from sympde.expr import find, EssentialBC

from psydac.api.discretization import discretize
from psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL

# Get the mesh directory
import psydac.cad.mesh as mesh_mod
mesh_dir = Path(mesh_mod.__file__).parent

x, y = symbols('x, y', real=True)

# backend to activate multi threading
PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP           = PSYDAC_BACKEND_GPYCCEL.copy()
PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP['openmp'] = True
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
def run_poisson_2d(filename, solution, f, dir_zero_boundary,
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
    np.set_printoptions(precision=3, linewidth=200)
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

#==============================================================================
# 2D Poisson's equation with identity map
#==============================================================================
def test_poisson_2d_identity_dir0_1234():

    filename = os.path.join(mesh_dir, 'identity_2d.h5')
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)

    dir_zero_boundary    = get_boundaries(1, 2, 3, 4)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries()

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error =  0.00021808678604159413
    expected_h1_error =  0.013023570720357957

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_identity_dir0_234_neu0_1():

    filename = os.path.join(mesh_dir, 'identity_2d.h5')
    solution = cos(0.5*pi*x)*sin(pi*y)
    f        = (5./4.)*pi**2*solution

    dir_zero_boundary    = get_boundaries(2, 3, 4)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries(1)

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error =  0.00015546057795986509
    expected_h1_error =  0.009269302784527035

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_identity_dir0_134_neu0_2():

    filename = os.path.join(mesh_dir, 'identity_2d.h5')
    solution = sin(0.5*pi*x)*sin(pi*y)
    f        = (5./4.)*pi**2*solution

    dir_zero_boundary    = get_boundaries(1, 3, 4)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries(2)

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error =  0.00015546057795095866
    expected_h1_error =  0.009269302784528054

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_identity_dir0_124_neu0_3():

    filename = os.path.join(mesh_dir, 'identity_2d.h5')
    solution = sin(pi*x)*cos(0.5*pi*y)
    f        = (5./4.)*pi**2*solution

    dir_zero_boundary    = get_boundaries(1, 2, 4)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries(3)

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error =  0.00015546057796188848
    expected_h1_error =  0.009269302784527448

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_identity_dir0_123_neu0_4():

    filename = os.path.join(mesh_dir, 'identity_2d.h5')
    solution = sin(pi*x)*sin(0.5*pi*y)
    f        = (5./4.)*pi**2*solution

    dir_zero_boundary    = get_boundaries(1, 2, 3)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries(4)

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error =  0.00015546057795073548
    expected_h1_error =  0.009269302784522822

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_identity_dir0_24_neu0_13():

    filename = os.path.join(mesh_dir, 'identity_2d.h5')
    solution = cos(0.5*pi*x)*cos(0.5*pi*y)
    f        = (1./2.)*pi**2*solution

    dir_zero_boundary    = get_boundaries(2, 4)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries(1, 3)

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error =  2.6119892693464717e-05
    expected_h1_error =  0.0016032430287989195

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_identity_dir0_4_neu0_123():

    filename = os.path.join(mesh_dir, 'identity_2d.h5')
    solution = cos(pi*x)*cos(0.5*pi*y)
    f        = 5./4.*pi**2*solution

    dir_zero_boundary    = get_boundaries(4)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries(1, 2, 3)

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error =  0.00015492540684276186
    expected_h1_error =  0.009242166615517364

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_identity_dir0_234_neui_1():

    filename = os.path.join(mesh_dir, 'identity_2d.h5')
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    dir_zero_boundary    = get_boundaries(2, 3, 4)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries(1)

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error = 0.00021786960671761908
    expected_h1_error = 0.01302350067761177

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_identity_dir0_134_neui_2():

    filename = os.path.join(mesh_dir, 'identity_2d.h5')
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    dir_zero_boundary    = get_boundaries(1, 3, 4)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries(2)

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error = 0.00021786960671761908
    expected_h1_error = 0.01302350067761177

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_identity_dir0_124_neui_3():

    filename = os.path.join(mesh_dir, 'identity_2d.h5')
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    dir_zero_boundary    = get_boundaries(1, 2, 4)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries(3)

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error = 0.00021786960671761908
    expected_h1_error = 0.01302350067761177

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_identity_dir0_123_neui_4():

    filename = os.path.join(mesh_dir, 'identity_2d.h5')
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    dir_zero_boundary    = get_boundaries(1, 2, 3)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries(4)

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error = 0.00021786960671761908
    expected_h1_error = 0.01302350067761177

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_identity_dir0_123_diri_4():

    filename = os.path.join(mesh_dir, 'identity_2d.h5')
    solution = sin(pi * x) * sin(0.5*pi * y)
    f        = 5/4*pi**2 * solution

    dir_zero_boundary    = get_boundaries(1, 2, 3)
    dir_nonzero_boundary = get_boundaries(4)
    neumann_boundary     = get_boundaries()

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error = 0.0001529221571156830
    expected_h1_error = 0.009293161646612863

    assert abs(l2_error - expected_l2_error) < 1.e-7
    assert abs(h1_error - expected_h1_error) < 1.e-7

#------------------------------------------------------------------------------
def test_poisson_2d_identity_dir0_13_diri_24():

    filename = os.path.join(mesh_dir, 'identity_2d.h5')
    solution = sin(3*pi/2 * x) * sin(3*pi/2 * y)
    f        = 9/2*pi**2 * solution

    dir_zero_boundary    = get_boundaries(1, 3)
    dir_nonzero_boundary = get_boundaries(2, 4)
    neumann_boundary     = get_boundaries()

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error = 0.0007786454571731944
    expected_h1_error = 0.0449669071240554

    assert abs(l2_error - expected_l2_error) < 1.e-7
    assert abs(h1_error - expected_h1_error) < 1.e-7

#==============================================================================
# 2D Poisson's equation with "Collela" map
#==============================================================================
def test_poisson_2d_collela_dir0_1234():

    filename = os.path.join(mesh_dir, 'collela_2d.h5')
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)

    dir_zero_boundary    = get_boundaries(1, 2, 3, 4)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries()

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error =  0.03032933682661518
    expected_h1_error =  0.41225081526853247

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_collela_dir0_234_neu0_1():

    filename = os.path.join(mesh_dir, 'collela_2d.h5')
    solution = sin(0.25*pi*(x-1))*sin(pi*y)
    f        = (17./16.)*pi**2*solution

    dir_zero_boundary    = get_boundaries(2, 3, 4)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries(1)

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error =  0.013540717397796734
    expected_h1_error =  0.19789463571596025

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_collela_dir0_134_neu0_2():

    filename = os.path.join(mesh_dir, 'collela_2d.h5')
    solution = sin(0.25*pi*(x+1.))*sin(pi*y)
    f        = (17./16.)*pi**2*solution

    dir_zero_boundary    = get_boundaries(1, 3, 4)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries(2)

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error =  0.012890849094111699
    expected_h1_error =  0.19553563279728328

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_collela_dir0_124_neu0_3():

    filename = os.path.join(mesh_dir, 'collela_2d.h5')
    solution = sin(0.25*pi*(y-1))*sin(pi*x)
    f        = (17./16.)*pi**2*solution

    dir_zero_boundary    = get_boundaries(1, 2, 4)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries(3)

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error =  0.013540717397817427
    expected_h1_error =  0.19789463571595994

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_collela_dir0_123_neu0_4():

    filename = os.path.join(mesh_dir, 'collela_2d.h5')
    solution = sin(0.25*pi*(y+1.))*sin(pi*x)
    f        = (17./16.)*pi**2*solution

    dir_zero_boundary    = get_boundaries(1, 2, 3)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries(4)

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error =  0.012890849094111942
    expected_h1_error =  0.19553563279728325

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_collela_dir0_234_neui_1():

    filename = os.path.join(mesh_dir, 'collela_2d.h5')
    solution = sin(pi/3 * (1 - x)) * cos(pi/2 * y)
    f        = (13/36)*pi**2 * solution

    dir_zero_boundary    = get_boundaries(2, 3, 4)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries(1)

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error = 0.002701799327716594
    expected_h1_error = 0.043091889730461796

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_collela_dir0_134_neui_2():

    filename = os.path.join(mesh_dir, 'collela_2d.h5')
    solution = sin(pi/3 * (1 + x)) * cos(pi/2 * y)
    f        = (13/36)*pi**2 * solution

    dir_zero_boundary    = get_boundaries(1, 3, 4)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries(2)

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error = 0.0026061796931093253
    expected_h1_error = 0.04400143055955451

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_collela_dir0_124_neui_3():

    filename = os.path.join(mesh_dir, 'collela_2d.h5')
    solution = cos(pi/2 * x) * sin(pi/3 * (1 - y))
    f        = (13/36)*pi**2 * solution

    dir_zero_boundary    = get_boundaries(1, 2, 4)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries(3)

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error = 0.002701799327724046
    expected_h1_error = 0.04309188973046055

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
@pytest.mark.parametrize('backend',  [None, PSYDAC_BACKEND_GPYCCEL, PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP])
def test_poisson_2d_collela_dir0_123_neui_4(backend):

    filename = os.path.join(mesh_dir, 'collela_2d.h5')
    solution = cos(pi/2 * x) * sin(pi/3 * (1 + y))
    f        = (13/36)*pi**2 * solution

    dir_zero_boundary    = get_boundaries(1, 2, 3)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries(4)

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary,comm=MPI.COMM_WORLD, backend=backend)

    expected_l2_error = 0.0026061796931066174
    expected_h1_error = 0.04400143055955377

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#------------------------------------------------------------------------------
def test_poisson_2d_collela_dir0_123_diri_4():

    filename = os.path.join(mesh_dir, 'collela_2d.h5')
    solution = cos(pi/2 * x) * sin(pi/3 * (1 + y))
    f        = (13/36)*pi**2 * solution

    dir_zero_boundary    = get_boundaries(1, 2, 3)
    dir_nonzero_boundary = get_boundaries(4)
    neumann_boundary     = get_boundaries()

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error = 0.0025850223987204306
    expected_h1_error = 0.04401691486495642

    assert abs(l2_error - expected_l2_error) < 1.e-7
    assert abs(h1_error - expected_h1_error) < 1.e-7

#------------------------------------------------------------------------------
@pytest.mark.parametrize('backend',  [None, PSYDAC_BACKEND_GPYCCEL, PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP])
def test_poisson_2d_collela_dir0_13_diri_24(backend):

    filename = os.path.join(mesh_dir, 'collela_2d.h5')
    solution = sin(pi/3 * (1 + x)) * sin(pi/3 * (1 + y))
    f        = (2/9)*pi**2 * solution

    dir_zero_boundary    = get_boundaries(1, 3)
    dir_nonzero_boundary = get_boundaries(2, 4)
    neumann_boundary     = get_boundaries()

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary, comm=MPI.COMM_WORLD, backend=backend)

    expected_l2_error = 0.0012801077606328381
    expected_h1_error = 0.02314405549486328

    assert abs(l2_error - expected_l2_error) < 1.e-7
    assert abs(h1_error - expected_h1_error) < 1.e-7

#------------------------------------------------------------------------------
def test_poisson_2d_collela_diri_1234():

    filename = os.path.join(mesh_dir, 'collela_2d.h5')
    solution = cos(pi/3 * x) * cos(pi/3 * y)
    f        = (2/9)*pi**2 * solution

    dir_zero_boundary    = get_boundaries()
    dir_nonzero_boundary = get_boundaries(1, 2, 3, 4)
    neumann_boundary     = get_boundaries()

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error = 0.0014604231101091047
    expected_h1_error = 0.025023352115363873

    assert abs(l2_error - expected_l2_error) < 1.e-7
    assert abs(h1_error - expected_h1_error) < 1.e-7

#==============================================================================
# 2D Poisson's equation on quarter annulus
#==============================================================================

def test_poisson_2d_quarter_annulus_dir0_1234():

    filename = os.path.join(mesh_dir, 'quarter_annulus.h5')
    c        = pi / (1. - 0.5**2)
    r2       = 1. - x**2 - y**2
    solution = x*y*sin(c * r2)
    f        = 4.*c**2*x*y*(x**2 + y**2)*sin(c * r2) + 12.*c*x*y*cos(c * r2)

    dir_zero_boundary    = get_boundaries(1, 2, 3, 4)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries()

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error =  0.00010289930281268989
    expected_h1_error =  0.009473407914765117

    assert abs(l2_error - expected_l2_error) < 1.e-7
    assert abs(h1_error - expected_h1_error) < 1.e-7

#------------------------------------------------------------------------------
@pytest.mark.parametrize('backend',  [None, PSYDAC_BACKEND_GPYCCEL, PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP])
def test_poisson_2d_quarter_annulus_dir0_12_diri_34(backend):

    filename = os.path.join(mesh_dir, 'quarter_annulus.h5')
    solution = sin(pi * x) * sin(pi * y)
    f        = 2*pi**2 * solution

    dir_zero_boundary    = get_boundaries(1, 2)
    dir_nonzero_boundary = get_boundaries(3, 4)
    neumann_boundary     = get_boundaries()

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary, comm=MPI.COMM_WORLD, backend=backend)

    expected_l2_error = 0.0005982761090480573
    expected_h1_error = 0.021271053089631443

    assert abs(l2_error - expected_l2_error) < 1.e-7
    assert abs(h1_error - expected_h1_error) < 1.e-7

#------------------------------------------------------------------------------
def test_poisson_2d_quarter_annulus_diri_1234():

    filename = os.path.join(mesh_dir, 'quarter_annulus.h5')
    solution = sin(pi*x + pi/4) * sin(pi*y + pi/4)
    f        = 2*pi**2 * solution

    dir_zero_boundary    = get_boundaries()
    dir_nonzero_boundary = get_boundaries(1, 2, 3, 4)
    neumann_boundary     = get_boundaries()

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error = 0.0006536882827670037
    expected_h1_error = 0.02592026831798558

    assert abs(l2_error - expected_l2_error) < 1.e-7
    assert abs(h1_error - expected_h1_error) < 1.e-7

#------------------------------------------------------------------------------
@pytest.mark.parametrize('backend',  [None, PSYDAC_BACKEND_GPYCCEL, PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP])
def test_poisson_2d_quarter_annulus_diri_34_neui_12(backend):

    filename = os.path.join(mesh_dir, 'quarter_annulus.h5')
    solution = sin(pi*x + pi/4) * sin(pi*y + pi/4)
    f        = 2*pi**2 * solution

    dir_zero_boundary    = get_boundaries()
    dir_nonzero_boundary = get_boundaries(3, 4)
    neumann_boundary     = get_boundaries(1, 2)

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary, comm=MPI.COMM_WORLD, backend=backend)

    expected_l2_error = 0.0006527836834289991
    expected_h1_error = 0.025919435390680808

    assert abs(l2_error - expected_l2_error) < 1.e-7
    assert abs(h1_error - expected_h1_error) < 1.e-7

#------------------------------------------------------------------------------
def test_poisson_2d_quarter_annulus_diri_12_neui_34():

    filename = os.path.join(mesh_dir, 'quarter_annulus.h5')
    solution = sin(pi*x + pi/4) * sin(pi*y + pi/4)
    f        = 2*pi**2 * solution

    dir_zero_boundary    = get_boundaries()
    dir_nonzero_boundary = get_boundaries(1, 2)
    neumann_boundary     = get_boundaries(3, 4)

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary)

    expected_l2_error = 0.0006663772402662598
    expected_h1_error = 0.025906309642232804

    assert abs(l2_error - expected_l2_error) < 1.e-7
    assert abs(h1_error - expected_h1_error) < 1.e-7

#==============================================================================
# 2D Poisson's equation on circle
#==============================================================================
@pytest.mark.parametrize('backend',  [None, PSYDAC_BACKEND_GPYCCEL, PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP])
def test_poisson_2d_circle_dir0(backend):

    filename = os.path.join(mesh_dir, 'circle.h5')
    solution = (1 - (x**2 + y**2)) * cos(2*pi*x) * cos(2*pi*y)
    f        = -laplace(solution)

    dir_zero_boundary    = get_boundaries(2) # only boundary is at r = r_max
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries()

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary, comm=MPI.COMM_WORLD, backend=backend)

    expected_l2_error = 0.0015245737751297718
    expected_h1_error = 0.06653900724243668

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
# 2D Poisson's equation on pipe
#==============================================================================
@pytest.mark.parametrize('backend',  [None, PSYDAC_BACKEND_GPYCCEL, PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP])
def test_poisson_2d_pipe_dir_1234(backend):

    filename = os.path.join(mesh_dir, 'pipe.h5')
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)

    dir_zero_boundary    = get_boundaries()
    dir_nonzero_boundary = get_boundaries(1, 2, 3, 4)
    neumann_boundary     = get_boundaries()

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary, comm=MPI.COMM_WORLD, backend=backend)

    expected_l2_error =  0.0008629074796755705
    expected_h1_error =  0.038151393401512884
 

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

###############################################################################
#            PARALLEL TESTS
###############################################################################

@pytest.mark.mpi
def test_poisson_2d_identity_dir0_1234_parallel():

    filename = os.path.join(mesh_dir, 'identity_2d.h5')
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)

    dir_zero_boundary    = get_boundaries(1, 2, 3, 4)
    dir_nonzero_boundary = get_boundaries()
    neumann_boundary     = get_boundaries()

    l2_error, h1_error = run_poisson_2d(filename, solution, f,
            dir_zero_boundary, dir_nonzero_boundary, neumann_boundary,
            comm=MPI.COMM_WORLD)

    expected_l2_error =  0.00021808678604159413
    expected_h1_error =  0.013023570720357957

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
