from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve

from sympde.topology import Derham
from sympde.topology import element_of, elements_of
from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping

from sympde.expr.expr import LinearForm, BilinearForm
from sympde.expr.expr import integral
from sympde.expr      import Norm

from psydac.linalg.iterative_solvers import cg, pcg
from psydac.linalg.utilities         import array_to_stencil
from psydac.fem.basic                import FemField

from psydac.feec.multipatch.fem_linear_operators import IdLinearOperator, ComposedLinearOperator
from psydac.feec.multipatch.api                  import discretize
from psydac.feec.multipatch.operators            import get_patch_index_from_face

from psydac.api.essential_bc import apply_essential_bc_stencil

#==============================================================================
# 2D Poisson's equation
#==============================================================================
def test_poisson_2d_2_patch_dirichlet_0():

    A = Square('A',bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi))

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    D1     = mapping_1(A)
    D2     = mapping_2(B)

    domain = D1.join(D2, name = 'domain',
                bnd_minus = D1.get_boundary(axis=1, ext=1),
                bnd_plus  = D2.get_boundary(axis=1, ext=-1), direction=1)

    x,y = domain.coordinates
    solution = x**2 + y**2
    f        = -4

    l2_error, h1_error = run_poisson_2d(solution, f, domain, ncells=[2**2, 2**2], degree=[2,2])

    expected_l2_error = 2.4666711343436988e-11
    expected_h1_error = 5.816934665846925e-09


    assert ( abs(l2_error - expected_l2_error) < 1e-7)
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

#------------------------------------------------------------------------------
def test_poisson_2d_2_patch_dirichlet_1():

    from sympy import sin, cos, pi

    A = Square('A',bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi))

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    D1     = mapping_1(A)
    D2     = mapping_2(B)

    domain = D1.join(D2, name = 'domain',
                bnd_minus = D1.get_boundary(axis=1, ext=1),
                bnd_plus  = D2.get_boundary(axis=1, ext=-1), direction=1)

    x,y = domain.coordinates
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    l2_error, h1_error = run_poisson_2d(solution, f, domain, ncells=[2**2,2**2], degree=[2,2])

    expected_l2_error = 0.008675206239954483
    expected_h1_error = 0.12126000405442837

    assert ( abs(l2_error - expected_l2_error) < 1e-7 )
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

#------------------------------------------------------------------------------
def test_poisson_2d_2_patch_dirichlet_2():

    A = Square('A',bounds1=(0.5, 1.), bounds2=(-1., 0.))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
#    C = Square('C',bounds1=(-1., 0.), bounds2=(0.5, 1.))

    mapping_1 = IdentityMapping('M1', 2)
    mapping_2 = PolarMapping   ('M2', 2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    #mapping_3 = IdentityMapping('M3', 2)

    D1     = mapping_1(A)
    D2     = mapping_2(B)
    #D3     = mapping_3(C)

    D1D2        = D1.join(D2, name = 'AB',
                bnd_minus = D1.get_boundary(axis=1, ext=1),
                bnd_plus  = D2.get_boundary(axis=1, ext=-1), direction=1)

    #D1D2D3       = D1D2.join(D3, name = 'ABC',
    #            bnd_minus = D2.get_boundary(axis=1, ext=1),
    #            bnd_plus  = D3.get_boundary(axis=0, ext=1))

    x,y       = D1D2.coordinates
    solution  = x**2 + y**2
    f         = -4

    l2_error, h1_error = run_poisson_2d(solution, f, D1D2, ncells=[2**2,2**2], degree=[2,2])

    expected_l2_error = 4.014657795437355e-10
    expected_h1_error = 1.1492725536080688e-08

    assert ( abs(l2_error - expected_l2_error) < 1e-7)
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()
