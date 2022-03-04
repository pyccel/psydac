import pytest
import numpy as np
from mpi4py import MPI

from sympy import Matrix

from sympde.calculus      import grad, dot, curl
from sympde.calculus      import minus, plus, cross
from sympde.topology      import Derham
from sympde.topology      import elements_of
from sympde.topology      import NormalVector
from sympde.topology      import Square
from sympde.topology      import IdentityMapping, PolarMapping
from sympde.expr.expr     import LinearForm, BilinearForm
from sympde.expr.expr     import integral
from sympde.expr.expr     import Norm
from sympde.expr.equation import find, EssentialBC

from psydac.feec.multipatch.api import discretize  # TODO: when possible, use line above

from psydac.linalg.iterative_solvers import cg, pcg
from psydac.linalg.utilities         import array_to_stencil

from psydac.fem.basic          import FemField
from psydac.feec.pull_push     import push_2d_hcurl, pull_2d_hcurl

from psydac.feec.multipatch.fem_linear_operators import FemLinearOperator, IdLinearOperator
from psydac.feec.multipatch.fem_linear_operators import SumLinearOperator, MultLinearOperator, ComposedLinearOperator
from psydac.feec.multipatch.operators import ConformingProjection_V1

from scipy.sparse.linalg import cg
#==============================================================================
# 2D Time Harmonic Maxwell equation
#==============================================================================
def test_maxwell_2d_2_patch_dirichlet_0():
    from sympy import cos, sin, pi, Tuple

    bounds1   = (0.5, 1.)
    bounds2_A = (0, np.pi/2)
    bounds2_B = (np.pi/2, np.pi)

    A = Square('A',bounds1=bounds1, bounds2=bounds2_A)
    B = Square('B',bounds1=bounds1, bounds2=bounds2_B)

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    D1     = mapping_1(A)
    D2     = mapping_2(B)

    domain = D1.join(D2, name = 'domain',
                bnd_minus = D1.get_boundary(axis=1, ext=1),
                bnd_plus  = D2.get_boundary(axis=1, ext=-1), direction=1)

    x,y    = domain.coordinates
    alpha    = 1.
    uex      = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
    f        = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y), 
                     alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))

    l2_error = run_conga_maxwell_2d(uex, f, alpha, domain, ncells=[2**3, 2**3], degree=[2,2], tol=1e-12)
    expected_l2_error = 0.006038546952550085

    assert ( abs(l2_error - expected_l2_error) < 1e-7)

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()
