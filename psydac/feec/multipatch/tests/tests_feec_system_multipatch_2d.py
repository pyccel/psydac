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
from psydac.feec.multipatch.operators import BrokenMass
from psydac.feec.multipatch.operators import ConformingProjection_V1

from scipy.sparse.linalg import cg
comm = MPI.COMM_WORLD
#==============================================================================
def run_conga_maxwell_2d(uex, f, alpha, domain, ncells, degree, tol, use_scipy=True):
    """
    - assemble and solve a Maxwell problem on a multipatch domain, using Conga approach
    - this problem is adapted from the single patch test_api_system_3_2d_dir_1
    """

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    # Multipatch de Rham sequence:
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])

    u, v, F  = elements_of(derham.V1, names='u, v, F')

    nn           = NormalVector('nn')
    penalization = 10**7
 
    ab    = BilinearForm((u,v), integral(domain.boundary, penalization * cross(u, nn) * cross(v, nn)))

    l = LinearForm(v, integral(domain, dot(f,v)) + integral(domain.boundary, penalization * cross(uex, nn) * cross(v, nn)))

    # Error norm
    error   = Matrix([F[0]-uex[0],F[1]-uex[1]])
    l2_norm = Norm(error, domain, kind='l2')

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    domain_h = discretize(domain, ncells=ncells, comm=comm)
    derham_h = discretize(derham, domain_h, degree=degree)

    V1h = derham_h.V1
    V2h = derham_h.V2

    ab_h  = discretize(ab, domain_h, [V1h, V1h])

    lh = discretize(l, domain_h, V1h)

    # Discretize error norm
    l2_norm_h  = discretize(l2_norm, domain_h, V1h)

    #+++++++++++++++++++++++++++++++
    # 3. Assembly
    #+++++++++++++++++++++++++++++++

    # Mass matrices for broken spaces (block-diagonal)
    M1 = BrokenMass(V1h, domain_h, is_scalar=False)
    M2 = BrokenMass(V2h, domain_h, is_scalar=True)

    # Broken derivative operators
    bD0, bD1 = derham_h.broken_derivatives_as_operators

    # Conforming projection V1 -> V1
    cP1 = ConformingProjection_V1(V1h, domain_h)

    I1  = IdLinearOperator(V1h)

    A1 = 1e10*ComposedLinearOperator([I1-cP1,I1-cP1]) + ComposedLinearOperator([cP1, bD1.transpose(), M2, bD1, cP1]) + alpha * M1

    A_b = FemLinearOperator(fem_domain=V1h, fem_codomain=V1h, matrix=ab_h.assemble())

    A   = A1 + A_b

    b   = lh.assemble()

    #+++++++++++++++++++++++++++++++
    # 4. Solution
    #+++++++++++++++++++++++++++++++

    # Solve linear system

    if use_scipy:
        A = A.to_sparse_matrix()
        b = b.toarray()

        x,info   = cg(A, b, tol=tol, atol=tol)
        u_coeffs   = array_to_stencil(x, V1h.vector_space)

    else:
        u_coeffs, info = cg( A, b, tol=tol)

    uh = FemField(V1h, coeffs=u_coeffs)
    uh = cP1(uh)

    # Compute error norms
    l2_error = l2_norm_h.assemble(F=uh)

    return l2_error

###############################################################################
#            SERIAL TESTS
###############################################################################

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
