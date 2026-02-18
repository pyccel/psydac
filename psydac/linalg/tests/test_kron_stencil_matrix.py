#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from   functools import reduce

import pytest
import numpy as np
from   mpi4py          import MPI
from   scipy.sparse    import kron

from   sympde.calculus import inner
from   sympde.expr     import integral, BilinearForm
from   sympde.topology import elements_of, Square, Line, Derham

from   psydac.api.discretization     import discretize
from   psydac.api.settings           import PSYDAC_BACKEND_GPYCCEL
from   psydac.ddm.cart               import DomainDecomposition, CartDecomposition
from   psydac.linalg.kron            import KroneckerStencilMatrix
from   psydac.linalg.block           import BlockLinearOperator
from   psydac.linalg.stencil         import StencilVectorSpace
from   psydac.linalg.stencil         import StencilVector
from   psydac.linalg.stencil         import StencilMatrix
from   psydac.linalg.tests.utilities import check_linop_equality_using_rng

#===============================================================================
def compute_global_starts_ends(domain_decomposition, npts):
    ndims         = len(npts)
    global_starts = [None]*ndims
    global_ends   = [None]*ndims

    for axis in range(ndims):
        ee = domain_decomposition.global_element_ends  [axis]

        global_ends  [axis]     = ee.copy()
        global_ends  [axis][-1] = npts[axis]-1
        global_starts[axis]     = np.array([0] + (global_ends[axis][:-1]+1).tolist())

    return tuple(global_starts), tuple(global_ends)

#==============================================================================
@pytest.mark.parametrize('dtype', [float,complex])
@pytest.mark.parametrize('npts', [(5, 7, 8)])
@pytest.mark.parametrize('pads', [(2, 3, 5)])
@pytest.mark.parametrize('periodic', [(True, False, False)])

def test_KroneckerStencilMatrix(dtype, npts, pads, periodic):

    # Extract input parameters
    n1, n2, n3 = npts
    p1, p2, p3 = pads
    P1, P2, P3 = periodic

    # Define data type with a factor
    if dtype==complex:
        factor=1j
    else:
        factor=1

    # Create domain decomposition
    D = DomainDecomposition([n1-1,n2-1, n3-1], periods=[P1,P2,P3])

    # Partition the points
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2,p3], shifts=[1,1,1])

    # 3D vector space and element
    W = StencilVectorSpace( cart, dtype=dtype)
    w = StencilVector(W)

    # 1D vector space

    D1 = DomainDecomposition([n1-1], periods=[P1])
    D2 = DomainDecomposition([n2-1], periods=[P2])
    D3 = DomainDecomposition([n3-1], periods=[P3])

    # Partition the points
    global_starts1, global_ends1 = compute_global_starts_ends(D1, [n1])
    global_starts2, global_ends2 = compute_global_starts_ends(D2, [n2])
    global_starts3, global_ends3 = compute_global_starts_ends(D3, [n3])

    cart1 = CartDecomposition(D1, [n1], global_starts1, global_ends1, pads=[p1], shifts=[1])
    cart2 = CartDecomposition(D2, [n2], global_starts2, global_ends2, pads=[p2], shifts=[1])
    cart3 = CartDecomposition(D3, [n3], global_starts3, global_ends3, pads=[p3], shifts=[1])

    V1 = StencilVectorSpace( cart1, dtype=dtype )
    V2 = StencilVectorSpace( cart2, dtype=dtype )
    V3 = StencilVectorSpace( cart3, dtype=dtype )

    # 1D stencil matrices
    M1 = StencilMatrix(V1, V1)
    M2 = StencilMatrix(V2, V2)
    M3 = StencilMatrix(V3, V3)

    # ...
    # Fill in stencil matrix values
    for k1 in range(-p1, p1+1):
        M1[:, k1] = 10 + k1*factor

    for k2 in range(-p2, p2+1):
        M2[:, k2] = 20 + k2*factor

    for k3 in range(-p3, p3+1):
        M3[:, k3] = 40 + k3*factor

    M1.remove_spurious_entries()
    M2.remove_spurious_entries()
    M3.remove_spurious_entries()
    # ...

    # Fill in vector values
    w[:, :, :] = factor

    # Create Kronecker matrix 
    M = KroneckerStencilMatrix(W, W, M1, M2, M3)

    # Scipy sparse matrices used for comparison
    M1_sp = M1.tosparse().tocsr()
    M2_sp = M2.tosparse().tocsr()
    M3_sp = M3.tosparse().tocsr()
    M_sp  = reduce(kron, (M1_sp, M2_sp, M3_sp)).tocsr()

    # Test transpose
    assert (M_sp.T - M.T.tosparse().tocsr()).count_nonzero() == 0

    # Test dot product
    assert np.array_equal(M_sp.dot(w.toarray()), M.dot(w).toarray())

#==============================================================================
def test_KroneckerStencilMatrix_diagonal(comm=None):
    """We create three mass matrices (Stencil/Block and Kronecker) belonging to a 2D de Rham sequence, and compare their diagonals."""

    ncells   = [6, 7]
    degree   = [3, 2]
    mult     = [1, 2]
    periodic = [False, True]

    backend = PSYDAC_BACKEND_GPYCCEL

    # 1. Obtain StencilMatrix / BlockLinearOperator (of StencilMatrices) mass matrices

    domain = Square('S', bounds1=(0,1), bounds2=(0,2))
    derham = Derham(domain, sequence=['h1', 'hcurl', 'l2'])

    domain_h = discretize(domain, ncells=ncells, periodic=periodic, comm=comm)
    derham_h = discretize(derham, domain_h, degree=degree, multiplicity=mult)

    V0, V1, V2       = derham.spaces
    V0h, V1h, V2h    = derham_h.spaces
    V0cs, V1cs, V2cs = [Vh.coeff_space for Vh in derham_h.spaces]

    u0, v0 = elements_of(V0, names='u0, v0')
    u1, v1 = elements_of(V1, names='u1, v1')
    u2, v2 = elements_of(V2, names='u2, v2')

    a0 = BilinearForm((u0, v0), integral(domain, u0*v0))
    a1 = BilinearForm((u1, v1), integral(domain, inner(u1, v1)))
    a2 = BilinearForm((u2, v2), integral(domain, u2*v2))

    a0h = discretize(a0, domain_h, (V0h, V0h), backend=backend)
    a1h = discretize(a1, domain_h, (V1h, V1h), backend=backend)
    a2h = discretize(a2, domain_h, (V2h, V2h), backend=backend)

    M0 = a0h.assemble()
    M1 = a1h.assemble()
    M2 = a2h.assemble()

    # 2. Obtain KroneckerStencilMatrix / BlockLinearOperator (of KroneckerStencilMatrices) mass matrices

    domain_1d_x = Line('L', bounds=domain.bounds1)
    domain_1d_y = Line('L', bounds=domain.bounds2)
    domains_1d  = (domain_1d_x, domain_1d_y)

    M0s_1d = []
    M1s_1d = []

    for n, d, m, p, domain_1d in zip(ncells, degree, mult, periodic, domains_1d):
        derham_1d = Derham(domain_1d)

        domain_1d_h = discretize(domain_1d, ncells=[n, ], periodic=[p, ])
        derham_1d_h = discretize(derham_1d, domain_1d_h, degree=[d, ], multiplicity=[m, ])

        V0_1d,  V1_1d  = derham_1d.spaces
        V0h_1d, V1h_1d = derham_1d_h.spaces
        
        u0_1d, v0_1d = elements_of(V0_1d, names='u0, v0')
        u1_1d, v1_1d = elements_of(V1_1d, names='u1, v1')

        a0_1d = BilinearForm((u0_1d, v0_1d), integral(domain_1d, u0_1d*v0_1d))
        a1_1d = BilinearForm((u1_1d, v1_1d), integral(domain_1d, u1_1d*v1_1d))

        a0h_1d = discretize(a0_1d, domain_1d_h, (V0h_1d, V0h_1d))
        a1h_1d = discretize(a1_1d, domain_1d_h, (V1h_1d, V1h_1d))

        M0s_1d.append(a0h_1d.assemble())
        M1s_1d.append(a1h_1d.assemble())

    M0_kron = KroneckerStencilMatrix(V0cs, V0cs, *M0s_1d)
    M1_kron = BlockLinearOperator(V1cs, V1cs, [[KroneckerStencilMatrix(V1cs[0], V1cs[0], M1s_1d[0], M0s_1d[1]), None],
                                               [None, KroneckerStencilMatrix(V1cs[1], V1cs[1], M0s_1d[0], M1s_1d[1])]])
    M2_kron = KroneckerStencilMatrix(V2cs, V2cs, *M1s_1d)

    # 3. Test whether M0/1/2.diagonal() is equal to M0/1/2_kron.diagonal() for all possible kwargs

    for M, M_kron in zip((M0, M1, M2), (M0_kron, M1_kron, M2_kron)):
        options = [True, False]

        for inverse in options:
            for sqrt in options:
                M_diag = M.diagonal(inverse=inverse, sqrt=sqrt)
                M_kron_diag = M_kron.diagonal(inverse=inverse, sqrt=sqrt)

                check_linop_equality_using_rng(M_diag, M_kron_diag, tol=1e-13)

#==============================================================================
@pytest.mark.mpi
def test_KroneckerStencilMatrix_diagonal_parallel():
    comm = MPI.COMM_WORLD
    test_KroneckerStencilMatrix_diagonal(comm=comm)
