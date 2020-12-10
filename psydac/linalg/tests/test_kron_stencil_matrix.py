from functools import reduce

import pytest
import numpy as np
from scipy.sparse import kron

from psydac.linalg.stencil import StencilVectorSpace
from psydac.linalg.stencil import StencilVector
from psydac.linalg.stencil import StencilMatrix
from psydac.linalg.kron    import KroneckerStencilMatrix

#==============================================================================
@pytest.mark.parametrize('npts', [(5, 7, 8)])
@pytest.mark.parametrize('pads', [(2, 3, 5)])
@pytest.mark.parametrize('periodic', [(True, False, False)])

def test_KroneckerStencilMatrix(npts, pads, periodic):

    # Extract input parameters
    n1, n2, n3 = npts
    p1, p2, p3 = pads
    P1, P2, P3 = periodic

    # 3D vector space and element
    W = StencilVectorSpace([n1, n2, n3], [p1, p2, p3], [P1, P2, P3])
    w = StencilVector(W)

    # 1D vector space
    V1 = StencilVectorSpace([n1], [p1], [P1])
    V2 = StencilVectorSpace([n2], [p2], [P2])
    V3 = StencilVectorSpace([n3], [p3], [P3])

    # 1D stencil matrices
    M1 = StencilMatrix(V1, V1)
    M2 = StencilMatrix(V2, V2)
    M3 = StencilMatrix(V3, V3)

    # ...
    # Fill in stencil matrix values
    for k1 in range(-p1, p1+1):
        M1[:, k1] = 10 + k1

    for k2 in range(-p2, p2+1):
        M2[:, k2] = 20 + k2

    for k3 in range(-p3, p3+1):
        M3[:, k3] = 40 + k3

    M1.remove_spurious_entries()
    M2.remove_spurious_entries()
    M3.remove_spurious_entries()
    # ...

    # Fill in vector values
    w[:, :, :] = 1.0

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
