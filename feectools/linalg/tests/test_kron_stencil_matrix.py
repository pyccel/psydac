from functools import reduce

import pytest
import numpy as np
from scipy.sparse import kron

from feectools.ddm.cart       import DomainDecomposition, CartDecomposition
from feectools.linalg.stencil import StencilVectorSpace
from feectools.linalg.stencil import StencilVector
from feectools.linalg.stencil import StencilMatrix
from feectools.linalg.kron    import KroneckerStencilMatrix
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
@pytest.mark.parametrize('dtype', [float])
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
