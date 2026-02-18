#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import pytest
import numpy as np

from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix, StencilInterfaceMatrix
from psydac.api.settings   import *
from psydac.ddm.cart import DomainDecomposition, CartDecomposition

#===============================================================================
def compute_global_starts_ends(domain_decomposition, npts, pads):
    ndims         = len(npts)
    global_starts = [None]*ndims
    global_ends   = [None]*ndims

    for axis in range(ndims):
        ee = domain_decomposition.global_element_ends  [axis]

        global_ends  [axis]     = ee.copy()
        global_ends  [axis][-1] = npts[axis]-1
        global_starts[axis]     = np.array([0] + (global_ends[axis][:-1]+1).tolist())

    for s, e, p in zip(global_starts, global_ends, pads):
        assert all(e - s + 1 >= p)

    return tuple(global_starts), tuple(global_ends)

#===============================================================================
def get_minus_starts_ends(plus_starts, plus_ends, minus_npts, plus_npts, minus_axis, plus_axis,
                          minus_ext, plus_ext, minus_pads, plus_pads, minus_shifts, plus_shifts,
                          diff):
    """
    Compute the coefficients needed by the minus patch in a given interface.
    """
    starts = [max(0,s-m*p) for s,m,p in zip(plus_starts, minus_shifts, minus_pads)]
    ends   = [min(n,e+m*p) for e,n,m,p in zip(plus_ends, minus_npts, minus_shifts, minus_pads)]
    starts[minus_axis] = 0 if minus_ext == -1 else ends[minus_axis]-minus_pads[minus_axis]
    ends[minus_axis]   = ends[minus_axis] if minus_ext == 1 else minus_pads[minus_axis]
    return starts, ends

#------------------------------------------------------------------------------
def get_plus_starts_ends(minus_starts, minus_ends, minus_npts, plus_npts, minus_axis, plus_axis,
                         minus_ext, plus_ext, minus_pads, plus_pads, minus_shifts, plus_shifts,
                         diff):
    """
    Compute the coefficients needed by the plus patch in a given interface.
    """
    starts = [max(0,s-m*p) for s,m,p in zip(minus_starts, plus_shifts, plus_pads)]
    ends   = [min(n,e+m*p) for e,n,m,p in zip(minus_ends, plus_npts, plus_shifts, plus_pads)]
    starts[plus_axis] = 0 if plus_ext == -1 else ends[plus_axis]-plus_pads[plus_axis]
    ends[plus_axis]   = ends[plus_axis] if plus_ext == 1 else plus_pads[plus_axis]
    return starts, ends

# TODO : Add test about dot, div, transpose, toarray, tosparse, copy, basic operation, max, exchange_assembly_data, set_backend
# ===============================================================================
# SERIAL TESTS
# ===============================================================================

@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('axis', [0])
@pytest.mark.parametrize('ext', [-1, 1])
@pytest.mark.parametrize('n1', [7, 15])
@pytest.mark.parametrize('p1', [2, 4])
@pytest.mark.parametrize('s1', [1, 2])
def test_stencil_interface_matrix_1d_serial_init(dtype, n1, p1, s1, axis, ext, P1=True):
    # Create domain decomposition
    D = DomainDecomposition([n1], periods=[P1])

    # Partition the points
    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1])
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[s1])

    # Create a vector space V and a matrix M from V to V
    V = StencilVectorSpace(cart, dtype=dtype)
    V.set_interface(axis, ext, cart)
    W = StencilVectorSpace(cart, dtype=dtype)
    W.set_interface(axis, -ext, cart)
    M = StencilInterfaceMatrix(V, W, global_starts, global_starts, axis, axis, ext, -ext)

    # Check properties of this matrix
    assert M.domain == V
    assert M.codomain == W
    assert M.dtype == dtype
    assert M.domain_axis == axis
    assert M.codomain_axis == axis
    assert M.domain_ext == ext
    assert M.codomain_ext == -ext
    assert M.dim == 1
    assert M.domain_start == (0,) * M.dim
    assert M.codomain_start == (0,) * M.dim
    assert M.flip == (1,) * M.dim
    assert np.array_equal(M.permutation, [0])
    assert M.pads == (p1,)
    assert M.backend == None
    assert M._data.shape == (p1 + 1 + 2 * p1 * s1, 1 + 2 * p1)
    assert M.shape == (n1, n1)

# ===============================================================================

@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('axis1', [0, 1])
@pytest.mark.parametrize('axis2', [0, 1])
@pytest.mark.parametrize('ext1', [-1, 1])
@pytest.mark.parametrize('ext2', [-1, 1])
@pytest.mark.parametrize('n1', [7, 15])
@pytest.mark.parametrize('n2', [7, 15])
@pytest.mark.parametrize('p1', [2, 3])
@pytest.mark.parametrize('p2', [2, 3])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [1, 2])
def test_stencil_interface_matrix_2d_serial_init(dtype, n1, n2, p1, p2, s1, s2, axis1, axis2, ext1, ext2, P1=True,
                                                 P2=True):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create a vector space V and a matrix M from V to V
    V = StencilVectorSpace(cart, dtype=dtype)
    V.set_interface(axis1, ext1, cart)
    W = StencilVectorSpace(cart, dtype=dtype)
    W.set_interface(axis2, ext2, cart)
    M = StencilInterfaceMatrix(V, W, global_starts, global_starts, axis1, axis2, ext1, ext2)

    # Check properties of this matrix
    assert M.domain == V
    assert M.codomain == W
    assert M.dtype == dtype
    assert M.domain_axis == axis1
    assert M.codomain_axis == axis2
    assert M.domain_ext == ext1
    assert M.codomain_ext == ext2
    assert M.dim == 2
    assert M.domain_start == (0,) * M.dim
    assert M.codomain_start == (0,) * M.dim
    assert M.flip == (1,) * M.dim
    assert M.pads == (p1, p2)
    assert M.backend == None
    if axis2 == 0:
        assert M._data.shape == (p1 + 1 + 2 * p1 * s1, n2 + 2 * p2 * s2, 1 + 2 * p1, 1 + 2 * p2)
    elif axis2 == 1:
        assert M._data.shape == (n1 + 2 * p1 * s1, p2 + 1 + 2 * p2 * s2, 1 + 2 * p1, 1 + 2 * p2)
    if axis1 == axis2:
        assert np.array_equal(M.permutation, [0, 1])
    else:
        assert np.array_equal(M.permutation, [1, 0])
    assert M.shape == (n1 * n2, n1 * n2)

# ===============================================================================

@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('axis1', [0, 1, 2])
@pytest.mark.parametrize('axis2', [0, 1, 2])
@pytest.mark.parametrize('ext1', [-1, 1])
@pytest.mark.parametrize('ext2', [-1, 1])
@pytest.mark.parametrize('n1', [7, 15])
@pytest.mark.parametrize('n2', [7, 15])
@pytest.mark.parametrize('n3', [6])
@pytest.mark.parametrize('p1', [2, 3])
@pytest.mark.parametrize('p2', [2, 3])
@pytest.mark.parametrize('p3', [1])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [1, 2])
@pytest.mark.parametrize('s3', [3])
def test_stencil_interface_matrix_3d_serial_init(dtype, n1, n2, n3, p1, p2, p3, s1, s2, s3, axis1, axis2,
                                                 ext1, ext2, P1=True, P2=True, P3=False):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2, n3], periods=[P1, P2, P3])

    # Partition the points
    npts = [n1, n2, n3]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2, p3])
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2, p3], shifts=[s1, s2, s3])

    # Create a vector space V and a matrix M from V to V
    V = StencilVectorSpace(cart, dtype=dtype)
    V.set_interface(axis1, ext1, cart)
    W = StencilVectorSpace(cart, dtype=dtype)
    W.set_interface(axis2, ext2, cart)
    M = StencilInterfaceMatrix(V, W, global_starts, global_starts, axis1, axis2, ext1, ext2)

    # Check properties of this matrix
    assert M.domain == V
    assert M.codomain == W
    assert M.dtype == dtype
    assert M.domain_axis == axis1
    assert M.codomain_axis == axis2
    assert M.domain_ext == ext1
    assert M.codomain_ext == ext2
    assert M.dim == 3
    assert M.domain_start == (0,) * M.dim
    assert M.codomain_start == (0,) * M.dim
    assert M.flip == (1,) * M.dim
    assert M.pads == (p1, p2, p3)
    assert M.backend == None
    if axis2 == 0:
        assert M._data.shape == (
        p1 + 1 + 2 * p1 * s1, n2 + 2 * p2 * s2, n3 + 2 * p3 * s3, 1 + 2 * p1, 1 + 2 * p2, 1 + 2 * p3)
    elif axis2 == 1:
        assert M._data.shape == (
        n1 + 2 * p1 * s1, p2 + 1 + 2 * p2 * s2, n3 + 2 * p3 * s3, 1 + 2 * p1, 1 + 2 * p2, 1 + 2 * p3)
    elif axis2 == 2:
        assert M._data.shape == (
        n1 + 2 * p1 * s1, n2 + 2 * p2 * s2, p3 + 1 + 2 * p3 * s3, 1 + 2 * p1, 1 + 2 * p2, 1 + 2 * p3)
    if axis1 == axis2:
        assert np.array_equal(M.permutation, [0, 1, 2])
    else:
        permutation = [0, 1, 2]
        permutation[axis1], permutation[axis2] = permutation[axis2], permutation[axis1]
        assert np.array_equal(M.permutation, permutation)
    assert M.shape == (n1 * n2 * n3, n1 * n2 * n3)
#===============================================================================
# Parallel TESTS
#===============================================================================
@pytest.mark.parametrize("n1,n2,p1,p2,expected", [(8,8,1,1, 827301207168.0), 
                                                  (8,8,2,2, 4824719287396.0),
                                                  (8,8,3,3, 13615010842712.0),
                                                  (12,12,1,1, 3023467041788.0),
                                                  (12,12,2,2, 19555497680544.0),
                                                  (12,12,3,3, 62573623909332.0)])
@pytest.mark.mpi
def test_stencil_interface_matrix_2d_parallel_dot(n1, n2, p1, p2, expected):

    from mpi4py              import MPI
    from psydac.ddm.cart     import MultiPatchDomainDecomposition, CartDecomposition, create_interfaces_cart
    from psydac.linalg.block import BlockVectorSpace, BlockVector, BlockLinearOperator

    # Number of patches
    N = 2

    nc = [[n1-p1,n2-p2] for i in range(N)]

    # Periodicity
    P = [[False, False] for i in range(N)]

    axis         = 0
    connectivity = {(0,1):((axis,1),(axis,-1))}

    comm = MPI.COMM_WORLD
    # Parallel info

    domain_decomposition =  MultiPatchDomainDecomposition(nc, P, comm=comm)

    # Number of elements
    n = [[n1,n2] for i in range(N)]

    # Padding ('thickness' of ghost region)
    p = [[p1,p2] for i in range(N)]

    carts = []
    for i in range(N):
        global_starts = [None]*2
        global_ends   = [None]*2
        for j in range(2):
            es = domain_decomposition.domains[i].global_element_starts[j]
            ee = domain_decomposition.domains[i].global_element_ends  [j]

            global_ends  [j]     = ee.copy()
            global_ends  [j][-1] = n[i][j]-1
            global_starts[j]     = np.array([0] + (global_ends[j][:-1]+1).tolist())

        carts.append(CartDecomposition(
                        domain_decomposition      = domain_decomposition.domains[i],
                        npts          = n[i],
                        global_starts = global_starts,
                        global_ends   = global_ends,
                        pads          = p[i],
                        shifts        = [1,1]))
    carts = tuple(carts)

    communication_info = (get_minus_starts_ends, get_plus_starts_ends)
    interface_carts = create_interfaces_cart(domain_decomposition, carts, connectivity, communication_info=communication_info)

   # Create vector spaces
    Vs  = [StencilVectorSpace( ci ) for ci in carts]

    # Create the interface spaces
    for i,j in connectivity:

        if not carts[i].is_comm_null and not carts[j].is_comm_null:
            cart_minus = carts[i]
            cart_plus  = carts[j]
        elif (i,j) in interface_carts:
            cart_minus = interface_carts[i,j]
            cart_plus  = interface_carts[i,j]
        else:
            continue

        ((axis_minus, ext_minus), (axis_plus , ext_plus)) = connectivity[i, j]
        # set interface space for the minus space

        Vs[i].set_interface(axis_minus, ext_minus, cart_minus)

        # set interface space for the plus space
        Vs[j].set_interface(axis_plus , ext_plus, cart_plus)

    V   = BlockVectorSpace(*Vs, connectivity=connectivity)


    # ...
    # Fill in vector with some values, then update ghost regions
    x   = BlockVector( V )
    for i,ci in enumerate(carts):
        if ci.is_comm_null:continue
        s1,s2 = ci.starts
        e1,e2 = ci.ends
        for i1 in range(s1,e1+1):
            for i2 in range(s2,e2+1):
                x[i][i1,i2] = 1

    # ...
    # Fill in the Matrix with some values
    A   = BlockLinearOperator( V, V )

    # Fill-in pattern
    fill_in = lambda i, i1, i2, k1, k2: 10000*i + 1000*i1 + 100*i2 + 10*abs(k1) + abs(k2)

    for i,ci in enumerate(carts):
        if ci.is_comm_null:continue
        Aii = StencilMatrix( Vs[i], Vs[i]) 
        s1,s2 = ci.starts
        e1,e2 = ci.ends
        # Fill in stencil matrix
        for i1 in range(s1, e1+1):
            for i2 in range(s2, e2+1):
                for k1 in range(-p1, p1+1):
                    for k2 in range(-p2, p2+1):
                        Aii[i1, i2, k1, k2] = fill_in( i, i1, i2, k1, k2 )

        Aii.remove_spurious_entries()
        A[i,i] = Aii

    # Fill in stencil interface matrix if the process is on the boundary
    if not carts[0].is_comm_null and (not carts[1].is_comm_null or not interface_carts[0,1].is_comm_null):
        s_d = 0
        s_c = n[0][axis]-p[0][axis]-1-Vs[0].starts[axis]
        A01 = StencilInterfaceMatrix(Vs[1], Vs[0], s_d, s_c, d_axis=axis, c_axis=axis, d_ext=-1, c_ext=1)

        s1,s2 = carts[0].starts
        e1,e2 = carts[0].ends
        p1,p2 = Vs[0].pads

        for i2 in range(s2, e2+1):
            A01._data[2*p1,i2+p2-s2,0,p2] = -i2-1

        A[0,1] = A01

    if not carts[1].is_comm_null and (not carts[0].is_comm_null or not interface_carts[0,1].is_comm_null):
        s_d = n[0][axis]-p[0][axis]-1-Vs[0].starts[axis]
        s_c = 0
        A10 = StencilInterfaceMatrix(Vs[0], Vs[1], s_d, s_c, d_axis=axis, c_axis=axis, d_ext=1, c_ext=-1)

        s1,s2 = carts[1].starts
        e1,e2 = carts[1].ends
        p1,p2 = Vs[1].pads
        for i2 in range(s2, e2+1):
            A10._data[p1,i2+p2-s2,2*p1,p2] = -i2-1

        A[1,0] = A10

    A = A.T.T
    # Updateh ghost regions and compute matrix-vector product
    x.update_ghost_regions()
    y = A.dot(x)

    # Check the results
    assert y.inner(y) == expected

#===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================
if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
