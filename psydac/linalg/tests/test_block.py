# -*- coding: UTF-8 -*-
#
import pytest
import numpy as np
import scipy.sparse as spa
from random import random, seed

from psydac.linalg.direct_solvers import SparseSolver
from psydac.linalg.stencil        import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.block          import BlockVectorSpace, BlockVector
from psydac.linalg.block          import BlockLinearOperator, BlockDiagonalSolver, BlockMatrix
from psydac.linalg.utilities      import array_to_stencil
from psydac.linalg.kron           import KroneckerLinearSolver
from psydac.api.settings          import PSYDAC_BACKEND_GPYCCEL
from psydac.ddm.cart              import DomainDecomposition, CartDecomposition

#===============================================================================
def compute_global_starts_ends(domain_h, npts):
    global_starts = [None]*2
    global_ends   = [None]*2

    for axis in range(2):
        es = domain_h.global_element_starts[axis]
        ee = domain_h.global_element_ends  [axis]

        global_ends  [axis]     = ee.copy()
        global_ends  [axis][-1] = npts[axis]-1
        global_starts[axis]     = np.array([0] + (global_ends[axis][:-1]+1).tolist())

    return global_starts, global_ends

#===============================================================================
# SERIAL TESTS
#===============================================================================
@pytest.mark.parametrize( 'n1', [8,16] )
@pytest.mark.parametrize( 'n2', [8,12] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'p2', [1,2,3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )

def test_block_linear_operator_serial_dot( n1, n2, p1, p2, P1, P2  ):
    # set seed for reproducibility
    seed(n1*n2*p1*p2)

    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector spaces, stencil matrices, and stencil vectors
    V = StencilVectorSpace( cart )
    M1 = StencilMatrix( V, V )
    M2 = StencilMatrix( V, V )
    M3 = StencilMatrix( V, V )
    x1 = StencilVector( V )
    x2 = StencilVector( V )

    # Fill in stencil matrices based on diagonal index
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            M1[:,:,k1,k2] = 10*k1+k2
            M2[:,:,k1,k2] = 10*k1+k2+2.
            M3[:,:,k1,k2] = 10*k1+k2+5.
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()
    M3.remove_spurious_entries()

    W = BlockVectorSpace(V, V)

    # Construct a BlockLinearOperator object containing M1, M2, M, using 3 ways
    #     |M1  M2|
    # L = |      |
    #     |M3  0 |

    dict_blocks = {(0,0):M1, (0,1):M2, (1,0):M3}
    list_blocks = [[M1, M2], [M3, None]]

    L1 = BlockLinearOperator( W, W, blocks=dict_blocks )
    L2 = BlockLinearOperator( W, W, blocks=list_blocks )

    L3 = BlockLinearOperator( W, W )
    L3[0,0] = M1
    L3[0,1] = M2
    L3[1,0] = M3

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1):
        for i2 in range(n2):
            x1[i1,i2] = 2.0*random() - 1.0
            x2[i1,i2] = 5.0*random() - 1.0
    x1.update_ghost_regions()
    x2.update_ghost_regions()

    # Construct a BlockVector object containing x1 and x2
    #     |x1|
    # X = |  |
    #     |x2|

    X = BlockVector(W)
    X[0] = x1
    X[1] = x2

    # Compute BlockLinearOperator product
    Y1 = L1.dot(X)
    Y2 = L2.dot(X)
    Y3 = L3.dot(X)

    # Compute matrix-vector products for each block
    y1 = M1.dot(x1) + M2.dot(x2)
    y2 = M3.dot(x1)

    # Check data in 1D array
    assert np.allclose( Y1.blocks[0].toarray(), y1.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Y1.blocks[1].toarray(), y2.toarray(), rtol=1e-14, atol=1e-14 )

    assert np.allclose( Y2.blocks[0].toarray(), y1.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Y2.blocks[1].toarray(), y2.toarray(), rtol=1e-14, atol=1e-14 )

    assert np.allclose( Y3.blocks[0].toarray(), y1.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Y3.blocks[1].toarray(), y2.toarray(), rtol=1e-14, atol=1e-14 )

#===============================================================================
@pytest.mark.parametrize( 'n1', [8,16] )
@pytest.mark.parametrize( 'n2', [8,12] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'p2', [1,2,3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )
def test_block_diagonal_solver_serial_dot( n1, n2, p1, p2, P1, P2  ):
    # set seed for reproducibility
    seed(n1*n2*p1*p2)

    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector spaces, stencil matrices, and stencil vectors
    V = StencilVectorSpace( cart )

    # Fill in stencil matrices based on diagonal index
    m11 = np.zeros((n1, n1))
    m12 = np.zeros((n2, n2))
    for j in range(n1):
        for i in range(-p1,p1+1):
            m11[j, max(0, min(n1-1, j+i))] = 10*j+i
    for j in range(n2):
        for i in range(-p2,p2+1):
            m12[j, max(0, min(n2-1, j+i))] = 20*j+5*i+2.
    
    m21 = np.zeros((n1, n1))
    m22 = np.zeros((n2, n2))
    for j in range(n1):
        for i in range(-p1,p1+1):
            m21[j, max(0, min(n1-1, j+i))] = 10*j**2+i**3
    for j in range(n2):
        for i in range(-p2,p2+1):
            m22[j, max(0, min(n2-1, j+i))] = 20*j**2+i**3+2.
    
    M11 = SparseSolver( spa.csc_matrix(m11) )
    M12 = SparseSolver( spa.csc_matrix(m12) )
    M21 = SparseSolver( spa.csc_matrix(m21) )
    M22 = SparseSolver( spa.csc_matrix(m22) )
    M1 = KroneckerLinearSolver(V, [M11,M12])
    M2 = KroneckerLinearSolver(V, [M21,M22])
    x1 = StencilVector( V )
    x2 = StencilVector( V )

    W = BlockVectorSpace(V, V)

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1):
        for i2 in range(n2):
            x1[i1,i2] = 2.0*random() - 1.0
            x2[i1,i2] = 5.0*random() - 1.0
    x1.update_ghost_regions()
    x2.update_ghost_regions()

    # Construct a BlockVector object containing x1 and x2
    #     |x1|
    # X = |  |
    #     |x2|

    X = BlockVector(W)
    X[0] = x1
    X[1] = x2

    # Construct a BlockDiagonalSolver object containing M1, M2 using 3 ways
    #     |M1  0 |
    # L = |      |
    #     |0   M2|

    dict_blocks = {0:M1, 1:M2}
    list_blocks = [M1, M2]

    L1 = BlockDiagonalSolver( W, blocks=dict_blocks )
    L2 = BlockDiagonalSolver( W, blocks=list_blocks )

    L3 = BlockDiagonalSolver( W )

    # Test for not allowing undefinedness
    errresult = False
    try:
        L3.solve(X)
    except NotImplementedError:
        errresult = True
    assert errresult

    L3[0] = M1
    L3[1] = M2

    # Compute BlockDiagonalSolver product
    Y1 = L1.solve(X)
    Y2 = L2.solve(X)
    Y3 = L3.solve(X)

    # Transposed
    Yt = L1.solve(X, transposed=True)

    # Test other in/out methods
    Y4a = W.zeros()
    Y4b = L1.solve(X, out=Y4a)
    assert Y4b is Y4a

    Y5a = W.zeros()
    Y5a[0] = x1.copy()
    Y5a[1] = x2.copy()
    Y5b = L1.solve(Y5a, out=Y5a)
    assert Y5b is Y5a

    # Solve linear equations for each block
    y1 = M1.solve(x1)
    y2 = M2.solve(x2)

    y1t = M1.solve(x1, transposed=True)
    y2t = M2.solve(x2, transposed=True)

    # Check data in 1D array
    assert np.allclose( Y1.blocks[0].toarray(), y1.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Y1.blocks[1].toarray(), y2.toarray(), rtol=1e-14, atol=1e-14 )

    assert np.allclose( Y2.blocks[0].toarray(), y1.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Y2.blocks[1].toarray(), y2.toarray(), rtol=1e-14, atol=1e-14 )

    assert np.allclose( Y3.blocks[0].toarray(), y1.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Y3.blocks[1].toarray(), y2.toarray(), rtol=1e-14, atol=1e-14 )

    assert np.allclose( Y4a.blocks[0].toarray(), y1.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Y4a.blocks[1].toarray(), y2.toarray(), rtol=1e-14, atol=1e-14 )

    assert np.allclose( Y5a.blocks[0].toarray(), y1.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Y5a.blocks[1].toarray(), y2.toarray(), rtol=1e-14, atol=1e-14 )

    assert np.allclose( Yt.blocks[0].toarray(), y1t.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Yt.blocks[1].toarray(), y2t.toarray(), rtol=1e-14, atol=1e-14 )

#===============================================================================
@pytest.mark.parametrize( 'n1', [8,16] )
@pytest.mark.parametrize( 'n2', [8,12] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'p2', [1,2,3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )

def test_block_matrix( n1, n2, p1, p2, P1, P2  ):
    # set seed for reproducibility
    seed(n1*n2*p1*p2)

    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector spaces, stencil matrices, and stencil vectors
    V = StencilVectorSpace( cart )
    M1 = StencilMatrix( V, V )
    M2 = StencilMatrix( V, V )
    M3 = StencilMatrix( V, V )
    M4 = StencilMatrix( V, V )
    x1 = StencilVector( V )
    x2 = StencilVector( V )

    # Fill in stencil matrices based on diagonal index
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            M1[:,:,k1,k2] = 10*k1+k2+1.
            M2[:,:,k1,k2] = 10*k1+k2+2.
            M3[:,:,k1,k2] = 10*k1+k2+5.
            M4[:,:,k1,k2] = 10*k1+k2+7.
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()
    M3.remove_spurious_entries()
    M4.remove_spurious_entries()

    W = BlockVectorSpace(V, V)
    # Construct a BlockMatrix object containing M1, M2, M3 and M4, using 3 ways
    #     |M1  M2|
    # L = |      |
    #     |M3  M4|
    dict_blocks = {(0,0):M1, (0,1):M2, (1,0):M3, (1,1):M4}
    list_blocks = ((M1,M2),(M3,M4))

    L1 = BlockMatrix( W, W, blocks=dict_blocks )
    L2 = BlockMatrix( W, W, blocks=list_blocks )

    L3 = BlockMatrix( W, W )
    L3[0,0] = M1
    L3[0,1] = M2
    L3[1,0] = M3
    L3[1,1] = M4

    # Convert L1, L2 and L3 to COO form
    coo1 = L1.tosparse().tocoo()
    coo2 = L2.tosparse().tocoo()
    coo3 = L3.tosparse().tocoo()

    assert np.array_equal( coo1.col , coo2.col  )
    assert np.array_equal( coo1.row , coo2.row  )
    assert np.array_equal( coo1.data, coo2.data )

    assert np.array_equal( coo1.col , coo3.col  )
    assert np.array_equal( coo1.row , coo3.row  )
    assert np.array_equal( coo1.data, coo3.data )

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1):
        for i2 in range(n2):
            x1[i1,i2] = 2.0*random() + 1.0
            x2[i1,i2] = 5.0*random() - 1.0
    x1.update_ghost_regions()
    x2.update_ghost_regions()

    # Compute dots and compare results
    y1 = M1.dot(x1) + M2.dot(x2)
    y2 = M3.dot(x1) + M4.dot(x2)
    yref = np.concatenate((y1.toarray(), y2.toarray()))

    x = np.concatenate((x1.toarray(), x2.toarray()))
    y = coo1.dot(x)

    assert np.allclose( y, yref, rtol=1e-12, atol=1e-12 )

#===============================================================================
@pytest.mark.parametrize( 'n1', [8,16] )
@pytest.mark.parametrize( 'n2', [8,12] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'p2', [1,2,3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )

def test_block_2d_array_to_stencil_1( n1, n2, p1, p2, P1, P2 ):
    # set seed for reproducibility
    seed(n1*n2*p1*p2)

    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector spaces, and stencil vectors
    V1 = StencilVectorSpace( cart )
    V2 = StencilVectorSpace( cart )

    W = BlockVectorSpace(V1, V2)

    x = BlockVector(W)

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1):
        for i2 in range(n2):
            x[0][i1,i2] = 2.0*random() + 1.0
            x[1][i1,i2] = 5.0*random() - 1.0
    x.update_ghost_regions()

    xa = x.toarray()
    v  = array_to_stencil(xa, W)

    assert np.allclose( xa , v.toarray() )

#===============================================================================
@pytest.mark.parametrize( 'n1', [8,16] )
@pytest.mark.parametrize( 'n2', [8,12] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'p2', [1,2,3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )

def test_block_2d_array_to_stencil_2( n1, n2, p1, p2, P1, P2 ):
    # set seed for reproducibility
    seed(n1*n2*p1*p2)

    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector spaces, and stencil vectors
    V1 = StencilVectorSpace( cart )
    V2 = StencilVectorSpace( cart )

    W = BlockVectorSpace(V1, V2)
    W = BlockVectorSpace(W, W)

    x = BlockVector(W)

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1):
        for i2 in range(n2):
            x[0][0][i1,i2] = 2.0*random() + 1.0
            x[0][1][i1,i2] = 5.0*random() - 1.0
            x[1][0][i1,i2] = 2.0*random() + 1.0
            x[1][1][i1,i2] = 5.0*random() - 1.0
    x.update_ghost_regions()

    xa = x.toarray()
    v  = array_to_stencil(xa, W)

    assert np.allclose( xa , v.toarray() )

#===============================================================================
@pytest.mark.parametrize( 'n1', [8,16] )
@pytest.mark.parametrize( 'n2', [8,32] )
@pytest.mark.parametrize( 'p1', [1,3] )
@pytest.mark.parametrize( 'p2', [1,2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )

def test_block_matrix_operator_dot_backend( n1, n2, p1, p2, P1, P2 ):

    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace( cart )

    M1 = StencilMatrix( V, V , backend=PSYDAC_BACKEND_GPYCCEL)
    M2 = StencilMatrix( V, V , backend=PSYDAC_BACKEND_GPYCCEL)
    M3 = StencilMatrix( V, V , backend=PSYDAC_BACKEND_GPYCCEL)
    M4 = StencilMatrix( V, V , backend=PSYDAC_BACKEND_GPYCCEL)
    x1 = StencilVector( V )
    x2 = StencilVector( V )

    s1,s2 = V.starts
    e1,e2 = V.ends

    # Fill in stencil matrix values based on diagonal index (periodic!)
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            M1[:,:,k1,k2] = k1+k2+10.
            M2[:,:,k1,k2] = 2.*k1+k2
            M3[:,:,k1,k2] = 5*k1+k2
            M4[:,:,k1,k2] = 10*k1+k2

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()
    M3.remove_spurious_entries()
    M4.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(s1,e1+1):
        for i2 in range(s2,e2+1):
            x1[i1,i2] = 2.0*random() + 1.0
            x2[i1,i2] = 5.0*random() - 1.0
    x1.update_ghost_regions()
    x2.update_ghost_regions()

    # Create and Fill Block objects
    W = BlockVectorSpace(V, V)
    L = BlockMatrix( W, W )
    L[0,0] = M1
    L[0,1] = M2
    L[1,0] = M3
    L[1,1] = M4

    L.set_backend(PSYDAC_BACKEND_GPYCCEL)

    X = BlockVector(W)
    X[0] = x1
    X[1] = x2

    # Compute Block-vector product
    Y = L.dot(X)

    # Compute matrix-vector products for each block
    y1 = M1.dot(x1) + M2.dot(x2)
    y2 = M3.dot(x1) + M4.dot(x2)

    # Check data in 1D array
    assert np.allclose( Y.blocks[0].toarray(), y1.toarray(), rtol=1e-13, atol=1e-13 )
    assert np.allclose( Y.blocks[1].toarray(), y2.toarray(), rtol=1e-13, atol=1e-13 )

#===============================================================================
# PARALLEL TESTS
#===============================================================================
@pytest.mark.parametrize( 'n1', [8,16] )
@pytest.mark.parametrize( 'n2', [8,32] )
@pytest.mark.parametrize( 'p1', [1,3] )
@pytest.mark.parametrize( 'p2', [1,2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )
@pytest.mark.parametrize( 'reorder', [True, False] )
@pytest.mark.parallel

def test_block_linear_operator_parallel_dot( n1, n2, p1, p2, P1, P2, reorder ):
    # set seed for reproducibility
    seed(n1*n2*p1*p2)

    from mpi4py       import MPI

    comm = MPI.COMM_WORLD
    D = DomainDecomposition([n1,n2], periods=[P1,P2], comm=comm)

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace( cart )
    M1 = StencilMatrix( V, V )
    M2 = StencilMatrix( V, V )
    M3 = StencilMatrix( V, V )
    M4 = StencilMatrix( V, V )
    x1 = StencilVector( V )
    x2 = StencilVector( V )

    s1,s2 = V.starts
    e1,e2 = V.ends

    # Fill in stencil matrix values based on diagonal index (periodic!)
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            M1[:,:,k1,k2] = k1+k2+10.
            M2[:,:,k1,k2] = 2.*k1+k2
            M3[:,:,k1,k2] = 5*k1+k2
            M4[:,:,k1,k2] = 10*k1+k2

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()
    M3.remove_spurious_entries()
    M4.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(s1,e1+1):
        for i2 in range(s2,e2+1):
            x1[i1,i2] = 2.0*random() + 1.0
            x2[i1,i2] = 5.0*random() - 1.0
    x1.update_ghost_regions()
    x2.update_ghost_regions()

    # Create and Fill Block objects
    W = BlockVectorSpace(V, V)
    L = BlockLinearOperator( W, W )
    L[0,0] = M1
    L[0,1] = M2
    L[1,0] = M3
    L[1,1] = M4

    X = BlockVector(W)
    X[0] = x1
    X[1] = x2

    # Compute Block-vector product
    Y = L.dot(X)

    # Compute matrix-vector products for each block
    y1 = M1.dot(x1) + M2.dot(x2)
    y2 = M3.dot(x1) + M4.dot(x2)

    # Check data in 1D array
    assert np.allclose( Y.blocks[0].toarray(), y1.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Y.blocks[1].toarray(), y2.toarray(), rtol=1e-14, atol=1e-14 )

#===============================================================================
@pytest.mark.parametrize( 'n1', [8,16] )
@pytest.mark.parametrize( 'n2', [8,32] )
@pytest.mark.parametrize( 'p1', [1,3] )
@pytest.mark.parametrize( 'p2', [1,2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )
@pytest.mark.parametrize( 'reorder', [True, False] )
@pytest.mark.parallel
def test_block_diagonal_solver_parallel_dot( n1, n2, p1, p2, P1, P2, reorder  ):
    # set seed for reproducibility
    seed(n1*n2*p1*p2)

    from mpi4py       import MPI
    comm = MPI.COMM_WORLD
    D = DomainDecomposition([n1,n2], periods=[P1,P2], comm=comm)

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector spaces, stencil matrices, and stencil vectors
    V = StencilVectorSpace( cart )

    s1,s2 = V.starts
    e1,e2 = V.ends

    # Fill in stencil matrices based on diagonal index
    m11 = np.zeros((n1, n1))
    m12 = np.zeros((n2, n2))
    for j in range(n1):
        for i in range(-p1,p1+1):
            m11[j, max(0, min(n1-1, j+i))] = 10*j+i
    for j in range(n2):
        for i in range(-p2,p2+1):
            m12[j, max(0, min(n2-1, j+i))] = 20*j+5*i+2.
    
    m21 = np.zeros((n1, n1))
    m22 = np.zeros((n2, n2))
    for j in range(n1):
        for i in range(-p1,p1+1):
            m21[j, max(0, min(n1-1, j+i))] = 10*j**2+i**3
    for j in range(n2):
        for i in range(-p2,p2+1):
            m22[j, max(0, min(n2-1, j+i))] = 20*j**2+i**3+2.
    
    M11 = SparseSolver( spa.csc_matrix(m11) )
    M12 = SparseSolver( spa.csc_matrix(m12) )
    M21 = SparseSolver( spa.csc_matrix(m21) )
    M22 = SparseSolver( spa.csc_matrix(m22) )
    M1 = KroneckerLinearSolver(V, [M11,M12])
    M2 = KroneckerLinearSolver(V, [M21,M22])
    x1 = StencilVector( V )
    x2 = StencilVector( V )

    W = BlockVectorSpace(V, V)

    # Fill in vector with random values, then update ghost regions
    for i1 in range(s1,e1+1):
        for i2 in range(s2,e2+1):
            x1[i1,i2] = 2.0*random() - 1.0
            x2[i1,i2] = 5.0*random() - 1.0
    x1.update_ghost_regions()
    x2.update_ghost_regions()

    # Construct a BlockVector object containing x1 and x2
    #     |x1|
    # X = |  |
    #     |x2|

    X = BlockVector(W)
    X[0] = x1
    X[1] = x2

    # Construct a BlockDiagonalSolver object containing M1, M2 using 3 ways
    #     |M1  0 |
    # L = |      |
    #     |0   M2|

    dict_blocks = {0:M1, 1:M2}
    list_blocks = [M1, M2]

    L1 = BlockDiagonalSolver( W, blocks=dict_blocks )
    L2 = BlockDiagonalSolver( W, blocks=list_blocks )

    L3 = BlockDiagonalSolver( W )

    # Test for not allowing undefinedness
    errresult = False
    try:
        L3.solve(X)
    except NotImplementedError:
        errresult = True
    assert errresult

    L3[0] = M1
    L3[1] = M2

    # Compute BlockDiagonalSolver product
    Y1 = L1.solve(X)
    Y2 = L2.solve(X)
    Y3 = L3.solve(X)

    # Transposed
    Yt = L1.solve(X, transposed=True)

    # Test other in/out methods
    Y4a = W.zeros()
    Y4b = L1.solve(X, out=Y4a)
    assert Y4b is Y4a

    Y5a = W.zeros()
    Y5a[0] = x1.copy()
    Y5a[1] = x2.copy()
    Y5b = L1.solve(Y5a, out=Y5a)
    assert Y5b is Y5a

    # Solve linear equations for each block
    y1 = M1.solve(x1)
    y2 = M2.solve(x2)

    y1t = M1.solve(x1, transposed=True)
    y2t = M2.solve(x2, transposed=True)

    # Check data in 1D array
    assert np.allclose( Y1.blocks[0].toarray(), y1.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Y1.blocks[1].toarray(), y2.toarray(), rtol=1e-14, atol=1e-14 )

    assert np.allclose( Y2.blocks[0].toarray(), y1.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Y2.blocks[1].toarray(), y2.toarray(), rtol=1e-14, atol=1e-14 )

    assert np.allclose( Y3.blocks[0].toarray(), y1.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Y3.blocks[1].toarray(), y2.toarray(), rtol=1e-14, atol=1e-14 )

    assert np.allclose( Y4a.blocks[0].toarray(), y1.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Y4a.blocks[1].toarray(), y2.toarray(), rtol=1e-14, atol=1e-14 )

    assert np.allclose( Y5a.blocks[0].toarray(), y1.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Y5a.blocks[1].toarray(), y2.toarray(), rtol=1e-14, atol=1e-14 )

    assert np.allclose( Yt.blocks[0].toarray(), y1t.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Yt.blocks[1].toarray(), y2t.toarray(), rtol=1e-14, atol=1e-14 )

@pytest.mark.parametrize( 'n1', [8,16] )
@pytest.mark.parametrize( 'n2', [8,32] )
@pytest.mark.parametrize( 'p1', [1,3] )
@pytest.mark.parametrize( 'p2', [1,2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )
@pytest.mark.parametrize( 'reorder', [True, False] )
@pytest.mark.parallel

def test_block_matrix_operator_parallel_dot_backend( n1, n2, p1, p2, P1, P2, reorder ):
    # set seed for reproducibility

    from mpi4py       import MPI

    comm = MPI.COMM_WORLD
    D = DomainDecomposition([n1,n2], periods=[P1,P2], comm=comm)

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace( cart )
    M1 = StencilMatrix( V, V , backend=PSYDAC_BACKEND_GPYCCEL)
    M2 = StencilMatrix( V, V , backend=PSYDAC_BACKEND_GPYCCEL)
    M3 = StencilMatrix( V, V , backend=PSYDAC_BACKEND_GPYCCEL)
    M4 = StencilMatrix( V, V , backend=PSYDAC_BACKEND_GPYCCEL)
    x1 = StencilVector( V )
    x2 = StencilVector( V )

    s1,s2 = V.starts
    e1,e2 = V.ends

    # Fill in stencil matrix values based on diagonal index (periodic!)
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            M1[:,:,k1,k2] = k1+k2+10.
            M2[:,:,k1,k2] = 2.*k1+k2
            M3[:,:,k1,k2] = 5*k1+k2
            M4[:,:,k1,k2] = 10*k1+k2

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()
    M3.remove_spurious_entries()
    M4.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(s1,e1+1):
        for i2 in range(s2,e2+1):
            x1[i1,i2] = 2.0*random() + 1.0
            x2[i1,i2] = 5.0*random() - 1.0
    x1.update_ghost_regions()
    x2.update_ghost_regions()

    # Create and Fill Block objects
    W = BlockVectorSpace(V, V)
    L = BlockMatrix( W, W )
    L[0,0] = M1
    L[0,1] = M2
    L[1,0] = M3
    L[1,1] = M4

    L.set_backend(PSYDAC_BACKEND_GPYCCEL)

    X = BlockVector(W)
    X[0] = x1
    X[1] = x2

    # Compute Block-vector product
    Y = L.dot(X)

    # Compute matrix-vector products for each block
    y1 = M1.dot(x1) + M2.dot(x2)
    y2 = M3.dot(x1) + M4.dot(x2)

    # Check data in 1D array
    assert np.allclose( Y.blocks[0].toarray(), y1.toarray(), rtol=1e-13, atol=1e-13 )
    assert np.allclose( Y.blocks[1].toarray(), y2.toarray(), rtol=1e-13, atol=1e-13 )

#===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================
if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
