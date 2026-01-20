#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import pytest
import numpy as np
from scipy.sparse import csr_matrix

from psydac.linalg.direct_solvers import SparseSolver
from psydac.linalg.stencil        import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.block          import BlockVectorSpace, BlockVector
from psydac.linalg.block          import BlockLinearOperator
from psydac.linalg.utilities      import array_to_psydac, petsc_to_psydac
from psydac.linalg.sparse         import SparseMatrixLinearOperator
from psydac.api.settings          import PSYDAC_BACKEND_GPYCCEL
from psydac.ddm.cart              import DomainDecomposition, CartDecomposition

#===============================================================================
def compute_global_starts_ends(domain_decomposition, npts):
    global_starts = [None]*len(npts)
    global_ends   = [None]*len(npts)

    for axis in range(len(npts)):
        es = domain_decomposition.global_element_starts[axis]
        ee = domain_decomposition.global_element_ends  [axis]

        global_ends  [axis]     = ee.copy()
        global_ends  [axis][-1] = npts[axis]-1
        global_starts[axis]     = np.array([0] + (global_ends[axis][:-1]+1).tolist())

    return global_starts, global_ends

#===============================================================================
# SERIAL TESTS
#===============================================================================
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'n1', [8, 16] )
@pytest.mark.parametrize( 'n2', [8, 12] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [1, 3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True] )

def test_2D_block_vector_space_serial_init( dtype, n1, n2, p1, p2, P1, P2  ):

    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector spaces, stencil matrices, and stencil vectors
    V = StencilVectorSpace( cart, dtype=dtype )
    W = BlockVectorSpace(V, V)

    assert W.dimension == 2*n1*n2
    assert W.dtype == dtype
    assert W.spaces == (V,V)
    assert W.parallel == False
    assert W.starts == [(0,0),(0,0)]
    assert W.ends == [(n1-1,n2-1),(n1-1,n2-1)]
    assert W.pads == (p1,p2)
    assert W.n_blocks == 2
    assert W.connectivity== {}

#===============================================================================
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'n1', [8, 16] )
@pytest.mark.parametrize( 'n2', [8, 12] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [1, 3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True] )

def test_2D_block_vector_serial_init( dtype, n1, n2, p1, p2, P1, P2  ):

    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector spaces, stencil matrices, and stencil vectors
    V = StencilVectorSpace( cart, dtype=dtype )
    x1 = StencilVector( V )
    x2 = StencilVector( V )

    W = BlockVectorSpace(V, V)
    x = BlockVector(W, blocks=[x1,x2])
    assert x.dtype == dtype
    assert x.space == W
    assert x.n_blocks == 2
    assert x.blocks == (x1, x2)

#===============================================================================
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'n1', [8, 16] )
@pytest.mark.parametrize( 'n2', [8, 12] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [1, 3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True] )

def test_2D_block_linear_operator_serial_init( dtype, n1, n2, p1, p2, P1, P2  ):

    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector spaces, stencil matrices, and stencil vectors
    V = StencilVectorSpace( cart, dtype=dtype )
    M1 = StencilMatrix( V, V)
    M2 = StencilMatrix( V, V )
    M3 = StencilMatrix( V, V )
    x1 = StencilVector( V )
    x2 = StencilVector( V )

    # Fill in stencil matrices based on diagonal index
    if dtype==complex:
        f=lambda k1,k2: 10j*k1+k2
    else:
        f=lambda k1,k2: 10*k1+k2

    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            M1[:,:,k1,k2] = f(k1,k2)
            M2[:,:,k1,k2] = f(k1,k2)+2.
            M3[:,:,k1,k2] = f(k1,k2)+5.

    M1.remove_spurious_entries()
    M2.remove_spurious_entries()
    M3.remove_spurious_entries()

    # Initialize random number generator (set seed for reproducibility)
    rng = np.random.default_rng(seed=n1*n2*p1*p2)

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1):
        for i2 in range(n2):
            x1[i1,i2] = 2.0*rng.random() - 1.0
            x2[i1,i2] = 5.0*rng.random() - 1.0
    x1.update_ghost_regions()
    x2.update_ghost_regions()

    W = BlockVectorSpace(V, V)

    # Construct a BlockLinearOperator object containing M1, M2, M, using 3 ways
    #     |M1  M2|
    # L = |      |
    #     |M3  0 |

    dict_blocks = {(0,0):M1, (0,1):M2, (1,0):M3}
    list_blocks = [[M1, M2], [M3, None]]

    L1 = BlockLinearOperator( W, W, blocks=dict_blocks )
    assert L1.domain == W
    assert L1.codomain == W
    assert L1.dtype == dtype
    assert L1.blocks == ((M1,M2),(M3,None))
    assert L1.n_block_rows == 2
    assert L1.n_block_cols == 2
    assert L1.nonzero_block_indices == ((0,0),(0,1),(1,0))
    assert L1.backend()==None

    # Test copy method with an out
    L4 = BlockLinearOperator( W, W )
    L1.copy(out=L4)
    assert L1.domain == W
    assert L1.codomain == W
    assert L1.dtype == dtype
    assert L1.n_block_rows == 2
    assert L1.n_block_cols == 2
    assert L1.nonzero_block_indices == ((0,0),(0,1),(1,0))
    assert L1.backend()==None

    L2 = BlockLinearOperator( W, W, blocks=list_blocks )
    L3 = BlockLinearOperator( W, W )

    L3[0,0] = M1
    L3[0,1] = M2
    L3[1,0] = M3

    # Convert L1, L2, L3 and L4 to COO form
    coo1 = L1.tosparse().tocoo()
    coo2 = L2.tosparse().tocoo()
    coo3 = L3.tosparse().tocoo()
    coo4 = L4.tosparse().tocoo()

    # Check if the data are in the same place
    assert np.array_equal( coo1.col , coo2.col  )
    assert np.array_equal( coo1.row , coo2.row  )
    assert np.array_equal( coo1.data, coo2.data )

    assert np.array_equal( coo1.col , coo3.col  )
    assert np.array_equal( coo1.row , coo3.row  )
    assert np.array_equal( coo1.data, coo3.data )

    assert np.array_equal( coo1.col , coo4.col  )
    assert np.array_equal( coo1.row , coo4.row  )
    assert np.array_equal( coo1.data, coo4.data )
    
    dict_blocks = {(0,0):M1, (0,1):M2}

    L1 = BlockLinearOperator( W, W, blocks=dict_blocks )

    # Test transpose
    LT1 = L1.transpose()
    LT2 = BlockLinearOperator( W, W )
    L1.transpose(out=LT2)

    assert LT1.domain == W
    assert LT1.codomain == W
    assert LT1.dtype == dtype
    assert LT1.n_block_rows == 2
    assert LT1.n_block_cols == 2
    assert LT1.nonzero_block_indices == ((0,0),(1,0))
    assert LT1.backend()==None

    assert LT2.domain == W
    assert LT2.codomain == W
    assert LT2.dtype == dtype
    assert LT2.n_block_rows == 2
    assert LT2.n_block_cols == 2
    assert LT2.nonzero_block_indices == ((0,0),(1,0))
    assert LT2.backend()==None

    #convert to scipy for tests
    LT1_sp = LT1.tosparse()
    LT2_sp = LT2.tosparse()
    L1_spT  = L1.tosparse().T

    # Check if the data are in the same place
    assert abs(LT1_sp - L1_spT).max()< 1e-14
    assert abs(LT2_sp - L1_spT).max()< 1e-14

#===============================================================================
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'ndim', [1, 2, 3] )
@pytest.mark.parametrize( 'p', [1, 2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )
@pytest.mark.parametrize( 'P3', [True] )
def test_block_serial_dimension( ndim, p, P1, P2, P3, dtype ):

    if ndim == 1:
        npts = [12]
        ps = [p]
        Ps = [P1]
        shifts = [1]

    elif ndim == 2:
        npts =[12, 15]
        ps = [p, p]
        Ps = [P1, P2]
        shifts = [1, 1]

    else:
        npts = [12, 15, 9]
        ps = [p, p, p]
        Ps = [P1, P2, P3]
        shifts = [1, 1, 1]

    # set seed for reproducibility
    D = DomainDecomposition(npts, periods=Ps)

    # Partition the points
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=ps, shifts=shifts)

    # Create vector spaces, stencil matrices, and stencil vectors
    V = StencilVectorSpace( cart, dtype=dtype)
    if dtype==complex:
        cst=1j
    else:
        cst=1

    x1 = StencilVector( V )
    x2 = StencilVector( V )
    y1 = StencilVector( V )
    y2 = StencilVector( V )

    W = BlockVectorSpace(V, V)

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in vector with random values, then update ghost regions
    if ndim==1:
        x1[:] = cst*2.0*rng.random((npts[0]+2*p))
        x2[:] = cst*5.0*rng.random((npts[0]+2*p))
        y1[:] = cst*2.0*rng.random((npts[0]+2*p))
        y2[:] = cst*3.0*rng.random((npts[0]+2*p))

    elif ndim==2:
        x1[:,:] = cst*2.0*rng.random((npts[0]+2*p,npts[1]+2*p))
        x2[:,:] = cst*5.0*rng.random((npts[0]+2*p,npts[1]+2*p))
        y1[:,:] = cst*2.0*rng.random((npts[0]+2*p,npts[1]+2*p))
        y2[:,:] = cst*3.0*rng.random((npts[0]+2*p,npts[1]+2*p))

    else:
        x1[:,:,:] = cst*2.0*rng.random((npts[0]+2*p,npts[1]+2*p,npts[2]+2*p))
        x2[:,:,:] = cst*5.0*rng.random((npts[0]+2*p,npts[1]+2*p,npts[2]+2*p))
        y1[:,:,:] = cst*2.0*rng.random((npts[0]+2*p,npts[1]+2*p,npts[2]+2*p))
        y2[:,:,:] = cst*3.0*rng.random((npts[0]+2*p,npts[1]+2*p,npts[2]+2*p))

    x1.update_ghost_regions()
    x2.update_ghost_regions()
    y1.update_ghost_regions()
    y2.update_ghost_regions()

    # Construct a BlockVector object containing x1 and x2
    #     |x1|
    # X = |  |
    #     |x2|

    X = BlockVector(W)
    X[0] = x1
    X[1] = x2

    Y = BlockVector(W)
    Y[0] = y1
    Y[1] = y2

    # Test dot product
    exact_inner = V.inner(x1, y1) + V.inner(x2, y2)

    assert X.dtype == dtype
    assert np.allclose(W.inner(X, Y), exact_inner,  rtol=1e-14, atol=1e-14 )

    # Test axpy product
    axpy_exact = X + np.pi * cst * Y
    X.mul_iadd(np.pi * cst, Y)
    assert np.allclose(X[0]._data, axpy_exact[0]._data,  rtol=1e-10, atol=1e-10 )
    assert np.allclose(X[1]._data, axpy_exact[1]._data,  rtol=1e-10, atol=1e-10 )

    M1 = StencilMatrix(V, V)
    M2 = StencilMatrix(V, V)
    M3 = StencilMatrix(V, V)

    # Fill in stencil matrices based on diagonal index
    if ndim==1:
        f = lambda k1: 10 * k1
        for k1 in range(-ps[0], ps[0] + 1):
                M1[:, k1] = f(k1)
                M2[:, k1] = f(k1) + 2.
                M3[:, k1] = f(k1) + 5.
    if ndim==2:
        f = lambda k1,k2: 10 * k1 + 100*k2
        for k1 in range(-ps[0], ps[0] + 1):
            for k2 in range(-ps[1], ps[1] + 1):
                M1[:, k1] = f(k1,k2)
                M2[:, k1] = f(k1,k2) + 2.
                M3[:, k1] = f(k1,k2) + 5.
    if ndim==3:
        f = lambda k1, k2, k3: 10 * k1 + 100*k2+1000*k3
        for k1 in range(-ps[0], ps[0] + 1):
            for k2 in range(-ps[1], ps[1] + 1):
                for k3 in range(-ps[1], ps[1] + 1):
                    M1[:, k1] = f(k1,k2,k3)
                    M2[:, k1] = f(k1,k2,k3) + 2.
                    M3[:, k1] = f(k1,k2,k3) + 5.

    M1.remove_spurious_entries()
    M2.remove_spurious_entries()
    M3.remove_spurious_entries()

    M = BlockLinearOperator(W, W, blocks=[[M1, M2], [M3, None]])

    Y[0] = M1.dot(x1) + M2.dot(x2)
    Y[1] = M3.dot(x1)

    assert M.dtype == dtype
    assert np.allclose((M.dot(X)).toarray(), Y.toarray(),  rtol=1e-14, atol=1e-14 )

#===============================================================================
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'npts', [[6, 8, 9]] )
@pytest.mark.parametrize( 'p', [[1,1,1], [2,3,4]] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )
@pytest.mark.parametrize( 'P3', [True] )
def test_3D_block_serial_basic_operator( dtype, npts, p, P1, P2, P3 ):

    # set seed for reproducibility
    D = DomainDecomposition(npts, periods=[P1,P2,P3])

    # Partition the points
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=p, shifts=[1,1,1])

    # Create vector spaces, stencil matrices, and stencil vectors
    V = StencilVectorSpace( cart, dtype=dtype)

    x1 = StencilVector( V )
    x2 = StencilVector( V )
    y1 = StencilVector( V )
    y2 = StencilVector( V )

    W = BlockVectorSpace(V, V)

    # Initialize random number generator
    rng = np.random.default_rng()

    if dtype==complex:
        x1[:,:,:] = 2.0*rng.random((npts[0]+2*p[0],npts[1]+2*p[1],npts[2]+2*p[2]))+1j*rng.random((npts[0]+2*p[0],npts[1]+2*p[1],npts[2]+2*p[2]))
        x2[:,:,:] = 5.0*rng.random((npts[0]+2*p[0],npts[1]+2*p[1],npts[2]+2*p[2]))+2j*rng.random((npts[0]+2*p[0],npts[1]+2*p[1],npts[2]+2*p[2]))
    else:
        x1[:,:,:] = 2.0*rng.random((npts[0]+2*p[0],npts[1]+2*p[1],npts[2]+2*p[2]))
        x2[:,:,:] = 5.0*rng.random((npts[0]+2*p[0],npts[1]+2*p[1],npts[2]+2*p[2]))

    x1.update_ghost_regions()
    x2.update_ghost_regions()
    y1.update_ghost_regions()
    y2.update_ghost_regions()

    # Construct a BlockVector object containing x1 and x2
    #     |x1|
    # X = |  |
    #     |x2|

    X = BlockVector(W)
    X[0] = x1
    X[1] = x2

    Y = BlockVector(W)

    Y +=X
    assert Y.dtype == dtype
    assert np.allclose(Y.blocks[0]._data, (x1)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(Y.blocks[1]._data, (x2)._data,  rtol=1e-14, atol=1e-14 )

    Y -=2*X
    assert np.allclose(Y.blocks[0]._data, -(x1)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(Y.blocks[1]._data, -(x2)._data,  rtol=1e-14, atol=1e-14 )

    Y *=6
    assert np.allclose(Y.blocks[0]._data, -6*(x1)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(Y.blocks[1]._data, -6*(x2)._data,  rtol=1e-14, atol=1e-14 )

    Y /=-6
    assert np.allclose(Y.blocks[0]._data, (x1)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(Y.blocks[1]._data, (x2)._data,  rtol=1e-14, atol=1e-14 )

    Y[0]=x2
    Y[1]=-x1

    Z1=Y+X
    assert isinstance(Z1,BlockVector)
    assert np.allclose(Z1.blocks[0]._data, (x1+x2)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(Z1.blocks[1]._data, (x2-x1)._data,  rtol=1e-14, atol=1e-14 )

    Z2=Y-X
    assert isinstance(Z2,BlockVector)
    assert np.allclose(Z2.blocks[0]._data, (x2-x1)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(Z2.blocks[1]._data, (-x2-x1)._data,  rtol=1e-14, atol=1e-14 )

    Z3=3*Y
    assert isinstance(Z3,BlockVector)
    assert np.allclose(Z3.blocks[0]._data, 3*(x2)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(Z3.blocks[1]._data, 3*(-x1)._data,  rtol=1e-14, atol=1e-14 )

    Z4=Y/4
    assert isinstance(Z4,BlockVector)
    assert np.allclose(Z4.blocks[0]._data, (x2)._data/4,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(Z4.blocks[1]._data, (-x1)._data/4,  rtol=1e-14, atol=1e-14 )

    M1 = StencilMatrix(V, V)
    M2 = StencilMatrix(V, V)
    M3 = StencilMatrix(V, V)

    if dtype==complex:
        f = lambda k1, k2, k3: 10 * k1 + 100j*k2+1000*k3
    else:
        f = lambda k1, k2, k3: 10 * k1 + 100*k2+1000*k3

    for k1 in range(-p[0], p[0] + 1):
        for k2 in range(-p[1], p[1] + 1):
            for k3 in range(-p[1], p[1] + 1):
                M1[:, k1] = f(k1,k2,k3)
                M2[:, k1] = f(k1,k2,k3) + 2.
                M3[:, k1] = f(k1,k2,k3) + 5.

    M1.remove_spurious_entries()
    M2.remove_spurious_entries()
    M3.remove_spurious_entries()

    M = BlockLinearOperator(W, W, blocks=[[M1, M2], [M3, None]])
    A = BlockLinearOperator(W, W)

    A +=M
    assert A.dtype == dtype
    assert np.allclose(A.blocks[0][0]._data, (M1)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(A.blocks[0][1]._data, (M2)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(A.blocks[1][0]._data, (M3)._data,  rtol=1e-14, atol=1e-14 )
    assert A.blocks[1][1]==None

    A -= 2*M
    assert np.allclose(A.blocks[0][0]._data, -(M1)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(A.blocks[0][1]._data, -(M2)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(A.blocks[1][0]._data, -(M3)._data,  rtol=1e-14, atol=1e-14 )
    assert A.blocks[1][1]==None

    A *= 5
    assert np.allclose(A.blocks[0][0]._data, -5*(M1)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(A.blocks[0][1]._data, -5*(M2)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(A.blocks[1][0]._data, -5*(M3)._data,  rtol=1e-14, atol=1e-14 )
    assert A.blocks[1][1]==None

    A /= -5
    assert np.allclose(A.blocks[0][0]._data, (M1)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(A.blocks[0][1]._data, (M2)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(A.blocks[1][0]._data, (M3)._data,  rtol=1e-14, atol=1e-14 )
    assert A.blocks[1][1]==None

    A= BlockLinearOperator(W, W, blocks=[[None, M3], [M2, M1]])

    A1=A+M
    assert isinstance(A1,BlockLinearOperator)
    assert np.allclose(A1.blocks[0][0]._data, (M1)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(A1.blocks[0][1]._data, (M2+M3)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(A1.blocks[1][0]._data, (M3+M2)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(A1.blocks[1][1]._data, (M1)._data,  rtol=1e-14, atol=1e-14 )

    A2=A-M
    assert isinstance(A2,BlockLinearOperator)
    assert np.allclose(A2.blocks[0][0]._data, (-M1)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(A2.blocks[0][1]._data, (M3-M2)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(A2.blocks[1][0]._data, (M2-M3)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(A2.blocks[1][1]._data, (M1)._data,  rtol=1e-14, atol=1e-14 )

    A3=6*A
    assert isinstance(A3,BlockLinearOperator)
    assert np.allclose(A3.blocks[0][1]._data, 6*(M3)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(A3.blocks[1][0]._data, 6*(M2)._data,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(A3.blocks[1][1]._data, 6*(M1)._data,  rtol=1e-14, atol=1e-14 )

    A4=A/5
    assert isinstance(A4,BlockLinearOperator)
    assert np.allclose(A4.blocks[0][1]._data, (M3)._data/5,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(A4.blocks[1][0]._data, (M2)._data/5,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(A4.blocks[1][1]._data, (M1)._data/5,  rtol=1e-14, atol=1e-14 )

#===============================================================================
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'npts', [[6, 8]] )
@pytest.mark.parametrize( 'p', [[1,1], [2,3]] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True] )
def test_2D_block_serial_math( dtype, npts, p, P1, P2 ):

    # set seed for reproducibility
    D = DomainDecomposition(npts, periods=[P1,P2])

    # Partition the points
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=p, shifts=[1,1])

    # Create vector spaces, stencil matrices, and stencil vectors
    V = StencilVectorSpace( cart, dtype=dtype)

    x1 = StencilVector( V )
    x2 = StencilVector( V )

    W = BlockVectorSpace(V, V)

    # Initialize random number generator
    rng = np.random.default_rng()

    if dtype==complex:
        x1[:,:,:] = 2.0*rng.random((npts[0]+2*p[0],npts[1]+2*p[1]))+1j*rng.random((npts[0]+2*p[0],npts[1]+2*p[1]))
        x2[:,:,:] = 5.0*rng.random((npts[0]+2*p[0],npts[1]+2*p[1]))+2j*rng.random((npts[0]+2*p[0],npts[1]+2*p[1]))
    else:
        x1[:,:,:] = 2.0*rng.random((npts[0]+2*p[0],npts[1]+2*p[1]))
        x2[:,:,:] = 5.0*rng.random((npts[0]+2*p[0],npts[1]+2*p[1]))

    x1.update_ghost_regions()
    x2.update_ghost_regions()

    x1a = x1.toarray().conj()
    x2a = x2.toarray().conj()

    # Construct a BlockVector object containing x1 and x2
    #     |x1|
    # X = |  |
    #     |x2|

    X = BlockVector(W)
    X[0] = x1
    X[1] = x2

    Xc=X.conjugate()
    assert np.allclose(Xc.blocks[0].toarray(), x1a, rtol=1e-14, atol=1e-14 )
    assert np.allclose(Xc.blocks[1].toarray(), x2a, rtol=1e-14, atol=1e-14 )

    Xc=X.conj()
    assert np.allclose(Xc.blocks[0].toarray(), x1a, rtol=1e-14, atol=1e-14 )
    assert np.allclose(Xc.blocks[1].toarray(), x2a, rtol=1e-14, atol=1e-14 )


    M1 = StencilMatrix(V, V)
    M2 = StencilMatrix(V, V)
    M3 = StencilMatrix(V, V)

    if dtype==complex:
        f = lambda k1, k2: 10 * k1 + 100j*k2
    else:
        f = lambda k1, k2: 10 * k1 + 100*k2

    for k1 in range(-p[0], p[0] + 1):
        for k2 in range(-p[1], p[1] + 1):
            M1[:, k1] = f(k1,k2)
            M2[:, k1] = f(k1,k2) + 2.
            M3[:, k1] = f(k1,k2) + 5.

    M1.remove_spurious_entries()
    M2.remove_spurious_entries()
    M3.remove_spurious_entries()

    M1a = M1.toarray().conjugate()
    M2a = M2.toarray().conjugate()
    M3a = M3.toarray().conjugate()

    # Construct a BlockLinearOperator object containing M1, M2, M3
    #     |M1   M2|
    # M = |       |
    #     |M3    0|

    M = BlockLinearOperator(W, W, blocks=[[M1, M2], [M3, None]])

    Mc = M.conjugate()
    assert np.allclose(Mc.blocks[0][0].toarray(), M1a,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(Mc.blocks[0][1].toarray(), M2a,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(Mc.blocks[1][0].toarray(), M3a,  rtol=1e-14, atol=1e-14 )

    Mc = M.conj()
    assert np.allclose(Mc.blocks[0][0].toarray(), M1a,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(Mc.blocks[0][1].toarray(), M2a,  rtol=1e-14, atol=1e-14 )
    assert np.allclose(Mc.blocks[1][0].toarray(), M3a,  rtol=1e-14, atol=1e-14 )

#===============================================================================
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'n1', [8, 16] )
@pytest.mark.parametrize( 'n2', [8, 12] )
@pytest.mark.parametrize( 'p1', [1, 3] )
@pytest.mark.parametrize( 'p2', [1, 2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True] )

def test_block_linear_operator_serial_dot( dtype, n1, n2, p1, p2, P1, P2  ):

    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector spaces, stencil matrices, and stencil vectors
    V = StencilVectorSpace( cart, dtype=dtype )
    M1 = StencilMatrix( V, V)
    M2 = StencilMatrix( V, V )
    M3 = StencilMatrix( V, V )
    x1 = StencilVector( V )
    x2 = StencilVector( V )

    # Fill in stencil matrices based on diagonal index
    if dtype==complex:
        f=lambda k1,k2: 10j*k1+k2
    else:
        f=lambda k1,k2: 10*k1+k2

    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            M1[:,:,k1,k2] = f(k1,k2)
            M2[:,:,k1,k2] = f(k1,k2)+2.
            M3[:,:,k1,k2] = f(k1,k2)+5.

    M1.remove_spurious_entries()
    M2.remove_spurious_entries()
    M3.remove_spurious_entries()

    # Initialize random number generator (set seed for reproducibility)
    rng = np.random.default_rng(seed=n1*n2*p1*p2)

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1):
        for i2 in range(n2):
            x1[i1,i2] = 2.0*rng.random() - 1.0
            x2[i1,i2] = 5.0*rng.random() - 1.0
    x1.update_ghost_regions()
    x2.update_ghost_regions()

    W = BlockVectorSpace(V, V)

    # Construct a BlockLinearOperator object containing M1, M2, M, using 3 ways
    #     |M1  M2|
    # L = |      |
    #     |M3  0 |

    dict_blocks = {(0,0):M1, (0,1):M2, (1,0):M3}

    L = BlockLinearOperator( W, W, blocks=dict_blocks )

    # Construct a BlockVector object containing x1 and x2
    #     |x1|
    # X = |  |
    #     |x2|

    X = BlockVector(W)
    X[0] = x1
    X[1] = x2

    # Compute BlockLinearOperator product
    Y = L.dot(X)

    # Compute matrix-vector products for each block
    y1 = M1.dot(x1) + M2.dot(x2)
    y2 = M3.dot(x1)

    # Check data in 1D array
    assert np.allclose( Y.blocks[0].toarray(), y1.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Y.blocks[1].toarray(), y2.toarray(), rtol=1e-14, atol=1e-14 )
#===============================================================================
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'n1', [8, 16] )
@pytest.mark.parametrize( 'n2', [8, 12] )
@pytest.mark.parametrize( 'p1', [1, 3] )
@pytest.mark.parametrize( 'p2', [1, 2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True] )

def test_sparse_matrix_linear_operator_serial_dot( dtype, n1, n2, p1, p2, P1, P2  ):

    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector spaces, stencil matrices, and stencil vectors
    V = StencilVectorSpace( cart, dtype=dtype )
    M1 = StencilMatrix( V, V)
    M2 = StencilMatrix( V, V )
    M3 = StencilMatrix( V, V )
    x1 = StencilVector( V )
    x2 = StencilVector( V )

    # Fill in stencil matrices based on diagonal index
    if dtype==complex:
        f=lambda k1,k2: 10j*k1+k2
    else:
        f=lambda k1,k2: 10*k1+k2

    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            M1[:,:,k1,k2] = f(k1,k2)
            M2[:,:,k1,k2] = f(k1,k2)+2.
            M3[:,:,k1,k2] = f(k1,k2)+5.

    M1.remove_spurious_entries()
    M2.remove_spurious_entries()
    M3.remove_spurious_entries()

    # Initialize random number generator (set seed for reproducibility)
    rng = np.random.default_rng(seed=n1*n2*p1*p2)

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1):
        for i2 in range(n2):
            x1[i1,i2] = 2.0*rng.random() - 1.0
            x2[i1,i2] = 5.0*rng.random() - 1.0
    x1.update_ghost_regions()
    x2.update_ghost_regions()

    W = BlockVectorSpace(V, V)

    # Construct a BlockLinearOperator object containing M1, M2, M, using 3 ways
    #     |M1  M2|
    # L = |      |
    #     |M3  0 |

    dict_blocks = {(0,0):M1, (0,1):M2, (1,0):M3}

    L = BlockLinearOperator( W, W, blocks=dict_blocks )
    Lm = SparseMatrixLinearOperator(W, W, L.tosparse().tocsr())

    # Construct a BlockVector object containing x1 and x2
    #     |x1|
    # X = |  |
    #     |x2|

    X = BlockVector(W)
    X[0] = x1
    X[1] = x2

    # Compute BlockLinearOperator product
    Y = L.dot(X)

    Ym = Lm.dot(X)

    # Check data in 1D array
    assert np.allclose( Ym.toarray(), Y.toarray(), rtol=1e-12, atol=1e-12 )
#===============================================================================
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'n1', [8, 16] )
@pytest.mark.parametrize( 'n2', [8, 12] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [1, 3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True] )

def test_block_2d_serial_array_to_psydac( dtype, n1, n2, p1, p2, P1, P2 ):
    #Define a factor for the data
    if dtype==complex:
        factor=1j
    else:
        factor=1

    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector spaces, and stencil vectors
    V1 = StencilVectorSpace( cart ,dtype=dtype)
    V2 = StencilVectorSpace( cart ,dtype=dtype)

    W = BlockVectorSpace(V1, V2)
    W2 = BlockVectorSpace(W, W)

    x = BlockVector(W)
    x2 = BlockVector(W2)

    # Initialize random number generator (set seed for reproducibility)
    rng = np.random.default_rng(seed=n1*n2*p1*p2)

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1):
        for i2 in range(n2):
            x[0][i1,i2] = 2.0*factor*rng.random() + 1.0
            x[1][i1,i2] = 5.0*factor*rng.random() - 1.0
            x2[0][0][i1,i2] = 2.0*factor*rng.random() + 1.0
            x2[0][1][i1,i2] = 5.0*factor*rng.random() - 1.0
            x2[1][0][i1,i2] = 2.0*factor*rng.random() + 1.0
            x2[1][1][i1,i2] = 5.0*factor*rng.random() - 1.0
    x.update_ghost_regions()
    x2.update_ghost_regions()

    xa = x.toarray()
    x2a = x2.toarray()
    v  = array_to_psydac(xa, W)
    v2  = array_to_psydac(x2a, W2)

    assert np.allclose( xa , v.toarray() )
    assert np.allclose( x2a , v2.toarray() )

#===============================================================================
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'n1', [8, 16] )
@pytest.mark.parametrize( 'n2', [8, 12] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [1, 3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True] )
@pytest.mark.petsc

def test_block_vector_2d_serial_topetsc( dtype, n1, n2, p1, p2, P1, P2 ):
    #Define a factor for the data
    if dtype==complex:
        factor=1j
    else:
        factor=1

    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector spaces, and stencil vectors
    V1 = StencilVectorSpace( cart ,dtype=dtype)
    V2 = StencilVectorSpace( cart ,dtype=dtype)

    W = BlockVectorSpace(V1, V2)

    x = BlockVector(W)

    # Initialize random number generator (set seed for reproducibility)
    rng = np.random.default_rng(seed=n1*n2*p1*p2)

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1):
        for i2 in range(n2):
            x[0][i1,i2] = 2.0*factor*rng.random() + 1.0
            x[1][i1,i2] = 5.0*factor*rng.random() - 1.0

    x.update_ghost_regions()

    v = x.topetsc()
    v = petsc_to_psydac(v, W)

    # The vectors can only be compared in the serial case
    assert np.allclose( x.toarray() , v.toarray() )

#===============================================================================
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'n1', [8, 16] )
@pytest.mark.parametrize( 'n2', [8, 12] )
@pytest.mark.parametrize( 'p1', [1, 3] )
@pytest.mark.parametrize( 'p2', [1, 2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True] )
@pytest.mark.petsc

def test_block_linear_operator_2d_serial_topetsc( dtype, n1, n2, p1, p2, P1, P2  ):

    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector spaces, stencil matrices, and stencil vectors
    V = StencilVectorSpace( cart, dtype=dtype )
    M1 = StencilMatrix( V, V)
    M2 = StencilMatrix( V, V )
    M3 = StencilMatrix( V, V )

    # Fill in stencil matrices based on diagonal index
    if dtype==complex:
        f=lambda k1,k2: 10j*k1+k2
    else:
        f=lambda k1,k2: 10*k1+k2

    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            M1[:,:,k1,k2] = f(k1,k2)
            M2[:,:,k1,k2] = f(k1,k2)+2.
            M3[:,:,k1,k2] = f(k1,k2)+5.

    M1.remove_spurious_entries()
    M2.remove_spurious_entries()
    M3.remove_spurious_entries()

    W = BlockVectorSpace(V, V)

    # Construct a BlockLinearOperator object containing M1, M2, M, using 3 ways
    #     |M1  M2|
    # L = |      |
    #     |M3  0 |

    dict_blocks = {(0,0):M1, (0,1):M2, (1,0):M3}

    L = BlockLinearOperator( W, W, blocks=dict_blocks )

    Lp = L.topetsc()
    indptr, indices, data = Lp.getValuesCSR()
    if dtype == float:
        data = data.real #PETSc with installation complex configuration only handles complex dtype
    Lp = csr_matrix((data, indices, indptr), shape=Lp.size)   
    L = L.tosparse().tocsr()

    # The operators can only be compared in the serial case
    assert (L-Lp).data.size == 0

#===============================================================================
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'n1', [8, 16] )
@pytest.mark.parametrize( 'n2', [8, 32] )
@pytest.mark.parametrize( 'p1', [1, 3] )
@pytest.mark.parametrize( 'p2', [2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True] )
@pytest.mark.parametrize( 'backend', [None, PSYDAC_BACKEND_GPYCCEL] )

def test_block_linear_operator_dot_backend( dtype, n1, n2, p1, p2, P1, P2, backend ):
    # Define a factor for the data
    if dtype == complex:
        factor = 1j
    else:
        factor = 1

    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace( cart, dtype=dtype )

    M1 = StencilMatrix( V, V , backend=backend)
    M2 = StencilMatrix( V, V , backend=backend)
    M3 = StencilMatrix( V, V , backend=backend)
    M4 = StencilMatrix( V, V , backend=backend)
    x1 = StencilVector( V )
    x2 = StencilVector( V )

    s1,s2 = V.starts
    e1,e2 = V.ends

    # Fill in stencil matrix values based on diagonal index (periodic!)
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            M1[:,:,k1,k2] = factor*k1+k2+10.
            M2[:,:,k1,k2] = factor*2.*k1+k2
            M3[:,:,k1,k2] = factor*5*k1+k2
            M4[:,:,k1,k2] = factor*10*k1+k2

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()
    M3.remove_spurious_entries()
    M4.remove_spurious_entries()

    # Fill in vector, then update ghost regions
    for i1 in range(s1,e1+1):
        for i2 in range(s2,e2+1):
            x1[i1,i2] = 2.0*factor * i1 + i2
            x2[i1,i2] = 5.0*factor * i2 - i1
    x1.update_ghost_regions()
    x2.update_ghost_regions()

    # Create and Fill Block objects
    W = BlockVectorSpace(V, V)
    L = BlockLinearOperator( W, W )
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
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'n1', [8, 16] )
@pytest.mark.parametrize( 'n2', [8, 32] )
@pytest.mark.parametrize( 'p1', [1, 3] )
@pytest.mark.parametrize( 'p2', [2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True] )
@pytest.mark.mpi

def test_block_linear_operator_parallel_dot( dtype, n1, n2, p1, p2, P1, P2 ):
    # Define a factor for the data
    if dtype == complex:
        factor = 1j
    else:
        factor = 1

    from mpi4py       import MPI

    comm = MPI.COMM_WORLD
    D = DomainDecomposition([n1,n2], periods=[P1,P2], comm=comm)

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace( cart, dtype=dtype )
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
            M1[:,:,k1,k2] = factor*k1+k2+10.
            M2[:,:,k1,k2] = factor*2.*k1+k2
            M3[:,:,k1,k2] = factor*5*k1+k2
            M4[:,:,k1,k2] = factor*10*k1+k2

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()
    M3.remove_spurious_entries()
    M4.remove_spurious_entries()

    # Initialize random number generator (set seed for reproducibility)
    rng = np.random.default_rng(seed=n1*n2*p1*p2)

    # Fill in vector with random values, then update ghost regions
    for i1 in range(s1,e1+1):
        for i2 in range(s2,e2+1):
            x1[i1,i2] = 2.0*factor*rng.random() + 1.0
            x2[i1,i2] = 5.0*factor*rng.random() - 1.0
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

    # Test copy with an out 
    # Create random matrix 
    N1 = StencilMatrix( V, V )
    N2 = StencilMatrix( V, V )
    N3 = StencilMatrix( V, V )
    N4 = StencilMatrix( V, V )

    # Initialize random number generator
    rng = np.random.default_rng()

    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            N1[:,:,k1,k2] = factor*rng.random()
            N2[:,:,k1,k2] = factor*rng.random()
            N3[:,:,k1,k2] = factor*rng.random()
            N4[:,:,k1,k2] = factor*rng.random()

    K = BlockLinearOperator( W, W )
    N = BlockLinearOperator( W, W )

    K[0,0] = N1
    K[0,1] = N2
    K[1,0] = N3
    K[1,1] = N4

    #replace the random entries to check they are really overwritten
    K.copy(out=N)
    L.copy(out=K)

    # Compute Block-vector product
    K.dot(X, out= Y)

    # Check data in 1D array
    assert np.allclose( Y.blocks[0].toarray(), y1.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Y.blocks[1].toarray(), y2.toarray(), rtol=1e-14, atol=1e-14 )

    # Test transpose with an out, check that we overwrite the random entries
    L.transpose(out = N)
    
    # Compute Block-vector product
    Z = N.dot(X)

    # Compute matrix-vector products for each block
    y1 = M1.T.dot(x1) + M3.T.dot(x2)
    y2 = M2.T.dot(x1) + M4.T.dot(x2)

    # Check data in 1D array
    assert np.allclose( Z.blocks[0].toarray(), y1.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Z.blocks[1].toarray(), y2.toarray(), rtol=1e-14, atol=1e-14 )

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [10, 17])
@pytest.mark.parametrize('n2', [13, 7])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [1])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True])
@pytest.mark.mpi

def test_block_vector_2d_parallel_array_to_psydac(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    npts = [n1, n2]   

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # Create domain decomposition
    D = DomainDecomposition(npts, periods=[P1, P2], comm=comm)

    # Partition the points
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create Vector spaces and Vectors
    V = StencilVectorSpace(C, dtype=dtype)
    W = BlockVectorSpace(V, V)
    W2 = BlockVectorSpace(W, W, V)
    x = W.zeros()
    x2 = W2.zeros()

    # Fill the vector with data
    if dtype == complex:
        f = lambda i1, i2: 10j * i1 + i2
    else:
        f = lambda i1, i2: 10 * i1 + i2
    for i1 in range(V.starts[0], V.ends[0]+1):
        for i2 in range(V.starts[1], V.ends[1]+1):
            x[0][i1, i2] = f(i1, i2)
            x[1][i1, i2] = -13*f(i1, i2)
            x2[0] = x
            x2[1] = 2*x
            x2[2][i1, i2] = 7*f(i1, i2)

    x.update_ghost_regions()
    x2.update_ghost_regions()

    # Convert vectors to arrays
    xa = x.toarray()
    x2a = x2.toarray()

    # Apply array_to_psydac as left inverse of toarray
    w = array_to_psydac(xa, W)
    w2 = array_to_psydac(x2a, W2)

    # Initialize random number generator
    rng = np.random.default_rng()

    # Apply array_to_psydac first, and toarray next
    xa_r_inv = rng.random(xa.size) * xa # the vector must be distributed as xa
    if dtype == complex:
        xa_r_inv += 1j * rng.random(xa.size) * xa
    x_r_inv = array_to_psydac(xa_r_inv, W)
    x_r_inv.update_ghost_regions()
    va_r_inv = x_r_inv.toarray()

    x2a_r_inv = rng.random(x2a.size) * x2a # the vector must be distributed as xa
    if dtype == complex:
        x2a_r_inv += 1j * rng.random(x2a.size) * x2a
    x2_r_inv = array_to_psydac(x2a_r_inv, W2)
    x2_r_inv.update_ghost_regions()
    v2a_r_inv = x2_r_inv.toarray()

    ## Check that array_to_psydac is the inverse of .toarray():
    # left inverse:
    assert isinstance(w, BlockVector)
    assert w.space is W
    assert isinstance(w2, BlockVector)
    assert w2.space is W2    
    for i in range(2):
        assert np.array_equal(x[i]._data, w[i]._data)
        for j in range(2):
            assert np.array_equal(x2[i][j]._data, w2[i][j]._data)
            assert np.array_equal(x2[i][j]._data, w2[i][j]._data)

    assert np.array_equal(x2[2]._data, w2[2]._data)

    # right inverse:
    assert np.array_equal(xa_r_inv, va_r_inv)
    assert np.array_equal(x2a_r_inv, v2a_r_inv)

#===============================================================================    
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'n1', [8, 16] )
@pytest.mark.parametrize( 'n2', [8, 12] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [1, 3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True] )
@pytest.mark.mpi
@pytest.mark.petsc

def test_block_vector_2d_parallel_topetsc( dtype, n1, n2, p1, p2, P1, P2 ):
    #Define a factor for the data
    if dtype==complex:
        factor=1j
    else:
        factor=1

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    D = DomainDecomposition([n1,n2], periods=[P1,P2], comm=comm)

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    D2 = DomainDecomposition([n1+1,n2+1], periods=[P1,P2], comm=comm)
    npts2 = [n1+1,n2+1]
    global_starts2, global_ends2 = compute_global_starts_ends(D2, npts2)
    cart2 = CartDecomposition(D2, npts2, global_starts2, global_ends2, pads=[p1+1,p2+1], shifts=[1,1])

    # Create vector spaces, and stencil vectors
    V1 = StencilVectorSpace( cart ,dtype=dtype)
    V2 = StencilVectorSpace( cart2 ,dtype=dtype)

    W = BlockVectorSpace(V1, V2)
    #TODO: implement conversion to PETSc recursively to treat case of block of blocks 

    x = BlockVector(W)

    # Initialize random number generator (set seed for reproducibility)
    rng = np.random.default_rng(seed=n1*n2*p1*p2)

    # Fill in vector with random values, then update ghost regions
    for i0 in range(len(W.starts)):
        for i1 in range(W.starts[i0][0], W.ends[i0][0] + 1):
            for i2 in range(W.starts[i0][1], W.ends[i0][1] + 1):
                x[i0][i1,i2] = 2.0*factor*rng.random() + 1.0

    x.update_ghost_regions()

    v = petsc_to_psydac(x.topetsc(), W)

    assert np.allclose( x.toarray() , v.toarray(), rtol=1e-12, atol=1e-12 )

#=============================================================================== 
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'n1', [8, 16] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.mpi
@pytest.mark.petsc

def test_block_linear_operator_1d_parallel_topetsc( dtype, n1, p1, P1):

    from mpi4py import MPI

    D = DomainDecomposition([n1], periods=[P1], comm=MPI.COMM_WORLD)

    # Partition the points
    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[1])

    # Create vector spaces, stencil matrices, and stencil vectors
    V = StencilVectorSpace( cart, dtype=dtype )
    M1 = StencilMatrix( V, V )
    M2 = StencilMatrix( V, V )

    # Fill in stencil matrices based on diagonal index
    if dtype==complex:
        f=lambda k1: 10j*k1
    else:
        f=lambda k1: 10*k1

    for k1 in range(-p1, p1+1):
        M1[:,k1] = f(k1)
        M2[:,k1] = f(k1)+2.

    M1.remove_spurious_entries()
    M2.remove_spurious_entries()

    W = BlockVectorSpace(V, V)

    # Construct a BlockLinearOperator object containing M1, M2, M3: 
    #     |M1  M2|
    # L = |      |
    #     |M3  0 |

    dict_blocks = {(0,0):M1, (0,1):M2}

    L = BlockLinearOperator( W, V, blocks=dict_blocks )
    x = BlockVector(W)

    # Initialize random number generator (set seed for reproducibility)
    rng = np.random.default_rng(seed=n1*p1)

    # Fill in vector with random values, then update ghost regions
    for i0 in range(len(W.starts)):
        for i1 in range(W.starts[i0][0], W.ends[i0][0] + 1):
            x[i0][i1] = 2.0*rng.random() + (1j if dtype==complex else 1.)
    x.update_ghost_regions()

    y = L.dot(x)

    # Cast operator to PETSc Mat format
    Lp = L.topetsc()

    # Create Vec to allocate the result of the dot product
    y_petsc = Lp.createVecLeft()
    # Compute dot product
    Lp.mult(x.topetsc(), y_petsc)
    # Cast result back to PSYDAC BlockVector format
    y_p = petsc_to_psydac(y_petsc, V)
    
    assert np.allclose(y_p.toarray(), y.toarray(), rtol=1e-12, atol=1e-12)

#===============================================================================    
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'n1', [8, 16] )
@pytest.mark.parametrize( 'n2', [8, 12] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [1, 3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True] )
@pytest.mark.mpi
@pytest.mark.petsc

def test_block_linear_operator_2d_parallel_topetsc( dtype, n1, n2, p1, p2, P1, P2):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    D = DomainDecomposition([n1,n2], periods=[P1,P2], comm=comm)

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector spaces, stencil matrices, and stencil vectors
    V = StencilVectorSpace( cart, dtype=dtype )
    M1 = StencilMatrix( V, V )
    M2 = StencilMatrix( V, V )
    M3 = StencilMatrix( V, V )

    # Fill in stencil matrices based on diagonal index
    if dtype==complex:
        f=lambda k1,k2: 10j*k1+k2
    else:
        f=lambda k1,k2: 10*k1+k2

    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            M1[:,:,k1,k2] = f(k1,k2)
            M2[:,:,k1,k2] = f(k1,k2)+2.
            M3[:,:,k1,k2] = -f(k1,k2)+1.

    M1.remove_spurious_entries()
    M2.remove_spurious_entries()
    M3.remove_spurious_entries()

    W = BlockVectorSpace(V, V)

    # Construct a BlockLinearOperator object containing M1, M2, M3: 
    #     
    # L = |M1  M2|
    #     |M3  0 |

    dict_blocks = {(0,0):M1, (0,1):M2, (1,0):M3}

    L = BlockLinearOperator( W, W, blocks=dict_blocks )
    x = BlockVector(W)

    # Initialize random number generator (set seed for reproducibility)
    rng = np.random.default_rng(seed=n1*n2*p1*p2)

    # Fill in vector with random values, then update ghost regions
    for i0 in range(len(W.starts)):
        for i1 in range(W.starts[i0][0], W.ends[i0][0] + 1):
            for i2 in range(W.starts[i0][1], W.ends[i0][1] + 1):
                x[i0][i1,i2] = 2.0*rng.random() + (1j if dtype==complex else 1.)
    x.update_ghost_regions()

    y = L.dot(x)

    # Cast operator to PETSc Mat format
    Lp = L.topetsc()

    # Create Vec to allocate the result of the dot product
    y_petsc = Lp.createVecLeft()
    # Compute dot product
    Lp.mult(x.topetsc(), y_petsc)
    # Cast result back to PSYDAC BlockVector format
    y_p = petsc_to_psydac(y_petsc, L.codomain)
    
    assert np.allclose(y_p.toarray(), y.toarray(), rtol=1e-12, atol=1e-12)

#===============================================================================

@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'n1', [8, 16] )
@pytest.mark.parametrize( 'n2', [8, 32] )
@pytest.mark.parametrize( 'p1', [1, 3] )
@pytest.mark.parametrize( 'p2', [2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True] )
@pytest.mark.mpi

def test_block_matrix_operator_parallel_dot_backend( dtype, n1, n2, p1, p2, P1, P2 ):
    # Define a factor for the data
    if dtype == complex:
        factor = 1j
    else:
        factor = 1
    # set seed for reproducibility

    from mpi4py       import MPI

    comm = MPI.COMM_WORLD
    D = DomainDecomposition([n1,n2], periods=[P1,P2], comm=comm)

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace( cart, dtype=dtype)
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
            M1[:,:,k1,k2] = k1*factor+k2+10.
            M2[:,:,k1,k2] = 2.*factor*k1+k2
            M3[:,:,k1,k2] = 5*factor*k1+k2
            M4[:,:,k1,k2] = 10*factor*k1+k2

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()
    M3.remove_spurious_entries()
    M4.remove_spurious_entries()

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(s1,e1+1):
        for i2 in range(s2,e2+1):
            x1[i1,i2] = 2.0*factor*rng.random() + 1.0
            x2[i1,i2] = 5.0*factor*rng.random() - 1.0
    x1.update_ghost_regions()
    x2.update_ghost_regions()

    # Create and Fill Block objects
    W = BlockVectorSpace(V, V)
    L = BlockLinearOperator( W, W )
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

    #Test axpy method in parallel
    z3 = X + 5 * factor * Y
    X.mul_iadd(5 * factor, Y)

    # Test exact value and symetry of the scalar product
    assert np.allclose(X[0]._data, z3[0]._data)

    # Check data in 1D array
    assert np.allclose( Y.blocks[0].toarray(), y1.toarray(), rtol=1e-13, atol=1e-13 )
    assert np.allclose( Y.blocks[1].toarray(), y2.toarray(), rtol=1e-13, atol=1e-13 )

#===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================
if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
