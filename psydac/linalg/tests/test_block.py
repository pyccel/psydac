# -*- coding: UTF-8 -*-
#
import pytest
import numpy as np
import scipy.sparse as spa
from random import random, seed

from psydac.linalg.direct_solvers import SparseSolver
from psydac.linalg.stencil        import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.block          import BlockVectorSpace, BlockVector
from psydac.linalg.block          import BlockLinearOperator, BlockDiagonalSolver
from psydac.linalg.utilities      import array_to_psydac
from psydac.linalg.kron           import KroneckerLinearSolver
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
    # set seed for reproducibility
    seed(n1*n2*p1*p2)

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
    # set seed for reproducibility
    seed(n1*n2*p1*p2)

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
    # set seed for reproducibility
    seed(n1*n2*p1*p2)

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

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1):
        for i2 in range(n2):
            x1[i1,i2] = 2.0*random() - 1.0
            x2[i1,i2] = 5.0*random() - 1.0
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

    L2 = BlockLinearOperator( W, W, blocks=list_blocks )
    L3 = BlockLinearOperator( W, W )

    L3[0,0] = M1
    L3[0,1] = M2
    L3[1,0] = M3

    # Convert L1, L2 and L3 to COO form
    coo1 = L1.tosparse().tocoo()
    coo2 = L2.tosparse().tocoo()
    coo3 = L3.tosparse().tocoo()

    # Check if the data are in the same place
    assert np.array_equal( coo1.col , coo2.col  )
    assert np.array_equal( coo1.row , coo2.row  )
    assert np.array_equal( coo1.data, coo2.data )

    assert np.array_equal( coo1.col , coo3.col  )
    assert np.array_equal( coo1.row , coo3.row  )
    assert np.array_equal( coo1.data, coo3.data )
#===============================================================================
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'n1', [8, 16] )
@pytest.mark.parametrize( 'n2', [8, 12] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [1, 3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True] )
def test_2D_block_diagonal_solver_serial_init( dtype, n1, n2, p1, p2, P1, P2  ):
    # set seed for reproducibility
    seed(n1*n2*p1*p2)

    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector spaces, stencil matrices, and stencil vectors
    V = StencilVectorSpace( cart, dtype=dtype)

    # Fill in stencil matrices based on diagonal index
    if dtype==complex:
        f1=lambda k1,k2: 10j*k1+k2
    else:
        f1=lambda k1,k2: 10*k1+k2

    m11 = np.zeros((n1, n1), dtype=dtype)
    m12 = np.zeros((n2, n2), dtype=dtype)
    for j in range(n1):
        for i in range(-p1,p1+1):
            m11[j, max(0, min(n1-1, j+i))] = f1(j,i)
    for j in range(n2):
        for i in range(-p2,p2+1):
            m12[j, max(0, min(n2-1, j+i))] = f1(j,5*i)+2.


    if dtype==complex:
        f2=lambda k1,k2: 10j*k1**2+k2**3
    else:
        f2=lambda k1,k2: 10*k1**2+k2**3

    m21 = np.zeros((n1, n1), dtype=dtype)
    m22 = np.zeros((n2, n2), dtype=dtype)

    for j in range(n1):
        for i in range(-p1,p1+1):
            m21[j, max(0, min(n1-1, j+i))] = f2(j,i)
    for j in range(n2):
        for i in range(-p2,p2+1):
            m22[j, max(0, min(n2-1, j+i))] = f2(j,2*i)+2.

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
    assert L3.space == W
    assert L3.blocks == (M1, M2)
    assert L3.n_blocks == 2
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


    # Fill in vector with random values, then update ghost regions
    if ndim==1:
        x1[:] = cst*2.0*np.random.random((npts[0]+2*p))
        x2[:] = cst*5.0*np.random.random((npts[0]+2*p))
        y1[:] = cst*2.0*np.random.random((npts[0]+2*p))
        y2[:] = cst*3.0*np.random.random((npts[0]+2*p))

    elif ndim==2:
        x1[:,:] = cst*2.0*np.random.random((npts[0]+2*p,npts[1]+2*p))
        x2[:,:] = cst*5.0*np.random.random((npts[0]+2*p,npts[1]+2*p))
        y1[:,:] = cst*2.0*np.random.random((npts[0]+2*p,npts[1]+2*p))
        y2[:,:] = cst*3.0*np.random.random((npts[0]+2*p,npts[1]+2*p))

    else:
        x1[:,:,:] = cst*2.0*np.random.random((npts[0]+2*p,npts[1]+2*p,npts[2]+2*p))
        x2[:,:,:] = cst*5.0*np.random.random((npts[0]+2*p,npts[1]+2*p,npts[2]+2*p))
        y1[:,:,:] = cst*2.0*np.random.random((npts[0]+2*p,npts[1]+2*p,npts[2]+2*p))
        y2[:,:,:] = cst*3.0*np.random.random((npts[0]+2*p,npts[1]+2*p,npts[2]+2*p))

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
    exact_dot = x1.dot(y1)+x2.dot(y2)

    assert X.dtype == dtype
    assert np.allclose(X.dot(Y), exact_dot,  rtol=1e-14, atol=1e-14 )

    # Test axpy product
    axpy_exact = X + np.pi * cst * Y
    X.axpy(Y, np.pi * cst)
    print((X-axpy_exact)[0]._data)
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

    Y[0]=M1.dot(x1)+M2.dot(x2)
    Y[1]=M3.dot(x1)

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
    if dtype==complex:
        x1[:,:,:] = 2.0*np.random.random((npts[0]+2*p[0],npts[1]+2*p[1],npts[2]+2*p[2]))+1j*np.random.random((npts[0]+2*p[0],npts[1]+2*p[1],npts[2]+2*p[2]))
        x2[:,:,:] = 5.0*np.random.random((npts[0]+2*p[0],npts[1]+2*p[1],npts[2]+2*p[2]))+2j*np.random.random((npts[0]+2*p[0],npts[1]+2*p[1],npts[2]+2*p[2]))
    else:
        x1[:,:,:] = 2.0*np.random.random((npts[0]+2*p[0],npts[1]+2*p[1],npts[2]+2*p[2]))
        x2[:,:,:] = 5.0*np.random.random((npts[0]+2*p[0],npts[1]+2*p[1],npts[2]+2*p[2]))


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
    if dtype==complex:
        x1[:,:,:] = 2.0*np.random.random((npts[0]+2*p[0],npts[1]+2*p[1]))+1j*np.random.random((npts[0]+2*p[0],npts[1]+2*p[1]))
        x2[:,:,:] = 5.0*np.random.random((npts[0]+2*p[0],npts[1]+2*p[1]))+2j*np.random.random((npts[0]+2*p[0],npts[1]+2*p[1]))
    else:
        x1[:,:,:] = 2.0*np.random.random((npts[0]+2*p[0],npts[1]+2*p[1]))
        x2[:,:,:] = 5.0*np.random.random((npts[0]+2*p[0],npts[1]+2*p[1]))


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
    # set seed for reproducibility
    seed(n1*n2*p1*p2)

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

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1):
        for i2 in range(n2):
            x1[i1,i2] = 2.0*random() - 1.0
            x2[i1,i2] = 5.0*random() - 1.0
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
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [1, 3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True] )
def test_block_diagonal_solver_serial_dot( dtype, n1, n2, p1, p2, P1, P2 ):
    # set seed for reproducibility
    seed(n1*n2*p1*p2)

    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector spaces, stencil matrices, and stencil vectors
    V = StencilVectorSpace( cart, dtype=dtype)

    # Fill in stencil matrices based on diagonal index
    if dtype==complex:
        f1=lambda k1,k2: 10j*k1+k2
    else:
        f1=lambda k1,k2: 10*k1+k2

    m11 = np.zeros((n1, n1), dtype=dtype)
    m12 = np.zeros((n2, n2), dtype=dtype)
    for j in range(n1):
        for i in range(-p1,p1+1):
            m11[j, max(0, min(n1-1, j+i))] = f1(j,i)
    for j in range(n2):
        for i in range(-p2,p2+1):
            m12[j, max(0, min(n2-1, j+i))] = f1(j,5*i)+2.


    if dtype==complex:
        f2=lambda k1,k2: 10j*k1**2+k2**3
    else:
        f2=lambda k1,k2: 10*k1**2+k2**3

    m21 = np.zeros((n1, n1), dtype=dtype)
    m22 = np.zeros((n2, n2), dtype=dtype)
    for j in range(n1):
        for i in range(-p1,p1+1):
            m21[j, max(0, min(n1-1, j+i))] = f2(j,i)
    for j in range(n2):
        for i in range(-p2,p2+1):
            m22[j, max(0, min(n2-1, j+i))] = f2(j,2*i)+2.
    
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
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'n1', [8, 16] )
@pytest.mark.parametrize( 'n2', [8, 12] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [1, 3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True] )

def test_block_2d_array_to_psydac_1( dtype, n1, n2, p1, p2, P1, P2 ):
    #Define a factor for the data
    if dtype==complex:
        factor=1j
    else:
        factor=1
    # set seed for reproducibility
    seed(n1*n2*p1*p2)

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

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1):
        for i2 in range(n2):
            x[0][i1,i2] = 2.0*factor*random() + 1.0
            x[1][i1,i2] = 5.0*factor*random() - 1.0
    x.update_ghost_regions()

    xa = x.toarray()
    v  = array_to_psydac(xa, W)

    assert np.allclose( xa , v.toarray() )
#===============================================================================
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'n1', [8, 16] )
@pytest.mark.parametrize( 'n2', [8, 12] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True] )

def test_block_2d_array_to_psydac_2( dtype, n1, n2, p1, p2, P1, P2 ):
    # Define a factor for the data
    if dtype == complex:
        factor = 1j
    else:
        factor = 1
    # set seed for reproducibility
    seed(n1*n2*p1*p2)

    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector spaces, and stencil vectors
    V1 = StencilVectorSpace( cart ,dtype=dtype)
    V2 = StencilVectorSpace( cart ,dtype=dtype)

    W = BlockVectorSpace(V1, V2)
    W = BlockVectorSpace(W, W)

    x = BlockVector(W)

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1):
        for i2 in range(n2):
            x[0][0][i1,i2] = 2.0*factor*random() + 1.0
            x[0][1][i1,i2] = 5.0*factor*random() - 1.0
            x[1][0][i1,i2] = 2.0*factor*random() + 1.0
            x[1][1][i1,i2] = 5.0*factor*random() - 1.0
    x.update_ghost_regions()

    xa = x.toarray()
    v  = array_to_psydac(xa, W)

    assert np.allclose( xa , v.toarray() )
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

    # Fill in vector with random values, then update ghost regions
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
@pytest.mark.parallel

def test_block_linear_operator_parallel_dot( dtype, n1, n2, p1, p2, P1, P2 ):
    # Define a factor for the data
    if dtype == complex:
        factor = 1j
    else:
        factor = 1

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

    # Fill in vector with random values, then update ghost regions
    for i1 in range(s1,e1+1):
        for i2 in range(s2,e2+1):
            x1[i1,i2] = 2.0*factor*random() + 1.0
            x2[i1,i2] = 5.0*factor*random() - 1.0
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
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'n1', [8,16] )
@pytest.mark.parametrize( 'n2', [8,32] )
@pytest.mark.parametrize( 'p1', [1,3] )
@pytest.mark.parametrize( 'p2', [2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True] )
@pytest.mark.parallel
def test_block_diagonal_solver_parallel_dot( dtype, n1, n2, p1, p2, P1, P2  ):
    # Define a factor for the data
    if dtype == complex:
        factor = 1j
    else:
        factor = 1
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
    V = StencilVectorSpace( cart, dtype=dtype )

    s1,s2 = V.starts
    e1,e2 = V.ends

    # Fill in stencil matrices based on diagonal index
    m11 = np.zeros((n1, n1),dtype=dtype)
    m12 = np.zeros((n2, n2),dtype=dtype)
    for j in range(n1):
        for i in range(-p1,p1+1):
            m11[j, max(0, min(n1-1, j+i))] = 10*factor*j+i
    for j in range(n2):
        for i in range(-p2,p2+1):
            m12[j, max(0, min(n2-1, j+i))] = 20*factor*j+5*i+2.
    
    m21 = np.zeros((n1, n1),dtype=dtype)
    m22 = np.zeros((n2, n2),dtype=dtype)
    for j in range(n1):
        for i in range(-p1,p1+1):
            m21[j, max(0, min(n1-1, j+i))] = 10*factor*j**2+i**3
    for j in range(n2):
        for i in range(-p2,p2+1):
            m22[j, max(0, min(n2-1, j+i))] = 20*factor*j**2+i**3+2.
    
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
            x1[i1,i2] = 2.0*factor*random() - 1.0
            x2[i1,i2] = 5.0*factor*random() - 1.0
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
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'n1', [8, 16] )
@pytest.mark.parametrize( 'n2', [8, 32] )
@pytest.mark.parametrize( 'p1', [1, 3] )
@pytest.mark.parametrize( 'p2', [2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True] )
@pytest.mark.parallel

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

    # Fill in vector with random values, then update ghost regions
    for i1 in range(s1,e1+1):
        for i2 in range(s2,e2+1):
            x1[i1,i2] = 2.0*factor*random() + 1.0
            x2[i1,i2] = 5.0*factor*random() - 1.0
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
    X.axpy(Y, 5*factor)

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
