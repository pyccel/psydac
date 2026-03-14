#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import pytest
import numpy as np

from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.utilities import petsc_to_psydac
from psydac.api.settings import PSYDAC_BACKENDS
from psydac.ddm.cart import DomainDecomposition, CartDecomposition


# Backends
PSYDAC_BACKEND_PYTHON  = PSYDAC_BACKENDS['python']     # Pure Python
PSYDAC_BACKEND_GPYCCEL = PSYDAC_BACKENDS['pyccel-gcc'] # Pyccel w/ Fortran and GCC

# backend to activate multi threading
PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP           = PSYDAC_BACKEND_GPYCCEL.copy()
PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP['openmp'] = True

# ===============================================================================
def compute_global_starts_ends(domain_decomposition, npts, pads):
    ndims = len(npts)
    global_starts = [None] * ndims
    global_ends = [None] * ndims

    for axis in range(ndims):
        ee = domain_decomposition.global_element_ends[axis]
        
        global_ends[axis] = ee.copy()
        global_ends[axis][-1] = npts[axis] - 1
        global_starts[axis] = np.array([0] + (global_ends[axis][:-1] + 1).tolist())

    for s, e, p in zip(global_starts, global_ends, pads):
        assert all(e - s + 1 >= p)

    return tuple(global_starts), tuple(global_ends)


# TODO : Add test remove_spurious_entries, update_ghost_regions, exchange-assembly_data, diagonal,
#        ghost_regions_in_sync
# TODO : check if toarray() is working with a shift greater than 1 and idem for all other function that need toaaray

# ===============================================================================
# SERIAL TESTS
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [7, 15])
@pytest.mark.parametrize('p1', [2, 4])
@pytest.mark.parametrize('s1', [1, 2])
def test_stencil_matrix_1d_serial_init(dtype, n1, p1, s1, P1=True):
    # Create domain decomposition
    D = DomainDecomposition([n1], periods=[P1])

    # Partition the points
    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1])
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[s1])

    # Create a vector space V and a matrix M from V to V
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)

    # Check properties of this matrix
    assert M.domain == V
    assert M.codomain == V
    assert M.dtype == dtype
    assert M.pads == (p1,)
    assert M.backend == None
    assert M._data.shape == (n1 + 2 * p1 * s1, 1 + 2 * p1)
    assert M.shape == (n1 , n1 )

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [7, 15])
@pytest.mark.parametrize('n2', [8, 12])
@pytest.mark.parametrize('p1', [2, 4])
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('s1', [1, 3])
@pytest.mark.parametrize('s2', [1])
def test_stencil_matrix_2d_serial_init(dtype, n1, n2, p1, p2, s1, s2, P1=True, P2=False):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create a vector space V and a matrix M from V to V
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)

    # Check properties of this matrix
    assert M.domain == V
    assert M.codomain == V
    assert M.dtype == dtype
    assert M.pads == (p1, p2)
    assert M.backend == None
    assert M._data.shape == (n1 + 2 * p1 * s1, n2 + 2 * p2 * s2, 1 + 2 * p1, 1 + 2 * p2)
    assert M.shape == (n1 * n2, n1 * n2)

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [7, 9])
@pytest.mark.parametrize('n2', [8, 10])
@pytest.mark.parametrize('n3', [8])
@pytest.mark.parametrize('p1', [2, 3])
@pytest.mark.parametrize('p2', [2, 3])
@pytest.mark.parametrize('p3', [3])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [1, 2])
@pytest.mark.parametrize('s3', [2])
def test_stencil_matrix_3d_serial_init(dtype, n1, n2, n3, p1, p2, p3, s1, s2, s3, P1=True, P2=False, P3=False):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2, n3], periods=[P1, P2, P3])

    # Partition the points
    npts = [n1, n2, n3]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2, p3])
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2, p3], shifts=[s1, s2, s3])

    # Create a vector space V and a matrix M from V to V
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)

    # Check properties of this matrix
    assert M.domain == V
    assert M.codomain == V
    assert M.dtype == dtype
    assert M.pads == (p1, p2, p3)
    assert M.backend == None
    assert M._data.shape == (n1 + 2 * p1 * s1, n2 + 2 * p2 * s2, n3+2*p3*s3, 1 + 2 * p1, 1 + 2 * p2, 1+2*p3)
    assert M.shape == (n1 * n2 * n3, n1 * n2 * n3)

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [7, 15])
@pytest.mark.parametrize('n2', [8])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [2])
def test_stencil_matrix_2d_copy(dtype, n1, n2, p1, p2,s1,s2, P1=True, P2=False):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create a vector space V and a matrix M from V to V
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)

    # Take random data, but determinize it
    rng = np.random.default_rng(2)

    if dtype == complex:
        M._data[:] = rng.random(M._data.shape)+ 2j*rng.random(M._data.shape)
    else:
        M._data[:] = rng.random(M._data.shape)

    # Create a copy of the matrix
    M1 = M.copy()

    # check properties of the copy
    assert M1.domain == V
    assert M1.codomain == V
    assert M1.dtype == dtype
    assert M1.pads == (p1, p2)
    assert M1.backend == None
    assert M1._data.shape == (n1 + 2 * p1 * s1, n2 + 2 * p2 * s2, 1 + 2 * p1, 1 + 2 * p2)
    assert M1.shape == (n1 * n2, n1 * n2)
    assert np.array_equal(M1._data, M._data)

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [7, 15])
@pytest.mark.parametrize('n2', [8, 12])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [1])
def test_stencil_matrix_2d_basic_ops(dtype, n1, n2, p1, p2, s1, s2, P1=True, P2=False):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create a vector space V and a matrix M from V to V
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)

    # take random data, but determinize it
    rng = np.random.default_rng(2)
    if dtype == complex:
        M._data[:] = rng.random(M._data.shape)+ 2j*rng.random(M._data.shape)
    else:
        M._data[:] = rng.random(M._data.shape)

    # Check that the basic operation return the correct StencilMatrix
    assert isinstance(M * 2, StencilMatrix)
    assert (M * 2).dtype == dtype
    assert np.array_equal((M * 2)._data, M._data * 2)
    assert isinstance(M / 2, StencilMatrix)
    assert (M / 2).dtype == dtype
    assert np.array_equal((M / 2)._data, M._data / 2)
    assert isinstance(M + M, StencilMatrix)
    assert (M + M).dtype == dtype
    assert np.array_equal((M + M)._data, M._data + M._data)
    assert isinstance(M - M, StencilMatrix)
    assert (M - M).dtype == dtype
    assert np.array_equal((M - M)._data, M._data - M._data)

    # Check that the basic operation work with complex
    if dtype==complex:
        assert (2j*M).dtype == dtype
        assert np.array_equal((M*2j)._data, M._data*2j)

    # check that the internal properties on the matrix work
    M1 = M.copy()
    M1 *= 2
    M2 = M.copy()
    M2 /= 2
    M3 = M.copy()
    M3 += M
    M4 = M.copy()
    M4 -= M

    #check that copy works with out 
    M5 = StencilMatrix(V, V)
    M3.copy(out=M5)

    # Check that the internal basic operation return the correct StencilMatrix
    assert isinstance(M1, StencilMatrix)
    assert M1.dtype==dtype
    assert np.array_equal(M1._data, M._data * 2)
    assert isinstance(M2, StencilMatrix)
    assert M2.dtype==dtype
    assert np.array_equal(M2._data, M._data / 2)
    assert isinstance(M3, StencilMatrix)
    assert M3.dtype==dtype
    assert np.array_equal(M3._data, M._data + M._data)
    assert isinstance(M4, StencilMatrix)
    assert M4.dtype==dtype
    assert np.array_equal(M4._data, M._data - M._data)
    assert isinstance(M5, StencilMatrix)
    assert M5.dtype==dtype
    assert np.array_equal(M5._data, M._data + M._data)

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [7, 15])
@pytest.mark.parametrize('n2', [8, 12])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [1])
def test_stencil_matrix_2d_math(dtype, n1, n2, p1, p2, s1, s2, P1=True, P2=False):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create a vector space V and a matrix M from V to V
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)

    # take random data, but determinize it
    rng = np.random.default_rng(2)
    if dtype == complex:
        M._data[:] = rng.random(M._data.shape) + 10j * rng.random(M._data.shape)
    else:
        M._data[:] = -1*rng.random(M._data.shape)

    # Check that the max function work
    assert M.max()==M._data.max()

    # Create the array with the exact value
    M1_exa = abs(M._data)
    M2_exa = M._data.conjugate()

    # Apply the function at our matrixes
    M1 = abs(M)
    M2 = M.conjugate()
    M3 = M.conj()
    M4 = M.conj(out=M)

    # Check that the math operation conjugate and abs return the correct StencilMatrix
    for (m, m_exa) in zip([M1, M2, M3, M4], [M1_exa, M2_exa, M2_exa, M2_exa]):
        assert isinstance(m, StencilMatrix)
        assert m.domain == V
        assert m.codomain == V
        assert m.dtype == dtype
        assert m.pads == (p1, p2)
        assert m.backend == None
        assert m._data.shape == (n1 + 2 * p1 * s1, n2 + 2 * p2 * s2, 1 + 2 * p1, 1 + 2 * p2)
        assert m.shape == (n1 * n2, n1 * n2)
        assert np.allclose(m._data, m_exa)

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [5, 10])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('P1', [True, False])
def test_stencil_matrix_1d_serial_spurious_entries( dtype, n1, p1, s1, P1):

    # Select non-zero values based on diagonal index
    nonzero_values = dict()
    if dtype==complex:
        for k1 in range(-p1, p1 + 1):
            nonzero_values[k1] = k1
    else:
        for k1 in range(-p1, p1 + 1):
            nonzero_values[k1] = k1

    # Create domain decomposition
    D = DomainDecomposition([n1], periods=[P1])

    # Partition the points
    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1])
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[s1])

    # Create vector space and stencil matrix
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)

    # Fill in stencil matrix values
    for k1 in range(-p1, p1 + 1):
        M[:, k1] = nonzero_values[k1]

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Extract the data without the ghost region
    Ma=M._data[p1*s1:-p1*s1,:]

    # Construction of the exact data array by hand
    A = np.zeros(Ma.shape, dtype=dtype)
    for k1 in range(-p1, p1 + 1):
        A[:, k1+p1] = nonzero_values[k1]

    # Remove the periodic part by hand
    if (not P1) :
        for k1 in range( 0, p1 ):
            A[k1, slice(-p1+p1,-k1+p1)] = 0
        for k1 in range( n1-p1, n1 ):
            A[k1, slice(n1-k1+p1, p1+p1+1)] = 0

    # Check shape and data in data array
    assert A.shape == (n1,2*p1+1)
    assert np.array_equal(Ma, A)

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [1, 3])
@pytest.mark.parametrize('s1', [1, 3])
@pytest.mark.parametrize('s2', [1, 2])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True, False])
def test_stencil_matrix_2d_serial_spurious_entries( dtype, p1, p2, s1, s2, P1, P2, n1=15, n2=15):

    # Select non-zero values based on diagonal index
    nonzero_values = dict()
    if dtype==complex:
        for k1 in range(-p1, p1 + 1):
            for k2 in range(-p2, p2 + 1):
                nonzero_values[k1, k2] = 10j * k1 + k2
    else:
        for k1 in range(-p1, p1 + 1):
            for k2 in range(-p2, p2 + 1):
                nonzero_values[k1, k2] = 10 * k1 + k2

    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil matrix
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)

    # Fill in stencil matrix values
    for k1 in range(-p1, p1 + 1):
        for k2 in range(-p2, p2 + 1):
            M[:, :, k1, k2] = nonzero_values[k1, k2]

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Extract the data without the ghost region
    Ma=M._data[p1*s1:-p1*s1,p2*s2:-p2*s2,:,:]

    # Construct exact data array by hand
    A = np.zeros(Ma.shape, dtype=dtype)
    for k1 in range(-p1, p1 + 1):
        for k2 in range(-p2, p2 + 1):
            A[:, :, k1+p1, k2+p2] = nonzero_values[k1, k2]

    # Remove the periodic part in case of non-periodicity
    if (not P1) :
        for k1 in range( 0, p1 ):
            A[k1, :, slice(-p1+p1,-k1+p1),:] = 0
        for k1 in range( n1-p1, n1 ):
            A[k1, :, slice(n1-k1+p1, p1+p1+1),:] = 0

    if (not P2):
        for k2 in range(0, p2):
            A[:, k2, :, slice(-p2 + p2, -k2 + p2)] = 0
        for k2 in range(n2 - p2, n2):
            A[:, k2, :, slice(n2 - k2 + p2, p2 + p2 + 1)] = 0

    # Check shape and data in data array
    assert A.shape == (n1, n2, 2*p1+1, 2*p2+1)
    assert np.array_equal(Ma, A)

# ===============================================================================
@pytest.mark.parametrize('dtype', [float])
@pytest.mark.parametrize('n1', [7])
@pytest.mark.parametrize('p1', [1,2])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('P1', [True, False])
def test_stencil_matrix_1d_serial_toarray( dtype, n1, p1, s1, P1):
    # Select non-zero values based on diagonal index
    nonzero_values = dict()
    if dtype==complex:
        for k1 in range(-p1, p1 + 1):
            nonzero_values[k1] = 1j*k1
    else:
        for k1 in range(-p1, p1 + 1):
            nonzero_values[k1] = k1

    # Create domain decomposition
    D = DomainDecomposition([n1], periods=[P1])

    # Partition the points
    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1])

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[s1])

    # Create vector space and stencil matrix
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)

    # Fill in stencil matrix values
    for i1 in range(0, n1):
        for k1 in range(0, 2*p1+1):
            M._data[p1*s1+i1,k1] = nonzero_values[k1-p1]

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Convert stencil matrix to 2D array
    Ma = M.toarray()

    # Construct exact matrix by hand
    A = np.zeros(M.shape, dtype=dtype)
    for i1 in range(n1):
        for k1 in range(-p1, p1 + 1):
            j1 = (i1 + k1) % n1
            i = i1
            j = j1
            if (P1 or 0 <= i1 + k1 < n1) :
                A[i, j] = nonzero_values[k1]
    # Check shape and data in 2D array
    assert Ma.shape == M.shape
    assert np.array_equal(Ma, A)

# TODO : understand why it's not working when s>1
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [7, 15])
@pytest.mark.parametrize('n2', [8, 7])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [1, 3])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [1, 2])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True, False])
def test_stencil_matrix_2d_serial_toarray( dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # Select non-zero values based on diagonal index
    nonzero_values = dict()
    if dtype==complex:
        for k1 in range(-p1, p1 + 1):
            for k2 in range(-p2, p2 + 1):
                nonzero_values[k1, k2] = 10j * k1 + k2
    else:
        for k1 in range(-p1, p1 + 1):
            for k2 in range(-p2, p2 + 1):
                nonzero_values[k1, k2] = 10 * k1 + k2

    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil matrix
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)

    # Fill in stencil matrix values
    for k1 in range(-p1, p1 + 1):
        for k2 in range(-p2, p2 + 1):
            M[:, :, k1, k2] = nonzero_values[k1, k2]

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Convert stencil matrix to 2D array
    Ma = M.toarray()

    # Construct exact matrix by hand
    A = np.zeros(M.shape, dtype=dtype)
    for i1 in range(n1):
        for i2 in range(n2):
            for k1 in range(-p1, p1 + 1):
                for k2 in range(-p2, p2 + 1):
                    j1 = (i1 + k1) % n1
                    j2 = (i2 + k2) % n2
                    i = i1 * (n2) + i2
                    j = j1 * (n2) + j2
                    if (P1 or 0 <= i1 + k1 < n1) and (P2 or 0 <= i2 + k2 < n2):
                        A[i, j] = nonzero_values[k1, k2]
    # Check shape and data in 2D array
    assert Ma.shape == M.shape
    assert np.array_equal(Ma, A)

# TODO : understand why it's not working when s>1
    
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [7, 15])
@pytest.mark.parametrize('n2', [8, 7])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [1, 3])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True, False])
@pytest.mark.petsc

def test_stencil_matrix_2d_serial_topetsc( dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # Select non-zero values based on diagonal index
    nonzero_values = dict()
    if dtype==complex:
        for k1 in range(-p1, p1 + 1):
            for k2 in range(-p2, p2 + 1):
                nonzero_values[k1, k2] = 10j * k1 + k2
    else:
        for k1 in range(-p1, p1 + 1):
            for k2 in range(-p2, p2 + 1):
                nonzero_values[k1, k2] = 10 * k1 + k2

    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil matrix
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)

    # Fill in stencil matrix values
    for k1 in range(-p1, p1 + 1):
        for k2 in range(-p2, p2 + 1):
            M[:, :, k1, k2] = nonzero_values[k1, k2]

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Convert stencil matrix to PETSc.Mat
    Mp = M.topetsc()
    # Convert PETSc.Mat to sparse CSR matrix   
    indptr, indices, data = Mp.getValuesCSR()
    if dtype == float:
        data = data.real #PETSc with installation complex configuration only handles complex dtype

    M = M.tosparse().tocsr()

    assert (indptr == M.indptr).all()
    assert (indices == M.indices).all()
    assert np.array_equal(data, M.data)

# ===============================================================================
    
@pytest.mark.parametrize('n1', [5])
@pytest.mark.parametrize('n2', [7])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('p2', [1, 3])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True, False])
@pytest.mark.petsc

def test_mass_matrix_2d_serial_topetsc(n1, n2, p1, p2, P1, P2):
    from sympde.topology import Square, ScalarFunctionSpace, element_of
    from sympde.expr     import BilinearForm, integral
    from psydac.api.settings            import PSYDAC_BACKENDS
    from psydac.api.discretization      import discretize

    domain = Square()
    V = ScalarFunctionSpace('V', domain)

    u = element_of(V, name='u')
    v = element_of(V, name='v')    

    a = BilinearForm((u, v), integral(domain, u * v))
    domain_h = discretize(domain, ncells=[n1,n2], periodic=[P1,P2])
    V_h = discretize(V, domain_h, degree=[p1,p2])
    a_h = discretize(a, domain_h, [V_h, V_h], backend=PSYDAC_BACKENDS['pyccel-gcc'])    
    M = a_h.assemble()

    # Convert stencil matrix to PETSc.Mat
    Mp = M.topetsc()

    # Convert PETSc.Mat to sparse CSR matrix   
    indptr, indices, data = Mp.getValuesCSR()
    data = data.real #PETSc with installation complex configuration only handles complex dtype

    M = M.tosparse().tocsr()

    # The matrices can only be compared in the serial case.
    assert (indptr == M.indptr).all()
    assert (indices == M.indices).all()
    assert np.array_equal(data, M.data)

# ===============================================================================

@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [7, 15])
@pytest.mark.parametrize('n2', [8, 7])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('p2', [1])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True])
def test_stencil_matrix_2d_serial_tosparse( dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # Select non-zero values based on diagonal index
    nonzero_values = dict()
    if dtype==complex:
        for k1 in range(-p1, p1 + 1):
            for k2 in range(-p2, p2 + 1):
                nonzero_values[k1, k2] = 10j * k1 + k2
    else:
        for k1 in range(-p1, p1 + 1):
            for k2 in range(-p2, p2 + 1):
                nonzero_values[k1, k2] = 10 * k1 + k2

    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil matrix
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)

    # Fill in stencil matrix values
    for k1 in range(-p1, p1 + 1):
        for k2 in range(-p2, p2 + 1):
            M[:, :, k1, k2] = nonzero_values[k1, k2]

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Convert stencil matrix to 2D array
    Ms = M.tosparse()

    # Construct exact matrix by hand
    data=[]
    i_vec=[]
    j_vec=[]
    for i1 in range(n1):
        for i2 in range(n2):
            for k1 in range(-p1, p1 + 1):
                for k2 in range(-p2, p2 + 1):
                    j1 = (i1 + k1) % n1
                    j2 = (i2 + k2) % n2
                    i = i1 * (n2) + i2
                    j = j1 * (n2) + j2
                    if (P1 or 0 <= i1 + k1 < n1) and (P2 or 0 <= i2 + k2 < n2):
                        if not nonzero_values[k1,k2]==0:
                            data.append(nonzero_values[k1,k2])
                            i_vec.append(i)
                            j_vec.append(j)

    from scipy.sparse import coo_matrix
    Ms_exa=coo_matrix((data, (i_vec, j_vec)), shape=[n1*n2,n1*n2], dtype=dtype)

    # Check shape and data in 2D array
    assert Ms.shape == M.shape
    assert np.array_equal(Ms.toarray(), Ms_exa.toarray())

# TODO: verify for s>1
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [10, 32])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('P1', [True, False])
def test_stencil_matrix_1d_serial_dot(dtype, n1, p1, s1, P1):
    # Create domain decomposition
    D = DomainDecomposition([n1], periods=[P1])

    # Partition the points
    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1])

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[s1])

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)
    x = StencilVector(V)

    # Fill in stencil matrix values based on diagonal index (periodic!)
    if dtype==complex:
        for k1 in range(-p1, p1 + 1):
            M[:, k1] = (1+1j)*k1
    else:
        for k1 in range(-p1, p1 + 1):
            M[:, k1] = k1

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    rng = np.random.default_rng()
    if dtype==complex:
        for i1 in range(n1):
            x[i1] = (2.0+3j) * rng.random() - 1.0
    else:
        for i1 in range(n1):
            x[i1] = 2.0 * rng.random() - 1.0

    x.update_ghost_regions()

    # Compute matrix-vector product
    y = M.dot(x)

    assert isinstance(y, StencilVector)
    assert y.space is x.space

    # Convert stencil objects to Scipy sparse matrix and 1D Numpy arrays
    Ms = M.tosparse()
    xa = x.toarray()
    ya = y.toarray()

    # Exact result using Scipy sparse dot product
    ya_exact = Ms.dot(xa)

    # Check data in 1D array
    assert y.dtype == dtype
    assert np.allclose(ya, ya_exact, rtol=1e-14, atol=1e-14)

# TODO: verify for s>1
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [5, 15])
@pytest.mark.parametrize('n2', [5, 7])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('p2', [1, 2])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True, False])
# Case where domain and codomain are the same and matrix pads are the same
def test_stencil_matrix_2d_serial_dot_1(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)
    x = StencilVector(V)

    # Fill in stencil matrix values based on diagonal index (periodic!)
    if dtype==complex:
        for k1 in range(-p1, p1 + 1):
            for k2 in range(-p2, p2 + 1):
                M[:, :, k1, k2] = 10 * k1 + 1j*k2
    else:
        for k1 in range(-p1, p1 + 1):
            for k2 in range(-p2, p2 + 1):
                M[:, :, k1, k2] = 10 * k1 + k2

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in vector with random values, then update ghost regions
    if dtype==complex:
        for i1 in range(n1):
            for i2 in range(n2):
                x[i1, i2] = 2.0 * rng.random() - 1.0j
    else:
        for i1 in range(n1):
            for i2 in range(n2):
                x[i1, i2] = 2.0 * rng.random() - 1.0
    x.update_ghost_regions()

    # Compute matrix-vector product
    y = M.dot(x)

    # Convert stencil objects to Numpy arrays
    Ma = M.toarray()
    xa = x.toarray()
    ya = y.toarray()

    # Exact result using Numpy dot product
    ya_exact = np.dot(Ma, xa)

    # Check data in 1D array
    assert y.dtype==dtype
    assert np.allclose(ya, ya_exact, rtol=1e-12, atol=1e-12)

# TODO: verify for s>1
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [5, 15])
@pytest.mark.parametrize('n2', [5, 7])
@pytest.mark.parametrize('p1', [2, 3])
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [False])
@pytest.mark.parametrize('P2', [False])
# Case where domain and codomain have different size for both dimension and matrix pads are the same
def test_stencil_matrix_2d_serial_dot_2(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # Create domain decomposition
    D = DomainDecomposition([n1 - 1, n2 - 1], periods=[P1, P2])

    # Partition the points
    npts1 = [n1, n2]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1, [p1, p2])

    npts2 = [n1 - 1, n2 - 1]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2, [p1, p2])

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1, p2], shifts=[s1, s2])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space, stencil matrix, and stencil vector
    V1 = StencilVectorSpace(cart1, dtype=dtype)
    V2 = StencilVectorSpace(cart2, dtype=dtype)
    M1 = StencilMatrix(V1, V2, pads=(p1, p2))
    M2 = StencilMatrix(V2, V1, pads=(p1, p2))
    x1 = StencilVector(V1)
    x2 = StencilVector(V2)

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in stencil matrix values based on diagonal index (periodic!)
    if dtype==complex:
        M1._data[p1:-p1, p2:-p2, :, :] = rng.random(M1._data[p1:-p1, p2:-p2, :, :].shape)+1j*rng.random(M1._data[p1:-p1, p2:-p2, :, :].shape)
        M2._data[p1:-p1, p2:-p2, :, :] = rng.random(M2._data[p1:-p1, p2:-p2, :, :].shape)+1j*rng.random(M2._data[p1:-p1, p2:-p2, :, :].shape)
    else:
        M1._data[p1:-p1, p2:-p2, :, :] = rng.random(M1._data[p1:-p1, p2:-p2, :, :].shape)
        M2._data[p1:-p1, p2:-p2, :, :] = rng.random(M2._data[p1:-p1, p2:-p2, :, :].shape)

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    if dtype ==complex:
        for i1 in range(n1):
            for i2 in range(n2 - 1):
                x1[i1, i2] = (2.0+1j) * rng.random() - 1.0
        for i1 in range(n1 - 1):
            for i2 in range(n2 - 1):
                x2[i1, i2] = (2.0+1j)  * rng.random() - 1.0
    else:
        for i1 in range(n1):
            for i2 in range(n2 - 1):
                x1[i1, i2] = 2.0 * rng.random() - 1.0
        for i1 in range(n1 - 1):
            for i2 in range(n2 - 1):
                x2[i1, i2] = 2.0 * rng.random() - 1.0
    x1.update_ghost_regions()
    x2.update_ghost_regions()

    # Compute matrix-vector product
    y1 = M1.dot(x1)
    y2 = M2.dot(x2)

    # Convert stencil objects to Numpy arrays
    M1a = M1.toarray()
    x1a = x1.toarray()
    y1a = y1.toarray()

    M2a = M2.toarray()
    x2a = x2.toarray()
    y2a = y2.toarray()

    # Exact result using Numpy dot product
    y1a_exact = np.dot(M1a, x1a)
    y2a_exact = np.dot(M2a, x2a)

    # Check data in 1D array
    assert y1.dtype == dtype
    assert y2.dtype == dtype
    assert np.allclose(y1a, y1a_exact, rtol=1e-13, atol=1e-13)
    assert np.allclose(y2a, y2a_exact, rtol=1e-13, atol=1e-13)

# TODO: verify for s>1
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [5, 15])
@pytest.mark.parametrize('n2', [5, 7])
@pytest.mark.parametrize('p1', [2, 3])
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [False])
@pytest.mark.parametrize('P2', [False])
# Case where domain and codomain have different size for first dimension and matrix pads are different
def test_stencil_matrix_2d_serial_dot_3(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # Create domain decomposition
    D = DomainDecomposition([n1 - 1, n2 - 1], periods=[P1, P2])

    # Partition the points
    npts1 = [n1, n2 - 1]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1, [p1, p2])

    npts2 = [n1 - 1, n2 - 1]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2, [p1, p2])

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1, p2], shifts=[s1, s2])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space, stencil matrix, and stencil vector
    V1 = StencilVectorSpace(cart1, dtype=dtype)
    V2 = StencilVectorSpace(cart2, dtype=dtype)
    M1 = StencilMatrix(V1, V2, pads=(p1, p2 - 1))
    M2 = StencilMatrix(V2, V1, pads=(p1, p2 - 1))
    x1 = StencilVector(V1)
    x2 = StencilVector(V2)

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in stencil matrix values based on diagonal index (periodic!)
    if dtype==complex:
        M1[0:n1 - 1, 0:n2 - 1, :, :] = rng.random((n1 - 1, n2 - 1, 2 * p1 + 1, 2 * p2 - 1))+1j*rng.random((n1 - 1, n2 - 1, 2 * p1 + 1, 2 * p2 - 1))
        M2[0:n1, 0:n2 - 1, :, :] = rng.random((n1, n2 - 1, 2 * p1 + 1, 2 * p2 - 1))+1j*rng.random((n1, n2 - 1, 2 * p1 + 1, 2 * p2 - 1))
    else:
        M1[0:n1 - 1, 0:n2 - 1, :, :] = rng.random((n1 - 1, n2 - 1, 2 * p1 + 1, 2 * p2 - 1))
        M2[0:n1, 0:n2 - 1, :, :] = rng.random((n1, n2 - 1, 2 * p1 + 1, 2 * p2 - 1))

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    if dtype == complex:
        for i1 in range(n1):
            for i2 in range(n2 - 1):
                x1[i1, i2] = (2.0+1j) * rng.random() - 1.0
        for i1 in range(n1 - 1):
            for i2 in range(n2 - 1):
                x2[i1, i2] = (2.0+1j) * rng.random() - 1.0
    else:
        for i1 in range(n1):
            for i2 in range(n2 - 1):
                x1[i1, i2] = 2.0 * rng.random() - 1.0
        for i1 in range(n1 - 1):
            for i2 in range(n2 - 1):
                x2[i1, i2] = 2.0 * rng.random() - 1.0
    x1.update_ghost_regions()
    x2.update_ghost_regions()

    # Compute matrix-vector product
    y1 = M1.dot(x1)
    y2 = M2.dot(x2)

    # Convert stencil objects to Numpy arrays
    M1a = M1.toarray()
    x1a = x1.toarray()
    y1a = y1.toarray()

    M2a = M2.toarray()
    x2a = x2.toarray()
    y2a = y2.toarray()

    # Exact result using Numpy dot product
    y1a_exact = np.dot(M1a, x1a)
    y2a_exact = np.dot(M2a, x2a)

    # Check data in 1D array
    assert y1.dtype==dtype
    assert y2.dtype==dtype
    assert np.allclose(y1a, y1a_exact, rtol=1e-13, atol=1e-13)
    assert np.allclose(y2a, y2a_exact, rtol=1e-13, atol=1e-13)

# TODO: verify for s>1
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [5, 15])
@pytest.mark.parametrize('n2', [5, 7])
@pytest.mark.parametrize('p1', [2, 3])
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [False])
@pytest.mark.parametrize('P2', [False])
# Case where domain and codomain have different size for second dimension and matrix pads are different
def test_stencil_matrix_2d_serial_dot_4(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # Create domain decomposition
    D = DomainDecomposition([n1 - 1, n2 - 1], periods=[P1, P2])

    # Partition the points
    npts1 = [n1 - 1, n2]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1, [p1, p2])

    npts2 = [n1 - 1, n2 - 1]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2, [p1, p2])

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1, p2], shifts=[s1, s2])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space, stencil matrix, and stencil vector
    V1 = StencilVectorSpace(cart1, dtype=dtype)
    V2 = StencilVectorSpace(cart2, dtype=dtype)
    M1 = StencilMatrix(V1, V2, pads=(p1 - 1, p2))
    M2 = StencilMatrix(V2, V1, pads=(p1 - 1, p2))
    x1 = StencilVector(V1)
    x2 = StencilVector(V2)

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in stencil matrix values based on diagonal index (periodic!)
    if dtype==complex:
        M1[0:n1 - 1, 0:n2 - 1, :, :] = rng.random((n1 - 1, n2 - 1, 2 * p1 - 1, 2 * p2 + 1))+1j*rng.random((n1 - 1, n2 - 1, 2 * p1 - 1, 2 * p2 + 1))
        M2[0:n1 - 1, 0:n2, :, :] = rng.random((n1 - 1, n2, 2 * p1 - 1, 2 * p2 + 1))+1j*rng.random((n1 - 1, n2, 2 * p1 - 1, 2 * p2 + 1))
    else:
        M1[0:n1 - 1, 0:n2 - 1, :, :] = rng.random((n1 - 1, n2 - 1, 2 * p1 - 1, 2 * p2 + 1))
        M2[0:n1 - 1, 0:n2, :, :] = rng.random((n1 - 1, n2, 2 * p1 - 1, 2 * p2 + 1))

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions

    if dtype==complex:
        for i1 in range(n1 - 1):
            for i2 in range(n2):
                x1[i1, i2] = (2.0+1j) * rng.random() - 1.0
        for i1 in range(n1 - 1):
            for i2 in range(n2 - 1):
                x2[i1, i2] = (2.0+1j) * rng.random() - 1.0
    else:
        for i1 in range(n1 - 1):
            for i2 in range(n2):
                x1[i1, i2] = 2.0 * rng.random() - 1.0
        for i1 in range(n1 - 1):
            for i2 in range(n2 - 1):
                x2[i1, i2] = 2.0 * rng.random() - 1.0
    x1.update_ghost_regions()
    x2.update_ghost_regions()

    # Compute matrix-vector product
    y1 = M1.dot(x1)
    y2 = M2.dot(x2)

    # Convert stencil objects to Numpy arrays
    M1a = M1.toarray()
    x1a = x1.toarray()
    y1a = y1.toarray()

    M2a = M2.toarray()
    x2a = x2.toarray()
    y2a = y2.toarray()

    # Exact result using Numpy dot product
    y1a_exact = np.dot(M1a, x1a)
    y2a_exact = np.dot(M2a, x2a)

    # Check data in 1D array

    assert np.allclose(y1a, y1a_exact, rtol=1e-13, atol=1e-13)
    assert np.allclose(y2a, y2a_exact, rtol=1e-13, atol=1e-13)

# TODO: verify for s>1
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [5, 15])
@pytest.mark.parametrize('n2', [5, 7])
@pytest.mark.parametrize('p1', [2, 3])
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [False])
@pytest.mark.parametrize('P2', [False])
# Case where domain and codomain have different size for second dimension and matrix pads are the same
def test_stencil_matrix_2d_serial_dot_5(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # Create domain decomposition
    D = DomainDecomposition([n1 - 1, n2 - 1], periods=[P1, P2])

    # Partition the points
    npts1 = [n1 - 1, n2]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1, [p1, p2])

    npts2 = [n1 - 1, n2 - 1]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2, [p1, p2])

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1, p2], shifts=[s1, s2])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space, stencil matrix, and stencil vector
    V1 = StencilVectorSpace(cart1, dtype=dtype)
    V2 = StencilVectorSpace(cart2, dtype=dtype)
    M1 = StencilMatrix(V1, V2, pads=(p1, p2))
    M2 = StencilMatrix(V2, V1, pads=(p1, p2))
    x1 = StencilVector(V1)
    x2 = StencilVector(V2)

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in stencil matrix values based on diagonal index (periodic!)
    if dtype==complex:
        M1[0:n1 - 1, 0:n2 - 1, :, :] = rng.random((n1 - 1, n2 - 1, 2 * p1 + 1, 2 * p2 + 1))+1j*rng.random((n1 - 1, n2 - 1, 2 * p1 + 1, 2 * p2 + 1))
        M2[0:n1 - 1, 0:n2, :, :] = rng.random((n1 - 1, n2, 2 * p1 + 1, 2 * p2 + 1))+1j*rng.random((n1 - 1, n2, 2 * p1 + 1, 2 * p2 + 1))
    else:
        M1[0:n1 - 1, 0:n2 - 1, :, :] = rng.random((n1 - 1, n2 - 1, 2 * p1 + 1, 2 * p2 + 1))
        M2[0:n1 - 1, 0:n2, :, :] = rng.random((n1 - 1, n2, 2 * p1 + 1, 2 * p2 + 1))

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    if dtype==complex:
        for i1 in range(n1 - 1):
            for i2 in range(n2):
                x1[i1, i2] = (2.0+1j) * rng.random() - 1.0
        for i1 in range(n1 - 1):
            for i2 in range(n2 - 1):
                x2[i1, i2] = (2.0+1j) * rng.random() - 1.0
    else:
        for i1 in range(n1 - 1):
            for i2 in range(n2):
                x1[i1, i2] = 2.0 * rng.random() - 1.0
        for i1 in range(n1 - 1):
            for i2 in range(n2 - 1):
                x2[i1, i2] = 2.0 * rng.random() - 1.0

    x1.update_ghost_regions()
    x2.update_ghost_regions()

    # Compute matrix-vector product
    y1 = M1.dot(x1)
    y2 = M2.dot(x2)

    # Convert stencil objects to Numpy arrays
    M1a = M1.toarray()
    x1a = x1.toarray()
    y1a = y1.toarray()

    M2a = M2.toarray()
    x2a = x2.toarray()
    y2a = y2.toarray()

    # Exact result using Numpy dot product
    y1a_exact = np.dot(M1a, x1a)
    y2a_exact = np.dot(M2a, x2a)

    # Check data in 1D array
    assert np.allclose(y1a, y1a_exact, rtol=1e-13, atol=1e-13)
    assert np.allclose(y2a, y2a_exact, rtol=1e-13, atol=1e-13)

# TODO: verify for s>1
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [5, 15])
@pytest.mark.parametrize('n2', [5, 7])
@pytest.mark.parametrize('p1', [2, 3])
@pytest.mark.parametrize('p2', [2, 3])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [True])
@pytest.mark.parametrize('P2', [True])
# Case where domain and codomain have different size for both dimension and matrix pads are different
def test_stencil_matrix_2d_serial_dot_6(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # Create domain decomposition
    D = DomainDecomposition([n1 - 1, n2 - 1], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V, pads=(p1 - 1, p2 - 1))
    x = StencilVector(V)

    # Fill in stencil matrix values based on diagonal index (periodic!)
    if dtype==complex:
        for k1 in range(-p1 + 1, p1):
            for k2 in range(-p2 + 1, p2):
                M[:, :, k1, k2] = 10 * k1 + 1j*k2
    else:
        for k1 in range(-p1 + 1, p1):
            for k2 in range(-p2 + 1, p2):
                M[:, :, k1, k2] = 10 * k1 + k2

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in vector with random values, then update ghost regions
    if dtype==complex:
        for i1 in range(n1):
            for i2 in range(n2):
                x[i1, i2] = 2.0 * rng.random() - 1.0j
    else:
        for i1 in range(n1):
            for i2 in range(n2):
                x[i1, i2] = 2.0 * rng.random() - 1.0
    x.update_ghost_regions()

    # Compute matrix-vector product
    y = M.dot(x)

    # Convert stencil objects to Numpy arrays
    Ma = M.toarray()
    xa = x.toarray()
    ya = y.toarray()

    # Exact result using Numpy dot product
    ya_exact = np.dot(Ma, xa)

    # Check data in 1D array
    assert np.allclose(ya, ya_exact, rtol=1e-13, atol=1e-13)

# TODO: verify for s>1
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [5, 15])
@pytest.mark.parametrize('n2', [5, 7])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('p2', [1, 2])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True, False])
# Case where domain and codomain are the same and matrix pads are the same
def test_stencil_matrix_2d_serial_vdot(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)
    x = StencilVector(V)

    # Fill in stencil matrix values based on diagonal index (periodic!)
    if dtype==complex:
        for k1 in range(-p1, p1 + 1):
            for k2 in range(-p2, p2 + 1):
                M[:, :, k1, k2] = 10 * k1 + 1j*k2
    else:
        for k1 in range(-p1, p1 + 1):
            for k2 in range(-p2, p2 + 1):
                M[:, :, k1, k2] = 10 * k1 + k2

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in vector with random values, then update ghost regions
    if dtype==complex:
        for i1 in range(n1):
            for i2 in range(n2):
                x[i1, i2] = 2.0 * rng.random() - 1.0j
    else:
        for i1 in range(n1):
            for i2 in range(n2):
                x[i1, i2] = 2.0 * rng.random() - 1.0
    x.update_ghost_regions()

    # Convert stencil objects to Numpy arrays
    Ma = M.toarray()
    xa = x.toarray()

    # Compute matrix-vector product
    y = M.vdot(x)
    ya = y.toarray()

    # Exact result using Numpy dot product
    ya_exact = np.dot(np.conjugate(Ma), xa)
    # Check data in 1D array
    assert y.dtype == dtype
    assert np.allclose(ya, ya_exact, rtol=1e-12, atol=1e-12)

# TODO: verify for s>1
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [4, 32])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('s1', [1])
def test_stencil_matrix_1d_serial_transpose(dtype, n1, p1, s1, P1):
    # Create domain decomposition
    D = DomainDecomposition([n1 - 1], periods=[P1])

    # Partition the points
    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1])

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[s1])

    # Create vector space and stencil matrix
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in matrix values with random numbers between 0 and 1
    if dtype==complex:
        M[0:n1, -p1:p1 + 1] = rng.random((n1, 2 * p1 + 1))+1j*np.random.random((n1, 2 * p1 + 1))
    else:
        M[0:n1, -p1:p1 + 1] = np.random.random((n1, 2 * p1 + 1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Numpy array
    Ta = M.transpose().toarray()

    # Exact result: convert to Numpy array, then transpose
    Ta_exact = M.toarray().transpose()

    # Check data
    assert M.transpose().dtype==dtype
    assert np.array_equal(Ta, Ta_exact)

    #Same test with using out of transpose
    Mt = StencilMatrix(V, V)
    M.transpose(out = Mt)
    Ta = Mt.toarray()

    assert Mt.dtype==dtype
    assert np.array_equal(Ta, Ta_exact)

# TODO: verify for s>1
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [5, 15])
@pytest.mark.parametrize('n2', [5, 12])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('p2', [1, 2])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True, False])
# Case where domain and codomain have the same size for both dimension and matrix pads the same
def test_stencil_matrix_2d_serial_transpose_1(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # Create domain decomposition
    D = DomainDecomposition([n1 - 1, n2 - 1], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil matrix
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in matrix values with random numbers between 0 and 1
    if dtype == complex:
        M[0:n1, 0:n2, :, :] = rng.random((n1, n2, 2 * p1 + 1, 2 * p2 + 1))+1j*rng.random((n1, n2, 2 * p1 + 1, 2 * p2 + 1))
    else:
        M[0:n1, 0:n2, :, :] = rng.random((n1, n2, 2 * p1 + 1, 2 * p2 + 1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().tosparse()

    # Exact result: convert to Scipy sparse format, then transpose
    Ts_exact = M.tosparse().transpose()

    # Check data
    assert M.transpose().dtype==dtype
    assert Ts.dtype==dtype
    assert abs(Ts - Ts_exact).max() < 1e-14
    assert abs(Ts - M.T.tosparse()).max() < 1e-14

    #Same test with using out of transpose
    Mt = StencilMatrix(V, V)
    M.transpose(out = Mt)
    Ts = Mt.tosparse()

    assert Mt.dtype==dtype
    assert Ts.dtype==dtype
    assert abs(Ts - Ts_exact).max() < 1e-14
    assert abs(Ts - M.T.tosparse()).max() < 1e-14

# TODO: verify for s>1
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [5, 12])
@pytest.mark.parametrize('n2', [6, 8])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('p2', [1, 2])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [False])
@pytest.mark.parametrize('P2', [False])
# Case where domain and codomain have different size for both dimension and matrix pads the same
def test_stencil_matrix_2d_serial_transpose_2(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # This should only work with non periodic boundaries

    # Create domain decomposition
    D = DomainDecomposition([n1 - 1, n2 - 1], periods=[P1, P2])

    # Partition the points
    npts1 = [n1, n2]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1, [p1, p2])

    npts2 = [n1 - 1, n2 - 1]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2, [p1, p2])

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1, p2], shifts=[s1, s2])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil matrix
    V1 = StencilVectorSpace(cart1, dtype=dtype)
    V2 = StencilVectorSpace(cart2, dtype=dtype)
    M = StencilMatrix(V1, V2, pads=(p1, p2))

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in matrix values with random numbers between 0 and 1
    if dtype == complex:
        M[0:n1- 1, 0:n2, :, :] = rng.random((n1- 1, n2, 2 * p1 + 1, 2 * p2 + 1))+1j*rng.random((n1- 1, n2, 2 * p1 + 1, 2 * p2 + 1))
    else:
        M[0:n1- 1, 0:n2, :, :] = rng.random((n1- 1, n2, 2 * p1 + 1, 2 * p2 + 1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().tosparse()

    # Exact result: convert to Scipy sparse format, then transpose
    Ts_exact = M.tosparse().transpose()

    # Check data
    assert M.transpose().dtype==dtype
    assert Ts.dtype==dtype
    assert abs(Ts - Ts_exact).max() < 1e-14

# TODO: verify for s>1
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [5, 12])
@pytest.mark.parametrize('n2', [6, 8])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('p2', [1, 2])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [False])
@pytest.mark.parametrize('P2', [False])
# Case where domain and codomain have different size for the first dimension and matrix pads the same
def test_stencil_matrix_2d_serial_transpose_3(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # This should only work with non periodic boundaries

    # Create domain decomposition
    D = DomainDecomposition([n1 - 1, n2 - 1], periods=[P1, P2])

    # Partition the points
    npts1 = [n1, n2]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1, [p1, p2])

    npts2 = [n1 - 1, n2]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2, [p1, p2])

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1, p2], shifts=[s1, s2])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil matrix
    V1 = StencilVectorSpace(cart1, dtype=dtype)
    V2 = StencilVectorSpace(cart2, dtype=dtype)
    M = StencilMatrix(V1, V2, pads=(p1, p2))

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in matrix values with random numbers between 0 and 1
    if dtype == complex:
        M[0:n1- 1, 0:n2, :, :] = rng.random((n1- 1, n2, 2 * p1 + 1, 2 * p2 + 1))+1j*rng.random((n1- 1, n2, 2 * p1 + 1, 2 * p2 + 1))
    else:
        M[0:n1- 1, 0:n2, :, :] = rng.random((n1- 1, n2, 2 * p1 + 1, 2 * p2 + 1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().tosparse()

    # Exact result: convert to Scipy sparse format, then transpose
    Ts_exact = M.tosparse().transpose()

    # Check data
    assert abs(Ts - Ts_exact).max() < 1e-14
    assert abs(Ts - M.T.tosparse()).max() < 1e-14

# TODO: verify for s>1
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [5, 12])
@pytest.mark.parametrize('n2', [6, 8])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('p2', [1, 2])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [False])
@pytest.mark.parametrize('P2', [False])
# Case where domain and codomain have different size for the second dimension and matrix pads the same
def test_stencil_matrix_2d_serial_transpose_4(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # This should only work with non periodic boundaries

    # Create domain decomposition
    D = DomainDecomposition([n1 - 1, n2 - 1], periods=[P1, P2])

    # Partition the points
    npts1 = [n1, n2]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1, [p1, p2])

    npts2 = [n1, n2 - 1]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2, [p1, p2])

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1, p2], shifts=[s1, s2])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil matrix
    V1 = StencilVectorSpace(cart1, dtype=dtype)
    V2 = StencilVectorSpace(cart2, dtype=dtype)
    M = StencilMatrix(V1, V2, pads=(p1, p2))

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in matrix values with random numbers between 0 and 1
    if dtype == complex:
        M[0:n1, 0:n2 - 1, :, :] = rng.random((n1, n2 - 1, 2 * p1 + 1, 2 * p2 + 1))+1j*rng.random((n1, n2 - 1, 2 * p1 + 1, 2 * p2 + 1))
    else:
        M[0:n1, 0:n2 - 1, :, :] = rng.random((n1, n2 - 1, 2 * p1 + 1, 2 * p2 + 1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().tosparse()

    # Exact result: convert to Scipy sparse format, then transpose
    Ts_exact = M.tosparse().transpose()

    # Check data
    assert abs(Ts - Ts_exact).max() < 1e-14
    assert abs(Ts - M.T.tosparse()).max() < 1e-14

# TODO: verify for s>1
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [7, 12])
@pytest.mark.parametrize('n2', [7, 8])
@pytest.mark.parametrize('p1', [2, 3])
@pytest.mark.parametrize('p2', [2, 3])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [False])
@pytest.mark.parametrize('P2', [False])
# Case where domain and codomain have same size for both dimension and matrix pads different
def test_stencil_matrix_2d_serial_transpose_5(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # This should only work with non periodic boundaries

    # Create domain decomposition
    D = DomainDecomposition([n1 - 1, n2 - 1], periods=[P1, P2])

    # Partition the points
    npts1 = [n1, n2 - 1]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1, [p1, p2])

    npts2 = [n1, n2 - 1]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2, [p1, p2])

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1, p2], shifts=[s1, s2])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil matrix
    V1 = StencilVectorSpace(cart1, dtype=dtype)
    V2 = StencilVectorSpace(cart2, dtype=dtype)
    M = StencilMatrix(V1, V2, pads=(p1, p2 - 1))

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in matrix values with random numbers between 0 and 1
    if dtype == complex:
        M[0:n1, 0:n2 - 1, :, :] = rng.random((n1, n2 - 1, 2 * p1 + 1, 2 * p2 - 1))+1j*rng.random((n1, n2 - 1, 2 * p1 + 1, 2 * p2 - 1))
    else:
        M[0:n1, 0:n2 - 1, :, :] = rng.random((n1, n2 - 1, 2 * p1 + 1, 2 * p2 - 1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().tosparse()

    # Exact result: convert to Scipy sparse format, then transpose
    Ts_exact = M.tosparse().transpose()

    # Check data
    assert abs(Ts - Ts_exact).max() < 1e-14

# TODO: verify for s>1
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [7, 12])
@pytest.mark.parametrize('n2', [7, 8])
@pytest.mark.parametrize('p1', [2, 3])
@pytest.mark.parametrize('p2', [2, 3])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [False])
@pytest.mark.parametrize('P2', [False])
# Case where domain and codomain have different size for the first dimension and matrix pads are different
def test_stencil_matrix_2d_serial_transpose_6(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # This should only work with non periodic boundaries

    # Create domain decomposition
    D = DomainDecomposition([n1 - 1, n2 - 1], periods=[P1, P2])

    # Partition the points
    npts1 = [n1 - 1, n2 - 1]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1, [p1, p2])

    npts2 = [n1, n2 - 1]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2, [p1, p2])

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1, p2], shifts=[s1, s2])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil matrix
    V1 = StencilVectorSpace(cart1, dtype=dtype)
    V2 = StencilVectorSpace(cart2, dtype=dtype)
    M = StencilMatrix(V1, V2, pads=(p1, p2 - 1))

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in matrix values with random numbers between 0 and 1
    if dtype == complex:
        M[0:n1, 0:n2 - 1, :, :] = rng.random((n1, n2 - 1, 2 * p1 + 1, 2 * p2 - 1))+1j*rng.random((n1, n2 - 1, 2 * p1 + 1, 2 * p2 - 1))
    else:
        M[0:n1, 0:n2 - 1, :, :] = rng.random((n1, n2 - 1, 2 * p1 + 1, 2 * p2 - 1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().tosparse()

    # Exact result: convert to Scipy sparse format, then transpose
    Ts_exact = M.tosparse().transpose()

    # Check data
    assert abs(Ts - Ts_exact).max() < 1e-14

# TODO: verify for s>1
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [7, 12])
@pytest.mark.parametrize('n2', [7, 8])
@pytest.mark.parametrize('p1', [2, 3])
@pytest.mark.parametrize('p2', [2, 3])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [True])
@pytest.mark.parametrize('P2', [True])
# Case where domain and codomain have same size for both dimension and matrix pads are different
def test_stencil_matrix_2d_serial_transpose_7(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # Create domain decomposition
    D = DomainDecomposition([n1 - 1, n2 - 1], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil matrix
    V1 = StencilVectorSpace(cart, dtype=dtype)
    V2 = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V1, V2, pads=(p1, p2 - 1))

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in matrix values with random numbers between 0 and 1
    if dtype == complex:
        M[0:n1, 0:n2, :, :] = rng.random((n1, n2, 2 * p1 + 1, 2 * p2 - 1))+1j*rng.random((n1, n2, 2 * p1 + 1, 2 * p2 - 1))
    else:
        M[0:n1, 0:n2, :, :] = rng.random((n1, n2, 2 * p1 + 1, 2 * p2 - 1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().tosparse()

    # Exact result: convert to Scipy sparse format, then transpose
    Ts_exact = M.tosparse().transpose()

    # Check data
    assert abs(Ts - Ts_exact).max() < 1e-14

# TODO: verify for s>1
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [7, 12])
@pytest.mark.parametrize('n2', [7, 8])
@pytest.mark.parametrize('p1', [2, 3])
@pytest.mark.parametrize('p2', [2, 3])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [True])
@pytest.mark.parametrize('P2', [True])
# Case where domain and codomain have same size for both dimension and matrix pads are different
def test_stencil_matrix_2d_serial_transpose_8(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # Create domain decomposition
    D = DomainDecomposition([n1 - 1, n2 - 1], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil matrix
    V1 = StencilVectorSpace(cart, dtype=dtype)
    V2 = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V1, V2, pads=(p1 - 1, p2 - 1))

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in matrix values with random numbers between 0 and 1
    if dtype == complex:
        M[0:n1, 0:n2, :, :] = rng.random((n1, n2, 2 * p1 - 1, 2 * p2 - 1))+1j*rng.random((n1, n2, 2 * p1 - 1, 2 * p2 - 1))
    else:
        M[0:n1, 0:n2, :, :] = rng.random((n1, n2, 2 * p1 - 1, 2 * p2 - 1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().tosparse()

    # Exact result: convert to Scipy sparse format, then transpose
    Ts_exact = M.tosparse().transpose()

    # Check data
    assert abs(Ts - Ts_exact).max() < 1e-14

# TODO: verify for s>1
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [7, 12])
@pytest.mark.parametrize('n2', [7, 8])
@pytest.mark.parametrize('p1', [2, 3])
@pytest.mark.parametrize('p2', [2, 3])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [False])
@pytest.mark.parametrize('P2', [False])
# Case where domain and codomain have same size for both dimension and matrix pads are the same
def test_stencil_matrix_2d_serial_transpose_9(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # This should only work with non periodic boundaries

    # Create domain decomposition
    D = DomainDecomposition([n1 - 1, n2 - 1], periods=[P1, P2])

    # Partition the points
    npts = [n1 - 1, n2 - 1]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil matrix
    V1 = StencilVectorSpace(cart, dtype=dtype)
    V2 = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V1, V2)

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in matrix values with random numbers between 0 and 1
    if dtype == complex:
        M[0:n1 - 1, 0:n2 - 1, :, :] = rng.random((n1 - 1, n2 - 1, 2 * p1 + 1, 2 * p2 + 1))+1j*rng.random((n1 - 1, n2 - 1, 2 * p1 + 1, 2 * p2 + 1))
    else:
        M[0:n1 - 1, 0:n2 - 1, :, :] = rng.random((n1 - 1, n2 - 1, 2 * p1 + 1, 2 * p2 + 1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().tosparse()

    # Exact result: convert to Scipy sparse format, then transpose
    Ts_exact = M.tosparse().transpose()

    # Check data
    assert abs(Ts - Ts_exact).max() < 1e-14

# TODO: verify for s>1
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [6, 8])
@pytest.mark.parametrize('n2', [6, 10])
@pytest.mark.parametrize('n3', [7])
@pytest.mark.parametrize('p1', [2, 3])
@pytest.mark.parametrize('p2', [2, 3])
@pytest.mark.parametrize('p3', [2])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('s3', [1])
def test_stencil_matrix_3d_serial_transpose_1(dtype, n1, n2, n3, p1, p2, p3, s1, s2, s3, P1=False, P2=False, P3=False):
    # This should only work with non periodic boundaries

    # Create domain decomposition
    D = DomainDecomposition([n1 - 1, n2 - 1, n3 - 1], periods=[P1, P2, P3])

    # Partition the points
    npts1 = [n1 - 1, n2 - 1, n3 - 1]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1, [p1, p2])

    npts2 = [n1, n2 - 1, n3 - 1]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2, [p1, p2])

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1, p2, p3], shifts=[s1, s2, s3])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1, p2, p3], shifts=[s1, s2, s3])

    # Create vector space and stencil matrix
    V1 = StencilVectorSpace(cart1, dtype=dtype)
    V2 = StencilVectorSpace(cart2, dtype=dtype)
    M = StencilMatrix(V1, V2, pads=(p1, p2 - 1, p3 - 1))

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in matrix values with random numbers between 0 and 1
    if dtype == complex:
        M[0:n1, 0:n2 - 1, 0:n3 - 1, :, :, :] = rng.random((n1, n2 - 1, n3 - 1, 2 * p1 + 1, 2 * p2 - 1, 2 * p3 - 1))\
                                               +1j*rng.random((n1, n2 - 1, n3 - 1, 2 * p1 + 1, 2 * p2 - 1, 2 * p3 - 1))
    else:
        M[0:n1, 0:n2 - 1, 0:n3 - 1, :, :, :] = rng.random((n1, n2 - 1, n3 - 1, 2 * p1 + 1, 2 * p2 - 1, 2 * p3 - 1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().transpose().tosparse()
    Mt = M.transpose().tosparse()

    # Exact result: convert to Scipy sparse format, then transpose
    Ts_exact = M.tosparse()
    Mt_exact = Ts_exact.transpose()

    # Check data
    assert abs(Ts - Ts_exact).max() < 1e-14
    assert abs(Mt - Mt_exact).max() < 1e-14

# TODO: verify for s>1
# ===============================================================================
# BACKENDS TESTS
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [8])
@pytest.mark.parametrize('n2', [9])
@pytest.mark.parametrize('p1', [2])
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [False])
@pytest.mark.parametrize('P2', [False])
@pytest.mark.parametrize('backend', [PSYDAC_BACKEND_GPYCCEL])
def test_stencil_matrix_2d_serial_backend_dot_1(dtype, n1, n2, p1, p2, s1, s2, P1, P2, backend):
   # Create domain decomposition
    D = DomainDecomposition([n1 - 1, n2 - 1], periods=[P1, P2])

    # Partition the points
    npts1 = [n1 - 1, n2]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1, [p1, p2])

    npts2 = [n1 - 1, n2 - 1]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2, [p1, p2])

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1, p2], shifts=[s1, s2])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space, stencil matrix, and stencil vector
    V1 = StencilVectorSpace(cart1, dtype=dtype)
    V2 = StencilVectorSpace(cart2, dtype=dtype)
    M1 = StencilMatrix(V1, V2, pads=(p1, p2), backend=backend)
    M2 = StencilMatrix(V2, V1, pads=(p1, p2), backend=backend)
    x1 = StencilVector(V1)
    x2 = StencilVector(V2)

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in stencil matrix values based on diagonal index (periodic!)
    if dtype==complex:
        M1[0:n1 - 1, 0:n2 - 1, :, :] = rng.random((n1 - 1, n2 - 1, 2 * p1 + 1, 2 * p2 + 1))+1j*rng.random((n1 - 1, n2 - 1, 2 * p1 + 1, 2 * p2 + 1))
        M2[0:n1 - 1, 0:n2, :, :] = rng.random((n1 - 1, n2, 2 * p1 + 1, 2 * p2 + 1)) +1j*rng.random((n1 - 1, n2, 2 * p1 + 1, 2 * p2 + 1))
    else:
        M1[0:n1 - 1, 0:n2 - 1, :, :] = rng.random((n1 - 1, n2 - 1, 2 * p1 + 1, 2 * p2 + 1))
        M2[0:n1 - 1, 0:n2, :, :] = rng.random((n1 - 1, n2, 2 * p1 + 1, 2 * p2 + 1))

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    if dtype==complex :
        for i1 in range(n1 - 1):
            for i2 in range(n2):
                x1[i1, i2] = 2.0j * rng.random() - 1.0
        for i1 in range(n1 - 1):
            for i2 in range(n2 - 1):
                x2[i1, i2] = 2.0 * rng.random() - 1.0j
    else:
        for i1 in range(n1 - 1):
            for i2 in range(n2):
                x1[i1, i2] = 2.0 * rng.random() - 1.0
        for i1 in range(n1 - 1):
            for i2 in range(n2 - 1):
                x2[i1, i2] = 2.0 * rng.random() - 1.0

    x1.update_ghost_regions()
    x2.update_ghost_regions()

    # Compute matrix-vector product
    y1 = M1.dot(x1)
    y2 = M2.dot(x2)

    # Convert stencil objects to Numpy arrays
    M1a = M1.toarray()
    x1a = x1.toarray()
    y1a = y1.toarray()

    M2a = M2.toarray()
    x2a = x2.toarray()
    y2a = y2.toarray()

    # Exact result using Numpy dot product
    y1a_exact = np.dot(M1a, x1a)
    y2a_exact = np.dot(M2a, x2a)

    #Check that copy with out give the same result
    Mo   = StencilMatrix(V1, V2, pads=(p1, p2), backend=backend)
    M1.copy(out=Mo)
    y1o  = Mo.dot(x1)
    y1oa = y1o.toarray()

    # Check data in 1D array
    assert y1.dtype ==dtype
    assert y1o.dtype==dtype
    assert y2.dtype ==dtype
    assert np.allclose(y1a, y1a_exact, rtol=1e-13, atol=1e-13)
    assert np.allclose(y1oa, y1a_exact, rtol=1e-13, atol=1e-13)
    assert np.allclose(y2a, y2a_exact, rtol=1e-13, atol=1e-13)

# ===============================================================================
@pytest.mark.parametrize('dtype', [float,complex])
@pytest.mark.parametrize('n1', [8])
@pytest.mark.parametrize('n2', [9])
@pytest.mark.parametrize('p1', [2])
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [True])
@pytest.mark.parametrize('P2', [True])
@pytest.mark.parametrize('backend', [PSYDAC_BACKEND_PYTHON, PSYDAC_BACKEND_GPYCCEL])
def test_stencil_matrix_2d_serial_backend_dot_2(dtype, n1, n2, p1, p2, s1, s2, P1, P2, backend):
    # Create domain decomposition
    D = DomainDecomposition([n1 - 1, n2 - 1], periods=[P1, P2])

    # Partition the points
    npts = [n1 - 1, n2 - 1]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V, pads=(p1 - 1, p2 - 1), backend=backend)
    x = StencilVector(V)

    # Fill in stencil matrix values based on diagonal index (periodic!)
    if dtype == complex:
        f = lambda i1, i2: 10j * i1 + i2
    else:
        f = lambda i1, i2: 10 * i1 + i2
    for k1 in range(-p1 + 1, p1):
        for k2 in range(-p2 + 1, p2):
            M[:, :, k1, k2] = f(k1,k2)

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in vector with random values, then update ghost regions
    if dtype == complex:
        for i1 in range(n1):
            for i2 in range(n2):
                x[i1, i2] = 2.0j * rng.random() - 1.0
    else:
        for i1 in range(n1):
            for i2 in range(n2):
                x[i1, i2] = 2.0 * rng.random() - 1.0
    x.update_ghost_regions()

    # Compute matrix-vector product
    y = M.dot(x)

    # Convert stencil objects to Numpy arrays
    Ma = M.toarray()
    xa = x.toarray()
    ya = y.toarray()

    # Exact result using Numpy dot product
    ya_exact = np.dot(Ma, xa)

    # Check data in 1D array
    assert np.allclose(ya, ya_exact, rtol=1e-13, atol=1e-13)

    # tests for backend propagation
    assert y.dtype == dtype
    assert M.backend is backend
    assert (M + M).backend is M.backend
    assert (2 * M).backend is M.backend

# ===============================================================================
@pytest.mark.parametrize('dtype', [float,complex])
@pytest.mark.parametrize('n1', [5])
@pytest.mark.parametrize('n2', [5])
@pytest.mark.parametrize('p1', [2])
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [False])
@pytest.mark.parametrize('P2', [False])
@pytest.mark.parametrize('backend', [PSYDAC_BACKEND_GPYCCEL])
def test_stencil_matrix_2d_serial_backend_dot_3(dtype, n1, n2, p1, p2, s1, s2, P1, P2, backend):
    # Create domain decomposition
    D = DomainDecomposition([n1 - 1, n2 - 1], periods=[P1, P2])

    # Partition the points
    npts1 = [n1 - 1, n2]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1, [p1, p2])

    npts2 = [n1 - 1, n2 - 1]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2, [p1, p2])

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1, p2], shifts=[s1, s2])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space, stencil matrix, and stencil vector
    V1 = StencilVectorSpace(cart1, dtype=dtype)
    V2 = StencilVectorSpace(cart2, dtype=dtype)
    M1 = StencilMatrix(V1, V2, pads=(p1 - 1, p2), backend=backend)
    M2 = StencilMatrix(V2, V1, pads=(p1 - 1, p2), backend=backend)
    x1 = StencilVector(V1)
    x2 = StencilVector(V2)

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in stencil matrix values based on diagonal index (periodic!)
    if dtype == complex:
        M1[0:n1 - 1, 0:n2 - 1, :, :] = rng.random((n1 - 1, n2 - 1, 2 * p1 - 1, 2 * p2 + 1)) \
                                       + 1j * rng.random((n1 - 1, n2 - 1, 2 * p1 - 1, 2 * p2 + 1))
        M2[0:n1 - 1, 0:n2, :, :] = rng.random((n1 - 1, n2, 2 * p1 - 1, 2 * p2 + 1)) \
                                   + 1j * rng.random((n1 - 1, n2, 2 * p1 - 1, 2 * p2 + 1))
    else:
        M1[0:n1 - 1, 0:n2 - 1, :, :] = rng.random((n1 - 1, n2 - 1, 2 * p1 - 1, 2 * p2 + 1))
        M2[0:n1 - 1, 0:n2, :, :] = rng.random((n1 - 1, n2, 2 * p1 - 1, 2 * p2 + 1))

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    if dtype==complex:
        for i1 in range(n1 - 1):
            for i2 in range(n2):
                x1[i1, i2] = 2.0j * rng.random() - 1.0
        for i1 in range(n1 - 1):
            for i2 in range(n2 - 1):
                x2[i1, i2] = 2.0j * rng.random() - 1.0
    else:
        for i1 in range(n1 - 1):
            for i2 in range(n2):
                x1[i1, i2] = 2.0 * rng.random() - 1.0
        for i1 in range(n1 - 1):
            for i2 in range(n2 - 1):
                x2[i1, i2] = 2.0 * rng.random() - 1.0
    x1.update_ghost_regions()
    x2.update_ghost_regions()

    # Compute matrix-vector product
    y1 = M1.dot(x1)
    y2 = M2.dot(x2)

    # Convert stencil objects to Numpy arrays
    M1a = M1.toarray()
    x1a = x1.toarray()
    y1a = y1.toarray()

    M2a = M2.toarray()
    x2a = x2.toarray()
    y2a = y2.toarray()

    # Exact result using Numpy dot product
    y1a_exact = np.dot(M1a, x1a)
    y2a_exact = np.dot(M2a, x2a)

    # Check data in 1D array
    assert np.allclose(y1a, y1a_exact, rtol=1e-13, atol=1e-13)
    assert np.allclose(y2a, y2a_exact, rtol=1e-13, atol=1e-13)

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [15])
@pytest.mark.parametrize('n2', [12])
@pytest.mark.parametrize('p1', [2])
@pytest.mark.parametrize('p2', [3])
@pytest.mark.parametrize('s1', [1])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [True])
@pytest.mark.parametrize('P2', [True])
@pytest.mark.parametrize('backend', [None, PSYDAC_BACKEND_PYTHON, PSYDAC_BACKEND_GPYCCEL])
@pytest.mark.parametrize('backend2', [None, PSYDAC_BACKEND_PYTHON, PSYDAC_BACKEND_GPYCCEL])
def test_stencil_matrix_2d_serial_backend_switch(dtype, n1, n2, p1, p2, s1, s2, P1, P2, backend, backend2):
    # Create domain decomposition
    D = DomainDecomposition([n1 - 1, n2 - 1], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V, pads=(p1 - 1, p2 - 1), backend=backend)
    x = StencilVector(V)

    # Fill in stencil matrix values based on diagonal index (periodic!)
    for k1 in range(-p1 + 1, p1):
        for k2 in range(-p2 + 1, p2):
            M[:, :, k1, k2] = 10 * k1 + k2

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1):
        for i2 in range(n2):
            x[i1, i2] = 2.0 * rng.random() - 1.0
    x.update_ghost_regions()

    assert M.backend is backend
    M.dot(x)
    M.set_backend(backend2)

    assert M.backend is backend2
    M.dot(x)

# ===============================================================================
# PARALLEL TESTS
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [20, 67])
@pytest.mark.parametrize('p1', [1, 2, 3])
@pytest.mark.parametrize('sh1', [1, 2])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.mpi
def test_stencil_matrix_1d_parallel_toarray(dtype, n1, p1, sh1, P1):
    # Select non-zero values based on diagonal index
    nonzero_values = dict()
    if dtype == complex:
        for k1 in range(-p1, p1 + 1):
            nonzero_values[k1] = 1j * k1
    else:
        for k1 in range(-p1, p1 + 1):
            nonzero_values[k1] = k1

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1 - 1], periods=[P1], comm=comm)

    # Partition the points
    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1])
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[sh1])

    # Create vector space, a stencil matrix, and a stencil vector
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)

    s1, = V.starts
    e1, = V.ends

    # Fill in stencil matrix values
    for k1 in range(-p1, p1 + 1):
        M[s1:e1, k1] = nonzero_values[k1]

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Convert stencil matrix to 2D array
    Ma = M.toarray()
    # Maw = M.toarray(with_pads=True)

    # Construct exact matrix by hand
    A = np.zeros(M.shape, dtype=dtype)
    for i1 in range(s1,e1):
        for k1 in range(-p1, p1 + 1):
            j1 = (i1 + k1) % n1
            i = i1
            j = j1
            if (P1 or 0 <= i1 + k1 < n1) :
                A[i, j] = nonzero_values[k1]

    # Check data in 2D array
    assert A.shape==Ma.shape
    # assert A.shape==Maw.shape
    assert np.allclose(Ma, A, rtol=1e-14, atol=1e-14)
    # assert np.allclose(Maw, A, rtol=1e-14, atol=1e-14)

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [20, 67])
@pytest.mark.parametrize('p1', [1, 2, 3])
@pytest.mark.parametrize('sh1', [1])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.mpi
def test_stencil_matrix_1d_parallel_dot(dtype, n1, p1, sh1, P1):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1 - 1], periods=[P1], comm=comm)

    # Partition the points
    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1])
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[sh1])

    # Create vector space, a stencil matrix, and a stencil vector
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)
    x = StencilVector(V)

    s1, = V.starts
    e1, = V.ends

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in stencil matrix values based on diagonal index (periodic!)
    if dtype == complex:
        M[s1:e1, :] = rng.random((e1-s1, 2 * p1 + 1)) \
                                       + 1j * rng.random((e1-s1, 2 * p1 + 1))
    else:
        M[s1:e1, :] = rng.random((e1-s1, 2 * p1 + 1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    if dtype==complex:
        for i1 in range(x.starts[0], x.ends[0] + 1):
            x[i1] = 2.0j * rng.random() - 1.0
    else:
        for i1 in range(x.starts[0], x.ends[0] + 1):
            x[i1] = 2.0 * rng.random() - 1.0
    x.update_ghost_regions()

    # Compute matrix-vector product
    y = M.dot(x)

    assert isinstance(y, StencilVector)
    assert y.dtype==dtype
    assert y.space is x.space

    # Convert stencil objects to Scipy sparse matrix and 1D Numpy arrays
    Ms = M.tosparse()
    xa = x.toarray(with_pads=True)
    ya = y.toarray()

    # Exact result using Scipy sparse dot product
    ya_exact = Ms.dot(xa)

    # Check data in 1D array
    assert np.allclose(ya, ya_exact, rtol=1e-14, atol=1e-14)

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [8, 21])
@pytest.mark.parametrize('n2', [13, 32])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('p2', [1, 2])
@pytest.mark.parametrize('sh1', [1])
@pytest.mark.parametrize('sh2', [1])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True, False])
@pytest.mark.mpi
def test_stencil_matrix_2d_parallel_dot(dtype, n1, n2, p1, p2, sh1, sh2, P1, P2):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2], comm=comm)

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[sh1, sh2])

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)
    x = StencilVector(V)

    s1, s2 = V.starts
    e1, e2 = V.ends

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in stencil matrix values based on diagonal index (periodic!)
    if dtype == complex:
        M[s1:e1, s2:e2, :, :] = rng.random((e1-s1, e2-s2, 2 * p1 + 1, 2 * p2 + 1)) \
                                       + 1j * rng.random((e1-s1, e2-s2, 2 * p1 + 1, 2 * p2 + 1))
    else:
        M[s1:e1, s2:e2, :, :] = rng.random((e1-s1, e2-s2, 2 * p1 + 1, 2 * p2 + 1))

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    if dtype == complex:
        for i1 in range(s1, e1 + 1):
            for i2 in range(s2, e2 + 1):
                x[i1, i2] = 2.0j * rng.random() - 1.0
    else:
        for i1 in range(s1, e1 + 1):
            for i2 in range(s2, e2 + 1):
                x[i1, i2] = 2.0 * rng.random() - 1.0
    x.update_ghost_regions()

    # Compute matrix-vector product
    y = M.dot(x)

    assert isinstance(y, StencilVector)
    assert y.space is x.space

    # Convert stencil objects to Numpy arrays
    Ma = M.toarray()
    xa = x.toarray(with_pads=True)
    ya = y.toarray()

    # Exact result using Numpy dot product
    ya_exact = np.dot(Ma, xa)

    # Check data in 1D array
    assert np.allclose(ya, ya_exact, rtol=1e-13, atol=1e-13)

# ===============================================================================
@pytest.mark.parametrize('n1', [20, 67])
@pytest.mark.parametrize('p1', [1, 2, 3])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('sh1', [1])
@pytest.mark.mpi
def test_stencil_matrix_1d_parallel_sync( n1, p1, sh1, P1):
    from mpi4py import MPI
    from psydac.ddm.cart import CartDecomposition

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1], periods=[P1], comm=comm)

    # Partition the points
    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1])
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[sh1])

    V = StencilVectorSpace(cart, dtype=int)
    M = StencilMatrix(V, V)

    s1, = V.starts
    e1, = V.ends

    # Fill-in pattern
    fill_in = lambda i1, k1: 10 * i1 + k1

    # Fill in stencil matrix
    for i1 in range(s1, e1 + 1):
        for k1 in range(-p1, p1 + 1):
            M[i1, k1] = fill_in(i1, k1)

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: update ghost regions
    M.update_ghost_regions()

    # Convert stencil object to 1D Numpy array
    Ma = M.toarray(with_pads=True)

    # Create exact solution
    Me = np.zeros((n1, n1), dtype=V.dtype)

    for i1 in range(n1):
        for k1 in range(-p1, p1 + 1):

            # Get column index
            j1 = i1 + k1

            # If j1 is outside matrix limits, apply periodic BCs or skip entry
            if not 0 <= j1 < n1:
                if P1:
                    j1 = j1 % n1
                else:
                    continue

            # Fill in matrix element
            Me[i1, j1] = fill_in(i1, k1)

    # Compare local solution to global
    i1_min = max(0, s1 - p1)
    i1_max = min(e1 + p1 + 1, n1)

#   for i in range( comm.size ):
#       if i == comm.rank:
#           print( "RANK {}:".format( i ) )
#           print( M._data.shape )
#           print( Ma.shape )
#           print( Ma )
#           print( "PASSED" )
#           print( flush=True )
#       comm.Barrier()
    assert np.array_equal(Ma[i1_min:i1_max, :], Me[i1_min:i1_max, :])

# ===============================================================================
@pytest.mark.parametrize('n1', [21, 67])
@pytest.mark.parametrize('n2', [13, 32])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('p2', [1, 2])
@pytest.mark.parametrize('sh1', [1])
@pytest.mark.parametrize('sh2', [1])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True, False])
@pytest.mark.mpi
def test_stencil_matrix_2d_parallel_sync(n1, n2, p1, p2, sh1, sh2, P1, P2):
    from mpi4py import MPI
    from psydac.ddm.cart import CartDecomposition

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2], comm=comm)

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[sh1, sh2])

    V = StencilVectorSpace(cart, dtype=int)
    M = StencilMatrix(V, V)

    s1, s2 = V.starts
    e1, e2 = V.ends

    # Fill-in pattern
    fill_in = lambda i1, i2, k1, k2: 1000 * i1 + 100 * i2 + 10 * abs(k1) + abs(k2)

    # Fill in stencil matrix
    for i1 in range(s1, e1 + 1):
        for i2 in range(s2, e2 + 1):
            for k1 in range(-p1, p1 + 1):
                for k2 in range(-p2, p2 + 1):
                    M[i1, i2, k1, k2] = fill_in(i1, i2, k1, k2)

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: update ghost regions
    M.update_ghost_regions()

    # Convert stencil object to 1D Numpy array
    Ma = M.toarray(with_pads=True)

    # Create exact solution
    Me = np.zeros((n1 * n2, n1 * n2), dtype=V.dtype)

    for i1 in range(n1):
        for i2 in range(n2):
            for k1 in range(-p1, p1 + 1):
                for k2 in range(-p2, p2 + 1):

                    # Get column multi-index
                    j1 = i1 + k1
                    j2 = i2 + k2

                    # If j1 is outside matrix limits,
                    # apply periodic BCs or skip entry
                    if not 0 <= j1 < n1:
                        if P1:
                            j1 = j1 % n1
                        else:
                            continue

                    # If j2 is outside matrix limits,
                    # apply periodic BCs or skip entry
                    if not 0 <= j2 < n2:
                        if P2:
                            j2 = j2 % n2
                        else:
                            continue

                    # Get matrix indices assuming C ordering
                    i = i1 * n2 + i2
                    j = j1 * n2 + j2

                    # Fill in matrix element
                    Me[i, j] = fill_in(i1, i2, k1, k2)

#       #++++++++++++++++++++++++++++++++++++++
#       # DEBUG
#        #++++++++++++++++++++++++++++++++++++++
#        np.set_printoptions( linewidth=200 )
#
#       if comm.rank == 0:
#           print( 'Me' )
#           print( Me )
#           print( flush=True )
#        comm.Barrier()
#
#       for i in range(comm.size):
#            if i == comm.rank:
#              print( 'RANK {}'.format( i ) )
#                print( Ma )
#                print( flush=True )
#            comm.Barrier()
#       #++++++++++++++++++++++++++++++++++++++

    # Compare local solution to global, row by row
    i1_min = max(0, s1 - p1)
    i1_max = min(e1 + p1 + 1, n1)

    i2_min = max(0, s2 - p2)
    i2_max = min(e2 + p2 + 1, n2)

    for i1 in range(i1_min, i1_max):
        for i2 in range(i2_min, i2_max):
            i = i1 * n2 + i2
            assert np.array_equal(Ma[i, :], Me[i, :])

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [20, 67])
@pytest.mark.parametrize('p1', [1, 2, 3])
@pytest.mark.parametrize('sh1', [1])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.mpi
def test_stencil_matrix_1d_parallel_transpose(dtype, n1, p1, sh1, P1):
    from mpi4py import MPI
    from psydac.ddm.cart import CartDecomposition

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1 - 1], periods=[P1], comm=comm)

    # Partition the points
    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1])

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[sh1])

    # Create vector space and stencil matrix
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)

    s1, = V.starts
    e1, = V.ends

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in matrix values with random numbers between 0 and 1
    if dtype==complex:
        M[s1:e1 + 1, -p1:p1 + 1] = rng.random((e1 - s1 + 1, 2 * p1 + 1))+ 1j* rng.random((e1 - s1 + 1, 2 * p1 + 1))
    else:
        M[s1:e1 + 1, -p1:p1 + 1] = np.random.random((e1 - s1 + 1, 2 * p1 + 1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Numpy array
    Ta = M.transpose().toarray()

    # Exact result: convert to Numpy array including padding, then transpose,
    # hence remove entries that do not belong to current process.
    Ta_exact = M.toarray(with_pads=True).transpose()
    Ta_exact[:s1, :] = 0.0
    Ta_exact[e1 + 1:, :] = 0.0

    # Check data
    assert M.transpose().dtype
    assert np.array_equal(Ta, Ta_exact)

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [8, 21])
@pytest.mark.parametrize('n2', [13, 32])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('p2', [1, 2])
@pytest.mark.parametrize('sh1', [1])
@pytest.mark.parametrize('sh2', [1])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True, False])
@pytest.mark.mpi
def test_stencil_matrix_2d_parallel_transpose(dtype, n1, n2, p1, p2, sh1, sh2, P1, P2):
    from mpi4py import MPI
    from psydac.ddm.cart import CartDecomposition

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2], comm=comm)

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[sh1, sh2])

    # Create vector space and stencil matrix
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)

    s1, s2 = V.starts
    e1, e2 = V.ends

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in matrix values with random numbers between 0 and 1
    if dtype==complex:
        M[s1:e1 + 1, s2:e2 + 1, -p1:p1 + 1, -p2:p2 + 1] = rng.random(
            (e1 - s1 + 1, e2 - s2 + 1, 2 * p1 + 1, 2 * p2 + 1)) + 1j * rng.random(
            (e1 - s1 + 1, e2 - s2 + 1, 2 * p1 + 1, 2 * p2 + 1))
    else:
        M[s1:e1 + 1, s2:e2 + 1, -p1:p1 + 1, -p2:p2 + 1] = rng.random(
            (e1 - s1 + 1, e2 - s2 + 1, 2 * p1 + 1, 2 * p2 + 1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().tosparse()
    # Test using out option
    To = StencilMatrix(V, V)
    M.transpose(out=To)
    Tos = To.tosparse()

    # Exact result: convert to Scipy sparse format with padding, then transpose
    Ts_exact = M.tosparse(with_pads=True).transpose()

    # ...
    Ts_exact = Ts_exact.tocsr()
    for i, j in zip(*Ts_exact.nonzero()):
        i1, i2 = np.unravel_index(i, shape=[n1, n2], order='C')
        if not (s1 <= i1 <= e1 and s2 <= i2 <= e2):
            Ts_exact[i, j] = 0.0
    Ts_exact = Ts_exact.tocoo()
    Ts_exact.eliminate_zeros()
    # ...

    # Check data
    assert abs(Ts - Ts_exact).max() < 1e-14
    assert abs(Tos - Ts_exact).max() < 1e-14

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [7, 11])
@pytest.mark.parametrize('n2', [5, 8])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('p2', [1, 2])
@pytest.mark.parametrize('sh1', [1])
@pytest.mark.parametrize('sh2', [1])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True, False])
@pytest.mark.mpi
@pytest.mark.petsc

def test_stencil_matrix_2d_parallel_topetsc(dtype, n1, n2, p1, p2, sh1, sh2, P1, P2):
    from mpi4py import MPI

    # Select non-zero values based on diagonal index
    nonzero_values = dict()
    if dtype==complex:
        for k1 in range(-p1, p1 + 1):
            for k2 in range(-p2, p2 + 1):
                nonzero_values[k1, k2] = 10j * k1 + k2 + 7
    else:
        for k1 in range(-p1, p1 + 1):
            for k2 in range(-p2, p2 + 1):
                nonzero_values[k1, k2] = 10 * k1 + k2 + 7

    # Create domain decomposition: decomposes the coefficients
    comm = MPI.COMM_WORLD
    D = DomainDecomposition([n1, n2], periods=[P1, P2], comm=comm)

    # Partition the coefficients
    npts = [n1, n2] #Number of cells
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])

    # In cart, npts must be the number of coefficients.
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[sh1, sh2])

    # Create vector space and stencil matrix
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)
    x = StencilVector(V)

    # Fill in stencil matrix values
    for k1 in range(-p1, p1 + 1):
        for k2 in range(-p2, p2 + 1):
            M[:, :, k1, k2] = nonzero_values[k1, k2]

    s1, s2 = V.starts
    e1, e2 = V.ends

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in vector with random values, then update ghost regions
    if dtype == complex:
        for i1 in range(s1, e1 + 1):
            for i2 in range(s2, e2 + 1):
                x[i1, i2] = 2.0j * rng.random() - 1.0
    else:
        for i1 in range(s1, e1 + 1):
            for i2 in range(s2, e2 + 1):
                x[i1, i2] = 2.0 * rng.random() - 1.0
    x.update_ghost_regions()

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Convert stencil matrix to PETSc.Mat
    Mp = M.topetsc()

    y = M.dot(x)
    
    # Convert stencil matrix to PETSc.Mat
    Mp = M.topetsc()
    # Create Vec to allocate the result of the dot product
    y_petsc = Mp.createVecLeft()
    # Compute dot product
    Mp.mult(x.topetsc(), y_petsc)
    # Cast result back to PSYDAC StencilVector format
    y_p = petsc_to_psydac(y_petsc, V)

    assert np.allclose(y_p.toarray(), y.toarray(), rtol=1e-12, atol=1e-12)

# ===============================================================================

@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [13, 15])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('sh1', [1])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.mpi
@pytest.mark.petsc

def test_stencil_matrix_1d_parallel_topetsc(dtype, n1, p1, sh1, P1):
    from mpi4py import MPI

    # Select non-zero values based on diagonal index
    nonzero_values = dict()
    if dtype==complex:
        for k1 in range(-p1, p1 + 1):
                nonzero_values[k1] = 10j * k1 + 7
    else:
        for k1 in range(-p1, p1 + 1):
                nonzero_values[k1] = 10 * k1 + 7

    # Create domain decomposition: decomposes the coefficients
    comm = MPI.COMM_WORLD
    D = DomainDecomposition([n1], periods=[P1], comm=comm)

    # Partition the coefficients
    npts = [n1] #Number of cells
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1])

    # In cart, npts must be the number of coefficients.
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[sh1])

    # Create vector space and stencil matrix
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V)
    x = StencilVector(V)

    s1, = V.starts
    e1, = V.ends    

    # Fill in stencil matrix values
    for i1 in range(s1, e1 + 1):
        for k1 in range(-p1, p1 + 1):
                M[i1, k1] = (i1+1)*nonzero_values[k1]

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in vector with random values, then update ghost regions
    if dtype == complex:
        for i1 in range(s1, e1 + 1):
            x[i1] = 2.0j * rng.random() - 1.0
    else:
        for i1 in range(s1, e1 + 1):
            x[i1] = 2.0 * rng.random() - 1.0
    x.update_ghost_regions()

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Convert stencil matrix to PETSc.Mat
    Mp = M.topetsc()

    y = M.dot(x)
    
    # Convert stencil matrix to PETSc.Mat
    Mp = M.topetsc()
    # Create Vec to allocate the result of the dot product
    y_petsc = Mp.createVecLeft()
    # Compute dot product
    Mp.mult(x.topetsc(), y_petsc)
    # Cast result back to PSYDAC StencilVector format
    y_p = petsc_to_psydac(y_petsc, V)

    assert np.allclose(y_p.toarray(), y.toarray(), rtol=1e-12, atol=1e-12)

# ===============================================================================

@pytest.mark.parametrize('n1', [4,7])
@pytest.mark.parametrize('n2', [3,5])
@pytest.mark.parametrize('p1', [2])
@pytest.mark.parametrize('p2', [1])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True, False])
@pytest.mark.mpi
@pytest.mark.petsc

def test_mass_matrix_2d_parallel_topetsc(n1, n2, p1, p2, P1, P2):
    from sympde.topology import Square, ScalarFunctionSpace, element_of
    from sympde.expr     import BilinearForm, integral
    from psydac.api.settings            import PSYDAC_BACKENDS
    from psydac.api.discretization      import discretize
    from mpi4py import MPI

    domain = Square()
    V = ScalarFunctionSpace('V', domain)

    u = element_of(V, name='u')
    v = element_of(V, name='v')    

    a = BilinearForm((u, v), integral(domain, u * v))
    comm = MPI.COMM_WORLD
    domain_h = discretize(domain, ncells=[n1,n2], periodic=[P1,P2], comm=comm)
    Vh = discretize(V, domain_h, degree=[p1,p2])
    ah = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKENDS['pyccel-gcc'])    
    M = ah.assemble()

    x = Vh.coeff_space.zeros()

    s1, s2 = Vh.coeff_space.starts
    e1, e2 = Vh.coeff_space.ends

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(s1, e1 + 1):
        for i2 in range(s2, e2 + 1):
            x[i1, i2] = 2.0 * rng.random() - 1.0
    x.update_ghost_regions()

    y = M.dot(x)
    
    # Convert stencil matrix to PETSc.Mat
    Mp = M.topetsc()
    # Create Vec to allocate the result of the dot product
    y_petsc = Mp.createVecLeft()
    # Compute dot product
    Mp.mult(x.topetsc(), y_petsc)
    # Cast result back to PSYDAC StencilVector format
    y_p = petsc_to_psydac(y_petsc, Vh.coeff_space)

    assert np.allclose(y_p.toarray(), y.toarray(), rtol=1e-12, atol=1e-12)

# ===============================================================================

@pytest.mark.parametrize('n1', [4,7])
@pytest.mark.parametrize('n2', [3,5])
@pytest.mark.parametrize('n3', [3,4])
@pytest.mark.parametrize('p1', [2])
@pytest.mark.parametrize('p2', [1])
@pytest.mark.parametrize('p3', [1])
@pytest.mark.parametrize('P1', [False])
@pytest.mark.parametrize('P2', [True])
@pytest.mark.parametrize('P3', [True, False])
@pytest.mark.mpi
@pytest.mark.petsc

def test_mass_matrix_3d_parallel_topetsc(n1, n2, n3, p1, p2, p3, P1, P2, P3):
    from sympde.topology import Cube, ScalarFunctionSpace, element_of
    from sympde.expr     import BilinearForm, integral
    from psydac.api.settings            import PSYDAC_BACKENDS
    from psydac.api.discretization      import discretize
    from mpi4py import MPI

    domain = Cube()
    V = ScalarFunctionSpace('V', domain)

    u = element_of(V, name='u')
    v = element_of(V, name='v')    

    a = BilinearForm((u, v), integral(domain, u * v))
    comm = MPI.COMM_WORLD
    domain_h = discretize(domain, ncells=[n1,n2,n3], periodic=[P1,P2,P3], comm=comm)
    Vh = discretize(V, domain_h, degree=[p1,p2,p3])
    ah = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKENDS['pyccel-gcc'])    
    M = ah.assemble()

    x = Vh.coeff_space.zeros()

    s1, s2, s3 = Vh.coeff_space.starts
    e1, e2, e3 = Vh.coeff_space.ends

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(s1, e1 + 1):
        for i2 in range(s2, e2 + 1):
            for i3 in range(s3, e3 + 1):
                x[i1, i2, i3] = 2.0 * rng.random() - 1.0
    x.update_ghost_regions()

    y = M.dot(x)
    
    # Convert stencil matrix to PETSc.Mat
    Mp = M.topetsc()
    # Create Vec to allocate the result of the dot product
    y_petsc = Mp.createVecLeft()
    # Compute dot product
    Mp.mult(x.topetsc(), y_petsc)
    # Cast result back to PSYDAC StencilVector format
    y_p = petsc_to_psydac(y_petsc, Vh.coeff_space)

    assert np.allclose(y_p.toarray(), y.toarray(), rtol=1e-12, atol=1e-12)
    
# ===============================================================================    

@pytest.mark.parametrize('n1', [15,17])
@pytest.mark.parametrize('p1', [2])
@pytest.mark.parametrize('P1', [True])
@pytest.mark.mpi
@pytest.mark.petsc

def test_mass_matrix_1d_parallel_topetsc(n1, p1, P1):
    from sympde.topology import Line, ScalarFunctionSpace, element_of
    from sympde.expr     import BilinearForm, integral
    from psydac.api.settings            import PSYDAC_BACKENDS
    from psydac.api.discretization      import discretize
    from mpi4py import MPI

    domain = Line()
    V = ScalarFunctionSpace('V', domain)

    u = element_of(V, name='u')
    v = element_of(V, name='v')    

    a = BilinearForm((u, v), integral(domain, u * v))
    comm = MPI.COMM_WORLD
    domain_h = discretize(domain, ncells=[n1], periodic=[P1], comm=comm)
    Vh = discretize(V, domain_h, degree=[p1])
    ah = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKENDS['pyccel-gcc'])    
    M = ah.assemble()

    x = Vh.coeff_space.zeros()

    s1, = Vh.coeff_space.starts
    e1, = Vh.coeff_space.ends

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(s1, e1 + 1):
        x[i1] = 2.0 * rng.random() - 1.0
    x.update_ghost_regions()

    y = M.dot(x)
    
    # Convert stencil matrix to PETSc.Mat
    Mp = M.topetsc()

    # Create Vec to allocate the result of the dot product
    y_petsc = Mp.createVecLeft()

    x_petsc = x.topetsc()
    # Compute dot product
    Mp.mult(x_petsc, y_petsc)
    # Cast result back to PSYDAC StencilVector format
    y_p = petsc_to_psydac(y_petsc, Vh.coeff_space)

    assert np.allclose(y_p.toarray(), y.toarray(), rtol=1e-12, atol=1e-12)

# ===============================================================================
# PARALLEL BACKENDS TESTS
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [20])
@pytest.mark.parametrize('p1', [1])
@pytest.mark.parametrize('sh1', [1])
@pytest.mark.parametrize('P1', [True])
@pytest.mark.parametrize('backend', [PSYDAC_BACKEND_GPYCCEL])
@pytest.mark.mpi
def test_stencil_matrix_1d_parallel_backend_dot(dtype, n1, p1, sh1, P1, backend):
    from mpi4py import MPI
    from psydac.ddm.cart import CartDecomposition

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1 - 1], periods=[P1], comm=comm)

    # Partition the points
    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1])

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[sh1])

    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V, backend=backend)
    x = StencilVector(V)

    # Fill in stencil matrix values based on diagonal index (periodic!)
    for k1 in range(-p1, p1 + 1):
        M[:, k1] = k1

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(x.starts[0], x.ends[0] + 1):
        x[i1] = 2.0 * rng.random() - 1.0
    x.update_ghost_regions()

    # Compute matrix-vector product
    y = M.dot(x)

    assert isinstance(y, StencilVector)
    assert y.dtype==dtype
    assert y.space is x.space

    # Convert stencil objects to Scipy sparse matrix and 1D Numpy arrays
    Ms = M.tosparse()
    xa = x.toarray(with_pads=True)
    ya = y.toarray()

    # Exact result using Scipy sparse dot product
    ya_exact = Ms.dot(xa)

    # Check data in 1D array
    assert np.allclose(ya, ya_exact, rtol=1e-14, atol=1e-14)

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [8])
@pytest.mark.parametrize('n2', [13])
@pytest.mark.parametrize('p1', [1])
@pytest.mark.parametrize('p2', [1])
@pytest.mark.parametrize('sh1', [1])
@pytest.mark.parametrize('sh2', [1])
@pytest.mark.parametrize('P1', [False])
@pytest.mark.parametrize('P2', [True])
@pytest.mark.parametrize('backend', [PSYDAC_BACKEND_GPYCCEL, PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP])
@pytest.mark.mpi
def test_stencil_matrix_2d_parallel_backend_dot(dtype, n1, n2, p1, p2, sh1, sh2, P1, P2, backend):
    from mpi4py import MPI
    from psydac.ddm.cart import CartDecomposition

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2], comm=comm)

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts, [p1, p2])

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[sh1, sh2])

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace(cart, dtype=dtype)
    M = StencilMatrix(V, V, backend=backend)
    x = StencilVector(V)

    s1, s2 = V.starts
    e1, e2 = V.ends

    # Fill in stencil matrix values based on diagonal index (periodic!)
    if dtype==complex:
        fill_in=lambda i1,i2: 10j * i1 + i2
    else:
        fill_in=lambda i1,i2: 10 * i1 + i2

    for k1 in range(-p1, p1 + 1):
        for k2 in range(-p2, p2 + 1):
            M[:, :, k1, k2] = fill_in(k1,k2)

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Initialize random number generator
    rng = np.random.default_rng()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(s1, e1 + 1):
        for i2 in range(s2, e2 + 1):
            x[i1, i2] = 2.0 * rng.random() - 1.0
    x.update_ghost_regions()

    # Compute matrix-vector product
    y = M.dot(x)

    assert isinstance(y, StencilVector)
    assert y.space is x.space

    # Convert stencil objects to Numpy arrays
    Ma = M.toarray()
    xa = x.toarray(with_pads=True)
    ya = y.toarray()

    # Exact result using Numpy dot product
    ya_exact = np.dot(Ma, xa)

    # Check data in 1D array
    assert np.allclose(ya, ya_exact, rtol=1e-13, atol=1e-13)

    # tests for backend propagation
    assert M.backend is backend
    assert M.T.backend is M.backend
    assert (M + M).backend is M.backend
    assert (2 * M).backend is M.backend

# ===============================================================================
# SCRIPT FUNCTIONALITY
# ===============================================================================
if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
