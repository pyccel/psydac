#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import pytest
import numpy as np

from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.utilities import array_to_psydac, petsc_to_psydac
from psydac.ddm.cart import DomainDecomposition, CartDecomposition

# TODO : test update ghost region interface
# TODO : add test exchange_assembly_data

# ===============================================================================
def compute_global_starts_ends(domain_decomposition, npts):
    ndims = len(npts)
    global_starts = [None] * ndims
    global_ends = [None] * ndims

    for axis in range(ndims):
        es = domain_decomposition.global_element_starts[axis]
        ee = domain_decomposition.global_element_ends[axis]

        global_ends[axis] = ee.copy()
        global_ends[axis][-1] = npts[axis] - 1
        global_starts[axis] = np.array([0] + (global_ends[axis][:-1] + 1).tolist())

    return global_starts, global_ends

# ===============================================================================
# SERIAL TESTS
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [1, 7])
@pytest.mark.parametrize('n2', [1, 5])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [2])
def test_stencil_vector_2d_serial_init(dtype, n1, n2, p1, p2, s1, s2, P1=True, P2=False):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil vector
    V = StencilVectorSpace(C, dtype=dtype)
    x = StencilVector(V)

    # Test properties of the vector
    assert x.space is V
    assert x.dtype == dtype
    assert x.starts == tuple(global_starts)
    assert x.ends == tuple(global_ends)
    assert x.pads == (p1, p2)
    assert x._data.shape == (n1 + 2 * p1 * s1, n2 + 2 * p2 * s2)
    assert x._data.dtype == dtype
    assert not x.ghost_regions_in_sync

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [1, 7])
@pytest.mark.parametrize('n2', [1, 5])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [2])
def test_stencil_vector_2d_serial_copy(dtype, n1, n2, p1, p2, s1, s2, P1=True, P2=False):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil vector
    V = StencilVectorSpace(C, dtype=dtype)
    x = StencilVector(V)

    # Take random data, but determinize it
    np.random.seed(2)
    if dtype == complex:
        x._data[:] = np.random.random(x._data.shape) + 1j * np.random.random(x._data.shape)
    else:
        x._data[:] = np.random.random(x._data.shape)

    # Compute the copy
    z = x.copy()

    # Test the properties of the copy
    assert isinstance(z, StencilVector)
    assert z.space is V
    assert z._data is not x._data
    assert z.dtype == dtype
    assert np.array_equal(x._data, z._data)

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [7, 15])
@pytest.mark.parametrize('n2', [8, 12])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('p2', [3])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [2])
def test_stencil_vector_2d_basic_ops(dtype, n1, n2, p1, p2, s1, s2, P1=True, P2=False):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil vector
    V = StencilVectorSpace(C, dtype=dtype)
    M = StencilVector(V)

    # take random data, but determinize it
    np.random.seed(2)
    if dtype == complex:
        M._data[:] = np.random.random(M._data.shape) + 1j * np.random.random(M._data.shape)
    else:
        M._data[:] = np.random.random(M._data.shape)

    # Test classical basic operation
    assert (M * 2).dtype == dtype
    assert np.array_equal((M * 2)._data, M._data * 2)
    assert (M / 2).dtype == dtype
    assert np.array_equal((M / 2)._data, M._data / 2)
    assert (M + M).dtype == dtype
    assert np.array_equal((M + M)._data, M._data + M._data)
    assert (M - M).dtype == dtype
    assert np.array_equal((M - M)._data, M._data - M._data)

    M1 = M.copy()
    M1 *= 2
    M2 = M.copy()
    M2 /= 2
    M3 = M.copy()
    M3 += M
    M4 = M.copy()
    M4 -= M

    # test inplace operation
    for (m, mex) in zip([M1, M2, M3, M4], [M._data * 2, M._data / 2, M._data + M._data, M._data - M._data]):
        assert isinstance(m, StencilVector)
        assert m.dtype == dtype
        assert m.space is V
        assert np.array_equal(m._data, mex)

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [1, 7])
@pytest.mark.parametrize('n2', [1, 5])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [1])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [1])
def test_stencil_vector_2d_serial_toarray(dtype, n1, n2, p1, p2, s1, s2, P1=True, P2=False):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil vector
    V = StencilVectorSpace(C, dtype=dtype)
    x = StencilVector(V)

    #Fill vector with some data
    if dtype == complex:
        f = lambda i1, i2: 10j * i1 + i2
    else:
        f = lambda i1, i2: 10 * i1 + i2

    for i1 in range(n1):
        for i2 in range(n2):
            x[i1, i2] = f(i1,i2)

    # Convert StencilVector into array (in serial only order has an impact)
    xc = x.toarray()
    xf = x.toarray(order='F')

    # Create our exact arrays
    zc = np.zeros((n1 * n2),dtype=dtype)
    zf = np.zeros((n1 * n2),dtype=dtype)
    for i1 in range(n1):
        for i2 in range(n2):
            zc[i1 * n2 + i2] = f(i1,i2)
            zf[i1 + i2 * n1] = f(i1,i2)

    # Verify toarray() with and without padding
    for (x, z) in zip([xc, xf], [zc, zf]):
        assert x.shape == (n1*n2,)
        assert x.dtype == dtype
        assert np.array_equal(xc, zc)

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [1, 7])
@pytest.mark.parametrize('n2', [1, 5])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [1])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [1])
def test_stencil_vector_2d_serial_math(dtype, n1, n2, p1, p2, s1, s2, P1=True, P2=False):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil vector
    V = StencilVectorSpace(C, dtype=dtype)
    x = StencilVector(V)
    y = StencilVector(V)

    # take random data, but determinize it
    np.random.seed(2)
    if dtype == complex:
        x._data[:] = np.random.random(x._data.shape) + 1j * np.random.random(x._data.shape)
    else:
        x._data[:] = np.random.random(x._data.shape)

    y[:, :] = 42.0

    # Compute new StencilVectors by basics operation
    r1 = x + y
    r2 = x - y
    r3 = 2 * x
    r4 = x * 2
    xa = x.toarray()
    ya = y.toarray()

    # Create exact array
    r1_exact = xa + ya
    r2_exact = xa - ya
    r3_exact = 2 * xa
    r4_exact = xa * 2

    # Compare value contain in StencilVector and exact array
    for (r, rex) in zip([r1, r2, r3, r4], [r1_exact, r2_exact, r3_exact, r4_exact]):
        assert isinstance(r, StencilVector)
        assert r.space is V
        assert r.dtype == dtype
        assert np.array_equal(r.toarray(), rex)

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [1, 7])
@pytest.mark.parametrize('n2', [1, 5])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [1])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [1])
def test_stencil_vector_2d_serial_dot(dtype, n1, n2, p1, p2, s1, s2, P1=True, P2=False):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil vectors
    V = StencilVectorSpace(C, dtype)
    x = StencilVector(V)
    y = StencilVector(V)

    if dtype == complex:
        f1 = lambda i1, i2: 100j * i1 + i2
        f2 = lambda i1, i2: 10j* i2 - i1
    else:
        f1 = lambda i1, i2: 100 * i1 + i2
        f2 = lambda i1, i2: 10 * i2 - i1

    # Fill the vectors with data
    for i1 in range(n1):
        for i2 in range(n2):
            x[i1, i2] = f1(i1,i2)
            y[i1, i2] = f2(i1,i2)

    # Create inner vector product (x,y) and (y,x)
    z1 = x.inner(y)
    z2 = y.inner(x)

    # Exact value by Numpy dot and vdot
    if dtype==complex:
        z_exact = np.vdot(x.toarray(), y.toarray())
    else:
        z_exact = np.dot(x.toarray(), y.toarray())

    # Compute axpy exact sol
    if dtype == complex:
        cst = 5j
    else:
        cst = 5

    z3 = x + cst * y
    x.mul_iadd(cst, y)

    # Test exact value and symmetry of the scalar product
    assert z1.dtype == dtype
    assert z2.dtype == dtype
    assert z1 == z_exact
    assert z2 == z_exact.conjugate()
    assert np.allclose(x._data, z3._data)

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [1, 7])
@pytest.mark.parametrize('n2', [1, 5])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [1])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [1])
def test_stencil_vector_2d_serial_conjugate(dtype, n1, n2, p1, p2, s1, s2, P1=True, P2=False):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil vector
    V = StencilVectorSpace(C, dtype)
    x = StencilVector(V)

    # Fill the vector with data
    if dtype == complex:
        f = lambda i1, i2: 10j * i1 + i2
    else:
        f = lambda i1, i2: 10 * i1 + i2

    for i1 in range(n1):
        for i2 in range(n2):
            x[i1, i2] = f(i1,i2)

    # Create the conjugate of the vector
    z1 = x.conjugate()
    z2 = StencilVector(V)
    x.conjugate(out=z2)

    # Compute exact value with Numpy conjugate
    z_exact = x._data.conjugate()

    # Test the exact value
    assert z1.dtype == dtype
    assert np.array_equal(z1._data, z_exact)
    assert np.array_equal(z2._data, z_exact)

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [1, 7])
@pytest.mark.parametrize('n2', [1, 5])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [1])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True])
def test_stencil_vector_2d_serial_array_to_psydac(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil vector
    V = StencilVectorSpace(C, dtype=dtype)
    x = StencilVector(V)

    # Fill the vector with data

    if dtype == complex:
        f = lambda i1, i2: 10j * i1 + i2
    else:
        f = lambda i1, i2: 10 * i1 + i2
    for i1 in range(n1):
        for i2 in range(n2):
            x[i1, i2] = f(i1,i2)

    # Convert vector to array
    xa = x.toarray()

    # Convert array to vector of V
    v = array_to_psydac(xa, V)

    # Test properties of v and data contained
    assert v.space is V
    assert v.dtype == dtype
    assert v.starts == (0, 0)
    assert v.ends == (n1 - 1, n2 - 1)
    assert v.pads == (p1, p2)
    assert v._data.shape == (n1 + 2 * p1 * s1, n2 + 2 * p2 * s2)
    assert v._data.dtype == dtype
    assert np.array_equal(xa, v.toarray())

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [1, 7])
@pytest.mark.parametrize('n2', [1, 5])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [1])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True])
@pytest.mark.petsc
def test_stencil_vector_2d_serial_topetsc(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil vector
    V = StencilVectorSpace(C, dtype=dtype)
    x = StencilVector(V)

    # Fill the vector with data
    if dtype == complex:
        f = lambda i1, i2: 10j * i1 + i2
    else:
        f = lambda i1, i2: 10 * i1 + i2
    for i1 in range(n1):
        for i2 in range(n2):
            x[i1, i2] = f(i1,i2)

    # Convert vector to PETSc.Vec
    v = x.topetsc()

    # Convert PETSc.Vec to StencilVector of V
    v = petsc_to_psydac(v, V)

    # Test properties of v and data contained
    assert v.space is V
    assert v.dtype == dtype
    assert v.starts == (0, 0)
    assert v.ends == (n1 - 1, n2 - 1)
    assert v.pads == (p1, p2)
    assert v._data.shape == (n1 + 2 * p1 * s1, n2 + 2 * p2 * s2)
    assert v._data.dtype == dtype
    assert np.array_equal(x.toarray(), v.toarray())

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [5, 7])
@pytest.mark.parametrize('n2', [5, 9])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [1])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True])
def test_stencil_vector_2d_serial_update_ghost_region_interior(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil vector
    V = StencilVectorSpace(C, dtype=dtype)
    x = StencilVector(V)

    # Fill vector with data
    if dtype == complex:
        f = lambda i1, i2: 10j * i1 + i2
    else:
        f = lambda i1, i2: 10 * i1 + i2
    for i1 in range(n1):
        for i2 in range(n2):
            x[i1, i2] = f(i1,i2)

    # Update the vector ghost region if the concerned domain is periodic
    x.update_ghost_regions()
    data = x._data

    # Test if _sync properties was changed
    assert x._sync

    # Compare vectors ghost region to the exact value
    if P1:
        # Left region with corner
        assert np.array_equal(data[0:p1 * s1, :], data[n1:n1 + p1 * s1, :])
        # Right region with corner
        assert np.array_equal(data[n1 + p1 * s1:n1 + 2 * p1 * s1, :], data[p1 * s1:2 * p1 * s1, :])
    else:
        # Left region with corner
        assert np.array_equal(data[0:p1 * s1, :], np.zeros((p1 * s1, n2 + 2 * p2 * s2), dtype=dtype))
        # Right region with corner
        assert np.array_equal(data[n1 + p1 * s1:n1 + 2 * p1 * s1, :],
                              np.zeros((p1 * s1, n2 + 2 * p2 * s2), dtype=dtype))
    if P2:
        # Left region with corner
        assert np.array_equal(data[:, 0:p2 * s2], data[:, n2:n2 + p2 * s2])
        # Right region with corner
        assert np.array_equal(data[:, n2 + p2 * s2:n2 + 2 * p2 * s2], data[:, p2 * s2:2 * p2 * s2])
    else:
        # Left region
        assert np.array_equal(data[:, 0:p2 * s2], np.zeros((n1 + 2 * p1 * s1, p2 * s2), dtype=dtype))
        # Right region with corner
        assert np.array_equal(data[:, n2 + p2 * s2:n2 + 2 * p2 * s2],
                              np.zeros((n1 + 2 * p1 * s1, p2 * s2), dtype=dtype))

# ===============================================================================
# PARALLEL TESTS
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [12, 22])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.mpi
def test_stencil_vector_1d_parallel_init(dtype, n1, p1, s1, P1=True):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # Create domain decomposition
    D = DomainDecomposition([n1], periods=[P1], comm=comm)

    # Partition the points
    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[s1])

    # Create vector space and stencil vector
    V = StencilVectorSpace(cart, dtype=dtype)
    x = StencilVector(V)

    # Test properties of the vector
    assert x.space is V
    assert x.dtype == dtype
    assert tuple(x.starts) == tuple(V.starts)
    assert tuple(x.ends) == tuple(V.ends)
    assert x.pads == (p1, )
    assert x._data.shape == (V.ends[0]-V.starts[0]+1 + 2 * p1 * s1, )
    assert x._data.dtype == dtype
    assert not x.ghost_regions_in_sync

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [10, 15])
@pytest.mark.parametrize('n2', [6, 12])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [3])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [2])
@pytest.mark.mpi
def test_stencil_vector_2d_parallel_init(dtype, n1, n2, p1, p2, s1, s2, P1=True, P2=False):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2], comm=comm)

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil vector
    V = StencilVectorSpace(cart, dtype=dtype)
    x = StencilVector(V)

    # Test properties of the vector
    assert x.space is V
    assert x.dtype == dtype
    assert tuple(x.starts) == tuple(V.starts)
    assert tuple(x.ends) == tuple(V.ends)
    assert x.pads == (p1, p2)
    assert x._data.shape == (V.ends[0]-V.starts[0]+1 + 2 * p1 * s1, V.ends[1]-V.starts[1]+1 + 2 * p2 * s2)
    assert x._data.dtype == dtype
    assert not x.ghost_regions_in_sync

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [20, 32])
@pytest.mark.parametrize('n2', [24, 40])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [2])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True])
@pytest.mark.mpi
@pytest.mark.petsc
def test_stencil_vector_2d_parallel_topetsc(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2], comm=comm)

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil vector
    V = StencilVectorSpace(C, dtype=dtype)
    x = StencilVector(V)

    # Fill the vector with data
    if dtype == complex:
        f = lambda i1, i2: 10j * i1 + i2
    else:
        f = lambda i1, i2: 10 * i1 + i2

    # Initialize distributed 2D stencil vector
    for i1 in range(V.starts[0], V.ends[0] + 1):
        for i2 in range(V.starts[1], V.ends[1] + 1):
            x[i1, i2] = f(i1,i2)

    # Convert vector to PETSc.Vec
    v = x.topetsc()

    # Convert PETSc.Vec to StencilVector of V
    v = petsc_to_psydac(v, V)

    assert np.array_equal(x.toarray(), v.toarray())
    
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [20, 32])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.mpi
@pytest.mark.petsc
def test_stencil_vector_1d_parallel_topetsc(dtype, n1, p1, s1, P1):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # Create domain decomposition
    D = DomainDecomposition([n1], periods=[P1], comm=comm)

    # Partition the points
    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[s1])

    # Create vector space and stencil vector
    V = StencilVectorSpace(C, dtype=dtype)
    x = StencilVector(V)

    # Fill the vector with data
    if dtype == complex:
        f = lambda i1: 10j * i1 + 3
    else:
        f = lambda i1: 10 * i1 + 3

    # Initialize distributed 2D stencil vector
    for i1 in range(V.starts[0], V.ends[0] + 1):
            x[i1] = f(i1)

    # Convert vector to PETSc.Vec
    v = x.topetsc()

    # Convert PETSc.Vec to StencilVector of V
    v = petsc_to_psydac(v, V)

    assert np.array_equal(x.toarray(), v.toarray())

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [20, 32])
@pytest.mark.parametrize('n2', [24, 40])
@pytest.mark.parametrize('n3', [7, 12])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('p3', [1])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [2])
@pytest.mark.parametrize('s3', [1])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True])
@pytest.mark.parametrize('P3', [False])

@pytest.mark.mpi
@pytest.mark.petsc
def test_stencil_vector_3d_parallel_topetsc(dtype, n1, n2, n3, p1, p2, p3, s1, s2, s3, P1, P2, P3):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # Create domain decomposition
    D = DomainDecomposition([n1, n2, n3], periods=[P1, P2, P3], comm=comm)

    # Partition the points
    npts = [n1, n2, n3]
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2, p3], shifts=[s1, s2, s3])

    # Create vector space and stencil vector
    V = StencilVectorSpace(C, dtype=dtype)
    x = StencilVector(V)

    # Fill the vector with data

    if dtype == complex:
        f = lambda i1, i2, i3: 10j * i1 + i2 - i3 
    else:
        f = lambda i1, i2, i3: 10 * i1 + i2 - i3

    # Initialize distributed 2D stencil vector
    for i1 in range(V.starts[0], V.ends[0] + 1):
        for i2 in range(V.starts[1], V.ends[1] + 1):
            for i3 in range(V.starts[2], V.ends[2] + 1):
                x[i1, i2, i3] = f(i1, i2, i3)

    # Convert vector to PETSc.Vec
    v = x.topetsc()

    # Convert PETSc.Vec to StencilVector of V
    v = petsc_to_psydac(v, V)

    assert np.array_equal(x.toarray(), v.toarray())

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [6, 15])
@pytest.mark.parametrize('n2', [10, 18])
@pytest.mark.parametrize('n3', [12])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('p3', [1])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [3])
@pytest.mark.parametrize('s3', [1])
@pytest.mark.mpi
def test_stencil_vector_3d_parallel_init(dtype, n1, n2, n3, p1, p2, p3, s1, s2, s3, P1=True, P2=False, P3=True):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # Create domain decomposition
    D = DomainDecomposition([n1, n2, n3], periods=[P1, P2, P3], comm=comm)

    # Partition the points
    npts = [n1, n2, n3]
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2, p3], shifts=[s1, s2, s3])

    # Create vector space and stencil vector
    V = StencilVectorSpace(cart, dtype=dtype)
    x = StencilVector(V)

    # Test properties of the vector
    assert x.space is V
    assert x.dtype == dtype
    assert tuple(x.starts) == tuple(V.starts)
    assert tuple(x.ends) == tuple(V.ends)
    assert x.pads == (p1, p2, p3)
    assert x._data.shape == (V.ends[0]-V.starts[0]+1 + 2 * p1 * s1, V.ends[1]-V.starts[1]+1 + 2 * p2 * s2,
                             V.ends[2]-V.starts[2]+1 + 2 * p3 * s3)
    assert x._data.dtype == dtype
    assert not x.ghost_regions_in_sync

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [20, 32])
@pytest.mark.parametrize('n2', [24, 40])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [2])
@pytest.mark.mpi
def test_stencil_vector_2d_parallel_toarray(dtype, n1, n2, p1, p2, s1, s2, P1=True, P2=False):
    # Create domain decomposition
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    D = DomainDecomposition([n1, n2], periods=[P1, P2], comm=comm)

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil vector
    V = StencilVectorSpace(cart, dtype=dtype)
    x = StencilVector(V)

    # Values in 2D grid (global indexing)
    if dtype == complex:
        f = lambda i1, i2: 100j * i1 + i2
    else:
        f = lambda i1, i2: 100 * i1 + i2

    # Initialize distributed 2D stencil vector
    for i1 in range(V.starts[0], V.ends[0] + 1):
        for i2 in range(V.starts[1], V.ends[1] + 1):
            x[i1, i2] = f(i1, i2)

    x.update_ghost_regions()

    assert x.dtype == dtype

    # Construct local 2D array manually
    z1 = np.zeros((n1, n2), dtype=dtype)
    z2 = np.zeros((n2, n1), dtype=dtype)
    for i1 in range(cart.starts[0], cart.ends[0] + 1):
        for i2 in range(cart.starts[1], cart.ends[1] + 1):
            z1[i1, i2] = f(i1, i2)
            z2[i2, i1] = f(i1, i2)

    # Verify toarray() without padding
    xa1 = x.toarray()
    xa2 = x.toarray(order="F")
    za1 = z1.reshape(-1)
    za2 = z2.reshape(-1)

    assert xa1.dtype == dtype
    assert xa1.shape == (n1 * n2,)
    assert np.array_equal(xa1, za1)
    assert np.array_equal(xa2, za2)

    # # Verify toarray() with padding: internal region should not change
    # xe = x.toarray(with_pads=True)
    # xe = xe.reshape(n1, n2)
    #
    # assert xe.dtype == dtype
    # assert xe.shape == (n1, n2)
    # assert np.array_equal(xe, z1)

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
def test_stencil_vector_2d_parallel_array_to_psydac(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    npts = [n1, n2]   

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # Create domain decomposition
    D = DomainDecomposition(npts, periods=[P1, P2], comm=comm)

    # Partition the points
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil vector
    V = StencilVectorSpace(C, dtype=dtype)
    x = StencilVector(V)

    # Fill the vector with data
    if dtype == complex:
        f = lambda i1, i2: 10j * i1 + i2
    else:
        f = lambda i1, i2: 10 * i1 + i2
    for i1 in range(V.starts[0], V.ends[0]+1):
        for i2 in range(V.starts[1], V.ends[1]+1):
            x[i1, i2] = f(i1, i2)

    x.update_ghost_regions()

    # Convert vector to array
    xa = x.toarray()

    # Apply array_to_psydac as left inverse of toarray
    v_l_inv = array_to_psydac(xa, V)

    # Apply array_to_psydac first, and toarray next
    xa_r_inv = np.array(np.random.rand(xa.size), dtype=dtype)*xa # the vector must be distributed as xa
    x_r_inv = array_to_psydac(xa_r_inv, V)
    x_r_inv.update_ghost_regions()
    va_r_inv = x_r_inv.toarray()

    ## Check that array_to_psydac is the inverse of .toarray():
    # left inverse:
    assert isinstance(v_l_inv, StencilVector)
    assert v_l_inv.space is V    
    assert np.array_equal(x._data, v_l_inv._data)
    # right inverse:
    assert np.array_equal(xa_r_inv, va_r_inv)

# TODO: test that ghost regions have been properly copied to 'xe' array
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [6, 10])
@pytest.mark.parametrize('n2', [12, 15])
@pytest.mark.parametrize('p1', [1, 4])
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [1])
@pytest.mark.mpi
def test_stencil_vector_2d_parallel_dot(dtype, n1, n2, p1, p2, s1, s2, P1=True, P2=False):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2], comm=comm)

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])

    # Create vector space and stencil vectors
    V = StencilVectorSpace(cart, dtype=dtype)
    x = StencilVector(V)
    y = StencilVector(V)

    if dtype == complex:
        f1 = lambda i1, i2: 100j * i1 + i2
        f2 = lambda i1, i2: 10j * i2 - i1
    else:
        f1 = lambda i1, i2: 100 * i1 + i2
        f2 = lambda i1, i2: 10 * i2 - i1

    # Fill the vectors with data
    for i1 in range(V.starts[0], V.ends[0] + 1):
        for i2 in range(V.starts[1], V.ends[1] + 1):
            x[i1, i2] = f1(i1,i2)
            y[i1, i2] = f2(i1,i2)

    # Create scalar product (x,y) and (y,x)
    res1 = x.inner(y)
    res2 = y.inner(x)

    # Compute exact value with Numpy dot
    if dtype==complex:
        res_ex1 = comm.allreduce(np.vdot(x.toarray(), y.toarray()))
        res_ex2 = comm.allreduce(np.vdot(y.toarray(), x.toarray()))
    else:
        res_ex1 = comm.allreduce(np.dot(x.toarray(), y.toarray()))
        res_ex2 = res_ex1

    # Compute axpy exact sol
    if dtype == complex:
        cst = 5j
    else:
        cst = 5

    z3 = x + cst * y
    x.mul_iadd(cst, y)

    # Test exact value and symmetry of the scalar product
    assert np.allclose(x._data, z3._data)
    assert res1 == res_ex1
    assert res2 == res_ex2

# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [12, 24])
@pytest.mark.parametrize('n2', [9, 15])
@pytest.mark.parametrize('n3', [8])
@pytest.mark.parametrize('p1', [1, 3])
@pytest.mark.parametrize('p2', [1, 2])
@pytest.mark.parametrize('p3', [4])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [1, 2])
@pytest.mark.parametrize('s3', [1])
@pytest.mark.mpi
def test_stencil_vector_3d_parallel_dot(dtype, n1, n2, n3, p1, p2, p3, s1, s2, s3, P1=True, P2=False, P3=True):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1, n2, n3], periods=[P1, P2, P3], comm=comm)

    # Partition the points
    npts = [n1, n2, n3]
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2, p3], shifts=[s1, s2, s3])

    # Create vector space and stencil vectors
    V = StencilVectorSpace(cart, dtype=dtype)
    x = StencilVector(V)
    y = StencilVector(V)

    # Fill the vectors with data
    if dtype == complex:
        f1 = lambda i1, i2, i3: 100 * i1 + i2+ 1j * i3
        f2 = lambda i1, i2, i3: 10 * i3 - i1- 10j * i2
    else:
        f1 = lambda i1, i2, i3: 100 * i1 + i2+ i3
        f2 = lambda i1, i2, i3: 10 * i3 - i1- i2

    for i1 in range(V.starts[0], V.ends[0] + 1):
        for i2 in range(V.starts[1], V.ends[1] + 1):
            for i3 in range(V.starts[2], V.ends[2] + 1):
                x[i1, i2, i3] = f1(i1,i2,i3)
                x[i1, i2, i3] = f2(i1,i2,i3)

    # Create scalar product (x,y) and (y,x)
        res1 = x.inner(y)
        res2 = y.inner(x)
    # Compute exact value with Numpy dot

    if dtype == complex:
        res_ex1 = comm.allreduce(np.vdot(x.toarray(), y.toarray()))
        res_ex2 = comm.allreduce(np.vdot(y.toarray(), x.toarray()))
    else:
        res_ex1 = comm.allreduce(np.dot(x.toarray(), y.toarray()))
        res_ex2 = res_ex1

    # Compute axpy exact sol
    if dtype == complex:
        cst = 5j
    else:
        cst = 5

    z3 = x + cst * y
    x.mul_iadd(cst, y)

    # Test exact value and symmetry of the scalar product
    assert np.allclose(x._data, z3._data)

    assert res1 == res_ex1
    assert res2 == res_ex2

#===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================
if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
