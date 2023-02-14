# coding: utf-8

import pytest
import numpy as np

from psydac.linalg.stencil import StencilVectorSpace, StencilVector
from psydac.linalg.utilities import array_to_psydac
from psydac.ddm.cart import DomainDecomposition, CartDecomposition


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
@pytest.mark.parametrize('p2', [1, 2])
def test_stencil_vector_2d_serial_init(dtype, n1, n2, p1, p2, P1=True, P2=False):
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[1, 1])
    V = StencilVectorSpace(C, dtype=dtype)
    x = StencilVector(V)

    assert x.space is V
    assert x.dtype == dtype
    assert x.starts == (0, 0)
    assert x.ends == (n1 - 1, n2 - 1)
    assert x.pads == (p1, p2)
    assert x._data.shape == (n1 + 2 * p1, n2 + 2 * p2)
    assert x._data.dtype == dtype
    assert not x.ghost_regions_in_sync


# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [1, 7])
@pytest.mark.parametrize('n2', [1, 5])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [1, 2])
def test_stencil_vector_2d_serial_copy(dtype, n1, n2, p1, p2, P1=True, P2=False):
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[1, 1])

    V = StencilVectorSpace(C, dtype=dtype)
    x = StencilVector(V)

    for i1 in range(n1):
        for i2 in range(n2):
            x[i1, i2] = 10 * i1 + i2

    z = x.copy()

    assert isinstance(z, StencilVector)
    assert z.space is V
    assert z._data is not x._data
    assert z.dtype == dtype
    assert np.array_equal(x._data, z._data)


# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [7, 15])
@pytest.mark.parametrize('n2', [8, 12])
@pytest.mark.parametrize('p1', [1, 2, 3])
@pytest.mark.parametrize('p2', [1, 2, 3])
def test_stencil_vector_2d_basic_ops(dtype, n1, n2, p1, p2, P1=True, P2=False):
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[1, 1])

    V = StencilVectorSpace(C, dtype=dtype)
    M = StencilVector(V)

    # take random data, but determinize it
    np.random.seed(2)
    if dtype == float:
        M._data[:] = np.random.random(M._data.shape)
    else:
        M._data[:] = np.random.random(M._data.shape) + 1j * np.random.random(M._data.shape)

    # we try to go for equality here...
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
@pytest.mark.parametrize('p2', [1, 2])
def test_stencil_matrix_2d_serial_toarray(dtype, n1, n2, p1, p2, P1=True, P2=False):
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[1, 1])

    V = StencilVectorSpace(C, dtype=dtype)
    x = StencilVector(V)

    for i1 in range(n1):
        for i2 in range(n2):
            x[i1, i2] = 10 * i1 + i2

    xc = x.toarray()
    xf = x.toarray(order='F')
    xcp = x.toarray(with_pads=True)
    xfp = x.toarray(order='F', with_pads=True)

    zc = np.zeros((n1 * n2))
    zf = np.zeros((n1 * n2))
    zcp = np.zeros(((n1 + 2 * p1) * (n2 + 2 * p2)))
    zfp = np.zeros(((n1 + 2 * p1) * (n2 + 2 * p2)))
    for i1 in range(n1):
        for i2 in range(n2):
            zc[i1 * n2 + i2] = 10 * i1 + i2
            zf[i1 + i2 * n1] = 10 * i1 + i2
    # Verify toarray() with and without padding
    for (x, z) in zip([xc, xf, xcp, xc], [zc, zf, zc, zf]):
        assert x.shape == (n1 * n2,)
        assert x.dtype == dtype
        assert np.array_equal(xc, zc)


# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [1, 7])
@pytest.mark.parametrize('n2', [1, 5])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [1, 2])
def test_stencil_vector_2d_serial_math(dtype, n1, n2, p1, p2, P1=True, P2=False):
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[1, 1])

    V = StencilVectorSpace(C, dtype=dtype)
    x = StencilVector(V)
    y = StencilVector(V)
    if dtype == complex:
        for i1 in range(n1):
            for i2 in range(n2):
                x[i1, i2] = 10 * i1 + 1j * i2
    else:
        for i1 in range(n1):
            for i2 in range(n2):
                x[i1, i2] = 10 * i1 + i2

    y[:, :] = 42.0

    r1 = x + y
    r2 = x - y
    r3 = 2 * x
    r4 = x * 2

    xa = x.toarray()
    ya = y.toarray()

    r1_exact = xa + ya
    r2_exact = xa - ya
    r3_exact = 2 * xa
    r4_exact = xa * 2

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
@pytest.mark.parametrize('p2', [1, 2])
def test_stencil_vector_2d_serial_dot(dtype, n1, n2, p1, p2, P1=True, P2=False):
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[1, 1])

    V = StencilVectorSpace(C, dtype)
    x = StencilVector(V)
    y = StencilVector(V)
    if dtype == complex:
        for i1 in range(n1):
            for i2 in range(n2):
                x[i1, i2] = 10j * i1 + i2
                y[i1, i2] = 10j * i2 - i1
    else:
        for i1 in range(n1):
            for i2 in range(n2):
                x[i1, i2] = 10 * i1 + i2
                y[i1, i2] = 10 * i2 - i1

    z1 = x.dot(y)
    z2 = y.dot(x)

    z_exact = np.dot(x.toarray(), y.toarray())

    assert z1.dtype == dtype
    assert z2.dtype == dtype
    assert z1 == z_exact
    assert z2 == z_exact


# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [1, 7])
@pytest.mark.parametrize('n2', [1, 5])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [1, 2])
def test_stencil_vector_2d_serial_vdot(dtype, n1, n2, p1, p2, P1=True, P2=False):
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[1, 1])

    V = StencilVectorSpace(C, dtype)
    x = StencilVector(V)
    y = StencilVector(V)
    if dtype == complex:
        for i1 in range(n1):
            for i2 in range(n2):
                x[i1, i2] = 10j * i1 + i2
                y[i1, i2] = 10j * i2 - i1
    else:
        for i1 in range(n1):
            for i2 in range(n2):
                x[i1, i2] = 10 * i1 + i2
                y[i1, i2] = 10 * i2 - i1

    z1 = x.vdot(y)
    z2 = y.vdot(x)

    z_exact = np.vdot(x.toarray(), y.toarray())

    assert z1.dtype == dtype
    assert z2.dtype == dtype
    assert z1 == z_exact
    assert z2 == z_exact.conjugate()


# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [1, 7])
@pytest.mark.parametrize('n2', [1, 5])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [1, 2])
def test_stencil_vector_2d_serial_conjugate(dtype, n1, n2, p1, p2, P1=True, P2=False):
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[1, 1])

    V = StencilVectorSpace(C, dtype)
    x = StencilVector(V)
    if dtype == complex:
        for i1 in range(n1):
            for i2 in range(n2):
                x[i1, i2] = 10j * i1 + i2
    else:
        for i1 in range(n1):
            for i2 in range(n2):
                x[i1, i2] = 10 * i1 + i2

    z = x.conjugate()

    z_exact = x._data.conjugate()

    assert z.dtype == dtype
    assert np.array_equal(z._data, z_exact)


# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [1, 7])
@pytest.mark.parametrize('n2', [1, 5])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [1, 2])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True, False])
def test_stencil_2d_array_to_psydac(dtype, n1, n2, p1, p2, P1, P2):
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[1, 1])

    V = StencilVectorSpace(C, dtype=dtype)
    x = StencilVector(V)

    if dtype == complex:
        for i1 in range(n1):
            for i2 in range(n2):
                x[i1, i2] = 10j * i1 + i2
    else:
        for i1 in range(n1):
            for i2 in range(n2):
                x[i1, i2] = 10 * i1 + i2

    xa = x.toarray()
    v = array_to_psydac(xa, V)

    assert v.dtype == dtype
    # assert np.allclose( xa , v.toarray() )
    assert np.array_equal(xa, v.toarray())


# ===============================================================================
# PARALLEL TESTS
# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [12, 22])
@pytest.mark.parametrize('n2', [12, 24])
@pytest.mark.parametrize('p1', [1, 3, 4])
@pytest.mark.parametrize('p2', [1, 3, 4])
@pytest.mark.parallel
def test_stencil_vector_2d_parallel_init(dtype, n1, n2, p1, p2, P1=True, P2=False):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    D = DomainDecomposition([n1, n2], periods=[P1, P2], comm=comm)

    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[1, 1])

    V = StencilVectorSpace(cart, dtype=dtype)
    x = StencilVector(V)

    assert x.space is V
    assert x.dtype == dtype
    assert x.starts == (0, 0)
    assert x.ends == (n1 - 1, n2 - 1)
    assert x.pads == (p1, p2)
    assert x._data.shape == (n1 + 2 * p1, n2 + 2 * p2)
    assert x._data.dtype == dtype
    assert not x.ghost_regions_in_sync


# ===============================================================================
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [20, 64])
@pytest.mark.parametrize('n2', [24, 64])
@pytest.mark.parametrize('p1', [1, 3, 4])
@pytest.mark.parametrize('p2', [1, 3, 4])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True, False])
@pytest.mark.parallel
def test_stencil_vector_2d_parallel_toarray(dtype, n1, n2, p1, p2, P1, P2):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    D = DomainDecomposition([n1, n2], periods=[P1, P2], comm=comm)

    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[1, 1])

    V = StencilVectorSpace(cart, dtype=dtype)
    x = StencilVector(V)

    # Values in 2D grid (global indexing)
    if dtype==complex:
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
    z = np.zeros((n1, n2), dtype=dtype)
    for i1 in range(cart.starts[0], cart.ends[0] + 1):
        for i2 in range(cart.starts[1], cart.ends[1] + 1):
            z[i1, i2] = f(i1, i2)

    # Verify toarray() without padding
    xa = x.toarray()
    za = z.reshape(-1)

    assert xa.dtype == dtype
    assert xa.shape == (n1 * n2,)
    assert np.array_equal(xa, za)

    # Verify toarray() with padding: internal region should not change
    xe = x.toarray(with_pads=True)
    xe = xe.reshape(n1, n2)
    index = tuple(slice(s, e + 1) for s, e in zip(cart.starts, cart.ends))

    # print()
    # print(z)
    # print()
    # print(xe.reshape(n1, n2))

    assert xe.dtype == dtype
    assert xe.shape == (n1 ,n2)
    assert np.all(xe[index] == z[index])
    # assert np.array_equal(xe[index] == z[index])

    # TODO: test that ghost regions have been properly copied to 'xe' array
    # TODO: x.toarray ne marche pas pour les complexes
# ===============================================================================
@pytest.mark.parametrize('n1', [12, 24])
@pytest.mark.parametrize('n2', [12, 24])
@pytest.mark.parametrize('p1', [1, 3, 4])
@pytest.mark.parametrize('p2', [1, 3, 4])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True, False])
@pytest.mark.parallel
def test_stencil_vector_2d_parallel_dot(n1, n2, p1, p2, P1, P2):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    D = DomainDecomposition([n1, n2], periods=[P1, P2], comm=comm)

    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[1, 1])

    V = StencilVectorSpace(cart)
    x = StencilVector(V)
    y = StencilVector(V)

    for i1 in range(V.starts[0], V.ends[0] + 1):
        for i2 in range(V.starts[1], V.ends[1] + 1):
            x[i1, i2] = 10 * i1 + i2
            y[i1, i2] = 10 * i2 - i1

    res1 = x.dot(y)
    res2 = y.dot(x)
    res_ex = comm.allreduce(np.dot(x.toarray(), y.toarray()))

    assert res1 == res_ex
    assert res2 == res_ex


# TODO: add test str, topestc, update_ghost_region, exchange_assembly_data, right multiplication
# ===============================================================================
if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
