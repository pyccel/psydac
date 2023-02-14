import pytest
import numpy as np

from psydac.linalg.stencil import StencilVectorSpace
from psydac.ddm.cart import DomainDecomposition, CartDecomposition, find_mpi_type


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
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('s1', [1, 2])

def test_stencil_vector_space_1d_serial_init(dtype, n1, p1, P1, s1):
    D = DomainDecomposition([n1], periods=[P1])

    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[s1])
    V = StencilVectorSpace(C, dtype=dtype)

    assert V.dimension == n1
    assert V.dtype == dtype
    assert V.mpi_type == find_mpi_type(dtype)
    assert V.shape == (n1 + 2 * p1*s1,)
    assert not V.parallel
    assert V.cart == C
    assert V.npts == (n1,)
    assert V.starts == (0,)
    assert V.ends == (n1 - 1,)
    assert V.parent_starts == (None,)
    assert V.parent_ends == (None,)
    assert V.pads == (p1,)
    assert V.periods == (P1,)
    assert V.shifts == (s1,)
    assert V.ndim == 1
    assert V.interfaces == type(type.__dict__)({})
# ===============================================================================

@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [1, 7])
@pytest.mark.parametrize('n2', [1, 5])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [1, 2])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True, False])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [1, 2])

def test_stencil_vector_space_2d_serial_init(dtype, n1, n2, p1, p2, P1, P2, s1, s2):
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])
    V = StencilVectorSpace(C, dtype=dtype)

    assert V.dimension == n1 * n2
    assert V.dtype == dtype
    assert V.mpi_type == find_mpi_type(dtype)
    assert V.shape == ((n1 + 2 * p1*s1), (n2 + 2 * p2*s2))
    assert not V.parallel
    assert V.cart == C
    assert V.npts == (n1, n2)
    assert V.starts == (0, 0)
    assert V.ends == (n1 - 1, n2 - 1)
    assert V.parent_starts == (None, None)
    assert V.parent_ends == (None, None)
    assert V.pads == (p1, p2)
    assert V.periods == (P1,P2)
    assert V.shifts == (s1, s2)
    assert V.ndim == 2
    assert V.interfaces == type(type.__dict__)({})
# ===============================================================================

@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [1, 9])
@pytest.mark.parametrize('n2', [1, 7])
@pytest.mark.parametrize('n3', [1, 5])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [1, 2])
@pytest.mark.parametrize('p3', [1, 2])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True, False])
@pytest.mark.parametrize('P3', [True, False])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [1, 2])
@pytest.mark.parametrize('s3', [1, 2])

def test_stencil_vector_space_3d_serial_init(dtype, n1, n2, n3, p1, p2, p3, P1, P2, P3, s1, s2, s3):
    D = DomainDecomposition([n1,n2,n3], periods=[P1, P2, P3])

    npts = [n1,n2, n3]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2, p3], shifts=[s1, s2, s3])
    V = StencilVectorSpace(C, dtype=dtype)

    assert V.dimension == n1*n2*n3
    assert V.dtype == dtype
    assert V.mpi_type == find_mpi_type(dtype)
    assert V.shape == (n1 + 2 * p1*s1, n2 + 2 * p2*s2, n3 + 2 * p3*s3)
    assert not V.parallel
    assert V.cart == C
    assert V.npts == (n1, n2, n3)
    assert V.starts == (0, 0, 0)
    assert V.ends == (n1 - 1, n2 - 1, n3 - 1)
    assert V.parent_starts == (None, None, None)
    assert V.parent_ends == (None, None, None)
    assert V.pads == (p1, p2, p3)
    assert V.periods == (P1, P2, P3)
    assert V.shifts == (s1, s2, s3)
    assert V.ndim == 3
    assert V.interfaces == type(type.__dict__)({})
# ===============================================================================

@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [2, 9])
@pytest.mark.parametrize('n2', [2, 7])
@pytest.mark.parametrize('n3', [2, 5])

def test_stencil_vector_space_3D_serial_parent(dtype, n1, n2, n3, P1=True, P2=False, P3=True):
    D = DomainDecomposition([n1, n2, n3], periods=[P1, P2, P3])

    npts_red = [1, 1, 1]
    npts = [n1, n2, n3]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[1, 1, 1], shifts=[1, 1, 1])
    Cred=C.reduce_npts(npts_red, global_starts, global_ends, [1, 1, 1])
    V = StencilVectorSpace(Cred, dtype=dtype)

    assert V.dimension == 1
    assert V.dtype == dtype
    assert V.starts == (0, 0, 0)
    assert V.ends == (n1-1, n2-1, n3-1)
    assert V.parent_starts == (0, 0, 0)
    assert V.parent_ends == (n1-1, n2-1, n3-1)
# ===============================================================================

@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [2, 9])
@pytest.mark.parametrize('n2', [2, 7])

def test_stencil_vector_space_2D_serial_zeros(dtype, n1, n2, p1=1, p2=1, P1=True, P2=False):
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[1, 1])
    V = StencilVectorSpace(C, dtype=dtype)
    x = V.zeros()

    assert x.space is V
    assert x.dtype == dtype
    assert x.starts == (0, 0)
    assert x.ends   == (n1-1, n2-1)
    assert x._data.shape == (n1+2*p1, n2+2*p2)
    assert x.pads == (p1, p2)
    assert x._data.dtype == dtype
    assert np.array_equal(x._data, np.zeros((n1+2*p1, n2+2*p2), dtype=dtype))


# TODO : test for set_interface


# ===============================================================================
# PARALLEL TESTS
# ===============================================================================

@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [1, 7])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parallel

def test_stencil_vector_space_1d_serial_init(dtype, n1, p1, P1, s1):

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    D = DomainDecomposition([n1], periods=[P1], comm=comm)

    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[s1])
    V = StencilVectorSpace(C, dtype=dtype)

    assert V.dimension == n1
    assert V.dtype == dtype
    assert V.mpi_type == find_mpi_type(dtype)
    assert V.shape == (n1 + 2 * p1*s1,)
    assert V.parallel
    assert V.cart == C
    assert V.npts == (n1,)
    assert V.starts == (0,)
    assert V.ends == (n1 - 1,)
    assert V.parent_starts == (None,)
    assert V.parent_ends == (None,)
    assert V.pads == (p1,)
    assert V.periods == (P1,)
    assert V.shifts == (s1,)
    assert V.ndim == 1
    assert V.interfaces == type(type.__dict__)({})
# ===============================================================================

@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [1, 7])
@pytest.mark.parametrize('n2', [1, 5])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [1, 2])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True, False])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [1, 2])
@pytest.mark.parallel

def test_stencil_vector_space_2d_serial_init(dtype, n1, n2, p1, p2, P1, P2, s1, s2):

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    D = DomainDecomposition([n1, n2], periods=[P1, P2], comm=comm)

    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])
    V = StencilVectorSpace(C, dtype=dtype)

    assert V.dimension == n1 * n2
    assert V.dtype == dtype
    assert V.mpi_type == find_mpi_type(dtype)
    assert V.shape == ((n1 + 2 * p1*s1), (n2 + 2 * p2*s2))
    assert V.parallel
    assert V.cart == C
    assert V.npts == (n1, n2)
    assert V.starts == (0, 0)
    assert V.ends == (n1 - 1, n2 - 1)
    assert V.parent_starts == (None, None)
    assert V.parent_ends == (None, None)
    assert V.pads == (p1, p2)
    assert V.periods == (P1,P2)
    assert V.shifts == (s1, s2)
    assert V.ndim == 2
    assert V.interfaces == type(type.__dict__)({})
# ===============================================================================

@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [1, 9])
@pytest.mark.parametrize('n2', [1, 7])
@pytest.mark.parametrize('n3', [1, 5])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [1, 2])
@pytest.mark.parametrize('p3', [1, 2])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True, False])
@pytest.mark.parametrize('P3', [True, False])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [1, 2])
@pytest.mark.parametrize('s3', [1, 2])
@pytest.mark.parallel

def test_stencil_vector_space_3d_serial_init(dtype, n1, n2, n3, p1, p2, p3, P1, P2, P3, s1, s2, s3):

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    D = DomainDecomposition([n1, n2, n3], periods=[P1, P2, P3], comm=comm)

    npts = [n1,n2, n3]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2, p3], shifts=[s1, s2, s3])
    V = StencilVectorSpace(C, dtype=dtype)

    assert V.dimension == n1*n2*n3
    assert V.dtype == dtype
    assert V.mpi_type == find_mpi_type(dtype)
    assert V.shape == (n1 + 2 * p1*s1, n2 + 2 * p2*s2, n3 + 2 * p3*s3)
    assert V.parallel
    assert V.cart == C
    assert V.npts == (n1, n2, n3)
    assert V.starts == (0, 0, 0)
    assert V.ends == (n1 - 1, n2 - 1, n3 - 1)
    assert V.parent_starts == (None, None, None)
    assert V.parent_ends == (None, None, None)
    assert V.pads == (p1, p2, p3)
    assert V.periods == (P1, P2, P3)
    assert V.shifts == (s1, s2, s3)
    assert V.ndim == 3
    assert V.interfaces == type(type.__dict__)({})
# ===============================================================================

@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [2, 9])
@pytest.mark.parametrize('n2', [2, 7])
@pytest.mark.parametrize('n3', [2, 5])
@pytest.mark.parallel

def test_stencil_vector_space_3D_serial_parent(dtype, n1, n2, n3, P1=True, P2=False, P3=True):

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    D = DomainDecomposition([n1, n2, n3], periods=[P1, P2, P3], comm=comm)

    npts_red = [1, 1, 1]
    npts = [n1, n2, n3]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[1, 1, 1], shifts=[1, 1, 1])
    Cred=C.reduce_npts(npts_red, global_starts, global_ends, [1, 1, 1])
    V = StencilVectorSpace(Cred, dtype=dtype)

    assert V.dimension == 1
    assert V.dtype == dtype
    assert V.starts == (0, 0, 0)
    assert V.ends == (n1-1, n2-1, n3-1)
    assert V.parent_starts == (0, 0, 0)
    assert V.parent_ends == (n1-1, n2-1, n3-1)
# ===============================================================================

@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [2, 9])
@pytest.mark.parametrize('n2', [2, 7])
@pytest.mark.parallel

def test_stencil_vector_space_2D_serial_zeros(dtype, n1, n2, p1=1, p2=1, P1=True, P2=False):

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    D = DomainDecomposition([n1, n2], periods=[P1, P2], comm=comm)

    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[1, 1])
    V = StencilVectorSpace(C, dtype=dtype)
    x = V.zeros()

    assert x.space is V
    assert x.dtype == dtype
    assert x.starts == (0, 0)
    assert x.ends   == (n1-1, n2-1)
    assert x._data.shape == (n1+2*p1, n2+2*p2)
    assert x.pads == (p1, p2)
    assert x._data.dtype == dtype
    assert np.array_equal(x._data, np.zeros((n1+2*p1, n2+2*p2), dtype=dtype))


# TODO : test for set_interface
