#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import pytest
import numpy as np

from psydac.linalg.stencil import StencilVectorSpace, StencilVector
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
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('P1', [True, False])

def test_stencil_vector_space_1d_serial_init(dtype, n1, p1, s1, P1):
    # Create domain decomposition
    D = DomainDecomposition([n1], periods=[P1])

    # Partition the points
    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    # Create cart and  vector space
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[s1])
    V = StencilVectorSpace(C, dtype=dtype)

    # Test properties of the vector space
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
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [2])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True])

def test_stencil_vector_space_2d_serial_init(dtype, n1, n2, p1, p2, s1, s2, P1, P2):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    # Create cart and  vector space
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])
    V = StencilVectorSpace(C, dtype=dtype)

    # Test properties of the vector space
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
@pytest.mark.parametrize('n3', [5])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('p3', [1])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [2])
@pytest.mark.parametrize('s3', [1])

def test_stencil_vector_space_3d_serial_init(dtype, n1, n2, n3, p1, p2, p3, s1, s2, s3, P1=True, P2=False, P3=True):
    # Create domain decomposition
    D = DomainDecomposition([n1,n2,n3], periods=[P1, P2, P3])

    # Partition the points
    npts = [n1,n2, n3]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    # Create cart and  vector space
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2, p3], shifts=[s1, s2, s3])
    V = StencilVectorSpace(C, dtype=dtype)

    # Test properties of the vector space
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
@pytest.mark.parametrize('n1', [2])
@pytest.mark.parametrize('n2', [2])
@pytest.mark.parametrize('n3', [2])

def test_stencil_vector_space_3D_serial_parent(dtype, n1, n2, n3, P1=True, P2=False, P3=True):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2, n3], periods=[P1, P2, P3])

    # Partition the points for our domain and our reduced domain
    npts_red = [1, 1, 1]
    npts = [n1, n2, n3]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    # Create a cart
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[1, 1, 1], shifts=[1, 1, 1])

    # Create q reduced cart and vector space on it
    Cred = C.reduce_npts(npts_red, global_starts, global_ends, [1, 1, 1])
    V = StencilVectorSpace(Cred, dtype=dtype)

    # Test properties of the vector space
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
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [2])

def test_stencil_vector_space_2D_serial_zeros(dtype, n1, n2, p1, p2, s1, s2, P1=True, P2=False):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    # Create cart and  vector space
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])
    V = StencilVectorSpace(C, dtype=dtype)

    # Create a zero vector on this vector space
    x = V.zeros()

    # Test properties of the vector
    assert x.space is V
    assert x.dtype == dtype
    assert x.starts == (0, 0)
    assert x.ends   == (n1-1, n2-1)
    assert x._data.shape == (n1+2*p1*s1, n2+2*p2*s2)
    assert x.pads == (p1, p2)
    assert x._data.dtype == dtype
    assert np.array_equal(x._data, np.zeros((n1+2*p1*s1, n2+2*p2*s2), dtype=dtype))
# ===============================================================================

@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [5, 9])
@pytest.mark.parametrize('n2', [5, 7])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [2])
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('ext', [-1, 1])

def test_stencil_vector_space_2D_serial_set_interface(dtype, n1, n2, p1, p2, s1, s2, axis, ext, P1=True, P2=False):
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2])

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    # Create cart and  vector space
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])
    V = StencilVectorSpace(C, dtype=dtype)

    # Create an interface on this vector space
    V.set_interface(axis, ext, C)
    V_inter=V.interfaces[axis, ext]

    # Test the propertiesof this interface
    assert isinstance(V_inter, StencilVectorSpace)
    assert V_inter.dimension == n1 * n2
    assert V_inter.dtype == dtype
    assert V_inter.mpi_type == find_mpi_type(dtype)
    assert not V_inter.parallel
    assert isinstance(V_inter.cart, CartDecomposition)
    assert V_inter.npts == (n1, n2)
    assert V_inter.parent_starts == (None, None)
    assert V_inter.parent_ends == (None, None)
    assert V_inter.pads == (p1, p2)
    assert V_inter.periods == (P1, P2)
    assert V_inter.shifts == (s1, s2)
    assert V_inter.ndim == 2
    assert V_inter.interfaces == type(type.__dict__)({})

    if axis == 0:
        assert V_inter.shape == ((p1+1 + 2 * p1 * s1), (n2 + 2 * p2 * s2))
        if ext == 1:
            assert V_inter.starts == (n1-1-p1, 0)
            assert V_inter.ends == (n1-1, n2-1)
        else:
            assert V_inter.starts == (0, 0)
            assert V_inter.ends == (p1, n2-1)
    else:
        assert V_inter.shape == ((n1 + 2 * p1 * s1), p2+1+2*p2*s2)
        if ext == 1:
            assert V_inter.starts == (0, n2-1-p2)
            assert V_inter.ends == (n1-1, n2-1)
        else:
            assert V_inter.starts == (0, 0)
            assert V_inter.ends == (n1-1, p2)
# ===============================================================================
# PARALLEL TESTS
# ===============================================================================

@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [15, 30])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.mpi

def test_stencil_vector_space_1d_parallel_init(dtype, n1, p1, s1, P1):

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1], periods=[P1], comm=comm)

    # Partition the points
    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    # Create cart and  vector space
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[s1])
    V = StencilVectorSpace(C, dtype=dtype)


    # Test properties of the vector space
    assert V.dimension == n1
    assert V.dtype == dtype
    assert V.mpi_type == find_mpi_type(dtype)
    assert V.shape == (V.ends[0]+1-V.starts[0] + 2 * p1*s1,)
    assert V.parallel
    assert V.cart == C
    assert V.npts == (n1,)
    assert V.starts == C.starts
    assert V.ends == C.ends
    assert V.parent_starts == (None,)
    assert V.parent_ends == (None,)
    assert V.pads == (p1,)
    assert V.periods == (P1,)
    assert V.shifts == (s1,)
    assert V.ndim == 1
    assert V.interfaces == type(type.__dict__)({})
# ===============================================================================

@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [15, 30])
@pytest.mark.parametrize('n2', [20, 40])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [2])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [2])
@pytest.mark.parametrize('P1', [True, False])
@pytest.mark.parametrize('P2', [True])
@pytest.mark.mpi

def test_stencil_vector_space_2d_parallel_init(dtype, n1, n2, p1, p2, s1, s2, P1, P2):

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2], comm=comm)

    # Partition the points
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    # Create cart and  vector space
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[s1, s2])
    V = StencilVectorSpace(C, dtype=dtype)

    # Test properties of the vector space
    assert V.dimension == n1 * n2
    assert V.dtype == dtype
    assert V.mpi_type == find_mpi_type(dtype)
    assert V.shape == ((V.ends[0]+1-V.starts[0] + 2 * p1*s1), (V.ends[1]+1-V.starts[1] + 2 * p2*s2))
    assert V.parallel
    assert V.cart == C
    assert V.npts == (n1, n2)
    assert V.starts == C.starts
    assert V.ends == C.ends
    assert V.parent_starts == (None, None)
    assert V.parent_ends == (None, None)
    assert V.pads == (p1, p2)
    assert V.periods == (P1,P2)
    assert V.shifts == (s1, s2)
    assert V.ndim == 2
    assert V.interfaces == type(type.__dict__)({})
# ===============================================================================

@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [15, 30])
@pytest.mark.parametrize('n2', [20, 40])
@pytest.mark.parametrize('n3', [10, 25])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parametrize('p2', [1, 2])
@pytest.mark.parametrize('p3', [1])
@pytest.mark.parametrize('s1', [1, 2])
@pytest.mark.parametrize('s2', [1, 2])
@pytest.mark.parametrize('s3', [1])
@pytest.mark.mpi

def test_stencil_vector_space_3d_parallel_init(dtype, n1, n2, n3, p1, p2, p3, s1, s2, s3, P1=True, P2=False, P3=True):

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1, n2, n3], periods=[P1, P2, P3], comm=comm)

    # Partition the points
    npts = [n1,n2, n3]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    # Create cart and  vector space
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2, p3], shifts=[s1, s2, s3])
    V = StencilVectorSpace(C, dtype=dtype)

    # Test properties of the vector space
    assert V.dimension == n1*n2*n3
    assert V.dtype == dtype
    assert V.mpi_type == find_mpi_type(dtype)
    assert V.shape == (V.ends[0]+1-V.starts[0] + 2 * p1*s1, V.ends[1]+1-V.starts[1] + 2 * p2*s2, V.ends[2]+1-V.starts[2] + 2 * p3*s3)
    assert V.parallel
    assert V.cart == C
    assert V.npts == (n1, n2, n3)
    assert V.starts == C.starts
    assert V.ends == C.ends
    assert V.parent_starts == (None, None, None)
    assert V.parent_ends == (None, None, None)
    assert V.pads == (p1, p2, p3)
    assert V.periods == (P1, P2, P3)
    assert V.shifts == (s1, s2, s3)
    assert V.ndim == 3
    assert V.interfaces == type(type.__dict__)({})
# ===============================================================================

@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('n1', [15])
@pytest.mark.parametrize('n2', [20])
@pytest.mark.mpi

def test_stencil_vector_space_2D_parallel_parent(dtype, n1, n2, P1=True, P2=False):

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1, n2], periods=[P1, P2], comm=comm)

    # Partition the points for our domain and its reduced version
    npts_red = [1, 1]
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    # Create the cart
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[1, 1], shifts=[1, 1])

    # Create the cart reduced and a vector space
    Cred = C.reduce_npts(npts_red, global_starts, global_ends, [1, 1])
    V = StencilVectorSpace(Cred, dtype=dtype)

    # Test properties of the vector space
    assert V.dimension == 1
    assert V.dtype == dtype
    assert V.starts == Cred.starts
    assert V.ends == Cred.ends
    assert V.parent_starts == Cred.parent_starts
    assert V.parent_ends == Cred.parent_ends
# ===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================
if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
