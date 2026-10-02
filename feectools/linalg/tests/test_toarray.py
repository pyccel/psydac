# -*- coding: UTF-8 -*-
#
# Tests for the generic LinearOperator.toarray() and LinearOperator.tosparse(),
# which assemble the matrix of any linear operator from its dot() method.
#
import pytest
import cunumpy as xp

from feectools.ddm.mpi import mpi as MPI
from feectools.ddm.cart import DomainDecomposition, CartDecomposition
from feectools.linalg.basic import LinearOperator, MatrixFreeLinearOperator, IdentityOperator
from feectools.linalg.stencil import StencilVectorSpace, StencilMatrix
from feectools.linalg.block import BlockVectorSpace, BlockLinearOperator
from feectools.linalg.solvers import inverse
from feectools.linalg.tests.test_block import compute_global_starts_ends

SPARSE_FORMATS = ['csr', 'csc', 'bsr', 'lil', 'dok', 'coo', 'dia']

#===============================================================================
# HELPERS
#===============================================================================
def get_space(npts, pads, periods, comm=None):
    D = DomainDecomposition(npts, periods=periods, comm=comm)
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=pads, shifts=[1] * len(npts))
    return StencilVectorSpace(C)

def get_random_matrix(V, W, seed):
    """ Random StencilMatrix, identical on all ranks in the global numbering. """
    rng = xp.random.default_rng(seed)
    M = StencilMatrix(V, W)
    # Draw the full (global) band and keep the local part, so the matrix
    # does not depend on the domain decomposition.
    shape = tuple(W.npts) + M._data.shape[W.ndim:]
    band = rng.random(shape) - 0.5
    idx = tuple(slice(s, e + 1) for s, e in zip(W.starts, W.ends))
    M[idx] = band[idx]
    M.remove_spurious_entries()
    return M

def matrix_free(A):
    """ Hide the explicit matrix of A, so that the generic toarray() is used. """
    return MatrixFreeLinearOperator(A.domain, A.codomain, lambda v, out=None: A.dot(v, out=out))

def reference(A):
    """ Global dense matrix of A, from its own (row-local in parallel) tosparse(). """
    local = A.tosparse().toarray()
    comm = A.domain.cart.comm if isinstance(A.domain, StencilVectorSpace) else A.domain.spaces[0].cart.comm
    if A.domain.parallel:
        glob = xp.zeros_like(local)
        comm.Allreduce(local, glob, op=MPI.SUM)
        return glob
    return local

def check_all(O, ref):
    """ Check dense, in-place and all sparse outputs of the generic toarray()/tosparse(). """
    assert xp.allclose(O.toarray(), ref)

    out = xp.full(O.shape, 7.0)
    res = O.toarray(out=out)
    assert res is out
    assert xp.allclose(out, ref)

    for fmt in SPARSE_FORMATS:
        S = O.toarray(is_sparse=True, format=fmt)
        assert S.format == fmt
        assert S.shape == O.shape
        assert xp.allclose(S.toarray(), ref)

    S = O.tosparse()
    assert S.format == 'csr'
    assert xp.allclose(S.toarray(), ref)
    assert O.tosparse('csc').format == 'csc'

def stencil_case(periods, comm):
    V = get_space([6, 5], [2, 1], periods, comm=comm)
    return get_random_matrix(V, V, seed=0)

def block_case(periods, comm):
    """ 2x2 block operator, and a nested block operator [[B, 0], [0, A]]. """
    V = get_space([6, 5], [2, 1], periods, comm=comm)
    A = get_random_matrix(V, V, seed=0)
    A01 = get_random_matrix(V, V, seed=1)
    VV = BlockVectorSpace(V, V)
    B = BlockLinearOperator(VV, VV, blocks=[[A, A01], [None, A]])
    Vn = BlockVectorSpace(VV, V)
    N = BlockLinearOperator(Vn, Vn, blocks=[[B, None], [None, A]])
    return A, B, N

#===============================================================================
# SERIAL TESTS
#===============================================================================
@pytest.mark.parametrize('periods', [[False, False], [True, False], [True, True]])
def test_toarray_stencil(periods):
    A = stencil_case(periods, comm=None)
    check_all(matrix_free(A), reference(A))

@pytest.mark.parametrize('periods', [[False, False], [True, True]])
def test_toarray_block(periods):
    A, B, N = block_case(periods, comm=None)
    refA, refB = reference(A), reference(B)
    check_all(matrix_free(B), refB)

    Z = xp.zeros((refB.shape[0], refA.shape[1]))
    refN = xp.block([[refB, Z], [Z.T, refA]])
    check_all(matrix_free(N), refN)

def test_toarray_composite_operators():
    """ Operators that used to raise NotImplementedError in toarray()/tosparse(). """
    A = stencil_case([True, False], comm=None)
    ref = reference(A)
    V = A.domain

    assert xp.allclose((A @ A).toarray(), ref @ ref)
    assert xp.allclose((A ** 2).toarray(), ref @ ref)
    assert xp.allclose((A ** 2).tosparse().toarray(), ref @ ref)

    M = A.T @ A + IdentityOperator(V)
    refM = ref.T @ ref + xp.eye(ref.shape[0])
    Minv = inverse(M, 'cg', tol=1e-13, maxiter=1000)
    assert xp.allclose(Minv.toarray(), xp.linalg.inv(refM), atol=1e-8)

def test_toarray_invalid_input():
    O = matrix_free(stencil_case([False, False], comm=None))
    with pytest.raises(AssertionError):
        O.toarray(out=xp.zeros((O.shape[0], O.shape[1] + 1)))
    with pytest.raises(AssertionError):
        O.toarray(out=xp.zeros(O.shape), is_sparse=True)
    with pytest.raises(AssertionError):
        O.toarray(is_sparse=True, format='xyz')

#===============================================================================
# PARALLEL TESTS
#===============================================================================
@pytest.mark.parametrize('periods', [[False, False], [True, False], [True, True]])
@pytest.mark.parallel
def test_toarray_stencil_parallel(periods):
    A = stencil_case(periods, comm=MPI.COMM_WORLD)
    check_all(matrix_free(A), reference(A))

@pytest.mark.parametrize('periods', [[False, False], [True, True]])
@pytest.mark.parallel
def test_toarray_block_parallel(periods):
    A, B, N = block_case(periods, comm=MPI.COMM_WORLD)
    refA, refB = reference(A), reference(B)
    check_all(matrix_free(B), refB)

    Z = xp.zeros((refB.shape[0], refA.shape[1]))
    refN = xp.block([[refB, Z], [Z.T, refA]])
    check_all(matrix_free(N), refN)

@pytest.mark.parallel
def test_toarray_composite_operators_parallel():
    A = stencil_case([True, False], comm=MPI.COMM_WORLD)
    ref = reference(A)
    assert xp.allclose((A @ A).toarray(), ref @ ref)
    assert xp.allclose((A ** 2).tosparse().toarray(), ref @ ref)

#===============================================================================
if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
