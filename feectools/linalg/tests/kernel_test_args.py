"""Arguments of the stencil kernels for the parity tests.

Each builder returns the arguments of its kernel on the active backend, as ``StencilMatrix`` and
``StencilVectorSpace`` pass them, from random data that is the same on both backends (seeded with the case index).
The parity cases built from them are in :mod:`feectools.linalg.tests.cuda_parity_cases`.
"""
import numpy as np
import cunumpy as xp

from feectools.ddm.cart import DomainDecomposition, CartDecomposition
from feectools.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix

# (npts of the domain, npts of the codomain, pads): square matrices, and rectangular ones whose spaces
# differ by one point in a direction, which makes `add` zero (or one) there, as for derivative operators.
MATRIX_CASES = {
    1: [((24,), (24,), (2,)), ((23,), (24,), (2,)), ((24,), (23,), (3,))],
    2: [((10, 12), (10, 12), (2, 3)), ((11, 10), (12, 10), (2, 2)), ((12, 9), (11, 10), (1, 2))],
    3: [((7, 8, 9), (7, 8, 9), (1, 2, 3)), ((8, 8, 9), (9, 8, 10), (1, 2, 2)), ((6, 5, 7), (5, 6, 7), (2, 1, 1))],
}

# (npts, pads) of the vector spaces
VECTOR_CASES = {
    1: [((24,), (2,)), ((7,), (1,))],
    2: [((10, 12), (2, 3)), ((5, 9), (1, 1))],
    3: [((7, 8, 9), (1, 2, 3)), ((6, 5, 4), (2, 1, 1))],
}


def make_space(npts, pads, dtype=float):
    """A serial StencilVectorSpace with periodic directions."""
    ndim = len(npts)
    D = DomainDecomposition(list(npts), periods=[True] * ndim)
    global_starts, global_ends = [], []
    for axis in range(ndim):
        ee = D.global_element_ends[axis].copy()
        ee[-1] = npts[axis] - 1
        global_ends.append(ee)
        global_starts.append(xp.array([0] + (ee[:-1] + 1).tolist()))
    C = CartDecomposition(D, list(npts), global_starts, global_ends, pads=list(pads), shifts=[1] * ndim)
    return StencilVectorSpace(C, dtype=dtype)


def random_like(array, rng):
    """Random data of the shape and dtype of `array`, on the active backend."""
    shape = tuple(int(n) for n in array.shape)
    return xp.asarray(rng.random(shape).astype(array.dtype))


def stencil_matrix(npts_domain, npts_codomain, pads, rng):
    """A StencilMatrix with random entries (spurious entries removed), its domain and its codomain."""
    V = make_space(npts_domain, pads)
    W = V if npts_domain == npts_codomain else make_space(npts_codomain, pads)
    A = StencilMatrix(V, W)
    A._data[...] = random_like(A._data, rng)
    A.remove_spurious_entries()
    return A, V, W


def stencil_vector(V, rng):
    """A StencilVector of `V` with random entries and up-to-date ghost regions."""
    v = StencilVector(V)
    v._data[...] = random_like(v._data, rng)
    v.update_ghost_regions()
    return v


def dot_arguments(npts_domain, npts_codomain, pads, seed):
    """The arguments of ``stencil_dot_<n>d`` as ``StencilMatrix.dot`` passes them."""
    rng = np.random.default_rng(seed)
    A, V, W = stencil_matrix(npts_domain, npts_codomain, pads, rng)
    v = stencil_vector(V, rng)
    out = StencilVector(W)
    return (A._data, v._data, out._data, *A._args.values())


def transpose_arguments(npts_domain, npts_codomain, pads, seed):
    """The arguments of ``stencil_transpose_<n>d`` as ``StencilMatrix.transpose`` passes them."""
    rng = np.random.default_rng(seed)
    A, V, W = stencil_matrix(npts_domain, npts_codomain, pads, rng)
    out = StencilMatrix(W, V)
    return (A._data, out._data, *A._transpose_args.values())


def inner_arguments(npts, pads, seed):
    """The arguments of ``stencil_inner_<n>d`` as ``StencilVectorSpace.inner`` passes them."""
    rng = np.random.default_rng(seed)
    V = make_space(npts, pads)
    x = stencil_vector(V, rng)
    y = stencil_vector(V, rng)
    res = xp.zeros(1)
    return (x._data, y._data, *V._inner_consts, res)


def axpy_arguments(npts, pads, seed):
    """The arguments of ``stencil_axpy_<n>d`` as ``StencilVectorSpace.axpy`` passes them."""
    rng = np.random.default_rng(seed)
    V = make_space(npts, pads)
    x = stencil_vector(V, rng)
    y = stencil_vector(V, rng)
    return (float(rng.random()) - 0.5, x._data, y._data)
