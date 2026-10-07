#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
Tests that distributed results are *absolutely* correct, not merely
self-consistent.

Comparing a distributed run against another distributed run of the same code
hides whole classes of bug: if both sides are wrong in the same way they still
agree. In particular, CuPy kernels run asynchronously while MPI knows nothing
about the CuPy stream, so a ghost exchange started before the producing kernels
finish sends stale data -- and every rank agrees on the wrong answer. The tests
below therefore pin distributed results to values computed from the global
field, and check that they do not depend on the decomposition.

Run with, e.g.::

    mpirun -np 4 python -m pytest test_mpi_device.py --with-mpi
"""
import numpy as np
import pytest
import cunumpy as xp

from feectools.ddm.mpi import mpi as MPI
from feectools.ddm.cart import DomainDecomposition, CartDecomposition
from feectools.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix

pytestmark = pytest.mark.mpi

NPTS = (16, 12)
PADS = (1, 2)


# ===============================================================================
def make_space(npts, pads, dtype=float, comm=None):
    ndim = len(npts)
    D = DomainDecomposition(list(npts), periods=[True] * ndim, comm=comm)
    gs, ge = [], []
    for axis in range(ndim):
        ee = D.global_element_ends[axis].copy()
        ee[-1] = npts[axis] - 1
        ge.append(ee)
        gs.append(xp.array([0] + (ee[:-1] + 1).tolist()))
    C = CartDecomposition(D, list(npts), gs, ge, pads=list(pads),
                          shifts=[1] * ndim)
    return StencilVectorSpace(C, dtype=dtype)


def global_field(npts):
    """A deterministic global field, independent of any decomposition."""
    i1 = np.arange(npts[0])[:, None]
    i2 = np.arange(npts[1])[None, :]
    return ((i1 + 2 * i2) % 17 + 1).astype(float)


def scatter(V, glob):
    """Put this rank's part of the global field into a new vector."""
    v = StencilVector(V)
    owned = tuple(slice(int(s), int(e) + 1)
                  for s, e in zip(V.starts, V.ends))
    local = tuple(slice(int(p), int(p) + sl.stop - sl.start)
                  for p, sl in zip(V.pads, owned))
    v._data[local] = xp.asarray(glob[owned])
    v.update_ghost_regions()
    return v


def laplacian(V, diag=4.5):
    A = StencilMatrix(V, V)
    A[:, :, 0, 0] = diag
    for axis in range(2):
        for shift in (-1, 1):
            key = [slice(None)] * 2 + [0, 0]
            key[2 + axis] = shift
            A[tuple(key)] = -1.0
    A.remove_spurious_entries()
    return A


def reference_apply(glob, diag=4.5):
    """The same periodic stencil applied to the global field."""
    out = diag * glob
    for axis in range(2):
        for shift in (-1, 1):
            out = out - np.roll(glob, -shift, axis=axis)
    return out


# ===============================================================================
def test_ghost_regions_have_the_right_values():
    """Every entry of the local array, ghosts included, must equal the global
    field at the corresponding (periodic) global index."""
    comm = MPI.COMM_WORLD
    V = make_space(NPTS, PADS, comm=comm)
    glob = global_field(NPTS)
    v = scatter(V, glob)

    data = v._data
    data = xp.to_numpy(data)
    s0, s1 = int(V.starts[0]), int(V.starts[1])
    p0, p1 = int(V.pads[0]), int(V.pads[1])

    expected = np.empty_like(data)
    for k0 in range(data.shape[0]):
        for k1 in range(data.shape[1]):
            expected[k0, k1] = glob[(s0 - p0 + k0) % NPTS[0],
                                    (s1 - p1 + k1) % NPTS[1]]

    assert np.allclose(data, expected, rtol=0.0, atol=1e-14)


# ===============================================================================
def test_matvec_matches_global_reference():
    """A @ v must equal the stencil applied to the global field, whatever the
    decomposition. This is the check that catches an unsynchronized ghost
    exchange: a self-consistency check between two distributed runs does not,
    because both would be wrong identically."""
    comm = MPI.COMM_WORLD
    V = make_space(NPTS, PADS, comm=comm)
    glob = global_field(NPTS)
    v = scatter(V, glob)
    A = laplacian(V)

    w = A.dot(v)
    ref = reference_apply(glob)

    # Compare through a global reduction, so the check is decomposition-free.
    got = float(w.inner(v))
    expected = float((ref * glob).sum())
    assert abs(got - expected) <= 1e-9 * abs(expected)

    # And entry by entry on the rows this rank owns
    data = w._data
    data = xp.to_numpy(data)
    p0, p1 = int(V.pads[0]), int(V.pads[1])
    for i1 in range(int(V.starts[0]), int(V.ends[0]) + 1):
        for i2 in range(int(V.starts[1]), int(V.ends[1]) + 1):
            k0 = p0 + i1 - int(V.starts[0])
            k1 = p1 + i2 - int(V.starts[1])
            assert abs(data[k0, k1] - ref[i1, i2]) <= 1e-12


# ===============================================================================
def test_matvec_after_device_kernels_without_explicit_sync():
    """The exchange must be safe when the vector was just written by kernels
    and the ghost update happens implicitly inside `A.dot` -- the ordering the
    PCG loop produces.

    A missing synchronization here is a data race, so the test has to force it
    rather than hope for it: a long chain of asynchronous work on the vector is
    queued and the exchange is triggered immediately afterwards, leaving the
    stream busy while MPI reads the buffer.
    """
    comm = MPI.COMM_WORLD
    npts, pads = (512, 512), (1, 2)
    V = make_space(npts, pads, comm=comm)
    glob = global_field(npts)
    A = laplacian(V)

    r = scatter(V, glob)

    # Queue work that writes r, mathematically the identity so the expected
    # result is unchanged. On a device the arrays are large and the chain long
    # enough that kernels are still queued when the exchange starts -- which is
    # what makes the race reproducible rather than occasional. There is no race
    # on the host, so one pass is enough there.
    passes = 200 if xp.is_gpu(r._data) else 1
    for _ in range(passes):
        r._data *= 2.0
        r._data *= 0.5

    r.ghost_regions_in_sync = False
    w = A.dot(r)   # triggers the implicit ghost update

    ref = reference_apply(glob)
    got = float(w.inner(r))
    expected = float((ref * glob).sum())
    assert abs(got - expected) <= 1e-9 * abs(expected)


# ===============================================================================
def test_ghost_exchange_synchronizes_before_mpi(monkeypatch):
    """
    The exchangers must call `synchronize_for_mpi` before giving a buffer to
    MPI.

    This is checked structurally rather than by observing corrupted data,
    because the underlying race is not deterministic: whether MPI actually
    reads a half-written buffer depends on which internal protocol it picks for
    the message, and some of those happen to synchronize with the CuPy stream
    by accident. Relying on that accident is exactly the bug, so the contract
    is what gets tested.
    """
    import feectools.ddm.blocking_data_exchanger as blocking
    import feectools.ddm.nonblocking_data_exchanger as nonblocking

    calls = []
    for module in (blocking, nonblocking):
        monkeypatch.setattr(module, 'synchronize_for_mpi',
                            lambda *args: calls.append(args))

    V = make_space(NPTS, PADS, comm=MPI.COMM_WORLD)
    v = StencilVector(V)
    v.ghost_regions_in_sync = False
    v.update_ghost_regions()

    assert calls, 'ghost exchange handed a buffer to MPI without synchronizing'
    assert any(v._data is arg for args in calls for arg in args), \
        'the synchronized buffer was not the one being exchanged'


# ===============================================================================
def test_axpy_then_matvec_is_correct():
    """`mul_iadd` writes on the device; the following exchange must see it."""
    comm = MPI.COMM_WORLD
    V = make_space(NPTS, PADS, comm=comm)
    glob = global_field(NPTS)
    A = laplacian(V)

    x = scatter(V, glob)
    y = scatter(V, glob)
    x.mul_iadd(2.0, y)              # x = 3 * glob
    w = A.dot(x)

    ref = reference_apply(3.0 * glob)
    got = float(w.inner(x))
    expected = float((ref * (3.0 * glob)).sum())
    assert abs(got - expected) <= 1e-9 * abs(expected)


# ===============================================================================
def test_inner_matches_global_reference():
    """Reductions must equal the value computed from the global field."""
    comm = MPI.COMM_WORLD
    V = make_space(NPTS, PADS, comm=comm)
    glob = global_field(NPTS)
    other = np.flipud(glob).copy()

    x = scatter(V, glob)
    y = scatter(V, other)

    assert abs(float(x.inner(y)) - float((glob * other).sum())) <= 1e-9
    assert abs(float(x.inner(x)) - float((glob * glob).sum())) <= 1e-9


# ===============================================================================
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, '-v', '--with-mpi']))
