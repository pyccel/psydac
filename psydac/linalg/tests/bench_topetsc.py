#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#

import sys
import time

import numpy as np
from mpi4py import MPI

from psydac.ddm.cart import DomainDecomposition, CartDecomposition
from psydac.linalg.stencil import StencilVectorSpace, StencilVector

from psydac.linalg import topetsc as topetsc_new
from psydac.linalg import topetsc_old


def compute_global_starts_ends(domain_decomposition, npts):
    ndims = len(npts)
    global_starts = [None] * ndims
    global_ends = [None] * ndims
    for axis in range(ndims):
        ee = domain_decomposition.global_element_ends[axis]
        global_ends[axis] = ee.copy()
        global_ends[axis][-1] = npts[axis] - 1
        global_starts[axis] = np.array([0] + (global_ends[axis][:-1] + 1).tolist())
    return global_starts, global_ends


def make_vector(n1, n2, p1=2, p2=2, comm=None):
    D = DomainDecomposition([n1, n2], periods=[False, False], comm=comm)
    npts = [n1, n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[1, 1])
    V = StencilVectorSpace(C, dtype=float)
    x = StencilVector(V)
    rng = np.random.default_rng(0)
    x._data[:] = rng.random(x._data.shape)
    return x


def timeit(fn, n_repeat=5):
    # Warm-up call: lazily initializes PETSc and populates caches.
    fn()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        fn()
    return (time.perf_counter() - t0) / n_repeat


def check_same_result(x, comm):
    v_old = topetsc_old.vec_topetsc(x)
    v_new = topetsc_new.vec_topetsc(x)

    rstart_old, rend_old = v_old.getOwnershipRange()
    rstart_new, rend_new = v_new.getOwnershipRange()
    assert (rstart_old, rend_old) == (rstart_new, rend_new), \
        f"Ownership ranges differ: old={(rstart_old, rend_old)}, new={(rstart_new, rend_new)}"

    a_old = v_old.getArray(readonly=True)
    a_new = v_new.getArray(readonly=True)
    local_ok = np.allclose(a_old, a_new)

    all_ok = comm.allreduce(local_ok, op=MPI.LAND) if comm else local_ok
    if not local_ok:
        diff = np.abs(a_old - a_new)
        raise AssertionError(
            f"[rank {comm.Get_rank() if comm else 0}] vec_topetsc mismatch: "
            f"max |old - new| = {diff.max():.3e} at local index {int(diff.argmax())}"
        )
    assert all_ok, "vec_topetsc mismatch detected on another rank"

    v_old.destroy()
    v_new.destroy()


def main():
    comm = MPI.COMM_WORLD
    sizes = [int(a) for a in sys.argv[1:]] or [20, 40, 80, 160, 320]

    if comm.Get_rank() == 0:
        print(f"{'n1 x n2':>12} {'local size':>12} {'old [s]':>12} {'new [s]':>12} {'speedup':>10}   check")

    for n in sizes:
        x = make_vector(n, n, comm=comm)
        localsize = int(np.prod([e - s + 1 for s, e in zip(x.space.starts, x.space.ends)]))

        # Correctness check: must pass before the timing numbers are trusted.
        check_same_result(x, comm)

        t_old = timeit(lambda: topetsc_old.vec_topetsc(x))
        t_new = timeit(lambda: topetsc_new.vec_topetsc(x))

        comm.Barrier()
        if comm.Get_rank() == 0:
            print(f"{n:>5}x{n:<6} {localsize:>12} {t_old:>12.5f} {t_new:>12.5f} {t_old / t_new:>9.1f}x   OK")


if __name__ == '__main__':
    main()
