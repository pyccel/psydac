# coding: utf-8
"""
Bookkeeping of the memory occupied by the stencil matrices that are currently alive.

Every :class:`~feectools.linalg.stencil.StencilMatrix` that allocates its data array registers
itself (weakly) in the module-level :data:`stencil_matrix_memory` tracker, so that an application
can report how much memory its matrices actually take, without having to walk its own data
structures. Matrices created with ``dry_run=True`` do not allocate anything and are not registered.
"""

import weakref

__all__ = ('MatrixMemoryTracker', 'stencil_matrix_memory')


class MatrixMemoryTracker:
    """Weak registry of allocated matrices; matrices that are garbage collected drop out of it."""

    def __init__(self):
        self._matrices = weakref.WeakSet()

    def register(self, matrix):
        """Add a matrix to the registry (does not keep it alive)."""
        self._matrices.add(matrix)

    def clear(self):
        """Forget all registered matrices."""
        self._matrices.clear()

    @property
    def n_matrices(self):
        """Number of currently alive registered matrices."""
        return len(self._matrices)

    @property
    def nbytes(self):
        """Local (per-MPI-rank) memory footprint, in bytes, of all currently alive registered matrices."""
        return int(sum(matrix.nbytes for matrix in self._matrices))


stencil_matrix_memory = MatrixMemoryTracker()
