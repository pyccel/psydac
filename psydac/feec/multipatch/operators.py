import os
import numpy as np

from scipy.sparse import save_npz, load_npz
from scipy.sparse import block_diag
from scipy.sparse.linalg import inv

from sympde.topology import element_of, elements_of
from sympde.topology.space import ScalarFunction
from sympde.calculus import dot
from sympde.expr.expr import BilinearForm
from sympde.expr.expr import integral

from psydac.api.settings import PSYDAC_BACKENDS

from psydac.feec.derivatives import Gradient_2D, ScalarCurl_2D
from psydac.feec.multipatch.fem_linear_operators import FemLinearOperator

# ===============================================================================
class HodgeOperator(FemLinearOperator):
    """
    Change of basis operator: dual basis -> primal basis

        self._matrix: matrix of the primal Hodge = this is the mass matrix !
        self.dual_Hodge_matrix: this is the INVERSE mass matrix

    Parameters
    ----------
    Vh: <FemSpace>
     The discrete space

    domain_h: <Geometry>
     The discrete domain of the projector

    metric : <str>
     the metric of the de Rham complex

    backend_language: <str>
     The backend used to accelerate the code

    load_dir: <str>
     storage files for the primal and dual Hodge sparse matrice

    load_space_index: <str>
      the space index in the derham sequence

    Notes
    -----
     Either we use a storage, or these matrices are only computed on demand
     # todo: we compute the sparse matrix when to_sparse_matrix is called -- but never the stencil matrix (should be fixed...)
     We only support the identity metric, this implies that the dual Hodge is the inverse of the primal one.
     # todo: allow for non-identity metrics
    """

    def __init__(
            self,
            Vh,
            domain_h,
            metric='identity',
            backend_language='python',
            load_dir=None,
            load_space_index=''):

        FemLinearOperator.__init__(self, fem_domain=Vh)
        self._domain_h = domain_h
        self._backend_language = backend_language
        self._dual_Hodge_sparse_matrix = None

        assert metric == 'identity'
        self._metric = metric

        if load_dir and isinstance(load_dir, str):
            if not os.path.exists(load_dir):
                os.makedirs(load_dir)
            assert str(load_space_index) in ['0', '1', '2', '3']
            primal_Hodge_storage_fn = load_dir + \
                '/H{}_m.npz'.format(load_space_index)
            dual_Hodge_storage_fn = load_dir + \
                '/dH{}_m.npz'.format(load_space_index)

            primal_Hodge_is_stored = os.path.exists(primal_Hodge_storage_fn)
            dual_Hodge_is_stored = os.path.exists(dual_Hodge_storage_fn)
            if dual_Hodge_is_stored:
                assert primal_Hodge_is_stored
                print(
                    " ...            loading dual Hodge sparse matrix from " +
                    dual_Hodge_storage_fn)
                self._dual_Hodge_sparse_matrix = load_npz(
                    dual_Hodge_storage_fn)
                print(
                    "[HodgeOperator] loading primal Hodge sparse matrix from " +
                    primal_Hodge_storage_fn)
                self._sparse_matrix = load_npz(primal_Hodge_storage_fn)
            else:
                assert not primal_Hodge_is_stored
                print(
                    "[HodgeOperator] assembling both sparse matrices for storage...")
                self.assemble_primal_Hodge_matrix()
                print(
                    "[HodgeOperator] storing primal Hodge sparse matrix in " +
                    primal_Hodge_storage_fn)
                save_npz(primal_Hodge_storage_fn, self._sparse_matrix)
                self.assemble_dual_Hodge_matrix()
                print(
                    "[HodgeOperator] storing dual Hodge sparse matrix in " +
                    dual_Hodge_storage_fn)
                save_npz(dual_Hodge_storage_fn, self._dual_Hodge_sparse_matrix)
        else:
            # matrices are not stored, we will probably compute them later
            pass

    def to_sparse_matrix(self):
        """
        the Hodge matrix is the patch-wise multi-patch mass matrix
        it is not stored by default but assembled on demand
        """

        if (self._sparse_matrix is not None) or (self._matrix is not None):
            return FemLinearOperator.to_sparse_matrix(self)

        self.assemble_primal_Hodge_matrix()

        return self._sparse_matrix

    def assemble_primal_Hodge_matrix(self):
        """
        the Hodge matrix is the patch-wise multi-patch mass matrix
        it is not stored by default but assembled on demand
        """
        from psydac.api.discretization import discretize

        if self._matrix is None:
            Vh = self.fem_domain
            assert Vh == self.fem_codomain

            V = Vh.symbolic_space
            domain = V.domain
            # domain_h = V0h.domain:  would be nice...
            u, v = elements_of(V, names='u, v')

            if isinstance(u, ScalarFunction):
                expr = u * v
            else:
                expr = dot(u, v)

            a = BilinearForm((u, v), integral(domain, expr))
            ah = discretize(a, self._domain_h, [Vh, Vh], backend=PSYDAC_BACKENDS[self._backend_language])

            self._matrix = ah.assemble()  # Mass matrix in stencil format
            self._sparse_matrix = self._matrix.tosparse()

    def get_dual_Hodge_sparse_matrix(self):
        if self._dual_Hodge_sparse_matrix is None:
            self.assemble_dual_Hodge_matrix()

        return self._dual_Hodge_sparse_matrix

    def assemble_dual_Hodge_matrix(self):
        """
        the dual Hodge matrix is the patch-wise inverse of the multi-patch mass matrix
        it is not stored by default but computed on demand, by local (patch-wise) inversion of the mass matrix
        """

        if self._dual_Hodge_sparse_matrix is None:
            if not self._matrix:
                self.assemble_primal_Hodge_matrix()

            M = self._matrix  # mass matrix of the (primal) basis
            nrows = M.n_block_rows
            ncols = M.n_block_cols

            inv_M_blocks = []
            for i in range(nrows):
                Mii = M[i, i].tosparse()
                inv_Mii = inv(Mii.tocsc())
                inv_Mii.eliminate_zeros()
                inv_M_blocks.append(inv_Mii)

            inv_M = block_diag(inv_M_blocks)
            self._dual_Hodge_sparse_matrix = inv_M

