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
#from psydac.api.discretization import discretize

# from psydac.feec.derivatives import Gradient_2D, ScalarCurl_2D
#from psydac.feec.multipatch.fem_linear_operators import FemLinearOperator
from psydac.linalg.basic import LinearOperator
from psydac.linalg.utilities import SparseMatrixLinearOperator

# ===============================================================================
class HodgeOperator:
    """
    Change of basis operator: dual basis -> primal basis

        self._linop: matrix (LinearOperator) of the primal Hodge = this is the mass matrix !
        self.dual_linop: this is the INVERSE mass matrix (LinearOperator)

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

    def __init__(self, Vh, domain_h, metric='identity', backend_language='python', load_dir=None, load_space_index=''):

        self._fem_domain = Vh
        self._fem_codomain = Vh

        # FemLinearOperators
        self._primal_Hodge = None
        self._dual_Hodge = None

        # LinearOperators
        self._linop = None
        self._dual_linop = None

        # Sparse matrices
        self._sparse_matrix = None
        self._dual_sparse_matrix = None

        self._domain_h = domain_h
        self._backend_language = backend_language

        assert metric == 'identity'
        self._metric = metric

        if load_dir and isinstance(load_dir, str):
            if not os.path.exists(load_dir):
                os.makedirs(load_dir)
            assert str(load_space_index) in ['0', '1', '2', '3']
            primal_Hodge_storage_fn = load_dir + '/H{}_m.npz'.format(load_space_index)
            dual_Hodge_storage_fn   = load_dir + '/dH{}_m.npz'.format(load_space_index)

            primal_Hodge_is_stored = os.path.exists(primal_Hodge_storage_fn)
            dual_Hodge_is_stored = os.path.exists(dual_Hodge_storage_fn)
            if dual_Hodge_is_stored:
                assert primal_Hodge_is_stored
                print(" ...            loading dual Hodge sparse matrix from " + dual_Hodge_storage_fn)
                self._dual_sparse_matrix = load_npz(dual_Hodge_storage_fn)
                print("[HodgeOperator] loading primal Hodge sparse matrix from " + primal_Hodge_storage_fn)
                self._sparse_matrix = load_npz(primal_Hodge_storage_fn)
            else:
                assert not primal_Hodge_is_stored
                print("[HodgeOperator] assembling both sparse matrices for storage...")
                self.assemble_matrix()
                print("[HodgeOperator] storing primal Hodge sparse matrix in " + primal_Hodge_storage_fn)
                save_npz(primal_Hodge_storage_fn, self._sparse_matrix)
                self.assemble_dual_sparse_matrix()
                print("[HodgeOperator] storing dual Hodge sparse matrix in " + dual_Hodge_storage_fn)
                save_npz(dual_Hodge_storage_fn, self._dual_sparse_matrix)
        else:
            # matrices are not stored, we will probably compute them later
            pass

    def assemble_matrix(self):
        """
        the Hodge matrix is the patch-wise multi-patch mass matrix
        it is not stored by default but assembled on demand
        """
        from psydac.api.discretization import discretize
        from psydac.fem.basic          import FemLinearOperator

        if self._linop is None:
            Vh = self._fem_domain
            assert Vh == self._fem_codomain

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

            self._linop = ah.assemble()  # Mass matrix in stencil format
            self._sparse_matrix = self._linop.tosparse()

            self._primal_Hodge = FemLinearOperator(self._fem_domain, self._fem_codomain, linop=self._linop, sparse_matrix=self._sparse_matrix)

    # which of the two assemblys for the sparse dual matrix is better? For now use the exact one.
    def assemble_dual_sparse_matrix(self):
        """
        the dual Hodge sparse matrix is the patch-wise inverse of the multi-patch mass matrix
        it is not stored by default but computed on demand, by local (patch-wise) exact inversion of the mass matrix
        """
        from psydac.fem.basic import FemLinearOperator
        
        if self._dual_sparse_matrix is None:
            if not self._linop:
                self.assemble_matrix()

            M = self._linop  # mass matrix of the (primal) basis

            if self._fem_domain.is_multipatch:
                nrows = M.n_block_rows
                ncols = M.n_block_cols

                inv_M_blocks = []
                for i in range(nrows):
                    Mii = M[i, i].tosparse()
                    inv_Mii = inv(Mii.tocsc())
                    inv_Mii.eliminate_zeros()
                    inv_M_blocks.append(inv_Mii)

                inv_M = block_diag(inv_M_blocks)

                self._dual_sparse_matrix = inv_M
                self._dual_linop = SparseMatrixLinearOperator(M.codomain, M.domain, inv_M)
                self._dual_Hodge = FemLinearOperator(self._fem_codomain, self._fem_domain, linop=self._dual_linop, sparse_matrix=self._dual_sparse_matrix)

            else:
                M_m = M.tosparse()
                inv_M = inv(M_m.tocsc())
                inv_M.eliminate_zeros()   

                self._dual_sparse_matrix = inv_M
                self._dual_linop = SparseMatrixLinearOperator(M.codomain, M.domain, inv_M)
                self._dual_Hodge = FemLinearOperator(self._fem_codomain, self._fem_domain, linop=self._dual_linop, sparse_matrix=self._dual_sparse_matrix)
 

    def assemble_dual_matrix(self, solver ='gmres', **kwargs):
        """
        the dual Hodge matrix is the patch-wise inverse of the multi-patch mass matrix
        it is not stored by default but computed on demand, by approximate local (patch-wise) inversion of the mass matrix
        """
        from psydac.linalg.solvers import inverse
        from psydac.linalg.block   import BlockLinearOperator
        from psydac.fem.basic      import FemLinearOperator
        
        if self._dual_linop is None:
            if not self._linop:
                self.assemble_matrix()

            M = self._linop  # mass matrix of the (primal) basis

            if self._fem_domain.is_multipatch:

                nrows = M.n_block_rows
                ncols = M.n_block_cols

                inv_M_blocks = [list(b) for b in M.blocks]
                for i in range(nrows):
                    Mii = M[i, i]
                    inv_Mii = inverse(M[i,i], solver=solver, **kwargs)
                    inv_M_blocks[i][i] = inv_Mii

                self._dual_linop = BlockLinearOperator(M.codomain, M.domain, blocks=inv_M_blocks)
                self._dual_sparse_matrix = None
                self._dual_Hodge = FemLinearOperator(self._fem_codomain, self._fem_domain, linop=self._dual_linop, sparse_matrix=self._dual_sparse_matrix)
            
            else:
                inv_M = inverse(M, solver=solver, **kwargs)
                self._dual_sparse_matrix = None
                self._dual_Hodge = FemLinearOperator(self._fem_codomain, self._fem_domain, linop=self._dual_linop, sparse_matrix=self._dual_sparse_matrix)
            
    @property
    def linop(self):
        if self._linop is None:
            self.assemble_matrix()

        return self._linop
        
    @property
    def tosparse(self):
        if self._sparse_matrix is None:
            self.assemble_matrix()

        return self._sparse_matrix

    @property
    def dual_linop(self):
        if self._dual_linop is None:
            self.assemble_dual_sparse_matrix()

        return self._dual_linop
        
    @property
    def dual_tosparse(self):
        if self._dual_sparse_matrix is None:
            self.assemble_dual_sparse_matrix()

        return self._dual_sparse_matrix

    @property
    def Hodge(self):
        if self._linop is None:
            self.assemble_matrix()

        return self._primal_Hodge

    @property
    def dual_Hodge(self):
        if self._dual_linop is None:
            self.assemble_dual_sparse_matrix()

        return self._dual_Hodge