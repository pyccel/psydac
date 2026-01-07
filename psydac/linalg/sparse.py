#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from scipy.sparse import sparray, csr_array, bsr_array
from scipy.sparse import spmatrix, csr_matrix, bsr_matrix

from psydac.linalg.basic   import LinearOperator
from psydac.linalg.basic   import VectorSpace, Vector, LinearOperator
from psydac.linalg.stencil import StencilVector
from psydac.linalg.block   import BlockVector

__all__ = (
    'SparseMatrixLinearOperator',
)

class SparseMatrixLinearOperator(LinearOperator):
    """ 
    LinearOperator representation of a sparse matrix.

    Parameters
    ----------
    domain : VectorSpace
        The domain of the operator.

    codomain : VectorSpace
        The codomain of the operator.

    sparse_matrix : scipy.sparse.sparray | scipy.sparse.spmatrix
        The sparse SciPy matrix representing the operator. Recommended formats are
        CSR and BSR. Any other format will be converted to CSR (csr_array).
    """
        
    def __init__(self, domain, codomain, sparse_matrix):

        assert isinstance(domain, VectorSpace)
        assert isinstance(codomain, VectorSpace)
        assert isinstance(sparse_matrix, (sparray, spmatrix))

        if not isinstance(sparse_matrix,
                          (csr_array, csr_matrix,
                           bsr_array, bsr_matrix)):
            sparse_matrix = sparse_matrix.tocsr()

        if domain.parallel:
            raise NotImplementedError('Parallel SparseMatrixLinearOperator not supported yet.')

        self._domain = domain
        self._codomain = codomain
        self._matrix = sparse_matrix

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._matrix.dtype

    def toarray(self):
        return self._matrix.toarray()

    def tosparse(self):
        return self._matrix

    def transpose(self, conjugate=False):
        if conjugate:
            return SparseMatrixLinearOperator(self.codomain, self.domain, self._matrix.getH().tocsr())
        else:
            return SparseMatrixLinearOperator(self.codomain, self.domain, self._matrix.T.tocsr())

    def dot(self, v, out=None):
        assert isinstance(v, Vector)
        assert v.space is self.domain

        if out is not None:
            assert isinstance(out, Vector)
            assert out.space is self.codomain
            out *= 0
        else:
            out = self.codomain.zeros()

        self._dot_recursive(v, out=out)

        return out

    def _dot_recursive(self, v, out, ind_V=0, ind_W=0):
        V = v.space
        W = out.space

        if isinstance(v, StencilVector):
            index_global_W = tuple(slice(s, e+1) for s, e in zip(W.starts, W.ends))
            index_global_V = tuple(slice(s, e+1) for s, e in zip(V.starts, V.ends))

            dim_W = W.dimension
            dim_V = V.dimension

            out[index_global_W].flat += self._matrix[ind_W:ind_W+dim_W, ind_V:ind_V+dim_V] @ v[index_global_V].flat

        elif isinstance(v, BlockVector):

            offset_i = ind_W
            for (i, Wi) in enumerate(W.spaces):
                
                offset_j = ind_V
                for (j, Vj) in enumerate(V.spaces):

                    self._dot_recursive(v[j], out[i], ind_V=offset_j, ind_W=offset_i)

                    offset_j += Vj.dimension

                offset_i += Wi.dimension
