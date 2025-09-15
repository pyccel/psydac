# coding: utf-8

import numpy as np
from math import sqrt

from psydac.linalg.basic   import VectorSpace, Vector, MatrixFreeLinearOperator
from psydac.linalg.stencil import StencilVectorSpace, StencilVector
from psydac.linalg.block   import BlockVector, BlockVectorSpace
from psydac.linalg.topetsc import petsc_local_to_psydac, get_npts_per_block

from scipy.sparse import csr_matrix

__all__ = (
    'array_to_psydac',
    'petsc_to_psydac',
    '_sym_ortho',
    'SparseMatrixLinearOperator',
)

#==============================================================================
def array_to_psydac(x, V):
    """ 
    Convert a NumPy array to a Vector of the space V. This function is designed to be the inverse of the method .toarray() of the class Vector.
    Note: This function works in parallel but it is very costly and should be avoided if performance is a priority.

    Parameters
    ----------
    x : numpy.ndarray
        Array to be converted. It only contains the true data, the ghost regions must not be included.

    V : psydac.linalg.stencil.StencilVectorSpace or psydac.linalg.block.BlockVectorSpace
        Space of the final Psydac Vector.

    Returns
    -------
    u : psydac.linalg.stencil.StencilVector or psydac.linalg.block.BlockVector
        Element of space V, the coefficients of which (excluding ghost regions) are the entries of x. The ghost regions of u are up to date.

    """

    assert x.ndim == 1, 'Array must be 1D.'
    if x.dtype==complex:
        assert V.dtype==complex, 'Complex array cannot be converted to a real StencilVector'
    assert x.size == V.dimension, 'Array must have the same global size as the space.'

    u = V.zeros()
    _array_to_psydac_recursive(x, u)
    u.update_ghost_regions()

    return u


def _array_to_psydac_recursive(x, u):
    """
    Recursive function filling in the coefficients of each block of u.
    """
    assert isinstance(u, Vector)
    V = u.space

    assert x.ndim == 1, 'Array must be 1D.'
    if x.dtype==complex:
        assert V.dtype==complex, 'Complex array cannot be converted to a real StencilVector'
    assert x.size == V.dimension, 'Array must have the same global size as the space.'    

    if isinstance(V, BlockVectorSpace):
        for i, V_i in enumerate(V.spaces):
            x_i = x[:V_i.dimension]
            x   = x[V_i.dimension:]
            u_i = u[i]
            _array_to_psydac_recursive(x_i, u_i)

    elif isinstance(V, StencilVectorSpace):
        index_global = tuple(slice(s, e+1) for s, e in zip(V.starts, V.ends))
        u[index_global] = x.reshape(V.npts)[index_global]

    else:
        raise NotImplementedError(f'Can only handle StencilVector or BlockVector spaces, got {type(V)} instead')
    
#==============================================================================
def petsc_to_psydac(x, Xh, out=None):
    """
    Convert a PETSc.Vec object to a StencilVector or BlockVector. It assumes that PETSc was installed with the configuration for complex numbers.
    Uses the index conversion functions in psydac.linalg.topetsc.py.

    Parameters
    ----------
    x : PETSc.Vec
      PETSc vector

    Xh : psydac.linalg.stencil.StencilVectorSpace | psydac.linalg.block.BlockVectorSpace
      Space of the coefficients of the Psydac vector.

    out : psydac.linalg.stencil.StencilVector | psydac.linalg.block.BlockVector, optional
      The Psydac vector where to store the result.

    Returns
    -------
    u : psydac.linalg.stencil.StencilVector | psydac.linalg.block.BlockVector
        Psydac vector. In the case of a BlockVector, the blocks must be StencilVector. The general case is not yet implemented.
    """
    
    if isinstance(Xh, BlockVectorSpace):
        if any([isinstance(Xh.spaces[b], BlockVectorSpace) for b in range(len(Xh.spaces))]):
            raise NotImplementedError('Block of blocks not implemented.')
        
        if out is not None:
            assert isinstance(out, BlockVector)
            assert out.space is Xh
            u = out
        else:
            u = BlockVector(Xh)

        comm       = x.comm
        dtype      = Xh._dtype
        localsize, globalsize = x.getSizes()
        assert globalsize == u.shape[0], 'Sizes of global vectors do not match'

        # Find shift for process k:
        # ..get number of points for each block, each process and each dimension:
        npts_local_per_block_per_process = np.array(get_npts_per_block(Xh)) #indexed [b,k,d] for block b and process k and dimension d
        # ..get local sizes for each block and each process:
        local_sizes_per_block_per_process = np.prod(npts_local_per_block_per_process, axis=-1) #indexed [b,k] for block b and process k
        # ..sum the sizes over all the blocks and the previous processes:
        index_shift = 0 + np.sum(local_sizes_per_block_per_process[:,:comm.Get_rank()], dtype=int) #global variable

        for local_petsc_index in range(localsize):
            block_index, psydac_index = petsc_local_to_psydac(Xh, local_petsc_index)
            # Get value of local PETSc vector passing the global PETSc index
            value = x.getValue(local_petsc_index + index_shift) 
            if value != 0:
                u[block_index[0]]._data[psydac_index] = value if dtype is complex else value.real # PETSc always handles dtype specified in the installation configuration
        
    elif isinstance(Xh, StencilVectorSpace):

        if out is not None:
            assert isinstance(out, StencilVector)
            assert out.space is Xh
            u = out
        else:
            u = StencilVector(Xh)

        comm       = x.comm
        dtype      = Xh.dtype
        localsize, globalsize = x.getSizes()
        assert globalsize == u.shape[0], 'Sizes of global vectors do not match'

        # Find shift for process k:
        # ..get number of points for each process and each dimension:
        npts_local_per_block_per_process = np.array(get_npts_per_block(Xh))[0] #indexed [k,d] for process k and dimension d
        # ..get local sizes for each process:
        local_sizes_per_block_per_process = np.prod(npts_local_per_block_per_process, axis=-1) #indexed [k] for process k
        # ..sum the sizes over all the previous processes:
        index_shift = 0 + np.sum(local_sizes_per_block_per_process[:comm.Get_rank()], dtype=int) #global variable

        for local_petsc_index in range(localsize):
            block_index, psydac_index = petsc_local_to_psydac(Xh, local_petsc_index) 
            # Get value of local PETSc vector passing the global PETSc index
            value = x.getValue(local_petsc_index + index_shift)
            if value != 0:
                u._data[psydac_index] = value if dtype is complex else value.real # PETSc always handles dtype specified in the installation configuration            

    else:
        raise ValueError('Xh must be a StencilVectorSpace or a BlockVectorSpace')

    u.update_ghost_regions()

    return u

#==============================================================================
def _sym_ortho(a, b):
    """
    Stable implementation of Givens rotation.
    This function was taken from the scipy repository
    https://github.com/scipy/scipy/blob/master/scipy/sparse/linalg/isolve/lsqr.py

    Notes
    -----
    The routine 'SymOrtho' was added for numerical stability. This is
    recommended by S.-C. Choi in [1]_.  It removes the unpleasant potential of
    ``1/eps`` in some important places (see, for example text following
    "Compute the next plane rotation Qk" in minres.py).

    References
    ----------
    .. [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations
           and Least-Squares Problems", Dissertation,
           http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf
    """
    if b == 0:
        return np.sign(a), 0, abs(a)
    elif a == 0:
        return 0, np.sign(b), abs(b)
    elif abs(b) > abs(a):
        tau = a / b
        s = np.sign(b) / sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = np.sign(a) / sqrt(1+tau*tau)
        s = c * tau
        r = a / c
    return c, s, r

#==============================================================================
class SparseMatrixLinearOperator(MatrixFreeLinearOperator):
    """ 
    Wrap a sparse matrix into a MatrixFreeLinearOperator
    """
        
    def __init__(self, domain, codomain, sparse_matrix):

        assert isinstance(domain, VectorSpace)
        assert isinstance(codomain, VectorSpace)
        assert isinstance(sparse_matrix, csr_matrix)

        self._domain = domain
        self._codomain = codomain
        self._matrix = sparse_matrix

    def tosparse(self):
        return self._matrix

    def toarray(self):
        return self._matrix.toarray()

    def transpose(self, conjugate=False):
        return SparseMatrixLinearOperator(self.codomain, self.domain, self._matrix.T)

    def dot(self, v, out=None):

        assert v.space is self.domain

        if out is not None:

            assert out.space is self.codomain

            out *= 0.0
        else:
            out = self.codomain.zeros()

        if not v.ghost_regions_in_sync:
            v.update_ghost_regions()

        if self.domain.parallel:
            raise NotImplementedError('Parallel case not implemented yet.')
        else:
            self._dot_recursive(v, out)

        out.ghost_regions_in_sync = False

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
