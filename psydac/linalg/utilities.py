#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from math import sqrt

import numpy as np

from psydac.linalg.basic   import Vector
from psydac.linalg.stencil import StencilVector, StencilVectorSpace
from psydac.linalg.block   import BlockVector, BlockVectorSpace
from psydac.linalg.topetsc import get_npts_local

__all__ = (
    'array_to_psydac',
    'petsc_to_psydac',
    '_sym_ortho',
)

#==============================================================================
def array_to_psydac(x, V):
    """ 
    Convert a NumPy array to a Vector of the space V. This function is designed
    to be the inverse of the method .toarray() of the class Vector.

    Parameters
    ----------
    x : numpy.ndarray
        Array to be converted. It only contains the true data, the ghost regions must not be included.

    V : psydac.linalg.stencil.StencilVectorSpace or psydac.linalg.block.BlockVectorSpace
        Space of the final PSYDAC Vector.

    Returns
    -------
    u : psydac.linalg.stencil.StencilVector or psydac.linalg.block.BlockVector
        Element of space V, the coefficients of which (excluding ghost regions) are the entries of x. The ghost regions of u are up to date.

    Notes
    -----
    This function works in parallel but it is very costly and should be avoided
    if performance is a priority.
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
    Convert a PETSc.Vec object to a StencilVector or BlockVector. It assumes
    that PETSc was installed with the configuration for complex numbers. It
    uses the index conversion functions in psydac.linalg.topetsc.

    Parameters
    ----------
    x : PETSc.Vec
      PETSc vector

    Xh : psydac.linalg.stencil.StencilVectorSpace | psydac.linalg.block.BlockVectorSpace
      Space of the coefficients of the PSYDAC vector.

    out : psydac.linalg.stencil.StencilVector | psydac.linalg.block.BlockVector, optional
      The PSYDAC vector where to store the result.

    Returns
    -------
    u : psydac.linalg.stencil.StencilVector | psydac.linalg.block.BlockVector
        PSYDAC vector. In the case of a BlockVector, the blocks must be of type
        StencilVector. The general case is not yet implemented.
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

        dtype      = Xh._dtype
        localsize, globalsize = x.getSizes()
        assert globalsize == u.shape[0], 'Sizes of global vectors do not match'

        # Local PETSc data (this process only), ordered block-by-block (see vec_topetsc):
        values = x.getArray(readonly=True)

        # Local shape of this process, per block:
        npts_local_per_block = get_npts_local(Xh) # indexed [b][d]

        offset = 0
        for bb in range(Xh.n_blocks):
            npts_local = npts_local_per_block[bb]
            n = int(np.prod(npts_local))
            block_values = values[offset:offset + n]
            offset += n

            block_indices = _local_petsc_indices_to_data_indices(
                np.arange(n), npts_local, Xh.spaces[bb].pads, Xh.spaces[bb].shifts,
            )
            u[bb]._data[block_indices] = block_values if dtype is complex else block_values.real # PETSc always handles dtype specified in the installation configuration

    elif isinstance(Xh, StencilVectorSpace):

        if out is not None:
            assert isinstance(out, StencilVector)
            assert out.space is Xh
            u = out
        else:
            u = StencilVector(Xh)

        dtype      = Xh.dtype
        localsize, globalsize = x.getSizes()
        assert globalsize == u.shape[0], 'Sizes of global vectors do not match'

        # Local PETSc data (this process only):
        values = x.getArray(readonly=True)
        npts_local = get_npts_local(Xh)[0]

        indices = _local_petsc_indices_to_data_indices(np.arange(localsize), npts_local, Xh.pads, Xh.shifts)
        u._data[indices] = values if dtype is complex else values.real # PETSc always handles dtype specified in the installation configuration

    else:
        raise ValueError('Xh must be a StencilVectorSpace or a BlockVectorSpace')

    u.update_ghost_regions()

    return u

#==============================================================================
def _local_petsc_indices_to_data_indices(local_petsc_indices, npts_local, pads, shifts):
    """ Vectorized equivalent of calling `petsc_local_to_psydac` for every index in
    `local_petsc_indices` (all belonging to the same block), returning a tuple of
    integer arrays directly usable to index a StencilVector's `._data` array.
    """
    ndim = len(npts_local)

    if ndim == 1:
        i0 = local_petsc_indices + pads[0] * shifts[0]
        return (i0,)

    elif ndim == 2:
        i0 = local_petsc_indices // npts_local[1] + pads[0] * shifts[0]
        i1 = local_petsc_indices % npts_local[1] + pads[1] * shifts[1]
        return (i0, i1)

    elif ndim == 3:
        n1n2 = npts_local[1] * npts_local[2]
        i0 = local_petsc_indices // n1n2 + pads[0] * shifts[0]
        rem = local_petsc_indices % n1n2
        i1 = rem // npts_local[2] + pads[1] * shifts[1]
        i2 = rem % npts_local[2] + pads[2] * shifts[2]
        return (i0, i1, i2)

    else:
        raise NotImplementedError("Cannot handle more than 3 dimensions.")

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
