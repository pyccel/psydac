import numpy as np

from psydac.linalg.block import BlockVectorSpace, BlockVector, BlockLinearOperator
from psydac.linalg.stencil import StencilVectorSpace, StencilVector
from scipy.sparse import coo_matrix, bmat

from mpi4py import MPI

__all__ = ('flatten_vec', 'vec_topetsc', 'mat_topetsc')

def flatten_vec( vec ):
    """ Return the flattened 1D array values and indices owned by the process of the given vector.

    Parameters
    ----------
    vec : <Vector>
      Psydac Vector to be flattened

    Returns
    -------
    indices: numpy.ndarray
        The global indices the data array collapsed into one dimension.

    array : numpy.ndarray
        A copy of the data array collapsed into one dimension.

    """

    if isinstance(vec, StencilVector):
        npts = vec.space.npts
        idx = tuple( slice(m*p,-m*p) for m,p in zip(vec.pads, vec.space.shifts) )
        shape = vec._data[idx].shape
        starts = vec.space.starts
        indices = np.array([np.ravel_multi_index( [s+x for s,x in zip(starts, xx)], dims=npts,  order='C' ) for xx in np.ndindex(*shape)] )
        data = vec._data[idx].flatten()
        vec = coo_matrix(
                    (data,(indices,indices)),
                    shape = [vec.space.dimension,vec.space.dimension],
                    dtype = vec.space.dtype)

    elif isinstance(vec, BlockVector):
        vecs = [flatten_vec(b) for b in vec.blocks]
        vecs = [coo_matrix((v[1],(v[0],v[0])),
                shape=[vs.space.dimension,vs.space.dimension],
                dtype=vs.space.dtype) for v,vs in zip(vecs, vec.blocks)]

        blocks = [[None]*len(vecs) for v in vecs]
        for i,v in enumerate(vecs):
            blocks[i][i] = v

        vec = bmat(blocks,format='coo')

    else:
        raise TypeError("Expected StencilVector or BlockVector, found instead {}".format(type(vec)))

    array   = vec.data
    indices = vec.row
    return indices, array

def vec_topetsc( vec ):
    """ Convert Psydac Vector to PETSc data structure.

    Parameters
    ----------
    vec : <Vector>
      Distributed Psydac vector.

    Returns
    -------
    gvec : PETSc.Vec
        Distributed PETSc vector.

    """

    from petsc4py import PETSc
    comm = vec.space.spaces[0].cart.global_comm if isinstance(vec, BlockVector) else vec.space.cart.global_comm
    globalsize = vec.space.dimension
    indices, data = flatten_vec(vec)
    gvec  = PETSc.Vec().create(comm=comm)
    gvec.setSizes(globalsize)
    gvec.setFromOptions()
    gvec.setValues(indices, data)
    gvec.assemble()
    return gvec

def mat_topetsc( mat ):
    """ Convert Psydac Matrix to PETSc data structure.

    Parameters
    ----------
    vec : <Matrix>
      Distributed Psydac Matrix.

    Returns
    -------
    gmat : PETSc.Mat
        Distributed PETSc matrix.
    """

    from petsc4py import PETSc

    comm = mat.domain.spaces[0].cart.global_comm if isinstance(mat, BlockLinearOperator) else mat.domain.cart.global_comm
    mat_coo = mat.tosparse()
    ncols = mat.domain.dimension
    nrows = mat.codomain.dimension
    gmat  = PETSc.Mat().create(comm=comm)
    gmat.setSizes((nrows, ncols))
    gmat.setType("mpiaij")
    gmat.setFromOptions()

    rows, cols, data = mat_coo.row, mat_coo.col, mat_coo.data
    NNZ = comm.allreduce(data.size, op=MPI.SUM)
    gmat.setPreallocationNNZ(NNZ)
    for i in range(len(rows)):
        gmat.setValues(rows[i], cols[i], data[i])

    gmat.assemble()
    return gmat
