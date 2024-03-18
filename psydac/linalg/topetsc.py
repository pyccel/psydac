import numpy as np

from psydac.linalg.block import BlockVectorSpace, BlockVector
from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from scipy.sparse import coo_matrix, bmat

from mpi4py import MPI

__all__ = ('flatten_vec', 'vec_topetsc', 'mat_topetsc')

def flatten_vec( vec ):
    """ Return the flattened 1D array values and indices owned by the process of the given vector.

    Parameters
    ----------
    vec : psydac.linalg.stencil.StencilVector | psydac.linalg.block.BlockVector
      Psydac vector to be flattened

    Returns
    -------
    indices: numpy.ndarray
        The global indices of the data array collapsed into one dimension.

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
    """ Convert vector from Psydac format to a PETSc.Vec object.

    Parameters
    ----------
    vec : psydac.linalg.stencil.StencilVector | psydac.linalg.block.BlockVector
      Psydac StencilVector or BlockVector.

    Returns
    -------
    gvec : PETSc.Vec
        PETSc vector
    """
    from petsc4py import PETSc

    if isinstance(vec, StencilVector):
        comm = vec.space.cart.global_comm
    elif isinstance(vec.space.spaces[0], StencilVectorSpace):
        comm = vec.space.spaces[0].cart.global_comm
    elif isinstance(vec.space.spaces[0], BlockVectorSpace):
        comm = vec.space.spaces[0][0].cart.global_comm

    globalsize = vec.space.dimension
    indices, data = flatten_vec(vec)
    gvec  = PETSc.Vec().create(comm=comm)
    # Set global size
    gvec.setSizes(globalsize)
    gvec.setFromOptions()
    # Set values of the vector. They are stored in a cache, so the assembly is necessary to use the vector.
    gvec.setValues(indices, data)

    # Assemble vector
    gvec.assemble() # Here PETSc exchanges global communication. The block corresponding to a certain process is not necessarily the same block in the Psydac StencilVector.
    return gvec

def mat_topetsc( mat ):
    """ Convert operator from Psydac format to a PETSc.Mat object.

    Parameters
    ----------
    mat : psydac.linalg.stencil.StencilMatrix | psydac.linalg.basic.LinearOperator | psydac.linalg.block.BlockLinearOperator
      Psydac operator

    Returns
    -------
    gmat : PETSc.Mat
        PETSc Matrix
    """

    from petsc4py import PETSc

    if isinstance(mat, StencilMatrix):
        comm = mat.domain.cart.global_comm
    elif isinstance(mat.domain.spaces[0], StencilVectorSpace):
        comm = mat.domain.spaces[0].cart.global_comm
    elif isinstance(mat.domain.spaces[0], BlockVectorSpace):
        comm = mat.domain.spaces[0][0].cart.global_comm

    mat_coo = mat.tosparse()

    gmat  = PETSc.Mat().create(comm=comm)

    if comm:
        # Set PETSc sparse parallel matrix type
        gmat.setType("mpiaij")
    else:
        # Set PETSc sequential matrix type
        gmat.setType("seqaij")

    # Set GLOBAL matrix size
    gmat.setSizes(mat.shape)        
    gmat.setFromOptions()

    rows, cols, data = mat_coo.row, mat_coo.col, mat_coo.data

    if comm:
        # Preallocate number of nonzeros
        row_lengths = np.count_nonzero(rows[None,:] == np.unique(rows)[:,None], axis=1).max()
        # NNZ is the number of non-zeros per row for the local portion of the matrix
        NNZ = comm.allreduce(row_lengths, op=MPI.MAX)
        gmat.setPreallocationNNZ(NNZ)

    # Fill-in matrix values
    for i in range(rows.size):
        # The values have to be set in "addition mode", otherwise the default just takes the new value.
        # This is here necessary, since the COO format can contain repeated entries.
        gmat.setValues(rows[i], cols[i], data[i], addv=PETSc.InsertMode.ADD_VALUES)

    # Process inserted matrix entries
    ################################################
    # Note 12.03.2024:
    # In the assembly PETSc uses global communication to distribute the matrix in a different way than Psydac.
    # For this reason, at the moment we cannot compare directly each distributed 'chunck' of the Psydac and the PETSc matrices.
    # In the future we would like that PETSc uses the partition from Psydac, 
    # which might involve passing a DM Object.
    ################################################
    gmat.assemble()
    return gmat
