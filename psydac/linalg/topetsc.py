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
    gvec.setSizes(globalsize)
    gvec.setFromOptions()
    gvec.setValues(indices, data)
    gvec.assemble()
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
        NNZ = comm.allreduce(data.size, op=MPI.SUM)
        gmat.setPreallocationNNZ(NNZ)

    # Fill-in matrix values
    for i in range(rows.size):
        # The values have to be set in "addition mode", otherwise the default just takes the new value.
        # This is here necessary, since the COO format can contain repeated entries and they must be added.
        gmat.setValues(rows[i], cols[i], data[i], addv=PETSc.InsertMode.ADD_VALUES)

    # Process inserted matrix entries
    gmat.assemble()
    return gmat

'''def mat_topetsc( mat ):
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

    mat_coo = mat.tosparse() #has the shape of the global operator
    print('global shape', mat.shape)
    print('dense matrix', mat_coo.todense())
    print('mat_coo.row', mat_coo.row)
    print('mat_coo.col', mat_coo.col)
    mat_csr = mat_coo.tocsr()
    print('mat_csr.indptr', mat_csr.indptr)
    print('mat_csr.indices', mat_csr.indices)

    gmat = PETSc.Mat().create(comm=comm)


 
    #gmat.setSizes(mat.shape)
    #mat.setSizes([[nrl, nrg], [ncl, ncg]])
    gmat.setSizes([[mat.shape[0]//2, mat.shape[0]], [mat.shape[1]//2, mat.shape[1]]])

    #gmat.setSizes((nrows, ncols))

    # Set sparse matrix type
    gmat.setType("mpiaij")   
    gmat.setFromOptions()

    '''if comm:
        # Preallocate number of nonzeros based on CSR structure
        gmat.setPreallocationCSR((mat_csr.indptr, mat_csr.indices))
        #NNZ = comm.allreduce(mat_csr.size, op=MPI.SUM)
        #gmat.setPreallocationNNZ(NNZ)

    # Fill-in matrix values from CSR data
        #indptr: Stores accumulated number of non-zero entries
        #indices: Stores column index of entries
        #data: Stores non-zero entries

    nrows = len(mat_csr.indptr)-1 #indptr always has a 0 in the first position to avoid void array
    for r in range(nrows):
        # get number of non zero entries to fill in row r
        num_non_zero = mat_csr.indptr[r+1] - mat_csr.indptr[r]
        for k in range(num_non_zero):
            # set the value in correct column
            gmat.setValues(r, mat_csr.indices[r+k], mat_csr.data[r+k])  
        #cols = mat_csr.indices[mat_csr.indptr[i]:mat_csr.indptr[i+1]]
        #col_data = mat_csr.data[mat_csr.indptr[i]:mat_csr.indptr[i+1]]
        #gmat.setValues(i, cols, col_data)  
   
    # Process inserted matrix entries
    gmat.assemble()
    return gmat
'''