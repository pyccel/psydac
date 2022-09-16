import numpy as np

from psydac.linalg.block import BlockVectorSpace, BlockVector, BlockMatrix
from psydac.linalg.stencil import StencilVectorSpace, StencilVector
from scipy.sparse import coo_matrix, bmat

from mpi4py import MPI

def get_npts( space ):

    if isinstance(space, StencilVectorSpace):
        npts = np.product(space.npts)
    elif isinstance(space, BlockVectorSpace):
        npts = sum([get_npts(s) for s in space.spaces])
    else:
        raise TypeError("Expected StencilVectorSpace or BlockVectorSpace, found instead {}".format(type(space)))

    return npts

def vec_tocoo( vec ):

    if isinstance(vec, StencilVector):
        npts = vec.space.npts
        idx = tuple( slice(m*p,-m*p) for m,p in zip(vec.pads, vec.space.shifts) )
        shape = vec._data[idx].shape
        starts = vec.space.starts
        indices = np.array([np.ravel_multi_index( [s+x for s,x in zip(starts, xx)], dims=npts,  order='C' ) for xx in np.ndindex(*shape)] )
        data = vec._data[idx].flatten()
        vec = coo_matrix(
                    (data,(indices,indices)),
                    shape = [np.prod(npts),np.prod(npts)],
                    dtype = vec.space.dtype)
    elif isinstance(vec, BlockVector):
        vecs = [vec_tocoo(b) for b in vec.blocks]
        blocks = [[None]*len(vecs) for v in vecs]
        for i,v in enumerate(vecs):
            blocks[i][i] = v
        return bmat(blocks,format='coo')
    else:
        raise TypeError("Expected StencilVector or BlockVector, found instead {}".format(type(vec)))
    return vec

def vec_topetsc( vec ):
    """ Convert to petsc data structure.
    """

    from petsc4py import PETSc
    comm = vec.space.spaces[0].cart.global_comm if isinstance(vec, BlockVector) else vec.space.cart.global_comm
    globalsize = get_npts(vec.space)
    vec = vec_tocoo(vec)
    gvec  = PETSc.Vec().create(comm=comm)
    gvec.setSizes(globalsize)
    gvec.setFromOptions()
    gvec.setValues(vec.row, vec.data)
    gvec.assemble()
    return gvec

def mat_topetsc( mat ):
    """ Convert to petsc data structure.
    """

    from petsc4py import PETSc

    comm = mat.domain.spaces[0].cart.global_comm if isinstance(mat, BlockMatrix) else mat.domain.cart.global_comm
    mat_coo = mat.tosparse()
    ncols = get_npts(mat.domain)
    nrows = get_npts(mat.codomain)
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

