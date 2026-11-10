#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from itertools import product as cartesian_prod

import numpy as np

from psydac.linalg.basic   import VectorSpace
from psydac.linalg.block   import BlockVectorSpace, BlockVector, BlockLinearOperator
from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.kernels.stencil2IJV_kernels import stencil2IJV_1d_C, stencil2IJV_2d_C, stencil2IJV_3d_C

__all__ = (
    'petsc_local_to_psydac',
    'psydac_to_petsc_global',
    'get_npts_local',
    'get_npts_per_block',
    'vec_topetsc',
    'mat_topetsc'
)

# Dictionary used to select the correct kernel function based on dimensionality
kernels = {
    'stencil2IJV': {'F': None,
                    'C': (None,   stencil2IJV_1d_C,   stencil2IJV_2d_C,   stencil2IJV_3d_C)}
}


def get_index_shift_per_block_per_process(V):
    npts_local_per_block_per_process = np.array(get_npts_per_block(V)) #indexed [b,k,d] for block b and process k and dimension d
    local_sizes_per_block_per_process = np.prod(npts_local_per_block_per_process, axis=-1) #indexed [b,k] for block b and process k

    n_blocks = npts_local_per_block_per_process.shape[0]
    n_procs = npts_local_per_block_per_process.shape[1]

    index_shift_per_block_per_process = [[0 + np.sum(local_sizes_per_block_per_process[:,:k]) + np.sum(local_sizes_per_block_per_process[:b,k]) for k in range(n_procs)] for b in range(n_blocks)]

    return index_shift_per_block_per_process #Global variable indexed as [b][k] fo block b, process k


def toIJVrowmap(mat_block, bd, bc, I, J, V, rowmap, dspace, cspace, dnpts_block, cnpts_block, dshift_block, cshift_block, order='C'):
    # Extract Cartesian decomposition of the Block where the node is:
    dspace_block = dspace if isinstance(dspace, StencilVectorSpace) else dspace.spaces[bd]
    cspace_block = cspace if isinstance(cspace, StencilVectorSpace) else cspace.spaces[bc]       

    # Shortcuts
    cnl = [np.int64(n) for n in get_npts_local(cspace_block)[0]] 
    dng = [np.int64(n) for n in dspace_block.cart.npts]
    cs = [np.int64(s) for s in cspace_block.cart.starts]
    cp = [np.int64(p) for p in cspace_block.cart.pads]
    cm = [np.int64(m) for m in cspace_block.cart.shifts]
    dsh = np.array(dshift_block, dtype='int64')
    csh = np.array(cshift_block, dtype='int64')

    dgs = [np.array(gs, dtype='int64') for gs in dspace_block.cart.global_starts] # Global variable
    dge = [np.array(ge, dtype='int64') for ge in dspace_block.cart.global_ends] # Global variable
    cgs = [np.array(gs, dtype='int64') for gs in cspace_block.cart.global_starts] # Global variable
    cge = [np.array(ge, dtype='int64') for ge in cspace_block.cart.global_ends] # Global variable

    dnlb = [np.array([n[d] for n in dnpts_block], dtype='int64') for d in range(dspace_block.cart.ndim)] 
    cnlb = [np.array([n[d] for n in cnpts_block] , dtype='int64') for d in range(cspace_block.cart.ndim)]

    # Range of data owned by local process (no ghost regions)
    local = tuple( [slice(m*p,-m*p) for p,m in zip(cp, cm)] + [slice(None)] * dspace_block.cart.ndim )
    shape  = mat_block._data[local].shape
    nrows = np.prod(shape[0:dspace_block.cart.ndim])
    nentries = np.prod(shape)

    # locally block I, J, V, rowmap storage
    Ib = np.zeros(nrows + 1, dtype='int64')
    Jb = np.zeros(nentries, dtype='int64')
    rowmapb = np.zeros(nrows, dtype='int64')
    Vb = np.zeros(nentries, dtype=mat_block._data.dtype)

    Ib[0] += I[-1]

    stencil2IJV = kernels['stencil2IJV'][order][dspace_block.cart.ndim]

    nnz_rows, nnz = stencil2IJV(mat_block._data, Ib, Jb, Vb, rowmapb,
                      *cnl, *dng, *cs, *cp, *cm,
                      dsh, csh, *dgs, *dge, *cgs, *cge, *dnlb, *cnlb
                      )

    I += list(Ib[1:nnz_rows + 1])
    rowmap += list(rowmapb[:nnz_rows])
    J += list(Jb[:nnz])
    V += list(Vb[:nnz])

    return I, J, V, rowmap


def petsc_local_to_psydac(
    V : VectorSpace,
    petsc_index : int):
    """
    Convert the PETSc local index (starting from 0 in each process) to a PSYDAC local index (natural multi-index, as grid coordinates).

    Parameters
    ----------
    V : VectorSpace
        The vector space to which the PSYDAC vector belongs.
        This defines the number of blocks, the size of each block,
        and how each block is distributed across MPI processes.

    petsc_index : int
        The local PETSc index. The 0 index is only owned by every process.

    Returns
    -------
    block: tuple
        The block where the PSYDAC multi-index belongs to.

    psydac_index : tuple
        The PSYDAC local multi-index. This index is local the block.
    """
    # Get the number of points for each block and each dimension local to the current process:
    npts_local_per_block = np.array(get_npts_local(V)) # indexed [b,d] for block b and dimension d
    # Get the local size of the current process for each block:
    local_sizes_per_block = np.prod(npts_local_per_block, axis=-1)  # indexed [b] for block b
    # Compute the accumulated local size of the current process for each block:
    accumulated_local_sizes_per_block = np.concatenate((np.zeros((1,), dtype=int), np.cumsum(local_sizes_per_block, axis=0))) #indexed [b+1] for block b

    n_blocks = local_sizes_per_block.size

    # Find the block where the index belongs to:
    bb = np.nonzero(
            np.array(
                [petsc_index in range(accumulated_local_sizes_per_block[b], accumulated_local_sizes_per_block[b+1]) 
                    for b in range(n_blocks)]
            ))[0][0]

    if isinstance(V, BlockVectorSpace):
        V = V.spaces[bb]

    ndim = V.ndim
    p = V.pads
    m = V.shifts    

    # Get the number of points for each dimension local to the current process and block:
    npts_local = npts_local_per_block[bb]
    
    # Get the PETSc index local within the block:
    petsc_index -= accumulated_local_sizes_per_block[bb]
    
    ii = np.zeros((ndim,), dtype=int)
    if ndim == 1:
        ii[0] = petsc_index + p[0]*m[0]

    elif ndim == 2:
        ii[0] = petsc_index // npts_local[1] + p[0]*m[0]
        ii[1] = petsc_index % npts_local[1] + p[1]*m[1]

    elif ndim == 3:
        ii[0] = petsc_index // (npts_local[1]*npts_local[2]) + p[0]*m[0]
        ii[1] = petsc_index // npts_local[2] + p[1]*m[1] - npts_local[1]*(ii[0] - p[0]*m[0])
        ii[2] = petsc_index % npts_local[2] + p[2]*m[2]

    else:
        raise NotImplementedError( "Cannot handle more than 3 dimensions." )

    return (bb,), tuple(ii)


def psydac_to_petsc_global(
        V : VectorSpace, 
        block_indices, 
        ndarray_indices) -> int:
    """
    Convert the PSYDAC local index (natural multi-index, as grid coordinates) to a PETSc global index. Perform a search to find the process owning the multi-index.

    Parameters
    ----------
    V : VectorSpace
        The vector space to which the PSYDAC vector belongs.
        This defines the number of blocks, the size of each block,
        and how each block is distributed across MPI processes.

    block_indices : tuple[int]
        The indices which identify the block in a (possibly nested) block vector.
        In the case of a StencilVector this is an empty tuple.

    ndarray_indices : tuple[int]
        The multi-index which identifies an element in the _data array,
        excluding the ghost regions.

    Returns
    -------
    petsc_index : int
        The global PETSc index. The 0 index is only owned by the first process.
    """
    bb = block_indices[0]
    # Get the number of points per block, per process and per dimension:
    npts_local_per_block_per_process = np.array(get_npts_per_block(V)) #indexed [b,k,d] for block b and process k and dimension d
    # Get the local sizes per block and per process:
    local_sizes_per_block_per_process = np.prod(npts_local_per_block_per_process, axis=-1) #indexed [b,k] for block b and process k

    # Extract Cartesian decomposition of the Block where the node is:
    if isinstance(V, BlockVectorSpace):
        V = V.spaces[bb]

    cart = V.cart

    nprocs = cart.nprocs # Number of processes in each dimension
    ndim = cart.ndim

    # Get global starts and ends to find process owning the node.
    gs = cart.global_starts # Global variable
    ge = cart.global_ends # Global variable

    jj = ndarray_indices

    if ndim == 1:
        if cart.comm:
            # Find to which process the node belongs to:
            proc_index = np.nonzero(np.array([jj[0] in range(gs[0][k],ge[0][k]+1) for k in range(gs[0].size)]))[0][0]
        else:
            proc_index = 0
       
        # Find the index shift corresponding to the block and the owner process:
        index_shift = 0 + np.sum(local_sizes_per_block_per_process[:,:proc_index]) + np.sum(local_sizes_per_block_per_process[:bb,proc_index])

        # Compute the global PETSc index:
        global_index = index_shift + jj[0] - gs[0][proc_index]

    elif ndim == 2:
        if cart.comm:
            # Find to which process the node belongs to:
            proc_x = np.nonzero(np.array([jj[0] in range(gs[0][k],ge[0][k]+1) for k in range(gs[0].size)]))[0][0]
            proc_y = np.nonzero(np.array([jj[1] in range(gs[1][k],ge[1][k]+1) for k in range(gs[1].size)]))[0][0]
        else:
            proc_x = 0
            proc_y = 0

        proc_index = proc_y + proc_x*nprocs[1]
        # Find the index shift corresponding to the block and the owner process:
        index_shift = 0 + np.sum(local_sizes_per_block_per_process[:,:proc_index]) + np.sum(local_sizes_per_block_per_process[:bb,proc_index])

        # Compute the global PETSc index:
        global_index = index_shift + jj[1] - gs[1][proc_y] + (jj[0] - gs[0][proc_x]) * npts_local_per_block_per_process[bb,proc_index,1]

    elif ndim == 3:
        if cart.comm:
            # Find to which process the node belongs to:
            proc_x = np.nonzero(np.array([jj[0] in range(gs[0][k],ge[0][k]+1) for k in range(gs[0].size)]))[0][0]
            proc_y = np.nonzero(np.array([jj[1] in range(gs[1][k],ge[1][k]+1) for k in range(gs[1].size)]))[0][0]
            proc_z = np.nonzero(np.array([jj[2] in range(gs[2][k],ge[2][k]+1) for k in range(gs[2].size)]))[0][0]
        else:
            proc_x = 0
            proc_y = 0
            proc_z = 0

        proc_index = proc_z + proc_y*nprocs[2] + proc_x*nprocs[1]*nprocs[2]

        # Find the index shift corresponding to the block and the owner process:
        index_shift = 0 + np.sum(local_sizes_per_block_per_process[:,:proc_index]) + np.sum(local_sizes_per_block_per_process[:bb,proc_index])

        # Compute the global PETSc index:
        global_index = index_shift \
                    +  jj[2] - gs[2][proc_z] \
                    + (jj[1] - gs[1][proc_y]) * npts_local_per_block_per_process[bb][proc_index][2] \
                    + (jj[0] - gs[0][proc_x]) * npts_local_per_block_per_process[bb][proc_index][1] * npts_local_per_block_per_process[bb][proc_index][2]

    else:
        raise NotImplementedError( "Cannot handle more than 3 dimensions." )

    return global_index 


def get_npts_local(V : VectorSpace) -> list:
    """
    Compute the local number of nodes per dimension owned by the actual process. 
    This is a local variable, its value will be different for each process.

    Parameters
    ----------
    V : VectorSpace
        The distributed PSYDAC vector space.

    Returns
    -------
    list
        Local number of nodes per dimension owned by the actual process.
        In case of a StencilVectorSpace the list contains a single list with length equal the number of dimensions in the domain.
        In case of a BlockVectorSpace the list has length equal the number of blocks.
    """        
    if isinstance(V, StencilVectorSpace):
        s = V.starts
        e = V.ends        
        npts_local = [ e - s + 1 for s, e in zip(s, e)] #Number of points in each dimension within each process. Different for each process.
        return [npts_local]

    npts_local_per_block = []
    for b in range(V.n_blocks):
        npts_local_b = get_npts_local(V.spaces[b])
        if isinstance(V.spaces[b], StencilVectorSpace):
            npts_local_b = npts_local_b[0]
        npts_local_per_block.append(npts_local_b)

    return npts_local_per_block


def get_npts_per_block(V : VectorSpace) -> list:
    """
    Compute the number of nodes per block, process and dimension. 
    This is a global variable, its value is the same for all processes.

    Parameters
    ----------
    V : VectorSpace
        The distributed PSYDAC vector space.

    Returns
    -------
    list
        Number of nodes per block, process and dimension.
    """ 
    if isinstance(V, StencilVectorSpace):
        gs = V.cart.global_starts # Global variable
        ge = V.cart.global_ends # Global variable
        npts_local_perprocess = [ ge_i - gs_i + 1 for gs_i, ge_i in zip(gs, ge)] #Global variable
        
        #if V.cart.comm:
        npts_local_perprocess = [*cartesian_prod(*npts_local_perprocess)] #Global variable

        return [npts_local_perprocess]

    npts_local_per_block = [] 
    for b in range(V.n_blocks):
        npts_b = get_npts_per_block(V.spaces[b])
        if isinstance(V.spaces[b], StencilVectorSpace):
            npts_b = npts_b[0]
        npts_local_per_block.append(npts_b)

    return npts_local_per_block


def vec_topetsc(vec):
    """ Convert vector from PSYDAC format to a PETSc.Vec object.

    Parameters
    ----------
    vec : psydac.linalg.stencil.StencilVector | psydac.linalg.block.BlockVector
      PSYDAC StencilVector or BlockVector. In the case of a BlockVector, only the case where the blocks are StencilVector is implemented.

    Returns
    -------
    gvec : PETSc.Vec
        PETSc vector
    """
    from petsc4py import PETSc

    if isinstance(vec.space, BlockVectorSpace) and any([isinstance(vec.space.spaces[b], BlockVectorSpace) for b in range(len(vec.space.spaces))]):
        raise NotImplementedError('Conversion for block of blocks not implemented.')

    if isinstance(vec, StencilVector):
        carts = [vec.space.cart]
    elif isinstance(vec.space, BlockVectorSpace):
        carts = []
        for b in range(vec.n_blocks):
            carts.append(vec.space.spaces[b].cart)

    n_blocks = 1 if isinstance(vec, StencilVector) else vec.n_blocks

    # Get the number of points local to the current process:
    npts_local = get_npts_local(vec.space) # indexed [block, dimension]. Different for each process.

    # Number of dimensions for each cart:
    ndims = [cart.ndim for cart in carts]

    globalsize = vec.space.dimension

    # Sum over the blocks to get the total local size
    localsize = np.sum(np.prod(npts_local, axis=1))

    gvec  = PETSc.Vec().create(comm=carts[0].global_comm)    

    # Set global and local size:
    gvec.setSizes(size=(localsize, globalsize))

    gvec.setFromOptions()
    gvec.setUp()

    petsc_indices = []
    petsc_data = []

    vec_block = vec

    for b in range(n_blocks): 
        if isinstance(vec, BlockVector):
            vec_block = vec.blocks[b]
        
        s = carts[b].starts
        ghost_size = [pi*mi for pi,mi in zip(carts[b].pads, carts[b].shifts)]

        if ndims[b] == 1:
            for i1 in range(npts_local[b][0]):
                value = vec_block._data[i1 + ghost_size[0]]
                if value != 0:
                    i1_n = s[0] + i1
                    i_g = psydac_to_petsc_global(vec.space, (b,), (i1_n,))
                    petsc_indices.append(i_g)
                    petsc_data.append(value)        

        elif ndims[b] == 2:
            for i1 in range(npts_local[b][0]):
                for i2 in range(npts_local[b][1]):
                    value = vec_block._data[i1 + ghost_size[0], i2 + ghost_size[1]]
                    if value != 0:
                        i1_n = s[0] + i1
                        i2_n = s[1] + i2                    
                        i_g = psydac_to_petsc_global(vec.space, (b,), (i1_n, i2_n))
                        petsc_indices.append(i_g)
                        petsc_data.append(value)

        elif ndims[b] == 3:
            for i1 in np.arange(npts_local[b][0]):             
                for i2 in np.arange(npts_local[b][1]):
                    for i3 in np.arange(npts_local[b][2]):
                        value = vec_block._data[i1 + ghost_size[0], i2 + ghost_size[1], i3 + ghost_size[2]]
                        if value != 0:
                            i1_n = s[0] + i1
                            i2_n = s[1] + i2
                            i3_n = s[2] + i3    
                            i_g = psydac_to_petsc_global(vec.space, (b,), (i1_n, i2_n, i3_n))                    
                            petsc_indices.append(i_g)
                            petsc_data.append(value)        

    # Set the values. The values are stored in a cache memory.
    gvec.setValues(petsc_indices, petsc_data, addv=PETSc.InsertMode.ADD_VALUES) #The addition mode the values is necessary when periodic BC

    # Assemble vector with the values from the cache. Here it is where PETSc exchanges global communication.
    gvec.assemble()

    return gvec


def mat_topetsc(mat):
    """ Convert operator from PSYDAC format to a PETSc.Mat object.

    Parameters
    ----------
    mat : psydac.linalg.stencil.StencilMatrix | psydac.linalg.block.BlockLinearOperator
      PSYDAC operator. In the case of a BlockLinearOperator, only the case where the blocks are StencilMatrix is implemented.

    Returns
    -------
    gmat : PETSc.Mat
        PETSc Matrix
    """

    from petsc4py import PETSc

    assert isinstance(mat, StencilMatrix) or isinstance(mat, BlockLinearOperator), 'Conversion only implemented for StencilMatrix and BlockLinearOperator.'

    if (isinstance(mat.domain, BlockVectorSpace) and any([isinstance(mat.domain.spaces[b], BlockVectorSpace) for b in range(len(mat.domain.spaces))]))\
        or (isinstance(mat.codomain, BlockVectorSpace) and any([isinstance(mat.codomain.spaces[b], BlockVectorSpace) for b in range(len(mat.codomain.spaces))])):
        raise NotImplementedError('Conversion for block of blocks not implemented.')

    if isinstance(mat.domain, StencilVectorSpace):
        comm = mat.domain.cart.global_comm
    elif isinstance(mat.domain, BlockVectorSpace):
        comm = mat.domain.spaces[0].cart.global_comm

    nonzero_block_indices = ((0,0),) if isinstance(mat, StencilMatrix) else mat.nonzero_block_indices

    mat.update_ghost_regions()
    mat.remove_spurious_entries()

    # Get the number of points local to the current process:
    dnpts_local = get_npts_local(mat.domain) # indexed [block, dimension]. Different for each process.
    cnpts_local = get_npts_local(mat.codomain) # indexed [block, dimension]. Different for each process. 

    # Get the number of points per block, per process and per dimension:
    dnpts_per_block_per_process = np.array(get_npts_per_block(mat.domain)) # global variable, indexed as [block, process, dimension]
    cnpts_per_block_per_process = np.array(get_npts_per_block(mat.codomain)) # global variable, indexed as [block, process, dimension]

    # Get the index shift for each block and each process:
    dindex_shift = get_index_shift_per_block_per_process(mat.domain) # global variable, indexed as [block, process, dimension]
    cindex_shift = get_index_shift_per_block_per_process(mat.codomain) # global variable, indexed as [block, process, dimension]

    globalsize = mat.shape

    # Sum over the blocks to get the total local size
    localsize = (np.sum(np.prod(cnpts_local, axis=1)), np.sum(np.prod(dnpts_local, axis=1)))

    gmat  = PETSc.Mat().create(comm=comm)

    # Set global and local sizes: size=((local_rows, rows), (local_columns, columns))
    gmat.setSizes(size=((localsize[0], globalsize[0]), (localsize[1], globalsize[1])))
    
    if comm:
        # Set PETSc sparse parallel matrix type
        gmat.setType("mpiaij")
    else:
        # Set PETSc sparse sequential matrix type
        gmat.setType("seqaij")

    gmat.setFromOptions()
    gmat.setUp()

    I = [0] # Row pointers
    J = [] # Column indices
    V = [] # Values
    rowmap = [] # Row indices of rows containing non-zeros

    mat_block = mat

    for bc, bd in nonzero_block_indices:
        if isinstance(mat, BlockLinearOperator):
            mat_block = mat.blocks[bc][bd]
        dnpts_block = dnpts_per_block_per_process[bd]
        cnpts_block = cnpts_per_block_per_process[bc]
        dshift_block = dindex_shift[bd]
        cshift_block = cindex_shift[bc]

        I,J,V,rowmap = toIJVrowmap(mat_block, bd, bc, I, J, V, rowmap, mat.domain, mat.codomain, dnpts_block, cnpts_block, dshift_block, cshift_block)

    # Set the values using IJV&rowmap format. The values are stored in a cache memory.
    gmat.setValuesIJV(I, J, V, rowmap=rowmap, addv=PETSc.InsertMode.ADD_VALUES) # The addition mode is necessary when periodic BC

    # Assemble the matrix with the values from the cache. Here it is where PETSc exchanges global communication.
    gmat.assemble()
      
    return gmat
