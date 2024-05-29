import numpy as np

from psydac.linalg.block import BlockVectorSpace, BlockVector, BlockLinearOperator
from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.basic import VectorSpace
from scipy.sparse import coo_matrix, bmat
from itertools import product as cartesian_prod

from mpi4py import MPI

__all__ = ('petsc_local_to_psydac', 'psydac_to_petsc_global', 'get_npts_local', 'get_npts_per_block', 'vec_topetsc', 'mat_topetsc')

from .kernels.stencil2IJV_kernels import stencil2IJV_1d_C, stencil2IJV_2d_C, stencil2IJV_3d_C
# Dictionary used to select correct kernel functions based on dimensionality
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

def toIJVrowmap(mat_block, bd, bc, I, J, V, rowmap, dspace, cspace, order='C'):

    '''# Get the number of points per block, per process and per dimension:
    dnpts_local_per_block_per_process = np.array(get_npts_per_block(dspace)) #indexed [b,k,d] for block b and process k and dimension d
    cnpts_local_per_block_per_process = np.array(get_npts_per_block(cspace)) 
    # Get the local sizes per block and per process:
    dlocal_sizes_per_block_per_process = np.prod(dnpts_local_per_block_per_process, axis=-1) #indexed [b,k] for block b and process k
    clocal_sizes_per_block_per_process = np.prod(cnpts_local_per_block_per_process, axis=-1)

    dn_blocks = dnpts_local_per_block_per_process.shape[0]
    dn_procs = dnpts_local_per_block_per_process.shape[1]
    dindex_shift = [[0 + np.sum(dlocal_sizes_per_block_per_process[:,:k]) + np.sum(dlocal_sizes_per_block_per_process[:b,k]) for k in range(dn_procs)] for b in range(dn_blocks)]

    cn_blocks = cnpts_local_per_block_per_process.shape[0]
    cn_procs = cnpts_local_per_block_per_process.shape[1]
    cindex_shift = [[0 + np.sum(clocal_sizes_per_block_per_process[:,:k]) + np.sum(clocal_sizes_per_block_per_process[:b,k]) for k in range(cn_procs)] for b in range(cn_blocks)]
    '''

    dnpts_local_per_block_per_process = np.array(get_npts_per_block(dspace))
    cnpts_local_per_block_per_process = np.array(get_npts_per_block(cspace))

    dindex_shift = get_index_shift_per_block_per_process(dspace)
    cindex_shift = get_index_shift_per_block_per_process(cspace)
    # Extract Cartesian decomposition of the Block where the node is:
    dspace_block = dspace if isinstance(dspace, StencilVectorSpace) else dspace.spaces[bd]
    cspace_block = cspace if isinstance(cspace, StencilVectorSpace) else cspace.spaces[bc]       

    # Shortcuts
    cnl = [np.int64(n) for n in get_npts_local(cspace_block)[0]] 
    dng = [np.int64(n) for n in dspace_block.cart.npts]
    cs = [np.int64(s) for s in cspace_block.cart.starts]
    cp = [np.int64(p) for p in cspace_block.cart.pads]
    cm = [np.int64(m) for m in cspace_block.cart.shifts]
    dsh = np.array(dindex_shift[bd], dtype='int64') #[np.array(sh, dtype='int64') for sh in dindex_shift[bd]]
    csh = np.array(cindex_shift[bc], dtype='int64') #[np.array(sh, dtype='int64') for sh in cindex_shift[bc]]

    dgs = [np.array(gs, dtype='int64') for gs in dspace_block.cart.global_starts] # Global variable
    dge = [np.array(ge, dtype='int64') for ge in dspace_block.cart.global_ends] # Global variable
    cgs = [np.array(gs, dtype='int64') for gs in cspace_block.cart.global_starts] # Global variable
    cge = [np.array(ge, dtype='int64') for ge in cspace_block.cart.global_ends] # Global variable

    #dnlb = [np.array(n, dtype='int64') for n in dnpts_local_per_block_per_process[bd]] 
    #cnlb = [np.array(n, dtype='int64') for n in cnpts_local_per_block_per_process[bc]] 

    dnlb = [np.array([n[d] for n in dnpts_local_per_block_per_process[bd]], dtype='int64') for d in range(dspace_block.cart.ndim)] 
    cnlb = [np.array([n[d] for n in cnpts_local_per_block_per_process[bc]] , dtype='int64') for d in range(cspace_block.cart.ndim)]

    # Range of data owned by local process (no ghost regions)
    local = tuple( [slice(m*p,-m*p) for p,m in zip(cp, cm)] + [slice(None)] * dspace_block.cart.ndim )
    shape  = mat_block._data[local].shape
    
    nrows = np.prod(shape[0:dspace_block.cart.ndim])
    nentries = np.prod(shape)
    # I, J, V, rowmap storage
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
    Convert the PETSc local index (starting from 0 in each process) to a Psydac local index (natural multi-index, as grid coordinates).

    Parameters
    -----------
    V : VectorSpace
        The vector space to which the Psydac vector belongs.
        This defines the number of blocks, the size of each block,
        and how each block is distributed across MPI processes.

    petsc_index : int
        The local PETSc index. The 0 index is only owned by every process.

    Returns
    --------
    block: tuple
        The block where the Psydac multi-index belongs to.
    psydac_index : tuple
        The Psydac local multi-index. This index is local the block.
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
        block_indices : 'tuple[int]', 
        ndarray_indices : 'tuple[int]') -> int:
    """
    Convert the Psydac local index (natural multi-index, as grid coordinates) to a PETSc global index. Performs a search to find the process owning the multi-index.

    Parameters
    -----------
    V : VectorSpace
        The vector space to which the Psydac vector belongs.
        This defines the number of blocks, the size of each block,
        and how each block is distributed across MPI processes.

    block_indices : tuple[int]
        The indices which identify the block in a (possibly nested) block vector.
        In the case of a StencilVector this is an empty tuple.

    ndarray_indices : tuple[int]
        The multi-index which identifies an element in the _data array,
        excluding the ghost regions.

    Returns
    --------
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

    Parameter
    ---------
    V : VectorSpace
        The distributed Psydac vector space.

    Returns
    --------
    list
        Local number of nodes per dimension owned by the actual process.
        In case of a StencilVectorSpace the list has length equal the number of dimensions in the domain.
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

    Parameter
    ---------
    V : VectorSpace
        The distributed Psydac vector space.

    Returns
    --------
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

def vec_topetsc( vec ):
    """ Convert vector from Psydac format to a PETSc.Vec object.

    Parameters
    ----------
    vec : psydac.linalg.stencil.StencilVector | psydac.linalg.block.BlockVector
      Psydac StencilVector or BlockVector. In the case of a BlockVector, only the case where the blocks are StencilVector is implemented.

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

def mat_topetsc( mat ):
    """ Convert operator from Psydac format to a PETSc.Mat object.

    Parameters
    ----------
    mat : psydac.linalg.stencil.StencilMatrix | psydac.linalg.block.BlockLinearOperator
      Psydac operator. In the case of a BlockLinearOperator, only the case where the blocks are StencilMatrix is implemented.

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
        dcarts = [mat.domain.cart]
    elif isinstance(mat.domain, BlockVectorSpace):
        dcarts = []
        for b in range(len(mat.domain.spaces)):
                dcarts.append(mat.domain.spaces[b].cart)

    if isinstance(mat.codomain, StencilVectorSpace):
        ccarts = [mat.codomain.cart]
    elif isinstance(mat.codomain, BlockVectorSpace):
        ccarts = []
        for b in range(len(mat.codomain.spaces)):
                ccarts.append(mat.codomain.spaces[b].cart)

    nonzero_block_indices = ((0,0),) if isinstance(mat, StencilMatrix) else mat.nonzero_block_indices

    mat.update_ghost_regions()
    mat.remove_spurious_entries()

    # Number of dimensions for each cart:
    dndims = [dcart.ndim for dcart in dcarts]
    cndims = [ccart.ndim for ccart in ccarts]
    # Get global number of points per block:
    dnpts =  [dcart.npts for dcart in dcarts] # indexed [block, dimension]. Same for all processes.   

    # Get the number of points local to the current process:
    dnpts_local = get_npts_local(mat.domain) # indexed [block, dimension]. Different for each process.
    cnpts_local = get_npts_local(mat.codomain) # indexed [block, dimension]. Different for each process. 

    globalsize = mat.shape

    # Sum over the blocks to get the total local size
    localsize = (np.sum(np.prod(cnpts_local, axis=1)), np.sum(np.prod(dnpts_local, axis=1)))

    gmat  = PETSc.Mat().create(comm=dcarts[0].global_comm)

    # Set global and local sizes: size=((local_rows, rows), (local_columns, columns))
    gmat.setSizes(size=((localsize[0], globalsize[0]), (localsize[1], globalsize[1])))
    
    if dcarts[0].global_comm:
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

    import time

    output = open('output.txt', 'a')
    comm=dcarts[0].global_comm

    time_loop = np.empty((comm.Get_size(),))
    time_setValues = np.empty((comm.Get_size(),)) 
    time_assemble = np.empty((comm.Get_size(),))
                             

    t_prev = time.time()
    for bc, bd in nonzero_block_indices:
        if isinstance(mat, BlockLinearOperator):
            mat_block = mat.blocks[bc][bd]
        I,J,V,rowmap = toIJVrowmap(mat_block, bd, bc, I, J, V, rowmap, mat.domain, mat.codomain)

    """    cs = ccarts[bc].starts
        cghost_size = [pi*mi for pi,mi in zip(ccarts[bc].pads, ccarts[bc].shifts)]

        if dndims[bd] == 1 and cndims[bc] == 1:
            I,J,V,rowmap = toIJVrowmap(mat_block, bd, bc, I, J, V, rowmap, mat.domain, mat.codomain)
            '''for i1 in range(cnpts_local[bc][0]):
                nnz_in_row = 0
                i1_n = cs[0] + i1
                i_g = psydac_to_petsc_global(mat.codomain, (bc,), (i1_n,))

                stencil_size = mat_block._data[i1 + cghost_size[0],:].shape

                for k1 in range(stencil_size[0]):
                    value = mat_block._data[i1 + cghost_size[0], k1]

                    j1_n = (i1_n + k1 - stencil_size[0]//2) % dnpts[bd][0] # modulus is necessary for periodic BC
                    
                    if value != 0:
                        j_g = psydac_to_petsc_global(mat.domain, (bd,), (j1_n, ))

                        if nnz_in_row == 0:
                            rowmap.append(i_g)  

                        J.append(j_g)           
                        V.append(value)  

                        nnz_in_row += 1

                if nnz_in_row > 0:
                    I.append(I[-1] + nnz_in_row)'''
                
        elif dndims[bd] == 2 and cndims[bc] == 2:
            I,J,V,rowmap = toIJVrowmap(mat_block, bd, bc, I, J, V, rowmap, mat.domain, mat.codomain)
            '''for i1 in np.arange(cnpts_local[bc][0]):              
                for i2 in np.arange(cnpts_local[bc][1]):

                    nnz_in_row = 0

                    i1_n = cs[0] + i1
                    i2_n = cs[1] + i2
                    i_g = psydac_to_petsc_global(mat.codomain, (bc,), (i1_n, i2_n))

                    stencil_size = mat_block._data[i1 + cghost_size[0], i2 + cghost_size[1],:,:].shape

                    for k1 in range(stencil_size[0]):                    
                        for k2 in range(stencil_size[1]):
                            value = mat_block._data[i1 + cghost_size[0], i2 + cghost_size[1], k1, k2]

                            j1_n = (i1_n + k1 - stencil_size[0]//2) % dnpts[bd][0] # modulus is necessary for periodic BC
                            j2_n = (i2_n + k2 - stencil_size[1]//2) % dnpts[bd][1] # modulus is necessary for periodic BC

                            if value != 0:
                                j_g = psydac_to_petsc_global(mat.domain, (bd,), (j1_n, j2_n))

                                if nnz_in_row == 0:
                                    rowmap.append(i_g)     

                                J.append(j_g)
                                V.append(value)  

                                nnz_in_row += 1

                    if nnz_in_row > 0:
                        I.append(I[-1] + nnz_in_row)'''

        elif dndims[bd] == 3 and cndims[bc] == 3: 
            for i1 in np.arange(cnpts_local[bc][0]):             
                for i2 in np.arange(cnpts_local[bc][1]):
                    for i3 in np.arange(cnpts_local[bc][2]):
                        nnz_in_row = 0
                        i1_n = cs[0] + i1
                        i2_n = cs[1] + i2
                        i3_n = cs[2] + i3
                        i_g = psydac_to_petsc_global(mat.codomain, (bc,), (i1_n, i2_n, i3_n))

                        stencil_size = mat_block._data[i1 + cghost_size[0], i2 + cghost_size[1], i3 + cghost_size[2],:,:,:].shape

                        for k1 in range(stencil_size[0]):                    
                            for k2 in range(stencil_size[1]):
                                for k3 in range(stencil_size[2]):
                                    value = mat_block._data[i1 + cghost_size[0], i2 + cghost_size[1], i3 + cghost_size[2], k1, k2, k3]

                                    j1_n = (i1_n + k1 - stencil_size[0]//2) % dnpts[bd][0] # modulus is necessary for periodic BC
                                    j2_n = (i2_n + k2 - stencil_size[1]//2) % dnpts[bd][1] # modulus is necessary for periodic BC
                                    j3_n = (i3_n + k3 - stencil_size[2]//2) % dnpts[bd][2] # modulus is necessary for periodic BC

                                    if value != 0:
                                        j_g = psydac_to_petsc_global(mat.domain, (bd,), (j1_n, j2_n, j3_n))

                                        if nnz_in_row == 0: 
                                            rowmap.append(i_g)     
                                
                                        J.append(j_g)
                                        V.append(value)  

                                        nnz_in_row += 1

                        if nnz_in_row > 0:
                            I.append(I[-1] + nnz_in_row)
    """
    time_loop[comm.Get_rank()] = time.time() - t_prev

    print('Time for the loop: ', time.time() - t_prev)
    t_prev = time.time()
    # Set the values using IJV&rowmap format. The values are stored in a cache memory.
    gmat.setValuesIJV(I, J, V, rowmap=rowmap, addv=PETSc.InsertMode.ADD_VALUES) # The addition mode is necessary when periodic BC

    time_setValues[comm.Get_rank()] = time.time() - t_prev

    print('Time for the setValuesIJV: ', time.time() - t_prev)

    t_prev = time.time()
    # Assemble the matrix with the values from the cache. Here it is where PETSc exchanges global communication.
    gmat.assemble()
    time_assemble[comm.Get_rank()] = time.time() - t_prev
    print('Time for the assemble: ', time.time() - t_prev)


    if comm.Get_rank() == 0:
        print(f'\nProcess & global size & local size & Time loop & Time setValuesIJV & Time assemble ', file=output, flush=True)

    for k in range(comm.Get_size()):
        if k == comm.Get_rank():
            ls, gs = gmat.getSizes()[0]
            print(f'{k} & {gs} & {ls} & {time_loop[k]:.2f} & {time_setValues[k]:.2f} & {time_assemble[k]:.2f}', file=output, flush=True)
        comm.Barrier()

    avg_time_loop = comm.reduce(time_loop)
    avg_time_setValues = comm.reduce(time_setValues, op=MPI.SUM, root=0)
    avg_time_assemble = comm.reduce(time_assemble, op=MPI.SUM, root=0)
    
    if comm.Get_rank() == 0:
        print(f'Average & {np.mean(avg_time_loop):.2f} & {np.mean(avg_time_setValues):.2f} & {np.mean(avg_time_assemble):.2f}', file=output, flush=True)   

    output.close() 
      
    return gmat
