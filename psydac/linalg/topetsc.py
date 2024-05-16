import numpy as np

from psydac.linalg.block import BlockVectorSpace, BlockVector
from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.basic import VectorSpace
from scipy.sparse import coo_matrix, bmat
from itertools import product as cartesian_prod

from mpi4py import MPI

__all__ = ('flatten_vec', 'vec_topetsc', 'mat_topetsc')


def psydac_to_petsc_local(
     V : VectorSpace,
     block_indices : tuple[int],
     ndarray_indices : tuple[int]) -> int :
    """
    Convert the Psydac local index to a PETSc local index.

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
        The local PETSc index, which is equivalent to the global PETSc index
        but starts from 0. 
    """

    ndim = V.ndim
    starts = V.starts
    ends = V.ends
    pads = V.pads
    shifts = V.shifts
    shape = V.shape

    ii = ndarray_indices

    npts_local = [ e - s + 1 for s, e in zip(starts, ends)] #Number of points in each dimension within each process. Different for each process.
    
    assert all([ii[d] >= pads[d]*shifts[d] and ii[d] < shape[d] - pads[d]*shifts[d] for d in range(ndim)]), 'ndarray_indices within the ghost region'

    if ndim == 1:
        petsc_index = ii[0] - pads[0]*shifts[0] # global index starting from 0 in each process
    elif ndim == 2:
        petsc_index = npts_local[1] * (ii[0] - pads[0]*shifts[0]) + ii[1] - pads[1]*shifts[1] # global index starting from 0 in each process
    elif ndim == 3:
        petsc_index = npts_local[1] * npts_local[2] * (ii[0] - pads[0]*shifts[0]) + npts_local[2] * (ii[1] - pads[1]*shifts[1]) + ii[2] - pads[2]*shifts[2]
    else:
        raise NotImplementedError( "Cannot handle more than 3 dimensions." )
    
    return petsc_index

def get_petsc_local_to_global_shift(V : VectorSpace) -> int:
    """
    Compute the correct integer shift (process dependent) in order to convert
    a PETSc local index to the corresponding global index.

    Parameter
    ---------
    V : VectorSpace
        The distributed Psydac vector space.

    Returns
    --------
    int
        The integer shift which must be added to a local index
        in order to get a global index.
    """

    cart = V.cart
    comm = cart.global_comm

    if comm is None:
        return 0
    
    gstarts = cart.global_starts # Global variable
    gends = cart.global_ends # Global variable

    npts_local_perprocess = [ ge - gs + 1 for gs, ge in zip(gstarts, gends)] #Global variable
    npts_local_perprocess = [*cartesian_prod(*npts_local_perprocess)] #Global variable
    localsize_perprocess = [np.prod(npts_local_perprocess[k]) for k in range(comm.Get_size())] #Global variable
    index_shift = 0 + np.sum(localsize_perprocess[0:comm.Get_rank()], dtype=int) #Global variable

    return index_shift

def petsc_to_psydac_local(
    V : VectorSpace,
    petsc_index : int) :#-> tuple(tuple[int], tuple[int]) :
    """
    Convert the PETSc local index to a Psydac local index.
    This is the inverse of `psydac_to_petsc_local`.
    """

    ndim = V.ndim
    starts = V.starts
    ends = V.ends
    pads = V.pads
    shifts = V.shifts    

    npts_local = [ e - s + 1 for s, e in zip(starts, ends)] #Number of points in each dimension within each process. Different for each process.

    ii = np.zeros((ndim,), dtype=int)
    if ndim == 1:
        ii[0] = petsc_index + pads[0]*shifts[0] # global index starting from 0 in each process

    elif ndim == 2:
        ii[0] = petsc_index // npts_local[1] + pads[0]*shifts[0]
        ii[1] = petsc_index % npts_local[1] + pads[1]*shifts[1]

    elif ndim == 3:
        ii[0] = petsc_index // (npts_local[1]*npts_local[2]) + pads[0]*shifts[0]
        ii[1] = petsc_index // npts_local[2] + pads[1]*shifts[1] - npts_local[1]*(ii[0] - pads[0]*shifts[0])
        ii[2] = petsc_index % npts_local[2] + pads[2]*shifts[2]

    else:
        raise NotImplementedError( "Cannot handle more than 3 dimensions." )

    return tuple(tuple(ii))

def psydac_to_global(V : VectorSpace, ndarray_indices : tuple[int]) -> int:
    '''From Psydac natural multi-index (grid coordinates) to global PETSc single-index.
    Performs a search to find the process owning the multi-index.'''
    ndim = V.ndim
    s = V.starts
    e = V.ends
    p = V.pads
    m = V.shifts
    dnpts = V.cart.npts
    nprocs = V.cart.nprocs
    #dnpts_local = [ e - s + 1 for s, e in zip(s, e)] #Number of points in each dimension within each process. Different for each process.

    gs = V.cart.global_starts # Global variable
    ge = V.cart.global_ends # Global variable

    npts_local_perprocess = [ ge_i - gs_i + 1 for gs_i, ge_i in zip(gs, ge)] #Global variable
    npts_local_perprocess = [*cartesian_prod(*npts_local_perprocess)] #Global variable
    localsize_perprocess = [np.prod(npts_local_perprocess[k]) for k in range(V.cart.comm.Get_size())] #Global variable

    jj = ndarray_indices
    if ndim == 1:
        #global_index = (jj[0] - p[0]*m[0])%dnpts[0] 
        proc_index = np.nonzero(np.array([jj[0] in range(gs[0][k],ge[0][k]+1) for k in range(gs[0].size)]))[0][0]
        index_shift = 0 + np.sum(localsize_perprocess[0:proc_index], dtype=int) #Global variable
        global_index = index_shift + jj[0] - gs[0][proc_index]

    elif ndim == 2:
        proc_x = np.nonzero(np.array([jj[0] in range(gs[0][k],ge[0][k]+1) for k in range(gs[0].size)]))[0][0]
        proc_y = np.nonzero(np.array([jj[1] in range(gs[1][k],ge[1][k]+1) for k in range(gs[1].size)]))[0][0]

        proc_index = proc_y + proc_x*nprocs[1]#proc_x + proc_y*nprocs[0]
        index_shift = 0 + np.sum(localsize_perprocess[0:proc_index], dtype=int) #Global variable
        #global_index = jj[0] - gs[0][proc_x] + (jj[1] - gs[1][proc_y]) * npts_local_perprocess[proc_index][0] + index_shift 
        global_index = index_shift + jj[1] - gs[1][proc_y] + (jj[0] - gs[0][proc_x]) * npts_local_perprocess[proc_index][1]

        #x_proc_ranges = np.array([range(gs[0][k],ge[0][k]+1) for k in range(gs[0].size)])

        #hola = np.where(jj[0] in x_proc_ranges)#, gs[0], -1)
        '''for k in range(V.cart.comm.Get_size()):
            if k == V.cart.comm.Get_rank():
                print('\nRank', k, '\njj=', jj)
                print('proc_x=', proc_x)
                print('proc_y=', proc_y)
                print('proc_index=', proc_index)
                print('index_shift=', index_shift)
                print('global_index=', global_index)
                print('npts_local_perprocess=', npts_local_perprocess)
            V.cart.comm.Barrier()'''
    elif ndim == 3:
        proc_x = np.nonzero(np.array([jj[0] in range(gs[0][k],ge[0][k]+1) for k in range(gs[0].size)]))[0][0]
        proc_y = np.nonzero(np.array([jj[1] in range(gs[1][k],ge[1][k]+1) for k in range(gs[1].size)]))[0][0]
        proc_z = np.nonzero(np.array([jj[2] in range(gs[2][k],ge[2][k]+1) for k in range(gs[2].size)]))[0][0]

        proc_index = proc_z + proc_y*nprocs[2] + proc_x*nprocs[1]*nprocs[2] #proc_x + proc_y*nprocs[0]
        index_shift = 0 + np.sum(localsize_perprocess[0:proc_index], dtype=int) #Global variable
        global_index = index_shift \
                     +  jj[2] - gs[2][proc_z] \
                     + (jj[1] - gs[1][proc_y]) * npts_local_perprocess[proc_index][2] \
                     + (jj[0] - gs[0][proc_x]) * npts_local_perprocess[proc_index][1] * npts_local_perprocess[proc_index][2]


    else:
        raise NotImplementedError( "Cannot handle more than 3 dimensions." )


    return global_index

def psydac_to_singlenatural(V : VectorSpace, ndarray_indices : tuple[int]) -> int:
    ndim = V.ndim
    dnpts = V.cart.npts


    jj = ndarray_indices
    if ndim == 1:
        singlenatural_index = 0
    elif ndim == 2:
        singlenatural_index = jj[1] + jj[0] * dnpts[1] 
        

        #x_proc_ranges = np.array([range(gs[0][k],ge[0][k]+1) for k in range(gs[0].size)])

        #hola = np.where(jj[0] in x_proc_ranges)#, gs[0], -1)
        '''for k in range(V.cart.comm.Get_size()):
            if k == V.cart.comm.Get_rank():
                print('\nRank', k, '\njj=', jj)
                print('proc_x=', proc_x)
                print('proc_y=', proc_y)
                print('proc_index=', proc_index)
                print('index_shift=', index_shift)
                print('global_index=', global_index)
                print('npts_local_perprocess=', npts_local_perprocess)
            V.cart.comm.Barrier()'''

     
    else:
        raise NotImplementedError( "Cannot handle more than 3 dimensions." )


    return singlenatural_index    

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
        cart = vec.space.cart
    elif isinstance(vec.space.spaces[0], StencilVectorSpace):
        cart = vec.space.spaces[0].cart
    elif isinstance(vec.space.spaces[0], BlockVectorSpace):
        cart = vec.space.spaces[0][0].cart

    comm = cart.global_comm
    globalsize = vec.space.dimension #integer
    """    print('\n\nVEC:\nglobalsize=', globalsize)    
    gvec.setDM(Dmda)

    # Set local and global size
    gvec.setSizes(size=(ownership_ranges[comm.Get_rank()], globalsize))

    '''ownership_ranges = [comm.allgather(cart.domain_decomposition.local_ncells[k]) for k in range(cart.ndim)]
    boundary_type = [(PETSc.DM.BoundaryType.PERIODIC if cart.domain_decomposition.periods[k] else PETSc.DM.BoundaryType.NONE) for k in range(cart.ndim)]

    #ownership_ranges = [ dcart.global_ends[0][k] - dcart.global_starts[0][k] + 1 for k in range(dcart.global_starts[0].size)]
    print('VECTOR: OWNership_ranges=', ownership_ranges)
    #Dmda = PETSc.DMDA().create(dim=2, sizes=mat.shape, proc_sizes=(comm.Get_size(),1), ownership_ranges=(ownership_ranges, mat.shape[1]), comm=comm)
    # proc_sizes = [ len]
    Dmda = PETSc.DMDA().create(dim=cart.ndim, sizes=cart.domain_decomposition.ncells, proc_sizes=cart.domain_decomposition.nprocs, 
                               ownership_ranges=ownership_ranges, comm=comm, stencil_type=PETSc.DMDA.StencilType.BOX, boundary_type=boundary_type)'''
    
    ### SPLITTING COEFFS
    ownership_ranges = [ 1 + cart.global_ends[0][k] - cart.global_starts[0][k] for k in range(cart.global_starts[0].size)] 
    #ownership_ranges = [comm.allgather(dcart.domain_decomposition.local_ncells[k]) for k in range(dcart.ndim)]
    
    print('OWNership_ranges=', ownership_ranges)
    print('dcart.domain_decomposition.nprocs=', *cart.domain_decomposition.nprocs)

    boundary_type = [(PETSc.DM.BoundaryType.PERIODIC if cart.domain_decomposition.periods[k] else PETSc.DM.BoundaryType.NONE) for k in range(cart.ndim)]

    Dmda = PETSc.DMDA().create(dim=1, sizes=(globalsize,), proc_sizes=cart.domain_decomposition.nprocs, comm=comm, 
                               ownership_ranges=[ownership_ranges], boundary_type=boundary_type)
    

    indices, data = flatten_vec(vec)
    for k in range(comm.Get_size()):
        if comm.Get_rank() == k:
            print('Rank ', k)
            print('vec.toarray()=\n', vec.toarray())
            print('VEC_indices=', indices)
            print('VEC_data=', data)
        comm.Barrier()



    gvec  = PETSc.Vec().create(comm=comm)

    gvec.setDM(Dmda)

    # Set local and global size
    gvec.setSizes(size=(ownership_ranges[comm.Get_rank()], globalsize))

    '''if comm:
        cart_petsc = cart.topetsc()
        gvec.setLGMap(cart_petsc.l2g_mapping)'''


    gvec.setFromOptions()
    gvec.setUp()
    # Set values of the vector. They are stored in a cache, so the assembly is necessary to use the vector.
    gvec.setValues(indices, data, addv=PETSc.InsertMode.ADD_VALUES)"""

    ndim = vec.space.ndim
    starts = vec.space.starts
    ends = vec.space.ends
    pads = vec.space.pads
    shifts = vec.space.shifts
    #npts = vec.space.npts

    #cart = vec.space.cart

    npts_local = [ e - s + 1 for s, e in zip(starts, ends)] #Number of points in each dimension within each process. Different for each process.
    '''npts_local_perprocess = [ ge - gs + 1 for gs, ge in zip(cart.global_starts, cart.global_ends)] #Global variable
    npts_local_perprocess = [*cartesian_prod(*npts_local_perprocess)] #Global variable
    localsize_perprocess = [np.prod(npts_local_perprocess[k]) for k in range(comm.Get_size())] #Global variable'''
    index_shift = get_petsc_local_to_global_shift(vec.space) #Global variable

    '''for k in range(comm.Get_size()):
        if k == comm.Get_rank():   
            print('\nRank ', k)
            print('starts=', starts)
            print('ends=', ends)
            print('npts=', npts)
            print('pads=', pads)
            print('shifts=', shifts)
            print('npts_local=', npts_local)
            print('cart.global_starts=', cart.global_starts)
            print('cart.global_ends=', cart.global_ends)
            print('npts_local_perprocess=', npts_local_perprocess)
            print('localsize_perprocess=', localsize_perprocess)
            print('index_shift=', index_shift)

            print('vec._data.shape=', vec._data.shape)
            print('vec._data=', vec._data)
            #print('vec.toarray()=', vec.toarray())
        comm.Barrier()'''

    gvec  = PETSc.Vec().create(comm=comm)

    localsize = np.prod(npts_local)
    gvec.setSizes(size=(localsize, globalsize))#size=(ownership_ranges[comm.Get_rank()], globalsize))

    gvec.setFromOptions()
    gvec.setUp()

    petsc_indices = []
    petsc_data = []

    if ndim == 1:
        for i1 in range(pads[0]*shifts[0], pads[0]*shifts[0] + npts_local[0]):
            value = vec._data[i1]
            if value != 0:
                index = psydac_to_petsc_local(vec.space, [], (i1,)) # global index starting from 0 in each process
                index += index_shift #starts[0] # global index starting from NOT 0 in each process
                petsc_indices.append(index)
                petsc_data.append(value)        

    elif ndim == 2:
        for i1 in range(pads[0]*shifts[0], pads[0]*shifts[0] + npts_local[0]):
            for i2 in range(pads[1]*shifts[1], pads[1]*shifts[1] + npts_local[1]):
                value = vec._data[i1,i2]
                if value != 0:
                    #index = npts_local[1] * (i1 - pads[0]*shifts[0]) + i2 - pads[1]*shifts[1] # global index starting from 0 in each process
                    index = psydac_to_petsc_local(vec.space, [], (i1,i2)) # global index starting from 0 in each process
                    index += index_shift # global index starting from NOT 0 in each process
                    petsc_indices.append(index)
                    petsc_data.append(value)

    elif ndim == 3:
        for i1 in range(pads[0]*shifts[0], pads[0]*shifts[0] + npts_local[0]):
            for i2 in range(pads[1]*shifts[1], pads[1]*shifts[1] + npts_local[1]):
                for i3 in range(pads[2]*shifts[2], pads[2]*shifts[2] + npts_local[2]):
                    value = vec._data[i1, i2, i3]
                    if value != 0:
                        #index = npts_local[1] * npts_local[2] * (i1 - pads[0]*shifts[0]) + npts_local[2] * (i2 - pads[1]*shifts[1]) + i3 - pads[2]*shifts[2]
                        index = psydac_to_petsc_local(vec.space, [], (i1,i2,i3)) 
                        index += index_shift # global index starting from NOT 0 in each process
                        petsc_indices.append(index)
                        petsc_data.append(value)        

    gvec.setValues(petsc_indices, petsc_data)#, addv=PETSc.InsertMode.ADD_VALUES)
    # Assemble vector
    gvec.assemble() # Here PETSc exchanges global communication. The block corresponding to a certain process is not necessarily the same block in the Psydac StencilVector.

    '''if comm is not None:
        vec_arr = vec.toarray()
        for k in range(comm.Get_size()):
            if k == comm.Get_rank():   
                print('\nRank ', k)
                #print('petsc_indices=', petsc_indices)
                #print('petsc_data=', petsc_data)
                #print('\ngvec.array=', gvec.array.real)
                print('vec.toarray()=', vec_arr)
                #print('gvec.getSizes()=', gvec.getSizes())
            comm.Barrier()
        print('================================')'''

    
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
        dcart = mat.domain.cart
        ccart = mat.codomain.cart
    elif isinstance(mat.domain.spaces[0], StencilVectorSpace):
        dcart = mat.domain.spaces[0].cart
        ccart = mat.codomain.spaces[0].cart
    elif isinstance(mat.domain.spaces[0], BlockVectorSpace):
        dcart = mat.domain.spaces[0][0].cart
        ccart = mat.codomain.spaces[0][0].cart

    dcomm = dcart.global_comm
    ccomm = ccart.global_comm

    mat.update_ghost_regions()
    mat.remove_spurious_entries()

    dndim = dcart.ndim
    dstarts = dcart.starts
    dends = dcart.ends
    dpads = dcart.pads
    dshifts = dcart.shifts
    dnpts = dcart.npts

    cndim = ccart.ndim
    cstarts = ccart.starts
    cends = ccart.ends
    #cpads = ccart.pads
    #cshifts = ccart.shifts
    cnpts = ccart.npts 

    dnpts_local = [ e - s + 1 for s, e in zip(dstarts, dends)] #Number of points in each dimension within each process. Different for each process.
    cnpts_local = [ e - s + 1 for s, e in zip(cstarts, cends)]

    
    dindex_shift = get_petsc_local_to_global_shift(mat.domain) #Global variable
    cindex_shift = get_petsc_local_to_global_shift(mat.codomain) #Global variable


    mat_dense = mat.tosparse().todense()
    for k in range(dcomm.Get_size()):
        if k == dcomm.Get_rank():
            print('\nRank ', k)
            print('dstarts=', dstarts)
            print('dends=', dends)
            print('cstarts=', cstarts)
            print('cends=', cends)
            print('dnpts=', dnpts)            
            print('cnpts=', cnpts)
            print('dnpts_local=', dnpts_local)
            print('cnpts_local=', cnpts_local)
            #print('mat_dense=\n', mat_dense[:3])
            print('mat._data.shape=\n', mat._data.shape)
            print('dindex_shift=', dindex_shift)
            print('cindex_shift=', cindex_shift)
            print('ccart.global_starts=', ccart.global_starts)
            print('ccart.global_ends=', ccart.global_ends)
            #print('mat._data=\n', mat._data)

        dcomm.Barrier() 
        ccomm.Barrier()  


    globalsize = (np.prod(dnpts), np.prod(cnpts)) #Tuple of integers
    localsize = (np.prod(dnpts_local), np.prod(cnpts_local))

    gmat  = PETSc.Mat().create(comm=dcomm)

    gmat.setSizes(size=((localsize[0], globalsize[0]), (localsize[1], globalsize[1]))) #((local_rows, rows), (local_columns, columns))
    
    if dcomm:
        # Set PETSc sparse parallel matrix type
        gmat.setType("mpiaij")
    else:
        gmat.setType("seqaij")

    gmat.setFromOptions()
    gmat.setUp()

    print('gmat.getSizes()=', gmat.getSizes())

    '''mat_coo = mat.tosparse()
    rows_coo, cols_coo, data_coo = mat_coo.row, mat_coo.col, mat_coo.data

    mat_csr = mat_coo.tocsr()
    mat_csr.eliminate_zeros()
    data, indices, indptr = mat_csr.data, mat_csr.indices, mat_csr.indptr
    #indptr_chunk = indptr[indptr >= dcart.starts[0] and indptr <= dcart.ends[0]]
    #indices_chunk = indices[indices >= dcart.starts[1] and indices <= dcart.ends[1]]

    mat_coo_local = mat.tocoo_local()
    rows_coo_local, cols_coo_local, data_coo_local = mat_coo_local.row, mat_coo_local.col, mat_coo_local.data'''


    #mat.remove_spurious_entries()

    I = [0]
    J = []
    V = []
    J2 = []
    rowmap = []
    rowmap2 = []

    #dindices = [np.arange(p*m, p*m + n) for p, m, n in zip(dpads, dshifts, dnpts_local)]

    #[[ dcomm.Get_rank()*dnpts_local[1] + n2 + dnpts[1]*n1 for n2 in np.arange(dnpts_local[1])]  for n1 in np.arange(dnpts_local[0])]
                        
    #cindices = [np.arange(2*p*m + 1) for p, m in zip(dpads, dshifts)]

    #prod_indices = np.empty((max(dnpts_local) * max(cnpts_local), 3))
    '''prod_indices = []
    for d in range(len(dindices)):
        #prod_indices[:, d] = [*cartesian_prod(dindices[d], cindices[d])]
        prod_indices.append([*cartesian_prod(dindices[d], cindices[d])])
    '''

    #matd = mat.tosparse().todense()
    s = dstarts
    p = dpads
    m = dshifts
    ghost_size = [pi*mi for pi,mi in zip(p,m)]
   

    if dndim == 1 and cndim == 1:
        for i1 in np.arange(dnpts_local[0]):
            nnz_in_row = 0
            i1_n = s[0] + i1
            i_g = psydac_to_global(mat.codomain, (i1_n,))

            for k1 in range(-p[0]*m[0], p[0]*m[0] + 1):
                value = mat._data[i1 + ghost_size[0], (k1 + ghost_size[0])%(2*p[0]*m[0] + 1)]
                j1_n = (i1_n + k1)%dnpts[0] 
                
                if value != 0:

                    j_g = psydac_to_global(mat.domain, (j1_n, ))

                    if nnz_in_row == 0:
                        rowmap.append(i_g)  

                    J.append(j_g)           
                    V.append(value)  

                    nnz_in_row += 1

            I.append(I[-1] + nnz_in_row)
                
    elif dndim == 2 and cndim == 2:
        #ghost_size = (p[0]*m[0], p[1]*m[1])
        for i1 in np.arange(dnpts_local[0]):#dindices[0]:  #range(dpads[0]*dshifts[0] + dnpts_local[0]):               
            for i2 in np.arange(dnpts_local[1]):#dindices[1]: #range(dpads[1]*dshifts[1] + dnpts_local[1]): 

                nnz_in_row = 0
                #local_row = psydac_to_petsc_local(mat.domain, [], (i1, i2))
                #local_row += (local_row // dnpts_local[1])*dnpts_local[1]

                #cindices1 = np.arange( max(0, id1 - dindices[0][0] - dpads[0]*dshifts[0]), min(2*dpads[0]*dshifts[0], id1 - dindices[0][0] + dpads[0]*dshifts[0]) + 1)
                #cindices2 = np.arange( max(0, id2 - dindices[1][0] - dpads[1]*dshifts[1]), min(2*dpads[1]*dshifts[1], id2 - dindices[1][0] + dpads[1]*dshifts[1]) + 1)
                #cindices1 = np.arange( max(dpads[0]*dshifts[0], id1), min(2*dpads[0]*dshifts[0] + 1, id1 + 2*dpads[0]*dshifts[0]) + 1)
                #cindices2 = np.arange( max(dpads[1]*dshifts[1], id2), min(2*dpads[1]*dshifts[1] + 1, id2 + 2*dpads[1]*dshifts[1]) + 1)

                #cindices = [*cartesian_prod(cindices1, cindices2)] 
                #cindices = [[(ic1, ic2) for ic2 in np.arange(id2 - int(np.ceil(dpads[1]*dshifts[1]/2)), id2 + int(np.floor(dpads[1]*dshifts[1]/2)) + 1) - dpads[1]*dshifts[1] ] 
                #                        for ic1 in np.arange(id1 - int(np.ceil(dpads[0]*dshifts[0]/2)), id1 + int(np.floor(dpads[0]*dshifts[0]/2)) + 1) - dpads[0]*dshifts[0] ] 

                #ravel_ind_0_col = 2*dpads[1]*dshifts[1] + 1 + 2*dpads[0]*dshifts[0] - local_row #becomes negative for large row index
                #ravel_ind_0_col = ((2*dpads[1]*dshifts[1] + 1) * (2*dpads[0]*dshifts[0] + 1) ) // 2 - local_row #becomes negative for large row index

                #cindices1 = [np.arange(max(0, (4*np.prod(dpads)*np.prod(dshifts) - local_row), ) for p, m, n in zip(dpads, dshifts, dnpts_local)]

                '''if dcomm.Get_rank() == 0:
                    print('Rank 0: mat._data[',id1, ',' , id2 , ']=\n', mat._data[id1, id2])
                elif dcomm.Get_rank() == 1:
                    print('Rank 1: mat._data[',id1, ',' , id2 , ']=\n', mat._data[id1, id2])
                #dcomm.Barrier()'''

                i1_n = s[0] + i1
                i2_n = s[1] + i2
                i_g = psydac_to_global(mat.codomain, (i1_n, i2_n))
                i_n = psydac_to_singlenatural(mat.codomain, (i1_n,i2_n))

                for k in range(dcomm.Get_size()):
                    if k == dcomm.Get_rank():
                        print(f'Rank {k}: ({i1_n}, {i2_n}), i_n= {i_n}, i_g= {i_g}')
                        #print(f'global_row= {global_row}')
                    #dcomm.Barrier()
                    #ccomm.Barrier()




                for k1 in range(- p[0]*m[0], p[0]*m[0] + 1):                    
                    for k2 in range(- p[1]*m[1], p[1]*m[1] + 1):
                    #for ic1, ic2 in cindices:

                        value = mat._data[i1 + ghost_size[0], i2 + ghost_size[1], (k1 + ghost_size[0])%(2*p[0]*m[0] + 1), (k2 + ghost_size[1])%(2*p[1]*m[1] + 1)]

                        '''i1_n = s[0] + i1
                        i2_n = s[1] + i2
                        #(j1_n, j2_n) is the Psydac natural multi-index (like a grid)
                        j1_n = i1_n + k1 - p[0]
                        j2_n = i2_n + k2 - p[1]'''
                        

                        #(j1_n, j2_n) is the Psydac natural multi-index (like a grid)
                        j1_n = (i1_n + k1)%dnpts[0] #- p[0]*m[0]
                        j2_n = (i2_n + k2)%dnpts[1] #- p[1]*m[1]

                        

                        #print('i1,i2,k1,k2=', i1,i2,k1,k2)
                        #print('i1_n,i2_n,j1_n,j2_n,value=', i1_n,i2_n,j1_n,j2_n,value)


                    

                        if value != 0: #and j1_n in range(dnpts[0]) and j2_n in range(dnpts[1]):

                            j_g = psydac_to_global(mat.domain, (j1_n, j2_n))

                            global_col = psydac_to_global(mat.domain, (j1_n, j2_n))
                            #print('row,id1,id2,ic1,ic2=', local_row, id1, id2, ic1, ic2)
                            '''dindex_petsc = psydac_to_petsc_local(mat.domain, [], (id1,id2)) # global index starting from 0 in each process
                            cindex_petsc = (id1 + ic1 - 2*dpads[0]*dshifts[0]) % (2*dpads[0]*dshifts[0]) 


                            dindex_petsc += dindex_shift # global index NOT starting from 0 in each process
                            cindex_petsc += cindex_shift # global index NOT starting from 0 in each process
                            petsc_row_indices.append(dindex_petsc)
                            petsc_col_indices.append(cindex_petsc)
                            petsc_data.append(value)  

                            nnz_in_row += 1
                            J.append(cindex_petsc)
                            V.append(value)'''
                            
                            #local_row = psydac_to_petsc_local(mat.domain, [], (id1, id2))

                            if nnz_in_row == 0:
                                #rowmap.append(dindex_shift + local_row)  
                                #rowmap.append(dindex_shift + local_row)   
                                rowmap.append(i_g)     
                                rowmap2.append(psydac_to_singlenatural(mat.domain, (i1_n,i2_n)))                 
                            #J.append( (dindex_shift + local_row + ic1*(2*dpads[1]*dshifts[1] + 1) + ic2 - 2*dpads[0]*dshifts[0] - 2*dpads[1]*dshifts[1] ) \
                            #          % np.prod(dnpts) )
                            #num_zeros_0row = (2*dpads[0]*dshifts[0] + 1)*(2*dpads[1]*dshifts[1] + 1) // 2
                            #J.append( (dindex_shift + local_row \
                            #           + (ic1*(2*dpads[1]*dshifts[1] + 1) + ic2)
                            #          - num_zeros_0row \
                            #            ) \
                            #           % (np.prod(dnpts)) )

                            #ravel_ind = ic2 + (2*dpads[1]*dshifts[1] + 1) * ic1
                            #col_index = ic2 - dpads[1]*dshifts[1] + (ic1 - dpads[0]*dshifts[0])*dnpts[1]
                            #col_index = ic2 - dpads[1]*dshifts[1] + (dindex_shift+local_row) % dnpts[1] \
                            #            + (ic1 - dpads[0]*dshifts[0] + (dindex_shift+local_row) // dnpts[1]) * dnpts[1]
                            
                            J.append(j_g)
                            J2.append(psydac_to_singlenatural(mat.domain, (j1_n,j2_n)))

                            V.append(value)  

                            nnz_in_row += 1

                I.append(I[-1] + nnz_in_row)

    elif dndim == 3 and cndim == 3:
        for i1 in np.arange(dnpts_local[0]):             
            for i2 in np.arange(dnpts_local[1]):
                for i3 in np.arange(dnpts_local[2]):
                    nnz_in_row = 0
                    i1_n = s[0] + i1
                    i2_n = s[1] + i2
                    i3_n = s[2] + i3
                    i_g = psydac_to_global(mat.codomain, (i1_n, i2_n, i3_n))

                    for k1 in range(-p[0]*m[0], p[0]*m[0] + 1):                    
                        for k2 in range(-p[1]*m[1], p[1]*m[1] + 1):
                            for k3 in range(-p[2]*m[2], p[2]*m[2] + 1):
                                value = mat._data[i1 + ghost_size[0], i2 + ghost_size[1], i3 + ghost_size[2], (k1 + ghost_size[0])%(2*p[0]*m[0] + 1), (k2 + ghost_size[1])%(2*p[1]*m[1] + 1), (k3 + ghost_size[2])%(2*p[2]*m[2] + 1)]

                                j1_n = (i1_n + k1)%dnpts[0] #- ghost_size[0]
                                j2_n = (i2_n + k2)%dnpts[1] # - ghost_size[1]
                                j3_n = (i3_n + k3)%dnpts[2] # - ghost_size[2]

                                if value != 0: #and j1_n in range(dnpts[0]) and j2_n in range(dnpts[1]) and j3_n in range(dnpts[2]):
                                    j_g = psydac_to_global(mat.domain, (j1_n, j2_n, j3_n))

                                    if nnz_in_row == 0: 
                                        rowmap.append(i_g)     
                            
                                    J.append(j_g)
                                    V.append(value)  
                                    nnz_in_row += 1

                    I.append(I[-1] + nnz_in_row)




    if not dcomm:
        print()
        #print('mat_dense=', mat_dense)
        #print('gmat_dense=', gmat_dense)
    else:
        for k in range(dcomm.Get_size()):
            if k == dcomm.Get_rank():
                print('\n\nRank ', k)
                #print('mat_dense=\n', mat_dense)
                #print('petsc_row_indices=\n', petsc_row_indices)
                #print('petsc_col_indices=\n', petsc_col_indices)
                #print('petsc_data=\n', petsc_data)
                #print('owned_rows=', owned_rows)
                print('mat_dense=\n', mat_dense)
                print('I=', I)
                print('rowmap=', rowmap)
                print('rowmap2=', rowmap2)
                print('J=\n', J)
                print('J2=\n', J2)
                #print('V=\n', V)
                #print('gmat_dense=\n', gmat_dense)     
                print('\n\n============')   
            dcomm.Barrier()  
    
    import time
    t_prev = time.time() 
    '''for k in range(len(rows_coo)):
        gmat.setValues(rows_coo[k] - dcart.global_starts[0][comm.Get_rank()], cols_coo[k], data_coo[k])
    '''
    #gmat.setValuesCSR([r - dcart.global_starts[0][comm.Get_rank()] for r in indptr[1:]], indices, data)
    #gmat.setValuesLocalCSR(local_indptr, indices, data)#, addv=PETSc.InsertMode.ADD_VALUES)
    gmat.setValuesIJV(I, J, V, rowmap=rowmap, addv=PETSc.InsertMode.ADD_VALUES) # The addition mode is necessary when periodic BC


    print('Rank ', dcomm.Get_rank() if dcomm else '-', ': duration of setValuesIJV :', time.time()-t_prev)
  

  
    # Process inserted matrix entries
    ################################################
    # Note 12.03.2024:
    # In the assembly PETSc uses global communication to distribute the matrix in a different way than Psydac.
    # For this reason, at the moment we cannot compare directly each distributed 'chunck' of the Psydac and the PETSc matrices.
    # In the future we would like that PETSc uses the partition from Psydac, 
    # which might involve passing a DM Object.
    ################################################
    t_prev = time.time() 
    gmat.assemble()
    print('Rank ', dcomm.Get_rank() if dcomm else '-', ': duration of Mat assembly :', time.time()-t_prev)

    
    '''
    if not dcomm:
        gmat_dense = gmat.getDenseArray()
    else:   
        gmat_dense = gmat.getDenseLocalMatrix()
        dcomm.Barrier()
    '''
      
    return gmat


def mat_topetsc_old( mat ):
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
        dcart = mat.domain.cart
        ccart = mat.codomain.cart
    elif isinstance(mat.domain.spaces[0], StencilVectorSpace):
        dcart = mat.domain.spaces[0].cart
        ccart = mat.codomain.spaces[0].cart
    elif isinstance(mat.domain.spaces[0], BlockVectorSpace):
        dcart = mat.domain.spaces[0][0].cart
        ccart = mat.codomain.spaces[0][0].cart

    comm = dcart.global_comm


    #print('mat.shape = ', mat.shape)
    #print('rank: ', comm.Get_rank(), ', local_ncells=', dcart.domain_decomposition.local_ncells)
    #print('rank: ', comm.Get_rank(), ', nprocs=', dcart.domain_decomposition.nprocs)


    #recvbuf = np.empty(shape=(dcart.domain_decomposition.nprocs[0],1))
    #comm.allgather(sendbuf=dcart.domain_decomposition.local_ncells, recvbuf=recvbuf)
    
    ####################################
    
    ### SPLITTING DOMAIN
    #ownership_ranges = [comm.allgather(dcart.domain_decomposition.local_ncells[k]) for k in range(dcart.ndim)]
    
    '''boundary_type = [(PETSc.DM.BoundaryType.PERIODIC if mat.domain.periods[k] else PETSc.DM.BoundaryType.NONE) for k in range(dcart.ndim)]


    dim = dcart.ndim
    sizes = dcart.npts
    proc_sizes = dcart.nprocs
    #ownership_ranges = [[ 1 + dcart.global_ends[p][k] - dcart.global_starts[p][k] 
    #                                                    for k in range(dcart.global_starts[p].size)] 
    #                                                        for p in range(len(dcart.global_starts))]
    ownership_ranges = [[e - s + 1 for s,e in zip(starts, ends)] for starts, ends in zip(dcart.global_starts, dcart.global_ends)]
    print('OWNership_ranges=', ownership_ranges)
    Dmda = PETSc.DMDA().create(dim=dim, sizes=sizes, proc_sizes=proc_sizes, 
                               ownership_ranges=ownership_ranges, comm=comm, 
                               stencil_type=PETSc.DMDA.StencilType.BOX, boundary_type=boundary_type)'''
    

    '''### SPLITTING COEFFS
    ownership_ranges = [[ 1 + dcart.global_ends[p][k] - dcart.global_starts[p][k] for k in range(dcart.global_starts[p].size)] for p in range(len(dcart.global_starts))]
    #ownership_ranges = [comm.allgather(dcart.domain_decomposition.local_ncells[k]) for k in range(dcart.ndim)]
    
    print('MAT: ownership_ranges=', ownership_ranges)
    print('MAT: dcart.domain_decomposition.nprocs=', *dcart.domain_decomposition.nprocs)

    boundary_type = [(PETSc.DM.BoundaryType.PERIODIC if mat.domain.periods[k] else PETSc.DM.BoundaryType.NONE) for k in range(dcart.ndim)]

    Dmda = PETSc.DMDA().create(dim=1, sizes=mat.shape, proc_sizes=[*dcart.domain_decomposition.nprocs,1], comm=comm, 
                               ownership_ranges=[ownership_ranges[0], [mat.shape[1]]], stencil_type=PETSc.DMDA.StencilType.BOX, boundary_type=boundary_type)
    '''

    '''    if comm:
        dcart_petsc = dcart.topetsc()
        d_LG_map    = dcart_petsc.l2g_mapping

        ccart_petsc = ccart.topetsc()
        c_LG_map    = ccart_petsc.l2g_mapping

        print('Rank', comm.Get_rank(), ': dcart_petsc.local_size = ', dcart_petsc.local_size)
        print('Rank', comm.Get_rank(), ': dcart_petsc.local_shape = ', dcart_petsc.local_shape)
        print('Rank', comm.Get_rank(), ': ccart_petsc.local_size = ', ccart_petsc.local_size)
        print('Rank', comm.Get_rank(), ': ccart_petsc.local_shape = ', ccart_petsc.local_shape)

    if not comm:
        print('')
    else:
        for k in range(comm.Get_size()):
            if comm.Get_rank() == k:
                print('\nRank ', k)
                print('mat=\n', mat.tosparse().toarray())
                print('dcart.local_ncells=', dcart.domain_decomposition.local_ncells)
                print('ccart.local_ncells=', ccart.domain_decomposition.local_ncells)
                print('dcart._grids=', dcart._grids)
                print('ccart._grids=', ccart._grids)
                print('dcart.starts =', dcart.starts)
                print('dcart.ends =', dcart.ends)
                print('ccart.starts =', ccart.starts)
                print('ccart.ends =', ccart.ends)
                print('dcart.shape=', dcart.shape)
                print('dcart.npts=', dcart.npts)
                print('ccart.shape=', ccart.shape)
                print('ccart.npts=', ccart.npts)
                print('\ndcart.indices=', dcart_petsc.indices)
                print('ccart.indices=', ccart_petsc.indices)
                print('dcart.global_starts=', dcart.global_starts)
                print('dcart.global_ends=', dcart.global_ends)
                print('ccart.global_starts=', ccart.global_starts)
                print('ccart.global_ends=', ccart.global_ends)
            comm.Barrier()
    '''

    print('Dmda.getOwnershipRanges()=', Dmda.getOwnershipRanges())
    print('Dmda.getRanges()=', Dmda.getRanges())

    #LGmap = PETSc.LGMap().create(indices=)

    #dm = PETSc.DM().create(comm=comm)
    gmat  = PETSc.Mat().create(comm=comm)

    gmat.setDM(Dmda)
    # Set GLOBAL matrix size
    #gmat.setSizes(mat.shape)

    #gmat.setSizes(size=((dcart.domain_decomposition.local_ncells[0],mat.shape[0]), (mat.shape[1],mat.shape[1])),
    #              bsize=None)
    #gmat.setSizes([[dcart_petsc.local_size, mat.shape[0]], [ccart_petsc.local_size, mat.shape[1]]])      #mat.setSizes([[nrl, nrg], [ncl, ncg]])
    
    local_rows = np.prod([e - s + 1 for s, e in zip(ccart.starts, ccart.ends)])
    #local_columns = np.prod([p*m for p, m in zip(mat.domain.pads, mat.domain.shifts)])
    local_columns = np.prod([e - s + 1 for s, e in zip(dcart.starts, dcart.ends)])    
    rows = mat.shape[0]
    columns = mat.shape[1]
    gmat.setSizes(size=((local_rows, rows), (local_columns, columns))) #((local_rows, rows), (local_columns, columns))
    
    if comm:
        # Set PETSc sparse parallel matrix type
        gmat.setType("mpiaij")
        #gmat.setLGMap(c_LG_map, d_LG_map)
    else:
        # Set PETSc sequential matrix type
        gmat.setType("seqaij")

    gmat.setFromOptions()
    gmat.setUp()

    print('gmat.getSizes()=', gmat.getSizes())

    mat_coo = mat.tosparse()
    rows_coo, cols_coo, data_coo = mat_coo.row, mat_coo.col, mat_coo.data

    mat_csr = mat_coo.tocsr()
    mat_csr.eliminate_zeros()
    data, indices, indptr = mat_csr.data, mat_csr.indices, mat_csr.indptr
    #indptr_chunk = indptr[indptr >= dcart.starts[0] and indptr <= dcart.ends[0]]
    #indices_chunk = indices[indices >= dcart.starts[1] and indices <= dcart.ends[1]]

    mat_coo_local = mat.tocoo_local()
    rows_coo_local, cols_coo_local, data_coo_local = mat_coo_local.row, mat_coo_local.col, mat_coo_local.data

    local_petsc_index = psydac_to_petsc_local(mat.domain, [], [2,0])
    global_petsc_index = get_petsc_local_to_global_shift(mat.domain)


    print('dcart.global_starts=', dcart.global_starts)
    print('dcart.global_ends=', dcart.global_ends)
    
    '''for k in range(comm.Get_size()):
        if comm.Get_rank() == k:
            local_indptr = indptr[1 + dcart.global_starts[0][comm.Get_rank()]:2+dcart.global_ends[0][comm.Get_rank()]]
            local_indptr = [row_pter - dcart.global_starts[0][comm.Get_rank()] for row_pter in local_indptr]
            local_indptr = [0, *local_indptr]
        comm.Barrier()'''
    
    '''if comm.Get_rank() == 0:
        local_indptr = [3,6]
    else:
        local_indptr = [3]'''
    #local_indptr = indptr[1+ dcart.global_starts[comm.Get_rank()][0]:dcart.global_ends[comm.Get_rank()][0]+2]
    #local_indptr = [0, *local_indptr]

    if not comm:
        print('178:indptr = ', indptr)
        print('178:indices = ', indices)
        print('178:data = ', data)
    else:
        for k in range(comm.Get_size()):
            if comm.Get_rank() == k:
                print('\nRank ', k)
                print('mat=\n', mat_csr.toarray())
                print('CSR: indptr = ', indptr)
                #print('local_indptr = ', local_indptr)
                print('CSR: indices = ', indices)
                print('CSR: data = ', data)  


                '''print('data_coo_local=', data_coo_local)
                print('rows_coo_local=', rows_coo_local)
                print('cols_coo_local=', cols_coo_local)

                print('data_coo=', data_coo)
                print('rows_coo=', rows_coo)
                print('cols_coo=', cols_coo)'''

            comm.Barrier()      

    '''rows, cols, data = mat_coo.row, mat_coo.col, mat_coo.data
    for k in range(comm.Get_size()):
        if k == comm.Get_rank():
            print('\nRank ', k, ': data.size =', data.size)
            print('rows=', rows)
            print('cols=', cols)
            print('data=', data)
        comm.Barrier()'''
    

    import time
    t_prev = time.time() 
    '''for k in range(len(rows_coo)):
        gmat.setValues(rows_coo[k] - dcart.global_starts[0][comm.Get_rank()], cols_coo[k], data_coo[k])
    '''
    #gmat.setValuesCSR([r - dcart.global_starts[0][comm.Get_rank()] for r in indptr[1:]], indices, data)
    #gmat.setValuesLocalCSR(local_indptr, indices, data)#, addv=PETSc.InsertMode.ADD_VALUES)
    
    r = gmat.Stencil(0,0,0)
    c = gmat.Stencil(0,0,0)
    s = np.prod([p*m for p, m in zip(mat.domain.pads, mat.domain.shifts)])
    print('r=', r)
    for k in range(comm.Get_size()):
        if comm.Get_rank() == k:
            print('\nrank ', k)
            print('mat._data=', mat._data)
            
            print('mat.domain.pads=', mat.domain.pads)
            print('mat.domain.shifts=', mat.domain.shifts)
            print('mat._data (without ghost)=', mat._data[s:-s])

        comm.Barrier()

    gmat.setValuesStencil(mat._data[s:-s])


    print('Rank ', comm.Get_rank() if comm else '-', ': duration of setValuesCSR :', time.time()-t_prev)
    
    '''if comm:
        # Preallocate number of nonzeros per row
        #row_lengths = np.count_nonzero(rows[None,:] == np.unique(rows)[:,None], axis=1).max()
        
        ##very slow:
        #row_lengths = 0
        #for r in np.unique(rows):
        #    row_lengths = max(row_lengths, np.nonzero(rows==r)[0].size)
        
        row_lengths = np.unique(rows, return_counts=True)[1].max()

        # NNZ is the number of non-zeros per row for the local portion of the matrix
        t_prev = time.time() 
        NNZ = comm.allreduce(row_lengths, op=MPI.MAX)
        print('Rank ', comm.Get_rank() , ': duration of comm.allreduce :', time.time()-t_prev)

        t_prev = time.time() 
        gmat.setPreallocationNNZ(NNZ)
        print('Rank ', comm.Get_rank() , ': duration of setPreallocationNNZ :', time.time()-t_prev)

    t_prev = time.time() 
    # Fill-in matrix values
    for i in range(rows.size):
        # The values have to be set in "addition mode", otherwise the default just takes the new value.
        # This is here necessary, since the COO format can contain repeated entries.
        gmat.setValues(rows[i], cols[i], data[i])#, addv=PETSc.InsertMode.ADD_VALUES)

    print('Rank ', comm.Get_rank() , ': duration of setValues :', time.time()-t_prev)'''
  
    # Process inserted matrix entries
    ################################################
    # Note 12.03.2024:
    # In the assembly PETSc uses global communication to distribute the matrix in a different way than Psydac.
    # For this reason, at the moment we cannot compare directly each distributed 'chunck' of the Psydac and the PETSc matrices.
    # In the future we would like that PETSc uses the partition from Psydac, 
    # which might involve passing a DM Object.
    ################################################
    t_prev = time.time() 
    gmat.assemble()
    print('Rank ', comm.Get_rank() if comm else '-', ': duration of Mat assembly :', time.time()-t_prev)

    return gmat
