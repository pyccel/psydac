# coding: utf-8

import numpy as np
from math import sqrt

from psydac.linalg.basic   import Vector
from psydac.linalg.stencil import StencilVectorSpace, StencilVector
from psydac.linalg.block   import BlockVector, BlockVectorSpace
from psydac.linalg.topetsc import psydac_to_petsc_local, get_petsc_local_to_global_shift, petsc_to_psydac_local, global_to_psydac, get_npts_per_block

__all__ = (
    'array_to_psydac',
    'petsc_to_psydac',
    '_sym_ortho'
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
def petsc_to_psydac(x, Xh):
    """Convert a PETSc.Vec object to a StencilVector or BlockVector. It assumes that PETSc was installed with the configuration for complex numbers.
        We gather the petsc global vector in all the processes and extract the chunk owned by the Psydac Vector.
        .. warning: This function will not work if the global vector does not fit in the process memory.

    Parameters
    ----------
    x : PETSc.Vec
      PETSc vector

    Returns
    -------
    u : psydac.linalg.stencil.StencilVector | psydac.linalg.block.BlockVector
        Psydac vector
    """
    
    if isinstance(Xh, BlockVectorSpace):
        u = BlockVector(Xh)

        comm       = x.comm#u[0][0].space.cart.global_comm
        dtype      = Xh._dtype#u[0][0].space.dtype
        n_blocks   = Xh.n_blocks
        #sendcounts = np.array(comm.allgather(len(x.array))) if comm else np.array([len(x.array)])
        #recvbuf    = np.empty(sum(sendcounts), dtype='complex') # PETSc installed with complex configuration only handles complex vectors
        localsize, globalsize = x.getSizes()
        #assert globalsize == u.shape[0], 'Sizes of global vectors do not match'


        # Find shifts for process k:
        npts_local_per_block_per_process = np.array(get_npts_per_block(Xh)) #indexed [b,k,d] for block b and process k and dimension d
        local_sizes_per_block_per_process = np.prod(npts_local_per_block_per_process, axis=-1) #indexed [b,k] for block b and process k

        index_shift = 0 + np.sum(local_sizes_per_block_per_process[:,:comm.Get_rank()]) #+ np.sum(local_sizes_per_block_per_process[:b,comm.Get_rank()]) for b in range(n_blocks)]

        #index_shift_per_block =  [0 + np.sum(local_sizes_per_block_per_process[b][0:x.comm.Get_rank()], dtype=int) for b in range(n_blocks)] #Global variable

        print(f'rk={comm.Get_rank()}, local_sizes_per_block_per_process={local_sizes_per_block_per_process}, index_shift={index_shift}, u[0]._data={u[0]._data.shape}, u[1]._data={u[1]._data.shape}')


        local_petsc_indices = np.arange(localsize)
        global_petsc_indices = []
        psydac_indices = []
        block_indices = []
        for petsc_index in local_petsc_indices:

            block_index, psydac_index = global_to_psydac(Xh, petsc_index)#, comm=x.comm)            
            psydac_indices.append(psydac_index)
            block_indices.append(block_index)


            global_petsc_indices.append(petsc_index + index_shift)

            print(f'rank={comm.Get_rank()}, psydac_index = {psydac_index}, local_petsc_index={petsc_index}, petsc_global_index={global_petsc_indices[-1]}')

        
        for block_index, psydac_index, petsc_index in zip(block_indices, psydac_indices, global_petsc_indices):
            value = x.getValue(petsc_index) # Global index
            if value != 0:
                u[block_index[0]]._data[psydac_index] = value if dtype is complex else value.real
        



        '''if comm:
            # Gather the global array in all the processors
            ################################################
            # Note 12.03.2024:
            # This global communication is at the moment necessary since PETSc distributes matrices and vectors different than Psydac. 
            # In order to avoid it, we would need that PETSc uses the partition from Psydac, 
            # which might involve passing a DM Object.
            ################################################
            comm.Allgatherv(sendbuf=x.array, recvbuf=(recvbuf, sendcounts))
        else:
            recvbuf[:] = x.array

        inds = 0
        for d in range(len(Xh.spaces)):
            starts = [np.array(V.starts) for V in Xh.spaces[d].spaces]
            ends   = [np.array(V.ends)   for V in Xh.spaces[d].spaces]

            for i in range(len(starts)):
                idx = tuple( slice(m*p,-m*p) for m,p in zip(u.space.spaces[d].spaces[i].pads, u.space.spaces[d].spaces[i].shifts) )
                shape = tuple(ends[i]-starts[i]+1)
                npts  = Xh.spaces[d].spaces[i].npts
                # compute the global indices of the coefficents owned by the process using starts and ends
                indices = np.array([np.ravel_multi_index( [s+x for s,x in zip(starts[i], xx)], dims=npts,  order='C' ) for xx in np.ndindex(*shape)] )
                vals = recvbuf[indices+inds]

                # With PETSc installation configuration for complex, all the numbers are by default complex. 
                # In the float case, the imaginary part must be truncated to avoid warnings.
                u[d][i]._data[idx] = (vals if dtype is complex else vals.real).reshape(shape)

                inds += np.prod(npts)

        else:
        comm       = u[0].space.cart.global_comm
        dtype      = u[0].space.dtype
        sendcounts = np.array(comm.allgather(len(x.array))) if comm else np.array([len(x.array)])
        recvbuf    = np.empty(sum(sendcounts), dtype='complex') # PETSc installed with complex configuration only handles complex vectors

        if comm:
            # Gather the global array in all the procs
            # TODO: Avoid this global communication with a DM Object (see note above).
            comm.Allgatherv(sendbuf=x.array, recvbuf=(recvbuf, sendcounts))
        else:
            recvbuf[:] = x.array

        inds = 0
        starts = [np.array(V.starts) for V in Xh.spaces]
        ends   = [np.array(V.ends)   for V in Xh.spaces]
        for i in range(len(starts)):
            idx = tuple( slice(m*p,-m*p) for m,p in zip(u.space.spaces[i].pads, u.space.spaces[i].shifts) )
            shape = tuple(ends[i]-starts[i]+1)
            npts  = Xh.spaces[i].npts
            # compute the global indices of the coefficents owned by the process using starts and ends
            indices = np.array([np.ravel_multi_index( [s+x for s,x in zip(starts[i], xx)], dims=npts,  order='C' ) for xx in np.ndindex(*shape)] )
            vals = recvbuf[indices+inds]

            # With PETSc installation configuration for complex, all the numbers are by default complex. 
            # In the float case, the imaginary part must be truncated to avoid warnings.
            u[i]._data[idx] = (vals if dtype is complex else vals.real).reshape(shape)

            inds += np.prod(npts)'''

    elif isinstance(Xh, StencilVectorSpace):

        u          = StencilVector(Xh)
        comm       = u.space.cart.global_comm
        dtype      = u.space.dtype
        localsize, globalsize = x.getSizes()
        assert globalsize == u.shape[0], 'Sizes of global vectors do not match'

        '''index_shift = get_petsc_local_to_global_shift(Xh)
        petsc_local_indices = np.arange(localsize) 
        petsc_indices = petsc_local_indices #+ index_shift
        psydac_indices = [petsc_to_psydac_local(Xh, petsc_index) for petsc_index in petsc_indices]'''


        # Find shifts for process k:
        npts_local_per_block_per_process = np.array(get_npts_per_block(Xh)) #indexed [b,k,d] for block b and process k and dimension d
        local_sizes_per_block_per_process = np.prod(npts_local_per_block_per_process, axis=-1) #indexed [b,k] for block b and process k

        index_shift = 0 + np.sum(local_sizes_per_block_per_process[0][0:x.comm.Get_rank()], dtype=int) #Global variable


        
        local_petsc_indices = np.arange(localsize)
        global_petsc_indices = []
        psydac_indices = []
        block_indices = []
        for petsc_index in local_petsc_indices:

            block_index, psydac_index = global_to_psydac(Xh, petsc_index)#, comm=x.comm)            
            psydac_indices.append(psydac_index)
            block_indices.append(block_index)
            global_petsc_indices.append(petsc_index + index_shift)




        #psydac_indices = [global_to_psydac(Xh, petsc_index, comm=x.comm) for petsc_index in petsc_indices]


        '''if comm is not None:
            for k in range(comm.Get_size()):
                if k == comm.Get_rank():
                    print('\nRank ', k)
                    print('petsc_indices=\n', petsc_indices)
                    print('psydac_indices=\n', psydac_indices)
                    print('index_shift=', index_shift)
                comm.Barrier()'''

        for block_index, psydac_index, petsc_index in zip(block_indices, psydac_indices, global_petsc_indices):
            value = x.getValue(petsc_index) # Global index
            if value != 0:
                u._data[psydac_index] = value if dtype is complex else value.real
        
        '''sendcounts = np.array(comm.allgather(len(x.array))) if comm else np.array([len(x.array)])
        recvbuf    = np.empty(sum(sendcounts), dtype='complex') # PETSc installed with complex configuration only handles complex vectors 

        if comm:
            # Gather the global array in all the procs
            # TODO: Avoid this global communication with a DM Object (see note above).
            comm.Allgatherv(sendbuf=x.array, recvbuf=(recvbuf, sendcounts))
        else:
            recvbuf[:] = x.array

        # compute the global indices of the coefficents owned by the process using starts and ends
        starts = np.array(Xh.starts)
        ends   = np.array(Xh.ends)
        shape  = tuple(ends-starts+1)
        npts   = Xh.npts
        indices = np.array([np.ravel_multi_index( [s+x for s,x in zip(starts, xx)], dims=npts,  order='C' ) for xx in np.ndindex(*shape)] )
        idx = tuple( slice(m*p,-m*p) for m,p in zip(u.space.pads, u.space.shifts) )
        vals = recvbuf[indices]

        # With PETSc installation configuration for complex, all the numbers are by default complex. 
        # In the float case, the imaginary part must be truncated to avoid warnings.
        u._data[idx] = (vals if dtype is complex else vals.real).reshape(shape)'''

    else:
        raise ValueError('Xh must be a StencilVectorSpace or a BlockVectorSpace')

    u.update_ghost_regions()

    '''if comm is not None:
        u_arr = u.toarray()
        x_arr = x.array.real
        for k in range(comm.Get_size()):
            if k == comm.Get_rank():
                print('\nRank ', k)
                print('u.toarray()=\n', u_arr)
                #print('x.array=\n', x_arr)
                #print('u._data=\n', u._data)

            comm.Barrier()'''

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
