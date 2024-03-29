# coding: utf-8

import numpy as np
from math import sqrt

from psydac.linalg.stencil import StencilVectorSpace, StencilVector
from psydac.linalg.block   import BlockVector, BlockVectorSpace

__all__ = (
    'array_to_psydac',
    'petsc_to_psydac',
    '_sym_ortho'
)

#==============================================================================
def array_to_psydac(x, Xh):
    """ converts a numpy array to StencilVector or BlockVector format"""

    if isinstance(Xh, BlockVectorSpace):
        u = BlockVector(Xh)
        if isinstance(Xh.spaces[0], BlockVectorSpace):
            for d in range(len(Xh.spaces)):
                starts = [np.array(V.starts) for V in Xh.spaces[d].spaces]
                ends   = [np.array(V.ends)   for V in Xh.spaces[d].spaces]

                for i in range(len(starts)):
                    g = tuple(slice(s,e+1) for s,e in zip(starts[i], ends[i]))
                    shape = tuple(ends[i]-starts[i]+1)
                    u[d][i][g] = x[:np.prod(shape) ].reshape(shape)
                    x          = x[ np.prod(shape):]

        else:
            starts = [np.array(V.starts) for V in Xh.spaces]
            ends   = [np.array(V.ends)   for V in Xh.spaces]

            for i in range(len(starts)):
                g = tuple(slice(s,e+1) for s,e in zip(starts[i], ends[i]))
                shape = tuple(ends[i]-starts[i]+1)
                u[i][g] = x[:np.prod(shape) ].reshape(shape)
                x       = x[ np.prod(shape):]

    elif isinstance(Xh, StencilVectorSpace):

        u =  StencilVector(Xh)
        starts = np.array(Xh.starts)
        ends   = np.array(Xh.ends)
        g = tuple(slice(s, e+1) for s,e in zip(starts, ends))
        shape = tuple(ends-starts+1)
        u[g] = x[:np.prod(shape)].reshape(shape)
    else:
        raise ValueError('Xh must be a StencilVectorSpace or a BlockVectorSpace')

    u.update_ghost_regions()
    return u

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
        if isinstance(Xh.spaces[0], BlockVectorSpace):

            comm       = u[0][0].space.cart.global_comm
            dtype      = u[0][0].space.dtype
            sendcounts = np.array(comm.allgather(len(x.array))) if comm else np.array([len(x.array)])
            recvbuf    = np.empty(sum(sendcounts), dtype='complex') # PETSc installed with complex configuration only handles complex vectors

            if comm:
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

                inds += np.prod(npts)

    elif isinstance(Xh, StencilVectorSpace):

        u          = StencilVector(Xh)
        comm       = u.space.cart.global_comm
        dtype      = u.space.dtype
        sendcounts = np.array(comm.allgather(len(x.array))) if comm else np.array([len(x.array)])
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
        u._data[idx] = (vals if dtype is complex else vals.real).reshape(shape)

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
