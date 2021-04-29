#coding = utf-8
from functools import reduce

import numpy as np
from scipy.sparse import kron

from psydac.linalg.basic   import LinearOperator, LinearSolver, Matrix
from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix

__all__ = ['KroneckerStencilMatrix',
           'KroneckerLinearSolver',
           'kronecker_solve_2d_par',
           'kronecker_solve_3d_par',
           'kronecker_solve']

##==============================================================================
#def kron_dot(starts, ends, pads, X, X_tmp, Y, A, B):
#    s1 = starts[0]
#    s2 = starts[1]
#    e1 = ends[0]
#    e2 = ends[1]
#    p1 = pads[0]
#    p2 = pads[1]

#    for j1 in range(s1-p1, e1+p1+1):
#        for i2 in range(s2, e2+1):
#             
#             X_tmp[j1+p1-s1, i2-s2+p2] = sum(X[j1+p1-s1, i2-s2+k]*B[i2,k] for k in range(2*p2+1))
#    
#    for i1 in range(s1, e1+1):
#        for i2 in range(s2, e2+1):
#             Y[i1-s1+p1,i2-s2+p2] = sum(A[i1, k]*X_tmp[i1-s1+k, i2-s2+p2] for k in range(2*p1+1))
#    return Y

##==============================================================================
#class KroneckerStencilMatrix_2D( Matrix ):

#    def __init__( self, V, W, A1, A2 ):

#        assert isinstance( V, StencilVectorSpace )
#        assert isinstance( W, StencilVectorSpace )
#        assert V is W
#        assert V.ndim == 2

#        assert isinstance( A1, StencilMatrix )
#        assert A1.domain.ndim == 1
#        assert A1.domain.npts[0] == V.npts[0]

#        assert isinstance( A2, StencilMatrix )
#        assert A2.domain.ndim == 1
#        assert A2.domain.npts[0] == V.npts[1]

#        self._space = V
#        self._A1    = A1
#        self._A2    = A2
#        self._w     = StencilVector( V )

#    #--------------------------------------
#    # Abstract interface
#    #--------------------------------------
#    @property
#    def domain( self ):
#        return self._space

#    # ...
#    @property
#    def codomain( self ):
#        return self._space

#    # ...
#    def dot( self, v, out=None ):

#        dot = np.dot

#        assert isinstance( v, StencilVector )
#        assert v.space is self.domain

#        if out is not None:
#            assert isinstance( out, StencilVector )
#            assert out.space is self.codomain
#        else:
#            out = StencilVector( self.codomain )

#        [s1, s2] = self._space.starts
#        [e1, e2] = self._space.ends
#        [p1, p2] = self._space.pads

#        A1 = self._A1
#        A2 = self._A2
#        w  = self._w

#        for j1 in range(s1-p1, e1+p1+1):
#            for i2 in range(s2, e2+1):
#                 w[j1,i2] = dot( v[j1,i2-p2:i2+p2+1], A2[i2,:])

#        for i1 in range(s1, e1+1):
#            for i2 in range(s2, e2+1):
#                 out[i1,i2] = dot( A1[i1,:], w[i1-p1:i1+p1+1,i2] )

#        out.update_ghost_regions()

#        return out

#    #--------------------------------------
#    # Other properties/methods
#    #--------------------------------------
#    @property
#    def starts( self ):
#        return self._space.starts

#    # ...
#    @property
#    def ends( self ):
#        return self._space.ends

#    # ...
#    @property
#    def pads( self ):
#        return self._space.pads

#    # ...
#    def __getitem__(self, key):
#        raise NotImplementedError('TODO')

#    # ...
#    def tosparse( self ):
#        raise NotImplementedError('TODO')

#    #...
#    def tocsr( self ):
#        return self.tosparse().tocsr()

#    #...
#    def toarray( self ):
#        return self.tosparse().toarray()

#    #...
#    def copy( self ):
#        M = KroneckerStencilMatrix_2D( self.domain, self.codomain, self._A1, self._A2 )
#        return M

#==============================================================================
class KroneckerStencilMatrix( Matrix ):
    """ Kronecker product of 1D stencil matrices.
    """

    def __init__( self,V, W, *args ):

        assert isinstance( V, StencilVectorSpace )
        assert isinstance( W, StencilVectorSpace )

        for i,A in enumerate(args):
            assert isinstance( A, Matrix )
            assert A.domain.ndim == 1
            assert A.domain.npts[0] == V.npts[i]

        self._domain   = V
        self._codomain = W
        self._mats     = args
        self._ndim     = len(args)

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def domain( self ):
        return self._domain

    # ...
    @property
    def codomain( self ):
        return self._codomain

    # ...
    @property
    def ndim( self ):
        return self._ndim
        
    # ...
    @property
    def mats( self ):
        return self._mats

    # ...
    def dot( self, x, out=None ):

        dot = np.dot

        assert isinstance( x, StencilVector )
        assert x.space is self.domain

        # Necessary if vector space is periodic or distributed across processes
        if not x.ghost_regions_in_sync:
            x.update_ghost_regions()

        if out is not None:
            assert isinstance( out, StencilVector )
            assert out.space is self.codomain
        else:
            out = StencilVector( self.codomain )

        starts = self._codomain.starts
        ends   = self._codomain.ends
        pads   = self._codomain.pads

        mats   = self.mats
        
        nrows  = tuple(e-s+1 for s,e in zip(starts, ends))
        pnrows = tuple(2*p+1 for p in pads)
        
        for ii in np.ndindex(*nrows):
            v = 0.
            xx = tuple(i+p for i,p in zip(ii, pads))

            for jj in np.ndindex(*pnrows):
                i_mats = [mat._data[s, j] for s,j,mat in zip(xx, jj, mats)]
                ii_jj = tuple(i+j for i,j in zip(ii, jj))
                v += x._data[ii_jj]*np.product(i_mats)

            out._data[xx] = v

        # IMPORTANT: flag that ghost regions are not up-to-date
        out.ghost_regions_in_sync = False
        return out

    # ...
    def copy(self):
        mats = [m.copy() for m in self.mats]
        return KroneckerStencilMatrix(self.domain, self.codomain, *mats)

    # ...
    def __neg__(self):
        mats = [-self.mats[0], *(m.copy() for m in self.mats[1:])]
        return KroneckerStencilMatrix(self.domain, self.codomain, *mats)

    # ...
    def __mul__(self, a):
        mats = [*(m.copy() for m in self.mats[:-1]), self.mats[-1] * a]
        return KroneckerStencilMatrix(self.domain, self.codomain, *mats)

    # ...
    def __rmul__(self, a):
        mats = [a * self.mats[0], *(m.copy() for m in self.mats[1:])]
        return KroneckerStencilMatrix(self.domain, self.codomain, *mats)

    # ...
    def __imul__(self, a):
        self.mats[-1] *= a
        return self

    # ...
    def __add__(self, m):
        raise NotImplementedError('Cannot sum Kronecker matrices')

    def __sub__(self, m):
        raise NotImplementedError('Cannot subtract Kronecker matrices')

    def __iadd__(self, m):
        raise NotImplementedError('Cannot sum Kronecker matrices')

    def __isub__(self, m):
        raise NotImplementedError('Cannot subtract Kronecker matrices')

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------

    def __getitem__(self, key):
        pads = self._codomain.pads
        rows = key[:self.ndim]
        cols = key[self.ndim:]
        mats = self.mats
        elements = [A[i,j] for A,i,j in zip(mats, rows, cols)]
        return np.product(elements)

    def tostencil(self):

        mats  = self.mats
        ssc   = self.codomain.starts
        eec   = self.codomain.ends
        ssd   = self.domain.starts
        eed   = self.domain.ends
        pads  = [A.pads[0] for A in self.mats]
        xpads = self.domain.pads

        # Number of rows in matrix (along each dimension)
        nrows       = [ed-s+1 for s,ed in zip(ssd, eed)]
        nrows_extra = [0 if ec<=ed else ec-ed for ec,ed in zip(eec,eed)]

        # create the stencil matrix
        M  = StencilMatrix(self.domain, self.codomain, pads=tuple(pads))

        mats = [mat._data for mat in mats]

        self._tostencil(M._data, mats, nrows, nrows_extra, pads, xpads)
        return M

    @staticmethod
    def _tostencil(M, mats, nrows, nrows_extra, pads, xpads):

        ndiags = [2*p + 1 for p in pads]
        diff   = [xp-p for xp,p in zip(xpads, pads)]
        ndim   = len(nrows)

        for xx in np.ndindex( *nrows ):

            ii = tuple(xp + x for xp, x in zip(xpads, xx) )

            for kk in np.ndindex( *ndiags ):

                values        = [mat[i,k] for mat,i,k in zip(mats, ii, kk)]
                M[(*ii, *kk)] = np.product(values)

    def tosparse(self):
        return reduce(kron, (m.tosparse() for m in self.mats))

    def toarray(self):
        return self.tosparse().toarray()

    def transpose(self):
        mats_tr = [Mi.transpose() for Mi in self.mats]
        return KroneckerStencilMatrix(self.codomain, self.domain, *mats_tr)

    @property
    def T(self):
        return self.transpose()

class KroneckerLinearSolver( LinearSolver ):
    """
    A solver for Ax=b, where A is a Kronecker matrix from arbirary dimension d,
    defined by d solvers. We also need information about the space of b.

    Parameters
    ----------
    V : StencilVectorSpace
        The space b will live in; i.e. which gives us information about the distribution of the right-hand sides.
    
    solvers : list of LinearSolver
        The components of A in each dimension.
    """
    def __init__(self, V, solvers):
        assert isinstance( V, StencilVectorSpace )
        assert hasattr( solvers, '__iter__' )
        for solver in solvers:
            assert isinstance( solver, LinearSolver  )

        assert V.ndim == len( solvers )

        # general arguments
        self._space = V
        self._solvers = solvers
        self._mpi_type = V._mpi_type
        self._parallel = self._space.parallel
        self._ndim = self._space.ndim

        # compute and setup solver arguments
        self._setup_solvers()

        # compute reordering permutations between the steps
        self._setup_permutations()

        # for now: allocate temporary arrays here (can be removed later)
        self._temp1, self._temp2 = self._allocate_temps()
    
    def _setup_solvers( self ):
        """
        Computes the distribution of elements and sets up the solvers (which potentially utilize MPI).
        """
        # slice sizes
        starts = np.array(self._space.starts)
        ends = np.array(self._space.ends) + 1
        self._slice = tuple([slice(s, e) for s,e in zip(starts, ends)])

        # local and global sizes
        nglobals = self._space.npts
        nlocals = ends - starts
        self._localsize = np.product(nlocals)
        mglobals = self._localsize // nlocals
        self._nlocals = nlocals

        # solver passes (and mlocal size)
        solver_passes = [None] * self._ndim

        tempsize = self._localsize
        self._allserial = True
        for i in range(self._ndim):
            # decide for each direction individually, if we should use a serial or a parallel/distributed sovler
            # useful e.g. if we have little data in some directions (and thus no data distributed there)

            if not self._parallel or self._space.cart.subcomm[i].size <= 1:
                # serial solve
                solver_passes[i] = KroneckerSolverSerialPass(self._solvers[i], nglobals[i], mglobals[i])
            else:
                # TODO: also implement a pass using Alltoall, in case that the data is regular enough
                # for the parallel case, use Alltoallv
                solver_passes[i] = KroneckerSolverParallelPass(self._solvers[i], self._space._mpi_type, i, self._space.cart, mglobals[i], nglobals[i], nlocals[i], self._localsize)

                # we have a parallel solve pass now, so we are not completely local any more
                self._allserial = False
            
            # update memory requirements
            tempsize = max(tempsize, solver_passes[i].required_memory())
        
        # we want to start with the last dimension
        self._solver_passes = list(reversed(solver_passes))
        self._tempsize = tempsize

    def _setup_permutations(self):
        # permutation and shape for reordering

        # we want for the permutations:
        # a) re-order as little as possible
        # b) all permutations should, when applied in the right order, lead to 
        #
        # so we can do:
        # if we have (1,...,n) in the beginning, then do:
        # (1,...,n,n-1) first (i.e. swap -1th with -2nd component)
        # (1,...,n,n-1,n-2) second (i.e. swap -1th with -3rd component)
        # (1,...,n,n-2,n-1,n3) third
        # until we get to (n, 2, ..., n-1, 1)
        # combining all these permutations, we have: (2,3,...,n-1,n,1)
        # the inverse of this last permutation is (n,1,2,...,n-1)

        # this way, we avoid too large strides
        self._perm = [None] * self._ndim
        for i in range(self._ndim - 1):
            # permutation which swaps -i-2 with -1
            self._perm[i] = np.arange(self._ndim)
            self._perm[i][-i-2], self._perm[i][-1] = self._perm[i][-1], self._perm[i][-i-2]
        # last permutation
        self._perm[-1] = np.arange(self._ndim)
        self._perm[-1][1:] = self._perm[-1][:-1]
        self._perm[-1][0] = self._ndim - 1

        # re-order the shapes based on the permutations
        self._shapes = [None] * self._ndim
        self._shapes[0] = self._nlocals
        for i in range(1, self._ndim):
            self._shapes[i] = self._shapes[i-1][self._perm[i-1]]
    
    def _allocate_temps( self ):
        """
        Allocates all temporary data needed for the solve operation.
        """
        temp1 = np.empty((self._tempsize,))
        if self._ndim <= 1 and self._allserial:
            # if ndim==1 and we have no parallelism, we can avoid allocating a second temp array
            temp2 = None
        else:
            temp2 = np.empty((self._tempsize,))
        return temp1, temp2
    
    @property
    def space( self ):
        """
        Returns the space associated to this solver (i.e. where the information about the cartesian distribution is taken from).
        """
        return self._space

    def solve( self, rhs, out=None, transposed=False ):
        """
        Solves Ax=b where A is a Kronecker product matrix (and represented as such),
        and b is a suitable vector.
        """

        # type checks
        assert rhs.space is self._space

        if out is not None:
            assert isinstance( out, StencilVector )
            assert out.space is self._space
        else:
            out = StencilVector( rhs.space )
        
        inslice = rhs[self._slice]
        outslice = out[self._slice]

        # call the actual kernel
        self._solve_nd(inslice, outslice, transposed)
        
        out.update_ghost_regions()
        return out
 
    def _solve_nd(self, inslice, outslice, transposed):
        """
        The internal solve loop. Can handle arbitrary dimensions.
        """
        temp1 = self._temp1
        temp2 = self._temp2

        # copy input
        self._inslice_to_temp(inslice, temp1)

        # internal passes
        for i in range(self._ndim - 1):
            # solve direction
            self._solver_passes[i].solve_pass(temp1, temp2, transposed)

            # reorder and swap
            self._reorder_temp_to_temp(temp1, temp2, i)
            temp1, temp2 = temp2, temp1
        
        # last pass
        self._solver_passes[-1].solve_pass(temp1, temp2, transposed)

        # copy to output
        self._reorder_temp_to_outslice(temp1, outslice)

    def _inslice_to_temp(self, inslice, target):
        """
        Copies data to an internal, 1-dimensional temporary array.
        """
        targetview = target[:self._localsize]
        targetview.shape = inslice.shape

        targetview[:] = inslice
    
    def _reorder_temp_to_temp(self, source, target, i):
        """
        Reorders the dimensions of the temporary arrays, and copies data from one to another.
        """
        sourceview = source[:self._localsize]
        sourceview.shape = self._shapes[i]

        targetview = target[:self._localsize]
        targetview.shape = self._shapes[i+1]

        targetview[:] = sourceview.transpose(self._perm[i])
    
    def _reorder_temp_to_outslice(self, source, outslice):
        """
        Reorders the dimensions of the temporary array for a final time, and copies it to the output.
        """
        sourceview = source[:self._localsize]
        sourceview.shape = self._shapes[-1]

        outslice[:] = sourceview.transpose(self._perm[-1])

class KroneckerSolverSerialPass:
    """
    Solves several linear equations at the same time, given that the data is already in memory.

    Not intended for outside use.
    """
    def __init__(self, solver, nglobal, mglobal):
        self._numrhs = mglobal
        self._dimrhs = nglobal
        self._datasize = nglobal*mglobal
        self._solver = solver
        self._view = None
    
    def required_memory(self):
        return self._datasize

    def solve_pass(self, workmem, tempmem, transposed):
        # reshape necessary memory in column-major
        view = workmem[:self._datasize]
        view.shape = (self._numrhs,self._dimrhs)

        # the solvers want the FORTRAN-contiguous format
        # (TODO: push this into the DirectSolver?)
        view_T = view.transpose()

        # call solver in in-place mode
        self._solver.solve(view_T, out=view_T, transposed=transposed)

class KroneckerSolverParallelPass:
    """
    Solves several linear equations at the same time, using an Alltoallv operation to distribute the data.

    Not intended for outside use.
    """

    # To understand the following, here is a short explaination. Consider two processes like this:
    #
    # Pr1 | Pr2
    # 0 1 | 2 3
    # 4 5 | 6 7
    # 8 9 | A B 
    # C D | E F
    #
    # i.e. Pr1 has 0 1 4 5 8 9 C D; Pr2 has 2 3 6 7 A B E F
    #
    # We now would like to get each line on at least one process. So, we do an AlltoAll like this:
    #
    # Pr1 | Pr2
    # 0 1 | 2 3 | to Pr1
    # 4 5 | 6 7 | to Pr1
    # ------------------
    # 8 9 | A B | to Pr2
    # C D | E F | to Pr2
    #
    # But the data is transported per process, i.e. we get in this order:
    # 0 1 4 5 2 3 6 7 on Pr1
    # 8 9 C D A B E F on Pr2
    #
    # so we still need to re-order (i.e. partially transpose) locally to finally get what we want.
    # 0 1 2 3 4 5 6 7 on Pr1
    # 8 9 A B C D E F on Pr2
    #
    
    def __init__(self, solver, mpi_type, i, cart, mglobal, nglobal, nlocal, localsize):
        self._nglobal = nglobal

        # cartesian distribution
        comm = cart.subcomm[i]
        cartend = cart.global_ends[i] + 1
        cartstart = cart.global_starts[i]
        cartsize = cartend - cartstart

        # source MPI
        mlocal_pre = mglobal // comm.size
        mlocal_add = mglobal % comm.size
        sourcesizes = np.full((comm.size,), mlocal_pre, dtype=int)
        sourcesizes[:mlocal_add] += 1
        mlocal = sourcesizes[comm.rank]
        sourcesizes *= nlocal

        sourcedisps = np.zeros((comm.size+1,), dtype=int)
        np.cumsum(sourcesizes, out=sourcedisps[1:])
        sourcedisps = sourcedisps[:-1]

        # target MPI (mlocal is the same over all processes in the communicator)
        targetsizes = cartsize * mlocal
        targetdisps = cartstart * mlocal

        # setting all arguments to keep
        self._mlocal = mlocal
        self._localsize = localsize
        self._datasize = mlocal * nglobal
        self._source_transfer = (sourcesizes, sourcedisps)
        self._target_transfer = (targetsizes, targetdisps)
        self._mpi_type = mpi_type
        self._cartstart = cartstart
        self._cartend = cartend
        self._comm = comm
        self._serialsolver = KroneckerSolverSerialPass(solver, nglobal, mlocal)

    def required_memory(self):
        return max(self._datasize, self._localsize)

    def _order_blocked(self, source, target):
        blocked_view = source[:self._datasize]
        blocked_view.shape = (self._mlocal,self._nglobal)
        for start, end in zip(self._cartstart, self._cartend):
            targetpart = target[start*self._mlocal:end*self._mlocal]
            targetpart.shape = (self._mlocal,end-start)
            blocked_view[:,start:end] = targetpart
    
    def _unorder_blocked(self, source, target):
        blocked_view = source[:self._datasize]
        blocked_view.shape = (self._mlocal,self._nglobal)
        for start, end in zip(self._cartstart, self._cartend):
            targetpart = target[start*self._mlocal:end*self._mlocal]
            targetpart.shape = (self._mlocal,end-start)
            targetpart[:] = blocked_view[:,start:end]

    def solve_pass(self, workmem, tempmem, transposed):
        # preparation
        sourceargs = [workmem[:self._localsize], self._source_transfer, self._mpi_type]
        targetargs = [tempmem[:self._datasize], self._target_transfer, self._mpi_type]

        # parts of stripes -> blocked stripes
        self._comm.Alltoallv(sourceargs, targetargs)

        # blocked stripes -> ordered stripes
        self._order_blocked(workmem, tempmem)

        # actual solve (source contains the data)
        self._serialsolver.solve_pass(workmem, tempmem, transposed)

        # ordered stripes -> blocked stripes
        self._unorder_blocked(workmem, tempmem)

        # blocked stripes -> parts of stripes
        self._comm.Alltoallv(targetargs, sourceargs)

#==============================================================================
def kronecker_solve_2d_par( A1, A2, rhs, out=None ):
    """
    Solve linear system Ax=b with A=kron(A2,A1).

    Parameters
    ----------
    A1 : LinearSolver
        Solve linear system A1 x1 = b1: x1=A1.solve(b1).

    A2 : LinearSolver
        Solve linear system A2 x2 = b2: x2=A2.solve(b2).

    rhs : StencilVector 2D
        Right hand side vector of linear system Ax=b.

    """
    assert isinstance( A1 , LinearSolver  )
    assert isinstance( A2 , LinearSolver  )
    assert isinstance( rhs, StencilVector )

    if out is not None:
        assert isinstance( out, StencilVector )
        assert out.space is rhs.space
    else:
        out = StencilVector( rhs.space )

    # ...
    V = rhs.space

    s1, s2 = V.starts
    e1, e2 = V.ends
    n1, n2 = V.npts

    subcomm_1 = V.cart.subcomm[0]
    subcomm_2 = V.cart.subcomm[1]

    disps1 = V.cart.global_starts[0]
    disps2 = V.cart.global_starts[1]

    sizes1 = V.cart.global_ends[0] - V.cart.global_starts[0] + 1
    sizes2 = V.cart.global_ends[1] - V.cart.global_starts[1] + 1

    # TODO: make MPI type available through property
    mpi_type = V._mpi_type
    # ...

    # ...
    # 2D slices
    X = rhs[s1:e1+1,s2:e2+1]
    Y = out[s1:e1+1,s2:e2+1]

    # 1D global arrays
    X_glob_1 = np.zeros( n1 )
    Y_glob_2 = np.zeros( n2 )
    # ...

    for i2 in range(e2-s2+1):
        X_loc = X[:,i2].copy()  # need 1D contiguous copy
        subcomm_1.Allgatherv( X_loc, [X_glob_1, sizes1, disps1, mpi_type] )
        Y[:,i2] = A1.solve( X_glob_1 )[s1:e1+1]

    for i1 in range(e1-s1+1):
        Y_loc = Y[i1,:]  # 1D contiguous slice
        subcomm_2.Allgatherv( Y_loc, [Y_glob_2, sizes2, disps2, mpi_type] )
        Y[i1,:] = A2.solve( Y_glob_2 )[s2:e2+1]

    # ...
    out.update_ghost_regions()
    # ...

    return out

#==============================================================================
def kronecker_solve_3d_par_old( A1, A2, A3, rhs, out=None ):
    """
    Solve linear system Ax=b with A=kron(A3,A2,A1).

    Parameters
    ----------
    A1 : LinearSolver
        Solve linear system A1 x1 = b1: x1=A1.solve(b1).

    A2 : LinearSolver
        Solve linear system A2 x2 = b2: x2=A2.solve(b2).

    A3 : LinearSolver
        Solve linear system A3 x3 = b3: x3=A3.solve(b3).

    rhs : StencilVector 3D
        Right hand side vector of linear system Ax=b.

    """
    assert isinstance( A1 , LinearSolver  )
    assert isinstance( A2 , LinearSolver  )
    assert isinstance( A3 , LinearSolver  )
    assert isinstance( rhs, StencilVector )

    if out is not None:
        assert isinstance( out, StencilVector )
        assert out.space is rhs.space
    else:
        out = StencilVector( rhs.space )

    # ...
    V = rhs.space

    s1, s2, s3 = V.starts
    e1, e2, e3 = V.ends
    n1, n2, n3 = V.npts

    subcomm_1 = V.cart.subcomm[0]
    subcomm_2 = V.cart.subcomm[1]
    subcomm_3 = V.cart.subcomm[2]

    disps1 = V.cart.global_starts[0]
    disps2 = V.cart.global_starts[1]
    disps3 = V.cart.global_starts[2]

    sizes1 = V.cart.global_ends[0] - V.cart.global_starts[0] + 1
    sizes2 = V.cart.global_ends[1] - V.cart.global_starts[1] + 1
    sizes3 = V.cart.global_ends[2] - V.cart.global_starts[2] + 1

    # TODO: make MPI type available through property
    mpi_type = V._mpi_type
    # ...

    # ...
    # 3D slices
    X = rhs[s1:e1+1, s2:e2+1, s3:e3+1]
    Y = out[s1:e1+1, s2:e2+1, s3:e3+1]

    # 1D local arrays
    X_loc_1 = np.zeros(X.shape[0])
    Y_loc_2 = np.zeros(Y.shape[1])
    # Y_loc_3 is not needed

    # 1D global arrays
    X_glob_1 = np.zeros( n1 )
    Y_glob_2 = np.zeros( n2 )
    Y_glob_3 = np.zeros( n3 )
    # ...

    for i3 in range(e3-s3+1):
        for i2 in range(e2-s2+1):
            X_loc_1[:] = X[:, i2, i3]  # need 1D contiguous copy
            subcomm_1.Allgatherv( X_loc_1, [X_glob_1, sizes1, disps1, mpi_type] )
            Y[:, i2, i3] = A1.solve( X_glob_1, out=X_glob_1 )[s1:e1+1]

    for i3 in range(e3-s3+1):
        for i1 in range(e1-s1+1):
            Y_loc_2[:] = Y[i1, :, i3]  # need 1D contiguous copy
            subcomm_2.Allgatherv( Y_loc_2, [Y_glob_2, sizes2, disps2, mpi_type] )
            Y[i1, :, i3] = A2.solve( Y_glob_2, out=Y_glob_2 )[s2:e2+1]

    for i2 in range(e2-s2+1):
        for i1 in range(e1-s1+1):
            Y_loc_3 = Y[i1, i2, :]  # 1D contiguous slice
            subcomm_3.Allgatherv( Y_loc_3, [Y_glob_3, sizes3, disps3, mpi_type] )
            Y[i1, i2, :] = A3.solve( Y_glob_3, out=Y_glob_3 )[s3:e3+1]

    # ...
    out.update_ghost_regions()
    # ...

    return out

def kronecker_solve_3d_par( A1, A2, A3, rhs, out=None ):
    """
    Solve linear system Ax=b with A=kron(A3,A2,A1).

    Soon to be replaced by the KroneckerLinearSolver from above.

    Parameters
    ----------
    A1 : LinearSolver
        Solve linear system A1 x1 = b1: x1=A1.solve(b1).

    A2 : LinearSolver
        Solve linear system A2 x2 = b2: x2=A2.solve(b2).

    A3 : LinearSolver
        Solve linear system A3 x3 = b3: x3=A3.solve(b3).

    rhs : StencilVector 3D
        Right hand side vector of linear system Ax=b.

    """
    assert isinstance( A1 , LinearSolver  )
    assert isinstance( A2 , LinearSolver  )
    assert isinstance( A3 , LinearSolver  )
    assert isinstance( rhs, StencilVector )

    if out is not None:
        assert isinstance( out, StencilVector )
        assert out.space is rhs.space
    else:
        out = StencilVector( rhs.space )

    # ...
    V = rhs.space

    s1, s2, s3 = V.starts
    e1, e2, e3 = V.ends
    n1, n2, n3 = V.npts

    subcomm_1 = V.cart.subcomm[0]
    subcomm_2 = V.cart.subcomm[1]
    subcomm_3 = V.cart.subcomm[2]

    disps1 = V.cart.global_starts[0]
    disps2 = V.cart.global_starts[1]
    disps3 = V.cart.global_starts[2]

    sizes1 = V.cart.global_ends[0] - V.cart.global_starts[0] + 1
    sizes2 = V.cart.global_ends[1] - V.cart.global_starts[1] + 1
    sizes3 = V.cart.global_ends[2] - V.cart.global_starts[2] + 1

    # TODO: make MPI type available through property
    mpi_type = V._mpi_type
    # ...

    # ...
    # 3D slices
    X = rhs[s1:e1+1, s2:e2+1, s3:e3+1]
    Y = out[s1:e1+1, s2:e2+1, s3:e3+1]

    # local chunk size
    l1 = e1-s1+1
    l2 = e2-s2+1
    l3 = e3-s3+1

    # slice size (i.e. divide local chunk by communicator size again)
    def roundcomm(num, comm):
        return num // comm.size + (comm.rank < (num % comm.size))
    s1 = roundcomm(l2*l3, subcomm_1)
    s2 = roundcomm(l1*l3, subcomm_2)
    s3 = roundcomm(l1*l2, subcomm_3)

    def makedists(source, target, comm, keepd, shortd, longd, cartdisps, cartsizes):
        # creates the sizes and disps arrays used for MPI. We need a new one for each direction

        # source (parts of stripes)
        share = keepd // comm.size # keepd == l1*l2; locald == l1*l2/commsize; shortd == l3; longd == n3
        addmax = keepd % comm.size
        sourcesizes = np.full((comm.size,), share, dtype=int)
        sourcesizes[:addmax] += 1
        # keepd=7, comm.size=3
        # 2+1 2 2
        locald = sourcesizes[comm.rank]
        sourcesizes *= shortd
        sourcedisps = np.zeros((comm.size+1,), dtype=int)
        np.cumsum(sourcesizes, out=sourcedisps[1:])
        size = sourcedisps[-1]
        sourcedisps = sourcedisps[:-1]

        # target (blocked stripes)
        targetsizes = np.array(cartsizes) * locald
        targetdisps = np.array(cartdisps) * locald

        return [source[:size], (sourcesizes, sourcedisps), mpi_type], [target[:longd*locald], (targetsizes, targetdisps), mpi_type], locald
    
    def solve_onedir_serial(target, solver, nrhs, dimrhs):
        # reshape and call solver
        view = target[:dimrhs*nrhs].reshape((dimrhs,nrhs), order='F')
        solver.solve(view, out=view)

    # does the MPI distribute, solve, and distribute back for one dimension
    def solve_onedir_par(source, target, solver, comm, keepd, shortd, longd, cartdisps, cartsizes):
        sourceargs, targetargs, locald = makedists(source, target, comm, keepd, shortd, longd, cartdisps, cartsizes)

        # parts of stripes -> blocked stripes
        comm.Alltoallv(sourceargs, targetargs)

        # blocked stripes -> ordered stripes
        preview = source[:longd*locald].reshape((locald,longd))
        for disp, size in zip(cartdisps, cartsizes):
            preview[:,disp:(disp+size)] = target[disp*locald:(disp+size)*locald].reshape((locald,size))

        # actual solve
        solve_onedir_serial(source, solver, locald, longd)

        # ordered stripes -> blocked stripes
        preview = source[:longd*locald].reshape((locald,longd))
        for disp, size in zip(cartdisps, cartsizes):
            target[disp*locald:(disp+size)*locald] = preview[:,disp:(disp+size)].flat

        # blocked stripes -> parts of stripes
        comm.Alltoallv(targetargs, sourceargs)

    # we need this to avoid the padding (or shall we send the padding as well?)
    tempsize = max(l1*l2*l3,s1*n1,s2*n2,s3*n3)
    temp1 = np.empty(tempsize)
    temp2 = np.empty(tempsize)

    # TODO check array order...

    # (1,2,3)
    temp1[:l1*l2*l3] = X.flat

    solve_onedir_par(temp1, temp2, A3, subcomm_3, l1*l2, l3, n3, disps3, sizes3)

    # (1,2,3) -> (3,1,2)
    temp2[:l1*l2*l3] = temp1[:l1*l2*l3].reshape((l1,l2,l3)).transpose((2,0,1)).flat

    solve_onedir_par(temp2, temp1, A2, subcomm_2, l1*l3, l2, n2, disps2, sizes2)

    # (3,1,2) -> (2,3,1)
    temp1[:l1*l2*l3] = temp2[:l1*l2*l3].reshape((l3,l1,l2)).transpose((2,0,1)).flat

    solve_onedir_par(temp1, temp2, A1, subcomm_1, l2*l3, l1, n1, disps1, sizes1)

    # (2,3,1) -> (1,2,3)
    Y[:] = temp1[:l1*l2*l3].reshape((l2,l3,l1)).transpose((2,0,1))

    # ...
    out.update_ghost_regions()
    # ...

    return out

#==============================================================================
def kronecker_solve( solvers, rhs, out=None ):
    """
    Solve linear system Ax=b with A=kron( A_n, A_{n-1}, ..., A_2, A_1 ), given
    $n$ separate linear solvers $L_n$ for the 1D problems $A_n x_n = b_n$:

    x_n = L_n.solve( b_n )

    Parameters
    ----------
    solvers : list( LinearSolver )
        List of linear solvers along each direction: [L_1, L_2, ..., L_n].

    rhs : StencilVector
        Right hand side vector of linear system Ax=b.

    """
    assert hasattr( solvers, '__iter__' )
    for solver in solvers:
        assert isinstance( solver, LinearSolver  )

    assert isinstance( rhs, StencilVector )
    assert rhs.space.ndim == len( solvers )

    if out is not None:
        assert isinstance( out, StencilVector )
        assert out.space is rhs.space
    else:
        out = StencilVector( rhs.space )

    space = rhs.space

    kronsolver = KroneckerLinearSolver(space, solvers)
    kronsolver.solve(rhs, out=out)
    return out

