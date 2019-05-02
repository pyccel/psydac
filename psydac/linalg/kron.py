#coding = utf-8
import numpy as np

from psydac.linalg.basic   import LinearOperator, LinearSolver
from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from scipy.sparse          import kron

__all__ = ['KroneckerStencilMatrix_2D',
           'kronecker_solve_2d_par',
           'kronecker_solve_3d_par',
           'kronecker_solve']
           
def kron_dot(starts, ends, pads, X, X_tmp, Y, A, B):
    s1 = starts[0]
    s2 = starts[1]
    e1 = ends[0]
    e2 = ends[1]
    p1 = pads[0]
    p2 = pads[1]

    for j1 in range(s1-p1, e1+p1+1):
        for i2 in range(s2, e2+1):
             
             X_tmp[j1+p1-s1, i2-s2+p2] = sum(X[j1+p1-s1, i2-s2+k]*B[i2,k] for k in range(2*p2+1))
    
    for i1 in range(s1, e1+1):
        for i2 in range(s2, e2+1):
             Y[i1-s1+p1,i2-s2+p2] = sum(A[i1, k]*X_tmp[i1-s1+k, i2-s2+p2] for k in range(2*p1+1))
    return Y
#==============================================================================
class KroneckerStencilMatrix( LinearOperator ):

    def __init__( self,V, W, *args ):

        assert isinstance( V, StencilVectorSpace )
        assert isinstance( W, StencilVectorSpace )

        for i,A in enumerate(args):
            assert isinstance( A, StencilMatrix )
            assert A.domain.ndim == 1
            assert A.domain.npts[0] == V.npts[i]


        self._domain   = V
        self._codomain = W
        self._mats     = args
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
    def mats(self):
        return self._mats

    # ...
    def dot( self, x, out=None ):

        dot = np.dot

        assert isinstance( x, StencilVector )
        assert x.space is self.domain

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
            xx = tuple(i+p for i,p in zip(ii,pads))
            for jj in np.ndindex(*pnrows):
                i_mats = [mat._data[x,j] for x,j,mat in zip(xx,jj,mats)]
                ii_jj = tuple(i+j for i,j in zip(ii,jj))
                v += x._data[ii_jj]*np.product(i_mats)

            out._data[xx] = v

        out.update_ghost_regions()

        return out

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------

    def __getitem__(self, key):
        rows = key[:2]
        cols = key[2:]
        A = self._A1
        B = self._A2
        return A[rows[0],cols[0]]*B[rows[1],cols[1]]

    # ...
    def tocoo( self ):
        raise NotImplementedError('TODO')

    #...
    def tocsr( self ):
        return self.tocoo().tocsr()

    #...
    def toarray( self ):
        # TODO improve by using tocoo.toarray()
        A = self._A1.toarray()
        B = self._A2.toarray()
        mat = np.kron(A, B)
        return mat

    #...
    def copy( self ):
        M = KroneckerStencilMatrix_2D( self.domain, self.codomain, self._A1, self._A2 )
        return M


class KroneckerStencilMatrix( LinearOperator ):

    def __init__( self,V, W, *args ):

        assert isinstance( V, StencilVectorSpace )
        assert isinstance( W, StencilVectorSpace )

        for i,A in enumerate(args):
            assert isinstance( A, StencilMatrix )
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
            xx = tuple(i+p for i,p in zip(ii,pads))
            for jj in np.ndindex(*pnrows):
                i_mats = [mat._data[x,j] for x,j,mat in zip(xx,jj,mats)]
                ii_jj = tuple(i+j for i,j in zip(ii,jj))
                v += x._data[ii_jj]*np.product(i_mats)

            out._data[xx] = v
        out.update_ghost_regions()

        return out

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------

    # ...
    def __getitem__(self, key):
        pads = self._codomain.pads
        rows = key[:self.ndim]
        cols = key[self.ndim:]
        mats = self.mats
        elements = [A[i,j] for A,i,j in zip(mats, rows, cols)]
        return np.product(elements)
    # ...
    def tosparse( self ):

        mat = self.mats[0].tosparse()
        for i in range(len(self.mats)-1):
            mat = kron(mat, self.mats[i+1].tosparse())
        return mat

    #...
    def tocsr( self ):
        return self.tosparse().tocsr()

    #...
    def toarray( self ):
        return self.tosparse().toarray()
    #...
    def copy( self ):
        M = KroneckerStencilMatrix( self.domain, self.codomain, *self.mats )
        return M

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
def kronecker_solve_3d_par( A1, A2, A3, rhs, out=None ):
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

    # 1D global arrays
    X_glob_1 = np.zeros( n1 )
    Y_glob_2 = np.zeros( n2 )
    Y_glob_3 = np.zeros( n3 )
    # ...

    for i3 in range(e3-s3+1):
        for i2 in range(e2-s2+1):
            X_loc = X[:, i2, i3].copy()  # need 1D contiguous copy
            subcomm_1.Allgatherv( X_loc, [X_glob_1, sizes1, disps1, mpi_type] )
            Y[:, i2, i3] = A1.solve( X_glob_1 )[s1:e1+1]

    for i3 in range(e3-s3+1):
        for i1 in range(e1-s1+1):
            Y_loc = Y[i1, :, i3].copy()  # need 1D contiguous copy
            subcomm_2.Allgatherv( Y_loc, [Y_glob_2, sizes2, disps2, mpi_type] )
            Y[i1, :, i3] = A2.solve( Y_glob_2 )[s2:e2+1]

    for i2 in range(e2-s2+1):
        for i1 in range(e1-s1+1):
            Y_loc = Y[i1, i2, :]  # 1D contiguous slice
            subcomm_3.Allgatherv( Y_loc, [Y_glob_3, sizes3, disps3, mpi_type] )
            Y[i1, i2, :] = A3.solve( Y_glob_3 )[s3:e3+1]

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

    # 1D case
    # TODO: should also work in parallel
    if space.ndim == 1:
        if space.parallel:
            raise NotImplementedError( "1D Kronecker solver only works in serial." )
        else:
            solver[0].solve( rhs, out )

    # 2D/3D cases
    # TODO: should also work in serial
    # TODO: should work in any number of dimensions
    else:
        if not space.parallel:
            raise NotImplementedError( "Multi-dimensional Kronecker solver only works in parallel." )

        if   space.ndim == 2: kronecker_solve_2d_par( *solvers, rhs=rhs, out=out )
        elif space.ndim == 3: kronecker_solve_3d_par( *solvers, rhs=rhs, out=out )
        else:
            raise NotImplementedError( "Kronecker solver does not work in more than 3 dimensions." )

    return out

