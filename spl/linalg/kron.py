#coding = utf-8
import numpy as np

from spl.linalg.basic   import LinearOperator, LinearSolver
from spl.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix

__all__ = ['KroneckerStencilMatrix_2D', \
           'kronecker_solve_2d_par', 'kronecker_solve_3d_par']

#==============================================================================
class KroneckerStencilMatrix_2D( LinearOperator ):

    def __init__( self, V, W, A1, A2 ):

        assert isinstance( V, StencilVectorSpace )
        assert isinstance( W, StencilVectorSpace )
        assert V is W
        assert V.ndim == 2

        assert isinstance( A1, StencilMatrix )
        assert A1.domain.ndim == 1
        assert A1.domain.npts[0] == V.npts[0]

        assert isinstance( A2, StencilMatrix )
        assert A2.domain.ndim == 1
        assert A2.domain.npts[0] == V.npts[1]

        self._space = V
        self._A1    = A1
        self._A2    = A2
        self._Y     = StencilVector( V )

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def domain( self ):
        return self._space

    # ...
    @property
    def codomain( self ):
        return self._space

    # ...
    def dot( self, X, out=None ):

        dot = np.dot

        assert isinstance( X, StencilVector )
        assert X.space is self.domain

        if out is not None:
            assert isinstance( out, StencilVector )
            assert out.space is self.codomain
        else:
            out = StencilVector( self.codomain )

        [s1, s2] = self._space.starts
        [e1, e2] = self._space.ends
        [p1, p2] = self._space.pads

        A1 = self._A1
        A2 = self._A2
        Y  = self._Y

        for j1 in range(s1-p1, e1+p1+1):
            for i2 in range(s2, e2+1):
                 Y[j1,i2] = dot( X[j1,i2-p2:i2+p2+1], A2[i2,:])

        for i1 in range(s1, e1+1):
            for i2 in range(s2, e2+1):
                 out[i1,i2] = dot( A1[i1,:], Y[i1-p1:i1+p1+1,i2] )

        out.update_ghost_regions()

        return out

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------
    @property
    def starts( self ):
        return self._space.starts

    # ...
    @property
    def ends( self ):
        return self._space.ends

    # ...
    @property
    def pads( self ):
        return self._space.pads

    # ...
    def __getitem__(self, key):
        raise NotImplementedError('TODO')

    # ...
    def tocoo( self ):
        raise NotImplementedError('TODO')

    #...
    def tocsr( self ):
        return self.tocoo().tocsr()

    #...
    def toarray( self ):
        return self.tocoo().toarray()

    #...
    def copy( self ):
        M = KroneckerStencilMatrix_2D( self.domain, self.codomain, self._A1, self._A2 )
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

