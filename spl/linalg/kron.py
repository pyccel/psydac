#coding = utf-8

from spl.linalg.basic   import LinearOperator
from spl.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix

__all__ = ['KroneckerStencilMatrix_2D', 'kronecker_solve_2d_par']

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
def kronecker_solve_2d_par( solve1, solve2, rhs, out=None ):
    """
    Solve linear system Ax=b with A=kron(A2,A1).

    Parameters
    ----------
    solve1 : callable
        Solve linear system A1 x1 = b1: x1=solve1(b1).

    solve2 : callable
        Solve linear system A2 x2 = b2: x2=solve2(b2).

    rhs : StencilVector 2D
        Right hand side vector of linear system Ax=b.

    """

    if out is not None:
        assert isinstance( out, StencilVector )
        assert out.space is rhs.space
    else:
        out = StencilVector( rhs.space )

    # ...
    V = rhs.space
    X = StencilVector( V )

    s1, s2 = V.starts
    e1, e2 = V.ends
    n1, n2 = V.npts

    subcomm_1 = V.cart.subcomm[0]
    subcomm_2 = V.cart.subcomm[1]
    # ...

    # ...
    Y_glob_1 = np.zeros((n1))

    Ytmp_glob_1 = np.zeros((n1, e2-s2+1))
    Ytmp_glob_2 = np.zeros((n2))

    X_glob_2 = np.zeros((e1-s1+1, n2))
    # ...

    for i2 in range(e2-s2+1):
        Y_loc = rhs[s1:e1+1, s2+i2].copy()
        subcomm_1.Allgatherv( Y_loc, Y_glob_1 )
        Ytmp_glob_1[:,i2] = solve1( Y_glob_1 )

    for i1 in range(e1-s1+1):
        Ytmp_loc = Ytmp_glob_1[s1+i1, 0:e2+1-s2].copy()
        subcomm_2.Allgatherv( Ytmp_loc, Ytmp_glob_2 )
        X_glob_2[i1,:] = solve2( Ytmp_glob_2 )

    out[s1:e1+1,s2:e2+1] = X_glob_2[:, s2:e2+1]

    return out
