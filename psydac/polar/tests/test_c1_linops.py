#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import pytest
import numpy as np

from psydac.polar.dense    import DenseVectorSpace, DenseVector, DenseMatrix
from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.block   import BlockVectorSpace, BlockVector, BlockLinearOperator
from psydac.ddm.cart       import DomainDecomposition, CartDecomposition

from psydac.polar.c1_linops import LinearOperator_StencilToDense
from psydac.polar.c1_linops import LinearOperator_DenseToStencil

#==============================================================================
@pytest.mark.parametrize( 'n0'  , [2,3,5] )
@pytest.mark.parametrize( 'npts', [(7,8),(12,13),(19,25)] )
@pytest.mark.parametrize( 'pads', [(2,3),(3,2),(3,3)] )

def test_c1_linops( n0, npts, pads, verbose=False ):

    if verbose:
        np.set_printoptions( precision=2, linewidth=200 )

    n1, n2 = npts
    p1, p2 = pads
    P1,P2  = False, True
    nc1, nc2 = n1-p1*(1-P1), n2-p2*(1-P2) 

    # Spaces
    U = DenseVectorSpace( n0 )
    domain_decomposition = DomainDecomposition(ncells=[nc1, nc2], periods=[P1,P2], comm=None)

    npts          = [n1-2,n2]
    global_starts = [None]*2
    global_ends   = [None]*2
    for axis in range(2):
        es = domain_decomposition.global_element_starts[axis]
        ee = domain_decomposition.global_element_ends  [axis]

        global_ends  [axis]     = (ee+1)-1
        global_ends  [axis][-1] = npts[axis]-1
        global_starts[axis]     = np.array([0] + (global_ends[axis][:-1]+1).tolist())

    # Decomposition of Cartesian domain
    cart = CartDecomposition(
            domain_decomposition      = domain_decomposition,
            npts          = [n1-2,n2],
            global_starts = global_starts,
            global_ends   = global_ends,
            pads          = [p1,p2],
            shifts        = [1,1],
    )


    V = StencilVectorSpace( cart=cart, dtype=float )
    W = BlockVectorSpace( U, V )

    s1, s2 = V.starts
    e1, e2 = V.ends

    # 4 matrix blocks:
    #      | A  B |
    # M =  |      |
    #      | C  D |
    Aa = np.random.random( (n0, n0) )
    Ba = np.random.random( (n0, p1, e2-s2+1+2*p2) )
    Ca = np.random.random( (p1, e2-s2+1, n0) )

    # Linear operators
    A = DenseMatrix( U, U, Aa )
    B = LinearOperator_StencilToDense( V, U, Ba )
    C = LinearOperator_DenseToStencil( U, V, Ca )

    D = StencilMatrix( V, V )
    D[s1:e1+1, s2:e2+1, :, :] = np.random.random( (e1-s1+1, e2-s2+1, 2*p1+1, 2*p2+1) )
    D.remove_spurious_entries()

    M  = BlockLinearOperator( W, W, blocks=[[A,B],[C,D]] )

    # Vectors
    u = DenseVector( U, np.arange( n0, dtype=float ) )
    v = StencilVector( V )
    v[s1:e1+1,s2:e2+1] = np.random.random( (e1-s1+1, e2-s2+1) )
    v.update_ghost_regions()
    w = BlockVector( W, [u,v] )

    # Check individual dot products
    x  = M.dot( w )
    x0 = A.dot( u ) + B.dot( v )
    x1 = C.dot( u ) + D.dot( v )

    tols = {'rtol': 1e-12, 'atol': 1e-12}
    assert np.allclose( x[0].toarray(), x0.toarray(), **tols )
    assert np.allclose( x[1].toarray(), x1.toarray(), **tols )

    # Convert to arrays and compare result to numpy dot
    Ma = M.toarray()
    wa = w.toarray()
    xa = x.toarray()

    assert np.allclose( xa, Ma.dot( wa ), **tols )

    if verbose:
        print( "PASSED" )
        return locals()

#==============================================================================
if __name__ == "__main__":
    namespace = test_c1_linops( n0=5, npts=(7,8), pads=(2,2), verbose=True )
    globals().update( namespace )
