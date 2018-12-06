# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

import numpy as np

from spl.mapping.discrete import SplineMapping
from spl.linalg.stencil   import StencilVectorSpace, StencilVector, StencilMatrix
from spl.linalg.block     import ProductSpace, BlockVector, BlockMatrix
from spl.polar .dense     import DenseVectorSpace, DenseVector, DenseLinearOperator
from spl.polar.c1_spaces  import new_c1_vector_space
from spl.polar.c1_linops  import LinearOperator_StencilToDense
from spl.polar.c1_linops  import LinearOperator_DenseToStencil

#===============================================================================
class C1Projector:
    """
         +---+---------+
         |   |         |
         | L |    0    |
         |   |         |
         +---+---------+
    E =  |   |         |
         |   |         |
         | 0 |    I    |
         |   |         |
         |   |         |
         +---+---------+
    """
    def __init__( self, mapping ):

        assert isinstance( mapping, SplineMapping )

        S = mapping.space.vector_space

        assert isinstance( S, StencilVectorSpace )

        # Store matrix L with 3 indices
        lamb = self.compute_lambda( mapping )
        self._L = np.array( lamb )

        # Vector spaces
        self._S = S
        self._P = new_c1_vector_space( S, radial_dim=0 )

    # ...
    def compute_lambda( self, mapping ):

        s1, s2 = mapping.space.vector_space.starts
        e1, e2 = mapping.space.vector_space.ends

        x = mapping.control_points[0:2,s2:e2+1,0]
        y = mapping.control_points[0:2,s2:e2+1,1]

        SQRT3     = np.sqrt(3.0)
        ONE_THIRD = 1.0/3.0
        (x0,y0)   = (x[0,0],y[0,0])

        # Define equilateral triangle enclosing first row of control points
        # TODO: use MPI_MAX
        tau = max( np.max( -2*(x[1,:]-x0) ),
                   np.max( x[1,:]-x0-SQRT3*(y[1,:]-y0) ),
                   np.max( x[1,:]-x0+SQRT3*(y[1,:]-y0) ) )

        # Coordinates of vertices of equilateral triangle
        vrtx = [(x0-tau/2, y0-SQRT3*tau/2), 
                (x0+tau  , y0            ),
                (x0-tau/2, y0+SQRT3*tau/2)]

        # Define barycentric coordinates with respect to smallest circle
        # enclosing first row of control points
        lambda_0  = ONE_THIRD*(1.0 + 2.0*(x-x0)                /tau)
        lambda_1  = ONE_THIRD*(1.0 -    ((x-x0) - SQRT3*(y-y0))/tau)
        lambda_2  = ONE_THIRD*(1.0 -    ((x-x0) + SQRT3*(y-y0))/tau)
        lamb = (lambda_0, lambda_1, lambda_2)

        # STORE INFO
        self._tau = tau
        self._vrtx = vrtx
        self._lamb = lamb

        return lamb

    # ...
    @property
    def L( self ):
        L = self._L[:,:]
        L.setflags( write=False )
        return L

    # ...
    @property
    def tensor_space( self ):
        return self._S

    # ...
    @property
    def c1_space( self ):
        return self._P

    #---------------------------------------------------------------------------
    def change_matrix_basis( self, G ):
        """
        Compute G' = E^t * G * E.

        """

        assert isinstance( G, StencilMatrix )
        assert G.domain   == self.tensor_space
        assert G.codomain == self.tensor_space

        L = self._L
        P = self.c1_space

        n1, n2 = G.domain.npts
        s1, s2 = G.starts
        e1, e2 = G.ends
        p1, p2 = G.pads

        #****************************************
        # Compute A' = L^T A L
        #****************************************
        Ap = np.zeros( (3,3) )

        # Compute product (A L) and store it
        AL = np.zeros( (3,2,n2) )
        for v in [0,1,2]:
            for i1 in [0,1]:
                for i2 in range( s2, e2+1 ):

                    # Sum over j1 and j2
                    temp = 0
                    for j1 in [0,1]:
                        for k2 in range( -p2, p2+1 ):
                            j2 = (i2+k2) % n2
                            temp += G[i1,i2,j1-i1,k2] * L[v,j1,j2]

                    AL[v,i1,i2] = temp

        # Compute product A' = L^T (A L)
        for u in range( 3 ):
            for v in range( 3 ):
                Ap[u,v] = np.dot( L[u,:,:].flat, AL[v,:,:].flat )

        # Create linear operator
        Ap = DenseLinearOperator( P[0], P[0], Ap )

        #****************************************
        # Compute B' = L^T B
        #****************************************
        Bp = np.zeros( (3,p1,n2) )

        for u in [0,1,2]:

            for i1 in [0,1]:
                for i2 in range( s2, e2+1 ):

                    for j1 in range( 2, i1+p1+1 ):
                        for k2 in range( -p2, p2+1 ):
                            j2 = (i2+k2) % n2

                            Bp[u,j1-2,j2] += L[u,i1,i2] * G[i1,i2,j1-i1,k2]

        # Create linear operator
        Bp = LinearOperator_StencilToDense( P[1], P[0], Bp )

        #****************************************
        # Compute C' = C L
        #****************************************
        Cp = np.zeros( (p1,n2,3) )

        for i1 in range( 2, 2+p1 ):
            for i2 in range( s2, e2+1 ):

                for v in [0,1,2]:

                    # Sum over j1 and j2
                    temp = 0
                    for j1 in range( max(0,i1-p1), 2 ):
                        for k2 in range( -p2, p2+1 ):
                            j2 = (i2+k2) % n2
                            temp += G[i1,i2,j1-i1,k2] * L[v,j1,j2]

                    Cp[i1-2,i2,v] = temp

        # Create linear operator
        Cp = LinearOperator_DenseToStencil( P[0], P[1], Cp )

        #****************************************
        # Store D' = D
        #****************************************

        # Create linear operator
        Dp = StencilMatrix( P[1], P[1] )

        # Copy all data from G, but skip values for i1 = 0, 1
        Dp[0:,:,:,:] = G[2:,:,:,:]

        # Remove values with negative j1 which may have polluted ghost region
        Dp.remove_spurious_entries()

        #****************************************
        # Consistency checks
        #****************************************

        # Is it true that C'=(B')^T ?
        assert np.allclose( Cp._data, np.moveaxis( Bp._data, 0, -1 ), rtol=1e-14, atol=1e-14 )

        #****************************************
        # Block linear operator G' = E^T G E
        #****************************************

        return BlockMatrix( P, P, blocks = [[Ap, Bp], [Cp, Dp]] )

    #---------------------------------------------------------------------------
    def change_rhs_basis( self, b ):
        """
        Compute b' = E^t * b.

        We assume that b was obtained by L^2-projection of a callable function
        onto the tensor-product spline space.

        """
        # TODO: make this work in parallel

        assert isinstance( b, StencilVector )
        assert b.space == self.tensor_space

        L = self._L
        P = self.c1_space

        s1, s2 = b.starts
        e1, e2 = b.ends

        bp0 = np.zeros( 3 )
        for u in [0,1,2]:
            for i1 in [0,1]:
                for i2 in range( s2, e2+1 ):
                    bp0[u] += L[u,i1,i2] * b[i1,i2]

        bp0 =   DenseVector( P[0], bp0 )
        bp1 = StencilVector( P[1] )
        bp1[0:,:,:,:] = b[2:,:,:,:]

        bp1.update_ghost_regions()

        return BlockVector( P, blocks = [bp0, bp1] )

    #---------------------------------------------------------------------------
    def convert_to_tensor_basis( self, vp ):
        """
        Compute v = E * v'

        """
        # TODO: make this work in parallel

        assert isinstance( vp, BlockVector )
        assert vp.space == self.c1_space

        L = self._L

        v = StencilVector( self.tensor_space )
        v[2:,:] = vp[1][0:,:]

        s1, s2 = v.starts
        e1, e2 = v.ends

        vp0 = vp[0].toarray()
        for i1 in [0,1]:
            for i2 in range( s2, e2+1 ):
                for u in [0,1,2]:
                    v[i1,i2] = np.dot( L[:,i1,i2], vp0 )

        v.update_ghost_regions()

        return v

