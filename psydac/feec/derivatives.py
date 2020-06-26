# -*- coding: UTF-8 -*-

import numpy as np

from psydac.linalg.stencil  import StencilMatrix, StencilVectorSpace
from psydac.linalg.kron     import KroneckerStencilMatrix
from psydac.linalg.block    import ProductSpace, BlockVector, BlockLinearOperator, BlockMatrix
from psydac.fem.vector      import ProductFemSpace
from psydac.linalg.identity import IdentityLinearOperator, IdentityMatrix

def d_matrix(n, p, P):
    """creates a 1d incidence matrix.
    The final matrix will have a shape of (n,n-1)

    n: int
        number of nodes
        
    p : int
        pads
        
    P : bool
        periodicity
    """
    
    V1 = StencilVectorSpace([n], [p], [P])
    V2 = StencilVectorSpace([n-1], [p], [P])
    M = StencilMatrix(V1, V2)

    for i in range(n):
        M._data[p+i,p] = -1.
        M._data[p+i,p+1] = 1.
    return M

class Grad(object):
    def __init__(self, V0_h, V1_h):
        """
        V0_h : TensorFemSpace
        
        V1_h : ProductFemSpace
        
        """
        
        dim     = V0_h.ldim
        npts    = [V.nbasis for V in V0_h.spaces]
        pads    = [V.degree for V in V0_h.spaces]
        periods = [V.periodic for V in V0_h.spaces]

        d_matrices = [d_matrix(n, p, P) for n,p,P in zip(npts, pads, periods)]
        identities = [IdentityMatrix(V.vector_space) for V in V0_h.spaces]

        mats = []
        for i in range(dim):
            args = []
            for j in range(dim):
                if i==j:
                    args.append(d_matrices[j])
                else:
                    args.append(identities[j])
            
            if dim == 1:
                mats += args
            else: 
                mats += [KroneckerStencilMatrix(V0_h.vector_space, V1_h.vector_space.spaces[i], *args)]

        C0 = ProductSpace(V0_h.vector_space)
       
        if dim == 1:
            C1 = ProductSpace(V1_h.vector_space)
        else:
            C1 = V1_h.vector_space

        self._matrix = BlockMatrix( C0, C1, blocks=[[mat] for mat in mats] )

    def __call__(self, x):
        x = BlockVector(ProductSpace(x.space), blocks=[x])
        return self._matrix.dot(x)

class Curl(object):
    def __init__(self, Curl_Vh, Div_Vh):
        """
        
        Curl_Vh : ProductFemSpace
        
        Div_Vh  : ProductFemSpace
        
        """

        dim     = Vh.ldim
        npts    =  [V.nbasis for V in Vh.spaces]
        pads    =  [V.degree for V in Vh.spaces]
        periods =  [V.periodic for V in Vh.spaces]

        d_matrices   = [d_matrix(n, p, P) for n,p,P in zip(npts, pads, periods)]
        identities_0 = [identity(n, p, P) for n,p,P in zip(npts, pads, periods)]
        identities_1 = [identity(n-1, p, P) for n,p,P in zip(npts, pads, periods)]
        
        mats = []    
            
        if dim == 3:
            mats = [[None, None, None],
                    [None,None,None],
                    [None,None,None]]
                    
            args       = [-identities_0[0],identities_1[1],d_matrices[2]]
            mats[0][1] = Stencil_kron(Curl_Vh.spaces[1],Div_Vh.spaces[0],*args)
            
            args       = [identities_0[0],d_matrices[1],identities_1[2]]
            mats[0][2] = Stencil_kron(Curl_Vh.spaces[2],Div_Vh.spaces[0],*args)
            # ...
            
            # ...
            args       = [identities_1[0],identities_0[1],d_matrices[2]]
            mats[1][0] = Stencil_kron(Curl_Vh.spaces[0],Div_Vh.spaces[1], *args)
            
            args       = [-d_matrices[0],identities_0[1],identities_1[2]]
            mats[1][2] = Stencil_kron(Curl_Vh.spaces[2],Div_Vh.spaces[1], *args)
            # ...
            
            # ...
            args       = [-identities_1[0],d_matrices[1],identities_0[2]]
            mats[2][0] = Stencil_kron(Curl_Vh.spaces[0],Div_Vh.spaces[2], *args)
            
            args       = [d_matrices[0],identities_1[1],identities_0[2]]
            mats[2][1] = Stencil_kron(Curl_Vh.spaces[1],Div_Vh.spaces[2], *args)
            
            Vh = Curl_Vh
            Wh = Div_Vh

        elif dim == 2:
            mats = [[None , None]]
        
            args = [-identities_1[0], d_matrices[1]]
            mats[0][0] = Stencil_kron(Curl_Vh.spaces[0], Div_Vh, *args)

            args = [d_matrices[0], identities_1[1]]
            mats[0][1] = Stencil_kron(Curl_Vh.spaces[1], Div_Vh, *args)
            
            Vh = Curl_Vh
            Wh = ProductSpace( Div_Vh )
        
        else:
            raise NotImplementedError('TODO')
            
        Mat = BlockLinearOperator( Vh, Wh, blocks=mats )
        self._matrix = Mat


    def __call__(self, x):
        return self._matrix.dot(x)

class Div(object):
    def __init__(self, Vh, Div_Vh, L2_Vh):
        """
        Vh      : TensorFemSpace
        
        Div_Vh : StencilVectorSpace
        
        L2_Vh  : StencilVectorSpace
        
        """

        dim     = Vh.ldim
        npts    =  [V.nbasis for V in Vh.spaces]
        pads    =  [V.degree for V in Vh.spaces]
        periods =  [V.periodic for V in Vh.spaces]
        
        d_matrices = [d_matrix(n, p, P) for n,p,P in zip(npts, pads, periods)]
        identities = [identity(n-1, p, P) for n,p,P in zip(npts, pads, periods)]
            
        mats = []
        for i in range(dim):
            args = []
            for j in range(dim):
                if i==j:
                    args.append(d_matrices[j])
                else:
                    args.append(identities[j])
                    
            mats += [Stencil_kron(Div_Vh.spaces[i], L2_Vh, *args)]
        
        Mat = BlockLinearOperator( Div_Vh, ProductSpace(L2_Vh), blocks=[mats])
        self._matrix = Mat

    def __call__(self, x):
        return self._matrix.dot(x)

class Rot(object):
    def __init__(self, Vh, Rot_Vh):
        """
        Vh : TensorFemSpace
        
        Grad_Vh : StencilVectorSpace
        
        """
        
        dim     = Vh.ldim
        
        if dim != 2:
            raise ValueError('only dimension 2 is available')
            
        npts    = [V.nbasis for V in Vh.spaces]
        pads    = [V.degree for V in Vh.spaces]
        periods = [V.periodic for V in Vh.spaces]

        d_matrices = [d_matrix(n, p, P) for n,p,P in zip(npts, pads, periods)]
        identities = [identity(n, p, P) for n,p,P in zip(npts, pads, periods)]
         
        mats = []
        mats += [Stencil_kron(Vh.vector_space, Grad_Vh.spaces[0], *args)]
        mats += [Stencil_kron(Vh.vector_space, Grad_Vh.spaces[1], *args)]

        Vh = Vh.vector_space
        Vh = ProductSpace(Vh)
        
        mats = [[mats[1]],[-mats[0]]]
        
        Mat = BlockLinearOperator( Vh, Grad_Vh, blocks=mats )
        self._matrix = Mat



    def __call__(self, x):
        return self._matrix.dot(x)
# user friendly function that returns all discrete derivatives
def discrete_derivatives(Vh):
    """."""
    
