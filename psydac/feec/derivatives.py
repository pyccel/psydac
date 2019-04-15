# -*- coding: UTF-8 -*-

import numpy as np

from psydac.linalg         import StencilMatrix, StencilVectorSpace, KroneckerStencilMatrix
from psydac.linalg.block   import ProductSpace, BlockVector, BlockLinearOperator, BlockMatrix
from psydac.fem.vector     import ProductFemSpace

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
    
    V = StencilVectorSpace([n], [p], [P])
    M = StencilVector(V, V)
    
    M._data[0, p] = 1.
    for i in range(1, n):
        M._data[i,p] = 1.
        M[i,p-1] = -1.
        
    return M
    
def identity(n, p, P):
    """creates a 1d identity matrix.
    The final matrix will have a shape of (n,n)

    n: int
        number of nodes
        
    p : int
        pads
        
    P : bool
        periodicity
    """
    
    V = StencilVectorSpace([n], [p], [P])
    M = StencilVector(V, V)
    
    for i in range(0, n):
        M._data[i, p] = 1.
        
    return M


class Grad(object):
    def __init__(self, Vh):
        """
        Vh : TensorFemSpace
        
        """
        
        dim     = Vh.ldim
        npts    = [V.nbasis for V in Vh.spaces]
        pads    = [V.degree for V in Vh.spaces]
        periods = [V.periodic for V in Vh.spaces]
        
        d_matrices = [d_matrix(n, p, P) for n,p,P in zip(npts, pads, periods)]
        identities = [identity(n, p, P) for n,p,P in zip(npts, pads, periods)]
        
        if dim == 1:
            Grad_Vh = Vh.reduce_degree(axes=[0])
        elif dim == 2:
            spaces = [Vh.reduce_degree(axes=[0]), Vh.reduce_degree(axes=[1])]
            Grad_Vh = ProductFemSpace(*spaces)
        elif dim == 3:
            spaces = [Vh.reduce_degree(axes=[0]), Vh.reduce_degree(axes=[1]), Vh.reduce_degree(axes=[2])]
            Grad_Vh = ProductFemSpace(*spaces)
        else:
            raise NotImplementedError('TODO')
         
        mats = []
        for i in range(dim):
            args = []
            for j in range(dim):
                if i==j:
                    args.append(d_matrices[j])
                else:
                    args.append(idnetities[j])
            mats.append(KroneckerStencilMatrix(Vh, Vh.reduce(axes=[i]), *args))

        Mat = BlockLinearOperator( Vh, Grad_Vh, blocks=[mats] )
        self._matrix = Mat



    def __call__(self, x):
        return self._matrix.dot(x)

class Curl(object):
    def __init__(self, Vh):
        """
        Vh : TensorFemSpace
        
        """
        dim     = Vh.ldim
        npts    =  [V.nbasis for V in Vh.spaces]
        pads    =  [V.degree for V in Vh.spaces]
        periods =  [V.periodic for V in Vh.spaces]
        
        d_matrices = [d_matrix(n, p, P) for n,p,P in zip(npts, pads, periods)]
        identities = [identity(n-1, p, P) for n,p,P in zip(npts, pads, periods)]
        

        if dim == 2:
            spaces = [Vh.reduce_degree(axes=[1]), Vh.reduce_degree(axes=[0])]
            Curl_Vh = ProductFemSpace(*spaces)
        elif dim == 3:
            spaces = [Vh.reduce_degree(axes=[1,2]), Vh.reduce_degree(axes=[0,2]), Vh.reduce_degree(axes=[0,1])]
            Curl_Vh = ProductFemSpace(*spaces)
        else:
            raise NotImplementedError('TODO')
            
        mats = []
        for i in range(dim):
            args = []
            for j in range(dim):
                if i==j:
                    args.append(d_matrices[j])
                else:
                    args.append(idnetities[j])
            mats.append(KroneckerStencilMatrix(Vh, Vh.reduce(axes=[i]), *args))
            
        if dim == 3:
            mats = [[None,-mats[2],mats[1]],
                    [mats[2],None,-mats[0]],
                    [-mats[1],mats[0],0]]
        elif dim == 2:
            mats = [[mats[1],-mats[0]]]
        
        Mat = BlockLinearOperator( Vh, Gurl_Vh, blocks=mats )
        self._matrix = Mat

    def __call__(self, x):
        return self._matrix.dot(x)

class Div(object):
    def __init__(self, Vh):
        """
        Vh : TensorFemSpace
        
        """
        dim     = Vh.ldim
        npts    =  [V.nbasis for V in Vh.spaces]
        pads    =  [V.degree for V in Vh.spaces]
        periods =  [V.periodic for V in Vh.spaces]
        
        d_matrices = [d_matrix(n, p, P) for n,p,P in zip(npts, pads, periods)]
        identities = [identity(n-1, p, P) for n,p,P in zip(npts, pads, periods)]
        

        if dim == 1:
            Div_Vh = Vh.reduce(axes=[0])
        if dim == 2:
            Div_Vh = Vh.reduce_degree(axes=[0,1])
        elif dim == 3:
            Div_Vh = Vh.reduce_degree(axes=[0,1,2])
        else:
            raise NotImplementedError('TODO')
            
        mats = []
        for i in range(dim):
            args = []
            for j in range(dim):
                if i==j:
                    args.append(d_matrices[j])
                else:
                    args.append(idnetities[j])
            mats.append(KroneckerStencilMatrix(Vh, Vh.reduce(axes=[i]), *args))
            
        mats = [[mat] for mat in mats]
        
        Mat = BlockLinearOperator( Vh, Div_Vh, blocks=mats )
        self._matrix = Mat

    def __call__(self, x):
        return self._matrix.dot(x)


# user friendly function that returns all discrete derivatives
def discrete_derivatives(Vh):
    """."""
    
