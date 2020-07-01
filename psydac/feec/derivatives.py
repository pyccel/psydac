# -*- coding: UTF-8 -*-

import numpy as np

from psydac.linalg.stencil  import StencilMatrix, StencilVectorSpace
from psydac.linalg.kron     import KroneckerStencilMatrix
from psydac.linalg.block    import ProductSpace, BlockVector, BlockLinearOperator, BlockMatrix
from psydac.fem.vector      import ProductFemSpace
from psydac.linalg.identity import IdentityLinearOperator, IdentityStencilMatrix as IdentityMatrix

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

        d_matrices = [d_matrix(V.nbasis, V.degree, V.periodic) for V in V0_h.spaces]
        identities = [IdentityMatrix(V.vector_space)           for V in V0_h.spaces]

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

        VS0 = ProductSpace(V0_h.vector_space)

        if dim == 1:
            VS1 = ProductSpace(V1_h.vector_space)
        else:
            VS1 = V1_h.vector_space

        self._matrix = BlockMatrix( VS0, VS1, blocks=[[mat] for mat in mats] )

    def __call__(self, x):
        x = BlockVector(ProductSpace(x.space), blocks=[x])
        return self._matrix.dot(x)

class Curl(object):
    def __init__(self, V1_h, V2_h):
        """
        
        V1_h  : ProductFemSpace
        
        V2_h  : ProductFemSpace
        
        """

        D_basis = [V.spaces[i] for i,V in enumerate(V1_h.spaces)]
        dim     = len(D_basis)
        if dim == 2:
            N_basis = [V1_h.spaces[1].spaces[0], V1_h.spaces[0].spaces[1]]
        elif dim == 3:
            N_basis = [V1_h.spaces[1].spaces[0], V1_h.spaces[0].spaces[1], V1_h.spaces[0].spaces[2]]

        d_matrices   = [d_matrix(V.nbasis, V.degree, V.periodic)   for V in N_basis]
        identities_0 = [IdentityMatrix(V.vector_space) for V in N_basis]
        identities_1 = [IdentityMatrix(V.vector_space) for V in D_basis]
        
        mats = []    
            
        if dim == 3:
            mats = [[None, None, None],
                    [None,None,None],
                    [None,None,None]]
                    
            args       = [-identities_0[0], identities_1[1], d_matrices[2]]
            mats[0][1] = KroneckerStencilMatrix(V1_h.vector_space.spaces[1], V2_h.vector_space.spaces[0],*args)
            
            args       = [identities_0[0], d_matrices[1], identities_1[2]]
            mats[0][2] = KroneckerStencilMatrix(V1_h.vector_space.spaces[2], V2_h.vector_space.spaces[0],*args)
            # ...
            
            # ...
            args       = [identities_1[0], identities_0[1], d_matrices[2]]
            mats[1][0] = KroneckerStencilMatrix(V1_h.vector_space.spaces[0], V2_h.vector_space.spaces[1], *args)
            
            args       = [-d_matrices[0], identities_0[1], identities_1[2]]
            mats[1][2] = KroneckerStencilMatrix(V1_h.vector_space.spaces[2], V2_h.vector_space.spaces[1], *args)
            # ...
            
            # ...
            args       = [-identities_1[0], d_matrices[1], identities_0[2]]
            mats[2][0] = KroneckerStencilMatrix(V1_h.vector_space.spaces[0], V2_h.vector_space.spaces[2], *args)
            
            args       = [d_matrices[0], identities_1[1], identities_0[2]]
            mats[2][1] = KroneckerStencilMatrix(V1_h.vector_space.spaces[1], V2_h.vector_space.spaces[2], *args)

            self._matrix = BlockLinearOperator( V1_h.vector_space, V2_h.vector_space, blocks=mats )

        elif dim == 2:
            mats = [[None , None]]
        
            args = [-identities_1[0], d_matrices[1]]
            mats[0][0] = KroneckerStencilMatrix(V1_h.vector_space.spaces[0], V2_h.vector_space, *args)

            args = [d_matrices[0], identities_1[1]]
            mats[0][1] = KroneckerStencilMatrix(V1_h.vector_space.spaces[1], V2_h.vector_space, *args)

            self._matrix = BlockMatrix( V1_h.vector_space, ProductSpace( V2_h.vector_space ), blocks=mats )
        
        else:
            raise NotImplementedError('TODO')

    def __call__(self, x):
        return self._matrix.dot(x)

class Div(object):
    def __init__(self, V2_h, V3_h):
        """
        V2_h : ProductFemSpace

        V3_h  : TensorFemSpace
        
        """

        dim        = V2_h.ldim
        N_basis    = [V.spaces[i] for i,V in enumerate(V2_h.spaces)]

        d_matrices = [d_matrix(V.nbasis, V.degree, V.periodic)   for V in N_basis]
        identities = [IdentityMatrix(V.vector_space) for V in V3_h.spaces]
            
        mats = []
        for i in range(dim):
            args = []
            for j in range(dim):
                if i==j:
                    args.append(d_matrices[j])
                else:
                    args.append(identities[j])
                    
            mats += [KroneckerStencilMatrix(V2_h.spaces[i].vector_space, V3_h.vector_space, *args)]
        
        Mat = BlockMatrix( V2_h.vector_space, ProductSpace(V3_h.vector_space), blocks=[mats])
        self._matrix = Mat

    def __call__(self, x):
        return self._matrix.dot(x)

class Rot(object):
    def __init__(self, V0_h, V1_h):
        """
        V0_h : TensorFemSpace
        
        V1_h : ProductFemSpace
        
        """
        
        if V0_h.ldim != 2:
            raise ValueError('only dimension 2 is available')


        d_matrices = [d_matrix(V.nbasis, V.degree, V.periodic) for V in V0_h.spaces]
        identities = [IdentityMatrix(V.vector_space) for V in V0_h.spaces]
         
        mats = [[None],[None]]
        mats[0][0] = KroneckerStencilMatrix(V0_h.vector_space, V1_h.spaces[0].vector_space, *[identities[0],d_matrices[1]])
        mats[1][0] = KroneckerStencilMatrix(V0_h.vector_space, V1_h.spaces[1].vector_space, *[-d_matrices[0],identities[1]])

        Mat = BlockMatrix( ProductSpace(V0_h.vector_space), V1_h.vector_space, blocks=mats )
        self._matrix = Mat

    def __call__(self, x):
        x = BlockVector(ProductSpace(x.space), blocks=[x])
        return self._matrix.dot(x)

    
