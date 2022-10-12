import numpy as np
from numpy import ndarray

#from psydac.linalg2.basic import VectorSpace, Vector, LinearOperator
from psydac.linalg2.ndarray import NdarrayVectorSpace, NdarrayVector, NdarrayLinearOperator

__all__ = ("SumLinearOperator", "ConvLinearOperator", "ScalLinearOperator", "ZeroOperator", "IdOperator",)

class SumLinearOperator( NdarrayLinearOperator ):
    def __init__(self, A, B):
        assert isinstance(A, NdarrayLinearOperator)
        assert isinstance(B, NdarrayLinearOperator)
        assert A._domain == B._domain
        assert A._codomain == B._codomain
        NdarrayLinearOperator.__init__(self, domain=A._domain, codomain=A.codomain)
        if isinstance(A,SumLinearOperator):
            addends = A._addends
            if isinstance(B,SumLinearOperator):
                addends = np.append(addends, B._addends)
            else:
                addends= np.append(addends, B)
        elif isinstance(B,SumLinearOperator):
            addends = np.append(A, B._addends)
        else:
            addends = np.array([A,B])

        self._addends = addends
        
    @property
    def addends( self ):
        return self._addends

    @property
    def length( self ):
        return np.size(self._addends)

    def dot( self, v, mat=None, fac=1, out=None ):
        if out is not None:
            level = 1
        else:
            level = 0
            out = NdarrayVector(space=self.codomain)
        if mat is None:
            mat = np.zeros((self._codomain._dim,self._domain._dim),dtype=float)
        assert isinstance(v,NdarrayVector)
        assert v.space == self._domain
        if mat is not None:
            assert isinstance(mat,np.ndarray)
        assert isinstance(out,NdarrayVector)
        assert out.space == self._codomain

        # Gathering all the indices!
        index_mat = np.where([self._addends[i]._matrix is not None for i in range(self.length)])
        index_CLO = np.where([isinstance(self._addends[i],ConvLinearOperator) for i in range(self.length)])
        index_ScLO = np.where([isinstance(self._addends[i],ScalLinearOperator) for i in range(self.length)])
        index_union = np.append(index_mat[0],index_CLO[0])
        index_union = np.append(index_union,index_ScLO[0])
        index_else = np.setdiff1d(range(self.length),index_union,assume_unique=True)

        #print('Index of matrix LOs')
        #print(index_mat[0])       
        #print('Index of CLOs')
        #print(index_CLO[0])        
        #print('Index of ScLOs')
        #print(index_ScLO[0])        
        #print('Index of other LOs')
        #print(index_else)

        # Operators with matrix representation       
        len = np.size(index_mat[0])
        if len != 0:
            tmpmat = self._addends[index_mat[0][0]]._matrix.copy()
            if len > 1:
                for i in range(1,len):#index_mat[0]:
                    tmpmat += self._addends[index_mat[0][i]]._matrix
            mat += np.multiply(fac,tmpmat)
        
        # Operators without matrix representation that are also not CLOs or ScLOs
        len = np.size(index_else)
        if len != 0:
            tmpout = self._addends[index_else[0]].dot(v)
            if len > 1:
                for i in range(1,len):
                    tmpout._data += self._addends[index_else[i]].dot(v)._data

        # Operators that are of class ConvolutionalLinearOperator
        len = np.size(index_CLO[0])
        if len != 0:
            if tmpout:
                tmpout += self._addends[index_CLO[0][0]].dot(v)
            else:
                tmpout = self._addends[index_CLO[0][0]].dot(v)
            if len > 1:
                for i in range(1,len):
                    tmpout._data += self._addends[index_CLO[0][i]].dot(v)._data

        if tmpout:
            out._data += np.multiply(fac,tmpout._data)

        # Operators that are of class ScalLinearOperator
        len = np.size(index_ScLO[0])
        if len != 0:
            for i in range(len):
                fac_new = fac*self._addends[index_ScLO[0][i]]._scalar                    
                #check that fac*._scalar != 1!
                if isinstance(self._addends[index_ScLO[0][i]]._operator,SumLinearOperator):
                    [mat,out] = self._addends[index_ScLO[0][i]]._operator.dot(v,mat=mat,fac=fac_new,out=out)
                elif self._addends[index_ScLO[0][i]]._operator._matrix is not None:
                    mat += np.multiply(fac_new,self._addends[index_ScLO[0][i]]._operator._matrix)
                else:
                    out._data += np.multiply(fac_new,self._addends[index_ScLO[0][i]]._operator.dot(v)._data)

        if level == 0:
            if not(np.array_equal(mat,np.zeros((self._codomain._dim,self._domain._dim),dtype=float))):
                out._data += np.dot(mat,v._data) # iadd
            return out
        elif level == 1:
            return [mat,out]

class ConvLinearOperator( NdarrayLinearOperator ):
    def __init__(self, A, B):
        assert isinstance(A, NdarrayLinearOperator)
        assert isinstance(B, NdarrayLinearOperator)
        assert A._domain == B._codomain
        NdarrayLinearOperator.__init__(self, domain=B._domain, codomain=A._codomain)
        if isinstance(A,ConvLinearOperator):
            multiplicants = A._multiplicants
            if isinstance(B,ConvLinearOperator):
                multiplicants = np.append(B._multiplicants, multiplicants)
            else:
                multiplicants = np.append(B, multiplicants)
        elif isinstance(B,ConvLinearOperator):
            multiplicants = np.append(B._multiplicants, A)
        else:
            multiplicants = np.array([B,A])

        self._multiplicants = multiplicants

    @property
    def multiplicants( self ):
        return self._multiplicants

    @property
    def length( self ):
        return np.size(self._multiplicants)

    def dot( self, v ):
        assert isinstance(v,NdarrayVector)
        assert v.space == self._domain
        len = self.length
        out = self._multiplicants[0].dot(v)
        for i in range(1,len):
            out = self._multiplicants[i].dot(out)
        return out

class ScalLinearOperator( NdarrayLinearOperator ):
    def __init__( self, c, A ):
        assert np.isscalar(c)
        assert isinstance(A, NdarrayLinearOperator)
        NdarrayLinearOperator.__init__(self, domain=A._domain, codomain=A._codomain)
        if isinstance(A,ScalLinearOperator):
            scalar = A._scalar*c
        else:
            scalar = c
        self._operator = A
        self._scalar = scalar

    @property
    def scalar( self ):
        return self._scalar

    def dot( self, v, mat=None, fac=1, out=None ):   # INSTEAD RAISE ERROR
        assert isinstance(v,NdarrayVector)
        assert v.space == self._domain        
        out = self._operator.dot(v, mat=mat, fac=fac*self._scalar, out=out)
        return out

class ZeroOperator( NdarrayLinearOperator ):
    def __init__(self, domain=None, codomain=None, matrix=None):

        NdarrayLinearOperator.__init__(self, domain=domain, codomain=codomain, matrix=matrix)

    def dot( self, v, mat=None, fac=None, out=None ):#, fac=None ):
        assert isinstance(v, NdarrayVector)
        assert v.space == self. domain
        return NdarrayVector(space=self._codomain)

    def idot( self, v, out ):
        assert isinstance(v, NdarrayVector)
        assert v.space == self. domain
        assert isinstance(out, NdarrayVector)
        assert out.space == self._codomain
        return out

class IdOperator( NdarrayLinearOperator ):
    def __init__(self, domain=None, codomain=None, matrix=None):

        NdarrayLinearOperator.__init__(self, domain=domain, codomain=domain, matrix=matrix)

    def dot( self, v, mat=None, fac=None, out=None ):#, fac=1 ):
        assert isinstance(v, NdarrayVector)
        assert v.space == self. domain
        return v

    def idot( self, v, out ):#, fac=None ):
        assert isinstance(v, NdarrayVector)
        assert v.space == self. domain
        assert isinstance(out, NdarrayVector)
        assert out.space == self._codomain
        return v+out