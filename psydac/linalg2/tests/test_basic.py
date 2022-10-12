
from multiprocessing.sharedctypes import Value
import numpy as np
from psydac.linalg2.basic import VectorSpace, Vector, LinearOperator
from abc   import ABCMeta, abstractmethod
from numpy import ndarray

class NdarrayVectorSpace( VectorSpace ):
    """
    space of real ndarrays, dtype only allowing float right now
    """
    def __init__( self, dim, dtype=float ):
        self._dim = dim
        self._dtype = dtype

    @property
    def dimension( self ):
        return self._dim

    @property
    def dtype( self ):
        return self._dtype

    def zeros( self ):
        return NdarrayVector(space=self)

    def ones( self ):
        return NdarrayVector(space=self, data=np.ones(self._dim, dtype=self._dtype))

class NdarrayVector( Vector ):
    def __init__( self, space, data=None ):
        
        self._space = space        

        if data is None:
            self._data = np.zeros(space.dimension, dtype=space.dtype)
        elif isinstance(data, np.ndarray):
            assert data.shape == (space.dimension, ), f"data neither scalar nor of right dimension"
            assert data.dtype == space.dtype, f"data of wrong dtype"                                     
            self._data = data
        elif np.isscalar(data):
            self._data = np.full(shape=space.dimension, fill_value=data, dtype=space.dtype)
        else:
            raise ValueError(data)

    @property
    def space( self ):
        return self._space

    @property
    def dtype( self ):
        return self._dtype

    def dot( self, v ):
        assert isinstance(v, NdarrayVector), f"v is not a NdarrayVector"
        assert self.space is v.space, f"v and self dont belong to the same space"
        return np.dot(self._data, v._data)

    def __mul__( self, a ):
        assert np.isscalar(a), f"a is not a scalar"
        return NdarrayVector(space=self._space, data=np.multiply(self._data,a))

    def __rmul__( self, a ):
        assert np.isscalar(a), f"a is not a scalar"
        return NdarrayVector(space=self._space, data=np.multiply(self._data,a))

    def __add__( self, v ):
        assert isinstance(v, NdarrayVector), f"v is not NdarrayVector"
        assert self.space is v.space, f"v space is not self space"
        return NdarrayVector(space=self._space, data=self._data+v._data)

class NdarrayLinearOperator( LinearOperator ):
    def __init__( self, domain=None, codomain=None, matrix=None ):

        assert domain
        assert isinstance(domain,NdarrayVectorSpace)
        self._domain = domain
        if codomain:
            assert isinstance(codomain,NdarrayVectorSpace)
            self._codomain = codomain
        else:
            self._codomain = domain
        if matrix is not None:
            assert np.shape(matrix)[1] == self._domain.dimension
            assert np.shape(matrix)[0] == self._codomain.dimension
            self._matrix = matrix
            self._dtype = matrix.dtype
        else:
            self._matrix = None

    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def dtype( self ):
        if self._matrix is not None:
            return self._dtype
        else:
            raise NotImplementedError('Class does not provide a dtype method without a matrix')

    def dot( self, v ):
        assert isinstance(v, NdarrayVector)
        assert v.space == self.domain
        if self._matrix is not None:
            return NdarrayVector(space=self._codomain, data=np.dot(self._matrix,v._data))
        else:
            raise NotImplementedError('Class does not provide a dot() method without a matrix')

    def __add__( self, B ):
        return SumLinearOperator(self,B)

    def __matmul__( self, B ):
        return ConvLinearOperator(self,B)

    def __mul__( self, c ):
        assert np.isscalar(c)
        if c==0:
            return ZeroOperator(domain=self._domain, codomain=self._codomain)
        elif c == 1:
            return self
        else:
            return ScalLinearOperator(c, self)

    def __rmul__( self, c ):
        assert np.isscalar(c)
        if c==0:
            return ZeroOperator(domain=self._domain, codomain=self._codomain)
        elif c == 1:
            return self
        else:
            return ScalLinearOperator(c, self)

    #-------------------------------------
    # Methods with default implementation
    #-------------------------------------
    def idot( self, v, out ):
        assert isinstance(v, NdarrayVector)
        assert v.space == self._domain
        if out is not None:
            assert isinstance(out, NdarrayVector)
            assert out.space == self._codomain
            out += self.dot(v)
            return out
        else:
            return self.dot(v)

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

#===============================================================================
if __name__ == "__main__":
    """
    # NdarrayVector and NdarrayVectorSpace __init__
    V = NdarrayVectorSpace(dim=6, dtype=float)
    a = NdarrayVector(space=V)
    b = NdarrayVector(space=V, data=1)
    c = NdarrayVector(space=V, data=np.array([1,2,3,4,5,6], dtype=float))
    try:
        NdarrayVector(space=V, data=np.array([1,2,3,4,5,6,7], dtype=float))
    except AssertionError as msg:
        print('NdarrayVector: Error if wrong dimension (too high)')
        print('data neither scalar nor of right dimension' == str(msg))
    try:
        NdarrayVector(space=V, data=np.array([1,2], dtype=float))
    except AssertionError as msg:
        print('NdarrayVector: Error if wrong dimension (too low)')
        print('data neither scalar nor of right dimension' == str(msg))
    try:
        NdarrayVector(space=V, data=np.array([1,2,3,4,5,6], dtype=int))
    except AssertionError as msg:
        print('NdarrayVector: Error if wrong data type')
        print('data of wrong dtype' == str(msg))
    
    W = NdarrayVectorSpace(dim=6, dtype=int)
    x = NdarrayVector(space=W, data=1)
    y = np.array([2,3,2,3,2,3], dtype=float)

    # dot
    b.dot(c)
    try:
        b.dot(x)
    except AssertionError as msg:
        print('NdarrayVector.dot: Error if wrong space')
        print('v and self dont belong to the same space' == str(msg))

    try:
        b.dot(y)
    except AssertionError as msg:
        print('NdarrayVector.dot: Error if np.array instead of NdarrayVector')
        print('v is not a NdarrayVector' == str(msg))

    # __mul__
    u = 3*x
    v = x*3
    print('NdarrayVector.__mul__: Scalar multiplication from left and right works as expected')
    print(u._data == v._data)
    print(u._data == 3*np.ones(6, dtype=int))
    try:
        x*x
    except AssertionError as msg:
        print('NdarrayVector.__mul__: Error if not scalar')
        print('a is not a scalar' == str(msg))

    # __add__
    a+b
    try:
        c+y
    except AssertionError as msg:
        print('NdarrayVector.__add__: Error if not NdarrayVector')
        print('v is not NdarrayVector' == str(msg))

    try:
        a+x
    except AssertionError as msg:
        print('NdarrayVector.__add__: Error if wrong space')
        print('v space is not self space' == str(msg))

    # NdarrayLinearOperator __init__
    U = NdarrayVectorSpace( dim = 2 )
    u = NdarrayVector(U,1)
    A = NdarrayLinearOperator( domain = V, codomain = U, matrix = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]], dtype=float) )
    Z = ZeroOperator( domain = V, codomain = U )
    print(A.dot(b)._data == np.array([1,1], dtype=float))
    print(A.idot(b,u)._data == np.array([2,2], dtype=float))
    
    S = A+Z
    print('Basic SumLinearOperator Test:')
    S1 = S.addends[0]
    S2 = S.addends[1]
    print(S1.dot(b)._data == np.ones(2, dtype=float))
    print(S1.idot(b,u)._data == np.array([2,2], dtype=float))
    print(S2.dot(b)._data == np.zeros(2, dtype=float))
    print(S2.idot(b,u)._data == np.ones(2, dtype=float))
    print(S.domain)
    print(S.codomain)
    print(S.shape == (2,6))
    print(S.addends[0]._matrix)
    print(S.addends[1]._matrix)
    print(S.length)
    index = np.where([S._addends[i]._matrix is not None for i in range(S.length) ])
    print(index[0])

    try:
        S.dtype
    except NotImplementedError as msg:
        print('Class does not provide a dtype method without a matrix' == str(msg))

    #T = A+A
    #print(isinstance(T,SumLinearOperator))
    #print(T.length)
    #print("B is SLO")
    #R = A+T
    #print(isinstance(R,SumLinearOperator))
    #print(R.length)
    #print("A is SLO")
    #H = T+A
    #print(isinstance(H,SumLinearOperator))
    #print(H.length)
    #T = A+A+Z+A
    #print("First SLO.dot test:")
    #print(T.dot(b)._data)
    #print(T.length)
    """
    print('Implementation Test:')
    print()

    print('1. Creating VectorSpaces:')
    U = NdarrayVectorSpace(1)
    V = NdarrayVectorSpace(2)
    W = NdarrayVectorSpace(3)
    print('VectorSpaces U,V,W of dim 1,2,3 successfully created.')
    print()

    print('2. Creating LOs from V->W with matrix representation:')
    a = np.array([[1,0],[0,1],[1,1]], dtype=float)
    b = np.array([[0,2],[2,0],[3,3]], dtype=float)
    A = NdarrayLinearOperator(domain=V, codomain=W, matrix=a)
    B = NdarrayLinearOperator(domain=V, codomain=W, matrix=b)
    print('Successfully created LOs A and B, matrix.dimesion = (3,2).')
    print()

    print('3. Creating LOs from V->W without matrix representation:')
    Z = ZeroOperator(domain=V, codomain=W)
    I1 = IdOperator(V)
    I2 = IdOperator(W)
    print('Sucessfully created three LOs without matrix representation, namely two IdOperator and one ZeroOperator.')
    print()

    print('4. Creating convolutions of LOs from V->W:')
    c = np.array([[1],[0],[1]], dtype=float)
    d = np.array([[1,-1]], dtype=float)
    e = np.array([[1,-1],[0,-2],[3,0]], dtype=float)
    f = np.array([[0,1],[-1,0]], dtype=float)
    C = NdarrayLinearOperator(domain=U, codomain=W, matrix=c)
    D = NdarrayLinearOperator(domain=V, codomain=U, matrix=d)    
    E = NdarrayLinearOperator(domain=V, codomain=W, matrix=e)
    F = NdarrayLinearOperator(domain=V, codomain=V, matrix=f)
    G = C@D@I1
    H = I2@E@I1@F
    print('Convolution LOs G and H have been successfully created, both including identity operators without matrix representation')
    print()

    print('5. Testing both creation of ~arbitrary~ combinations of the given operators A,B,Z,I1,I2,G,H as well as')
    print('the evaluation of such operator at v=(1,1):')
    print()
    T1 = 2*(A+Z+G+Z+H+B)
    print('5.1 Successfully created operator T1 = 2*(A+Z+G+Z+H+B)')
    v = NdarrayVector(space=V, data=np.ones(2,dtype=float))
    print('    Successfully created vector (1,1) of space V')
    y1 = T1.dot(v)
    print('    Successfully evaluated T1.dot(v) if [ True True True]')
    print(y1._data==np.array([10,10,22],dtype=float))
    print()
    T2 = 1*(T1+0*T1) + 2*0.5*I2@T1 + Z@(I1@I1+I1)
    print('5.2 Successfully created operator T2 = 1*(T1+0*T1) + 2*0.5*Id2@T1 + Z1@(Id1@Id1+Id1)')
    y2 = T2.dot(v)
    print('    Successfully evaluated T2.dot(v) if [ True True True]')
    print(y2._data==np.array([20,20,44],dtype=float))
    print()