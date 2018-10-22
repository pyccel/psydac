# coding: utf-8
from collections        import OrderedDict

from spl.linalg.basic   import VectorSpace, Vector, LinearOperator

__all__ = [ 'ProductSpace', 'BlockVector', 'BlockLinearOperator']

#===============================================================================
class ProductSpace(VectorSpace ):
    """
    Product Vector Space V of two Vector Spaces or more.


    Parameters
    ----------
    list_spaces : list
        A list of Vector Spaces (spl.linalg.basic.VectorSpace)

    """

    def __init__( self,  *args ):
        assert isinstance(args[0], list)
        self._list_spaces  = args[0]
        for V in self._list_spaces:
            if V is not None:
                assert isinstance( V, VectorSpace )

        self._dim   = len(self._list_spaces)

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def dimension( self ):
        """
        The dimension of a product space V= [V1, V2, ...] is the cardinality
        (i.e. the number of vectors) of a basis of V over its base field.

        """
        return sum(self._list_spaces[:].dimension())

    def zeros( self ):
        """
        Get a copy of the null element of the product space [V1, V2, ...]

        Returns
        -------
        null : ProductVector
            A new vector object with all components equal to zero.

        """

        return ProductSpace(self._list_spaces)

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------
    @property
    def list_spaces( self ):
        return self._list_spaces

    @property
    def dim( self ):
        return self._dim

    #...
    def __eq__( self, block_space ):
        assert self._dim == block_space.dim
        res = True
        for i in range(self._dim):
            res = res and  self._block_list[i] == block_space._list[i]

        return res

#===============================================================================
class BlockVector( Vector ):
    """
    Block of Vectors

    Parameters
    ----------
    list_vectors : list
        List of Vectors.

    n_block : integer
        Number of blocks.
    """
    def __init__( self,  *args ):
        # ... Data are given via a list of Vectors
        if  isinstance(args[0], list):
            self._block_list = args[0]
            self._n_blocks = len(self._block_list)

            l_spaces = []
            for i in range(self._n_blocks):
                l_spaces = l_spaces + [self._block_list[i].space]
            self._space = ProductSpace(l_spaces)
        # ...
        elif isinstance(args[0], int):
            self._n_blocks = args[0]
            self._block_list = [None]*self._n_blocks
            self._space = ProductSpace([None]*self._n_blocks)

        else:
            raise TypeError('Unexpected argument.')

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def space( self ):
        return self._space

    # ...
    @property
    def block_list( self ):
        return self._block_list

    # ...
    @property
    def n_blocks( self ):
        return self._n_blocks

    #...
    def dot( self, v ):
        assert isinstance( v, BlockVector )
        assert v._n_blocks == self._n_blocks

        res = 0.
        for i in range(self._n_blocks):
            assert v._space.list_spaces[i] is self._space.list_spaces[i]
            res = res + self._block_list[i].dot(v._block_list[i])

        return res

    #...
    def copy( self ):
        w = BlockVector( self._block_list )
        return w

    #...
    def __mul__( self, a ):
        w_block_list = [None]*self._n_blocks
        for i in range(self._n_blocks):
            w_block_list[i] = self._block_list[i]*a

        w = BlockVector( w_block_list )
        return w

    #...
    def __rmul__( self, a ):
        w_block_list = [None]*self._n_blocks
        for i in range(self._n_blocks):
            w_block_list[i] =  a*self._block_list[i]

        w = BlockVector( w_block_list )
        return w

    #...
    def __add__( self, v ):
        assert isinstance( v, BlockVector )
        assert v._n_blocks == self._n_blocks

        w_block_list = [None]*self._n_blocks
        for i in range(self._n_blocks):
            assert v._space.list_spaces[i] is self._space.list_spaces[i]
            w_block_list[i] = self._block_list[i] + v._block_list[i]

        w = BlockVector(w_block_list)
        return w

    #...
    def __sub__( self, v ):
        assert isinstance( v, BlockVector )
        assert v._n_blocks == self._n_blocks

        w_block_list = [None]*self._n_blocks
        for i in range(self._n_blocks):
            assert v._space.list_spaces[i] is self._space.list_spaces[i]
            w_block_list[i] = self._block_list[i] - v._block_list[i]

        w = BlockVector(w_block_list)
        return w

    #...
    def __imul__( self, a ):
        self._data *= a
        for i in range(self._n_blocks):
            self._block_list[i] *= a

        return self

    #...
    def __iadd__( self, v ):
        assert isinstance( v, BlockVector )
        assert v._n_blocks == self._n_blocks
        for i in range(self._n_blocks):
            assert v._space.list_spaces[i] is self._space.list_spaces[i]
            self._block_list[i] += v.block_list[i]

        return self

    #...
    def __isub__( self, v ):
        assert isinstance( v, BlockVector )
        assert v._n_blocks == self._n_blocks
        for i in range(self._n_blocks):
            assert v._space.list_spaces[i] is self._space.list_spaces[i]
            self._block_list[i] -= v.block_list[i]

        return self

#===============================================================================
class BlockLinearOperator(LinearOperator):
    """
    Linear operator that can be written as blocks of other Linear Operators.

    Parameters
    ----------
    block_dict : collection.OrderedDict
        key   = tuple (i, j), i and j are two integers >= 0.
        value = corresponding LinearOperator Lij.

    n_block_rows : integer
        Number of row blocks.

    n_block_cols : integer
        Number of column blocks.

    """

    def __init__(self, *args):

        # ... Data are given via a dictionary
        if len(args) == 1:

            assert isinstance(args[0], dict)
            self._block_dict = OrderedDict(args[0])

            # TODO assert keys are integers >=0.
            row_min = min(self._block_dict.keys(), key=lambda k: k[0])[0]
            row_max = max(self._block_dict.keys(), key=lambda k: k[0])[0]
            col_min = min(self._block_dict.keys(), key=lambda k: k[1])[1]
            col_max = max(self._block_dict.keys(), key=lambda k: k[1])[1]

            self._n_block_rows = row_max - row_min + 1
            self._n_block_cols = col_max - col_min + 1

            self._domain   = ProductSpace([None] * self._n_block_cols)
            self._codomain = ProductSpace([None] * self._n_block_rows)

            for ij, Lij in self._block_dict.items():
                # ... Check spaces
                if isinstance(Lij, LinearOperator):
                    i = ij[0]
                    j = ij[1]

                    if self.domain._list_spaces[j] is None:
                        self.domain._list_spaces[j] = Lij.domain
                    else:
                        assert self.domain._list_spaces[j] == Lij.domain

                    if self.codomain._list_spaces[i] is None:
                        self.codomain._list_spaces[i] = Lij.codomain
                    else:
                        assert self.codomain._list_spaces[i] == Lij.codomain
                else:
                    raise TypeError('Unexpected type.')

        # ...  Data structure is initialised by the given block rows and cols
        elif len(args) == 2:
            n_block_rows = args[0]
            n_block_cols = args[1]

            assert n_block_rows > 0 and  n_block_rows > 0

            self._n_block_rows  = n_block_rows
            self._n_block_cols  = n_block_cols
            self._domain   = ProductSpace([None] * n_block_cols)
            self._codomain = ProductSpace([None] * n_block_rows)

            self._block_dict = OrderedDict({})
        else:
            raise TypeError('Unexpected argument.')


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
    def block_dict( self ):
        return self._block_dict

    # ...
    def dot( self, v, out=None ):
        assert isinstance( v, BlockVector )
        pass

    # ...
    def __getitem__(self, key):
        if isinstance( key, tuple ):
            assert len(key) == 2
        else:
            raise TypeError('A tuple is expected.')

        i = key[0]
        j = key[1]
        assert 0 <= i < self._n_block_rows
        assert 0 <= j < self._n_block_cols

        return self._block_dict[i, j]

    # ...
    def __setitem__(self, key, value):
        if isinstance( key, tuple ):
            assert len(key) == 2
        else:
            raise TypeError('A tuple is expected.')

        i = key[0]
        j = key[1]
        assert 0 <= i < self._n_block_rows
        assert 0 <= j < self._n_block_cols

        if isinstance(value, LinearOperator):
            if self.domain._list_spaces[i] is None:
                self.domain._list_spaces[i] = value.domain
            else:
                assert self.domain._list_spaces[i] == value.domain

            if self.codomain._list_spaces[i] is None:
                self.codomain._list_spaces[i] = value.codomain
            else:
                assert self.codomain._list_spaces[i] == value.codomain

            self._block_dict[i,j] = value
        else:
            raise TypeError('Unexpected argument.')

#===============================================================================
