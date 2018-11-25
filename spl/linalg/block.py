# coding: utf-8
#
# Copyright 2018 Jalal Lakhlili, Yaman Güçlü

from collections        import OrderedDict

from spl.linalg.basic   import VectorSpace, Vector, LinearOperator
from spl.linalg.stencil import StencilMatrix

__all__ = ['ProductSpace', 'BlockVector', 'BlockLinearOperator', 'BlockMatrix']

#===============================================================================
class ProductSpace( VectorSpace ):
    """
    Product Vector Space V of two Vector Spaces (V1,V2) or more.

    Parameters
    ----------
    *spaces : spl.linalg.basic.VectorSpace
        A list of Vector Spaces.

    """
    def __init__( self,  *spaces ):

        assert all( isinstance( Vi, VectorSpace ) for Vi in spaces )

        # We store the spaces in a Tuple because they will not be changed
        self._spaces = tuple( spaces )

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def dimension( self ):
        """
        The dimension of a product space V = (V1, V2, ...] is the cardinality
        (i.e. the number of vectors) of a basis of V over its base field.

        """
        return sum( Vi.dimension for Vi in self._spaces )

    # ...
    def zeros( self ):
        """
        Get a copy of the null element of the product space V = [V1, V2, ...]

        Returns
        -------
        null : BlockVector
            A new vector object with all components equal to zero.

        """
        return BlockVector( self, blocks=[Vi.zeros() for Vi in self._spaces] )

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------
    @property
    def spaces( self ):
        return self._spaces

    @property
    def n_blocks( self ):
        return len( self._spaces )

    def __getitem__( self, key ):
        return self._spaces[key]

#===============================================================================
class BlockVector( Vector ):
    """
    Block of Vectors, which is an element of a ProductSpace.

    Parameters
    ----------
    V : spl.linalg.block.ProductSpace
        Space to which the new vector belongs.

    blocks : list or tuple (spl.linalg.basic.Vector)
        List of Vector objects, belonging to the correct spaces (optional).

    """
    def __init__( self,  V, blocks=None ):

        assert isinstance( V, ProductSpace )
        self._space = V

        # We store the blocks in a List so that we can change them later.
        if blocks:
            self.set_blocks( blocks )
        else:
            # TODO: Each block is a 'zeros' vector of the correct space for now,
            # but in the future we would like 'empty' vectors of the same space.
            self._blocks = [Vi.zeros() for Vi in V.spaces]

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def space( self ):
        return self._space

    #...
    def dot( self, v ):

        assert isinstance( v, BlockVector )
        assert v._space is self._space

        return sum( b1.dot( b2 ) for b1,b2 in zip( self._blocks, v._blocks ) )

    #...
    def copy( self ):
        return BlockVector( self._space, [b.copy() for b in self._blocks] )

    #...
    def __mul__( self, a ):
        return BlockVector( self._space, [b*a for b in self._blocks] )

    #...
    def __rmul__( self, a ):
        return BlockVector( self._space, [a*b for b in self._blocks] )

    #...
    def __add__( self, v ):
        assert isinstance( v, BlockVector )
        assert v._space is self._space
        return BlockVector( self._space, [b1+b2 for b1,b2 in zip( self._blocks, v._blocks )] )

    #...
    def __sub__( self, v ):
        assert isinstance( v, BlockVector )
        assert v._space is self._space
        return BlockVector( self._space, [b1-b2 for b1,b2 in zip( self._blocks, v._blocks )] )

    #...
    def __imul__( self, a ):
        for b in self._blocks:
            b *= a
        return self

    #...
    def __iadd__( self, v ):
        assert isinstance( v, BlockVector )
        assert v._space is self._space
        for b1,b2 in zip( self._blocks, v._blocks ):
            b1 += b2
        return self

    #...
    def __isub__( self, v ):
        assert isinstance( v, BlockVector )
        assert v._space is self._space
        for b1,b2 in zip( self._blocks, v._blocks ):
            b1 -= b2
        return self

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------
    def __getitem__( self, key ):
        return self._blocks[key]

    # ...
    def __setitem__( self, key, value ):
        assert value.space == self.space[key]
        self._blocks[key] = value

    # ...
    @property
    def n_blocks( self ):
        return len( self._blocks )

    # ...
    @property
    def blocks( self ):
        return tuple( self._blocks )

    # ...
    def set_blocks( self, blocks ):
        """
        Parameters
        ----------
        blocks : list or tuple (spl.linalg.basic.Vector)
        List of Vector objects, belonging to the correct spaces.
        """

        assert isinstance( blocks, (list, tuple) )
        V = self.space
        # Verify that vectors belong to correct spaces and store them
        assert all( (Vi is b.space) for Vi,b in zip( V.spaces, blocks ) )
        self._blocks = list( blocks )

#===============================================================================
class BlockLinearOperator(LinearOperator):
    """
    Linear operator that can be written as blocks of other Linear Operators.

    Parameters
    ----------
    V1 : spl.linalg.block.ProductSpace
        Domain of the new linear operator.

    V2 : spl.linalg.block.ProductSpace
        Codomain of the new linear operator.

    block : dict
        key   = tuple (i, j), i and j are two integers >= 0.
        value = corresponding LinearOperator Lij (belonging to the correct spaces).
        (optional).
    """

    def __init__(self, V1, V2, blocks=None):
        assert isinstance( V1, ProductSpace )
        assert isinstance( V2, ProductSpace )

        self._domain   = V1
        self._codomain = V2

        # We store the blocks in a OrderedDict  (that we can change them later).
        if blocks:
            self.set_blocks(blocks)
        else:
            self._blocks = OrderedDict({})

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
    def blocks( self ):
        return self._blocks

    # ...
    @property
    def n_block_rows( self ):
        return self._codomain.n_blocks

    # ...
    @property
    def n_block_cols( self ):
        return self._domain.n_blocks

    # ...
    def set_blocks(self, blocks):
        """
        Parameters
        ----------
        block : dict
            key   = tuple (i, j), i and j are two integers >= 0.
            value = corresponding LinearOperator Lij (belonging to the correct spaces).
            (optional).
        """

        # Verify that blocks belong to correct spaces and store them
        assert isinstance( blocks, dict )
        for (i,j), Lij in blocks.items():
            assert isinstance( Lij, LinearOperator )
            assert Lij.domain   is self.domain  [j]
            assert Lij.codomain is self.codomain[i]

        self._blocks = OrderedDict( blocks )

    # ...
    def dot( self, v, out=None ):
        assert isinstance( v, BlockVector )
        assert v.space is self._codomain
        assert all( v.blocks )

        if out is not None:
            assert isinstance( out, BlockVector )
            assert out.space is self._domain
            out *= 0.0
        else:
            out = BlockVector( self._domain )

        for (i,j), Lij in self._blocks.items():
            out[i] += Lij.dot( v[j] )

        return out

    # ...
    def __getitem__(self, key):
        return self._blocks[key]

    # ...
    def __setitem__(self, key, value):

        if isinstance( key, tuple ):
            assert len(key) == 2
        else:
            raise TypeError('A tuple is expected.')

        i = key[0]
        j = key[1]
        assert 0 <= i < self.n_block_rows
        assert 0 <= j < self.n_block_cols

        assert isinstance( value, LinearOperator )
        assert value.domain   is self.domain  [j]
        assert value.codomain is self.codomain[i]

        self._blocks[i,j] = value

#===============================================================================

#===============================================================================
# TODO  allow numpy and sparse scipy matrices
class BlockMatrix( BlockLinearOperator ):
    """
    Linear operator that can be written as blocks of Stencil Matrices.

    Parameters
    ----------
    V1 : spl.linalg.block.ProductSpace
        Domain of the new linear operator.

    V2 : spl.linalg.block.ProductSpace
        Codomain of the new linear operator.

    block : dict
        key   = tuple (i, j), i and j are two integers >= 0.
        value = corresponding StencilMatrix Mij (belonging to the correct spaces).
        (optional).
    """

    def __init__(self, V1, V2, blocks=None):
        if blocks:
            for M in blocks.values():
                if not isinstance(M, StencilMatrix):
                    raise typeerror('>>> Expecting a StencilMatrix')

        BlockLinearOperator.__init__(self, V1, V2, blocks)

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------

    # ...
    def __setitem__( self, key, value ):

        if isinstance(value, StencilMatrix):
            assert value.domain   is self.domain.spaces[key[1]]
            assert value.codomain is self.codomain.spaces[key[0]]
            self._blocks[key] = value
        else:
            raise typeerror('>>> Expecting a StencilMatrix')

    #  ...
    def tocoo(self):
        from numpy import zeros
        from scipy.sparse import coo_matrix

        # ...
        n_block_rows = self.n_block_rows
        n_block_cols = self.n_block_cols

        matrices = {}
        for k, M in list(self.blocks.items()):
            if isinstance( M, StencilMatrix ):
                matrices[k] = M.tocoo()
            else:
                raise NotImplementedError('TODO')
        # ...

        # ... compute the global nnz
        nnz = 0
        for i in range(0, n_block_rows):
            for j in range(0, n_block_cols):
                nnz += matrices[i,j].nnz
        # ...

        # ... compute number of rows and cols per block
        n_rows = zeros(n_block_rows, dtype=int)
        n_cols = zeros(n_block_cols, dtype=int)

        for i in range(0, n_block_rows):
            n = 0
            for j in range(0, n_block_cols):
                if not(matrices[i,j] is None):
                    n = matrices[i,j].shape[0]
                    break
            if n == 0:
                raise ValueError('At least one block must be non empty per row')
            n_rows[i] = n

        for j in range(0, n_block_cols):
            n = 0
            for i in range(0, n_block_rows):
                if not(matrices[i,j] is None):
                    n = matrices[i,j].shape[1]
                    break
            if n == 0:
                raise ValueError('At least one block must be non empty per col')
            n_cols[j] = n
        # ...

        # ...
        data = zeros(nnz)
        rows = zeros(nnz, dtype=int)
        cols = zeros(nnz, dtype=int)
        # ...

        # ...
        n = 0
        for ir in range(0, n_block_rows):
            for ic in range(0, n_block_cols):
                if not(matrices[ir,ic] is None):
                    A = matrices[ir,ic]

                    n += A.nnz

                    shift_row = 0
                    if ir > 0:
                        shift_row = sum(n_rows[:ir])

                    shift_col = 0
                    if ic > 0:
                        shift_col = sum(n_cols[:ic])

                    rows[n-A.nnz:n] = A.row[:] + shift_row
                    cols[n-A.nnz:n] = A.col[:] + shift_col
                    data[n-A.nnz:n] = A.data
        # ...

        # ...
        nr = n_rows.sum()
        nc = n_cols.sum()

        coo = coo_matrix((data, (rows, cols)), shape=(nr, nc))
        coo.eliminate_zeros()
        # ...

        return coo
    # ...
#===============================================================================
