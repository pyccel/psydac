# coding: utf-8
#
# Copyright 2018 Jalal Lakhlili, Yaman Güçlü

import numpy as np
from collections        import OrderedDict
from scipy.sparse       import bmat

from psydac.linalg.basic   import VectorSpace, Vector, LinearOperator, Matrix

__all__ = ['ProductSpace', 'BlockVector', 'BlockLinearOperator', 'BlockMatrix']

#===============================================================================
class ProductSpace( VectorSpace ):
    """
    Product Vector Space V of two Vector Spaces (V1,V2) or more.

    Parameters
    ----------
    *spaces : psydac.linalg.basic.VectorSpace
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
        return BlockVector( self, [Vi.zeros() for Vi in self._spaces] )

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
    V : psydac.linalg.block.ProductSpace
        Space to which the new vector belongs.

    blocks : list or tuple (psydac.linalg.basic.Vector)
        List of Vector objects, belonging to the correct spaces (optional).

    """
    def __init__( self,  V, blocks=None ):

        assert isinstance( V, ProductSpace )
        self._space = V

        # We store the blocks in a List so that we can change them later.
        if blocks:
            # Verify that vectors belong to correct spaces and store them
            assert isinstance( blocks, (list, tuple) )
            assert all( (Vi is bi.space) for Vi,bi in zip( V.spaces, blocks ) )
            self._blocks = list( blocks )
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
    def toarray( self ):
        return np.concatenate( [bi.toarray() for bi in self._blocks] )

#===============================================================================
class BlockLinearOperator( LinearOperator ):
    """
    Linear operator that can be written as blocks of other Linear Operators.

    Parameters
    ----------
    V1 : psydac.linalg.block.ProductSpace
        Domain of the new linear operator.

    V2 : psydac.linalg.block.ProductSpace
        Codomain of the new linear operator.

    blocks : dict | (list of lists) | (tuple of tuples)
        LinearOperator objects (optional).

        a) 'blocks' can be dictionary with
            . key   = tuple (i, j), where i and j are two integers >= 0
            . value = corresponding LinearOperator Lij

        b) 'blocks' can be list of lists (or tuple of tuples) where blocks[i][j]
            is the LinearOperator Lij (if None, we assume null operator)

    """
    def __init__( self, V1, V2, blocks=None ):

        assert isinstance( V1, ProductSpace )
        assert isinstance( V2, ProductSpace )

        self._domain   = V1
        self._codomain = V2
        self._blocks   = OrderedDict()

        # Store blocks in OrderedDict (hence they can be manually changed later)
        if blocks:

            if isinstance( blocks, dict ):
                for (i,j), Lij in blocks.items():
                    self[i,j] = Lij

            elif isinstance( blocks, (list, tuple) ):
                blocks = np.array( blocks, dtype=object )
                for (i,j), Lij in np.ndenumerate( blocks ):
                    self[i,j] = Lij

            else:
                raise ValueError( "Blocks can only be given as dict or 2D list/tuple." )

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
    def dot( self, v, out=None ):

        assert isinstance( v, BlockVector )
        
        for i in range(len(v.space[:])):
            assert v.space[i] is self._domain[i]
            
        assert all( v.blocks )

        if out is not None:
            assert isinstance( out, BlockVector )
            assert out.space is self._codomain
            out *= 0.0
        else:
            out = BlockVector( self._codomain )

        for (i,j), Lij in self._blocks.items():
            out[i] += Lij.dot( v[j] )

        return out

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------
    @property
    def blocks( self ):
        """ Immutable 2D view (tuple of tuples) of the linear operator,
            including the empty blocks as 'None' objects.
        """
        return tuple(
               tuple( self._blocks.get( (i,j), None ) for j in range( self.n_block_cols ) )
                                                      for i in range( self.n_block_rows ) )

    # ...
    @property
    def n_block_rows( self ):
        return self._codomain.n_blocks

    # ...
    @property
    def n_block_cols( self ):
        return self._domain.n_blocks

    # ...
    def __getitem__( self, key ):

        assert isinstance( key, tuple )
        assert len( key ) == 2
        assert 0 <= key[0] < self.n_block_rows
        assert 0 <= key[1] < self.n_block_cols

        return self._blocks.get( key, None )

    # ...
    def __setitem__( self, key, value ):

        assert isinstance( key, tuple )
        assert len( key ) == 2
        assert 0 <= key[0] < self.n_block_rows
        assert 0 <= key[1] < self.n_block_cols

        if value is None:
            self._blocks.pop( key, None )
            return

        i,j = key
        assert isinstance( value, LinearOperator )
        assert value.domain   == self.domain  [j]
        assert value.codomain == self.codomain[i]

        self._blocks[i,j] = value

#===============================================================================
class BlockMatrix( BlockLinearOperator, Matrix ):
    """
    Linear operator that can be written as blocks of other Linear Operators,
    with the additional capability to be converted to a 2D Numpy array
    or to a Scipy sparse matrix.

    Parameters
    ----------
    V1 : psydac.linalg.block.ProductSpace
        Domain of the new linear operator.

    V2 : psydac.linalg.block.ProductSpace
        Codomain of the new linear operator.

    blocks : dict | (list of lists) | (tuple of tuples)
        Matrix objects (optional).

        a) 'blocks' can be dictionary with
            . key   = tuple (i, j), where i and j are two integers >= 0
            . value = corresponding Matrix Mij

        b) 'blocks' can be list of lists (or tuple of tuples) where blocks[i][j]
            is the Matrix Mij (if None, we assume all entries are zeros)

    """
    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    def toarray( self ):
        """ Convert to 2D Numpy array.
        """
        return self.tosparse().toarray()

    # ...
    def tosparse( self ):
        """ Convert to Scipy sparse matrix.
        """
        # Convert all blocks to Scipy sparse format
        blocks_sparse = [[None for j in range( self.n_block_cols )] for i in range( self.n_block_rows )]
        for (i,j), Mij in self._blocks.items():
            blocks_sparse[i][j] = Mij.tosparse()

        # Create sparse matrix from sparse blocks
        M = bmat( blocks_sparse )
        M.eliminate_zeros()

        # Sanity check
        assert M.shape[0] == self.codomain.dimension
        assert M.shape[1] == self.  domain.dimension

        return M

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------
    def __setitem__( self, key, value ):

        i,j = key

        if value is None:
            pass

        elif not isinstance( value, LinearOperator ):
            msg = "Block ({},{}) must be 'Matrix' from module 'psydac.linalg.basic'.".format( i,j )
            raise TypeError( msg )

        BlockLinearOperator.__setitem__( self, key, value )
