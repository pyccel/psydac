# coding: utf-8
#
# Copyright 2018 Jalal Lakhlili, Yaman Güçlü

import numpy as np
from scipy.sparse import bmat, lil_matrix

from psydac.linalg.basic import VectorSpace, Vector, LinearOperator, LinearSolver, Matrix
from psydac.ddm.cart     import InterfaceCartDataExchanger, InterfaceCartDecomposition

__all__ = ['BlockVectorSpace', 'BlockVector', 'BlockLinearOperator', 'BlockMatrix', 'BlockDiagonalSolver']

#===============================================================================
class BlockVectorSpace( VectorSpace ):
    """
    Product Vector Space V of two Vector Spaces (V1,V2) or more.

    Parameters
    ----------
    *spaces : psydac.linalg.basic.VectorSpace
        A list of Vector Spaces.

    """
    def __new__(cls, *spaces):

        # Check that all input arguments are vector spaces
        if not all(isinstance(Vi, VectorSpace) for Vi in spaces):
            raise TypeError('All input spaces must be VectorSpace objects')

        # If no spaces are passed, raise an error
        if len(spaces) == 0:
            raise ValueError('Cannot create a BlockVectorSpace of zero spaces')

        # If only one space is passed, return it without creating a new object
        if len(spaces) == 1:
            return spaces[0]

        # Create a new BlockVectorSpace object
        return VectorSpace.__new__(cls)

    # ...
    def __init__(self,  *spaces):

        # Store spaces in a Tuple, because they will not be changed
        self._spaces = tuple(spaces)

        if all(np.dtype(s.dtype)==np.dtype(spaces[0].dtype) for s in spaces):
            self._dtype  = spaces[0].dtype
        else:
            self._dtype = tuple(s.dtype for s in spaces)

        self._interfaces = {}
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
    @property
    def dtype( self ):
        return self._dtype

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
    def parallel( self ):
        """ Returns True if the memory is distributed."""
        return self._spaces[0].parallel

    @property
    def starts( self ):
        return [s.starts for s in self._spaces]

    @property
    def ends( self ):
        return [s.ends for s in self._spaces]

    @property
    def pads( self ):
        return self._spaces[0].pads

    @property
    def n_blocks( self ):
        return len( self._spaces )

    def __getitem__( self, key ):
        return self._spaces[key]

#===============================================================================
class BlockVector( Vector ):
    """
    Block of Vectors, which is an element of a BlockVectorSpace.

    Parameters
    ----------
    V : psydac.linalg.block.BlockVectorSpace
        Space to which the new vector belongs.

    blocks : list or tuple (psydac.linalg.basic.Vector)
        List of Vector objects, belonging to the correct spaces (optional).

    """
    def __init__( self,  V, blocks=None ):

        assert isinstance( V, BlockVectorSpace )
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

        # TODO: distinguish between different directions
        self._sync  = False

        self._data_exchangers = {}
        self._interface_buf   = {}

        if not V.parallel:return

        # Prepare the data exchangers for the interface data
        for i,j in V._interfaces:
            axis_i,axis_j = V._interfaces[i,j][0]
            ext_i,ext_j   = V._interfaces[i,j][1]

            Vi = V.spaces[i]
            Vj = V.spaces[j]
            self._data_exchangers[i,j] = []

            if isinstance(Vi, BlockVectorSpace) and isinstance(Vj, BlockVectorSpace):
                # case of a system of equations
                for k,(Vik,Vjk) in enumerate(zip(Vi.spaces, Vj.spaces)):
                    cart_i = Vik.cart
                    cart_j = Vjk.cart

                    if cart_i.is_comm_null and cart_j.is_comm_null:continue
                    if not cart_i.is_comm_null and not cart_j.is_comm_null:continue
                    if not (axis_i, ext_i) in Vik._interfaces:continue
                    cart_ij = Vik._interfaces[axis_i, ext_i].cart
                    assert isinstance(cart_ij, InterfaceCartDecomposition)
                    self._data_exchangers[i,j].append(InterfaceCartDataExchanger(cart_ij, self.dtype))

            elif  not isinstance(Vi, BlockVectorSpace) and not isinstance(Vj, BlockVectorSpace):
                # case of scalar equations
                cart_i = Vi.cart
                cart_j = Vj.cart
                if cart_i.is_comm_null and cart_j.is_comm_null:continue
                if not cart_i.is_comm_null and not cart_j.is_comm_null:continue
                if not (axis_i, ext_i) in Vi._interfaces:continue

                cart_ij = Vi._interfaces[axis_i, ext_i].cart
                assert isinstance(cart_ij, InterfaceCartDecomposition)
                self._data_exchangers[i,j].append(InterfaceCartDataExchanger(cart_ij, self.dtype))
            else:
                raise NotImplementedError("This case is not treated")

        for i,j in V._interfaces:
            if len(self._data_exchangers.get((i,j), [])) == 0:
                self._data_exchangers.pop((i,j), None)

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def space( self ):
        return self._space

    #...
    @property
    def dtype( self ):
        return self.space.dtype

    #...
    def dot( self, v ):

        assert isinstance( v, BlockVector )
        assert v._space is self._space
        return sum( b1.dot( b2 ) for b1,b2 in zip( self._blocks, v._blocks ) )

    #...
    def copy( self ):
        w = BlockVector( self._space, [b.copy() for b in self._blocks] )
        w._sync = False
        return w

    #...
    def __neg__( self ):
        w = BlockVector( self._space, [-b for b in self._blocks] )
        w._sync = False
        return w

    #...
    def __mul__( self, a ):
        w = BlockVector( self._space, [b*a for b in self._blocks] )
        w._sync = False
        return w

    #...
    def __rmul__( self, a ):
        w = BlockVector( self._space, [a*b for b in self._blocks] )
        w._sync    = False
        return w

    #...
    def __add__( self, v ):
        assert isinstance( v, BlockVector )
        assert v._space is self._space
        w = BlockVector( self._space, [b1+b2 for b1,b2 in zip( self._blocks, v._blocks )] )
        w._sync = False
        return w

    #...
    def __sub__( self, v ):
        assert isinstance( v, BlockVector )
        assert v._space is self._space
        w = BlockVector( self._space, [b1-b2 for b1,b2 in zip( self._blocks, v._blocks )] )
        w._sync = False
        return w

    #...
    def __imul__( self, a ):
        for b in self._blocks:
            b *= a
        self._sync = False
        return self

    #...
    def __iadd__( self, v ):
        assert isinstance( v, BlockVector )
        assert v._space is self._space
        for b1,b2 in zip( self._blocks, v._blocks ):
            b1 += b2
        self._sync = False
        return self

    #...
    def __isub__( self, v ):
        assert isinstance( v, BlockVector )
        assert v._space is self._space
        for b1,b2 in zip( self._blocks, v._blocks ):
            b1 -= b2
        self._sync = False
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
    def ghost_regions_in_sync( self ):
        return self._sync

    # ...
    # NOTE: this property must be set collectively
    @ghost_regions_in_sync.setter
    def ghost_regions_in_sync( self, value ):
        assert isinstance( value, bool )
        self._sync = value

    # ...
    def update_ghost_regions( self, *, direction=None ):
        self.start_update_interface_ghost_regions()
        for vi in self.blocks:
            vi.update_ghost_regions(direction=direction)
        self.end_update_interface_ghost_regions()

        # Flag ghost regions as up-to-date
        self._sync = True

    def start_update_interface_ghost_regions( self ):
        self._collect_interface_buf()
        req = {}
        for (i,j) in self._data_exchangers:
            req[i,j] = [data_ex.start_update_ghost_regions(*bufs) for bufs,data_ex in zip(self._interface_buf[i,j], self._data_exchangers[i,j])]

        self._req = req

    def end_update_interface_ghost_regions( self ):

        for (i,j) in self._data_exchangers:
            for data_ex,bufs,req_ij in zip(self._data_exchangers[i,j], self._interface_buf[i,j], self._req[i,j]):
                data_ex.end_update_ghost_regions(req_ij)

    def _collect_interface_buf( self ):
        V = self.space
        if not V.parallel:return
        for i,j in V._interfaces:
            if not (i,j) in self._data_exchangers:continue
            axis_i,axis_j = V._interfaces[i,j][0]
            ext_i,ext_j   = V._interfaces[i,j][1]
            Vi = V.spaces[i]
            Vj = V.spaces[j]
            self._interface_buf[i,j]   = []
            if isinstance(Vi, BlockVectorSpace) and isinstance(Vj, BlockVectorSpace):
                # case of a system of equations
                for k,(Vik,Vjk) in enumerate(zip(Vi.spaces, Vj.spaces)):

                    cart_i = Vik.cart
                    cart_j = Vjk.cart

                    buf = [None]*2
                    if cart_i.is_comm_null:
                        buf[0] = self._blocks[i]._blocks[k]._interface_data[axis_i, ext_i]
                    else:
                        buf[0] = self._blocks[i]._blocks[k]._data

                    if cart_j.is_comm_null:
                        buf[1] = self._blocks[j]._blocks[k]._interface_data[axis_j, ext_j]
                    else:
                        buf[1] = self._blocks[j]._blocks[k]._data

                    self._interface_buf[i,j].append(tuple(buf))
            elif  not isinstance(Vi, BlockVectorSpace) and not isinstance(Vj, BlockVectorSpace):
                # case of scalar equations
                cart_i = Vi.cart
                cart_j = Vj.cart

                buf = [None]*2
                if cart_i.is_comm_null:
                    buf[0] = self._blocks[i]._interface_data[axis_i, ext_i]
                else:
                    buf[0] = self._blocks[i]._data

                if cart_j.is_comm_null:
                    buf[1] = self._blocks[j]._interface_data[axis_j, ext_j]
                else:
                    buf[1] = self._blocks[j]._data

                self._interface_buf[i,j].append(tuple(buf))

    # ...
    @property
    def n_blocks( self ):
        return len( self._blocks )

    # ...
    @property
    def blocks( self ):
        return tuple( self._blocks )

    # ...
    def toarray( self, order='C' ):
        return np.concatenate( [bi.toarray(order=order) for bi in self._blocks] )

    def toarray_local( self, order='C' ):
        """ Convert to petsc Nest vector.
        """

        blocks    = [v.toarray_local(order=order) for v in self._blocks]
        return np.block([blocks])[0]

    def topetsc( self ):
        """ Convert to petsc Nest vector.
        """

        blocks    = [v.topetsc() for v in self._blocks]
        cart      = self._space.spaces[0].cart
        petsccart = cart.topetsc()
        vec       = petsccart.petsc.Vec().createNest(blocks, comm=cart.comm)
        return vec

#===============================================================================
class BlockLinearOperator( LinearOperator ):
    """
    Linear operator that can be written as blocks of other Linear Operators.
    Either the domain or the codomain of this operator, or both, should be of
    class BlockVectorSpace.

    Parameters
    ----------
    V1 : psydac.linalg.block.VectorSpace
        Domain of the new linear operator.

    V2 : psydac.linalg.block.VectorSpace
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

        assert isinstance( V1, VectorSpace )
        assert isinstance( V2, VectorSpace )

        if not (isinstance(V1, BlockVectorSpace) or isinstance(V2, BlockVectorSpace)):
            raise TypeError("Either domain or codomain must be of type BlockVectorSpace")

        self._domain   = V1
        self._codomain = V2
        self._blocks   = {}

        self._nrows = V2.n_blocks if isinstance(V2, BlockVectorSpace) else 1
        self._ncols = V1.n_blocks if isinstance(V1, BlockVectorSpace) else 1

        # Store blocks in dict (hence they can be manually changed later)
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

        self._args = {}
        self._blocks_as_args = self._blocks
        self._args['n_rows'] = self._nrows
        self._args['n_cols'] = self._ncols
        self._func           = self._dot

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
    def dtype( self ):
        return self.domain.dtype

    # ...
    def dot( self, v, out=None ):

        if self.n_block_cols == 1:
            assert isinstance(v, Vector)
        else:
            assert isinstance(v, BlockVector)

        assert v.space is self.domain

        if out is not None:
            if self.n_block_rows == 1:
                assert isinstance(out, Vector)
            else:
                assert isinstance(out, BlockVector)
            assert out.space is self.codomain
            out *= 0.0
        else:
            out = self.codomain.zeros()

        if not v.ghost_regions_in_sync:
            v.update_ghost_regions()

        self._func(self._blocks_as_args, v, out, **self._args)

        out.ghost_regions_in_sync = False
        return out

    #...
    @staticmethod
    def _dot(blocks, v, out, n_rows, n_cols):

        if n_rows == 1:
            for (_, j), L0j in blocks.items():
                out += L0j.dot(v[j])
        elif n_cols == 1:
            for (i, _), Li0 in blocks.items():
                out[i] += Li0.dot(v)
        else:
            for (i, j), Lij in blocks.items():
                out[i] += Lij.dot(v[j])

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
        return self._nrows

    # ...
    @property
    def n_block_cols( self ):
        return self._ncols

    # ...
    def update_ghost_regions( self ):
        for Lij in self._blocks.values():
            Lij.update_ghost_regions()

    # ...
    def remove_spurious_entries( self ):
        for Lij in self._blocks.values():
            Lij.remove_spurious_entries()

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

        # Check domain of rhs
        if self.n_block_cols == 1:
            assert value.domain is self.domain
        else:
            assert value.domain is self.domain[j]

        # Check codomain of rhs
        if self.n_block_rows == 1:
            assert value.codomain is self.codomain
        else:
            assert value.codomain is self.codomain[i]

        self._blocks[i,j] = value
    
    def transform(self, operation):
        """
        Applies an operation on each block in this BlockLinearOperator.

        Parameters
        ----------
        operation : LinearOperator -> LinearOperator
            The operation which transforms each block.
        """
        blocks = {ij: operation(Bij) for ij, Bij in self._blocks.items()}
        return BlockLinearOperator(self.domain, self.codomain, blocks=blocks)

    def tomatrix(self):
        """
        Returns a BlockMatrix with the same blocks as this BlockLinearOperator.
        Does only work, if all blocks are matrices (i.e. type Matrix) as well.
        """
        return BlockMatrix(self.domain, self.codomain, blocks=self._blocks)

    # ...
    def transpose(self):
        blocks = {(j, i): b.transpose() for (i, j), b in self._blocks.items()}
        return BlockLinearOperator(self.codomain, self.domain, blocks=blocks)

    # ...
    @property
    def T(self):
        return self.transpose()

#===============================================================================
class BlockMatrix( BlockLinearOperator, Matrix ):
    """
    Linear operator that can be written as blocks of other Linear Operators,
    with the additional capability to be converted to a 2D Numpy array
    or to a Scipy sparse matrix.

    Parameters
    ----------
    V1 : psydac.linalg.block.BlockVectorSpace
        Domain of the new linear operator.

    V2 : psydac.linalg.block.BlockVectorSpace
        Codomain of the new linear operator.

    blocks : dict | (list of lists) | (tuple of tuples)
        Matrix objects (optional).

        a) 'blocks' can be dictionary with
            . key   = tuple (i, j), where i and j are two integers >= 0
            . value = corresponding Matrix Mij

        b) 'blocks' can be list of lists (or tuple of tuples) where blocks[i][j]
            is the Matrix Mij (if None, we assume all entries are zeros)

    """

    def __init__( self, V1, V2, blocks=None ):
        super(BlockMatrix, self).__init__(V1, V2, blocks=blocks)
        self._backend = None

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    def toarray( self, **kwargs ):
        """ Convert to Numpy 2D array. """
        return self.tosparse(**kwargs).toarray()

    def backend( self ):
        return self._backend

    # ...
    def tosparse( self, **kwargs ):
        """ Convert to any Scipy sparse matrix format. """
        # Shortcuts

        nrows = self.n_block_rows
        ncols = self.n_block_cols

        # Utility functions: get domain of blocks on column j, get codomain of blocks on row i
        block_domain   = (lambda j: self.domain  [j]) if ncols > 1 else (lambda j: self.domain)
        block_codomain = (lambda i: self.codomain[i]) if nrows > 1 else (lambda i: self.codomain)

        # Convert all blocks to Scipy sparse format
        blocks_sparse = [[None for j in range(ncols)] for i in range(nrows)]
        for i in range(nrows):
            for j in range(ncols):
                if (i, j) in self._blocks:
                    blocks_sparse[i][j] = self._blocks[i, j].tosparse(**kwargs)
                else:
                    m = block_codomain(i).dimension
                    n = block_domain  (j).dimension
                    blocks_sparse[i][j] = lil_matrix((m, n))

        # Create sparse matrix from sparse blocks
        M = bmat( blocks_sparse )
        M.eliminate_zeros()

        # Sanity check
        assert M.shape[0] == self.codomain.dimension
        assert M.shape[1] == self.  domain.dimension

        return M

    # ...
    def copy(self):
        blocks = {ij: Bij.copy() for ij, Bij in self._blocks.items()}
        mat = BlockMatrix(self.domain, self.codomain, blocks=blocks)
        if self._backend is not None:
            mat._func = self._func
            mat._args = self._args
            mat._blocks_as_args = [mat._blocks[key]._data for key in self._blocks]
            mat._backend = self._backend
        return mat

    # ...
    def __neg__(self):
        blocks = {ij: -Bij for ij, Bij in self._blocks.items()}
        mat    = BlockMatrix(self.domain, self.codomain, blocks=blocks)
        if self._backend is not None:
            mat._func = self._func
            mat._args = self._args
            mat._blocks_as_args = [mat._blocks[key]._data for key in self._blocks]
            mat._backend = self._backend
        return mat
    # ...
    def __mul__(self, a):
        blocks = {ij: Bij * a for ij, Bij in self._blocks.items()}
        mat = BlockMatrix(self.domain, self.codomain, blocks=blocks)
        if self._backend is not None:
            mat._func = self._func
            mat._args = self._args
            mat._blocks_as_args = [mat._blocks[key]._data for key in self._blocks]
            mat._backend = self._backend
        return mat
    # ...
    def __rmul__(self, a):
        blocks = {ij: a * Bij for ij, Bij in self._blocks.items()}
        mat = BlockMatrix(self.domain, self.codomain, blocks=blocks)
        if self._backend is not None:
            mat._func = self._func
            mat._args = self._args
            mat._blocks_as_args = [mat._blocks[key]._data for key in self._blocks]
            mat._backend = self._backend
        return mat
    # ...
    def __add__(self, M):
        assert isinstance(M, BlockMatrix)
        assert M.  domain is self.  domain
        assert M.codomain is self.codomain
        blocks  = {}
        for ij in set(self._blocks.keys()) | set(M._blocks.keys()):
            Bij = self[ij]
            Mij = M[ij]
            if   Bij is None: blocks[ij] = Mij.copy()
            elif Mij is None: blocks[ij] = Bij.copy()
            else            : blocks[ij] = Bij + Mij
        mat = BlockMatrix(self.domain, self.codomain, blocks=blocks)
        if len(mat._blocks) != len(self._blocks):
            mat.set_backend(self._backend)
        elif self._backend is not None:
            mat._func = self._func
            mat._args = self._args
            mat._blocks_as_args = [mat._blocks[key]._data for key in self._blocks]
            mat._backend = self._backend
        return mat
    # ...
    def __sub__(self, M):
        assert isinstance(M, BlockMatrix)
        assert M.  domain is self.  domain
        assert M.codomain is self.codomain
        blocks  = {}
        for ij in set(self._blocks.keys()) | set(M._blocks.keys()):
            Bij = self[ij]
            Mij = M[ij]
            if   Bij is None: blocks[ij] = -Mij
            elif Mij is None: blocks[ij] =  Bij.copy()
            else            : blocks[ij] =  Bij - Mij
        mat = BlockMatrix(self.domain, self.codomain, blocks=blocks)
        if len(mat._blocks) != len(self._blocks):
            mat.set_backend(self._backend)
        elif self._backend is not None:
            mat._func = self._func
            mat._args = self._args
            mat._blocks_as_args = [mat._blocks[key]._data for key in self._blocks]
            mat._backend = self._backend
        return mat

    # ...
    def __imul__(self, a):
        for Bij in self._blocks.values():
            Bij *= a
        return self

    # ...
    def __iadd__(self, M):
        assert isinstance(M, BlockMatrix)
        assert M.  domain is self.  domain
        assert M.codomain is self.codomain

        for ij in set(self._blocks.keys()) | set(M._blocks.keys()):

            Mij = M[ij]
            if Mij is None:
                continue

            Bij = self[ij]
            if Bij is None:
                self[ij] = Mij.copy()
            else:
                Bij += Mij

        return self

    # ...
    def __isub__(self, M):
        assert isinstance(M, BlockMatrix)
        assert M.  domain is self.  domain
        assert M.codomain is self.codomain

        for ij in set(self._blocks.keys()) | set(M._blocks.keys()):

            Mij = M[ij]
            if Mij is None:
                continue

            Bij = self[ij]
            if Bij is None:
                self[ij] = -Mij
            else:
                Bij -= Mij

        return self
    #--------------------------------------
    # Other properties/methods
    #--------------------------------------
    def __setitem__( self, key, value ):

        i,j = key

        if value is None:
            pass

        elif not isinstance( value, Matrix ):
            msg = "Block ({},{}) must be 'Matrix' from module 'psydac.linalg.basic'.".format( i,j )
            raise TypeError( msg )

        BlockLinearOperator.__setitem__( self, key, value )

    # ...
    def transpose(self):
        blocks = {(j, i): b.transpose() for (i, j), b in self._blocks.items()}
        mat = BlockMatrix(self.codomain, self.domain, blocks=blocks)
        mat.set_backend(self._backend)
        return mat

    # ...
    def topetsc( self ):
        """ Convert to petsc Nest Matrix.
        """
        # Convert all blocks to petsc format
        blocks = [[None for j in range( self.n_block_cols )] for i in range( self.n_block_rows )]
        for (i,j), Mij in self._blocks.items():
            blocks[i][j] = Mij.topetsc()

        if self.n_block_cols == 1:
            cart = self.domain.cart
        else:
            cart = self.domain.spaces[0].cart

        petsccart = cart.topetsc()

        return petsccart.petsc.Mat().createNest(blocks, comm=cart.comm)

    def update_ghost_regions(self):
        blocks = self._blocks.copy()
        if not self.codomain.parallel:
            return blocks

        from mpi4py import MPI
        from psydac.linalg.stencil import StencilInterfaceMatrix

        V = self.codomain
        for i,j in V._interfaces:

            axis_i,axis_j = V._interfaces[i,j][0]
            ext_i,ext_j   = V._interfaces[i,j][1]

            Vi = V.spaces[i]
            Vj = V.spaces[j]

            # case of scalar equations
            cart_i = Vi.cart
            cart_j = Vj.cart
            if cart_i.is_comm_null and cart_j.is_comm_null:continue
            if not cart_i.is_comm_null and not cart_j.is_comm_null:continue
            if not (axis_i, ext_i) in Vi._interfaces:continue
            cart_ij = Vi._interfaces[axis_i, ext_i].cart
            assert isinstance(cart_ij, InterfaceCartDecomposition)
            block_ij = self[i,j] is not None
            block_ji = self[j,i] is not None

            if not cart_i.is_comm_null:
                if cart_ij.intercomm.rank == 0:
                    root = MPI.ROOT
                else:
                    root = MPI.PROC_NULL
            else:
                root = 0

            block_ij = cart_ij.intercomm.bcast(block_ij, root= root) or block_ij

            if block_ij:
                if not cart_i.is_comm_null:
                    block_ij = self[i,j]
                    cart_ij.intercomm.bcast((block_ij.d_start, block_ij.c_start, block_ij.flip, block_ij.pads), root= root)
                else:
                    (s_d, s_c, flip, pads) = cart_ij.intercomm.bcast(None, root=root)
                    block_ij = StencilInterfaceMatrix(Vj, Vi._interfaces[axis_i, ext_i], s_d, s_c, axis_j, axis_i, ext_j, ext_i, flip=flip)

                data_exchanger = InterfaceCartDataExchanger(cart_ij, self.dtype, coeff_shape = tuple(2*p+1 for p in block_ij.pads))
                data_exchanger.update_ghost_regions(array_minus=block_ij._data)

            if not cart_j.is_comm_null:
                if cart_ij.intercomm.rank == 0:
                    root = MPI.ROOT
                else:
                    root = MPI.PROC_NULL
            else:
                root = 0

            block_ji =  cart_ij.intercomm.bcast(block_ji, root= root) or block_ji
            if block_ji:
                if not cart_j.is_comm_null:
                    block_ji = self[j,i]
                    cart_ij.intercomm.bcast((block_ji.d_start, block_ji.c_start, block_ji.flip, block_ji.pads), root= root)
                else:
                    (s_d, s_c, flip, pads) = cart_ij.intercomm.bcast(None, root=root)
                    block_ji = StencilInterfaceMatrix(Vi, Vj._interfaces[axis_j, ext_j], s_d, s_c, axis_i, axis_j, ext_i, ext_j, flip=flip)

                data_exchanger = InterfaceCartDataExchanger(cart_ij, self.dtype, coeff_shape = tuple(2*p+1 for p in block_ji.pads))
                data_exchanger.update_ghost_regions(array_plus=block_ji._data)


    def set_backend(self, backend):
        if isinstance(self.domain, BlockVectorSpace) and isinstance(self.domain.spaces[0], BlockVectorSpace):
            return

        if isinstance(self.codomain, BlockVectorSpace) and isinstance(self.codomain.spaces[0], BlockVectorSpace):
            return

        if backend is None:return
        if backend is self._backend:return
        self._backend=backend

        from psydac.api.ast.linalg import LinearOperatorDot, TransposeOperator, InterfaceTransposeOperator
        from psydac.linalg.stencil import StencilInterfaceMatrix

        block_shape = (self.n_block_rows, self.n_block_cols)
        keys        = tuple(self._blocks.keys())
        ndim        = self._blocks[keys[0]]._ndim
        c_starts    = []
        d_starts    = []

        interface   = isinstance(self._blocks[keys[0]], StencilInterfaceMatrix)
        if interface:
            interface_axis  = self._blocks[keys[0]]._c_axis
            d_ext   = self._blocks[keys[0]]._d_ext
            d_axis   = self._blocks[keys[0]]._d_axis
            flip_axis       = self._blocks[keys[0]]._flip
            permutation     = self._blocks[keys[0]]._permutation

            for key in keys:
                c_starts.append(self._blocks[key]._c_start)
                d_starts.append(self._blocks[key]._d_start)

            c_starts = tuple(c_starts)
            d_starts = tuple(d_starts)
        else:
            interface_axis = None
            flip_axis      = (1,)*ndim
            permutation    = None
            c_starts       = None
            d_starts       = None

        if interface:
            transpose = InterfaceTransposeOperator(ndim, backend=frozenset(backend.items()))
        else:
            transpose = TransposeOperator(ndim, backend=frozenset(backend.items()))

        for k,key in enumerate(keys):
            self._blocks[key]._transpose_func = transpose.func
            self._blocks[key]._transpose_args  = self._blocks[key]._transpose_args_null.copy()
            nrows   = self._blocks[key]._transpose_args.pop('nrows')
            ncols   = self._blocks[key]._transpose_args.pop('ncols')
            gpads   = self._blocks[key]._transpose_args.pop('gpads')
            pads    = self._blocks[key]._transpose_args.pop('pads')
            ndiags  = self._blocks[key]._transpose_args.pop('ndiags')
            ndiagsT = self._blocks[key]._transpose_args.pop('ndiagsT')
            si      = self._blocks[key]._transpose_args.pop('si')
            sk      = self._blocks[key]._transpose_args.pop('sk')
            sl      = self._blocks[key]._transpose_args.pop('sl')

            if interface:
                args = dict([('n{i}',nrows),('nc{i}', ncols),('gp{i}', gpads),('p{i}',pads )
                          ,('nd{i}', ndiags),('ndT{i}', ndiagsT),('si{i}', si),
                          ('sk{i}', sk),('sl{i}', sl)])

                self._blocks[key]._transpose_args            = {}
                self._blocks[key]._transpose_args['d_start'] =  np.int64(d_starts[k])
                self._blocks[key]._transpose_args['c_start'] =  np.int64(c_starts[k])
                self._blocks[key]._transpose_args['dim']     =  np.int64(interface_axis)
            else:
                dm      = self._blocks[key]._transpose_args.pop('dm')
                cm      = self._blocks[key]._transpose_args.pop('cm')
                args = dict([('n{i}',nrows),('nc{i}', ncols),('gp{i}', gpads),('p{i}',pads ),
                                ('dm{i}', dm),('cm{i}', cm),('nd{i}', ndiags),
                                ('ndT{i}', ndiagsT),('si{i}', si),('sk{i}', sk),('sl{i}', sl)])
                self._blocks[key]._transpose_args = {}

            for arg_name, arg_val in args.items():
                for i in range(len(nrows)):
                    self._blocks[key]._transpose_args[arg_name.format(i=i+1)] = np.int64(arg_val[i]) if isinstance(arg_val[i], int) else arg_val[i]

        starts      = []
        nrows       = []
        nrows_extra = []
        gpads       = []
        pads        = []
        dm          = []
        cm          = []
        for key in keys:
            nrows.append(self._blocks[key]._dotargs_null['nrows'])
            nrows_extra.append(self._blocks[key]._dotargs_null['nrows_extra'])
            gpads.append(self._blocks[key]._dotargs_null['gpads'])
            pads.append(self._blocks[key]._dotargs_null['pads'])
            if interface:
                cm.append((1,)*ndim)
                dm.append((1,)*ndim)
                starts.append(None)
            else:
                starts.append(self._blocks[key]._dotargs_null['starts'])
                cm.append(self._blocks[key]._dotargs_null['cm'])
                dm.append(self._blocks[key]._dotargs_null['dm'])

        if self.domain.parallel:
            comm = self.codomain.spaces[0].cart.comm if isinstance(self.codomain, BlockVectorSpace) else self.codomain.cart.comm
            if self.domain == self.codomain:
                # In this case nrows_extra[i] == 0 for all i
                dot = LinearOperatorDot(ndim,
                                block_shape=block_shape,
                                keys=keys,
                                comm=comm,
                                backend=frozenset(backend.items()),
                                nrows_extra = tuple(nrows_extra),
                                gpads=tuple(gpads),
                                pads=tuple(pads),
                                dm=tuple(dm),
                                cm=tuple(cm),
                                interface=interface,
                                flip_axis=flip_axis,
                                interface_axis=interface_axis,
                                d_start=d_starts,
                                c_start=c_starts)

                self._args = {}
                if not interface:
                    for k,key in enumerate(keys):
                        key_str = ''.join(str(i) for i in key)
                        starts_k = starts[k]
                        for i in range(len(starts_k)):
                            self._args['s{}_{}'.format(key_str, i+1)] = np.int64(starts_k[i])

                for k,key in enumerate(keys):
                    key_str = ''.join(str(i) for i in key)
                    nrows_k  = nrows[k]
                    for i in range(len(nrows_k)):
                        self._args['n{}_{}'.format(key_str, i+1)] = np.int64(nrows_k[i])

            else:
                dot = LinearOperatorDot(ndim,
                                        block_shape=block_shape,
                                        keys=keys,
                                        comm=comm,
                                        backend=frozenset(backend.items()),
                                        gpads=gpads,
                                        pads=pads,
                                        dm=dm,
                                        cm=cm,
                                        interface=interface,
                                        flip_axis=flip_axis,
                                        interface_axis=interface_axis,
                                        d_start=d_starts,
                                        c_start=c_starts)

                self._args = {}
                if not interface:
                    for k,key in enumerate(keys):
                        key_str       = ''.join(str(i) for i in key)
                        starts_k      = starts[k]
                        for i in range(len(starts_k)):
                            self._args['s{}_{}'.format(key_str, i+1)] = np.int64(starts_k[i])

                for k,key in enumerate(keys):
                    key_str       = ''.join(str(i) for i in key)
                    nrows_k       = nrows[k]
                    for i in range(len(nrows_k)):
                        self._args['n{}_{}'.format(key_str, i+1)] = np.int64(nrows_k[i])

                for k,key in enumerate(keys):
                    key_str       = ''.join(str(i) for i in key)
                    nrows_extra_k = nrows_extra[k]
                    for i in range(len(nrows_extra_k)):
                        self._args['ne{}_{}'.format(key_str, i+1)] = np.int64(nrows_extra_k[i])

        else:
            dot = LinearOperatorDot(ndim,
                                    block_shape=block_shape,
                                    keys=keys,
                                    comm=None,
                                    backend=frozenset(backend.items()),
                                    starts=tuple(starts),
                                    nrows=tuple(nrows),
                                    nrows_extra=tuple(nrows_extra),
                                    gpads=tuple(gpads),
                                    pads=tuple(pads),
                                    dm=tuple(dm),
                                    cm=tuple(cm),
                                    interface=interface,
                                    flip_axis=flip_axis,
                                    interface_axis=interface_axis,
                                    d_start=d_starts,
                                    c_start=c_starts)
            self._args = {}

        self._blocks_as_args = [self._blocks[key]._data for key in keys]
        dot = dot.func

        if interface:
            def func(blocks, v, out, **args):
                    vs   = [vi._interface_data[d_axis, d_ext] for vi in v.blocks] if isinstance(v, BlockVector) else v._data
                    outs = [outi._data for outi in out.blocks] if isinstance(out, BlockVector) else out._data
                    dot(*blocks, *vs, *outs, **args)
        else:
            def func(blocks, v, out, **args):
                vs   = [vi._data for vi in v.blocks] if isinstance(v, BlockVector) else v._data
                outs = [outi._data for outi in out.blocks] if isinstance(out, BlockVector) else out._data
                dot(*blocks, *vs, *outs, **args)

        self._func = func

#===============================================================================
class BlockDiagonalSolver( LinearSolver ):
    """
    A LinearSolver that can be written as blocks of other LinearSolvers,
    i.e. it can be seen as a solver for linear equations with block-diagonal matrices.

    The space of this solver has to be of the type BlockVectorSpace.

    Parameters
    ----------
    V : psydac.linalg.block.BlockVectorSpace
        Space of the new blocked linear solver.

    blocks : dict | list | tuple
        LinearSolver objects (optional).

        a) 'blocks' can be dictionary with
            . key   = integer i >= 0
            . value = corresponding LinearSolver Lii

        b) 'blocks' can be list of LinearSolvers (or tuple of these) where blocks[i]
            is the LinearSolver Lii (if None, we assume null operator)

    """
    def __init__( self, V, blocks=None ):

        assert isinstance( V, BlockVectorSpace )

        self._space   = V
        self._nblocks = V.n_blocks

        # Store blocks in list (hence, they can be manually changed later)
        self._blocks   = [None] * self._nblocks

        if blocks:

            if isinstance( blocks, dict ):
                for i, L in blocks.items():
                    self[i] = L

            elif isinstance( blocks, (list, tuple) ):
                for i, L in enumerate( blocks ):
                    self[i] = L

            else:
                raise ValueError( "Blocks can only be given as dict or 1D list/tuple." )

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def space( self ):
        """
        The space this BlockDiagonalSolver works on.
        """
        return self._space

    # ...
    def solve( self, rhs, out=None, transposed=False ):
        """
        Solves the linear system for the given right-hand side rhs.
        An out vector can be supplied, otherwise a new vector will be allocated.

        This operation supports in-place operations, given that the underlying solvers
        do as well.

        Parameters
        ----------
        rhs : BlockVector
            The input right-hand side.
        
        out : BlockVector | NoneType
            The output vector, or None.
        
        transposed : Bool
            If true, and supported by the underlying solvers,
            rhs is solved against the transposed right-hand sides.
        
        Returns
        -------
        out : BlockVector
            Either `out`, if given as input parameter, or a newly-allocated vector.
            In all cases, it holds the result of the computation.
        """
        assert isinstance(rhs, BlockVector)
        assert rhs.space is self.space
        if out is None:
            out = self.space.zeros()
        
        assert isinstance(out, BlockVector)
        assert out.space is self.space

        rhs.update_ghost_regions()

        for i, L in enumerate(self._blocks):
            if L is None:
                raise NotImplementedError('All solvers have to be defined.')
            L.solve(rhs[i], out=out[i], transposed=transposed)

        return out

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------
    @property
    def blocks( self ):
        """ Immutable 1D view (tuple) of the linear solvers,
            including the empty blocks as 'None' objects.
        """
        return tuple(self._blocks)
    # ...
    @property
    def n_blocks( self ):
        """
        The number of blocks in the matrix.
        """
        return self._nblocks

    # ...
    def __getitem__( self, key ):
        assert 0 <= key < self._nblocks

        return self._blocks.get( key, None )

    # ...
    def __setitem__( self, key, value ):
        assert 0 <= key < self._nblocks

        assert isinstance( value, LinearSolver )

        # Check domain of rhs
        assert value.space is self.space[key]

        self._blocks[key] = value
