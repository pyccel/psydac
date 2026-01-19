#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from types import MappingProxyType

import numpy as np
from scipy.sparse import bmat, lil_matrix

from psydac.linalg.basic    import VectorSpace, Vector, LinearOperator
from psydac.linalg.stencil  import StencilMatrix
from psydac.ddm.cart        import InterfaceCartDecomposition
from psydac.ddm.utilities   import get_data_exchanger

__all__ = ('BlockVectorSpace', 'BlockVector', 'BlockLinearOperator')

#===============================================================================
class BlockVectorSpace(VectorSpace):
    """
    Product Vector Space V of two Vector Spaces (V1,V2) or more.

    Parameters
    ----------
    *spaces : psydac.linalg.basic.VectorSpace
        A list of Vector Spaces.

    """
    def __new__(cls, *spaces, connectivity=None):

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
    def __init__(self, *spaces, connectivity=None):

        # Store spaces in a Tuple, because they will not be changed
        self._spaces = tuple(spaces)

        if all(np.dtype(s.dtype)==np.dtype(spaces[0].dtype) for s in spaces):
            self._dtype  = spaces[0].dtype
        else:
            raise NotImplementedError("The matrices domains don't have the same data type.")

        self._connectivity = connectivity or {}
        self._connectivity_readonly = MappingProxyType(self._connectivity)

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def dimension(self):
        """
        The dimension of a product space V = (V1, V2, ...] is the cardinality
        (i.e. the number of vectors) of a basis of V over its base field.

        """
        return sum(Vi.dimension for Vi in self._spaces)

    # ...
    @property
    def dtype(self):
        return self._dtype

    # ...
    def zeros(self):
        """
        Get a copy of the null element of the product space V = [V1, V2, ...]

        Returns
        -------
        null : BlockVector
            A new vector object with all components equal to zero.

        """
        return BlockVector(self, [Vi.zeros() for Vi in self._spaces])

    # ...
    def inner(self, x, y):
        """
        Evaluate the inner vector product between two vectors of this space V.

        If the field of V is real, compute the classical scalar product.
        If the field of V is complex, compute the classical sesquilinear
        product with linearity on the second vector.

        TODO [YG 01.05.2025]: Currently, the first vector is conjugated. We
        want to reverse this behavior in order to align with the convention
        of FEniCS.

        Parameters
        ----------
        x : Vector
            The first vector in the scalar product. In the case of a complex
            field, the inner product is antilinear w.r.t. this vector (hence
            this vector is conjugated).

        y : Vector
            The second vector in the scalar product. The inner product is
            linear w.r.t. this vector.

        Returns
        -------
        float | complex
            The scalar product of the two vectors. Note that inner(x, x) is
            a non-negative real number which is zero if and only if x = 0.

        """

        assert isinstance(x, BlockVector)
        assert isinstance(y, BlockVector)
        assert x.space is self
        assert y.space is self
        return sum(Vi.inner(xi, yi) for Vi, xi, yi in zip(self.spaces, x.blocks, y.blocks))

    #...
    def axpy(self, a, x, y):
        """
        Increment the vector y with the a-scaled vector x, i.e. y = a * x + y,
        provided that x and y belong to the same vector space V (self).
        The scalar value a may be real or complex, depending on the field of V.

        Parameters
        ----------
        a : scalar
            The scaling coefficient needed for the operation.

        x : BlockVector
            The vector which is not modified by this function.

        y : BlockVector
            The vector modified by this function (incremented by a * x).
        """

        assert isinstance(x, BlockVector)
        assert isinstance(y, BlockVector)
        assert x.space is self
        assert y.space is self

        for Vi, xi, yi in zip(self.spaces, x.blocks, y.blocks):
            Vi.axpy(a, xi, yi)

        x._sync = x._sync and y._sync

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------
    @property
    def spaces(self):
        return self._spaces

    @property
    def parallel(self):
        """ Returns True if the memory is distributed."""
        return self._spaces[0].parallel

    @property
    def starts(self):
        return [s.starts for s in self._spaces]

    @property
    def ends(self):
        return [s.ends for s in self._spaces]

    @property
    def pads(self):
        return self._spaces[0].pads

    @property
    def n_blocks(self):
        return len(self._spaces)

    @property
    def connectivity(self):
        return self._connectivity_readonly

    def __getitem__(self, key):
        return self._spaces[key]

#===============================================================================
class BlockVector(Vector):
    """
    Block of Vectors, which is an element of a BlockVectorSpace.

    Parameters
    ----------
    V : psydac.linalg.block.BlockVectorSpace
        Space to which the new vector belongs.

    blocks : list or tuple (psydac.linalg.basic.Vector)
        List of Vector objects, belonging to the correct spaces (optional).

    """
    def __init__(self, V, blocks=None):

        assert isinstance(V, BlockVectorSpace)
        self._space = V

        # We store the blocks in a List so that we can change them later.
        if blocks:
            # Verify that vectors belong to correct spaces and store them
            assert isinstance(blocks, (list, tuple))
            assert all((isinstance(b, Vector)) for b in blocks)
            assert all((Vi is bi.space) for Vi,bi in zip(V.spaces, blocks))

            self._blocks = list(blocks)
        else:
            # TODO: Each block is a 'zeros' vector of the correct space for now,
            # but in the future we would like 'empty' vectors of the same space.
            self._blocks = [Vi.zeros() for Vi in V.spaces]

        # TODO: distinguish between different directions
        self._sync = False

        self._data_exchangers = {}
        self._interface_buf   = {}

        if not V.parallel: return

        # Prepare the data exchangers for the interface data
        for i, j in V.connectivity:
            ((axis_i, ext_i),(axis_j, ext_j)) = V.connectivity[i, j]

            Vi = V.spaces[i]
            Vj = V.spaces[j]
            self._data_exchangers[i, j] = []

            if isinstance(Vi, BlockVectorSpace) and isinstance(Vj, BlockVectorSpace):
                # case of a system of equations
                for k, (Vik, Vjk) in enumerate(zip(Vi.spaces, Vj.spaces)):
                    cart_i = Vik.cart
                    cart_j = Vjk.cart

                    if cart_i.is_comm_null and cart_j.is_comm_null: continue
                    if not cart_i.is_comm_null and not cart_j.is_comm_null: continue
                    if not (axis_i, ext_i) in Vik.interfaces: continue
                    cart_ij = Vik.interfaces[axis_i, ext_i].cart
                    assert isinstance(cart_ij, InterfaceCartDecomposition)
                    self._data_exchangers[i, j].append(get_data_exchanger(cart_ij, self.dtype))

            elif  not isinstance(Vi, BlockVectorSpace) and not isinstance(Vj, BlockVectorSpace):
                # case of scalar equations
                cart_i = Vi.cart
                cart_j = Vj.cart
                if cart_i.is_comm_null and cart_j.is_comm_null: continue
                if not cart_i.is_comm_null and not cart_j.is_comm_null: continue
                if not (axis_i, ext_i) in Vi.interfaces: continue

                cart_ij = Vi.interfaces[axis_i, ext_i].cart
                assert isinstance(cart_ij, InterfaceCartDecomposition)
                self._data_exchangers[i, j].append(get_data_exchanger(cart_ij, self.dtype))
            else:
                raise NotImplementedError("This case is not treated")

        for i, j in V.connectivity:
            if len(self._data_exchangers.get((i, j), [])) == 0:
                self._data_exchangers.pop((i, j), None)

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def space(self):
        """ Vector space to which this vector belongs. """
        return self._space

    # ...
    def toarray(self, *, order='C'):
        """ Convert to Numpy 1D array. """
        return np.concatenate([bi.toarray(order=order) for bi in self._blocks])

    #...
    def copy(self, out=None):
        if self is out:
            return self
        w = out or BlockVector(self._space)#, [b.copy() for b in self._blocks])
        for n, b in enumerate(self._blocks):
            b.copy(out=w[n])
        w._sync = self._sync
        return w

    #...
    def conjugate(self, out=None):
        if out is not None:
            assert isinstance(out, BlockVector)
            assert out.space is self.space
        else:
            out = BlockVector(self.space)

        for (Lij, Lij_out) in zip(self.blocks, out.blocks):
            Lij.conjugate(out=Lij_out)
        out._sync = self._sync
        return out

    #...
    def __neg__(self):
        w = BlockVector(self._space, [-b for b in self._blocks])
        w._sync = self._sync
        return w

    #...
    def __mul__(self, a):
        w = BlockVector(self._space, [b * a for b in self._blocks])
        w._sync = self._sync
        return w

    #...
    def __add__(self, v):
        assert isinstance(v, BlockVector)
        assert v._space is self._space
        w = BlockVector(self._space, [b1 + b2 for b1, b2 in zip(self._blocks, v._blocks)])
        w._sync = self._sync and v._sync
        return w

    #...
    def __sub__(self, v):
        assert isinstance(v, BlockVector)
        assert v._space is self._space
        w = BlockVector(self._space, [b1 - b2 for b1, b2 in zip(self._blocks, v._blocks)])
        w._sync = self._sync and v._sync
        return w

    #...
    def __imul__(self, a):
        for b in self._blocks:
            b *= a
        return self

    #...
    def __iadd__(self, v):
        assert isinstance(v, BlockVector)
        assert v._space is self._space
        for b1, b2 in zip(self._blocks, v._blocks):
            b1 += b2
        self._sync = self._sync and v._sync
        return self

    #...
    def __isub__(self, v):
        assert isinstance(v, BlockVector)
        assert v._space is self._space
        for b1, b2 in zip(self._blocks, v._blocks):
            b1 -= b2
        self._sync = self._sync and v._sync
        return self

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------
    @property
    def blocks(self):
        return tuple(self._blocks)

    #...
    @property
    def n_blocks(self):
        return len(self._blocks)

    # ...
    def __getitem__(self, key):
        return self._blocks[key]

    # ...
    def __setitem__(self, key, value):
        assert value.space == self.space[key]
        assert isinstance(value, Vector)
        self._blocks[key] = value

    # ...
    @property
    def ghost_regions_in_sync(self):
        return self._sync

    # ...
    # NOTE: this property must be set collectively
    @ghost_regions_in_sync.setter
    def ghost_regions_in_sync(self, value):
        assert isinstance(value, bool)
        self._sync = value
        for vi in self.blocks:
            vi.ghost_regions_in_sync = value

    # ...
    def update_ghost_regions(self):

        req = self.start_update_interface_ghost_regions()

        for vi in self.blocks:
            vi.update_ghost_regions()

        self.end_update_interface_ghost_regions(req)

        # Flag ghost regions as up-to-date
        self._sync = True

    def start_update_interface_ghost_regions(self):
        self._collect_interface_buf()
        req = {}
        for (i, j) in self._data_exchangers:
            req[i, j] = [data_ex.start_update_ghost_regions(*bufs) for bufs, data_ex in zip(self._interface_buf[i, j], self._data_exchangers[i, j])]

        return req

    def end_update_interface_ghost_regions(self, req):

        for (i, j) in self._data_exchangers:
            for data_ex, bufs, req_ij in zip(self._data_exchangers[i, j], self._interface_buf[i, j], req[i, j]):
                data_ex.end_update_ghost_regions(req_ij)

    def _collect_interface_buf(self):
        V = self.space
        if not V.parallel:return
        for i, j in V.connectivity:
            if (i, j) not in self._data_exchangers:
                continue
            ((axis_i, ext_i), (axis_j, ext_j)) = V.connectivity[i, j]

            Vi = V.spaces[i]
            Vj = V.spaces[j]

            # The process that owns the patch i will use block i to send data and receive in block j
            self._interface_buf[i, j] = []
            if isinstance(Vi, BlockVectorSpace) and isinstance(Vj, BlockVectorSpace):
                # case of a system of equations
                for k, (Vik, Vjk) in enumerate(zip(Vi.spaces, Vj.spaces)):

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

                if cart_i.is_comm_null:
                    read_buffer = self._blocks[i]._interface_data[axis_i, ext_i]
                else:
                    read_buffer = self._blocks[i]._data

                if cart_j.is_comm_null:
                    write_buffer = self._blocks[j]._interface_data[axis_j, ext_j]
                else:
                    write_buffer = self._blocks[j]._data

                self._interface_buf[i, j].append((read_buffer, write_buffer))

    # ...
    def exchange_assembly_data(self):
        for vi in self.blocks:
            vi.exchange_assembly_data()

    # ...
    def toarray_local(self, order='C'):
        """ Convert to petsc Nest vector.
        """

        blocks = [v.toarray_local(order=order) for v in self._blocks]
        return np.block([blocks])[0]

    # ...
    def topetsc(self):
        """ Convert to petsc data structure.
        """
        from psydac.linalg.topetsc import vec_topetsc
        vec = vec_topetsc( self )
        return vec

#===============================================================================
class BlockLinearOperator(LinearOperator):
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
    def __init__(self, V1, V2, blocks=None):

        assert isinstance(V1, VectorSpace)
        assert isinstance(V2, VectorSpace)

        if not (isinstance(V1, BlockVectorSpace) or isinstance(V2, BlockVectorSpace)):
            raise TypeError("Either domain or codomain must be of type BlockVectorSpace")

        self._domain   = V1
        self._codomain = V2
        self._blocks   = {}

        self._nrows = V2.n_blocks if isinstance(V2, BlockVectorSpace) else 1
        self._ncols = V1.n_blocks if isinstance(V1, BlockVectorSpace) else 1

        # Store blocks in dict (hence they can be manually changed later)
        if blocks:

            if isinstance(blocks, dict):
                for (i, j), Lij in blocks.items():
                    self[i, j] = Lij

            elif isinstance(blocks, (list, tuple)):
                blocks = np.array(blocks, dtype=object)
                for (i, j), Lij in np.ndenumerate(blocks):
                    self[i, j] = Lij

            else:
                raise ValueError( "Blocks can only be given as dict or 2D list/tuple." )

        self._args           = {}
        self._blocks_as_args = self._blocks
        self._increment      = self._codomain.zeros()
        self._args['inc']    = self._increment
        self._args['n_rows'] = self._nrows
        self._args['n_cols'] = self._ncols
        self._func           = self._dot
        self._sync           = False
        self._backend        = None

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def domain(self):
        return self._domain

    # ...
    @property
    def codomain(self):
        return self._codomain

    # ...
    @property
    def dtype(self):
        return self.domain.dtype

    def conjugate(self, out=None):
        if out is not None:
            assert isinstance(out, BlockLinearOperator)
            assert out.domain is self.domain
            assert out.codomain is self.codomain
        else:
            out = BlockLinearOperator(self.domain, self.codomain)

        for (i, j), Lij in self._blocks.items():
            assert isinstance(Lij, (StencilMatrix, BlockLinearOperator))
            if out[i,j]==None:
                out[i, j] = Lij.conjugate()
            else:
                Lij.conjugate(out=out[i,j])

        return out

    def conj(self, out=None):
        return self.conjugate(out=out)
        
# NOTE [YG 27.03.2023]:
# NOTE as part of PR 279, this method was added to facilitate comparisons in tests,
# NOTE but then commented out as deemed unnecessary.
#    def __eq__(self, B):
#        """
#        Return True if self and B are mathematically the same, else return False.
#        Also returns False if at least one block is not the same object and the entries can't be accessed and compared using toarray().
#
#        """
#        assert isinstance(B, BlockLinearOperator)
#
#        if self is B:
#            return True
#
#        nrows = self._nrows
#        ncols = self._ncols
#        if not ((B.n_block_cols == ncols) & (B.n_block_rows == nrows)):
#            return False
#
#        for i in range(nrows):
#            for j in range(ncols):
#                A_ij = self[i, j]
#                B_ij = B[i, j]
#                if not ( A_ij is B_ij ):
#                    if not (((A_ij is None) or (isinstance(A_ij, ZeroOperator))) & ((B_ij is None) or (isinstance(B_ij, ZeroOperator)))):
#                        if not ( np.array_equal(A_ij.toarray(), B_ij.toarray()) ):
#                            return False
#        return True

    # ...
    def tosparse(self, **kwargs):
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
    def toarray(self, **kwargs):
        """ Convert to Numpy 2D array. """
        return self.tosparse(**kwargs).toarray()

    # ...
    def dot(self, v, out=None):

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
    def _dot(blocks, v, out, n_rows, n_cols, inc):

        if n_rows == 1:
            for (_, j), L0j in blocks.items():
                out += L0j.dot(v[j], out=inc)
        elif n_cols == 1:
            for (i, _), Li0 in blocks.items():
                out[i] += Li0.dot(v, out=inc[i])
        else:
            for (i, j), Lij in blocks.items():
                out[i] += Lij.dot(v[j], out=inc[i])

    # ...
    def transpose(self, conjugate=False, out=None):
        """"
        Return the transposed BlockLinearOperator, or the Hermitian Transpose if conjugate==True

        Parameters
        ----------
        conjugate : Bool(optional)
            True to get the Hermitian adjoint.

        out : BlockLinearOperator(optional)
            Optional out for the transpose to avoid temporaries
        """
        if out is not None:
            assert isinstance(out, BlockLinearOperator)
            assert out.codomain is self.domain
            assert out.domain is self.codomain
            for (i, j), Lij in self._blocks.items():
                if out[j,i]==None:
                    out[j, i] = Lij.transpose(conjugate=conjugate)
                else:
                    Lij.transpose(conjugate=conjugate, out=out[j,i])
        else:
            blocks, blocks_T = self.compute_interface_matrices_transpose()
            blocks = {(j, i): b.transpose(conjugate=conjugate) for (i, j), b in blocks.items()}
            blocks.update(blocks_T)
            out = BlockLinearOperator(self.codomain, self.domain, blocks=blocks)

        out.set_backend(self._backend)
        return out

    #--------------------------------------
    # Overridden properties/methods
    #--------------------------------------
    def __neg__(self):
        blocks = {ij: -Bij for ij, Bij in self._blocks.items()}
        mat    = BlockLinearOperator(self.domain, self.codomain, blocks=blocks)
        if self._backend is not None:
            mat._func = self._func
            mat._args = self._args
            mat._blocks_as_args = [mat._blocks[key]._data for key in self._blocks]
            mat._backend = self._backend
        return mat

    # ...
    def __mul__(self, a):
        blocks = {ij: Bij * a for ij, Bij in self._blocks.items()}
        mat = BlockLinearOperator(self.domain, self.codomain, blocks=blocks)
        if self._backend is not None:
            mat._func = self._func
            mat._args = self._args
            mat._blocks_as_args = [mat._blocks[key]._data for key in self._blocks]
            mat._backend = self._backend
        return mat

    # ...
    def __add__(self, M):
        if not isinstance(M, BlockLinearOperator):
            return LinearOperator.__add__(self, M)

        assert M.  domain is self.domain
        assert M.codomain is self.codomain
        blocks  = {}
        for ij in set(self._blocks.keys()) | set(M._blocks.keys()):
            Bij = self[ij]
            Mij = M[ij]
            if   Bij is None: blocks[ij] = Mij.copy()
            elif Mij is None: blocks[ij] = Bij.copy()
            else            : blocks[ij] = Bij + Mij
        mat = BlockLinearOperator(self.domain, self.codomain, blocks=blocks)
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
        if not isinstance(M, BlockLinearOperator):
            return LinearOperator.__sub__(self, M)

        assert M.  domain is self.  domain
        assert M.codomain is self.codomain
        blocks  = {}
        for ij in set(self._blocks.keys()) | set(M._blocks.keys()):
            Bij = self[ij]
            Mij = M[ij]
            if   Bij is None: blocks[ij] = -Mij
            elif Mij is None: blocks[ij] =  Bij.copy()
            else            : blocks[ij] =  Bij - Mij
        mat = BlockLinearOperator(self.domain, self.codomain, blocks=blocks)
        if len(mat._blocks) != len(self._blocks):
            mat.set_backend(self._backend)
        elif self._backend is not None:
            mat._func = self._func
            mat._args = self._args
            mat._blocks_as_args = [mat._blocks[key]._data for key in self._blocks]
            mat._backend = self._backend
        return mat

    #--------------------------------------
    # New properties/methods
    #--------------------------------------
    def diagonal(self, *, inverse = False, sqrt = False, out = None):
        """Get the coefficients on the main diagonal as another BlockLinearOperator object.

        Parameters
        ----------
        inverse : bool
            If True, get the inverse of the diagonal. (Default: False).
            Can be combined with sqrt to get the inverse square root.

        sqrt : bool
            If True, get the square root of the diagonal. (Default: False).
            Can be combined with inverse to get the inverse square root.

        out : BlockLinearOperator
            If provided, write the diagonal entries into this matrix. (Default: None).

        Returns
        -------
        BlockLinearOperator
            The matrix which contains the main diagonal of self (or its inverse).

        """
        # Determine domain and codomain of result
        V, W = self.domain, self.codomain
        if inverse:
            V, W = W, V

        # Check the `out` argument, if `None` create a new BlockLinearOperator
        if out is not None:
            assert isinstance(out, BlockLinearOperator)
            assert out.domain is V
            assert out.codomain is W

            # Set any off-diagonal blocks to zero
            for i, j in out.nonzero_block_indices:
                if i != j:
                    out[i, j] = None
        else:
            out = BlockLinearOperator(V, W)

        # Store the diagonal (or its inverse) into `out`
        for i, j in self.nonzero_block_indices:
            if i == j:
                out[i, i] = self[i, i].diagonal(inverse = inverse, sqrt = sqrt, out = out[i, i])

        return out

    # ...
    @property
    def blocks(self):
        """ Immutable 2D view (tuple of tuples) of the linear operator,
            including the empty blocks as 'None' objects.
        """
        return tuple(
               tuple(self._blocks.get((i, j), None) for j in range(self.n_block_cols))
                                                    for i in range(self.n_block_rows))

    # ...
    @property
    def n_block_rows(self):
        return self._nrows

    # ...
    @property
    def n_block_cols(self):
        return self._ncols

    @property
    def nonzero_block_indices(self):
        """
        Tuple of (i, j) pairs which identify the non-zero blocks:
        i is the row index, j is the column index.
        """
        return tuple(self._blocks)

    # ...
    def update_ghost_regions(self):
        for Lij in self._blocks.values():
            Lij.update_ghost_regions()

    # ...
    def exchange_assembly_data(self):
        for Lij in self._blocks.values():
            Lij.exchange_assembly_data()

    # ...
    def remove_spurious_entries(self ):
        for Lij in self._blocks.values():
            Lij.remove_spurious_entries()

    @property
    def ghost_regions_in_sync(self):
        return self._sync

    @ghost_regions_in_sync.setter
    def ghost_regions_in_sync( self, value ):
        assert isinstance( value, bool )
        self._sync = value
        for Lij in self._blocks.values():
            Lij.ghost_regions_in_sync = value

    # ...
    def __getitem__(self, key):

        assert isinstance( key, tuple )
        assert len( key ) == 2
        assert 0 <= key[0] < self.n_block_rows
        assert 0 <= key[1] < self.n_block_cols

        return self._blocks.get( key, None )

    # ...
    def __setitem__(self, key, value):

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

    # ...
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

    # ...
    def backend(self):
        return self._backend

    # ...
    def copy(self, out=None):
        """
        Create a copy of self, that can potentially be stored in a given BlockLinearOperator.

        Parameters
        ----------
        out : BlockLinearOperator(optional)
            The existing BlockLinearOperator in which we want to copy self.

        Returns
        -------
        BlockLinearOperator
            The copy of `self`, either stored in the given BlockLinearOperator `out`
            (if provided) or in a new one. In the corner case where `out=self` the
            `self` object is immediately returned.
        """
        if out is not None:
            if out is self:
                return self
            assert isinstance(out, BlockLinearOperator)
            assert out.domain is self.domain
            assert out.codomain is self.codomain
        else:
            out = BlockLinearOperator(self.domain, self.codomain)

        for (i, j), Lij in self._blocks.items():
            if out[i, j] is None:
                out[i, j] = Lij.copy()
            else:
                Lij.copy(out = out[i, j])

        out.set_backend(self._backend)

        return out
        
    # ...
    def __imul__(self, a):
        for Bij in self._blocks.values():
            Bij *= a
        return self

    # ...
    def __iadd__(self, M):
        if not isinstance(M, BlockLinearOperator):
            return LinearOperator.__add__(self, M)

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
        if not isinstance(M, BlockLinearOperator):
            return LinearOperator.__sub__(self, M)

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
            
    # ...
    def topetsc(self):
        """ Convert to petsc data structure.
        """
        from psydac.linalg.topetsc import mat_topetsc
        mat = mat_topetsc( self )
        return mat

    def compute_interface_matrices_transpose(self):
        blocks = self._blocks.copy()
        blocks_T = {}
        if not self.codomain.parallel:
            return blocks, blocks_T

        from mpi4py import MPI
        from psydac.linalg.stencil import StencilInterfaceMatrix

        if not isinstance(self.codomain, BlockVectorSpace):
            return blocks, blocks_T

        V = self.codomain 

        for i,j in V.connectivity:
            ((axis_i,ext_i), (axis_j,ext_j)) = V.connectivity[i,j]

            Vi = V.spaces[i]
            Vj = V.spaces[j]

            if isinstance(Vi, BlockVectorSpace) and isinstance(Vj, BlockVectorSpace):
                # case of a system of equations
                block_ij_exists = False
                blocks_T[j,i] = BlockLinearOperator(Vi, Vj)
                block_ij = blocks.get((i,j))._blocks.copy() if self[i,j] else None
                for k1,Vik1 in enumerate(Vi.spaces):
                    for k2,Vjk2 in enumerate(Vj.spaces):
                        cart_i = Vik1.cart
                        cart_j = Vjk2.cart

                        if cart_i.is_comm_null and cart_j.is_comm_null:break
                        if not cart_i.is_comm_null and not cart_j.is_comm_null:break
                        if not (axis_i, ext_i) in Vik1.interfaces: break
                        cart_ij = Vik1.interfaces[axis_i, ext_i].cart
                        assert isinstance(cart_ij, InterfaceCartDecomposition)

                        if not cart_i.is_comm_null:
                            if cart_ij.intercomm.rank == 0:
                                root = MPI.ROOT
                            else:
                                root = MPI.PROC_NULL

                        else:
                            root = 0

                        if not block_ij_exists:
                            block_ij_exists = self[i,j] is not None
                            block_ij_exists = cart_ij.intercomm.bcast(block_ij_exists, root= root) or block_ij_exists

                        if not block_ij_exists:break
                        blocks.pop((i,j), None)
                        block_ij_k1k2 = block_ij is not None and (k1,k2) in block_ij is not None
                        block_ij_k1k2 = cart_ij.intercomm.bcast(block_ij_k1k2, root= root) or block_ij_k1k2

                        if block_ij_k1k2:
                            if not cart_i.is_comm_null:
                                block_ij_k1k2 = block_ij.pop((k1,k2))
                                info = (block_ij_k1k2.domain_start, block_ij_k1k2.codomain_start, block_ij_k1k2.flip, block_ij_k1k2.pads)
                                cart_ij.intercomm.bcast(info, root= root)
                            else:
                                info = cart_ij.intercomm.bcast(None, root=root)
                                block_ij_k1k2 = StencilInterfaceMatrix(Vjk2, Vik1.interfaces[axis_i, ext_i], info[0], info[1], axis_j, axis_i, ext_j, ext_i, flip=info[2], pads=info[3])
                                block_ji_k2k1 = StencilInterfaceMatrix(Vik1, Vjk2, info[1], info[0], axis_i, axis_j, ext_i, ext_j, flip=info[2], pads=info[3])

                            data_exchanger = get_data_exchanger(cart_ij, self.dtype, coeff_shape = block_ij_k1k2._data.shape[block_ij_k1k2._ndim:])
                            data_exchanger.update_ghost_regions(array_minus=block_ij_k1k2._data)

                            if cart_i.is_comm_null:
                                blocks_T[j,i][k2,k1] = block_ij_k1k2.transpose(out=block_ji_k2k1)
                    else:
                        continue

                    break

                if (j,i) in blocks_T and len(blocks_T[j,i]._blocks) == 0:
                    blocks_T.pop((j,i))
                if (i,j) in blocks and len(blocks[i,j]._blocks) == 0:
                    blocks.pop((i,j))

                block_ji_exists = False
                blocks_T[i,j] = BlockLinearOperator(Vj, Vi)
                block_ji = blocks.get((j,i))._blocks.copy() if self[j,i] else None
                for k1,Vik1 in enumerate(Vi.spaces):
                    for k2,Vjk2 in enumerate(Vj.spaces):
                        cart_i = Vik1.cart
                        cart_j = Vjk2.cart

                        if cart_i.is_comm_null and cart_j.is_comm_null:break
                        if not cart_i.is_comm_null and not cart_j.is_comm_null:break
                        if not (axis_i, ext_i) in Vik1.interfaces: break
                        interface_cart_i = Vik1.interfaces[axis_i, ext_i].cart
                        interface_cart_j = Vjk2.interfaces[axis_j, ext_j].cart
                        assert isinstance(interface_cart_i, InterfaceCartDecomposition)
                        assert isinstance(interface_cart_j, InterfaceCartDecomposition)

                        if not cart_j.is_comm_null:
                            if interface_cart_i.intercomm.rank == 0:
                                root = MPI.ROOT
                            else:
                                root = MPI.PROC_NULL

                        else:
                            root = 0

                        if not block_ji_exists:
                            block_ji_exists = self[j,i] is not None
                            block_ji_exists = interface_cart_i.intercomm.bcast(block_ji_exists, root= root) or block_ji_exists

                        if not block_ji_exists:break
                        blocks.pop((j,i), None)

                        block_ji_k2k1 = block_ji is not None and (k2,k1) in block_ji is not None
                        block_ji_k2k1 = interface_cart_i.intercomm.bcast(block_ji_k2k1, root= root) or block_ji_k2k1

                        if block_ji_k2k1:
                            if not cart_j.is_comm_null:
                                block_ji_k2k1 = block_ji.pop((k2,k1))
                                info = (block_ji_k2k1.domain_start, block_ji_k2k1.codomain_start, block_ji_k2k1.flip, block_ji_k2k1.pads)
                                interface_cart_i.intercomm.bcast(info, root= root)
                            else:
                                info = interface_cart_i.intercomm.bcast(None, root=root)
                                block_ji_k2k1 = StencilInterfaceMatrix(Vik1, Vjk2.interfaces[axis_j, ext_j], info[0], info[1], axis_i, axis_j, ext_i, ext_j, flip=info[2], pads=info[3])
                                block_ij_k1k2 = StencilInterfaceMatrix(Vjk2, Vik1, info[1], info[0], axis_j, axis_i, ext_j, ext_i, flip=info[2], pads=info[3])

                            interface_cart_i.comm.Barrier()
                            data_exchanger = get_data_exchanger(interface_cart_j, self.dtype, coeff_shape = block_ji_k2k1._data.shape[block_ji_k2k1._ndim:])

                            data_exchanger.update_ghost_regions(array_plus=block_ji_k2k1._data)

                            if cart_j.is_comm_null:
                                blocks_T[i,j][k1,k2] = block_ji_k2k1.transpose(out=block_ij_k1k2)

                    else:
                        continue

                    break


                if (i,j) in blocks_T and len(blocks_T[i,j]._blocks) == 0:
                    blocks_T.pop((i,j))
                if (j,i) in blocks and len(blocks[j,i]._blocks) == 0:
                    blocks.pop((j,i))

            elif not isinstance(Vi, BlockVectorSpace) and not isinstance(Vj, BlockVectorSpace):

                # case of scalar equations
                cart_i = Vi.cart
                cart_j = Vj.cart
                if cart_i.is_comm_null and cart_j.is_comm_null:continue
                if not cart_i.is_comm_null and not cart_j.is_comm_null:continue
                if not (axis_i, ext_i) in Vi.interfaces: continue
                cart_ij = Vi.interfaces[axis_i, ext_i].cart
                assert isinstance(cart_ij, InterfaceCartDecomposition)

                if not cart_i.is_comm_null:
                    if cart_ij.intercomm.rank == 0:
                        root = MPI.ROOT
                    else:
                        root = MPI.PROC_NULL

                else:
                    root = 0

                block_ij_exists = self[i,j] is not None
                block_ij_exists = cart_ij.intercomm.bcast(block_ij_exists, root= root) or block_ij_exists

                if block_ij_exists:
                    if not cart_i.is_comm_null:
                        block_ij = blocks.pop((i,j))
                        info = (block_ij.domain_start, block_ij.codomain_start, block_ij.flip, block_ij.pads)
                        cart_ij.intercomm.bcast(info, root= root)
                    else:
                        info = cart_ij.intercomm.bcast(None, root=root)
                        block_ij = StencilInterfaceMatrix(Vj, Vi.interfaces[axis_i, ext_i], info[0], info[1], axis_j, axis_i, ext_j, ext_i, flip=info[2], pads=info[3])
                        block_ji = StencilInterfaceMatrix(Vi, Vj, info[1], info[0], axis_i, axis_j, ext_i, ext_j, flip=info[2], pads=info[3])

                    data_exchanger = get_data_exchanger(cart_ij, self.dtype, coeff_shape = block_ij._data.shape[block_ij._ndim:])
                    data_exchanger.update_ghost_regions(array_minus=block_ij._data)

                    if cart_i.is_comm_null:
                        blocks_T[j,i] = block_ij.transpose(out=block_ji)

                if not cart_j.is_comm_null:
                    if cart_ij.intercomm.rank == 0:
                        root = MPI.ROOT
                    else:
                        root = MPI.PROC_NULL

                else:
                    root = 0

                block_ji_exists = self[j,i] is not None
                block_ji_exists =  cart_ij.intercomm.bcast(block_ji_exists, root= root) or block_ji_exists
                if block_ji_exists:
                    if not cart_j.is_comm_null:
                        block_ji = blocks.pop((j,i))
                        info = (block_ji.domain_start, block_ji.codomain_start, block_ji.flip, block_ji.pads)
                        cart_ij.intercomm.bcast((block_ji.domain_start, block_ji.codomain_start, block_ji.flip, block_ji.pads), root= root)
                    else:
                        info = cart_ij.intercomm.bcast(None, root=root)
                        block_ji = StencilInterfaceMatrix(Vi, Vj.interfaces[axis_j, ext_j], info[0], info[1], axis_i, axis_j, ext_i, ext_j, flip=info[2], pads=info[3])
                        block_ij = StencilInterfaceMatrix(Vj, Vi, info[1], info[0], axis_j, axis_i, ext_j, ext_i, flip=info[2], pads=info[3])

                    data_exchanger = get_data_exchanger(cart_ij, self.dtype, coeff_shape = block_ji._data.shape[block_ji._ndim:])
                    data_exchanger.update_ghost_regions(array_plus=block_ji._data)

                    if cart_j.is_comm_null:
                        blocks_T[i,j] = block_ji.transpose(out=block_ij)

        return blocks, blocks_T

    def set_backend(self, backend):
        if isinstance(self.domain, BlockVectorSpace) and isinstance(self.domain.spaces[0], BlockVectorSpace):
            return

        if isinstance(self.codomain, BlockVectorSpace) and isinstance(self.codomain.spaces[0], BlockVectorSpace):
            return

        if backend is None:return
        if backend is self._backend:return

        from psydac.api.ast.linalg import LinearOperatorDot
        from psydac.linalg.stencil import StencilInterfaceMatrix, StencilMatrix

        if not all(isinstance(b, (StencilMatrix, StencilInterfaceMatrix)) for b in self._blocks.values()):
            for b in self._blocks.values():
                b.set_backend(backend)
            return

        block_shape = (self.n_block_rows, self.n_block_cols)

        keys        = self.nonzero_block_indices
        ndim        = self._blocks[keys[0]]._ndim
        c_starts    = []
        d_starts    = []

        interface   = isinstance(self._blocks[keys[0]], StencilInterfaceMatrix)
        if interface:
            interface_axis  = self._blocks[keys[0]]._codomain_axis
            d_ext           = self._blocks[keys[0]]._domain_ext
            d_axis          = self._blocks[keys[0]]._domain_axis
            flip_axis       = self._blocks[keys[0]]._flip
            permutation     = self._blocks[keys[0]]._permutation

            for key in keys:
                c_starts.append(self._blocks[key]._codomain_start)
                d_starts.append(self._blocks[key]._domain_start)

            c_starts = tuple(c_starts)
            d_starts = tuple(d_starts)
        else:
            interface_axis = None
            flip_axis      = (1,)*ndim
            permutation    = None
            c_starts       = None
            d_starts       = None

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
            starts.append(self._blocks[key]._dotargs_null['starts'])
            cm.append(self._blocks[key]._dotargs_null['cm'])
            dm.append(self._blocks[key]._dotargs_null['dm'])

        if self.domain.parallel:
            if interface:
                comm = self.domain.spaces[0].interfaces[d_axis, d_ext].cart.local_comm if isinstance(self.domain, BlockVectorSpace) else self.domain.interfaces[d_axis, d_ext].cart.local_comm 
            else:
                comm = self.codomain.spaces[0].cart.comm if isinstance(self.codomain, BlockVectorSpace) else self.codomain.cart.comm
            if self.domain == self.codomain:
                # In this case nrows_extra[i] == 0 for all i
                dot = LinearOperatorDot(ndim,
                                block_shape=block_shape,
                                keys=keys,
                                comm=comm,
                                backend=frozenset(backend.items()),
                                gpads=tuple(gpads),
                                pads=tuple(pads),
                                dm=tuple(dm),
                                cm=tuple(cm),
                                interface=interface,
                                flip_axis=flip_axis,
                                interface_axis=interface_axis,
                                d_start=d_starts,
                                c_start=c_starts,
                                dtype=self._domain.dtype)

                self._args = {}
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


                for k,key in enumerate(keys):
                    key_str       = ''.join(str(i) for i in key)
                    nrows_extra_k = nrows_extra[k]
                    for i in range(len(nrows_extra_k)):
                        self._args['ne{}_{}'.format(key_str, i+1)] = np.int64(nrows_extra_k[i])

            else:
                dot = LinearOperatorDot(ndim,
                                        block_shape=block_shape,
                                        keys=keys,
                                        comm=comm,
                                        backend=frozenset(backend.items()),
                                        gpads=tuple(gpads),
                                        pads=tuple(pads),
                                        dm=tuple(dm),
                                        cm=tuple(cm),
                                        interface=interface,
                                        flip_axis=flip_axis,
                                        interface_axis=interface_axis,
                                        d_start=d_starts,
                                        c_start=c_starts,
                                        dtype=self._domain.dtype)

                self._args = {}

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
                                    c_start=c_starts,
                                    dtype=self._domain.dtype)
            self._args = {}

        self._blocks_as_args = [self._blocks[key]._data for key in keys]
        dot = dot.func

        if interface:
            def func(blocks, v, out, **args):
                    vs   = [vi._interface_data[d_axis, d_ext] for vi in v.blocks] if isinstance(v, BlockVector) else [v._data]
                    outs = [outi._data for outi in out.blocks] if isinstance(out, BlockVector) else [out._data]
                    dot(*blocks, *vs, *outs, **args)
        else:
            def func(blocks, v, out, **args):
                vs   = [vi._data for vi in v.blocks] if isinstance(v, BlockVector) else [v._data]
                outs = [outi._data for outi in out.blocks] if isinstance(out, BlockVector) else [out._data]
                dot(*blocks, *vs, *outs, **args)

        self._func    = func
        self._backend = backend
