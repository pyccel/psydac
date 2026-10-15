#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np
from mpi4py import MPI
from scipy.sparse import coo_matrix

from psydac.linalg.basic import VectorSpace, Vector, LinearOperator

__all__ = ('DenseVectorSpace', 'DenseVector', 'DenseMatrix')

#==============================================================================
class DenseVectorSpace(VectorSpace):
    """
    Space of one-dimensional global arrays, NOT distributed across processes.

    Typical examples are the right-hand side vector b and solution vector x
    of a linear system A*x=b.
 
    Parameters
    ----------
    n : int
        Number of vector components.

    dtype : data-type, optional
        Desired data-type for the arrays (default is numpy.float64).

    cart : psydac.ddm.cart.CartDecomposition, optional
        N-dimensional Cartesian communicator with N >= 2 (default is None).

    radial_dim : int, optional
        Dimension index for radial variable (default is 0).

    angle_dim : int, optional
        Dimension index for angle variable (default is 1).

    Notes
    -----

    - The current implementation is tailored to the algorithm for imposing C^1
      continuity of a field on a domain with a polar singularity (O-point).

    - Given an N-dimensional Cartesian communicator (N=2+M), each process
      belongs to 3 different subcommunicators:

        1. An 'angular' 1D subcommunicator, where all processes share the same
           identical (M+1)-dimensional array;

        2. A 'radial' 1D subcommunicator, where only the 'master' process has
           access to the data array (other processes store a 0-length array);

        3. A 'tensor' M-dimensional subcommunicator, where the data array
           is distributed among processes and requires the usual StencilVector
           communication pattern.

    - When computing the dot product between two vectors, the following
      operations will be performed in sequence:

        1. Processes with radial coordinate = 0 compute a local dot product;

        2. Processes with radial coordinate = 0 perform an MPI_ALLREDUCE
           operation on the 'tensor' subcommunicator;

        3. All processes perform an MPI_BCAST operation on the 'radial'
           subcommunicator (process with radial coordinate = 0 is the root);

    """
    def __init__(self, n, *, dtype=np.float64, cart=None, radial_dim=0, angle_dim=1):

        self._n     = n
        self._dtype = dtype
        self._cart  = cart

        if cart is not None:

            # TODO: perform checks on input arguments

            # Angle sub-communicator (1D)
            angle_comm = cart.subcomm[angle_dim]

            # Radial sub-communicator (1D)
            radial_comm   = cart.subcomm[radial_dim]
            radial_master = (cart.coords[radial_dim] == 0)
            radial_root   = radial_comm.allreduce(radial_comm.rank if radial_master else 0)

            # Tensor sub-communicator (M-dimensional)
            remain_dims = [d not in (radial_dim, angle_dim) for d in range(cart.ndim)]
            tensor_comm = cart.comm_cart.Sub(remain_dims)

            # Calculate dimension of linear space
            tensor_shape = [cart.npts[i] for i, d in enumerate(remain_dims) if d]
            dimension    = n * np.prod(tensor_shape, dtype=int)

            # Store info
            self._radial_dim  = radial_dim
            self._radial_comm = radial_comm
            self._radial_root = radial_root
            self._angle_dim   = angle_dim
            self._angle_comm  = angle_comm
            self._tensor_comm = tensor_comm
            self._dimension   = dimension

        else:

            # TODO: remove inconsistency between serial and parallel cases

            # For now, in the serial case we assume that the dimension of the
            # linear space is equal to the number of components
            self._dimension = n

    #-------------------------------------
    # Abstract interface
    #-------------------------------------
    @property
    def dimension(self):
        """ The dimension of a vector space V is the cardinality
            (i.e. the number of vectors) of a basis of V over its base field.
        """
        return self._dimension

    # ...
    @property
    def dtype(self):
        return self._dtype

    # ...
    def zeros(self):
        """
        Get a copy of the null element of the DenseVectorSpace V.

        Returns
        -------
        null : DenseVector
            A new vector object with all components equal to zero.

        """
        data = np.zeros(self.ncoeff, dtype=self.dtype)
        return DenseVector(self, data)

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

        TODO [MCP 07.05.2025]: Actually, there is currently no conjugation, 
        since `numpy.dot` is being called.

        Parameters
        ----------
        x : Vector
            The first vector in the scalar product. In the case of a complex
            field, the inner product is antilinear w.r.t. this vector (hence
            this vector is conjugated).
            NOTE [MCP 07.05.2025]: currently without conjugate, see np.dot

        y : Vector
            The second vector in the scalar product. The inner product is
            linear w.r.t. this vector.

        Returns
        -------
        float | complex
            The scalar product of the two vectors. Note that inner(x, x) is
            a non-negative real number which is zero if and only if x = 0.

        """
        assert isinstance(x, DenseVector)
        assert isinstance(y, DenseVector)
        assert x.space is self
        assert y.space is self

        # 1. Local dot product
        res = np.dot(x._data, y._data)

        V = self
        if V.parallel:
            # 2. MPI_ALLREDUCE operation on the M-2 dimensional tensor subcomm.
            if (V.tensor_comm != MPI.COMM_NULL) and (V.radial_comm.rank == V.radial_root):
                res = V.tensor_comm.allreduce(res)
            # 3. MPI_BCAST operation on the 1D radial subcommunicator
            res = V.radial_comm.bcast(res, root=V.radial_root)

        return res

    # ...
    def axpy(self, a, x, y):

        assert isinstance(a, (int, float, complex))
        assert isinstance(x, DenseVector)
        assert isinstance(y, DenseVector)
        assert x.space is self
        assert y.space is self

        y += a * x

    #-------------------------------------
    # Other properties/methods
    #-------------------------------------
    @property
    def parallel(self):
        return (self._cart is not None)

    # ...
    @property
    def ncoeff(self):
        """ Local number of coefficients. """
        # TODO: maybe keep this number global, and add local 'dshape' property
        if self.parallel:
            return self._n if self._radial_comm.rank == self._radial_root else 0
        else:
            return self._n

    # ...
    @property
    def tensor_comm(self):
        return self._tensor_comm

    # ...
    @property
    def angle_comm(self):
        return self._angle_comm

    # ...
    @property
    def radial_comm(self):
        return self._radial_comm

    # ...
    @property
    def radial_root(self):
        return self._radial_root

#==============================================================================
class DenseVector(Vector):

    def __init__(self, V, data):

        assert isinstance(V, DenseVectorSpace)

        data = np.asarray(data)
        assert data.ndim  == 1
        assert data.shape ==(V.ncoeff,)
        assert data.dtype == V.dtype

        self._space = V
        self._data  = data

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def space(self):
        return self._space

    # ...
    def toarray(self, **kwargs):
        return self._data.copy()

    # ...
    def copy(self, out=None):
        if self is out:
            return self
        if out is not None:
            assert isinstance(out, DenseVector)
            assert self.space is out.space
            np.copyto(out._data, self._data, casting='no')
            return out
        else:
            return DenseVector(self._space, self._data.copy())

    # ...
    def conjugate(self, out=None):
        if out is not None:
            assert isinstance(out, DenseVector)
            assert out.space is self.space
        else:
            out = DenseVector(self.space)
        np.conjugate(self._data, out=out._data, casting='no')
        return out

    # ...
    def __neg__(self):
        return DenseVector(self._space, -self._data)

    # ...
    def __mul__(self, a):
        return DenseVector(self._space, self._data * a)

    # ...
    def __add__(self, v):
        assert isinstance(v, DenseVector)
        assert v._space is self._space
        return DenseVector(self._space, self._data + v._data)

    # ...
    def __sub__(self, v):
        assert isinstance(v, DenseVector)
        assert v._space is self._space
        return DenseVector(self._space, self._data - v._data)

    # ...
    def __imul__(self, a):
        self._data *= a
        return self

    # ...
    def __iadd__(self, v):
        assert isinstance(v, DenseVector)
        assert v._space is self._space
        self._data += v._data
        return self

    # ...
    def __isub__(self, v):
        assert isinstance(v, DenseVector)
        assert v._space is self._space
        self._data -= v._data
        return self

    #-------------------------------------
    # Other properties/methods
    #-------------------------------------
    def update_ghost_regions(self, *, direction=None):
        pass

    # ...
    @property
    def ghost_regions_in_sync(self):
        return True

    # ...
    @ghost_regions_in_sync.setter
    def ghost_regions_in_sync(self, value):
        pass

#==============================================================================
class DenseMatrix(LinearOperator):

    def __init__(self, V, W, data):

        assert isinstance(V, DenseVectorSpace)
        assert isinstance(W, DenseVectorSpace)

        data = np.asarray(data)
        assert data.ndim  == 2
        assert data.shape == (W.ncoeff, V.ncoeff)
#        assert data.dfype == #???

        self._domain   = V
        self._codomain = W
        self._data     = data

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

    def transpose(self, conjugate=False):
        raise NotImplementedError()

    # ...
    @property
    def dtype(self):
        return self.domain.dtype

    def __truediv__(self, a):
        """ Divide by scalar. """
        return self * (1.0 / a)

    def __itruediv__(self, a):
        """ Divide by scalar, in place. """
        self *= 1.0 / a
        return self

    # ...
    def dot(self, v, out=None):

        assert isinstance(v, DenseVector)
        assert v.space is self._domain

        if out:
            assert isinstance(out, DenseVector)
            assert out.space is self._codomain
            np.dot(self._data, v._data, out=out._data)
        else:
            W    = self._codomain
            data = np.dot(self._data, v._data)
            out  = DenseVector(W, data)

        return out

    # ...
    def toarray(self , **kwargs):
        return self._data.copy()

    # ...
    def tosparse(self , **kwargs):
        return coo_matrix(self._data)

    # ...
    def copy(self):
        return DenseMatrix(self.domain, self.codomain, self._data.copy())

    # ...
    def __neg__(self):
        return DenseMatrix(self.domain, self.codomain, -self._data)

    # ...
    def __mul__(self, a):
        return DenseMatrix(self.domain, self.codomain, self._data * a)

    # ...
    def __add__(self, m):
        assert isinstance(m, DenseMatrix)
        assert self.  domain == m.  domain
        assert self.codomain == m.codomain
        return DenseMatrix(self.domain, self.codomain, self._data + m._data)

    # ...
    def __sub__(self, m):
        assert isinstance(m, DenseMatrix)
        assert self.  domain == m.  domain
        assert self.codomain == m.codomain
        return DenseMatrix(self.domain, self.codomain, self._data - m._data)

    # ...
    def __imul__(self, a):
        self._data *= a
        return self

    # ...
    def __iadd__(self, m):
        assert isinstance(m, DenseMatrix)
        assert self.  domain == m.  domain
        assert self.codomain == m.codomain
        self._data += m._data
        return self

    # ...
    def __isub__(self, m):
        assert isinstance(m, DenseMatrix)
        assert self.  domain == m.  domain
        assert self.codomain == m.codomain
        self._data -= m._data
        return self
