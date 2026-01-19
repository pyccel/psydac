#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from itertools import repeat

import numpy as np
from scipy.sparse import coo_matrix

from psydac.linalg.basic   import LinearOperator
from psydac.linalg.stencil import StencilVectorSpace, StencilVector
from psydac.polar .dense   import DenseVectorSpace, DenseVector

__all__ = ('LinearOperator_StencilToDense', 'LinearOperator_DenseToStencil')

#==============================================================================
class LinearOperator_StencilToDense(LinearOperator):

    def __init__(self, V, W, data):

        assert isinstance(V, StencilVectorSpace)
        assert isinstance(W,   DenseVectorSpace)

        # V space must be 2D for now (TODO: extend to higher dimensions)
        # W space must have 3 components for now (TODO: change to arbitrary n)

        s1, s2 = V.starts
        e1, e2 = V.ends
        p1, p2 = V.pads
        n0     = W.ncoeff

        data = np.asarray(data)
        assert data.shape == (n0, p1, e2-s2+1 + 2*p2)

        # Store information in object
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

        assert isinstance(v, StencilVector)
        assert v.space is self._domain

        if out:
            assert isinstance(out, DenseVector)
            assert out.space is self._codomain
        else:
            out = self._codomain.zeros()

        V = self._domain
        W = self._codomain

        s1, s2 = V.starts
        e1, e2 = V.ends
        p1, p2 = V.pads
        n0     = W.ncoeff

        B_sd = self._data
        y    =  out._data

        # IMPORTANT: algorithm uses data in ghost regions
        if not v.ghost_regions_in_sync:
            v.update_ghost_regions()

        # Compute local contribution to global dot product
        for i in range(n0):
            y[i] = np.dot(B_sd[i, :, :].flat, v[0:p1, :].flat)

        # Sum contributions from all processes that share data at r=0
        if out.space.parallel:
            from mpi4py import MPI
            U = out.space
            if U.radial_comm.rank == U.radial_root:
                U.angle_comm.Allreduce(MPI.IN_PLACE, y, op=MPI.SUM)

        return out

    # ...
    def toarray(self , **kwargs):

        n0     = self.codomain.ncoeff

        n1, n2 = self.domain.npts
        p1, p2 = self.domain.pads
        s1, s2 = self.domain.starts
        e1, e2 = self.domain.ends

        a  = np.zeros((n0, n1*n2), dtype=self.codomain.dtype)
        d  = self._data

        for i in range(n0):
            for j1 in range(p1):
                j_start = j1*n2 + s2
                j_stop  = j1*n2 + e2 + 1
                a[i, j_start:j_stop] = d[i, j1, :]

        return a

    # ...
    def tosparse(self , **kwargs):
        return self.tocoo()

    # ...
    def copy(self):
        return LinearOperator_StencilToDense(self.domain, self.codomain, self._data.copy())

    # ...
    def __neg__(self):
        return LinearOperator_StencilToDense(self.domain, self.codomain, -self._data)

    # ...
    def __mul__(self, a):
        return LinearOperator_StencilToDense(self.domain, self.codomain, self._data * a)

    # ...
    def __add__(self, m):
        assert isinstance(m, LinearOperator_StencilToDense)
        assert self.  domain == m.  domain
        assert self.codomain == m.codomain
        return LinearOperator_StencilToDense(self.domain, self.codomain, self._data + m._data)

    # ...
    def __sub__(self, m):
        assert isinstance(m, LinearOperator_StencilToDense)
        assert self.  domain == m.  domain
        assert self.codomain == m.codomain
        return LinearOperator_StencilToDense(self.domain, self.codomain, self._data - m._data)

    # ...
    def __imul__(self, a):
        self._data *= a
        return self

    # ...
    def __iadd__(self, m):
        assert isinstance(m, LinearOperator_StencilToDense)
        assert self.  domain == m.  domain
        assert self.codomain == m.codomain
        self._data += m._data
        return self

    # ...
    def __isub__(self, m):
        assert isinstance(m, LinearOperator_StencilToDense)
        assert self.  domain == m.  domain
        assert self.codomain == m.codomain
        self._data -= m._data
        return self

    #-------------------------------------
    # Other properties/methods
    #-------------------------------------
    def tocoo(self):

        # Extract relevant information from vector spaces
        n0     = self.codomain.ncoeff
        n1, n2 = self.domain.npts
        p1, p2 = self.domain.pads
        s1, s2 = self.domain.starts
        e1, e2 = self.domain.ends

        # Compute 1D arrays 'data', 'rows', 'cols': data[i] = mat[rows[i],cols[i]]
        data  = []  # non-zero matrix entries
        rows  = []  # corresponding row indices i
        cols  = []  # corresponding column indices j
        for i in range(n0):
            for j1 in range(p1):
                data += self._data[i, j1, :].flat
                rows += repeat(i, e2-s2+1+2*p2)
                cols += (j1*n2 + i2%n2 for i2 in range(s2-p2, e2+1+p2))

        # Create Scipy sparse matrix in COO format
        coo = coo_matrix((data, (rows,cols)), shape=(n0, n1*n2), dtype=self.codomain.dtype)
        coo.eliminate_zeros()

        return coo

    def transpose(self, conjugate=False):
        raise NotImplementedError()

#==============================================================================
class LinearOperator_DenseToStencil(LinearOperator):

    def __init__(self, V, W, data):

        assert isinstance(V,   DenseVectorSpace)
        assert isinstance(W, StencilVectorSpace)

        # V space must have 3 components for now (TODO: change to arbitrary n)
        # W space must be 2D for now (TODO: extend to higher dimensions)

        s1, s2 = W.starts
        e1, e2 = W.ends
        p1, p2 = W.pads
        n0     = V.ncoeff

        data = np.asarray(data)
        assert data.shape == (p1, e2-s2+1, n0)

        # Store information in object
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
            assert isinstance(out, StencilVector)
            assert out.space is self._codomain
            out *= 0.0
        else:
            out = self._codomain.zeros()

        V = self._domain
        W = self._codomain

        s1, s2 = W.starts
        e1, e2 = W.ends
        p1, p2 = W.pads
        n0     = V.ncoeff

        B_ds = self._data
        x    =    v._data

        if n0 > 0:
            out[0:p1, s2:e2+1] = np.dot(B_ds, x)

        # IMPORTANT: flag that ghost regions are not up-to-date
        out.ghost_regions_in_sync = False

        return out

    # ...
    def toarray(self , **kwargs):

        n0     = self.domain.ncoeff

        n1, n2 = self.codomain.npts
        p1, p2 = self.codomain.pads
        s1, s2 = self.codomain.starts
        e1, e2 = self.codomain.ends

        a  = np.zeros((n1*n2, n0), dtype=self.codomain.dtype)
        d  = self._data

        for i1 in range(p1):
            for i2 in range(s2, e2+1):
                i = i1*n2 + i2
                a[i, :] = d[i1, i2, :]

        return a

    # ...
    def tosparse(self , **kwargs):
        return self.tocoo()

    # ...
    def copy(self):
        return LinearOperator_DenseToStencil(self.domain, self.codomain, self._data.copy())

    # ...
    def __neg__(self):
        return LinearOperator_DenseToStencil(self.domain, self.codomain, -self._data)

    # ...
    def __mul__(self, a):
        return LinearOperator_DenseToStencil(self.domain, self.codomain, self._data * a)

    # ...
    def __add__(self, m):
        assert isinstance(m, LinearOperator_DenseToStencil)
        assert self.  domain == m.  domain
        assert self.codomain == m.codomain
        return LinearOperator_DenseToStencil(self.domain, self.codomain, self._data + m._data)

    # ...
    def __sub__(self, m):
        assert isinstance(m, LinearOperator_DenseToStencil)
        assert self.  domain == m.  domain
        assert self.codomain == m.codomain
        return LinearOperator_DenseToStencil(self.domain, self.codomain, self._data - m._data)

    # ...
    def __imul__(self, a):
        self._data *= a
        return self

    # ...
    def __iadd__(self, m):
        assert isinstance(m, LinearOperator_DenseToStencil)
        assert self.  domain == m.  domain
        assert self.codomain == m.codomain
        self._data += m._data
        return self

    # ...
    def __isub__(self, m):
        assert isinstance(m, LinearOperator_DenseToStencil)
        assert self.  domain == m.  domain
        assert self.codomain == m.codomain
        self._data -= m._data
        return self
    #-------------------------------------
    # Other properties/methods
    #-------------------------------------
    def tocoo(self):

        # Extract relevant information from vector spaces
        n0     = self.domain.ncoeff
        n1, n2 = self.codomain.npts
        p1, p2 = self.codomain.pads
        s1, s2 = self.codomain.starts
        e1, e2 = self.codomain.ends

        # Compute 1D arrays 'data', 'rows', 'cols': data[i] = mat[rows[i],cols[i]]
        data  = []  # non-zero matrix entries
        rows  = []  # corresponding row indices i
        cols  = []  # corresponding column indices j
        for i1 in range(p1):
            for i2 in range(s2, e2+1):
                data += self._data[i1, i2, :].flat
                rows += repeat(i1*n2+i2, n0)
                cols += range(n0)

        # Create Scipy sparse matrix in COO format
        coo = coo_matrix((data, (rows, cols)), shape=(n1*n2, n0), dtype=self.codomain.dtype)
        coo.eliminate_zeros()

        return coo

    def transpose(self, conjugate=False):
        raise NotImplementedError()
