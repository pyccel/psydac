# coding: utf-8

import numpy as np

# ...
class Matrix(object):
    """
    Class that represents a stencil matrix.
    """

    def __init__(self, starts, ends, pads):

        assert( len(starts) == len(ends) == len(pads) )

        self._starts = tuple(starts)
        self._ends   = tuple(ends)
        self._pads   = tuple(pads)
        self._ndim   = len(starts)

        sizes = [e-s+1 for s,e in zip(starts, ends)]
        pads  = [2*p+1 for p in pads]
        shape =  sizes + pads

        self._data = np.zeros(shape)

    @property
    def starts(self):
        return self._starts

    @property
    def ends(self):
        return self._ends

    @property
    def pads(self):
        return self._pads

    @property
    def ndim(self):
        return self._ndim

    # ...
    def __getitem__(self, key):
        nd = self._ndim
        ii = key[:nd]
        kk = key[nd:]

        index = []

        for (i,s) in zip(ii, self._starts):
            if isinstance(i, slice):
                start = None if i.start is None else s + i.start
                stop  = None if i.stop  is None else s + i.stop
                l = slice(start, stop, i.step)
            else:
                l = s + i
            index.append(l)

        for (k,p) in zip(kk, self._pads):
            if isinstance(k, slice):
                start = None if k.start is None else p + k.start
                stop  = None if k.stop  is None else p + k.stop
                l = slice(start, stop, k.step)
            else:
                l = p + k
            index.append(l)

        return self._data[tuple(index)]

    # ...
    def __setitem__(self, key, value):
        nd = self._ndim
        ii = key[:nd]
        kk = key[nd:]

        index = []

        for (i,s) in zip(ii, self._starts):
            if isinstance( i, slice ):
                start = None if i.start is None else s+i.start
                stop  = None if i.stop  is None else s+i.stop
                l = slice(start, stop, i.step)
            else:
                l = s + i
            index.append(l)

        for (k,p) in zip(kk, self._pads):
            if isinstance(k, slice):
                start = None if k.start is None else p+k.start
                stop  = None if k.stop  is None else p+k.stop
                l = slice(start, stop, k.step)
            else:
                l = p + k
            index.append(l)

        self._data[tuple(index)] = value

    def __str__(self):
        return str(self._data)
    # ...

    # ...
    def dot(self, v):

        if not isinstance(v, Vector):
            raise TypeError("v must be a Vector")

        # TODO check shapes

        [s1, s2] = self.starts
        [e1, e2] = self.ends
        [p1, p2] = self.pads

        # ...
        res = v.zeros_like()

        for i1 in range(s1, e1+1):
            for i2 in range(s2, e2+1):
                for k1 in range(-p1, p1+1):
                    for k2 in range(-p2, p2+1):
                        j1 = k1+i1
                        j2 = k2+i2
                        res[i1,i2] = res[i1,i2] + self[i1,i2,k1,k2] * v[j1,j2]
        # ...

        return res
    # ...

    # ...
    def tocoo(self):
        """
        Convert the stencil data to sparce matrix in the COOrdinate form
        """

        from scipy.sparse import coo_matrix

        s1 = self.starts[0]
        e1 = self.ends[0]

        s2 = self.starts[0]
        e2 = self.ends[1]

        p1 = self.pads[0]
        p2 = self.pads[1]

        n1 = e1 - s1 + 1
        n2 = e2 - s2 + 1

        rows = []
        cols = []
        vals = []

        # ...
        for i1 in range(s1, e1+1):
            for i2 in range(s2, e2+1):
                for k1 in range(-p1, p1+1):
                    for k2 in range(-p2, p2+1):
                        j1 = (k1 + i1)%n1
                        j2 = (k2 + i2)%n2
                        irow = i1 + n1*i2
                        icol = j1 + n1*j2

                        rows.append(irow)
                        cols.append(icol)
                        vals.append(self[i1, i2, k1, k2])
        # ...

        rows = np.array(rows)
        cols = np.array(cols)
        vals = np.array(vals)
        mat  = coo_matrix((vals, (rows, cols)), shape=(n1*n2, n1*n2))

        mat.eliminate_zeros()

        return mat
    # ...

# ...

# ... 
class Vector(object):
    """
    Class that represents a stencil  vector.
    """

    def __init__(self, starts, ends, pads):

        assert( len(starts) == len(ends) == len(pads) )

        self._starts = tuple(starts)
        self._ends   = tuple(ends)
        self._pads   = tuple(pads)
        self._ndim   = len(starts)

        self._starts = starts
        self._ends   = ends
        self._pads   = pads

        sizes = [e-s+2*p+1 for s,e,p in zip(starts, ends, pads)]
        self._data = np.zeros(sizes)

    @property
    def starts(self):
        return self._starts

    @property
    def ends(self):
        return self._ends

    @property
    def pads(self):
        return self._pads

    # ...
    def __getitem__(self, key):
        index = []

        for (i,s,p) in zip(key, self._starts, self._pads):
            if isinstance(i, slice):
                start = None if i.start is None else i.start + s - p
                stop  = None if i.stop  is None else i.stop  + s - p
                l = slice(start, stop, i.step)
            else:
                l = i + s - p

            index.append(l)

        return self._data[tuple(index)]
    # ...

    # ...
    def __setitem__(self, key, value):
        index = []

        for (i,s,p) in zip(key, self._starts, self._pads):
            if isinstance(i, slice):
                start = None if i.start is None else i.start + s - p
                stop  = None if i.stop  is None else i.stop  + s - p
                l = slice(start, stop, i.step)
            else:
                l = i + s - p

            index.append(l)

        self._data[tuple(index)] = value
    # ...

    # ...
    def __str__(self):
        return str(self._data)
    # ...

    # ...
    def zeros_like(self):
        """
        Return a Vector of zeros with the same shape a given Vector.
        """

        res = Vector(self.starts, self.ends, self.pads)
        res[:, :] = 0.

        return res
    # ...

    # ...
    def add(self, v):

        # ...
        if isinstance(v, int):
            self[:, :] = self[:, :] + v

        elif isinstance(v, float):
            self[:, :] = self[:, :] + v

        elif isinstance(v, Vector):
            # ... TODO check shapes
            self[:, :] = self[:, :] + v[:, :]

        else:
            raise TypeError("passed type must be int, float or Vector.")
        # ...
    # ...

    # ...
    def sub(self, v):

        # ...
        if isinstance(v, int):
            self[:, :] = self[:, :] - v

        elif isinstance(v, float):
            self[:, :] = self[:, :] - v

        elif isinstance(v, Vector):
            # ... TODO check shapes
            self[:, :] = self[:, :] - v[:, :]

        else:
            raise TypeError("passed type must be int, float or Vector")
        # ...
    # ...

    # ...
    def mul(self, v):

        # ...
        if isinstance(v, int):
            self[:, :] = v * self[:, :]

        elif isinstance(v, float):
            self[:, :] = v * self[:, :]
        else:
            raise TypeError("passed must be int, float")
        # ...
    # ...

    # ...
    def dot(self, v):

        if not isinstance(v, Vector):
            raise TypeError("passed type must be a Vector")

        # TODO check shapes

        [s1, s2] = self.starts
        [e1, e2] = self.ends
        [p1, p2] = self.pads

        # ...
        res = 0.0
        for i1 in range(s1, e1+1):
            for i2 in range(s2, e2+1):
                for k1 in range(-p1, p1+1):
                    for k2 in range(-p2, p2+1):
                        res += self[k1,i1] * v[k2,i2]
        # ...

        return res
    # ...

    # ...
    def copy(self):
        """
        Return a Vector copy of the given Vector.
        """

        res = Vector(self.starts, self.ends, self.pads)

        res[:, :] = self[:, :]

        return res
    # ...

    # ...
    def toarray(self):
        """
        Return a numpy ndarray corresponding to the given Vector.
        """

        [s1, s2] = self.starts
        [e1, e2] = self.ends
        [p1, p2] = self.pads
        n1 = e1 - s1 + 1
        n2 = e2 - s2 + 1
        res = np.zeros(n1*n2)

        # ...
        for i1 in range(s1, e1+1):
            for i2 in range(s2, e2+1):
                for k1 in range(-p1, p1+1):
                    for k2 in range(-p2, p2+1):
                        j1 = (k1 + i1)%n1
                        j2 = (k2 + i2)%n2
                        icol = j1 + n1*j2
                        res[icol] = self[j1,j2]
        # ...

        return res
    # ...
