# coding: utf-8

import numpy as np

class Stencil(object):
    """
    Class that represents a stencil matrix.
    """

    def __init__(self, starts, ends, pads):

        dim = 1
        if (isinstance(starts, (list, tuple)) and isinstance(ends, (list, tuple))
            and  isinstance(pads, (list, tuple))):
            dim = len(starts)
            assert(len(ends) == dim)
            assert(len(pads) == dim)

        if not isinstance(starts, (list, tuple)):
            starts = list(starts)

        if not isinstance(ends, (list, tuple)):
            ends = list(ends)

        if not isinstance(pads, (list, tuple)):
            pads = list(pads)

        self._starts = starts
        self._ends   = ends
        self._pads   = pads

        sizes = [e-s+1 for s,e in zip(self.starts, self.ends)]
        pads  = [2*p+1 for p in pads]
        shape = pads + sizes

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

    # ... TODO shift also the nx and ny index using (end - start)
    def __getitem__(self, *args):
        indx = list(*args)

        for i, z in enumerate(zip(self.pads, indx)):
            if isinstance(z[1], slice):
                s1 = None
                s2 = None
                s3 = z[1].step

                if z[1].start is not None:
                    s1 = z[1].start + z[0]
                if z[1].stop is not None:
                    s2 = z[1].stop  + z[0]

                indx[i] = slice(s1, s2, s3)
            else:
                indx[i] = z[0] + z[1]

        return self._data[tuple(indx)]

    # ... TODO shift also the nx and ny index using (end - start)
    def __setitem__(self, *args):
        item = args[-1]
        indx = args[:-1]
        indx = list(*indx)

        for i, z in enumerate(zip(self.pads, indx)):
            if isinstance(z[1], slice):
                s1 = None
                s2 = None
                s3 = p[1].step

                if z[1].start is not None:
                    s1 = z[1].start + z[0]
                if z[1].stop is not None:
                    s2 = z[1].stop  + z[0]

                indx[i] = slice(s1, s2, s3)
            else:
                indx[i] = z[0] + z[1]

        self._data[tuple(indx)] = item

    def __str__(self):
        return str(self._data)

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
                        vals.append(self[k1, k2, i1, i2])
        # ...

        rows = np.array(rows)
        cols = np.array(cols)
        vals = np.array(vals)
        mat  = coo_matrix((vals, (rows, cols)), shape=(n1*n2, n1*n2))

        mat.eliminate_zeros()

        return mat
    # ...

