# coding: utf-8

import numpy as np

# TODO: to improve
class Vector(object):
    """
    Class that represents a stencil  vector.
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

        sizes = [e-s+2*p+1 for s,e,p in zip(self.starts, self.ends, self.pads)]
        print '>>> V size: ', sizes
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
    def __getitem__(self, *args):
        indx = list(*args)

        for i, z in enumerate(zip(self.starts, self.pads, indx)):
            if isinstance(z[2], slice):
                s1 = None
                s2 = None
                s3 = z[2].step

                if z[2].start is not None:
                    s1 = z[2].start - z[0] + z[1]
                if z[2].stop is not None:
                    s2 = z[2].stop  - z[0] + z[1]

                indx[i] = slice(s1, s2, s3)
            else:
                indx[i] = z[2] -  z[0] +z[1]

        return self._data[tuple(indx)]

    # ...
    def __setitem__(self, *args):
        item = args[-1]
        indx = args[:-1]
        indx = list(*indx)

        for i, z in enumerate(zip(self.starts, self.pads, indx)):
            if isinstance(z[2], slice):
                s1 = None
                s2 = None
                s3 = z[2].step

                if z[2].start is not None:
                    s1 = z[2].start - z[0] + z[1]
                if z[2].stop is not None:
                    s2 = z[2].stop  - z[0] + z[1]

                indx[i] = slice(s1, s2, s3)
            else:
                indx[i] = z[2] -  z[0] + z[1]

        self._data[tuple(indx)] = item


    def __str__(self):
        return str(self._data)

    # ...
    def toarray(self):
        """
        Convert the stencil data to sparce matrix in the array form
        """
        # TODO
        pass
    # ...
