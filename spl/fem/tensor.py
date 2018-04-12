# coding: utf-8

"""
We assume here that a tensor space is the product of fem spaces whom basis are
of compact support
"""

from spl.linalg.stencil import VectorSpace
from spl.fem.basic      import SpaceBase


#===============================================================================
class TensorSpace( SpaceBase ):
    """
    Generic Finite Element space V.

    """

    def __init__(self, V1, V2, V3=None):
        """."""
        if V3:
            self._spaces = (V1, V2, V3)
        else:
            self._spaces = (V1, V2)

        # serial case
        starts = [0 for V in self.spaces]
        ends = [V.dimension for V in self.spaces]
        pads = [V.degree for V in self.spaces]
        self._vector_space = VectorSpace(starts, ends, pads)

        # TODO parallel case

    @property
    def vector_space(self):
        """Returns the topological associated vector space."""
        return self._vector_space

    @property
    def spaces( self ):
        return self._spaces

    @property
    def dimension(self):
        dims = [V.dimension for V in self.spaces]
        dim = 1
        for d in dims:
            dim *= d
        return dim

    def __str__(self):
        """Pretty printing"""
        txt  = '\n'
        txt += '> Dimension  :: {dim}\n'.format(dim=self.dimension)

        dims = ', '.join(str(V.dimension) for V in self.spaces)
        txt += '> Dimensions :: ({dims})\n'.format(dims=dims)
        return txt

