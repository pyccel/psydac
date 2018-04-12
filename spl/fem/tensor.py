# coding: utf-8

"""
We assume here that a tensor space is the product of fem spaces whom basis are
of compact support
"""

from spl.linalg.stencil import VectorSpace as StencilVectorSpace
from spl.fem.basic      import FemSpace


#===============================================================================
class TensorSpace( FemSpace ):
    """
    Generic Finite Element space V.

    """

    def __init__( self, *args, **kwargs ):
        """."""
        self._spaces = tuple(args)

        # serial case
        starts = [0 for V in self.spaces]
        ends = [V.nbasis for V in self.spaces]
        pads = [V.degree for V in self.spaces]
        self._vector_space = StencilVectorSpace(starts, ends, pads)

        # TODO parallel case

    @property
    def vector_space(self):
        """Returns the topological associated vector space."""
        return self._vector_space

    @property
    def spaces( self ):
        return self._spaces

    @property
    def nbasis(self):
        dims = [V.nbasis for V in self.spaces]
        dim = 1
        for d in dims:
            dim *= d
        return dim

    @property
    def degree(self):
        return [V.degree for V in self.spaces]

    def __str__(self):
        """Pretty printing"""
        txt  = '\n'
        txt += '> total nbasis  :: {dim}\n'.format(dim=self.nbasis)

        dims = ', '.join(str(V.nbasis) for V in self.spaces)
        txt += '> nbasis :: ({dims})\n'.format(dims=dims)
        return txt

