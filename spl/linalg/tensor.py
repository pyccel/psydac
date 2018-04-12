# coding: utf-8

from spl.linalg.basic import VectorSpace

# TODO shall we keep it?


#===============================================================================
class TensorSpace( VectorSpace ):
    """
    Generic Finite Element space V.

    """

    def __init__(self, V1, V2, V3=None):
        """."""
        if V3:
            self._spaces = (V1, V2, V3)
        else:
            self._spaces = (V1, V2)

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
