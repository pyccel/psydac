# coding: utf-8

"""
We assume here that a tensor space is the product of fem spaces whom basis are
of compact support
"""

from mpi4py             import MPI
from spl.linalg.stencil import StencilVectorSpace
from spl.fem.basic      import FemSpace
from spl.ddm.cart       import Cart

#===============================================================================
class TensorFemSpace( FemSpace ):
    """
    Generic Finite Element space V.

    """

    def __init__( self, *args, **kwargs ):
        """."""
        self._spaces = tuple(args)

        npts = [V.nbasis for V in self.spaces]
        pads = [V.degree for V in self.spaces]
        periods = [V.periodic for V in self.spaces]

        if kwargs == {}:
            # serial case
            self._vector_space = StencilVectorSpace(npts, pads, periods)

        else:
            # parallel case
            for key, value in kwargs.items():
                if key == 'comm':
                    # TODO add reorder as param ?
                    assert isinstance(value, MPI.Comm)
                    cart = Cart(npts    = npts,
                                pads    = pads,
                                periods = periods,
                                reorder = False,
                                comm    = value)

                    self._vector_space = StencilVectorSpace(cart)

                else:
                    # ... TODO add case if cart in given as argument
                    print('Not yet implemented.')


    @property
    def pdim( self ):
        """ Parametric dimension.
        """
        return sum([V.pdim for V in self.spaces])

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

    @property
    def periodic(self):
        return [V.periodic for V in self.spaces]

    @property
    def ncells(self):
        return [V.ncells for V in self.spaces]

    def __str__(self):
        """Pretty printing"""
        txt  = '\n'
        txt += '> pdim   :: {pdim}\n'.format(pdim=self.pdim)
        txt += '> total nbasis  :: {dim}\n'.format(dim=self.nbasis)

        dims = ', '.join(str(V.nbasis) for V in self.spaces)
        txt += '> nbasis :: ({dims})\n'.format(dims=dims)
        return txt

