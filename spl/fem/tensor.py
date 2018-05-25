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

        if 'comm' in kwargs:
            # parallel case
            comm = kwargs['comm']
            assert isinstance(comm, MPI.Comm)

            cart = Cart(npts = npts,
                        pads    = pads,
                        periods = periods,
                        reorder = True,
                        comm    = comm)

            self._vector_space = StencilVectorSpace(cart)

        else:
            # serial case
            self._vector_space = StencilVectorSpace(npts, pads, periods)

    #--------------------------------------------------------------------------
    # Abstract interface
    #--------------------------------------------------------------------------
    @property
    def ldim( self ):
        """ Parametric dimension.
        """
        return sum([V.ldim for V in self.spaces])

    @property
    def periodic(self):
        return [V.periodic for V in self.spaces]

    @property
    def mapping(self):
        return None

    @property
    def vector_space(self):
        """Returns the topological associated vector space."""
        return self._vector_space

    @property
    def is_scalar(self):
        return True

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
    def ncells(self):
        return [V.ncells for V in self.spaces]

    #--------------------------------------------------------------------------
    # Other properties and methods
    #--------------------------------------------------------------------------
    @property
    def spaces( self ):
        return self._spaces

    def __str__(self):
        """Pretty printing"""
        txt  = '\n'
        txt += '> ldim   :: {ldim}\n'.format(ldim=self.ldim)
        txt += '> total nbasis  :: {dim}\n'.format(dim=self.nbasis)

        dims = ', '.join(str(V.nbasis) for V in self.spaces)
        txt += '> nbasis :: ({dims})\n'.format(dims=dims)
        return txt

