# coding: utf-8

# TODO: - have a block version for VectorSpace when all component spaces are the same

from spl.linalg.stencil import StencilVectorSpace
from spl.fem.basic      import FemSpace

from numpy import unique, asarray, allclose


#===============================================================================
class VectorFemSpace( FemSpace ):
    """
    FEM space with a vector basis

    """

    def __init__( self, *args, **kwargs ):
        """."""
        self._spaces = tuple(args)

        # ... make sure that all spaces have the same parametric dimension
        pdims = [V.pdim for V in self.spaces]
        assert (len(unique(pdims)) == 1)

        self._pdim = pdims[0]
        # ...

        # ... make sure that all spaces have the same number of cells
        ncells = [V.ncells for V in self.spaces]
        if self.pdim == 1:
            assert( len(unique(ncells)) == 1 )
        else:
            ns = asarray(ncells[0])
            for ms in ncells[1:]:
                assert( allclose(ns, asarray(ms)) )

        self._ncells = ncells[0]
        # ...

        # TODO serial case
        self._vector_space = None

        # TODO parallel case

    @property
    def pdim( self ):
        """ Parametric dimension.
        """
        return self._pdim

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
        return sum(dims)

    @property
    def degree(self):
        return [V.degree for V in self.spaces]

    @property
    def ncells(self):
        return self._ncells

    @property
    def is_block(self):
        """Returns True if all components are identical spaces."""
        # TODO - improve this tests. for the moment, we only check the degree,
        #      - shall we check the bc too?

        degree = [V.degree for V in self.spaces]
        if self.pdim == 1:
            return len(unique(degree)) == 1
        else:
            ns = asarray(degree[0])
            for ms in degree[1:]:
                if not( allclose(ns, asarray(ms)) ): return False
            return True


    def __str__(self):
        """Pretty printing"""
        txt  = '\n'
        txt += '> pdim   :: {pdim}\n'.format(pdim=self.pdim)
        txt += '> total nbasis  :: {dim}\n'.format(dim=self.nbasis)

        dims = ', '.join(str(V.nbasis) for V in self.spaces)
        txt += '> nbasis :: ({dims})\n'.format(dims=dims)
        return txt

