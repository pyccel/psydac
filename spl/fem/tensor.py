# coding: utf-8

"""
We assume here that a tensor space is the product of fem spaces whom basis are
of compact support

"""
from mpi4py import MPI
import numpy as np

from spl.linalg.stencil import StencilVectorSpace
from spl.fem.basic      import FemSpace, FemField
from spl.fem.splines    import SplineSpace
from spl.ddm.cart       import Cart
from spl.core.bsplines  import find_span, basis_funs

#===============================================================================
class TensorFemSpace( FemSpace ):
    """
    Tensor-product Finite Element space V.

    Notes
    -----
    For now we assume that this tensor-product space can ONLY be constructed
    from 1D spline spaces.

    """

    def __init__( self, *args, **kwargs ):
        """."""
        assert all( isinstance( s, SplineSpace ) for s in args )
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

        # Shortcut
        v = self._vector_space

        # Compute support of basis functions local to process
        degrees  = [V.degree for V in self.spaces]
        ncells   = [V.ncells for V in self.spaces]
        spans    = [V.spans  for V in self.spaces]
        supports = [[k for k in range( nc )
            if any( s <= i%nb <= e for i in range( span[k]-p, span[k]+1 ) )]
            for (s,e,p,nb,nc,span) in zip( v.starts, v.ends, degrees, npts, ncells, spans )]

        self._supports = tuple( tuple( np.unique( sup ) ) for sup in supports )

        # Determine portion of logical domain local to process
        coords = v.cart.coords if v.parallel else tuple( [0]*v.ndim )
        nprocs = v.cart.nprocs if v.parallel else tuple( [1]*v.ndim )

        iterator = lambda: zip( v.starts, v.ends, v.pads, coords, nprocs )

        self._element_starts = [(s   if c == 0    else s-p+1) for s,e,p,c,np in iterator()]
        self._element_ends   = [(e-p if c == np-1 else e-p+1) for s,e,p,c,np in iterator()]

        # Create (empty) dictionary that will contain all fields in this space
        self._fields = {}

    #--------------------------------------------------------------------------
    # Abstract interface: read-only attributes
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
    def fields( self ):
        """Dictionary containing all FemField objects associated to this space."""
        return self._fields

    #--------------------------------------------------------------------------
    # Abstract interface: evaluation methods
    #--------------------------------------------------------------------------
    def eval_field( self, field, *eta ):

        assert isinstance( field, FemField )
        assert field.space is self
        assert len( eta ) == self.ldim

        bases = []
        index = []

        for (x, space) in zip( eta, self.spaces ):

            knots  = space.knots
            degree = space.degree
            span   =  find_span( knots, degree, x )
            basis  = basis_funs( knots, degree, x, span )

            bases.append( basis )
            index.append( slice( span-degree, span+1 ) )

        # Get contiguous copy of the spline coefficients required for evaluation
        index  = tuple( index )
        coeffs = field.coeffs[index].copy()

        # Evaluation of multi-dimensional spline
        # TODO: optimize

        # Option 1: contract indices one by one and store intermediate results
        #   - Pros: small number of Python iterations = ldim
        #   - Cons: we create ldim-1 temporary objects of decreasing size
        #
        res = coeffs
        for basis in bases[::-1]:
            res = np.dot( res, basis )

#        # Option 2: cycle over each element of 'coeffs' (touched only once)
#        #   - Pros: no temporary objects are created
#        #   - Cons: large number of Python iterations = number of elements in 'coeffs'
#        #
#        res = 0.0
#        for idx,c in np.ndenumerate( coeffs ):
#            ndbasis = np.prod( [b[i] for i,b in zip( idx, bases )] )
#            res    += c * ndbasis

        return res

    # ...
    def eval_field_gradient( self, field, *eta ):

        assert isinstance( field, FemField )
        assert field.space is self
        assert len( eta ) == self.ldim

        raise NotImplementedError()

    #--------------------------------------------------------------------------
    # Other properties and methods
    #--------------------------------------------------------------------------
    @property
    def is_scalar(self):
        return True

    #TODO: return tuple instead of product?
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

    @property
    def spaces( self ):
        return self._spaces

    @property
    def local_support( self ):
        """
        Support of all the basis functions local to the process, in the form
        of ldim tuples with the element indices along each direction.

        Thanks to the presence of ghost values, this is also equivalent to the
        region over which the coefficients of all non-zero basis functions are
        available and hence a field can be evaluated.

        Returns
        -------
        element_supports : tuple of (tuple of int)
            Along each dimension, the basis support is a tuple of element indices.

        """
        return self._supports

    @property
    def local_domain( self ):
        """
        Logical domain local to the process, assuming the global domain is
        decomposed across processes without any overlapping.

        This information is fundamental for avoiding double-counting when computing
        integrals over the global domain.

        Returns
        -------
        element_starts : tuple of int
            Start element index along each direction.

        element_ends : tuple of int
            End element index along each direction.

        """
        return self._element_starts, self._element_ends

    @property
    def __str__(self):
        """Pretty printing"""
        txt  = '\n'
        txt += '> ldim   :: {ldim}\n'.format(ldim=self.ldim)
        txt += '> total nbasis  :: {dim}\n'.format(dim=self.nbasis)

        dims = ', '.join(str(V.nbasis) for V in self.spaces)
        txt += '> nbasis :: ({dims})\n'.format(dims=dims)
        return txt

