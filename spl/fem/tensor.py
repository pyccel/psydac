# coding: utf-8

"""
We assume here that a tensor space is the product of fem spaces whom basis are
of compact support

"""
from mpi4py import MPI
import numpy as np
import itertools
import h5py

from spl.linalg.stencil import StencilVectorSpace
from spl.linalg.kron    import kronecker_solve
from spl.fem.basic      import FemSpace, FemField
from spl.fem.splines    import SplineSpace
from spl.fem.grid       import FemAssemblyGrid
from spl.ddm.cart       import Cart
from spl.core.bsplines  import find_span, basis_funs, basis_funs_1st_der

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

#        iterator = lambda: zip( v.starts, v.ends, v.pads, coords, nprocs )
#
#        self._element_starts = [(s   if c == 0    else s-p+1) for s,e,p,c,np in iterator()]
#        self._element_ends   = [(e-p if c == np-1 else e-p+1) for s,e,p,c,np in iterator()]

        self._element_starts = []
        self._element_ends   = []
        for (s,e,p,period,c,npr) in zip( v.starts, v.ends, v.pads, v.periods, coords, nprocs ):
            if period:
                start = s
                end   = e
            else:
                start = s   if c == 0     else s-p+1
                end   = e-p if c == npr-1 else e-p+1
            self._element_starts.append( start )
            self._element_ends  .append( end   )

        # Compute limits of eta_0, eta_1, eta_2, etc... in subdomain local to process
        self._eta_limits = tuple( (space.breaks[s], space.breaks[e+1])
           for s,e,space in zip( self._element_starts, self._element_ends, self.spaces ) )

        # Create (empty) dictionary that will contain all fields in this space
        self._fields = {}

        # Store flag: object NOT YET prepared for interpolation
        self._collocation_ready = False

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

        for (x, xlim, space) in zip( eta, self.eta_lims, self.spaces ):

            knots  = space.knots
            degree = space.degree
            span   =  find_span( knots, degree, x )
            #-------------------------------------------------#
            # Fix span for boundaries between subdomains      #
            #-------------------------------------------------#
            # TODO: Use local knot sequence instead of global #
            #       one to get correct span in all situations #
            #-------------------------------------------------#
            if x == xlim[1] and x != knots[-1-degree]:
                span -= 1
            #-------------------------------------------------#
            basis  = basis_funs( knots, degree, x, span )

            # Determine local span
            wrap_x   = space.periodic and x > xlim[1]
            loc_span = span - space.nbasis if wrap_x else span

            bases.append( basis )
            index.append( slice( loc_span-degree, loc_span+1 ) )

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

        bases_0 = []
        bases_1 = []
        index   = []

        for (x, xlim, space) in zip( eta, self.eta_lims, self.spaces ):

            knots   = space.knots
            degree  = space.degree
            span    =  find_span( knots, degree, x )
            #-------------------------------------------------#
            # Fix span for boundaries between subdomains      #
            #-------------------------------------------------#
            # TODO: Use local knot sequence instead of global #
            #       one to get correct span in all situations #
            #-------------------------------------------------#
            if x == xlim[1] and x != knots[-1-degree]:
                span -= 1
            #-------------------------------------------------#
            basis_0 = basis_funs( knots, degree, x, span )
            basis_1 = basis_funs_1st_der( knots, degree, x, span )

            # Determine local span
            wrap_x   = space.periodic and x > xlim[1]
            loc_span = span - space.nbasis if wrap_x else span

            bases_0.append( basis_0 )
            bases_1.append( basis_1 )
            index.append( slice( loc_span-degree, loc_span+1 ) )

        # Get contiguous copy of the spline coefficients required for evaluation
        index  = tuple( index )
        coeffs = field.coeffs[index].copy()

        # Evaluate each component of the gradient using algorithm described in "Option 1" above
        grad = []
        for d in range( self.ldim ):
            bases = [(bases_1[d] if i==d else bases_0[i]) for i in range( self.ldim )]
            res   = coeffs
            for basis in bases[::-1]:
                res = np.dot( res, basis )
            grad.append( res )

        return grad

    # ...
    def integral( self, f ):

        assert hasattr( f, '__call__' )

        # Compute quadrature data
        self.init_fem()

        # Extract and store quadrature data
        nq      = [V.quad_order   for V in self.spaces]
        points  = [V.quad_points  for V in self.spaces]
        weights = [V.quad_weights for V in self.spaces]

        # Get element range
        sk, ek  = self.local_domain

        # Iterator over multi-index k (equivalent to nested loops over each dimension)
        multi_range = lambda starts, ends: \
                itertools.product( *[range(s,e+1) for s,e in zip(starts,ends)] )

        # Shortcut: Numpy product of all elements in a list
        np_prod = np.prod

        # Perform Gaussian quadrature in multiple dimensions
        c = 0.0
        for k in multi_range( sk, ek ):

            x = [ points_i[k_i,:] for  points_i,k_i in zip( points,k)]
            w = [weights_i[k_i,:] for weights_i,k_i in zip(weights,k)]

            for q in np.ndindex( *nq ):

                y  = [x_i[q_i] for x_i,q_i in zip(x,q)]
                v  = [w_i[q_i] for w_i,q_i in zip(w,q)]

                c += f(*y) * np_prod( v )

        # All reduce (MPI_SUM)
        # TODO: verify that it is OK to access private attribute
        if self.vector_space.parallel:
            mpi_comm = self.vector_space.cart._comm
            c = mpi_comm.allreduce( c )

        return c

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
    def eta_lims( self ):
        """
        Eta limits of domain local to the process (for field evaluation).

        Returns
        -------
        eta_limits: tuple of (2-tuple of float)
            Along each dimension i, limits are given as (eta^i_{min}, eta^i_{max}).

        """
        return self._eta_limits

    # ...
    def init_fem( self ):
        for V in self.spaces:
            if V.quad_basis is None:
                V.init_fem()

        # NEW
        spaces = self.spaces
        starts = self.vector_space.starts
        ends   = self.vector_space.ends

        self.quad_grids = tuple( FemAssemblyGrid( V,s,e )
                                 for V,s,e in zip( spaces, starts, ends ) )

    # ...
    def init_collocation( self ):
        for V in self.spaces:
            # TODO: check if OK to access private attribute...
            if not V._collocation_ready:
                V.init_collocation()

    # ...
    def compute_interpolant( self, values, field ):
        """
        Compute field (i.e. update its spline coefficients) such that it
        interpolates a certain function $f(x1,x2,..)$ at the Greville points.

        Parameters
        ----------
        values : StencilVector
            Function values $f(x_i)$ at the n-dimensional tensor grid of
            Greville points $x_i$, to be interpolated.

        field : FemField
            Input/output argument: tensor spline that has to interpolate the given
            values.

        """
        assert values.space is self.vector_space
        assert isinstance( field, FemField )
        assert field.space is self

        if not self._collocation_ready:
            self.init_collocation()

        # TODO: check if OK to access private attribute '_interpolator' in self.spaces[i]
        kronecker_solve(
            solvers = [V._interpolator for V in self.spaces],
            rhs     = values,
            out     = field.coeffs,
        )

    # ...
    def export_fields( self, filename, **fields ):
        """ Write spline coefficients of given fields to HDF5 file.
        """
        assert isinstance( filename, str )
        assert all( field.space is self for field in fields.values() )

        V    = self.vector_space
        comm = V.cart.comm

        # Multi-dimensional index range local to process
        index = tuple( slice( s, e+1 ) for s,e in zip( V.starts, V.ends ) )

        # Create HDF5 file (in parallel mode if MPI communicator size > 1)
        kwargs = dict( driver='mpio', comm=comm ) if comm.size > 1 else {}
        h5 = h5py.File( filename, mode='w', **kwargs )

        # Add field coefficients as named datasets
        for name,field in fields.items():
            dset = h5.create_dataset( name, shape=V.npts, dtype=V.dtype )
            dset[index] = field.coeffs[index]

        # Close HDF5 file
        h5.close()

    # ...
    def import_fields( self, filename, *field_names ):
        """
        Load fields from HDF5 file containing spline coefficients.

        Parameters
        ----------
        filename : str
            Name of HDF5 input file.

        field_names : list of str
            Names of the datasets with the required spline coefficients.

        Results
        -------
        fields : list of FemSpace objects
            Distributed fields, given in the same order of the names.

        """
        assert isinstance( filename, str )
        assert all( isinstance( name, str ) for name in field_names )

        V    = self.vector_space
        comm = V.cart.comm

        # Multi-dimensional index range local to process
        index = tuple( slice( s, e+1 ) for s,e in zip( V.starts, V.ends ) )

        # Open HDF5 file (in parallel mode if MPI communicator size > 1)
        kwargs = dict( driver='mpio', comm=comm ) if comm.size > 1 else {}
        h5 = h5py.File( filename, mode='r', **kwargs )

        # Create fields and load their coefficients from HDF5 datasets
        fields = []
        for name in field_names:
            dset = h5[name]
            if dset.shape != V.npts:
                h5.close()
                raise TypeError( 'Dataset not compatible with spline space.' )
            field = FemField( self, name )
            field.coeffs[index] = dset[index]
            field.coeffs.update_ghost_regions()
            fields.append( field )

        # Close HDF5 file
        h5.close()

        return fields

    # ...
    def plot_2d_decomposition( self, mapping=None, refine=10 ):

        import matplotlib.pyplot as plt
        from matplotlib.patches  import Polygon, Patch
        from spl.utilities.utils import refine_array_1d

        if mapping is None:
            mapping = lambda eta: eta
        else:
            assert mapping.ldim == self.ldim == 2
            assert mapping.pdim == self.ldim == 2

        assert refine >= 1
        N = int( refine )
        V1, V2 = self.spaces

        mpi_comm = self.vector_space.cart.comm
        mpi_rank = mpi_comm.rank

        # Local grid, refined
        [sk1,sk2], [ek1,ek2] = self.local_domain
        eta1 = refine_array_1d( V1.breaks[sk1:ek1+2], N )
        eta2 = refine_array_1d( V2.breaks[sk2:ek2+2], N )
        pcoords = np.array( [[mapping( [e1,e2] ) for e2 in eta2] for e1 in eta1] )

        # Local domain as Matplotlib polygonal patch
        AB = pcoords[   :,    0, :] # eta2 = min
        BC = pcoords[  -1,    :, :] # eta1 = max
        CD = pcoords[::-1,   -1, :] # eta2 = max (points must be reversed)
        DA = pcoords[   0, ::-1, :] # eta1 = min (points must be reversed)
        xy = np.concatenate( [AB, BC, CD, DA], axis=0 )
        poly = Polygon( xy, edgecolor='None' )

        # Gather polygons on master process
        polys = mpi_comm.gather( poly )

        #-------------------------------
        # Non-master processes stop here
        if mpi_rank != 0:
            return
        #-------------------------------

        # Global grid, refined
        eta1    = refine_array_1d( V1.breaks, N )
        eta2    = refine_array_1d( V2.breaks, N )
        pcoords = np.array( [[mapping( [e1,e2] ) for e2 in eta2] for e1 in eta1] )
        xx      = pcoords[:,:,0]
        yy      = pcoords[:,:,1]

        # Plot decomposed domain
        fig, ax = plt.subplots( 1, 1 )
        colors  = itertools.cycle( plt.rcParams['axes.prop_cycle'].by_key()['color'] )
        handles = []
        for i, (poly, color) in enumerate( zip( polys, colors ) ):
            # Add patch
            poly.set_facecolor( color )
            ax.add_patch( poly )
            # Create legend entry
            handle = Patch( color=color, label='Rank {}'.format(i) )
            handles.append( handle )

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( 'Domain decomposition' )
        ax.plot( xx[:,::N]  , yy[:,::N]  , 'k' )
        ax.plot( xx[::N,:].T, yy[::N,:].T, 'k' )
        ax.set_aspect('equal')
        ax.legend( handles=handles, bbox_to_anchor=(1.05, 1), loc=2 )
        fig.tight_layout()
        fig.show()

    # ...
    def __str__(self):
        """Pretty printing"""
        txt  = '\n'
        txt += '> ldim   :: {ldim}\n'.format(ldim=self.ldim)
        txt += '> total nbasis  :: {dim}\n'.format(dim=self.nbasis)

        dims = ', '.join(str(V.nbasis) for V in self.spaces)
        txt += '> nbasis :: ({dims})\n'.format(dims=dims)
        return txt

