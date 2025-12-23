#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, dia_matrix

from sympde.topology.space import BasicFunctionSpace

from psydac.linalg.stencil        import StencilVectorSpace
from psydac.linalg.direct_solvers import BandedSolver, SparseSolver
from psydac.fem.basic             import FemSpace, FemField
from psydac.core.bsplines         import (
        find_span,
        basis_funs,
        collocation_matrix,
        histopolation_matrix,
        breakpoints,
        greville,
        make_knots,
        elevate_knots,
        basis_integrals,
        )

from psydac.utilities.utils import unroll_edges, refine_array_1d
from psydac.ddm.cart        import DomainDecomposition, CartDecomposition


__all__ = ('SplineSpace',)

#===============================================================================
class SplineSpace( FemSpace ):
    """
    a 1D Splines Finite Element space

    Parameters
    ----------
    degree : int
        Polynomial degree.

    knots : array_like
        Coordinates of knots (clamped or extended by periodicity).

    grid: array_like
        Coordinates of the grid. Used to construct the knots sequence, if not given.

    multiplicity: int
        Multiplicity of the knots in the knot sequence.
 
    parent_multiplicity: int
        Multiplicity of the parent knot sequence, if the space is reduced space.
 
    periodic : bool
        True if domain is periodic, False otherwise.
        Default: False

    dirichlet : tuple, list
        True if using homogeneous dirichlet boundary conditions, False
        otherwise. Must be specified for each bound
        Default: (False, False)

    basis : str
        Set to "B" for B-splines (have partition of unity)
        Set to "M" for M-splines (have unit integrals)

    """
    def __init__(self, degree, knots=None, grid=None, multiplicity=None, parent_multiplicity=None,
                 periodic=False, dirichlet=(False, False), basis='B', pads=None):

        if basis not in ['B', 'M']:
            raise ValueError(" only options for basis functions are B or M ")

        if (knots is not None) and (grid is not None):
            raise ValueError( 'Cannot provide both grid and knots.' )

        if (knots is None) and (grid is None):
            raise ValueError('Either knots or grid must be provided.')

        if (knots is not None) and (multiplicity is not None):
            raise ValueError( 'Cannot provide both knots and multiplicity.' )

        if (multiplicity is not None) and multiplicity<1:
            raise ValueError('multiplicity should be >=1')

        if (parent_multiplicity is not None) and parent_multiplicity<1:
            raise ValueError('parent_multiplicity should be >=1')

        if knots is None:
            if multiplicity is None:multiplicity = 1
            knots = make_knots( grid, degree, periodic, multiplicity )

        if grid is None:
            grid = breakpoints(knots, degree)

        indices = np.where(np.diff(knots[degree:len(knots)-degree])>1e-15)[0]

        if len(indices)>0:
            multiplicity = np.diff(indices).max(initial=1)
        else:
            multiplicity = max(1,len(knots[degree+1:-degree-1]))

        if parent_multiplicity is None:
            parent_multiplicity = multiplicity

        assert parent_multiplicity >= multiplicity

        # TODO: verify that user-provided knots make sense in periodic case

        # Number of basis function in space (= cardinality)
        if periodic:
            nbasis = len(knots) - 2*degree - 2 + multiplicity
        else:
            defect = 0
            if dirichlet[0]: defect += 1
            if dirichlet[1]: defect += 1
            nbasis = len(knots) - degree - 1 - defect

        # Coefficients to convert B-splines to M-splines (if needed)
        if basis == 'M':
            scaling_array = 1 / basis_integrals(knots, degree)
        else:
            scaling_array = None

        # Store attributes in object
        self._degree        = degree
        self._pads          = pads or degree
        self._knots         = knots
        self._periodic      = periodic # this is a scalar bool
        self._multiplicity  = multiplicity
        self._dirichlet     = dirichlet
        self._basis         = basis
        self._nbasis        = nbasis
        self._breaks        = grid
        self._ncells        = len(grid) - 1
        self._greville      = greville(knots, degree, periodic, multiplicity = multiplicity)
        self._ext_greville  = greville(elevate_knots(knots, degree, periodic, multiplicity=multiplicity), degree+1, periodic, multiplicity = multiplicity)
        self._scaling_array = scaling_array
        self._parent_multiplicity  = parent_multiplicity
        self._histopolation_grid   = unroll_edges(self.domain, self.ext_greville)

        # Create space of spline coefficients
        domain_decomposition = DomainDecomposition([self._ncells], [periodic])
        cart     = CartDecomposition(domain_decomposition, [nbasis], [np.array([0])],[np.array([nbasis-1])], [self._pads], [multiplicity])
        self._coeff_space = StencilVectorSpace(cart)

        # Store flag: object NOT YET prepared for interpolation / histopolation
        self._interpolation_ready = False
        self._histopolation_ready = False

        self._symbolic_space      = None
        # ...

    # ...
    @property
    def histopolation_grid(self):
        """
        Coordinates of the N+1 points x[i] that define the N 1D edges
        (x[i], x[i+1]) for histopolation, where N is equal to the number of
        basis functions (i.e. the cardinality of the space).

        In the non-periodic case x is simply the array of extended Greville
        points. In the periodic case we "unroll" the 1D edges to ensure that
        they correspond to positive, well-defined intervals with x[i] < x[i+1].

        """
        return self._histopolation_grid

    # ...
    def init_interpolation( self, dtype=float ):
        """
        Compute the 1D collocation matrix and factorize it, in preparation
        for the calculation of a spline interpolant given the values at the
        Greville points.

        """
        imat = collocation_matrix(
            knots    = self.knots,
            degree   = self.degree,
            periodic = self.periodic,
            normalization = self.basis,
            xgrid    = self.greville,
            multiplicity = self.multiplicity
        )

        if self.periodic:
            # Convert to CSC format and compute sparse LU decomposition
            self._interpolator = SparseSolver( csc_matrix( imat ) )
        else:
            # Convert to LAPACK banded format (see DGBTRF function)
            dmat = dia_matrix( imat )
            l = abs( dmat.offsets.min() )
            u =      dmat.offsets.max()
            cmat = csr_matrix( dmat )
            bmat = np.zeros( (1+u+2*l, cmat.shape[1]), dtype=dtype )
            for i,j in zip( *cmat.nonzero() ):
                bmat[u+l+i-j,j] = cmat[i,j]
            self._interpolator = BandedSolver( u, l, bmat )
        self.imat = imat

        # Store flag
        self._interpolation_ready = True

    # ...
    def init_histopolation( self, dtype=float):
        """
        Compute the 1D histopolation matrix and factorize it, in preparation
        for the calculation of a spline interpolant given the integrals within
        the cells defined by the extended Greville points.

        """
        imat = histopolation_matrix(
            knots    = self.knots,
            degree   = self.degree,
            periodic = self.periodic,
            normalization = self.basis,
            xgrid    = self.ext_greville,
            multiplicity = self._multiplicity
        )

        self.hmat= imat
        if self.periodic:
            # Convert to CSC format and compute sparse LU decomposition
            self._histopolator = SparseSolver( csc_matrix( imat ) )
        else:
            # Convert to LAPACK banded format (see DGBTRF function)
            dmat = dia_matrix( imat )
            l = abs( dmat.offsets.min() )
            u =      dmat.offsets.max()
            cmat = csr_matrix( dmat )
            bmat = np.zeros( (1+u+2*l, cmat.shape[1]), dtype=dtype)
            for i,j in zip( *cmat.nonzero() ):
                bmat[u+l+i-j,j] = cmat[i,j]
            self._histopolator = BandedSolver( u, l, bmat )

        # Store flag
        self._histopolation_ready = True

    #--------------------------------------------------------------------------
    # Abstract interface: read-only attributes
    #--------------------------------------------------------------------------
    @property
    def ldim( self ):
        """ Parametric dimension.
        """
        return 1

    @property
    def periodic( self ):
        """ True if domain is periodic, False otherwise.
        """
        # [YG, 28.03.2025]: according to the abstract interface of FemSpace,
        # this property should return a tuple of `ldim` booleans. Instead, this
        # property returns a single boolean.
        return self._periodic
    
    @property
    def pads( self ):
        """ Padding for potential parallel assembly.
        """
        return self._pads

    @property
    def mapping( self ):
        """ Assume identity mapping for now.
        """
        # [YG, 28.03.2025]: not clear why there should be no mapping here...
        # Clearly this property is never used in PSYDAC.
        return None

    @property
    def coeff_space( self ):
        """Returns the topological associated vector space."""
        return self._coeff_space

    @property
    def symbolic_space( self ):
        return self._symbolic_space

    @symbolic_space.setter
    def symbolic_space( self, symbolic_space ):
        assert isinstance(symbolic_space, BasicFunctionSpace)
        self._symbolic_space = symbolic_space

    @property
    def is_multipatch(self):
        return False

    @property
    def is_vector_valued(self):
        return False

    @property
    def patch_spaces(self):
        return (self,)

    @property
    def component_spaces(self):
        return (self,)

    @property
    def axis_spaces(self):
        return (self,)

    #--------------------------------------------------------------------------
    # Abstract interface: evaluation methods
    #--------------------------------------------------------------------------
    def eval_field(self, field, *eta , weights=None):
        assert isinstance( field, FemField )
        assert field.space is self
        assert len(eta) == 1

        eta = eta[0]

        span = find_span( self.knots, self.degree, eta)

        basis_array = basis_funs( self.knots, self.degree, eta, span)
        index = slice(span-self.degree, span + 1)

        if self.basis == 'M':
            basis_array *= self._scaling_array[index]

        coeffs = field.coeffs[index].copy()

        if weights:
            coeffs *= weights[index]

        return np.dot(coeffs,basis_array)

    # ...
    def eval_field_gradient( self, field, *eta , weights=None):

        assert isinstance( field, FemField )
        assert field.space is self
        assert len( eta ) == 1

        raise NotImplementedError()

    #--------------------------------------------------------------------------
    # Other properties
    #--------------------------------------------------------------------------
    @property
    def basis( self ):
        return self._basis

    @property
    def interpolation_grid( self ):
        if self.basis == 'B':
            return self.greville
        elif self.basis == 'M':
            return self.ext_greville
        else:
            raise NotImplementedError()

    @property
    def nbasis( self ):
        """ Number of basis functions, i.e. cardinality of spline space.
        """
        return self._nbasis

    @property
    def degree( self ):
        """ Spline degree.
        """
        return self._degree

    @property
    def ncells( self ):
        """ Number of cells in domain.
        """
        return self._ncells

    @property
    def dirichlet( self ):
        """ True if using homogeneous dirichlet boundary conditions, False otherwise.
        """
        return self._dirichlet

    @property
    def knots( self ):
        """ Knot sequence.
        """
        return self._knots

    @property
    def multiplicity( self ):
        return self._multiplicity

    @property
    def parent_multiplicity( self ):
        return self._parent_multiplicity

    @property
    def breaks( self ):
        """ List of breakpoints.
        """
        return self._breaks

    @property
    def domain( self ):
        """ Domain boundaries [a,b].
        """
        breaks = self.breaks
        return breaks[0], breaks[-1]

    @property
    def greville( self ):
        """ Coordinates of all Greville points. Used for interpolation.
        """
        return self._greville

    @property
    def ext_greville( self ):
        """ Greville coordinates of 'extended' space with degree p+1.
            Used for histopolation.
        """
        return self._ext_greville

    @property
    def scaling_array(self):
        """
        If self.basis=='M', return array used to rescale B-splines to M-splines
        If self.basis=='B', return None.

        The length of the scaling array is (len(knots)-degree-1).
        """
        return self._scaling_array

    #--------------------------------------------------------------------------
    # Other methods
    #--------------------------------------------------------------------------
    def compute_interpolant( self, values, field ):
        """
        Compute field (i.e. update its spline coefficients) such that it
        interpolates a certain function $f(x)$ at the Greville points.

        Parameters
        ----------
        values : array_like (nbasis,)
            Function values $f(x_i)$ at the 'nbasis' Greville points $x_i$,
            to be interpolated.

        field : FemField
            Input/output argument: spline that has to interpolate the given
            values.

        """
        assert len( values ) == self.nbasis
        assert isinstance( field, FemField )
        assert field.space is self

        if not self._interpolation_ready:
            self.init_interpolation()

        n = self.nbasis
        c = field.coeffs

        c[0:n] = self._interpolator.solve( values )
        c.update_ghost_regions()

    # ...
    def compute_histopolant( self, values, field ):
        """
        Compute field (i.e. update its spline coefficients) such that its
        integrals between the extended Greville points match the given
        values.

        Parameters
        ----------
        values : array_like (nbasis,)
            Integral values between the 'nbasis' extended Greville cells
            $[x_i, x_{i+1}]$, to be matched by the spline.

        field : FemField
            Input/output argument: spline that has to match the given
            integral values.

        """
        assert len( values ) == self.nbasis
        assert isinstance( field, FemField )
        assert field.space is self

        if not self._histopolation_ready:
            self.init_histopolation()

        n = self.nbasis
        c = field.coeffs

        c[0:n] = self._histopolator.solve( values )
        c.update_ghost_regions()

    # ...
    def refine(self, ncells):
        """
        Create a refined 1D spline space with the given number of cells.

        Parameters
        ----------
        ncells : int
            Number of cells of refined space. Must be multiple of self.ncells.

        Returns
        -------
        SplineSpace
            Refined 1D spline space which contains the original space.

        """

        # Sanity checks
        if int(ncells) != ncells:
            msg = f"{ncells} is not an integer"
        elif ncells < self.ncells:
            msg = f"{ncells} is smaller than minimum value {self.ncells}"
        elif ncells % self.ncells != 0:
            msg = f"{ncells} is not multiple of {self.ncells}"
        else:
            msg = None

        if msg:
            raise ValueError("Wrong number of cells: " + msg)

        if ncells == self.ncells:
            return self

        refinement_factor = ncells // self.ncells
        grid = refine_array_1d(self.breaks, refinement_factor)

        return SplineSpace(self.degree,
                           grid=grid,
                           multiplicity=self.multiplicity,
                           parent_multiplicity=self.parent_multiplicity,
                           periodic=self.periodic,
                           dirichlet=self.dirichlet,
                           basis=self.basis,
                           pads=self.pads)

    # ...
    def __str__(self):
        """Pretty printing"""
        txt  = '\n'
        txt += '> ldim   :: {ldim}\n'.format( ldim=self.ldim )
        txt += '> nbasis :: {dim} \n'.format( dim=self.nbasis )
        txt += '> degree :: {degree}'.format( degree=self.degree )
        return txt

    def draw(self):
        from scipy.interpolate import BSpline
        import matplotlib.pyplot as plt
        d = self.degree
        n = self.nbasis + d*self.periodic
        knots = self.knots
        fig, ax = plt.subplots()
        xx = np.linspace(knots[0], knots[-1], 200)
        for i in range(n):
            c = [0]*n
            c[i] = 1
            spl = BSpline(knots, c, d)
            ax.plot(xx, spl(xx), label='N{}'.format(i))
        ax.grid(True)
        ax.legend()
        plt.show()
