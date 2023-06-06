# coding: utf-8

"""
We assume here that a tensor space is the product of fem spaces whom basis are
of compact support

"""
from mpi4py import MPI
import numpy as np
import itertools
import h5py
import os

from types import MappingProxyType

from sympde.topology.space import BasicFunctionSpace

from psydac.linalg.stencil   import StencilVectorSpace
from psydac.linalg.kron      import kronecker_solve
from psydac.fem.basic        import FemSpace, FemField
from psydac.fem.splines      import SplineSpace
from psydac.fem.grid         import FemAssemblyGrid
from psydac.fem.partitioning import create_cart, partition_coefficients
from psydac.ddm.cart         import DomainDecomposition

from psydac.core.bsplines  import (find_span,
                                   basis_funs,
                                   basis_funs_1st_der,
                                   basis_ders_on_quad_grid,
                                   elements_spans,
                                   cell_index,
                                   basis_ders_on_irregular_grid)

from psydac.core.kernels import (eval_fields_2d_no_weights,
                                 eval_fields_2d_irregular_no_weights,
                                 eval_fields_2d_weighted,
                                 eval_fields_2d_irregular_weighted,
                                 eval_fields_3d_no_weights,
                                 eval_fields_3d_irregular_no_weights,
                                 eval_fields_3d_weighted,
                                 eval_fields_3d_irregular_weighted)

__all__ = ('TensorFemSpace',)

#===============================================================================
class TensorFemSpace( FemSpace ):
    """
    Tensor-product Finite Element space V.

    Notes
    -----
    For now we assume that this tensor-product space can ONLY be constructed
    from 1D spline spaces.

    """

    def __init__( self, domain_decomposition, *spaces, vector_space=None, cart=None, quad_order=None ):
        """."""
        assert isinstance(domain_decomposition, DomainDecomposition)
        assert all( isinstance( s, SplineSpace ) for s in spaces )
        self._domain_decomposition = domain_decomposition
        self._spaces = tuple(spaces)

        if cart is not None:
            self._vector_space = StencilVectorSpace(cart)
        elif vector_space is not None:
            self._vector_space = vector_space
        else:
            cart               = create_cart([domain_decomposition], [self._spaces])
            self._vector_space = StencilVectorSpace(cart[0])

        # Shortcut
        v = self._vector_space

        if quad_order is None:
            self._quad_order = [sp.degree for sp in self.spaces]
        else:
            self._quad_order = quad_order

        self._symbolic_space = None
        self._refined_space  = {}
        self._interfaces     = {}
        self._interfaces_readonly = MappingProxyType(self._interfaces)

        if self._vector_space.parallel and self._vector_space.cart.is_comm_null:return

        starts = self._vector_space.cart.domain_decomposition.starts
        ends   = self._vector_space.cart.domain_decomposition.ends

        # Compute extended 1D quadrature grids (local to process) along each direction
        self._quad_grids = tuple( FemAssemblyGrid( V,s,e, nderiv=V.degree, quad_order=q)
                                  for V,s,e,q in zip( self.spaces, starts, ends, self._quad_order ) )

        # Determine portion of logical domain local to process
        self._element_starts = starts
        self._element_ends   = ends

        # Compute limits of eta_0, eta_1, eta_2, etc... in subdomain local to process
        self._eta_limits = tuple( (space.breaks[s], space.breaks[e+1])
           for s,e,space in zip( self._element_starts, self._element_ends, self.spaces ) )

        # Store flag: object NOT YET prepared for interpolation
        self._interpolation_ready = False
        # Compute the local domains for every process

        # ...
        self._global_element_starts = domain_decomposition.global_element_starts
        self._global_element_ends   = domain_decomposition.global_element_ends

        self.set_refined_space(self.ncells, self)
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
    def domain_decomposition(self):
        return self._domain_decomposition

    @property
    def mapping(self):
        return None

    @property
    def vector_space(self):
        """Returns the topological associated vector space."""
        return self._vector_space

    @property
    def is_product(self):
        return False

    @property
    def symbolic_space( self ):
        return self._symbolic_space 

    @property
    def interfaces( self ):
        return self._interfaces_readonly

    @symbolic_space.setter
    def symbolic_space( self, symbolic_space ):
        assert isinstance(symbolic_space, BasicFunctionSpace)
        self._symbolic_space = symbolic_space

    #--------------------------------------------------------------------------
    # Abstract interface: evaluation methods
    #--------------------------------------------------------------------------
    def eval_field( self, field, *eta, weights=None):

        assert isinstance( field, FemField )
        assert field.space is self
        assert len( eta ) == self.ldim
        if weights:
            assert weights.space == field.coeffs.space

        bases = []
        index = []

        # Necessary if vector coeffs is distributed across processes
        if not field.coeffs.ghost_regions_in_sync:
            field.coeffs.update_ghost_regions()

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
            basis = basis_funs( knots, degree, x, span)

            # If needed, rescale B-splines to get M-splines
            if space.basis == 'M':
                basis *= space.scaling_array[span-degree : span+1]

            # Determine local span
            wrap_x   = space.periodic and x > xlim[1]
            loc_span = span - space.nbasis if wrap_x else span

            bases.append( basis )
            index.append( slice( loc_span-degree, loc_span+1 ) )
        # Get contiguous copy of the spline coefficients required for evaluation
        index  = tuple( index )
        coeffs = field.coeffs[index].copy()
        if weights:
            coeffs *= weights[index]

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
    def preprocess_regular_tensor_grid(self, grid, der=0, overlap=0):
        """Returns all the quantities needed to evaluate fields on a regular tensor-grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        der : int, default=0
            Number of derivatives of the basis functions to pre-compute.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        degree : tuple of int
            Degree in each direction
        global_basis : List of ndarray
            List of 4D arrays, one per direction, containing the values of the p+1 non-vanishing
            basis functions (and their derivatives) at each grid point.
            The array for direction xi has shape (ne_xi, der + 1, p+1,  nv_xi).

        global_spans : List of ndarray
            List of 1D arrays, one per direction, containing the index of the last non-vanishing
            basis function in each cell. The array for direction xi has shape (ne_xi,).
        
        local_shape : List of tuple
            Shape of what is local to this instance. 
        """
        # Check the grid
        assert len(grid) == self.ldim

        # Get the local domain
        v = self.vector_space
        starts, ends = self.local_domain

        # Add the overlap if we are in parallel
        if v.parallel:
            starts = tuple(s - overlap if s!=0 else s for s in starts)
            ends = tuple(e + overlap for e in ends)

        # Compute the basis functions and spans.
        local_shape = [] 
        global_basis = []
        global_spans = []
        for i in range(self.ldim):
            # We only care about the local grid
            grid_local = grid[i][slice(starts[i], ends[i] + 1)]

            # Compute basis functions and spans
            global_basis_i = basis_ders_on_quad_grid(self.knots[i], self.degree[i], grid_local, der, self.spaces[i].basis, offset=starts[i])
            global_spans_i = elements_spans(self.knots[i], self.degree[i])[slice(starts[i], ends[i] + 1)] - v.starts[i] + v.shifts[i] * v.pads[i]

            local_shape.append(grid_local.shape)
            global_basis.append(global_basis_i)
            global_spans.append(global_spans_i)
        return self.degree, global_basis, global_spans, local_shape

    #...
    def preprocess_irregular_tensor_grid(self, grid, der=0, overlap=0):
        """Returns all the quantities needed to evaluate fields on a regular tensor-grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 1D arrays representing each direction of the grid.

        der : int, default=0
            Number of derivatives of the basis functions to pre-compute.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        pads : tuple of int
            Padding in each direction
        degree : tuple of int
            Degree in each direction
        global_basis : List of ndarray
            List of 4D arrays, one per direction, containing the values of the p+1 non-vanishing
            basis functions (and their derivatives) at each grid point.
            The array for direction xi has shape (n_xi, p+1, der + 1).

        global_spans : List of ndarray
            List of 1D arrays, one per direction, containing the index of the last non-vanishing
            basis function in each cell. The array for direction xi has shape (n_xi,).
        
        cell_indexes : list of ndarray
            List of 1D arrays, one per direction, containing the index of the cell in which
            the corresponding point in grid is.
        
        local_shape : List of tuple
            Shape of what is local to this instance. 
        """
        # Check the grid
        assert len(grid) == self.ldim

        # Get the local domain
        v = self.vector_space
        starts, ends = self.local_domain

        # Add the overlap if we are in parallel
        if v.parallel:
            starts = tuple(s - overlap if s!=0 else s for s in starts)
            ends = tuple(e + overlap for e in ends)

        # Compute the basis functions and spans and cell indexes.
        global_basis = []
        global_spans = []
        cell_indexes = []
        local_shape = []
        for i in range(self.ldim):
            # Check the that the grid is sorted.
            grid_i = grid[i]
            assert all(grid_i[j] <= grid_i[j + 1] for j in range(len(grid_i) - 1))

            # Get the cell indexes
            cell_index_i = cell_index(self.breaks[i], grid_i)
            min_idx = np.searchsorted(cell_index_i, starts[i], side='left')
            max_idx = np.searchsorted(cell_index_i, ends[i], side='right')
            # We only care about the local cells.
            cell_index_i = cell_index_i[min_idx:max_idx]
            grid_local_i = grid_i[min_idx:max_idx]

            # basis functions and spans
            global_basis_i = basis_ders_on_irregular_grid(self.knots[i], self.degree[i], grid_local_i, cell_index_i, der, self.spaces[i].basis)
            global_spans_i = elements_spans(self.knots[i], self.degree[i])[slice(starts[i], ends[i] + 1)] - v.starts[i] + v.shifts[i] * v.pads[i]

            local_shape.append(len(grid_local_i))
            global_basis.append(global_basis_i)
            global_spans.append(global_spans_i)
            
            # starts[i] is cell 0 of the local domain 
            cell_indexes.append(cell_index_i  - starts[i])

        return self.degree, global_basis, global_spans, cell_indexes, local_shape

    # ...
    def eval_fields(self, grid, *fields, weights=None, npts_per_cell=None, overlap=0):
        """Evaluate one or several fields at the given location(s) grid.

        Parameters
        ----------
        grid : List of ndarray
            Grid on which to evaluate the fields

        *fields : tuple of psydac.fem.basic.FemField
            Fields to evaluate

        weights : psydac.fem.basic.FemField or None, optional
            Weights field.

        npts_per_cell: int or tuple of int or None, optional
            number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        List of ndarray of floats
            List of the evaluated fields.
        """

        assert all(f.space is self for f in fields)
        for f in fields:
            # Necessary if vector coeffs is distributed across processes
            if not f.coeffs.ghost_regions_in_sync:
                f.coeffs.update_ghost_regions()
        
        if weights is not None:
            assert weights.space is self
            assert all(f.coeffs.space is weights.coeffs.space for f in fields)
            if not weights.coeffs.ghost_regions_in_sync:
                weights.coeffs.update_ghost_regions()
        
        assert len(grid) == self.ldim
        grid = [np.asarray(grid[i]) for i in range(self.ldim)]
        assert all(grid[i].ndim == grid[i + 1].ndim for i in range(self.ldim - 1))

        # --------------------------
        # Case 1. Scalar coordinates
        if (grid[0].size == 1) or grid[0].ndim == 0:
            if weights is not None:
                return [self.eval_field(f, *grid, weights=weights.coeffs) for f in fields]
            else:
                return [self.eval_field(f, *grid) for f in fields]

        # Case 2. 1D array of coordinates and no npts_per_cell is given
        # -> grid is tensor-product, but npts_per_cell is not the same in each cell
        elif grid[0].ndim == 1 and npts_per_cell is None:
            out_fields = self.eval_fields_irregular_tensor_grid(grid, *fields, weights=weights, overlap=overlap)
            return [np.ascontiguousarray(out_fields[..., i]) for i in range(len(fields))]

        # Case 3. 1D arrays of coordinates and npts_per_cell is a tuple or an integer
        # -> grid is tensor-product, and each cell has the same number of evaluation points
        elif grid[0].ndim == 1 and npts_per_cell is not None:
            if isinstance(npts_per_cell, int):
                npts_per_cell = (npts_per_cell,) * self.ldim
            for i in range(self.ldim):
                ncells_i = len(self.breaks[i]) - 1
                grid[i] = np.reshape(grid[i], newshape=(ncells_i, npts_per_cell[i]))
            out_fields = self.eval_fields_regular_tensor_grid(grid, *fields, weights=weights, overlap=overlap)
            # return a list
            return [np.ascontiguousarray(out_fields[..., i]) for i in range(len(fields))]

        # Case 4. (self.ldim)D arrays of coordinates and no npts_per_cell
        # -> unstructured grid
        elif grid[0].ndim == self.ldim and npts_per_cell is None:
            raise NotImplementedError("Unstructured grids are not supported yet.")

        # Case 5. Nonsensical input
        else:
            raise ValueError("This combination of argument isn't understood. The 4 cases understood are :\n"
                             "Case 1. Scalar coordinates\n"
                             "Case 2. 1D array of coordinates and no npts_per_cell is given\n"
                             "Case 3. 1D arrays of coordinates and npts_per_cell is a tuple or an integer\n"
                             "Case 4. {0}D arrays of coordinates and no npts_per_cell".format(self.ldim))

    # ...
    def eval_fields_regular_tensor_grid(self, grid, *fields, weights=None, overlap=0):
        """Evaluate fields on a regular tensor grid

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        *fields : tuple of psydac.fem.basic.FemField
            Fields to evaluate on `grid`.

        weights : psydac.fem.basic.FemField or None, optional
            Weights to apply to our fields.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        List of ndarray of float
            Values of the fields on the regular tensor grid
        """
        degree, global_basis, global_spans, local_shape = self.preprocess_regular_tensor_grid(grid, der=0, overlap=overlap)
        ncells = [local_shape[i][0] for i in range(self.ldim)]
        n_eval_points = [local_shape[i][1] for i in range(self.ldim)]
        out_fields = np.zeros((*(tuple(ncells[i] * n_eval_points[i] for i in range(self.ldim))), len(fields)))

        glob_arr_coeffs = np.zeros(shape=(*fields[0].coeffs._data.shape, len(fields)))

        for i in range(len(fields)):
            glob_arr_coeffs[..., i] = fields[i].coeffs._data

        if self.ldim == 2:
            if weights is None:
                eval_fields_2d_no_weights(ncells[0], ncells[1], degree[0], degree[1],
                                          n_eval_points[0], n_eval_points[1], global_basis[0], global_basis[1],
                                          global_spans[0], global_spans[1], glob_arr_coeffs, out_fields)
            else:

                global_weight_coeff = weights.coeffs._data

                eval_fields_2d_weighted(ncells[0], ncells[1], degree[0], degree[1],
                                        n_eval_points[0], n_eval_points[1], global_basis[0], global_basis[1],
                                        global_spans[0], global_spans[1], glob_arr_coeffs, global_weight_coeff,
                                        out_fields)

        elif self.ldim == 3:
            if weights is None:

                eval_fields_3d_no_weights(ncells[0], ncells[1], ncells[2], degree[0],
                                          degree[1], degree[2], n_eval_points[0], n_eval_points[1],
                                          n_eval_points[2], global_basis[0], global_basis[1], global_basis[2],
                                          global_spans[0], global_spans[1], global_spans[2], glob_arr_coeffs,
                                          out_fields)

            else:
                global_weight_coeff = weights.coeffs._data

                eval_fields_3d_weighted(ncells[0], ncells[1], ncells[2], degree[0],
                                        degree[1], degree[2], n_eval_points[0], n_eval_points[1], n_eval_points[2],
                                        global_basis[0], global_basis[1], global_basis[2], global_spans[0],
                                        global_spans[1], global_spans[2], glob_arr_coeffs, global_weight_coeff,
                                        out_fields)
        else:
            raise NotImplementedError("1D not Implemented")

        return out_fields

# ...
    def eval_fields_irregular_tensor_grid(self, grid, *fields, weights=None, overlap=0):
        """Evaluate fields on a regular tensor grid

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        *fields : tuple of psydac.fem.basic.FemField
            Fields to evaluate on `grid`.

        weights : psydac.fem.basic.FemField or None, optional
            Weights to apply to our fields.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        List of ndarray of float
            Values of the fields on the regular tensor grid
        """
        degree, global_basis, global_spans, cell_indexes, local_shape = \
            self.preprocess_irregular_tensor_grid(grid, overlap=overlap)
        out_fields = np.zeros(tuple(local_shape) + (len(fields),))

        glob_arr_coeffs = np.zeros(shape=(*fields[0].coeffs._data.shape, len(fields)))

        npoints = local_shape

        for i in range(len(fields)):
            glob_arr_coeffs[..., i] = fields[i].coeffs._data

        if self.ldim == 2:
            if weights is None:
                eval_fields_2d_irregular_no_weights(*npoints, *degree, *cell_indexes, *global_basis,
                                                    *global_spans, glob_arr_coeffs, out_fields)
            else:

                global_weight_coeff = weights.coeffs._data

                eval_fields_2d_irregular_weighted(*npoints, *degree, *cell_indexes, *global_basis,
                                                  *global_spans, glob_arr_coeffs, global_weight_coeff, out_fields)

        elif self.ldim == 3:
            if weights is None:

                eval_fields_3d_irregular_no_weights(*npoints, *degree, *cell_indexes, *global_basis,
                                                    *global_spans, glob_arr_coeffs, out_fields)

            else:
                global_weight_coeff = weights.coeffs._data

                eval_fields_3d_irregular_weighted(*npoints, *degree, *cell_indexes, *global_basis,
                                                  *global_spans, glob_arr_coeffs, global_weight_coeff, out_fields)
        else:
            raise NotImplementedError("1D not Implemented")

        return out_fields

    # ...
    def eval_field_gradient( self, field, *eta , weights=None):

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
            basis_0 = basis_funs(knots, degree, x, span)
            basis_1 = basis_funs_1st_der(knots, degree, x, span)

            # If needed, rescale B-splines to get M-splines
            if space.basis == 'M':
                scaling  = space.scaling_array[span-degree : span+1]
                basis_0 *= scaling
                basis_1 *= scaling

            # Determine local span
            wrap_x   = space.periodic and x > xlim[1]
            loc_span = span - space.nbasis if wrap_x else span

            bases_0.append( basis_0 )
            bases_1.append( basis_1 )
            index.append( slice( loc_span-degree, loc_span+1 ) )

        # Get contiguous copy of the spline coefficients required for evaluation
        index  = tuple( index )
        coeffs = field.coeffs[index].copy()
        if weights:
            coeffs *=  weights[index]

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

        # Extract and store quadrature data
        nq      = [g.num_quad_pts for g in self.quad_grids]
        points  = [g.points       for g in self.quad_grids]
        weights = [g.weights      for g in self.quad_grids]

        # Get local element range
        sk = [g.local_element_start for g in self.quad_grids]
        ek = [g.local_element_end   for g in self.quad_grids]

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
        if self.vector_space.parallel:
            mpi_comm = self.vector_space.cart.comm
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
    def knots(self):
        return [V.knots for V in self.spaces]

    @property
    def breaks(self):
        return [V.breaks for V in self.spaces]

    @property
    def degree(self):
        return [V.degree for V in self.spaces]

    @property
    def multiplicity(self):
        return [V.multiplicity for V in self.spaces]

    @property
    def pads(self):
        return self.vector_space.pads

    @property
    def ncells(self):
        return [V.ncells for V in self.spaces]

    @property
    def spaces( self ):
        return self._spaces
    
    @property
    def quad_order( self ):
        return self._quad_order

    @property
    def quad_grids( self ):
        """
        List of 'FemAssemblyGrid' objects (one for each direction)
        containing all 1D information local to process that is necessary
        for the correct assembly of the l.h.s. matrix and r.h.s. vector
        in a finite element method.

        """
        return self._quad_grids

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
    def global_element_starts( self ):
        return self._global_element_starts

    @property
    def global_element_ends( self ):
        return self._global_element_ends

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
    def init_interpolation( self ):
        for V in self.spaces:
            # TODO: check if OK to access private attribute...
            if not V._interpolation_ready:
                V.init_interpolation()

    # ...
    def init_histopolation( self ):
        for V in self.spaces:
            # TODO: check if OK to access private attribute...
            if not V._histopolation_ready:
                V.init_histopolation()

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

        if not self._interpolation_ready:
            self.init_interpolation()

        # TODO: check if OK to access private attribute '_interpolator' in self.spaces[i]
        kronecker_solve(
            solvers = [V._interpolator for V in self.spaces],
            rhs     = values,
            out     = field.coeffs,
        )

    # ...
    def reduce_grid(self, axes=(), knots=()):
        """ 
        Create a new TensorFemSpace object with a coarser grid than the original one
        we do that by giving a new knot sequence in the desired dimension.
            
        Parameters
        ----------
        axes : List of int
            Dimensions where we want to coarsen the grid.

        knots : List or tuple
            New knot sequences in each dimension.
 
        Returns
        -------
        V : TensorFemSpace
            New space with a coarser grid.

        """
        assert len(axes) == len(knots)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        v = self._vector_space
        spaces = list(self.spaces)

        global_starts = v._cart._global_starts.copy()
        global_ends   = v._cart._global_ends.copy()
        global_domains_ends  = self._global_element_ends

        for i, axis in enumerate(axes):
            space    = spaces[axis]
            degree   = space.degree
            periodic = space.periodic
            breaks   = space.breaks
            T        = list(knots[i]).copy()
            elements_ends = global_domains_ends[axis]
            boundaries    = breaks[elements_ends+1].tolist()

            for b in boundaries:
                if b not in T:
                    T.append(b)
            T.sort()

            new_space = SplineSpace(degree, knots=T, periodic=periodic,
                                    dirichlet=space.dirichlet, basis=space.basis)
            spaces[axis] = new_space
            breaks = new_space.breaks.tolist()
            elements_ends = np.array([breaks.index(bd) for bd in boundaries])-1
            elements_starts = np.array([0] + (elements_ends[:-1]+1).tolist())

            if periodic:
                global_starts[axis] = elements_starts
                global_ends[axis]   = elements_ends
            else:
                global_starts[axis] = elements_starts + degree - 1
                global_ends[axis]   = elements_ends   + degree - 1
                global_ends[axis][-1] += 1
                global_starts[axis][0] = 0

        cart = v._cart.reduce_grid(global_starts, global_ends)
        V    = TensorFemSpace(*spaces, cart=cart, quad_order=self._quad_order)
        return V

    # ...
    def export_fields( self, filename, **fields ):
        """ Write spline coefficients of given fields to HDF5 file.
        """
        assert isinstance( filename, str )
        assert all( field.space is self for field in fields.values() )

        V    = self.vector_space
        comm = V.cart.comm if V.parallel else None

        # Multi-dimensional index range local to process
        index = tuple( slice( s, e+1 ) for s,e in zip( V.starts, V.ends ) )

        # Create HDF5 file (in parallel mode if MPI communicator size > 1)
        kwargs = {}
        if comm is not None:
            if comm.size > 1:
                kwargs.update( driver='mpio', comm=comm )
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

        Returns
        -------
        fields : list of FemSpace objects
            Distributed fields, given in the same order of the names.

        """
        assert isinstance( filename, str )
        assert all( isinstance( name, str ) for name in field_names )

        V    = self.vector_space
        comm = V.cart.comm if V.parallel else None

        # Multi-dimensional index range local to process
        index = tuple( slice( s, e+1 ) for s,e in zip( V.starts, V.ends ) )

        # Open HDF5 file (in parallel mode if MPI communicator size > 1)
        kwargs = {}
        if comm is not None:
            if comm.size > 1:
                kwargs.update( driver='mpio', comm=comm )
        h5 = h5py.File( filename, mode='r', **kwargs )

        # Create fields and load their coefficients from HDF5 datasets
        fields = []
        for name in field_names:
            dset = h5[name]
            if dset.shape != V.npts:
                h5.close()
                raise TypeError( 'Dataset not compatible with spline space.' )
            field = FemField( self )
            field.coeffs[index] = dset[index]
            field.coeffs.update_ghost_regions()
            fields.append( field )

        # Close HDF5 file
        h5.close()

        return fields

    # ...
    def reduce_degree(self, axes, multiplicity=None, basis='B'):

        if isinstance(axes, int):
            axes = [axes]

        if isinstance(multiplicity, int):
            multiplicity = [multiplicity]

        if multiplicity is None:
            multiplicity = [self.multiplicity[i] for i in axes]

        v = self._vector_space

        spaces = list(self.spaces)

        for m, axis in zip(multiplicity, axes):
            space = spaces[axis]

            reduced_space = SplineSpace(
                degree    = space.degree - 1,
                pads      = space.pads,
                grid      = space.breaks,
                multiplicity= m,
                parent_multiplicity=space.multiplicity,
                periodic  = space.periodic,
                dirichlet = space.dirichlet,
                basis     = basis
            )
            spaces[axis] = reduced_space

        npts         = [s.nbasis for s in spaces]
        multiplicity = [s.multiplicity for s in spaces]

        global_starts, global_ends = partition_coefficients(v.cart.domain_decomposition, spaces)

        # create new CartDecomposition
        red_cart   = v.cart.reduce_npts(npts, global_starts, global_ends, shifts=multiplicity)

        # create new TensorFemSpace
        tensor_vec = TensorFemSpace(self._domain_decomposition, *spaces, cart=red_cart, quad_order=self._quad_order)

        tensor_vec._interpolation_ready = False

        for key in self._refined_space:
            if key == tuple(self.ncells):
                tensor_vec.set_refined_space(key, tensor_vec)
            else:
                tensor_vec.set_refined_space(key, self._refined_space[key].reduce_degree(axes, multiplicity, basis))
        return tensor_vec

    # ...
    def add_refined_space(self, ncells):
        """ refine the space with new ncells and add it to the dictionary of refined_space"""

        ncells = tuple(ncells)
        if ncells in self._refined_space: return
        if ncells == tuple(self.ncells):
            self.set_refined_space(ncells, self)
            return

        spaces = [s.refine(n) for s,n in zip(self.spaces, ncells)]
        npts   = [s.nbasis for s in spaces]
        domain = self.domain_decomposition
        new_global_starts = []
        new_global_ends   = []
        for i in range(domain.ndim):
            gs = domain.global_element_starts[i]
            ge = domain.global_element_ends  [i]
            new_global_starts.append([])
            new_global_ends  .append([])
            for s,e in zip(gs, ge):
                bs = self.spaces[i].breaks[s]
                be = self.spaces[i].breaks[e+1]
                s  = spaces[i].breaks.tolist().index(bs)
                e  = spaces[i].breaks.tolist().index(be)
                new_global_starts[-1].append(s)
                new_global_ends  [-1].append(e-1)

            new_global_starts[-1] = np.array(new_global_starts[-1])
            new_global_ends  [-1] = np.array(new_global_ends  [-1])

        domain = domain.refine(ncells, new_global_starts, new_global_ends)

        FS     = TensorFemSpace(domain, *spaces, quad_order=self.quad_order)
        self.set_refined_space(ncells, FS)

    # ...
    def create_interface_space(self, axis, ext, cart):
        """ Create a new interface fem space along a given axis and extremity.

        Parameters
        ----------
         axis : int
          The axis of the new Interface space.

         ext: int
          The extremity of the new Interface space.
          the values must be 1 or -1.

         cart: CartDecomposition
          The cart of the new space, needed in the parallel case.
        """
        axis = int(axis)
        ext  = int(ext)
        assert axis<self.ldim
        assert ext in [-1,1]

        if cart.is_comm_null or self._interfaces.get((axis, ext), None): return
        spaces       = self.spaces
        vector_space = self.vector_space
        quad_order   = self.quad_order

        vector_space.set_interface(axis, ext, cart)
        space = TensorFemSpace( self._domain_decomposition, *spaces, vector_space=vector_space.interfaces[axis, ext], quad_order=self.quad_order)
        self._interfaces[axis, ext] = space

    def get_refined_space(self, ncells):
        return self._refined_space[tuple(self.ncells)]

    def set_refined_space(self, ncells, new_space):
        self._refined_space[tuple(self.ncells)] = new_space

    # ...
    def plot_2d_decomposition( self, mapping=None, refine=10 ):

        import matplotlib.pyplot as plt
        from matplotlib.patches  import Polygon, Patch
        from psydac.utilities.utils import refine_array_1d

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
        pcoords = np.array([[mapping(e1, e2) for e2 in eta2] for e1 in eta1])

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
        pcoords = np.array([[mapping(e1, e2) for e2 in eta2] for e1 in eta1])
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
