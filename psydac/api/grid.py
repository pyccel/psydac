# coding: utf-8

import numpy as np

from psydac.core.bsplines          import find_span
from psydac.core.bsplines          import basis_funs_all_ders
from psydac.core.bsplines          import basis_ders_on_quad_grid
from psydac.fem.splines            import SplineSpace
from psydac.fem.tensor             import TensorFemSpace
from psydac.fem.vector             import ProductFemSpace
#==============================================================================
class QuadratureGrid():
    def __init__( self, V , axis=None, ext=None):

        if isinstance(V, ProductFemSpace):
            V = V.spaces[0]

        self._fem_grid            = V.quad_grids
        self._n_elements          = [g.num_elements        for g in V.quad_grids]
        self._local_element_start = [g.local_element_start for g in V.quad_grids]
        self._local_element_end   = [g.local_element_end   for g in V.quad_grids]
        self._points              = [g.points              for g in V.quad_grids]
        self._weights             = [g.weights             for g in V.quad_grids]
        self._axis                = axis
        self._ext                 = ext

        if axis is not None:
            assert ext is not None
            points  = self.points
            weights = self.weights

            # ...
            if V.ldim == 1 and isinstance(V, SplineSpace):
                bounds = {-1: V.domain[0],
                           1: V.domain[1]}
            elif isinstance(V, TensorFemSpace):
                bounds = {-1: V.spaces[axis].domain[0],
                           1: V.spaces[axis].domain[1]}
            else:
                raise ValueError('Incompatible type(V) = {} in {} dimensions'.format(
                    type(V), V.ldim))

            points [axis] = np.asarray([[bounds[ext]]])
            weights[axis] = np.asarray([[1.]])
            # ...
            self._points  = points
            self._weights = weights

    @property
    def fem_grid(self):
        return self._fem_grid

    @property
    def n_elements(self):
        return self._n_elements

    @property
    def local_element_start( self ):
        """ Local index of first element owned by process.
        """
        return self._local_element_start

    @property
    def local_element_end( self ):
        """ Local index of last element owned by process.
        """
        return self._local_element_end

    @property
    def points(self):
        return self._points

    @property
    def weights(self):
        return self._weights

    @property
    def quad_order(self):
        return [w.shape[1] for w in self.weights]

    @property
    def axis(self):
        return self._axis

    @property
    def ext(self):
        return self._ext

#==============================================================================
class BasisValues():
    """ Compute the basis values and the spans.

    Parameters
    ----------
    V : FemSpace
       the space that contains the basis values and the spans.

    nderiv : int
        the maximum number of derivatives needed for the basis values.

    trial : bool, optional
        the trial parameter indicates if the FemSpace represents the trial space or the test space.

    grid : QuadratureGrid, optional
        needed for the basis values on the boundary to indicate the boundary over an axis.

    ext : int, optional
        needed for the basis values on the boundary to indicate the boundary over an axis.

    Attributes
    ----------
    basis : list
        The basis values.
    spans : list
        The spans of the basis functions.

    """
    def __init__( self, V, nderiv , trial=False, grid=None, ext=None):

        self._space = V

        if isinstance(V, ProductFemSpace):
            starts = V.vector_space.starts
            V      = V.spaces
        else:
            starts = [V.vector_space.starts]
            V      = [V]

        spans = []
        basis = []
        for si,Vi in zip(starts,V):
            quad_grids  = Vi.quad_grids
            spans_i     = []
            basis_i     = []

            for sij,g,p,vij in zip(si, quad_grids, Vi.vector_space.pads, Vi.spaces):
                sp = g.spans-sij
                bs = g.basis
                if not trial and vij.periodic and vij.degree <= p:
                    bs                 = bs.copy()
                    sp                 = sp.copy()
                    bs[0:p-vij.degree] = 0.
                    sp[0:p-vij.degree] = sp[p-vij.degree]

                spans_i.append(sp)
                basis_i.append(bs)

            spans.append(spans_i)
            basis.append(basis_i)

        self._spans = spans
        self._basis = basis

        if grid and grid.axis is not None:
            axis = grid.axis
            for i,Vi in enumerate(V):
                space  = Vi.spaces[axis]
                points = grid.points[axis]
                boundary_basis = basis_ders_on_quad_grid(
                        space.knots, space.degree, points, nderiv, space.basis)

                self._basis[i][axis] = self._basis[i][axis].copy()
                self._basis[i][axis][0:1, :, 0:nderiv+1, 0:1] = boundary_basis
                if ext == 1:
                    self._spans[i][axis]    = self._spans[i][axis].copy()
                    self._spans[i][axis][0] = self._spans[i][axis][-1]

    @property
    def basis(self):
        return self._basis

    @property
    def spans(self):
        return self._spans

    @property
    def space(self):
        return self._space

#==============================================================================
# TODO have a parallel version of this function, as done for fem
def create_collocation_basis( glob_points, space, nderiv=1 ):

    T    = space.knots      # knots sequence
    p    = space.degree     # spline degree
    n    = space.nbasis     # total number of control points
    grid = space.breaks     # breakpoints
    nc   = space.ncells     # number of cells in domain (nc=len(grid)-1)

    #-------------------------------------------
    # GLOBAL GRID
    #-------------------------------------------

    # List of basis function values on each element
    nq = len(glob_points)
    glob_spans = np.zeros( nq, dtype='int' )
#    glob_basis = np.zeros( (p+1,nderiv+1,nq) ) # TODO use this for local basis fct
    glob_basis = np.zeros( (n+p,nderiv+1,nq) ) # n+p for ghosts
    for iq,xq in enumerate(glob_points):
        span = find_span( T, p, xq )
        glob_spans[iq] = span

        ders = basis_funs_all_ders( T, p, xq, span, nderiv )
        glob_basis[span:span+p+1,:,iq] = ders.transpose()

    return glob_points, glob_spans, glob_basis


#==============================================================================
# TODO experimental
class CollocationBasisValues():
    def __init__( self, points, V, nderiv ):

        assert(isinstance(V, TensorFemSpace))

        _points  = []
        _basis = []
        _spans = []

        for i,W in enumerate(V.spaces):
            ps, sp, bs = create_collocation_basis( points[i], W, nderiv=nderiv )
            _points.append(ps)
            _basis.append(bs)
            _spans.append(sp)

        self._spans = _spans
        self._basis = _basis

    @property
    def basis(self):
        return self._basis

    @property
    def spans(self):
        return self._spans
