# coding: utf-8

import numpy as np

from functools import reduce

from psydac.core.bsplines          import find_span
from psydac.core.bsplines          import basis_funs_all_ders
from psydac.fem.splines            import SplineSpace
from psydac.fem.tensor             import TensorFemSpace
from psydac.fem.vector             import ProductFemSpace

def get_points_weights(spaces, axis, e):
    for s in spaces:
        if e in s.quad_grids[axis].indices:
            i = np.where(s.quad_grids[axis].indices==e)[0][0]
            return s.quad_grids[axis].points[i:i+1], s.quad_grids[axis].weights[i:i+1]
#==============================================================================
class QuadratureGrid():
    def __init__( self, V , axis=None, ext=None, trial_space=None):

        n_elements          = []
        indices             = []
        local_element_start = []
        local_element_end   = []
        points              = []
        weights             = []

        if isinstance(V, ProductFemSpace):
            V1 = V.spaces[0]
            spaces = list(V.spaces)
        else:
            V1 = V
            spaces = [V]

        if trial_space  and not isinstance(trial_space, ProductFemSpace):
            spaces.append(trial_space)
        elif isinstance(trial_space, ProductFemSpace):
            spaces = spaces + list(trial_space.spaces)

        # calculate the union of the indices in quad_grids, and make sure that all the grids match for each space.
        for i in range(len(V1.spaces)):

            indices.append(reduce(np.union1d,[s.quad_grids[i].indices for s in spaces]))
            local_element_start.append(V1.quad_grids[i].local_element_start)
            local_element_end.append  (V1.quad_grids[i].local_element_end)
            points.append(V1.quad_grids[i].points)
            weights.append(V1.quad_grids[i].weights)

            for e in np.setdiff1d(indices[-1], V1.quad_grids[i].indices):
                if e<V1.quad_grids[i].indices[0]:
                    local_element_start[-1] +=1
                    local_element_end  [-1] +=1
                    p,w = get_points_weights(spaces, i, e)
                    points[-1]  = np.concatenate((p, points[-1]))
                    weights[-1] = np.concatenate((w, weights[-1]))
                elif e>V1.quad_grids[i].indices[-1]:
                    p,w = get_points_weights(spaces, i, e)
                    points[-1]  = np.concatenate((points[-1],p))
                    weights[-1] = np.concatenate((weights[-1],w))
                else:
                    raise ValueError("Could not contsruct indices")

        self._n_elements          = [p.shape[0] for p in points]
        self._indices             = indices
        self._local_element_start = local_element_start
        self._local_element_end   = local_element_end
        self._points              = points
        self._weights             = weights
        self._axis                = axis
        self._ext                 = ext

        if axis is not None:
            assert ext is not None
            points  = self.points
            weights = self.weights

            # ...
            if V1.ldim == 1 and isinstance(V1, SplineSpace):
                bounds = {-1: V1.domain[0],
                           1: V1.domain[1]}
            elif isinstance(V1, TensorFemSpace):
                bounds = {-1: V1.spaces[axis].domain[0],
                           1: V1.spaces[axis].domain[1]}
            else:
                raise ValueError('Incompatible type(V) = {} in {} dimensions'.format(
                    type(V1), V1.ldim))

            points [axis] = np.asarray([[bounds[ext]]])
            weights[axis] = np.asarray([[1.]])
            # ...
            self._points  = points
            self._weights = weights

    @property
    def n_elements(self):
        return self._n_elements

    @property
    def indices(self):
        return self._indices

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
    def nquads(self):
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

    Attributes
    ----------
    basis : list
        The basis values.
    spans : list
        The spans of the basis functions.

    """
    def __init__( self, V, nderiv , trial=False, grid=None):

        self._space = V
        assert grid is not None
        if isinstance(V, ProductFemSpace):
            starts = V.vector_space.starts
            V      = V.spaces
        else:
            starts = [V.vector_space.starts]
            V      = [V]

        spans = []
        basis = []

        weights = grid.weights
        for si,Vi in zip(starts,V):
            quad_grids  = Vi.quad_grids
            spans_i     = []
            basis_i     = []

            for sij,g,w,p,vij in zip(si, quad_grids, weights, Vi.vector_space.pads, Vi.spaces):
                sp = g.spans-sij
                bs = g.basis[:,:,:nderiv+1,:].copy()

                if not trial:
                    bs  = bs.copy()
                    bs *= w[:, None, None, :]
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
                local_span = find_span(space.knots, space.degree, points[0, 0])
                boundary_basis = basis_funs_all_ders(space.knots, space.degree,
                                                     points[0, 0], local_span, nderiv, space.basis)

                self._basis[i][axis] = np.transpose(boundary_basis)[None, :, :, None].copy()
                index = 0 if grid.ext == -1 else -1
                self._spans[i][axis] = np.array([self._spans[i][axis][index]])

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
