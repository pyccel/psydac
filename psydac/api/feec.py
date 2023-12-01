from sympde.topology.mapping import Mapping

from psydac.api.basic              import BasicDiscrete
from psydac.feec.derivatives       import Derivative_1D, Gradient_2D, Gradient_3D
from psydac.feec.derivatives       import ScalarCurl_2D, VectorCurl_2D, Curl_3D
from psydac.feec.derivatives       import Divergence_2D, Divergence_3D
from psydac.feec.global_projectors import Projector_H1, Projector_Hcurl, Projector_H1vec
from psydac.feec.global_projectors import Projector_Hdiv, Projector_L2
from psydac.feec.pull_push         import pull_1d_h1, pull_1d_l2
from psydac.feec.pull_push         import pull_2d_h1, pull_2d_hcurl, pull_2d_hdiv, pull_2d_l2, pull_2d_h1vec
from psydac.feec.pull_push         import pull_3d_h1, pull_3d_hcurl, pull_3d_hdiv, pull_3d_l2, pull_3d_h1vec
from psydac.fem.basic              import FemSpace
from psydac.fem.vector             import VectorFemSpace

__all__ = ('DiscreteDerham',)

#==============================================================================
class DiscreteDerham(BasicDiscrete):
    """ A discrete de Rham sequence built over a single-patch geometry.

    Parameters
    ----------
    mapping : Mapping or None
        Symbolic mapping from the logical space to the physical space, if any.

    *spaces : list of FemSpace
        The discrete spaces of the de Rham sequence.

    Notes
    -----
    - The basic type Mapping is defined in module sympde.topology.mapping.
      A discrete mapping (spline or NURBS) may be attached to it.

    - This constructor should not be called directly, but rather from the
      `discretize_derham` function in `psydac.api.discretization`.

    - For the multipatch counterpart of this class please see
      `MultipatchDiscreteDerham` in `psydac.feec.multipatch.api`.
    """
    def __init__(self, mapping, *spaces):

        assert (mapping is None) or isinstance(mapping, Mapping)
        assert all(isinstance(space, FemSpace) for space in spaces)

        self.has_vec = isinstance(spaces[-1], VectorFemSpace)

        if self.has_vec : 
            dim          = len(spaces) - 2
            self._spaces = spaces[:-1]
            self._H1vec    = spaces[-1]

        else :
            dim           = len(spaces) - 1
            self._spaces  = spaces

        self._dim     = dim
        self._mapping = mapping
        self._callable_mapping = mapping.get_callable_mapping() if mapping else None

        if dim == 1:
            D0 = Derivative_1D(spaces[0], spaces[1])
            spaces[0].diff = spaces[0].grad = D0

        elif dim == 2:
            kind = spaces[1].symbolic_space.kind.name

            if kind == 'hcurl':

                D0 =   Gradient_2D(spaces[0], spaces[1])
                D1 = ScalarCurl_2D(spaces[1], spaces[2])

                spaces[0].diff = spaces[0].grad = D0
                spaces[1].diff = spaces[1].curl = D1

            elif kind == 'hdiv':

                D0 = VectorCurl_2D(spaces[0], spaces[1])
                D1 = Divergence_2D(spaces[1], spaces[2])

                spaces[0].diff = spaces[0].rot = D0
                spaces[1].diff = spaces[1].div = D1

        elif dim == 3:

            D0 =   Gradient_3D(spaces[0], spaces[1])
            D1 =       Curl_3D(spaces[1], spaces[2])
            D2 = Divergence_3D(spaces[2], spaces[3])

            spaces[0].diff = spaces[0].grad = D0
            spaces[1].diff = spaces[1].curl = D1
            spaces[2].diff = spaces[2].div  = D2

        else:
            raise ValueError('Dimension {} is not available'.format(dim))

    #--------------------------------------------------------------------------
    @property
    def dim(self):
        """Dimension of the physical and logical domains, which are assumed to be the same."""
        return self._dim

    @property
    def V0(self):
        """First space of the de Rham sequence : H1 space"""
        return self._spaces[0]

    @property
    def V1(self):
        """Second space of the de Rham sequence :
        - 1d : L2 space
        - 2d : either Hdiv or Hcurl space
        - 3d : Hcurl space"""
        return self._spaces[1]

    @property
    def V2(self):
        """Third space of the de Rham sequence :
        - 2d : L2 space
        - 3d : Hdiv space"""
        return self._spaces[2]

    @property
    def V3(self):
        """Fourth space of the de Rham sequence : L2 space in 3d"""
        return self._spaces[3]

    @property
    def H1vec(self):
        """Vector-valued H1 space built as the Cartesian product of N copies of V0,
        where N is the dimension of the (logical) domain."""
        assert self.has_vec
        return self._H1vec

    @property
    def spaces(self):
        """Spaces of the proper de Rham sequence (excluding Hvec)."""
        return self._spaces

    @property
    def mapping(self):
        """The mapping from the logical space to the physical space."""
        return self._mapping

    @property
    def callable_mapping(self):
        """The mapping as a callable."""
        return self._callable_mapping

    @property
    def derivatives_as_matrices(self):
        """Differential operators of the De Rham sequence as LinearOperator objects."""
        return tuple(V.diff.matrix for V in self.spaces[:-1])

    @property
    def derivatives(self):
        """Differential operators of the De Rham sequence as `DiffOperator` objects.

        Those are objects with `domain` and `codomain` properties that are `FemSpace`, 
        they act on `FemField` (they take a `FemField` of their `domain` as input and return 
        a `FemField` of their `codomain`.
        """
        return tuple(V.diff for V in self.spaces[:-1])

    #--------------------------------------------------------------------------
    def projectors(self, *, kind='global', nquads=None):
        """Projectors mapping callable functions of the physical coordinates to a 
        corresponding `FemField` object in the De Rham sequence.

        Parameters
        ----------
        kind : str
            Type of the projection : at the moment, only global is accepted and
            returns geometric commuting projectors based on interpolation/histopolation 
            for the De Rham sequence (GlobalProjector objects).

        nquads : list(int) | tuple(int)
            Number of quadrature points along each direction, to be used in Gauss
            quadrature rule for computing the (approximated) degrees of freedom.

        Returns
        -------
        P0, ..., Pn : callables
            Projectors that can be called on any callable function that maps 
            from the physical space to R (scalar case) or R^d (vector case) and
            returns a FemField belonging to the i-th space of the De Rham sequence
        """

        if not (kind == 'global'):
            raise NotImplementedError('only global projectors are available')

        if self.dim == 1:
            P0 = Projector_H1(self.V0)
            P1 = Projector_L2(self.V1, nquads)
            if self.mapping:
                P0_m = lambda f: P0(pull_1d_h1(f, self.callable_mapping))
                P1_m = lambda f: P1(pull_1d_l2(f, self.callable_mapping))
                return P0_m, P1_m
            return P0, P1

        elif self.dim == 2:
            P0 = Projector_H1(self.V0)
            P2 = Projector_L2(self.V2, nquads)

            kind = self.V1.symbolic_space.kind.name
            if kind == 'hcurl':
                P1 = Projector_Hcurl(self.V1, nquads)
            elif kind == 'hdiv':
                P1 = Projector_Hdiv(self.V1, nquads)
            else:
                raise TypeError('projector of space type {} is not available'.format(kind))

            if self.has_vec : 
                Pvec = Projector_H1vec(self.H1vec, nquads)

            if self.mapping:
                P0_m = lambda f: P0(pull_2d_h1(f, self.callable_mapping))
                P2_m = lambda f: P2(pull_2d_l2(f, self.callable_mapping))
                if kind == 'hcurl':
                    P1_m = lambda f: P1(pull_2d_hcurl(f, self.callable_mapping))
                elif kind == 'hdiv':
                    P1_m = lambda f: P1(pull_2d_hdiv(f, self.callable_mapping))
                if self.has_vec : 
                    Pvec_m = lambda f: Pvec(pull_2d_h1vec(f, self.callable_mapping))
                    return P0_m, P1_m, P2_m, Pvec_m
                else : 
                    return P0_m, P1_m, P2_m

            if self.has_vec :
                return P0, P1, P2, Pvec
            else : 
                return P0, P1, P2

        elif self.dim == 3:
            P0 = Projector_H1   (self.V0)
            P1 = Projector_Hcurl(self.V1, nquads)
            P2 = Projector_Hdiv (self.V2, nquads)
            P3 = Projector_L2   (self.V3, nquads)
            if self.has_vec : 
                Pvec = Projector_H1vec(self.H1vec)
            if self.mapping:
                P0_m = lambda f: P0(pull_3d_h1   (f, self.callable_mapping))
                P1_m = lambda f: P1(pull_3d_hcurl(f, self.callable_mapping))
                P2_m = lambda f: P2(pull_3d_hdiv (f, self.callable_mapping))
                P3_m = lambda f: P3(pull_3d_l2   (f, self.callable_mapping))
                if self.has_vec : 
                    Pvec_m = lambda f: Pvec(pull_3d_h1vec(f, self.callable_mapping))
                    return P0_m, P1_m, P2_m, P3_m, Pvec_m
                else : 
                    return P0_m, P1_m, P2_m, P3_m

            if self.has_vec :
                return P0, P1, P2, P3, Pvec
            else : 
                return P0, P1, P2, P3

