from sympde.topology.mapping import Mapping

from psydac.api.basic              import BasicDiscrete
from psydac.feec.derivatives       import Derivative_1D, Gradient_2D, Gradient_3D
from psydac.feec.derivatives       import ScalarCurl_2D, VectorCurl_2D, Curl_3D
from psydac.feec.derivatives       import Divergence_2D, Divergence_3D
from psydac.feec.global_projectors import Projector_H1, Projector_Hcurl
from psydac.feec.global_projectors import Projector_Hdiv, Projector_L2
from psydac.feec.pull_push         import pull_1d_h1, pull_1d_l2
from psydac.feec.pull_push         import pull_2d_h1, pull_2d_hcurl, pull_2d_hdiv, pull_2d_l2
from psydac.feec.pull_push         import pull_3d_h1, pull_3d_hcurl, pull_3d_hdiv, pull_3d_l2

__all__ = ('DiscreteDerham',)

#==============================================================================
class DiscreteDerham(BasicDiscrete):
    """ Represent the discrete De Rham sequence.
    """
    def __init__(self, mapping, *spaces):

        assert (mapping is None) or isinstance(mapping, Mapping)

        dim           = len(spaces) - 1
        self._dim     = dim
        self._spaces  = spaces
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
        return self._dim

    @property
    def V0(self):
        return self._spaces[0]

    @property
    def V1(self):
        return self._spaces[1]

    @property
    def V2(self):
        return self._spaces[2]

    @property
    def V3(self):
        return self._spaces[3]

    @property
    def spaces(self):
        return self._spaces

    @property
    def mapping(self):
        return self._mapping

    @property
    def callable_mapping(self):
        return self._callable_mapping

    @property
    def derivatives_as_matrices(self):
        return tuple(V.diff.matrix for V in self.spaces[:-1])

    @property
    def derivatives_as_operators(self):
        return tuple(V.diff for V in self.spaces[:-1])

    #--------------------------------------------------------------------------
    def projectors(self, *, kind='global', nquads=None):

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

            if self.mapping:
                P0_m = lambda f: P0(pull_2d_h1(f, self.callable_mapping))
                P2_m = lambda f: P2(pull_2d_l2(f, self.callable_mapping))
                if kind == 'hcurl':
                    P1_m = lambda f: P1(pull_2d_hcurl(f, self.callable_mapping))
                elif kind == 'hdiv':
                    P1_m = lambda f: P1(pull_2d_hdiv(f, self.callable_mapping))
                return P0_m, P1_m, P2_m
            return P0, P1, P2

        elif self.dim == 3:
            P0 = Projector_H1   (self.V0)
            P1 = Projector_Hcurl(self.V1, nquads)
            P2 = Projector_Hdiv (self.V2, nquads)
            P3 = Projector_L2   (self.V3, nquads)
            if self.mapping:
                P0_m = lambda f: P0(pull_3d_h1   (f, self.callable_mapping))
                P1_m = lambda f: P1(pull_3d_hcurl(f, self.callable_mapping))
                P2_m = lambda f: P2(pull_3d_hdiv (f, self.callable_mapping))
                P3_m = lambda f: P3(pull_3d_l2   (f, self.callable_mapping))
                return P0_m, P1_m, P2_m, P3_m
            return P0, P1, P2, P3
