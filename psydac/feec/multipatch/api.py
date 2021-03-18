# coding: utf-8

from sympde.topology import Derham

from psydac.api.discretization        import discretize as discretize_single_patch
from psydac.api.discretization        import discretize_space
from psydac.api.discretization        import DiscreteDerham
from psydac.feec.multipatch.operators import BrokenGradient_2D
from psydac.feec.multipatch.operators import BrokenScalarCurl_2D
from psydac.feec.multipatch.operators import Multipatch_Projector_H1
from psydac.feec.multipatch.operators import Multipatch_Projector_Hcurl
from psydac.feec.multipatch.operators import Multipatch_Projector_L2

__all__ = ('DiscreteDerhamMultipatch', 'discretize')

#==============================================================================
class DiscreteDerhamMultipatch(DiscreteDerham):

    def __init__(self, *, mapping, spaces, sequence=None):

        dim = len(spaces) - 1
        self._dim     = dim
        self._mapping = mapping
        self._spaces  = tuple(spaces)

        if sequence:
            if len(sequence) != dim + 1:
                raise ValueError('Expected len(sequence) = {}, got {} instead'.
                        format(dim + 1, len(sequence)))

        if dim == 1:
            self._sequence = ('h1', 'l2')
            raise NotImplementedError('1D FEEC multipatch non available yet')

        elif dim == 2:
            if sequence is None:
                raise ValueError('Sequence must be specified in 2D case')

            elif tuple(sequence) == ('h1', 'hcurl', 'l2'):
                self._sequence = tuple(sequence)
                self._broken_diff_ops = (
                    BrokenGradient_2D(self.V0, self.V1),
                    BrokenScalarCurl_2D(self.V1, self.V2),  # None,
                )

            elif tuple(sequence) == ('h1', 'hdiv', 'l2'):
                self._sequence = tuple(sequence)
                raise NotImplementedError('2D sequence with H-div not available yet')

            else:
                raise ValueError('2D sequence not understood')

        elif dim == 3:
            self._sequence = ('h1', 'hcurl', 'hdiv', 'l2')
            raise NotImplementedError('3D FEEC multipatch non available yet')

        else:
            raise ValueError('Dimension {} is not available'.format(dim))

    #--------------------------------------------------------------------------
    @property
    def sequence(self):
        return self._sequence

    # ...
    @property
    def broken_derivatives_as_operators(self):
        return self._broken_diff_ops

    # ...
    @property
    def broken_derivatives_as_matrices(self):
        return tuple(b_diff.matrix for b_diff in self._broken_diff_ops)

    #--------------------------------------------------------------------------
    def projectors(self, *, kind='global', nquads=None):

        if not (kind == 'global'):
            raise NotImplementedError('only global projectors are available')

        if self.dim == 1:
            pass # TODO
#            P0 = Multipatch_Projector_H1(self.V0)
#            P1 = Multipatch_Projector_L2(self.V1, nquads=nquads)
#            return P0, P1

        elif self.dim == 2:
            P0 = Multipatch_Projector_H1(self.V0)

            if self.sequence[1] == 'hcurl':
                P1 = Multipatch_Projector_Hcurl(self.V1, nquads=nquads)
            else:
                P1 = None # TODO: Multipatch_Projector_Hdiv(self.V1, nquads=nquads)
                raise NotImplementedError('2D sequence with H-div not available yet')

            P2 = Multipatch_Projector_L2(self.V2, nquads=nquads)
            return P0, P1, P2

        elif self.dim == 3:
            pass # TODO
#            P0 = Multipatch_Projector_H1   (self.V0)
#            P1 = Multipatch_Projector_Hcurl(self.V1, nquads=nquads)
#            P2 = Multipatch_Projector_Hdiv (self.V2, nquads=nquads)
#            P3 = Multipatch_Projector_L2   (self.V3, nquads=nquads)
#            return P0, P1, P2, P3

#==============================================================================
def discretize_derham_multipatch(derham, domain_h, *args, **kwargs):

    ldim     = derham.shape
    mapping  = derham.spaces[0].domain.mapping

    bases  = ['B'] + ldim * ['M']
    spaces = [discretize_space(V, domain_h, *args, basis=basis, **kwargs) \
            for V, basis in zip(derham.spaces, bases)]

    return DiscreteDerhamMultipatch(
        mapping  = mapping,
        spaces   = spaces,
        sequence = [V.kind.name for V in derham.spaces]
    )

#==============================================================================
def discretize(expr, *args, **kwargs):

    if isinstance(expr, Derham) and expr.V0.is_broken:
        return discretize_derham_multipatch(expr, *args, **kwargs)

    else:
        return discretize_single_patch(expr, *args, **kwargs)
