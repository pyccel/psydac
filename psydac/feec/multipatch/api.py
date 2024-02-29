# coding: utf-8
import os

from sympde.topology import Derham
from sympde.topology  import element_of, elements_of
from sympde.topology.space  import ScalarFunction
from sympde.calculus  import grad, dot, inner, rot, div
from sympde.calculus  import laplace, bracket, convect
from sympde.calculus  import jump, avg, Dn, minus, plus
from sympde.expr.expr import LinearForm, BilinearForm, integral

from psydac.api.settings             import PSYDAC_BACKENDS

from psydac.api.discretization        import discretize as discretize_single_patch
from psydac.api.discretization        import discretize_space
from psydac.api.discretization        import DiscreteDerham
from psydac.feec.multipatch.operators import BrokenGradient_2D
from psydac.feec.multipatch.operators import BrokenScalarCurl_2D
from psydac.feec.multipatch.operators import Multipatch_Projector_H1
from psydac.feec.multipatch.operators import Multipatch_Projector_Hcurl
from psydac.feec.multipatch.operators import Multipatch_Projector_L2
from psydac.feec.multipatch.operators import ConformingProjection_V0
from psydac.feec.multipatch.operators import ConformingProjection_V1
from psydac.feec.multipatch.fem_linear_operators import IdLinearOperator


__all__ = ('DiscreteDerhamMultipatch', 'discretize', 'discretize_derham_multipatch')

#==============================================================================
class DiscreteDerhamMultipatch(DiscreteDerham):
    """ Represents the discrete De Rham sequence for multipatch domains.
        It only works when the number of patches>1

    Parameters
    ----------
    mapping: <Mapping>
     The mapping of the multipatch domain, the multipatch mapping contains the mapping of each patch 

    domain_h: <Geometry>
     The discrete domain

    spaces: <list,tuple>
      The discrete spaces that are contained in the De Rham sequence

    sequence: <list,tuple>
      The space kind of each space in the De Rham sequence
    """
    
    def __init__(self, *, mapping, domain_h, spaces, sequence=None):


        dim = len(spaces) - 1
        self._dim     = dim
        self._mapping = mapping
        self._spaces  = tuple(spaces)
        self._domain_h = domain_h

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
        """
        This method returns the patch-wise commuting projectors on the broken multi-patch space

        Parameters
        ----------
        kind: <str>
          The projectors kind, can be global or local

        nquads: <list,tuple>
          The number of quadrature points.

        Returns
        -------
        P0: <Multipatch_Projector_H1>
         Patch wise H1 projector

        P1: <Multipatch_Projector_Hcurl>
         Patch wise Hcurl projector

        P2: <Multipatch_Projector_L2>
         Patch wise L2 projector

        Notes
        -----
            - when applied to smooth functions they return conforming fields
            - default 'global projectors' correspond to geometric interpolation/histopolation operators on Greville grids
            - here 'global' is a patch-level notion, as the interpolation-type problems are solved on each patch independently
        """
        if not (kind == 'global'):
            raise NotImplementedError('only global projectors are available')

        if self.dim == 1:
            raise NotImplementedError("1D projectors are not available")

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
            raise NotImplementedError("3D projectors are not available")

        #--------------------------------------------------------------------------
    def conforming_projection(self, space, hom_bc=False, backend_language="python", load_dir=None):
        """
        return the conforming projectors of the broken multi-patch space

        Parameters
        ----------
        space : <str>
          The space of the projector

        hom_bc: <bool>
          Apply homogenous boundary conditions if True

        backend_language: <str>
          The backend used to accelerate the code

        load_dir: <str|None>
          Filename for storage in sparse matrix format

        Returns
        -------
        Cp: <FemLinearOperator>
          The conforming projector

        """
        if hom_bc is None:
            raise ValueError('please provide a value for "hom_bc" argument')

        if isinstance(load_dir, str):
            if not os.path.exists(load_dir):
                os.makedirs(load_dir)
            if space == 'V0':
                P_name = 'cP0'
            elif space == 'V1':
                P_name = 'cP1'
            elif space == 'V2':
                P_name = 'cP2'
            else:
                raise ValueError(space)

            if hom_bc:
                storage_fn = load_dir + '/{}_hom_m.npz'.format(P_name)
            else:
                storage_fn = load_dir + '/{}_m.npz'.format(P_name)
        else:
            storage_fn = None

        cP = None
        if self.dim == 1:
            raise NotImplementedError("1D projectors are not available")

        elif self.dim == 2:
            if space == 'V0':
                cP = ConformingProjection_V0(self.V0, self._domain_h, hom_bc=hom_bc, backend_language=backend_language, storage_fn=storage_fn)
            elif space == 'V1':
                if self.sequence[1] == 'hcurl':
                    cP = ConformingProjection_V1(self.V1, self._domain_h, hom_bc=hom_bc, backend_language=backend_language, storage_fn=storage_fn)
                else:
                    raise NotImplementedError('2D sequence with H-div not available yet')

            elif space == 'V2':
                cP = IdLinearOperator(self.V2)  # no storage needed!
            else:
                raise ValueError('Invalid value for "space" argument: {}'.format(space))

        elif self.dim == 3:
            raise NotImplementedError("3D projectors are not available")

        return cP

    def get_dual_dofs(self, space, f, backend_language="python", return_format='stencil_array'):
        """
        return the dual dofs tilde_sigma_i(f) = < Lambda_i, f >_{L2} i = 1, .. dim(V^k)) of a given function f, as a stencil array or numpy array

        Parameters
        ----------
        space : <str>
          The space of the dual dofs

        f : <sympy.Expr>
         The function used for evaluation

        backend_language: <str>
          The backend used to accelerate the code

        return_format: <str>
         The format of the dofs, can be 'stencil_array' or 'numpy_array'

        Returns
        -------
        tilde_f:<Vector|ndarray>
         The dual dofs
        """
        if space == 'V0':
            Vh = self.V0
        elif space == 'V1':
            Vh = self.V1
        elif space == 'V2':
            Vh = self.V2
        else:
            raise NotImplementedError("The space of kind {} is not available".format(space))

        V  = Vh.symbolic_space
        v  = element_of(V, name='v')

        if isinstance(v, ScalarFunction):
            expr   = f*v
        else:
            expr   = dot(f,v)

        l        = LinearForm(v, integral( V.domain, expr))
        lh       = discretize(l, self._domain_h, Vh, backend=PSYDAC_BACKENDS[backend_language])
        tilde_f  = lh.assemble()

        if return_format == 'numpy_array':
            return tilde_f.toarray()
        else:
            return tilde_f

#==============================================================================
def discretize_derham_multipatch(derham, domain_h, *args, **kwargs):

    ldim     = derham.shape
    mapping  = derham.spaces[0].domain.mapping

    bases  = ['B'] + ldim * ['M']
    spaces = [discretize_space(V, domain_h, *args, basis=basis, **kwargs) \
            for V, basis in zip(derham.spaces, bases)]

    return DiscreteDerhamMultipatch(
        mapping  = mapping,
        domain_h = domain_h,
        spaces   = spaces,
        sequence = [V.kind.name for V in derham.spaces]
    )

#==============================================================================
def discretize(expr, *args, **kwargs):

    if isinstance(expr, Derham) and expr.V0.is_broken:
        return discretize_derham_multipatch(expr, *args, **kwargs)

    else:
        return discretize_single_patch(expr, *args, **kwargs)
