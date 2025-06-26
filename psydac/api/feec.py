import os

from sympde.topology.mapping import Mapping
from sympde.topology.space   import ScalarFunction

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

from psydac.feec.multipatch.operators import BrokenGradient_2D
from psydac.feec.multipatch.operators import BrokenScalarCurl_2D
from psydac.feec.multipatch.operators import Multipatch_Projector_H1
from psydac.feec.multipatch.operators import Multipatch_Projector_Hcurl
from psydac.feec.multipatch.operators import Multipatch_Projector_L2
from psydac.feec.multipatch.operators import ConformingProjection_V0
from psydac.feec.multipatch.operators import ConformingProjection_V1
from psydac.feec.multipatch.fem_linear_operators import IdLinearOperator

from sympde.expr.expr import LinearForm, integral
from sympde.calculus  import dot
from sympde.topology  import element_of

from psydac.api.settings import PSYDAC_BACKENDS

__all__ = ('DiscreteDerham', 'DiscreteDerhamMultipatch',)

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

        if nquads is None:
            nquads = [p + 1 for p in self.V0.degree]
        elif isinstance(nquads, int):
            nquads = [nquads] * self.dim
        else:
            assert hasattr(nquads, '__iter__')
            nquads = list(nquads)

        assert all(isinstance(nq, int) for nq in nquads)
        assert all(nq >= 1 for nq in nquads)

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
