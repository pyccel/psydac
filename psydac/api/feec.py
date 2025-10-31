import numpy as np

from scipy.sparse                               import dia_matrix

from sympde.expr                                import integral, BilinearForm
from sympde.topology                            import elements_of, Line, Derham

from psydac.api.basic                           import BasicDiscrete

from psydac.feec.derivatives                    import Derivative1D, Gradient2D, Gradient3D
from psydac.feec.derivatives                    import ScalarCurl2D, VectorCurl2D, Curl3D
from psydac.feec.derivatives                    import Divergence2D, Divergence3D
from psydac.feec.derivatives                    import BrokenGradient2D
from psydac.feec.derivatives                    import BrokenScalarCurl2D

from psydac.feec.global_geometric_projectors    import GlobalGeometricProjectorH1
from psydac.feec.global_geometric_projectors    import GlobalGeometricProjectorHcurl
from psydac.feec.global_geometric_projectors    import GlobalGeometricProjectorH1vec
from psydac.feec.global_geometric_projectors    import GlobalGeometricProjectorHdiv
from psydac.feec.global_geometric_projectors    import GlobalGeometricProjectorL2
from psydac.feec.global_geometric_projectors    import MultipatchGeometricProjector

from psydac.feec.conforming_projectors          import ConformingProjectionV0
from psydac.feec.conforming_projectors          import ConformingProjectionV1

from psydac.feec.hodge                          import HodgeOperator

from psydac.feec.pull_push                      import pull_1d_h1, pull_1d_l2
from psydac.feec.pull_push                      import pull_2d_h1, pull_2d_hcurl
from psydac.feec.pull_push                      import pull_2d_hdiv, pull_2d_l2, pull_2d_h1vec
from psydac.feec.pull_push                      import pull_3d_h1, pull_3d_hcurl
from psydac.feec.pull_push                      import pull_3d_hdiv, pull_3d_l2, pull_3d_h1vec

from psydac.fem.basic                           import FemSpace, FemLinearOperator
from psydac.fem.vector                          import VectorFemSpace

from psydac.linalg.basic                        import LinearOperator, IdentityOperator
from psydac.linalg.block                        import BlockLinearOperator
from psydac.linalg.direct_solvers               import BandedSolver
from psydac.linalg.kron                         import KroneckerLinearSolver, KroneckerStencilMatrix
from psydac.linalg.stencil                      import StencilVectorSpace

__all__ = ('DiscreteDeRham', 'DiscreteDeRhamMultipatch',)

#==============================================================================
class DiscreteDeRham(BasicDiscrete):
    """ A discrete de Rham sequence built over a single-patch geometry.

    Parameters
    ----------
    domain_h : Geometry
        The discretized domain, which is a single-patch geometry.

    *spaces : list of FemSpace
        The discrete spaces of the de Rham sequence.

    Notes
    -----
    - This constructor should not be called directly, but rather from the
      `discretize_derham` function in `psydac.api.discretization`.
    """
    def __init__(self, domain_h, *spaces):

        assert all(isinstance(space, FemSpace) for space in spaces)

        self.has_vec = isinstance(spaces[-1], VectorFemSpace)

        if self.has_vec : 
            dim          = len(spaces) - 2
            self._spaces = spaces[:-1]
            self._H1vec    = spaces[-1]

        else :
            dim           = len(spaces) - 1
            self._spaces  = spaces

        self._domain_h = domain_h
        self._sequence = tuple(space.symbolic_space.kind.name for space in spaces)
        self._dim     = dim
        self._mapping = domain_h.domain.mapping
        self._callable_mapping = self._mapping.get_callable_mapping() if self._mapping else None

        if dim == 1:
            D0 = Derivative1D(spaces[0], spaces[1])

            spaces[0].diff = spaces[0].grad = D0

            self._derivatives = (D0,)

        elif dim == 2:
            kind = spaces[1].symbolic_space.kind.name

            if kind == 'hcurl':

                D0 =   Gradient2D(spaces[0], spaces[1])
                D1 = ScalarCurl2D(spaces[1], spaces[2])

                spaces[0].diff = spaces[0].grad = D0
                spaces[1].diff = spaces[1].curl = D1

                self._derivatives = (D0, D1)

            elif kind == 'hdiv':

                D0 = VectorCurl2D(spaces[0], spaces[1])
                D1 = Divergence2D(spaces[1], spaces[2])

                spaces[0].diff = spaces[0].rot = D0
                spaces[1].diff = spaces[1].div = D1

                self._derivatives = (D0, D1)


        elif dim == 3:

            D0 =   Gradient3D(spaces[0], spaces[1])
            D1 =       Curl3D(spaces[1], spaces[2])
            D2 = Divergence3D(spaces[2], spaces[3])

            spaces[0].diff = spaces[0].grad = D0
            spaces[1].diff = spaces[1].curl = D1
            spaces[2].diff = spaces[2].div  = D2

            self._derivatives = (D0, D1, D2)

        else:
            raise ValueError('Dimension {} is not available'.format(dim))

        self._hodge_operators = ()
        self._conf_proj = ()
    #--------------------------------------------------------------------------
    @property
    def dim(self):
        """Dimension of the physical and logical domains, which are assumed to be the same."""
        return self._dim
    
    @property
    def domain_h(self):
        """Discretized domain."""
        return self._domain_h

    @property
    def spaces(self):
        """Spaces of the proper de Rham sequence (excluding Hvec)."""
        return self._spaces

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
    def sequence(self):
        return self._sequence

    @property
    def H1vec(self):
        """Vector-valued H1 space built as the Cartesian product of N copies of V0,
        where N is the dimension of the (logical) domain."""
        assert self.has_vec
        return self._H1vec

    @property
    def mapping(self):
        """The mapping from the logical space to the physical space."""
        return self._mapping

    @property
    def callable_mapping(self):
        """The mapping as a callable."""
        return self._callable_mapping

    #--------------------------------------------------------------------------
    def projectors(self, *, kind='global', nquads=None):
        """Projectors mapping callable functions of the physical coordinates to a 
        corresponding `FemField` object in the de Rham sequence.

        Parameters
        ----------
        kind : str
            Type of the projection : at the moment, only global is accepted and
            returns geometric commuting projectors based on interpolation/histopolation 
            for the de Rham sequence (GlobalProjector objects).

        nquads : list(int) | tuple(int)
            Number of quadrature points along each direction, to be used in Gauss
            quadrature rule for computing the (approximated) degrees of freedom.

        Returns
        -------
        P0, ..., Pn : callables
            Projectors that can be called on any callable function that maps 
            from the physical space to R (scalar case) or R^d (vector case) and
            returns a FemField belonging to the i-th space of the de Rham sequence
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
            P0 = GlobalGeometricProjectorH1(self.V0)
            P1 = GlobalGeometricProjectorL2(self.V1, nquads)
            if self.mapping:
                P0_m = lambda f: P0(pull_1d_h1(f, self.callable_mapping))
                P1_m = lambda f: P1(pull_1d_l2(f, self.callable_mapping))
                return P0_m, P1_m
            return P0, P1

        elif self.dim == 2:
            P0 = GlobalGeometricProjectorH1(self.V0)
            P2 = GlobalGeometricProjectorL2(self.V2, nquads)

            kind = self.V1.symbolic_space.kind.name
            if kind == 'hcurl':
                P1 = GlobalGeometricProjectorHcurl(self.V1, nquads)
            elif kind == 'hdiv':
                P1 = GlobalGeometricProjectorHdiv(self.V1, nquads)
            else:
                raise TypeError('projector of space type {} is not available'.format(kind))

            if self.has_vec : 
                Pvec = GlobalGeometricProjectorH1vec(self.H1vec, nquads)

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
            P0 = GlobalGeometricProjectorH1   (self.V0)
            P1 = GlobalGeometricProjectorHcurl(self.V1, nquads)
            P2 = GlobalGeometricProjectorHdiv (self.V2, nquads)
            P3 = GlobalGeometricProjectorL2   (self.V3, nquads)
            if self.has_vec : 
                Pvec = GlobalGeometricProjectorH1vec(self.H1vec)
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

    #--------------------------------------------------------------------------
    def derivatives(self, kind='femlinop'):
        if kind == 'femlinop':
            return self._derivatives
        elif kind == 'linop': 
            return tuple(b_diff.linop for b_diff in self._derivatives)
    
    #--------------------------------------------------------------------------
    def dirichlet_projectors(self, kind='femlinop'):
        """
        Returns operators that apply the correct Dirichlet boundary conditions.

        Parameters
        ----------
        kind : str
            The kind of the projector, can be 'femlinop' or 'linop'.
            - 'femlinop' returns a psydac FemLinearOperator (default)
            - 'linop' returns a psydac LinearOperator

        Returns
        -------
        d_projectors : list
            List of <psydac.fem.basic.FemLinearOperator> or <psydac.linalg.basic.LinearOperator>
            The Dirichlet boundary projectors of each space and in desired form.

        Notes
        -----
        See examples/vector_potential_3d.py for a use case of these operators in LinearOperator form.
        
        """
        assert kind in ('femlinop', 'linop')

        from psydac.linalg.tests.test_solvers import DirichletBoundaryProjector
        d_projectors = [DirichletBoundaryProjector(Vh) for Vh in self.spaces[:-1]]

        if kind == 'femlinop':
            d_projectors = [FemLinearOperator(fem_domain=Vh, fem_codomain=Vh, linop=d_projector) for Vh, d_projector in zip(self.spaces[:-1], d_projectors)]

        return d_projectors

    #--------------------------------------------------------------------------
    def LST_preconditioner(self, M0=None, M1=None, M2=None, M3=None, hom_bc=False):
        """
        LST (Loli, Sangalli, Tani) preconditioners are mass matrix preconditioners of the form
        pc = D_inv_sqrt @ D_log_sqrt @ M_log_kron_solver @ D_log_sqrt @ D_inv_sqrt, where

        D_inv_sqrt          is the diagonal matrix of the square roots of the inverse diagonal entries of the mass matrix M,
        D_log_sqrt          is the diagonal matrix of the square roots of the diagonal entries of the mass matrix on the logical domain,
        M_log_kron_solver   is the Kronecker Solver of the mass matrix on the logical domain.

        These preconditioners work very well even on complex domains as numerical experiments have shown.

        Upon choosing hom_bc=True, preconditioner for the modified mass matrices M{i}_0 are being returned.
        The preconditioner for the last mass matrix of the sequence remains identical as there are no BCs to take care of.
        M{i}_0 is a mass matrix of the form
        M{i}_0 = DBP @ M{i} @ DBP + (I - DBP)
        where DBP and I are the corresponding DirichletBoundaryProjector and IdentityOperator.
        See examples/vector_potential_3d.

        Parameters
        ----------
        M0, M1, M2, M3 : psydac.linalg.stencil.StencilMatrix | psydac.linalg.block.BlockLinearOperator | None
            H1, Hcurl/Hdiv, L2 (2D) or H1, Hcurl, Hdiv, L2 mass matrices or None. 
            Returns only preconditioners for passed mass matrices.

        hom_bc : bool
            If True, return LST preconditioner for modified M{i}_0 = DBP @ M{i} @ DBP + (I - DBP) mass matrix (i=0,1 (2D), i=0,1,2 (3D)).
            The arguments M{i} in that case remain the same (M{i}, not M{i}_0). DBP and I are DirichletBoundaryProjector and IdentityOperator.
            Default False

        Returns
        -------
        psydac.linalg.stencil.StencilMatrix | psydac.linalg.block.BlockLinearOperator | list
            LST preconditioner(s) for passed M{i}s (hom_bc=False) or M{i}_0s (hom_b=True).
        
        """
        # To avoid circular imports
        from psydac.api.discretization                   import discretize
        from psydac.linalg.tests.test_kron_direct_solver import matrix_to_bandsolver

        dim = self.dim
        # dim=1 makes hardly any sense (because of the Kronecker solver that is no more Kronecker solver in 1D)
        assert dim in (2, 3)

        if hom_bc == True:
            # We require a numpy array represenation of the modified 1D mass matrices
            def toarray_1d(A):
                """
                Obtain a numpy array representation of a (1D) LinearOperator (which has not implemented toarray()).
                
                We fill an empty numpy array row by row by repeatedly applying unit vectors
                to the transpose of A. In order to obtain those unit vectors in Stencil format,
                we make use of an auxiliary function that takes periodicity into account.
                """
                
                assert isinstance(A, LinearOperator)
                W = A.codomain
                assert isinstance(W, StencilVectorSpace)

                def get_unit_vector_1d(v, periodic, n1, npts1, pads1):

                    v *= 0.0
                    v._data[pads1+n1] = 1.

                    if periodic:
                        if n1 < pads1:
                            v._data[-pads1+n1] = 1.
                        if n1 >= npts1-pads1:
                            v._data[n1-npts1+pads1] = 1.
                    
                    return v

                periods  = W.periods
                periodic = periods[0]

                w = W.zeros()
                At = A.T

                A_arr = np.zeros(A.shape, dtype=A.dtype)

                npts1,  = W.npts
                pads1,  = W.pads
                for n1 in range(npts1):
                    e_n1 = get_unit_vector_1d(w, periodic, n1, npts1, pads1)
                    A_n1 = At @ e_n1
                    A_arr[n1, :] = A_n1.toarray()

                return A_arr

            def M0_0_1d_to_bandsolver(M0_0_1d):
                """
                Converts the M0_0_1d StencilMatrix to a BandedSolver.

                Closely resembles a combination of the two functions
                matrix_to_bandsolver & to_bnd
                found in test_kron_direct_solver,
                the difference being that M0_0_1d neither has a 
                remove_spurious_entries() nor a toarray() function.
                
                """

                dmat = dia_matrix(toarray_1d(M0_0_1d), dtype=M0_0_1d.dtype)
                la   = abs(dmat.offsets.min())
                ua   = dmat.offsets.max()
                cmat = dmat.tocsr()

                M0_0_1d_bnd = np.zeros((1+ua+2*la, cmat.shape[1]), M0_0_1d.dtype)

                for i,j in zip(*cmat.nonzero()):
                    M0_0_1d_bnd[la+ua+i-j, j] = cmat[i,j]

                return BandedSolver(ua, la, M0_0_1d_bnd)

        domain_h    = self.domain_h
        domain      = domain_h.domain

        ncells,     = domain_h.ncells.values()
        degree      = self.V0.degree
        periodic,   = domain_h.periodic.values()
        
        logical_domain = domain.logical_domain

        Ms = [M0, M1, M2] if dim == 2 else [M0, M1, M2, M3]

        # ----- Gather D_inv_sqrt

        D_inv_sqrt_arr = []

        for M in Ms:
            if M is not None:
                D_inv_sqrt_arr.append(M.diagonal(inverse=True, sqrt=True))
            else:
                D_inv_sqrt_arr.append(None)

        # ----- Gather M_log_kron_solver

        M_log_kron_solver_arr = []

        logical_domain_1d_x = Line('L', bounds=logical_domain.bounds1)
        logical_domain_1d_y = Line('L', bounds=logical_domain.bounds2)
        if dim == 3:
            logical_domain_1d_z = Line('L', bounds=logical_domain.bounds3)

        logical_domain_1d_list = [logical_domain_1d_x, logical_domain_1d_y]
        if dim == 3:
            logical_domain_1d_list += [logical_domain_1d_z]

        M0_1d_solvers = []
        M1_1d_solvers = []
        # We gather all (2x3=6) 1D mass matrices.
        # Those will be used to obtain D_log_sqrt using the new
        # diagonal function for KroneckerStencilMatrices.
        M0s_1d = []
        M1s_1d = []

        for ncells_1d, degree_1d, periodic_1d, logical_domain_1d in zip(ncells, degree, periodic, logical_domain_1d_list):

            derham_1d = Derham(logical_domain_1d)

            logical_domain_1d_h = discretize(logical_domain_1d, ncells=[ncells_1d, ], periodic=[periodic_1d, ])
            derham_1d_h = discretize(derham_1d, logical_domain_1d_h, degree=[degree_1d, ])

            V0_1d,  V1_1d  = derham_1d.spaces
            V0h_1d, V1h_1d = derham_1d_h.spaces

            u0, v0 = elements_of(V0_1d, names='u0, v0')
            u1, v1 = elements_of(V1_1d, names='u1, v1')

            a0_1d = BilinearForm((u0, v0), integral(logical_domain_1d, u0*v0))
            a1_1d = BilinearForm((u1, v1), integral(logical_domain_1d, u1*v1))

            a0h_1d = discretize(a0_1d, logical_domain_1d_h, (V0h_1d, V0h_1d))
            a1h_1d = discretize(a1_1d, logical_domain_1d_h, (V1h_1d, V1h_1d))

            M0_1d = a0h_1d.assemble()
            M1_1d = a1h_1d.assemble()

            M0s_1d.append(M0_1d)
            M1s_1d.append(M1_1d)

            # In order to obtain a good preconditioner for modified mass matrices 
            # M{i}_0 = DBP @ M{i} @ DBP + (I - DBP) (see docstring)
            # the Kronecker solver of M_log must be modified as well
            if hom_bc == True:
                DBP0,   = derham_1d_h.dirichlet_projectors(kind='linop')
                
                if DBP0 is not None:
                    I0      = IdentityOperator(V0h_1d.coeff_space)
                    M0_0_1d = DBP0 @ M0_1d @ DBP0 + (I0 - DBP0)

                    M0_0_1d_solver = M0_0_1d_to_bandsolver(M0_0_1d)
                    M0_1d_solvers.append(M0_0_1d_solver)
                else:
                    M0_1d_solver = matrix_to_bandsolver(M0_1d)
                    M0_1d_solvers.append(M0_1d_solver)
            else:
                M0_1d_solver = matrix_to_bandsolver(M0_1d)
                M0_1d_solvers.append(M0_1d_solver)

            M1_1d_solver = matrix_to_bandsolver(M1_1d)
            M1_1d_solvers.append(M1_1d_solver)

        if dim == 2:
            V0_cs, V1_cs, V2_cs = [Vh.coeff_space for Vh in self.spaces]

            if M0 is not None:
                M0_log_kron_solver = KroneckerLinearSolver(V0_cs, V0_cs, (M0_1d_solvers[0], M0_1d_solvers[1]))
                M_log_kron_solver_arr.append(M0_log_kron_solver)
            else:
                M_log_kron_solver_arr.append(None)

            if M1 is not None:
                if self.sequence[1] == 'hcurl':
                    M1_0_log_kron_solver = KroneckerLinearSolver(V1_cs[0], V1_cs[0], (M1_1d_solvers[0], M0_1d_solvers[1]))
                    M1_1_log_kron_solver = KroneckerLinearSolver(V1_cs[1], V1_cs[1], (M0_1d_solvers[0], M1_1d_solvers[1]))
                    M1_log_kron_solver = BlockLinearOperator(V1_cs, V1_cs, [[M1_0_log_kron_solver, None],
                                                                            [None, M1_1_log_kron_solver]])
                elif self.sequence[1] == 'hdiv':
                    M1_0_log_kron_solver = KroneckerLinearSolver(V1_cs[0], V1_cs[0], (M0_1d_solvers[0], M1_1d_solvers[1]))
                    M1_1_log_kron_solver = KroneckerLinearSolver(V1_cs[1], V1_cs[1], (M1_1d_solvers[0], M0_1d_solvers[1]))
                    M1_log_kron_solver = BlockLinearOperator(V1_cs, V1_cs, [[M1_0_log_kron_solver, None],
                                                                            [None, M1_1_log_kron_solver]])
                else:
                    raise ValueError(f'The second space in the sequence {self.sequence} must be either "hcurl" or "hdiv".')
                M_log_kron_solver_arr.append(M1_log_kron_solver)
            else:
                M_log_kron_solver_arr.append(None)

            if M2 is not None:
                M2_log_kron_solver = KroneckerLinearSolver(V2_cs, V2_cs, (M1_1d_solvers[0], M1_1d_solvers[1]))
                M_log_kron_solver_arr.append(M2_log_kron_solver)
            else:
                M_log_kron_solver_arr.append(None)
        else:
            V0_cs, V1_cs, V2_cs, V3_cs = [Vh.coeff_space for Vh in self.spaces]

            if M0 is not None:
                M0_log_kron_solver = KroneckerLinearSolver(V0_cs, V0_cs, (M0_1d_solvers[0], M0_1d_solvers[1], M0_1d_solvers[2]))
                M_log_kron_solver_arr.append(M0_log_kron_solver)
            else:
                M_log_kron_solver_arr.append(None)

            if M1 is not None:
                M1_0_log_kron_solver = KroneckerLinearSolver(V1_cs[0], V1_cs[0], (M1_1d_solvers[0], M0_1d_solvers[1], M0_1d_solvers[2]))
                M1_1_log_kron_solver = KroneckerLinearSolver(V1_cs[1], V1_cs[1], (M0_1d_solvers[0], M1_1d_solvers[1], M0_1d_solvers[2]))
                M1_2_log_kron_solver = KroneckerLinearSolver(V1_cs[2], V1_cs[2], (M0_1d_solvers[0], M0_1d_solvers[1], M1_1d_solvers[2]))
                M1_log_kron_solver = BlockLinearOperator(V1_cs, V1_cs, [[M1_0_log_kron_solver, None, None],
                                                                        [None, M1_1_log_kron_solver, None],
                                                                        [None, None, M1_2_log_kron_solver]])
                M_log_kron_solver_arr.append(M1_log_kron_solver)
            else:
                M_log_kron_solver_arr.append(None)
            
            if M2 is not None:
                M2_0_log_kron_solver = KroneckerLinearSolver(V2_cs[0], V2_cs[0], (M0_1d_solvers[0], M1_1d_solvers[1], M1_1d_solvers[2]))
                M2_1_log_kron_solver = KroneckerLinearSolver(V2_cs[1], V2_cs[1], (M1_1d_solvers[0], M0_1d_solvers[1], M1_1d_solvers[2]))
                M2_2_log_kron_solver = KroneckerLinearSolver(V2_cs[2], V2_cs[2], (M1_1d_solvers[0], M1_1d_solvers[1], M0_1d_solvers[2]))
                M2_log_kron_solver = BlockLinearOperator(V2_cs, V2_cs, [[M2_0_log_kron_solver, None, None],
                                                                        [None, M2_1_log_kron_solver, None],
                                                                        [None, None, M2_2_log_kron_solver]])
                M_log_kron_solver_arr.append(M2_log_kron_solver)
            else:
                M_log_kron_solver_arr.append(None)

            if M3 is not None:
                M3_log_kron_solver = KroneckerLinearSolver(V3_cs, V3_cs, (M1_1d_solvers[0], M1_1d_solvers[1], M1_1d_solvers[2]))
                M_log_kron_solver_arr.append(M3_log_kron_solver)
            else:
                M_log_kron_solver_arr.append(None)

        # ----- Gather D_log_sqrt

        D_log_sqrt_arr = []

        M0_log = KroneckerStencilMatrix(V0_cs, V0_cs, *M0s_1d)
        if dim == 2:
            if self.sequence[1] == 'hcurl':
                M1_0_log = KroneckerStencilMatrix(V1_cs[0], V1_cs[0], M1s_1d[0], M0s_1d[1])
                M1_1_log = KroneckerStencilMatrix(V1_cs[1], V1_cs[1], M0s_1d[0], M1s_1d[1])
            else:
                M1_0_log = KroneckerStencilMatrix(V1_cs[0], V1_cs[0], M0s_1d[0], M1s_1d[1])
                M1_1_log = KroneckerStencilMatrix(V1_cs[1], V1_cs[1], M1s_1d[0], M0s_1d[1])
            M1_log = BlockLinearOperator(V1_cs, V1_cs, [[M1_0_log, None],
                                                        [None, M1_1_log]])
        else:
            M1_0_log = KroneckerStencilMatrix(V1_cs[0], V1_cs[0], M1s_1d[0], M0s_1d[1], M0s_1d[2])
            M1_1_log = KroneckerStencilMatrix(V1_cs[1], V1_cs[1], M0s_1d[0], M1s_1d[1], M0s_1d[2])
            M1_2_log = KroneckerStencilMatrix(V1_cs[2], V1_cs[2], M0s_1d[0], M0s_1d[1], M1s_1d[2])
            M1_log = BlockLinearOperator(V1_cs, V1_cs, [[M1_0_log, None, None],
                                                        [None, M1_1_log, None],
                                                        [None, None, M1_2_log]])
        if dim == 2:
            M2_log = KroneckerStencilMatrix(V2_cs, V2_cs, *M1s_1d)
        else:
            M2_0_log = KroneckerStencilMatrix(V2_cs[0], V2_cs[0], M0s_1d[0], M1s_1d[1], M1s_1d[2])
            M2_1_log = KroneckerStencilMatrix(V2_cs[1], V2_cs[1], M1s_1d[0], M0s_1d[1], M1s_1d[2])
            M2_2_log = KroneckerStencilMatrix(V2_cs[2], V2_cs[2], M1s_1d[0], M1s_1d[1], M0s_1d[2])
            M2_log = BlockLinearOperator(V2_cs, V2_cs, [[M2_0_log, None, None],
                                                        [None, M2_1_log, None],
                                                        [None, None, M2_2_log]])
        if dim == 3:
            M3_log = KroneckerStencilMatrix(V3_cs, V3_cs, *M1s_1d)

        Ms_log = [M0_log, M1_log, M2_log]
        if dim == 3:
            Ms_log += [M3_log]

        for M, M_log in zip(Ms, Ms_log):
            if M is not None:
                D_log_sqrt_arr.append(M_log.diagonal(inverse=False, sqrt=True))
            else:
                D_log_sqrt_arr.append(None)

        # --------------------------------

        M_pc_arr = []

        for M, D_inv_sqrt, D_log_sqrt, M_log_kron_solver in zip(Ms, D_inv_sqrt_arr, D_log_sqrt_arr, M_log_kron_solver_arr):
            if M is not None:
                M_pc = D_inv_sqrt @ D_log_sqrt @ M_log_kron_solver @ D_log_sqrt @ D_inv_sqrt
                M_pc_arr.append(M_pc)

        return M_pc_arr

    #--------------------------------------------------------------------------
    def conforming_projectors(self, kind='femlinop', mom_pres=False, p_moments=-1, hom_bc=False):
        """
        return the conforming projectors of the broken multi-patch space

        Parameters
        ----------

        p_moments : <int>
            The number of moments preserved by the projector.

        hom_bc: <bool>
          Apply homogenous boundary conditions if True

        kind : <str>
            The kind of the projector, can be 'femlinop' or 'linop'.
            - 'femlinop' returns a psydac FemLinearOperator (default)
            - 'linop' returns a psydac LinearOperator

        Returns
        -------
        cP0, cP1, cP2 : Tuple of <psydac.fem.basic.FemLinearOperator> or <psydac.linalg.basic.LinearOperator>
          The conforming projectors of each space and in desired form.

        """

        if hom_bc is None:
            raise ValueError('please provide a value for "hom_bc" argument')

        if self.dim == 1:
            raise NotImplementedError("1D projectors are not available")

        elif self.dim == 2:
            if self.sequence[1] != 'hcurl':
                raise NotImplementedError('2D sequence with H-div not available yet')

            else:

                if not self._conf_proj: 

                    cP0 = ConformingProjectionV0(self.V0, mom_pres=mom_pres, p_moments=p_moments, hom_bc=hom_bc)
                    cP1 = ConformingProjectionV1(self.V1, mom_pres=mom_pres, p_moments=p_moments, hom_bc=hom_bc)

                    I2 = IdentityOperator(self.V2.coeff_space)
                    cP2 = FemLinearOperator(fem_domain=self.V2, fem_codomain=self.V2, linop=I2)

                    self._conf_proj = (cP0, cP1, cP2)
                
                if kind == 'femlinop':
                    return self._conf_proj[0], self._conf_proj[1], self._conf_proj[2]
                elif kind == 'linop': 
                    return self._conf_proj[0].linop, self._conf_proj[1].linop, self._conf_proj[2].linop

        elif self.dim == 3:
            raise NotImplementedError("3D projectors are not available")

    #--------------------------------------------------------------------------
    def _init_hodge_operators(self, backend_language='python'):
        """
        Initialize the Hodge operator for the multipatch de Rham sequence.

        Parameters
        ----------

        backend_language: <str>
          The backend used to accelerate the code

        """
        if not self._hodge_operators: 

            if self.dim == 1:
                    H0 = HodgeOperator(self.V0, self.domain_h, backend_language=backend_language)
                    H1 = HodgeOperator(self.V1, self.domain_h, backend_language=backend_language)
                    
                    self._hodge_operators = (H0, H1)

            elif self.dim == 2:

                    H0 = HodgeOperator(self.V0, self.domain_h, backend_language=backend_language)
                    H1 = HodgeOperator(self.V1, self.domain_h, backend_language=backend_language)
                    H2 = HodgeOperator(self.V2, self.domain_h, backend_language=backend_language)

                    self._hodge_operators = (H0, H1, H2)

            elif self.dim == 3:

                    H0 = HodgeOperator(self.V0, self.domain_h, backend_language=backend_language)
                    H1 = HodgeOperator(self.V1, self.domain_h, backend_language=backend_language)
                    H2 = HodgeOperator(self.V2, self.domain_h, backend_language=backend_language)
                    H3 = HodgeOperator(self.V3, self.domain_h, backend_language=backend_language)

                    self._hodge_operators = (H0, H1, H2, H3)

    #--------------------------------------------------------------------------
    def _get_hodge_operator(self, H, dual=False, kind='femlinop'):
        """
        Helper function to return the Hodge operator in the specified form.
        
        Parameters
        ----------
            H : <HodgeOperator>

            dual : <bool>
                If True, returns the dual Hodge operator

            kind : <str>
                The kind of the projector, can be 'femlinop' or 'linop'.
                - 'femlinop' returns a psydac FemLinearOperator (default)
                - 'linop' returns a psydac LinearOperator

        Returns
        -------
        Hodge operator in the specified form.
        """

        if not dual: 
            if kind == 'femlinop':
                return H.hodge
            elif kind == 'linop':
                return H.linop
        else: 
            if kind == 'femlinop':
                return H.dual_hodge
            elif kind == 'linop':
                return H.dual_linop

    #--------------------------------------------------------------------------
    def hodge_operator(self, space=None, dual=False, kind='femlinop', backend_language='python'):
        """
        Returns the Hodge operator for the given space and specified kind.
        
        Parameters
        ----------
        space : str or None
            The space for which to return the Hodge operator, can be 'V0', 'V1', 'V2' or None.
            If None, returns a tuple with all three Hodge operators.

        dual : bool
            If True, returns the dual Hodge operator.

        kind : <str>
            The kind of the projector, can be 'femlinop' or 'linop'.
            - 'femlinop' returns a psydac FemLinearOperator (default)
            - 'linop' returns a psydac LinearOperator

        backend_language : str
            The backend used to accelerate the code, default is 'python'.

        Returns
        -------
        The Hodge operator of the space of the specified kind.

        H : <psydac.fem.basic.FemLinearOperator> or <psydac.linalg.basic.LinearOperator>
        """

        if not self._hodge_operators:
            self._init_hodge_operators(backend_language=backend_language)
            
        if space == 'V0':
           return self._get_hodge_operator(self._hodge_operators[0], dual=dual, kind=kind)

        elif space == 'V1':
            return self._get_hodge_operator(self._hodge_operators[1], dual=dual, kind=kind)

        elif space == 'V2':
            return self._get_hodge_operator(self._hodge_operators[2], dual=dual, kind=kind)

        elif space == 'V3':
            return self._get_hodge_operator(self._hodge_operators[3], dual=dual, kind=kind)

    #--------------------------------------------------------------------------
    def hodge_operators(self, dual=False, kind='femlinop', backend_language='python'):
        """
        Returns the Hodge operators for the specified kind.
        
        Parameters
        ----------
        dual : bool
            If True, returns the dual Hodge operator.

        kind : <str>
            The kind of the projector, can be 'femlinop' or 'linop'.
            - 'femlinop' returns a psydac FemLinearOperator (default)
            - 'linop' returns a psydac LinearOperator

        backend_language : str
            The backend used to accelerate the code, default is 'python'.

        Returns
        -------
        The Hodge operators of all spaces and of the specified kind.
        """

        if not self._hodge_operators:
            self._init_hodge_operators(backend_language=backend_language)
            
        return tuple(self._get_hodge_operator(H, dual=dual, kind=kind) for H in self._hodge_operators)


#==============================================================================
class DiscreteDeRhamMultipatch(DiscreteDeRham):
    """ Represents the discrete de Rham sequence for multipatch domains.
        It only works when the number of patches>1.

    Parameters
    ----------
    domain_h: <Geometry>
     The discrete domain

    spaces: <list,tuple>
      The discrete spaces that are contained in the de Rham sequence
    """
    
    def __init__(self, *, domain_h, spaces):

        dim           = len(spaces) - 1
        self._spaces  = tuple(spaces)
        self._dim     = dim
        self._mapping = domain_h.domain.mapping
        self._callable_mapping = [m.get_callable_mapping() for m in self._mapping.mappings.values()] if self._mapping else None
        self._domain_h = domain_h
        self._sequence = tuple(space.symbolic_space.kind.name for space in spaces)


        if dim == 1:
            raise NotImplementedError('1D FEEC multipatch non available yet')

        elif dim == 2:

            if self._sequence[1] == 'hcurl':

                self._derivatives = (
                    BrokenGradient2D(self.V0, self.V1),
                    BrokenScalarCurl2D(self.V1, self.V2),  # None,
                )

            elif self._sequence[1] == 'hdiv':
                raise NotImplementedError('2D sequence with H-div not available yet')

            else:
                raise ValueError('2D sequence not understood')

        elif dim == 3:
            raise NotImplementedError('3D FEEC multipatch non available yet')

        else:
            raise ValueError('Dimension {} is not available'.format(dim))

        self._hodge_operators = ()
        self._conf_proj = ()

    #--------------------------------------------------------------------------
    @property
    def H1vec(self):
        raise NotImplementedError('Not implemented for Multipatch de Rham sequences.')

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
        P0: <MultipatchGeometricProjector>
         Patch wise H1 projector

        P1: <MultipatchGeometricProjector>
         Patch wise Hcurl projector

        P2: <MultipatchGeometricProjector>
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
            P0 = MultipatchGeometricProjector(self.V0, GlobalGeometricProjectorH1)

            if self.sequence[1] == 'hcurl':
                P1 = MultipatchGeometricProjector(self.V1, GlobalGeometricProjectorHcurl, nquads=nquads)
            else:
                P1 = MultipatchGeometricProjector(self.V1, GlobalGeometricProjectorHdiv, nquads=nquads)

            P2 = MultipatchGeometricProjector(self.V2, GlobalGeometricProjectorL2, nquads=nquads)

            if self.mapping:
                P0_m = lambda f : P0([pull_2d_h1(f, m) for m in self.callable_mapping])

                if self.sequence[1] == 'hcurl':
                    P1_m = lambda f : P1([pull_2d_hcurl(f, m) for m in self.callable_mapping])
                else:
                    raise NotImplementedError('2D sequence with H-div not available yet')

                P2_m = lambda f : P2([pull_2d_l2(f, m) for m in self.callable_mapping])

                return P0_m, P1_m, P2_m

            return P0, P1, P2

        elif self.dim == 3:
            raise NotImplementedError("3D projectors are not available")
