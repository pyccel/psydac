import numpy as np

from psydac.linalg.stencil import StencilMatrix
from psydac.linalg.block import BlockLinearOperator
from psydac.linalg.basic import Vector
from psydac.fem.basic import FemSpace
from psydac.fem.tensor import TensorFemSpace
from psydac.feec.global_projectors import GlobalProjector
from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL
from psydac.linalg.basic import LinearOperator
from psydac.feec import basis_projection_kernels
from psydac.utilities.quadratures import gauss_legendre
from psydac.fem.basic import FemField
from psydac.utilities.utils import roll_edges



class BasisProjectionOperator(LinearOperator):
    """
    Class for "basis projection operators" PI_ijk(fun Lambda_mno) in the general form BP * P * DOF * EV^T * BV^T.

    Parameters
    ----------
    P : struphy.psydac_api.projectors.Projector
        Global commuting projector mapping into TensorFemSpace/ProductFemSpace W = P.space (codomain of operator).

    V : psydac.fem.basic.FemSpace
        Finite element spline space (domain, input space).

    fun : list
        Weight function(s) (callables) in a 2d list of shape corresponding to number of components of domain/codomain.

    transposed : bool
        Whether to assemble the transposed operator.
    """

    def __init__(self, P, V, fun, transposed=False):

        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL['flags'] = '-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none'

        assert isinstance(P, GlobalProjector)
        assert isinstance(V, FemSpace)

        self._P = P
        self._V = V


        self._fun = fun
        self._transposed = transposed
        self._dtype = V.vector_space.dtype

        # set domain and codomain symbolic names
        if hasattr(P.space.symbolic_space, 'name'):
            P_name = P.space.symbolic_space.name
        else:
            P_name = 'H1vec'

        if hasattr(V.symbolic_space, 'name'):
            V_name = V.symbolic_space.name
        else:
            V_name = 'H1vec'

        if transposed:
            self._domain_symbolic_name = P_name
            self._codomain_symbolic_name = V_name
        else:
            self._domain_symbolic_name = V_name
            self._codomain_symbolic_name = P_name

        # ============= assemble tensor-product dof matrix =======
        dof_mat = BasisProjectionOperator.assemble_mat(
            P, V, fun)
        # ========================================================

        self._dof_operator = dof_mat

        if transposed:
            self._dof_operator = self._dof_operator.transpose()

        # set domain and codomain
        self._domain = self.dof_operator.domain
        self._codomain = self.dof_operator.codomain

        # temporary vectors for dot product
        self._tmp_dom = self._dof_operator.domain.zeros()
        self._tmp_codom = self._dof_operator.codomain.zeros()

    @property
    def domain(self):
        """ Domain vector space (input) of the operator.
        """
        return self._domain

    @property
    def codomain(self):
        """ Codomain vector space (input) of the operator.
        """
        return self._codomain

    @property
    def dtype(self):
        """ Datatype of the operator.
        """
        return self._dtype

    @property
    def tosparse(self):
        raise NotImplementedError()

    @property
    def toarray(self):
        raise NotImplementedError()

    @property
    def transposed(self):
        """ If the transposed operator is in play.
        """
        return self._transposed

    @property
    def dof_operator(self):
        """ The degrees of freedom operator as composite linear operator containing polar extraction and boundary operators. 
        """
        return self._dof_operator

    def dot(self, v, out=None):
        """
        Applies the basis projection operator to the FE coefficients v.

        Parameters
        ----------
        v : psydac.linalg.basic.Vector
            Vector the operator shall be applied to.

        out : psydac.linalg.basic.Vector, optional
            If given, the output will be written in-place into this vector.

        Returns
        -------
         out : psydac.linalg.basic.Vector
            The output (codomain) vector.
        """

        assert isinstance(v, Vector)
        assert v.space == self.domain

        if out is None:

            if self.transposed:
                # 1. apply inverse transposed inter-/histopolation matrix, 2. apply transposed dof operator
                out = self.dof_operator.dot(self._P.solver.solve(v, transposed=True))
            else:
                # 1. apply dof operator, 2. apply inverse inter-/histopolation matrix
                out = self._P.solver.solve(self.dof_operator.dot(v))

        else:

            assert isinstance(out, Vector)
            assert out.space == self.codomain

            if self.transposed:
                # 1. apply inverse transposed inter-/histopolation matrix, 2. apply transposed dof operator
                self._P.solver.solve(v, out=self._tmp_dom, transposed=True)
                self.dof_operator.dot(self._tmp_dom, out=out)
            else:
                # 1. apply dof operator, 2. apply inverse inter-/histopolation matrix
                self.dof_operator.dot(v, out=self._tmp_codom)
                self._P.solver.solve(self._tmp_codom, out=out)

        return out

    def transpose(self):
        """
        Returns the transposed operator.
        """
        return BasisProjectionOperator(self._P, self._V, self._fun, not self.transposed)

    @staticmethod
    def assemble_mat(P, V, fun):
        """
        Assembles the tensor-product DOF matrix sigma_i(fun*Lambda_j), where i=(i1, i2, ...) and j=(j1, j2, ...) depending on the number of spatial dimensions (1d, 2d or 3d).

        Parameters
        ----------
        P : GlobalProjector
            The psydac global tensor product projector defining the space onto which the input shall be projected.

        V : TensorFemSpace | ProductFemSpace
            The spline space which shall be projected.

        fun : list
            Weight function(s) (callables) in a 2d list of shape corresponding to number of components of domain/codomain.

        Returns
        -------
        dof_mat : StencilMatrix | BlockLinearOperator
            Degrees of freedom matrix in the full tensor product setting.
        """

        # input space: 3d StencilVectorSpaces and 1d SplineSpaces of each component
        if isinstance(V, TensorFemSpace):
            _Vspaces = [V.vector_space]
            _V1ds = [V.spaces]
        else:
            _Vspaces = V.vector_space
            _V1ds = [comp.spaces for comp in V.spaces]

        # output space: 3d StencilVectorSpaces and 1d SplineSpaces of each component
        if isinstance(P.space, TensorFemSpace):
            _Wspaces = [P.space.vector_space]
            _W1ds = [P.space.spaces]
        else:
            _Wspaces = P.space.vector_space
            _W1ds = [comp.spaces for comp in P.space.spaces]

        # retrieve number of quadrature points of each component (=1 for interpolation)
        _nqs = [[P.grid_x[comp][direction].shape[1]
                 for direction in range(V.ldim)] for comp in range(len(_W1ds))]

        # blocks of dof matrix
        blocks = []

        # ouptut vector space (codomain), row of block
        for Wspace, W1d, nq, fun_line in zip(_Wspaces, _W1ds, _nqs, fun):
            blocks += [[]]
            _Wdegrees = [space.degree for space in W1d]

            # input vector space (domain), column of block
            for Vspace, V1d, f in zip(_Vspaces, _V1ds, fun_line):

                # instantiate cell of block matrix
                dofs_mat = StencilMatrix(
                    Vspace, Wspace, backend=PSYDAC_BACKEND_GPYCCEL)

                _starts_in = np.array(dofs_mat.domain.starts)
                _ends_in = np.array(dofs_mat.domain.ends)
                _pads_in = np.array(dofs_mat.domain.pads)

                _starts_out = np.array(dofs_mat.codomain.starts)
                _ends_out = np.array(dofs_mat.codomain.ends)
                _pads_out = np.array(dofs_mat.codomain.pads)

                _ptsG, _wtsG, _spans, _bases, _subs = prepare_projection_of_basis(
                    V1d, W1d, _starts_out, _ends_out, nq)

                _ptsG = [pts.flatten() for pts in _ptsG]
                _Vnbases = [space.nbasis for space in V1d]

                # Evaluate weight function at quadrature points
                pts = np.meshgrid(*_ptsG, indexing='ij')
                #needs to be improved when passing FemFields, 
                #a lot of evaluations that are probably not needed
                if isinstance(f, FemField):
                    assert(isinstance(f.space,TensorFemSpace))
                    _fun_q = f.space.eval_fields_irregular_tensor_grid(_ptsG, f) 
                    _fun_q = np.squeeze(_fun_q) #since we only evaluate one field the result is a 3D
                    #array with last dim 1, we need to squeeze it in order to use the pyccelized kernels
                elif isinstance(f, float) or isinstance(f, int):
                    shape_grid = tuple([len(pts_i) for pts_i in _ptsG])
                    _fun_q = np.full(shape_grid, f)
                else : 
                    f = np.vectorize(f)
                    _fun_q = f(*pts).copy() #this formulation does not work atm for FemFields

                # Call the kernel if weight function is not zero
                if np.any(np.abs(_fun_q) > 1e-14):

                    kernel = getattr(
                        basis_projection_kernels, 'assemble_dofs_for_weighted_basisfuns_' + str(V.ldim) + 'd')

                    kernel(dofs_mat._data, _starts_in, _ends_in, _pads_in, _starts_out, _ends_out,
                           _pads_out, _fun_q, *_wtsG, *_spans, *_bases, *_subs, *_Vnbases, *_Wdegrees)

                    blocks[-1] += [dofs_mat]

                else:
                    blocks[-1] += [None]

        # build BlockLinearOperator (if necessary) and return
        if len(blocks) == len(blocks[0]) == 1:
            if blocks[0][0] is not None:
                return blocks[0][0]
            else:
                return dofs_mat
        else:
            return BlockLinearOperator(V.vector_space, P.space.vector_space, blocks)


def prepare_projection_of_basis(V1d, W1d, starts_out, ends_out, n_quad=None):
    '''Obtain knot span indices and basis functions evaluated at projection point sets of a given space.

    Parameters
    ----------
    V1d : 3-list
        Three SplineSpace objects from Psydac from the input space (to be projected).

    W1d : 3-list
        Three SplineSpace objects from Psydac from the output space (projected onto).

    starts_out : 3-list
        Global starting indices of process. 

    ends_out : 3-list
        Global ending indices of process.

    n_quad : 3_list
        Number of quadrature points per histpolation interval. If not given, is set to V1d.degree + 1.

    Returns
    -------
    ptsG : 3-tuple of 2d float arrays
        Quadrature points (or Greville points for interpolation) in each dimension in format (interval, quadrature point).

    wtsG : 3-tuple of 2d float arrays
        Quadrature weights (or ones for interpolation) in each dimension in format (interval, quadrature point).

    spans : 3-tuple of 2d int arrays
        Knot span indices in each direction in format (n, nq).

    bases : 3-tuple of 3d float arrays
        Values of p + 1 non-zero eta basis functions at quadrature points in format (n, nq, basis).'''

    import psydac.core.bsplines as bsp


    x_grid, subs, pts, wts, spans, bases = [], [], [], [], [], []

    # Loop over direction, prepare point sets and evaluate basis functions
    direction = 0
    for space_in, space_out, s, e in zip(V1d, W1d, starts_out, ends_out):

        greville_loc = space_out.greville[s: e + 1].copy()
        histopol_loc = space_out.histopolation_grid[s: e + 2].copy()

        # make sure that greville points used for interpolation are in [0, 1]
        #assert np.all(np.logical_and(greville_loc >= 0., greville_loc <= 1.))

        # interpolation
        if space_out.basis == 'B':
            x_grid = greville_loc
            pts += [greville_loc[:, None]]
            wts += [np.ones(pts[-1].shape, dtype=float)]
            # sub-interval index is always 0 for interpolation.
            subs += [np.zeros(pts[-1].shape[0], dtype=int)]

        # histopolation
        elif space_out.basis == 'M':

            """if space_out.degree % 2 == 0:
                union_breaks = space_out.breaks
            else:
                union_breaks = space_out.breaks[:-1]

            # Make union of Greville and break points
            tmp = set(np.round_(space_out.histopolation_grid, decimals=14)).union(
                np.round_(union_breaks, decimals=14))

            tmp = list(tmp)
            tmp.sort()
            tmp_a = np.array(tmp)

            x_grid = tmp_a[np.logical_and(tmp_a >= np.min(
                histopol_loc) - 1e-14, tmp_a <= np.max(histopol_loc) + 1e-14)]"""

            x_grid = space_out.histopolation_grid

            # determine subinterval index (= 0 or 1):
            subs += [np.zeros(len(x_grid), dtype=int)]
            for n, x_h in enumerate(x_grid):
                add = 1
                for x_g in histopol_loc:
                    if abs(x_h - x_g) < 1e-14:
                        add = 0
                subs[-1][n] += add

            # Gauss - Legendre quadrature points and weights
            if n_quad is None:
                # products of basis functions are integrated exactly
                nq = space_in.degree + 1
            else:
                nq = n_quad[direction]
            pts_loc, wts_loc = gauss_legendre(nq-1)
            pts_loc, wts_loc = pts_loc[::-1], wts_loc[::-1]
            global_quad_x, global_quad_w = bsp.quadrature_grid(x_grid, pts_loc, wts_loc)
            #"roll" back points to the interval to ensure that the quadrature points are
            #in the domain. Probably only usefull on periodic cases
            roll_edges(space_out.domain, global_quad_x) 
            x = global_quad_x#[s:e+1]
            w = global_quad_w#[s:e+1]
            pts += [x]
            wts += [w]

        # Knot span indices and V-basis functions evaluated at W-point sets
        s, b = get_span_and_basis(pts[-1], space_in)

        spans += [s]
        bases += [b]

        direction += 1

    return tuple(pts), tuple(wts), tuple(spans), tuple(bases), tuple(subs)


def get_span_and_basis(pts, space):
    '''Compute the knot span index and the values of p + 1 basis function at each point in pts.

    Parameters
    ----------
    pts : np.array
        2d array of points (interval, quadrature point).

    space : SplineSpace
        Psydac object, the 1d spline space to be projected.

    Returns
    -------
    span : np.array
        2d array indexed by (n, nq), where n is the interval and nq is the quadrature point in the interval.

    basis : np.array
        3d array of values of basis functions indexed by (n, nq, basis function). 
    '''

    import psydac.core.bsplines as bsp

    # Extract knot vectors, degree and kind of basis
    T = space.knots
    p = space.degree

    span = np.zeros(pts.shape, dtype=int)
    basis = np.zeros((*pts.shape, p + 1), dtype=float)

    for n in range(pts.shape[0]):
        for nq in range(pts.shape[1]):
            # avoid 1. --> 0. for clamped interpolation
            x = pts[n, nq] #% (1. + 1e-14)
            span_tmp = bsp.find_span(T, p, x)
            basis[n, nq, :] = bsp.basis_funs_all_ders(
                T, p, x, span_tmp, 0, normalization=space.basis)
            span[n, nq] = span_tmp  # % space.nbasis

    return span, basis
