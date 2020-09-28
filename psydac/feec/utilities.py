# -*- coding: UTF-8 -*-


from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.sparse import identity
from psydac.linalg.kron import kronecker_solve
from psydac.fem.tensor  import TensorFemSpace
from psydac.fem.vector  import ProductFemSpace
from psydac.linalg.direct_solvers import BandedSolver, SparseSolver
from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector
from itertools import product
from psydac.core.bsplines import quadrature_grid
from psydac.utilities.quadratures import gauss_legendre
from psydac.linalg.utilities import array_to_stencil

from numpy import zeros
from numpy import array
import numpy as np

class Kron:

    def __init__(self, *args):
        self._args = args
        
    @property
    def args(self):
        return self._args
        
    def solve(self, rhs, out=None):
       args = [SparseSolver(arg) for arg in self.args]
       out  = kronecker_solve(args, rhs, out)
       return out
       
    def toarray(self):
        M1 = self.args[0].toarray()
        for M2 in self.args[1:]:
            M1 = np.kron(M1, M2.toarray())
        return M1
        

class block_diag:

    def __init__(self, *args):
        self._args = args
        
    @property
    def args(self):
        return self._args
        
    def solve(self, rhs, out=None):
        if out is None:
            out = [None]*len(rhs)

        for i,arg in enumerate(self.args):
            out[i] = arg.solve(rhs[i], out[i])

        return out
# ...
def build_kron_matrix(p, n, T, kind):
    """."""
    from psydac.core.interface import collocation_matrix
    from psydac.core.interface import histopolation_matrix
    from psydac.core.interface import compute_greville

    if not isinstance(p, (tuple, list)) or not isinstance(n, (tuple, list)):
        raise TypeError('Wrong type for n and/or p. must be tuple or list')

    assert(len(kind) == len(T))

    grid = [compute_greville(_p, _n, _T) for (_n,_p,_T) in zip(n, p, T)]

    Ms = []
    for i in range(0, len(p)):
        _p = p[i]
        _n = n[i]
        _T = T[i]
        _grid = grid[i]
        _kind = kind[i]

        if _kind == 'interpolation':
            _kind = 'collocation'
        else:
            assert(_kind == 'histopolation')

        func = eval('{}_matrix'.format(_kind))
        M = func(_p, _n, _T, _grid)
        M = csr_matrix(M)

        Ms.append(M) # kron expects dense matrices

    return Kron(*Ms)
# ...
def _interpolation_matrices_3d(p, n, T):
    """."""

    # H1
    M0 = build_kron_matrix(p, n, T, kind=['interpolation', 'interpolation', 'interpolation'])

    # H-curl
    A = build_kron_matrix(p, n, T, kind=['histopolation', 'interpolation', 'interpolation'])
    B = build_kron_matrix(p, n, T, kind=['interpolation', 'histopolation', 'interpolation'])
    C = build_kron_matrix(p, n, T, kind=['interpolation', 'interpolation', 'histopolation'])
    M1 = block_diag(A, B, C)
    
    # H-div
    A = build_kron_matrix(p, n, T, kind=['interpolation', 'histopolation', 'histopolation'])
    B = build_kron_matrix(p, n, T, kind=['histopolation', 'interpolation', 'histopolation'])
    C = build_kron_matrix(p, n, T, kind=['histopolation', 'histopolation', 'interpolation'])
    M2 = block_diag(A, B, C)

    # L2
    M3 = build_kron_matrix(p, n, T, kind=['histopolation', 'histopolation','histopolation'])

    return M0, M1, M2, M3

def _interpolation_matrices_2d(p, n, T):
    """."""

    # H1
    M0 = build_kron_matrix(p, n, T, kind=['interpolation', 'interpolation'])

    # H-curl
    A = build_kron_matrix(p, n, T, kind=['histopolation', 'interpolation'])
    B = build_kron_matrix(p, n, T, kind=['interpolation', 'histopolation'])
    M1 = block_diag(A, B)

    # L2
    M2 = build_kron_matrix(p, n, T, kind=['histopolation', 'histopolation'])

    return M0, M1, M2


def interpolation_matrices(Vh):
    """Returns all interpolation matrices.
    This is a user-friendly function.
    """
    # 1d case
    assert isinstance(Vh, TensorFemSpace)
    
    T = [V.knots for V in Vh.spaces]
    p = Vh.degree
    n = [V.nbasis for V in Vh.spaces] 
    if isinstance(p, int):
        from psydac.core.interface import compute_greville
        from psydac.core.interface import collocation_matrix
        from psydac.core.interface import histopolation_matrix

        grid = compute_greville(p, n, T)

        M = collocation_matrix(p, n, T, grid)
        H = histopolation_matrix(p, n, T, grid)

        return M, H

    if not isinstance(p, (list, tuple)):
        raise TypeError('Expecting p to be int or list/tuple')

    if len(p) == 2:
        return _interpolation_matrices_2d(p, n, T)
        
    elif len(p) == 3:
        return _interpolation_matrices_3d(p, n, T)

    raise NotImplementedError('only 1d 2D and 3D are available')


class H1_Projector:

    def __init__(self, H1):

        # Quadrature grids in cells defined by consecutive Greville points

        points = [V.greville for V in H1.spaces]

        H1.init_interpolation()

        # Collocation matrices for N-splines in each direction
        self.N   = [V._interpolator for V in H1.spaces]
        n_basis  = [V.nbasis for V in H1.spaces]
        self.rhs = StencilVector(H1.vector_space)
        slices   = tuple(slice(p,-p) for p in H1.degree)

        self.space = H1
        self.args  = (*n_basis, *points, self.rhs._data[slices])

        if len(self.N) == 1:
            self.func = interpolate_1d
        elif len(self.N) == 2:
            self.func = interpolate_2d
        elif len(self.N) == 3:
            self.func = interpolate_3d
        else:
            raise ValueError('H1 projector of dimension {} not available'.format(str(len(self.N))))

    # ======================================
    def __call__(self, fun):

        '''
        Projection on the space V0 via interpolation.

        Parameters
        ----------
        fun : callable
            fun(x) \in R is the 0-form to be projected.

        Returns
        -------
        coeffs : 1D array_like
            Finite element coefficients obtained by projection.
        '''

        self.func(*self.args, fun)
        if len(self.N)==1:
            rhs = self.rhs.toarray()
            return array_to_stencil(self.N[0].solve(rhs), self.space.vector_space)
        out = kronecker_solve(solvers = self.N, rhs = self.rhs)
        return out

class Hcurl_Projector:

    def __init__(self, Hcurl, n_quads=None):

        if n_quads:
            uw = [gauss_legendre( k-1 ) for k in n_quads]
            uw = [(u[::-1], w[::-1]) for u,w in uw]
        else:
            uw = [(V.quad_grids[i].quad_rule_x,V.quad_grids[i].quad_rule_w) for i,V in enumerate(Hcurl.spaces)]

        dim = len(n_quads)

        self.rhs = BlockVector(Hcurl.vector_space)
        self.out = BlockVector(Hcurl.vector_space)

        for V in Hcurl.spaces:
            V.init_interpolation()
            V.init_histopolation()

        if dim == 3:
            Ns       = [Hcurl.spaces[1].spaces[0], Hcurl.spaces[0].spaces[1], Hcurl.spaces[0].spaces[2]]
            Ds       = [Hcurl.spaces[0].spaces[0], Hcurl.spaces[1].spaces[1], Hcurl.spaces[2].spaces[2]]

            self.DNN = [Ds[0]._histopolator, Ns[1]._interpolator, Ns[2]._interpolator]
            self.NDN = [Ns[0]._interpolator, Ds[1]._histopolator, Ns[2]._interpolator]
            self.NND = [Ns[0]._interpolator, Ns[1]._interpolator, Ds[2]._histopolator]

            n_basis = [V.nbasis for V in Ns]

            quads  = [quadrature_grid( V.ext_greville , u, w ) for V,(u,w) in zip(Ds, uw)]
            points = [V.greville for V in Ns]

            i_points, i_weights = list(zip(*quads))

            slices   = tuple(slice(p,-p) for p in Hcurl.spaces[0].vector_space.pads)

            self.args   =  (*n_basis, *n_quads, *i_weights, *i_points, *points,
                            self.rhs[0]._data[slices], self.rhs[1]._data[slices], self.rhs[2]._data[slices])

            self.func = Hcurl_projection_3d
            self.Ns = Ns
            self.Ds = Ds

    # ======================================
    def __call__(self, fun):

        self.func(*self.args, *fun)

        self.out[0] = kronecker_solve(solvers = self.DNN, rhs = self.rhs[0])
        self.out[1] = kronecker_solve(solvers = self.NDN, rhs = self.rhs[1])
        self.out[2] = kronecker_solve(solvers = self.NND, rhs = self.rhs[2])

        return self.out

class Hdiv_Projector:

    def __init__(self, Hdiv, n_quads=None):

        if n_quads:
            uw = [gauss_legendre( k-1 ) for k in n_quads]
            uw = [(u[::-1], w[::-1]) for u,w in uw]
        else:
            uw = [(V.quad_grids[i].quad_rule_x,V.quad_grids[i].quad_rule_w) for i,V in enumerate(Hdiv.spaces)]

        dim = len(n_quads)

        self.rhs = BlockVector(Hdiv.vector_space)
        self.out = BlockVector(Hdiv.vector_space)

        for V in Hdiv.spaces:
            V.init_interpolation()
            V.init_histopolation()

        if dim == 3:
            Ns       = [Hdiv.spaces[0].spaces[0], Hdiv.spaces[1].spaces[1], Hdiv.spaces[2].spaces[2]]
            Ds       = [Hdiv.spaces[1].spaces[0], Hdiv.spaces[0].spaces[1], Hdiv.spaces[0].spaces[2]]

            self.NDD = [Ns[0]._interpolator, Ds[1]._histopolator, Ds[2]._histopolator]
            self.DND = [Ds[0]._histopolator, Ns[1]._interpolator, Ds[2]._histopolator]
            self.DDN = [Ds[0]._histopolator, Ds[1]._histopolator, Ns[2]._interpolator]

            n_basis = [V.nbasis for V in Ns]

            quads  = [quadrature_grid( V.ext_greville , u, w ) for V,(u,w) in zip(Ds, uw)]
            points = [V.greville for V in Ns]

            i_points, i_weights = list(zip(*quads))

            slices   = tuple(slice(p,-p) for p in Hdiv.spaces[0].vector_space.pads)

            self.args   =  (*n_basis, *n_quads, *i_weights, *i_points, *points,
                            self.rhs[0]._data[slices], self.rhs[1]._data[slices], self.rhs[2]._data[slices])

            self.func = Hdiv_projection_3d
            self.Ns = Ns
            self.Ds = Ds

    # ======================================
    def __call__(self, fun):

        self.func(*self.args, *fun)

        self.out[0] = kronecker_solve(solvers = self.NDD, rhs = self.rhs[0])
        self.out[1] = kronecker_solve(solvers = self.DND, rhs = self.rhs[1])
        self.out[2] = kronecker_solve(solvers = self.DDN, rhs = self.rhs[2])

        return self.out

class L2_Projector:

    def __init__(self, L2, quads=None):

        # Quadrature grids in cells defined by consecutive Greville points
        if quads:
            uw = [gauss_legendre( k-1 ) for k in quads]
            uw = [(u[::-1], w[::-1]) for u,w in uw]
        else:
            uw = [(V.quad_rule_x,V.quad_rule_w) for V in L2.quad_grids]

        quads           = [quadrature_grid( V.ext_greville , u, w ) for V,(u,w) in zip(L2.spaces, uw)]
        points, weights = list(zip(*quads))

        L2.init_histopolation()

        # Histopolation matrices for D-splines in each direction
        self.D = [V._histopolator for V in L2.spaces]

        self.rhs = StencilVector(L2.vector_space)
        slices   = tuple(slice(p+1,-p-1) for p in L2.degree)

        if len(self.D) == 1:
            self.func = integrate_1d
        elif len(self.D) == 2:
            self.func = integrate_2d
        elif len(self.D) == 3:
            self.func = integrate_3d
        else:
            raise ValueError('H1 projector of dimension {} not available'.format(str(len(self.N))))

        self.space = L2
        self.args  = (*points, *weights, self.rhs._data[slices])

    def __call__(self, fun):

        '''
        Projection on the space V1 via histopolation.

        Parameters
        ----------
        fun : callable
            fun(x) \in R is the 1-form to be projected.

        Returns
        -------
        coeffs : Vector
            Finite element coefficients obtained by projection.
        '''

        self.func(*self.args, fun)
        if len(self.D)==1:
            rhs = self.rhs.toarray()
            return array_to_stencil(self.D[0].solve(rhs), self.space.vector_space)
        out = kronecker_solve(solvers = self.D, rhs = self.rhs)
        return out

def interpolate_1d(n1, points_1, F, f):
    for i1 in range(n1):
        F[i1] = f(points_1[i1])
        
def interpolate_2d(n1, n2, points_1, points_2, F, f):
    for i1 in range(n1):
        for i2 in range(n2):
            F[i1,i2] = f(points_1[i1], points_2[i2])

def interpolate_3d(n1, n2, n3, points_1, points_2, points_3, F, f):
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F[i1, i2, i3] = f(points_1[i1], points_2[i2], points_3[i3])

def Hcurl_projection_2d(n1, n2, k1, k2, weights_1, weights_2, ipoints_1, ipoints_2, points_1, points_2, F1, F2, f1, f2):

    for i2 in range(n1[1]):
        for i1 in range(n1[0]):
            for g1 in range(k1):
                F1[i1, i2] += weights_1[g1, i1]*f1(ipoints_1[g1, i1], points_2[i2])
                
    for i1 in range(n2[0]):
        for i2 in range(n2[1]):
            for g2 in range(k2):
                F2[i1, i2] += weights_2[g2, i2]*f2(points_1[i1], ipoints_2[g2, i2])
                
def Hcurl_projection_3d(n1, n2, n3, k1, k2, k3, weights_1, weights_2, weights_3, ipoints_1, ipoints_2, ipoints_3,           
                                                  points_1, points_2, points_3, F1, F2, F3, f1, f2, f3):
    for i2 in range(n2):
        for i3 in range(n3):
            for i1 in range(n1-1):
                for g1 in range(k1):
                    F1[i1, i2, i3] += weights_1[i1, g1]*f1(ipoints_1[i1, g1], points_2[i2], points_3[i3])

    for i1 in range(n1):
        for i3 in range(n3):
            for i2 in range(n2-1):
                for g2 in range(k2):
                    F2[i1, i2, i3] += weights_2[i2, g2]*f2(points_1[i1],ipoints_2[i2, g2], points_3[i3])

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3-1):
                for g3 in range(k3):
                    F3[i1, i2, i3] += weights_3[i3, g3]*f3(points_1[i1],points_2[i2], ipoints_3[i3, g3])
                    
def Hdiv_projection_2d(n1, n2, k1, k2, weights_1, weights_2, ipoints_1, ipoints_2, points_1, points_2, F1, F2, f1, f2):
    for i1 in range(n1[0]):
        for i2 in range(n1[1]):
            for g2 in range(k2):
                F1[i1, i2] += weights_2[i2, g2]*f1(points_1[i1],ipoints_2[i2, g2])

    for i2 in range(n2[1]):
        for i1 in range(n2[0]):          
            for g1 in range(k1):
                F2[i1, i2] += weights_1[g1, i1]*f2(ipoints_1[g1,i1],points_2[i2])
                
def Hdiv_projection_3d(n1, n2, n3, k1, k2, k3, weights_1, weights_2, weights_3, ipoints_1, ipoints_2, ipoints_3,
                                                  points_1, points_2, points_3, F1, F2, F3, f1, f2, f3):
    for i1 in range(n1):
        for i2 in range(n2-1):
            for i3 in range(n3-1):
                for g2 in range(k2):
                    for g3 in range(k3):
                        F1[i1, i2, i3] += weights_2[i2, g2]*weights_3[i3, g3]*f1(points_1[i1],
                                                                        ipoints_2[i2, g2], ipoints_3[i3, g3])                        
    for i2 in range(n2):
        for i1 in range(n1-1):
            for i3 in range(n3-1):
                for g1 in range(k1):
                    for g3 in range(k3):
                        F2[i1, i2, i3] += weights_1[i1, g1]*weights_3[i3, g3]*f2(ipoints_1[i1, g1],
                                                                points_2[i2], ipoints_3[i3, g3])
    for i3 in range(n3):
        for i1 in range(n1-1):
            for i2 in range(n2-1):
                for g1 in range(k1):
                    for g2 in range(k2):
                        F3[i1, i2, i3] += weights_1[i1, g1]*weights_2[i2, g2]*f3(ipoints_1[i1, g1],
                                                        ipoints_2[i2, g2], points_3[i3])


def integrate_1d(points, weights, F, fun):
    """Integrates the function f over the quadrature grid
    defined by (points,weights) in 1d.

    points: np.array
        a multi-dimensional array describing the quadrature points mapped onto
        the grid. it must be constructed using construct_quadrature_grid

    weights: np.array
        a multi-dimensional array describing the quadrature weights (scaled) mapped onto
        the grid. it must be constructed using construct_quadrature_grid

    Examples

    >>> from psydac.core.interface import make_open_knots
    >>> from psydac.core.interface import construct_grid_from_knots
    >>> from psydac.core.interface import construct_quadrature_grid
    >>> from psydac.core.interface import compute_greville
    >>> from psydac.utilities.quadratures import gauss_legendre

    >>> n_elements = 8
    >>> p = 2                    # spline degree
    >>> n = n_elements + p - 1   # number of control points
    >>> T = make_open_knots(p, n)
    >>> grid = compute_greville(p, n, T)
    >>> u, w = gauss_legendre(p)  # gauss-legendre quadrature rule
    >>> k = len(u)
    >>> ne = len(grid) - 1        # number of elements
    >>> points, weights = construct_quadrature_grid(ne, k, u, w, grid)
    >>> f = lambda u: u*(1.-u)
    >>> f_int = integrate(points, weights, f)
    >>> f_int
    [0.00242954 0.01724976 0.02891156 0.03474247 0.03474247 0.02891156
     0.01724976 0.00242954]
    n = points.shape[0]
    k = points.shape[1]
    """
    n1 = points.shape[0]
    k1 = points.shape[1]

    for ie1 in range(n1):
        for g1 in range(k1):
            F[ie1] += weights[ie1, g1]*fun(points[ie1, g1])

def integrate_2d(points_1, points_2, weights_1, weights_2, F, fun):

    """Integrates the function f over the quadrature grid
    defined by (points,weights) in 2d.

    points: list, tuple
        list of quadrature points, as they should be passed for `integrate`

    weights: list, tuple
        list of quadrature weights, as they should be passed for `integrate`

    Examples

    """

    n1 = points_1.shape[0]
    n2 = points_2.shape[0]
    k1 = points_1.shape[1]
    k2 = points_2.shape[1]

    for ie1 in range(n1):
        for ie2 in range(n2):
            for g1 in range(k1):
                for g2 in range(k2):
                    F[ie1, ie2] += weights_1[ie1, g1]*weights_2[ie2, g2]*fun(points_1[ie1, g1], points_2[ie2, g2])


def integrate_3d(points_1, points_2, points_3,  weights_1, weights_2, weights_3, F, fun):

    n1 = points_1.shape[0]
    n2 = points_2.shape[0]
    n3 = points_3.shape[0]
    k1 = points_1.shape[1]
    k2 = points_2.shape[1]
    k3 = points_3.shape[1]

    for ie1 in range(n1):
        for ie2 in range(n2):
            for ie3 in range(n3):
                for g1 in range(k1):
                    for g2 in range(k2):
                        for g3 in range(k3):
                            F[ie1, ie2, ie3] += weights_1[ie1, g1]*weights_2[ie2, g2]*weights_3[ie3, g3]\
                                                       *fun(points_1[ie1, g1], points_2[ie2, g2], points_3[ie3, g3])
