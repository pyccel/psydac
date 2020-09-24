# -*- coding: UTF-8 -*-


from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.sparse import identity
from psydac.linalg.kron import kronecker_solve
from psydac.fem.tensor  import TensorFemSpace
from psydac.fem.vector  import ProductFemSpace
from psydac.linalg.direct_solvers import BandedSolver, SparseSolver
from psydac.linalg.stencil import StencilVector
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

    def __init__(self, Hcurl, quads=None):

        if quads:
            uw = [gauss_legendre( k-1 ) for k in quads]
            uw = [(u[::-1], w[::-1]) for u,w in uw]
        else:
            uw = [(V.quad_grids[i].quad_rule_x,V.quad_grids[i].quad_rule_w) for i,V in enumerate(Hcurl.spaces)]

        dim = len(quads)

        self.rhs = StencilVector(Hcurl.vector_space)

        for V in Hcurl.spaces:
            V.init_interpolation()
            V.init_histopolation()

        if dim == 3:
            self.DNN = [Hcurl.spaces[0]._histopolator, Hcurl.spaces[0]._interpolator, Hcurl.spaces[0]._interpolator]
            self.NDN = [Hcurl.spaces[1]._interpolator, Hcurl.spaces[1]._histopolator, Hcurl.spaces[1]._interpolator]
            self.NND = [Hcurl.spaces[2]._interpolator, Hcurl.spaces[2]._interpolator, Hcurl.spaces[2]._histopolator]

            n_basis1  = [V.nbasis for V in Hcurl.spaces[0].spaces]
            n_basis2  = [V.nbasis for V in Hcurl.spaces[1].spaces]
            n_basis3  = [V.nbasis for V in Hcurl.spaces[2].spaces]

        slices   = tuple(slice(p,-p) for p in Hcurl.degree)

        self.args1  = (*n_basis, *points, self.rhs._data[slices])
        self.args2  = (*n_basis, *points, self.rhs._data[slices])
        self.args3  = (*n_basis, *points, self.rhs._data[slices])

    # ======================================
    def __call__(self, fun):

        self.func(*self.args, fun)

        out = kronecker_solve(solvers = self.N, rhs = self.rhs)   
        return out

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
    # ======================================
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

class Interpolation(object):

    def __init__(self, **spaces):

        from psydac.utilities.integrate import Integral
        from psydac.utilities.integrate import Interpolation as Inter

        for Vh in spaces.values():
            assert isinstance(Vh, (TensorFemSpace, ProductFemSpace))
        
        H1 = spaces.pop('H1')
        
        if len(spaces) == 0:
            kind = 'H1'
        else:
            kind = list(spaces.keys())[0]
        
        T = [V.knots for V in H1.spaces]
        p = H1.degree
        n = [V.nbasis for V in H1.spaces]   


        Is = []
        Hs = []
        
        if kind == 'H1':
            for i in range(0, len(p)):
                _interpolation = Inter(p[i], n[i], T[i])
                Is.append(_interpolation)
                
        elif kind == 'L2':
            for i in range(0, len(p)):
                _integration   = Integral(p[i], n[i], T[i], kind='greville')
                Hs.append(_integration)
        else:
            for i in range(0, len(p)):
                _interpolation = Inter(p[i], n[i], T[i])
                _integration   = Integral(p[i], n[i], T[i], kind='greville')
                Is.append(_interpolation)
                Hs.append(_integration)

        self._interpolate = Is
        self._integrate   = Hs

        self._p = p
        self._n = n
        self._T = T
        self._spaces = spaces
        self._kind   = kind
        self._dim = len(p)

    @property
    def sites(self):
        return [i.sites for i in self._interpolate]
        
    @property
    def dim(self):
        return self._dim
        
    @property
    def spaces(self):
        return self._spaces
        
    @property
    def kind(self):
        return self._kind

    def __call__(self, f):
        """Computes the integral of the function f over each element of the grid."""
        
        kind = self.kind
        space = self.spaces[kind]
        slices = tuple(slice(p,-p) for p in self._p)
        
        if kind == 'H1':
            F = StencilVector(space.vector_space)
            if self.dim == 3:
                interpolate_3d(*self._n, *self.sites, F._data[slices], f)         
            elif self.dim == 2:
                interpolate_2d(*self._n, *self.sites, F._data[slices], f)                        
            elif self.dim == 1:
                interpolate_1d(*self._n, *self.sites, F._data[slices], f)               
                
            return F

        elif kind == 'Hcurl':
            if self.dim == 3:
                n1 = (self._n[0]-1, self._n[1], self._n[2])
                n2 = (self._n[0], self._n[1]-1, self._n[2])
                n3 = (self._n[0], self._n[1], self._n[2]-1)
                
                ipoints_1 = self._integrate[0]._points
                ipoints_2 = self._integrate[1]._points
                ipoints_3 = self._integrate[2]._points
 
                weights_1 = self._integrate[0]._weights
                weights_2 = self._integrate[1]._weights
                weights_3 = self._integrate[2]._weights

                points_1 = self.sites[0]
                points_2 = self.sites[1]
                points_3 = self.sites[2]
                
                k1 = len(weights_1)
                k2 = len(weights_2)
                k3 = len(weights_3)
                
                F1 = StencilVector(space.spaces[0].vector_space)
                F2 = StencilVector(space.spaces[1].vector_space)
                F3 = StencilVector(space.spaces[2].vector_space)
                

                f1 = f[0]
                f2 = f[1]
                f3 = f[2]
                Hcurl_projection_3d(n1, n2, n3, k1, k2, k3, weights_1, weights_2, weights_3, ipoints_1, ipoints_2, 
                                    ipoints_3, points_1, points_2, points_3, F1._data[slices], F2._data[slices], F3._data[slices],
                                     f1, f2, f3)
                                
                return F1, F2, F3
            
            elif self.dim == 2:
                n1 = (self._n[0]-1, self._n[1])
                n2 = (self._n[0], self._n[1]-1)
                
                ipoints_1 = self._integrate[0]._points
                ipoints_2 = self._integrate[1]._points
 
                weights_1 = self._integrate[0]._weights
                weights_2 = self._integrate[1]._weights

                points_1 = self.sites[0]
                points_2 = self.sites[1]
                
                k1 = len(weights_1)
                k2 = len(weights_2)
                
                F1 = StencilVector(space.spaces[0].vector_space)
                F2 = StencilVector(space.spaces[1].vector_space)
 
                f1 = f[0]
                f2 = f[1]
                
                Hcurl_projection_2d(n1, n2, k1, k2, weights_1, weights_2, ipoints_1, ipoints_2, 
                                                    points_1, points_2, F1._data[slices], F2._data[slices], f1, f2)
                            
                return F1 ,F2
            
            
        elif kind == 'Hdiv':
            if self.dim == 3:
                n1 = (self._n[0], self._n[1]-1, self._n[2]-1)
                n2 = (self._n[0]-1, self._n[1], self._n[2]-1)
                n3 = (self._n[0]-1, self._n[1]-1, self._n[2])
                
                ipoints_1 = self._integrate[0]._points
                ipoints_2 = self._integrate[1]._points
                ipoints_3 = self._integrate[2]._points
 
                weights_1 = self._integrate[0]._weights
                weights_2 = self._integrate[1]._weights
                weights_3 = self._integrate[2]._weights

                points_1 = self.sites[0]
                points_2 = self.sites[1]
                points_3 = self.sites[2]
                
                k1 = len(weights_1)
                k2 = len(weights_2)
                k3 = len(weights_3)
                

                F1 = StencilVector(space.spaces[0].vector_space)
                F2 = StencilVector(space.spaces[1].vector_space)
                F3 = StencilVector(space.spaces[2].vector_space)

                f1 = f[0]
                f2 = f[1]
                f3 = f[2]
                
                Hdiv_projection_3d(n1, n2, n3, k1, k2, k3, weights_1, weights_2, weights_3,ipoints_1, ipoints_2,  
                                   ipoints_3, points_1, points_2, points_3, F1._data[slices], F2._data[slices], F3._data[slices]
                                   , f1, f2, f3)
                                
                return F1, F2, F3
            
            elif self.dim == 2:
                n1 = (self._n[0], self._n[1]-1)
                n2 = (self._n[0]-1, self._n[1])
                
                ipoints_1 = self._integrate[0]._points
                ipoints_2 = self._integrate[1]._points
 
                weights_1 = self._integrate[0]._weights
                weights_2 = self._integrate[1]._weights

                points_1 = self.sites[0]
                points_2 = self.sites[1]
                
                k1 = len(weights_1)
                k2 = len(weights_2)
                
                F1 = StencilVector(space.spaces[0].vector_space)
                F2 = StencilVector(space.spaces[1].vector_space)

                f1 = f[0]
                f2 = f[1]
                Hdiv_projection_2d(n1, n2, k1, k2, weights_1, weights_2, ipoints_1, ipoints_2, 
                                   points_1, points_2, F1._data[slices], F2._data[slices], f1, f2)
                            
                return F1, F2

        elif kind == 'L2':

            points  = tuple(self._integrate[i]._points for i in range(self._dim))
            weights = tuple(self._integrate[i]._weights for i in range(self._dim))
            F       = StencilVector(space.vector_space)
            if self._dim == 1:
                integrate_1d(points, weights, F._data[slices], f)
            elif self._dim == 2:
                integrate_2d(points, weights, F._data[slices], f)
            elif self._dim == 3:
                integrate_3d(points, weights, F._data[slices], f)
            return F
        else:
            raise NotImplementedError('Only H1, Hcurl ,Hdiv and L2 are available')

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
    for i2 in range(n1[1]):
        for i3 in range(n1[2]):
            for i1 in range(n1[0]):
                for g1 in range(k1):
                    F1[i1, i2, i3] += weights_1[g1, i1]*f1(ipoints_1[g1, i1],points_2[i2], points_3[i3])

    for i1 in range(n2[0]):          
        for i3 in range(n2[2]):
            for i2 in range(n2[1]): 
                for g2 in range(k2):
                    F2[i1, i2, i3] += weights_2[g2, i2]*f2(points_1[i1],ipoints_2[g2, i2], points_3[i3])

    for i1 in range(n3[0]):
        for i2 in range(n3[1]):
            for i3 in range(n3[2]):
                for g3 in range(k3):
                    F3[i1, i2, i3] += weights_3[g3, i3]*f3(points_1[i1],points_2[i2], ipoints_3[g3, i3])
                    
def Hdiv_projection_2d(n1, n2, k1, k2, weights_1, weights_2, ipoints_1, ipoints_2, points_1, points_2, F1, F2, f1, f2):
    for i1 in range(n1[0]):
        for i2 in range(n1[1]):
            for g2 in range(k2):
                F1[i1, i2] += weights_2[g2, i2]*f1(points_1[i1],ipoints_2[g2,i2])

    for i2 in range(n2[1]):
        for i1 in range(n2[0]):          
            for g1 in range(k1):
                F2[i1, i2] += weights_1[g1, i1]*f2(ipoints_1[g1,i1],points_2[i2])
                
def Hdiv_projection_3d(n1, n2, n3, k1, k2, k3, weights_1, weights_2, weights_3, ipoints_1, ipoints_2, ipoints_3,           
                                                  points_1, points_2, points_3, F1, F2, F3, f1, f2, f3):
    for i1 in range(n1[0]):
        for i2 in range(n1[1]):
            for i3 in range(n1[2]):
                for g2 in range(k2):
                    for g3 in range(k3):
                        F1[i1, i2, i3] += weights_2[g2, i2]*weights_3[g3, i3]*f1(points_1[i1],
                                                                        ipoints_2[g2,i2], ipoints_3[g3,i3])
                                                                        
    for i2 in range(n2[1]):
        for i1 in range(n2[0]):          
            for i3 in range(n2[2]):
                for g1 in range(k1):
                    for g3 in range(k3):
                        F2[i1, i2, i3] += weights_1[g1, i1]*weights_3[g3, i3]*f2(ipoints_1[g1,i1],
                                                                points_2[i2], ipoints_3[g3, i3])
    for i3 in range(n3[2]):
        for i1 in range(n3[0]):
            for i2 in range(n3[1]):
                for g1 in range(k1):
                    for g2 in range(k2):
                        F3[i1, i2, i3] += weights_1[g1, i1]*weights_2[g2, i2]*f3(ipoints_1[g1,i1],
                                                        ipoints_2[g2,i2], points_3[i3])


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

    n1 = points_1.shape[1]
    n2 = points_2.shape[1]
    k1 = points_1.shape[0]
    k2 = points_2.shape[0]

    for ie1 in range(n1):
        for ie2 in range(n2):
            for g1 in range(k1):
                for g2 in range(k2):
                    F[ie1, ie2] += weights_1[g1,ie1]*weights_2[g2, ie2]*fun(points_1[g1, ie1], points_2[g2, ie2])


def integrate_3d(points_1, points_2, points_3,  weights_1, weights_2, weights_3, F, fun):

    n1 = points_0.shape[1]
    n2 = points_1.shape[1]
    n3 = points_2.shape[1]
    k1 = points_0.shape[0]
    k2 = points_1.shape[0]
    k3 = points_2.shape[0]

    for ie1 in range(n1):
        for ie2 in range(n2):
            for ie3 in range(n3):
                for g1 in range(k1):
                    for g2 in range(k2):
                        for g3 in range(k3):
                            F[ie1, ie2, ie3] += weights_1[g1, ie1]*weights_2[g2, ie2]*weights_3[g3, ie3]\
                                                       *fun(pts_1[g1, ie1], pts_2[g2, ie2], pts_3[g3, ie3])
