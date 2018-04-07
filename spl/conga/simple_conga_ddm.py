# -*- coding: UTF-8 -*-

__author__ = 'campos'

import numpy as np

from scipy.integrate import quad
from scipy.interpolate import BSpline
from scipy.interpolate import splev
from scipy.special.orthogonal import p_roots
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import splu

import matplotlib
matplotlib.use('Agg')   # backend configuration: high quality PNG images using the Anti-Grain Geometry engine
import matplotlib.pyplot as plt

from spl.core.interface import make_open_knots
from spl.core.interface import construct_grid_from_knots
from spl.core.interface import construct_quadrature_grid
from spl.utilities.quadratures import gauss_legendre
from spl.utilities import Contribution
from spl.feec import get_tck
from spl.feec import mass_matrices


# import os.path
# import spl.core as core
# print ('writing path : ')
# print (os.path.abspath(core.__file__))
# exit()

# ...
def solve(M, x):
    """Solve y:= Mx using SuperLU."""
    M = csc_matrix(M)
    M_op = splu(M)
    return M_op.solve(x)
# ...


class SimpleCongaDDM:
    """
    class for a simple Conga method with 2 domains in 1D: [0, 0.5] and [0.5, 1]


    Here the discontinous space consists of two spline spaces, one for each domain.
    The smooth space is a std spline space on the whole domain, with a grid that is the union of the two grids.
    """

    def __init__(self,
                 p,
                 m,
                 N_cells_sub,
                 watch_your_steps=False,
                 n_checks=5,
                 use_macro_elem_duals=True,
                 ):
        """
        p: int
            spline degree

        m: int
            degree of the moments preserved by the smoothing operator

        N_cells_sub: int
            number of cells on each subdomain, must be a multiple (> 1) of m+p+1 the macro-elements size
        """
        if not isinstance(p, int) or not isinstance(m, int) or not isinstance(N_cells_sub, int):
            raise TypeError('Wrong type for N_cells_sub and/or p. must be int')

        # degree of the moments preserved by the smoothing operator
        self._m = m

        # size of macro-elements
        self._M_p = m+p+1

        if not np.mod(N_cells_sub,self._M_p) == 0 or N_cells_sub/self._M_p < 2:
            raise ValueError('Wrong value for N_cells_sub, must be a multiple (>1) of m+p+1')

        self._p = p
        self._N_cells = 2 * N_cells_sub
        self._N_cells_left = N_cells_sub
        self._N_cells_right = N_cells_sub

        self._x_min = 0
        self._x_max = 1
        self._x_sep = 0.5

        self._h = (self._x_max-self._x_min)*(1./self._N_cells)

        # number of spline functions:
        self._n_smooth = self._N_cells + p
        self._n_left = self._N_cells_left + p
        self._n_right = self._N_cells_right + p

        # number of discontinuous (pw spline) functions:
        self._n_disc = self._n_left + self._n_right

        # self._T_smooth = self._x_min + (self._x_max - self._x_min) * make_open_knots(p, self._n_smooth)
        self._T_smooth, h_check = np.linspace(
            self._x_min - self._h*self._p,
            self._x_max + self._h*self._p,
            num=self._N_cells + 2*self._p+1,
            endpoint=True, retstep=True)
        if not np.allclose(h_check, self._h):
            print("error : "+repr(h_check)+" "+repr(self._h))
            assert np.allclose(h_check, self._h)

        # knot vector for the left spline space:
        self._T_left, h_check = np.linspace(
            self._x_min - self._h*self._p,
            self._x_sep + self._h*self._p,
            num=self._N_cells_left + 2*self._p+1,
            endpoint=True, retstep=True)
        if not np.allclose(h_check, self._h):
            print("error : "+repr(h_check)+" "+repr(self._h))
            assert np.allclose(h_check, self._h)

        # knot vector for the right spline space:
        self._T_right, h_check = np.linspace(
            self._x_sep - self._h*self._p,
            self._x_max + self._h*self._p,
            num=self._N_cells_right + 2*self._p+1,
            endpoint=True, retstep=True)
        if not np.allclose(h_check, self._h):
            print("error : "+repr(h_check)+" "+repr(self._h))
            assert np.allclose(h_check, self._h)

        # self._T_left = self._x_min + (self._x_sep - self._x_min) * make_open_knots(p, self._n_left)  # regular open vector
        #
        # here I am changing the left knot vector on its right end to get truncated regular splines on the sub-domain boundary
        # but we should study the other option: defining the discontinuous space as a sum of disjoint open splines
        # assert abs(self._T_left[self._n_left] - self._x_sep) < 1e-10
        # for i in range(1, self._p+1):
        #     self._T_left[self._n_left+i] = self._x_sep + i*self._h

        # self._T_right = self._x_sep + (self._x_max - self._x_sep) * make_open_knots(p, self._n_right)
        # I am also changing the right knot vector on its left end -- same remark as above
        # assert abs(self._T_right[self._p] - self._x_sep) < 1e-10
        # for i in range(self._p):
        #     self._T_right[i] = self._x_sep + (i-self._p)*self._h

        self.grid = construct_grid_from_knots(p, self._n_smooth, self._T_smooth)
        assert len(self.grid) == self._N_cells + 1

        # spline coefs
        self.coefs_smooth = np.zeros(self._n_smooth, dtype=np.double)
        self.coefs_left = np.zeros(self._n_left, dtype=np.double)
        self.coefs_right = np.zeros(self._n_right, dtype=np.double)
        # self.coefs_disc = np.zeros(self._n_disc, dtype=np.double)

        self._use_macro_elem_duals = use_macro_elem_duals

        # Duality products
        self.duality_prods = np.zeros((self._n_smooth, self._n_smooth, self._N_cells), dtype=np.double)

        # -- construction of the dual functions --
        self._alpha = np.zeros(self._p+1)
        for i in range(self._p+1):
            self._alpha[i] = _my_gauss_quad(lambda x: self._q_pieces(i, x), 0, 1, n=self._p)
        # self._alpha[0] = 1.

        self._pre_I_left_B_products = None
        self._pre_I_right_B_products = None

        # compute a change of basis matrix (from q to r pieces)
        temp_mass_q = np.zeros((self._p + 1, self._p + 1))
        for i in range(self._p + 1):
            qi = lambda x: self._q_pieces(i, x)
            for j in range(i, self._p + 1):
                qj = lambda x: self._q_pieces(j, x)
                temp_mass_q[i, j] = _my_L2_prod(qi, qj, xmin=0, xmax=1)
            for j in range(i):
                # j<i so the j-th row is computed already
                temp_mass_q[i, j] = temp_mass_q[j, i]
        self._q_to_r_matrix = np.linalg.inv(temp_mass_q)

        if watch_your_steps:
            self._check_q_r_duality()
            self._check_ref_pw_spline_duality(n_checks=n_checks)
            self._check_pw_spline_duality(n_checks=n_checks)

        # for the macro-interior duals
        self._mu_coefs = None

        print("compute_macro_int_duals...")
        self._compute_macro_int_duals()

        if watch_your_steps:
            self._check_macro_element_duality()
            self._check_macro_elem_spline_duality(n_checks=n_checks)

        print("compute duality products...")
        self.compute_duality_prods()

        if watch_your_steps:
            self._check_duality_prods()

        print("Ok, construction done.")

    def _check_q_r_duality(self):
        print("## checking the duality of the polynomial basis...")
        prod_qr = np.zeros((self._p + 1, self._p + 1))
        for i in range(self._p + 1):
            for j in range(self._p + 1):
                prod_qr[i,j] = _my_L2_prod(lambda x: self._q_pieces(i, x), lambda x: self._r_pieces(j, x), xmin=0, xmax=1)
        if np.allclose(prod_qr, np.identity(self._p + 1), rtol=1e-9):
            print("PASSED")
        else:
            print("NOT passed, here is the duality matrix:")
            print(prod_qr)

    def _check_ref_pw_spline_duality(self, n_checks=5):
        print("## checking the duality of the ref pw pol dual basis...")
        check_xmin = -self._p - 10
        check_xmax = n_checks + 10
        singular_points = range(check_xmin, check_xmax+1)  # maybe less
        prods = np.zeros((n_checks, n_checks))
        for i in range(n_checks):
            for j in range(n_checks):
                prods[i, j] = _my_L2_prod(
                    lambda x: self._ref_conservative_dual_function(x-i),
                    lambda x: self._ref_b_spline(x-j),
                    sing_points=singular_points, xmin=check_xmin, xmax=check_xmax
                )
        if np.allclose(prods, np.identity(n_checks), rtol=1e-9):
            print("PASSED")
        else:
            print("NOT passed, here is the duality matrix:")
            print(prods)

    def _check_pw_spline_duality(self, n_checks=5):
        print("## checking the duality of the pw pol dual basis...")
        i_node_min = -self._p
        i_node_max = n_checks + 1
        nb_nodes = i_node_max - i_node_min + 1
        check_xmin = i_node_min * self._h
        check_xmax = i_node_max * self._h
        singular_points, h_aux = np.linspace(check_xmin, check_xmax, num=nb_nodes, endpoint=True, retstep=True)
        assert np.allclose(h_aux, self._h)

        prods = np.zeros((n_checks, n_checks))
        for i in range(n_checks):
            for j in range(n_checks):
                prods[i, j] = _my_L2_prod(
                    lambda x: self._pw_pol_dual_function(i, x),
                    lambda x: self._reg_spline(j, x),
                    sing_points=singular_points, xmin=check_xmin, xmax=check_xmax
                )
        if np.allclose(prods, np.identity(n_checks), rtol=1e-9):
            print("PASSED")
        else:
            print("NOT passed, here is the duality matrix:")
            print(prods)

    def _check_duality_prods(self, n_checks=5):
        print("## checking the duality products D_i,j,k...")

        prods = np.zeros((n_checks, n_checks))
        for i in range(n_checks):
            for j in range(n_checks):
                val = 0
                for k in range(self._N_cells):
                    val += self.duality_prods[i+self._p, j+self._p, k]
                prods[i, j] = val
        if np.allclose(prods, np.identity(n_checks), rtol=1e-9):
            print("PASSED")
        else:
            print("NOT passed, here is the duality matrix:")
            print(prods)

    def _check_macro_elem_spline_duality(self, n_checks=5):
        print("## checking the duality of the macro-element dual basis...")
        i_node_min = -self._p
        i_node_max = self._N_cells + 1
        nb_nodes = i_node_max - i_node_min + 1
        check_xmin = i_node_min * self._h
        check_xmax = i_node_max * self._h
        singular_points, h_aux = np.linspace(check_xmin, check_xmax, num=nb_nodes, endpoint=True, retstep=True)
        assert np.allclose(h_aux, self._h)

        prods = np.zeros((n_checks, n_checks))
        for i in range(n_checks):
            for j in range(n_checks):
                prods[i, j] = _my_L2_prod(
                    lambda x: self._macro_elem_dual_function(i, x),
                    lambda x: self._reg_spline(j, x), sing_points=singular_points, xmin=check_xmin, xmax=check_xmax
                )
        if np.allclose(prods, np.identity(n_checks), rtol=1e-9):
            print("PASSED")
        else:
            print("NOT passed, here is the duality matrix:")
            print(prods)

    def _check_macro_element_duality(self):
        print("## checking the duality of the macro-element basis functions...")
        pre_int_prods = np.zeros((self._m+1, self._m+1))
        int_prods = np.zeros((self._m+1, self._m+1))
        singular_points = range(-self._M_p, 2*self._M_p+1)  # maybe less
        for i in range(self._m+1):
            for j in range(self._m+1):
                pre_int_prods[i, j] = _my_L2_prod(
                    lambda x: self._pre_ref_macro_int_dual_function(i, x), lambda x: self._ref_macro_int_splines(j, x),
                    sing_points=singular_points, xmin=0, xmax=self._M_p
                )
                int_prods[i, j] = _my_L2_prod(
                    lambda x: self._ref_macro_int_dual_function(i, x), lambda x: self._ref_macro_int_splines(j, x),
                    sing_points=singular_points, xmin=0, xmax=self._M_p
                )
        print("1. macro-element duality check for the products : pre_int_duals * int_splines...")
        if np.allclose(pre_int_prods, np.identity(self._m + 1), rtol=1e-9):
            print("PASSED")
        else:
            print("NOT passed, here is the duality matrix:")
            print(pre_int_prods)

        print("2. macro-element duality check for the products : int_duals * int_splines...")
        if np.allclose(int_prods, np.identity(self._m + 1), rtol=1e-9):
            print("PASSED")
        else:
            print("NOT passed, here is the duality matrix:")
            print(int_prods)

        # check the B-B duality:
        bound_prods = np.zeros((self._p, self._p))
        for i in range(self._p):
            for j in range(self._p):
                # all the _ref_macro_bound_splines are supported in [-p,p]
                bound_prods[i, j] = _my_L2_prod(
                    lambda x: self._ref_macro_bound_dual_function(i, x), lambda x: self._ref_macro_bound_splines(j, x),
                    sing_points=singular_points, xmin=-self._p, xmax=self._p
                )
        print("3. macro-element duality check for the products : bound_duals * bound_splines...")
        if np.allclose(bound_prods, np.identity(self._p), rtol=1e-9):
            print("PASSED")
        else:
            print("NOT passed, here is the duality matrix:")
            print(bound_prods)

        # check the I-B duality:
        pre_int_bound_prods = np.zeros((self._m+1, self._p))
        int_bound_prods = np.zeros((self._m+1, self._p))
        bound_int_prods = np.zeros((self._p, self._m+1))
        for i in range(self._m+1):
            for j in range(self._p):
                # all the _ref_macro_bound_splines are supported in [-p,p]
                pre_int_bound_prods[i, j] = _my_L2_prod(
                    lambda x: self._pre_ref_macro_int_dual_function(i, x), lambda x: self._ref_macro_bound_splines(j, x),
                    sing_points=singular_points, xmin=-self._p, xmax=self._p
                )
                int_bound_prods[i, j] = _my_L2_prod(
                    lambda x: self._ref_macro_int_dual_function(i, x), lambda x: self._ref_macro_bound_splines(j, x),
                    sing_points=singular_points, xmin=-self._p, xmax=self._p
                )
                bound_int_prods[j, i] = _my_L2_prod(
                    lambda x: self._ref_macro_bound_dual_function(j, x), lambda x: self._ref_macro_int_splines(i, x),
                    sing_points=singular_points, xmin=-self._p, xmax=self._p
                )

        print("4. macro-element duality check for the products : bound_duals * int_splines")
        if np.allclose(bound_int_prods, np.zeros((self._p, self._m+1)), rtol=1e-9):
            print("PASSED")
        else:
            print("NOT passed, here is the duality matrix:")
            print(bound_int_prods)

        print("5. macro-element duality check for the products : int_duals * bound_splines")
        if np.allclose(int_bound_prods, np.zeros((self._m+1, self._p)), rtol=1e-9):
            print("PASSED")
        else:
            print("NOT passed, here is the duality matrix:")
            print(int_bound_prods)
            print(" -- note: the problem may come from the products : pre_int_duals * bound_splines (which should NOT be zero)")
            print(pre_int_bound_prods)

    def _ref_macro_int_splines(self, i, x):
        """
        the regular splines whose support [i, i+p+1] is inside the reference macro-element [0, m+p+1]
        """
        assert i in range(0, self._m+1)
        return self._ref_b_spline(x-i)

    def _ref_macro_bound_splines(self, i, x):
        """
        the regular splines whose open support ]i-p, i+1[ contains the (left) reference macro-vertex 0
        """
        assert i in range(0, self._p)
        return self._ref_b_spline(x-(i-self._p))

    def _ref_macro_int_aux_basis(self, i, x):
        assert i in range(0, self._m+1)
        # bernstein restricted to the reference macro-element, but could be any basis
        if 0 < x < self._M_p:
            return (x**i) * ((1-x)**(self._m-i))
        else:
            return 0

    def _compute_macro_int_duals(self):
        # step 1: compute the preliminary interior duals
        temp_interior_matrix = np.zeros((self._m+1,self._m+1), dtype=np.double)
        n_quad_nodes = int(np.ceil(0.5*(self._p+self._m+1)))  # so that 2n-1 >= p+m
        for i in range(self._m+1):
            for j in range(self._m+1):
                # use Gauss quadratures with n_nodes in each sub-interval [k, k+1] of [0, M_p]
                val = 0
                for k in range(self._M_p):
                    val += _my_gauss_quad(
                        lambda x: self._ref_macro_int_aux_basis(i, x)*self._ref_macro_int_splines(j, x),
                        k, k+1, n=n_quad_nodes
                    )
                temp_interior_matrix[i, j] = val
        if abs(np.linalg.det(temp_interior_matrix)) < 1e-8:
            print(" WARNING: temp_interior_matrix has a small determinant: "+repr(np.linalg.det(temp_interior_matrix)))
            print(" temp_interior_matrix = "+repr(temp_interior_matrix))
        self._mu_coefs = np.linalg.inv(temp_interior_matrix)

        # step 2: compute the correction factors:
        #   products of the form
        #       < pre_ref_dual_I(i, x), ref_boundary_spline(j, x) > corresponding to the left macro-vertex
        #   and
        #       < pre_ref_dual_I(i, x), ref_boundary_spline(j, x-M_p) > corresponding to the right macro-vertex
        self._pre_I_left_B_products = np.zeros((self._m+1, self._p), dtype=np.double)
        self._pre_I_right_B_products = np.zeros((self._m+1, self._p), dtype=np.double)
        for i in range(self._m+1):
            for j in range(self._p):
                # use Gauss quadratures with n_nodes in each sub-interval [k, k+1] of [0, M_p]
                left_prod = 0
                right_prod = 0
                for k in range(self._M_p):
                    left_prod += _my_gauss_quad(
                        lambda x: self._pre_ref_macro_int_dual_function(i, x) * self._ref_macro_bound_splines(j, x),
                        k, k+1, n=n_quad_nodes
                    )
                    right_prod += _my_gauss_quad(
                        lambda x: self._pre_ref_macro_int_dual_function(i, x) * self._ref_macro_bound_splines(j, x-self._M_p),
                        k, k+1, n=n_quad_nodes
                    )
                self._pre_I_left_B_products[i, j] = left_prod
                self._pre_I_right_B_products[i, j] = right_prod

    def _ref_macro_bound_dual_function(self, i, x):
        """
        dual functions of boundary type associated with the macro-vertex 0 in the reference macro-element [0, M_p]
        -- denoted psi_B in prospline code
        """
        assert i in range(self._p)
        # note: this dual function corresponds to the "alpha" (or "conservative") choice in the prospline notes
        return self._ref_conservative_dual_function(x-(i-self._p))

    def _pre_ref_macro_int_dual_function(self, i, x):
        """
        preliminary dual function of interior type on the reference macro-element
        -- denoted tilde_psi_I in prospline code
        """
        assert self._mu_coefs is not None
        assert i in range(self._m+1)
        val = 0
        for j in range(self._m+1):
            val += self._mu_coefs[i, j] * self._ref_macro_int_aux_basis(j, x)
        return val

    def _ref_macro_int_dual_function(self, i, x):
        """
        dual function of interior type associated with the reference macro-element  [0, M_p]
        -- denoted psi_I in prospline code
        """
        assert self._mu_coefs is not None
        assert self._pre_I_left_B_products is not None
        assert self._pre_I_right_B_products is not None
        assert i in range(self._m+1)
        val = self._pre_ref_macro_int_dual_function(i, x)
        for j in range(self._p):
            val -= self._pre_I_left_B_products[i, j] * self._ref_macro_bound_dual_function(j, x)
            val -= self._pre_I_right_B_products[i, j] * self._ref_macro_bound_dual_function(j, x-self._M_p)
        return val

    def _ref_b_spline(self, x):
        """
        the regular (cardinal) B-spline of degree p -- with support [0, p+1]
        """
        val = 0
        if 0 < x < self._p+1:
            reg_knots = range(self._p+2)
            b = BSpline.basis_element(reg_knots, extrapolate=False)
            val = b(x)
        return val

    def _reg_spline(self, i, x):
        """
        evaluate at x the i-th regular spline of degree p, on this grid
        its support is [(i-p)*h, (i+1)*h] with h = 1/N_cells
        and away from the domain boundaries we have reg_spline(i) == phi[i]
        """
        return self._ref_b_spline(x/self._h - (i-self._p))

    def _q_pieces(self, i, x):
        """
        restriction of the ref B-spline on the interval [i,i+1], and translated back to [0,1].
        for i = 0, .. p, they span a basis of P_p([0,1])
        """
        assert i in range(self._p+1)
        if 0 < x < 1:
            reg_knots = range(self._p+2)
            b = BSpline.basis_element(reg_knots, extrapolate=False)
            return b(x+i)
        else:
            return 0

    def _r_pieces(self, i, x):
        # note: this one is restricted to [0,1], unlike in the prospline code
        assert i in range(self._p+1)
        if 0 < x < 1:
            val = 0
            for j in range(self._p+1):
                val += self._q_to_r_matrix[i, j] * self._q_pieces(j, x)
            return val
        else:
            return 0

    def _ref_conservative_dual_function(self, x):
        """
        pw pol (degree p) dual function -- with support [0, p+1]
        """
        val = 0
        for l in range(self._p+1):
            val += self._alpha[l] * self._r_pieces(l, x-l)
        return val

    def _pw_pol_dual_function(self, i, x):
        """
        pw pol (degree p) dual function on the grid, dual to reg_spline(i, x)
        """
        return (1./self._h) * self._ref_conservative_dual_function(x/self._h-(i-self._p))

    def _macro_elem_dual_function(self, i, x):
        q = np.mod(i, self._M_p)
        ell_M_p = i - q
        if q < self._p:
            # dual function is of macro-boundary type, associated with macro vertex ell_M_p = ell * M_p
            val = self._ref_macro_bound_dual_function(q, x/self._h - ell_M_p)
        else :
            # dual function is of macro-interior type, associated with macro element [ell_M_p, ell_M_p + M_p]
            val = self._ref_macro_int_dual_function(q-self._p, x/self._h - ell_M_p)
        return val/self._h

    def compute_duality_prods(self):

        # u_quad, w_quad = gauss_legendre(self._p)  # gauss-legendre quadrature rule
        # k_quad = len(u_quad)
        # nb_element = 1

        # note: we may only need these products close to the inner subdomain boundaries

        # computing quadratures on the cells [k,k+1]*h, product is polynomial of degree <= p + max( p, m ) <= 2*p
        n_quad_nodes = self._p+1  # so that 2n-1 >= 2*p
        for k in range(self._N_cells):

            # cell = [self.grid[k], self.grid[k+1]]
            # points, weights = construct_quadrature_grid(nb_element, k_quad, u_quad, w_quad, cell)

            # d = 1                     # number of derivatives
            # basis = eval_on_grid_splines_ders(p, n, k, d, T, points)

            # support of phi_i: [i-p, i+1]*h, so
            i_min = k
            i_max = k + self._p
            for i in range(i_min, i_max+1):
                assert i in range(self._n_smooth)

                # phi_i = lambda x: self._reg_spline(i, x)
                # phi_i_on_quad_points = list(map(phi_i, points))

                if self._use_macro_elem_duals:
                    # support of tilde_phi_j:
                    #   [Ml-p, M(k+1)+p]*h   if j in {Ml+p, .. Ml+M-1}  (in this case tilde_phi_j is a macro_interior dual function)
                    #   [j-p, j+1]           if j in {Ml, .. Ml+p-1}    (in this case tilde_phi_j is a macro_boundary dual function)
                    # so, we can take all the j's associated to the macro-element [Ml,Ml+1]*h and its neighbors
                    ell = int(k/self._M_p)     # int division
                    assert ell*self._M_p <= k < (ell+1)*self._M_p
                    j_min = max((ell-1)*self._M_p,   0)
                    j_max = min((ell+2)*self._M_p-1, self._n_smooth-1)
                else:
                    j_min = k
                    j_max = k+self._p
                for j in range(j_min, j_max+1):
                    assert j in range(self._n_smooth)

                    def tilde_phi_j(x):
                        if self._use_macro_elem_duals:
                            return self._macro_elem_dual_function(j, x)
                        else:
                            return self._pw_pol_dual_function(j, x)

                    self.duality_prods[i, j, k] = _my_gauss_quad(
                        lambda x: self._reg_spline(i, x) * tilde_phi_j(x),
                        self.grid[k], self.grid[k+1], n=n_quad_nodes
                    )

    def smooth_projection(self):
        """
        computes the coefs of a smooth projection in the smooth spline space, from the coefs in the discontinuous space
        """
        self.coefs_smooth[:] = 0
        # project the left function
        delta = 0  # global position of the 1st node in LEFT domain
        for i in range(self._n_left):
            assert 0 <= i+delta < self._n_smooth
            if i+1 <= self._N_cells_left:
                # then support of phi_left[i] is in left domain, so phi_left[i] == phi[i+delta]
                # and coefs_left[i] only contributes to coefs_smooth[i+delta]
                j = i+delta
                self.coefs_smooth[j] += self.coefs_left[i]
            else:
                for j in range(self._n_smooth):     # todo: localize
                    pi_ij = 0
                    for k in range(self._N_cells_left):     # todo: localize
                        assert 0 <= k+delta < self._N_cells
                        pi_ij += self.duality_prods[i+delta, j, k+delta]
                    self.coefs_smooth[j] += pi_ij * self.coefs_left[i]

        # project the right function
        delta = self._N_cells_left  # global position of the 1st node in RIGHT domain
        for i in range(self._n_right):
            assert 0 <= i+delta < self._n_smooth
            if i-self._p >= 0:
                # then support of phi_right[i] is in right domain, so phi_right[i] == phi[i+delta]
                # and coefs_right[i] only contributes to coefs_smooth[i+delta]
                j = i+delta
                self.coefs_smooth[j] += self.coefs_right[i]
            else:
                for j in range(self._n_smooth):     # todo: localize
                    pi_ij = 0
                    for k in range(self._N_cells_right):     # todo: localize
                        assert 0 <= k+delta < self._N_cells
                        pi_ij += self.duality_prods[i+delta, j, k+delta]
                    self.coefs_smooth[j] += pi_ij * self.coefs_right[i]

    def l2_project_on_sub_domain(self, f, sub='left'):
        """
        L2 projection (??) derived from test_projector_1d by ARA
        """
        if sub == 'left':
            n_sub = self._n_left
            T_sub = self._T_left
        else:
            assert sub == 'right'
            n_sub = self._n_right
            T_sub = self._T_right
        print("L2 proj on subdomain "+sub)
        print("n_sub = "+repr(n_sub))
        print("T_sub = "+repr(T_sub))
        mass_0, mass_1 = mass_matrices(self._p, n_sub, T_sub)   # works with a knot vector that is not open ??
        contribution = Contribution(self._p, n_sub, T_sub)
        f_l2 = solve(mass_1, contribution(f))
        tck = get_tck('L2', self._p, n_sub, T_sub, f_l2)    # H1, L2 ??
        assert len(tck[1]) == n_sub
        if sub == 'left':
            self.coefs_left = tck[1]
        else:
            assert sub == 'right'
            self.coefs_right = tck[1]

    def eval_discontinuous_spline(self, x):
        if x <= self._x_sep:
            tck = [self._T_left, self.coefs_left, self._p]
        else:
            tck = [self._T_right, self.coefs_right, self._p]
        return splev(x, tck)

    def eval_continuous_spline(self, x):
        tck = [self._T_smooth, self.coefs_smooth, self._p]
        return splev(x, tck)

    def plot_spline(
            self,
            filename,
            spline_type='continuous',
            N_points=100,
            ltype='-',
            legend=None,
            title=None,
            legend_loc='lower left'):

        vis_grid, vis_h = np.linspace(self._x_min, self._x_max, N_points, endpoint=True, retstep=True)
        if spline_type == 'continuous':
            vals = [self.eval_continuous_spline(xi) for xi in vis_grid]
        else:
            assert spline_type == 'discontinuous'
            vals = [self.eval_discontinuous_spline(xi) for xi in vis_grid]
        fig = plt.figure()
        plt.clf()
        image = plt.plot(vis_grid, vals, ltype, color='b', label=spline_type)
        if title is not None:
            plt.title(title)
        plt.legend(loc=legend_loc)
        fig.savefig(filename)
        plt.clf()


# local utilities, maybe discard

def _my_L2_prod(f, g, xmin=0, xmax=1, sing_points=None, eps=None):
    """
    L2 product of f and g, using a scipy quadrature

    sing_points:
        sequence of break points in the bounded integration interval
        where local difficulties of the integrand may occur (e.g., singularities, discontinuities)
    eps:
        tolerance
    """
    if sing_points is not None:
        valid_points = list(map(lambda x: min(xmax, max(xmin, x)), sing_points))
    else:
        valid_points = None

    if eps is not None:
        epsabs = eps
        epsrel = eps
    else:
        epsabs = 1e-10
        epsrel = 1e-10
    return quad(lambda x: f(x)*g(x), xmin, xmax, points=valid_points, epsabs=epsabs, epsrel=epsrel, limit=100)[0]


def _my_gauss_quad(func, a, b, n=5):
    """
    from scipy.integrate.fixed_quad
    ****
    Compute a definite integral using fixed-order Gaussian quadrature.
    Integrate `func` from `a` to `b` using Gaussian quadrature of
    order `n`.
    Parameters
    ----------
    func : callable
        A Python function or method to integrate (must accept vector inputs).
        If integrating a vector-valued function, the returned array must have
        shape ``(..., len(x))``.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    n : int, optional
        Order of quadrature integration. Default is 5.
    Returns
    -------
    val : float
        Gaussian quadrature approximation to the integral
    """
    x, w = p_roots(n)
    x = np.real(x)
    if np.isinf(a) or np.isinf(b):
        raise ValueError("Gaussian quadrature is only available for "
                         "finite limits.")
    y = (b-a)*(x+1)/2.0 + a
    return (b-a)/2.0 * np.sum([w[i]*func(y[i]) for i in range(len(y))])
