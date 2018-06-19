# -*- coding: UTF-8 -*-

__author__ = 'campos'

import numpy as np

from scipy.integrate import quad
from scipy.interpolate import BSpline
from scipy.interpolate import splev
from scipy.special import comb
from scipy.integrate import quadrature
# from scipy.special.orthogonal import p_roots
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import splu

import matplotlib
matplotlib.use('Agg')   # backend configuration: high quality PNG images using the Anti-Grain Geometry engine
import matplotlib.pyplot as plt

from spl.core.interface import make_open_knots
from spl.core.interface import construct_grid_from_knots
from spl.core.interface import construct_quadrature_grid
from spl.utilities.quadratures import gauss_legendre

from spl.utilities.integrate import Integral
from spl.utilities.integrate import Interpolation
from spl.utilities.integrate import Contribution

from spl.feec.utilities import interpolation_matrices
from spl.feec.utilities import get_tck
from spl.feec.utilities import mass_matrices
from spl.feec.utilities import scaling_matrix


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

class trialCongaDDM:
    """
    trial class for a Conga method with 2 subdomains in 1D: [0, 0.5] and [0.5, 1]

    The smooth space is a std spline space on the whole domain, with open knots.
    The discontinous space consists of two smooth spline spaces, one for each subdomain.
    """

    def __init__(self,
                 p,
                 m,
                 N_cells_sub,
                 watch_my_steps=False,
                 n_checks=5,
                 use_macro_elem_duals=False,
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

        if use_macro_elem_duals and (not np.mod(N_cells_sub,self._M_p) == 0):
            raise ValueError('Wrong value for N_cells_sub, must be a multiple of m+p+1 for a macro-element dual basis')

        self._p = p
        self._N_cells_left = N_cells_sub
        self._N_cells_right = N_cells_sub
        self._N_cells = self._N_cells_left + self._N_cells_right

        self._N_macro_cells = self._N_cells // self._M_p

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

        # open knot vectors for the three smooth spaces
        self._T_smooth = self._x_min + (self._x_max - self._x_min) * make_open_knots(self._p, self._n_smooth)
        self._T_left = self._x_min + (self._x_sep - self._x_min) * make_open_knots(self._p, self._n_left)
        self._T_right = self._x_sep + (self._x_max - self._x_sep) * make_open_knots(self._p, self._n_right)

        self.grid = construct_grid_from_knots(p, self._n_smooth, self._T_smooth)
        assert len(self.grid) == self._N_cells + 1

        # spline coefs
        self.coefs_smooth = np.zeros(self._n_smooth, dtype=np.double)
        self.coefs_left = np.zeros(self._n_left, dtype=np.double)
        self.coefs_right = np.zeros(self._n_right, dtype=np.double)

        # flag
        self._use_macro_elem_duals = use_macro_elem_duals

        # Duality products
        self.duality_prods = np.zeros((self._n_smooth, self._n_smooth, self._N_cells), dtype=np.double)

        self._tilde_phi_P_coefs = [np.zeros((self._p + 1, self._p + 1)) for k in range(self._N_cells)]  # todo: try sequences of None
        self._tilde_phi_M_aux_coefs = [None for ell in range(self._N_macro_cells)]
        # self._tilde_phi_M_aux_coefs = [np.zeros((self._m + 1, self._m + 1)) for ell in range(self._N_macro_cells)]
        self._left_correction_products_tilde_phi_M_aux = [np.zeros((self._m+1, self._p)) for ell in range(self._N_macro_cells)]
        self._right_correction_products_tilde_phi_M_aux = [np.zeros((self._m+1, self._p)) for ell in range(self._N_macro_cells)]

        # -- construction of the P dual basis ---

        # change-of-basis matrices for each I_k
        temp_matrix = np.zeros((self._p + 1, self._p + 1))
        for k in range(self._N_cells):
            temp_matrix[:,:] = 0
            for a in range(self._p + 1):
                bern_ak = lambda x: self._bernstein_P(a, k, x)
                for b in range(self._p + 1):
                    j = k + b
                    phi_jk = lambda x: self._phi(j, x)  # could be phi_pieces(j,k,x) but we only evaluate it on I_k so its the same
                    temp_matrix[a, b] = _my_L2_prod(bern_ak, phi_jk, xmin=self._T_smooth[k+p], xmax=self._T_smooth[k+p+1])
            self._tilde_phi_P_coefs[k] = np.linalg.inv(temp_matrix)

        # alpha coefficients
        self._alpha = np.zeros((self._n_smooth, self._N_cells))
        int_phi = np.zeros(self._n_smooth)
        for i in range(self._n_smooth):
            for a in range(self._p+1):
                int_phi[i] += quadrature(
                    lambda x: self._phi(i, x),
                    self._T_smooth[i+a],
                    self._T_smooth[i+a+1],
                    maxiter=self._p+1,
                    vec_func=False,
                )[0]
            print("i = ", i, "  --  int_phi[i] = ", int_phi[i])

        for k in range(self._N_cells):
            for a in range(self._p+1):
                i = k + a
                assert i < self._n_smooth
                self._alpha[i,k] = quadrature(
                    lambda x: self._phi(i, x),
                    self._T_smooth[k+p],
                    self._T_smooth[k+p+1],
                    maxiter=self._p+1,
                    vec_func=False,
                )[0]/int_phi[i]

        if self._use_macro_elem_duals:
            M = self._M_p
            m = self._m

            # change-of-basis coefs for the macro element dual functions
            temp_matrix = np.zeros((m + 1, m + 1))
            for ell in range(self._N_macro_cells):
                temp_matrix[:,:] = 0
                for a in range(m + 1):
                    bern_a_ell = lambda x: self._bernstein_M(a, ell, x)
                    for b in range(m + 1):
                        j = self._global_index_of_macro_element_dof(ell,b)
                        phi_jk = lambda x: self._phi(j, x)
                        for k in range(ell*M, (ell+1)*M):
                            temp_matrix[a, b] += quadrature(
                                lambda x: bern_a_ell(x) * phi_jk(x),
                                self._T_smooth[k + p],
                                self._T_smooth[k+1 + p],
                                maxiter=2*m,
                                vec_func=False,
                            )[0]
                self._tilde_phi_M_aux_coefs[ell] = np.linalg.inv(temp_matrix)

            if 0:
                print("check -- MM ")
                grid = construct_grid_from_knots(self._p, self._n_smooth, self._T_smooth)
                ell = 0
                coef_check = np.zeros((m+1,m+1))
                for a in range(m + 1):
                    i = self._global_index_of_macro_element_dof(ell, a)
                    for b in range(m + 1):
                        j = self._global_index_of_macro_element_dof(ell,b)
                        coef_check[a,b] = _my_L2_prod(
                            lambda x:self._tilde_phi_M_aux(i, x),
                            lambda x:self._phi(j, x),
                            sing_points=grid,
                        )
                print(coef_check)
                print('check done -- 847876474')
                exit()


            # correction coefs for the macro element dual functions
            for ell in range(self._N_macro_cells):
                for a in range(m + 1):
                    i = self._global_index_of_macro_element_dof(ell, a)
                    tilde_phi_M_aux_i = lambda x: self._tilde_phi_M_aux(i, x)
                    for b in range(p):

                        # left macro-vertex:
                        j = self._global_index_of_macro_vertex_dof(ell, b)
                        phi_j =  lambda x: self._phi(j, x)
                        self._left_correction_products_tilde_phi_M_aux[ell][a,b] = _my_L2_prod(
                            tilde_phi_M_aux_i,
                            phi_j,
                            xmin=self._T_smooth[ell*M + p],
                            xmax=self._T_smooth[(ell+1)*M + p]
                        )

                        # right macro-vertex:
                        j = self._global_index_of_macro_vertex_dof(ell+1, b)
                        phi_j =  lambda x: self._phi(j, x)
                        self._right_correction_products_tilde_phi_M_aux[ell][a,b] = _my_L2_prod(
                            tilde_phi_M_aux_i,
                            phi_j,
                            xmin=self._T_smooth[ell*M + p],
                            xmax=self._T_smooth[(ell+1)*M + p]
                        )

        #     self._pre_I_left_B_products = None
        #     self._pre_I_right_B_products = None
        #
        #     # compute a change of basis matrix (from q to r pieces)
        #     temp_mass_q = np.zeros((self._p + 1, self._p + 1))
        #     for i in range(self._p + 1):
        #         qi = lambda x: self._q_pieces(i, x)
        #         for j in range(i, self._p + 1):
        #             qj = lambda x: self._q_pieces(j, x)
        #             temp_mass_q[i, j] = _my_L2_prod(qi, qj, xmin=0, xmax=1)
        #         for j in range(i):
        #             # j<i so the j-th row is computed already
        #             temp_mass_q[i, j] = temp_mass_q[j, i]
        #     self._q_to_r_matrix = np.linalg.inv(temp_mass_q)
        #
        #     if watch_my_steps:
        #         self._check_q_r_duality()
        #         self._check_ref_pw_spline_duality(n_checks=n_checks)
        #         self._check_pw_spline_duality(n_checks=n_checks)
        #
        #     # for the macro-interior duals
        #     self._mu_coefs = None
        #
        #     print("compute_macro_int_duals...")
        #     self._compute_macro_int_duals()
        #
        #     if watch_my_steps:
        #         self._check_macro_element_duality()
        #         self._check_macro_elem_spline_duality(n_checks=n_checks)
        #
        # if 0:
        #     print("compute duality products...")
        #     self.compute_duality_prods()
        #
        # if watch_my_steps:
        #     self._check_duality_prods()

        print("Ok, construction done, n_dofs (smooth space) = ", self._n_smooth)

    # def _check_q_r_duality(self):
    #     print("## checking the duality of the polynomial basis...")
    #     prod_qr = np.zeros((self._p + 1, self._p + 1))
    #     for i in range(self._p + 1):
    #         for j in range(self._p + 1):
    #             prod_qr[i,j] = _my_L2_prod(lambda x: self._q_pieces(i, x), lambda x: self._r_pieces(j, x), xmin=0, xmax=1)
    #     if np.allclose(prod_qr, np.identity(self._p + 1), rtol=1e-9):
    #         print("PASSED")
    #     else:
    #         print("NOT passed, here is the duality matrix:")
    #         print(prod_qr)
    #
    # def _check_ref_pw_spline_duality(self, n_checks=5):
    #     print("## checking the duality of the ref pw pol dual basis...")
    #     check_xmin = -self._p - 10
    #     check_xmax = n_checks + 10
    #     singular_points = range(check_xmin, check_xmax+1)  # maybe less
    #     prods = np.zeros((n_checks, n_checks))
    #     for i in range(n_checks):
    #         for j in range(n_checks):
    #             prods[i, j] = _my_L2_prod(
    #                 lambda x: self._ref_conservative_dual_function(x-i),
    #                 lambda x: self._ref_b_spline(x-j),
    #                 sing_points=singular_points, xmin=check_xmin, xmax=check_xmax
    #             )
    #     if np.allclose(prods, np.identity(n_checks), rtol=1e-9):
    #         print("PASSED")
    #     else:
    #         print("NOT passed, here is the duality matrix:")
    #         print(prods)
    #
    # def _check_pw_spline_duality(self, n_checks=5):
    #     print("## checking the duality of the pw pol dual basis...")
    #     i_node_min = -self._p
    #     i_node_max = n_checks + 1
    #     nb_nodes = i_node_max - i_node_min + 1
    #     check_xmin = i_node_min * self._h
    #     check_xmax = i_node_max * self._h
    #     singular_points, h_aux = np.linspace(check_xmin, check_xmax, num=nb_nodes, endpoint=True, retstep=True)
    #     assert np.allclose(h_aux, self._h)
    #
    #     prods = np.zeros((n_checks, n_checks))
    #     for i in range(n_checks):
    #         for j in range(n_checks):
    #             prods[i, j] = _my_L2_prod(
    #                 lambda x: self._pw_pol_dual_function(i, x),
    #                 lambda x: self._reg_spline(j, x),
    #                 sing_points=singular_points, xmin=check_xmin, xmax=check_xmax
    #             )
    #     if np.allclose(prods, np.identity(n_checks), rtol=1e-9):
    #         print("PASSED")
    #     else:
    #         print("NOT passed, here is the duality matrix:")
    #         print(prods)
    #
    # def _check_duality_prods(self, n_checks=5):
    #     print("## checking the duality products D_i,j,k...")
    #
    #     prods = np.zeros((n_checks, n_checks))
    #     for i in range(n_checks):
    #         for j in range(n_checks):
    #             val = 0
    #             for k in range(self._N_cells):
    #                 val += self.duality_prods[i+self._p, j+self._p, k]
    #             prods[i, j] = val
    #     if np.allclose(prods, np.identity(n_checks), rtol=1e-9):
    #         print("PASSED")
    #     else:
    #         print("NOT passed, here is the duality matrix:")
    #         print(prods)
    #
    # def _check_macro_elem_spline_duality(self, n_checks=5):
    #     print("## checking the duality of the macro-element dual basis...")
    #     i_node_min = -self._p
    #     i_node_max = self._N_cells + 1
    #     nb_nodes = i_node_max - i_node_min + 1
    #     check_xmin = i_node_min * self._h
    #     check_xmax = i_node_max * self._h
    #     singular_points, h_aux = np.linspace(check_xmin, check_xmax, num=nb_nodes, endpoint=True, retstep=True)
    #     assert np.allclose(h_aux, self._h)
    #
    #     prods = np.zeros((n_checks, n_checks))
    #     for i in range(n_checks):
    #         for j in range(n_checks):
    #             prods[i, j] = _my_L2_prod(
    #                 lambda x: self._macro_elem_dual_function(i, x),
    #                 lambda x: self._reg_spline(j, x), sing_points=singular_points, xmin=check_xmin, xmax=check_xmax
    #             )
    #     if np.allclose(prods, np.identity(n_checks), rtol=1e-9):
    #         print("PASSED")
    #     else:
    #         print("NOT passed, here is the duality matrix:")
    #         print(prods)
    #
    # def _check_macro_element_duality(self):
    #     print("## checking the duality of the macro-element basis functions...")
    #     pre_int_prods = np.zeros((self._m+1, self._m+1))
    #     int_prods = np.zeros((self._m+1, self._m+1))
    #     singular_points = range(-self._M_p, 2*self._M_p+1)  # maybe less
    #     for i in range(self._m+1):
    #         for j in range(self._m+1):
    #             pre_int_prods[i, j] = _my_L2_prod(
    #                 lambda x: self._pre_ref_macro_int_dual_function(i, x), lambda x: self._ref_macro_int_splines(j, x),
    #                 sing_points=singular_points, xmin=0, xmax=self._M_p
    #             )
    #             int_prods[i, j] = _my_L2_prod(
    #                 lambda x: self._ref_macro_int_dual_function(i, x), lambda x: self._ref_macro_int_splines(j, x),
    #                 sing_points=singular_points, xmin=0, xmax=self._M_p
    #             )
    #     print("1. macro-element duality check for the products : pre_int_duals * int_splines...")
    #     if np.allclose(pre_int_prods, np.identity(self._m + 1), rtol=1e-9):
    #         print("PASSED")
    #     else:
    #         print("NOT passed, here is the duality matrix:")
    #         print(pre_int_prods)
    #
    #     print("2. macro-element duality check for the products : int_duals * int_splines...")
    #     if np.allclose(int_prods, np.identity(self._m + 1), rtol=1e-9):
    #         print("PASSED")
    #     else:
    #         print("NOT passed, here is the duality matrix:")
    #         print(int_prods)
    #
    #     # check the B-B duality:
    #     bound_prods = np.zeros((self._p, self._p))
    #     for i in range(self._p):
    #         for j in range(self._p):
    #             # all the _ref_macro_bound_splines are supported in [-p,p]
    #             bound_prods[i, j] = _my_L2_prod(
    #                 lambda x: self._ref_macro_bound_dual_function(i, x), lambda x: self._ref_macro_bound_splines(j, x),
    #                 sing_points=singular_points, xmin=-self._p, xmax=self._p
    #             )
    #     print("3. macro-element duality check for the products : bound_duals * bound_splines...")
    #     if np.allclose(bound_prods, np.identity(self._p), rtol=1e-9):
    #         print("PASSED")
    #     else:
    #         print("NOT passed, here is the duality matrix:")
    #         print(bound_prods)
    #
    #     # check the I-B duality:
    #     pre_int_bound_prods = np.zeros((self._m+1, self._p))
    #     int_bound_prods = np.zeros((self._m+1, self._p))
    #     bound_int_prods = np.zeros((self._p, self._m+1))
    #     for i in range(self._m+1):
    #         for j in range(self._p):
    #             # all the _ref_macro_bound_splines are supported in [-p,p]
    #             pre_int_bound_prods[i, j] = _my_L2_prod(
    #                 lambda x: self._pre_ref_macro_int_dual_function(i, x), lambda x: self._ref_macro_bound_splines(j, x),
    #                 sing_points=singular_points, xmin=-self._p, xmax=self._p
    #             )
    #             int_bound_prods[i, j] = _my_L2_prod(
    #                 lambda x: self._ref_macro_int_dual_function(i, x), lambda x: self._ref_macro_bound_splines(j, x),
    #                 sing_points=singular_points, xmin=-self._p, xmax=self._p
    #             )
    #             bound_int_prods[j, i] = _my_L2_prod(
    #                 lambda x: self._ref_macro_bound_dual_function(j, x), lambda x: self._ref_macro_int_splines(i, x),
    #                 sing_points=singular_points, xmin=-self._p, xmax=self._p
    #             )
    #
    #     print("4. macro-element duality check for the products : bound_duals * int_splines")
    #     if np.allclose(bound_int_prods, np.zeros((self._p, self._m+1)), rtol=1e-9):
    #         print("PASSED")
    #     else:
    #         print("NOT passed, here is the duality matrix:")
    #         print(bound_int_prods)
    #
    #     print("5. macro-element duality check for the products : int_duals * bound_splines")
    #     if np.allclose(int_bound_prods, np.zeros((self._m+1, self._p)), rtol=1e-9):
    #         print("PASSED")
    #     else:
    #         print("NOT passed, here is the duality matrix:")
    #         print(int_bound_prods)
    #         print(" -- note: the problem may come from the products : pre_int_duals * bound_splines (which should NOT be zero)")
    #         print(pre_int_bound_prods)


    # -- indices of macro elements, vertices and associated dofs --

    def dof_index_is_macro_vertex(self, i):
        return np.mod(i,self._M_p) < self._p

    def dof_index_is_macro_element(self, i):
        return not self.dof_index_is_macro_vertex(i)

    def macro_vertex_index_of_dof(self, i):
        assert self.dof_index_is_macro_vertex(i)
        ell = i // self._M_p
        assert 0 <= i - ell * self._M_p < self._p
        return ell

    def macro_element_index_of_dof(self, i):
        assert self.dof_index_is_macro_element(i)
        ell = i // self._M_p
        assert self._p <= i - ell * self._M_p < self._p + self._m + 1
        return ell

    def dof_indices_of_macro_vertex(self, ell):
        return [ell*self._M_p + a for a in range(self._p)]

    def dof_indices_of_macro_element(self, ell):
        return [ell*self._M_p + self._p + a for a in range(self._m+1)]

    def _local_index_of_macro_vertex_dof(self, i, ell):
        assert 0 <= i < self._n_smooth
        assert 0 <= ell <= self._N_macro_cells
        a = i - ell*self._M_p
        assert 0 <= a < self._p
        return a

    def _local_index_of_macro_element_dof(self, i, ell):
        assert 0 <= i < self._n_smooth
        assert 0 <= ell < self._N_macro_cells
        a = i - ell*self._M_p - self._p
        assert 0 <= a <= self._m
        return a

    def _global_index_of_macro_vertex_dof(self, ell, a):
        assert 0 <= ell <= self._N_macro_cells
        assert 0 <= a < self._p
        i = ell*self._M_p + a
        assert 0 <= i < self._n_smooth
        return i

    def _global_index_of_macro_element_dof(self, ell, a):
        assert 0 <= ell < self._N_macro_cells
        assert 0 <= a <= self._m
        i = ell*self._M_p + self._p + a
        assert 0 <= i < self._n_smooth
        return i

    # def _ref_macro_int_splines(self, i, x):
    #     """
    #     the regular splines whose support [i, i+p+1] is inside the reference macro-element [0, m+p+1]
    #     """
    #     assert i in range(0, self._m+1)
    #     return self._ref_b_spline(x-i)

    # def _ref_macro_bound_splines(self, i, x):
    #     """
    #     the regular splines whose open support ]i-p, i+1[ contains the (left) reference macro-vertex 0
    #     """
    #     assert i in range(0, self._p)
    #     return self._ref_b_spline(x-(i-self._p))

    # def _ref_macro_int_aux_basis(self, i, x):
    #     assert i in range(0, self._m+1)
    #     # bernstein restricted to the reference macro-element, but could be any basis
    #     if 0 < x < self._M_p:
    #         return (x**i) * ((1-x)**(self._m-i))
    #     else:
    #         return 0

    # def _compute_macro_int_duals(self):
    #     # step 1: compute the preliminary interior duals
    #     temp_interior_matrix = np.zeros((self._m+1,self._m+1), dtype=np.double)
    #     n_quad_nodes = int(np.ceil(0.5*(self._p+self._m+1)))  # so that 2n-1 >= p+m
    #     for i in range(self._m+1):
    #         for j in range(self._m+1):
    #             # use Gauss quadratures with n_nodes in each sub-interval [k, k+1] of [0, M_p]
    #             val = 0
    #             for k in range(self._M_p):
    #                 val += quadrature(
    #                     lambda x: self._ref_macro_int_aux_basis(i, x)*self._ref_macro_int_splines(j, x),
    #                     k, k+1, maxiter=n_quad_nodes,
    #                     vec_func=False,
    #                 )[0]
    #             temp_interior_matrix[i, j] = val
    #     if abs(np.linalg.det(temp_interior_matrix)) < 1e-8:
    #         print(" WARNING: temp_interior_matrix has a small determinant: "+repr(np.linalg.det(temp_interior_matrix)))
    #         print(" temp_interior_matrix = "+repr(temp_interior_matrix))
    #     self._mu_coefs = np.linalg.inv(temp_interior_matrix)
    #
    #     # step 2: compute the correction factors:
    #     #   products of the form
    #     #       < pre_ref_dual_I(i, x), ref_boundary_spline(j, x) > corresponding to the left macro-vertex
    #     #   and
    #     #       < pre_ref_dual_I(i, x), ref_boundary_spline(j, x-M_p) > corresponding to the right macro-vertex
    #     self._pre_I_left_B_products = np.zeros((self._m+1, self._p), dtype=np.double)
    #     self._pre_I_right_B_products = np.zeros((self._m+1, self._p), dtype=np.double)
    #     for i in range(self._m+1):
    #         for j in range(self._p):
    #             # use Gauss quadratures with n_nodes in each sub-interval [k, k+1] of [0, M_p]
    #             left_prod = 0
    #             right_prod = 0
    #             for k in range(self._M_p):
    #                 left_prod += quadrature(
    #                     lambda x: self._pre_ref_macro_int_dual_function(i, x) * self._ref_macro_bound_splines(j, x),
    #                     k, k+1, maxiter=n_quad_nodes, vec_func=False,
    #                 )[0]
    #                 right_prod += quadrature(
    #                     lambda x: self._pre_ref_macro_int_dual_function(i, x) * self._ref_macro_bound_splines(j, x-self._M_p),
    #                     k, k+1, maxiter=n_quad_nodes, vec_func=False,
    #                 )[0]
    #             self._pre_I_left_B_products[i, j] = left_prod
    #             self._pre_I_right_B_products[i, j] = right_prod

    # def _ref_macro_bound_dual_function(self, i, x):
    #     """
    #     dual functions of boundary type associated with the macro-vertex 0 in the reference macro-element [0, M_p]
    #     -- denoted psi_B in prospline code
    #     """
    #     assert i in range(self._p)
    #     # note: this dual function corresponds to the "alpha" (or "conservative") choice in the prospline notes
    #     return self._ref_conservative_dual_function(x-(i-self._p))

    # def _pre_ref_macro_int_dual_function(self, i, x):
    #     """
    #     preliminary dual function of interior type on the reference macro-element
    #     -- denoted tilde_psi_I in prospline code
    #     """
    #     assert self._mu_coefs is not None
    #     assert i in range(self._m+1)
    #     val = 0
    #     for j in range(self._m+1):
    #         val += self._mu_coefs[i, j] * self._ref_macro_int_aux_basis(j, x)
    #     return val

    # def _ref_macro_int_dual_function(self, i, x):
    #     """
    #     dual function of interior type associated with the reference macro-element  [0, M_p]
    #     -- denoted psi_I in prospline code
    #     """
    #     assert self._mu_coefs is not None
    #     assert self._pre_I_left_B_products is not None
    #     assert self._pre_I_right_B_products is not None
    #     assert i in range(self._m+1)
    #     val = self._pre_ref_macro_int_dual_function(i, x)
    #     for j in range(self._p):
    #         val -= self._pre_I_left_B_products[i, j] * self._ref_macro_bound_dual_function(j, x)
    #         val -= self._pre_I_right_B_products[i, j] * self._ref_macro_bound_dual_function(j, x-self._M_p)
    #     return val

    # def _ref_b_spline(self, x):
    #     """
    #     the regular (cardinal) B-spline of degree p -- with support [0, p+1]
    #     """
    #     val = 0
    #     if 0 < x < self._p+1:
    #         reg_knots = range(self._p+2)
    #         b = BSpline.basis_element(reg_knots, extrapolate=False)
    #         val = b(x)
    #     return val

    # def _reg_spline(self, i, x):
    #     """
    #     evaluate at x the i-th regular spline of degree p, on this grid
    #     its support is [(i-p)*h, (i+1)*h] with h = 1/N_cells
    #     and away from the domain boundaries we have reg_spline(i) == phi[i]
    #     """
    #     return self._ref_b_spline(x/self._h - (i-self._p))
    #
    # def _q_pieces(self, i, x):
    #     """
    #     restriction of the ref B-spline on the interval [i,i+1], and translated back to [0,1].
    #     for i = 0, .. p, they span a basis of P_p([0,1])
    #     """
    #     assert i in range(self._p+1)
    #     if 0 < x < 1:
    #         reg_knots = range(self._p+2)
    #         b = BSpline.basis_element(reg_knots, extrapolate=False)
    #         return b(x+i)
    #     else:
    #         return 0

    def _bernstein_P(self, a, k, x):
        """
        NEW
        a-th Bernstein polynomial of degree p on the interval I_k = [t_{k+p},t_{k+p+1}] -- else, 0
        """
        p = self._p
        assert a in range(p+1)
        t0 = self._T_smooth[k+p]
        t1 = self._T_smooth[k+p+1]
        if t0 <= x <= t1:
            t = (x-t0)/(t1-t0)
            return comb(p, a) * t**a * (1 - t)**(p - a)
        else:
            return 0

    def _bernstein_M(self, a, ell, x):
        """
        NEW
        a-th Bernstein polynomial of degree m (the degree of preserved moments)
        on the macro-element hat I_k = [t_{ell*M+p},t_{(ell+1)*M+p}] -- else, 0
        """
        p = self._p
        m = self._m
        assert a in range(m+1)
        t0 = self._T_smooth[ell*self._M_p+p]    # todo: would be clearer with grid[ell*self._M_p] ...
        t1 = self._T_smooth[(ell+1)*self._M_p+p]
        if t0 <= x <= t1:
            t = (x-t0)/(t1-t0)
            return comb(m, a) * t**a * (1 - t)**(m - a)
        else:
            return 0

    def _phi(self, i, x):
        """
        NEW
        the B-spline phi_i = B_i^p
        """
        assert i in range(self._n_smooth)
        p = self._p
        val = 0
        if self._T_smooth[i] <= x < self._T_smooth[i+p+1]:
            t = self._T_smooth[i:i+p+2]
            b = BSpline.basis_element(t)
            val = b(x)
        return val

    def _phi_pieces(self, i, k, x):
        """
        NEW
        polynomial pieces of the B-splines on the smooth space (\varphi_{i,k} in my notes)
        defined as the restriction of the B-spline phi_i = B_i^p on the interval I_k = [t_{k+p},t_{k+p+1}]
        Note:
            since phi_i is supported on [t_i,t_{i+p+1}], this piece is zero unless k <= i <= k+p
            moreover for i = k, .. k+p they span a basis of P_p(I_k)
        """
        assert i in range(self._n_smooth)
        p = self._p
        val = 0
        if 0 <= k < self._N_cells and k <= i <= k+p and self._T_smooth[k+p] <= x < self._T_smooth[k+p+1]:
            val = self._phi(i, x)
        return val

    def _tilde_phi_P_pieces(self, i, k, x):
        """
        NEW
        local duals to the _phi_pieces, computed using Bernstein basis polynomials
        """
        assert i in range(self._n_smooth)
        p = self._p
        val = 0
        if 0 <= k < self._N_cells and k <= i <= k+p and self._T_smooth[k+p] <= x < self._T_smooth[k+p+1]:
            a = i - k
            for b in range(p+1):
                val += self._tilde_phi_P_coefs[k][a,b] * self._bernstein_P(b,k,x)
        return val

    def _tilde_phi_P(self, i, x):
        """
        NEW
        duals to the _phi B-splines, of type P
        """
        p = self._p
        val = 0
        if self._T_smooth[i] <= x < self._T_smooth[i+p+1]:
            # x is in one cell I_k = [t_{k+p},t_{k+p+1}] with i <= k+p <= i+p
            for a in range(p+1):
                k = i - a
                if 0 <= k < self._N_cells and self._T_smooth[k+p] <= x < self._T_smooth[k+p+1]:
                    val = self._alpha[i,k] * self._tilde_phi_P_pieces(i,k,x)
        return val

    def _tilde_phi_M_aux(self, i, x):
        """
        NEW
        For i a dof index associated with a macro-element, these auxiliary functions form a basis of PP_m
        and they are duals to the splines phi_j (for j an index associated to the same macro-element)

        They are expressed in a Bernstein basis of the macro-element ell
        """
        assert i in range(self._n_smooth)
        p = self._p
        M = self._M_p
        m = self._m
        ell = self.macro_element_index_of_dof(i)
        val = 0
        if self._T_smooth[ell*M + p] <= x < self._T_smooth[(ell+1)*M + p]:
            a = self._local_index_of_macro_element_dof(i, ell)
            for b in range(m+1):
                val += self._tilde_phi_M_aux_coefs[ell][a,b] * self._bernstein_M(b,ell,x)
        return val

    def _tilde_phi_M(self, i, x):
        """
        NEW
        """
        # here we assume that the M dual basis is of MP type
        if self.dof_index_is_macro_vertex(i):
            val = self._tilde_phi_P(i,x)
        else:
            val = self._tilde_phi_M_aux(i,x)
            ell = self.macro_element_index_of_dof(i)
            a = self._local_index_of_macro_element_dof(i, ell)
            for b in range(self._p):
                # corrections with left macro-vertex duals
                j = self._global_index_of_macro_vertex_dof(ell, b)
                val -= self._left_correction_products_tilde_phi_M_aux[ell][a,b] * self._tilde_phi_M(j,x)
                # corrections with right macro-vertex duals
                j = self._global_index_of_macro_vertex_dof(ell+1, b)
                val -= self._right_correction_products_tilde_phi_M_aux[ell][a,b] * self._tilde_phi_M(j,x)
        return val


    def _tilde_phi(self, i, x, type='P'):
        """
        NEW
        duals to the _phi B-splines
        """
        assert i in range(self._n_smooth)
        val = 0
        if type == 'P':
            # then this dual function has the same support as phi_i
            val = self._tilde_phi_P(i,x)
        elif type == 'M':
            val = self._tilde_phi_M(i,x)
        else:
            raise ValueError("dual type unknown: "+repr(type))
        return val

    # def _r_pieces(self, i, x):
    #     # note: this one is restricted to [0,1], unlike in the prospline code
    #     assert i in range(self._p+1)
    #     if 0 < x < 1:
    #         val = 0
    #         for j in range(self._p+1):
    #             val += self._q_to_r_matrix[i, j] * self._q_pieces(j, x)
    #         return val
    #     else:
    #         return 0
    #
    # def _ref_conservative_dual_function(self, x):
    #     """
    #     pw pol (degree p) dual function -- with support [0, p+1]
    #     """
    #     val = 0
    #     for l in range(self._p+1):
    #         val += self._alpha[l] * self._r_pieces(l, x-l)
    #     return val
    #
    # def _pw_pol_dual_function(self, i, x):
    #     """
    #     pw pol (degree p) dual function on the grid, dual to reg_spline(i, x)
    #     """
    #     return (1./self._h) * self._ref_conservative_dual_function(x/self._h-(i-self._p))

    # def _macro_elem_dual_function(self, i, x):
    #     q = np.mod(i, self._M_p)
    #     ell_M_p = i - q
    #     if q < self._p:
    #         # dual function is of macro-boundary type, associated with macro vertex ell_M_p = ell * M_p
    #         val = self._ref_macro_bound_dual_function(q, x/self._h - ell_M_p)
    #     else :
    #         # dual function is of macro-interior type, associated with macro element [ell_M_p, ell_M_p + M_p]
    #         val = self._ref_macro_int_dual_function(q-self._p, x/self._h - ell_M_p)
    #     return val/self._h
    #
    # def compute_duality_prods(self):
    #
    #     # u_quad, w_quad = gauss_legendre(self._p)  # gauss-legendre quadrature rule
    #     # k_quad = len(u_quad)
    #     # nb_element = 1
    #
    #     # note: we may only need these products close to the inner subdomain boundaries
    #
    #     # computing quadratures on the cells [k,k+1]*h, product is polynomial of degree <= p + max( p, m ) <= 2*p
    #     n_quad_nodes = self._p+1  # so that 2n-1 >= 2*p
    #     for k in range(self._N_cells):
    #
    #         # cell = [self.grid[k], self.grid[k+1]]
    #         # points, weights = construct_quadrature_grid(nb_element, k_quad, u_quad, w_quad, cell)
    #
    #         # d = 1                     # number of derivatives
    #         # basis = eval_on_grid_splines_ders(p, n, k, d, T, points)
    #
    #         # support of phi_i: [i-p, i+1]*h, so
    #         i_min = k
    #         i_max = k + self._p
    #         for i in range(i_min, i_max+1):
    #             assert i in range(self._n_smooth)
    #
    #             # phi_i = lambda x: self._reg_spline(i, x)
    #             # phi_i_on_quad_points = list(map(phi_i, points))
    #
    #             if self._use_macro_elem_duals:
    #                 # support of tilde_phi_j:
    #                 #   [Ml-p, M(k+1)+p]*h   if j in {Ml+p, .. Ml+M-1}  (in this case tilde_phi_j is a macro_interior dual function)
    #                 #   [j-p, j+1]           if j in {Ml, .. Ml+p-1}    (in this case tilde_phi_j is a macro_boundary dual function)
    #                 # so, we can take all the j's associated to the macro-element [Ml,Ml+1]*h and its neighbors
    #                 ell = int(k/self._M_p)     # int division
    #                 assert ell*self._M_p <= k < (ell+1)*self._M_p
    #                 j_min = max((ell-1)*self._M_p,   0)
    #                 j_max = min((ell+2)*self._M_p-1, self._n_smooth-1)
    #             else:
    #                 j_min = k
    #                 j_max = k+self._p
    #             for j in range(j_min, j_max+1):
    #                 assert j in range(self._n_smooth)
    #
    #                 def tilde_phi_j(x):
    #                     if self._use_macro_elem_duals:
    #                         return self._macro_elem_dual_function(j, x)
    #                     else:
    #                         return self._pw_pol_dual_function(j, x)
    #
    #                 self.duality_prods[i, j, k] = quadrature(
    #                     lambda x: self._reg_spline(i, x) * tilde_phi_j(x),
    #                     self.grid[k], self.grid[k+1],
    #                     maxiter=n_quad_nodes, vec_func=False,
    #                 )[0]

    # def smooth_projection(self):
    #     """
    #     computes the coefs of a smooth projection in the smooth spline space, from the coefs in the discontinuous space
    #     """
    #     self.coefs_smooth[:] = 0
    #     # project the left function
    #     delta = 0  # global position of the 1st node in LEFT domain
    #     for i in range(self._n_left):
    #         assert 0 <= i+delta < self._n_smooth
    #         if i+1 <= self._N_cells_left:
    #             # then support of phi_left[i] is in left domain, so phi_left[i] == phi[i+delta]
    #             # and coefs_left[i] only contributes to coefs_smooth[i+delta]
    #             j = i+delta
    #             self.coefs_smooth[j] += self.coefs_left[i]
    #         else:
    #             for j in range(self._n_smooth):     # todo: localize
    #                 pi_ij = 0
    #                 for k in range(self._N_cells_left):     # todo: localize
    #                     assert 0 <= k+delta < self._N_cells
    #                     pi_ij += self.duality_prods[i+delta, j, k+delta]
    #                 self.coefs_smooth[j] += pi_ij * self.coefs_left[i]
    #
    #     # project the right function
    #     delta = self._N_cells_left  # global position of the 1st node in RIGHT domain
    #     for i in range(self._n_right):
    #         assert 0 <= i+delta < self._n_smooth
    #         if i-self._p >= 0:
    #             # then support of phi_right[i] is in right domain, so phi_right[i] == phi[i+delta]
    #             # and coefs_right[i] only contributes to coefs_smooth[i+delta]
    #             j = i+delta
    #             self.coefs_smooth[j] += self.coefs_right[i]
    #         else:
    #             for j in range(self._n_smooth):     # todo: localize
    #                 pi_ij = 0
    #                 for k in range(self._N_cells_right):     # todo: localize
    #                     assert 0 <= k+delta < self._N_cells
    #                     pi_ij += self.duality_prods[i+delta, j, k+delta]
    #                 self.coefs_smooth[j] += pi_ij * self.coefs_right[i]

    def local_smooth_proj(self, f, type='P', check=False):
        """
        NEW
        """
        grid = construct_grid_from_knots(self._p, self._n_smooth, self._T_smooth)
        for i in range(self._n_smooth):
            self.coefs_smooth[i] = _my_L2_prod(
                lambda x:f(x),
                lambda x:self._tilde_phi(i, x, type=type),
                sing_points=grid,
            )

        if check:
            print("check -- "+repr(type)+" duals:  <tilde phi_i, phi_j>")
            coef_check = np.zeros((self._n_smooth,self._n_smooth))
            for i in range(self._n_smooth):
                for j in range(self._n_smooth):
                    coef_check[i,j] = _my_L2_prod(
                        lambda x:self._tilde_phi(i,x, type=type),
                        lambda x:self._phi(j, x),
                        sing_points=grid,
                    )
            print(coef_check)

                # ,
                # sing_points=self._T_smooth)

    def l2_project_on_sub_domain(self, f, sub='left'):
        """
        L2 projection (??) derived from test_projector_1d by ARA
        """
        if sub == 'left':
            n = self._n_left
            T = self._T_left
        else:
            assert sub == 'right'
            n = self._n_right
            T = self._T_right
        p = self._p
        print("L2 proj on subdomain "+sub)
        print("n = "+repr(n))
        print("T = "+repr(T))
        mass_0, mass_1 = mass_matrices(p, n, T)   # works with a knot vector that is not open ??
        contribution = Contribution(p, n, T)
        f_l2 = solve(mass_1, contribution(f))
        tck = get_tck('L2', p, n, T, f_l2)    # H1, L2 ??
        assert len(tck[1]) == n
        if sub == 'left':
            self.coefs_left = tck[1]
        else:
            assert sub == 'right'
            self.coefs_right = tck[1]

    def histopolation_on_sub_domain(self, f, sub='left'):
        """
        histopolation (??) derived from test_projector_1d by ARA
        """
        if sub == 'left':
            n = self._n_left
            T = self._T_left
        else:
            assert sub == 'right'
            n = self._n_right
            T = self._T_right
        p = self._p
        print("Histopolation on subdomain "+sub)
        print("n = "+repr(n))
        print("T = "+repr(T))
        I0, I1 = interpolation_matrices(p, n, T)
        histopolation = Integral(p, n, T, kind='greville')
        f_1 = solve(I1, histopolation(f))
        # scale fh_1 coefficients
        S = scaling_matrix(p, n, T, kind='L2')
        f_1 = S.dot(f_1)
        tck = get_tck('L2', p, n, T, f_1)
        assert len(tck[1]) == n    # assert FAILS ?
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

    def eval_continuous_spline_splev(self, x):
        tck = [self._T_smooth, self.coefs_smooth, self._p]
        return splev(x, tck)

    def eval_continuous_spline(self, x):
        val = 0
        for i in range(self._n_smooth):
            val += self.coefs_smooth[i] * self._phi(i, x)
        return val

    def plot_spline(
            self,
            filename,
            spline_type='continuous',
            N_points=100,
            ltype='-',
            f_ref=None,
            legend=None,
            title=None,
            legend_loc='lower left'):

        vis_grid, vis_h = np.linspace(self._x_min, self._x_max, N_points, endpoint=False, retstep=True)
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
        if f_ref is not None:
            vals = [f_ref(xi) for xi in vis_grid]
            image = plt.plot(vis_grid, vals, ltype, color='r', label="f ref")
        plt.legend(loc=legend_loc)
        fig.savefig(filename)
        plt.clf()


# local utilities, derived from some scipy quadrature functions

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
    #print("fg(0.6) = ", f(0.6)*g(0.6) )
    #print("valid_points = ", valid_points)
    return quad(lambda x: f(x)*g(x), xmin, xmax, points=valid_points, epsabs=epsabs, epsrel=epsrel, limit=100)[0]
