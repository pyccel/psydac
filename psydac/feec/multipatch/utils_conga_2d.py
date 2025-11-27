#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os
import datetime

import numpy as np

from sympy import lambdify
from sympde.topology import Derham
from sympde.topology.callable_mapping import BasicCallableMapping

from psydac.api.settings import PSYDAC_BACKENDS
from psydac.feec.pull_push import pull_2d_h1, pull_2d_hcurl, pull_2d_l2

from psydac.api.discretization import discretize
from psydac.feec.multipatch.utilities import time_count  
from psydac.linalg.utilities import array_to_psydac
from psydac.fem.basic import FemField
from psydac.fem.plotting_utilities import get_plotting_grid, get_grid_quad_weights, get_grid_vals

from scipy.sparse import kron, block_diag
from psydac.core.bsplines import collocation_matrix, histopolation_matrix
from psydac.linalg.solvers import inverse


# commuting projections on the physical domain (should probably be in the
# interface)
def P0_phys(f_phys, P0, domain):
    f = lambdify(domain.coordinates, f_phys)

    return P0(f)


def P1_phys(f_phys, P1, domain):
    f_x = lambdify(domain.coordinates, f_phys[0])
    f_y = lambdify(domain.coordinates, f_phys[1])

    return P1([f_x, f_y])


def P2_phys(f_phys, P2, domain):
    f = lambdify(domain.coordinates, f_phys)

    return P2(f)


def get_kind(space='V*'):
    # temp helper
    if space == 'V0':
        kind = 'h1'
    elif space == 'V1':
        kind = 'hcurl'
    elif space == 'V2':
        kind = 'l2'
    else:
        raise ValueError(space)
    return kind

# ===============================================================================
def get_K0_and_K0_inv(V0h, uniform_patches=False):
    """
    Compute the change of basis matrices K0 and K0^{-1} in V0h.

    With
    K0_ij = sigma^0_i(B_j) = B_jx(n_ix) * B_jy(n_iy)
    where sigma_i is the geometric (interpolation) dof
    and B_j is the tensor-product B-spline
    """
    if uniform_patches:
        print(' [[WARNING -- hack in get_K0_and_K0_inv: using copies of 1st-patch matrices in every patch ]] ')

    V0 = V0h.symbolic_space   # VOh is FemSpace
    domain = V0.domain
    K0_blocks = []
    K0_inv_blocks = []
    for k, D in enumerate(domain.interior):
        if uniform_patches and k > 0:
            K0_k = K0_blocks[0].copy()
            K0_inv_k = K0_inv_blocks[0].copy()

        else:
            V0_k = V0h.spaces[k]  # fem space on patch k: (TensorFemSpace)
            K0_k_factors = [None, None]
            for d in [0, 1]:
                # 1d fem space alond dim d (SplineSpace)
                V0_kd = V0_k.spaces[d]
                K0_k_factors[d] = collocation_matrix(
                    knots=V0_kd.knots,
                    degree=V0_kd.degree,
                    periodic=V0_kd.periodic,
                    normalization=V0_kd.basis,
                    xgrid=V0_kd.greville
                )
            K0_k = kron(*K0_k_factors)
            K0_k.eliminate_zeros()
            K0_inv_k = inv(K0_k.tocsc())
            K0_inv_k.eliminate_zeros()

        K0_blocks.append(K0_k)
        K0_inv_blocks.append(K0_inv_k)
    K0 = block_diag(K0_blocks)
    K0_inv = block_diag(K0_inv_blocks)
    return K0, K0_inv


# ===============================================================================
def get_K1_and_K1_inv(V1h, uniform_patches=False):
    """
    Compute the change of basis matrices K1 and K1^{-1} in Hcurl space V1h.

    With
    K1_ij = sigma^1_i(B_j) = int_{e_ix}(M_jx) * B_jy(n_iy)
    if i = horizontal edge [e_ix, n_iy] and j = (M_jx o B_jy)  x-oriented MoB spline
    or
    = B_jx(n_ix) * int_{e_iy}(M_jy)
    if i = vertical edge [n_ix, e_iy]  and  j = (B_jx o M_jy)  y-oriented BoM spline
    (above, 'o' denotes tensor-product for functions)
    """
    if uniform_patches:
        print(' [[WARNING -- hack in get_K1_and_K1_inv: using copies of 1st-patch matrices in every patch ]] ')

    V1 = V1h.symbolic_space   # V1h is FemSpace
    domain = V1.domain
    K1_blocks = []
    K1_inv_blocks = []
    for k, D in enumerate(domain.interior):
        if uniform_patches and k > 0:
            K1_k = K1_blocks[0].copy()
            K1_inv_k = K1_inv_blocks[0].copy()

        else:
            # fem space on patch k:
            V1_k = V1h.spaces[k]
            K1_k_blocks = []
            for c in [0, 1]:    # dim of component
                # fem space for comp. dc (TensorFemSpace)
                V1_kc = V1_k.spaces[c]
                K1_kc_factors = [None, None]
                for d in [0, 1]:    # dim of variable
                    # 1d fem space for comp c alond dim d (SplineSpace)
                    V1_kcd = V1_kc.spaces[d]
                    if c == d:
                        K1_kc_factors[d] = histopolation_matrix(
                            knots=V1_kcd.knots,
                            degree=V1_kcd.degree,
                            periodic=V1_kcd.periodic,
                            normalization=V1_kcd.basis,
                            xgrid=V1_kcd.ext_greville
                        )
                    else:
                        K1_kc_factors[d] = collocation_matrix(
                            knots=V1_kcd.knots,
                            degree=V1_kcd.degree,
                            periodic=V1_kcd.periodic,
                            normalization=V1_kcd.basis,
                            xgrid=V1_kcd.greville
                        )
                K1_kc = kron(*K1_kc_factors)
                K1_kc.eliminate_zeros()
                K1_k_blocks.append(K1_kc)
            K1_k = block_diag(K1_k_blocks)
            K1_k.eliminate_zeros()
            K1_inv_k = inv(K1_k.tocsc())
            K1_inv_k.eliminate_zeros()

        K1_blocks.append(K1_k)
        K1_inv_blocks.append(K1_inv_k)

    K1 = block_diag(K1_blocks)
    K1_inv = block_diag(K1_inv_blocks)
    return K1, K1_inv

# ===============================================================================


def ortho_proj_Hcurl(EE, V1h, domain_h, M1, backend_language='python'):
    """
    return orthogonal projection of E on V1h, given M1 the mass matrix
    """
    assert isinstance(EE, Tuple)
    V1 = V1h.symbolic_space
    v = element_of(V1, name='v')
    l = LinearForm(v, integral(V1.domain, dot(v, EE)))
    lh = discretize(
        l,
        domain_h,
        V1h,
        backend=PSYDAC_BACKENDS[backend_language])
    b = lh.assemble()
    M1_inv = inverse(M1.mat(), 'pcg', pc='jacobi', tol=1e-10)
    sol_coeffs = M1_inv @ b

    return FemField(V1h, coeffs=sol_coeffs)


# ===============================================================================
class DiagGrid():
    """
    Class storing:
        - a diagnostic cell-centered grid
        - writing / quadrature utilities
        - a ref solution
    
    to compare solutions from different FEM spaces on same domain
    """

    def __init__(self, mappings=None, N_diag=None):

        mappings_list = list(mappings.values())
        etas, xx, yy, patch_logvols = get_plotting_grid(
            mappings, N=N_diag, centered_nodes=True, return_patch_logvols=True)
        quad_weights = get_grid_quad_weights(
            etas, patch_logvols, mappings_list)

        self.etas = etas
        self.xx = xx
        self.yy = yy
        self.patch_logvols = patch_logvols
        self.quad_weights = quad_weights
        self.mappings_list = mappings_list

        self.sol_ref = {}  # Fem fields
        self.sol_vals = {}   # values on diag grid
        self.sol_ref_vals = {}   # values on diag grid

    def grid_vals_h1(self, v):
        return get_grid_vals(v, self.etas, self.mappings_list, space_kind='h1')

    def grid_vals_hcurl(self, v):
        return get_grid_vals(
            v,
            self.etas,
            self.mappings_list,
            space_kind='hcurl')

    def create_ref_fem_spaces(self, domain=None, ref_nc=None, ref_deg=None):
        print('[DiagGrid] Discretizing the ref FEM space...')
        degree = [ref_deg, ref_deg]
        derham = Derham(domain, ["H1", "Hcurl", "L2"])
        ref_nc = {patch.name: [ref_nc, ref_nc] for patch in domain.interior}

        domain_h = discretize(domain, ncells=ref_nc)
        # , backend=PSYDAC_BACKENDS[backend_language])
        derham_h = discretize(derham, domain_h, degree=degree)
        self.V0h = derham_h.V0
        self.V1h = derham_h.V1

    def import_ref_sol_from_coeffs(self, sol_ref_filename=None, space='V*'):
        print('[DiagGrid] loading coeffs of ref_sol from {}...'.format(
            sol_ref_filename))
        if space == 'V0':
            Vh = self.V0h
        elif space == 'V1':
            Vh = self.V1h
        else:
            raise ValueError(space)
        try:
            coeffs = np.load(sol_ref_filename)
        except OSError:
            print("-- WARNING: file not found, setting sol_ref = 0")
            coeffs = np.zeros(Vh.nbasis)
        if space in self.sol_ref:
            print(
                'WARNING !! sol_ref[{}] exists -- will be overwritten !! '.format(space))
            print('use refined labels if several solutions are needed in the same space')
        self.sol_ref[space] = FemField(
            Vh, coeffs=array_to_psydac(
                coeffs, Vh.coeff_space))

    def write_sol_values(self, v, space='V*'):
        """
        v: FEM field
        """
        if space in self.sol_vals:
            print(
                'WARNING !! sol_vals[{}] exists -- will be overwritten !! '.format(space))
            print('use refined labels if several solutions are needed in the same space')
        self.sol_vals[space] = get_grid_vals(
            v, self.etas, self.mappings_list, space_kind=get_kind(space))

    def write_sol_ref_values(self, v=None, space='V*'):
        """
        if no FemField v is provided, then use the self.sol_ref (must have been imported)
        """
        if space in self.sol_vals:
            print(
                'WARNING !! sol_ref_vals[{}] exists -- will be overwritten !! '.format(space))
            print('use refined labels if several solutions are needed in the same space')
        if v is None:
            # then sol_ref must have been imported
            v = self.sol_ref[space]
        self.sol_ref_vals[space] = get_grid_vals(
            v, self.etas, self.mappings_list, space_kind=get_kind(space))

    def compute_l2_error(self, space='V*'):
        if space in ['V0', 'V2']:
            u = self.sol_ref_vals[space]
            uh = self.sol_vals[space]
            abs_u = [np.abs(p) for p in u]
            abs_uh = [np.abs(p) for p in uh]
            errors = [np.abs(p - q) for p, q in zip(u, uh)]
        elif space == 'V1':
            u_x, u_y = self.sol_ref_vals[space]
            uh_x, uh_y = self.sol_vals[space]
            abs_u = [np.sqrt((u1)**2 + (u2)**2) for u1, u2 in zip(u_x, u_y)]
            abs_uh = [np.sqrt((u1)**2 + (u2)**2) for u1, u2 in zip(uh_x, uh_y)]
            errors = [np.sqrt((u1 - v1)**2 + (u2 - v2)**2)
                      for u1, v1, u2, v2 in zip(u_x, uh_x, u_y, uh_y)]
        else:
            raise ValueError(space)

        l2_norm_uh = (
            np.sum([J_F * v**2 for v, J_F in zip(abs_uh, self.quad_weights)]))**0.5
        l2_norm_u = (
            np.sum([J_F * v**2 for v, J_F in zip(abs_u, self.quad_weights)]))**0.5
        l2_error = (
            np.sum([J_F * v**2 for v, J_F in zip(errors, self.quad_weights)]))**0.5

        return l2_norm_uh, l2_norm_u, l2_error

    def get_diags_for(self, v, space='V*', print_diags=True):
        self.write_sol_values(v, space)
        sol_norm, sol_ref_norm, l2_error = self.compute_l2_error(space)
        rel_l2_error = l2_error / (max(sol_norm, sol_ref_norm))
        diags = {
            'sol_norm': sol_norm,
            'sol_ref_norm': sol_ref_norm,
            'rel_l2_error': rel_l2_error,
        }
        if print_diags:
            print(' .. l2 norms (computed via quadratures on diag_grid): ')
            print(diags)

        return diags


def get_Vh_diags_for(
        v=None,
        v_ref=None,
        M_m=None,
        print_diags=True,
        msg='error between ?? and ?? in Vh'):
    """
    v, v_ref: FemField
    M_m: mass matrix in scipy format
    """
    uh_c = v.coeffs.toarray()
    uh_ref_c = v_ref.coeffs.toarray()
    err_c = uh_c - uh_ref_c
    l2_error = np.dot(err_c, M_m.dot(err_c))**0.5
    sol_norm = np.dot(uh_c, M_m.dot(uh_c))**0.5
    sol_ref_norm = np.dot(uh_ref_c, M_m.dot(uh_ref_c))**0.5
    rel_l2_error = l2_error / (max(sol_norm, sol_ref_norm))
    diags = {
        'sol_norm': sol_norm,
        'sol_ref_norm': sol_ref_norm,
        'rel_l2_error': rel_l2_error,
    }
    if print_diags:
        print(' .. l2 norms ({}): '.format(msg))
        print(diags)

    return diags


def write_diags_to_file(diags, script_filename, diag_filename, params=None):
    """ write diagnostics to file """
    print(' -- writing diags to file {} --'.format(diag_filename))
    if not os.path.exists(diag_filename):
        open(diag_filename, 'w')

    with open(diag_filename, 'a') as a_writer:
        a_writer.write('\n')
        a_writer.write(
            ' ---------- ---------- ---------- ---------- ---------- ---------- \n')
        a_writer.write(' run script:  \n   {}\n'.format(script_filename))
        a_writer.write(
            ' executed on: \n   {}\n\n'.format(
                datetime.datetime.now()))
        a_writer.write(' params:  \n')
        for key, value in params.items():
            a_writer.write('   {}: {} \n'.format(key, value))
        a_writer.write('\n')
        a_writer.write(' diags:  \n')
        for key, value in diags.items():
            a_writer.write('   {}: {} \n'.format(key, value))
        a_writer.write(
            ' ---------- ---------- ---------- ---------- ---------- ---------- \n')
        a_writer.write('\n')
