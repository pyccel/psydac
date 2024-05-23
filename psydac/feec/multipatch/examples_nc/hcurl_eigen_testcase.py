"""
    Runner script for solving the eigenvalue problem for the H(curl) operator for different discretizations.
"""

import os
import numpy as np

from psydac.feec.multipatch.examples_nc.hcurl_eigen_pbms_nc import hcurl_solve_eigen_pbm_nc
from psydac.feec.multipatch.examples_nc.hcurl_eigen_pbms_dg import hcurl_solve_eigen_pbm_dg
from psydac.feec.multipatch.utilities import time_count, get_run_dir, get_plot_dir, get_mat_dir, get_sol_dir, diag_fn
from psydac.feec.multipatch.utils_conga_2d import write_diags_to_file
from psydac.api.postprocessing import OutputManager, PostProcessManager

t_stamp_full = time_count()

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#
# test-case and numerical parameters:
method = 'feec'
# method = 'dg'

operator = 'curl-curl'
degree = [3, 3]  # shared across all patches

# pretzel_f (18 patches)
# domain_name = 'pretzel_f'
# ncells = np.array([8, 8, 16, 16, 8, 4, 4, 4, 4, 4, 2, 2, 4, 16, 16, 8, 2, 2, 2])
# ncells = np.array([4 for _ in range(18)])

# domain onlyneeded for square like domains
domain = [[0, np.pi], [0, np.pi]]  # interval in x- and y-direction

# refined square domain
# domain_name = 'refined_square'
# the shape of ncells gives the shape of the domain,
# while the entries describe the isometric number of cells in each patch
# 2x2 = 4 patches
# ncells = np.array([[8, 4],
#                    [4, 4]])
# 3x3= 9 patches
# ncells = np.array([[4, 2, 4],
#                    [2, 4, 2],
#                    [4, 2, 4]])

# L-shaped domain
# domain_name = 'square_L_shape'
# domain=[[-1, 1],[-1, 1]] # interval in x- and y-direction

# The None indicates the patches to leave out
# 2x2 = 4 patches
# ncells = np.array([[None, 2],
#                    [2, 2]])
# 4x4 = 16 patches
# ncells = np.array([[None, None, 4, 2],
#                    [None, None, 8, 4],
#                    [4,     8,   8, 4],
#                    [2,     4,   4, 2]])
# 8x8 = 64 patches
# ncells = np.array([[None, None, None, None, 2, 2, 2,1 2],
#                    [None, None, None, None, 2, 2, 2, 2],
#                    [None, None, None, None, 2, 2, 2, 2],
#                    [None, None, None, None, 4, 4, 2, 2],
#                    [2,      2,    2,    4,  8, 4, 2, 2],
#                    [2,      2,    2,    4,  4, 4, 2, 2],
#                    [2,      2,    2,    2,  2, 2, 2, 2],
#                    [2,      2,    2,    2,  2, 2, 2, 2]])

# Curved L-shape domain
domain_name = 'curved_L_shape'
domain = [[1, 3], [0, np.pi / 4]]  # interval in x- and y-direction


ncells = np.array([[None, 5],
                   [5, 10]])


# ncells = np.array([[None, None, 2, 2],
#                    [None, None, 4, 2],
#                    [ 2,    4,   8, 4],
#                    [ 2,    2,   4, 4]])

# ncells = np.array([[None, None, None,   2, 2, 2],
#                    [None, None, None,  4, 4, 2],
#                    [None, None, None,  8, 4, 2],
#                    [2,     4,     8,   8, 4, 2],
#                    [2,     4,     4,   4, 4, 2],
#                    [2,     2,     2,   2, 2, 2]])

# ncells = np.array([[None, None, None,  None,  2, 2, 2, 2],
#                    [None, None, None,  None,  4, 4, 4, 2],
#                    [None, None, None,  None,  8, 8, 4, 2],
#                    [None, None, None,  None, 16, 8, 4, 2],
#                    [2,     4,     8,   16,   16, 8, 4, 2],
#                    [2,     4,     8,    8,    8, 8, 4, 2],
#                    [2,     4,     4,    4,    4, 4, 4, 2],
#                    [2,     2,     2,    2,    2, 2, 2, 2]])

# all kinds of different square refinements and constructions are possible, eg
# doubly connected domains
# ncells = np.array([[4,  2,    2,   4],
#                    [2, None, None, 2],
#                    [2, None, None, 2],
#                    [4,  2,    2,   4]])

gamma_h = 0
# solves generalized eigenvalue problem with:  B(v,w) = <Pv,Pw> +
# <(I-P)v,(I-P)w> in rhs
generalized_pbm = True

if operator == 'curl-curl':
    nu = 0
    mu = 1
else:
    raise ValueError(operator)

case_dir = 'eigenpbm_' + operator + '_' + method
ref_case_dir = case_dir

ref_sigmas = None
sigma = None
nb_eigs_solve = None
nb_eigs_plot = None
skip_eigs_threshold = None
diags = None
eigenvalues = None

if domain_name == 'refined_square':
    assert domain == [[0, np.pi], [0, np.pi]]
    ref_sigmas = [
        1, 1,
        2,
        4, 4,
        5, 5,
        8,
        9, 9,
    ]
    sigma = 5
    nb_eigs_solve = 10
    nb_eigs_plot = 10
    skip_eigs_threshold = 1e-7

elif domain_name == 'square_L_shape':
    assert domain == [[-1, 1], [-1, 1]]
    ref_sigmas = [
        1.47562182408,
        3.53403136678,
        9.86960440109,
        9.86960440109,
        11.3894793979,
    ]
    sigma = 6
    nb_eigs_solve = 5
    nb_eigs_plot = 5
    skip_eigs_threshold = 1e-7

elif domain_name == 'curved_L_shape':
    # ref eigenvalues from Monique Dauge benchmark page
    assert domain == [[1, 3], [0, np.pi / 4]]
    ref_sigmas = [
        0.181857115231E+01,
        0.349057623279E+01,
        0.100656015004E+02,
        0.101118862307E+02,
        0.124355372484E+02,
    ]
    sigma = 7
    nb_eigs_solve = 7
    nb_eigs_plot = 7
    skip_eigs_threshold = 1e-7

elif domain_name in ['pretzel_f']:
    if operator == 'curl-curl':
        # ref sigmas computed with nc=20 and deg=6 and gamma = 0 (and
        # generalized ev-pbm)
        ref_sigmas = [
            0.1795339843,
            0.1992261261,
            0.6992717244,
            0.8709410438,
            1.1945106937,
            1.2546992683,
        ]

        sigma = .8
        nb_eigs_solve = 10
        nb_eigs_plot = 5
        skip_eigs_threshold = 1e-7

#
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

common_diag_filename = './' + case_dir + '_diags.txt'


params = {
    'domain_name': domain_name,
    'domain': domain,
    'operator': operator,
    'mu': mu,
    'nu': nu,
    'ncells': ncells,
    'degree': degree,
    'gamma_h': gamma_h,
    'generalized_pbm': generalized_pbm,
    'nb_eigs_solve': nb_eigs_solve,
    'skip_eigs_threshold': skip_eigs_threshold
}

print(params)

# backend_language = 'numba'
backend_language = 'pyccel-gcc'

dims = ncells.shape
sz = ncells[ncells is not None].sum()
print(dims)
# get_run_dir(domain_name, nc, deg)
run_dir = domain_name + str(dims) + 'patches_' + 'size_{}'.format(sz)
plot_dir = get_plot_dir(case_dir, run_dir)
diag_filename = plot_dir + '/' + diag_fn()
common_diag_filename = './' + case_dir + '_diags.txt'

# to save and load matrices
# m_load_dir = get_mat_dir(domain_name, nc, deg)
m_load_dir = None

print('\n --- --- --- --- --- --- --- --- --- --- --- --- --- --- \n')
print(' Calling hcurl_solve_eigen_pbm() with params = {}'.format(params))
print('\n --- --- --- --- --- --- --- --- --- --- --- --- --- --- \n')

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# calling eigenpbm solver for:
#
# find lambda in R and u in H0(curl), such that
#   A u   = lambda * u    on \Omega
# with
#
#   A u := mu * curl curl u  -  nu * grad div u
#
# note:
#   - we look for nb_eigs_solve eigenvalues close to sigma (skip zero eigenvalues if skip_zero_eigs==True)
#   - we plot nb_eigs_plot eigenvectors
if method == 'feec':
    diags, eigenvalues = hcurl_solve_eigen_pbm_nc(
        ncells=ncells, degree=degree,
        gamma_h=gamma_h,
        generalized_pbm=generalized_pbm,
        nu=nu,
        mu=mu,
        sigma=sigma,
        ref_sigmas=ref_sigmas,
        skip_eigs_threshold=skip_eigs_threshold,
        nb_eigs_solve=nb_eigs_solve,
        nb_eigs_plot=nb_eigs_plot,
        domain_name=domain_name, domain=domain,
        backend_language=backend_language,
        plot_dir=plot_dir,
        hide_plots=True,
        m_load_dir=m_load_dir,
    )
elif method == 'dg':
    diags, eigenvalues = hcurl_solve_eigen_pbm_dg(
        ncells=ncells, degree=degree,
        gamma_h=gamma_h,
        generalized_pbm=generalized_pbm,
        nu=nu,
        mu=mu,
        sigma=sigma,
        ref_sigmas=ref_sigmas,
        skip_eigs_threshold=skip_eigs_threshold,
        nb_eigs_solve=nb_eigs_solve,
        nb_eigs_plot=nb_eigs_plot,
        domain_name=domain_name, domain=domain,
        backend_language=backend_language,
        plot_dir=plot_dir,
        hide_plots=True,
        m_load_dir=m_load_dir,
    )


if ref_sigmas is not None:
    errors = []
    n_errs = min(len(ref_sigmas), len(eigenvalues))
    for k in range(n_errs):
        diags['error_{}'.format(k)] = abs(eigenvalues[k] - ref_sigmas[k])
#
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

write_diags_to_file(
    diags,
    script_filename=__file__,
    diag_filename=diag_filename,
    params=params)
write_diags_to_file(
    diags,
    script_filename=__file__,
    diag_filename=common_diag_filename,
    params=params)

# PM = PostProcessManager(geometry_file=, )
time_count(t_stamp_full, msg='full program')
