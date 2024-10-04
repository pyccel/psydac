"""
    Runner script for solving the time-domain Maxwell problem.
"""

import numpy as np

from psydac.feec.multipatch.examples.timedomain_maxwell import solve_td_maxwell_pbm
from psydac.feec.multipatch.utilities import time_count, FEM_sol_fn, get_run_dir, get_plot_dir, get_mat_dir, get_sol_dir, diag_fn
from psydac.feec.multipatch.utils_conga_2d import write_diags_to_file

t_stamp_full = time_count()

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#
# main test-cases and parameters used for the ppc paper:

test_case = 'E0_pulse_no_source'   # used in paper
# test_case = 'Issautier_like_source'  # used in paper
# test_case = 'transient_to_harmonic'  # actually, not used in paper

# J_proj_case = 'P_geom'
J_proj_case = 'P_L2'
# J_proj_case = 'tilde Pi_1'

#
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Parameters to be changed in the batch run
deg = 3

# Common simulation parameters
# domain_name = 'square_6'
# ncells = [4,4,4,4,4,4]
# domain_name = 'pretzel_f'

# non-conf domains
domain = [[0, 2 * np.pi], [0, 2 * np.pi]]  # interval in x- and y-direction
domain_name = 'refined_square'
# use isotropic meshes (probably with a square domain)
# 4x8= 64 patches
# care for the transpose
# ncells = np.array([[16, 16],
#                    [16, 16]])

ncells = np.array([[8,8,16,8],
                 [8,8,16,8],
                 [8,8,16,8],
                 [8,8,16,8]])
# ncells = np.array([[8,8,8,8],
#                    [8,8,8,8],
#                    [8,8,8,8],
#                    [8,8,8,8]])
# ncells = np.array([[8,8,16,8,8,8],
#                      [8,8,16,8,8,8],
#                      [8,8,16,8,8,8],
#                      [8,8,16,8,8,8]])

# ncells = np.array([[4, 4, 4],
#                     [4, 8, 4],
#                     [8, 16, 8],
#                     [4, 8, 4],
#                     [4, 4, 4]])
# ncells = np.array([[4, 4, 4, 4],
#                     [4, 8, 8, 4],
#                     [8, 16, 16, 8],
#                     [4, 8, 8, 4],
#                     [4, 4, 4, 4]]).transpose()
# ncells = np.array([[4, 4, 4, 4],
#                     [4, 4, 4, 4],
#                     [4, 8, 8, 4],
#                     [8, 16, 16, 8],
#                     [8, 16, 16, 8],
#                     [4, 8, 8, 4],
#                     [4, 4, 4, 4],
#                     [4, 4, 4, 4]])


cfl_max = 0.8
# 'P_geom'  # projection used for initial E0 (B0 = 0 in all cases)
E0_proj = 'P_geom'
backend = 'pyccel-gcc'
project_sol = True  # whether cP1 E_h is plotted instead of E_h
# multiplicative parameter for quadrature order in (bi)linear forms
# discretizaion
quad_param = 4
gamma_h = 0     # jump dissipation parameter (not used in paper)
# 'BSP' # type of conforming projection operators (averaging B-spline or Geometric-splines coefficients)
conf_proj = 'GSP'
hide_plots = True
plot_divE = True
# time interval between scalar diagnostics (if None, compute every time step)
diag_dt = None

# Parameters that depend on test case
if test_case == 'E0_pulse_no_source':

    E0_type = 'pulse_2'   # non-zero initial conditions
    source_type = 'zero'    # no current source
    source_omega = None
    final_time = 9.02  # wave transit time in domain is > 4
    dt_max = None
    plot_source = False

    plot_a_lot = True
    if plot_a_lot:
        plot_time_ranges = [[[0, final_time], 0.1]]
    else:
        plot_time_ranges = [
            [[0, 2], 0.1],
            [[final_time - 1, final_time], 0.1],
        ]

    cb_min_sol = 0
    cb_max_sol = 5

# TODO: check
elif test_case == 'Issautier_like_source':

    E0_type = 'zero'      # zero initial conditions
    source_type = 'Il_pulse'
    source_omega = None
    final_time = 20
    plot_source = True
    dt_max = None
    if deg_s == [3] and final_time == 20:

        plot_time_ranges = [
            [[1.9, 2], 0.1],
            [[4.9, 5], 0.1],
            [[9.9, 10], 0.1],
            [[19.9, 20], 0.1],
        ]

        # plot_time_ranges = [
        #     ]
        # if nc_s == [8]:
        #     Nt_pp = 10

    cb_min_sol = 0  # None
    cb_max_sol = 0.3  # None

# TODO: check
elif test_case == 'transient_to_harmonic':

    E0_type = 'th_sol'
    source_type = 'elliptic_J'
    source_omega = np.sqrt(50)  # source time pulsation
    plot_source = True

    source_period = 2 * np.pi / source_omega
    nb_t_periods = 100
    Nt_pp = 20

    dt_max = source_period / Nt_pp
    final_time = nb_t_periods * source_period

    plot_time_ranges = [
        [[(nb_t_periods - 2) * source_period, final_time], dt_max]
    ]

    cb_min_sol = 0
    cb_max_sol = 1

else:
    raise ValueError(test_case)


# projection used for the source J
if J_proj_case == 'P_geom':
    source_proj = 'P_geom'
    filter_source = False

elif J_proj_case == 'P_L2':
    source_proj = 'P_L2'
    filter_source = False

elif J_proj_case == 'tilde Pi_1':
    source_proj = 'P_L2'
    filter_source = True

else:
    raise ValueError(J_proj_case)

case_dir = 'nov14_' + test_case + '_J_proj=' + \
    J_proj_case + '_qp{}'.format(quad_param)
if filter_source:
    case_dir += '_Jfilter'
else:
    case_dir += '_Jnofilter'
if not project_sol:
    case_dir += '_E_noproj'

if source_omega is not None:
    case_dir += f'_omega={source_omega}'

case_dir += f'_tend={final_time}'

#
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

common_diag_filename = './' + case_dir + '_diags.txt'


run_dir = get_run_dir(
    domain_name,
    sum(ncells),
    deg,
    source_type=source_type,
    conf_proj=conf_proj)
plot_dir = get_plot_dir(case_dir, run_dir)
diag_filename = plot_dir + '/' + \
    diag_fn(source_type=source_type, source_proj=source_proj)

# to save and load matrices
m_load_dir = get_mat_dir(domain_name, sum(ncells), deg, quad_param=quad_param)

if E0_type == 'th_sol':
    # initial E0 will be loaded from time-harmonic FEM solution
    th_case_dir = 'maxwell_hom_eta=50'
    th_sol_dir = get_sol_dir(th_case_dir, domain_name, sum(ncells), deg)
    th_sol_filename = th_sol_dir + '/' + \
        FEM_sol_fn(source_type=source_type, source_proj=source_proj)
else:
    # no initial solution to load
    th_sol_filename = ''

params = {
    'nc': ncells,
    'deg': deg,
    'final_time': final_time,
    'cfl_max': cfl_max,
    'dt_max': dt_max,
    'domain_name': domain_name,
    'backend': backend,
    'source_type': source_type,
    'source_omega': source_omega,
    'source_proj': source_proj,
    'conf_proj': conf_proj,
    'gamma_h': gamma_h,
    'project_sol': project_sol,
    'filter_source': filter_source,
    'quad_param': quad_param,
    'E0_type': E0_type,
    'E0_proj': E0_proj,
    'hide_plots': hide_plots,
    'plot_dir': plot_dir,
    'plot_time_ranges': plot_time_ranges,
    'plot_source': plot_source,
    'plot_divE': plot_divE,
    'diag_dt': diag_dt,
    'cb_min_sol': cb_min_sol,
    'cb_max_sol': cb_max_sol,
    'm_load_dir': m_load_dir,
    'th_sol_filename': th_sol_filename,
    'domain_lims': domain
}

print('\n --- --- --- --- --- --- --- --- --- --- --- --- --- --- \n')
print(' Calling solve_td_maxwell_pbm() with params = {}'.format(params))
print('\n --- --- --- --- --- --- --- --- --- --- --- --- --- --- \n')

diags = solve_td_maxwell_pbm(**params)

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

time_count(t_stamp_full, msg='full program')
