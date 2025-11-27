#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
    Runner script for solving the time-domain Maxwell problem.
"""

import numpy as np

from psydac.feec.multipatch.examples.timedomain_maxwell import solve_td_maxwell_pbm
from psydac.feec.multipatch.utilities import get_run_dir, get_plot_dir

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#

test_case = 'E0_pulse_no_source'  
# test_case = 'Issautier_like_source'
# J_proj_case = 'P_geom'
J_proj_case = 'P_L2'

#
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Parameters to be changed in the batch run
deg = 3

# Common simulation parameters
# domain_name = 'square_6'
# ncells = [4,4,4,4,4,4]
# domain_name = 'pretzel_f'

# non-conf domains
domain = [[0, np.pi], [0, np.pi]]  # interval in x- and y-direction
domain_name = 'refined_square'
# use isotropic meshes (probably with a square domain)
# 4x8= 64 patches
# care for the transpose
ncells =  np.array([[10, 10, 10],
                    [10, 20, 10],
                    [10, 10, 10]])

cfl_max = 0.8

# 'P_geom'  # projection used for initial E0 (B0 = 0 in all cases)
E0_proj = 'P_L2'
backend = 'pyccel-gcc'
project_sol = True  # whether cP1 E_h is plotted instead of E_h

# Parameters that depend on test case
if test_case == 'E0_pulse_no_source':

    E0_type = 'pulse_2'   # non-zero initial conditions
    source_type = 'zero'    # no current source
    source_omega = None
    final_time = 2  # wave transit time in domain is > 4
    dt_max = None

    plot_a_lot = True
    if plot_a_lot:
        plot_time_ranges = [[[0, final_time], 0.1]]
    else:
        plot_time_ranges = [
            [[0, 2], 0.1],
            [[final_time - 1, final_time], 0.1],
        ]

# TODO: check
elif test_case == 'Issautier_like_source':

    E0_type = 'zero'      # zero initial conditions
    source_type = 'Il_pulse'
    source_omega = None
    final_time = 20
    dt_max = None

    if deg == 3 and final_time == 20:

        plot_time_ranges = [
            [[1.9, 2], 0.1],
            [[4.9, 5], 0.1],
            [[9.9, 10], 0.1],
            [[19.9, 20], 0.1],
        ]

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

case_dir = 'tdmaxwell_' + test_case + '_J_proj=' + J_proj_case 

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

run_dir = get_run_dir(
    domain_name,
    sum(ncells),
    deg,
    source_type=source_type,
    conf_proj="")

plot_dir = get_plot_dir(case_dir, run_dir)


#
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
    'project_sol': project_sol,
    'filter_source': filter_source,
    'E0_type': E0_type,
    'E0_proj': E0_proj,
    'plot_dir': plot_dir,
    'plot_time_ranges': plot_time_ranges,
    'domain_lims': domain
}

print('\n --- --- --- --- --- --- --- --- --- --- --- --- --- --- \n')
print(' Calling solve_td_maxwell_pbm() with params = {}'.format(params))
print('\n --- --- --- --- --- --- --- --- --- --- --- --- --- --- \n')

solve_td_maxwell_pbm(**params)
