#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
    Runner script for solving the H(curl) source problem.
"""

import os
import numpy as np
from psydac.feec.multipatch.examples.hcurl_source_pbms_conga_2d import solve_hcurl_source_pbm

from psydac.feec.multipatch.utilities import time_count, FEM_sol_fn, get_run_dir, get_plot_dir, get_mat_dir, get_sol_dir, diag_fn
from psydac.feec.multipatch.utils_conga_2d import write_diags_to_file

t_stamp_full = time_count()

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#
# main test-cases used for the ppc paper:

# test_case = 'maxwell_hom_eta=50'   # used in paper
#test_case = 'maxwell_hom_eta=170'   # used in paper
test_case = 'maxwell_inhom'   # used in paper

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# numerical parameters:
domain_name = 'pretzel_f'
# domain_name = 'curved_L_shape'

# currently only 'tilde_Pi' is implemented
source_proj = 'tilde_Pi'

# nc_s = [np.array([16 for _ in range(18)])]

# corners in pretzel [2, 2, 2*,2*, 2, 1, 1, 1, 1, 1, 0, 0, 1, 2*, 2*, 2, 0, 0 ]
nc_s = [np.array([16, 16, 16, 16, 16, 8, 8, 8, 8,
                 8, 8, 8, 8, 16, 16, 16, 8, 8])]
# nc_s = [10]
# refine handles only
# nc_s = [np.array([16, 16, 16, 16, 16, 8, 8, 8, 8, 4, 2, 2, 4, 16, 16, 16, 2, 2])]

# refine source
# nc_s = [np.array([32, 8, 8, 32, 32, 32, 32, 8, 8, 8, 8, 8, 8, 32, 8, 8, 8, 8])]

deg_s = [3]

if test_case == 'maxwell_hom_eta=50':
    homogeneous = True
    source_type = 'elliptic_J'
    omega = np.sqrt(50)  # source time pulsation

elif test_case == 'maxwell_hom_eta=170':
    homogeneous = True
    source_type = 'elliptic_J'
    omega = np.sqrt(170)  # source time pulsation

elif test_case == 'maxwell_inhom':
    homogeneous = False
    source_type = 'manu_maxwell_inhom'
    omega = np.pi

else:
    raise ValueError(test_case)

case_dir = test_case

eta = int(-omega**2 * roundoff) / roundoff

project_sol = True  # True #   (use conf proj of solution for visualization)
gamma_h = 10

#
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

common_diag_filename = './' + case_dir + '_diags.txt'

for nc in nc_s:
    for deg in deg_s:

        params = {
            'domain_name': domain_name,
            'nc': nc,
            'deg': deg,
            'homogeneous': homogeneous,
            'source_type': source_type,
            'source_proj': source_proj,
            'project_sol': project_sol,
            'omega': omega,
            'gamma_h': gamma_h,
        }
        # backend_language = 'numba'
        backend_language = 'pyccel-gcc'

        run_dir = get_run_dir(domain_name, nc, deg, source_type=source_type)
        plot_dir = get_plot_dir(case_dir, run_dir)
        diag_filename = plot_dir + '/' + \
            diag_fn(source_type=source_type, source_proj=source_proj)



        print('\n --- --- --- --- --- --- --- --- --- --- --- --- --- --- \n')
        print(' Calling solve_hcurl_source_pbm() with params = {}'.format(params))
        print('\n --- --- --- --- --- --- --- --- --- --- --- --- --- --- \n')

        # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
        # calling solver for:
        #
        # find u in H(curl), s.t.
        #   A u = f             on \Omega
        #   n x u = n x u_bc    on \partial \Omega
        # with
        #   A u := eta * u  +  mu * curl curl u  -  nu * grad div u

        diags = solve_hcurl_source_pbm(
            nc=nc, deg=deg,
            eta=eta,
            nu=0,
            mu=1,
            domain_name=domain_name,
            source_type=source_type,
            source_proj=source_proj,
            backend_language=backend_language,
            project_sol=project_sol,
            gamma_h=gamma_h,
            plot_dir=plot_dir,
        )

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

time_count(t_stamp_full, msg='full program')
