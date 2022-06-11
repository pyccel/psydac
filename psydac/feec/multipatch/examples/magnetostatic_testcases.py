from cProfile import run
import os
import datetime
from unittest import case
import numpy as np
from psydac.feec.multipatch.examples.mixed_source_pbms_conga_2d import solve_magnetostatic_pbm
from psydac.feec.multipatch.utilities                   import time_count, FEM_sol_fn, get_run_dir, get_plot_dir, get_mat_dir, get_sol_dir, diag_fn
from psydac.feec.multipatch.utils_conga_2d              import write_diags_to_file
t_stamp_full = time_count()


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
#
# main test-cases used for the ppc paper:

# test_case = 'magnetostatic_metal'   # used in paper
test_case = 'magnetostatic_vacuum'   # used in paper

#
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

source_type = 'dipole_J'
source_proj = 'P_L2_wcurl_J'
assert source_proj in ['P_geom', 'P_L2', 'P_L2_wcurl_J']

domain_name = 'pretzel_f'
dim_harmonic_space = 3

# nc_s = [2,4,8,16]
# deg_s = [2,3,4,5]

# nc_s = [2,8]
nc_s = [16]   ## 
deg_s = [3]  ##

# nc_s = [20]
# deg_s = [6]

if test_case == 'magnetostatic_metal':
    bc_type = 'metallic'
    cb_min_sol = 0
    cb_max_sol = 0.08

elif test_case == 'magnetostatic_vacuum':
    bc_type = 'pseudo-vacuum'
    cb_min_sol = 0
    cb_max_sol = 0.1

else:
    raise ValueError(test_case)
    # domain_name = 'curved_L_shape'

case_dir = test_case
ref_case_dir = case_dir

# ref solution (if no exact solution)
ref_nc = 20
ref_deg = 6
# ref_nc = 2
# ref_deg = 2


#
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

common_diag_filename = './'+case_dir+'_diags.txt'

for nc in nc_s:
    for deg in deg_s:

        params = {
            'domain_name': domain_name,
            'nc': nc,
            'deg': deg,
            'bc_type': bc_type,
            'source_type': source_type,
            'source_proj': source_proj, 
            'ref_nc': ref_nc,
            'ref_deg': ref_deg,
        }

        # backend_language = 'numba'
        backend_language='pyccel-gcc'

        run_dir = get_run_dir(domain_name, nc, deg, source_type=source_type)
        plot_dir = get_plot_dir(case_dir, run_dir)
        diag_filename = plot_dir+'/'+diag_fn(source_type=source_type, source_proj=source_proj)

        # to save and load matrices
        m_load_dir = get_mat_dir(domain_name, nc, deg)
        # to save the FEM sol and diags
        sol_dir = get_sol_dir(case_dir, domain_name, nc, deg)
        sol_filename = sol_dir+'/'+FEM_sol_fn(source_type=source_type, source_proj=source_proj)
        if not os.path.exists(sol_dir):
            os.makedirs(sol_dir)

        # to load the ref FEM sol
        sol_ref_dir = get_sol_dir(ref_case_dir, domain_name, ref_nc, ref_deg)
        sol_ref_filename = sol_ref_dir+'/'+FEM_sol_fn(source_type=source_type, source_proj=source_proj)

        print('\n --- --- --- --- --- --- --- --- --- --- --- --- --- --- \n')
        print(' Calling solve_magnetostatic_pbm() with params = {}'.format(params))
        print('\n --- --- --- --- --- --- --- --- --- --- --- --- --- --- \n')

        # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
        # calling ms solver
        
        diags = solve_magnetostatic_pbm(
            nc=nc, deg=deg,
            domain_name=domain_name,
            source_type=source_type,
            source_proj=source_proj,
            bc_type=bc_type,
            backend_language=backend_language,
            dim_harmonic_space=dim_harmonic_space,
            plot_source=True,
            plot_dir=plot_dir,
            # plot_dir='./plots/magnetostatic_runs/'+run_dir,
            hide_plots=True,
            cb_min_sol=cb_min_sol,
            cb_max_sol=cb_max_sol,
            m_load_dir=m_load_dir,
            sol_filename=sol_filename,
            sol_ref_filename=sol_ref_filename,
            ref_nc=ref_nc,
            ref_deg=ref_deg,    
        )

        #
        # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

        write_diags_to_file(diags, script_filename=__file__, diag_filename=diag_filename, params=params)
        write_diags_to_file(diags, script_filename=__file__, diag_filename=common_diag_filename, params=params)

time_count(t_stamp_full, msg='full program')