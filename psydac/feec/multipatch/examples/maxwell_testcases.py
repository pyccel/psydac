import os
import numpy as np
from psydac.feec.multipatch.examples.hcurl_source_pbms_conga_2d import solve_hcurl_source_pbm
from psydac.feec.multipatch.utilities                   import time_count, FEM_sol_fn, get_run_dir, get_plot_dir, get_mat_dir, get_sol_dir, diag_fn
from psydac.feec.multipatch.utils_conga_2d              import write_diags_to_file

t_stamp_full = time_count()

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
#
# main test-cases used for the ppc paper:

# test_case = 'maxwell_hom_eta=50'   # used in paper
# test_case = 'maxwell_hom_eta=170'   # used in paper
test_case = 'maxwell_inhom'   # used in paper

compute_ref_sol = False  # (not needed for inhomogeneous test-case, as exact solution is known)

#
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


# numerical parameters:
domain_name = 'pretzel_f'

source_proj = 'tilde_Pi'
# other values are: 
#   source_proj = 'P_L2'    # L2 projection in broken space
#   source_proj = 'P_geom'  # geometric projection (primal commuting proj)


if compute_ref_sol:
    nc_s = [20]
    deg_s = [6]
    save_sol = True

else:
    nc_s = [4,8,16]
    deg_s = [3] #[2,3,4,5]
    save_sol = False

# nc_s = [4]
# deg_s = [2]
# nc_s = [8]
# deg_s = [4]

if test_case == 'maxwell_hom_eta=50':
    homogeneous = True
    source_type = 'elliptic_J'
    omega = np.sqrt(50) # source time pulsation

    cb_min_sol = 0
    cb_max_sol = 1

    # ref solution (no exact solution)
    ref_nc = 20
    ref_deg = 6

elif test_case == 'maxwell_hom_eta=170':
    homogeneous = True
    source_type = 'elliptic_J'
    omega = np.sqrt(170) # source time pulsation

    cb_min_sol = 0
    cb_max_sol = 1

    # ref solution (no exact solution)
    ref_nc = 20
    ref_deg = 6

    
elif test_case == 'maxwell_inhom':

    homogeneous = False # 
    source_type = 'manu_maxwell_inhom'
    omega = np.pi 

    cb_min_sol = 0
    cb_max_sol = 1

    # dummy ref solution (there is an exact solution)
    ref_nc = 2
    ref_deg = 2

else:
    raise ValueError(test_case)

case_dir = test_case
ref_case_dir = case_dir

roundoff = 1e4
eta = int(-omega**2 * roundoff)/roundoff

project_sol = False # True #   (use conf proj of solution for visualization)
gamma_h = 10

#
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

common_diag_filename = './'+case_dir+'_diags.txt'

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
        # to save the FEM sol
        if save_sol:
            sol_dir = get_sol_dir(case_dir, domain_name, nc, deg)
            sol_filename = sol_dir+'/'+FEM_sol_fn(source_type=source_type, source_proj=source_proj)
            if not os.path.exists(sol_dir):
                os.makedirs(sol_dir)
        else:
            sol_filename = ''
        # to load the ref FEM sol
        sol_ref_dir = get_sol_dir(ref_case_dir, domain_name, ref_nc, ref_deg)
        sol_ref_filename = sol_ref_dir+'/'+FEM_sol_fn(source_type=source_type, source_proj=source_proj)

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
            plot_source=True,
            project_sol=project_sol,
            gamma_h=gamma_h,
            plot_dir=plot_dir,
            hide_plots=True,
            skip_plot_titles=True,
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