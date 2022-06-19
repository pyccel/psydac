import os
import numpy as np
# from psydac.feec.multipatch.examples.hcurl_eigen_pbms_conga_2d import hcurl_solve_eigen_pbm
from psydac.feec.multipatch.examples.hcurl_eigen_pbms_conga_2d_nc import hcurl_solve_eigen_pbm
from psydac.feec.multipatch.utilities                   import time_count, get_run_dir, get_plot_dir, get_mat_dir, get_sol_dir, diag_fn
from psydac.feec.multipatch.utils_conga_2d              import write_diags_to_file

t_stamp_full = time_count()

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
#
# test-case and numerical parameters:
# '2patch_nc'  :  2-patch domain for the square, with non-conforming grids: left patch is fine (2*nc), right patch is coarse (nc)
# '2patch_conf'  :  2-patch domain for the square, with conforming grids (both patches are 'coarse')

operator = 'curl-curl' # 'grad-div' # 
domain_name = '2patch_conf' # '2patch_nc' # 'curved_L_shape' # 'pretzel_f' # 
# domain_name = '2patch_nc' # '2patch_conf_mapped' # '2patch_nc_mapped' #'2patch_conf_mapped' # '2patch_nc' # 'curved_L_shape' # 'pretzel_f' # 

# nc_s = [2,4,8,16]
# deg_s = [2,3,4,5]

# nc_s = [8]
# deg_s = [4]

nc_s = [16]
deg_s = [3]
# nc_s = [20]
# nc_s = [20]
# deg_s = [5]

# nc_s = [4,8,16,20]
# deg_s = [3]

gamma_h = 0
generalized_pbm = True  # solves generalized eigenvalue problem with:  B(v,w) = <Pv,Pw> + <(I-P)v,(I-P)w> in rhs

if operator == 'curl-curl':
    nu=0
    mu=1
elif operator == 'grad-div':
    nu=1
    mu=0
else:
    raise ValueError(operator)

case_dir = 'eigenpbm_'+operator
ref_case_dir = case_dir

cb_min_sol = None
cb_max_sol = None

ref_sigmas = [
]
sigma = 8
nb_eigs_solve = 8
nb_eigs_plot = 5 
skip_eigs_threshold = 1e-7

if domain_name == 'curved_L_shape':    
    if operator == 'curl-curl':
        # ref eigenvalues from Monique Dauge benchmark page
        ref_sigmas = [
            0.181857115231E+01,
            0.349057623279E+01,
            0.100656015004E+02,
            0.101118862307E+02,
            0.124355372484E+02,
            ]
        sigma = 10
        nb_eigs_solve = 10 
        nb_eigs_plot = 5 

elif domain_name in ['pretzel_f']:
    if operator == 'curl-curl':
        # ref sigmas computed with nc=20 and deg=6 and gamma = 0 (and generalized ev-pbm)
        ref_sigmas = [
            0.1795339843,
            0.1992261261,
            0.6992717244, 
            0.8709410438, 
            1.1945106937, 
            1.2546992683,
        ]

        sigma = .8
        # sigma = .6
        nb_eigs_solve = 10 
        nb_eigs_plot = 5 

elif domain_name in ['2patch_nc', '2patch_conf']:
    assert operator == 'curl-curl'

    ref_sigmas = [
        1,
        1,
        2, 
        4, 
        4, 
        5,
        5,
        8,
        9,
        9,
    ]

    sigma = 5
    # sigma = .6
    nb_eigs_solve = 10 
    nb_eigs_plot = 10

#
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

common_diag_filename = './'+case_dir+'_diags.txt'

for nc in nc_s:
    for deg in deg_s:

        params = {
            'domain_name': domain_name,
            'operator': operator,
            'mu': mu,
            'nu': nu,
            'nc': nc,
            'deg': deg,            
            'gamma_h': gamma_h,
            'generalized_pbm': generalized_pbm,
            'nb_eigs_solve': nb_eigs_solve,
            'skip_eigs_threshold': skip_eigs_threshold
        }

        print(params)

        # backend_language = 'numba'
        backend_language='pyccel-gcc'

        run_dir = get_run_dir(domain_name, nc, deg)
        plot_dir = get_plot_dir(case_dir, run_dir)
        diag_filename = plot_dir+'/'+diag_fn()

        # to save and load matrices
        m_load_dir = get_mat_dir(domain_name, nc, deg)

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

        diags = hcurl_solve_eigen_pbm(
            nc=nc, deg=deg,
            gamma_h=gamma_h,
            generalized_pbm=generalized_pbm,
            nu=nu,
            mu=mu,
            sigma=sigma,
            ref_sigmas=ref_sigmas,
            skip_eigs_threshold=skip_eigs_threshold,
            nb_eigs_solve=nb_eigs_solve,
            nb_eigs_plot=nb_eigs_plot,
            domain_name=domain_name,
            backend_language=backend_language,
            plot_dir=plot_dir,
            hide_plots=True,
            m_load_dir=m_load_dir,
        )

        # diags = solve_hcurl_source_pbm(
        #     nc=nc, deg=deg,
        #     eta=eta,
        #     nu=0,
        #     mu=1,
        #     domain_name=domain_name,
        #     source_type=source_type,
        #     source_proj=source_proj,
        #     backend_language=backend_language,
        #     plot_source=True,
        #     project_sol=project_sol,
        #     gamma_h=gamma_h,
        #     filter_source=filter_source,
        #     plot_dir=plot_dir,
        #     hide_plots=True,
        #     cb_min_sol=cb_min_sol, 
        #     cb_max_sol=cb_max_sol,
        #     m_load_dir=m_load_dir,
        #     sol_filename=sol_filename,
        #     sol_ref_filename=sol_ref_filename,
        #     ref_nc=ref_nc,
        #     ref_deg=ref_deg,    
        # )

        #
        # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

        write_diags_to_file(diags, script_filename=__file__, diag_filename=diag_filename, params=params)
        write_diags_to_file(diags, script_filename=__file__, diag_filename=common_diag_filename, params=params)

time_count(t_stamp_full, msg='full program')