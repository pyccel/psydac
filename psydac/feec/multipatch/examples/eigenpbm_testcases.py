from multiprocessing.sharedctypes import Value
import os
import numpy as np
from psydac.feec.multipatch.examples.hcurl_eigen_pbms_conga_2d import hcurl_solve_eigen_pbm
from psydac.feec.multipatch.utilities                   import time_count, FEM_sol_fn, get_run_dir, get_plot_dir, get_mat_dir, get_sol_dir, diag_fn
from psydac.feec.multipatch.utils_conga_2d              import write_diags_to_file

t_stamp_full = time_count()


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
#
# main test-cases used for the ppc paper:

# test_case = 'cc_eigenpbm_pretzel'   # used in paper
test_case = 'cc_eigenpbm_L_shape'   # used in paper

#
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
# numerical parameters:

# nc_s = [2,4,8,16]
# deg_s = [2,3,4,5]

# to plot the eigenmodes on the curved L-shaped domain
nc_s = [56]
deg_s = [6]

# to plot the eigenmodes on the pretzel domain
# nc_s = [20]
# deg_s = [6]

# nc_s = [16]
# deg_s = [3]


operator = 'curl-curl' # 'grad-div' # 
gamma_h = 0
generalized_pbm = True  # solves generalized eigenvalue problem with:  B(v,w) = <Pv,Pw> + <(I-P)v,(I-P)w> in rhs

if test_case == 'cc_eigenpbm_pretzel':

    domain_name = 'pretzel_f' 

    # ref sigmas computed with nc=20 and deg=6 and gamma = 0 (and generalized ev-pbm)
    ref_sigmas = [
        0.1795339843,
        0.1992261261,
        0.6992717244, 
        0.8709410438, 
        1.1945106937, 
        1.2546992683,
    ]

    sigma = .8    # we look for eigenvalues close to that value
    nb_eigs_solve = 10 
    nb_eigs_plot = 5 
    
elif test_case == 'cc_eigenpbm_L_shape':    
    
    domain_name = 'curved_L_shape'

    # ref eigenvalues from Monique Dauge benchmark page
    ref_sigmas = [
        0.181857115231E+01,
        0.349057623279E+01,
        0.100656015004E+02,
        0.101118862307E+02,
        0.124355372484E+02,
        ]
    sigma = 10    # we look for eigenvalues close to that value
    nb_eigs_solve = 10 
    nb_eigs_plot = 5 

else:

    raise ValueError(test_case)

# small eigenvalues will be treated as zero
skip_eigs_threshold = 1e-7

if operator == 'curl-curl':
    nu=0
    mu=1
elif operator == 'grad-div':
    nu=1
    mu=0
else:
    raise ValueError(operator)

case_dir = test_case
ref_case_dir = case_dir

cb_min_sol = None
cb_max_sol = None

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
            skip_plot_titles=True,
            m_load_dir=m_load_dir,
        )

        #
        # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

        write_diags_to_file(diags, script_filename=__file__, diag_filename=diag_filename, params=params)
        write_diags_to_file(diags, script_filename=__file__, diag_filename=common_diag_filename, params=params)

time_count(t_stamp_full, msg='full program')