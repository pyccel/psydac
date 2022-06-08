from multiprocessing.sharedctypes import Value
import os
import numpy as np
from psydac.feec.multipatch.examples.td_maxwell_conga_2d import solve_td_maxwell_pbm
from psydac.feec.multipatch.utilities                   import time_count, FEM_sol_fn, get_run_dir, get_plot_dir, get_mat_dir, get_sol_dir, diag_fn
from psydac.feec.multipatch.utils_conga_2d              import write_diags_to_file

t_stamp_full = time_count()

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
#
# main test-cases and parameters used for the ppc paper:

# test_case = 'E0_pulse_no_source'   # used in paper
test_case = 'Issautier_like_source'  # used in paper
# test_case = 'transient_to_harmonic'  # actually, not used in paper

J_proj_case = 'P_geom'
# J_proj_case = 'P_L2'
# J_proj_case = 'tilde Pi_1' 

#
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

nc_s = [8] #16]
deg_s = [3]

# we use a t_period = (2*pi/omega) (sometimes denoted tau)
# this is relevant for oscillating sources but also for plotting
omega = 5*2*np.pi 
nb_t_periods = 10  #  # final time: T = nb_t_periods * t_period

# plotting ranges:
#   we give a list of ranges and plotting period: [[t_start, t_end], nt_plot_period]
#   with 
#       t_start, t_end: in t_periods (tau) units
#   and 
#       nt_plots_pp: nb of plots per period
plot_time_ranges = [
    [[nb_t_periods-1, nb_t_periods], 4]
    ]

# nb of time steps per period (if None, will be set from cfl)
Nt_pp = None
cfl = .8  

if test_case == 'E0_pulse_no_source':
    E0_type = 'pulse'
    source_type = 'zero'    # Issautier-like pulse
    source_is_harmonic = False
    
    nb_t_periods = 25 # final time: T = nb_t_periods * t_period
    plot_a_lot = False # True # 
    if plot_a_lot:
        plot_time_ranges = [
            [[0, nb_t_periods], 4]
        ]
    else:
        # plot only a few snapshots
        plot_time_ranges = [
            [[0, 2], 2],
            [[nb_t_periods-2, nb_t_periods], 2],
        ]

    cb_min_sol = 0
    cb_max_sol = 5

elif test_case == 'Issautier_like_source':
    E0_type = 'zero'
    source_type = 'Il_pulse_pp'  # 'Il_pulse' has a coarser rho
    source_is_harmonic = False

    nb_t_periods = 100  #  # final time: T = nb_t_periods * t_period
            
    if J_proj_case == 'P_geom':    
        cb_min_sol = None #
        cb_max_sol = None #
    else: 
        cb_min_sol = 0 # 
        cb_max_sol = .3 #

    if deg_s == [3] and nb_t_periods==100:
            
        plot_time_ranges = [
            [[9.5,10], 2],
            [[24.5,25], 2],
            [[49.5,50], 2],
            [[99.5,100], 2],
            ]

            # plot_time_ranges = [
            #     ]
            # if nc_s == [8]:
            #     Nt_pp = 10

elif test_case == 'transient_to_harmonic':
    E0_type = 'th_sol'
    source_type = 'elliptic_J'
    source_is_harmonic = True

    omega = np.sqrt(50) # source time pulsation
    nb_t_periods = 100 # final time: T = nb_t_periods * (2*pi/omega)
    # Nt_pp = 100 # time steps per time period  # CFL should be decided automatically...

    plot_time_ranges = [
        [[nb_t_periods-2,nb_t_periods], 4]
        ]
    cb_min_sol = 0
    cb_max_sol = 1

else:
    raise ValueError(test_case)

# projection used for initial E0 (B0 = 0 in all cases)
E0_proj = 'P_L2' # 'P_geom' # 

# whether cP1 E_h is plotted instead of E_h:
project_sol =  True #  False #   

# projection used for the source J
if J_proj_case == 'P_geom':
    source_proj = 'P_geom'
    filter_source =  False
elif J_proj_case == 'P_L2':
    source_proj = 'P_L2' 
    filter_source = False

elif J_proj_case == 'tilde Pi_1':
    source_proj = 'P_L2' 
    filter_source =  True 

else:
    raise ValueError(J_proj_case)

# multiplicative parameter for the quadrature order of the bilinear/linear forms discretizations:
quad_param = 4

# jump dissipation parameter (not used in paper)
gamma_h = 0

case_dir = '0_td_maxwell_' + test_case + '_J_proj=' + J_proj_case + '_qp{}'.format(quad_param)
if filter_source:
    case_dir += '_Jfilter'
else:
    case_dir += '_Jnofilter'
if not project_sol:
    case_dir += '_E_noproj'

case_dir += '_nb_tau={}'.format(nb_t_periods)

# diag_dtau: tau period for intermediate diags (time curves) plotting
diag_dtau = max(1,nb_t_periods//10)

domain_name = 'pretzel_f'

# type of conforming projection operators (averaging B-spline or Geometric-splines coefficients)
conf_proj = 'GSP' # 'BSP' # 

#
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

common_diag_filename = './'+case_dir+'_diags.txt'

for nc in nc_s:
    for deg in deg_s:

        params = {
            'domain_name': domain_name,
            'nc': nc,
            'deg': deg,
            'homogeneous': True,
            'E0_type': E0_type,
            'E0_proj': E0_proj,
            'source_type': source_type,
            'source_is_harmonic ': source_is_harmonic,
            'source_proj': source_proj,
            'conf_proj': conf_proj,
            'filter_source': filter_source, 
            'project_sol': project_sol,
            'omega': omega,
            'gamma_h': gamma_h,
            'quad_param': quad_param,
        }
        # backend_language = 'numba'
        backend_language='pyccel-gcc'

        run_dir = get_run_dir(domain_name, nc, deg, source_type=source_type, conf_proj=conf_proj)
        plot_dir = get_plot_dir(case_dir, run_dir)
        diag_filename = plot_dir+'/'+diag_fn(source_type=source_type, source_proj=source_proj)

        # to save and load matrices
        m_load_dir = get_mat_dir(domain_name, nc, deg, quad_param=quad_param)

        if E0_type == 'th_sol':
            # initial E0 will be loaded from time-harmonic FEM solution
            th_case_dir = 'maxwell_hom_eta=50'
            th_sol_dir = get_sol_dir(th_case_dir, domain_name, nc, deg)
            th_sol_filename = th_sol_dir+'/'+FEM_sol_fn(source_type=source_type, source_proj=source_proj)
        else:
            # no initial solution to load
            th_sol_filename = ''

        print('\n --- --- --- --- --- --- --- --- --- --- --- --- --- --- \n')
        print(' Calling solve_hcurl_source_pbm() with params = {}'.format(params))
        print('\n --- --- --- --- --- --- --- --- --- --- --- --- --- --- \n')
        
        # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
        # calling solver for time domain maxwell
        
        diags = solve_td_maxwell_pbm(
            nc=nc, deg=deg,
            Nt_pp=Nt_pp,
            cfl=cfl,
            source_is_harmonic=source_is_harmonic,
            nb_t_periods=nb_t_periods,
            omega=omega,
            domain_name=domain_name,
            E0_type=E0_type,
            E0_proj=E0_proj,
            source_type=source_type,
            source_proj=source_proj,
            backend_language=backend_language,
            quad_param=quad_param,
            plot_source=True,
            plot_divE=True,
            conf_proj=conf_proj,
            project_sol=project_sol,
            gamma_h=gamma_h,
            filter_source=filter_source,
            plot_dir=plot_dir,
            plot_time_ranges=plot_time_ranges,
            diag_dtau=diag_dtau,
            hide_plots=True,
            cb_min_sol=cb_min_sol, 
            cb_max_sol=cb_max_sol,
            m_load_dir=m_load_dir,
            th_sol_filename=th_sol_filename,
        )

        #
        # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

        write_diags_to_file(diags, script_filename=__file__, diag_filename=diag_filename, params=params)
        write_diags_to_file(diags, script_filename=__file__, diag_filename=common_diag_filename, params=params)

time_count(t_stamp_full, msg='full program')