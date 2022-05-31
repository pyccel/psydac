import os
import numpy as np
from psydac.feec.multipatch.examples.td_maxwell_conga_2d import solve_td_maxwell_pbm
from psydac.feec.multipatch.utilities                   import time_count, FEM_sol_fn, get_run_dir, get_plot_dir, get_mat_dir, get_sol_dir, diag_fn
from psydac.feec.multipatch.utils_conga_2d              import write_diags_to_file

t_stamp_full = time_count()

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
#
# test-case and numerical parameters:

E0_type = 'zero' # 'pulse' # 'th_sol' # 
E0_proj = 'P_L2' # 'P_geom' # 

# source_type = 'pulse' # 'elliptic_J' # 'cf_pulse' # 
source_type = 'Il_pulse'    #Issautier-like pulse
source_proj = 'P_L2' # 'P_geom' #
source_is_harmonic = False # True # 
filter_source =  True # False # 

project_sol =  True #  False #
gamma_h = 0

# nc_s = [2,4,8,16]
# deg_s = [2,3,4,5]

nc_s = [16]
deg_s = [5]
deg_s = [3]

# nc_s = [4]
# deg_s = [2]

# nc_s = [20]
# deg_s = [3]
# deg_s = [6]

# nc_s = [8]
# # deg_s = [5]
# deg_s = [3]


# Nt_pp = 400 # time steps per time period  # CFL should be decided automatically...
# Nt_pp = 500 # time steps per time period  # CFL should be decided automatically...
Nt_pp = None
cfl = .8

# Nt_pp = 54  # TMP for 20/6



if source_is_harmonic:

    nb_t_periods = 100 # final time: T = nb_t_periods * (2*pi/omega)
    Nt_pp = 100 # time steps per time period  # CFL should be decided automatically...
    omega = np.sqrt(50) # source time pulsation

    case_dir = 'td_maxwell_harmonic_J_'+source_proj+'_E0_' + E0_type

    if E0_type == 'th_sol':
        case_dir += '_PE0=' + E0_proj  
    else:
        assert E0_type == 'zero'

    cb_min_sol = 0
    cb_max_sol = 1

    plot_time_ranges = [
        [[0,nb_t_periods], None] #Nt_pp//10]
        # [[95,100], Nt_pp//10]
        # [[495,500], Nt_pp//5]
        ]

else: 
    
    case_dir = 'td_maxwell_E0_' + E0_type + '_' + E0_proj + '_J_' + source_type + '_' + source_proj

    if not project_sol:
        case_dir += '_E_noproj'

    # code will use t_period = (2*pi/omega), relevant for plotting if no source
    omega = 5*2*np.pi 

    if E0_type == 'pulse':
        nb_t_periods = 25 # final time: T = nb_t_periods * t_period
        cb_min_sol = 0
        cb_max_sol = 8

    elif E0_type == 'zero':
        # assert source_type == 'pulse'
        # case_dir += '_J_'+source_type+'_'+source_proj
        if filter_source:
            case_dir += '_Jfilter'
        else:
            case_dir += '_Jnofilter'
        
        if source_type == '_Il_pulse':
            nb_t_periods = 50  #  # final time: T = nb_t_periods * t_period
            
            case_dir += '_nb_tau={}'.format(nb_t_periods)+'_GE'

            # Nt_pp = 20
            cb_min_sol = None
            cb_max_sol = None

        else:
            nb_t_periods = 16 # final time: T = nb_t_periods * t_period   # 16 for paper ?

            cb_min_sol = 0
            cb_max_sol = .3
    
    else:
        raise ValueError


    if source_type == 'Il_pulse':
        plot_time_ranges = [
            [[nb_t_periods-1,nb_t_periods], None]
            # [[95,100], Nt_pp//10]
            # [[495,500], Nt_pp//5]
            ]
    else:
        plot_time_ranges = [
            [[0,nb_t_periods], None] #Nt_pp//10]
            # [[95,100], Nt_pp//10]
            # [[495,500], Nt_pp//5]
            ]

# plotting ranges:
#   we give a list of ranges and plotting period: [[t_start, t_end], nt_plot_period]
#   with 
#       t_start, t_end: in time periods units
#   and 
#       nt_plot_period: nb of time steps between two plots
# plot_time_ranges = [[[0,1], 2], [[90,100], Nt_pp//2]]
# plot_time_ranges = [[[10,20], Nt_pp//2], [[90,100], Nt_pp//2]]

# diag_dtau: tau period for intermediate diags plotting
diag_dtau = max(1,nb_t_periods//10)


# ref_case_dir = 'maxwell_hom_eta=50'    

# omega = np.sqrt(170) # source time pulsation

# case_dir = 'maxwell_hom_eta=50'

ref_case_dir = case_dir
domain_name = 'pretzel_f'

# domain_name = 'annulus_4'

conf_proj = 'GSP'

# ref solution (not used here)
# ref_nc = 20
# ref_deg = 6
ref_nc = 2
ref_deg = 2

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
            'ref_nc': ref_nc,
            'ref_deg': ref_deg,
        }
        # backend_language = 'numba'
        backend_language='pyccel-gcc'

        run_dir = get_run_dir(domain_name, nc, deg, source_type=source_type, conf_proj=conf_proj)
        plot_dir = get_plot_dir(case_dir, run_dir)
        diag_filename = plot_dir+'/'+diag_fn(source_type=source_type, source_proj=source_proj)

        # to save and load matrices
        m_load_dir = get_mat_dir(domain_name, nc, deg)

        if E0_type == 'th_sol':
            # initial E0 will be loaded from time-harmonic FEM solution
            th_case_dir = 'maxwell_hom_eta=50'
            th_sol_dir = get_sol_dir(th_case_dir, domain_name, nc, deg)
            th_sol_filename = th_sol_dir+'/'+FEM_sol_fn(source_type=source_type, source_proj=source_proj)
        else:
            # no initial solution to load
            th_sol_filename = ''

        # # to save the FEM sol
        # sol_dir = get_sol_dir(case_dir, domain_name, nc, deg)
        # sol_filename = sol_dir+'/'+FEM_sol_fn(source_type=source_type, source_proj=source_proj)
        # if not os.path.exists(sol_dir):
        #     os.makedirs(sol_dir)
        # to load the ref FEM sol
        sol_ref_dir = get_sol_dir(ref_case_dir, domain_name, ref_nc, ref_deg)
        sol_ref_filename = sol_ref_dir+'/'+FEM_sol_fn(source_type=source_type, source_proj=source_proj)

        print('\n --- --- --- --- --- --- --- --- --- --- --- --- --- --- \n')
        print(' Calling solve_hcurl_source_pbm() with params = {}'.format(params))
        print('\n --- --- --- --- --- --- --- --- --- --- --- --- --- --- \n')
        
        # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
        # calling solver for time domain maxwell
        
        diags = solve_td_maxwell_pbm(
            nc=nc, deg=deg,
            Nt_pp=Nt_pp,
            source_is_harmonic=source_is_harmonic,
            nb_t_periods=nb_t_periods,
            omega=omega,
            domain_name=domain_name,
            E0_type=E0_type,
            E0_proj=E0_proj,
            source_type=source_type,
            source_proj=source_proj,
            backend_language=backend_language,
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
            sol_ref_filename=sol_ref_filename,
            ref_nc=ref_nc,
            ref_deg=ref_deg,    
        )

        #
        # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

        write_diags_to_file(diags, script_filename=__file__, diag_filename=diag_filename, params=params)
        write_diags_to_file(diags, script_filename=__file__, diag_filename=common_diag_filename, params=params)

time_count(t_stamp_full, msg='full program')