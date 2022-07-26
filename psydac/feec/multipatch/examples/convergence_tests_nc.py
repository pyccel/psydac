import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from psydac.feec.multipatch.examples.multipatch_non_conf_examples import hcurl_solve_eigen_pbm_multipatch_nc

from psydac.feec.multipatch.utilities                   import time_count, get_run_dir, get_plot_dir, get_mat_dir, get_sol_dir, diag_fn
from psydac.feec.multipatch.utils_conga_2d              import write_diags_to_file

from psydac.api.postprocessing import OutputManager, PostProcessManager


def run_convergence_tests(domain_name, degree):
    t_stamp_full = time_count()

    operator = 'curl-curl'
    if domain_name == 'curved_L_shape':
        domain=[[1, 3],[0, np.pi/4]]
        ncells = [np.array([[None, 2],
                    [2, 2]]),
            np.array([[None, None, 2, 2],
                   [None, None, 4, 2],
                   [ 2,    4,   4, 2],
                   [ 2,    2,   2, 2]]),
            np.array([[None, None, None, 2, 2, 2],
                    [None, None, None, 4, 4, 2],
                   [None, None, None, 8, 4, 2],
                   [ 2, 4,    8,   8, 4, 2],
                   [ 2, 4,    4,   4, 4, 2],
                   [ 2, 2,    2,   2, 2, 2]]),
            np.array([[None, None, None, None, 2, 2, 2, 2],
                   [None, None, None, None, 4, 4, 4, 2],
                   [None, None, None, None, 8, 8, 4, 2],
                   [None, None, None, None, 16, 8, 4, 2],
                   [ 2,    4,   8, 16, 16, 8, 4, 2],
                   [ 2,    4,   8, 8, 8,   8, 4, 2],
                   [ 2,    4,   4, 4, 4, 4, 4, 2],
                   [ 2,    2,   2, 2, 2, 2, 2, 2]])
        ]

        levels = len(ncells)

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

    gamma_h = 0
    generalized_pbm = True  # solves generalized eigenvalue problem with:  B(v,w) = <Pv,Pw> + <(I-P)v,(I-P)w> in rhs

    if operator == 'curl-curl':
        nu=0
        mu=1
    else:
        raise ValueError(operator)

    case_dir = 'conv_runs'
    ref_case_dir = case_dir

    global_dof_errors = []
    dof_sizes = []

    for lvl in range(levels):
        print("level: {}".format(lvl))
        params = {
        'domain_name': domain_name,
        'domain': domain,
        'operator': operator,
        'mu': mu,
        'nu': nu,
        'ncells': ncells[lvl],
        'degree': degree,            
        'gamma_h': gamma_h,
        'generalized_pbm': generalized_pbm,
        'nb_eigs_solve': nb_eigs_solve,
        'skip_eigs_threshold': skip_eigs_threshold
        }
        common_diag_filename = './'+case_dir+'_diags.txt'

        # backend_language = 'numba'
        backend_language='pyccel-gcc'

        dims = ncells[lvl].shape
        sz = ncells[lvl][ncells[lvl] != None].sum()
        run_dir = domain_name+("_%dx%d_" %dims)+'patches_'+'size_{}'.format(sz) #get_run_dir(domain_name, nc, deg)
        plot_dir = get_plot_dir(case_dir, run_dir)
        diag_filename = plot_dir+'/'+diag_fn()
        common_diag_filename = './'+case_dir+'_diags.txt'

        # to save and load matrices
        #m_load_dir = get_mat_dir(domain_name, nc, deg)
        m_load_dir = None    

        diags, eigenvalues = hcurl_solve_eigen_pbm_multipatch_nc(
                            ncells=ncells[lvl], degree=degree,
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

        error_vec = []
        if ref_sigmas is not None:
            errors = []
            n_errs = min(len(ref_sigmas), len(eigenvalues))
            for k in range(n_errs):
                diags['error_{}'.format(k)] = abs(eigenvalues[k]-ref_sigmas[k])
                error_vec.append(abs(eigenvalues[k]-ref_sigmas[k]))

        global_dof_errors.append(error_vec)
        dof_sizes.append(sz)
        write_diags_to_file(diags, script_filename=__file__, diag_filename=diag_filename, params=params)
        write_diags_to_file(diags, script_filename=__file__, diag_filename=common_diag_filename, params=params)

    time_count(t_stamp_full, msg='full program')

    #Quickly plot error curves
    fig, ax = plt.subplots( )
    ax.loglog()
    ax.set_title("l1 error of eigenvalues over dofs")
    for i in range( len(global_dof_errors[0])):
        error_i = [ global_dof_errors[lvl][i] for lvl in range(len(global_dof_errors))]
        ax.plot(dof_sizes, error_i, label = "e-val {}".format(i))
    
    ax.plot(dof_sizes, [ (0.1/dof_sizes[i])**1 for i in range(len(dof_sizes))], label="order 1", linestyle='--')
    ax.plot(dof_sizes, [ (0.1/dof_sizes[i])**2 for i in range(len(dof_sizes))], label="order 2", linestyle='--')
    ax.plot(dof_sizes, [ (0.75/dof_sizes[i])**3 for i in range(len(dof_sizes))], label="order 3", linestyle='--')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('./plots/'+case_dir+"/errors", dpi=300)

if __name__ == '__main__':
    domain_name = 'curved_L_shape'
    degree = [2,2]
    run_convergence_tests(domain_name, degree)