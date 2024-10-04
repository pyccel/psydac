import os 
import numpy as np
import matplotlib.pyplot as plt

from psydac.feec.multipatch.examples.h1_source_pbms_conga_2d import solve_h1_source_pbm
from psydac.feec.multipatch.examples.hcurl_eigen_pbms_conga_2d import hcurl_solve_eigen_pbm

from sympde.utilities.utils import plot_domain
from psydac.feec.multipatch.multipatch_domain_utilities import build_cartesian_multipatch_domain

def poisson(base_dir):
    base_dir = base_dir + 'poisson_pretzel/'
    


    source_type = 'manu_poisson_elliptic'
    domain_name = 'pretzel_f'
    eta = 0
    mu = 1

    for u in [4, 8, 16]:
        for deg in [2, 3, 4, 5]:

            r = 2*u
            nc = np.array([r, u, u, r, r, r, r, u, u,
                        u, u, u, u, r, u, u, u, u])

            run_dir = '{}_{}_nc={}_deg={}/'.format(domain_name, source_type, nc, deg)
            plot_dir = base_dir + run_dir

            err = solve_h1_source_pbm(
                    nc=nc, deg=deg,
                    eta=eta,
                    mu=mu,
                    domain_name=domain_name,
                    source_type=source_type,
                    backend_language='pyccel-gcc',
                    plot_dir=plot_dir,
                    )

            print(nc, deg, err)

def hcurl_eigen(base_dir):
    method = 'feec'
    # method = 'dg'

    operator = 'curl-curl'
    nu = 0
    mu = 1

    generalized_pbm = True
    gamma_h = 0

    # domain_name = 'three_patch'
    # domain = [[0, np.pi], [0, np.pi]]
    # ncells = np.array([16, 32, 8])

    domain_name = 'curved_L_shape'
    domain = [[1, 3], [0, np.pi / 4]]  # interval in x- and y-direction
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


    base_dir = base_dir + 'hcurl_eigen_'+domain_name


    ncells = [np.array([[None, 4],
                            [4, 4]]),
        np.array([[None, None, 4, 4],
                    [None, None, 8, 4],
                    [ 4,    8,   8, 4],
                    [ 4,    4,   4, 4]]),
        np.array([[None, None, None,    4, 4, 4],
                   [None, None, None,    8, 8, 4],
                   [None, None, None,   16, 8, 4],
                   [4,     8,     16,   16, 8, 4],
                   [4,     8,     8,     8, 8, 4],
                   [4,     4,     4,     4, 4, 4]]),
        np.array([[None, None, None,  None,   4,   4,  4, 4],
                   [None, None, None,  None,   8,   8,  8, 4],
                   [None, None, None,  None,  16,  16,  8, 4],
                   [None, None, None,  None,  32,  16,  8, 4],
                   [4,     8,     16,    32,  32,  16,  8, 4],
                   [4,     8,     16,    16,  16,  16,  8, 4],
                   [4,     8,     8,     8,    8,   8,  8, 4],
                   [4,     4,     4,     4,    4,   4,  4, 4]])
    ]

    degree = [2, 3, 4]

    for nc in ncells:
        for deg in degree:

            run_dir = '_dims={}_nc={}_deg={}/'.format(nc.shape, nc[nc != None].sum(), deg)
            plot_dir = base_dir + run_dir

            deg = [deg, deg]

            diags, eigenvalues = hcurl_solve_eigen_pbm_nc(
                ncells=nc, degree=deg,
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
                backend_language='pyccel-gcc',
                plot_dir=plot_dir,
                hide_plots=True,
                m_load_dir=None,
            )

            for k in range(min(len(ref_sigmas), len(eigenvalues))):
                diags['error_{}'.format(k)] = abs(eigenvalues[k] - ref_sigmas[k])

            print(diags)

def hcurl_source(base_dir):
    domain_name = 'pretzel_f'
    source_proj = 'tilde_Pi'

    # corners in pretzel [2, 2, 2*,2*, 2, 1, 1, 1, 1, 1, 0, 0, 1, 2*, 2*, 2, 0, 0 ]
    # nc_s = [np.array([16, 16, 16, 16, 16, 8, 8, 8, 8,
    #                 8, 8, 8, 8, 16, 16, 16, 8, 8])]

    # refine handles only
    # nc_s = [np.array([16, 16, 16, 16, 16, 8, 8, 8, 8, 4, 2, 2, 4, 16, 16, 16, 2, 2])]

    # refine source
    # nc_s = [np.array([32, 8, 8, 32, 32, 32, 32, 8, 8, 8, 8, 8, 8, 32, 8, 8, 8, 8])]

    test_case = 'maxwell_inhom'

    homogeneous = False
    source_type = 'manu_maxwell_inhom'
    omega = np.pi


    eta = -omega**2 

    project_sol = True  #   (use conf proj of solution for visualization)
    gamma_h = 10

    for r in [1,2,3,4]:
        for deg in [2,3,4,5]:
            print(deg, r)
            # nc = np.array([r, r, r, r, r, r, r, r, r,
            #          r, r, r, r, r, r, r, r, r])

            # refinement: 

            nc = np.array([2**(r+1), 2**(r),   2**(r+1), 2**(r), 2**(r+1),
                        2**(r),   2**(r+1),   2**(r),   2**(r+1),   2**(r),
                        2**(r+1),   2**(r),   2**(r+1),   2**(r), 2**(r+1), 
                        2**(r), 2**(r+1), 2**(r)])


            run_dir = '_nc={}_deg={}/'.format(nc, deg)
            plot_dir = base_dir + run_dir

            diags = solve_hcurl_source_pbm_nc(
                nc=nc, deg=deg,
                eta=eta,
                nu=0,
                mu=1,
                domain_name=domain_name,
                source_type=source_type,
                source_proj=source_proj,
                backend_language='pyccel-gcc',
                plot_source=True,
                project_sol=project_sol,
                gamma_h=gamma_h,
                plot_dir=plot_dir,
                hide_plots=True,
                skip_plot_titles=False,
                test = True
            )

            print(diags)

from psydac.feec.multipatch.experimental.three_patch_ex import three_patch_domain
def plot_meshes(base_dir):
    # int_x, int_y = [[1, 3], [0, np.pi / 4]]
    n = 20 
    ncells =  ncells = np.array([[n, 2*n, n, 2*n],
                                [2*n, n, 2*n, n],
                                [ n,    2*n,   n, 2*n],
                                [ 2*n,    n,   2*n, n]])
    domain = build_cartesian_multipatch_domain(ncells, [0,1], [0,1])
    # domain = build_cartesian_multipatch_domain(ncells, int_x, int_y, mapping='polar')
    ncells = {patch.name: [ncells[int(patch.name[4])][int(patch.name[2])], ncells[int(patch.name[4])][int(patch.name[2])]] for patch in domain.interior}

    # domain = three_patch_domain()
    # ncells = [n, 2*n, n]
    # ncells = {patch.name: [ncells[i], ncells[i]] for (i, patch) in enumerate(domain.interior)}

    plot_domain(domain, draw=True, isolines=True, ncells=ncells)    

if __name__ == '__main__':
    base_dir = 'plots/paper_formp/'
    plot_meshes(base_dir)
    #poisson(base_dir)
