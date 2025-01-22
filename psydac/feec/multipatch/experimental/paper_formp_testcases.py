import os 
import numpy as np
import matplotlib.pyplot as plt

from psydac.feec.multipatch.examples.h1_source_pbms_conga_2d import solve_h1_source_pbm
from psydac.feec.multipatch.examples.hcurl_eigen_pbms_conga_2d import hcurl_solve_eigen_pbm
from psydac.feec.multipatch.examples.hcurl_source_pbms_conga_2d import solve_hcurl_source_pbm

from sympde.utilities.utils import plot_domain
from psydac.feec.multipatch.multipatch_domain_utilities import build_cartesian_multipatch_domain
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain

def poisson(base_dir):
    base_dir = base_dir + 'poisson_pretzel/'
    


    source_type = 'manu_poisson_elliptic'
    domain_name = 'pretzel_f'
    eta = 0
    mu = 1
    for deg in [5]:
        for k in [3, 4, 5]:

            # nc = np.array([2**(k+1), 2**k, 2**k, 2**(k+1), 2**(k+1), 2**(k+1), 2**(k+1), 2**k, 2**k,
            #             2**k, 2**k, 2**k, 2**k, 2**(k+1), 2**k, 2**k, 2**k, 2**k])

            nc = np.array([2**k, 2**k, 2**k, 2**k, 2**k, 2**k, 2**k, 2**k, 2**k,
                           2**k, 2**k, 2**k, 2**k, 2**k, 2**k, 2**k, 2**k, 2**k])

            # nc = np.array([2**(k+1), 4, 4, 2**(k+1), 2**(k+1), 2**(k+1), 2**(k+1), 4, 4,
            #                  4, 4, 4, 4, 2**(k+1), 4, 4, 4, 4])

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
            #plot_domain(domain, draw=False, isolines=True, ncells=ncells, path=plot_dir+'domain_'+str(np.sum(nc))+'.png')    

def hcurl_eigen(base_dir):
    
    method = 'dg'
    # method = 'dg'

    operator = 'curl-curl'
    nu = 0
    mu = 1

    generalized_pbm = True
    gamma_h = 0

    # domain_name = 'three_patch'
    # domain = [[0, np.pi], [0, np.pi]]
    # ncells = [np.array([32, 64, 32])]
    #         # np.array([8, 16, 8]), 
    #         # np.array([16, 32, 16]), 
    #         # np.array([32, 64, 32])]

    domain_name = 'refined_square'
    domain = [[0, np.pi], [0, np.pi]]
    ncells = [np.array([[4, 4],[ 4, 4]]),
            np.array([[8, 8],[ 8, 8]]), 
            np.array([[16, 16],[ 16, 16]]), 
            np.array([[32, 32],[ 32, 32]])]


    ref_sigmas = [
        1, 1,
        2,
        4, 4,
        5, 5,
        8,
        9, 9,
    ]
    sigma = 5
    nb_eigs_solve = 10
    nb_eigs_plot = 10
    skip_eigs_threshold = 1e-7


    # domain_name = 'curved_L_shape'
    # domain = [[1, 3], [0, np.pi / 4]]  # interval in x- and y-direction
    # ref_sigmas = [
    #     0.181857115231E+01,
    #     0.349057623279E+01,
    #     0.100656015004E+02,
    #     0.101118862307E+02,
    #     0.124355372484E+02,
    # ]
    # sigma = 7
    # nb_eigs_solve = 7
    # nb_eigs_plot = 7
    # skip_eigs_threshold = 1e-7


    # ncells = [np.array([[None, 4],
    #                         [4, 4]]),
    #     np.array([[None, None, 4, 4],
    #                 [None, None, 8, 4],
    #                 [ 4,    8,   8, 4],
    #                 [ 4,    4,   4, 4]]),
    #     np.array([[None, None, None,    4, 4, 4],
    #                [None, None, None,    8, 8, 4],
    #                [None, None, None,   16, 8, 4],
    #                [4,     8,     16,   16, 8, 4],
    #                [4,     8,     8,     8, 8, 4],
    #                [4,     4,     4,     4, 4, 4]]),
    #     np.array([[None, None, None,  None,   4,   4,  4, 4],
    #                [None, None, None,  None,   8,   8,  8, 4],
    #                [None, None, None,  None,  16,  16,  8, 4],
    #                [None, None, None,  None,  32,  16,  8, 4],
    #                [4,     8,     16,    32,  32,  16,  8, 4],
    #                [4,     8,     16,    16,  16,  16,  8, 4],
    #                [4,     8,     8,     8,    8,   8,  8, 4],
    #                [4,     4,     4,     4,    4,   4,  4, 4]])
    # ]

    degree = [3]
    base_dir = base_dir + 'hcurl_eigen_'+domain_name+method

    for nc in ncells:
        for deg in degree:

            run_dir = '_dims={}_nc={}_deg={}/'.format(nc.shape, nc[nc != None].sum(), deg)
            plot_dir = base_dir + run_dir

            deg = [deg, deg]

            diags, eigenvalues = hcurl_solve_eigen_pbm(
                ncells=nc, degree=deg,
                gamma_h=gamma_h,
                generalized_pbm=generalized_pbm,
                nu=nu,
                mu=mu,
                sigma=sigma,
                skip_eigs_threshold=skip_eigs_threshold,
                nb_eigs_solve=nb_eigs_solve,
                nb_eigs_plot=nb_eigs_plot,
                domain_name=domain_name, domain=domain,
                backend_language='pyccel-gcc',
                plot_dir=plot_dir,
                m_load_dir=None,
            )
            errors = []
            for k in range(min(len(ref_sigmas), len(eigenvalues))):
                e = abs(eigenvalues[k] - ref_sigmas[k])
                diags['error_{}'.format(k)] = e
                errors.append(e)

            print(nc, deg, errors)
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

            diags = solve_hcurl_source_pbm(
                nc=nc, deg=deg,
                eta=eta,
                nu=0,
                mu=1,
                domain_name=domain_name,
                source_type=source_type,
                source_proj=source_proj,
                backend_language='pyccel-gcc',
                project_sol=project_sol,
                gamma_h=gamma_h,
                plot_dir=plot_dir,
            )

            print(diags)

from psydac.feec.multipatch.experimental.three_patch_ex import three_patch_domain
def plot_meshes(base_dir, case):
    if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
    if case == "cc_ev_L_shape_grid":

        # ncells = np.array([[n, n,   n,   None, None, None],
        #                             [n, 2*n,  2*n, None, None, None],
        #                             [n, 2*n,  4*n, None, None, None],
        #                             [n, 2*n,  4*n, 4*n, 2*n, n],
        #                             [ n, 2*n,    2*n,  2*n, 2*n,  n], 
        #                             [ n, n,    n,  n, n,  n]]).transpose()
        ncells_ = [np.array([[None, 4],
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
        
        for n in range(4):
            ncells = np.rot90(ncells_[n])
            domain = build_cartesian_multipatch_domain(ncells, [1, 3], [0, np.pi / 4], mapping='polar')
            ncells = {patch.name: [ncells[int(patch.name[4])][int(patch.name[2])], ncells[int(patch.name[4])][int(patch.name[2])]] for patch in domain.interior}

            plot_domain(domain, draw=False, isolines=True, ncells=ncells, path=base_dir+case+'_'+str(n)+'.png')    

    elif case == "th_maxwell_pretzel_rand":
        for r in range(4):
            #random ref
            nc = np.array([2**(r+1), 2**(r),   2**(r+1), 2**(r), 2**(r+1),
                            2**(r),   2**(r+1),   2**(r),   2**(r+1),   2**(r),
                            2**(r+1),   2**(r),   2**(r+1),   2**(r), 2**(r+1), 
                            2**(r), 2**(r+1), 2**(r)])

            # refine source
            # nc = np.array([2**(r+1), 4, 4, 2**(r+1), 2**(r+1), 2**(r+1), 2**(r+1), 4, 4, 4, 4, 4, 4, 2**(r+1), 4, 4, 4, 4])
            domain = build_multipatch_domain(domain_name='pretzel_f')
            ncells = {patch.name: [nc[i], nc[i]] for (i, patch) in enumerate(domain.interior)}

            plot_domain(domain, draw=False, isolines=True, ncells=ncells, path=base_dir+case+'_'+str(r)+'.png')    

    elif case == "three_patch":
        n = 5
        domain = three_patch_domain(np.pi, np.pi)
        ncells = [n, 4*n, 2*n]
        ncells = {patch.name: [ncells[i], ncells[i]] for (i, patch) in enumerate(domain.interior)}

        plot_domain(domain, draw=False, isolines=True, ncells=ncells, path=base_dir+case+'_'+str(n)+'.png')    

    elif case == "refined_square":
        n = 6

        ncells = np.array([[n, n, n],
                        [n, 2*n,  2*n ],
                        [n, 2*n,  4*n]])

        domain = build_cartesian_multipatch_domain(ncells, [0,np.pi], [0,np.pi])
        ncells = {patch.name: [ncells[int(patch.name[4])][int(patch.name[2])], ncells[int(patch.name[4])][int(patch.name[2])]] for patch in domain.interior}
    
        plot_domain(domain, draw=False, isolines=True, ncells=ncells, path=base_dir+case+'_'+str(n)+'.png')    
    elif case == "wdiv_L_shape_grid":
         fac = 6
         for k in range(0,4):
            ncells = np.array([[None, fac * 2**k], [fac * 2**k, fac * 2**(k+1)]])
            ncells = np.rot90(ncells)
            domain = build_cartesian_multipatch_domain(ncells, [1, 3], [0, np.pi / 4], mapping='polar')
            ncells = {patch.name: [ncells[int(patch.name[4])][int(patch.name[2])], ncells[int(patch.name[4])][int(patch.name[2])]] for patch in domain.interior}

            plot_domain(domain, draw=False, isolines=True, ncells=ncells, path=base_dir+case+'_'+str(k)+'.png')    


if __name__ == '__main__':
    base_dir = 'plots/NumKin24/'
    plot_meshes(base_dir, "cc_ev_L_shape_grid")
   # poisson(base_dir)
    #hcurl_source(base_dir)
    #hcurl_eigen(base_dir)