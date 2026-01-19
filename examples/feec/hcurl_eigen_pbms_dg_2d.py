#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
    Solve the eigenvalue problem for the curl-curl operator in 2D with DG discretization, following
    A. Buffa and I. Perugia, “Discontinuous Galerkin Approximation of the Maxwell Eigenproblem”
    SIAM Journal on Numerical Analysis 44 (2006)
"""
import os
import numpy as np

from sympde.calculus import dot, curl, cross
from sympde.calculus import minus, plus
from sympde.topology import VectorFunctionSpace
from sympde.topology import elements_of
from sympde.topology import NormalVector
from sympde.expr.expr import LinearForm, BilinearForm
from sympde.expr.expr import integral

from psydac.linalg.utilities import array_to_psydac

from psydac.fem.basic import FemField

from psydac.feec.multipatch_domain_utilities import build_multipatch_domain, build_cartesian_multipatch_domain

from psydac.api.discretization import discretize
from psydac.api.postprocessing import OutputManager, PostProcessManager

from examples.feec.hcurl_eigen_pbms_conga_2d import get_eigenvalues

#==============================================================================
# Solver for curl-curl eigenvalue problems
#==============================================================================
def hcurl_solve_eigen_pbm_dg(ncells=np.array([[8, 4], [4, 4]]), degree=(3, 3), domain=([0, np.pi], [0, np.pi]), domain_name='refined_square', backend_language='pyccel-gcc', mu=1, nu=0,
                             sigma=5, nb_eigs_solve=8, nb_eigs_plot=5, skip_eigs_threshold=1e-7,
                             plot_dir=None,):
    """
    Solve the eigenvalue problem for the curl-curl operator in 2D with DG discretization

    Parameters
    ----------
    ncells : array
        Number of cells in each direction
    degree : tuple
        Degree of the basis functions
    domain : list
        Interval in x- and y-direction
    domain_name : str
        Name of the domain
    backend_language : str
        Language used for the backend
    mu : float
        Coefficient in the curl-curl operator
    nu : float
        Coefficient in the curl-curl operator
    sigma : float
        Calculate eigenvalues close to sigma
    nb_eigs_solve : int
        Number of eigenvalues to solve
    nb_eigs_plot : int
        Number of eigenvalues to plot
    skip_eigs_threshold : float
        Threshold for the eigenvalues to skip
    plot_dir : str
        Directory for the plots
    """

    if sigma is None:
        raise ValueError('please specify a value for sigma')

    print('---------------------------------------------------------------------------------------------------------')
    print('Starting hcurl_solve_eigen_pbm function with: ')
    print(' ncells = {}'.format(ncells))
    print(' degree = {}'.format(degree))
    print(' domain_name = {}'.format(domain_name))
    print(' backend_language = {}'.format(backend_language))
    print('---------------------------------------------------------------------------------------------------------')
    print('building symbolic and discrete domain...')

    int_x, int_y = domain
    if isinstance(ncells, int):
        domain = build_multipatch_domain(domain_name=domain_name)

    elif domain_name == 'refined_square' or domain_name == 'square_L_shape':
        domain = build_cartesian_multipatch_domain(ncells, int_x, int_y, mapping='identity')

    elif domain_name == 'curved_L_shape':
        domain = build_cartesian_multipatch_domain(ncells, int_x, int_y, mapping='polar')

    else:
        domain = build_multipatch_domain(domain_name=domain_name)

    if isinstance(ncells, int):
        ncells = [ncells, ncells]
    elif ncells.ndim == 1:
        ncells = {patch.name: [ncells[i], ncells[i]]
                    for (i, patch) in enumerate(domain.interior)}
    elif ncells.ndim == 2:
        ncells = {patch.name: [ncells[int(patch.name[2])][int(patch.name[4])], 
                ncells[int(patch.name[2])][int(patch.name[4])]] for patch in domain.interior}


    print(' .. discrete domain...')
    V = VectorFunctionSpace('V', domain, kind='hcurl')

    u, v, F = elements_of(V, names='u, v, F')
    nn = NormalVector('nn')

    I = domain.interfaces
    boundary = domain.boundary

    kappa = 10
    k = 1

    def jump(w): return plus(w) - minus(w)
    def avr(w): return 0.5 * plus(w) + 0.5 * minus(w)

    expr1_I = cross(nn, jump(v)) * curl(avr(u))\
        + k * cross(nn, jump(u)) * curl(avr(v))\
        + kappa * cross(nn, jump(u)) * cross(nn, jump(v))

    expr1 = curl(u) * curl(v)
    expr1_b = -cross(nn, v) * curl(u) - k * cross(nn, u) * \
        curl(v) + kappa * cross(nn, u) * cross(nn, v)
    # curl curl u = - omega**2 u

    expr2 = dot(u, v)
    # expr2_I = kappa*cross(nn, jump(u))*cross(nn, jump(v))
    # expr2_b = -k*cross(nn, u)*curl(v) + kappa * cross(nn, u) * cross(nn, v)

    # Bilinear form a: V x V --> R
    a = BilinearForm((u, v), integral(domain, expr1) +
                     integral(I, expr1_I) + integral(boundary, expr1_b))

    # Linear form l: V --> R
    # + integral(I, expr2_I) + integral(boundary, expr2_b))
    b = BilinearForm((u, v), integral(domain, expr2))

    # +++++++++++++++++++++++++++++++
    # 2. Discretization
    # +++++++++++++++++++++++++++++++

    domain_h = discretize(domain, ncells=ncells)
    Vh = discretize(V, domain_h, degree=degree)

    ah = discretize(a, domain_h, [Vh, Vh])
    Ah_m = ah.assemble().tosparse()

    bh = discretize(b, domain_h, [Vh, Vh])
    Bh_m = bh.assemble().tosparse()

    all_eigenvalues, all_eigenvectors_transp = get_eigenvalues(
        nb_eigs_solve, sigma, Ah_m, Bh_m)

    # Eigenvalue processing
    print('sorting out eigenvalues...')
    zero_eigenvalues = []
    if skip_eigs_threshold is not None:
        eigenvalues = []
        eigenvectors = []
        for val, vect in zip(all_eigenvalues, all_eigenvectors_transp.T):
            if abs(val) < skip_eigs_threshold:
                zero_eigenvalues.append(val)
                # we skip the eigenvector
            else:
                eigenvalues.append(val)
                eigenvectors.append(vect)
    else:
        eigenvalues = all_eigenvalues
        eigenvectors = all_eigenvectors_transp.T

    print('plotting the eigenmodes...')
    if plot_dir:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        OM = OutputManager(plot_dir + '/spaces.yml', plot_dir + '/fields.h5')
        OM.add_spaces(Vh=Vh)
        OM.export_space_info()

        nb_eigs = len(eigenvalues)
        for i in range(min(nb_eigs_plot, nb_eigs)):

            print('looking at emode i = {}... '.format(i))
            lambda_i = eigenvalues[i]
            emode_i = np.real(eigenvectors[i])
            norm_emode_i = np.dot(emode_i, Bh_m.dot(emode_i))
            eh_c = emode_i / norm_emode_i

            stencil_coeffs = array_to_psydac(eh_c, Vh.coeff_space)
            vh = FemField(Vh, coeffs=stencil_coeffs)
            OM.add_snapshot(i, i)
            OM.export_fields(vh=vh)

        OM.close()

        PM = PostProcessManager(
            domain=domain,
            space_file=plot_dir + '/spaces.yml',
            fields_file=plot_dir + '/fields.h5')
        PM.export_to_vtk(
            plot_dir + "/eigenvalues",
            grid=None,
            npts_per_cell=[6] * 2,
            snapshots='all',
            fields='vh')
        PM.close()

    return eigenvalues

if __name__ == '__main__':
    # Degree
    degree = [3, 3]

    # Refined square domain
    domain_name = 'refined_square'
    domain = [[0, np.pi], [0, np.pi]]
    ncells = np.array([[10, 5, 10],
                        [5, 10, 5],
                        [10, 5, 10]])

    # Curved L-shape domain
    # domain_name = 'curved_L_shape'
    # domain = [[1, 3], [0, np.pi / 4]]  # interval in x- and y-direction
    # ncells = np.array([[None, 5],
    #                 [5, 10]])

    # solves generalized eigenvalue problem with:  B(v,w) = <Pv,Pw> +
    # <(I-P)v,(I-P)w> in rhs
    generalized_pbm = True

    # curl-curl operator
    nu = 0
    mu = 1

    # reference eigenvalues for validation
    if domain_name == 'refined_square':
        assert domain == [[0, np.pi], [0, np.pi]]
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

    elif domain_name == 'curved_L_shape':
        # ref eigenvalues from Monique Dauge benchmark page
        assert domain == [[1, 3], [0, np.pi / 4]]
        ref_sigmas = [
            0.181857115231E+01,
            0.349057623279E+01,
            0.100656015004E+02,
            0.101118862307E+02,
            0.124355372484E+02,
        ]
        sigma = 7
        nb_eigs_solve = 5
        nb_eigs_plot = 5
        skip_eigs_threshold = 1e-7

    eigenvalues = hcurl_solve_eigen_pbm_dg(
        ncells=ncells, degree=degree,
        nu=nu,
        mu=mu,
        sigma=sigma,
        skip_eigs_threshold=skip_eigs_threshold,
        nb_eigs_solve=nb_eigs_solve,
        nb_eigs_plot=nb_eigs_plot,
        domain_name=domain_name, domain=domain,
    )

    if ref_sigmas is not None:
        n_errs = min(len(ref_sigmas), len(eigenvalues))
        for k in range(n_errs):
            print('error_{}: '.format(k), abs(eigenvalues[k] - ref_sigmas[k]))
