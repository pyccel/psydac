#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
    Solve the eigenvalue problem for the curl-curl operator in 2D with a FEEC discretization
"""
import os
import numpy as np

from sympde.topology import Derham

from psydac.api.discretization  import discretize
from psydac.api.postprocessing  import OutputManager, PostProcessManager

from psydac.linalg.basic        import IdentityOperator
from psydac.linalg.utilities    import array_to_psydac

from psydac.fem.basic import FemField

from psydac.feec.multipatch_domain_utilities import build_cartesian_multipatch_domain, build_multipatch_domain


#==============================================================================
# Solver for curl-curl eigenvalue problems
#==============================================================================
def hcurl_solve_eigen_pbm(ncells=np.array([[8, 4], [4, 4]]), degree=(3, 3), domain=([0, np.pi], [0, np.pi]), domain_name='refined_square', backend_language='pyccel-gcc', mu=1, nu=0, gamma_h=0,
                             generalized_pbm=False, sigma=5, nb_eigs_solve=8, nb_eigs_plot=5, skip_eigs_threshold=1e-7,
                             plot_dir=None):
    """
    Solve the eigenvalue problem for the curl-curl operator in 2D with a brokenFEEC discretization

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
    gamma_h : float
        Coefficient in the curl-curl operator
    generalized_pbm : bool
        If True, solve the generalized eigenvalue problem
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
    domain_h = discretize(domain, ncells=ncells)   # Vh space

    print('building symbolic and discrete derham sequences...')
    print(' .. derham sequence...')
    derham = Derham(domain, ["H1", "Hcurl", "L2"])

    print(' .. discrete derham sequence...')
    derham_h = discretize(derham, domain_h, degree=degree)

    V0h, V1h, V2h = derham_h.spaces
    print('dim(V0h) = {}'.format(V0h.nbasis))
    print('dim(V1h) = {}'.format(V1h.nbasis))
    print('dim(V2h) = {}'.format(V2h.nbasis))

    print('building the discrete operators:')
    print('commuting projection operators...')

    I1 = IdentityOperator(V1h.coeff_space)

    print('Hodge operators...')
    # multi-patch (broken) linear operators / matrices
    H0, H1, H2 = derham_h.hodge_operators(kind='linop', backend_language=backend_language)
    dH0, dH1, dH2 = derham_h.hodge_operators(kind='linop', dual=True, backend_language=backend_language)

    print('conforming projection operators...')
    # conforming Projections (should take into account the boundary conditions
    # of the continuous deRham sequence)
    cP0, cP1, cP2 = derham_h.conforming_projectors(kind='linop', hom_bc = True)

    print('broken differential operators...')
    bD0, bD1 = derham_h.derivatives(kind='linop')


    print('computing the full operator matrix...')

    # Conga (projection-based) stiffness matrices
    if mu != 0:
        # curl curl:
        print('mu = {}'.format(mu))
        print('curl-curl stiffness matrix...')

        CC = cP1.T @ bD1.T @ H2 @ bD1 @ cP1  # Conga stiffness matrix
        A = mu * CC

    if nu != 0:
        GD = - cP1.T @ H1 @ bD0 @ cP0 @ dH0 @ cP0.T @ bD0.T @ H1 @ cP1
        A -= nu * GD

    # jump stabilization in V1h:
    if gamma_h != 0 or generalized_pbm:
        print('jump stabilization matrix...')
        JS = (I1 - cP1).T @ H1 @ (I1 - cP1)
        A += gamma_h * JS

    if generalized_pbm:
        print('adding jump stabilization to RHS of generalized eigenproblem...')
        B = cP1.T @ H1 @ cP1 + JS
    else:
        B = H1

    print('solving matrix eigenproblem...')
    all_eigenvalues, all_eigenvectors_transp = get_eigenvalues(nb_eigs_solve, sigma, A.tosparse(), B.tosparse())
    
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
        OM.add_spaces(V1h=V1h)
        OM.export_space_info()

        nb_eigs = len(eigenvalues)
        H1_m = H1.tosparse()
        cP1_m = cP1.tosparse()

        for i in range(min(nb_eigs_plot, nb_eigs)):

            print('looking at emode i = {}... '.format(i))
            lambda_i = eigenvalues[i]
            emode_i = np.real(eigenvectors[i])
            norm_emode_i = np.dot(emode_i, H1_m.dot(emode_i))
            eh_c = emode_i / norm_emode_i

            stencil_coeffs = array_to_psydac(cP1_m @ eh_c, V1h.coeff_space)
            vh = FemField(V1h, coeffs=stencil_coeffs)
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

#==============================================================================
# Eigenvalue solver
#==============================================================================
def get_eigenvalues(nb_eigs, sigma, A_m, M_m):
    """
    Compute the eigenvalues of the matrix A close to sigma and right-hand-side M

    Parameters
    ----------
    nb_eigs : int
        Number of eigenvalues to compute
    sigma : float
        Value close to which the eigenvalues are computed
    A_m : sparse matrix
        Matrix A
    M_m : sparse matrix
        Matrix M
    """

    from scipy.sparse.linalg    import spilu, lgmres
    from scipy.sparse.linalg    import LinearOperator, eigsh, minres
    from scipy.linalg           import norm

    print('-----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  ----- ')
    print(
        'computing {0} eigenvalues (and eigenvectors) close to sigma={1} with scipy.sparse.eigsh...'.format(
            nb_eigs,
            sigma))
    mode = 'normal'
    which = 'LM'
    # from eigsh docstring:
    #   ncv = number of Lanczos vectors generated ncv must be greater than k and smaller than n;
    #   it is recommended that ncv > 2*k. Default: min(n, max(2*k + 1, 20))
    ncv = 4 * nb_eigs
    print('A_m.shape = ', A_m.shape)
    try_lgmres = True
    max_shape_splu = 24000   # OK for nc=20, deg=6 on pretzel_f
    if A_m.shape[0] < max_shape_splu:
        print('(via sparse LU decomposition)')
        OPinv = None
        tol_eigsh = 0
    else:

        OP_m = A_m - sigma * M_m
        tol_eigsh = 1e-7
        if try_lgmres:
            print(
                '(via SPILU-preconditioned LGMRES iterative solver for A_m - sigma*M1_m)')
            OP_spilu = spilu(OP_m, fill_factor=15, drop_tol=5e-5)
            preconditioner = LinearOperator(
                OP_m.shape, lambda x: OP_spilu.solve(x))
            tol = tol_eigsh
            OPinv = LinearOperator(
                matvec=lambda v: lgmres(OP_m, v, x0=None, tol=tol, atol=tol, M=preconditioner,
                                        callback=lambda x: print(
                                            'cg -- residual = ', norm(OP_m.dot(x) - v))
                                        )[0],
                shape=M_m.shape,
                dtype=M_m.dtype
            )

        else:
            # from https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html:
            # the user can supply the matrix or operator OPinv, which gives x = OPinv @ b = [A - sigma * M]^-1 @ b.
            # > here, minres: MINimum RESidual iteration to solve Ax=b
            # suggested in https://github.com/scipy/scipy/issues/4170
            print('(with minres iterative solver for A_m - sigma*M1_m)')
            OPinv = LinearOperator(
                matvec=lambda v: minres(
                    OP_m,
                    v,
                    tol=1e-10)[0],
                shape=M_m.shape,
                dtype=M_m.dtype)

    eigenvalues, eigenvectors = eigsh(
        A_m, k=nb_eigs, M=M_m, sigma=sigma, mode=mode, which=which, ncv=ncv, tol=tol_eigsh, OPinv=OPinv)

    print("done: eigenvalues found: " + repr(eigenvalues))
    return eigenvalues, eigenvectors

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

    # Jump stabilization parameter
    gamma_h = 0
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

    eigenvalues = hcurl_solve_eigen_pbm(
        ncells=ncells, degree=degree,
        gamma_h=gamma_h,
        generalized_pbm=generalized_pbm,
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
