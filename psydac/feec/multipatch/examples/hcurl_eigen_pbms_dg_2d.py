"""
    Solve the eigenvalue problem for the curl-curl operator in 2D with DG discretization, following
    A. Buffa and I. Perugia, “Discontinuous Galerkin Approximation of the Maxwell Eigenproblem”
    SIAM Journal on Numerical Analysis 44 (2006)
"""
import os
from mpi4py import MPI
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot

from scipy.sparse.linalg import LinearOperator, eigsh, minres

from sympde.calculus import grad, dot, curl, cross
from sympde.calculus import minus, plus
from sympde.topology import VectorFunctionSpace
from sympde.topology import elements_of
from sympde.topology import NormalVector
from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping
from sympde.expr.expr import LinearForm, BilinearForm
from sympde.expr.expr import integral
from sympde.expr.expr import Norm
from sympde.expr.equation import find, EssentialBC

from psydac.linalg.utilities import array_to_psydac
from psydac.fem.basic import FemField
from psydac.feec.pull_push import pull_2d_hcurl

from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain
from psydac.feec.multipatch.utilities import time_count
from psydac.api.discretization import discretize
from psydac.feec.multipatch.multipatch_domain_utilities import build_cartesian_multipatch_domain
from psydac.api.postprocessing import OutputManager, PostProcessManager


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

    diags = {}

    if sigma is None:
        raise ValueError('please specify a value for sigma')

    print('---------------------------------------------------------------------------------------------------------')
    print('Starting hcurl_solve_eigen_pbm function with: ')
    print(' ncells = {}'.format(ncells))
    print(' degree = {}'.format(degree))
    print(' domain_name = {}'.format(domain_name))
    print(' backend_language = {}'.format(backend_language))
    print('---------------------------------------------------------------------------------------------------------')
    t_stamp = time_count()
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

    mappings = OrderedDict([(P.logical_domain, P.mapping)
                           for P in domain.interior])
    mappings_list = list(mappings.values())

    t_stamp = time_count(t_stamp)
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

    all_eigenvalues_2, all_eigenvectors_transp_2 = get_eigenvalues(
        nb_eigs_solve, sigma, Ah_m, Bh_m)

    # Eigenvalue processing
    t_stamp = time_count(t_stamp)
    print('sorting out eigenvalues...')
    zero_eigenvalues = []
    if skip_eigs_threshold is not None:
        eigenvalues = []
        eigenvectors = []
        for val, vect in zip(all_eigenvalues_2, all_eigenvectors_transp_2.T):
            if abs(val) < skip_eigs_threshold:
                zero_eigenvalues.append(val)
                # we skip the eigenvector
            else:
                eigenvalues.append(val)
                eigenvectors.append(vect)
    else:
        eigenvalues = all_eigenvalues_2
        eigenvectors = all_eigenvectors_transp_2.T
    diags['DG'] = True
    for k, val in enumerate(eigenvalues):
        diags['eigenvalue2_{}'.format(k)] = val  # eigenvalues[k]

    for k, val in enumerate(zero_eigenvalues):
        diags['skipped eigenvalue2_{}'.format(k)] = val

    t_stamp = time_count(t_stamp)
    print('plotting the eigenmodes...')

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

    t_stamp = time_count(t_stamp)

    return diags, eigenvalues


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
