# small script written to test Conga operators on multipatch domains, using the piecewise (broken) de Rham sequences available on every space

from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import inv

from sympde.topology import Derham
from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping

from psydac.feec.multipatch.api import discretize  # TODO: when possible, use line above
from psydac.feec.pull_push     import pull_2d_hcurl

from psydac.linalg.utilities import array_to_stencil

from psydac.fem.basic   import FemField

from psydac.feec.multipatch.fem_linear_operators import FemLinearOperator, IdLinearOperator
from psydac.feec.multipatch.fem_linear_operators import SumLinearOperator, MultLinearOperator, ComposedLinearOperator
from psydac.feec.multipatch.operators import BrokenMass, ortho_proj_Hcurl
from psydac.feec.multipatch.operators import ConformingProjection_V0, ConformingProjection_V1
from psydac.feec.multipatch.plotting_utilities import get_grid_vals_scalar, get_grid_vals_vector
from psydac.feec.multipatch.plotting_utilities import get_plotting_grid, get_patch_knots_gridlines, my_small_plot
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain

comm = MPI.COMM_WORLD

import time
def time_count(t_stamp=None):
    new_t_stamp = time.time()
    if t_stamp:
        print('time elapsed: ', new_t_stamp - t_stamp)
    return new_t_stamp

def run_maxwell_2d_eigenproblem(nb_eigs, ncells, degree, alpha,
                                domain_name='square',
                                n_patches=2,
                                compute_kernel=False,
                                test_harmonic_field=False,
                                ref_sigmas=None,
                                show_all=False):
    """
    Maxwell eigenproblem solver, see eg
    Buffa, Perugia & Warburton, The Mortar-Discontinuous Galerkin Method for the 2D Maxwell Eigenproblem JSC 2009.

    :param nb_eigs: nb of eigenmodes to be computed
    :return: eigenvalues and eigenmodes
    """

    assert len(ncells) == 2 and ncells[0] == ncells[1]
    assert len(degree) == 2 and degree[0] == degree[1]

    t_stamp = time_count()

    print("building domain and spaces...")
    domain = build_multipatch_domain(domain_name=domain_name, n_patches=n_patches)

    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())
    x,y    = domain.coordinates
    nquads = [d + 1 for d in degree]
    # plotting
    etas, xx, yy = get_plotting_grid(mappings, N=20)

    # multipatch de Rham sequence:
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])

    domain_h = discretize(domain, ncells=ncells, comm=comm)
    derham_h = discretize(derham, domain_h, degree=degree)
    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2


    t_stamp = time_count(t_stamp)

    # Mass matrices for broken spaces (block-diagonal)
    print("assembling mass matrix operators...")
    M0 = BrokenMass(V0h, domain_h, is_scalar=True)
    M1 = BrokenMass(V1h, domain_h, is_scalar=False)
    M2 = BrokenMass(V2h, domain_h, is_scalar=True)

    t_stamp = time_count(t_stamp)
    print("assembling broken derivative operators...")

    bD0, bD1 = derham_h.broken_derivatives_as_operators

    t_stamp = time_count(t_stamp)
    print("assembling conf projection operators...")

    cP0 = ConformingProjection_V0(V0h, domain_h, hom_bc=True)
    cP1 = ConformingProjection_V1(V1h, domain_h, hom_bc=True)

    t_stamp = time_count(t_stamp)
    print("assembling conga derivative operators...")

    D0 = ComposedLinearOperator([bD0,cP0])
    # D0_t = ComposedLinearOperator([cP0, bD0.transpose()])
    D1 = ComposedLinearOperator([bD1,cP1])
    # D1_t = ComposedLinearOperator([cP1, bD1.transpose()])
    I1 = IdLinearOperator(V1h)

    if test_harmonic_field:
        print("testing harmonic field...")

        t_stamp = time_count(t_stamp)
        print("assembling projection operators...")
        P0, P1, P2 = derham_h.projectors(nquads=nquads)

        # testing fields in kernel for circular annulus...
        harmonic_field = 2

        if harmonic_field == 1:
            # 'diverging' harmonic field: hf = (-(sin theta)/r , (cos theta)/r) = (-y/r**2, x/r**2)
            hf_x = -y/(x**2 + y**2)
            hf_y =  x/(x**2 + y**2)
        else:
            # 'rotating' harmonic field: hf = (-(sin theta)/r , (cos theta)/r) = (-y/r**2, x/r**2)
            hf_x = x/(x**2 + y**2)
            hf_y = y/(x**2 + y**2)

        from sympy import lambdify
        hf_x   = lambdify(domain.coordinates, hf_x)
        hf_y   = lambdify(domain.coordinates, hf_y)
        hf_log = [pull_2d_hcurl([hf_x,hf_y], f) for f in mappings_list]

        hf = P1(hf_log)
        chf = D1(hf)

        grid_vals_h1 = lambda v: get_grid_vals_scalar(v, etas, mappings_list, space_kind='h1')
        grid_vals_hcurl = lambda v: get_grid_vals_vector(v, etas, mappings_list, space_kind='hcurl')

        hf_x_vals, hf_y_vals = grid_vals_hcurl(hf)
        chf_vals  = grid_vals_h1(chf)

        my_small_plot(
            title=r'diverging harmonic field and Conga curl',
            vals=[hf_x_vals, hf_y_vals, chf_vals],
            titles=[r'$v_x$', r'$v_y$' , r'$curl Pv$' ],
            surface_plot=True,
            xx=xx, yy=yy,
        )

    as_psydac_operator = False
    if as_psydac_operator:

        raise NotImplementedError
        A = (
                ComposedLinearOperator([M1, D0, M0_inv, D0_t, M1])
            + alpha*ComposedLinearOperator([I1-cP1,M1, I1-cP1])
            + ComposedLinearOperator([D1_t, M2, D1])
            )

        # convert anyhow, to use eigensolver from scipy.sparse
        A_m = A.to_sparse_matrix()
        M1_m = M1.to_sparse_matrix()

    else:

        t_stamp = time_count(t_stamp)
        print("converting in sparse matrices...")
        M0_m = M0.to_sparse_matrix()
        M1_m = M1.to_sparse_matrix()
        M2_m = M2.to_sparse_matrix()
        cP0_m = cP0.to_sparse_matrix()
        cP1_m = cP1.to_sparse_matrix()
        D0_m = D0.to_sparse_matrix()  # also possible as matrix product bD0 * cP0
        D1_m = D1.to_sparse_matrix()
        I1_m = I1.to_sparse_matrix()

        M0_minv = inv(M0_m.tocsc())  # todo: assemble patch-wise M0_inv, as Hodge operator

        ## building Hodge Laplacian matrix
        t_stamp = time_count(t_stamp)
        print("computing (sparse) Hodge-Laplacian matrix...")
        div_aux_m = D0_m.transpose() * M1_m  # note: the matrix of the (weak) div operator is:   - M0_minv * div_aux_m
        jump_penal_m = I1_m-cP1_m
        A_m = ( div_aux_m.transpose() * M0_minv * div_aux_m
            + alpha * jump_penal_m.transpose() * M1_m * jump_penal_m
            + D1_m.transpose() * M2_m * D1_m
            )

    # A = ComposedLinearOperator([I1-cP1,I1-cP1]) + ComposedLinearOperator([cP1, bD1.transpose(), M2, bD1, cP1])

        # + M1

        #
        # + 1000*ComposedLinearOperator([M1, D0, D0_t, M1])

    print('Finding eigenmodes and eigenvalues with scipy.sparse.eigsh ... ')

    if compute_kernel:
        sigma = 0
        mode = 'normal'
        which = 'LM'
    else:
        sigma = 2
        mode='cayley'
        # mode='buckling'
        which = 'LM'

    # from eigsh docstring:
    #   ncv = number of Lanczos vectors generated ncv must be greater than k and smaller than n;
    #   it is recommended that ncv > 2*k. Default: min(n, max(2*k + 1, 20))
    ncv = 4*nb_eigs
    # search mode: normal and buckling give a lot of zero eigenmodes. Cayley seems best for Maxwell.
    # mode='normal'

    t_stamp = time_count(t_stamp)
    print('computing eigenvalues and eigenvectors...' )
    eigenvalues, eigenvectors = eigsh(A_m, k=nb_eigs, M=M1_m, sigma=sigma, mode=mode, which=which, ncv=ncv)

    grid_vals_h1 = lambda v: get_grid_vals_scalar(v, etas, mappings_list, space_kind='h1')
    grid_vals_hcurl = lambda v: get_grid_vals_vector(v, etas, mappings_list, space_kind='hcurl')

    first_Pemodes_vals = []
    first_Pemodes_titles = []
    first_evalues = []

    k_eig = 0
    nb_eigs_found = 0   # we only look for curl-curl eigenmodes
    while k_eig < nb_eigs:
        t_stamp = time_count(t_stamp)
        print('looking at emode k = ', repr(k_eig), '... ')
        evalue  = eigenvalues[k_eig]
        emode_sp = np.real(eigenvectors[:,k_eig])
        aux_div_emode = div_aux_m.dot(emode_sp)
        ampl_aux_div_emode = np.dot(aux_div_emode, aux_div_emode)/np.dot(emode_sp,emode_sp)
        print('rel amplitude of aux_div_emode: ', repr(ampl_aux_div_emode))

        if ampl_aux_div_emode < 1e-5:
            print('seems to be a curl-curl eigenmode.')
            # normalize mode in L2
            Me = M1_m.dot(emode_sp)
            norm_emode = np.dot(emode_sp,Me)
            print('norm of computed eigenmode: ', norm_emode)

            emode_c = array_to_stencil(emode_sp/norm_emode, V1h.vector_space)
            emode = FemField(V1h, coeffs=emode_c)

            cP_emode = cP1(emode)
            curl_emode = D1(emode)

            eh_x_vals, eh_y_vals = grid_vals_hcurl(emode)
            cPeh_x_vals, cPeh_y_vals = grid_vals_hcurl(cP_emode)
            Peh_abs_vals = [np.sqrt(abs(Pex)**2 + abs(Pey)**2) for Pex, Pey in zip(cPeh_x_vals, cPeh_y_vals)]
            jumps_eh_vals = [np.sqrt(abs(ex-Pex)**2 + abs(ey-Pey)**2)
                             for ex, Pex, ey, Pey in zip (eh_x_vals, cPeh_x_vals, eh_y_vals, cPeh_y_vals)]
            curl_eh_vals = grid_vals_h1(curl_emode)

            if nb_eigs_found < 8:
                first_Pemodes_vals.append(Peh_abs_vals)
                first_Pemodes_titles.append(r'$\sigma=$'+'{0:0.2f}'.format(np.real(evalue)))
                first_evalues.append(np.real(evalue))
            else:
                print('warning: not plotting eigenmode for nb_eigs_found = ' + repr(nb_eigs_found))

            nb_eigs_found += 1
            is_curl_curl = 'Yes'
        else:
            print('does not seem to be a curl-curl eigenmode.')
            is_curl_curl = 'No'

        if show_all:
            my_small_plot(
                title=('mode k:'+repr(k_eig)+' -- eigenvalue: '+repr(evalue)+' -- is curl_curl: '+is_curl_curl),
                vals=[eh_x_vals, eh_y_vals, Peh_abs_vals, jumps_eh_vals, curl_eh_vals],
                titles=[r'$e^h_{k,x}$', r'$e^h_{k,y}$', r'$|P^1_c e^h_k|$', r'$|(I-P^1_c) e^h_k|$', r'curl$(e^h_k)$'],
                xx=xx,
                yy=yy,
            )

        k_eig += 1

    my_small_plot(
        title=r'Amplitude $|P^1_c e^h_k|$ of some eigenmodes found',
        vals=first_Pemodes_vals,
        titles=first_Pemodes_titles,
        xx=xx,
        yy=yy,
    )

    t_stamp = time_count(t_stamp)
    print('done -- summary: ')

    print("using jump penalization factor alpha = ", alpha )
    print('nb of spline cells per patch: ' + repr(ncells))
    h = 1/ncells[0]
    print('-- corresponding to h: '+ repr(h))
    print('degree: ' + repr(degree))

    nb_dofs = len(emode_sp)
    print(' -- nb of DOFS: ' + repr(nb_dofs))

    print('computed eigenvalues: ')
    print(first_evalues)

    if ref_sigmas is not None:
        errors = []
        n_errs = min(len(ref_sigmas), len(first_evalues))
        for k in range(n_errs):
            errors.append(abs(first_evalues[k]-ref_sigmas[k]))

    print('errors from reference eigenvalues: ')
    print(errors)

if __name__ == '__main__':

    nc = 2**5
    h = 1/nc
    deg = 3
    # jump penalization factor from Buffa, Perugia and Warburton  >> need to study
    DG_alpha = 10*(deg+1)**2/h
    alpha = DG_alpha

    nb_eigs = 8
    n_patches = None
    ref_sigmas = None

    domain_name = 'curved_L_shape'

    if domain_name == 'square':
        n_patches = 6
    elif domain_name == 'annulus':
        n_patches = 4
    elif domain_name == 'curved_L_shape':
        ref_sigmas = [
            0.181857115231E+01,
            0.349057623279E+01,
            0.100656015004E+02,
            0.101118862307E+02,
            0.124355372484E+02,
            ]
        nb_eigs=7  # need a bit more, to get rid of grad-div eigenmodes

    run_maxwell_2d_eigenproblem(
        nb_eigs=nb_eigs, ncells=[nc, nc], degree=[deg,deg], alpha=alpha,
        domain_name=domain_name, n_patches=n_patches,
        ref_sigmas=ref_sigmas, compute_kernel=True, show_all=True)