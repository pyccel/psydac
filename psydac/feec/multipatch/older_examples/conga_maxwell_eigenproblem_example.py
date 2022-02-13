# small script written to test Conga operators on multipatch domains, using the piecewise (broken) de Rham sequences available on every space

from mpi4py import MPI

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from sympde.expr     import TerminalExpr
from sympde.calculus import grad, dot, inner, rot, div, curl, cross
from sympde.calculus import minus, plus
from sympde.topology import NormalVector
from sympde.expr     import Norm

from sympde.topology import Derham
from sympde.topology import element_of, elements_of
from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping
from sympde.topology import VectorFunctionSpace

from sympde.expr.equation import find, EssentialBC

from sympde.expr.expr import LinearForm, BilinearForm
from sympde.expr.expr import integral

from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import LinearOperator, eigsh, minres, gmres

from scipy.sparse.linalg import inv
from scipy.linalg        import eig
from scipy.sparse import save_npz, load_npz

# from scikits.umfpack import splu    # import error



from sympde.topology import Derham
from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping

from psydac.feec.multipatch.api import discretize  # TODO: when possible, use line above
from psydac.feec.pull_push     import pull_2d_hcurl

from psydac.linalg.utilities import array_to_stencil

from psydac.fem.basic   import FemField

from psydac.api.settings        import PSYDAC_BACKENDS
from psydac.feec.multipatch.fem_linear_operators import FemLinearOperator, IdLinearOperator
from psydac.feec.multipatch.fem_linear_operators import SumLinearOperator, MultLinearOperator, ComposedLinearOperator
from psydac.feec.multipatch.operators import BrokenMass, ortho_proj_Hcurl
from psydac.feec.multipatch.operators import ConformingProjection_V0, ConformingProjection_V1, time_count
from psydac.feec.multipatch.plotting_utilities import get_grid_vals_scalar, get_grid_vals_vector
from psydac.feec.multipatch.plotting_utilities import get_plotting_grid, my_small_plot, my_small_streamplot
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain

comm = MPI.COMM_WORLD

# ---------------------------------------------------------------------------------------------------------------
# small utility for saving/loading sparse matrices, plots...
def get_fem_name(nitsche_method=None, k=None, domain_name=None,n_patches=None,nc=None,deg=None):
    assert domain_name and nc and deg
    assert nitsche_method is not None
    if nitsche_method:
        if k==1:
            method = 'nitsche_SIP'
        elif k==-1:
            method = 'nitsche_NIP'
        elif k==0:
            method = 'nitsche_IIP'
        else:
            raise NotImplementedError
    else:
        method = 'conga'
    if n_patches:
        np_suffix = '_'+repr(n_patches)
    else:
        np_suffix = ''
    return domain_name+np_suffix+'_nc'+repr(nc)+'_deg'+repr(deg)+'_'+method

def get_load_dir(nitsche_method=False, k=None, domain_name=None,n_patches=None,nc=None,deg=None,data='matrices'):
    assert data in ['matrices','solutions']
    fem_name = get_fem_name(nitsche_method=nitsche_method, k=k, domain_name=domain_name,n_patches=n_patches,nc=nc,deg=deg)
    return './saved_'+data+'/'+fem_name+'/'

# ---------------------------------------------------------------------------------------------------------------
def run_nitsche_maxwell_2d(gamma, domain, ncells, degree, kappa=None, k=None):

    from psydac.api.discretization import discretize
    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    V  = VectorFunctionSpace('V', domain, kind='hcurl')

    u, v, F  = elements_of(V, names='u, v, F')
    nn  = NormalVector('nn')

    I        = domain.interfaces
    boundary = domain.boundary

    kappa   = 10**3 if kappa is None else kappa*ncells[0]
    k       = 1     if k     is None else k

    jump = lambda w:plus(w)-minus(w)
    avr  = lambda w:(curl(plus(w)) + curl(minus(w)))/2

#    # Bilinear form a: V x V --> R

    expr   = curl(u)*curl(v) + gamma*dot(u,v)

#    expr_I  = kappa*cross(nn, jump(u))*cross(nn, jump(v))
#    expr_b =  kappa*cross(nn, u)*cross(nn, v)

#    expr_I  = cross(nn, jump(v))*curl(minus(u))\
#              +k*cross(nn, jump(u))*curl(minus(v))\
#              +kappa*cross(jump(u), nn)*cross(jump(v), nn)

    expr_I  =   cross(nn, jump(v))*avr(u)\
               +k*cross(nn, jump(u))*avr(v)\
               +kappa*cross(nn, jump(u))*cross(nn, jump(v))

    expr_b = -cross(nn, v) * curl(u) -k*cross(nn, u)*curl(v)  + kappa*cross(nn, u)*cross(nn, v)

    a = BilinearForm((u,v),  integral(domain, expr) + integral(I, expr_I) + integral(boundary, expr_b))

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    domain_h = discretize(domain, ncells=ncells, comm=comm)
    Vh       = discretize(V, domain_h, degree=degree,basis='M')

    a_h = discretize(a, domain_h, [Vh, Vh])

    A = a_h.assemble()
    return A
# ---------------------------------------------------------------------------------------------------------------

def run_maxwell_2d_eigenproblem_nitsche(nb_eigs, ncells, degree, gamma_jump,
                                domain_name='square',
                                n_patches=2,
                                load_dir=None,
                                save_dir=None,
                                plot_dir='',
                                fem_name='',
                                sigma=None,
                                test_harmonic_field=False,
                                ref_sigmas=None,
                                show_all=False,
                                ext_plots=False,
                                dpi='figure',
                                dpi_vf='figure',
                                kappa=None, k=None):
    """
    Maxwell eigenproblem solver, see eg
    Buffa, Perugia & Warburton, The Mortar-Discontinuous Galerkin Method for the 2D Maxwell Eigenproblem JSC 2009.

    :param nb_eigs: nb of eigenmodes to be computed
    :return: eigenvalues and eigenmodes
    """
 
    assert sigma is not None
    assert k     is not None

    mode =  {-1:'Non Symmetric Interior Penalty',
              0:'Incomplete Interior Penalty',
              1:'Symmetric Interior Penalty'}[k]

    print("Running Maxwell eigenproblem solver with the " + mode + " method")
    print("Looking for {nb_eigs} eigenvalues close to sigma={sigma}".format(nb_eigs=nb_eigs, sigma=sigma))

    t_stamp = time_count()
    print('building and discretizing the domain with ncells = '+repr(ncells)+'...' )
    # print("building domain and spaces...")
    domain = build_multipatch_domain(domain_name=domain_name, n_patches=n_patches)
    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())
    x,y    = domain.coordinates
    nquads = [d + 1 for d in degree]
    # plotting
    etas, xx, yy = get_plotting_grid(mappings, N=20)
    domain_h = discretize(domain, ncells=ncells, comm=comm)

    t_stamp = time_count(t_stamp)
    print('discretizing the de Rham seq with degree = '+repr(degree)+'...' )
    # multipatch de Rham sequence:
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])
    derham_h = discretize(derham, domain_h, degree=degree)
    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2
    print("V0h.nbasis = ", V0h.nbasis)
    print("V1h.nbasis = ", V1h.nbasis)
    print("V2h.nbasis = ", V2h.nbasis)

    TEST_DEBUG = False

    if TEST_DEBUG:
        # TEST V PLOT
        etas, xx, yy = get_plotting_grid(mappings, N=20)

        t_stamp = time_count(t_stamp)
        print("assembling commuting projection operators...")

        P0, P1, P2 = derham_h.projectors(nquads=nquads)

        t_stamp = time_count(t_stamp)

        hf_x = x/(x**2 + y**2)
        hf_y = y/(x**2 + y**2)

        from sympy import lambdify
        hf_x   = lambdify(domain.coordinates, hf_x)
        hf_y   = lambdify(domain.coordinates, hf_y)
        hf_log = [pull_2d_hcurl([hf_x,hf_y], f) for f in mappings_list]

        hf = P1(hf_log)

        grid_vals_hcurl = lambda v: get_grid_vals_vector(v, etas, mappings_list, space_kind='hcurl')

        hf_x_vals, hf_y_vals = grid_vals_hcurl(hf)

        my_small_streamplot(
            title=('test plot'),
            vals_x=hf_x_vals,
            vals_y=hf_y_vals,
            xx=xx,
            yy=yy,
        )

    t_stamp = time_count(t_stamp)

    print("assembling the system ...")
    A    = run_nitsche_maxwell_2d(sigma, domain, ncells, degree, kappa=kappa, k=k)
    A_m  = A.tosparse().tocsr()

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
    D1 = ComposedLinearOperator([bD1,cP1])
    I1 = IdLinearOperator(V1h)

    # Note: we could also assemble A as a psydac operator
    # D0_t = ComposedLinearOperator([cP0, bD0.transpose()])
    # D1_t = ComposedLinearOperator([cP1, bD1.transpose()])
    # A = (  ComposedLinearOperator([M1, D0, M0_inv, D0_t, M1])
    #     + gamma_jump*ComposedLinearOperator([I1-cP1,M1, I1-cP1])
    #     + ComposedLinearOperator([D1_t, M2, D1])
    #     )

    # and then convert to use eigensolver from scipy.sparse
    # A_m = A.to_sparse_matrix()

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

    L_option = 2

    compute_eigenvalues(t_stamp, domain, mappings_list, gamma_jump, sigma, ref_sigmas, 
                        A_m, M1_m, div_aux_m, cP1_m, D1_m, nb_eigs, derham_h, 
                        ncells, degree, etas, xx, yy, ext_plots, fem_name, test_harmonic_field, nitsche=True)

#----------------------------------------------------------------------------------------------------------------------------------
def run_maxwell_2d_eigenproblem_conga(nb_eigs, ncells, degree, gamma_jump,
                                domain_name='square',
                                n_patches=2,
                                load_dir=None,
                                save_dir=None,
                                plot_dir='',
                                fem_name='',
                                sigma=None,
                                test_harmonic_field=False,
                                ref_sigmas=None,
                                show_all=False,
                                ext_plots=False,
                                dpi='figure',
                                dpi_vf='figure',
                                kappa=None, k=None):
    """
    Maxwell eigenproblem solver, see eg
    Buffa, Perugia & Warburton, The Mortar-Discontinuous Galerkin Method for the 2D Maxwell Eigenproblem JSC 2009.

    :param nb_eigs: nb of eigenmodes to be computed
    :return: eigenvalues and eigenmodes
    """

    assert sigma is not None

    print("Running Maxwell eigenproblem solver with the conga method")
    print("Looking for {nb_eigs} eigenvalues close to sigma={sigma}".format(nb_eigs=nb_eigs, sigma=sigma))
    if load_dir:
        print(" -- will load matrices from " + load_dir)
    elif save_dir:
        print(" -- will save matrices in " + save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    t_stamp = time_count()
    print('building and discretizing the domain with ncells = '+repr(ncells)+'...' )
    # print("building domain and spaces...")
    domain = build_multipatch_domain(domain_name=domain_name, n_patches=n_patches)
    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())
    x,y    = domain.coordinates
    nquads = [d + 1 for d in degree]
    # plotting
    etas, xx, yy = get_plotting_grid(mappings, N=20)
    domain_h = discretize(domain, ncells=ncells, comm=comm)

    t_stamp = time_count(t_stamp)
    print('discretizing the de Rham seq with degree = '+repr(degree)+'...' )
    # multipatch de Rham sequence:
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])
    derham_h = discretize(derham, domain_h, degree=degree)
    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2
    print("V0h.nbasis = ", V0h.nbasis)
    print("V1h.nbasis = ", V1h.nbasis)
    print("V2h.nbasis = ", V2h.nbasis)

    TEST_DEBUG = False

    if TEST_DEBUG:
        # TEST V PLOT
        etas, xx, yy = get_plotting_grid(mappings, N=20)

        t_stamp = time_count(t_stamp)
        print("assembling commuting projection operators...")

        P0, P1, P2 = derham_h.projectors(nquads=nquads)

        t_stamp = time_count(t_stamp)

        hf_x = x/(x**2 + y**2)
        hf_y = y/(x**2 + y**2)

        from sympy import lambdify
        hf_x   = lambdify(domain.coordinates, hf_x)
        hf_y   = lambdify(domain.coordinates, hf_y)
        hf_log = [pull_2d_hcurl([hf_x,hf_y], f) for f in mappings_list]

        hf = P1(hf_log)

        grid_vals_hcurl = lambda v: get_grid_vals_vector(v, etas, mappings_list, space_kind='hcurl')

        hf_x_vals, hf_y_vals = grid_vals_hcurl(hf)

        my_small_streamplot(
            title=('test plot'),
            vals_x=hf_x_vals,
            vals_y=hf_y_vals,
            xx=xx,
            yy=yy,
        )

    t_stamp = time_count(t_stamp)

    if load_dir:
        print("loading sparse matrices...")
        M0_m = load_npz(load_dir+'M0_m.npz')
        M1_m = load_npz(load_dir+'M1_m.npz')
        M2_m = load_npz(load_dir+'M2_m.npz')
        M0_minv = load_npz(load_dir+'M0_minv.npz')
        cP0_m = load_npz(load_dir+'cP0_m.npz')
        cP1_m = load_npz(load_dir+'cP1_m.npz')
        D0_m = load_npz(load_dir+'D0_m.npz')
        D1_m = load_npz(load_dir+'D1_m.npz')
        I1_m = load_npz(load_dir+'I1_m.npz')
        if save_dir:
            print("(warning: save_dir argument is discarded)")

        print("loaded: M1_m.shape = " + repr(M1_m.shape))
    else:

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
        D1 = ComposedLinearOperator([bD1,cP1])
        I1 = IdLinearOperator(V1h)

        # Note: we could also assemble A as a psydac operator
        # D0_t = ComposedLinearOperator([cP0, bD0.transpose()])
        # D1_t = ComposedLinearOperator([cP1, bD1.transpose()])
        # A = (  ComposedLinearOperator([M1, D0, M0_inv, D0_t, M1])
        #     + gamma_jump*ComposedLinearOperator([I1-cP1,M1, I1-cP1])
        #     + ComposedLinearOperator([D1_t, M2, D1])
        #     )

        # and then convert to use eigensolver from scipy.sparse
        # A_m = A.to_sparse_matrix()

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

        if save_dir:
            t_stamp = time_count(t_stamp)
            print("saving sparse matrices to file...")
            save_npz(save_dir+'M0_m.npz', M0_m)
            save_npz(save_dir+'M1_m.npz', M1_m)
            save_npz(save_dir+'M2_m.npz', M2_m)
            save_npz(save_dir+'M0_minv.npz', M0_minv)
            save_npz(save_dir+'cP0_m.npz', cP0_m)
            save_npz(save_dir+'cP1_m.npz', cP1_m)
            save_npz(save_dir+'D0_m.npz', D0_m)
            save_npz(save_dir+'D1_m.npz', D1_m)
            save_npz(save_dir+'I1_m.npz', I1_m)

    ## building Hodge Laplacian matrix
    t_stamp = time_count(t_stamp)

    print("computing (sparse) Hodge-Laplacian matrix...")
    div_aux_m = D0_m.transpose() * M1_m  # note: the matrix of the (weak) div operator is:   - M0_minv * div_aux_m

    L_option = 2
    if L_option == 1:
        A_m = div_aux_m.transpose() * M0_minv * div_aux_m
    else:
        A_m = (div_aux_m * cP1_m).transpose() * M0_minv * div_aux_m * cP1_m


    jump_penal_m = I1_m-cP1_m
    A_m += (
            D1_m.transpose() * M2_m * D1_m
            + gamma_jump * jump_penal_m.transpose() * M1_m * jump_penal_m
    )

    compute_eigenvalues(t_stamp, domain, mappings_list, gamma_jump, sigma, ref_sigmas, 
                        A_m, M1_m, div_aux_m, cP1_m, D1_m, nb_eigs, derham_h, 
                        ncells, degree, etas, xx, yy, ext_plots, fem_name, test_harmonic_field, nitsche=False)

def compute_eigenvalues(t_stamp, domain, mappings_list, gamma_jump, sigma, ref_sigmas, 
                        A_m, M1_m, div_aux_m, cP1_m, D1_m, nb_eigs, derham_h, 
                        ncells, degree, etas, xx, yy, ext_plots, fem_name, test_harmonic_field, nitsche):

    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2

    if test_harmonic_field:
        print("testing harmonic field (for debugging purposes)...")

        t_stamp = time_count(t_stamp)
        print("assembling projection operators...")
        P0, P1, P2 = derham_h.projectors(nquads=nquads)

        # testing fields in kernel for circular annulus...
        harmonic_field = 2

        if harmonic_field == 1:
            # 'diverging' harmonic field: hf = ((cos theta)/r , (sin theta)/r) = (-y/r**2, x/r**2)
            hf_x = x/(x**2 + y**2)
            hf_y = y/(x**2 + y**2)
        else:
            # 'rotating' harmonic field: hf = (-(sin theta)/r , (cos theta)/r) = (-y/r**2, x/r**2)
            hf_x = -y/(x**2 + y**2)
            hf_y =  x/(x**2 + y**2)

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

    print('Finding eigenmodes and eigenvalues ... ')

    if sigma == 0:
        # computing kernel
        mode = 'normal'
        which = 'LM'
        # note: using scipy.linalg.null_space with A_m.todense() is way too slow
    else:
        mode = 'normal'
        # mode='cayley'
        # mode='buckling'
        which = 'LM'

    # from eigsh docstring:
    #   ncv = number of Lanczos vectors generated ncv must be greater than k and smaller than n;
    #   it is recommended that ncv > 2*k. Default: min(n, max(2*k + 1, 20))
    ncv = 4*nb_eigs
    # search mode: normal and buckling give a lot of zero eigenmodes. Cayley seems best for Maxwell.
    # mode='normal'

    t_stamp = time_count(t_stamp)
    print('A_m.shape = ', A_m.shape)

    print('computing eigenvalues and eigenvectors with scipy.sparse.eigsh...' )
    sigma_ref = ref_sigmas[len(ref_sigmas)//2] if nitsche else 0
    if A_m.shape[0] < 17000:   # max value for super_lu is >= 13200
        print('(with super_lu decomposition)')
        
        eigenvalues, eigenvectors = eigsh(A_m, k=nb_eigs, M=M1_m, sigma=sigma_ref, mode=mode, which=which, ncv=ncv)
    else:
        # from https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html:
        # the user can supply the matrix or operator OPinv, which gives x = OPinv @ b = [A - sigma * M]^-1 @ b.
        # > here, minres: MINimum RESidual iteration to solve Ax=b
        # suggested in https://github.com/scipy/scipy/issues/4170
        OP = A_m - sigma*M1_m
        print('(with minres iterative solver for A_m - sigma*M1_m)')
        OPinv = LinearOperator(matvec=lambda v: minres(OP, v, tol=1e-10)[0], shape=M1_m.shape, dtype=M1_m.dtype)
        # print('(with gmres iterative solver for A_m - sigma*M1_m)')
        # OPinv = LinearOperator(matvec=lambda v: gmres(OP, v, tol=1e-7)[0], shape=M1_m.shape, dtype=M1_m.dtype)
        # print('(with spsolve solver for A_m - sigma*M1_m)')
        # OPinv = LinearOperator(matvec=lambda v: spsolve(OP, v, use_umfpack=True), shape=M1_m.shape, dtype=M1_m.dtype)

        # lu = splu(OP)
        # OPinv = LinearOperator(matvec=lambda v: lu.solve(v), shape=M1_m.shape, dtype=M1_m.dtype)
        eigenvalues, eigenvectors = eigsh(A_m, k=nb_eigs, M=M1_m, sigma=sigma, mode=mode, which=which, ncv=ncv, tol=1e-10, OPinv=OPinv)

    t_stamp = time_count(t_stamp)
    print("done. eigenvalues found: " + repr(eigenvalues))

    grid_vals_h1 = lambda v: get_grid_vals_scalar(v, etas, mappings_list, space_kind='h1')
    grid_vals_hcurl = lambda v: get_grid_vals_vector(v, etas, mappings_list, space_kind='hcurl')

    curl_Pemodes_vals = []
    curl_Pemodes_titles = []
    curl_evalues = []
    other_evalues = []

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

        # normalize mode in L2
        Me = M1_m.dot(emode_sp)
        norm_emode = np.dot(emode_sp,Me)
        print('norm of computed eigenmode: ', norm_emode)

        emode      = FemField(V1h, coeffs=array_to_stencil(emode_sp/norm_emode, V1h.vector_space))
        cP_emode   = FemField(V1h, coeffs=array_to_stencil(cP1_m.dot(emode_sp), V1h.vector_space))
        curl_emode = FemField(V2h, coeffs=array_to_stencil(D1_m.dot(emode_sp), V2h.vector_space))
        # psydac version (ok if operators are there):
        # cP_emode_c = cP1(emode)
        # curl_emode = D1(emode)

        eh_x_vals, eh_y_vals = grid_vals_hcurl(emode)
        cPeh_x_vals, cPeh_y_vals = grid_vals_hcurl(cP_emode)
        Peh_abs_vals = [np.sqrt(abs(Pex)**2 + abs(Pey)**2) for Pex, Pey in zip(cPeh_x_vals, cPeh_y_vals)]
        jumps_eh_vals = [np.sqrt(abs(ex-Pex)**2 + abs(ey-Pey)**2)
                         for ex, Pex, ey, Pey in zip (eh_x_vals, cPeh_x_vals, eh_y_vals, cPeh_y_vals)]
        curl_eh_vals = grid_vals_h1(curl_emode)

        if ampl_aux_div_emode < 1e-5:
            print('seems to be a curl-curl eigenmode.')

            if nb_eigs_found < 8:
                curl_Pemodes_vals.append(Peh_abs_vals)
                curl_Pemodes_titles.append(r'$\sigma=$'+'{0:0.2f}'.format(np.real(evalue)))
                curl_evalues.append(np.real(evalue))
            else:
                print('warning: not plotting eigenmode for nb_eigs_found = ' + repr(nb_eigs_found))

            nb_eigs_found += 1
            is_curl_curl = 'Yes'
        else:
            print('does not seem to be a curl-curl eigenmode.')
            other_evalues.append(np.real(evalue))
            is_curl_curl = 'No'

        if show_all:
            if fem_name:
                fig_name=plot_dir+'HL_emode_k='+repr(k_eig)+'.png'  # +'_'+fem_name+'.png'
                fig_name_vf=plot_dir+'HL_emode_k='+repr(k_eig)+'_vf.png'   # +'_vf_'+fem_name+'.png'
            else:
                fig_name=None
                fig_name_vf=None

            print('len(Peh_abs_vals) = ',len(Peh_abs_vals))
            if ext_plots:
                title=('mode k:'+repr(k_eig)+' -- eigenvalue: '+repr(evalue)+' -- is curl_curl: '+is_curl_curl)
                vals=[eh_x_vals, eh_y_vals, Peh_abs_vals, jumps_eh_vals, curl_eh_vals]
                titles=[r'$e^h_{k,x}$', r'$e^h_{k,y}$', r'$|P^1_c e^h_k|$', r'$|(I-P^1_c) e^h_k|$', r'curl$(e^h_k)$']
            else:
                # lambda is std notation for eigenvalue
                title=('eigenmode for $\lambda_{k,h}$ = '+repr(evalue))
                vals=[Peh_abs_vals,]
                titles=[r'$|P^1_c e_{k,h}|$',]

            my_small_plot(
                title=title,
                vals=vals,
                titles=titles,
                xx=xx,
                yy=yy,
                cmap='hsv',
                save_fig=fig_name,
                dpi=dpi,
                show_xylabel=False,
            )

            my_small_streamplot(
                title=title,
                vals_x=eh_x_vals,
                vals_y=eh_y_vals,
                xx=xx,
                yy=yy,
                save_fig=fig_name_vf,
                dpi=dpi_vf
            )

        k_eig += 1

    if fem_name:
        fig_name=plot_dir+'HL_emodes.png'  # _'+fem_name+'.png'
    else:
        fig_name=None
    my_small_plot(
        title=r'Amplitude $|P^1_c e^h_k|$ of some curl eigenmodes for ncells = {nc} and degree = {deg}'.format(nc=ncells[0],deg=degree[0]),
        vals=curl_Pemodes_vals,
        titles=curl_Pemodes_titles,
        xx=xx,
        yy=yy,
        cmap='magma',
        save_fig=fig_name,
        show_xylabel=False,
    )


    t_stamp = time_count(t_stamp)
    print('done -- summary: ')

    print("using jump penalization factor gamma_jump = ", gamma_jump )
    print('nb of spline cells per patch: ' + repr(ncells))
    h = 1/ncells[0]
    print('-- corresponding to h: '+ repr(h))
    print('degree: ' + repr(degree))

    nb_dofs = len(emode_sp)
    print(' -- nb of DOFS: ' + repr(nb_dofs))

    print('computed eigenvalues for curl curl operator: ')
    print(curl_evalues)

    print('other eigenvalues (for grad div) operator: ')
    print(other_evalues)

    if ref_sigmas is not None:
        errors = []
        n_errs = min(len(ref_sigmas), len(curl_evalues))
        for k in range(n_errs):
            errors.append(abs(curl_evalues[k]-ref_sigmas[k]))

        print('errors from reference eigenvalues: ')
        print(errors)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        description     = "Solve 2D eigenvalue problem of the Time Harmonic Maxwell equations."
    )

    parser.add_argument('ncells',
        type = int,
        help = 'Number of cells in domain'
    )

    parser.add_argument('degree',
        type = int,
        help = 'Polynomial spline degree'
    )

    parser.add_argument( '--domain',
        choices = ['square', 'annulus', 'curved_L_shape', 'pretzel', 'pretzel_annulus', 'pretzel_debug'],
        default = 'curved_L_shape',
        help    = 'Domain'
    )

    parser.add_argument( '--mode',
        choices = ['conga', 'nitsche'],
        default = 'conga',
        help    = 'Maxwell solver'
    )

    parser.add_argument( '--k',
        type    = int,
        choices = [-1, 0, 1],
        default = 1,
        help    = 'Nitsche method (NIP, IIP, IP)'
    )

    parser.add_argument( '--kappa',
        type    = int,
        default = 10,
        help    = 'Nitsche stabilization term'
    )

    # Read input arguments
    args        = parser.parse_args()
    deg         = args.degree
    nc          = args.ncells
    domain_name = args.domain
    mode        = args.mode
    kappa       = args.kappa
    k           = args.k
    
    nitsche_method = mode == 'nitsche'
    if mode == 'conga':
        run_maxwell_2d_eigenproblem =  run_maxwell_2d_eigenproblem_conga
    else:
        assert nitsche_method
        run_maxwell_2d_eigenproblem = run_maxwell_2d_eigenproblem_nitsche

    # from scipy.sparse import rand
    # A = rand(m=14300, n=14300)
    # A = rand(m=15300, n=15300)
    # A = rand(m=22000, n=22000)
    # res = eigsh(A, 1, sigma=50);
    # print(res)
    # exit()

    # domain_name = 'pretzel'  #_debug'

    # valid parameters for curved_L_shape (V1 dofs around 10.000)
    # nc = 40; deg = 3
    # nc = 40; deg = 3
    # nc = 40; deg = 5
    # (nc, deg = 50, 2 is too large for super_lu)

    plot_dir_suffix = ''
    # valid parameters for pretzel (V1 dofs around 10.000)
    #nc = 2**4; deg = 2  # OK
    # nc = 20; deg = 2  # OK
    # nc = 20; deg = 4  # OK -- V1 dofs: 12144
    # nc = 20; deg = 5  # OK -- V1 dofs: 13200
    # nc = 20; deg = 8  # OK --
    # nc=20
    # nc=8
    #nc=10

    # (nc, deg = 30, 2 is too large for super_lu)

    # jump penalization factor from Buffa, Perugia and Warburton  >> need to study
    h = 1/nc
    DG_gamma = 10*(deg+1)**2/h
    # DG_gamma = 10*(deg)**2/h
    gamma_jump = DG_gamma

    show_all = False
    plot_all = True
    dpi = 400
    dpi_vf = 200
    # show_all = True
    # plot_all = False

    nb_eigs = 16
    n_patches = None
    ref_sigmas = None
    save_dir = None
    load_dir = None
    nitsche  = False

    if domain_name == 'square':
        n_patches = 6
        sigma = 0
    elif domain_name == 'annulus':
        n_patches = 4
        sigma = 0
    elif domain_name == 'curved_L_shape':
        sigma = 0
        ref_sigmas = [
            0.181857115231E+01,
            0.349057623279E+01,
            0.100656015004E+02,
            0.101118862307E+02,
            0.124355372484E+02,
            ]
        nb_eigs=14  # need a bit more, to get rid of grad-div eigenmodes
    elif domain_name in ['pretzel', 'pretzel_debug']:
        # radii used in the pretzel_J source test case
        sigma = 64
        plot_dir_suffix = '_sigma_64'
        if sigma == 0 and domain_name == 'pretzel':
            nb_eigs = 16
            ref_sigmas = [
                0,
                0,
                0,
                0.1795447761871659,
                0.19922705025897117,
                0.699286528403241,
                0.8709410737744409,
                1.1945444491250097,
            ]
        else:
            nb_eigs = 30
        # note: nc = 2**5 and deg = 2 gives a matrix too big for super_lu factorization...
    else:
        raise NotImplementedError


    fem_name = get_fem_name(nitsche_method=nitsche_method, k=k, domain_name=domain_name,n_patches=n_patches,nc=nc,deg=deg) #domain_name+np_suffix+'_nc'+repr(nc)+'_deg'+repr(deg)
    save_dir = load_dir = get_load_dir(domain_name=domain_name,n_patches=n_patches,nc=nc,deg=deg)  # './tmp_matrices/'+fem_name+'/'
    # save_dir = './tmp_matrices/'+domain_name+np_suffix+'_nc'+repr(nc)+'_deg'+repr(deg)+'/'
    # load_dir = save_dir

    plot_dir = './plots/'+fem_name+plot_dir_suffix+'/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if plot_all:
        show_all=True
        # will also use above value of fem_name
    else:
        # reset fem_name to disable plots
        fem_name = ''

    # possible domain shapes:
    assert domain_name in ['square', 'annulus', 'curved_L_shape', 'pretzel', 'pretzel_annulus', 'pretzel_debug']

    if load_dir and not os.path.exists(load_dir):
        print(' -- note: discarding absent load directory')
        load_dir = None

    run_maxwell_2d_eigenproblem(
        nb_eigs=nb_eigs, ncells=[nc, nc], degree=[deg,deg], gamma_jump=gamma_jump,
        domain_name=domain_name, n_patches=n_patches,
        save_dir=save_dir, load_dir=load_dir, plot_dir=plot_dir, fem_name=fem_name,
        ref_sigmas=ref_sigmas, sigma=sigma, show_all=show_all, dpi=dpi, dpi_vf=dpi_vf, kappa=kappa, k=k
    )

