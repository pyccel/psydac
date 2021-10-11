# small script written to test Conga operators on multipatch domains, using the piecewise (broken) de Rham sequences available on every space

from mpi4py import MPI

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from sympy import pi, cos, sin, Matrix, Tuple, Max, exp
from sympy import symbols
from sympy import lambdify

from sympde.expr     import TerminalExpr
from sympde.calculus import grad, dot, inner, rot, div, curl, cross
from sympde.calculus import minus, plus
from sympde.topology import NormalVector
from sympde.expr     import Norm

from sympde.topology import Derham
from sympde.topology import element_of, elements_of, Domain

from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping
from sympde.topology import VectorFunctionSpace

from sympde.expr.equation import find, EssentialBC

from sympde.expr.expr import LinearForm, BilinearForm
from sympde.expr.expr import integral

from scipy.sparse.linalg import spsolve, spilu, cg, lgmres
from scipy.sparse.linalg import LinearOperator, eigsh, minres, gmres

from scipy.sparse.linalg import inv
from scipy.sparse.linalg import norm as spnorm
from scipy.linalg        import eig, norm
from scipy.sparse import save_npz, load_npz

# from scikits.umfpack import splu    # import error



from sympde.topology import Derham
from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping

from psydac.feec.multipatch.api import discretize
from psydac.feec.pull_push     import pull_2d_hcurl

from psydac.linalg.utilities import array_to_stencil

from psydac.fem.basic   import FemField

from psydac.api.settings        import PSYDAC_BACKENDS
from psydac.feec.multipatch.fem_linear_operators import FemLinearOperator, IdLinearOperator
from psydac.feec.multipatch.fem_linear_operators import SumLinearOperator, MultLinearOperator, ComposedLinearOperator
from psydac.feec.multipatch.operators import BrokenMass, get_K0_and_K0_inv, get_K1_and_K1_inv, get_M_and_M_inv
from psydac.feec.multipatch.operators import ConformingProjection_V0, ConformingProjection_V1, time_count
from psydac.feec.multipatch.plotting_utilities import get_grid_vals_scalar, get_grid_vals_vector, get_grid_quad_weights
from psydac.feec.multipatch.plotting_utilities import get_plotting_grid, my_small_plot, my_small_streamplot
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain, get_ref_eigenvalues

comm = MPI.COMM_WORLD

# ---------------------------------------------------------------------------------------------------------------
# small utility for saving/loading sparse matrices, plots...
def rhs_fn(source_type,nbc=False):
    if nbc:
        # additional terms for nitsche bc
        fn = 'rhs_'+source_type+'_nbc.npz'
    else:
        fn = 'rhs_'+source_type+'.npz'
    return fn

def E_ref_fn(source_type, N_diag):
    return 'E_ref_'+source_type+'_N'+repr(N_diag)+'.npz'

def Eh_coeffs_fn(source_type, N_diag):
    return 'Eh_coeffs_'+source_type+'_N'+repr(N_diag)+'.npz'

def error_fn(source_type=None, method=None, k=None, domain_name=None,deg=None):
    return 'errors/error_'+domain_name+'_'+source_type+'_'+'_deg'+repr(deg)+'_'+get_method_name(method, k)+'.txt'

def get_method_name(method=None, k=None, geo_cproj=None, penal_regime=None):
    if method == 'nitsche':
        method_name = method
        if k==1:
            method_name += '_SIP'
        elif k==-1:
            method_name += '_NIP'
        elif k==0:
            method_name += '_IIP'
        else:
            assert k is None
    elif method == 'conga':
        method_name = method
        if geo_cproj is not None:
            if geo_cproj:
                method_name += '_GSP'  # Geometric-Spline-Projection
            else:
                method_name += '_BSP'  # B-Spline-Projection
    else:
        raise ValueError(method)
    if penal_regime is not None:
        method_name += '_pr'+repr(penal_regime)

    return method_name

def get_fem_name(method=None, k=None, DG_full=False, geo_cproj=None, domain_name=None,nc=None,deg=None):
    assert domain_name and nc and deg
    fn = domain_name+'_nc'+repr(nc)+'_deg'+repr(deg)
    if DG_full:
        fn += '_fDG'
    if method is not None:
        fn += '_'+get_method_name(method, k, geo_cproj)
    return fn

def get_load_dir(method=None, DG_full=False, domain_name=None,nc=None,deg=None,data='matrices'):
    assert data in ['matrices','solutions','rhs']
    if method is None:
        assert data == 'rhs'
    fem_name = get_fem_name(domain_name=domain_name,method=method, nc=nc,deg=deg, DG_full=DG_full)
    return './saved_'+data+'/'+fem_name+'/'


# ---------------------------------------------------------------------------------------------------------------
def nitsche_curl_curl_2d(domain_h, Vh, gamma_h=None, k=None, load_dir=None, backend_language='python', need_mass_matrix=False):
    """
    computes
        K_m the k-IP matrix of the curl-curl operator with penalization parameter gamma
        (as defined eg in Buffa, Houston & Perugia, JCAM 2007)

    :param k: parameter for SIP/NIP/IIP
    :return: matrices in sparse format
    """
    assert gamma_h is not None

    M_m = None
    got_mass_matrix = (not need_mass_matrix)

    if os.path.exists(load_dir):
        print(" -- load directory " + load_dir + " found -- will load the Nitsche matrices from there...")

        # unpenalized curl-curl matrix (main part and symmetrization term)
        CC_m = load_npz(load_dir+'CC_m.npz')
        CS_m = load_npz(load_dir+'CS_m.npz')
        # jump penalization matrix
        JP_m = load_npz(load_dir+'JP_m.npz')
        # mass matrix
        if need_mass_matrix:
            try:
                M_m = load_npz(load_dir+'M_m.npz')
                got_mass_matrix = True
            except:
                print(" -- (mass matrix not found)")
    else:
        print(" -- load directory " + load_dir + " not found -- will assemble the Nitsche matrices...")

        t_stamp = time_count()
        print('computing IP curl-curl matrix with k = {0} and penalization gamma_h = {1}'.format(k, gamma_h))

        #+++++++++++++++++++++++++++++++
        # Abstract IP model
        #+++++++++++++++++++++++++++++++

        V = Vh.symbolic_space
        domain = V.domain

        u, v  = elements_of(V, names='u, v')
        nn  = NormalVector('nn')

        I        = domain.interfaces
        boundary = domain.boundary

        jump = lambda w:plus(w)-minus(w)
        avr_curl = lambda w:(curl(plus(w)) + curl(minus(w)))/2

        # Bilinear forms a: V x V --> R

        # note (MCP): the IP formulation involves the tangential jumps [v]_T = n^- x v^- + n^+ x v^+
        # here this t-jump corresponds to -cross(nn, jump(v))
        expr   = curl(u)*curl(v)
        expr_I =   cross(nn, jump(v))*avr_curl(u)
        expr_b = -cross(nn, v) * curl(u)
        expr_Is =  cross(nn, jump(u))*avr_curl(v)   # symmetrization terms
        expr_bs = -cross(nn, u)*curl(v)

        # jump penalization terms:
        expr_jp_I = cross(nn, jump(u))*cross(nn, jump(v))
        expr_jp_b = cross(nn, u)*cross(nn, v)

        a_cc = BilinearForm((u,v),  integral(domain, expr) + integral(I, expr_I) + integral(boundary, expr_b))
        a_cs = BilinearForm((u,v),  integral(I, expr_Is) + integral(boundary, expr_bs))  # symmetrization terms
        a_jp = BilinearForm((u,v),  integral(I, expr_jp_I) + integral(boundary, expr_jp_b))

        #+++++++++++++++++++++++++++++++
        # 2. Discretization
        #+++++++++++++++++++++++++++++++

        # domain_h = discretize(domain, ncells=ncells, comm=comm)
        # Vh       = discretize(V, domain_h, degree=degree,basis='M')

        # unpenalized curl-curl matrix (incomplete)
        a_h = discretize(a_cc, domain_h, [Vh, Vh], backend=PSYDAC_BACKENDS[backend_language])
        A = a_h.assemble()
        CC_m  = A.tosparse().tocsr()

        # symmetrization part (for SIP or NIP curl-curl matrix)
        a_h = discretize(a_cs, domain_h, [Vh, Vh], backend=PSYDAC_BACKENDS[backend_language])
        A = a_h.assemble()
        CS_m  = A.tosparse().tocsr()

        # jump penalization matrix
        a_h = discretize(a_jp, domain_h, [Vh, Vh], backend=PSYDAC_BACKENDS[backend_language])
        A = a_h.assemble()
        JP_m  = A.tosparse().tocsr()

        print(" -- now saving these matrices in " + load_dir + "...")
        os.makedirs(load_dir)
        t_stamp = time_count(t_stamp)
        save_npz(load_dir+'CC_m.npz', CC_m)
        save_npz(load_dir+'CS_m.npz', CS_m)
        save_npz(load_dir+'JP_m.npz', JP_m)
        time_count(t_stamp)

    if not got_mass_matrix:
        print(" -- assembling the mass matrix (and saving to file)...")
        V = Vh.symbolic_space
        domain = V.domain
        u, v  = elements_of(V, names='u, v')
        expr   = dot(u,v)
        a_m  = BilinearForm((u,v),  integral(domain, expr))
        m_h = discretize(a_m, domain_h, [Vh, Vh], backend=PSYDAC_BACKENDS[backend_language])
        M = m_h.assemble()
        M_m  = M.tosparse().tocsr()
        save_npz(load_dir+'M_m.npz', M_m)

    K_m = CC_m + k*CS_m + gamma_h*JP_m

    return K_m, M_m


# ---------------------------------------------------------------------------------------------------------------
def get_elementary_conga_matrices(domain_h, derham_h, load_dir=None, backend_language='python', discard_non_hom_matrices=False):

    if os.path.exists(load_dir):
        print(" -- load directory " + load_dir + " found -- will load the CONGA matrices from there...")

        # print("loading sparse matrices...")
        M0_m = load_npz(load_dir+'M0_m.npz')
        M1_m = load_npz(load_dir+'M1_m.npz')
        M2_m = load_npz(load_dir+'M2_m.npz')
        M0_minv = load_npz(load_dir+'M0_minv.npz')
        cP0_m = load_npz(load_dir+'cP0_m.npz')
        cP1_m = load_npz(load_dir+'cP1_m.npz')
        cP0_hom_m = load_npz(load_dir+'cP0_hom_m.npz')
        cP1_hom_m = load_npz(load_dir+'cP1_hom_m.npz')
        bD0_m = load_npz(load_dir+'bD0_m.npz')
        bD1_m = load_npz(load_dir+'bD1_m.npz')
        I1_m = load_npz(load_dir+'I1_m.npz')

        # print('loaded.')
    else:
        print(" -- load directory " + load_dir + " not found -- will assemble the CONGA matrices...")

        V0h = derham_h.V0
        V1h = derham_h.V1
        V2h = derham_h.V2

        # Mass matrices for broken spaces (block-diagonal)
        t_stamp = time_count()
        print("assembling mass matrix operators...")

        M0 = BrokenMass(V0h, domain_h, is_scalar=True, backend_language=backend_language)
        M1 = BrokenMass(V1h, domain_h, is_scalar=False, backend_language=backend_language)
        M2 = BrokenMass(V2h, domain_h, is_scalar=True, backend_language=backend_language)

        t_stamp = time_count(t_stamp)
        print('----------     inv M0')
        # M0_m = M0.to_sparse_matrix()
        # M0_minv = inv(M0_m.tocsc())  # todo: assemble patch-wise M0_inv, as Hodge operator
        M0_minv = M0.get_sparse_inverse_matrix()

        t_stamp = time_count(t_stamp)
        print("assembling conf projection operators for V1...")
        # todo: disable the non-hom-bc operators for hom-bc pretzel test cases...
        cP1_hom = ConformingProjection_V1(V1h, domain_h, hom_bc=True, backend_language=backend_language)
        t_stamp = time_count(t_stamp)
        print("assembling conf projection operators for V0...")
        cP0_hom = ConformingProjection_V0(V0h, domain_h, hom_bc=True, backend_language=backend_language)
        t_stamp = time_count(t_stamp)
        if discard_non_hom_matrices:
            print('WARNING: discarding the non-homogeneous cP0 and cP1 projection operators!')
            cP0 = cP0_hom
            cP1 = cP1_hom
        else:
            cP0 = ConformingProjection_V0(V0h, domain_h, hom_bc=False, backend_language=backend_language)
            cP1 = ConformingProjection_V1(V1h, domain_h, hom_bc=False, backend_language=backend_language)

        t_stamp = time_count(t_stamp)
        print("assembling broken derivative operators...")
        bD0, bD1 = derham_h.broken_derivatives_as_operators

        # t_stamp = time_count(t_stamp)
        # print("assembling conga derivative operators...")

        # D0 = ComposedLinearOperator([bD0,cP0])
        # D1 = ComposedLinearOperator([bD1,cP1])
        I1 = IdLinearOperator(V1h)

        t_stamp = time_count(t_stamp)
        print("converting in sparse matrices...")
        M0_m = M0.to_sparse_matrix()
        M1_m = M1.to_sparse_matrix()
        M2_m = M2.to_sparse_matrix()
        cP0_m = cP0.to_sparse_matrix()
        cP1_m = cP1.to_sparse_matrix()
        cP0_hom_m = cP0_hom.to_sparse_matrix()
        cP1_hom_m = cP1_hom.to_sparse_matrix()
        bD0_m = bD0.to_sparse_matrix()  # broken (patch-local) differential
        bD1_m = bD1.to_sparse_matrix()
        I1_m = I1.to_sparse_matrix()
        t_stamp = time_count(t_stamp)


        print(" -- now saving these matrices in " + load_dir)
        os.makedirs(load_dir)

        t_stamp = time_count(t_stamp)
        save_npz(load_dir+'M0_m.npz', M0_m)
        save_npz(load_dir+'M1_m.npz', M1_m)
        save_npz(load_dir+'M2_m.npz', M2_m)
        save_npz(load_dir+'M0_minv.npz', M0_minv)
        save_npz(load_dir+'cP0_m.npz', cP0_m)
        save_npz(load_dir+'cP1_m.npz', cP1_m)
        save_npz(load_dir+'cP0_hom_m.npz', cP0_hom_m)
        save_npz(load_dir+'cP1_hom_m.npz', cP1_hom_m)
        save_npz(load_dir+'bD0_m.npz', bD0_m)
        save_npz(load_dir+'bD1_m.npz', bD1_m)
        save_npz(load_dir+'I1_m.npz', I1_m)
        time_count(t_stamp)

    print('ok, got the matrices. Some shapes are: \n M0_m = {0}\n M1_m = {1}\n M2_m = {2}'.format(M0_m.shape,M1_m.shape,M2_m.shape))

    V0h = derham_h.V0
    K0, K0_inv = get_K0_and_K0_inv(V0h, uniform_patches=True)
    V1h = derham_h.V1
    K1, K1_inv = get_K1_and_K1_inv(V1h, uniform_patches=True)

    print('  -- some more shapes: \n K0 = {0}\n K1_inv = {1}\n'.format(K0.shape,K1_inv.shape))

    M_mats = [M0_m, M1_m, M2_m, M0_minv]
    P_mats = [cP0_m, cP1_m, cP0_hom_m, cP1_hom_m]
    D_mats = [bD0_m, bD1_m]
    IK_mats = [I1_m, K0, K0_inv, K1, K1_inv]

    return M_mats, P_mats, D_mats, IK_mats


def conga_curl_curl_2d(M1_m=None, M2_m=None, cP1_m=None, cP1_hom_m=None, bD1_m=None, I1_m=None, epsilon=1, gamma_h=None, hom_bc=True):
    """
    computes
        K_hom_m (and K_m if not hom_bc)
        the CONGA stiffness matrix of the vector-valued curl-curl operator in V1, with (and without) homogeneous bc
    """

    if not hom_bc:
        assert cP1_m is not None
    print('computing Conga curl_curl matrix with penalization gamma_h = {}'.format(gamma_h))
    t_stamp = time_count()
    assert operator == 'curl_curl'  # todo: implement also the hodge-laplacian ?

    # curl_curl matrix (left-multiplied by M1_m) :
    D1_hom_m = bD1_m * cP1_hom_m
    jump_penal_hom_m = I1_m-cP1_hom_m
    # print(" warning -- WIP on K_hom_m -- 8767654659747644864")
    K_hom_m = (
                  (1/epsilon) * D1_hom_m.transpose() * M2_m * D1_hom_m
                + gamma_h * jump_penal_hom_m.transpose() * M1_m * jump_penal_hom_m
    )

    if not hom_bc:
        # then we also need the matrix of the non-homogeneous operator -- unpenalized
        D1_m = bD1_m * cP1_m
        K_bc_m = (
                D1_hom_m.transpose() * M2_m * D1_m
        )
    else:
        K_bc_m = None
    time_count(t_stamp)

    return K_hom_m, K_bc_m


def conga_operators_2d(domain, ncells, degree, operator='curl_curl', gamma_h=None, hom_bc=True):
    """
    OBSOLETE -- to update for single hodge_laplacian operator

    computes:
        K_hom_m and K_m the CONGA stiffness matrix of 'operator' in V1 with and without homogeneous bc
        (stiffness = left-multiplied by M1_m),
        with penalization parameter gamma
        with:
            K = curl curl               (if operator == 'curl_curl')
        or
            K = curl curl + grad div    (if operator == 'hodge_laplacian')
        as defined in Campos Pinto and Güçlü (preprint 2021)

    :return: matrices in sparse format
    """
    assert operator in ['curl_curl', 'hodge_laplacian']

    # t_stamp = time_count()
    print('computing Conga {0} matrix with penalization gamma_h = {1}'.format(operator, gamma_h))

    assert operator == 'curl_curl'  # todo: implement (verify) the hodge-laplacian

    # curl_curl matrix (left-multiplied by M1_m) :
    D1_hom_m = bD1_m * cP1_hom_m
    jump_penal_hom_m = I1_m-cP1_hom_m
    K_hom_m = (
            D1_hom_m.transpose() * M2_m * D1_hom_m
            + gamma_h * jump_penal_hom_m.transpose() * M1_m * jump_penal_hom_m
    )

    if not hom_bc:
        # then we also need the matrix of the non-homogeneous operator
        D1_m = bD1_m * cP1_m
        # jump_penal_m = I1_m-cP1_m
        K_bc_m = (
                D1_hom_m.transpose() * M2_m * D1_m
        )
    else:
        K_bc_m = None

    if operator == 'hodge_laplacian':
        D0_hom_m = bD0_m * cP0_hom_m
        div_aux_m = D0_hom_m.transpose() * M1_m  # note: the matrix of the (weak) div operator is:   - M0_minv * div_aux_m
        K_hom_m += div_aux_m.transpose() * M0_minv * div_aux_m

        if not hom_bc:
            raise NotImplementedError
            # todo: this is not correct -- find proper formulation for the non-homogeneous case
            D0_m = bD0_m * cP0_m
            div_aux_m = D0_m.transpose() * M1_m  # note: the matrix of the (weak) div operator is:   - M0_minv * div_aux_m
            K_bc_m += div_aux_m.transpose() * M0_minv * div_aux_m

    return K_hom_m, K_bc_m


# ---------------------------------------------------------------------------------------------------------------


def get_eigenvalues(nb_eigs, sigma, A_m, M_m):
    print('-----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  ----- ')
    print('computing {0} eigenvalues (and eigenvectors) close to sigma={1} with scipy.sparse.eigsh...'.format(nb_eigs, sigma) )

    if sigma == 0:
        # computing kernel
        mode = 'normal'
        which = 'LM'
    else:
        # ahah
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

    t_stamp = time_count()
    print('A_m.shape = ', A_m.shape)
    # print('getting sigma = ', sigma)
    # sigma_ref = ref_sigmas[len(ref_sigmas)//2] if nitsche else 0
    if A_m.shape[0] < 17000:   # max value for super_lu is >= 13200
        print('(with super_lu decomposition)')
        eigenvalues, eigenvectors = eigsh(A_m, k=nb_eigs, M=M_m, sigma=sigma, mode=mode, which=which, ncv=ncv)
    else:
        # from https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html:
        # the user can supply the matrix or operator OPinv, which gives x = OPinv @ b = [A - sigma * M]^-1 @ b.
        # > here, minres: MINimum RESidual iteration to solve Ax=b
        # suggested in https://github.com/scipy/scipy/issues/4170
        OP = A_m - sigma*M_m
        print('(with minres iterative solver for A_m - sigma*M1_m)')
        OPinv = LinearOperator(matvec=lambda v: minres(OP, v, tol=1e-10)[0], shape=M_m.shape, dtype=M_m.dtype)
        # print('(with gmres iterative solver for A_m - sigma*M1_m)')
        # OPinv = LinearOperator(matvec=lambda v: gmres(OP, v, tol=1e-7)[0], shape=M1_m.shape, dtype=M1_m.dtype)
        # print('(with spsolve solver for A_m - sigma*M1_m)')
        # OPinv = LinearOperator(matvec=lambda v: spsolve(OP, v, use_umfpack=True), shape=M1_m.shape, dtype=M1_m.dtype)

        # lu = splu(OP)
        # OPinv = LinearOperator(matvec=lambda v: lu.solve(v), shape=M1_m.shape, dtype=M1_m.dtype)
        eigenvalues, eigenvectors = eigsh(A_m, k=nb_eigs, M=M_m, sigma=sigma, mode=mode, which=which, ncv=ncv, tol=1e-10, OPinv=OPinv)

    time_count(t_stamp)
    print("done. eigenvalues found: " + repr(eigenvalues))
    return eigenvalues, eigenvectors


def get_source_and_solution(source_type, eta, domain, refsol_params=None):
    """
    get source and ref solution of time-Harmonic Maxwell equation
        curl curl E + eta E = J

    """

    assert refsol_params
    nc_ref, deg_ref, N_diag, method_ref = refsol_params

    # ref solution (values on diag grid)
    E_ref_vals = None

    # bc solution: describe the bc on boundary. Inside domain, values should not matter. Homogeneous bc will be used if None
    E_bc = None
    E_ex = None

    x,y    = domain.coordinates

    if source_type == 'manu_J':
        # use a manufactured solution, with ad-hoc (homogeneous or inhomogeneous) bc
        if domain_name in ['square_2', 'square_6', 'square_8', 'square_9']:
            theta = 1
        else:
            theta = pi

            # E_ex    = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
            # f      = Tuple(eta*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
            #                  eta*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))
        E_ex   = Tuple(sin(theta*y), sin(theta*x)*cos(theta*y))
        f      = Tuple(eta*sin(theta*y) - theta**2*sin(theta*y)*cos(theta*x) + theta**2*sin(theta*y),
                         eta*sin(theta*x)*cos(theta*y) + theta**2*sin(theta*x)*cos(theta*y))
        E_ex_x = lambdify(domain.coordinates, E_ex[0])
        E_ex_y = lambdify(domain.coordinates, E_ex[1])
        E_ex_log = [pull_2d_hcurl([E_ex_x,E_ex_y], f) for f in mappings_list]
        E_ref_x_vals, E_ref_y_vals   = grid_vals_hcurl_cdiag(E_ex_log)
        E_ref_vals = [E_ref_x_vals, E_ref_y_vals]
        # print(E_ex_x)

        # boundary condition: (here we only need to coincide with E_ex on the boundary !)
        if domain_name in ['square_2', 'square_6', 'square_9']:
            E_bc = None
        # elif domain_name == 'square_8':
        #     E_bc = Tuple(sin(theta*y) * (1+(x-pi/3)*(x-2*pi/3)*(y-pi/3)*(y-2*pi/3)), sin(theta*x)*cos(theta*y) * (1+(x-pi/3)*(x-2*pi/3)*(y-pi/3)*(y-2*pi/3)))
        else:
            E_bc = E_ex

    elif source_type == 'dp_J':
        # div-free J
        f = Tuple(10*sin(y), -10*sin(x))

    elif source_type == 'cf_J':
        # curl-free J
        f = Tuple(10*sin(x), -10*sin(y))

    elif source_type == 'ring_J':

        # 'rotating' (divergence-free) J field:
        #   J = j(r) * (-sin theta, cos theta)

        if domain_name in ['square_2', 'square_6', 'square_8', 'square_9']:
            r0 = np.pi/4
            dr = 0.1
            x0 = np.pi/2
            y0 = np.pi/2
            omega = 43/2
            # alpha  = -omega**2  # not a square eigenvalue
            J_factor = 100

        elif domain_name in ['curved_L_shape']:
            r0 = np.pi/4
            dr = 0.1
            x0 = np.pi/2
            y0 = np.pi/2
            omega = 43/2
            # alpha  = -omega**2  # not a square eigenvalue
            J_factor = 100

        else:
            # for pretzel

            # omega = 8  # ?
            # alpha  = -omega**2

            source_option = 2

            if source_option==1:
                # big circle:
                r0 = 2.4
                dr = 0.05
                x0 = 0
                y0 = 0.5
                J_factor = 10

            elif source_option==2:
                # small circle in corner:
                r0 = 1
                dr = 0.2
                x0 = 1.5
                y0 = 1.5
                J_factor = 10

            elif source_option==3:
                # small circle in corner, seems less interesting
                r0 = 0.0
                dr = 0.05
                x0 = 0.9
                y0 = 0.9
                J_factor = 10
            else:
                raise NotImplementedError

        # note: some other currents give sympde or numba errors, see below [1]
        J_x = -J_factor * (y-y0) * exp( - .5*(( (x-x0)**2 + (y-y0)**2 - r0**2 )/dr)**2 )   # /(x**2 + y**2)
        J_y =  J_factor * (x-x0) * exp( - .5*(( (x-x0)**2 + (y-y0)**2 - r0**2 )/dr)**2 )

        f = Tuple(J_x, J_y)

        E_ref_filename = get_load_dir(method=method_ref, domain_name=domain_name,nc=nc_ref,deg=deg_ref,data='solutions')+E_ref_fn(source_type, N_diag)
        if os.path.isfile(E_ref_filename):
            print("getting ref solution values from file "+E_ref_filename)
            with open(E_ref_filename, 'rb') as file:
                E_ref_vals = np.load(file)
                # check form of ref values
                # assert 'x_vals' in E_ref_vals; assert 'y_vals' in E_ref_vals
                E_ref_x_vals = E_ref_vals['x_vals']
                E_ref_y_vals = E_ref_vals['y_vals']
                assert isinstance(E_ref_x_vals, (list, np.ndarray)) and isinstance(E_ref_y_vals, (list, np.ndarray))
            E_ref_vals = [E_ref_x_vals, E_ref_y_vals]
        else:
            print("-- no ref solution file '"+E_ref_filename+"', skipping it")
    else:
        raise ValueError(source_type)

    return f, E_bc, E_ref_vals, E_ex

# --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * ---
# --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * --- * ---

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        description     = "Solve 2D curl-curl eigenvalue or source problem."
    )

    parser.add_argument('ncells',
        type = int,
        help = 'Number of cells in every patch'
    )

    parser.add_argument('degree',
        type = int,
        help = 'Polynomial spline degree'
    )

    parser.add_argument( '--domain',
        choices = ['square_2', 'square_6', 'square_8', 'square_9', 'annulus', 'curved_L_shape', 'pretzel', 'pretzel_f', 'pretzel_annulus', 'pretzel_debug'],
        default = 'curved_L_shape',
        help    = 'Domain'
    )

    parser.add_argument( '--method',
        choices = ['conga', 'nitsche'],
        default = 'conga',
        help    = 'Maxwell solver'
    )

    parser.add_argument( '--k',
        type    = int,
        choices = [-1, 0, 1],
        default = 1,
        help    = 'type of Nitsche IP method (NIP, IIP, SIP)'
    )

    parser.add_argument( '--DG_full',
        action  = 'store_true',
        help    = 'whether DG (Nitsche) method is used with full polynomials spaces'
    )

    parser.add_argument( '--proj_sol',
        action  = 'store_true',
        help    = 'whether cP1 is applied to solution of source problem'
    )

    parser.add_argument( '--no_plots',
        action  = 'store_true',
        help    = 'whether plots are done'
    )

    parser.add_argument( '--hide_plots',
        action  = 'store_true',
        help    = 'whether plots are hidden'
    )

    parser.add_argument( '--gamma',
        type    = float,
        default = 10,
        help    = 'penalization factor (Nitsche or conga)'
    )

    parser.add_argument( '--penal_regime',
        type    = int,
        choices = [0, 1, 2],
        default = 1,
        help    = 'penalization regime (Nitsche or conga)'
    )

    parser.add_argument( '--geo_cproj',
        action  = 'store_true',
        help    = 'whether cP is applied with the geometric (interpolation/histopolation) splines'
    )

    parser.add_argument( '--operator',
        choices = ['curl_curl', 'hodge_laplacian'],
        default = 'curl_curl',
        help    = 'second order differential operator'
    )

    parser.add_argument( '--epsilon',
        type    = float,
        default = 1,
        help    = 'inverse parameter for curl-curl term in source problem'
    )

    parser.add_argument( '--problem',
        choices = ['eigen_pbm', 'source_pbm'],
        default = 'source_pbm',
        help    = 'problem to be solved'
    )

    parser.add_argument( '--source',
        choices = ['manu_J', 'ring_J', 'df_J', 'cf_J'],
        default = 'manu_J',
        help    = 'type of source (manufactured or circular J)'
    )

    parser.add_argument( '--eta',
        type    = float,
        default = -64,
        help    = 'Constant parameter for zero-order term in source problem. Corresponds to -omega^2 for Maxwell harmonic'
    )

    # Read input arguments
    args         = parser.parse_args()
    deg          = args.degree
    nc           = args.ncells
    domain_name  = args.domain
    method       = args.method
    k            = args.k
    DG_full      = args.DG_full
    geo_cproj    = args.geo_cproj
    gamma        = args.gamma
    penal_regime = args.penal_regime
    proj_sol     = args.proj_sol
    operator     = args.operator  # only curl_curl for now
    problem      = args.problem
    source_type  = args.source
    eta          = args.eta
    epsilon      = args.epsilon
    no_plots     = args.no_plots
    hide_plots   = args.hide_plots

    do_plots = not no_plots

    ncells = [nc, nc]
    degree = [deg,deg]

    if domain_name in ['pretzel', 'pretzel_f'] and nc > 8:
        # backend_language='numba'
        backend_language='python'
    else:
        backend_language='python'
    print('[note: using '+backend_language+ ' backends in discretize functions]')

    if DG_full:
        raise NotImplementedError("DG_full spaces not implemented yet (eval error in sympde/topology/mapping.py)")

    print()
    print('--------------------------------------------------------------------------------------------------------------')
    t_overstamp = time_count()  # full run
    t_stamp = time_count()
    print('building the multipatch domain "'+domain_name+'"...' )
    domain = build_multipatch_domain(domain_name=domain_name)
    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())
    time_count(t_stamp)

    # plotting and diagnostics
    if domain_name == 'curved_L_shape':
        N_diag = 200
    else:
        N_diag = 100  # should match the grid resolution of the stored E_ref...

    # jump penalization factor:
    assert gamma >= 0

    h = 1/nc
    if penal_regime == 0:
        # constant penalization
        gamma_h = gamma
    elif penal_regime == 1:
        gamma_h = gamma/h
    elif penal_regime == 2:
        gamma_h = gamma * (deg+1)**2 /h  # DG std (see eg Buffa, Perugia and Warburton)
    else:
        raise ValueError(penal_regime)

    # node based grid (to better see the smoothness)
    etas, xx, yy = get_plotting_grid(mappings, N=N_diag)
    grid_vals_hcurl = lambda v: get_grid_vals_vector(v, etas, mappings_list, space_kind='hcurl')

    # cell-centered grid to compute approx L2 norm
    etas_cdiag, xx_cdiag, yy_cdiag, patch_logvols = get_plotting_grid(mappings, N=N_diag, centered_nodes=True, return_patch_logvols=True)
    # grid_vals_h1_cdiag = lambda v: get_grid_vals_scalar(v, etas_cdiag, mappings_list, space_kind='h1')
    grid_vals_hcurl_cdiag = lambda v: get_grid_vals_vector(v, etas_cdiag, mappings_list, space_kind='hcurl')

    # todo: add some identifiers for secondary parameters (eg gamma_h, proj_sol ...)
    fem_name = get_fem_name(method=method, DG_full=DG_full, geo_cproj=geo_cproj, k=k, domain_name=domain_name,nc=nc,deg=deg)
    plot_dir = './plots/'+source_type+'_'+fem_name+'/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    print('discretizing the domain with ncells = '+repr(ncells)+'...' )
    domain_h = discretize(domain, ncells=ncells, comm=comm)

    t_stamp = time_count()
    print('discretizing the de Rham seq with degree = '+repr(degree)+'...' )
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])
    derham_h = discretize(derham, domain_h, degree=degree, backend=PSYDAC_BACKENDS[backend_language])
    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2
    nquads = [d + 1 for d in degree]

    if DG_full:
        V_dg  = VectorFunctionSpace('V_dg', domain, kind='h1')
        Vh_dg = discretize(V_dg, domain_h, degree=degree) #, basis='M')
        print('Vh_dg.degree = ', Vh_dg.degree)
        print('V1h.degree = ', V1h.degree)

    else:
        Vh_dg = V1h

    # getting CONGA matrices -- also needed with nitsche method ????  (M1_m, and some other for post-processing)
    load_dir = get_load_dir(method='conga', domain_name=domain_name, nc=nc, deg=deg)
    M_mats, P_mats, D_mats, IK_mats = get_elementary_conga_matrices(
        domain_h, derham_h, load_dir=load_dir, backend_language=backend_language,
        discard_non_hom_matrices=(source_type=='ring_J')
    )
    [M0_m, M1_m, M2_m, M0_minv] = M_mats
    [bsp_P0_m, bsp_P1_m, bsp_P0_hom_m, bsp_P1_hom_m] = P_mats  # BSpline-based conf Projections
    [bD0_m, bD1_m] = D_mats
    [I1_m, K0, K0_inv, K1, K1_inv] = IK_mats

    gsp_P0_hom_m = K0_inv @ bsp_P0_hom_m @ K0
    gsp_P1_hom_m = K1_inv @ bsp_P1_hom_m @ K1
    gsp_P0_m = K0_inv @ bsp_P0_m @ K0
    gsp_P1_m = K1_inv @ bsp_P1_m @ K1

    if geo_cproj:
        print(' [* GSP-conga: using Geometric Spline conf Projections ]')
        cP0_hom_m = gsp_P0_hom_m
        cP0_m     = gsp_P0_m
        cP1_hom_m = gsp_P1_hom_m
        cP1_m     = gsp_P1_m
    else:
        print(' [* BSP-conga: using B-Spline conf Projections ]')
        cP0_hom_m = bsp_P0_hom_m
        cP0_m     = bsp_P0_m
        cP1_hom_m = bsp_P1_hom_m
        cP1_m     = bsp_P1_m

    # weak divergence matrices V1h -> V0h
    pw_div_m = - M0_minv @ bD0_m.transpose() @ M1_m   # patch-wise weak divergence
    bsp_D0_m = bD0_m @ bsp_P0_hom_m  # bsp-conga gradient on homogeneous space
    bsp_div_m = - M0_minv @ bsp_D0_m.transpose() @ M1_m   # gsp-conga divergence
    gsp_D0_m = bD0_m @ gsp_P0_hom_m  # gsp-conga gradient on homogeneous space
    gsp_div_m = - M0_minv @ gsp_D0_m.transpose() @ M1_m   # bsp-conga divergence

    def div_norm(u_c, type=None):
        if type is None:
            if geo_cproj:
                type = 'gsp'
            else:
                type = 'bsp'
        if type=='gsp':
            du_c = gsp_div_m.dot(u_c)
        elif type=='bsp':
            du_c = bsp_div_m.dot(u_c)
        elif type=='pw':
            du_c = pw_div_m.dot(u_c)
        else:
            print("WARNING: invalid value for weak divergence type (returning -1)")
            return -1

        return np.dot(du_c,M0_m.dot(du_c))**0.5

    def curl_norm(u_c):
        du_c = (bD1_m @ cP1_m).dot(u_c)
        return np.dot(du_c,M2_m.dot(du_c))**0.5

    # E_vals saved/loaded as point values on cdiag grid (mostly for error measure)
    f = None
    E_ex = None
    E_ref_vals = None
    save_E_vals = False
    E_vals_filename = None

    # Eh saved/loaded as numpy array of FEM coefficients (mostly for further diagnostics)
    save_Eh = False
    Eh = None

    if problem == 'source_pbm':

        print("***  Defining the source and ref solution *** ")

        # source and ref solution
        nc_ref = 32
        deg_ref = 6
        method_ref = 'conga'
        # source_type = 'ring_J'
        # source_type = 'manu_sol'

        f, E_bc, E_ref_vals, E_ex = get_source_and_solution(source_type=source_type, eta=eta, domain=domain, refsol_params=[nc_ref, deg_ref, N_diag, method_ref])
        if E_ref_vals is None:
            print('-- no ref solution found')

        # print("[[ FORCING: discard E_bc ]]")
        # E_bc = None

        # todo: discard if same as E_ref

        solutions_dir = get_load_dir(method=method, DG_full=DG_full, domain_name=domain_name,nc=nc,deg=deg,data='solutions')
        E_vals_filename = solutions_dir+E_ref_fn(source_type, N_diag)
        save_E_vals = True
        if not os.path.exists(solutions_dir):
            os.makedirs(solutions_dir)

        # disabled for now -- if with want to save the coeffs we need to store more parameters (gamma, proj_sol, etc...)
        save_Eh = False
        if save_Eh:
            Eh_filename = solutions_dir+Eh_coeffs_fn(source_type, N_diag)
            print("-- I will also save the present solution coefficients in file '"+Eh_filename+"' --")
            if not os.path.exists(solutions_dir):
                os.makedirs(solutions_dir)

        hom_bc = (E_bc is None)
    else:
        # eigenpbm is with homogeneous bc
        E_bc = None
        hom_bc = True

    assert operator == 'curl_curl'

    # ------------------------------------------------------------------------------------------
    #   curl-curl operator matrix
    # ------------------------------------------------------------------------------------------

    if method == 'conga':
        K_hom_m, K_bc_m = conga_curl_curl_2d(M1_m=M1_m, M2_m=M2_m, cP1_m=cP1_m, cP1_hom_m=cP1_hom_m, bD1_m=bD1_m, I1_m=I1_m,
                                             epsilon=epsilon, gamma_h=gamma_h, hom_bc=hom_bc)

    elif method == 'nitsche':
        load_dir = get_load_dir(method='nitsche', DG_full=DG_full, domain_name=domain_name, nc=nc, deg=deg)
        K_hom_m, M_m = nitsche_curl_curl_2d(domain_h, Vh=Vh_dg, gamma_h=gamma_h, k=k, load_dir=load_dir,
                                            need_mass_matrix=DG_full, backend_language=backend_language)
        if not DG_full:
            M_m = M1_m
        # no lifting of bc (for now):
        K_bc_m = None
    else:
        raise ValueError(method)

    if not DG_full:
        div_K = bsp_D0_m.transpose() @ K_hom_m
        print('****   [[[ spnorm(div_K) ]]] :', spnorm(div_K))

    if problem == 'eigen_pbm':

        print("***  Solving eigenvalue problem  *** ")
        if hom_bc:
            print('     (with homogeneous bc)')

        sigma, ref_sigmas = get_ref_eigenvalues(domain_name, operator)
        nb_eigs = max(10, len(ref_sigmas))

        eigenvalues, eigenvectors = get_eigenvalues(nb_eigs, sigma, K_hom_m, M1_m)

        if operator == 'curl_curl':
            # discard zero eigenvalues
            n = 0
            all_eigenvalues = eigenvalues
            eigenvalues = []
            while len(eigenvalues) < len(ref_sigmas):
                comment = '* checking computed eigenvalue #{:d}: {:15.10f}: '.format(n, all_eigenvalues[n])
                if n == len(all_eigenvalues):
                    print("Error: not enough computed eigenvalues...")
                    raise ValueError
                if abs(all_eigenvalues[n]) > 1e-6:
                    eigenvalues.append(all_eigenvalues[n])
                    print(comment+'keeping it')
                else:
                    print(comment+'discarding small eigenvalue')
                n += 1

        errors = []
        n_errs = min(len(ref_sigmas), len(eigenvalues))
        for n in range(n_errs):
            errors.append(abs(eigenvalues[n]-ref_sigmas[n]))

        print('errors from reference eigenvalues: ')
        print(errors)

    elif problem == 'source_pbm':

        print("***  Solving source problem  *** ")
        # print(" with 1/epsilon = ", repr(1/epsilon))

        # ------------------------------------------------------------------------------------------
        #   equation operator matrix
        # ------------------------------------------------------------------------------------------

        #  in homogeneous spaces // or in full space for nitsche... (todo: improve the notation and call that A_m // and A_bc_m the operator for the lifted bc if needed)
        if method == 'conga':
            # A_hom_m = (1/epsilon) * K_hom_m + eta * cP1_hom_m.transpose() @ M1_m @ cP1_hom_m
            A_hom_m = K_hom_m + eta * cP1_hom_m.transpose() @ M1_m @ cP1_hom_m
        else:
            assert method == 'nitsche'
            A_hom_m = (1/epsilon) * K_hom_m + eta * M_m

        lift_E_bc = (method == 'conga' and not hom_bc)
        if lift_E_bc:
            # equation operator for bc lifting
            assert K_bc_m is not None
            A_bc_m = (1/epsilon) * K_bc_m + eta * cP1_hom_m.transpose() @ M1_m @ cP1_m
        else:
            A_bc_m = None

        # ------------------------------------------------------------------------------------------
        #   assembling RHS
        # ------------------------------------------------------------------------------------------

        rhs_load_dir = get_load_dir(domain_name=domain_name,nc=nc,deg=deg,DG_full=DG_full,data='rhs')
        if not os.path.exists(rhs_load_dir):
            os.makedirs(rhs_load_dir)

        rhs_filename = rhs_load_dir+rhs_fn(source_type)
        if os.path.isfile(rhs_filename):
            print("getting rhs array from file "+rhs_filename)
            with open(rhs_filename, 'rb') as file:
                content = np.load(rhs_filename)
            b_c = content['b_c']
        else:
            print("-- no rhs file '"+rhs_filename+" -- so I will assemble the source")

            if source_type == 'cf_J':
                # J_h = P1-geometric projection of J
                P0, P1, P2 = derham_h.projectors(nquads=nquads)
                f_x = lambdify(domain.coordinates, f[0])
                f_y = lambdify(domain.coordinates, f[1])
                f_log = [pull_2d_hcurl([f_x, f_y], m) for m in mappings_list]
                f_h = P1(f_log)
                f_c = f_h.coeffs.toarray()
                b_c = M1_m.dot(f_c)
            else:
                # J_h = L2 projection of J
                v  = element_of(V1h.symbolic_space, name='v')
                expr = dot(f,v)
                l = LinearForm(v, integral(domain, expr))
                lh = discretize(l, domain_h, V1h, backend=PSYDAC_BACKENDS[backend_language])
                b  = lh.assemble()
                b_c = b.toarray()

            print("saving this rhs arrays (for future needs) in file "+rhs_filename)
            with open(rhs_filename, 'wb') as file:
                np.savez(file, b_c=b_c)

        # if method == 'conga':
        #     print("FILTERING RHS (FOR CONGA)")
        #     b_c = cP1_hom_m.transpose().dot(b_c)

        if method == 'nitsche' and not hom_bc:
            print("(non hom.) bc with nitsche: need some additional rhs arrays.")
            # need additional terms for the bc with nitsche
            rhs_filename = rhs_load_dir+rhs_fn(source_type, nbc=True)
            if os.path.isfile(rhs_filename):
                print("getting them from file "+rhs_filename)
                with open(rhs_filename, 'rb') as file:
                    content = np.load(rhs_filename)
                bs_c = content['bs_c']
                bp_c = content['bp_c']  # penalization term
            else:
                print("-- no rhs file '"+rhs_filename+" -- so I will assemble them...")
                nn  = NormalVector('nn')
                boundary = domain.boundary
                v  = element_of(V1h.symbolic_space, name='v')

                # expr_b = -k*cross(nn, E_bc)*curl(v) + gamma_h * cross(nn, E_bc) * cross(nn, v)

                # nitsche symmetrization term:
                expr_bs = cross(nn, E_bc)*curl(v)
                ls = LinearForm(v, integral(boundary, expr_bs))
                lsh = discretize(ls, domain_h, V1h, backend=PSYDAC_BACKENDS[backend_language])
                bs  = lsh.assemble()
                bs_c = bs.toarray()

                # nitsche penalization term:
                expr_bp = cross(nn, E_bc) * cross(nn, v)
                lp = LinearForm(v, integral(boundary, expr_bp))
                lph = discretize(lp, domain_h, V1h, backend=PSYDAC_BACKENDS[backend_language])
                bp  = lph.assemble()
                bp_c = bp.toarray()

                print("saving these rhs arrays (for future needs) in file "+rhs_filename)
                with open(rhs_filename, 'wb') as file:
                    np.savez(file, bs_c=bs_c, bp_c=bp_c)

            # full rhs for nitsche method with non-hom. bc
            b_c = b_c + (1/epsilon) * (-k * bs_c + gamma_h * bp_c)


        if lift_E_bc:
            t_stamp = time_count(t_stamp)
            print('lifting the boundary condition...')
            debug_plot = False

            # Projector on broken space
            # todo: we should probably apply P1 on E_bc -- it's a bit weird to call it on the list of (pulled back) logical fields.
            P0, P1, P2 = derham_h.projectors(nquads=nquads)
            E_bc_x = lambdify(domain.coordinates, E_bc[0])
            E_bc_y = lambdify(domain.coordinates, E_bc[1])
            E_bc_log = [pull_2d_hcurl([E_bc_x, E_bc_y], f) for f in mappings_list]
            # note: we only need the boundary dofs of E_bc (and Eh_bc)
            Eh_bc = P1(E_bc_log)
            Ebc_c = Eh_bc.coeffs.toarray()

            if debug_plot:
                Ebc_x_vals, Ebc_y_vals = grid_vals_hcurl(Eh_bc)
                my_small_plot(
                    title=r'full E for bc',
                    vals=[Ebc_x_vals, Ebc_y_vals],
                    titles=[r'Eb x', r'Eb y'],  # , r'$div_h J$' ],
                    surface_plot=False,
                    xx=xx, yy=yy,
                    save_fig=plot_dir+'full_Ebc.png',
                    hide_plot=hide_plots,
                    cmap='plasma',
                    dpi=400,
                )

            # removing internal dofs
            Ebc_c = cP1_m.dot(Ebc_c)-cP1_hom_m.dot(Ebc_c)
            b_c = b_c - A_bc_m.dot(Ebc_c)

            if debug_plot:
                Eh_bc = FemField(V1h, coeffs=array_to_stencil(Ebc_c, V1h.vector_space))
                Ebc_x_vals, Ebc_y_vals = grid_vals_hcurl(Eh_bc)
                my_small_plot(
                    title=r'E bc',
                    vals=[Ebc_x_vals, Ebc_y_vals],
                    titles=[r'Eb x', r'Eb y'],  # , r'$div_h J$' ],
                    surface_plot=False,
                    xx=xx, yy=yy,
                    save_fig=plot_dir+'Ebc.png',
                    hide_plot=hide_plots,
                    cmap='plasma',
                    dpi=400,
                )

                E_ex_x = lambdify(domain.coordinates, E_ex[0])
                E_ex_y = lambdify(domain.coordinates, E_ex[1])
                E_ex_log = [pull_2d_hcurl([E_ex_x, E_ex_y], f) for f in mappings_list]
                # note: we only need the boundary dofs of E_bc (and Eh_bc)
                Eh_ex = P1(E_ex_log)
                E_ex_c = Eh_ex.coeffs.toarray()

                E_diff_c = E_ex_c - Ebc_c
                Edh = FemField(V1h, coeffs=array_to_stencil(E_diff_c, V1h.vector_space))
                Ed_x_vals, Ed_y_vals = grid_vals_hcurl(Edh)
                my_small_plot(
                    title=r'E_exact - E_bc',
                    vals=[Ed_x_vals, Ed_y_vals],
                    titles=[r'(E_{ex}-E_{bc})_x', r'(E_{ex}-E_{bc})_y'],  # , r'$div_h J$' ],
                    surface_plot=False,
                    xx=xx, yy=yy,
                    save_fig=plot_dir+'diff_Ebc.png',
                    hide_plot=hide_plots,
                    cmap='plasma',
                    dpi=400,
                )

        print(' [[ source divergence: ')
        fh_c = spsolve(M1_m.tocsc(), b_c)
        fh_norm = np.dot(fh_c,M1_m.dot(fh_c))**0.5
        print("|| fh || = ", fh_norm)
        print("|| pw_div fh || / || fh ||  = ", div_norm(fh_c, type='pw')/fh_norm)
        print("|| bsp_div fh || / || fh || = ", div_norm(fh_c, type='bsp')/fh_norm)
        print("|| gsp_div fh || / || fh || = ", div_norm(fh_c, type='gsp')/fh_norm)
        print(' ]] ')

        print(' [[ source curl: ')
        print("|| curl fh || / || fh ||  = ", curl_norm(fh_c)/fh_norm)
        print(' ]] ')

        plot_source = True
        # plot_source = False
        if do_plots and plot_source:
            t_stamp = time_count(t_stamp)
            print('plotting the source...')
            # representation of discrete source:
            fh = FemField(V1h, coeffs=array_to_stencil(fh_c, V1h.vector_space))

            fh_x_vals, fh_y_vals = grid_vals_hcurl(fh)
            plot_full_fh=False
            if plot_full_fh:
                div_fh = FemField(V0h, coeffs=array_to_stencil(div_m.dot(fh_c), V0h.vector_space))
                div_fh_vals = grid_vals_h1(div_fh)
                my_small_plot(
                    title=r'discrete source term for Maxwell curl-curl problem',
                    vals=[np.abs(fh_x_vals), np.abs(fh_y_vals), np.abs(div_fh_vals)],
                    titles=[r'$|fh_x|$', r'$|fh_y|$', r'$|div_h fh|$'],  # , r'$div_h J$' ],
                    cmap='hsv',
                    save_fig=plot_dir+'full_Jh.png',
                    hide_plot=hide_plots,
                    surface_plot=False,
                    xx=xx, yy=yy,
                )
            else:
                abs_fh_vals = [np.sqrt(abs(fx)**2 + abs(fy)**2) for fx, fy in zip(fh_x_vals, fh_y_vals)]
                my_small_plot(
                    title=r'current source $J_h$ (amplitude)',
                    vals=[abs_fh_vals],
                    titles=[r'$|J_h|$'],  # , r'$div_h J$' ],
                    surface_plot=False,
                    xx=xx, yy=yy,
                    save_fig=plot_dir+'Jh.png',
                    hide_plot=hide_plots,
                    cmap='plasma',
                    dpi=400,
                )
            if domain_name in ['pretzel','pretzel_f']:
                vf_amp = .1
            else:
                vf_amp = .05
            my_small_streamplot(
                title=r'current source $J_h$ (vector field)',
                vals_x=fh_x_vals,
                vals_y=fh_y_vals,
                skip=10,
                xx=xx, yy=yy,
                save_fig=plot_dir+'Jh_vf.png',
                hide_plot=hide_plots,
                amplification=vf_amp,
            )

        # ------------------------------------------------------------------------------------------
        #   solving the matrix equation
        # ------------------------------------------------------------------------------------------

        t_stamp = time_count(t_stamp)
        try:
            print("trying direct solve with scipy spsolve...")   #todo: use for small problems [[ or: try catch ??]]
            Eh_c = spsolve(A_hom_m.tocsc(), b_c)
        except:
            ## for large problems:
            print("did not work -- trying with scipy lgmres...")
            A_hom_csc = A_hom_m.tocsc()
            print(" -- with approximate inverse using ILU decomposition -- ")
            A_hom_spilu = spilu(A_hom_csc)
            # A_hom_spilu = spilu(A_hom_csc, fill_factor=100, drop_tol=1e-6)  # better preconditionning, if matrix not too large
            # print('**** A: ',  A_hom_m.shape )

            preconditioner = LinearOperator(
                A_hom_m.shape, lambda x: A_hom_spilu.solve(x)
            )
            nb_iter = 0
            def f2_iter(x):
                global nb_iter
                print('lgmres -- iter = ', nb_iter, 'residual= ', norm(A_hom_m.dot(x)-b_c))
                nb_iter = nb_iter + 1
            tol = 1e-10
            Eh_c, info = lgmres(A_hom_csc, b_c, x0=None, tol=tol, atol=tol, M=preconditioner, callback=f2_iter)
                      # inner_m=30, outer_k=3, outer_v=None,
                      #                                           store_outer_Av=True)
            print(' -- convergence info:', info)

        # E_coeffs = array_to_stencil(Eh_c, V1h.vector_space)

        # print('**** cP1:',  cP1_hom_m.shape )
        # print('**** Eh:',  Eh_c.shape )
        print("... solver done.")
        time_count(t_stamp)

        if proj_sol:
            if method == 'conga':
                print("  (projecting the homogeneous Conga solution with cP1_hom_m)  ")
                Eh_c = cP1_hom_m.dot(Eh_c)
            else:
                print("  (projecting the Nitsche solution with cP1_m -- NOTE: THIS IS NONSTANDARD! )  ")
                Eh_c = cP1_m.dot(Eh_c)

        if lift_E_bc:
            print("lifting the solution with E_bc  ")
            Eh_c += Ebc_c


        # Eh = FemField(V1h, coeffs=array_to_stencil(PEh_c, V1h.vector_space))
        Eh = FemField(V1h, coeffs=array_to_stencil(Eh_c, V1h.vector_space))

        if save_Eh:
            # MCP: I think this should be discarded....
            if os.path.isfile(Eh_filename):
                print('(solution coeff array is already saved, no need to save it again)')
            else:
                print("saving solution coeffs (for future needs) in new file "+Eh_filename)
                with open(Eh_filename, 'wb') as file:
                    np.savez(file, array_coeffs=Eh_c)

        #+++++++++++++++++++++++++++++++
        # plotting and diagnostics
        #+++++++++++++++++++++++++++++++

        compute_div = True
        if compute_div:
            print(' [[ field divergence: ')
            Eh_norm = np.dot(Eh_c,M1_m.dot(Eh_c))**0.5
            print("|| Eh || = ", Eh_norm)
            print("|| pw_div Eh || / || Eh ||  = ", div_norm(Eh_c, type='pw')/Eh_norm)
            print("|| bsp_div Eh || / || Eh || = ", div_norm(Eh_c, type='bsp')/Eh_norm)
            print("|| gsp_div Eh || / || Eh || = ", div_norm(Eh_c, type='bsp')/Eh_norm)
            print(' ]] ')

            print(' [[ field curl: ')
            print("|| curl Eh || / || Eh ||  = ", curl_norm(Eh_c)/Eh_norm)
            print(' ]] ')

        if do_plots:
            # smooth plotting with node-valued grid
            Eh_x_vals, Eh_y_vals = grid_vals_hcurl(Eh)
            if domain_name in ['pretzel','pretzel_f']:
                vf_amp = 2
            else:
                vf_amp = 1
            my_small_streamplot(
                title=r'solution electric field $E_h$ (vector field)',  # for $\omega = $'+repr(omega),
                vals_x=Eh_x_vals,
                vals_y=Eh_y_vals,
                skip=10,
                xx=xx,
                yy=yy,
                amplification=vf_amp,
                save_fig=plot_dir+'Eh_vf.png',
                hide_plot=hide_plots,
                dpi = 200,
            )

            Eh_abs_vals = [np.sqrt(abs(ex)**2 + abs(ey)**2) for ex, ey in zip(Eh_x_vals, Eh_y_vals)]
            my_small_plot(
                title=r'solution electric field $E_h$ (amplitude)', # for $\omega = $'+repr(omega),
                vals=[Eh_abs_vals], #[Eh_x_vals, Eh_y_vals, Eh_abs_vals],
                titles=[r'$|E^h|$'], #[r'$E^h_x$', r'$E^h_y$', r'$|E^h|$'],
                xx=xx,
                yy=yy,
                surface_plot=False,
                # gridlines_x1=gridlines_x1,
                # gridlines_x2=gridlines_x2,
                save_fig=plot_dir+'Eh.png',
                hide_plot=hide_plots,
                cmap='hsv',
                dpi = 400,
            )

        # error measure with centered-valued grid
        Eh_x_vals, Eh_y_vals = grid_vals_hcurl_cdiag(Eh)

        if save_E_vals:
            print("saving solution values (on cdiag grid) in new file (for future needs)"+E_vals_filename)
            with open(E_vals_filename, 'wb') as file:
                np.savez(file, x_vals=Eh_x_vals, y_vals=Eh_y_vals)

        xx = xx_cdiag
        yy = yy_cdiag

        if E_ref_vals is None:
            E_x_vals = np.zeros_like(Eh_x_vals)
            E_y_vals = np.zeros_like(Eh_y_vals)
        else:
            E_x_vals, E_y_vals = E_ref_vals

        only_last_patch = False
        quad_weights = get_grid_quad_weights(etas_cdiag, patch_logvols, mappings_list)
        Eh_errors_cdiag = [np.sqrt( (u1-v1)**2 + (u2-v2)**2 )
                           for u1, v1, u2, v2 in zip(Eh_x_vals, E_x_vals, Eh_y_vals, E_y_vals)]
        if only_last_patch:
            print('WARNING ** WARNING : measuring error on last patch only !!' )
            warning_msg = ' [on last patch]'
            l2_error = (np.sum([J_F * err**2 for err, J_F in zip(Eh_errors_cdiag[-1:], quad_weights[-1:])]))**0.5
        else:
            warning_msg = ''
            l2_error = (np.sum([J_F * err**2 for err, J_F in zip(Eh_errors_cdiag, quad_weights)]))**0.5

        err_message = 'diag_grid error'+warning_msg+' for method = {0} with nc = {1}, deg = {2}, gamma = {3}, gamma_h = {4} and proj_sol = {5}: {6}\n'.format(
                    get_method_name(method, k, geo_cproj, penal_regime), nc, deg, gamma, gamma_h, proj_sol, l2_error
        )
        print('\n** '+err_message)

        check_err = True
        if E_ex is not None:
            # also assembling the L2 error with Psydac quadrature
            print(" -- * --  also computing L2 error with explicit (exact) solution, using Psydac quadratures...")
            F  = element_of(V1h.symbolic_space, name='F')
            error       = Matrix([F[0]-E_ex[0],F[1]-E_ex[1]])
            l2_norm     = Norm(error, domain, kind='l2')
            l2_norm_h   = discretize(l2_norm, domain_h, V1h, backend=PSYDAC_BACKENDS[backend_language])
            l2_error     = l2_norm_h.assemble(F=Eh)
            err_message_2 = 'l2_psydac error for method = {0} with nc = {1}, deg = {2}, gamma = {3}, gamma_h = {4} and proj_sol = {5} [*] : {6}\n'.format(
                    get_method_name(method, k, geo_cproj, penal_regime), nc, deg, gamma, gamma_h, proj_sol, l2_error
            )
            print('\n** '+err_message_2)
            if check_err:
                # since Ex is available, compute also the auxiliary error || Eh - P1 E || with M1 mass matrix
                P0, P1, P2 = derham_h.projectors(nquads=nquads)
                E_x = lambdify(domain.coordinates, E_ex[0])
                E_y = lambdify(domain.coordinates, E_ex[1])
                E_log = [pull_2d_hcurl([E_x, E_y], f) for f in mappings_list]
                Ex_h = P1(E_log)
                Ex_c = Ex_h.coeffs.toarray()
                err_c = Ex_c-Eh_c
                err_norm = np.dot(err_c,M1_m.dot(err_c))**0.5
                print('--- ** --- check: L2 discrete-error (in V1h): {}'.format(err_norm))

        else:
            err_message_2 = ''

        error_filename = error_fn(source_type=source_type, method=method, k=k, domain_name=domain_name,deg=deg)
        if not os.path.exists(error_filename):
            open(error_filename, 'w')
        with open(error_filename, 'a') as a_writer:
            a_writer.write(err_message)
            if err_message_2:
                a_writer.write(err_message_2)

        if do_plots:
            E_x_err = [(u1 - u2) for u1, u2 in zip(E_x_vals, Eh_x_vals)]
            E_y_err = [(u1 - u2) for u1, u2 in zip(E_y_vals, Eh_y_vals)]
            my_small_plot(
                title=r'approximation of solution $u$, $x$ component',
                vals=[E_x_vals, Eh_x_vals, E_x_err],
                titles=[r'$u^{ex}_x(x,y)$', r'$u^h_x(x,y)$', r'$|(u^{ex}-u^h)_x(x,y)|$'],
                xx=xx,
                yy=yy,
                save_fig=plot_dir+'err_Ex.png',
                hide_plot=hide_plots,
                # gridlines_x1=gridlines_x1,
                # gridlines_x2=gridlines_x2,
            )

            my_small_plot(
                title=r'approximation of solution $u$, $y$ component',
                vals=[E_y_vals, Eh_y_vals, E_y_err],
                titles=[r'$u^{ex}_y(x,y)$', r'$u^h_y(x,y)$', r'$|(u^{ex}-u^h)_y(x,y)|$'],
                xx=xx,
                yy=yy,
                save_fig=plot_dir+'err_Ey.png',
                hide_plot=hide_plots,
                # gridlines_x1=gridlines_x1,
                # gridlines_x2=gridlines_x2,
            )

    else:
        raise NotImplementedError

    print(" -- OK run done -- ")
    time_count(t_overstamp, msg='full run')
    print()
    exit()







