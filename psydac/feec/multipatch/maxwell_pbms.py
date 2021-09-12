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
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain, get_ref_eigenvalues

comm = MPI.COMM_WORLD

# ---------------------------------------------------------------------------------------------------------------
# small utility for saving/loading sparse matrices, plots...
def E_ref_fn(source_type, N_diag):
    return 'E_ref_'+source_type+'_N'+repr(N_diag)+'.npz'

def get_fem_name(method=None, k=None, domain_name=None,nc=None,deg=None):
    assert domain_name and nc and deg
    assert method is not None
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
    else:
        method_name = method
        assert method == 'conga'
    return domain_name+'_nc'+repr(nc)+'_deg'+repr(deg)+'_'+method_name

def get_load_dir(method=None, domain_name=None,nc=None,deg=None,data='matrices'):
    assert data in ['matrices','solutions']
    fem_name = get_fem_name(method=method, domain_name=domain_name,nc=nc,deg=deg)
    return './saved_'+data+'/'+fem_name+'/'


# ---------------------------------------------------------------------------------------------------------------
def nitsche_operators_2d(domain, ncells, degree, operator='curl_curl', gamma_h=None, k=None):
    """
    computes
        A_m the k-IP matrix of the curl-curl operator with penalization parameter gamma
        (as defined eg in Buffa, Houston & Perugia, JCAM 2007)
        and M_m the mass matrix

    :param k: parameter for SIP/NIP/IIP
    :return: matrices in sparse format
    """
    from psydac.api.discretization import discretize
    assert gamma_h is not None

    if load_dir:
        print(" -- will load matrices from " + load_dir)
    elif save_dir:
        print(" -- will save matrices in " + save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    if load_dir:
        print("loading sparse matrices...")
        # unpenalized curl-curl matrix (main part and symmetrization term)
        CC_m = load_npz(load_dir+'CC_m.npz')
        CS_m = load_npz(load_dir+'CS_m.npz')
        # jump penalization matrix
        JP_m = load_npz(load_dir+'JP_m.npz')
        # mass matrix
        M_m = load_npz(load_dir+'M_m.npz')

    else:
        t_stamp = time_count()
        print('computing IP curl-curl matrix with penalization gamma_h = {}'.format(gamma_h))
        #+++++++++++++++++++++++++++++++
        # 1. Abstract model
        #+++++++++++++++++++++++++++++++

        V  = VectorFunctionSpace('V', domain, kind='hcurl')

        u, v, F  = elements_of(V, names='u, v, F')
        nn  = NormalVector('nn')

        I        = domain.interfaces
        boundary = domain.boundary

        jump = lambda w:plus(w)-minus(w)
        avr_curl = lambda w:(curl(plus(w)) + curl(minus(w)))/2

        #    # Bilinear form a: V x V --> R
        assert operator == 'curl_curl'  # todo: write hodge-laplacian case

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


        # a = BilinearForm((u,v),  integral(domain, expr) + integral(I, expr_I) + integral(boundary, expr_b))

        a_cc = BilinearForm((u,v),  integral(domain, expr) + integral(I, expr_I) + integral(boundary, expr_b))
        a_cs = BilinearForm((u,v),  integral(I, expr_Is) + integral(boundary, expr_bs))  # symmetrization terms
        a_jp = BilinearForm((u,v),  integral(I, expr_jp_I) + integral(boundary, expr_jp_b))

        expr_m = dot(u, v)
        m = BilinearForm((u,v), integral(domain, expr_m))

        #+++++++++++++++++++++++++++++++
        # 2. Discretization
        #+++++++++++++++++++++++++++++++

        domain_h = discretize(domain, ncells=ncells, comm=comm)
        Vh       = discretize(V, domain_h, degree=degree,basis='M')

        # unpenalized curl-curl matrix
        a_h = discretize(a_cc, domain_h, [Vh, Vh])
        A = a_h.assemble()
        CC_m  = A.tosparse().tocsr()

        # symmetrization part
        a_h = discretize(a_cs, domain_h, [Vh, Vh])
        A = a_h.assemble()
        CS_m  = A.tosparse().tocsr()

        # jump penalization matrix
        a_h = discretize(a_jp, domain_h, [Vh, Vh])
        A = a_h.assemble()
        JP_m  = A.tosparse().tocsr()

        # mass matrix
        m_h = discretize(m, domain_h, [Vh, Vh])
        M = m_h.assemble()
        M_m  = M.tosparse().tocsr()

        if save_dir:
            t_stamp = time_count(t_stamp)
            print("saving sparse matrices to file...")
            save_npz(save_dir+'CC_m.npz', CC_m)
            save_npz(save_dir+'CS_m.npz', CS_m)
            save_npz(save_dir+'JP_m.npz', JP_m)
            save_npz(save_dir+'M_m.npz', M_m)

    if operator == 'curl_curl':
        A_m = CC_m + k*CS_m + gamma_h*JP_m
    else:
        raise NotImplementedError
    return A_m, M_m


# ---------------------------------------------------------------------------------------------------------------
def get_elementary_conga_matrices(domain, ncells, degree):

    if load_dir:
        print(" -- will load matrices from " + load_dir)
    elif save_dir:
        print(" -- will save matrices in " + save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    t_stamp = time_count()

    print('discretizing the domain with ncells = '+repr(ncells)+'...' )
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

        mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
        mappings_list = list(mappings.values())
        x,y    = domain.coordinates
        nquads = [d + 1 for d in degree]

        # plotting
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

    return M0_m, M1_m, M2_m, M0_minv, cP0_m, cP1_m, D0_m, D1_m, I1_m


def conga_operators_2d(domain, ncells, degree, operator='curl_curl', gamma_h=None):
    """
    computes:
        A_m the matrix of the CONGA A operator, with penalization parameter gamma
        (as defined eg in Campos Pinto and Güçlü, preprint 2021)
        with:
            A = curl curl
        or
            A = curl curl + grad div
        as specified by operator
        and M_m the mass matrix

    :return: matrices in sparse format
    """
    ## building Hodge Laplacian matrix

    M0_m, M1_m, M2_m, M0_minv, cP0_m, cP1_m, D0_m, D1_m, I1_m = get_elementary_conga_matrices(domain, ncells, degree)

    # t_stamp = time_count()

    jump_penal_m = I1_m-cP1_m
    A_m = (
            D1_m.transpose() * M2_m * D1_m
            + gamma_h * jump_penal_m.transpose() * M1_m * jump_penal_m
    )

    if operator == 'curl_curl':
        print('computing Conga curl-curl matrix with penalization gamma_h = {}'.format(gamma_h))

    elif operator == 'hodge_laplacian':
        print("computing Conga Hodge-Laplacian matrix...")
        div_aux_m = D0_m.transpose() * M1_m  # note: the matrix of the (weak) div operator is:   - M0_minv * div_aux_m

        L_option = 2
        if L_option == 1:
            A_m += div_aux_m.transpose() * M0_minv * div_aux_m
        else:
            A_m += (div_aux_m * cP1_m).transpose() * M0_minv * div_aux_m * cP1_m
    else:
        raise NotImplementedError

    return A_m, M1_m


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


def get_source_and_solution(source_type, gamma, domain, refsol_params=None):

    assert refsol_params
    nc_ref, deg_ref, N_diag, method_ref = refsol_params

    E_ref_vals = None   # ref solution
    x,y    = domain.coordinates

    if source_type == 'manu_sol':
        # use a manufactured solution, with ad-hoc (inhomogeneous) bc

        E_ex    = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
        f      = Tuple(gamma*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                         gamma*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))
        E_ex_x = lambdify(domain.coordinates, E_ex[0])
        E_ex_y = lambdify(domain.coordinates, E_ex[1])
        E_ex_log = [pull_2d_hcurl([E_ex_x,E_ex_y], f) for f in mappings_list]
        E_ref_x_vals, E_ref_y_vals   = grid_vals_hcurl_cdiag(E_ex_log)
        E_ref_vals = [E_ref_x_vals, E_ref_y_vals]

    elif source_type == 'ring_J':

        # 'rotating' (divergence-free) J field:
        #   J = j(r) * (-sin theta, cos theta)

        if domain_name in ['square_2', 'square_6', 'square_8']:
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

        # vis_J = False
        # if vis_J:
        #     tmp_plot_source(J_x,J_y, domain)
        E_ref_filename = get_load_dir(method=method_ref, domain_name=domain_name,nc=nc_ref,deg=deg_ref,data='solutions')+E_ref_fn(source_type, N_diag)
        # E_ex = None  # exact solution void by default
        # E_ref_x_vals = None
        # E_ref_y_vals = None
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
        raise NotImplementedError

    return f, E_ref_vals

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        description     = "Solve 2D curl-curl eigenvalue problem."
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
        choices = ['square', 'annulus', 'curved_L_shape', 'pretzel', 'pretzel_annulus', 'pretzel_debug'],
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

    parser.add_argument( '--gamma',
        type    = float,
        default = -1,
        help    = 'penalization term (Nitsche or conga)'
    )

    parser.add_argument( '--operator',
        choices = ['curl_curl', 'hodge_laplacian'],
        default = 'curl_curl',
        help    = 'second order differential operator'
    )

    parser.add_argument( '--problem',
        choices = ['eigen_pbm', 'source_pbm'],
        default = 'eigen_pbm',
        help    = 'problem to be solved'
    )

    parser.add_argument( '--eta',
        type    = int,
        default = -1,
        help    = 'Constant parameter for zero-order term in source problem. Corresponds to -omega^2 for Maxwell harmonic'
    )

    # Read input arguments
    args        = parser.parse_args()
    deg         = args.degree
    nc          = args.ncells
    domain_name = args.domain
    method      = args.method
    gamma       = args.gamma
    k           = args.k
    operator    = args.operator
    problem     = args.problem
    eta         = args.eta

    ncells = [nc, nc]
    degree = [deg,deg]

    fem_name = get_fem_name(method=method, k=k, domain_name=domain_name,nc=nc,deg=deg) #domain_name+np_suffix+'_nc'+repr(nc)+'_deg'+repr(deg)
    save_dir = load_dir = get_load_dir(method=method, domain_name=domain_name,nc=nc,deg=deg)  # './tmp_matrices/'+fem_name+'/'
    if load_dir and not os.path.exists(load_dir):
        print(' -- note: discarding absent load directory')
        load_dir = None

    print('--------------------------------------------------------------------------------------------------------------')
    t_stamp = time_count()
    print('building the multipatch domain...' )
    domain = build_multipatch_domain(domain_name=domain_name)
    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())

    print('discretizing the domain with ncells = '+repr(ncells)+'...' )
    domain_h = discretize(domain, ncells=ncells, comm=comm)

    t_stamp = time_count(t_stamp)
    print('discretizing the de Rham seq with degree = '+repr(degree)+'...' )
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])
    derham_h = discretize(derham, domain_h, degree=degree) #, backend=PSYDAC_BACKENDS['numba'])
    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2
    # todo: avoid building these spaces again

    # diag grid (cell-centered)
    N_diag = 100
    etas_cdiag, xx_cdiag, yy_cdiag, patch_logvols = get_plotting_grid(mappings, N=N_diag, centered_nodes=True, return_patch_logvols=True)
    # grid_vals_h1_cdiag = lambda v: get_grid_vals_scalar(v, etas_cdiag, mappings_list, space_kind='h1')
    grid_vals_hcurl_cdiag = lambda v: get_grid_vals_vector(v, etas_cdiag, mappings_list, space_kind='hcurl')

    # jump penalization factor:
    # todo: study different penalization regimes
    if gamma > 0:
        h = 1/nc
        # gamma_h = 10*(deg+1)**2/h  # DG std (see eg Buffa, Perugia and Warburton)
        gamma_h = gamma/h
    else:
        gamma_h = 10**3

    # build operator matrices
    if method == 'conga':
        A_m, M_m = conga_operators_2d(domain, ncells=ncells, degree=degree, operator=operator, gamma_h=gamma_h)
    else:
        assert method == 'nitsche'
        A_m, M_m = nitsche_operators_2d(domain, ncells=ncells, degree=degree, operator=operator, gamma_h=gamma_h, k=k)

    if problem == 'eigen_pbm':

        print("***  Solving eigenvalue problem  *** ")

        sigma, ref_sigmas = get_ref_eigenvalues(domain_name, operator)
        nb_eigs = max(10, len(ref_sigmas))

        eigenvalues, eigenvectors = get_eigenvalues(nb_eigs, sigma, A_m, M_m)

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

        # source and ref solution
        nc_ref = 20
        deg_ref = 8
        method_ref = 'conga'
        source_type = 'ring_J' #'manu_sol'
        f, E_ref_vals = get_source_and_solution(source_type=source_type, gamma=gamma, domain=domain, refsol_params=[nc_ref, deg_ref, N_diag, method_ref])

        if E_ref_vals is None:
            solutions_dir = get_load_dir(method=method, domain_name=domain_name,nc=nc,deg=deg,data='solutions')
            E_ref_filename = solutions_dir+E_ref_fn(source_type, N_diag)
            print("-- no ref solution, so I will save the present solution instead, in file '"+E_ref_filename+"' --")
            if not os.path.exists(solutions_dir):
                os.makedirs(solutions_dir)

        # discrete spaces
        u, v, F  = elements_of(V1h.symbolic_space, names='u, v, F')

        # regular (node based) plotting
        etas, xx, yy = get_plotting_grid(mappings, N=N_diag)
        grid_vals_hcurl = lambda v: get_grid_vals_vector(v, etas, mappings_list, space_kind='hcurl')

        # assembling RHS
        hom_bc = True

        expr   = dot(f,v)
        if hom_bc:
            l = LinearForm(v, integral(domain, expr))
        else:
            # todo
            raise NotImplementedError

        lh = discretize(l, domain_h, V1h) #, backend=PSYDAC_BACKENDS['numba'])
        b  = lh.assemble()
        b_c = b.toarray()

    plot_source = True
    if plot_source:
        # representation of discrete source:
        fh_c = spsolve(M_m.tocsc(), b_c)
        # fh_norm = np.dot(fh_c,M_m.dot(fh_c))**0.5
        # print("|| fh || = ", fh_norm)
        # print("|| div fh ||/|| fh || = ", div_norm(fh_c)/fh_norm)

        # if fem_name:
        #     fig_name=plot_dir+'Jh.png'  # +'_'+fem_name+'.png'
        #     fig_name_vf=plot_dir+'Jh_vf.png'   # +'_vf_'+fem_name+'.png'
        # else:
        fig_name=None
        fig_name_vf=None

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
                surface_plot=False,
                xx=xx, yy=yy,
            )
        else:
            abs_fh_vals = [np.sqrt(abs(fx)**2 + abs(fy)**2) for fx, fy in zip(fh_x_vals, fh_y_vals)]
            my_small_plot(
                title=r'source term $J_h$',
                vals=[abs_fh_vals],
                titles=[r'$|J_h|$'],  # , r'$div_h J$' ],
                surface_plot=False,
                xx=xx, yy=yy,
                cmap='plasma',
                dpi=400,
                save_fig=fig_name,
            )

        my_small_streamplot(
            title='source J',
            vals_x=fh_x_vals,
            vals_y=fh_y_vals,
            xx=xx, yy=yy,
            amplification=.05,
            save_fig=fig_name_vf,
        )






        t_stamp = time_count(t_stamp)
        print("solving with scipy...")
        Eh_c = spsolve(A_m.tocsc(), b_c)
        # E_coeffs = array_to_stencil(Eh_c, V1h.vector_space)

        # projected solution
        # PEh_c = cP1_m.dot(Eh_c)
        # jumpEh_c = Eh_c - PEh_c
        # Eh = FemField(V1h, coeffs=array_to_stencil(PEh_c, V1h.vector_space))
        Eh = FemField(V1h, coeffs=array_to_stencil(Eh_c, V1h.vector_space))
        Eh_x_vals, Eh_y_vals = grid_vals_hcurl_cdiag(Eh)

        xx = xx_cdiag
        yy = yy_cdiag

        # if fem_name:
        #     fig_name=plot_dir+'Eh.png'  # +'_'+fem_name+'.png'
        #     fig_name_vf=plot_dir+'Eh_vf.png'   # +'_vf_'+fem_name+'.png'
        # else:
        fig_name=None
        fig_name_vf=None

        if E_ref_vals:
            E_x_vals, E_y_vals = E_ref_vals
        else:
            E_x_vals = Eh_x_vals
            E_y_vals = Eh_y_vals

        E_x_err = [(u1 - u2) for u1, u2 in zip(E_x_vals, Eh_x_vals)]
        E_y_err = [(u1 - u2) for u1, u2 in zip(E_y_vals, Eh_y_vals)]
        # print(E_x_vals)
        # print(Eh_x_vals)
        # print(E_x_err)
        my_small_plot(
            title=r'approximation of solution $u$, $x$ component',
            vals=[E_x_vals, Eh_x_vals, E_x_err],
            titles=[r'$u^{ex}_x(x,y)$', r'$u^h_x(x,y)$', r'$|(u^{ex}-u^h)_x(x,y)|$'],
            xx=xx,
            yy=yy,
            # gridlines_x1=gridlines_x1,
            # gridlines_x2=gridlines_x2,
        )

        my_small_plot(
            title=r'approximation of solution $u$, $y$ component',
            vals=[E_y_vals, Eh_y_vals, E_y_err],
            titles=[r'$u^{ex}_y(x,y)$', r'$u^h_y(x,y)$', r'$|(u^{ex}-u^h)_y(x,y)|$'],
            xx=xx,
            yy=yy,
            # gridlines_x1=gridlines_x1,
            # gridlines_x2=gridlines_x2,
        )

        my_small_streamplot(
            title=r'discrete field $E_h$',  # for $\omega = $'+repr(omega),
            vals_x=Eh_x_vals,
            vals_y=Eh_y_vals,
            skip=1,
            xx=xx,
            yy=yy,
            amplification=1,
            save_fig=fig_name_vf,
            dpi = 200,
        )








    else:
        raise NotImplementedError

    print("OK OK")
    exit()

    show_all = False
    plot_all = True
    dpi = 400
    dpi_vf = 200
    # show_all = True
    # plot_all = False
    plot_dir_suffix = ''
    plot_dir = './plots/'+fem_name+plot_dir_suffix+'/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if plot_all:
        show_all=True
        # will also use above value of fem_name
    else:
        # reset fem_name to disable plots
        fem_name = ''








