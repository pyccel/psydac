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
    return 'errors/error_'+domain_name+'_'+'_deg'+repr(deg)+get_method_name(method, k)+'_'+source_type+'.txt'

def get_method_name(method=None, k=None):
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
    return method_name

def get_fem_name(method=None, k=None, domain_name=None,nc=None,deg=None):
    assert domain_name and nc and deg
    fn = domain_name+'_nc'+repr(nc)+'_deg'+repr(deg)
    if method is not None:
        fn += '_'+get_method_name(method, k)
    return fn

def get_load_dir(method=None, domain_name=None,nc=None,deg=None,data='matrices'):
    assert data in ['matrices','solutions','rhs']
    if method is None:
        assert data == 'rhs'
    fem_name = get_fem_name(domain_name=domain_name,method=method, nc=nc,deg=deg)
    return './saved_'+data+'/'+fem_name+'/'


# ---------------------------------------------------------------------------------------------------------------
def nitsche_curl_curl_2d(domain, V, ncells, degree, gamma_h=None, k=None, load_dir=None):
    """
    computes
        K_m the k-IP matrix of the curl-curl operator with penalization parameter gamma
        (as defined eg in Buffa, Houston & Perugia, JCAM 2007)

    :param k: parameter for SIP/NIP/IIP
    :return: matrices in sparse format
    """
    from psydac.api.discretization import discretize
    assert gamma_h is not None

    if os.path.exists(load_dir):
        print(" -- load directory " + load_dir + " found -- will load the Nitsche matrices from there...")

        # unpenalized curl-curl matrix (main part and symmetrization term)
        CC_m = load_npz(load_dir+'CC_m.npz')
        CS_m = load_npz(load_dir+'CS_m.npz')
        # jump penalization matrix
        JP_m = load_npz(load_dir+'JP_m.npz')

    else:
        print(" -- load directory " + load_dir + " not found -- will assemble the Nitsche matrices...")

        t_stamp = time_count()
        print('computing IP curl-curl matrix with k = {0} and penalization gamma_h = {1}'.format(k, gamma_h))
        #+++++++++++++++++++++++++++++++
        # 1. Abstract model
        #+++++++++++++++++++++++++++++++

        # V  = VectorFunctionSpace('V', domain, kind='hcurl')

        u, v, F  = elements_of(V, names='u, v, F')
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

        domain_h = discretize(domain, ncells=ncells, comm=comm)
        Vh       = discretize(V, domain_h, degree=degree,basis='M')

        # unpenalized curl-curl matrix (incomplete)
        a_h = discretize(a_cc, domain_h, [Vh, Vh])
        A = a_h.assemble()
        CC_m  = A.tosparse().tocsr()

        # symmetrization part (for SIP or NIP curl-curl matrix)
        a_h = discretize(a_cs, domain_h, [Vh, Vh])
        A = a_h.assemble()
        CS_m  = A.tosparse().tocsr()

        # jump penalization matrix
        a_h = discretize(a_jp, domain_h, [Vh, Vh])
        A = a_h.assemble()
        JP_m  = A.tosparse().tocsr()

        print(" -- now saving these matrices in " + load_dir)
        os.makedirs(load_dir)
        t_stamp = time_count(t_stamp)
        print("saving sparse matrices to file...")
        save_npz(load_dir+'CC_m.npz', CC_m)
        save_npz(load_dir+'CS_m.npz', CS_m)
        save_npz(load_dir+'JP_m.npz', JP_m)

    K_m = CC_m + k*CS_m + gamma_h*JP_m

    return K_m


# ---------------------------------------------------------------------------------------------------------------
def get_elementary_conga_matrices(domain_h, derham_h, load_dir=None):

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
        M0 = BrokenMass(V0h, domain_h, is_scalar=True)
        M1 = BrokenMass(V1h, domain_h, is_scalar=False)
        M2 = BrokenMass(V2h, domain_h, is_scalar=True)

        t_stamp = time_count(t_stamp)
        print("assembling broken derivative operators...")
        bD0, bD1 = derham_h.broken_derivatives_as_operators

        t_stamp = time_count(t_stamp)
        print("assembling conf projection operators...")
        # todo: disable the non-hom-bc operators for hom-bc pretzel test cases...
        cP0 = ConformingProjection_V0(V0h, domain_h, hom_bc=False)
        cP1 = ConformingProjection_V1(V1h, domain_h, hom_bc=False)
        cP0_hom = ConformingProjection_V0(V0h, domain_h, hom_bc=True)
        cP1_hom = ConformingProjection_V1(V1h, domain_h, hom_bc=True)

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
        time_count(t_stamp)

        M0_minv = inv(M0_m.tocsc())  # todo: assemble patch-wise M0_inv, as Hodge operator

        print(" -- now saving these matrices in " + load_dir)
        os.makedirs(load_dir)

        t_stamp = time_count()
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

    return M0_m, M1_m, M2_m, M0_minv, cP0_m, cP1_m, cP0_hom_m, cP1_hom_m, bD0_m, bD1_m, I1_m

def conga_curl_curl_2d(M1_m=None, M2_m=None, cP1_m=None, cP1_hom_m=None, bD1_m=None, I1_m=None, gamma_h=None, hom_bc=True):
    """
    computes
        K_hom_m (and K_m if not hom_bc)
        the CONGA stiffness matrix of the vector-valued curl-curl operator in V1, with (and without) homogeneous bc
    """

    if not hom_bc:
        assert cP1_m is not None
    print('computing Conga curl_curl matrix with penalization gamma_h = {}'.format(gamma_h))

    assert operator == 'curl_curl'  # todo: implement (verify) the hodge-laplacian

    # curl_curl matrix (left-multiplied by M1_m) :
    D1_hom_m = bD1_m * cP1_hom_m
    jump_penal_hom_m = I1_m-cP1_hom_m
    K_hom_m = (
            D1_hom_m.transpose() * M2_m * D1_hom_m
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

    x,y    = domain.coordinates

    if source_type == 'manu_J':
        # use a manufactured solution, with ad-hoc (inhomogeneous) bc

        E_ex    = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
        f      = Tuple(eta*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                         eta*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))
        E_ex_x = lambdify(domain.coordinates, E_ex[0])
        E_ex_y = lambdify(domain.coordinates, E_ex[1])
        E_ex_log = [pull_2d_hcurl([E_ex_x,E_ex_y], f) for f in mappings_list]
        E_ref_x_vals, E_ref_y_vals   = grid_vals_hcurl_cdiag(E_ex_log)
        E_ref_vals = [E_ref_x_vals, E_ref_y_vals]
        # print(E_ex_x)

        # boundary condition: (here we only need to coincide with E_ex on the boundary !)
        E_bc = E_ex

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

    return f, E_bc, E_ref_vals

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
        choices = ['square', 'annulus', 'curved_L_shape', 'pretzel', 'pretzel_f', 'pretzel_annulus', 'pretzel_debug'],
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

    parser.add_argument( '--proj_sol',
        action  = 'store_true',
        help    = 'whether cP1 is applied to solution of source problem'
    )

    parser.add_argument( '--hide_plots',
        action  = 'store_true',
        help    = 'whether plots are hidden (and saved)'
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
        default = 'source_pbm',
        help    = 'problem to be solved'
    )

    parser.add_argument( '--source',
        choices = ['manu_J', 'ring_J'],
        default = 'manu_J',
        help    = 'type of source (manufactured or circular J)'
    )

    parser.add_argument( '--eta',
        type    = int,
        default = -64,
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
    proj_sol    = args.proj_sol
    operator    = args.operator
    problem     = args.problem
    source_type = args.source
    eta         = args.eta
    hide_plots  = args.hide_plots

    ncells = [nc, nc]
    degree = [deg,deg]

    fem_name = get_fem_name(method=method, k=k, domain_name=domain_name,nc=nc,deg=deg) #domain_name+np_suffix+'_nc'+repr(nc)+'_deg'+repr(deg)

    print('--------------------------------------------------------------------------------------------------------------')
    t_stamp = time_count()
    print('building the multipatch domain...' )
    domain = build_multipatch_domain(domain_name=domain_name)
    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())

    # plotting and diagnostics
    N_diag = 100  # should match the grid resolution of the stored E_ref...

    # node based grid (to better see the smoothness)
    etas, xx, yy = get_plotting_grid(mappings, N=N_diag)
    grid_vals_hcurl = lambda v: get_grid_vals_vector(v, etas, mappings_list, space_kind='hcurl')

    # cell-centered grid to compute approx L2 norm
    etas_cdiag, xx_cdiag, yy_cdiag, patch_logvols = get_plotting_grid(mappings, N=N_diag, centered_nodes=True, return_patch_logvols=True)
    # grid_vals_h1_cdiag = lambda v: get_grid_vals_scalar(v, etas_cdiag, mappings_list, space_kind='h1')
    grid_vals_hcurl_cdiag = lambda v: get_grid_vals_vector(v, etas_cdiag, mappings_list, space_kind='hcurl')

    plot_dir = './plots/'+fem_name+'/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    print('discretizing the domain with ncells = '+repr(ncells)+'...' )
    domain_h = discretize(domain, ncells=ncells, comm=comm)

    t_stamp = time_count(t_stamp)
    print('discretizing the de Rham seq with degree = '+repr(degree)+'...' )
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])
    derham_h = discretize(derham, domain_h, degree=degree) #, backend=PSYDAC_BACKENDS['numba'])
    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2

    # getting CONGA matrices -- also needed with nitsche method (M1_m, and some other for post-processing)
    load_dir = get_load_dir(method='conga', domain_name=domain_name, nc=nc, deg=deg)
    # if load_dir and not os.path.exists(load_dir):
    #     print(' -- note: discarding absent load directory')
    #     load_dir = None
    M0_m, M1_m, M2_m, M0_minv, cP0_m, cP1_m, cP0_hom_m, cP1_hom_m, bD0_m, bD1_m, I1_m = get_elementary_conga_matrices(
        domain_h, derham_h, load_dir=load_dir
    )

    # jump penalization factor:
    # todo: study different penalization regimes
    if gamma > 0:
        h = 1/nc
        # gamma_h = 10*(deg+1)**2/h  # DG std (see eg Buffa, Perugia and Warburton)
        gamma_h = gamma/h
    else:
        gamma_h = 10**3

    # E_ref saved/loaded as point values on cdiag grid (mostly for error measure)
    save_E_ref = False
    E_ref_filename = None

    # Eh saved/loaded as numpy array of FEM coefficients (mostly for further diagnostics)
    save_Eh = False
    Eh = None

    if problem == 'source_pbm':

        print("***  Defining the source and ref solution *** ")

        # source and ref solution
        nc_ref = 30
        deg_ref = 8
        method_ref = 'conga'
        # source_type = 'ring_J'
        # source_type = 'manu_sol'

        f, E_bc, E_ref_vals = get_source_and_solution(source_type=source_type, eta=eta, domain=domain, refsol_params=[nc_ref, deg_ref, N_diag, method_ref])

        solutions_dir = get_load_dir(method=method, domain_name=domain_name,nc=nc,deg=deg,data='solutions')
        if E_ref_vals is None:
            E_ref_filename = solutions_dir+E_ref_fn(source_type, N_diag)
            print("-- no ref solution, so I will save the present solution instead, in file '"+E_ref_filename+"' --")
            # ok but why saving this only if no ref solution available ?
            save_E_ref = True
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

    # build operator matrices
    if method == 'conga':
        K_hom_m, K_bc_m = conga_curl_curl_2d(M1_m=M1_m, M2_m=M2_m, cP1_m=cP1_m, cP1_hom_m=cP1_hom_m, bD1_m=bD1_m, I1_m=I1_m, gamma_h=gamma_h, hom_bc=hom_bc)
    elif method == 'nitsche':
        load_dir = get_load_dir(method='nitsche', domain_name=domain_name, nc=nc, deg=deg)
        K_hom_m = nitsche_curl_curl_2d(domain, V=derham.V1, ncells=ncells, degree=degree, gamma_h=gamma_h, k=k, load_dir=load_dir)
        # no lifting of bc (for now):
        K_bc_m = None
    else:
        raise ValueError(method)

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

        # equation operator in homogeneous spaces
        A_hom_m = K_hom_m + eta * M1_m

        lift_E_bc = (method == 'conga' and not hom_bc)
        if lift_E_bc:
            # equation operator for bc lifting
            assert K_bc_m is not None
            A_bc_m = K_bc_m + eta * M1_m
        else:
            A_bc_m = None

        # assembling RHS
        rhs_load_dir = get_load_dir(domain_name=domain_name,nc=nc,deg=deg,data='rhs')
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

            u, v, F  = elements_of(V1h.symbolic_space, names='u, v, F')
            expr = dot(f,v)
            l = LinearForm(v, integral(domain, expr))
            lh = discretize(l, domain_h, V1h, backend=PSYDAC_BACKENDS['numba'])
            b  = lh.assemble()
            b_c = b.toarray()

            print("saving this rhs arrays (for future needs) in file "+rhs_filename)
            with open(rhs_filename, 'wb') as file:
                np.savez(file, b_c=b_c)

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
                # expr_b = -k*cross(nn, E_bc)*curl(v) + gamma_h * cross(nn, E_bc) * cross(nn, v)

                # nitsche symmetrization term:
                expr_bs = cross(nn, E_bc)*curl(v)
                ls = LinearForm(v, integral(boundary, expr_bs))
                lsh = discretize(ls, domain_h, V1h, backend=PSYDAC_BACKENDS['numba'])
                bs  = lsh.assemble()
                bs_c = bs.toarray()

                # nitsche penalization term:
                expr_bp = cross(nn, E_bc) * cross(nn, v)
                lp = LinearForm(v, integral(boundary, expr_bp))
                lph = discretize(ls, domain_h, V1h, backend=PSYDAC_BACKENDS['numba'])
                bp  = lph.assemble()
                bp_c = bp.toarray()

                print("saving these rhs arrays (for future needs) in file "+rhs_filename)
                with open(rhs_filename, 'wb') as file:
                    np.savez(file, bs_c=bs_c, bp_c=bp_c)

            # full rhs for nitsche method with non-hom. bc
            b_c += -k * bs_c + gamma_h * bp_c

        if lift_E_bc:
            # lift boundary condition
            debug_plot = False

            # Projector on broken space
            # todo: we should probably apply P1 on E_bc -- it's a bit weird to call it on the list of (pulled back) logical fields.
            nquads = [d + 1 for d in degree]
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
                Ebc_c_tmp = Ebc_c

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
                Ebc_c_tmp -= Ebc_c
                Eh_debug = FemField(V1h, coeffs=array_to_stencil(Ebc_c_tmp, V1h.vector_space))
                Ebc_x_vals, Ebc_y_vals = grid_vals_hcurl(Eh_debug)
                my_small_plot(
                    title=r'E_exact - E_bc',
                    vals=[Ebc_x_vals, Ebc_y_vals],
                    titles=[r'Eb x', r'Eb y'],  # , r'$div_h J$' ],
                    surface_plot=False,
                    xx=xx, yy=yy,
                    save_fig=plot_dir+'diff_Ebc.png',
                    hide_plot=hide_plots,
                    cmap='plasma',
                    dpi=400,
                )

        plot_source = True
        if plot_source:
            # representation of discrete source:
            fh_c = spsolve(M1_m.tocsc(), b_c)
            # fh_norm = np.dot(fh_c,M1_m.dot(fh_c))**0.5
            # print("|| fh || = ", fh_norm)
            # print("|| div fh ||/|| fh || = ", div_norm(fh_c)/fh_norm)

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
                    title=r'source term $J_h$',
                    vals=[abs_fh_vals],
                    titles=[r'$|J_h|$'],  # , r'$div_h J$' ],
                    surface_plot=False,
                    xx=xx, yy=yy,
                    save_fig=plot_dir+'Jh.png',
                    hide_plot=hide_plots,
                    cmap='plasma',
                    dpi=400,
                )

            my_small_streamplot(
                title='source J',
                vals_x=fh_x_vals,
                vals_y=fh_y_vals,
                xx=xx, yy=yy,
                save_fig=plot_dir+'Jh_vf.png',
                hide_plot=hide_plots,
                amplification=.05,
            )


        t_stamp = time_count(t_stamp)
        print("solving with scipy...")
        Eh_c = spsolve(A_hom_m.tocsc(), b_c)
        # E_coeffs = array_to_stencil(Eh_c, V1h.vector_space)
        print("... done.")
        time_count(t_stamp)
        if lift_E_bc:
            # add the lifted boundary condition
            Eh_c += Ebc_c


        # projected solution
        if proj_sol:
            Eh_c = cP1_m.dot(Eh_c)
        # jumpEh_c = Eh_c - PEh_c
        # Eh = FemField(V1h, coeffs=array_to_stencil(PEh_c, V1h.vector_space))
        Eh = FemField(V1h, coeffs=array_to_stencil(Eh_c, V1h.vector_space))

        if save_Eh:
            if os.path.isfile(Eh_filename):
                print('(solution coeff array is already saved, no need to save it again)')
            else:
                print("saving solution coeffs (for future needs) in new file "+Eh_filename)
                with open(Eh_filename, 'wb') as file:
                    np.savez(file, array_coeffs=Eh_c)

        #+++++++++++++++++++++++++++++++
        # plotting and diagnostics
        #+++++++++++++++++++++++++++++++

        # smooth plotting with node-valued grid
        Eh_x_vals, Eh_y_vals = grid_vals_hcurl(Eh)
        my_small_streamplot(
            title=r'discrete field $E_h$',  # for $\omega = $'+repr(omega),
            vals_x=Eh_x_vals,
            vals_y=Eh_y_vals,
            skip=1,
            xx=xx,
            yy=yy,
            amplification=1,
            save_fig=plot_dir+'Eh_vf.png',
            hide_plot=hide_plots,
            dpi = 200,
        )

        Eh_abs_vals = [np.sqrt(abs(ex)**2 + abs(ey)**2) for ex, ey in zip(Eh_x_vals, Eh_y_vals)]
        my_small_plot(
            title=r'amplitude of discrete field $E_h$', # for $\omega = $'+repr(omega),
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

        if save_E_ref:
            if os.path.isfile(E_ref_filename):
                print('(ref solution is already saved, no need to save it again)')
            else:
                print("saving solution values (on cdiag grid) in new file (for future needs)"+E_ref_filename)
                with open(E_ref_filename, 'wb') as file:
                    np.savez(file, x_vals=Eh_x_vals, y_vals=Eh_y_vals)

        xx = xx_cdiag
        yy = yy_cdiag

        if E_ref_vals:
            E_x_vals, E_y_vals = E_ref_vals
            quad_weights = get_grid_quad_weights(etas_cdiag, patch_logvols, mappings_list)
            Eh_errors_cdiag = [np.sqrt( (u1-v1)**2 + (u2-v2)**2 )
                               for u1, v1, u2, v2 in zip(Eh_x_vals, E_x_vals, Eh_y_vals, E_y_vals)]
            l2_error = (np.sum([J_F * err**2 for err, J_F in zip(Eh_errors_cdiag, quad_weights)]))**0.5
            err_message = 'error for nc = {0} with gamma_h = {1} and proj_sol = {2}: {3}\n'.format(nc, gamma_h, proj_sol, l2_error)
            print(err_message)
            error_filename = error_fn(source_type=source_type, method=method, k=k, domain_name=domain_name,deg=deg)
            if not os.path.exists(error_filename):
                open(error_filename, 'w')
            with open(error_filename, 'a') as a_writer:
                a_writer.write(err_message)

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








