"""
    solver for the problem: find u in H^1, such that

      A u = f             on \\Omega
        u = u_bc          on \\partial \\Omega

    where the operator

      A u := eta * u  -  mu * div grad u

    is discretized as  Ah: V0h -> V0h  in a broken-FEEC approach involving a discrete sequence on a 2D multipatch domain \\Omega,

      V0h  --grad->  V1h  -—curl-> V2h
"""

from mpi4py import MPI

import os
import numpy as np
from collections import OrderedDict

from sympy import lambdify
from scipy.sparse.linalg import spsolve

from sympde.calculus import dot
from sympde.expr.expr import LinearForm
from sympde.expr.expr import integral, Norm
from sympde.topology import Derham
from sympde.topology import element_of

from psydac.api.settings import PSYDAC_BACKENDS
from psydac.feec.multipatch.api import discretize
from psydac.feec.pull_push import pull_2d_h1
from psydac.feec.multipatch.utils_conga_2d import P0_phys

from psydac.feec.multipatch.fem_linear_operators import IdLinearOperator
from psydac.feec.multipatch.operators import HodgeOperator
from psydac.feec.multipatch.plotting_utilities import plot_field
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain
from psydac.feec.multipatch.examples.ppc_test_cases import get_source_and_solution_h1
from psydac.feec.multipatch.utilities import time_count
from psydac.feec.multipatch.non_matching_operators import construct_h1_conforming_projection, construct_hcurl_conforming_projection
from psydac.api.postprocessing import OutputManager, PostProcessManager

from psydac.linalg.utilities import array_to_psydac
from psydac.fem.basic import FemField

from psydac.api.postprocessing import OutputManager, PostProcessManager


def solve_h1_source_pbm(
        nc=4, deg=4, domain_name='pretzel_f', backend_language=None, source_type='manu_poisson_elliptic',
        eta=-10., mu=1., gamma_h=10., plot_dir=None,
):
    """
    solver for the problem: find u in H^1, such that

      A u = f             on \\Omega
        u = u_bc          on \\partial \\Omega

    where the operator

      A u := eta * u  -  mu * div grad u

    is discretized as  Ah: V0h -> V0h  in a broken-FEEC approach involving a discrete sequence on a 2D multipatch domain \\Omega,

      V0h  --grad->  V1h  -—curl-> V2h

    Examples:

      - Helmholtz equation with
          eta = -omega**2
          mu  = 1

      - Poisson equation with Laplace operator L = A,
          eta = 0
          mu  = 1

    :param nc: nb of cells per dimension, in each patch
    :param deg: coordinate degree in each patch
    :param domain_name: name of the domain
    :param backend_language: backend language for the operators
    :param source_type: must be implemented in get_source_and_solution_h1
    :param eta: coefficient of the elliptic operator
    :param mu: coefficient of the elliptic operator
    :param gamma_h: jump penalization parameter
    :param plot_dir: directory for the plots (if None, no plots are generated)
    """

    degree = [deg, deg]

    print('---------------------------------------------------------------------------------------------------------')
    print('Starting solve_h1_source_pbm function with: ')
    print(' ncells = {}'.format(nc))
    print(' degree = {}'.format(degree))
    print(' domain_name = {}'.format(domain_name))
    print(' backend_language = {}'.format(backend_language))
    print('---------------------------------------------------------------------------------------------------------')

    print('building the multipatch domain...')
    domain = build_multipatch_domain(domain_name=domain_name)
    mappings = OrderedDict([(P.logical_domain, P.mapping)
                           for P in domain.interior])
    mappings_list = list(mappings.values())

    if isinstance(ncells, int):
        ncells = [nc, nc]
    else:
        ncells = {patch.name: [nc[i], nc[i]]
                    for (i, patch) in enumerate(domain.interior)}

    domain_h = discretize(domain, ncells=ncells)

    print('building the symbolic and discrete deRham sequences...')
    derham = Derham(domain, ["H1", "Hcurl", "L2"])
    derham_h = discretize(derham, domain_h, degree=degree)

    # multi-patch (broken) spaces
    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2
    print('dim(V0h) = {}'.format(V0h.nbasis))
    print('dim(V1h) = {}'.format(V1h.nbasis))
    print('dim(V2h) = {}'.format(V2h.nbasis))

    print('broken differential operators...')
    # broken (patch-wise) differential operators
    bD0, bD1 = derham_h.broken_derivatives_as_operators
    bD0_m = bD0.to_sparse_matrix()

    print('building the discrete operators:')
    print('commuting projection operators...')
    nquads = [4 * (d + 1) for d in degree]
    P0, P1, P2 = derham_h.projectors(nquads=nquads)

    I0 = IdLinearOperator(V0h)
    I0_m = I0.to_sparse_matrix()

    print('Hodge operators...')
    # multi-patch (broken) linear operators / matrices
    H0 = HodgeOperator(V0h, domain_h, backend_language=backend_language)
    H1 = HodgeOperator(V1h, domain_h, backend_language=backend_language)

    H0_m = H0.to_sparse_matrix()                # = mass matrix of V0
    dH0_m = H0.get_dual_Hodge_sparse_matrix()  # = inverse mass matrix of V0
    H1_m = H1.to_sparse_matrix()                # = mass matrix of V1

    print('conforming projection operators...')
    # conforming Projections (should take into account the boundary conditions
    # of the continuous deRham sequence)
    cP0_m = construct_h1_conforming_projection(V0h, hom_bc=True)

    def lift_u_bc(u_bc):
        if u_bc is not None:
            print('lifting the boundary condition in V0h...  [warning: Not Tested Yet!]')
            d_ubc_c = derham_h.get_dual_dofs(space='V0', f=u_bc, backend_language=backend_language, return_format='numpy_array')
            ubc_c = dH0_m.dot(d_ubc_c)

            ubc_c = ubc_c - cP0_m.dot(ubc_c)
        else:
            ubc_c = None
        return ubc_c

    # Conga (projection-based) stiffness matrices:
    # div grad:
    pre_DG_m = - bD0_m.transpose() @ H1_m @ bD0_m

    # jump penalization:
    jump_penal_m = I0_m - cP0_m
    JP0_m = jump_penal_m.transpose() @ H0_m @ jump_penal_m

    # useful for the boundary condition (if present)
    pre_A_m = cP0_m.transpose() @ (eta * H0_m - mu * pre_DG_m)
    A_m = pre_A_m @ cP0_m + gamma_h * JP0_m

    print('getting the source and ref solution...')
    f_scal, u_bc, u_ex = get_source_and_solution_h1(
        source_type=source_type, eta=eta, mu=mu, domain=domain, domain_name=domain_name,
    )

    # compute approximate source f_h
    b_c =  derham_h.get_dual_dofs(space='V0', f=f_scal, backend_language=backend_language, return_format='numpy_array')
    # source in primal sequence for plotting
    f_c = dH0_m.dot(b_c)
    b_c = cP0_m.transpose() @ b_c

    ubc_c = lift_u_bc(u_bc)

    if ubc_c is not None:
        # modified source for the homogeneous pbm
        print('modifying the source with lifted bc solution...')
        b_c = b_c - pre_A_m.dot(ubc_c)

    # direct solve with scipy spsolve
    print('solving source problem with scipy.spsolve...')
    uh_c = spsolve(A_m, b_c)

    # project the homogeneous solution on the conforming problem space
    print('projecting the homogeneous solution on the conforming problem space...')
    uh_c = cP0_m.dot(uh_c)

    if ubc_c is not None:
        # adding the lifted boundary condition
        print('adding the lifted boundary condition...')
        uh_c += ubc_c

    print('getting and plotting the FEM solution from numpy coefs array...')

    if u_ex:
        u_ex_c = derham_h.get_dual_dofs(space='V0', f=u_ex, backend_language=backend_language, return_format='numpy_array')
        u_ex_c = dH0_m.dot(u_ex_c)

    if plot_dir is not None:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        OM = OutputManager(plot_dir + '/spaces.yml', plot_dir + '/fields.h5')
        OM.add_spaces(V0h=V0h)
        OM.set_static()

        stencil_coeffs = array_to_psydac(uh_c, V0h.vector_space)
        vh = FemField(V0h, coeffs=stencil_coeffs)
        OM.export_fields(vh=vh)

        stencil_coeffs = array_to_psydac(f_c, V0h.vector_space)
        fh = FemField(V0h, coeffs=stencil_coeffs)
        OM.export_fields(fh=fh)
        
        if u_ex:
            stencil_coeffs = array_to_psydac(u_ex_c, V0h.vector_space)
            uh_ex = FemField(V0h, coeffs=stencil_coeffs)
            OM.export_fields(uh_ex=uh_ex)

        OM.export_space_info()
        OM.close()

        PM = PostProcessManager(
            domain=domain,
            space_file=plot_dir + '/spaces.yml',
            fields_file=plot_dir + '/fields.h5')

        PM.export_to_vtk(
            plot_dir + "/u_h",
            grid=None,
            npts_per_cell=[6] * 2,
            snapshots='all',
            fields='vh')

        PM.export_to_vtk(
            plot_dir + "/f_h",
            grid=None,
            npts_per_cell=[6] * 2,
            snapshots='all',
            fields='fh')

        if u_ex:
            PM.export_to_vtk(
                plot_dir + "/uh_ex",
                grid=None,
                npts_per_cell=[6] * 2,
                snapshots='all',
                fields='uh_ex')

        PM.close()

    if u_ex:
        err = uh_c - u_ex_c
        rel_err = np.sqrt(np.dot(err, H0_m.dot(err)))/np.sqrt(np.dot(u_ex_c,H0_m.dot(u_ex_c)))
        
        return rel_err


if __name__ == '__main__':

    omega = np.sqrt(170)  # source
    eta = -omega**2 

    source_type = 'manu_poisson_elliptic'

    domain_name = 'pretzel_f'

    nc = 10
    deg = 2

    run_dir = '{}_{}_nc={}_deg={}/'.format(domain_name, source_type, nc, deg)
    solve_h1_source_pbm(
        nc=nc, deg=deg,
        eta=eta,
        mu=1,  # 1,
        domain_name=domain_name,
        source_type=source_type,
        backend_language='pyccel-gcc',
        plot_dir='./plots/h1_source_pbms_conga_2d/' + run_dir,
    )