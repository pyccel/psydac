"""
    solver for the problem: find u in H^1, such that

      A u = f             on \\Omega
        u = u_bc          on \\partial \\Omega

    where the operator

      A u := eta * u  -  mu * div grad u

    is discretized as  Ah: V0h -> V0h  in a broken-FEEC approach involving a discrete sequence on a 2D multipatch domain \\Omega,

      V0h  --grad->  V1h  -—curl-> V2h
"""
import os
import numpy as np

from sympde.topology import Derham

from psydac.api.discretization import discretize
from psydac.linalg.basic       import IdentityOperator
from psydac.linalg.solvers     import inverse

from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain
from psydac.feec.multipatch.examples.ppc_test_cases     import get_source_and_solution_h1

from psydac.fem.projectors import get_dual_dofs

from psydac.fem.basic          import FemField
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

    if isinstance(nc, int):
        ncells = [nc, nc]
    else:
        ncells = {patch.name: [nc[i], nc[i]]
                    for (i, patch) in enumerate(domain.interior)}

    domain_h = discretize(domain, ncells=ncells)

    print('building the symbolic and discrete deRham sequences...')
    derham = Derham(domain, ["H1", "Hcurl", "L2"])
    derham_h = discretize(derham, domain_h, degree=degree)

    V0h, V1h, V2h = derham_h.spaces
    print('dim(V0h) = {}'.format(V0h.nbasis))
    print('dim(V1h) = {}'.format(V1h.nbasis))
    print('dim(V2h) = {}'.format(V2h.nbasis))

    print('broken differential operators...')
    # broken (patch-wise) differential operators
    bD0, bD1 = derham_h.derivatives(kind='linop')

    print('Hodge operators...')
    # multi-patch (broken) linear operators 
    H0 = derham_h.Hodge_operators(space='V0', kind='linop', backend_language=backend_language)
    H1 = derham_h.Hodge_operators(space='V1', kind='linop', backend_language=backend_language)
    dH0 = derham_h.Hodge_operators(space='V0', kind='linop', dual=True, backend_language=backend_language)

    print('conforming projection operators...')
    # conforming Projections (should take into account the boundary conditions
    # of the continuous deRham sequence)
    cP0, cP1, cP2 = derham_h.conforming_projectors(kind='linop', hom_bc = True)

    print('building the discrete operators:')

    I0 = IdentityOperator(V0h.coeff_space)
    
    # div grad
    DG = - bD0.T @ H1 @ bD0

    # jump penalization:
    JP0 = (I0 - cP0).T @ H0 @ (I0 - cP0)

    # useful for the boundary condition (if present)
    pre_A = cP0.T @ (eta * H0 - mu * DG) 
    
    A = pre_A @ cP0 + gamma_h * JP0

  
    f_scal, u_bc, u_ex = get_source_and_solution_h1(source_type=source_type, eta=eta, mu=mu, domain=domain, domain_name=domain_name,)

    df = get_dual_dofs(Vh=V0h, f=f_scal, domain_h=domain_h, backend_language=backend_language)
    f = dH0.dot(df)
    df = cP0.T @ df

    def lift_u_bc(u_bc):
        if u_bc is not None:
            du_bc = get_dual_dofs(Vh=V0h, f=u_bc, domain_h = domain_h, backend_language=backend_language)
            u_bc = dH0.dot(du_bc)
            u_bc = u_bc - cP0.dot(u_bc)

        else:
            u_bc = None
            
        return u_bc

    ubc = lift_u_bc(u_bc)

    if ubc is not None:
        # modified source for the homogeneous pbm
        print('modifying the source with lifted bc solution...')
        df = df - pre_A.dot(ubc)

    # direct solve with scipy spsolve
    print('solving source problem with conjugate gradient...')
    solver = inverse(A, solver='cg', tol=1e-8)
    u = solver.solve(df)

    # project the homogeneous solution on the conforming problem space
    print('projecting the homogeneous solution on the conforming problem space...')
    u = cP0.dot(u)

    if ubc is not None:
        # adding the lifted boundary condition
        print('adding the lifted boundary condition...')
        u += ubc


    if u_ex:
        u_ex = get_dual_dofs(Vh=V0h, f=u_ex, domain_h=domain_h, backend_language=backend_language)
        u_ex = dH0.dot(u_ex)


    if plot_dir is not None:
        print('plotting the FEM solution...')

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        OM = OutputManager(plot_dir + '/spaces.yml', plot_dir + '/fields.h5')
        OM.add_spaces(V0h=V0h)
        OM.set_static()

        uh = FemField(V0h, coeffs=u)
        OM.export_fields(uh=uh)

        fh = FemField(V0h, coeffs=f)
        OM.export_fields(fh=fh)
        
        if u_ex:
            uh_ex = FemField(V0h, coeffs=u_ex)
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
            fields='uh')

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
        err = u - u_ex
        rel_err = np.sqrt( err.inner(H0.dot(err))) / np.sqrt(u_ex.inner(H0.dot(u_ex)))

        return rel_err


if __name__ == '__main__':

    omega = np.sqrt(170)  # source
    eta = -omega**2 
    mu=0
    gamma_h = 10

    source_type = 'manu_poisson_elliptic'

    domain_name = 'pretzel_f'

    nc = 4
    deg = 2

    run_dir = '{}_{}_nc={}_deg={}/'.format(domain_name, source_type, nc, deg)
    solve_h1_source_pbm(
        nc=nc, deg=deg,
        eta=eta,
        mu=mu,  # 1,
        domain_name=domain_name,
        source_type=source_type,
        backend_language='pyccel-gcc',
        plot_dir='./plots/h1_source_pbms_conga_2d/' + run_dir,
    )