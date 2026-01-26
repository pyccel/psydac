#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from logging import warning
import os

import numpy as np

from sympde.topology import Cube, Mapping

from psydac.api.postprocessing import OutputManager, PostProcessManager
from psydac.fem.basic import FemField
from psydac.utilities.utils import refine_array_1d

NUM_DIMS_LOGICAL  = 3
NUM_DIMS_PHYSICAL = 3

#==============================================================================
class Laplacian:

    def __init__(self, mapping):

        assert isinstance(mapping, Mapping)

        sym = mapping

        self._eta        = sym.logical_coordinates
        self._metric     = sym.metric_expr
        self._metric_det = sym.metric_det_expr

    # ...
    def __call__(self, phi):

        from sympy import sqrt, Matrix

        u      = self._eta
        G      = self._metric
        sqrt_g = sqrt( self._metric_det )

        # Store column vector of partial derivatives of phi w.r.t. uj
        dphi_du = Matrix( [phi.diff( uj ) for uj in u] )

        # Compute gradient of phi in tangent basis: A = G^(-1) dphi_du
        A = G.LUsolve( dphi_du )

        # Compute Laplacian of phi using formula for divergence of vector A
        lapl = sum( (sqrt_g*Ai).diff( ui ) for ui,Ai in zip( u,A ) ) / sqrt_g

        return lapl


#==============================================================================
# Define the Spherical coordinate system
class TargetTorusMapping(Mapping):
    """
    3D Torus with a polar cross-section like in the TargetMapping.
    """
    _expressions = {'x' : '(R0 + (1-k)*x1*cos(x2) - D*x1**2) * cos(x3)',
                    'y' : '(R0 + (1-k)*x1*cos(x2) - D*x1**2) * sin(x3)',
                    'z' : '(Z0 + (1+k)*x1*sin(x2))'}

    _ldim = 3
    _pdim = 3

#==============================================================================
def run_model(ncells, degree, comm=None, is_logical=False):

    from sympy import sin, pi, cos

    from sympde.calculus import laplace, dot, grad
    from sympde.topology import ScalarFunctionSpace, element_of, LogicalExpr, Union
    from sympde.expr     import BilinearForm, LinearForm, Norm
    from sympde.expr     import EssentialBC, find
    from sympde.expr     import integral

    from psydac.api.discretization import discretize
    from psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL


    # Backend to activate multi threading
    backend = PSYDAC_BACKEND_GPYCCEL.copy()
#    backend['openmp'] = True

    # Choose number of OpenMP threads
    os.environ['OMP_NUM_THREADS'] = "2"

    # Define topological domain
    r_in    = 0.05
    r_out   = 0.2
    A       = Cube('A', bounds1=(r_in, r_out), bounds2=(0, 2 * np.pi), bounds3=(0, 2* np.pi))
    mapping = TargetTorusMapping('M', 3, R0=1.0, Z0=0, k=0.3, D=0.2)
    Omega   = mapping(A)

    Omega_logical = Omega.logical_domain

    # Method of manufactured solutions: define exact
    # solution phi_e, then compute right-hand side f
    if not is_logical:
        print("Start creating physical expression", flush=True)
        x, y, z  = Omega.coordinates
        r_sq   = y**2 + z**2
        # arg = pi * (r_sq - r_in**2) / (r_out**2 - r_in**2)
        u_e = sin(pi * x) * sin(pi * y) * cos(pi * z)
        f = -laplace(u_e)

    else:
        print("Start creating Logical expression", flush=True)
        logical_laplace = Laplacian(mapping=mapping)
        x1, x2, x3 = Omega.logical_domain.coordinates
        x, y, z = Omega.mapping.expressions
        u_e_logical = (0.05 - x1) * (0.2 - x1) * sin(x2) * cos(x3)
        f_logical   = - logical_laplace(u_e_logical)

    # Define abstract model
    V = ScalarFunctionSpace('V', Omega, kind='h1')
    v = element_of(V, name='v')
    u = element_of(V, name='u')

    a = BilinearForm((u,v), integral(Omega, dot(grad(v), grad(u))))
    if not is_logical:
        l = LinearForm(v , integral(Omega, f * v))

    if is_logical:
        a_log = LogicalExpr(a, Omega)

        u = a_log.trial_functions[0]
        v = a_log.test_functions[0]

        V = u.space
        l_log =   LinearForm(v , integral(Omega_logical, f_logical * v * mapping.det_jacobian))

    if is_logical:
        bc = EssentialBC(u, 0, Union(Omega_logical.get_boundary(axis=0, ext=1), Omega_logical.get_boundary(axis=0, ext=-1)))
    else:
        bc = EssentialBC(u, u_e, Union(mapping(Omega_logical.get_boundary(axis=0, ext=1)), mapping(Omega_logical.get_boundary(axis=0, ext=-1))))

    if is_logical:
        equation = find(u, forall=v, lhs=a_log(u,v), rhs=l_log(v), bc=bc)
    else:
        equation = find(u, forall=v, lhs=a(u,v), rhs=l(v), bc=bc)

    # Define (abstract) error norms
    if is_logical:
        v2 = element_of(V, name='v2')
        l2norm_u_e = Norm(u_e_logical - v2, Omega_logical, kind='l2')

        l2norm_e = Norm(u - u_e_logical, Omega_logical, kind='l2')

    else:
        v2 = element_of(V, name='v2')
        l2norm_u_e = Norm(u_e - v2, Omega, kind='l2')

        l2norm_e = Norm(u - u_e, Omega, kind='l2')

    print("Start discretization", flush=True)
    # Create computational domain from topological domain
    periodic = [False, False, False]
    periodic_log = [False, True, True]
    Omega_h = discretize(Omega, ncells=ncells, periodic=periodic, comm=comm)
    Omega_log_h = discretize(Omega_logical, ncells=ncells, periodic=periodic_log, comm=comm)

    # Number of quadrature points to be used for assemblying bilinear and linear forms
    nquads = [p + 1 for p in degree]

    # Choose whether to work on logical or physical domain
    if is_logical:
        domain_h = Omega_log_h
    else:
        domain_h = Omega_h

    # Create discrete spline space
    Vh = discretize(V, domain_h, degree=degree)

    # Discretize equation
    equation_h = discretize(equation, domain_h, [Vh, Vh], nquads=nquads, backend=backend)

    # Discretize norms
    l2norm_u_e_h = discretize(l2norm_u_e, domain_h, Vh, nquads=nquads, backend=backend)
    l2norm_e_h   = discretize(l2norm_e  , domain_h, Vh, nquads=nquads, backend=backend)

    # Solve discrete equation to obtain finite element coefficients
    print('Start equation_h.solve()')
    equation_h.set_solver('cg', tol=1e-9, maxiter=10**5, info=True, verbose=True)

    u_h, info = equation_h.solve()
    if not info['success']:
        print(info, flush=True)

    # Compute error norms from solution field
    vh = FemField(Vh)

    l2_norm_ue = l2norm_u_e_h.assemble(v2=vh)
    l2_norm_e  = l2norm_e_h.assemble(u=u_h)

    return locals()

# =============================================================================
def save_model(ncells, degree, is_logical, namespace, comm):

    ne1, ne2, ne3 = ncells
    p1, p2, p3 = degree

    Om = OutputManager(f'spaces_{ne1}_{ne2}_{ne3}_{p1}_{p2}_{p3}_{is_logical}.yml', f'fields_{ne1}_{ne2}_{ne3}_{p1}_{p2}_{p3}_{is_logical}.h5', comm=comm)
    Om.add_spaces(V=namespace['Vh'])
    Om.export_space_info()
    Om.set_static()
    Om.export_fields(u=namespace['u_h'])
    Om.close()
    if comm:
        comm.Barrier()


# =============================================================================
def export_model(ncells, degree, is_logical, comm, npts_per_cell):
    # Recreate domain
    r_in  = 0.05
    r_out = 0.2
    A       = Cube('A', bounds1=(r_in, r_out), bounds2=(0, 2 * np.pi), bounds3=(0, 2* np.pi))
    mapping = TargetTorusMapping('M', 3, R0=1.0, Z0=0, k=0.3, D=0.2)
    Omega = mapping(A)

    p1, p2, p3 = degree
    ne1, ne2, ne3 = ncells
    Pm = PostProcessManager(domain=Omega,
                            space_file=f'spaces_{ne1}_{ne2}_{ne3}_{p1}_{p2}_{p3}_{is_logical}.yml',
                            fields_file=f'fields_{ne1}_{ne2}_{ne3}_{p1}_{p2}_{p3}_{is_logical}.h5',
                            comm=comm)
    if isinstance(npts_per_cell, int):
        npts_per_cell = [npts_per_cell] * 3
    if any( n <= 1 for n in npts_per_cell):
        warning.warn('Refinement must be at least 2\nSetting refinement to 2')
        npts_per_cell = [max(n, 2) for n in npts_per_cell]

    grid = [refine_array_1d(Pm.spaces['V'].breaks[i], n=npts_per_cell[i] - 1, remove_duplicates=False) for i in range(3)]

    u_e_logical = lambda x,y,z: (0.05 - x) * (0.2 - x) * np.sin(y) * np.cos(z)
    u_e_physical = lambda x,y,z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)

    phy_f = {}
    log_f = {}

    if is_logical:
        log_f['u_e_log'] =  u_e_logical
    else:
        phy_f['u_e_phy'] = u_e_physical


    Pm.export_to_vtk(f'poisson_3d_target_torus_{ne1}_{ne2}_{ne3}_{p1}_{p2}_{p3}_{npts_per_cell}_{is_logical}',
                     grid=grid,
                     npts_per_cell=npts_per_cell,
                     snapshots='none',
                     fields='u',
                     additional_physical_functions=phy_f,
                     additional_logical_functions=log_f)


#==============================================================================
def parse_input_arguments():

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        description     = "Solve Poisson's equation on a 3D domain with" +
                          " homogeneous Dirichlet boundary conditions."
    )

    parser.add_argument( '-d',
        type    = int,
        nargs   = 3,
        default = [2, 2, 2],
        metavar = ('P1', 'P2', 'P3'),
        dest    = 'degree',
        help    = 'Spline degree along each dimension'
    )

    parser.add_argument( '-n',
        type    = int,
        nargs   = 3,
        default = [10, 10, 10],
        metavar = ('N1', 'N2', 'N3'),
        dest    = 'ncells',
        help    = 'Number of grid cells (elements) along each dimension'
    )

    # parser.add_argument( '-v',
    #     action  = 'store_true',
    #     dest    = 'verbose',
    #     help    = 'Increase output verbosity'
    # )

    parser.add_argument('-l',
        action = 'store_true',
        dest   = 'is_logical',
        help   = 'Define problem in the logical domain'
    )

    parser.add_argument( '-p',
        action  = 'store_true',
        dest    = 'plots',
        help    = 'Plot exact solution and error'
    )

    parser.add_argument('-r',
        action='store',
        default=2,
        type=int,
        metavar='R',
        dest='refinement',
        help='Refinement of the exported model')

    parser.add_argument('-m',
        action='store_true',
        dest='run_m',
        help='Run the model')

    return parser.parse_args()

#==============================================================================
def main(degree, ncells, is_logical, plots, refinement, run_m):

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
    except ImportError:
        comm = None
        rank = 0
    if run_m:
        namespace = run_model(ncells, degree, comm, is_logical)

        if rank == 0:
            print()
            print('L2 Norm error = {}'.format(namespace['l2_norm_ue']))
            print('L2 Norm exact solution = {}'.format(namespace['l2_norm_e']))
            print(flush=True)

        save_model(ncells, degree, is_logical, namespace, comm)

    if plots:
        export_model(ncells, degree, is_logical, comm, refinement)
    if comm:
        comm.Barrier()

#==============================================================================
if __name__ == '__main__':

    args = parse_input_arguments()
    main( **vars( args ) )
