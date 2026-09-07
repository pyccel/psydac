import numpy as np

from sympde.topology import NCube
from sympde.topology import domain
from sympde.topology import Square, PolarMapping
from sympde.topology.mapping import BasicCallableMapping
from sympde.utilities.utils import plot_domain

from psydac.mapping.discrete import SplineMapping
from psydac.cad.geometry     import Geometry
from psydac.api.tests.build_domain import build_11_patch_pretzel
from psydac.fem.splines      import SplineSpace
from psydac.fem.tensor       import TensorFemSpace
from psydac.ddm.cart         import DomainDecomposition

import pytest


def spline_mapping_approx(
        F, min_coords=None, max_coords=None,
        degree=None, ncells=None, periodic=(False, False), mpi_comm=None):
    
    """ 
    should this function be merged with psydac.mapping.discrete.SplineMapping.from_mapping() ?
    """

    # Accept either a callable mapping directly or a symbolic mapping able
    # to produce one through get_callable_mapping().
    if not isinstance(F, BasicCallableMapping):
        if hasattr(F, 'get_callable_mapping'):
            F = F.get_callable_mapping()
        else:
            raise TypeError('F must be a BasicCallableMapping or expose get_callable_mapping().')
    assert degree is not None and ncells is not None, "degree and ncells must be provided for spline mapping approximation"   
    
    # Create uniform grids and 1d spline spaces         
    grids = [np.linspace(min_coords[d], max_coords[d], num=ncells[d]+1) for d in range(F.ldim)]
    V_spl = [SplineSpace(degree[d], grid=grids[d], periodic=periodic[d]) for d in range(F.ldim)]
    
    # Decompose domain across MPI processes and create tensor-product spline space, distributed
    dd = DomainDecomposition(ncells, periodic, comm=mpi_comm)
    V = TensorFemSpace(dd, *V_spl)

    F_h = SplineMapping.from_mapping(V, F) 

    # domain_h = Geometry.from_discrete_mapping(F_h, domain_log=domain_log, comm=mpi_comm)

    return F_h

@pytest.mark.parametrize('spline_mapping', [False, True])    
def test_poisson_mapping(spline_mapping):    
    # Define the topological geometry for each patch
    rmin, rmax = 0.3, 1.

    F_degree = (2, 2)
    F_ncells = (8, 8)

    # First quarter annulus
    domain_log_1 = Square('A_1', bounds1=(0., 1.), bounds2=(0., 1/2 * np.pi))
    F_1 = PolarMapping('F_1', dim=2, c1=0., c2=0., rmin=rmin, rmax=rmax)
    if spline_mapping:
        F_1s = spline_mapping_approx(F=F_1,
            min_coords=domain_log_1.min_coords, max_coords=domain_log_1.max_coords,
            degree=F_degree, ncells=F_ncells, periodic=(False, False),
            )
        F_1.set_callable_mapping(F_1s)
        Omega_1 = F_1(domain_log_1)
    else:
        Omega_1 = F_1(domain_log_1)
    
    # Second quarter annulus
    F_2 = PolarMapping('F_2', dim=2, c1=rmin+rmax, c2=0., rmin=rmax, rmax=rmin)   ## rmin > rmax ??? 
    domain_log_2 = Square('A_2', bounds1=(0., 1.), bounds2=(np.pi, 3/2 * np.pi))
    if spline_mapping:
        F_2s = spline_mapping_approx(F=F_2,
            min_coords=domain_log_2.min_coords, max_coords=domain_log_2.max_coords,
            degree=F_degree, ncells=F_ncells, periodic=(False, False)
            )
        F_2.set_callable_mapping(F_2s)
        Omega_2 = F_2(domain_log_2)
    else:
        Omega_2 = F_2(domain_log_2)

    # Join the patches
    from sympde.topology import Domain
    connectivity = [((0,1,-1),(1,1,-1), 1)]
    patches = [Omega_1, Omega_2]
    Omega = Domain.join(patches, connectivity, 'domain')

    for map in [Omega_1.mapping, Omega_2.mapping]:
        map_call = map.get_callable_mapping()
        print(f'map_call = {map_call}, {type(map_call)}')
        patch = Omega_1.interior
        linspace_0 = np.linspace(patch.min_coords[0], patch.max_coords[0], 10, endpoint=True)
        linspace_1 = np.linspace(patch.min_coords[1], patch.max_coords[1], 10, endpoint=True)

        # if isolines:
        mesh_grid = np.meshgrid(linspace_0, linspace_1, indexing='ij')

        # print(f'mesh_grid: {type(mesh_grid)}, {len(mesh_grid)}')
        # print(f'mesh_grid[0]: {type(mesh_grid[0])}, {len(mesh_grid[0])}, {mesh_grid[0].shape}')
        # for map_Xd in map_call._fields:
            # print(f'map_Xd = {map_Xd}, {type(map_Xd)}')

        print(f'single point evaluation:')
        XX, YY = map_call(1.1, 0.2)
        # XX, YY = map_call(*mesh_grid)
        print(f'XX = {XX}')
        print(f'YY = {YY}')

        print(f'array evaluation:')
        eta_1 = np.array([0.1, 0.4])
        eta_2 = np.array([0.2, 0.45])
        try:
            XX, YY = map_call(eta_1, eta_2)
        except TypeError as err:
            # Some callable mappings only support scalar evaluation.
            if 'length-1 arrays' not in str(err):
                raise
            values = [map_call(float(e1), float(e2)) for e1, e2 in zip(eta_1, eta_2)]
            XX = np.array([v[0] for v in values])
            YY = np.array([v[1] for v in values])
        print(f'XX = {XX}')
        print(f'YY = {YY}')

    # Simple visualization of the topological domain.
    # The spline callable mapping used for the solve path may only support
    # scalar evaluation, while plot_domain evaluates mappings on array grids.
    # An AnalyticMapping is its own array-capable callable, so temporarily
    # point each symbolic mapping back at itself for plotting, then restore.
    _plot_backups = []
    for _mapping in [Omega_1.mapping, Omega_2.mapping]:
        _callable = _mapping.get_callable_mapping()
        if isinstance(_callable, SplineMapping):
            _plot_backups.append((_mapping, _callable))
            _mapping.set_callable_mapping(_mapping)

    try:
        plot_domain(Omega, draw=False, isolines=True)
    finally:
        for _mapping, _callable in _plot_backups:
            _mapping.set_callable_mapping(_callable)

    from sympde.calculus import grad, dot
    from sympde.calculus import minus, plus
    from sympde.topology import Derham

    from sympde.expr.expr          import LinearForm, BilinearForm
    from sympde.expr.expr          import integral              
    from sympde.expr.expr          import Norm                       
    from sympde.expr               import find, EssentialBC

    from sympde.topology import ScalarFunctionSpace
    from sympde.topology import elements_of
    from sympde.topology import NormalVector

    from psydac.api.discretization import discretize
    from psydac.api.settings import PSYDAC_BACKENDS

    from    psydac.linalg.basic     import IdentityOperator
    from    psydac.linalg.solvers   import inverse
    from psydac.fem.plotting_utilities import plot_field_2d as plot_field

    from psydac.fem.basic import FemField
        
        
    # print(f'domain = {domain}, {type(domain)}')
    # print(f'mapping = {domain.mapping}, {type(domain.mapping)}')

    # Define the abstract model to solve Poisson's equation using the manufactured solution method
    x,y       = Omega.coordinates
    solution  = x**2 + y**2
    f         = -4
    
    derham   = Derham(Omega, sequence=['h1', 'hcurl'])

    V, V1, V2  = derham.spaces

    # V   = ScalarFunctionSpace('V', Omega, kind=None)

    u, v = elements_of(V, names='u, v')
    nn   = NormalVector('nn')

#     bc   = EssentialBC(u, solution, Omega.boundary)

    error  = u - solution

    I = Omega.interfaces

    kappa  = 10**3

    expr_I =- 0.5*dot(grad(plus(u)),nn)*minus(v)  + 0.5*dot(grad(minus(v)),nn)*plus(u)  - kappa*plus(u)*minus(v)\
            + 0.5*dot(grad(minus(u)),nn)*plus(v)  - 0.5*dot(grad(plus(v)),nn)*minus(u)  - kappa*plus(v)*minus(u)\
            - 0.5*dot(grad(minus(v)),nn)*minus(u) - 0.5*dot(grad(minus(u)),nn)*minus(v) + kappa*minus(u)*minus(v)\
            + 0.5*dot(grad(plus(v)),nn)*plus(u)   + 0.5*dot(grad(plus(u)),nn)*plus(v)   + kappa*plus(u)*plus(v)

    expr   = dot(grad(u),grad(v))

    a = BilinearForm((u,v),  integral(Omega, expr) + integral(I, expr_I))
#     a = BilinearForm((u,v),  integral(Omega, expr))
    l = LinearForm(v, integral(Omega, f*v))

#     equation = find(u, forall=v, lhs=a(u,v), rhs=l(v), bc=bc)

    l2norm = Norm(error, Omega, kind='l2')
    h1norm = Norm(error, Omega, kind='h1')

    backend = PSYDAC_BACKENDS['python']

    # Uncomment to use OpenMp
    # import os
    # os.environ['OMP_NUM_THREADS'] = "4"
    # backend['omp'] = True

    ncells = [10, 10]
    degree = [2, 2]
    periodic = [False, False]

    nquads = [p + 1 for p in degree]

    # MPI version
    # from mpi4py import MPI
    # comm = MPI.COMM_WORLD
    # Omega_h = discretize(Omega, ncells=ncells, comm=comm)
    Omega_h = discretize(Omega, ncells=ncells, periodic=periodic)

    derham_h = discretize(derham, Omega_h, degree=degree)
    Vh, V1h, V2h = derham_h.spaces
    # Vh        = discretize(V, Omega_h, degree=degree)

    # Discrete bilinear forms
    print(f'discretization of the bilinear form...')    
    a_h = discretize(a, Omega_h, (Vh, Vh), nquads=nquads, backend=backend)
    b_h = discretize(l, Omega_h, Vh, nquads=nquads, backend=backend)

    # Mass matrices (StencilMatrix or BlockLinearOperator objects)
    print(f'assembling of the matrix...')    
    A_pw = a_h.assemble()
    B = b_h.assemble()

    DP0, DP1, _ = derham_h.dirichlet_projectors(kind='linop')
    I0             = IdentityOperator(Vh.coeff_space)

    print(f'defining the inverse discrete operator...')
    A = DP0 @ A_pw @ DP0 + kappa * (I0 - DP0)
    A_inv = inverse(A, 'cg', maxiter=1000, tol=1e-15)

    uh_c = A_inv @ (DP0 @ B)

    uh = FemField(Vh, coeffs=uh_c)
    plot_field(fem_field=uh, domain=Omega, title='Poisson solution uh', hide_plot=False, filename=f'poisson_uh_splinemap={spline_mapping}.png')


if __name__ == '__main__':
    for spline_mapping in [True, False]:
        print(f'Running test_poisson_mapping with spline_mapping={spline_mapping}')
        test_poisson_mapping(spline_mapping=spline_mapping)
