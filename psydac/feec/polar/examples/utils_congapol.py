import numpy as np

from psydac.ddm.cart import DomainDecomposition
from psydac.fem.splines import SplineSpace
from psydac.fem.tensor import TensorFemSpace


def print_map_polar_coeffs(map_discrete):
    """
    Used for debugging purposes. Prints information about discrete polar coefficients
    """
    # Print spline mapping
    print('Spline mapping:')
    print(map_discrete)
    print('vars(map_discrete):')
    print(vars(map_discrete))
    print()
    print('vars(map_discrete._fields[0]) :')
    print(vars(map_discrete._fields[0]))

    print()
    print('vars(map_discrete._fields[0]._space._spaces[0]) :')
    print(vars(map_discrete._fields[0]._space._spaces[0]))
    print('vars(map_discrete._fields[0]._space._spaces[1]) :')
    print(vars(map_discrete._fields[0]._space._spaces[1]))
    n_s = map_discrete._fields[0]._space._spaces[0]._nbasis
    n_theta = map_discrete._fields[0]._space._spaces[1]._nbasis
    print('n_s = ', n_s)
    print('n_theta = ', n_theta)
    print()

    map_0_c = map_discrete._fields[0]._coeffs.toarray()
    map_1_c = map_discrete._fields[1]._coeffs.toarray()

    print('pole: ', map_0_c[0], map_1_c[0])
    map_0_c -= map_0_c[0]
    map_1_c -= map_1_c[0]

    # print(map_0_c**2 + map_1_c**2)
    radius_pole = np.sqrt(map_0_c[:n_theta] ** 2 + map_1_c[:n_theta] ** 2)
    radius_first_ring = np.sqrt(
        map_0_c[n_theta:2 * n_theta] ** 2 + map_1_c[n_theta:2 * n_theta] ** 2
    )
    rho_1 = radius_first_ring[0]

    cs = map_0_c[n_theta:2 * n_theta] / rho_1
    sn = map_1_c[n_theta:2 * n_theta] / rho_1
    theta = np.arctan2(sn, cs)

    print('radius_pole:', radius_pole)
    print('radius first ring:', radius_first_ring)
    print('D theta_j:', np.mod(theta[1:] - theta[:-1], 2 * np.pi))


def check_regular_ring_map(map_discrete, verbose=False):
    """
    Check the first two radial rings of a 2D spline mapping

    Performs 3 checks:
    1. that the pole ring (i=0) collapses to a single point
    2. the 1st ring (i=1) has constant radius
    3. the 1st ring is uniformly spaced in angle
    and prints diagnostic information. Used for debugging purposes.

    """
    n_s = map_discrete._fields[0]._space._spaces[0]._nbasis
    n_theta = map_discrete._fields[0]._space._spaces[1]._nbasis
    assert n_s == map_discrete._fields[1]._space._spaces[0]._nbasis
    assert n_theta == map_discrete._fields[1]._space._spaces[1]._nbasis

    if verbose:
        print('n_s = ', n_s)
        print('n_theta = ', n_theta)
        print()

    # read mapping coefficients
    map_0_c = map_discrete._fields[0]._coeffs.toarray()
    map_1_c = map_discrete._fields[1]._coeffs.toarray()

    # shift
    map_0_c -= map_0_c[0]
    map_1_c -= map_1_c[0]

    # distance of every coefficient in the pole ring from the first pole coefficient (should be 0)
    radius_pole = np.sqrt(map_0_c[:n_theta] ** 2 + map_1_c[:n_theta] ** 2)
    # distance from the pole of the first radial ring
    radius_first_ring = np.sqrt(
        map_0_c[n_theta:2 * n_theta] ** 2 + map_1_c[n_theta:2 * n_theta] ** 2
    )
    rho_1 = radius_first_ring[0]

    # compute angles of the 1st ring
    cs = map_0_c[n_theta:2 * n_theta] / rho_1
    sn = map_1_c[n_theta:2 * n_theta] / rho_1
    theta = np.arctan2(sn, cs)

    delta_theta = np.mod(theta[1:] - theta[:-1], 2 * np.pi)
    if verbose:
        print('radius_pole:', radius_pole)
        print('radius first ring:', radius_first_ring)
        print('D theta_j:', delta_theta)
        print()

    circle_error = np.linalg.norm(radius_pole[1:] - radius_pole[:-1])
    #check that 1st ring points are uniformly spaced in angle
    regularity_error = np.linalg.norm(delta_theta[1:] - delta_theta[:-1])
    first_ring_radius_error = np.linalg.norm(radius_first_ring[1:] - radius_first_ring[:-1])
    regularity_check = circle_error + regularity_error + first_ring_radius_error < 1e-12

    print('CHECK: spline mapping is regular: ', regularity_check)
    if not regularity_check:
        print(' - circle error on pole ring:', circle_error)
        print(' - regularity error on 1st ring:', regularity_error)
        print(' - first ring radius error:', first_ring_radius_error)
    print()

def add_colorbar(im, ax, **kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.2, pad=0.3)
    cbar = ax.get_figure().colorbar(im, cax=cax, **kwargs)
    return cbar

def create_tensor_spline_space(ncells, spline_degrees, periodic, bounds, mpi_comm=None):
    """
    Create a 2D tensor-product spline finite element space on a rectangular
    logical domain (e.g. with bounds ``[[0, R], [0, 2*pi]]``).

    Returns
    -------
    TensorFemSpace
        The 2D tensor-product spline finite element space.
    """

    # Number of elements and spline degree
    ne1, ne2 = ncells
    p1, p2 = spline_degrees

    # Create uniform grid
    grid_1 = np.linspace(*bounds[0], num=ne1 + 1)
    grid_2 = np.linspace(*bounds[1], num=ne2 + 1)

    # Create 1D finite element spaces
    S1 = SplineSpace(p1, grid=grid_1, periodic=periodic[0])
    S2 = SplineSpace(p2, grid=grid_2, periodic=periodic[1])

    # Create 2D tensor product finite element space
    domain_decomposition = DomainDecomposition(ncells, periodic, comm=mpi_comm)
    V = TensorFemSpace(domain_decomposition, S1, S2)
    return V


