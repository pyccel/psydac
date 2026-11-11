#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
r"""
Utility functions for 2D examples of broken-FEEC discretization on polar domains.
"""

import numpy as np

from psydac.ddm.cart import DomainDecomposition
from psydac.fem.splines import SplineSpace
from psydac.fem.tensor import TensorFemSpace

__all__ = (
    "add_colorbar",
    "check_regular_ring_map",
    "create_tensor_spline_space",
)


def check_regular_ring_map(map_discrete, radial_dim=0, verbose=False):
    r"""
    Check the first two radial rings of a 2D spline mapping

    Performs 3 checks:
    1. that the pole ring (i=0) collapses to a single point
    2. the 1st ring (i=1) has constant radius
    3. the 1st ring is uniformly spaced in angle
    and prints diagnostic information. Used for debugging purposes.

    Parameters
    ----------
    map_discrete : psydac.mapping.discrete.SplineMapping
        Discrete spline mapping whose coefficients are inspected.

    radial_dim : int, default=0
    angle_dim = 1 - radial_dim
    spaces = map_discrete.space.spaces  # shared by all components of the mapping
    n_s = spaces[radial_dim].nbasis
    n_theta = spaces[angle_dim].nbasis

    if verbose:
        print("n_s = ", n_s)
        print("n_theta = ", n_theta)
        print()

    # read mapping coefficients, flattened with the radial index first
    map_0_c = map_discrete.fields[0].coeffs.toarray()
    map_1_c = map_discrete.fields[1].coeffs.toarray()
    if radial_dim == 1:
        map_0_c = map_0_c.reshape(n_theta, n_s).T.ravel()
        map_1_c = map_1_c.reshape(n_theta, n_s).T.ravel()

    if verbose:
        print("n_s = ", n_s)
        print("n_theta = ", n_theta)
        print("pole position:", map_0_c[0], map_1_c[0])
        print()

    # shift
    map_0_c -= map_0_c[0]
    map_1_c -= map_1_c[0]

    # distance of every coefficient in the pole ring from the first pole coefficient (should be 0)
    radius_pole = np.hypot(map_0_c[:n_theta], map_1_c[:n_theta])
    # distance from the pole of the first radial ring
    radius_first_ring = np.hypot(
        map_0_c[n_theta : 2 * n_theta], map_1_c[n_theta : 2 * n_theta]
    )

    # compute angles of the 1st ring
    cs = map_0_c[n_theta : 2 * n_theta]
    sn = map_1_c[n_theta : 2 * n_theta]
    theta = np.arctan2(sn, cs)

    delta_theta = np.mod(theta[1:] - theta[:-1], 2 * np.pi)
    if verbose:
        print("radius_pole:", radius_pole)
        print("radius first ring:", radius_first_ring)
        print("D theta_j:", delta_theta)
        print()

    circle_error = np.linalg.norm(radius_pole[1:] - radius_pole[:-1])
    # check that 1st ring points are uniformly spaced in angle
    regularity_error = np.linalg.norm(delta_theta[1:] - delta_theta[:-1])
    first_ring_radius_error = np.linalg.norm(
        radius_first_ring[1:] - radius_first_ring[:-1]
    )
    regularity_check = circle_error + regularity_error + first_ring_radius_error < 1e-12

    print("CHECK: spline mapping is regular: ", regularity_check)
    if not regularity_check:
        print(" - circle error on pole ring:", circle_error)
        print(" - regularity error on 1st ring:", regularity_error)
        print(" - first ring radius error:", first_ring_radius_error)
    print()


def add_colorbar(im, ax, **kwargs):
    """
    Add a colorbar to the right of an axes.

    Parameters
    ----------
    im : matplotlib.cm.ScalarMappable
        Mappable object associated with the plot
    ax : matplotlib.axes.Axes
        Axes to which the colorbar is attached.
    **kwargs
        Additional keyword arguments passed to `Figure.colorbar`.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The created colorbar.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.2, pad=0.3)
    cbar = ax.get_figure().colorbar(im, cax=cax, **kwargs)
    return cbar


def create_tensor_spline_space(ncells, spline_degrees, periodic, bounds, mpi_comm=None):
    r"""
    Create a 2D tensor-product spline finite element space on a rectangular
    logical domain (e.g. with bounds `[[0, R], [0, 2*pi]]`).

    Parameters
    ----------
    ncells : sequence of int
        Number of cells in 2D spline space in each direction.
    spline_degrees: sequence of int
        Degree of the spline space in each direction.
    periodic : sequence of bool
        Periodicity of spline space in each direction.
    bounds : sequence of sequence of float
        Lower and upper bounds of the logical domain in each direction.
    mpi_comm : mpi4py.MPI.Comm, optional
        MPI communicator used for the domain decomposition.

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
