#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#

import numpy as np
from psydac.feec.multipatch_domain_utilities import TransposedPolarMapping
import pytest
from mpi4py import MPI
from numpy import pi
from sympde import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import Square
from sympde.topology.analytical_mapping import PolarMapping

from psydac.api.discretization import discretize
from psydac.feec.polar.conga_projections import (
    C0PolarProjection_V0,
    C0PolarProjection_V1,
    C0PolarProjection_V2,
    C1PolarProjection_U0,
    C1PolarProjection_U1,
    C1PolarProjection_U2,
    transpose_flat_index,
)
from psydac.fem.basic import FemField
from psydac.linalg.block import BlockVector, BlockVectorSpace
from psydac.utilities.parallel_utils import (
    gather_vlen_array,
    gather_vlen_arrays,
)

ATOL = 1e-12
RTOL = 1e-12


def get_domain(R, radial_dim=0):
    if radial_dim == 0:
        # Build physical domain - disk of radius R
        logical_domain = Square("Omega", bounds1=[0, R], bounds2=[0, 2 * pi])
        mapping = PolarMapping("F", c1=0, c2=0, rmin=0, rmax=1)
        domain = mapping(logical_domain)
    else:
        # Logical domain with swapped coordinates (theta, s). 
        logical_domain = Square("Omega", bounds1=[0, 2 * pi], bounds2=[0, R])
        mapping = TransposedPolarMapping("F", c1=0, c2=0, rmin=0, rmax=1)
        domain = mapping(logical_domain)

    return domain


def get_random_vector(space):
    [s1, s2] = space.coeff_space.starts
    [e1, e2] = space.coeff_space.ends

    phi = FemField(space)
    x = phi.coeffs
    x[s1 : e1 + 1, s2 : e2 + 1] = np.random.random([e1 - s1 + 1, e2 - s2 + 1])
    x.update_ghost_regions()
    return x


def get_random_block_vector(space):
    x = BlockVector(space.coeff_space)
    for i in (0, 1):
        [s1, s2] = space.coeff_space[i].starts
        [e1, e2] = space.coeff_space[i].ends
        x[i][s1 : e1 + 1, s2 : e2 + 1] = np.random.random([e1 - s1 + 1, e2 - s2 + 1])
    return x

def to_s_theta_index(k, coeff_space, radial_dim):
    """
    Map global flat indices `k` of 1-form coefficients to the (s, theta)
    ordering, where the s-component comes first and each component is
    flattened in (radial, angular) order.
    """
    if radial_dim == 0:
        return k

    # Number the coefficients in storage order (theta-component first, both
    # components with shape (angular, radial)), then list them in (s, theta) order
    theta_shape, s_shape = (V.npts for V in coeff_space.spaces)
    n_theta = np.prod(theta_shape)
    old_theta = np.arange(n_theta).reshape(theta_shape)
    old_s = n_theta + np.arange(np.prod(s_shape)).reshape(s_shape)
    old_index = np.concatenate([old_s.T.ravel(), old_theta.T.ravel()])

    new_index = np.empty_like(old_index)
    new_index[old_index] = np.arange(old_index.size)
    return new_index[k]

@pytest.mark.parametrize("radial_dim", [0, 1])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("hbc", [True, False])
@pytest.mark.parametrize("degree", [[2, 2], [2, 3]])
@pytest.mark.parametrize("ncells", [[8, 10], [15, 12]])
@pytest.mark.parametrize("R", [1])
@pytest.mark.parametrize("Projector", [C0PolarProjection_V0, C1PolarProjection_U0])
@pytest.mark.mpi
def test_PolarProjection_V0(
    Projector, R, ncells, degree, hbc, transposed, radial_dim, verbose=False
    ):
    mpi_comm = MPI.COMM_WORLD
    domain = get_domain(R, radial_dim)

    # ncells and degree are given in (radial, angular) order
    if radial_dim == 0:
        periodic = [False, True]
    else:
        ncells, degree, periodic = ncells[::-1], degree[::-1], [True, False]

    # Discrete physical domain and discrete space
    domain_h = discretize(domain, ncells=ncells, periodic=periodic, comm=mpi_comm)
    V0 = ScalarFunctionSpace("V0", domain)
    V0_h = discretize(V0, domain_h, degree=degree)

    P0 = Projector(V0_h, hbc=hbc, transposed=transposed, radial_dim=radial_dim)

    x = get_random_vector(V0_h)
    phiC = FemField(V0_h)
    y = phiC.coeffs
    P0.dot(x, out=y)

    # Checking projection property P0(P0(phi)) = P0(phi)
    assert np.allclose(P0.dot(y)[:, :], y[:, :], atol=ATOL, rtol=RTOL)

    sp_P0 = P0.tosparse().tocoo()
    if verbose:
        print("Projection P0 in matrix form on rank", mpi_comm.rank)
        print(sp_P0.toarray())

    nr = V0_h.coeff_space.npts[radial_dim]
    na = V0_h.coeff_space.npts[1 - radial_dim]
    assert sp_P0.shape == (nr * na, nr * na)

    # gather sparse matrix entries (rows, columns, data) on root process
    data = gather_vlen_array(sp_P0.data, mpi_comm)
    cols_rows = gather_vlen_arrays((sp_P0.col, sp_P0.row), mpi_comm)

    if mpi_comm.rank == 0:

        cols = cols_rows[0]
        rows = cols_rows[1]

        # Express rows and columns as (radial, angular) flat indices i*na + j,
        # so that the checks below are the same for both orderings
        if radial_dim == 1:
            rows = transpose_flat_index(rows, na, nr)
            cols = transpose_flat_index(cols, na, nr)

        # Check the number of non-zero entries in the sparse matrix
        if Projector == C0PolarProjection_V0:
            assert len(data) == na * na + na * (nr - (2 if hbc else 1))
        else:
            assert len(data) == 3 * na * na + na * (nr - (3 if hbc else 2))

        if transposed:
            rows, cols = cols, rows

        # Check all non-zero entries
        for i, j, v in zip(rows, cols, data):
            if i < na and j < na:
                assert np.isclose(v, 1 / na, atol=ATOL, rtol=RTOL)
            elif Projector == C1PolarProjection_U0 and j < na <= i < 2 * na:
                assert np.isclose(v, 1 / na, atol=ATOL, rtol=RTOL)
            elif (
                    Projector == C1PolarProjection_U0
                    and na <= i < 2 * na
                    and na <= j < 2 * na
            ):
                assert np.isclose(
                    v, 2 / na * np.cos((i - j) * 2 * np.pi / na), atol=ATOL, rtol=RTOL
                )
            else:
                # identity
                assert i == j
                assert np.isclose(v, 1.0, atol=ATOL, rtol=RTOL)

    # Comparing results of dot and tosparse
    x_global = mpi_comm.allreduce(x.toarray(), op=MPI.SUM)
    y_sp = sp_P0 @ x_global
    y = mpi_comm.allreduce(y.toarray(), op=MPI.SUM)
    y_sp = mpi_comm.allreduce(y_sp, op=MPI.SUM)

    assert np.allclose(y_sp, y, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("radial_dim", [0, 1])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("hbc", [True, False])
@pytest.mark.parametrize("degree", [[2, 2], [2, 3]])
@pytest.mark.parametrize("ncells", [[8, 10], [15, 12]])
@pytest.mark.parametrize("R", [1])
@pytest.mark.parametrize("Projector", [C0PolarProjection_V1])
@pytest.mark.mpi
def test_PolarProjection_V1(
    Projector, R, ncells, degree, hbc, transposed, radial_dim, verbose=False
):
    mpi_comm = MPI.COMM_WORLD
    domain = get_domain(R, radial_dim)

    # ncells and degree are given in (radial, angular) order
    if radial_dim == 0:
        periodic = [False, True]
    else:
        ncells, degree, periodic = ncells[::-1], degree[::-1], [True, False]

    # Discrete physical domain and discrete space
    domain_h = discretize(domain, ncells=ncells, periodic=periodic, comm=mpi_comm)
    V1 = VectorFunctionSpace("V1", domain, kind="hcurl")
    V1_h = discretize(V1, domain_h, degree=degree)

    P1 = Projector(V1_h, hbc=hbc, transposed=transposed, radial_dim=radial_dim)

    x = get_random_block_vector(V1_h)
    y = BlockVector(V1_h.coeff_space)
    P1.dot(x, out=y)
    z = P1.dot(y)
    # Checking projection property P1(P1(phi)) = P1(phi)
    assert np.allclose(z[0][:, :], y[0][:, :], atol=ATOL, rtol=RTOL)
    assert np.allclose(z[1][:, :], y[1][:, :], atol=ATOL, rtol=RTOL)

    sp_P1 = P1.tosparse().tocoo()
    if verbose:
        print("Projection P1 in matrix form on rank", mpi_comm.rank)
        print(sp_P1.toarray())

    # Number of radial basis functions of the s- and theta-components, and
    # number of angular basis functions (the same for both components)
    V_s = V1_h.coeff_space[radial_dim]
    V_theta = V1_h.coeff_space[1 - radial_dim]
    nr_s, na = V_s.npts[radial_dim], V_s.npts[1 - radial_dim]
    nr_theta = V_theta.npts[radial_dim]

    assert sp_P1.shape == (nr_s * na + nr_theta * na, nr_s * na + nr_theta * na)

    # gather sparse matrix entries (rows, columns, data) on root process
    data = gather_vlen_array(sp_P1.data, mpi_comm)
    cols_rows = gather_vlen_arrays((sp_P1.col, sp_P1.row), mpi_comm)

    if mpi_comm.rank == 0:

        # Express rows and columns as indices in the (s, theta) ordering,
        # so that the checks below are the same for both orderings
        cols = to_s_theta_index(cols_rows[0], V1_h.coeff_space, radial_dim)
        rows = to_s_theta_index(cols_rows[1], V1_h.coeff_space, radial_dim)

        # Check the number of non-zero entries in the sparse matrix
        if Projector == C0PolarProjection_V1:
            assert len(data) == nr_s * na + 2 * na + (nr_theta - (3 if hbc else 2)) * na
        else:
            assert (
                len(data)
                == 3 * na * na + na * (nr_s - 1) + (nr_theta - (3 if hbc else 2)) * na
            )

        if transposed:
            rows, cols = cols, rows

        # Check all non-zero entries
        for i, j, v in zip(rows, cols, data):
            if Projector == C0PolarProjection_V1:
                # Block s_s
                if i < nr_s * na:
                    # identity
                    assert i == j
                    assert np.isclose(v, 1.0, atol=ATOL, rtol=RTOL)
                # Block theta_s
                elif nr_s * na + na <= i < nr_s * na + 2 * na:
                    # d matrix
                    if j == i - na * (nr_s + 1):
                        assert np.isclose(v, -1.0, atol=ATOL, rtol=RTOL)
                    else:
                        assert np.isclose(v, 1.0, atol=ATOL, rtol=RTOL)
                        assert j == (i - na * (nr_s + 1) + 1) % na
                # Block theta_theta
                else:
                    assert i >= nr_s * na + 2 * na
                    assert j == i
                    assert np.isclose(v, 1.0, atol=ATOL, rtol=RTOL)

            if Projector == C1PolarProjection_U1:
                p_ij = 2 / na * np.cos((i - j) * 2 * np.pi / na)
                # Block s_s
                if i < na:
                    # matrix p
                    assert j < na
                    assert np.isclose(v, p_ij, atol=ATOL, rtol=RTOL)
                elif 2 * na > i >= na > j:
                    # matrix I - p
                    if i == j + na:
                        assert np.isclose(v, 1 - p_ij, atol=ATOL, rtol=RTOL)
                    else:
                        assert np.isclose(v, -p_ij, atol=ATOL, rtol=RTOL)
                elif na <= i < nr_s * na:
                    assert j == i
                    assert np.isclose(v, 1.0, atol=ATOL, rtol=RTOL)

                # Block theta_s
                elif j < na:
                    # matrix q
                    q_ij = 2 / na * np.cos((i + 1 - j) * 2 * np.pi / na) - p_ij
                    assert np.isclose(v, q_ij, atol=ATOL, rtol=RTOL)

                # Block theta_theta
                else:
                    assert i == j
                    assert i >= nr_s * na + 2 * na
                    assert np.isclose(v, 1.0, atol=ATOL, rtol=RTOL)

    # Comparing results of dot and tosparse
    x_global = mpi_comm.allreduce(x.toarray(), op=MPI.SUM)
    y_sp = sp_P1 @ x_global
    y = mpi_comm.allreduce(y.toarray(), op=MPI.SUM)
    y_sp = mpi_comm.allreduce(y_sp, op=MPI.SUM)

    assert np.allclose(y_sp, y, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("radial_dim", [0, 1])
@pytest.mark.parametrize("transposed", [True, False])
@pytest.mark.parametrize("degree", [[2, 2], [2, 3]])
@pytest.mark.parametrize("ncells", [[8, 10], [15, 12]])
@pytest.mark.parametrize("R", [1])
@pytest.mark.parametrize("Projector", [C0PolarProjection_V2, C1PolarProjection_U2])
@pytest.mark.mpi
def test_PolarProjection_V2(
        Projector, R, ncells, degree, transposed, radial_dim, verbose=False
):
    mpi_comm = MPI.COMM_WORLD
    domain = get_domain(R, radial_dim)

    # ncells and degree are given in (radial, angular) order
    if radial_dim == 0:
        periodic = [False, True]
    else:
        ncells, degree, periodic = ncells[::-1], degree[::-1], [True, False]

    # Discrete physical domain and discrete space
    domain_h = discretize(domain, ncells=ncells, periodic=periodic, comm=mpi_comm)
    V2 = ScalarFunctionSpace("V2", domain)
    V2_h = discretize(V2, domain_h, degree=degree)

    P2 = Projector(V2_h, transposed=transposed, radial_dim=radial_dim)
    # Checking that the transpose keeps the projector type and coordinate ordering
    P2_T = P2.T
    assert type(P2_T) is Projector
    assert P2_T.transposed == (not transposed)
    assert P2_T.radial_dim == radial_dim

    x = get_random_vector(V2_h)
    phiC = FemField(V2_h)
    y = phiC.coeffs
    P2.dot(x, out=y)

    # Checking projection property P2(P2(phi)) = P2(phi)
    assert np.allclose(P2.dot(y)[:, :], y[:, :], atol=ATOL, rtol=RTOL)

    sp_P2 = P2.tosparse().tocoo()
    if verbose:
        print("Projection P2 in matrix form on rank", mpi_comm.rank)
        print(sp_P2.toarray())

    nr = V2_h.coeff_space.npts[radial_dim]
    na = V2_h.coeff_space.npts[1 - radial_dim]
    assert sp_P2.shape == (nr * na, nr * na)

    # gather sparse matrix entries (rows, columns, data) on root process
    data = gather_vlen_array(sp_P2.data, mpi_comm)
    cols_rows = gather_vlen_arrays((sp_P2.col, sp_P2.row), mpi_comm)

    if mpi_comm.rank == 0:

        cols = cols_rows[0]
        rows = cols_rows[1]

        # Express rows and columns as (radial, angular) flat indices i*na + j,
        # so that the checks below are the same for both orderings
        if radial_dim == 1:
            rows = transpose_flat_index(rows, na, nr)
            cols = transpose_flat_index(cols, na, nr)

        # Check the number of non-zero entries in the sparse matrix
        assert len(data) == nr * na

        if transposed:
            rows, cols = cols, rows

        # Check all non-zero entries
        for i, j, v in zip(rows, cols, data):
            assert np.isclose(v, 1.0, atol=ATOL, rtol=RTOL)
            if j < na:
                assert i == j + na
            else:
                assert i == j

    # Comparing results of dot and tosparse
    x_global = mpi_comm.allreduce(x.toarray(), op=MPI.SUM)
    y_sp = sp_P2 @ x_global

    assert np.allclose(y_sp, y.toarray(), atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    test_PolarProjection_V0(
        C0PolarProjection_V0, 1, [3, 3], [1, 1], False, False, 1, verbose=True
    )
    test_PolarProjection_V1(
        C0PolarProjection_V1, 1, [2, 3], [1, 1], False, False, verbose=True
    )
    test_PolarProjection_V2(1, [8, 10], [2, 2], False, 1, verbose=True)
