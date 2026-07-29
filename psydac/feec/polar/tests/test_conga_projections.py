import numpy as np
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
    C1PolarProjection_V0,
    C1PolarProjection_V1,
    C1PolarProjection_V2,
)
from psydac.fem.basic import FemField
from psydac.linalg.block import BlockVector
from psydac.utilities.gather_variable_len_arrays import (
    gather_vlen_array,
    gather_vlen_arrays,
)

ATOL = 1e-12
RTOL = 1e-12


def get_domain(R):
    # Build physical domain - disk of radius R
    logical_domain = Square("Omega", bounds1=[0, R], bounds2=[0, 2 * pi])
    mapping = PolarMapping("F", c1=0, c2=0, rmin=0, rmax=1)
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


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("hbc", [True, False])
@pytest.mark.parametrize("degree", [[2, 2], [2, 3]])
@pytest.mark.parametrize("ncells", [[8, 10], [15, 12]])
@pytest.mark.parametrize("R", [1])
@pytest.mark.parametrize("Projector", [C0PolarProjection_V0, C1PolarProjection_V0])
@pytest.mark.mpi
def test_PolarProjection_V0(
    Projector, R, ncells, degree, hbc, transposed, verbose=False
):
    mpi_comm = MPI.COMM_WORLD
    domain = get_domain(R)

    # Discrete physical domain and discrete space
    domain_h = discretize(domain, ncells=ncells, periodic=[False, True], comm=mpi_comm)
    V0 = ScalarFunctionSpace("V0", domain)
    V0_h = discretize(V0, domain_h, degree=degree)

    P0 = Projector(V0_h, hbc=hbc, transposed=transposed)

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

    [n1, n2] = V0_h.coeff_space.npts
    assert sp_P0.shape == (n1 * n2, n1 * n2)

    # gather sparse matrix entries (rows, columns, data) on root process
    data = gather_vlen_array(sp_P0.data, mpi_comm)
    cols_rows = gather_vlen_arrays((sp_P0.col, sp_P0.row), mpi_comm)

    if mpi_comm.rank == 0:

        cols = cols_rows[0]
        rows = cols_rows[1]

        # Check the number of non-zero entries in the sparse matrix
        if Projector == C0PolarProjection_V0:
            assert len(data) == n2 * n2 + n2 * (n1 - (2 if hbc else 1))
        else:
            assert len(data) == 3 * n2 * n2 + n2 * (n1 - (3 if hbc else 2))

        if transposed:
            rows, cols = cols, rows

        # Check all non-zero entries
        for i, j, v in zip(rows, cols, data):
            if i < n2 and j < n2:
                assert np.isclose(v, 1 / n2, atol=ATOL, rtol=RTOL)
            elif Projector == C1PolarProjection_V0 and j < n2 <= i < 2 * n2:
                assert np.isclose(v, 1 / n2, atol=ATOL, rtol=RTOL)
            elif (
                Projector == C1PolarProjection_V0
                and n2 <= i < 2 * n2
                and n2 <= j < 2 * n2
            ):
                assert np.isclose(
                    v, 2 / n2 * np.cos((i - j) * 2 * np.pi / n2), atol=ATOL, rtol=RTOL
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


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("hbc", [True, False])
@pytest.mark.parametrize("degree", [[2, 2], [2, 3]])
@pytest.mark.parametrize("ncells", [[8, 10], [15, 12]])
@pytest.mark.parametrize("R", [1])
@pytest.mark.parametrize("Projector", [C0PolarProjection_V1, C1PolarProjection_V1])
@pytest.mark.mpi
def test_PolarProjection_V1(
    Projector, R, ncells, degree, hbc, transposed, verbose=False
):
    mpi_comm = MPI.COMM_WORLD
    domain = get_domain(R)

    # Discrete physical domain and discrete space
    domain_h = discretize(domain, ncells=ncells, periodic=[False, True], comm=mpi_comm)
    V1 = VectorFunctionSpace("V1", domain, kind="hcurl")
    V1_h = discretize(V1, domain_h, degree=degree)

    P1 = Projector(V1_h, hbc=hbc)
    if transposed:
        P1 = P1.T

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

    [n01, n02] = V1_h.coeff_space[0].npts
    [n11, n12] = V1_h.coeff_space[1].npts

    assert sp_P1.shape == (n01 * n02 + n11 * n12, n01 * n02 + n11 * n12)

    # gather sparse matrix entries (rows, columns, data) on root process
    data = gather_vlen_array(sp_P1.data, mpi_comm)
    cols_rows = gather_vlen_arrays((sp_P1.col, sp_P1.row), mpi_comm)

    if mpi_comm.rank == 0:

        cols = cols_rows[0]
        rows = cols_rows[1]

        # Check the number of non-zero entries in the sparse matrix
        if Projector == C0PolarProjection_V1:
            assert len(data) == n01 * n02 + 2 * n02 + (n11 - (3 if hbc else 2)) * n12
        else:
            assert (
                len(data)
                == 3 * n02 * n02 + n02 * (n01 - 1) + (n11 - (3 if hbc else 2)) * n12
            )

        if transposed:
            rows, cols = cols, rows

        # Check all non-zero entries
        for i, j, v in zip(rows, cols, data):
            if Projector == C0PolarProjection_V1:
                # Block P1_00
                if i < n01 * n02:
                    # identity
                    assert i == j
                    assert np.isclose(v, 1.0, atol=ATOL, rtol=RTOL)
                # Block P1_10
                elif n01 * n02 + n02 <= i < n01 * n02 + 2 * n02:
                    # d matrix
                    if j == i - n02 * (n01 + 1):
                        assert np.isclose(v, -1.0, atol=ATOL, rtol=RTOL)
                    else:
                        assert np.isclose(v, 1.0, atol=ATOL, rtol=RTOL)
                        assert j == (i - n02 * (n01 + 1) + 1) % n02
                # Block P1_11
                else:
                    assert i >= n01 * n02 + 2 * n02
                    assert j == i
                    assert np.isclose(v, 1.0, atol=ATOL, rtol=RTOL)

            if Projector == C1PolarProjection_V1:
                p_ij = 2 / n02 * np.cos((i - j) * 2 * np.pi / n02)
                # Block P1_00
                if i < n02:
                    # matrix p
                    assert j < n02
                    assert np.isclose(v, p_ij, atol=ATOL, rtol=RTOL)
                elif 2 * n02 > i >= n02 > j:
                    # matrix I - p
                    if i == j + n02:
                        assert np.isclose(v, 1 - p_ij, atol=ATOL, rtol=RTOL)
                    else:
                        assert np.isclose(v, -p_ij, atol=ATOL, rtol=RTOL)
                elif n02 <= i < n01 * n02:
                    assert j == i
                    assert np.isclose(v, 1.0, atol=ATOL, rtol=RTOL)

                # Block P1_10
                elif j < n02:
                    # matrix q
                    q_ij = 2 / n02 * np.cos((i + 1 - j) * 2 * np.pi / n02) - p_ij
                    assert np.isclose(v, q_ij, atol=ATOL, rtol=RTOL)

                # Block P1_11
                else:
                    assert i == j
                    assert i >= n01 * n02 + 2 * n02
                    assert np.isclose(v, 1.0, atol=ATOL, rtol=RTOL)

    # Comparing results of dot and tosparse
    x_global = mpi_comm.allreduce(x.toarray(), op=MPI.SUM)
    y_sp = sp_P1 @ x_global
    y = mpi_comm.allreduce(y.toarray(), op=MPI.SUM)
    y_sp = mpi_comm.allreduce(y_sp, op=MPI.SUM)

    assert np.allclose(y_sp, y, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("transposed", [True, False])
@pytest.mark.parametrize("degree", [[2, 2], [2, 3]])
@pytest.mark.parametrize("ncells", [[8, 10], [15, 12]])
@pytest.mark.parametrize("R", [1])
@pytest.mark.parametrize("Projector", [C0PolarProjection_V2, C1PolarProjection_V2])
@pytest.mark.mpi
def test_PolarProjection_V2(
        Projector, R, ncells, degree, transposed, verbose=False
):
    mpi_comm = MPI.COMM_WORLD
    domain = get_domain(R)

    # Discrete physical domain and discrete space
    domain_h = discretize(domain, ncells=ncells, periodic=[False, True], comm=mpi_comm)
    V2 = ScalarFunctionSpace("V2", domain)
    V2_h = discretize(V2, domain_h, degree=degree)

    P2 = Projector(V2_h, transposed=transposed)

    x = get_random_vector(V2_h)
    phiC = FemField(V2_h)
    y = phiC.coeffs
    P2.dot(x, out=y)

    # Checking projection property P0(P0(phi)) = P0(phi)
    assert np.allclose(P2.dot(y)[:, :], y[:, :], atol=ATOL, rtol=RTOL)

    # Comparing the global sparse matrix to reference file
    sp_P2 = P2.tosparse().tocoo()
    if verbose:
        print("Projection P2 in matrix form on rank", mpi_comm.rank)
        print(sp_P2.toarray())

    [n1, n2] = V2_h.coeff_space.npts
    assert sp_P2.shape == (n1 * n2, n1 * n2)

    data = gather_vlen_array(sp_P2.data, mpi_comm)
    cols_rows = gather_vlen_arrays((sp_P2.col, sp_P2.row), mpi_comm)

    if mpi_comm.rank == 0:

        cols = cols_rows[0]
        rows = cols_rows[1]

        # Check the number of non-zero entries in the sparse matrix
        assert len(data) == n1 * n2

        if transposed:
            rows, cols = cols, rows

        # Check all non-zero entries
        for i, j, v in zip(rows, cols, data):
            assert np.isclose(v, 1.0, atol=ATOL, rtol=RTOL)
            if j < n2:
                assert i == j + n2
            else:
                assert i == j

    # Comparing results of dot and tosparse
    x_global = mpi_comm.allreduce(x.toarray(), op=MPI.SUM)
    y_sp = sp_P2 @ x_global

    assert np.allclose(y_sp, y.toarray(), atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    test_PolarProjection_V0(
        C0PolarProjection_V0, 1, [3, 3], [1, 1], False, False, verbose=True
    )
    test_PolarProjection_V1(
        C0PolarProjection_V1, 1, [2, 3], [1, 1], False, False, verbose=True
    )
    test_PolarProjection_V2(1, [8, 10], [2, 2], False, verbose=True)
