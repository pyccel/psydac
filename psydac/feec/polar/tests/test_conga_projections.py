from itertools import product
from pathlib import Path

import h5py
import numpy as np
from numpy import pi

import pytest
from sympde import ScalarFunctionSpace, VectorFunctionSpace

from mpi4py import MPI

from sympde.topology import Square
from sympde.topology.analytical_mapping import PolarMapping

from psydac.api.discretization import discretize
from psydac.feec.polar.conga_projections import C0PolarProjection_V0, C0PolarProjection_V2, C0PolarProjection_V1, \
    C1PolarProjection_V0, C1PolarProjection_V1
from psydac.fem.basic import FemField
from psydac.linalg.block import BlockVector


def get_domain(R):
    # Build physical domain - disk of radius R
    logical_domain = Square('Omega', bounds1=[0, R], bounds2=[0, 2 * pi])
    mapping = PolarMapping('F', c1=0, c2=0, rmin=0, rmax=1)
    domain = mapping(logical_domain)

    return domain

def get_random_vector(space):
    [s1, s2] = space.coeff_space.starts
    [e1, e2] = space.coeff_space.ends

    phi = FemField(space)
    x = phi.coeffs
    x[s1:e1 + 1, s2:e2 + 1] = np.random.random([e1 - s1 + 1, e2 - s2 + 1])
    x.update_ghost_regions()
    return x

def get_random_block_vector(space):
    x = BlockVector(space.coeff_space)
    for i in (0, 1):
        [s1, s2] = space.coeff_space[i].starts
        [e1, e2] = space.coeff_space[i].ends
        x[i][s1:e1 + 1, s2:e2 + 1] = np.random.random([e1 - s1 + 1, e2 - s2 + 1])
    return x


def create_reference_file(proj_name, projections_list):
    """
    Save global sparse matrices (after checking their correctness by hand) in h5 format for tests
    This function is needed only if the parameters for the tests are changed
    """
    f = h5py.File(proj_name + '.h5', mode='w')
    params = dict(
        transposed=[False, True],
        hbc=[True, False],
        degree=[[2, 2], [2, 3]],
        ncells=[[8, 10], [15, 12]],
        projector=projections_list,
    )

    for (tr, hbc, deg, nc, proj) in product(params["transposed"], params["hbc"], params["degree"], params["ncells"],
                                            params["projector"]):

        if proj_name == "P2" and hbc is True:
            continue
        domain = get_domain(1)
        domain_h = discretize(domain, ncells=nc, periodic=[False, True])
        if proj_name == 'P1':
            V = VectorFunctionSpace('V', domain, kind='hcurl')
        else:
            V = ScalarFunctionSpace('V', domain)
        V_h = discretize(V, domain_h, degree=deg)

        if proj_name == 'P2': P = proj(V_h)
        else: P = proj(V_h, hbc=hbc)
        if tr: P = P.T

        key = f"{proj.__name__}/n{nc[0]}x{nc[1]}/p{deg[0]}-{deg[1]}/hbc{hbc}/T{tr}"
        if key in f:
            del f[key]
        g = f.require_group(key)
        g.create_dataset("P", data=P.toarray())
        print("saved", key)

    f.close()


@pytest.mark.parametrize('transposed', [False, True])
@pytest.mark.parametrize('hbc', [True, False])
@pytest.mark.parametrize('degree', [[2, 2], [2, 3]])
@pytest.mark.parametrize('ncells', [[8, 10], [15, 12]])
@pytest.mark.parametrize('R', [1])
@pytest.mark.parametrize('Projector', [C0PolarProjection_V0, C1PolarProjection_V0])
@pytest.mark.mpi

def test_PolarProjection_V0(Projector, R, ncells, degree, hbc, transposed):
    mpi_comm = MPI.COMM_WORLD
    domain = get_domain(R)

    # Discrete physical domain and discrete space
    domain_h = discretize(domain, ncells=ncells, periodic=[False, True], comm=mpi_comm)
    V0 = ScalarFunctionSpace('V0', domain)
    V0_h = discretize(V0, domain_h, degree=degree)

    P0 = Projector(V0_h, hbc=hbc, transposed=transposed)

    x = get_random_vector(V0_h)
    phiC = FemField(V0_h)
    y = phiC.coeffs
    P0.dot(x, out=y)

    # Checking projection property P0(P0(phi)) = P0(phi)
    assert np.allclose(P0.dot(y)[:, :], y[:, :])

    # Comparing the global sparse matrix to reference file
    sp_P0 = P0.tosparse()
    sp_P0_global = mpi_comm.allreduce(sp_P0.toarray(), op=MPI.SUM)
    if mpi_comm.rank == 0:
        name = Projector.__name__
        h5_path = Path(__file__).absolute().parent / "P0.h5"

        with h5py.File(h5_path, "r") as f:
            key = f"{name}/n{ncells[0]}x{ncells[1]}/p{degree[0]}-{degree[1]}/hbc{hbc}/T{transposed}/P"
            sp_P0_ref = f[key][()]
            assert np.allclose(sp_P0_global, sp_P0_ref)

    # Comparing results of dot and tosparse
    x_global = mpi_comm.allreduce(x.toarray(), op=MPI.SUM)
    y_sp = sp_P0 @ x_global
    y = mpi_comm.allreduce(y.toarray(), op=MPI.SUM)
    y_sp = mpi_comm.allreduce(y_sp, op=MPI.SUM)

    assert np.allclose(y_sp, y)



@pytest.mark.parametrize('transposed', [False, True])
@pytest.mark.parametrize('hbc', [True, False])
@pytest.mark.parametrize('degree', [[2, 2], [2, 3]])
@pytest.mark.parametrize('ncells', [[8, 10], [15, 12]])
@pytest.mark.parametrize('R', [1])
@pytest.mark.parametrize('Projector', [C0PolarProjection_V1, C1PolarProjection_V1])
@pytest.mark.mpi

def test_PolarProjection_V1(Projector, R, ncells, degree, hbc, transposed):
    mpi_comm = MPI.COMM_WORLD
    domain = get_domain(R)

    # Discrete physical domain and discrete space
    domain_h = discretize(domain, ncells=ncells, periodic=[False, True], comm=mpi_comm)
    V1 = VectorFunctionSpace('V1', domain, kind='hcurl')
    V1_h = discretize(V1, domain_h, degree=degree)

    P1 = Projector(V1_h, hbc=hbc)
    if transposed:
        P1 = P1.T

    x = get_random_block_vector(V1_h)
    y = BlockVector(V1_h.coeff_space)
    P1.dot(x, out=y)
    z = P1.dot(y)
    # Checking projection property P1(P1(phi)) = P1(phi)
    assert np.allclose(z[0][:, :], y[0][:, :])
    assert np.allclose(z[1][:, :], y[1][:, :])

    # Comparing the global sparse matrix to reference file
    sp_P1 = P1.tosparse()
    sp_P1_global = mpi_comm.allreduce(sp_P1.toarray(), op=MPI.SUM)
    if mpi_comm.rank == 0:
        name = Projector.__name__
        h5_path = (Path(__file__).absolute().parent / "P1.h5")

        with h5py.File(h5_path, "r") as f:
            key = f"{name}/n{ncells[0]}x{ncells[1]}/p{degree[0]}-{degree[1]}/hbc{hbc}/T{transposed}/P"
            sp_P1_ref = f[key][()]
            assert np.allclose(sp_P1_global, sp_P1_ref)

    # Comparing results of dot and tosparse
    x_global = mpi_comm.allreduce(x.toarray(), op=MPI.SUM)
    y_sp = sp_P1 @ x_global
    y = mpi_comm.allreduce(y.toarray(), op=MPI.SUM)
    y_sp = mpi_comm.allreduce(y_sp, op=MPI.SUM)

    assert np.allclose(y_sp, y)


@pytest.mark.parametrize( 'transposed', [True, False])
@pytest.mark.parametrize('degree', [[2, 2], [2, 3]])
@pytest.mark.parametrize('ncells', [[8, 10], [15, 12]])
@pytest.mark.parametrize( 'R', [1])
@pytest.mark.mpi

def test_PolarProjection_V2(R, ncells, degree, transposed):
    mpi_comm = MPI.COMM_WORLD
    domain = get_domain(R)

    # Discrete physical domain and discrete space
    domain_h = discretize(domain, ncells=ncells, periodic=[False, True], comm=mpi_comm)
    V2 = ScalarFunctionSpace('V2', domain)
    V2_h = discretize(V2, domain_h, degree=degree)

    P2 = C0PolarProjection_V2(V2_h, transposed=transposed)

    x = get_random_vector(V2_h)
    phiC = FemField(V2_h)
    y = phiC.coeffs
    P2.dot(x, out=y)

    # Checking projection property P0(P0(phi)) = P0(phi)
    assert np.allclose(P2.dot(y)[:, :], y[:, :])

    # Comparing the global sparse matrix to reference file
    sp_P2 = P2.tosparse()
    sp_P2_global = mpi_comm.allreduce(sp_P2.toarray(), op=MPI.SUM)
    if mpi_comm.rank == 0:

        name = 'C0PolarProjection_V2'
        h5_path = (Path(__file__).absolute().parent / "P2.h5")

        with h5py.File(h5_path, "r") as f:
            key = f"{name}/n{ncells[0]}x{ncells[1]}/p{degree[0]}-{degree[1]}/hbc{False}/T{transposed}/P"
            sp_P1_ref = f[key][()]
            assert np.allclose(sp_P2_global, sp_P1_ref)

    # Comparing results of dot and tosparse
    x_global = mpi_comm.allreduce(x.toarray(), op=MPI.SUM)
    y_sp = sp_P2 @ x_global

    assert np.allclose(y_sp, y.toarray())

if __name__ == '__main__':
    create_reference_file("P0", [C0PolarProjection_V0, C1PolarProjection_V0])
    create_reference_file("P1", [C0PolarProjection_V1, C1PolarProjection_V1])
    create_reference_file("P2", [C0PolarProjection_V2])
