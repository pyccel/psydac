import numpy as np
from numpy import pi

import pytest
from sympde import ScalarFunctionSpace, VectorFunctionSpace

from mpi4py import MPI

from sympde.topology import Square
from sympde.topology.analytical_mapping import PolarMapping

from psydac.api.discretization import discretize
from psydac.feec.polar.conga_projections import C0PolarProjection_V0, C0PolarProjection_V2, C0PolarProjection_V1
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


@pytest.mark.parametrize( 'R', [1])
@pytest.mark.parametrize( 'ncells', [[4, 8], [12, 12]])
@pytest.mark.parametrize( 'degree', [[1, 1], [2, 2]])
@pytest.mark.mpi

def test_C0PolarProjection_V0(R, ncells, degree):
    mpi_comm = MPI.COMM_WORLD
    domain = get_domain(R)

    # Discrete physical domain and discrete space
    domain_h = discretize(domain, ncells=ncells, periodic=[False, True], comm=mpi_comm)
    V0 = ScalarFunctionSpace('V0', domain)
    V0_h = discretize(V0, domain_h, degree=degree)

    P0 = C0PolarProjection_V0(V0_h, hbc=True)

    x = get_random_vector(V0_h)
    phiC = FemField(V0_h)
    y = phiC.coeffs
    P0.dot(x, out=y)

    # Checking projection property P0(P0(phi)) = P0(phi)
    assert np.allclose(P0.dot(y)[:, :], y[:, :])

    # Comparing results of dot and tosparse
    sp_P0 = P0.tosparse()
    y_sp = sp_P0 @ x.toarray()
    y = mpi_comm.allreduce(y.toarray(), op=MPI.SUM)
    y_sp = mpi_comm.allreduce(y_sp, op=MPI.SUM)

    assert np.allclose(y_sp, y)


@pytest.mark.parametrize( 'R', [1])
@pytest.mark.parametrize( 'ncells', [[4, 8], [12, 12]])
@pytest.mark.parametrize( 'degree', [[1, 1], [2, 2]])
@pytest.mark.mpi

def test_C0PolarProjection_V1(R, ncells, degree):
    mpi_comm = MPI.COMM_WORLD
    domain = get_domain(R)

    # Discrete physical domain and discrete space
    domain_h = discretize(domain, ncells=ncells, periodic=[False, True], comm=mpi_comm)
    V1 = VectorFunctionSpace('V1', domain, kind='hcurl')
    V1_h = discretize(V1, domain_h, degree=degree)

    P1 = C0PolarProjection_V1(V1_h, hbc=True)

    x = get_random_block_vector(V1_h)
    print(x)
    y = BlockVector(V1_h.coeff_space)
    P1.dot(x, out=y)

    # Checking projection property P1(P1(phi)) = P1(phi)
    assert np.allclose(P1.dot(y)[0][:, :], y[0][:, :])
    assert np.allclose(P1.dot(y)[1][:, :], y[1][:, :])

    # Comparing results of dot and tosparse
    sp_P1 = P1.tosparse()
    y_sp = sp_P1 @ x.toarray()

    assert np.allclose(y_sp, y.toarray())


@pytest.mark.parametrize( 'R', [1])
@pytest.mark.parametrize( 'ncells', [[4, 8], [12, 12]])
@pytest.mark.parametrize( 'degree', [[1, 1], [2, 2]])
@pytest.mark.mpi

def test_C0PolarProjection_V2(R, ncells, degree):
    mpi_comm = MPI.COMM_WORLD
    domain = get_domain(R)

    # Discrete physical domain and discrete space
    domain_h = discretize(domain, ncells=ncells, periodic=[False, True], comm=mpi_comm)
    V2 = ScalarFunctionSpace('V2', domain)
    V2_h = discretize(V2, domain_h, degree=degree)

    P2 = C0PolarProjection_V2(V2_h)

    x = get_random_vector(V2_h)
    phiC = FemField(V2_h)
    y = phiC.coeffs
    P2.dot(x, out=y)

    # Checking projection property P0(P0(phi)) = P0(phi)
    assert np.allclose(P2.dot(y)[:, :], y[:, :])

    # Comparing results of dot and tosparse
    sp_P2 = P2.tosparse()
    y_sp = sp_P2 @ x.toarray()

    assert np.allclose(y_sp, y.toarray())

