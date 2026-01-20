import numpy as np
from numpy import pi

import pytest
from sympde import ScalarFunctionSpace

from mpi4py import MPI

from sympde.topology import Square
from sympde.topology.analytical_mapping import PolarMapping

from psydac.api.discretization import discretize
from psydac.feec.polar.conga_projections import C0PolarProjection_V0
from psydac.fem.basic import FemField


@pytest.mark.parametrize( 'R', [1])
@pytest.mark.parametrize( 'ncells', [[4, 8], [12, 12]])
@pytest.mark.parametrize( 'degree', [[1, 1], [2, 2]])
@pytest.mark.parallel

def test_C0PolarProjection_V0(R, ncells, degree):
    mpi_comm = MPI.COMM_WORLD

    # Build physical domain - disk of radius R
    logical_domain = Square('Omega', bounds1=[0, R], bounds2=[0, 2 * pi])
    mapping = PolarMapping('F', c1=0, c2=0, rmin=0, rmax=1)
    domain = mapping(logical_domain)

    # Discrete physical domain and discrete space
    domain_h = discretize(domain, ncells=ncells, periodic=[False, True], comm=mpi_comm)
    V0 = ScalarFunctionSpace('V0', domain)
    V0_h = discretize(V0, domain_h, degree=degree)

    P0 = C0PolarProjection_V0(V0_h, hbc=True)

    [s1, s2] = V0_h.coeff_space.starts
    [e1, e2] = V0_h.coeff_space.ends

    phi = FemField(V0_h)
    x = phi.coeffs
    x[s1:e1 + 1, s2:e2 + 1] = np.random.random([e1 - s1 + 1, e2 - s2 + 1])
    x.update_ghost_regions()

    phiC = FemField(V0_h)
    y = phiC.coeffs
    P0.dot(x, out=y)

    # Checking projection property P0(P0(phi)) = P0(phi)
    assert np.allclose(P0.dot(y)[:,:], y[:,:])

    # Comparing results of dot and tosparse
    sp_P0 = P0.tosparse()
    y_sp = sp_P0 @ x.toarray()
    y = mpi_comm.allreduce(y.toarray(), op=MPI.SUM)
    y_sp = mpi_comm.allreduce(y_sp, op=MPI.SUM)

    print(np.allclose(y_sp, y))
    assert np.allclose(y_sp, y)
