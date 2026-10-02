#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os
from pathlib import Path

import pytest
import h5py as h5
import numpy as np

from sympde.topology import Domain

from psydac.api.discretization import discretize
from psydac.core.bsplines import cell_index
from psydac.utilities.utils import refine_array_1d

# Get the mesh directory
import psydac.cad.mesh as mesh_mod
mesh_dir = Path(mesh_mod.__file__).parent

#==============================================================================
# Tolerance for testing float equality
RTOL = 1e-15
ATOL = 1e-15

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#==============================================================================
@pytest.mark.mpi
@pytest.mark.parametrize('geometry', ('identity_2d.h5', 'identity_3d.h5', 'collela_2d.h5', 'collela_3d.h5', 'bent_pipe.h5'))
@pytest.mark.parametrize('npts_per_cell', [2, 3, 4, 6])
def test_eval_fields_regular(geometry, npts_per_cell):
    filename = os.path.join(mesh_dir, geometry)

    domain = Domain.from_file(filename=filename)
    domainh = discretize(domain, filename=filename, comm=comm)

    mapping = list(domainh.mappings.values())[0]

    space = mapping.space

    # Regular tensor grid.
    grid_reg = [refine_array_1d(space.breaks[i], npts_per_cell - 1, False) for i in range(space.ldim)]

    npts_per_cell = [npts_per_cell] * space.ldim

    # This calls eval_fields
    mesh_part = mapping.build_mesh(grid_reg, npts_per_cell)

    starts, ends = space.local_domain

    actual_starts = tuple(npts_per_cell[0] * s for s in starts)
    actual_ends   = tuple(npts_per_cell[0] * (e + 1) for e in ends)

    index = tuple(slice(s, e, 1) for s,e in zip(actual_starts, actual_ends))

    shape = tuple(len(grid_reg[i]) for i in range(space.ldim))

    # Saving in an hdf5 file to compare on root
    fh5 = h5.File(f'result_parallel.h5', mode='w', driver='mpio', comm=comm)

    for i, dset_name in enumerate(['x_mesh', 'y_mesh', 'z_mesh'][:space.ldim]):
        fh5.create_dataset(dset_name, shape=shape, dtype=float)

        fh5[dset_name][index] = mesh_part[i]

    fh5.close()

    # Check
    if rank == 0:
        domainh = discretize(domain, filename=filename, comm=None)
        mapping = list(domainh.mappings.values())[0]

        space = mapping.space

        mesh = mapping.build_mesh(grid_reg, npts_per_cell)

        fh5 = h5.File(f'result_parallel.h5', mode='r')

        for i, dset_name in enumerate(['x_mesh', 'y_mesh', 'z_mesh'][:space.ldim]):
            dset = fh5[dset_name][...]

            assert np.allclose(dset, mesh[i], atol=ATOL, rtol=RTOL)

        fh5.close()
        os.remove('result_parallel.h5')

#==============================================================================
@pytest.mark.mpi
@pytest.mark.parametrize('geometry', ('identity_2d.h5', 'identity_3d.h5', 'collela_2d.h5', 'collela_3d.h5', 'bent_pipe.h5'))
@pytest.mark.parametrize('npts_irregular', [2, 5, 10, 25])
def test_eval_fields_irregular(geometry, npts_irregular):
    filename = os.path.join(mesh_dir, geometry)

    domain = Domain.from_file(filename=filename)
    domainh = discretize(domain, filename=filename, comm=comm)

    mapping = list(domainh.mappings.values())[0]

    space = mapping.space
    # Irregular tensor grid
    grid_i = [np.linspace(space.breaks[i][0], space.breaks[i][-1], npts_irregular, True) for i in range(space.ldim)]

    # This calls eval_fields
    mesh_part = mapping.build_mesh(grid_i)

    cell_indexes = [cell_index(space.breaks[i], grid_i[i]) for i in range(space.ldim)]

    starts, ends = space.local_domain

    actual_starts = tuple(np.searchsorted(cell_indexes[i], starts[i], side='left')
                          for i in range(space.ldim))
    actual_ends = tuple(np.searchsorted(cell_indexes[i], ends[i], side='right')
                         for i in range(space.ldim))

    index = tuple(slice(s, e, 1) for s,e in zip(actual_starts, actual_ends))

    shape = tuple(len(grid_i[i]) for i in range(space.ldim))

    # Saving in an hdf5 file to compare on root
    fh5 = h5.File(f'result_parallel_i.h5', mode='w', driver='mpio', comm=comm)

    for i, dset_name in enumerate(['x_mesh', 'y_mesh', 'z_mesh'][:space.ldim]):
        fh5.create_dataset(dset_name, shape=shape, dtype=float)

        fh5[dset_name][index] = mesh_part[i]

    fh5.close()

    # Check
    if rank == 0:
        domainh = discretize(domain, filename=filename, comm=None)
        mapping = list(domainh.mappings.values())[0]

        space = mapping.space

        mesh = mapping.build_mesh(grid_i)

        fh5 = h5.File(f'result_parallel_i.h5', mode='r')

        for i, dset_name in enumerate(['x_mesh', 'y_mesh', 'z_mesh'][:space.ldim]):
            dset = fh5[dset_name][...]

            assert np.allclose(dset, mesh[i], atol=ATOL, rtol=RTOL)

        fh5.close()
        os.remove('result_parallel_i.h5')

#==============================================================================
@pytest.mark.mpi
def test_eval_field_roundoff():
    """
    Check field and gradient evaluation under round-off
    at periodic MPI boundaries
    """

    from psydac.ddm.cart import DomainDecomposition
    from psydac.fem.basic import FemField
    from psydac.fem.splines import SplineSpace
    from psydac.fem.tensor import TensorFemSpace

    ncells = [8, 16]
    degree = [2, 2]

    grid_s     = np.linspace(0.0, 1.0, ncells[0] + 1)
    grid_theta = np.linspace(0.0, 2.0 * np.pi, ncells[1] + 1)

    V_s = SplineSpace(degree=degree[0], grid=grid_s, periodic=False)
    V_theta = SplineSpace(degree=degree[1], grid=grid_theta, periodic=True)

    # Force MPI decomposition only in theta
    domain_decomposition = DomainDecomposition(
        ncells,
        periods=[False, True],
        comm=comm,
        mpi_dims_mask=[False, True],
    )

    V = TensorFemSpace(domain_decomposition,V_s, V_theta)
    u = FemField(V)
    u.coeffs.update_ghost_regions()

    # Evaluation point inside the local radial interval
    s = (V.eta_lims[0][0] + V.eta_lims[0][1]) / 2.0

    theta_max_local = V.eta_lims[1][1]
    theta_max_global = V_theta.breaks[-1]

    # Only test internal MPI boundaries
    if theta_max_local < theta_max_global:

        # Smallest representable float larger than the boundary
        theta_roundoff = np.nextafter(theta_max_local, theta_max_local + 1)

        # Check that TensorFemSpace field and gradient evaluation remains valid
        # when round-off places the point just beyond the local MPI boundary
        value_with_roundoff = u(s, theta_roundoff)
        grad_with_roundoff = u.gradient(s, theta_roundoff)
