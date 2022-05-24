import os
import pytest
import h5py as h5
import numpy as np

from sympde.topology import Domain

from psydac.api.discretization import discretize
from psydac.core.bsplines import cell_index
from psydac.utilities.utils import refine_array_1d


try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']

except:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')

# Tolerance for testing float equality
RTOL = 1e-15
ATOL = 1e-15

from mpi4py import MPI 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


@pytest.mark.parallel
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
    x_mesh_part, y_mesh_part, z_mesh_part = mapping.build_mesh(grid_reg, npts_per_cell)

    starts, ends = space.local_domain

    actual_starts = tuple(npts_per_cell[0] * s for s in starts)
    actual_ends   = tuple(npts_per_cell[0] * (e + 1) for e in ends)

    index = tuple(slice(s, e, 1) for s,e in zip(actual_starts, actual_ends))

    shape = tuple(len(grid_reg[i]) for i in range(space.ldim)) + (1,) * (3 - space.ldim)

    # Saving in an hdf5 file to compare on root
    fh5 = h5.File(f'result_parallel.h5', mode='w', driver='mpio', comm=comm)

    fh5.create_dataset('x_mesh', shape=shape, dtype=float)
    fh5.create_dataset('y_mesh', shape=shape, dtype=float)
    fh5.create_dataset('z_mesh', shape=shape, dtype=float)

    fh5['x_mesh'][index] = x_mesh_part
    fh5['y_mesh'][index] = y_mesh_part
    fh5['z_mesh'][index] = z_mesh_part
    fh5.close()

    # Check
    if rank == 0:
        domainh = discretize(domain, filename=filename, comm=None)
        mapping = list(domainh.mappings.values())[0]

        space = mapping.space
        
        x_mesh, y_mesh, z_mesh = mapping.build_mesh(grid_reg, npts_per_cell)

        fh5 = h5.File(f'result_parallel.h5', mode='r')

        x_mesh_par = fh5['x_mesh'][...]
        y_mesh_par = fh5['y_mesh'][...]
        z_mesh_par = fh5['z_mesh'][...]

        assert np.allclose(x_mesh_par, x_mesh, atol=ATOL, rtol=RTOL)
        assert np.allclose(y_mesh_par, y_mesh, atol=ATOL, rtol=RTOL)
        assert np.allclose(z_mesh_par, z_mesh, atol=ATOL, rtol=RTOL)

        fh5.close()
        os.remove('result_parallel.h5')


@pytest.mark.parallel
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
    x_mesh_i_part, y_mesh_i_part, z_mesh_i_part = mapping.build_mesh(grid_i) 

    
    cell_indexes = [cell_index(space.breaks[i], grid_i[i]) for i in range(space.ldim)]
    
    starts, ends = space.local_domain

    actual_starts = tuple(np.searchsorted(cell_indexes[i], starts[i], side='left') 
                          for i in range(space.ldim)) + (0,) * (3 - space.ldim)
    actual_ends = tuple(np.searchsorted(cell_indexes[i], ends[i], side='right') 
                         for i in range(space.ldim)) + (1,) * (3 - space.ldim)

    index = tuple(slice(s, e, 1) for s,e in zip(actual_starts, actual_ends))

    shape = tuple(len(grid_i[i]) for i in range(space.ldim)) + (1,) * (3 - space.ldim)

    # Saving in an hdf5 file to compare on root
    fh5 = h5.File(f'result_parallel_i.h5', mode='w', driver='mpio', comm=comm)

    fh5.create_dataset('x_mesh', shape=shape, dtype=float)
    fh5.create_dataset('y_mesh', shape=shape, dtype=float)
    fh5.create_dataset('z_mesh', shape=shape, dtype=float)

    fh5['x_mesh'][index] = x_mesh_i_part
    fh5['y_mesh'][index] = y_mesh_i_part
    fh5['z_mesh'][index] = z_mesh_i_part
    fh5.close()

    # Check
    if rank == 0:
        domainh = discretize(domain, filename=filename, comm=None)
        mapping = list(domainh.mappings.values())[0]

        space = mapping.space

        x_mesh, y_mesh, z_mesh = mapping.build_mesh(grid_i)

        fh5 = h5.File(f'result_parallel_i.h5', mode='r')
    
        x_mesh_par = fh5['x_mesh'][...]
        y_mesh_par = fh5['y_mesh'][...]
        z_mesh_par = fh5['z_mesh'][...]

        assert np.allclose(x_mesh_par, x_mesh, atol=ATOL, rtol=RTOL)
        assert np.allclose(y_mesh_par, y_mesh, atol=ATOL, rtol=RTOL)
        assert np.allclose(z_mesh_par, z_mesh, atol=ATOL, rtol=RTOL)

        fh5.close()
        os.remove('result_parallel_i.h5')
