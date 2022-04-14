import numpy as np
import pytest
import os
import h5py as h5

from sympde.topology import Domain
from psydac.api.discretization import discretize
from psydac.utilities.utils import refine_array_1d

from psydac.core.bsplines import quadrature_grid, basis_ders_on_quad_grid, elements_spans, cell_index
from psydac.utilities.quadratures import gauss_legendre

try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']
except KeyError:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')


@pytest.mark.parametrize('geometry_file', ['collela_3d.h5', 'collela_2d.h5', 'bent_pipe.h5'])
@pytest.mark.parametrize('npts_per_cell', [2, 3, 4])
def test_build_mesh_reg(geometry_file, npts_per_cell):
    filename = os.path.join(mesh_dir, geometry_file)

    domain = Domain.from_file(filename)
    domainh = discretize(domain, filename=filename)

    for mapping in domainh.mappings.values():
        space = mapping.space

        grid = [refine_array_1d(space.breaks[i], npts_per_cell - 1, remove_duplicates=False) for i in range(mapping.ldim)]

        x_mesh, y_mesh, z_mesh = mapping.build_mesh(grid, npts_per_cell=npts_per_cell)

        if mapping.ldim == 2:

            eta1, eta2 = grid

            pcoords = np.array([[mapping(e1, e2) for e2 in eta2] for e1 in eta1])

            x_mesh_l = pcoords[..., 0:1]
            y_mesh_l = pcoords[..., 1:2]
            z_mesh_l = np.zeros_like(x_mesh_l)

        elif mapping.ldim == 3:

            eta1, eta2, eta3 = grid

            pcoords = np.array([[[mapping(e1, e2, e3) for e3 in eta3] for e2 in eta2] for e1 in eta1])

            x_mesh_l = pcoords[..., 0]
            y_mesh_l = pcoords[..., 1]
            z_mesh_l = pcoords[..., 2]

        else:
            assert False

        assert x_mesh.flags['C_CONTIGUOUS'] and y_mesh.flags['C_CONTIGUOUS'] and z_mesh.flags['C_CONTIGUOUS']

        assert np.allclose(x_mesh, x_mesh_l)
        assert np.allclose(y_mesh, y_mesh_l)
        assert np.allclose(z_mesh, z_mesh_l)

@pytest.mark.parametrize('geometry_file', ['collela_3d.h5', 'collela_2d.h5', 'bent_pipe.h5'])
@pytest.mark.parametrize('npts_i', [2, 5, 10, 25])
def test_build_mesh_i(geometry_file, npts_i):
    filename = os.path.join(mesh_dir, geometry_file)

    domain = Domain.from_file(filename)
    domainh = discretize(domain, filename=filename)

    for mapping in domainh.mappings.values():
        space = mapping.space

        grid = [np.linspace(space.breaks[i][0], space.breaks[i][-1], npts_i) for i in range(mapping.ldim)]

        x_mesh, y_mesh, z_mesh = mapping.build_mesh(grid)

        if mapping.ldim == 2:

            eta1, eta2 = grid

            pcoords = np.array([[mapping(e1, e2) for e2 in eta2] for e1 in eta1])

            x_mesh_l = pcoords[..., 0:1]
            y_mesh_l = pcoords[..., 1:2]
            z_mesh_l = np.zeros_like(x_mesh_l)

        elif mapping.ldim == 3:

            eta1, eta2, eta3 = grid

            pcoords = np.array([[[mapping(e1, e2, e3) for e3 in eta3] for e2 in eta2] for e1 in eta1])

            x_mesh_l = pcoords[..., 0]
            y_mesh_l = pcoords[..., 1]
            z_mesh_l = pcoords[..., 2]

        else:
            assert False

        assert x_mesh.flags['C_CONTIGUOUS'] and y_mesh.flags['C_CONTIGUOUS'] and z_mesh.flags['C_CONTIGUOUS']

        assert np.allclose(x_mesh, x_mesh_l)
        assert np.allclose(y_mesh, y_mesh_l)
        assert np.allclose(z_mesh, z_mesh_l)

@pytest.mark.parallel
@pytest.mark.parametrize('geometry',  ['collela_3d.h5', 'collela_2d.h5', 'bent_pipe.h5'])
@pytest.mark.parametrize('npts_per_cell', [2, 3, 4, 6])
def test_parallel_jacobians_regular(geometry, npts_per_cell):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    filename = os.path.join(mesh_dir, geometry)

    domain = Domain.from_file(filename=filename)
    domainh = discretize(domain, filename=filename, comm=comm)

    mapping = list(domainh.mappings.values())[0]

    space = mapping.space

    # Regular tensor grid.
    grid_reg = [refine_array_1d(space.breaks[i], npts_per_cell - 1, False) for i in range(space.ldim)]

    npts_per_cell = [npts_per_cell] * space.ldim

    jacobian_matrix_p = mapping.jac_mat_grid(grid_reg, npts_per_cell=npts_per_cell)
    inv_jacobian_matrix_p = mapping.inv_jac_mat_grid(grid_reg, npts_per_cell=npts_per_cell)
    jacobian_determinants_p = mapping.jac_det_grid(grid_reg, npts_per_cell=npts_per_cell)

    starts, ends = space.local_domain

    actual_starts = tuple(npts_per_cell[0] * s for s in starts)
    actual_ends   = tuple(npts_per_cell[0] * (e + 1) for e in ends)

    index = tuple(slice(s, e, 1) for s,e in zip(actual_starts, actual_ends))

    shape_0 = tuple(len(grid_reg[i]) for i in range(space.ldim)) + (space.ldim, space.ldim)
    shape_1 = tuple(len(grid_reg[i]) for i in range(space.ldim))

    # Saving in an hdf5 file to compare on root
    fh5 = h5.File(f'result_parallel.h5', mode='w', driver='mpio', comm=comm)

    fh5.create_dataset('jac_mat', shape=shape_0, dtype=float)
    fh5.create_dataset('inv_jac', shape=shape_0, dtype=float)
    fh5.create_dataset('jac_dets', shape=shape_1, dtype=float)

    fh5['jac_mat'][index] = jacobian_matrix_p
    fh5['inv_jac'][index] = inv_jacobian_matrix_p
    fh5['jac_dets'][index] = jacobian_determinants_p
    fh5.close()

    # Check
    if rank == 0:
        domainh = discretize(domain, filename=filename, comm=None)
        mapping = list(domainh.mappings.values())[0]

        space = mapping.space
        
        jacobian_matrix = mapping.jac_mat_grid(grid_reg, npts_per_cell=npts_per_cell)
        inv_jacobian_matrix = mapping.inv_jac_mat_grid(grid_reg, npts_per_cell=npts_per_cell)
        jacobian_determinants = mapping.jac_det_grid(grid_reg, npts_per_cell=npts_per_cell)

        fh5 = h5.File(f'result_parallel.h5', mode='r')

        jac_mat_par = fh5['jac_mat'][...]
        inv_jac_mat_par = fh5['inv_jac'][...]
        jac_dets_par = fh5['jac_dets'][...]

        assert np.allclose(jac_mat_par, jacobian_matrix)
        assert np.allclose(inv_jac_mat_par, inv_jacobian_matrix)
        assert np.allclose(jac_dets_par, jacobian_determinants)

        fh5.close()
        os.remove('result_parallel.h5')


@pytest.mark.parallel
@pytest.mark.parametrize('geometry',  ['collela_3d.h5', 'collela_2d.h5', 'bent_pipe.h5'])
@pytest.mark.parametrize('npts_irregular', [2, 5, 10, 25])
def test_parallel_jacobians_irregular(geometry, npts_irregular):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    filename = os.path.join(mesh_dir, geometry)

    domain = Domain.from_file(filename=filename)
    domainh = discretize(domain, filename=filename, comm=comm)

    mapping = list(domainh.mappings.values())[0]

    space = mapping.space
    # Irregular tensor grid
    grid_i = [np.linspace(space.breaks[i][0], space.breaks[i][-1], npts_irregular, True) for i in range(space.ldim)]

    jacobian_matrix_p = mapping.jac_mat_grid(grid_i)
    inv_jacobian_matrix_p = mapping.inv_jac_mat_grid(grid_i)
    jacobian_determinants_p = mapping.jac_det_grid(grid_i)
    
    cell_indexes = [cell_index(space.breaks[i], grid_i[i]) for i in range(space.ldim)]
    
    starts, ends = space.local_domain

    actual_starts = tuple(np.searchsorted(cell_indexes[i], starts[i], side='left') 
                          for i in range(space.ldim))
    actual_ends = tuple(np.searchsorted(cell_indexes[i], ends[i], side='right') 
                         for i in range(space.ldim))

    index = tuple(slice(s, e, 1) for s,e in zip(actual_starts, actual_ends))

    shape_0 = tuple(len(grid_i[i]) for i in range(space.ldim)) + (space.ldim, space.ldim)
    shape_1 = tuple(len(grid_i[i]) for i in range(space.ldim))

    # Saving in an hdf5 file to compare on root
    fh5 = h5.File(f'result_parallel.h5', mode='w', driver='mpio', comm=comm)

    fh5.create_dataset('jac_mat', shape=shape_0, dtype=float)
    fh5.create_dataset('inv_jac', shape=shape_0, dtype=float)
    fh5.create_dataset('jac_dets', shape=shape_1, dtype=float)

    fh5['jac_mat'][index] = jacobian_matrix_p
    fh5['inv_jac'][index] = inv_jacobian_matrix_p
    fh5['jac_dets'][index] = jacobian_determinants_p
    fh5.close()

    # Check
    if rank == 0:
        domainh = discretize(domain, filename=filename, comm=None)
        mapping = list(domainh.mappings.values())[0]

        space = mapping.space
        
        jacobian_matrix = mapping.jac_mat_grid(grid_i)
        inv_jacobian_matrix = mapping.inv_jac_mat_grid(grid_i)
        jacobian_determinants = mapping.jac_det_grid(grid_i)

        fh5 = h5.File(f'result_parallel.h5', mode='r')

        jac_mat_par = fh5['jac_mat'][...]
        inv_jac_mat_par = fh5['inv_jac'][...]
        jac_dets_par = fh5['jac_dets'][...]

        assert np.allclose(jac_mat_par, jacobian_matrix)
        assert np.allclose(inv_jac_mat_par, inv_jacobian_matrix)
        assert np.allclose(jac_dets_par, jacobian_determinants)

        fh5.close()
        os.remove('result_parallel.h5')
