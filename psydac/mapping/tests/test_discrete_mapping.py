#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os
from pathlib import Path

import pytest
import numpy as np
import h5py as h5

from igakit.cad import circle, ruled

from sympde.topology import Domain

from psydac.api.discretization import discretize
from psydac.core.bsplines import cell_index
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.splines import SplineSpace
from psydac.mapping.discrete import NurbsMapping
from psydac.utilities.utils import refine_array_1d
from psydac.ddm.cart        import DomainDecomposition

# Get the mesh directory
import psydac.cad.mesh as mesh_mod
mesh_dir = Path(mesh_mod.__file__).parent

# Tolerance for testing float equality
RTOL = 1e-15
ATOL = 1e-15

#==============================================================================
@pytest.mark.parametrize('geometry_file', ['collela_3d.h5', 'collela_2d.h5', 'bent_pipe.h5'])
@pytest.mark.parametrize('npts_per_cell', [2, 3, 4])
def test_build_mesh_reg(geometry_file, npts_per_cell):
    filename = os.path.join(mesh_dir, geometry_file)

    domain = Domain.from_file(filename)
    domain_h = discretize(domain, filename=filename)
    for mapping in domain_h.mappings.values():
        space = mapping.space

        grid = [refine_array_1d(space.breaks[i], npts_per_cell - 1, remove_duplicates=False) for i in range(mapping.ldim)]


        if mapping.ldim == 2:
            x_mesh, y_mesh = mapping.build_mesh(grid, npts_per_cell=npts_per_cell)

            eta1, eta2 = grid

            pcoords = np.array([[mapping(e1, e2) for e2 in eta2] for e1 in eta1])

            x_mesh_l = pcoords[..., 0]
            y_mesh_l = pcoords[..., 1]

        elif mapping.ldim == 3:

            eta1, eta2, eta3 = grid
            x_mesh, y_mesh, z_mesh = mapping.build_mesh(grid, npts_per_cell=npts_per_cell)
            pcoords = np.array([[[mapping(e1, e2, e3) for e3 in eta3] for e2 in eta2] for e1 in eta1])

            x_mesh_l = pcoords[..., 0]
            y_mesh_l = pcoords[..., 1]
            z_mesh_l = pcoords[..., 2]

        else:
            assert False

        assert x_mesh.flags['C_CONTIGUOUS'] and y_mesh.flags['C_CONTIGUOUS']

        assert np.allclose(x_mesh, x_mesh_l, atol=ATOL, rtol=RTOL)
        assert np.allclose(y_mesh, y_mesh_l, atol=ATOL, rtol=RTOL)
        if mapping.ldim == 3:
            assert  z_mesh.flags['C_CONTIGUOUS']
            assert np.allclose(z_mesh, z_mesh_l, atol=ATOL, rtol=RTOL)

#==============================================================================
@pytest.mark.parametrize('geometry_file', ['collela_3d.h5', 'collela_2d.h5', 'bent_pipe.h5'])
@pytest.mark.parametrize('npts_i', [2, 5, 10, 25])
def test_build_mesh_i(geometry_file, npts_i):
    filename = os.path.join(mesh_dir, geometry_file)

    domain = Domain.from_file(filename)
    domain_h = discretize(domain, filename=filename)

    for mapping in domain_h.mappings.values():
        space = mapping.space

        grid = [np.linspace(space.breaks[i][0], space.breaks[i][-1], npts_i) for i in range(mapping.ldim)]


        if mapping.ldim == 2:
            x_mesh, y_mesh = mapping.build_mesh(grid)

            eta1, eta2 = grid

            pcoords = np.array([[mapping(e1, e2) for e2 in eta2] for e1 in eta1])

            x_mesh_l = pcoords[..., 0]
            y_mesh_l = pcoords[..., 1]

        elif mapping.ldim == 3:
            x_mesh, y_mesh, z_mesh = mapping.build_mesh(grid)

            eta1, eta2, eta3 = grid

            pcoords = np.array([[[mapping(e1, e2, e3) for e3 in eta3] for e2 in eta2] for e1 in eta1])

            x_mesh_l = pcoords[..., 0]
            y_mesh_l = pcoords[..., 1]
            z_mesh_l = pcoords[..., 2]

        else:
            assert False

        assert x_mesh.flags['C_CONTIGUOUS'] and y_mesh.flags['C_CONTIGUOUS']

        assert np.allclose(x_mesh, x_mesh_l, atol=ATOL, rtol=RTOL)
        assert np.allclose(y_mesh, y_mesh_l, atol=ATOL, rtol=RTOL)
        if mapping.ldim == 3:
            assert  z_mesh.flags['C_CONTIGUOUS']
            assert np.allclose(z_mesh, z_mesh_l, atol=ATOL, rtol=RTOL)

#==============================================================================
@pytest.mark.mpi
@pytest.mark.parametrize('geometry',  ['collela_3d.h5', 'collela_2d.h5', 'bent_pipe.h5'])
@pytest.mark.parametrize('npts_per_cell', [2, 3, 4, 6])
def test_parallel_jacobians_regular(geometry, npts_per_cell):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    filename = os.path.join(mesh_dir, geometry)

    domain = Domain.from_file(filename=filename)
    domain_h = discretize(domain, filename=filename, comm=comm)

    mapping = list(domain_h.mappings.values())[0]

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
        domain_h = discretize(domain, filename=filename, comm=None)
        mapping = list(domain_h.mappings.values())[0]

        space = mapping.space

        jacobian_matrix = mapping.jac_mat_grid(grid_reg, npts_per_cell=npts_per_cell)
        inv_jacobian_matrix = mapping.inv_jac_mat_grid(grid_reg, npts_per_cell=npts_per_cell)
        jacobian_determinants = mapping.jac_det_grid(grid_reg, npts_per_cell=npts_per_cell)

        fh5 = h5.File(f'result_parallel.h5', mode='r')

        jac_mat_par = fh5['jac_mat'][...]
        inv_jac_mat_par = fh5['inv_jac'][...]
        jac_dets_par = fh5['jac_dets'][...]

        assert np.allclose(jac_mat_par, jacobian_matrix, atol=ATOL, rtol=RTOL)
        assert np.allclose(inv_jac_mat_par, inv_jacobian_matrix, atol=ATOL, rtol=RTOL)
        assert np.allclose(jac_dets_par, jacobian_determinants, atol=ATOL, rtol=RTOL)

        fh5.close()
        os.remove('result_parallel.h5')

#==============================================================================
@pytest.mark.mpi
@pytest.mark.parametrize('geometry',  ['collela_3d.h5', 'collela_2d.h5', 'bent_pipe.h5'])
@pytest.mark.parametrize('npts_irregular', [2, 5, 10, 25])
def test_parallel_jacobians_irregular(geometry, npts_irregular):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    filename = os.path.join(mesh_dir, geometry)

    domain = Domain.from_file(filename=filename)
    domain_h = discretize(domain, filename=filename, comm=comm)

    mapping = list(domain_h.mappings.values())[0]

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
        domain_h = discretize(domain, filename=filename, comm=None)
        mapping = list(domain_h.mappings.values())[0]

        space = mapping.space

        jacobian_matrix = mapping.jac_mat_grid(grid_i)
        inv_jacobian_matrix = mapping.inv_jac_mat_grid(grid_i)
        jacobian_determinants = mapping.jac_det_grid(grid_i)

        fh5 = h5.File(f'result_parallel.h5', mode='r')

        jac_mat_par = fh5['jac_mat'][...]
        inv_jac_mat_par = fh5['inv_jac'][...]
        jac_dets_par = fh5['jac_dets'][...]

        assert np.allclose(jac_mat_par, jacobian_matrix, atol=ATOL, rtol=RTOL)
        assert np.allclose(inv_jac_mat_par, inv_jacobian_matrix, atol=ATOL, rtol=RTOL)
        assert np.allclose(jac_dets_par, jacobian_determinants, atol=ATOL, rtol=RTOL)

        fh5.close()
        os.remove('result_parallel.h5')

#==============================================================================
def test_nurbs_circle():
    rmin, rmax = 0.2, 1
    c1, c2 = 0, 0

    # Igakit
    c_ext = circle(radius=rmax, center=(c1, c2))
    c_int = circle(radius=rmin, center=(c1, c2))

    disk = ruled(c_ext, c_int).transpose()

    w  = disk.weights
    k = disk.knots
    control = disk.points
    d = disk.degree

    # PSYDAC
    spaces = [SplineSpace(degree, knot) for degree, knot in zip(d, k)]

    ncells = [len(space.breaks)-1 for space in spaces]
    periods = [space.periodic for space in spaces]

    domain_decomposition = DomainDecomposition(ncells=ncells, periods=periods, comm=None)
    T = TensorFemSpace(domain_decomposition, *spaces)
    mapping = NurbsMapping.from_control_points_weights(T, control_points=control[..., :2], weights=w)

    x1_pts = np.linspace(0, 1, 10)
    x2_pts = np.linspace(0, 1, 10)

    for x2 in x2_pts:
        for x1 in x1_pts:
            x_p, y_p = mapping(x1, x2)
            x_i, y_i, z_i = disk(x1, x2)

            assert np.allclose((x_p, y_p), (x_i, y_i), atol=ATOL, rtol=RTOL)

            J_p = mapping.jacobian(x1, x2)
            J_i = disk.gradient(u=x1, v=x2)

            assert np.allclose(J_i[:2], J_p, atol=ATOL, rtol=RTOL)
