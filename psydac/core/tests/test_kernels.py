import numpy as np
import pytest
import os
import itertools as it

from sympde.topology import Domain
from psydac.api.discretization import discretize
from psydac.utilities.utils import refine_array_1d
from psydac.core.kernels import (eval_jacobians_2d, eval_jacobians_3d, eval_jacobians_2d_weights,
                                 eval_jacobians_3d_weights, eval_jacobians_inv_2d_weights,
                                 eval_jacobians_inv_3d_weights, eval_jacobians_inv_2d, eval_jacobians_inv_3d)


# Get mesh directory
try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']
except KeyError:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')


@pytest.mark.parametrize(('geometry_file', 'ldim', 'is_nurbs'), [('identity_2d.h5', 2, False),
                                                                 ('identity_3d.h5', 3, False),
                                                                 ('bent_pipe.h5', 2, True),
                                                                 ('quarter_annulus.h5', 2, True),
                                                                 ('collela_2d.h5', 2, False),
                                                                 ('collela_3d.h5', 3, False)])
def test_jacobians(geometry_file, ldim, is_nurbs):

    filename = os.path.join(mesh_dir, geometry_file)

    domain = Domain.from_file(filename)
    domainh = discretize(domain, filename=filename)

    mapping = list(domainh.mappings.values())[0]

    grid = []
    ncells = []
    for i in range(ldim):
        grid_i_initial = mapping.space.breaks[i]
        ncells.append(len(grid_i_initial) - 1)
        grid.append(np.reshape(np.asarray(refine_array_1d(grid_i_initial, 1, remove_duplicates=False)),
                               (ncells[-1], 2)))
    n_evals_points = [2] * ldim

    pads, \
    degree, \
    global_basis, \
    global_spans = mapping.space.preprocess_regular_tensor_grid(grid, der=1)

    jac_mats = np.zeros((*(tuple(ncells[i] * n_evals_points[i] for i in range(ldim))), ldim, ldim))
    inv_jac_mats = np.zeros((*(tuple(ncells[i] * n_evals_points[i] for i in range(ldim))), ldim, ldim))

    if is_nurbs:
        global_arr_weights = mapping._weights_field.coeffs._data

        if ldim == 2:
            global_arr_x = mapping._fields[0].coeffs._data
            global_arr_y = mapping._fields[1].coeffs._data

            # Compute the jacobians
            eval_jacobians_2d_weights(ncells[0], ncells[1], pads[0], pads[1], degree[0], degree[1], n_evals_points[0],
                                      n_evals_points[1], global_basis[0], global_basis[1], global_spans[0],
                                      global_spans[1], global_arr_x, global_arr_y, global_arr_weights, jac_mats)

            # Compute the inverses of the jacobians
            eval_jacobians_inv_2d_weights(ncells[0], ncells[1], pads[0], pads[1], degree[0], degree[1],
                                          n_evals_points[0], n_evals_points[1], global_basis[0], global_basis[1],
                                          global_spans[0], global_spans[1], global_arr_x, global_arr_y,
                                          global_arr_weights, inv_jac_mats)

        if ldim == 3:
            global_arr_x = mapping._fields[0].coeffs._data
            global_arr_y = mapping._fields[1].coeffs._data
            global_arr_z = mapping._fields[2].coeffs._data

            # Compute the jacobians
            eval_jacobians_3d_weights(ncells[0], ncells[1], ncells[2], pads[0], pads[1], pads[2], degree[0], degree[1],
                                      degree[2], n_evals_points[0], n_evals_points[1], n_evals_points[2],
                                      global_basis[0], global_basis[1], global_basis[2], global_spans[0],
                                      global_spans[1], global_spans[2], global_arr_x, global_arr_y, global_arr_z,
                                      global_arr_weights, jac_mats)

            # Compute the inverses of the jacobians
            eval_jacobians_inv_3d_weights(ncells[0], ncells[1], ncells[2], pads[0], pads[1], pads[2], degree[0],
                                          degree[1], degree[2], n_evals_points[0], n_evals_points[1], n_evals_points[2],
                                          global_basis[0], global_basis[1], global_basis[2], global_spans[0],
                                          global_spans[1], global_spans[2], global_arr_x, global_arr_y, global_arr_z,
                                          global_arr_weights, inv_jac_mats)

    else:

        if mapping.ldim == 2:
            global_arr_x = mapping._fields[0].coeffs._data
            global_arr_y = mapping._fields[1].coeffs._data

            # Compute the jacobians
            eval_jacobians_2d(ncells[0], ncells[1], pads[0], pads[1], degree[0], degree[1], n_evals_points[0],
                              n_evals_points[1], global_basis[0], global_basis[1], global_spans[0], global_spans[1],
                              global_arr_x, global_arr_y, jac_mats)

            # Compute the inverses of the jacobians
            eval_jacobians_inv_2d(ncells[0], ncells[1], pads[0], pads[1], degree[0], degree[1], n_evals_points[0],
                                  n_evals_points[1], global_basis[0], global_basis[1], global_spans[0], global_spans[1],
                                  global_arr_x, global_arr_y, inv_jac_mats)

        if ldim == 3:

            global_arr_x = mapping._fields[0].coeffs._data
            global_arr_y = mapping._fields[1].coeffs._data
            global_arr_z = mapping._fields[2].coeffs._data

            # Compute the jacobians
            eval_jacobians_3d(ncells[0], ncells[1], ncells[2], pads[0], pads[1], pads[2], degree[0], degree[1],
                              degree[2], n_evals_points[0], n_evals_points[1], n_evals_points[2], global_basis[0],
                              global_basis[1], global_basis[2], global_spans[0], global_spans[1], global_spans[2],
                              global_arr_x, global_arr_y, global_arr_z, jac_mats)

            # Compute the inverses of the jacobians
            eval_jacobians_inv_3d(ncells[0], ncells[1], ncells[2], pads[0], pads[1], pads[2], degree[0], degree[1],
                                  degree[2], n_evals_points[0], n_evals_points[1], n_evals_points[2], global_basis[0],
                                  global_basis[1], global_basis[2], global_spans[0], global_spans[1], global_spans[2],
                                  global_arr_x, global_arr_y, global_arr_z, inv_jac_mats)

    if ldim == 2:
        for i, j in it.product(range(jac_mats.shape[0]), range(jac_mats.shape[1])):
            # Assert that the computed inverse is the inverse.
            assert np.allclose(np.dot(jac_mats[i, j], inv_jac_mats[i, j]), np.eye(ldim))

    if ldim == 3:

        for i, j, k in it.product(range(jac_mats.shape[0]), range(jac_mats.shape[1]), range(jac_mats.shape[2])):
            # Assert that the computed inverse is the inverse.
            assert np.allclose(np.dot(jac_mats[i, j, k], inv_jac_mats[i, j, k]), np.eye(ldim))
