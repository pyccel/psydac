import numpy as np
import pytest
import os
import itertools as it

from sympde.topology import Domain, ScalarFunctionSpace, VectorFunctionSpace, Square, Cube
from psydac.api.discretization import discretize
from psydac.utilities.utils import refine_array_1d
from psydac.fem.basic import FemField
from psydac.core.kernels import (eval_fields_2d_no_weights, eval_fields_3d_no_weights,
                                 eval_fields_2d_weighted, eval_fields_3d_weighted,
                                 eval_jacobians_2d, eval_jacobians_3d, eval_jacobians_2d_weights,
                                 eval_jacobians_3d_weights, eval_jacobians_inv_2d_weights,
                                 eval_jacobians_inv_3d_weights, eval_jacobians_inv_2d, eval_jacobians_inv_3d,
                                 eval_det_metric_2d_weights, eval_det_metric_3d_weights,
                                 eval_det_metric_2d, eval_det_metric_3d)


# Get mesh directory
try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']
except KeyError:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')


@pytest.mark.parametrize('degree', [(2, 2), (3, 3)])
@pytest.mark.parametrize('refine', (1, 2))
def test_eval_fields_2d(degree, refine):
    # SymPDE layer
    domain = Square()
    scal = ScalarFunctionSpace('S', domain)

    domainh = discretize(domain, ncells=[5, 5])

    # TensorFemSpace
    tensor_h = discretize(scal, domainh, degree=degree)
    th = FemField(tensor_h)

    # give random coefficients
    th.coeffs[:] = np.random.random(size=th.coeffs[:].shape)


    # Building the grid
    ncells = [len(tensor_h.breaks[i]) - 1 for i in range(2)]
    n_eval_points = [refine + 1] * 2
    grid_raw = [refine_array_1d(tensor_h.breaks[i], refine, remove_duplicates=False) for i in range(2)]
    grid_tensor = [np.reshape(grid_raw[i], (ncells[i], n_eval_points[i])) for i in range(2)]

    # preprocess
    pads, degrees, basis, spans = tensor_h.preprocess_regular_tensor_grid(grid_tensor, der=0)

    # Test without weights
    expected_no_weights = np.array([[tensor_h.eval_field(th, e1, e2) for e2 in grid_raw[1]] for e1 in grid_raw[0]])

    out_no_weights = np.zeros((*tuple(ncells[i] * n_eval_points[i] for i in range(2)), 1))

    eval_fields_2d_no_weights(ncells[0], ncells[1], pads[0], pads[1], degrees[0], degrees[1], n_eval_points[0],
                              n_eval_points[1], basis[0], basis[1], spans[0], spans[1], th.coeffs._data[:, :, None],
                              out_no_weights)

    assert np.allclose(expected_no_weights[:, :, None], out_no_weights)

    # With weights
    weights_field = FemField(tensor_h)
    weights_field.coeffs._data[:] = np.random.random(size=th.coeffs._data[:].shape)

    expected_weights = np.array([[tensor_h.eval_field(th, e1, e2, weights=weights_field.coeffs)/weights_field(e1, e2)
                                  for e2 in grid_raw[1]]
                                 for e1 in grid_raw[0]])

    out_weights = np.zeros((*tuple(ncells[i] * n_eval_points[i] for i in range(2)), 1))
    eval_fields_2d_weighted(ncells[0], ncells[1], pads[0], pads[1], degrees[0], degrees[1], n_eval_points[0],
                            n_eval_points[1], basis[0], basis[1], spans[0], spans[1], th.coeffs._data[:, :, None],
                            weights_field.coeffs._data, out_weights)

    assert np.allclose(expected_weights[:, :, None], out_weights)


@pytest.mark.parametrize('degree', [(2, 2, 2), (3, 3, 3)])
@pytest.mark.parametrize('refine', (1, 2))
def test_eval_fields_3d(degree, refine):
    # SymPDE layer
    domain = Cube()
    scal = ScalarFunctionSpace('S', domain)

    domainh = discretize(domain, ncells=[5, 5, 5])

    # TensorFemSpace
    tensor_h = discretize(scal, domainh, degree=degree)
    th = FemField(tensor_h)

    # give random coefficients
    th.coeffs[:] = np.random.random(size=th.coeffs[:].shape)


    # Building the grid
    ncells = [len(tensor_h.breaks[i]) - 1 for i in range(3)]
    n_eval_points = [refine + 1] * 3
    grid_raw = [refine_array_1d(tensor_h.breaks[i], refine, remove_duplicates=False) for i in range(3)]
    grid_tensor = [np.reshape(grid_raw[i], (ncells[i], n_eval_points[i])) for i in range(3)]

    # preprocess
    pads, degrees, basis, spans = tensor_h.preprocess_regular_tensor_grid(grid_tensor, der=0)

    # Test without weights
    expected_no_weights = np.array([[[tensor_h.eval_field(th, e1, e2, e3)
                                     for e3 in grid_raw[2]]
                                    for e2 in grid_raw[1]]
                                   for e1 in grid_raw[0]])

    out_no_weights = np.zeros((*tuple(ncells[i] * n_eval_points[i] for i in range(3)), 1))

    eval_fields_3d_no_weights(ncells[0], ncells[1], ncells[2], pads[0], pads[1], pads[2],
                              degrees[0], degrees[1], degrees[2], n_eval_points[0],
                              n_eval_points[1], n_eval_points[2], basis[0], basis[1], basis[2],
                              spans[0], spans[1], spans[2], th.coeffs._data[:, :, :, None],
                              out_no_weights)

    assert np.allclose(expected_no_weights[:, :, :, None], out_no_weights)

    # With weights
    weights_field = FemField(tensor_h)
    weights_field.coeffs._data[:] = np.random.random(size=th.coeffs._data[:].shape)

    expected_weights = np.array([[[
        tensor_h.eval_field(th, e1, e2, e3, weights=weights_field.coeffs)/weights_field(e1, e2, e3)
        for e3 in grid_raw[2]]
       for e2 in grid_raw[1]]
      for e1 in grid_raw[0]])

    out_weights = np.zeros((*tuple(ncells[i] * n_eval_points[i] for i in range(3)), 1))

    eval_fields_3d_weighted(ncells[0], ncells[1], ncells[2], pads[0], pads[1], pads[2],
                              degrees[0], degrees[1], degrees[2], n_eval_points[0],
                              n_eval_points[1], n_eval_points[2], basis[0], basis[1], basis[2],
                              spans[0], spans[1], spans[2], th.coeffs._data[:, :, :, None], weights_field.coeffs._data,
                              out_weights)

    assert np.allclose(expected_weights[:, :, :, None], out_weights)


@pytest.mark.parametrize(('geometry_file', 'ldim', 'is_nurbs'), [('identity_2d.h5', 2, False),
                                                                 ('identity_3d.h5', 3, False),
                                                                 ('bent_pipe.h5', 2, True),
                                                                 ('quarter_annulus.h5', 2, True),
                                                                 ('collela_2d.h5', 2, False),
                                                                 ('collela_3d.h5', 3, False)])
@pytest.mark.parametrize('refine', (1, 2))
def test_jacobian_matrix_inverse_determinant(geometry_file, ldim, is_nurbs, refine):

    filename = os.path.join(mesh_dir, geometry_file)

    domain = Domain.from_file(filename)
    domainh = discretize(domain, filename=filename)

    mapping = list(domainh.mappings.values())[0]

    grid = []
    ncells = []
    for i in range(ldim):
        grid_i_initial = mapping.space.breaks[i]
        ncells.append(len(grid_i_initial) - 1)
        grid.append(np.asarray(refine_array_1d(grid_i_initial, refine, remove_duplicates=False)))

    n_evals_points = [refine + 1] * ldim

    # Direct API of the mapping
    try:
        if ldim == 2:
            jacobians_matrix_direct = np.array([[mapping.jac_mat(e1, e2) for e2 in grid[1]] for e1 in grid[0]])

        if ldim == 3:
            jacobians_matrix_direct = np.array([[[mapping.jac_mat(e1, e2, e3)
                                                  for e3 in grid[2]]
                                                 for e2 in grid[1]]
                                                for e1 in grid[0]])
    except NotImplementedError:
        pass

    # Kernel functions
    tensor_grid = [np.reshape(grid[i], (ncells[i], n_evals_points[i])) for i in range(ldim)]
    pads, \
    degree, \
    global_basis, \
    global_spans = mapping.space.preprocess_regular_tensor_grid(tensor_grid, der=1)

    jac_mats = np.zeros((*(tuple(ncells[i] * n_evals_points[i] for i in range(ldim))), ldim, ldim))
    inv_jac_mats = np.zeros((*(tuple(ncells[i] * n_evals_points[i] for i in range(ldim))), ldim, ldim))
    metric_dets = np.zeros(tuple(ncells[i] * n_evals_points[i] for i in range(ldim)))


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

            # Compute the determinant of the jacobians
            eval_det_metric_2d_weights(ncells[0], ncells[1], pads[0], pads[1], degree[0], degree[1],
                                       n_evals_points[0], n_evals_points[1], global_basis[0], global_basis[1],
                                       global_spans[0], global_spans[1], global_arr_x, global_arr_y,
                                       global_arr_weights, metric_dets)

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

            # Compute the determinant of the jacobians
            eval_det_metric_3d_weights(ncells[0], ncells[1], ncells[2], pads[0], pads[1], pads[2], degree[0],
                                       degree[1], degree[2], n_evals_points[0], n_evals_points[1], n_evals_points[2],
                                       global_basis[0], global_basis[1], global_basis[2], global_spans[0],
                                       global_spans[1], global_spans[2], global_arr_x, global_arr_y, global_arr_z,
                                       global_arr_weights, metric_dets)
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

            # Compute the determinant of the jacobians
            eval_det_metric_2d(ncells[0], ncells[1], pads[0], pads[1], degree[0], degree[1],
                               n_evals_points[0], n_evals_points[1], global_basis[0], global_basis[1],
                               global_spans[0], global_spans[1], global_arr_x, global_arr_y,
                               metric_dets)

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

            # Compute the determinant of the jacobians
            eval_det_metric_3d(ncells[0], ncells[1], ncells[2], pads[0], pads[1], pads[2], degree[0],
                               degree[1], degree[2], n_evals_points[0], n_evals_points[1], n_evals_points[2],
                               global_basis[0], global_basis[1], global_basis[2], global_spans[0],
                               global_spans[1], global_spans[2], global_arr_x, global_arr_y, global_arr_z,
                               metric_dets)
    try:
        assert np.allclose(jacobians_matrix_direct, jac_mats)
    except NameError:
        pass


    if ldim == 2:
        for i, j in it.product(range(jac_mats.shape[0]), range(jac_mats.shape[1])):
            # Assert that the computed inverse is the inverse.
            assert np.allclose(np.dot(jac_mats[i, j], inv_jac_mats[i, j]), np.eye(ldim))
            # Assert that the computed determinant is the determinant
            assert np.allclose(np.linalg.det(jac_mats[i, j]), metric_dets[i, j])

    if ldim == 3:
        for i, j, k in it.product(range(jac_mats.shape[0]), range(jac_mats.shape[1]), range(jac_mats.shape[2])):
            # Assert that the computed inverse is the inverse.
            assert np.allclose(np.dot(jac_mats[i, j, k], inv_jac_mats[i, j, k]), np.eye(ldim))
            # Assert that the computed determinant is the determinant
            assert np.allclose(np.linalg.det(jac_mats[i, j, k]), metric_dets[i, j, k])

