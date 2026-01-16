#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os
import itertools as it
from pathlib import Path

import numpy as np
import pytest

from sympde.topology import Domain, ScalarFunctionSpace, Line, Square, Cube
from psydac.api.discretization import discretize
from psydac.fem.basic import FemField
from psydac.mapping.discrete import NurbsMapping
from psydac.core.bsplines import cell_index, basis_ders_on_irregular_grid, breakpoints, elements_spans, basis_ders_on_quad_grid

from psydac.core.field_evaluation_kernels import (eval_fields_1d_no_weights, eval_fields_2d_no_weights, eval_fields_3d_no_weights,
                                                  eval_fields_1d_irregular_no_weights, eval_fields_2d_irregular_no_weights, 
                                                  eval_fields_3d_irregular_no_weights,
                                                  eval_fields_1d_weighted, eval_fields_2d_weighted, eval_fields_3d_weighted,
                                                  eval_fields_1d_irregular_weighted, eval_fields_2d_irregular_weighted, 
                                                  eval_fields_3d_irregular_weighted,
                                                  eval_jacobians_2d, eval_jacobians_3d,
                                                  eval_jacobians_irregular_2d, eval_jacobians_irregular_3d,
                                                  eval_jacobians_2d_weights, eval_jacobians_3d_weights,
                                                  eval_jacobians_irregular_2d_weights, eval_jacobians_irregular_3d_weights,
                                                  eval_jacobians_inv_2d, eval_jacobians_inv_3d,
                                                  eval_jacobians_inv_irregular_2d, eval_jacobians_inv_irregular_3d,
                                                  eval_jacobians_inv_2d_weights, eval_jacobians_inv_3d_weights,
                                                  eval_jacobians_inv_irregular_2d_weights, eval_jacobians_inv_irregular_3d_weights,
                                                  eval_jac_det_2d, eval_jac_det_3d,
                                                  eval_jac_det_irregular_2d, eval_jac_det_irregular_3d,
                                                  eval_jac_det_2d_weights, eval_jac_det_3d_weights,
                                                  eval_jac_det_irregular_2d_weights, eval_jac_det_irregular_3d_weights,
                                                  pushforward_2d_l2, pushforward_3d_l2,
                                                  pushforward_2d_hdiv, pushforward_3d_hdiv,
                                                  pushforward_2d_hcurl, pushforward_3d_hcurl)



# Get the mesh directory
import psydac.cad.mesh as mesh_mod
mesh_dir = Path(mesh_mod.__file__).parent

# Tolerance for testing float equality
RTOL = 1e-14
ATOL = 1e-12

@pytest.mark.parametrize('geometry', ('identity_2d.h5', 'identity_3d.h5', 'bent_pipe.h5',
                                      'collela_2d.h5', 'collela_3d.h5'))
@pytest.mark.parametrize('npts_per_cell', [2, 3, 4])
def test_regular_jacobians(geometry, npts_per_cell):
    filename = os.path.join(mesh_dir, geometry)
    domain = Domain.from_file(filename)

    # Discretization
    domainh = discretize(domain, filename=filename)
    mapping = list(domainh.mappings.values())[0]
    ldim = mapping.ldim
    space_h = mapping.space
    # Preprocessing
    is_nurbs = isinstance(mapping, NurbsMapping)

    ncells = tuple(len(space_h.breaks[i]) - 1 for i in range(ldim))
    regular_grid = [np.concatenate(
                    [np.random.random(size=npts_per_cell) * (
                                                             space_h.breaks[i][j + 1]
                                                             - space_h.breaks[i][j]
                                                            )
                                                             + space_h.breaks[i][j]
                     for j in range(ncells[i])
                    ]
                    )
                    for i in range(ldim)]

    # Direct API
    if ldim == 2:
        jacobian_matrix_direct = np.array([[mapping.jacobian(e1, e2) for e2 in regular_grid[1]] for e1 in regular_grid[0]])

    if ldim == 3:
        jacobian_matrix_direct = np.array([[[mapping.jacobian(e1, e2, e3)
                                                for e3 in regular_grid[2]]
                                            for e2 in regular_grid[1]]
                                            for e1 in regular_grid[0]])

    # Mapping related quantities through kernel functions
    degree = space_h.degree
    knots = [space_h.spaces[i].knots for i in range(ldim)]

    global_basis = [basis_ders_on_quad_grid(knots[i],
                                            degree[i],
                                            np.reshape(regular_grid[i], (ncells[i], npts_per_cell)),
                                            1,
                                            space_h.spaces[i].basis) for i in range(ldim)
                    ]
    v = space_h.coeff_space
    global_spans = [elements_spans(knots[i], degree[i]) - v.starts[i] + v.shifts[i] * v.pads[i] for i in range(ldim)]

    shape_grid = tuple(ncells[i] * npts_per_cell for i in range(ldim))
    n_eval_points = (npts_per_cell,) * ldim
    jac_mats = np.zeros(shape_grid + (ldim, ldim))
    inv_jac_mats = np.zeros(shape_grid + (ldim, ldim))
    jac_dets = np.zeros(shape_grid)

    if is_nurbs:
        global_arr_weights = mapping._weights_field.coeffs._data

        if ldim == 2:
            global_arr_x = mapping._fields[0].coeffs._data
            global_arr_y = mapping._fields[1].coeffs._data

            # Compute the jacobians
            eval_jacobians_2d_weights(*ncells, *degree, *n_eval_points, *global_basis, *global_spans,
                                      global_arr_x, global_arr_y, global_arr_weights, jac_mats)

            # Compute the inverses of the jacobians
            eval_jacobians_inv_2d_weights(*ncells, *degree, *n_eval_points, *global_basis, *global_spans,
                                          global_arr_x, global_arr_y,
                                          global_arr_weights, inv_jac_mats)

            # Compute the determinant of the jacobians
            eval_jac_det_2d_weights(*ncells, *degree, *n_eval_points, *global_basis, *global_spans,
                                    global_arr_x, global_arr_y,
                                    global_arr_weights, jac_dets)

        if ldim == 3:
            global_arr_x = mapping._fields[0].coeffs._data
            global_arr_y = mapping._fields[1].coeffs._data
            global_arr_z = mapping._fields[2].coeffs._data

            # Compute the jacobians
            eval_jacobians_3d_weights(*ncells, *degree, *n_eval_points, *global_basis, *global_spans,
                                      global_arr_x, global_arr_y, global_arr_z,
                                      global_arr_weights, jac_mats)

            # Compute the inverses of the jacobians
            eval_jacobians_inv_3d_weights(*ncells, *degree, *n_eval_points, *global_basis, *global_spans,
                                          global_arr_x, global_arr_y, global_arr_z,
                                          global_arr_weights, inv_jac_mats)

            # Compute the determinant of the jacobians
            eval_jac_det_3d_weights(*ncells, *degree, *n_eval_points, *global_basis, *global_spans,
                                    global_arr_x, global_arr_y, global_arr_z,
                                    global_arr_weights, jac_dets)
    else:
        if ldim == 2:
            global_arr_x = mapping._fields[0].coeffs._data
            global_arr_y = mapping._fields[1].coeffs._data
            # Compute the jacobians
            eval_jacobians_2d(*ncells, *degree, *n_eval_points, *global_basis, *global_spans,
                              global_arr_x, global_arr_y, jac_mats)

            # Compute the inverses of the jacobians
            eval_jacobians_inv_2d(*ncells, *degree, *n_eval_points, *global_basis, *global_spans,
                                  global_arr_x, global_arr_y, inv_jac_mats)

            # Compute the determinant of the jacobians
            eval_jac_det_2d(*ncells, *degree, *n_eval_points, *global_basis, *global_spans,
                            global_arr_x, global_arr_y,
                            jac_dets)

        if ldim == 3:
            global_arr_x = mapping._fields[0].coeffs._data
            global_arr_y = mapping._fields[1].coeffs._data
            global_arr_z = mapping._fields[2].coeffs._data

            # Compute the jacobians
            eval_jacobians_3d(*ncells, *degree, *n_eval_points, *global_basis, *global_spans,
                              global_arr_x, global_arr_y, global_arr_z, jac_mats)

            # Compute the inverses of the jacobians
            eval_jacobians_inv_3d(*ncells, *degree, *n_eval_points, *global_basis, *global_spans,
                                  global_arr_x, global_arr_y, global_arr_z, inv_jac_mats)

            # Compute the determinant of the jacobians
            eval_jac_det_3d(*ncells, *degree, *n_eval_points, *global_basis, *global_spans,
                            global_arr_x, global_arr_y, global_arr_z,
                            jac_dets)
    print(np.max(jacobian_matrix_direct - jac_mats))
    assert np.allclose(jacobian_matrix_direct, jac_mats, atol=ATOL, rtol=RTOL)

    if ldim == 2:
        for i, j in it.product(range(jac_mats.shape[0]), range(jac_mats.shape[1])):
            # Assert that the computed inverse is the inverse.
            assert np.allclose(np.dot(jac_mats[i, j], inv_jac_mats[i, j]), np.eye(ldim), atol=ATOL, rtol=RTOL)
            # Assert that the computed Jacobian determinant is the Jacobian determinant
            assert np.allclose(np.linalg.det(jac_mats[i, j]), jac_dets[i, j], atol=ATOL, rtol=RTOL)

    if ldim == 3:
        for i, j, k in it.product(range(jac_mats.shape[0]), range(jac_mats.shape[1]), range(jac_mats.shape[2])):
            # Assert that the computed inverse is the inverse.
            assert np.allclose(np.dot(jac_mats[i, j, k], inv_jac_mats[i, j, k]), np.eye(ldim), atol=ATOL, rtol=RTOL)
            # Assert that the computed Jacobian determinant is the Jacobian determinant
            assert np.allclose(np.linalg.det(jac_mats[i, j, k]), jac_dets[i, j, k], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize('geometry', ('identity_2d.h5', 'identity_3d.h5', 'bent_pipe.h5',
                                      'collela_2d.h5', 'collela_3d.h5'))
@pytest.mark.parametrize('npts', [2, 10, 20])
def test_irregular_jacobians(geometry, npts):
    filename = os.path.join(mesh_dir, geometry)
    domain = Domain.from_file(filename)

    # Discretization
    domainh = discretize(domain, filename=filename)
    mapping = list(domainh.mappings.values())[0]
    ldim = mapping.ldim
    space_h = mapping.space
    # Preprocessing
    is_nurbs = isinstance(mapping, NurbsMapping)

    irregular_grid = [np.random.random(npts) for i in range(ldim)]

    # Direct API
    if ldim == 2:
        jacobian_matrix_direct = np.array([[mapping.jacobian(e1, e2) for e2 in irregular_grid[1]] for e1 in irregular_grid[0]])

    if ldim == 3:
        jacobian_matrix_direct = np.array([[[mapping.jacobian(e1, e2, e3)
                                                for e3 in irregular_grid[2]]
                                            for e2 in irregular_grid[1]]
                                            for e1 in irregular_grid[0]])

    # Mapping related quantities through kernel functions
    degree = space_h.degree
    knots = [space_h.spaces[i].knots for i in range(ldim)]
    cell_indexes =[cell_index(space_h.breaks[i], irregular_grid[i]) for i in range(ldim)]
    global_basis = [basis_ders_on_irregular_grid(knots[i],
                                                 degree[i],
                                                 irregular_grid[i],
                                                 cell_indexes[i],
                                                 1,
                                                 space_h.spaces[i].basis) for i in range(ldim)
                    ]
    v = space_h.coeff_space
    global_spans = [elements_spans(knots[i], degree[i]) - v.starts[i] + v.shifts[i] * v.pads[i] for i in range(ldim)]

    npts = (npts,) * ldim

    shape_grid = npts
    jac_mats = np.zeros(shape_grid + (ldim, ldim))
    inv_jac_mats = np.zeros(shape_grid + (ldim, ldim))
    jac_dets = np.zeros(shape_grid)

    if is_nurbs:
        global_arr_weights = mapping._weights_field.coeffs._data

        if ldim == 2:
            global_arr_x = mapping._fields[0].coeffs._data
            global_arr_y = mapping._fields[1].coeffs._data

            # Compute the jacobians
            eval_jacobians_irregular_2d_weights(*npts, *degree, *cell_indexes, *global_basis, *global_spans,
                                                global_arr_x, global_arr_y, global_arr_weights, jac_mats)

            # Compute the inverses of the jacobians
            eval_jacobians_inv_irregular_2d_weights(*npts, *degree, *cell_indexes, *global_basis, *global_spans,
                                                    global_arr_x, global_arr_y,
                                                    global_arr_weights, inv_jac_mats)

            # Compute the determinant of the jacobians
            eval_jac_det_irregular_2d_weights(*npts, *degree, *cell_indexes, *global_basis, *global_spans,
                                              global_arr_x, global_arr_y,
                                              global_arr_weights, jac_dets)

        if ldim == 3:
            global_arr_x = mapping._fields[0].coeffs._data
            global_arr_y = mapping._fields[1].coeffs._data
            global_arr_z = mapping._fields[2].coeffs._data

            # Compute the jacobians
            eval_jacobians_3d_weights(*npts, *degree, *cell_indexes, *global_basis, *global_spans,
                                      global_arr_x, global_arr_y, global_arr_z,
                                      global_arr_weights, jac_mats)

            # Compute the inverses of the jacobians
            eval_jacobians_inv_irregular_3d_weights(*npts, *degree, *cell_indexes, *global_basis, *global_spans,
                                                    global_arr_x, global_arr_y, global_arr_z,
                                                    global_arr_weights, inv_jac_mats)

            # Compute the determinant of the jacobians
            eval_jac_det_irregular_3d_weights(*npts, *degree, *cell_indexes, *global_basis, *global_spans,
                                              global_arr_x, global_arr_y, global_arr_z,
                                              global_arr_weights, jac_dets)
    else:
        if ldim == 2:
            global_arr_x = mapping._fields[0].coeffs._data
            global_arr_y = mapping._fields[1].coeffs._data
            # Compute the jacobians
            eval_jacobians_irregular_2d(*npts, *degree, *cell_indexes, *global_basis, *global_spans,
                                        global_arr_x, global_arr_y, jac_mats)

            # Compute the inverses of the jacobians
            eval_jacobians_inv_irregular_2d(*npts, *degree, *cell_indexes, *global_basis, *global_spans,
                                            global_arr_x, global_arr_y, inv_jac_mats)

            # Compute the determinant of the jacobians
            eval_jac_det_irregular_2d(*npts, *degree, *cell_indexes, *global_basis, *global_spans,
                                      global_arr_x, global_arr_y,
                                      jac_dets)

        if ldim == 3:
            global_arr_x = mapping._fields[0].coeffs._data
            global_arr_y = mapping._fields[1].coeffs._data
            global_arr_z = mapping._fields[2].coeffs._data

            # Compute the jacobians
            eval_jacobians_irregular_3d(*npts, *degree, *cell_indexes, *global_basis, *global_spans,
                                        global_arr_x, global_arr_y, global_arr_z, jac_mats)

            # Compute the inverses of the jacobians
            eval_jacobians_inv_irregular_3d(*npts, *degree, *cell_indexes, *global_basis, *global_spans,
                                            global_arr_x, global_arr_y, global_arr_z, inv_jac_mats)

            # Compute the determinant of the jacobians
            eval_jac_det_irregular_3d(*npts, *degree, *cell_indexes, *global_basis, *global_spans,
                                      global_arr_x, global_arr_y, global_arr_z,
                                      jac_dets)
    print(np.max(jacobian_matrix_direct - jac_mats))
    assert np.allclose(jacobian_matrix_direct, jac_mats, atol=ATOL, rtol=RTOL)

    if ldim == 2:
        for i, j in it.product(range(jac_mats.shape[0]), range(jac_mats.shape[1])):
            # Assert that the computed inverse is the inverse.
            assert np.allclose(np.dot(jac_mats[i, j], inv_jac_mats[i, j]), np.eye(ldim), atol=ATOL, rtol=RTOL)
            # Assert that the computed Jacobian determinant is the Jacobian determinant
            assert np.allclose(np.linalg.det(jac_mats[i, j]), jac_dets[i, j], atol=ATOL, rtol=RTOL)

    if ldim == 3:
        for i, j, k in it.product(range(jac_mats.shape[0]), range(jac_mats.shape[1]), range(jac_mats.shape[2])):
            # Assert that the computed inverse is the inverse.
            assert np.allclose(np.dot(jac_mats[i, j, k], inv_jac_mats[i, j, k]), np.eye(ldim), atol=ATOL, rtol=RTOL)
            # Assert that the computed Jacobian determinant is the Jacobian determinant
            assert np.allclose(np.linalg.det(jac_mats[i, j, k]), jac_dets[i, j, k], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("knots, ldim, degree",
    [([np.sort(np.concatenate((np.zeros(3), np.random.random(9), np.ones(3))))],                   1, [2]),
     ([np.sort(np.concatenate((np.zeros(4), np.random.random(9), np.ones(4))))],                   1, [3]),
     ([np.sort(np.concatenate((np.zeros(3), np.random.random(9), np.ones(3)))) for i in range(2)], 2, [2] * 2),
     ([np.sort(np.concatenate((np.zeros(4), np.random.random(9), np.ones(4)))) for i in range(2)], 2, [3] * 2),
     ([np.sort(np.concatenate((np.zeros(3), np.random.random(9), np.ones(3)))) for i in range(3)], 3, [2] * 3),
     ([np.sort(np.concatenate((np.zeros(4), np.random.random(9), np.ones(4)))) for i in range(3)], 3, [3] * 3),
     ([np.array([0.0] * 3 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 3)],     1, [2]),
     ([np.array([0.0] * 4 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 4)],     1, [3]),
     ([np.array([0.0] * 3 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 3)] * 2, 2, [2] * 2),
     ([np.array([0.0] * 4 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 4)] * 2, 2, [3] * 2),
     ([np.array([0.0] * 3 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 3)] * 3, 3, [2] * 3),
     ([np.array([0.0] * 4 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 4)] * 3, 3, [3] * 3),
     ([np.array([0.0] * 4 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 4),
       np.array([0.0] * 3 + [1.0] * 3)],
      2,
      [3, 2]),
     ([np.array([0.0] * 3 + [1.0] * 3),
       np.array([0.0] * 4 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 4)],
      2,
      [2, 3]),
     ([np.array([0.0] * 3 + [1.0] * 3),
       np.array([0.0] * 4 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 4),
       np.array([0.0] * 4 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 4)],
      3,
      [2, 3, 3]),
     ([np.array([0.0] * 4 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 4),
       np.array([0.0] * 3 + [1.0] * 3),
       np.array([0.0] * 4 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 4)],
      3,
      [3, 2, 3]),
     ([np.array([0.0] * 4 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 4),
       np.array([0.0] * 4 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 4),
       np.array([0.0] * 3 + [1.0] * 3)],
      3,
      [3, 3, 2]),
    ]
)
@pytest.mark.parametrize("npts_per_cell", [2, 3, 4])
def test_regular_evaluations(knots, ldim, degree, npts_per_cell):
    if ldim == 1:
        domain = Line()
    elif ldim == 2:
        domain = Square()
    else:
        domain = Cube()
    space = ScalarFunctionSpace('space', domain)

    ncells = [len(breakpoints(knots[i], degree[i])) - 1 for i in range(ldim)]

    domain_h = discretize(domain, ncells=ncells)

    space_h = discretize(space, domain_h, knots=knots, degree=degree)

    field = FemField(space_h)
    weight = FemField(space_h)

    field.coeffs._data[:] = np.random.random(field.coeffs._data.shape)
    weight.coeffs._data[:] = np.random.random(weight.coeffs._data.shape)

    regular_grid = [np.concatenate(
                    [np.random.random(size=npts_per_cell) * (
                                                             space_h.breaks[i][j + 1]
                                                             - space_h.breaks[i][j]
                                                            )
                                                             + space_h.breaks[i][j]
                     for j in range(ncells[i])
                    ]
                    )
                    for i in range(ldim)]

    # Direct eval
    if ldim == 1:
        # No weights
        f_direct = np.array([space_h.eval_fields([e1], field) for e1 in regular_grid[0]])

        # Weighted
        f_direct_w = np.array([np.array(space_h.eval_fields([e1], field, weights=weight))
                                / np.array(space_h.eval_fields([e1], weight))
                                for e1 in regular_grid[0]])

    if ldim == 2:
        # No weights
        f_direct = np.array([[space_h.eval_fields([e1, e2], field) for e2 in regular_grid[1]] for e1 in regular_grid[0]])

        # Weighted
        f_direct_w = np.array([[np.array(space_h.eval_fields([e1, e2], field, weights=weight))
                                / np.array(space_h.eval_fields([e1, e2], weight))
                                for e2 in regular_grid[1]]
                                for e1 in regular_grid[0]])

    if ldim == 3:
        # No weights
        f_direct = np.array([[[space_h.eval_fields([e1, e2, e3], field)
                                for e3 in regular_grid[2]]
                                for e2 in regular_grid[1]]
                                for e1 in regular_grid[0]])

        # Weighted
        f_direct_w = np.array([[[np.array(space_h.eval_fields([e1, e2, e3], field, weights=weight))
                                    / np.array(space_h.eval_fields([e1, e2, e3], weight))
                                for e3 in regular_grid[2]]
                                for e2 in regular_grid[1]]
                                for e1 in regular_grid[0]])

    global_basis = [basis_ders_on_quad_grid(knots[i],
                                            degree[i],
                                            np.reshape(regular_grid[i], (ncells[i], npts_per_cell)),
                                            0,
                                            space_h.spaces[i].basis) for i in range(ldim)
                    ]
    v = space_h.coeff_space
    global_spans = [elements_spans(knots[i], degree[i]) - v.starts[i] + v.shifts[i] * v.pads[i] for i in range(ldim)]

    n_eval_points = (npts_per_cell,) * ldim
    out_field = np.zeros(tuple(ncells[i] * n_eval_points[i] for i in range(ldim)) + (1,))
    out_field_w = np.zeros_like(out_field)

    global_arr_field = field.coeffs._data.reshape(field.coeffs._data.shape + (1,))
    global_arr_w = weight.coeffs._data

    if ldim == 1:
        # No weights
        eval_fields_1d_no_weights(*ncells, *degree, *n_eval_points, *global_basis,
                                  *global_spans, global_arr_field, out_field)

        # Weighted
        eval_fields_1d_weighted(*ncells, *degree, *n_eval_points, *global_basis,
                                *global_spans, global_arr_field, global_arr_w, out_field_w)

    if ldim == 2:
        # No weights
        eval_fields_2d_no_weights(*ncells, *degree, *n_eval_points, *global_basis,
                                  *global_spans, global_arr_field, out_field)

        # Weighted
        eval_fields_2d_weighted(*ncells, *degree, *n_eval_points, *global_basis,
                                *global_spans, global_arr_field, global_arr_w, out_field_w)

    if ldim == 3:
        # No weights
        eval_fields_3d_no_weights(*ncells, *degree, *n_eval_points, *global_basis,
                                  *global_spans, global_arr_field, out_field)

        # Weighted
        eval_fields_3d_weighted(*ncells, *degree, *n_eval_points, *global_basis,
                                *global_spans, global_arr_field, global_arr_w, out_field_w)

    assert np.allclose(out_field, f_direct, atol=ATOL, rtol=RTOL)
    assert np.allclose(out_field_w, f_direct_w, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("knots, ldim, degree",
    [([np.sort(np.concatenate((np.zeros(3), np.random.random(9), np.ones(3))))],                   1, [2]),
     ([np.sort(np.concatenate((np.zeros(4), np.random.random(9), np.ones(4))))],                   1, [3]),
     ([np.sort(np.concatenate((np.zeros(3), np.random.random(9), np.ones(3)))) for i in range(2)], 2, [2] * 2),
     ([np.sort(np.concatenate((np.zeros(4), np.random.random(9), np.ones(4)))) for i in range(2)], 2, [3] * 2),
     ([np.sort(np.concatenate((np.zeros(3), np.random.random(9), np.ones(3)))) for i in range(3)], 3, [2] * 3),
     ([np.sort(np.concatenate((np.zeros(4), np.random.random(9), np.ones(4)))) for i in range(3)], 3, [3] * 3),
     ([np.array([0.0] * 3 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 3)],     1, [2]),
     ([np.array([0.0] * 4 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 4)],     1, [3]),
     ([np.array([0.0] * 3 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 3)] * 2, 2, [2] * 2),
     ([np.array([0.0] * 4 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 4)] * 2, 2, [3] * 2),
     ([np.array([0.0] * 3 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 3)] * 3, 3, [2] * 3),
     ([np.array([0.0] * 4 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 4)] * 3, 3, [3] * 3),
     ([np.array([0.0] * 4 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 4),
       np.array([0.0] * 3 + [1.0] * 3)],
      2,
      [3, 2]),
     ([np.array([0.0] * 3 + [1.0] * 3),
       np.array([0.0] * 4 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 4)],
      2,
      [2, 3]),
     ([np.array([0.0] * 3 + [1.0] * 3),
       np.array([0.0] * 4 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 4),
       np.array([0.0] * 4 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 4)],
      3,
      [2, 3, 3]),
     ([np.array([0.0] * 4 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 4),
       np.array([0.0] * 3 + [1.0] * 3),
       np.array([0.0] * 4 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 4)],
      3,
      [3, 2, 3]),
     ([np.array([0.0] * 4 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 4),
       np.array([0.0] * 4 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 4),
       np.array([0.0] * 3 + [1.0] * 3)],
      3,
      [3, 3, 2]),
    ]
)
@pytest.mark.parametrize('npts', [2, 10, 20])
def test_irregular_evaluations(knots, ldim, degree, npts):
    if ldim == 1:
        domain = Line()
    elif ldim == 2:
        domain = Square()
    else:
        domain = Cube()
    space = ScalarFunctionSpace('space', domain)

    ncells = [len(breakpoints(knots[i], degree[i])) - 1 for i in range(ldim)]

    domain_h = discretize(domain, ncells=ncells)

    space_h = discretize(space, domain_h, knots=knots, degree=degree)

    field = FemField(space_h)
    weight = FemField(space_h)

    field.coeffs._data[:] = np.random.random(field.coeffs._data.shape)
    weight.coeffs._data[:] = np.random.random(weight.coeffs._data.shape)

    irregular_grid = [np.random.random(npts) for i in range(ldim)]

    for i in range(ldim):
        j_left = np.random.randint(low=0, high=len(irregular_grid[i]))
        j_right = np.random.randint(low=0, high=len(irregular_grid[i]))
        j_interior = np.random.randint(low=0, high=len(irregular_grid[i]) - 1)

        # left boundary inserted at j_left
        irregular_grid[i][j_left] = space_h.breaks[i][0]
        # right boundary inserted at j_right
        irregular_grid[i][j_right] = space_h.breaks[i][-1]

        try:
            j_bk = np.random.randint(low=1, high=len(space_h.breaks[i]) - 1)
            # random interior breakpoint inserted at j_interior and j_interior + 1
            irregular_grid[i][j_interior:j_interior+2] = space_h.breaks[i][j_bk]
        except ValueError:
            pass

    # Direct eval
    if ldim == 1:
        # No weights
        f_direct = np.array([space_h.eval_fields([e1], field) for e1 in irregular_grid[0]])

        # Weighted
        f_direct_w = np.array([np.array(space_h.eval_fields([e1], field, weights=weight))
                                / np.array(space_h.eval_fields([e1], weight))
                                for e1 in irregular_grid[0]])
        
    if ldim == 2:
        # No weights
        f_direct = np.array([[space_h.eval_fields([e1, e2], field) for e2 in irregular_grid[1]] for e1 in irregular_grid[0]])

        # Weighted
        f_direct_w = np.array([[np.array(space_h.eval_fields([e1, e2], field, weights=weight))
                                / np.array(space_h.eval_fields([e1, e2], weight))
                                for e2 in irregular_grid[1]]
                                for e1 in irregular_grid[0]])

    if ldim == 3:
        # No weights
        f_direct = np.array([[[space_h.eval_fields([e1, e2, e3], field)
                                for e3 in irregular_grid[2]]
                                for e2 in irregular_grid[1]]
                                for e1 in irregular_grid[0]])

        # Weighted
        f_direct_w = np.array([[[np.array(space_h.eval_fields([e1, e2, e3], field, weights=weight))
                                    / np.array(space_h.eval_fields([e1, e2, e3], weight))
                                for e3 in irregular_grid[2]]
                                for e2 in irregular_grid[1]]
                                for e1 in irregular_grid[0]])

    cell_indexes = [cell_index(space_h.breaks[i], irregular_grid[i]) for i in range(ldim)]
    global_basis = [basis_ders_on_irregular_grid(knots[i],
                                                 degree[i],
                                                 irregular_grid[i],
                                                 cell_indexes[i],
                                                 0,
                                                 space_h.spaces[i].basis) for i in range(ldim)
                    ]
    v = space_h.coeff_space
    global_spans = [elements_spans(knots[i], degree[i]) - v.starts[i] + v.shifts[i] * v.pads[i] for i in range(ldim)]

    npts = (npts,) * ldim

    out_field = np.zeros(npts + (1,))
    out_field_w = np.zeros_like(out_field)

    global_arr_field = field.coeffs._data.reshape(field.coeffs._data.shape + (1,))
    global_arr_w = weight.coeffs._data

    if ldim == 1:
        # No weights
        eval_fields_1d_irregular_no_weights(*npts,*degree, *cell_indexes, *global_basis,
                                            *global_spans, global_arr_field, out_field)

        # Weighted
        eval_fields_1d_irregular_weighted(*npts, *degree, *cell_indexes, *global_basis,
                                          *global_spans, global_arr_field, global_arr_w, out_field_w)

    if ldim == 2:
        # No weights
        eval_fields_2d_irregular_no_weights(*npts,*degree, *cell_indexes, *global_basis,
                                            *global_spans, global_arr_field, out_field)

        # Weighted
        eval_fields_2d_irregular_weighted(*npts, *degree, *cell_indexes, *global_basis,
                                          *global_spans, global_arr_field, global_arr_w, out_field_w)

    if ldim == 3:
        # No weights
        eval_fields_3d_irregular_no_weights(*npts, *degree, *cell_indexes, *global_basis,
                                            *global_spans, global_arr_field, out_field)

        # Weighted
        eval_fields_3d_irregular_weighted(*npts, *degree, *cell_indexes, *global_basis,
                                          *global_spans, global_arr_field, global_arr_w, out_field_w)

    assert np.allclose(out_field, f_direct, atol=ATOL, rtol=RTOL)
    assert np.allclose(out_field_w, f_direct_w, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize('jac_det, ldim, field_to_push', [(np.ones((5, 5)), 2, np.ones((5, 5, 1))),
                                                          (np.ones((5, 5, 5)), 3, np.ones((5, 5, 5, 1))),
                                                          (np.random.rand(5, 5), 2, np.random.rand(5, 5, 1)),
                                                          (np.random.rand(5, 5, 5), 3, np.random.rand(5, 5, 5, 1))])
def test_pushforwards_l2(ldim, jac_det, field_to_push):
    expected = field_to_push[..., 0] / jac_det
    out = np.zeros_like(field_to_push)
    if ldim == 2:
        pushforward_2d_l2(field_to_push, jac_det, out)
    if ldim == 3:
        pushforward_3d_l2(field_to_push, jac_det, out)

    assert np.allclose(expected, out[..., 0], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize('ldim', (2, 3))
def test_pushforwards_hdiv(ldim):
    jacobians = np.full((5,) * ldim + (ldim, ldim), np.eye(ldim))
    metric_dets = np.full((5,) * ldim, 1.0)
    field_to_push = np.random.rand(ldim, *((5, ) * ldim), 1)
    expected = np.moveaxis(field_to_push, -1, 0)
    out = np.zeros(expected.shape)
    if ldim == 2:
        pushforward_2d_hdiv(field_to_push, jacobians, metric_dets, out)
    if ldim == 3:
        pushforward_3d_hdiv(field_to_push, jacobians, metric_dets, out)

    assert np.allclose(expected, out, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize('ldim', (2, 3))
def test_pushforwards_hcurl(ldim):
    inv_jacobians = np.full((5,) * ldim + (ldim, ldim), np.eye(ldim))
    field_to_push = np.random.rand(ldim, *((5, ) * ldim), 1)
    expected = np.moveaxis(field_to_push, -1, 0)
    out = np.zeros(expected.shape)

    if ldim == 2:
        pushforward_2d_hcurl(field_to_push, inv_jacobians, out)
    if ldim == 3:
        pushforward_3d_hcurl(field_to_push, inv_jacobians, out)

    assert np.allclose(expected, out, atol=ATOL, rtol=RTOL)
