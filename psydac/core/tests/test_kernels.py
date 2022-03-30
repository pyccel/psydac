import numpy as np
import pytest
import os
import itertools as it

from sympde.topology import Domain, ScalarFunctionSpace, Square, Cube
from psydac.api.discretization import discretize
from psydac.utilities.utils import refine_array_1d
from psydac.fem.basic import FemField
from psydac.mapping.discrete import NurbsMapping
from psydac.core.bsplines import cell_index, basis_ders_on_irregular_grid, breakpoints, elements_spans
from psydac.core.kernels import (eval_fields_2d_no_weights, eval_fields_3d_no_weights,
                                 eval_fields_2d_irregular_no_weights, eval_fields_3d_irregular_no_weights,
                                 eval_fields_2d_weighted, eval_fields_3d_weighted,
                                 eval_fields_2d_irregular_weighted, eval_fields_3d_irregular_weighted,
                                 eval_jacobians_2d, eval_jacobians_3d, eval_jacobians_2d_weights,
                                 eval_jacobians_3d_weights, eval_jacobians_inv_2d_weights,
                                 eval_jacobians_inv_3d_weights, eval_jacobians_inv_2d, eval_jacobians_inv_3d,
                                 eval_jac_det_2d_weights, eval_jac_det_3d_weights,
                                 eval_jac_det_2d, eval_jac_det_3d,
                                 pushforward_2d_l2, pushforward_3d_l2,
                                 pushforward_2d_hdiv, pushforward_3d_hdiv,
                                 pushforward_2d_hcurl, pushforward_3d_hcurl)
        


# Get mesh directory
try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']
except KeyError:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')


@pytest.mark.parametrize('geometry', ('identity_2d.h5', 'identity_3d.h5', 'bent_pipe.h5',
                                      'collela_2d.h5', 'collela_3d.h5'))
@pytest.mark.parametrize('refine', (1, 2))
@pytest.mark.parametrize('kind', ('hcurl', 'hdiv', 'l2', 'h1'))
def test_regular_kernels(geometry, refine, kind):

    filename = os.path.join(mesh_dir, geometry)

    # SymPDE
    domain = Domain.from_file(filename)
    space = ScalarFunctionSpace('hcurl', domain, kind=kind)

    # Discretization
    domainh = discretize(domain, filename=filename)
    mapping = list(domainh.mappings.values())[0]
    ldim = mapping.ldim

    spaceh = discretize(space, domainh, degree=[2]*ldim)
    field = FemField(spaceh)
    weight = FemField(spaceh)

    if not field.coeffs.ghost_regions_in_sync:
        field.coeffs.update_ghost_regions()
    if not weight.coeffs.ghost_regions_in_sync:
        weight.coeffs.update_ghost_regions()


    # Giving random values
    if kind in ['hcurl', 'hdiv']:
        for i in range(ldim):
            field.fields[i].coeffs._data[:] = np.random.random(field.fields[i].coeffs._data.shape)
            weight.fields[i].coeffs._data[:] = np.random.random(weight.fields[i].coeffs._data.shape)
    else:
        field.coeffs._data[:] = np.random.random(field.coeffs._data.shape)
        weight.coeffs._data[:] = np.random.random(weight.coeffs._data.shape)

    # Preprocessing
    is_nurbs = isinstance(mapping, NurbsMapping)
    grid = []
    ncells = []
    for i in range(ldim):
        grid_i_initial = mapping.space.breaks[i]
        ncells.append(len(grid_i_initial) - 1)
        grid.append(np.asarray(refine_array_1d(grid_i_initial, refine, remove_duplicates=False)))

    n_eval_points = [refine + 1] * ldim
    shape_grid = tuple(grid[i].size for i in range(ldim))
    tensor_grid = [np.reshape(grid[i], (ncells[i], n_eval_points[i])) for i in range(ldim)]

    pads_m, \
    degree_m, \
    global_basis_m, \
    global_spans_m = mapping.space.preprocess_regular_tensor_grid(tensor_grid, der=1)

    if kind not in ['hcurl', 'hdiv']:
        pads_s, \
        degree_s, \
        global_basis_s, \
        global_spans_s = spaceh.preprocess_regular_tensor_grid(tensor_grid, der=0)

    # Direct API
    try:
        if ldim == 2:
            jacobian_matrix_direct = np.array([[mapping.jac_mat(e1, e2) for e2 in grid[1]] for e1 in grid[0]])

        if ldim == 3:
            jacobian_matrix_direct = np.array([[[mapping.jac_mat(e1, e2, e3)
                                                 for e3 in grid[2]]
                                                for e2 in grid[1]]
                                               for e1 in grid[0]])
    except NotImplementedError:
        pass
    if kind in ['hdiv', 'hcurl']:
        if ldim == 2:
            # No weights
            f_direct = np.array([[[spaceh.spaces[i].eval_fields([e1, e2], field.fields[i]) for i in range(ldim)]
                                  for e2 in grid[1]]
                                 for e1 in grid[0]])

            # Weighted
            f_direct_w = np.array([[[np.array(spaceh.spaces[i].eval_fields([e1, e2],
                                                                           field.fields[i],
                                                                           weights=weight.fields[i]))
                                     / np.array(spaceh.spaces[i].eval_fields([e1, e2], weight.fields[i]))
                                     for i in range(ldim)]
                                    for e2 in grid[1]]
                                   for e1 in grid[0]])


        if ldim == 3:
            # No weights
            f_direct = np.array([[[spaceh.eval_fields([e1, e2, e3], field)
                                   for e3 in grid[2]]
                                  for e2 in grid[1]]
                                 for e1 in grid[0]])

            # Weighted
            f_direct_w = np.array([[[np.array(spaceh.eval_fields([e1, e2, e3], field, weights=weight))
                                     / np.array(spaceh.eval_fields([e1, e2, e3], weight))
                                     for e3 in grid[2]]
                                    for e2 in grid[1]]
                                   for e1 in grid[0]])
    else:
        if ldim == 2:
            # No weights
            f_direct = np.array([[spaceh.eval_fields([e1, e2], field) for e2 in grid[1]] for e1 in grid[0]])

            # Weighted
            f_direct_w = np.array([[np.array(spaceh.eval_fields([e1, e2], field, weights=weight))
                                    / np.array(spaceh.eval_fields([e1, e2], weight))
                                    for e2 in grid[1]]
                                   for e1 in grid[0]])

        if ldim == 3:
            # No weights
            f_direct = np.array([[[spaceh.eval_fields([e1, e2, e3], field)
                                   for e3 in grid[2]]
                                  for e2 in grid[1]]
                                 for e1 in grid[0]])

            # Weighted
            f_direct_w = np.array([[[np.array(spaceh.eval_fields([e1, e2, e3], field, weights=weight))
                                     / np.array(spaceh.eval_fields([e1, e2, e3], weight))
                                     for e3 in grid[2]]
                                    for e2 in grid[1]]
                                   for e1 in grid[0]])

    # Mapping related quantities through kernel functions
    jac_mats = np.zeros(shape_grid + (ldim, ldim))
    inv_jac_mats = np.zeros(shape_grid + (ldim, ldim))
    jac_dets = np.zeros(shape_grid)

    if is_nurbs:
        global_arr_weights = mapping._weights_field.coeffs._data

        if ldim == 2:
            global_arr_x = mapping._fields[0].coeffs._data
            global_arr_y = mapping._fields[1].coeffs._data

            # Compute the jacobians
            eval_jacobians_2d_weights(*ncells, *pads_m, *degree_m, *n_eval_points, *global_basis_m, *global_spans_m,
                                      global_arr_x, global_arr_y, global_arr_weights, jac_mats)

            # Compute the inverses of the jacobians
            eval_jacobians_inv_2d_weights(*ncells, *pads_m, *degree_m, *n_eval_points, *global_basis_m, *global_spans_m,
                                          global_arr_x, global_arr_y,
                                          global_arr_weights, inv_jac_mats)

            # Compute the determinant of the jacobians
            eval_jac_det_2d_weights(*ncells, *pads_m, *degree_m, *n_eval_points, *global_basis_m, *global_spans_m,
                                    global_arr_x, global_arr_y,
                                    global_arr_weights, jac_dets)

        if ldim == 3:
            global_arr_x = mapping._fields[0].coeffs._data
            global_arr_y = mapping._fields[1].coeffs._data
            global_arr_z = mapping._fields[2].coeffs._data

            # Compute the jacobians
            eval_jacobians_3d_weights(*ncells, *pads_m, *degree_m, *n_eval_points, *global_basis_m, *global_spans_m,
                                      global_arr_x, global_arr_y, global_arr_z,
                                      global_arr_weights, jac_mats)

            # Compute the inverses of the jacobians
            eval_jacobians_inv_3d_weights(*ncells, *pads_m, *degree_m, *n_eval_points, *global_basis_m, *global_spans_m,
                                          global_arr_x, global_arr_y, global_arr_z,
                                          global_arr_weights, inv_jac_mats)

            # Compute the determinant of the jacobians
            eval_jac_det_3d_weights(*ncells, *pads_m, *degree_m, *n_eval_points, *global_basis_m, *global_spans_m,
                                    global_arr_x, global_arr_y, global_arr_z,
                                    global_arr_weights, jac_dets)
    else:
        if mapping.ldim == 2:
            global_arr_x = mapping._fields[0].coeffs._data
            global_arr_y = mapping._fields[1].coeffs._data

            # Compute the jacobians
            eval_jacobians_2d(*ncells, *pads_m, *degree_m, *n_eval_points, *global_basis_m, *global_spans_m,
                              global_arr_x, global_arr_y, jac_mats)

            # Compute the inverses of the jacobians
            eval_jacobians_inv_2d(*ncells, *pads_m, *degree_m, *n_eval_points, *global_basis_m, *global_spans_m,
                                  global_arr_x, global_arr_y, inv_jac_mats)

            # Compute the determinant of the jacobians
            eval_jac_det_2d(*ncells, *pads_m, *degree_m, *n_eval_points, *global_basis_m, *global_spans_m,
                            global_arr_x, global_arr_y,
                            jac_dets)

        if ldim == 3:
            global_arr_x = mapping._fields[0].coeffs._data
            global_arr_y = mapping._fields[1].coeffs._data
            global_arr_z = mapping._fields[2].coeffs._data

            # Compute the jacobians
            eval_jacobians_3d(*ncells, *pads_m, *degree_m, *n_eval_points, *global_basis_m, *global_spans_m,
                              global_arr_x, global_arr_y, global_arr_z, jac_mats)

            # Compute the inverses of the jacobians
            eval_jacobians_inv_3d(*ncells, *pads_m, *degree_m, *n_eval_points, *global_basis_m, *global_spans_m,
                                  global_arr_x, global_arr_y, global_arr_z, inv_jac_mats)

            # Compute the determinant of the jacobians
            eval_jac_det_3d(*ncells, *pads_m, *degree_m, *n_eval_points, *global_basis_m, *global_spans_m,
                            global_arr_x, global_arr_y, global_arr_z,
                            jac_dets)

    # Field related quantities through kernel functions
    if kind in ['hcurl', 'hdiv']:  # Product FemSpace
        out_field = np.zeros((ldim,) + shape_grid)
        out_field_w = np.zeros((ldim,) + shape_grid)

        global_arr_field = [field.fields[i].coeffs._data[:] for i in range(ldim)]
        global_arr_w = [weight.fields[i].coeffs._data[:] for i in range(ldim)]

        if ldim == 2:

            for i in range(2):
                pads_s, \
                degree_s, \
                global_basis_s, \
                global_spans_s = spaceh.spaces[i].preprocess_regular_tensor_grid(tensor_grid, der=0)
                eval_fields_2d_no_weights(*ncells, *pads_s, *degree_s, *n_eval_points, *global_basis_s, *global_spans_s,
                                          global_arr_field[i][:, :, None], out_field[i][:, :, None])

                eval_fields_2d_weighted(*ncells, *pads_s, *degree_s, *n_eval_points, *global_basis_s, *global_spans_s,
                                        global_arr_field[i][:, :, None], global_arr_w[i],
                                        out_field_w[i][:, :, None])

        if ldim == 3:
            for i in range(3):
                pads_s, \
                degree_s, \
                global_basis_s, \
                global_spans_s = spaceh.spaces[i].preprocess_regular_tensor_grid(tensor_grid, der=0)

                eval_fields_3d_no_weights(*ncells, *pads_s, *degree_s, *n_eval_points, *global_basis_s, *global_spans_s,
                                          global_arr_field[i][:, :, :, None], out_field[i][:, :, :, None])

                eval_fields_3d_weighted(*ncells, *pads_s, *degree_s, *n_eval_points, *global_basis_s, *global_spans_s,
                                        global_arr_field[i][:, :, :, None], global_arr_w[i],
                                        out_field_w[i][:, :, :, None])
    else:
        out_field = np.zeros(shape_grid + (1,))
        out_field_w = np.zeros(shape_grid + (1,))

        global_arr_field = field.coeffs._data.reshape(field.coeffs._data.shape + (1,))
        global_arr_w = weight.coeffs._data

        if ldim == 2:
            eval_fields_2d_no_weights(*ncells, *pads_s, *degree_s, *n_eval_points, *global_basis_s, *global_spans_s,
                                      global_arr_field, out_field)

            eval_fields_2d_weighted(*ncells, *pads_s, *degree_s, *n_eval_points, *global_basis_s, *global_spans_s,
                                    global_arr_field, global_arr_w, out_field_w)

        if ldim == 3:
            eval_fields_3d_no_weights(*ncells, *pads_s, *degree_s, *n_eval_points, *global_basis_s, *global_spans_s,
                                      global_arr_field, out_field)

            eval_fields_3d_weighted(*ncells, *pads_s, *degree_s, *n_eval_points, *global_basis_s, *global_spans_s,
                                    global_arr_field, global_arr_w, out_field_w)

    # First round of checks
    # Jacobian related arrays
    try:
        assert np.allclose(jacobian_matrix_direct, jac_mats)
    except NameError:
        pass

    if ldim == 2:
        for i, j in it.product(range(jac_mats.shape[0]), range(jac_mats.shape[1])):
            # Assert that the computed inverse is the inverse.
            assert np.allclose(np.dot(jac_mats[i, j], inv_jac_mats[i, j]), np.eye(ldim))
            # Assert that the computed Jacobian determinant is the Jacobian determinant
            assert np.allclose(np.linalg.det(jac_mats[i, j]), jac_dets[i, j])

    if ldim == 3:
        for i, j, k in it.product(range(jac_mats.shape[0]), range(jac_mats.shape[1]), range(jac_mats.shape[2])):
            # Assert that the computed inverse is the inverse.
            assert np.allclose(np.dot(jac_mats[i, j, k], inv_jac_mats[i, j, k]), np.eye(ldim))
            # Assert that the computed Jacobian determinant is the Jacobian determinant
            assert np.allclose(np.linalg.det(jac_mats[i, j, k]), jac_dets[i, j, k])

    # Field related arrays
    if kind in ['hdiv', 'hcurl']:
        assert np.allclose(f_direct[:, :, :, 0], np.moveaxis(out_field, 0, -1))
        assert np.allclose(f_direct_w[:, :, :, 0], np.moveaxis(out_field_w, 0, -1))
    else:
        assert np.allclose(f_direct, out_field)
        assert np.allclose(f_direct_w, out_field_w)


@pytest.mark.parametrize("knots, ldim, degree", 
    [([np.sort(np.concatenate((np.zeros(3), np.random.random(9), np.ones(3)))) for i in range(2)], 2, [2] * 2),
     ([np.sort(np.concatenate((np.zeros(4), np.random.random(9), np.ones(4)))) for i in range(2)], 2, [3] * 2),
     ([np.sort(np.concatenate((np.zeros(3), np.random.random(9), np.ones(3)))) for i in range(3)], 3, [2] * 3),
     ([np.sort(np.concatenate((np.zeros(4), np.random.random(9), np.ones(4)))) for i in range(3)], 3, [3] * 3),
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
def test_irregular_kernels(knots, ldim, degree):
    if ldim == 2:
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

    irregular_grid = [np.random.random(np.random.randint(low=5, high = 15)) for i in range(ldim)]
    
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
    global_basis = [basis_ders_on_irregular_grid(knots[i], degree[i], irregular_grid[i], cell_indexes[i], 0, space_h.spaces[i].basis) for i in range(ldim)]
    global_spans = [elements_spans(knots[i], degree[i]) for i in range(ldim)]
    pads = space_h.pads

    nps = tuple(irregular_grid[i].shape[0] for i in range(ldim))

    out_field = np.zeros(nps + (1,))
    out_field_w = np.zeros_like(out_field)

    global_arr_field = field.coeffs._data.reshape(field.coeffs._data.shape + (1,))
    global_arr_w = weight.coeffs._data

    if ldim == 2:
        # No weights
        eval_fields_2d_irregular_no_weights(*nps, *pads, *degree, *cell_indexes, *global_basis,
                                            *global_spans, global_arr_field, out_field)
        
        # Weighted
        eval_fields_2d_irregular_weighted(*nps, *pads, *degree, *cell_indexes, *global_basis,
                                          *global_spans, global_arr_field, global_arr_w, out_field_w)
    
    if ldim == 3:
        # No weights
        eval_fields_3d_irregular_no_weights(*nps, *pads, *degree, *cell_indexes, *global_basis,
                                            *global_spans, global_arr_field, out_field)
        
        # Weighted
        eval_fields_3d_irregular_weighted(*nps, *pads, *degree, *cell_indexes, *global_basis,
                                          *global_spans, global_arr_field, global_arr_w, out_field_w)
    
    print(f_direct.shape, out_field.shape)
    assert np.allclose(out_field, f_direct)
    assert np.allclose(out_field_w, f_direct_w)
            

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

    assert np.allclose(expected, out[..., 0])


@pytest.mark.parametrize('ldim', (2, 3))
def test_pushforwards_hdiv(ldim):
    jacobians = np.full((5,) * ldim + (ldim, ldim), np.eye(ldim))
    field_to_push = np.random.rand(ldim, *((5, ) * ldim), 1)
    expected = np.moveaxis(field_to_push, 0, -2)
    out = np.zeros(expected.shape)
    
    if ldim == 2:
        pushforward_2d_hdiv(field_to_push, jacobians, out)
    if ldim == 3:
        pushforward_3d_hdiv(field_to_push, jacobians, out)

    assert np.allclose(expected, out)


@pytest.mark.parametrize('ldim', (2, 3))
def test_pushforwards_hcurl(ldim):
    inv_jacobians = np.full((5,) * ldim + (ldim, ldim), np.eye(ldim))
    field_to_push = np.random.rand(ldim, *((5, ) * ldim), 1)
    expected = np.moveaxis(field_to_push, 0, -2)
    out = np.zeros(expected.shape)

    if ldim == 2:
        pushforward_2d_hcurl(field_to_push, inv_jacobians, out)
    if ldim == 3:
        pushforward_3d_hcurl(field_to_push, inv_jacobians, out)

    assert np.allclose(expected, out)
