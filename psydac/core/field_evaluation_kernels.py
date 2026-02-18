#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np
from typing import TypeVar

T = TypeVar('T', float, complex)
# =============================================================================
# Field evaluation functions
# =============================================================================
# -----------------------------------------------------------------------------
# 1: Regular tensor grid without weight
# -----------------------------------------------------------------------------
def eval_fields_3d_no_weights(nc1: int, nc2: int, nc3: int, f_p1: int, f_p2: int,
                              f_p3: int, k1: int, k2: int, k3: int, global_basis_1: 'float[:,:,:,:]',
                              global_basis_2: 'float[:,:,:,:]', global_basis_3: 'float[:,:,:,:]',
                              global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_spans_3: 'int[:]',
                              glob_arr_coeff: 'T[:,:,:,:]', out_fields: 'T[:,:,:,:]'):
    """
    Parameters
    ----------
    nc1: int
        Number of cells in the X1 direction
    nc2: int
        Number of cells in the X2 direction
    nc3: int
        Number of cells in the X3 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction
    f_p3: int
        Degree in the X3 direction

    k1: int
        Number of evaluation points in the X1 direction
    k2: int
        Number of evaluation points in the X2 direction
    k3: int
        Number of evaluation points in the X3 direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X1 direction
    global_basis_2: ndarray of floats
        Basis functions values at each cell and quadrature points in the X2 direction
    global_basis_3: ndarray of floats
        Basis functions values at each cell and quadrature points in the X3 direction

    global_spans_1: ndarray of ints
        Spans in the X1 direction
    global_spans_2: ndarray of ints
        Spans in the X2 direction
    global_spans_3: ndarray of ints
        Spans in the X3 direction

    glob_arr_coeff: ndarray of floats
        Coefficients of the fields in the X1,X2 and X3 directions

    out_fields: ndarray of floats
        Evaluated fields, filled with the correct values by the function
    """
    arr_coeff_fields = np.zeros_like(glob_arr_coeff, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3, out_fields.shape[3]))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]

            for i_cell_3 in range(nc3):
                span_3 = global_spans_3[i_cell_3]

                arr_coeff_fields[:, :, :, :] = glob_arr_coeff[span_1 - f_p1:1 + span_1,
                                                              span_2 - f_p2:1 + span_2,
                                                              span_3 - f_p3:1 + span_3,
                                                              :]

                for i_basis_1 in range(1 + f_p1):
                    for i_basis_2 in range(1 + f_p2):
                        for i_basis_3 in range(1 + f_p3):
                            coeff_fields = arr_coeff_fields[i_basis_1, i_basis_2, i_basis_3, :]

                            for i_quad_1 in range(k1):
                                spline_1 = global_basis_1[i_cell_1, i_basis_1, 0, i_quad_1]

                                for i_quad_2 in range(k2):
                                    spline_2 = global_basis_2[i_cell_2, i_basis_2, 0, i_quad_2]

                                    for i_quad_3 in range(k3):
                                        spline_3 = global_basis_3[i_cell_3, i_basis_3, 0, i_quad_3]

                                        spline = spline_1 * spline_2 * spline_3

                                        out_fields[i_cell_1 * k1 + i_quad_1,
                                                   i_cell_2 * k2 + i_quad_2,
                                                   i_cell_3 * k3 + i_quad_3,
                                                   :] += spline * coeff_fields


def eval_fields_2d_no_weights(nc1: int, nc2: int, f_p1: int, f_p2: int, k1: int, k2: int,
                              global_basis_1: 'float[:,:,:,:]', global_basis_2: 'float[:,:,:,:]',
                              global_spans_1: 'int[:]', global_spans_2: 'int[:]', glob_arr_coeff: 'T[:,:,:]',
                              out_fields: 'T[:,:,:]'):
    """
    Parameters
    ----------
    nc1: int
        Number of cells in the X1 direction
    nc2: int
        Number of cells in the X2 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction

    k1: int
        Number of evaluation points in the X1 direction
    k2: int
        Number of evaluation points in the X2 direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X1 direction
    global_basis_2: ndarray of floats
        Basis functions values at each cell and quadrature points in the X2 direction

    global_spans_1: ndarray of ints
        Spans in the X1 direction
    global_spans_2: ndarray of ints
        Spans in the X2 direction

    glob_arr_coeff: ndarray of floats
        Coefficients of the fields in the X1 and X2 directions

    out_fields: ndarray of floats
        Evaluated fields, filled with the correct values by the function
    """
    arr_coeff_fields = np.zeros_like(glob_arr_coeff, shape=(1 + f_p1, 1 + f_p2, out_fields.shape[2]))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]
            arr_coeff_fields[:, :, :] = glob_arr_coeff[span_1 - f_p1:1 + span_1,
                                                       span_2 - f_p2:1 + span_2,
                                                       :]
            for i_basis_1 in range(1 + f_p1):
                for i_basis_2 in range(1 + f_p2):
                    coeff_fields = arr_coeff_fields[i_basis_1, i_basis_2, :]

                    for i_quad_1 in range(k1):
                        spline_1 = global_basis_1[i_cell_1, i_basis_1, 0, i_quad_1]

                        for i_quad_2 in range(k2):
                            spline_2 = global_basis_2[i_cell_2, i_basis_2, 0, i_quad_2]

                            spline = spline_1 * spline_2

                            out_fields[i_cell_1 * k1 + i_quad_1,
                                       i_cell_2 * k2 + i_quad_2,
                                       :] += spline * coeff_fields


def eval_fields_1d_no_weights(nc1: int, f_p1: int, k1: int,
                              global_basis_1: 'float[:,:,:,:]',
                              global_spans_1: 'int[:]', 
                              glob_arr_coeff: 'T[:,:]',
                              out_fields: 'T[:,:]'):
    """
    Parameters
    ----------
    nc1: int
        Number of cells in the X1 direction

    f_p1: int
        Degree in the X1 direction

    k1: int
        Number of evaluation points in the X1 direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X1 direction

    global_spans_1: ndarray of ints
        Spans in the X1 direction

    glob_arr_coeff: ndarray of floats
        Coefficients of the 1D fields, stored in a 2D array.
        The first dimension loops over the coefficients, and
        the second dimension over the different fields.

    out_fields: ndarray of floats
        Evaluated fields, filled with the correct values by the function
    """
    arr_coeff_fields = np.zeros_like(glob_arr_coeff, shape=(1 + f_p1, out_fields.shape[1]))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]
        arr_coeff_fields[:, :] = glob_arr_coeff[span_1 - f_p1:1 + span_1, :]
        for i_basis_1 in range(1 + f_p1):
            coeff_fields = arr_coeff_fields[i_basis_1, :]
            for i_quad_1 in range(k1):
                spline = global_basis_1[i_cell_1, i_basis_1, 0, i_quad_1]
                out_fields[i_cell_1 * k1 + i_quad_1, :] += spline * coeff_fields

# -----------------------------------------------------------------------------
# 2: Irregular tensor grid without weights
# -----------------------------------------------------------------------------
def eval_fields_3d_irregular_no_weights(np1: int, np2: int, np3: int, f_p1: int, f_p2: int,
                                        f_p3: int, cell_index_1: 'int[:]', cell_index_2: 'int[:]', cell_index_3 : 'int[:]',
                                        global_basis_1: 'float[:,:,:]', global_basis_2: 'float[:,:,:]', global_basis_3: 'float[:,:,:]',
                                        global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_spans_3: 'int[:]',
                                        glob_arr_coeff: 'T[:,:,:,:]', out_fields: 'T[:,:,:,:]'):
    """
    Parameters
    ----------
    np1 : int
        Number of points in the X1 direction
    np2 : int
        Number of points in the X2 direction
    np3 : int
        Number of points in the X3 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction
    f_p3: int
        Degree in the X3 direction

    cell_index_1 : ndarray of ints
        Index of the cells in the X1 direction
    cell_index_2 : ndarray of ints
        Index of the cells in the X2 direction
    cell_index_3 : ndarray of ints
        Index of the cells in the X3 direction

    global_basis_1 : ndarray of floats
        Basis functions values at each point in the X1 direction
    global_basis_2 : ndarray of floats
        Basis functions values at each point in the X2 direction
    global_basis_3 : ndarray of floats
        Basis functions values at each point in the X3 direction

    global_spans_1 : ndarray of ints
        Spans in the X1 direction
    global_spans_2 : ndarray of ints
        Spans in the X2 direction
    global_spans_3 : ndarray of ints
        Spans in the X3 direction

    glob_arr_coeff : ndarray of floats
        Coefficients of the fields in the X1, X2 and X3 directions

    out_fields : ndarray of floats
        Evaluated fields, filled with the correct values by the function
    """
    arr_coeff_fields = np.zeros_like(glob_arr_coeff, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3, out_fields.shape[3]))

    for i_p_1 in range(np1):
        i_cell_1 = cell_index_1[i_p_1]
        span_1 = global_spans_1[i_cell_1]

        for i_p_2 in range(np2):
            i_cell_2 = cell_index_2[i_p_2]
            span_2 = global_spans_2[i_cell_2]

            for i_p_3 in range(np3):
                i_cell_3 = cell_index_3[i_p_3]
                span_3 = global_spans_3[i_cell_3]

                arr_coeff_fields[:, :, :, :] = glob_arr_coeff[span_1 - f_p1:1 + span_1,
                                                              span_2 - f_p2:1 + span_2,
                                                              span_3 - f_p3:1 + span_3,
                                                              :]

                for i_basis_1 in range(1 + f_p1):
                    spline_1 = global_basis_1[i_p_1, i_basis_1, 0]

                    for i_basis_2 in range(1 + f_p2):
                        spline_2 = global_basis_2[i_p_2, i_basis_2, 0]

                        for i_basis_3 in range(1 + f_p3):
                            spline_3 = global_basis_3[i_p_3, i_basis_3, 0]

                            spline = spline_1 * spline_2 * spline_3

                            coeff_fields = arr_coeff_fields[i_basis_1, i_basis_2, i_basis_3, :]

                            out_fields[i_p_1, i_p_2, i_p_3, :] += spline * coeff_fields


def eval_fields_2d_irregular_no_weights(np1: int, np2: int, f_p1: int, f_p2: int,
                                        cell_index_1: 'int[:]', cell_index_2: 'int[:]',
                                        global_basis_1: 'float[:,:,:]', global_basis_2: 'float[:,:,:]',
                                        global_spans_1: 'int[:]', global_spans_2: 'int[:]', glob_arr_coeff: 'T[:,:,:]',
                                        out_fields: 'T[:,:,:]'):
    """
    Parameters
    ----------
    np1: int
        Number of points in the X1 direction
    np2: int
        Number of points in the X2 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction

    cell_index_1 : ndarray of ints
        Index of the cells in the X1 direction
    cell_index_2 : ndarray of ints
        Index of the cells in the X2 direction

    global_basis_1: ndarray of floats
        Basis functions values at each point in the X1 direction
    global_basis_2: ndarray of floats
        Basis functions values at each point in the X2 direction

    global_spans_1: ndarray of ints
        Spans in the X1 direction
    global_spans_2: ndarray of ints
        Spans in the X2 direction

    glob_arr_coeff: ndarray of floats
        Coefficients of the fields in the X1 and X2 directions

    out_fields: ndarray of floats
        Evaluated fields, filled with the correct values by the function
    """
    arr_coeff_fields = np.zeros_like(glob_arr_coeff, shape=(1 + f_p1, 1 + f_p2, out_fields.shape[2]))

    for i_p_1 in range(np1):
        i_cell_1 = cell_index_1[i_p_1]
        span_1 = global_spans_1[i_cell_1]

        for i_p_2 in range(np2):
            i_cell_2 = cell_index_2[i_p_2]
            span_2 = global_spans_2[i_cell_2]

            arr_coeff_fields[:, :, :] = glob_arr_coeff[span_1 - f_p1:1 + span_1,
                                                       span_2 - f_p2:1 + span_2,
                                                       :]

            for i_basis_1 in range(1 + f_p1):
                spline_1 = global_basis_1[i_p_1, i_basis_1, 0]

                for i_basis_2 in range(1 + f_p2):
                    spline_2 = global_basis_2[i_p_2, i_basis_2, 0]

                    spline = spline_1 * spline_2

                    coeff_fields = arr_coeff_fields[i_basis_1, i_basis_2, :]

                    out_fields[i_p_1, i_p_2, :] += spline * coeff_fields


def eval_fields_1d_irregular_no_weights(np1: int, f_p1: int,
                                        cell_index_1: 'int[:]',
                                        global_basis_1: 'float[:,:,:]',
                                        global_spans_1: 'int[:]', 
                                        glob_arr_coeff: 'T[:,:]',
                                        out_fields: 'T[:,:]'):
    """
    Parameters
    ----------
    np1: int
        Number of points in the X1 direction

    f_p1: int
        Degree in the X1 direction

    cell_index_1 : ndarray of ints
        Index of the cells in the X1 direction

    global_basis_1: ndarray of floats
        Basis functions values at each point in the X1 direction

    global_spans_1: ndarray of ints
        Spans in the X1 direction

    glob_arr_coeff: ndarray of floats
        Coefficients of the scalar 1D fields, stored in a 2D array.
        The first dimension loops over the coefficients, and the
        second dimension over the different fields.

    out_fields: ndarray of floats
        Evaluated fields, filled with the correct values by the function
    """
    arr_coeff_fields = np.zeros_like(glob_arr_coeff, shape=(1 + f_p1, out_fields.shape[1]))

    for i_p_1 in range(np1):
        i_cell_1 = cell_index_1[i_p_1]
        span_1 = global_spans_1[i_cell_1]

        arr_coeff_fields[:, :] = glob_arr_coeff[span_1 - f_p1:1 + span_1, :]

        for i_basis_1 in range(1 + f_p1):
            spline_1 = global_basis_1[i_p_1, i_basis_1, 0]
            coeff_fields = arr_coeff_fields[i_basis_1, :]

            out_fields[i_p_1, :] += spline_1 * coeff_fields

# -----------------------------------------------------------------------------
# 3: Regular tensor grid with weights
# -----------------------------------------------------------------------------
def eval_fields_3d_weighted(nc1: int, nc2: int, nc3: int, f_p1: int, f_p2: int,
                            f_p3: int, k1: int, k2: int, k3: int, global_basis_1: 'float[:,:,:,:]',
                            global_basis_2: 'float[:,:,:,:]', global_basis_3: 'float[:,:,:,:]',
                            global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_spans_3: 'int[:]',
                            glob_arr_coeff: 'T[:,:,:,:]', global_arr_weights: 'float[:,:,:]',
                            out_fields: 'T[:,:,:,:]'):
    """
    Parameters
    ----------
    nc1: int
        Number of cells in the X1 direction
    nc2: int
        Number of cells in the X2 direction
    nc3: int
        Number of cells in the X3 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction
    f_p3: int
        Degree in the X3 direction

    k1: int
        Number of evaluation points in the X1 direction
    k2: int
        Number of evaluation points in the X2 direction
    k3: int
        Number of evaluation points in the X3 direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X1 direction
    global_basis_2: ndarray of floats
        Basis functions values at each cell and quadrature points in the X2 direction
    global_basis_3: ndarray of floats
        Basis functions values at each cell and quadrature points in the X3 direction

    global_spans_1: ndarray of ints
        Spans in the X1 direction
    global_spans_2: ndarray of ints
        Spans in the X2 direction
    global_spans_3: ndarray of ints
        Spans in the X3 direction

    glob_arr_coeff: ndarray of floats
        Coefficients of the fields in the X1,X2 and X3 directions

    global_arr_weights: ndarray of float
        Coefficients of the weight field in the X1, X2 and X3 directions

    out_fields: ndarray of floats
        Evaluated fields, filled with the correct values by the function
    """
    arr_coeff_fields = np.zeros_like(glob_arr_coeff, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3, out_fields.shape[3]))
    arr_coeff_weights = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))

    arr_fields = np.zeros_like(glob_arr_coeff, shape=(k1, k2, k3, out_fields.shape[3]))
    arr_weights = np.zeros((k1, k2, k3))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]

            for i_cell_3 in range(nc3):
                span_3 = global_spans_3[i_cell_3]

                arr_coeff_fields[:, :, :, :] = glob_arr_coeff[span_1 - f_p1:1 + span_1,
                                                              span_2 - f_p2:1 + span_2,
                                                              span_3 - f_p3:1 + span_3,
                                                              :]

                arr_coeff_weights[:, :, :] = global_arr_weights[span_1 - f_p1:1 + span_1,
                                                                span_2 - f_p2:1 + span_2,
                                                                span_3 - f_p3:1 + span_3]

                arr_fields[:, :, :, :] = 0.0
                arr_weights[:, :, :] = 0.0

                for i_quad_1 in range(k1):
                    for i_quad_2 in range(k2):
                        for i_quad_3 in range(k3):
                            for i_basis_1 in range(1 + f_p1):
                                spline_1 = global_basis_1[i_cell_1, i_basis_1, 0, i_quad_1]

                                for i_basis_2 in range(1 + f_p2):
                                    spline_2 = global_basis_2[i_cell_2, i_basis_2, 0, i_quad_2]

                                    for i_basis_3 in range(1 + f_p3):
                                        spline_3 = global_basis_3[i_cell_3, i_basis_3, 0, i_quad_3]

                                        spline = spline_1 * spline_2 * spline_3

                                        coeff_fields = arr_coeff_fields[i_basis_1, i_basis_2, i_basis_3, :]
                                        coeff_weight = arr_coeff_weights[i_basis_1, i_basis_2, i_basis_3]

                                        arr_fields[i_quad_1, i_quad_2, i_quad_3, :] += \
                                            spline * coeff_fields * coeff_weight

                                        arr_weights[i_quad_1, i_quad_2, i_quad_3] += spline * coeff_weight

                            fields = arr_fields[i_quad_1, i_quad_2, i_quad_3, :]
                            weight = arr_weights[i_quad_1, i_quad_2, i_quad_3]

                            out_fields[i_cell_1 * k1 + i_quad_1,
                                       i_cell_2 * k2 + i_quad_2,
                                       i_cell_3 * k3 + i_quad_3,
                                       :] += fields / weight


def eval_fields_2d_weighted(nc1: int, nc2: int, f_p1: int, f_p2: int, k1: int, k2: int,
                            global_basis_1: 'float[:,:,:,:]', global_basis_2: 'float[:,:,:,:]',
                            global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_arr_coeff: 'T[:,:,:]',
                            global_arr_weights: 'float[:,:]', out_fields: 'T[:,:,:]'):
    """
    Parameters
    ----------
    nc1: int
        Number of cells in the X1 direction
    nc2: int
        Number of cells in the X2 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction

    k1: int
        Number of evaluation points in the X1 direction
    k2: int
        Number of evaluation points in the X2 direction

    global_basis_1: ndarray of float
        Basis functions values at each cell and quadrature points in the X1 direction
    global_basis_2: ndarray of float
        Basis functions values at each cell and quadrature points in the X2 direction

    global_spans_1: ndarray of int
        Spans in the X1 direction
    global_spans_2: ndarray of int
        Spans in the X2 direction

    global_arr_coeff: ndarray of float
        Coefficients of the fields in the X1 and X2 directions

    global_arr_weights: ndarray of float
        Coefficients of the weight field in the X1 and X2 directions

    out_fields: ndarray of float
        Evaluated fields, filled with the correct values by the function
    """
    arr_coeff_fields = np.zeros_like(global_arr_coeff, shape=(1 + f_p1, 1 + f_p2, out_fields.shape[2]))
    arr_coeff_weights = np.zeros((1 + f_p1, 1 + f_p2))

    arr_fields = np.zeros_like(global_arr_coeff, shape=(k1, k2, out_fields.shape[2]))
    arr_weights = np.zeros((k1, k2))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]

            arr_coeff_fields[:, :, :] = global_arr_coeff[span_1 - f_p1:1 + span_1,
                                                         span_2 - f_p2:1 + span_2,
                                                         :]

            arr_coeff_weights[:, :] = global_arr_weights[span_1 - f_p1:1 + span_1,
                                                         span_2 - f_p2:1 + span_2]

            arr_fields[:, :, :] = 0.0
            arr_weights[:, :] = 0.0

            for i_quad_1 in range(k1):
                for i_quad_2 in range(k2):
                    for i_basis_1 in range(1 + f_p1):
                        spline_1 = global_basis_1[i_cell_1, i_basis_1, 0, i_quad_1]

                        for i_basis_2 in range(1 + f_p2):
                            spline_2 = global_basis_2[i_cell_2, i_basis_2, 0, i_quad_2]

                            spline = spline_1 * spline_2

                            coeff_fields = arr_coeff_fields[i_basis_1, i_basis_2, :]
                            coeff_weight = arr_coeff_weights[i_basis_1, i_basis_2]

                            arr_fields[i_quad_1, i_quad_2, :] += spline * coeff_fields * coeff_weight

                            arr_weights[i_quad_1, i_quad_2] += spline * coeff_weight

                    fields = arr_fields[i_quad_1, i_quad_2, :]
                    weight = arr_weights[i_quad_1, i_quad_2]

                    out_fields[i_cell_1 * k1 + i_quad_1,
                               i_cell_2 * k2 + i_quad_2,
                               :] += fields / weight


def eval_fields_1d_weighted(nc1: int, f_p1: int, k1: int, 
                            global_basis_1: 'float[:,:,:,:]',
                            global_spans_1: 'int[:]', global_arr_coeff: 'T[:,:]',
                            global_arr_weights: 'float[:]', out_fields: 'T[:,:]'):
    """
    Parameters
    ----------
    nc1: int
        Number of cells in the X1 direction

    f_p1: int
        Degree in the X1 direction

    k1: int
        Number of evaluation points in the X1 direction

    global_basis_1: ndarray of float
        Basis functions values at each cell and quadrature points in the X1 direction

    global_spans_1: ndarray of int
        Spans in the X1 direction

    global_arr_coeff: ndarray of float
        Coefficients of the scalar 1D fields, stored in a 2D array.
        The first dimension loops over the coefficients, and the
        second dimension over the different fields.

    global_arr_weights: ndarray of float
        Coefficients of the 1D scalar weight field.

    out_fields: ndarray of float
        Evaluated fields, filled with the correct values by the function
    """
    arr_coeff_fields = np.zeros_like(global_arr_coeff, shape=(1 + f_p1, out_fields.shape[1]))
    arr_coeff_weights = np.zeros((1 + f_p1))

    arr_fields = np.zeros_like(global_arr_coeff, shape=(k1, out_fields.shape[1]))
    arr_weights = np.zeros((k1))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        arr_coeff_fields[:, :] = global_arr_coeff[span_1 - f_p1:1 + span_1, :]

        arr_coeff_weights[:] = global_arr_weights[span_1 - f_p1:1 + span_1]

        arr_fields[:, :] = 0.0
        arr_weights[:] = 0.0

        for i_quad_1 in range(k1):
            for i_basis_1 in range(1 + f_p1):
                spline = global_basis_1[i_cell_1, i_basis_1, 0, i_quad_1]

                coeff_fields = arr_coeff_fields[i_basis_1, :]
                coeff_weight = arr_coeff_weights[i_basis_1]

                arr_fields[i_quad_1, :] += spline * coeff_fields * coeff_weight

                arr_weights[i_quad_1] += spline * coeff_weight

            fields = arr_fields[i_quad_1, :]
            weight = arr_weights[i_quad_1]

            out_fields[i_cell_1 * k1 + i_quad_1, :] += fields / weight


# -----------------------------------------------------------------------------
# 4: Iregular tensor grid with weights
# -----------------------------------------------------------------------------
def eval_fields_3d_irregular_weighted(np1: int, np2: int, np3: int, f_p1: int, f_p2: int,
                                      f_p3: int, cell_index_1: 'int[:]', cell_index_2: 'int[:]', cell_index_3 : 'int[:]',
                                      global_basis_1: 'float[:,:,:]', global_basis_2: 'float[:,:,:]', global_basis_3: 'float[:,:,:]',
                                      global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_spans_3: 'int[:]',
                                      glob_arr_coeff: 'T[:,:,:,:]', global_arr_weights: 'float[:,:,:]',
                                      out_fields: 'T[:,:,:,:]'):
    """
    Parameters
    ----------
    np1 : int
        Number of points in the X1 direction
    np2 : int
        Number of points in the X2 direction
    np3 : int
        Number of points in the X3 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction
    f_p3: int
        Degree in the X3 direction

    cell_index_1 : ndarray of ints
        Index of the cells in the X1 direction
    cell_index_2 : ndarray of ints
        Index of the cells in the X2 direction
    cell_index_3 : ndarray of ints
        Index of the cells in the X3 direction

    global_basis_1 : ndarray of floats
        Basis functions values at each point in the X1 direction
    global_basis_2 : ndarray of floats
        Basis functions values at each point in the X2 direction
    global_basis_3 : ndarray of floats
        Basis functions values at each point in the X3 direction

    global_spans_1 : ndarray of ints
        Spans in the X1 direction
    global_spans_2 : ndarray of ints
        Spans in the X2 direction
    global_spans_3 : ndarray of ints
        Spans in the X3 direction

    glob_arr_coeff : ndarray of floats
        Coefficients of the fields in the X1, X2 and X3 directions

    global_arr_weights: ndarray of float
        Coefficients of the weight field in the X1, X2 and X3 directions

    out_fields : ndarray of floats
        Evaluated fields, filled with the correct values by the function
    """
    arr_coeff_fields = np.zeros_like(glob_arr_coeff, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3, out_fields.shape[3]))
    arr_coeff_weights = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))

    temp_fields = np.zeros_like(glob_arr_coeff, shape=out_fields.shape[3])

    for i_p_1 in range(np1):
        i_cell_1 = cell_index_1[i_p_1]
        span_1 = global_spans_1[i_cell_1]

        for i_p_2 in range(np2):
            i_cell_2 = cell_index_2[i_p_2]
            span_2 = global_spans_2[i_cell_2]

            for i_p_3 in range(np3):
                i_cell_3 = cell_index_3[i_p_3]
                span_3 = global_spans_3[i_cell_3]

                arr_coeff_fields[:, :, :, :] = glob_arr_coeff[span_1 - f_p1:1 + span_1,
                                                              span_2 - f_p2:1 + span_2,
                                                              span_3 - f_p3:1 + span_3,
                                                              :]

                arr_coeff_weights[:, :, :] = global_arr_weights[span_1 - f_p1:1 + span_1,
                                                                span_2 - f_p2:1 + span_2,
                                                                span_3 - f_p3:1 + span_3]

                temp_fields[:] = 0.0
                temp_weight = 0.0

                for i_basis_1 in range(1 + f_p1):
                    spline_1 = global_basis_1[i_p_1, i_basis_1, 0]

                    for i_basis_2 in range(1 + f_p2):
                        spline_2 = global_basis_2[i_p_2, i_basis_2, 0]

                        for i_basis_3 in range(1 + f_p3):
                            spline_3 = global_basis_3[i_p_3, i_basis_3, 0]

                            spline = spline_1 * spline_2 * spline_3

                            coeff_fields = arr_coeff_fields[i_basis_1, i_basis_2, i_basis_3, :]

                            coeff_weight = arr_coeff_weights[i_basis_1, i_basis_2, i_basis_3]

                            temp_fields[:] += spline * coeff_fields * coeff_weight

                            temp_weight += spline * coeff_weight

                out_fields[i_p_1, i_p_2, i_p_3, :] += temp_fields / temp_weight


def eval_fields_2d_irregular_weighted(np1: int, np2: int, f_p1: int, f_p2: int,
                                      cell_index_1: 'int[:]', cell_index_2: 'int[:]', global_basis_1: 'float[:,:,:]',
                                      global_basis_2: 'float[:,:,:]', global_spans_1: 'int[:]', global_spans_2: 'int[:]',
                                      global_arr_coeff: 'T[:,:,:]', global_arr_weights: 'float[:,:]',
                                      out_fields: 'T[:,:,:]'):
    """
    Parameters
    ----------
    np1 : int
        Number of points in the X1 direction
    np2 : int
        Number of points in the X2 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction

    cell_index_1 : ndarray of ints
        Index of the cells in the X1 direction
    cell_index_2 : ndarray of ints
        Index of the cells in the X2 direction

    global_basis_1 : ndarray of floats
        Basis functions values at each point in the X1 direction
    global_basis_2 : ndarray of floats
        Basis functions values at each point in the X2 direction

    global_spans_1 : ndarray of ints
        Spans in the X1 direction
    global_spans_2 : ndarray of ints
        Spans in the X2 direction

    global_arr_coeff : ndarray of floats
        Coefficients of the fields in the X1 and X2 directions

    global_arr_weights: ndarray of float
        Coefficients of the weight field in the X1 and X2 directions

    out_fields : ndarray of floats
        Evaluated fields, filled with the correct values by the function
    """
    arr_coeff_fields = np.zeros_like(global_arr_coeff, shape=(1 + f_p1, 1 + f_p2, out_fields.shape[2]))
    arr_coeff_weights = np.zeros((1 + f_p1, 1 + f_p2))

    temp_fields = np.zeros_like(global_arr_coeff, shape=out_fields.shape[2])

    for i_p_1 in range(np1):
        i_cell_1 = cell_index_1[i_p_1]
        span_1 = global_spans_1[i_cell_1]

        for i_p_2 in range(np2):
            i_cell_2 = cell_index_2[i_p_2]
            span_2 = global_spans_2[i_cell_2]


            arr_coeff_fields[:, :, :] = global_arr_coeff[span_1 - f_p1:1 + span_1,
                                                         span_2 - f_p2:1 + span_2,
                                                         :]

            arr_coeff_weights[:, :] = global_arr_weights[span_1 - f_p1:1 + span_1,
                                                         span_2 - f_p2:1 + span_2]

            temp_fields[:] = 0.0
            temp_weight     = 0.0

            for i_basis_1 in range(1 + f_p1):
                spline_1 = global_basis_1[i_p_1, i_basis_1, 0]

                for i_basis_2 in range(1 + f_p2):
                    spline_2 = global_basis_2[i_p_2, i_basis_2, 0]

                    spline = spline_1 * spline_2

                    coeff_fields = arr_coeff_fields[i_basis_1, i_basis_2, :]

                    coeff_weight = arr_coeff_weights[i_basis_1, i_basis_2]

                    temp_fields[:] += spline * coeff_fields * coeff_weight

                    temp_weight += spline * coeff_weight

            out_fields[i_p_1, i_p_2, :] += temp_fields / temp_weight


def eval_fields_1d_irregular_weighted(np1: int, f_p1: int, 
                                      cell_index_1: 'int[:]', global_basis_1: 'float[:,:,:]',
                                      global_spans_1: 'int[:]', 
                                      global_arr_coeff: 'T[:,:]', global_arr_weights: 'float[:]',
                                      out_fields: 'T[:,:]'):
    """
    Parameters
    ----------
    np1 : int
        Number of points in the X1 direction

    f_p1: int
        Degree in the X1 direction

    cell_index_1 : ndarray of ints
        Index of the cells in the X1 direction

    global_basis_1 : ndarray of floats
        Basis functions values at each point in the X1 direction

    global_spans_1 : ndarray of ints
        Spans in the X1 direction

    global_arr_coeff : ndarray of floats
        Coefficients of the scalar 1D fields, stored in a 2D array.
        The first dimension loops over the coefficients, and the
        second dimension over the different fields.

    global_arr_weights: ndarray of float
        Coefficients of the 1D scalar weight field.

    out_fields : ndarray of floats
        Evaluated fields, filled with the correct values by the function
    """
    temp_fields = np.zeros_like(global_arr_coeff, shape=out_fields.shape[1])

    for i_p_1 in range(np1):
        i_cell_1 = cell_index_1[i_p_1]
        span_1 = global_spans_1[i_cell_1]

        temp_fields[:] = 0.0
        temp_weight    = 0.0

        for i_basis_1 in range(1 + f_p1):
            spline = global_basis_1[i_p_1, i_basis_1, 0]

            coeff_fields = global_arr_coeff[span_1 - f_p1 + i_basis_1, :]

            coeff_weight = global_arr_weights[span_1 - f_p1 + i_basis_1]

            temp_fields[:] += spline * coeff_fields * coeff_weight

            temp_weight += spline * coeff_weight

        out_fields[i_p_1, :] += temp_fields / temp_weight


# =============================================================================
# Evaluation of the Jacobian determinant
# =============================================================================
# -----------------------------------------------------------------------------
# 1: Regular tensor grid without weights
# -----------------------------------------------------------------------------
def eval_jac_det_3d(nc1: int, nc2: int, nc3: int, f_p1: int, f_p2: int, f_p3: int, k1: int, k2: int, k3: int,
                    global_basis_1: 'float[:,:,:,:]', global_basis_2: 'float[:,:,:,:]',
                    global_basis_3: 'float[:,:,:,:]', global_spans_1: 'int[:]', global_spans_2: 'int[:]',
                    global_spans_3: 'int[:]', global_arr_coeff_x: 'T[:,:,:]', global_arr_coeff_y: 'T[:,:,:]',
                    global_arr_coeff_z: 'T[:,:,:]', jac_det: 'T[:,:,:]'):

    """
    Parameters
    ----------
    nc1: int
        Number of cells in the X1 direction
    nc2: int
        Number of cells in the X2 direction
    nc3: int
        Number of cells in the X3 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction
    f_p3: int
        Degree in the X3 direction

    k1: int
        Number of evaluation points in the X1 direction
    k2: int
        Number of evaluation points in the X2 direction
    k3: int
        Number of evaluation points in the X3 direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X1 direction
    global_basis_2: ndarray of floats
        Basis functions values at each cell and quadrature points in the X2 direction
    global_basis_3: ndarray of floats
        Basis functions values at each cell and quadrature points in the X3 direction

    global_spans_1: ndarray of ints
        Spans in the X1 direction
    global_spans_2: ndarray of ints
        Spans in the X2 direction
    global_spans_3: ndarray of ints
        Spans in the X3 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field
    global_arr_coeff_z: ndarray of floats
        Coefficients of the X3 field

    jac_det: ndarray of floats
        Jacobian determinant on the grid.
    """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_z = np.zeros_like(global_arr_coeff_z, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))

    arr_x_x1 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2, k3))
    arr_x_x2 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2, k3))
    arr_x_x3 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2, k3))

    arr_y_x1 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2, k3))
    arr_y_x2 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2, k3))
    arr_y_x3 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2, k3))

    arr_z_x1 = np.zeros_like(global_arr_coeff_z, shape=(k1, k2, k3))
    arr_z_x2 = np.zeros_like(global_arr_coeff_z, shape=(k1, k2, k3))
    arr_z_x3 = np.zeros_like(global_arr_coeff_z, shape=(k1, k2, k3))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]

            for i_cell_3 in range(nc3):
                span_3 = global_spans_3[i_cell_3]

                arr_x_x1[:, :, :] = 0.0
                arr_x_x2[:, :, :] = 0.0
                arr_x_x3[:, :, :] = 0.0

                arr_y_x1[:, :, :] = 0.0
                arr_y_x2[:, :, :] = 0.0
                arr_y_x3[:, :, :] = 0.0

                arr_z_x1[:, :, :] = 0.0
                arr_z_x2[:, :, :] = 0.0
                arr_z_x3[:, :, :] = 0.0

                arr_coeffs_x[:, :, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_y[:, :, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_z[:, :, :] = global_arr_coeff_z[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]
                for i_quad_1 in range(k1):
                    for i_quad_2 in range(k2):
                        for i_quad_3 in range(k3):
                            for i_basis_1 in range(1 + f_p1):
                                spline_1 = global_basis_1[i_cell_1, i_basis_1, 0, i_quad_1]
                                spline_x1 = global_basis_1[i_cell_1, i_basis_1, 1, i_quad_1]

                                for i_basis_2 in range(1 + f_p2):
                                    spline_2 = global_basis_2[i_cell_2, i_basis_2, 0, i_quad_2]
                                    spline_x2 = global_basis_2[i_cell_2, i_basis_2, 1, i_quad_2]

                                    for i_basis_3 in range(1 + f_p3):
                                        spline_3 = global_basis_3[i_cell_3, i_basis_3, 0, i_quad_3]
                                        spline_x3 = global_basis_3[i_cell_3, i_basis_3, 1, i_quad_3]

                                        mapping_x1 = spline_x1 * spline_2 * spline_3
                                        mapping_x2 = spline_1 * spline_x2 * spline_3
                                        mapping_x3 = spline_1 * spline_2 * spline_x3

                                        coeff_x = arr_coeffs_x[i_basis_1, i_basis_2, i_basis_3]
                                        coeff_y = arr_coeffs_y[i_basis_1, i_basis_2, i_basis_3]
                                        coeff_z = arr_coeffs_z[i_basis_1, i_basis_2, i_basis_3]

                                        arr_x_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_x
                                        arr_x_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_x
                                        arr_x_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_x

                                        arr_y_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_y
                                        arr_y_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_y
                                        arr_y_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_y

                                        arr_z_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_z
                                        arr_z_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_z
                                        arr_z_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_z

                            x_x1 = arr_x_x1[i_quad_1, i_quad_2, i_quad_3]
                            x_x2 = arr_x_x2[i_quad_1, i_quad_2, i_quad_3]
                            x_x3 = arr_x_x3[i_quad_1, i_quad_2, i_quad_3]

                            y_x1 = arr_y_x1[i_quad_1, i_quad_2, i_quad_3]
                            y_x2 = arr_y_x2[i_quad_1, i_quad_2, i_quad_3]
                            y_x3 = arr_y_x3[i_quad_1, i_quad_2, i_quad_3]

                            z_x1 = arr_z_x1[i_quad_1, i_quad_2, i_quad_3]
                            z_x2 = arr_z_x2[i_quad_1, i_quad_2, i_quad_3]
                            z_x3 = arr_z_x3[i_quad_1, i_quad_2, i_quad_3]

                            jac_det[i_cell_1 * k1 + i_quad_1,
                                       i_cell_2 * k2 + i_quad_2,
                                       i_cell_3 * k3 + i_quad_3] = (+ x_x1 * y_x2 * z_x3
                                                                    + x_x2 * y_x3 * z_x1
                                                                    + x_x3 * y_x1 * z_x2
                                                                    - x_x1 * y_x3 * z_x2
                                                                    - x_x2 * y_x1 * z_x3
                                                                    - x_x3 * y_x2 * z_x1)


def eval_jac_det_2d(nc1: int, nc2: int, f_p1: int, f_p2: int, k1: int, k2: int, global_basis_1: 'float[:,:,:,:]',
                    global_basis_2: 'float[:,:,:,:]', global_spans_1: 'int[:]', global_spans_2: 'int[:]',
                    global_arr_coeff_x: 'T[:,:]', global_arr_coeff_y: 'T[:,:]',
                    jac_det: 'T[:,:]'):
    """
    Parameters
    ----------
    nc1: int
        Number of cells in the X1 direction
    nc2: int
        Number of cells in the X2 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction

    k1: int
        Number of evaluation points in the X1 direction
    k2: int
        Number of evaluation points in the X2 direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X1 direction
    global_basis_2: ndarray of floats
        Basis functions values at each cell and quadrature points in the X2 direction

    global_spans_1: ndarray of ints
        Spans in the X1 direction
    global_spans_2: ndarray of ints
        Spans in the X2 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field

    jac_det: ndarray of floats
        Jacobian determinant on the grid.
    """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2))

    arr_x_x1 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2))
    arr_x_x2 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2))

    arr_y_x1 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2))
    arr_y_x2 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]

            arr_x_x1[:, :] = 0.0
            arr_x_x2[:, :] = 0.0

            arr_y_x1[:, :] = 0.0
            arr_y_x2[:, :] = 0.0

            arr_coeffs_x[:, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                    span_2 - f_p2:1 + span_2]

            arr_coeffs_y[:, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                    span_2 - f_p2:1 + span_2]

            for i_quad_1 in range(k1):
                for i_quad_2 in range(k2):
                    for i_basis_1 in range(1 + f_p1):
                        spline_1 = global_basis_1[i_cell_1, i_basis_1, 0, i_quad_1]
                        spline_x1 = global_basis_1[i_cell_1, i_basis_1, 1, i_quad_1]

                        for i_basis_2 in range(1 + f_p2):
                            spline_2 = global_basis_2[i_cell_2, i_basis_2, 0, i_quad_2]
                            spline_x2 = global_basis_2[i_cell_2, i_basis_2, 1, i_quad_2]

                            mapping_x1 = spline_x1 * spline_2
                            mapping_x2 = spline_1 * spline_x2

                            coeff_x = arr_coeffs_x[i_basis_1, i_basis_2]
                            coeff_y = arr_coeffs_y[i_basis_1, i_basis_2]

                            arr_x_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_x
                            arr_x_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_x

                            arr_y_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_y
                            arr_y_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_y

                    x_x1 = arr_x_x1[i_quad_1, i_quad_2]
                    x_x2 = arr_x_x2[i_quad_1, i_quad_2]

                    y_x1 = arr_y_x1[i_quad_1, i_quad_2]
                    y_x2 = arr_y_x2[i_quad_1, i_quad_2]

                    jac_det[i_cell_1 * k1 + i_quad_1,
                            i_cell_2 * k2 + i_quad_2] = (x_x1 * y_x2 - x_x2 * y_x1)


# -----------------------------------------------------------------------------
# 2: Irregular tensor grid without weights
# -----------------------------------------------------------------------------
def eval_jac_det_irregular_3d(np1: int, np2: int, np3: int, f_p1: int, f_p2: int,
                              f_p3: int, cell_index_1: 'int[:]', cell_index_2: 'int[:]', cell_index_3 : 'int[:]',
                              global_basis_1: 'float[:,:,:]', global_basis_2: 'float[:,:,:]',
                              global_basis_3: 'float[:,:,:]', global_spans_1: 'int[:]', global_spans_2: 'int[:]',
                              global_spans_3: 'int[:]', global_arr_coeff_x: 'T[:,:,:]',
                              global_arr_coeff_y: 'T[:,:,:]', global_arr_coeff_z: 'T[:,:,:]',
                              jac_det: 'T[:,:,:]'):
    """
    Parameters
    ----------
    np1 : int
        Number of points in the X1 direction
    np2 : int
        Number of points in the X2 direction
    np3 : int
        Number of points in the X3 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction
    f_p3: int
        Degree in the X3 direction

    cell_index_1 : ndarray of ints
        Index of the cells in the X1 direction
    cell_index_2 : ndarray of ints
        Index of the cells in the X2 direction
    cell_index_3 : ndarray of ints
        Index of the cells in the X3 direction

    global_basis_1 : ndarray of floats
        Basis functions values at each point in the X1 direction
    global_basis_2 : ndarray of floats
        Basis functions values at each point in the X2 direction
    global_basis_3 : ndarray of floats
        Basis functions values at each point in the X3 direction

    global_spans_1 : ndarray of ints
        Spans in the X1 direction
    global_spans_2 : ndarray of ints
        Spans in the X2 direction
    global_spans_3 : ndarray of ints
        Spans in the X3 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field
    global_arr_coeff_z: ndarray of floats
        Coefficients of the X3 field

    jac_det: ndarray of floats
        Jacobian determinant on the grid.
    """


    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_z = np.zeros_like(global_arr_coeff_z, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))

    temp_x_x1 = arr_coeffs_x[0,0,0]-arr_coeffs_x[0,0,0]
    temp_x_x2 = arr_coeffs_y[0,0,0]-arr_coeffs_y[0,0,0]
    temp_x_x3 = arr_coeffs_z[0,0,0]-arr_coeffs_z[0,0,0]

    temp_y_x1 = arr_coeffs_x[0,0,0]-arr_coeffs_x[0,0,0]
    temp_y_x2 = arr_coeffs_y[0,0,0]-arr_coeffs_y[0,0,0]
    temp_y_x3 = arr_coeffs_z[0,0,0]-arr_coeffs_z[0,0,0]

    temp_z_x1 = arr_coeffs_x[0,0,0]-arr_coeffs_x[0,0,0]
    temp_z_x2 = arr_coeffs_y[0,0,0]-arr_coeffs_y[0,0,0]
    temp_z_x3 = arr_coeffs_z[0,0,0]-arr_coeffs_z[0,0,0]

    for i_p_1 in range(np1):
        i_cell_1 = cell_index_1[i_p_1]
        span_1 = global_spans_1[i_cell_1]

        for i_p_2 in range(np2):
            i_cell_2 = cell_index_2[i_p_2]
            span_2 = global_spans_2[i_cell_2]

            for i_p_3 in range(np3):
                i_cell_3 = cell_index_3[i_p_3]
                span_3 = global_spans_3[i_cell_3]

                arr_coeffs_x[:, :, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_y[:, :, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_z[:, :, :] = global_arr_coeff_z[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                temp_x_x1 *= 0.0
                temp_x_x2 *= 0.0
                temp_x_x3 *= 0.0

                temp_y_x1 *= 0.0
                temp_y_x2 *= 0.0
                temp_y_x3 *= 0.0

                temp_z_x1 *= 0.0
                temp_z_x2 *= 0.0
                temp_z_x3 *= 0.0

                for i_basis_1 in range(1 + f_p1):
                    spline_1 = global_basis_1[i_p_1, i_basis_1, 0]
                    spline_x1 = global_basis_1[i_p_1, i_basis_1, 1]

                    for i_basis_2 in range(1 + f_p2):
                        spline_2 = global_basis_2[i_p_2, i_basis_2, 0]
                        spline_x2 = global_basis_2[i_p_2, i_basis_2, 1]

                        for i_basis_3 in range(1 + f_p3):
                            spline_3 = global_basis_3[i_p_3, i_basis_3, 0]
                            spline_x3 = global_basis_3[i_p_3, i_basis_3, 1]

                            mapping_x1 = spline_x1 * spline_2 * spline_3
                            mapping_x2 = spline_1 * spline_x2 * spline_3
                            mapping_x3 = spline_1 * spline_2 * spline_x3

                            coeff_x = arr_coeffs_x[i_basis_1, i_basis_2, i_basis_3]
                            coeff_y = arr_coeffs_y[i_basis_1, i_basis_2, i_basis_3]
                            coeff_z = arr_coeffs_z[i_basis_1, i_basis_2, i_basis_3]

                            temp_x_x1 += mapping_x1 * coeff_x
                            temp_x_x2 += mapping_x2 * coeff_x
                            temp_x_x3 += mapping_x3 * coeff_x

                            temp_y_x1 += mapping_x1 * coeff_y
                            temp_y_x2 += mapping_x2 * coeff_y
                            temp_y_x3 += mapping_x3 * coeff_y

                            temp_z_x1 += mapping_x1 * coeff_z
                            temp_z_x2 += mapping_x2 * coeff_z
                            temp_z_x3 += mapping_x3 * coeff_z

                jac_det[i_p_1, i_p_2, i_p_3] = (+ temp_x_x1 * temp_y_x2 * temp_z_x3
                                                + temp_x_x2 * temp_y_x3 * temp_z_x1
                                                + temp_x_x3 * temp_y_x1 * temp_z_x2
                                                - temp_x_x1 * temp_y_x3 * temp_z_x2
                                                - temp_x_x2 * temp_y_x1 * temp_z_x3
                                                - temp_x_x3 * temp_y_x2 * temp_z_x1)


def eval_jac_det_irregular_2d(np1: int, np2: int, f_p1: int, f_p2: int, cell_index_1: 'int[:]',
                              cell_index_2: 'int[:]', global_basis_1: 'float[:,:,:]', global_basis_2: 'float[:,:,:]',
                              global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_arr_coeff_x: 'T[:,:]',
                              global_arr_coeff_y: 'T[:,:]', jac_det: 'T[:,:]'):
    """
    Parameters
    ----------
    np1 : int
        Number of points in the X1 direction
    np2 : int
        Number of points in the X2 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction

    cell_index_1 : ndarray of ints
        Index of the cells in the X1 direction
    cell_index_2 : ndarray of ints
        Index of the cells in the X2 direction

    global_basis_1 : ndarray of floats
        Basis functions values at each point in the X1 direction
    global_basis_2 : ndarray of floats
        Basis functions values at each point in the X2 direction

    global_spans_1 : ndarray of ints
        Spans in the X1 direction
    global_spans_2 : ndarray of ints
        Spans in the X2 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field
    """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2))

    temp_x_x1 = arr_coeffs_x[0,0]-arr_coeffs_x[0,0]
    temp_x_x2 = arr_coeffs_y[0,0]-arr_coeffs_y[0,0]

    temp_y_x1 = arr_coeffs_x[0,0]-arr_coeffs_x[0,0]
    temp_y_x2 = arr_coeffs_y[0,0]-arr_coeffs_y[0,0]

    for i_p_1 in range(np1):
        i_cell_1 = cell_index_1[i_p_1]
        span_1 = global_spans_1[i_cell_1]

        for i_p_2 in range(np2):
            i_cell_2 = cell_index_2[i_p_2]
            span_2 = global_spans_2[i_cell_2]

            arr_coeffs_x[:, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                       span_2 - f_p2:1 + span_2]

            arr_coeffs_y[:, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                       span_2 - f_p2:1 + span_2]

            temp_x_x1 *= 0.0
            temp_x_x2 *= 0.0

            temp_y_x1 *= 0.0
            temp_y_x2 *= 0.0

            for i_basis_1 in range(1 + f_p1):
                spline_1 = global_basis_1[i_p_1, i_basis_1, 0]
                spline_x1 = global_basis_1[i_p_1, i_basis_1, 1]

                for i_basis_2 in range(1 + f_p2):
                    spline_2 = global_basis_2[i_p_2, i_basis_2, 0]
                    spline_x2 = global_basis_2[i_p_2, i_basis_2, 1]

                    mapping_x1 = spline_x1 * spline_2
                    mapping_x2 = spline_1 * spline_x2

                    coeff_x = arr_coeffs_x[i_basis_1, i_basis_2]
                    coeff_y = arr_coeffs_y[i_basis_1, i_basis_2]

                    temp_x_x1 += mapping_x1 * coeff_x
                    temp_x_x2 += mapping_x2 * coeff_x

                    temp_y_x1 += mapping_x1 * coeff_y
                    temp_y_x2 += mapping_x2 * coeff_y

            jac_det[i_p_1, i_p_2] = temp_x_x1 * temp_y_x2 - temp_x_x2 * temp_y_x1


# -----------------------------------------------------------------------------
# 3: Regular tensor grid with weights
# -----------------------------------------------------------------------------

def eval_jac_det_3d_weights(nc1: int, nc2: int, nc3: int, f_p1: int, f_p2: int, f_p3: int,
                            k1: int, k2: int, k3: int, global_basis_1: 'float[:,:,:,:]',
                            global_basis_2: 'float[:,:,:,:]', global_basis_3: 'float[:,:,:,:]',
                            global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_spans_3: 'int[:]',
                            global_arr_coeff_x: 'T[:,:,:]', global_arr_coeff_y: 'T[:,:,:]',
                            global_arr_coeff_z: 'T[:,:,:]', global_arr_coeff_weights: 'float[:,:,:]',
                            jac_det: 'T[:,:,:]'):

    """
    Parameters
    ----------
    nc1: int
        Number of cells in the X1 direction
    nc2: int
        Number of cells in the X2 direction
    nc3: int
        Number of cells in the X3 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction
    f_p3: int
        Degree in the X3 direction

    k1: int
        Number of evaluation points in the X1 direction
    k2: int
        Number of evaluation points in the X2 direction
    k3: int
        Number of evaluation points in the X3 direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X1 direction
    global_basis_2: ndarray of floats
        Basis functions values at each cell and quadrature points in the X2 direction
    global_basis_3: ndarray of floats
        Basis functions values at each cell and quadrature points in the X3 direction

    global_spans_1: ndarray of ints
        Spans in the X1 direction
    global_spans_2: ndarray of ints
        Spans in the X2 direction
    global_spans_3: ndarray of ints
        Spans in the X3 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field
    global_arr_coeff_z: ndarray of floats
        Coefficients of the X3 field

    global_arr_coeff_weights: ndarray of floats
        Coefficients of the weight field

    jac_det: ndarray of floats
        Jacobian determinant on the grid
    """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_z = np.zeros_like(global_arr_coeff_z, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))

    arr_x = np.zeros_like(global_arr_coeff_x, shape=(k1, k2, k3))
    arr_y = np.zeros_like(global_arr_coeff_y, shape=(k1, k2, k3))
    arr_z = np.zeros_like(global_arr_coeff_z, shape=(k1, k2, k3))

    arr_x_x1 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2, k3))
    arr_x_x2 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2, k3))
    arr_x_x3 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2, k3))

    arr_y_x1 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2, k3))
    arr_y_x2 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2, k3))
    arr_y_x3 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2, k3))

    arr_z_x1 = np.zeros_like(global_arr_coeff_z, shape=(k1, k2, k3))
    arr_z_x2 = np.zeros_like(global_arr_coeff_z, shape=(k1, k2, k3))
    arr_z_x3 = np.zeros_like(global_arr_coeff_z, shape=(k1, k2, k3))

    arr_coeff_weights = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_weights = np.zeros((k1, k2, k3))

    arr_weights_x1 = np.zeros((k1, k2, k3))
    arr_weights_x2 = np.zeros((k1, k2, k3))
    arr_weights_x3 = np.zeros((k1, k2, k3))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]

            for i_cell_3 in range(nc3):
                span_3 = global_spans_3[i_cell_3]

                arr_x[:, :, :] = 0.0
                arr_y[:, :, :] = 0.0
                arr_z[:, :, :] = 0.0

                arr_x_x1[:, :, :] = 0.0
                arr_x_x2[:, :, :] = 0.0
                arr_x_x3[:, :, :] = 0.0

                arr_y_x1[:, :, :] = 0.0
                arr_y_x2[:, :, :] = 0.0
                arr_y_x3[:, :, :] = 0.0

                arr_z_x1[:, :, :] = 0.0
                arr_z_x2[:, :, :] = 0.0
                arr_z_x3[:, :, :] = 0.0

                arr_weights[:, :, :] = 0.0

                arr_weights_x1[:, :, :] = 0.0
                arr_weights_x2[:, :, :] = 0.0
                arr_weights_x3[:, :, :] = 0.0

                arr_coeffs_x[:, :, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_y[:, :, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_z[:, :, :] = global_arr_coeff_z[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeff_weights[:, :, :] = global_arr_coeff_weights[span_1 - f_p1:1 + span_1,
                                                                      span_2 - f_p2:1 + span_2,
                                                                      span_3 - f_p3:1 + span_3]

                for i_quad_1 in range(k1):
                    for i_quad_2 in range(k2):
                        for i_quad_3 in range(k3):
                            for i_basis_1 in range(1 + f_p1):
                                spline_1 = global_basis_1[i_cell_1, i_basis_1, 0, i_quad_1]
                                spline_x1 = global_basis_1[i_cell_1, i_basis_1, 1, i_quad_1]

                                for i_basis_2 in range(1 + f_p2):
                                    spline_2 = global_basis_2[i_cell_2, i_basis_2, 0, i_quad_2]
                                    spline_x2 = global_basis_2[i_cell_2, i_basis_2, 1, i_quad_2]

                                    for i_basis_3 in range(1 + f_p3):
                                        spline_3 = global_basis_3[i_cell_3, i_basis_3, 0, i_quad_3]
                                        spline_x3 = global_basis_3[i_cell_3, i_basis_3, 1, i_quad_3]

                                        mapping = spline_1 * spline_2 * spline_3
                                        mapping_x1 = spline_x1 * spline_2 * spline_3
                                        mapping_x2 = spline_1 * spline_x2 * spline_3
                                        mapping_x3 = spline_1 * spline_2 * spline_x3

                                        coeff_x = arr_coeffs_x[i_basis_1, i_basis_2, i_basis_3]
                                        coeff_y = arr_coeffs_y[i_basis_1, i_basis_2, i_basis_3]
                                        coeff_z = arr_coeffs_z[i_basis_1, i_basis_2, i_basis_3]

                                        coeff_weight = arr_coeff_weights[i_basis_1, i_basis_2, i_basis_3]

                                        arr_x[i_quad_1, i_quad_2, i_quad_3] += mapping * coeff_x * coeff_weight
                                        arr_y[i_quad_1, i_quad_2, i_quad_3] += mapping * coeff_y * coeff_weight
                                        arr_z[i_quad_1, i_quad_2, i_quad_3] += mapping * coeff_z * coeff_weight

                                        arr_x_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_x * coeff_weight
                                        arr_x_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_x * coeff_weight
                                        arr_x_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_x * coeff_weight

                                        arr_y_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_y * coeff_weight
                                        arr_y_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_y * coeff_weight
                                        arr_y_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_y * coeff_weight

                                        arr_z_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_z * coeff_weight
                                        arr_z_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_z * coeff_weight
                                        arr_z_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_z * coeff_weight

                                        arr_weights[i_quad_1, i_quad_2, i_quad_3] += mapping * coeff_weight

                                        arr_weights_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_weight
                                        arr_weights_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_weight
                                        arr_weights_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_weight

                            x = arr_x[i_quad_1, i_quad_2, i_quad_3]
                            y = arr_y[i_quad_1, i_quad_2, i_quad_3]
                            z = arr_z[i_quad_1, i_quad_2, i_quad_3]

                            x_x1 = arr_x_x1[i_quad_1, i_quad_2, i_quad_3]
                            x_x2 = arr_x_x2[i_quad_1, i_quad_2, i_quad_3]
                            x_x3 = arr_x_x3[i_quad_1, i_quad_2, i_quad_3]

                            y_x1 = arr_y_x1[i_quad_1, i_quad_2, i_quad_3]
                            y_x2 = arr_y_x2[i_quad_1, i_quad_2, i_quad_3]
                            y_x3 = arr_y_x3[i_quad_1, i_quad_2, i_quad_3]

                            z_x1 = arr_z_x1[i_quad_1, i_quad_2, i_quad_3]
                            z_x2 = arr_z_x2[i_quad_1, i_quad_2, i_quad_3]
                            z_x3 = arr_z_x3[i_quad_1, i_quad_2, i_quad_3]

                            weight = arr_weights[i_quad_1, i_quad_2, i_quad_3]

                            weight_x1 = arr_weights_x1[i_quad_1, i_quad_2, i_quad_3]
                            weight_x2 = arr_weights_x2[i_quad_1, i_quad_2, i_quad_3]
                            weight_x3 = arr_weights_x3[i_quad_1, i_quad_2, i_quad_3]

                            inv_weight = 1.0 / weight

                            x_x1 = (x_x1 - weight_x1 * x * inv_weight) * inv_weight
                            x_x2 = (x_x2 - weight_x2 * x * inv_weight) * inv_weight
                            x_x3 = (x_x3 - weight_x3 * x * inv_weight) * inv_weight

                            y_x1 = (y_x1 - weight_x1 * y * inv_weight) * inv_weight
                            y_x2 = (y_x2 - weight_x2 * y * inv_weight) * inv_weight
                            y_x3 = (y_x3 - weight_x3 * y * inv_weight) * inv_weight

                            z_x1 = (z_x1 - weight_x1 * z * inv_weight) * inv_weight
                            z_x2 = (z_x2 - weight_x2 * z * inv_weight) * inv_weight
                            z_x3 = (z_x3 - weight_x3 * z * inv_weight) * inv_weight

                            jac_det[i_cell_1 * k1 + i_quad_1,
                                    i_cell_2 * k2 + i_quad_2,
                                    i_cell_3 * k3 + i_quad_3] = (+ x_x1 * y_x2 * z_x3
                                                                 + x_x2 * y_x3 * z_x1
                                                                 + x_x3 * y_x1 * z_x2
                                                                 - x_x1 * y_x3 * z_x2
                                                                 - x_x2 * y_x1 * z_x3
                                                                 - x_x3 * y_x2 * z_x1)


def eval_jac_det_2d_weights(nc1: int, nc2: int, f_p1: int, f_p2: int, k1: int, k2: int,
                            global_basis_1: 'float[:,:,:,:]', global_basis_2: 'float[:,:,:,:]',
                            global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_arr_coeff_x: 'T[:,:]',
                            global_arr_coeff_y: 'T[:,:]', global_arr_coeff_weights: 'float[:,:]',
                            jac_det: 'T[:,:]'):
    """
    Parameters
    ----------
    nc1: int
        Number of cells in the X1 direction
    nc2: int
        Number of cells in the X2 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction

    k1: int
        Number of evaluation points in the X1 direction
    k2: int
        Number of evaluation points in the X2 direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X1 direction
    global_basis_2: ndarray of floats
        Basis functions values at each cell and quadrature points in the X2 direction

    global_spans_1: ndarray of ints
        Spans in the X1 direction
    global_spans_2: ndarray of ints
        Spans in the X2 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field

    global_arr_coeff_weights: ndarray of floats
        Coefficients of the weights field

    jac_det: ndarray of floats
        Jacobian determinant on the grid
    """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2))

    arr_x = np.zeros_like(global_arr_coeff_x, shape=(k1, k2))
    arr_y = np.zeros_like(global_arr_coeff_y, shape=(k1, k2))

    arr_x_x1 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2))
    arr_x_x2 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2))

    arr_y_x1 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2))
    arr_y_x2 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2))

    arr_coeff_weights = np.zeros((1 + f_p1, 1 + f_p2))

    arr_weights = np.zeros((k1, k2))

    arr_weights_x1 = np.zeros((k1, k2))
    arr_weights_x2 = np.zeros((k1, k2))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]

            arr_x[:, :] = 0.0
            arr_y[:, :] = 0.0

            arr_x_x1[:, :] = 0.0
            arr_x_x2[:, :] = 0.0

            arr_y_x1[:, :] = 0.0
            arr_y_x2[:, :] = 0.0

            arr_weights[:, :] = 0.0

            arr_weights_x1[:, :] = 0.0
            arr_weights_x2[:, :] = 0.0

            arr_coeffs_x[:, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                    span_2 - f_p2:1 + span_2]

            arr_coeffs_y[:, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                    span_2 - f_p2:1 + span_2]

            arr_coeff_weights[:, :] = global_arr_coeff_weights[span_1 - f_p1:1 + span_1,
                                                               span_2 - f_p2:1 + span_2]

            for i_quad_1 in range(k1):
                for i_quad_2 in range(k2):
                    for i_basis_1 in range(1 + f_p1):
                        spline_1 = global_basis_1[i_cell_1, i_basis_1, 0, i_quad_1]
                        spline_x1 = global_basis_1[i_cell_1, i_basis_1, 1, i_quad_1]

                        for i_basis_2 in range(1 + f_p2):
                            spline_2 = global_basis_2[i_cell_2, i_basis_2, 0, i_quad_2]
                            spline_x2 = global_basis_2[i_cell_2, i_basis_2, 1, i_quad_2]

                            mapping = spline_1 * spline_2
                            mapping_x1 = spline_x1 * spline_2
                            mapping_x2 = spline_1 * spline_x2

                            coeff_x = arr_coeffs_x[i_basis_1, i_basis_2]
                            coeff_y = arr_coeffs_y[i_basis_1, i_basis_2]

                            coeff_weight = arr_coeff_weights[i_basis_1, i_basis_2]

                            arr_x[i_quad_1, i_quad_2] += mapping * coeff_x * coeff_weight
                            arr_y[i_quad_1, i_quad_2] += mapping * coeff_y * coeff_weight

                            arr_x_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_x * coeff_weight
                            arr_x_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_x * coeff_weight

                            arr_y_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_y * coeff_weight
                            arr_y_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_y * coeff_weight

                            arr_weights[i_quad_1, i_quad_2] += mapping * coeff_weight

                            arr_weights_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_weight
                            arr_weights_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_weight

                    x = arr_x[i_quad_1, i_quad_2]
                    y = arr_y[i_quad_1, i_quad_2]

                    x_x1 = arr_x_x1[i_quad_1, i_quad_2]
                    x_x2 = arr_x_x2[i_quad_1, i_quad_2]

                    y_x1 = arr_y_x1[i_quad_1, i_quad_2]
                    y_x2 = arr_y_x2[i_quad_1, i_quad_2]

                    weight = arr_weights[i_quad_1, i_quad_2]

                    weight_x1 = arr_weights_x1[i_quad_1, i_quad_2]
                    weight_x2 = arr_weights_x2[i_quad_1, i_quad_2]

                    inv_weight = 1.0 / weight

                    x_x1 = (x_x1 - weight_x1 * x * inv_weight) * inv_weight
                    x_x2 = (x_x2 - weight_x2 * x * inv_weight) * inv_weight

                    y_x1 = (y_x1 - weight_x1 * y * inv_weight) * inv_weight
                    y_x2 = (y_x2 - weight_x2 * y * inv_weight) * inv_weight

                    jac_det[i_cell_1 * k1 + i_quad_1,
                            i_cell_2 * k2 + i_quad_2] = (x_x1 * y_x2 - x_x2 * y_x1)


# -----------------------------------------------------------------------------
# 4: Irregular tensor grid with weights
# -----------------------------------------------------------------------------
def eval_jac_det_irregular_3d_weights(np1: int, np2: int, np3: int, f_p1: int, f_p2: int,
                                      f_p3: int, cell_index_1: 'int[:]', cell_index_2: 'int[:]', cell_index_3 : 'int[:]',
                                      global_basis_1: 'float[:,:,:]', global_basis_2: 'float[:,:,:]',
                                      global_basis_3: 'float[:,:,:]', global_spans_1: 'int[:]', global_spans_2: 'int[:]',
                                      global_spans_3: 'int[:]', global_arr_coeff_x: 'T[:,:,:]',
                                      global_arr_coeff_y: 'T[:,:,:]', global_arr_coeff_z: 'T[:,:,:]',
                                      global_arr_coeff_weights: 'float[:,:, :]', jac_det: 'T[:,:,:]'):
    """
    Parameters
    ----------
    np1 : int
        Number of points in the X1 direction
    np2 : int
        Number of points in the X2 direction
    np3 : int
        Number of points in the X3 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction
    f_p3: int
        Degree in the X3 direction

    cell_index_1 : ndarray of ints
        Index of the cells in the X1 direction
    cell_index_2 : ndarray of ints
        Index of the cells in the X2 direction
    cell_index_3 : ndarray of ints
        Index of the cells in the X3 direction

    global_basis_1 : ndarray of floats
        Basis functions values at each point in the X1 direction
    global_basis_2 : ndarray of floats
        Basis functions values at each point in the X2 direction
    global_basis_3 : ndarray of floats
        Basis functions values at each point in the X3 direction

    global_spans_1 : ndarray of ints
        Spans in the X1 direction
    global_spans_2 : ndarray of ints
        Spans in the X2 direction
    global_spans_3 : ndarray of ints
        Spans in the X3 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field
    global_arr_coeff_z: ndarray of floats
        Coefficients of the X3 field

    global_arr_coeff_weights: ndarray of floats
        Coefficients of the weights field

    jac_det: ndarray of floats
        Jacobian determinant on the grid
    """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_z = np.zeros_like(global_arr_coeff_z, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))

    arr_coeffs_weights = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))

    temp_x = arr_coeffs_x[0,0,0]-arr_coeffs_x[0,0,0]
    temp_y = arr_coeffs_y[0,0,0]-arr_coeffs_y[0,0,0]
    temp_z = arr_coeffs_z[0,0,0]-arr_coeffs_z[0,0,0]

    temp_x_x1 = arr_coeffs_x[0,0,0]-arr_coeffs_x[0,0,0]
    temp_x_x2 = arr_coeffs_y[0,0,0]-arr_coeffs_y[0,0,0]
    temp_x_x3 = arr_coeffs_z[0,0,0]-arr_coeffs_z[0,0,0]

    temp_y_x1 = arr_coeffs_x[0,0,0]-arr_coeffs_x[0,0,0]
    temp_y_x2 = arr_coeffs_y[0,0,0]-arr_coeffs_y[0,0,0]
    temp_y_x3 = arr_coeffs_z[0,0,0]-arr_coeffs_z[0,0,0]

    temp_z_x1 = arr_coeffs_x[0,0,0]-arr_coeffs_x[0,0,0]
    temp_z_x2 = arr_coeffs_y[0,0,0]-arr_coeffs_y[0,0,0]
    temp_z_x3 = arr_coeffs_z[0,0,0]-arr_coeffs_z[0,0,0]

    for i_p_1 in range(np1):
        i_cell_1 = cell_index_1[i_p_1]
        span_1 = global_spans_1[i_cell_1]

        for i_p_2 in range(np2):
            i_cell_2 = cell_index_2[i_p_2]
            span_2 = global_spans_2[i_cell_2]

            for i_p_3 in range(np3):
                i_cell_3 = cell_index_3[i_p_3]
                span_3 = global_spans_3[i_cell_3]

                arr_coeffs_x[:, :, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_y[:, :, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_z[:, :, :] = global_arr_coeff_z[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_weights[:, :, :] = global_arr_coeff_weights[span_1 - f_p1:1 + span_1,
                                                                       span_2 - f_p2:1 + span_2,
                                                                       span_3 - f_p3:1 + span_3]

                temp_x *= 0.0
                temp_y *= 0.0
                temp_z *= 0.0

                temp_weight= 0.0

                temp_x_x1 *= 0.0
                temp_x_x2 *= 0.0
                temp_x_x3 *= 0.0

                temp_y_x1 *= 0.0
                temp_y_x2 *= 0.0
                temp_y_x3 *= 0.0

                temp_z_x1 *= 0.0
                temp_z_x2 *= 0.0
                temp_z_x3 *= 0.0

                temp_weight_x1 = 0.0
                temp_weight_x2 = 0.0
                temp_weight_x3 = 0.0

                for i_basis_1 in range(1 + f_p1):
                    spline_1 = global_basis_1[i_p_1, i_basis_1, 0]
                    spline_x1 = global_basis_1[i_p_1, i_basis_1, 1]

                    for i_basis_2 in range(1 + f_p2):
                        spline_2 = global_basis_2[i_p_2, i_basis_2, 0]
                        spline_x2 = global_basis_2[i_p_2, i_basis_2, 1]

                        for i_basis_3 in range(1 + f_p3):
                            spline_3 = global_basis_3[i_p_3, i_basis_3, 0]
                            spline_x3 = global_basis_3[i_p_3, i_basis_3, 1]

                            mapping = spline_1 * spline_2 * spline_3

                            mapping_x1 = spline_x1 * spline_2 * spline_3
                            mapping_x2 = spline_1 * spline_x2 * spline_3
                            mapping_x3 = spline_1 * spline_2 * spline_x3

                            coeff_x = arr_coeffs_x[i_basis_1, i_basis_2, i_basis_3]
                            coeff_y = arr_coeffs_y[i_basis_1, i_basis_2, i_basis_3]
                            coeff_z = arr_coeffs_z[i_basis_1, i_basis_2, i_basis_3]

                            coeff_weight = arr_coeffs_weights[i_basis_1, i_basis_2, i_basis_3]

                            temp_x += mapping * coeff_x * coeff_weight
                            temp_y += mapping * coeff_y * coeff_weight
                            temp_z += mapping * coeff_z * coeff_weight

                            temp_weight += mapping * coeff_weight

                            temp_x_x1 += mapping_x1 * coeff_x * coeff_weight
                            temp_x_x2 += mapping_x2 * coeff_x * coeff_weight
                            temp_x_x3 += mapping_x3 * coeff_x * coeff_weight

                            temp_y_x1 += mapping_x1 * coeff_y * coeff_weight
                            temp_y_x2 += mapping_x2 * coeff_y * coeff_weight
                            temp_y_x3 += mapping_x3 * coeff_y * coeff_weight

                            temp_z_x1 += mapping_x1 * coeff_z * coeff_weight
                            temp_z_x2 += mapping_x2 * coeff_z * coeff_weight
                            temp_z_x3 += mapping_x3 * coeff_z * coeff_weight

                            temp_weight_x1 += mapping_x1 * coeff_weight
                            temp_weight_x2 += mapping_x2 * coeff_weight
                            temp_weight_x3 += mapping_x3 * coeff_weight

                inv_weight = 1.0 / temp_weight

                x_x1 = (temp_x_x1 - temp_weight_x1 * temp_x * inv_weight) * inv_weight
                x_x2 = (temp_x_x2 - temp_weight_x2 * temp_x * inv_weight) * inv_weight
                x_x3 = (temp_x_x3 - temp_weight_x3 * temp_x * inv_weight) * inv_weight

                y_x1 = (temp_y_x1 - temp_weight_x1 * temp_y * inv_weight) * inv_weight
                y_x2 = (temp_y_x2 - temp_weight_x2 * temp_y * inv_weight) * inv_weight
                y_x3 = (temp_y_x3 - temp_weight_x3 * temp_y * inv_weight) * inv_weight

                z_x1 = (temp_z_x1 - temp_weight_x1 * temp_z * inv_weight) * inv_weight
                z_x2 = (temp_z_x2 - temp_weight_x2 * temp_z * inv_weight) * inv_weight
                z_x3 = (temp_z_x3 - temp_weight_x3 * temp_z * inv_weight) * inv_weight

                jac_det[i_p_1, i_p_2, i_p_3] = (+ x_x1 * y_x2 * z_x3
                                                + x_x2 * y_x3 * z_x1
                                                + x_x3 * y_x1 * z_x2
                                                - x_x1 * y_x3 * z_x2
                                                - x_x2 * y_x1 * z_x3
                                                - x_x3 * y_x2 * z_x1)


def eval_jac_det_irregular_2d_weights(np1: int, np2: int, f_p1: int, f_p2: int,
                                      cell_index_1: 'int[:]', cell_index_2: 'int[:]',
                                      global_basis_1: 'float[:,:,:]', global_basis_2: 'float[:,:,:]',
                                      global_spans_1: 'int[:]', global_spans_2: 'int[:]',
                                      global_arr_coeff_x: 'T[:,:]', global_arr_coeff_y: 'T[:,:]',
                                      global_arr_coeff_weights: 'float[:,:]',
                                      jac_det: 'T[:,:]'):
    """
    Parameters
    ----------
    np1 : int
        Number of points in the X1 direction
    np2 : int
        Number of points in the X2 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction

    cell_index_1 : ndarray of ints
        Index of the cells in the X1 direction
    cell_index_2 : ndarray of ints
        Index of the cells in the X2 direction

    global_basis_1 : ndarray of floats
        Basis functions values at each point in the X1 direction
    global_basis_2 : ndarray of floats
        Basis functions values at each point in the X2 direction

    global_spans_1 : ndarray of ints
        Spans in the X1 direction
    global_spans_2 : ndarray of ints
        Spans in the X2 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field

    global_arr_coeff_weights: ndarray of floats
        Coefficients of the weights field

    jac_det: ndarray of floats
        Jacobian determinant on the grid
    """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2))

    arr_coeffs_weights = np.zeros((1 + f_p1, 1 + f_p2))

    temp_x = arr_coeffs_x[0,0]-arr_coeffs_x[0,0]
    temp_y = arr_coeffs_y[0,0]-arr_coeffs_y[0,0]

    temp_x_x1 = arr_coeffs_x[0,0]-arr_coeffs_x[0,0]
    temp_x_x2 = arr_coeffs_y[0,0]-arr_coeffs_y[0,0]

    temp_y_x1 = arr_coeffs_x[0,0]-arr_coeffs_x[0,0]
    temp_y_x2 = arr_coeffs_y[0,0]-arr_coeffs_y[0,0]

    for i_p_1 in range(np1):
        i_cell_1 = cell_index_1[i_p_1]
        span_1 = global_spans_1[i_cell_1]

        for i_p_2 in range(np2):
            i_cell_2 = cell_index_2[i_p_2]
            span_2 = global_spans_2[i_cell_2]

            arr_coeffs_x[:, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                    span_2 - f_p2:1 + span_2]
            arr_coeffs_y[:, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                    span_2 - f_p2:1 + span_2]

            arr_coeffs_weights[:, :] = global_arr_coeff_weights[span_1 - f_p1:1 + span_1,
                                                                span_2 - f_p2:1 + span_2]
            temp_x *= 0.0
            temp_y *= 0.0

            temp_weight = 0.0

            temp_x_x1 *= 0.0
            temp_x_x2 *= 0.0

            temp_y_x1 *= 0.0
            temp_y_x2 *= 0.0

            temp_weight_x1 = 0.0
            temp_weight_x2 = 0.0

            for i_basis_1 in range(1 + f_p1):
                spline_1 = global_basis_1[i_p_1, i_basis_1, 0]
                spline_x1 = global_basis_1[i_p_1, i_basis_1, 1]

                for i_basis_2 in range(1 + f_p2):
                    spline_2 = global_basis_2[i_p_2, i_basis_2, 0]
                    spline_x2 = global_basis_2[i_p_2, i_basis_2, 1]

                    mapping = spline_1 * spline_2

                    mapping_x1 = spline_x1 * spline_2
                    mapping_x2 = spline_1 * spline_x2

                    coeff_x = arr_coeffs_x[i_basis_1, i_basis_2]
                    coeff_y = arr_coeffs_y[i_basis_1, i_basis_2]

                    coeff_weight = arr_coeffs_weights[i_basis_1, i_basis_2]

                    temp_x += mapping * coeff_x * coeff_weight
                    temp_y += mapping * coeff_y * coeff_weight

                    temp_weight += mapping * coeff_weight

                    temp_x_x1 += mapping_x1 * coeff_x * coeff_weight
                    temp_x_x2 += mapping_x2 * coeff_x * coeff_weight

                    temp_y_x1 += mapping_x1 * coeff_y * coeff_weight
                    temp_y_x2 += mapping_x2 * coeff_y * coeff_weight

                    temp_weight_x1 += mapping_x1 * coeff_weight
                    temp_weight_x2 += mapping_x2 * coeff_weight

            inv_weight = 1.0 / temp_weight

            x_x1 = (temp_x_x1 - temp_weight_x1 * temp_x * inv_weight) * inv_weight
            x_x2 = (temp_x_x2 - temp_weight_x2 * temp_x * inv_weight) * inv_weight

            y_x1 = (temp_y_x1 - temp_weight_x1 * temp_y * inv_weight) * inv_weight
            y_x2 = (temp_y_x2 - temp_weight_x2 * temp_y * inv_weight) * inv_weight

            jac_det[i_p_1, i_p_2] = x_x1 * y_x2 - x_x2 * y_x1


# =============================================================================
# Evaluation of the Jacobian matrices
# =============================================================================
# -----------------------------------------------------------------------------
# 1: Regular tensor grid without weights
# -----------------------------------------------------------------------------
def eval_jacobians_3d(nc1: int, nc2: int, nc3: int, f_p1: int, f_p2: int, f_p3: int,
                      k1: int, k2: int, k3: int, global_basis_1: 'float[:,:,:,:]', global_basis_2: 'float[:,:,:,:]',
                      global_basis_3: 'float[:,:,:,:]', global_spans_1: 'int[:]', global_spans_2: 'int[:]',
                      global_spans_3: 'int[:]', global_arr_coeff_x: 'T[:,:,:]', global_arr_coeff_y: 'T[:,:,:]',
                      global_arr_coeff_z: 'T[:,:,:]', jacobians: 'T[:,:,:,:,:]'):
    """
    Parameters
    ----------
    nc1: int
        Number of cells in the X1 direction
    nc2: int
        Number of cells in the X2 direction
    nc3: int
        Number of cells in the X3 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction
    f_p3: int
        Degree in the X3 direction

    k1: int
        Number of evaluation points in the X1 direction
    k2: int
        Number of evaluation points in the X2 direction
    k3: int
        Number of evaluation points in the X3 direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X1 direction
    global_basis_2: ndarray of floats
        Basis functions values at each cell and quadrature points in the X2 direction
    global_basis_3: ndarray of floats
        Basis functions values at each cell and quadrature points in the X3 direction

    global_spans_1: ndarray of ints
        Spans in the X1 direction
    global_spans_2: ndarray of ints
        Spans in the X2 direction
    global_spans_3: ndarray of ints
        Spans in the X3 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field
    global_arr_coeff_z: ndarray of floats
        Coefficients of the X3 field

    jacobians: ndarray of floats
        Jacobian matrix on the grid
    """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_z = np.zeros_like(global_arr_coeff_z, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))

    arr_x_x1 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2, k3))
    arr_x_x2 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2, k3))
    arr_x_x3 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2, k3))

    arr_y_x1 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2, k3))
    arr_y_x2 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2, k3))
    arr_y_x3 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2, k3))

    arr_z_x1 = np.zeros_like(global_arr_coeff_z, shape=(k1, k2, k3))
    arr_z_x2 = np.zeros_like(global_arr_coeff_z, shape=(k1, k2, k3))
    arr_z_x3 = np.zeros_like(global_arr_coeff_z, shape=(k1, k2, k3))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]

            for i_cell_3 in range(nc3):
                span_3 = global_spans_3[i_cell_3]

                arr_x_x1[:, :, :] = 0.0
                arr_x_x2[:, :, :] = 0.0
                arr_x_x3[:, :, :] = 0.0

                arr_y_x1[:, :, :] = 0.0
                arr_y_x2[:, :, :] = 0.0
                arr_y_x3[:, :, :] = 0.0

                arr_z_x1[:, :, :] = 0.0
                arr_z_x2[:, :, :] = 0.0
                arr_z_x3[:, :, :] = 0.0

                arr_coeffs_x[:, :, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_y[:, :, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_z[:, :, :] = global_arr_coeff_z[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]
                for i_quad_1 in range(k1):
                    for i_quad_2 in range(k2):
                        for i_quad_3 in range(k3):
                            for i_basis_1 in range(1 + f_p1):
                                spline_1 = global_basis_1[i_cell_1, i_basis_1, 0, i_quad_1]
                                spline_x1 = global_basis_1[i_cell_1, i_basis_1, 1, i_quad_1]

                                for i_basis_2 in range(1 + f_p2):
                                    spline_2 = global_basis_2[i_cell_2, i_basis_2, 0, i_quad_2]
                                    spline_x2 = global_basis_2[i_cell_2, i_basis_2, 1, i_quad_2]

                                    for i_basis_3 in range(1 + f_p3):
                                        spline_3 = global_basis_3[i_cell_3, i_basis_3, 0, i_quad_3]
                                        spline_x3 = global_basis_3[i_cell_3, i_basis_3, 1, i_quad_3]

                                        mapping_x1 = spline_x1 * spline_2 * spline_3
                                        mapping_x2 = spline_1 * spline_x2 * spline_3
                                        mapping_x3 = spline_1 * spline_2 * spline_x3

                                        coeff_x = arr_coeffs_x[i_basis_1, i_basis_2, i_basis_3]
                                        coeff_y = arr_coeffs_y[i_basis_1, i_basis_2, i_basis_3]
                                        coeff_z = arr_coeffs_z[i_basis_1, i_basis_2, i_basis_3]

                                        arr_x_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_x
                                        arr_x_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_x
                                        arr_x_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_x

                                        arr_y_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_y
                                        arr_y_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_y
                                        arr_y_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_y

                                        arr_z_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_z
                                        arr_z_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_z
                                        arr_z_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_z

                            x_x1 = arr_x_x1[i_quad_1, i_quad_2, i_quad_3]
                            x_x2 = arr_x_x2[i_quad_1, i_quad_2, i_quad_3]
                            x_x3 = arr_x_x3[i_quad_1, i_quad_2, i_quad_3]

                            y_x1 = arr_y_x1[i_quad_1, i_quad_2, i_quad_3]
                            y_x2 = arr_y_x2[i_quad_1, i_quad_2, i_quad_3]
                            y_x3 = arr_y_x3[i_quad_1, i_quad_2, i_quad_3]

                            z_x1 = arr_z_x1[i_quad_1, i_quad_2, i_quad_3]
                            z_x2 = arr_z_x2[i_quad_1, i_quad_2, i_quad_3]
                            z_x3 = arr_z_x3[i_quad_1, i_quad_2, i_quad_3]

                            jacobians[i_cell_1 * k1 + i_quad_1,
                                      i_cell_2 * k2 + i_quad_2,
                                      i_cell_3 * k3 + i_quad_3,
                                      :, :] = np.array([[x_x1, x_x2, x_x3],
                                                        [y_x1, y_x2, y_x3],
                                                        [z_x1, z_x2, z_x3]])


def eval_jacobians_2d(nc1: int, nc2: int, f_p1: int, f_p2: int, k1: int, k2: int,
                      global_basis_1: 'float[:,:,:,:]', global_basis_2: 'float[:,:,:,:]', global_spans_1: 'int[:]',
                      global_spans_2: 'int[:]', global_arr_coeff_x: 'T[:,:]', global_arr_coeff_y: 'T[:,:]',
                      jacobians: 'T[:,:,:,:]'):
    """
    Parameters
    ----------
    nc1: int
        Number of cells in the X1 direction
    nc2: int
        Number of cells in the X2 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction

    k1: int
        Number of evaluation points in the X1 direction
    k2: int
        Number of evaluation points in the X2 direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X1 direction
    global_basis_2: ndarray of floats
        Basis functions values at each cell and quadrature points in the X2 direction

    global_spans_1: ndarray of ints
        Spans in the X1 direction
    global_spans_2: ndarray of ints
        Spans in the X2 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field

    jacobians: ndarray of floats
        Jacobian matrix at every point of the grid
    """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2))

    arr_x_x1 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2))
    arr_x_x2 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2))

    arr_y_x1 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2))
    arr_y_x2 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]

            arr_x_x1[:, :] = 0.0
            arr_x_x2[:, :] = 0.0

            arr_y_x1[:, :] = 0.0
            arr_y_x2[:, :] = 0.0

            arr_coeffs_x[:, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                    span_2 - f_p2:1 + span_2]

            arr_coeffs_y[:, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                    span_2 - f_p2:1 + span_2]

            for i_quad_1 in range(k1):
                for i_quad_2 in range(k2):
                    for i_basis_1 in range(1 + f_p1):
                        spline_1 = global_basis_1[i_cell_1, i_basis_1, 0, i_quad_1]
                        spline_x1 = global_basis_1[i_cell_1, i_basis_1, 1, i_quad_1]

                        for i_basis_2 in range(1 + f_p2):
                            spline_2 = global_basis_2[i_cell_2, i_basis_2, 0, i_quad_2]
                            spline_x2 = global_basis_2[i_cell_2, i_basis_2, 1, i_quad_2]

                            mapping_x1 = spline_x1 * spline_2
                            mapping_x2 = spline_1 * spline_x2

                            coeff_x = arr_coeffs_x[i_basis_1, i_basis_2]
                            coeff_y = arr_coeffs_y[i_basis_1, i_basis_2]

                            arr_x_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_x
                            arr_x_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_x

                            arr_y_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_y
                            arr_y_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_y

                    x_x1 = arr_x_x1[i_quad_1, i_quad_2]
                    x_x2 = arr_x_x2[i_quad_1, i_quad_2]

                    y_x1 = arr_y_x1[i_quad_1, i_quad_2]
                    y_x2 = arr_y_x2[i_quad_1, i_quad_2]

                    jacobians[i_cell_1 * k1 + i_quad_1,
                              i_cell_2 * k2 + i_quad_2,
                              :, :] = np.array([[x_x1, x_x2],
                                                [y_x1, y_x2]])


# -----------------------------------------------------------------------------
# 2: Irregular tensor grid without weights
# -----------------------------------------------------------------------------
def eval_jacobians_irregular_3d(np1: int, np2: int, np3: int, f_p1: int, f_p2: int,
                                f_p3: int, cell_index_1: 'int[:]', cell_index_2: 'int[:]', cell_index_3 : 'int[:]',
                                global_basis_1: 'float[:,:,:]', global_basis_2: 'float[:,:,:]',
                                global_basis_3: 'float[:,:,:]', global_spans_1: 'int[:]', global_spans_2: 'int[:]',
                                global_spans_3: 'int[:]', global_arr_coeff_x: 'T[:,:,:]',
                                global_arr_coeff_y: 'T[:,:,:]', global_arr_coeff_z: 'T[:,:,:]',
                                jacobians: 'T[:,:,:,:,:]'):
    """
    Parameters
    ----------
    np1 : int
        Number of points in the X1 direction
    np2 : int
        Number of points in the X2 direction
    np3 : int
        Number of points in the X3 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction
    f_p3: int
        Degree in the X3 direction

    cell_index_1 : ndarray of ints
        Index of the cells in the X1 direction
    cell_index_2 : ndarray of ints
        Index of the cells in the X2 direction
    cell_index_3 : ndarray of ints
        Index of the cells in the X3 direction

    global_basis_1 : ndarray of floats
        Basis functions values at each point in the X1 direction
    global_basis_2 : ndarray of floats
        Basis functions values at each point in the X2 direction
    global_basis_3 : ndarray of floats
        Basis functions values at each point in the X3 direction

    global_spans_1 : ndarray of ints
        Spans in the X1 direction
    global_spans_2 : ndarray of ints
        Spans in the X2 direction
    global_spans_3 : ndarray of ints
        Spans in the X3 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field
    global_arr_coeff_z: ndarray of floats
        Coefficients of the X3 field

    jacobians: ndarray of floats
        Jacobian matrix on the grid
    """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_z = np.zeros_like(global_arr_coeff_z, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))

    temp_x_x1 = arr_coeffs_x[0,0,0]-arr_coeffs_x[0,0,0]
    temp_x_x2 = arr_coeffs_y[0,0,0]-arr_coeffs_y[0,0,0]
    temp_x_x3 = arr_coeffs_z[0,0,0]-arr_coeffs_z[0,0,0]

    temp_y_x1 = arr_coeffs_x[0,0,0]-arr_coeffs_x[0,0,0]
    temp_y_x2 = arr_coeffs_y[0,0,0]-arr_coeffs_y[0,0,0]
    temp_y_x3 = arr_coeffs_z[0,0,0]-arr_coeffs_z[0,0,0]

    temp_z_x1 = arr_coeffs_x[0,0,0]-arr_coeffs_x[0,0,0]
    temp_z_x2 = arr_coeffs_y[0,0,0]-arr_coeffs_y[0,0,0]
    temp_z_x3 = arr_coeffs_z[0,0,0]-arr_coeffs_z[0,0,0]

    for i_p_1 in range(np1):
        i_cell_1 = cell_index_1[i_p_1]
        span_1 = global_spans_1[i_cell_1]

        for i_p_2 in range(np2):
            i_cell_2 = cell_index_2[i_p_2]
            span_2 = global_spans_2[i_cell_2]

            for i_p_3 in range(np3):
                i_cell_3 = cell_index_3[i_p_3]
                span_3 = global_spans_3[i_cell_3]

                arr_coeffs_x[:, :, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_y[:, :, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_z[:, :, :] = global_arr_coeff_z[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                temp_x_x1 *= 0.0
                temp_x_x2 *= 0.0
                temp_x_x3 *= 0.0

                temp_y_x1 *= 0.0
                temp_y_x2 *= 0.0
                temp_y_x3 *= 0.0

                temp_z_x1 *= 0.0
                temp_z_x2 *= 0.0
                temp_z_x3 *= 0.0

                for i_basis_1 in range(1 + f_p1):
                    spline_1 = global_basis_1[i_p_1, i_basis_1, 0]
                    spline_x1 = global_basis_1[i_p_1, i_basis_1, 1]

                    for i_basis_2 in range(1 + f_p2):
                        spline_2 = global_basis_2[i_p_2, i_basis_2, 0]
                        spline_x2 = global_basis_2[i_p_2, i_basis_2, 1]

                        for i_basis_3 in range(1 + f_p3):
                            spline_3 = global_basis_3[i_p_3, i_basis_3, 0]
                            spline_x3 = global_basis_3[i_p_3, i_basis_3, 1]

                            mapping_x1 = spline_x1 * spline_2 * spline_3
                            mapping_x2 = spline_1 * spline_x2 * spline_3
                            mapping_x3 = spline_1 * spline_2 * spline_x3

                            coeff_x = arr_coeffs_x[i_basis_1, i_basis_2, i_basis_3]
                            coeff_y = arr_coeffs_y[i_basis_1, i_basis_2, i_basis_3]
                            coeff_z = arr_coeffs_z[i_basis_1, i_basis_2, i_basis_3]

                            temp_x_x1 += mapping_x1 * coeff_x
                            temp_x_x2 += mapping_x2 * coeff_x
                            temp_x_x3 += mapping_x3 * coeff_x

                            temp_y_x1 += mapping_x1 * coeff_y
                            temp_y_x2 += mapping_x2 * coeff_y
                            temp_y_x3 += mapping_x3 * coeff_y

                            temp_z_x1 += mapping_x1 * coeff_z
                            temp_z_x2 += mapping_x2 * coeff_z
                            temp_z_x3 += mapping_x3 * coeff_z

                jacobians[i_p_1, i_p_2, i_p_3, :, :] = np.array([[temp_x_x1, temp_x_x2, temp_x_x3],
                                                                 [temp_y_x1, temp_y_x2, temp_y_x3],
                                                                 [temp_z_x1, temp_z_x2, temp_z_x3]])


def eval_jacobians_irregular_2d(np1: int, np2: int, f_p1: int, f_p2: int, cell_index_1: 'int[:]',
                                cell_index_2: 'int[:]', global_basis_1: 'float[:,:,:]', global_basis_2: 'float[:,:,:]',
                                global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_arr_coeff_x: 'T[:,:]',
                                global_arr_coeff_y: 'T[:,:]', jacobians: 'T[:,:,:,:]'):
    """
    Parameters
    ----------
    np1 : int
        Number of points in the X1 direction
    np2 : int
        Number of points in the X2 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction

    cell_index_1 : ndarray of ints
        Index of the cells in the X1 direction
    cell_index_2 : ndarray of ints
        Index of the cells in the X2 direction

    global_basis_1 : ndarray of floats
        Basis functions values at each point in the X1 direction
    global_basis_2 : ndarray of floats
        Basis functions values at each point in the X2 direction

    global_spans_1 : ndarray of ints
        Spans in the X1 direction
    global_spans_2 : ndarray of ints
        Spans in the X2 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field

    jacobians: ndarray of floats
        Jacobian matrix on the grid
    """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2))

    temp_x_x1 = arr_coeffs_x[0,0]-arr_coeffs_x[0,0]
    temp_x_x2 = arr_coeffs_y[0,0]-arr_coeffs_y[0,0]

    temp_y_x1 = arr_coeffs_x[0,0]-arr_coeffs_x[0,0]
    temp_y_x2 = arr_coeffs_y[0,0]-arr_coeffs_y[0,0]

    for i_p_1 in range(np1):
        i_cell_1 = cell_index_1[i_p_1]
        span_1 = global_spans_1[i_cell_1]

        for i_p_2 in range(np2):
            i_cell_2 = cell_index_2[i_p_2]
            span_2 = global_spans_2[i_cell_2]

            arr_coeffs_x[:, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                       span_2 - f_p2:1 + span_2]

            arr_coeffs_y[:, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                       span_2 - f_p2:1 + span_2]

            temp_x_x1 *= 0.0
            temp_x_x2 *= 0.0

            temp_y_x1 *= 0.0
            temp_y_x2 *= 0.0

            for i_basis_1 in range(1 + f_p1):
                spline_1 = global_basis_1[i_p_1, i_basis_1, 0]
                spline_x1 = global_basis_1[i_p_1, i_basis_1, 1]

                for i_basis_2 in range(1 + f_p2):
                    spline_2 = global_basis_2[i_p_2, i_basis_2, 0]
                    spline_x2 = global_basis_2[i_p_2, i_basis_2, 1]

                    mapping_x1 = spline_x1 * spline_2
                    mapping_x2 = spline_1 * spline_x2

                    coeff_x = arr_coeffs_x[i_basis_1, i_basis_2]
                    coeff_y = arr_coeffs_y[i_basis_1, i_basis_2]

                    temp_x_x1 += mapping_x1 * coeff_x
                    temp_x_x2 += mapping_x2 * coeff_x

                    temp_y_x1 += mapping_x1 * coeff_y
                    temp_y_x2 += mapping_x2 * coeff_y

            jacobians[i_p_1, i_p_2, :, :] = np.array([[temp_x_x1, temp_x_x2],
                                                      [temp_y_x1, temp_y_x2]])

# -----------------------------------------------------------------------------
# 3: Regular tensor grid with weights
# -----------------------------------------------------------------------------
def eval_jacobians_3d_weights(nc1: int, nc2: int, nc3: int,  f_p1: int, f_p2: int,
                              f_p3: int, k1: int, k2: int, k3: int, global_basis_1: 'float[:,:,:,:]',
                              global_basis_2: 'float[:,:,:,:]', global_basis_3: 'float[:,:,:,:]',
                              global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_spans_3: 'int[:]',
                              global_arr_coeff_x: 'T[:,:,:]', global_arr_coeff_y: 'T[:,:,:]',
                              global_arr_coeff_z: 'T[:,:,:]', global_arr_coeff_weights: 'float[:,:,:]',
                              jacobians: 'T[:,:,:,:,:]'):

    """
    Parameters
    ----------
    nc1: int
        Number of cells in the X1 direction
    nc2: int
        Number of cells in the X2 direction
    nc3: int
        Number of cells in the X3 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction
    f_p3: int
        Degree in the X3 direction

    k1: int
        Number of evaluation points in the X1 direction
    k2: int
        Number of evaluation points in the X2 direction
    k3: int
        Number of evaluation points in the X3 direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X1 direction
    global_basis_2: ndarray of floats
        Basis functions values at each cell and quadrature points in the X2 direction
    global_basis_3: ndarray of floats
        Basis functions values at each cell and quadrature points in the X3 direction

    global_spans_1: ndarray of ints
        Spans in the X1 direction
    global_spans_2: ndarray of ints
        Spans in the X2 direction
    global_spans_3: ndarray of ints
        Spans in the X3 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field
    global_arr_coeff_z: ndarray of floats
        Coefficients of the X3 field

    global_arr_coeff_weights: ndarray of floats
        Coefficients of the weight field

    jacobians: ndarray of floats
        Jacobian matrix on the grid
    """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_z = np.zeros_like(global_arr_coeff_z, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))

    arr_x = np.zeros_like(global_arr_coeff_x, shape=(k1, k2, k3))
    arr_y = np.zeros_like(global_arr_coeff_y, shape=(k1, k2, k3))
    arr_z = np.zeros_like(global_arr_coeff_z, shape=(k1, k2, k3))

    arr_x_x1 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2, k3))
    arr_x_x2 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2, k3))
    arr_x_x3 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2, k3))

    arr_y_x1 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2, k3))
    arr_y_x2 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2, k3))
    arr_y_x3 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2, k3))

    arr_z_x1 = np.zeros_like(global_arr_coeff_z, shape=(k1, k2, k3))
    arr_z_x2 = np.zeros_like(global_arr_coeff_z, shape=(k1, k2, k3))
    arr_z_x3 = np.zeros_like(global_arr_coeff_z, shape=(k1, k2, k3))

    arr_coeff_weights = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_weights = np.zeros((k1, k2, k3))

    arr_weights_x1 = np.zeros((k1, k2, k3))
    arr_weights_x2 = np.zeros((k1, k2, k3))
    arr_weights_x3 = np.zeros((k1, k2, k3))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]

            for i_cell_3 in range(nc3):
                span_3 = global_spans_3[i_cell_3]

                arr_x[:, :, :] = 0.0
                arr_y[:, :, :] = 0.0
                arr_z[:, :, :] = 0.0

                arr_x_x1[:, :, :] = 0.0
                arr_x_x2[:, :, :] = 0.0
                arr_x_x3[:, :, :] = 0.0

                arr_y_x1[:, :, :] = 0.0
                arr_y_x2[:, :, :] = 0.0
                arr_y_x3[:, :, :] = 0.0

                arr_z_x1[:, :, :] = 0.0
                arr_z_x2[:, :, :] = 0.0
                arr_z_x3[:, :, :] = 0.0

                arr_weights[:, :, :] = 0.0

                arr_weights_x1[:, :, :] = 0.0
                arr_weights_x2[:, :, :] = 0.0
                arr_weights_x3[:, :, :] = 0.0

                arr_coeffs_x[:, :, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_y[:, :, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_z[:, :, :] = global_arr_coeff_z[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeff_weights[:, :, :] = global_arr_coeff_weights[span_1 - f_p1:1 + span_1,
                                                                      span_2 - f_p2:1 + span_2,
                                                                      span_3 - f_p3:1 + span_3]

                for i_quad_1 in range(k1):
                    for i_quad_2 in range(k2):
                        for i_quad_3 in range(k3):
                            for i_basis_1 in range(1 + f_p1):
                                spline_1 = global_basis_1[i_cell_1, i_basis_1, 0, i_quad_1]
                                spline_x1 = global_basis_1[i_cell_1, i_basis_1, 1, i_quad_1]

                                for i_basis_2 in range(1 + f_p2):
                                    spline_2 = global_basis_2[i_cell_2, i_basis_2, 0, i_quad_2]
                                    spline_x2 = global_basis_2[i_cell_2, i_basis_2, 1, i_quad_2]

                                    for i_basis_3 in range(1 + f_p3):
                                        spline_3 = global_basis_3[i_cell_3, i_basis_3, 0, i_quad_3]
                                        spline_x3 = global_basis_3[i_cell_3, i_basis_3, 1, i_quad_3]

                                        mapping = spline_1 * spline_2 * spline_3
                                        mapping_x1 = spline_x1 * spline_2 * spline_3
                                        mapping_x2 = spline_1 * spline_x2 * spline_3
                                        mapping_x3 = spline_1 * spline_2 * spline_x3

                                        coeff_x = arr_coeffs_x[i_basis_1, i_basis_2, i_basis_3]
                                        coeff_y = arr_coeffs_y[i_basis_1, i_basis_2, i_basis_3]
                                        coeff_z = arr_coeffs_z[i_basis_1, i_basis_2, i_basis_3]

                                        coeff_weight = arr_coeff_weights[i_basis_1, i_basis_2, i_basis_3]

                                        arr_x[i_quad_1, i_quad_2, i_quad_3] += mapping * coeff_x * coeff_weight
                                        arr_y[i_quad_1, i_quad_2, i_quad_3] += mapping * coeff_y * coeff_weight
                                        arr_z[i_quad_1, i_quad_2, i_quad_3] += mapping * coeff_z * coeff_weight

                                        arr_x_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_x * coeff_weight
                                        arr_x_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_x * coeff_weight
                                        arr_x_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_x * coeff_weight

                                        arr_y_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_y * coeff_weight
                                        arr_y_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_y * coeff_weight
                                        arr_y_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_y * coeff_weight

                                        arr_z_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_z * coeff_weight
                                        arr_z_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_z * coeff_weight
                                        arr_z_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_z * coeff_weight

                                        arr_weights[i_quad_1, i_quad_2, i_quad_3] += mapping * coeff_weight

                                        arr_weights_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_weight
                                        arr_weights_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_weight
                                        arr_weights_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_weight

                            x = arr_x[i_quad_1, i_quad_2, i_quad_3]
                            y = arr_y[i_quad_1, i_quad_2, i_quad_3]
                            z = arr_z[i_quad_1, i_quad_2, i_quad_3]

                            x_x1 = arr_x_x1[i_quad_1, i_quad_2, i_quad_3]
                            x_x2 = arr_x_x2[i_quad_1, i_quad_2, i_quad_3]
                            x_x3 = arr_x_x3[i_quad_1, i_quad_2, i_quad_3]

                            y_x1 = arr_y_x1[i_quad_1, i_quad_2, i_quad_3]
                            y_x2 = arr_y_x2[i_quad_1, i_quad_2, i_quad_3]
                            y_x3 = arr_y_x3[i_quad_1, i_quad_2, i_quad_3]

                            z_x1 = arr_z_x1[i_quad_1, i_quad_2, i_quad_3]
                            z_x2 = arr_z_x2[i_quad_1, i_quad_2, i_quad_3]
                            z_x3 = arr_z_x3[i_quad_1, i_quad_2, i_quad_3]

                            weight = arr_weights[i_quad_1, i_quad_2, i_quad_3]

                            weight_x1 = arr_weights_x1[i_quad_1, i_quad_2, i_quad_3]
                            weight_x2 = arr_weights_x2[i_quad_1, i_quad_2, i_quad_3]
                            weight_x3 = arr_weights_x3[i_quad_1, i_quad_2, i_quad_3]

                            inv_weight = 1.0 / weight

                            x_x1 = (x_x1 - weight_x1 * x * inv_weight) * inv_weight
                            x_x2 = (x_x2 - weight_x2 * x * inv_weight) * inv_weight
                            x_x3 = (x_x3 - weight_x3 * x * inv_weight) * inv_weight

                            y_x1 = (y_x1 - weight_x1 * y * inv_weight) * inv_weight
                            y_x2 = (y_x2 - weight_x2 * y * inv_weight) * inv_weight
                            y_x3 = (y_x3 - weight_x3 * y * inv_weight) * inv_weight

                            z_x1 = (z_x1 - weight_x1 * z * inv_weight) * inv_weight
                            z_x2 = (z_x2 - weight_x2 * z * inv_weight) * inv_weight
                            z_x3 = (z_x3 - weight_x3 * z * inv_weight) * inv_weight

                            jacobians[i_cell_1 * k1 + i_quad_1,
                                      i_cell_2 * k2 + i_quad_2,
                                      i_cell_3 * k3 + i_quad_3,
                                      :, :] = np.array([[x_x1, x_x2, x_x3],
                                                        [y_x1, y_x2, y_x3],
                                                        [z_x1, z_x2, z_x3]])


def eval_jacobians_2d_weights(nc1: int, nc2: int,  f_p1: int, f_p2: int, k1: int, k2: int,
                              global_basis_1: 'float[:,:,:,:]', global_basis_2: 'float[:,:,:,:]',
                              global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_arr_coeff_x: 'T[:,:]',
                              global_arr_coeff_y: 'T[:,:]', global_arr_coeff_weights: 'float[:,:]',
                              jacobians: 'T[:,:,:,:]'):
    """
    Parameters
    ----------
    nc1: int
        Number of cells in the X1 direction
    nc2: int
        Number of cells in the X2 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction

    k1: int
        Number of evaluation points in the X1 direction
    k2: int
        Number of evaluation points in the X2 direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X1 direction
    global_basis_2: ndarray of floats
        Basis functions values at each cell and quadrature points in the X2 direction

    global_spans_1: ndarray of ints
        Spans in the X1 direction
    global_spans_2: ndarray of ints
        Spans in the X2 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field

    global_arr_coeff_weights: ndarray of floats
        Coefficients of the weights field

    jacobians: ndarray of floats
        Jacobian matrix at every point of the grid
       """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2))

    arr_x = np.zeros_like(global_arr_coeff_x, shape=(k1, k2))
    arr_y = np.zeros_like(global_arr_coeff_y, shape=(k1, k2))

    arr_x_x1 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2))
    arr_x_x2 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2))

    arr_y_x1 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2))
    arr_y_x2 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2))

    arr_coeff_weights = np.zeros((1 + f_p1, 1 + f_p2))
    arr_weights = np.zeros((k1, k2))

    arr_weights_x1 = np.zeros((k1, k2))
    arr_weights_x2 = np.zeros((k1, k2))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]

            arr_x[:, :] = 0.0
            arr_y[:, :] = 0.0

            arr_x_x1[:, :] = 0.0
            arr_x_x2[:, :] = 0.0

            arr_y_x1[:, :] = 0.0
            arr_y_x2[:, :] = 0.0

            arr_weights[:, :] = 0.0

            arr_weights_x1[:, :] = 0.0
            arr_weights_x2[:, :] = 0.0

            arr_coeffs_x[:, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                    span_2 - f_p2:1 + span_2]

            arr_coeffs_y[:, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                    span_2 - f_p2:1 + span_2]

            arr_coeff_weights[:, :] = global_arr_coeff_weights[span_1 - f_p1:1 + span_1,
                                                               span_2 - f_p2:1 + span_2]

            for i_quad_1 in range(k1):
                for i_quad_2 in range(k2):
                    for i_basis_1 in range(1 + f_p1):
                        spline_1 = global_basis_1[i_cell_1, i_basis_1, 0, i_quad_1]
                        spline_x1 = global_basis_1[i_cell_1, i_basis_1, 1, i_quad_1]

                        for i_basis_2 in range(1 + f_p2):
                            spline_2 = global_basis_2[i_cell_2, i_basis_2, 0, i_quad_2]
                            spline_x2 = global_basis_2[i_cell_2, i_basis_2, 1, i_quad_2]

                            mapping = spline_1 * spline_2
                            mapping_x1 = spline_x1 * spline_2
                            mapping_x2 = spline_1 * spline_x2

                            coeff_x = arr_coeffs_x[i_basis_1, i_basis_2]
                            coeff_y = arr_coeffs_y[i_basis_1, i_basis_2]

                            coeff_weight = arr_coeff_weights[i_basis_1, i_basis_2]

                            arr_x[i_quad_1, i_quad_2] += mapping * coeff_x * coeff_weight
                            arr_y[i_quad_1, i_quad_2] += mapping * coeff_y * coeff_weight

                            arr_x_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_x * coeff_weight
                            arr_x_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_x * coeff_weight

                            arr_y_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_y * coeff_weight
                            arr_y_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_y * coeff_weight

                            arr_weights[i_quad_1, i_quad_2] += mapping * coeff_weight

                            arr_weights_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_weight
                            arr_weights_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_weight

                    x = arr_x[i_quad_1, i_quad_2]
                    y = arr_y[i_quad_1, i_quad_2]

                    x_x1 = arr_x_x1[i_quad_1, i_quad_2]
                    x_x2 = arr_x_x2[i_quad_1, i_quad_2]

                    y_x1 = arr_y_x1[i_quad_1, i_quad_2]
                    y_x2 = arr_y_x2[i_quad_1, i_quad_2]

                    weight = arr_weights[i_quad_1, i_quad_2]

                    weight_x1 = arr_weights_x1[i_quad_1, i_quad_2]
                    weight_x2 = arr_weights_x2[i_quad_1, i_quad_2]

                    inv_weight = 1.0 / weight

                    x_x1 = (x_x1 - weight_x1 * x * inv_weight) * inv_weight
                    x_x2 = (x_x2 - weight_x2 * x * inv_weight) * inv_weight

                    y_x1 = (y_x1 - weight_x1 * y * inv_weight) * inv_weight
                    y_x2 = (y_x2 - weight_x2 * y * inv_weight) * inv_weight

                    jacobians[i_cell_1 * k1 + i_quad_1,
                              i_cell_2 * k2 + i_quad_2,
                              :, :] = np.array([[x_x1, x_x2],
                                                [y_x1, y_x2]])


# -----------------------------------------------------------------------------
# 4: Irregular tensor grid with weights
# -----------------------------------------------------------------------------
def eval_jacobians_irregular_3d_weights(np1: int, np2: int, np3: int, f_p1: int, f_p2: int,
                                        f_p3: int, cell_index_1: 'int[:]', cell_index_2: 'int[:]', cell_index_3 : 'int[:]',
                                        global_basis_1: 'float[:,:,:]', global_basis_2: 'float[:,:,:]',
                                        global_basis_3: 'float[:,:,:]', global_spans_1: 'int[:]', global_spans_2: 'int[:]',
                                        global_spans_3: 'int[:]', global_arr_coeff_x: 'T[:,:,:]',
                                        global_arr_coeff_y: 'T[:,:,:]', global_arr_coeff_z: 'T[:,:,:]',
                                        global_arr_coeff_weights: 'float[:,:, :]', jacobians: 'T[:,:,:,:,:]'):
    """
    Parameters
    ----------
    np1 : int
        Number of points in the X1 direction
    np2 : int
        Number of points in the X2 direction
    np3 : int
        Number of points in the X3 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction
    f_p3: int
        Degree in the X3 direction

    cell_index_1 : ndarray of ints
        Index of the cells in the X1 direction
    cell_index_2 : ndarray of ints
        Index of the cells in the X2 direction
    cell_index_3 : ndarray of ints
        Index of the cells in the X3 direction

    global_basis_1 : ndarray of floats
        Basis functions values at each point in the X1 direction
    global_basis_2 : ndarray of floats
        Basis functions values at each point in the X2 direction
    global_basis_3 : ndarray of floats
        Basis functions values at each point in the X3 direction

    global_spans_1 : ndarray of ints
        Spans in the X1 direction
    global_spans_2 : ndarray of ints
        Spans in the X2 direction
    global_spans_3 : ndarray of ints
        Spans in the X3 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field
    global_arr_coeff_z: ndarray of floats
        Coefficients of the X3 field

    global_arr_coeff_weights: ndarray of floats
        Coefficients of the weight field

    jacobians: ndarray of floats
        Jacobian matrix on the grid
    """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_z = np.zeros_like(global_arr_coeff_z, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))

    arr_coeffs_weights = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))

    temp_x = arr_coeffs_x[0,0,0]-arr_coeffs_x[0,0,0]
    temp_y = arr_coeffs_y[0,0,0]-arr_coeffs_y[0,0,0]
    temp_z = arr_coeffs_z[0,0,0]-arr_coeffs_z[0,0,0]

    temp_x_x1 = arr_coeffs_x[0,0,0]-arr_coeffs_x[0,0,0]
    temp_x_x2 = arr_coeffs_y[0,0,0]-arr_coeffs_y[0,0,0]
    temp_x_x3 = arr_coeffs_z[0,0,0]-arr_coeffs_z[0,0,0]

    temp_y_x1 = arr_coeffs_x[0,0,0]-arr_coeffs_x[0,0,0]
    temp_y_x2 = arr_coeffs_y[0,0,0]-arr_coeffs_y[0,0,0]
    temp_y_x3 = arr_coeffs_z[0,0,0]-arr_coeffs_z[0,0,0]

    temp_z_x1 = arr_coeffs_x[0,0,0]-arr_coeffs_x[0,0,0]
    temp_z_x2 = arr_coeffs_y[0,0,0]-arr_coeffs_y[0,0,0]
    temp_z_x3 = arr_coeffs_z[0,0,0]-arr_coeffs_z[0,0,0]

    for i_p_1 in range(np1):
        i_cell_1 = cell_index_1[i_p_1]
        span_1 = global_spans_1[i_cell_1]

        for i_p_2 in range(np2):
            i_cell_2 = cell_index_2[i_p_2]
            span_2 = global_spans_2[i_cell_2]

            for i_p_3 in range(np3):
                i_cell_3 = cell_index_3[i_p_3]
                span_3 = global_spans_3[i_cell_3]

                arr_coeffs_x[:, :, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_y[:, :, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_z[:, :, :] = global_arr_coeff_z[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_weights[:, :, :] = global_arr_coeff_weights[span_1 - f_p1:1 + span_1,
                                                                       span_2 - f_p2:1 + span_2,
                                                                       span_3 - f_p3:1 + span_3]

                temp_x *= 0.0
                temp_y *= 0.0
                temp_z *= 0.0

                temp_weight = 0.0

                temp_x_x1 *= 0.0
                temp_x_x2 *= 0.0
                temp_x_x3 *= 0.0

                temp_y_x1 *= 0.0
                temp_y_x2 *= 0.0
                temp_y_x3 *= 0.0

                temp_z_x1 *= 0.0
                temp_z_x2 *= 0.0
                temp_z_x3 *= 0.0

                temp_weight_x1 = 0.0
                temp_weight_x2 = 0.0
                temp_weight_x3 = 0.0

                for i_basis_1 in range(1 + f_p1):
                    spline_1 = global_basis_1[i_p_1, i_basis_1, 0]
                    spline_x1 = global_basis_1[i_p_1, i_basis_1, 1]

                    for i_basis_2 in range(1 + f_p2):
                        spline_2 = global_basis_2[i_p_2, i_basis_2, 0]
                        spline_x2 = global_basis_2[i_p_2, i_basis_2, 1]

                        for i_basis_3 in range(1 + f_p3):
                            spline_3 = global_basis_3[i_p_3, i_basis_3, 0]
                            spline_x3 = global_basis_3[i_p_3, i_basis_3, 1]

                            mapping = spline_1 * spline_2 * spline_3

                            mapping_x1 = spline_x1 * spline_2 * spline_3
                            mapping_x2 = spline_1 * spline_x2 * spline_3
                            mapping_x3 = spline_1 * spline_2 * spline_x3

                            coeff_x = arr_coeffs_x[i_basis_1, i_basis_2, i_basis_3]
                            coeff_y = arr_coeffs_y[i_basis_1, i_basis_2, i_basis_3]
                            coeff_z = arr_coeffs_z[i_basis_1, i_basis_2, i_basis_3]

                            coeff_weight = arr_coeffs_weights[i_basis_1, i_basis_2, i_basis_3]

                            temp_x += mapping * coeff_x * coeff_weight
                            temp_y += mapping * coeff_y * coeff_weight
                            temp_z += mapping * coeff_z * coeff_weight

                            temp_weight += mapping * coeff_weight

                            temp_x_x1 += mapping_x1 * coeff_x * coeff_weight
                            temp_x_x2 += mapping_x2 * coeff_x * coeff_weight
                            temp_x_x3 += mapping_x3 * coeff_x * coeff_weight

                            temp_y_x1 += mapping_x1 * coeff_y * coeff_weight
                            temp_y_x2 += mapping_x2 * coeff_y * coeff_weight
                            temp_y_x3 += mapping_x3 * coeff_y * coeff_weight

                            temp_z_x1 += mapping_x1 * coeff_z * coeff_weight
                            temp_z_x2 += mapping_x2 * coeff_z * coeff_weight
                            temp_z_x3 += mapping_x3 * coeff_z * coeff_weight

                            temp_weight_x1 += mapping_x1 * coeff_weight
                            temp_weight_x2 += mapping_x2 * coeff_weight
                            temp_weight_x3 += mapping_x3 * coeff_weight

                inv_weight = 1.0 / temp_weight

                x_x1 = (temp_x_x1 - temp_weight_x1 * temp_x * inv_weight) * inv_weight
                x_x2 = (temp_x_x2 - temp_weight_x2 * temp_x * inv_weight) * inv_weight
                x_x3 = (temp_x_x3 - temp_weight_x3 * temp_x * inv_weight) * inv_weight

                y_x1 = (temp_y_x1 - temp_weight_x1 * temp_y * inv_weight) * inv_weight
                y_x2 = (temp_y_x2 - temp_weight_x2 * temp_y * inv_weight) * inv_weight
                y_x3 = (temp_y_x3 - temp_weight_x3 * temp_y * inv_weight) * inv_weight

                z_x1 = (temp_z_x1 - temp_weight_x1 * temp_z * inv_weight) * inv_weight
                z_x2 = (temp_z_x2 - temp_weight_x2 * temp_z * inv_weight) * inv_weight
                z_x3 = (temp_z_x3 - temp_weight_x3 * temp_z * inv_weight) * inv_weight

                jacobians[i_p_1, i_p_2, i_p_3, :, :] = np.array([[x_x1, x_x2, x_x3],
                                                                 [y_x1, y_x2, y_x3],
                                                                 [z_x1, z_x2, z_x3]])


def eval_jacobians_irregular_2d_weights(np1: int, np2: int, f_p1: int, f_p2: int,
                                        cell_index_1: 'int[:]', cell_index_2: 'int[:]',
                                        global_basis_1: 'float[:,:,:]', global_basis_2: 'float[:,:,:]',
                                        global_spans_1: 'int[:]', global_spans_2: 'int[:]',
                                        global_arr_coeff_x: 'T[:,:]', global_arr_coeff_y: 'T[:,:]',
                                        global_arr_coeff_weights: 'float[:,:]',
                                        jacobians: 'T[:,:,:,:]'):
    """
    Parameters
    ----------
    np1 : int
        Number of points in the X1 direction
    np2 : int
        Number of points in the X2 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction

    cell_index_1 : ndarray of ints
        Index of the cells in the X1 direction
    cell_index_2 : ndarray of ints
        Index of the cells in the X2 direction

    global_basis_1 : ndarray of floats
        Basis functions values at each point in the X1 direction
    global_basis_2 : ndarray of floats
        Basis functions values at each point in the X2 direction

    global_spans_1 : ndarray of ints
        Spans in the X1 direction
    global_spans_2 : ndarray of ints
        Spans in the X2 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field

    global_arr_coeff_weights: ndarray of floats
        Coefficients of the weight field

    jacobians: ndarray of floats
        Jacobian matrix on the grid
    """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2))

    arr_coeffs_weights = np.zeros((1 + f_p1, 1 + f_p2))


    temp_x = arr_coeffs_x[0,0]-arr_coeffs_x[0,0]
    temp_y = arr_coeffs_y[0,0]-arr_coeffs_y[0,0]

    temp_x_x1 = arr_coeffs_x[0,0]-arr_coeffs_x[0,0]
    temp_x_x2 = arr_coeffs_y[0,0]-arr_coeffs_y[0,0]

    temp_y_x1 = arr_coeffs_x[0,0]-arr_coeffs_x[0,0]
    temp_y_x2 = arr_coeffs_y[0,0]-arr_coeffs_y[0,0]

    for i_p_1 in range(np1):
        i_cell_1 = cell_index_1[i_p_1]
        span_1 = global_spans_1[i_cell_1]

        for i_p_2 in range(np2):
            i_cell_2 = cell_index_2[i_p_2]
            span_2 = global_spans_2[i_cell_2]

            arr_coeffs_x[:, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                    span_2 - f_p2:1 + span_2]
            arr_coeffs_y[:, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                    span_2 - f_p2:1 + span_2]

            arr_coeffs_weights[:, :] = global_arr_coeff_weights[span_1 - f_p1:1 + span_1,
                                                                span_2 - f_p2:1 + span_2]
            temp_x *= 0.0
            temp_y *= 0.0

            temp_weight = 0.0

            temp_x_x1 *= 0.0
            temp_x_x2 *= 0.0

            temp_y_x1 *= 0.0
            temp_y_x2 *= 0.0

            temp_weight_x1 = 0.0
            temp_weight_x2 = 0.0

            for i_basis_1 in range(1 + f_p1):
                spline_1 = global_basis_1[i_p_1, i_basis_1, 0]
                spline_x1 = global_basis_1[i_p_1, i_basis_1, 1]

                for i_basis_2 in range(1 + f_p2):
                    spline_2 = global_basis_2[i_p_2, i_basis_2, 0]
                    spline_x2 = global_basis_2[i_p_2, i_basis_2, 1]

                    mapping = spline_1 * spline_2

                    mapping_x1 = spline_x1 * spline_2
                    mapping_x2 = spline_1 * spline_x2

                    coeff_x = arr_coeffs_x[i_basis_1, i_basis_2]
                    coeff_y = arr_coeffs_y[i_basis_1, i_basis_2]

                    coeff_weight = arr_coeffs_weights[i_basis_1, i_basis_2]

                    temp_x += mapping * coeff_x * coeff_weight
                    temp_y += mapping * coeff_y * coeff_weight

                    temp_weight += mapping * coeff_weight

                    temp_x_x1 += mapping_x1 * coeff_x * coeff_weight
                    temp_x_x2 += mapping_x2 * coeff_x * coeff_weight

                    temp_y_x1 += mapping_x1 * coeff_y * coeff_weight
                    temp_y_x2 += mapping_x2 * coeff_y * coeff_weight

                    temp_weight_x1 += mapping_x1 * coeff_weight
                    temp_weight_x2 += mapping_x2 * coeff_weight

            inv_weight = 1.0 / temp_weight

            x_x1 = (temp_x_x1 - temp_weight_x1 * temp_x * inv_weight) * inv_weight
            x_x2 = (temp_x_x2 - temp_weight_x2 * temp_x * inv_weight) * inv_weight

            y_x1 = (temp_y_x1 - temp_weight_x1 * temp_y * inv_weight) * inv_weight
            y_x2 = (temp_y_x2 - temp_weight_x2 * temp_y * inv_weight) * inv_weight

            jacobians[i_p_1, i_p_2, :, :] = np.array([[x_x1, x_x2],
                                                      [y_x1, y_x2]])


# =============================================================================
# Evaluation of the inverse of the Jacobian matrix
# =============================================================================
# -----------------------------------------------------------------------------
# 1: Regular tensor grid without weights
# -----------------------------------------------------------------------------
def eval_jacobians_inv_3d(nc1: int, nc2: int, nc3: int,  f_p1: int, f_p2: int,
                          f_p3: int, k1: int, k2: int, k3: int, global_basis_1: 'float[:,:,:,:]',
                          global_basis_2: 'float[:,:,:,:]', global_basis_3: 'float[:,:,:,:]', global_spans_1: 'int[:]',
                          global_spans_2: 'int[:]', global_spans_3: 'int[:]', global_arr_coeff_x: 'T[:,:,:]',
                          global_arr_coeff_y: 'T[:,:,:]', global_arr_coeff_z: 'T[:,:,:]',
                          jacobians_inv: 'T[:,:,:,:,:]'):

    """
    Parameters
    ----------
    nc1: int
        Number of cells in the X1 direction
    nc2: int
        Number of cells in the X2 direction
    nc3: int
        Number of cells in the X3 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction
    f_p3: int
        Degree in the X3 direction

    k1: int
        Number of evaluation points in the X1 direction
    k2: int
        Number of evaluation points in the X2 direction
    k3: int
        Number of evaluation points in the X3 direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X1 direction
    global_basis_2: ndarray of floats
        Basis functions values at each cell and quadrature points in the X2 direction
    global_basis_3: ndarray of floats
        Basis functions values at each cell and quadrature points in the X3 direction

    global_spans_1: ndarray of ints
        Spans in the X1 direction
    global_spans_2: ndarray of ints
        Spans in the X2 direction
    global_spans_3: ndarray of ints
        Spans in the X3 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field
    global_arr_coeff_z: ndarray of floats
        Coefficients of the X3 field

    jacobians_inv: ndarray of floats
        Inverse of the Jacobian matrix on the grid
    """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_z = np.zeros_like(global_arr_coeff_z, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))

    arr_x_x1 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2, k3))
    arr_x_x2 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2, k3))
    arr_x_x3 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2, k3))

    arr_y_x1 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2, k3))
    arr_y_x2 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2, k3))
    arr_y_x3 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2, k3))

    arr_z_x1 = np.zeros_like(global_arr_coeff_z, shape=(k1, k2, k3))
    arr_z_x2 = np.zeros_like(global_arr_coeff_z, shape=(k1, k2, k3))
    arr_z_x3 = np.zeros_like(global_arr_coeff_z, shape=(k1, k2, k3))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]

            for i_cell_3 in range(nc3):
                span_3 = global_spans_3[i_cell_3]

                arr_x_x1[:, :, :] = 0.0
                arr_x_x2[:, :, :] = 0.0
                arr_x_x3[:, :, :] = 0.0

                arr_y_x1[:, :, :] = 0.0
                arr_y_x2[:, :, :] = 0.0
                arr_y_x3[:, :, :] = 0.0

                arr_z_x1[:, :, :] = 0.0
                arr_z_x2[:, :, :] = 0.0
                arr_z_x3[:, :, :] = 0.0

                arr_coeffs_x[:, :, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_y[:, :, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_z[:, :, :] = global_arr_coeff_z[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]
                for i_quad_1 in range(k1):
                    for i_quad_2 in range(k2):
                        for i_quad_3 in range(k3):
                            for i_basis_1 in range(1 + f_p1):
                                spline_1 = global_basis_1[i_cell_1, i_basis_1, 0, i_quad_1]
                                spline_x1 = global_basis_1[i_cell_1, i_basis_1, 1, i_quad_1]

                                for i_basis_2 in range(1 + f_p2):
                                    spline_2 = global_basis_2[i_cell_2, i_basis_2, 0, i_quad_2]
                                    spline_x2 = global_basis_2[i_cell_2, i_basis_2, 1, i_quad_2]

                                    for i_basis_3 in range(1 + f_p3):
                                        spline_3 = global_basis_3[i_cell_3, i_basis_3, 0, i_quad_3]
                                        spline_x3 = global_basis_3[i_cell_3, i_basis_3, 1, i_quad_3]

                                        mapping_x1 = spline_x1 * spline_2 * spline_3
                                        mapping_x2 = spline_1 * spline_x2 * spline_3
                                        mapping_x3 = spline_1 * spline_2 * spline_x3

                                        coeff_x = arr_coeffs_x[i_basis_1, i_basis_2, i_basis_3]
                                        coeff_y = arr_coeffs_y[i_basis_1, i_basis_2, i_basis_3]
                                        coeff_z = arr_coeffs_z[i_basis_1, i_basis_2, i_basis_3]

                                        arr_x_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_x
                                        arr_x_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_x
                                        arr_x_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_x

                                        arr_y_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_y
                                        arr_y_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_y
                                        arr_y_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_y

                                        arr_z_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_z
                                        arr_z_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_z
                                        arr_z_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_z

                            x_x1 = arr_x_x1[i_quad_1, i_quad_2, i_quad_3]
                            x_x2 = arr_x_x2[i_quad_1, i_quad_2, i_quad_3]
                            x_x3 = arr_x_x3[i_quad_1, i_quad_2, i_quad_3]

                            y_x1 = arr_y_x1[i_quad_1, i_quad_2, i_quad_3]
                            y_x2 = arr_y_x2[i_quad_1, i_quad_2, i_quad_3]
                            y_x3 = arr_y_x3[i_quad_1, i_quad_2, i_quad_3]

                            z_x1 = arr_z_x1[i_quad_1, i_quad_2, i_quad_3]
                            z_x2 = arr_z_x2[i_quad_1, i_quad_2, i_quad_3]
                            z_x3 = arr_z_x3[i_quad_1, i_quad_2, i_quad_3]

                            det = x_x1 * y_x2 * z_x3 + x_x2 * y_x3 * z_x1 + x_x3 * y_x1 * z_x2 \
                                  - x_x1 * y_x3 * z_x2 - x_x2 * y_x1 * z_x3 - x_x3 * y_x2 * z_x1

                            a_11 = y_x2 * z_x3 - y_x3 * z_x2
                            a_12 = - y_x1 * z_x3 + y_x3 * z_x1
                            a_13 = y_x1 * z_x2 - y_x2 * z_x1

                            a_21 = - x_x2 * z_x3 + x_x3 * z_x2
                            a_22 = x_x1 * z_x3 - x_x3 * z_x1
                            a_23 = - x_x1 * z_x2 + x_x2 * z_x1

                            a_31 = x_x2 * y_x3 - x_x3 * y_x2
                            a_32 = - x_x1 * y_x3 + x_x3 * y_x1
                            a_33 = x_x1 * y_x2 - x_x2 * y_x1

                            jacobians_inv[i_cell_1 * k1 + i_quad_1,
                                          i_cell_2 * k2 + i_quad_2,
                                          i_cell_3 * k3 + i_quad_3,
                                          :, :] = np.array([[a_11, a_21, a_31],
                                                            [a_12, a_22, a_32],
                                                            [a_13, a_23, a_33]]) / det


def eval_jacobians_inv_2d(nc1: int, nc2: int,  f_p1: int, f_p2: int, k1: int, k2: int,
                          global_basis_1: 'float[:,:,:,:]', global_basis_2: 'float[:,:,:,:]', global_spans_1: 'int[:]',
                          global_spans_2: 'int[:]', global_arr_coeff_x: 'T[:,:]', global_arr_coeff_y: 'T[:,:]',
                          jacobians_inv: 'T[:,:,:,:]'):
    """
    Parameters
    ----------
    nc1: int
        Number of cells in the X1 direction
    nc2: int
        Number of cells in the X2 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction

    k1: int
        Number of evaluation points in the X1 direction
    k2: int
        Number of evaluation points in the X2 direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X1 direction
    global_basis_2: ndarray of floats
        Basis functions values at each cell and quadrature points in the X2 direction

    global_spans_1: ndarray of ints
        Spans in the X1 direction
    global_spans_2: ndarray of ints
        Spans in the X2 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field

    jacobians_inv: ndarray of floats
        Inverse of the Jacobian matrix at every point of the grid
    """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2))

    arr_x_x1 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2))
    arr_x_x2 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2))

    arr_y_x1 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2))
    arr_y_x2 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]

            arr_x_x1[:, :] = 0.0
            arr_x_x2[:, :] = 0.0

            arr_y_x1[:, :] = 0.0
            arr_y_x2[:, :] = 0.0

            arr_coeffs_x[:, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                    span_2 - f_p2:1 + span_2]

            arr_coeffs_y[:, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                    span_2 - f_p2:1 + span_2]

            for i_quad_1 in range(k1):
                for i_quad_2 in range(k2):
                    for i_basis_1 in range(1 + f_p1):
                        spline_1 = global_basis_1[i_cell_1, i_basis_1, 0, i_quad_1]
                        spline_x1 = global_basis_1[i_cell_1, i_basis_1, 1, i_quad_1]

                        for i_basis_2 in range(1 + f_p2):
                            spline_2 = global_basis_2[i_cell_2, i_basis_2, 0, i_quad_2]
                            spline_x2 = global_basis_2[i_cell_2, i_basis_2, 1, i_quad_2]

                            mapping_x1 = spline_x1 * spline_2
                            mapping_x2 = spline_1 * spline_x2

                            coeff_x = arr_coeffs_x[i_basis_1, i_basis_2]
                            coeff_y = arr_coeffs_y[i_basis_1, i_basis_2]

                            arr_x_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_x
                            arr_x_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_x

                            arr_y_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_y
                            arr_y_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_y

                    x_x1 = arr_x_x1[i_quad_1, i_quad_2]
                    x_x2 = arr_x_x2[i_quad_1, i_quad_2]

                    y_x1 = arr_y_x1[i_quad_1, i_quad_2]
                    y_x2 = arr_y_x2[i_quad_1, i_quad_2]

                    det = x_x1 * y_x2 - x_x2 * y_x1

                    jacobians_inv[i_cell_1 * k1 + i_quad_1,
                                  i_cell_2 * k2 + i_quad_2,
                                  :, :] = np.array([[y_x2, - x_x2],
                                                    [- y_x1, x_x1]]) / det


# -----------------------------------------------------------------------------
# 2: Irregular tensor grid without weights
# -----------------------------------------------------------------------------
def eval_jacobians_inv_irregular_3d(np1: int, np2: int, np3: int, f_p1: int, f_p2: int,
                                   f_p3: int, cell_index_1: 'int[:]', cell_index_2: 'int[:]', cell_index_3 : 'int[:]',
                                   global_basis_1: 'float[:,:,:]', global_basis_2: 'float[:,:,:]',
                                   global_basis_3: 'float[:,:,:]', global_spans_1: 'int[:]', global_spans_2: 'int[:]',
                                   global_spans_3: 'int[:]', global_arr_coeff_x: 'T[:,:,:]',
                                   global_arr_coeff_y: 'T[:,:,:]', global_arr_coeff_z: 'T[:,:,:]',
                                   jacobians_inv: 'T[:,:,:,:,:]'):
    """
    Parameters
    ----------
    np1 : int
        Number of points in the X1 direction
    np2 : int
        Number of points in the X2 direction
    np3 : int
        Number of points in the X3 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction
    f_p3: int
        Degree in the X3 direction

    cell_index_1 : ndarray of ints
        Index of the cells in the X1 direction
    cell_index_2 : ndarray of ints
        Index of the cells in the X2 direction
    cell_index_3 : ndarray of ints
        Index of the cells in the X3 direction

    global_basis_1 : ndarray of floats
        Basis functions values at each point in the X1 direction
    global_basis_2 : ndarray of floats
        Basis functions values at each point in the X2 direction
    global_basis_3 : ndarray of floats
        Basis functions values at each point in the X3 direction

    global_spans_1 : ndarray of ints
        Spans in the X1 direction
    global_spans_2 : ndarray of ints
        Spans in the X2 direction
    global_spans_3 : ndarray of ints
        Spans in the X3 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field
    global_arr_coeff_z: ndarray of floats
        Coefficients of the X3 field

    jacobians_inv: ndarray of floats
        Inverse of the Jacobian matrix at every point of the grid
    """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_z = np.zeros_like(global_arr_coeff_z, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))

    temp_x_x1 = arr_coeffs_x[0,0,0]-arr_coeffs_x[0,0,0]
    temp_x_x2 = arr_coeffs_y[0,0,0]-arr_coeffs_y[0,0,0]
    temp_x_x3 = arr_coeffs_z[0,0,0]-arr_coeffs_z[0,0,0]

    temp_y_x1 = arr_coeffs_x[0,0,0]-arr_coeffs_x[0,0,0]
    temp_y_x2 = arr_coeffs_y[0,0,0]-arr_coeffs_y[0,0,0]
    temp_y_x3 = arr_coeffs_z[0,0,0]-arr_coeffs_z[0,0,0]

    temp_z_x1 = arr_coeffs_x[0,0,0]-arr_coeffs_x[0,0,0]
    temp_z_x2 = arr_coeffs_y[0,0,0]-arr_coeffs_y[0,0,0]
    temp_z_x3 = arr_coeffs_z[0,0,0]-arr_coeffs_z[0,0,0]

    for i_p_1 in range(np1):
        i_cell_1 = cell_index_1[i_p_1]
        span_1 = global_spans_1[i_cell_1]

        for i_p_2 in range(np2):
            i_cell_2 = cell_index_2[i_p_2]
            span_2 = global_spans_2[i_cell_2]

            for i_p_3 in range(np3):
                i_cell_3 = cell_index_3[i_p_3]
                span_3 = global_spans_3[i_cell_3]

                arr_coeffs_x[:, :, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_y[:, :, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_z[:, :, :] = global_arr_coeff_z[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                temp_x_x1 *= 0.0
                temp_x_x2 *= 0.0
                temp_x_x3 *= 0.0

                temp_y_x1 *= 0.0
                temp_y_x2 *= 0.0
                temp_y_x3 *= 0.0

                temp_z_x1 *= 0.0
                temp_z_x2 *= 0.0
                temp_z_x3 *= 0.0

                for i_basis_1 in range(1 + f_p1):
                    spline_1 = global_basis_1[i_p_1, i_basis_1, 0]
                    spline_x1 = global_basis_1[i_p_1, i_basis_1, 1]

                    for i_basis_2 in range(1 + f_p2):
                        spline_2 = global_basis_2[i_p_2, i_basis_2, 0]
                        spline_x2 = global_basis_2[i_p_2, i_basis_2, 1]

                        for i_basis_3 in range(1 + f_p3):
                            spline_3 = global_basis_3[i_p_3, i_basis_3, 0]
                            spline_x3 = global_basis_3[i_p_3, i_basis_3, 1]

                            mapping_x1 = spline_x1 * spline_2 * spline_3
                            mapping_x2 = spline_1 * spline_x2 * spline_3
                            mapping_x3 = spline_1 * spline_2 * spline_x3

                            coeff_x = arr_coeffs_x[i_basis_1, i_basis_2, i_basis_3]
                            coeff_y = arr_coeffs_y[i_basis_1, i_basis_2, i_basis_3]
                            coeff_z = arr_coeffs_z[i_basis_1, i_basis_2, i_basis_3]

                            temp_x_x1 += mapping_x1 * coeff_x
                            temp_x_x2 += mapping_x2 * coeff_x
                            temp_x_x3 += mapping_x3 * coeff_x

                            temp_y_x1 += mapping_x1 * coeff_y
                            temp_y_x2 += mapping_x2 * coeff_y
                            temp_y_x3 += mapping_x3 * coeff_y

                            temp_z_x1 += mapping_x1 * coeff_z
                            temp_z_x2 += mapping_x2 * coeff_z
                            temp_z_x3 += mapping_x3 * coeff_z

                det = (+ temp_x_x1 * temp_y_x2 * temp_z_x3
                       + temp_x_x2 * temp_y_x3 * temp_z_x1
                       + temp_x_x3 * temp_y_x1 * temp_z_x2
                       - temp_x_x1 * temp_y_x3 * temp_z_x2
                       - temp_x_x2 * temp_y_x1 * temp_z_x3
                       - temp_x_x3 * temp_y_x2 * temp_z_x1)

                a_11 = temp_y_x2 * temp_z_x3 - temp_y_x3 * temp_z_x2
                a_12 = - temp_y_x1 * temp_z_x3 + temp_y_x3 * temp_z_x1
                a_13 = temp_y_x1 * temp_z_x2 - temp_y_x2 * temp_z_x1

                a_21 = - temp_x_x2 * temp_z_x3 + temp_x_x3 * temp_z_x2
                a_22 = temp_x_x1 * temp_z_x3 - temp_x_x3 * temp_z_x1
                a_23 = - temp_x_x1 * temp_z_x2 + temp_x_x2 * temp_z_x1

                a_31 = temp_x_x2 * temp_y_x3 - temp_x_x3 * temp_y_x2
                a_32 = - temp_x_x1 * temp_y_x3 + temp_x_x3 * temp_y_x1
                a_33 = temp_x_x1 * temp_y_x2 - temp_x_x2 * temp_y_x1

                jacobians_inv[i_p_1,
                                i_p_2,
                                i_p_3,
                                :, :] = np.array([[a_11, a_21, a_31],
                                                  [a_12, a_22, a_32],
                                                  [a_13, a_23, a_33]]) / det


def eval_jacobians_inv_irregular_2d(np1: int, np2: int, f_p1: int, f_p2: int, cell_index_1: 'int[:]',
                                    cell_index_2: 'int[:]', global_basis_1: 'float[:,:,:]', global_basis_2: 'float[:,:,:]',
                                    global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_arr_coeff_x: 'T[:,:]',
                                    global_arr_coeff_y: 'T[:,:]', jacobians_inv: 'T[:,:,:,:]'):
    """
    Parameters
    ----------
    np1 : int
        Number of points in the X1 direction
    np2 : int
        Number of points in the X2 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction

    cell_index_1 : ndarray of ints
        Index of the cells in the X1 direction
    cell_index_2 : ndarray of ints
        Index of the cells in the X2 direction

    global_basis_1 : ndarray of floats
        Basis functions values at each point in the X1 direction
    global_basis_2 : ndarray of floats
        Basis functions values at each point in the X2 direction

    global_spans_1 : ndarray of ints
        Spans in the X1 direction
    global_spans_2 : ndarray of ints
        Spans in the X2 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field

    jacobians_inv: ndarray of floats
        Inverse of the Jacobian matrix at every point of the grid
    """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2))

    temp_x_x1 = arr_coeffs_x[0,0]-arr_coeffs_x[0,0]
    temp_x_x2 = arr_coeffs_y[0,0]-arr_coeffs_y[0,0]

    temp_y_x1 = arr_coeffs_x[0,0]-arr_coeffs_x[0,0]
    temp_y_x2 = arr_coeffs_y[0,0]-arr_coeffs_y[0,0]


    for i_p_1 in range(np1):
        i_cell_1 = cell_index_1[i_p_1]
        span_1 = global_spans_1[i_cell_1]

        for i_p_2 in range(np2):
            i_cell_2 = cell_index_2[i_p_2]
            span_2 = global_spans_2[i_cell_2]

            arr_coeffs_x[:, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                       span_2 - f_p2:1 + span_2]

            arr_coeffs_y[:, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                       span_2 - f_p2:1 + span_2]

            temp_x_x1 *= 0.0
            temp_x_x2 *= 0.0

            temp_y_x1 *= 0.0
            temp_y_x2 *= 0.0

            for i_basis_1 in range(1 + f_p1):
                spline_1 = global_basis_1[i_p_1, i_basis_1, 0]
                spline_x1 = global_basis_1[i_p_1, i_basis_1, 1]

                for i_basis_2 in range(1 + f_p2):
                    spline_2 = global_basis_2[i_p_2, i_basis_2, 0]
                    spline_x2 = global_basis_2[i_p_2, i_basis_2, 1]

                    mapping_x1 = spline_x1 * spline_2
                    mapping_x2 = spline_1 * spline_x2

                    coeff_x = arr_coeffs_x[i_basis_1, i_basis_2]
                    coeff_y = arr_coeffs_y[i_basis_1, i_basis_2]

                    temp_x_x1 += mapping_x1 * coeff_x
                    temp_x_x2 += mapping_x2 * coeff_x

                    temp_y_x1 += mapping_x1 * coeff_y
                    temp_y_x2 += mapping_x2 * coeff_y

            det = temp_x_x1 * temp_y_x2 - temp_y_x1 * temp_x_x2

            jacobians_inv[i_p_1, i_p_2, :, :] = np.array([[temp_y_x2, - temp_x_x2],
                                                          [- temp_y_x1, temp_x_x1]]) / det

# -----------------------------------------------------------------------------
# 3: Regular tensor grid with weights
# -----------------------------------------------------------------------------
def eval_jacobians_inv_3d_weights(nc1: int, nc2: int, nc3: int,  f_p1: int, f_p2: int,
                                  f_p3: int, k1: int, k2: int, k3: int, global_basis_1: 'float[:,:,:,:]',
                                  global_basis_2: 'float[:,:,:,:]', global_basis_3: 'float[:,:,:,:]',
                                  global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_spans_3: 'int[:]',
                                  global_arr_coeff_x: 'T[:,:,:]', global_arr_coeff_y: 'T[:,:,:]',
                                  global_arr_coeff_z: 'T[:,:,:]', global_arr_coeff_weigths: 'float[:,:,:]',
                                  jacobians_inv: 'T[:,:,:,:,:]'):

    """
    Parameters
    ----------
    nc1: int
        Number of cells in the X1 direction
    nc2: int
        Number of cells in the X2 direction
    nc3: int
        Number of cells in the X3 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction
    f_p3: int
        Degree in the X3 direction

    k1: int
        Number of evaluation points in the X1 direction
    k2: int
        Number of evaluation points in the X2 direction
    k3: int
        Number of evaluation points in the X3 direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X1 direction
    global_basis_2: ndarray of floats
        Basis functions values at each cell and quadrature points in the X2 direction
    global_basis_3: ndarray of floats
        Basis functions values at each cell and quadrature points in the X3 direction

    global_spans_1: ndarray of ints
        Spans in the X1 direction
    global_spans_2: ndarray of ints
        Spans in the X2 direction
    global_spans_3: ndarray of ints
        Spans in the X3 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field
    global_arr_coeff_z: ndarray of floats
        Coefficients of the X3 field

    global_arr_coeff_weigths: ndarray of floats
        Coefficients of the weight field

    jacobians_inv: ndarray of floats
        Inverse of the Jacobian matrix on the grid
    """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_z = np.zeros_like(global_arr_coeff_z, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))

    arr_x = np.zeros_like(global_arr_coeff_x, shape=(k1, k2, k3))
    arr_y = np.zeros_like(global_arr_coeff_y, shape=(k1, k2, k3))
    arr_z = np.zeros_like(global_arr_coeff_z, shape=(k1, k2, k3))

    arr_x_x1 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2, k3))
    arr_x_x2 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2, k3))
    arr_x_x3 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2, k3))

    arr_y_x1 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2, k3))
    arr_y_x2 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2, k3))
    arr_y_x3 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2, k3))

    arr_z_x1 = np.zeros_like(global_arr_coeff_z, shape=(k1, k2, k3))
    arr_z_x2 = np.zeros_like(global_arr_coeff_z, shape=(k1, k2, k3))
    arr_z_x3 = np.zeros_like(global_arr_coeff_z, shape=(k1, k2, k3))

    arr_coeff_weights = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_weights = np.zeros((k1, k2, k3))

    arr_weights_x1 = np.zeros((k1, k2, k3))
    arr_weights_x2 = np.zeros((k1, k2, k3))
    arr_weights_x3 = np.zeros((k1, k2, k3))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]

            for i_cell_3 in range(nc3):
                span_3 = global_spans_3[i_cell_3]

                arr_x[:, :, :] = 0.0
                arr_y[:, :, :] = 0.0
                arr_z[:, :, :] = 0.0

                arr_x_x1[:, :, :] = 0.0
                arr_x_x2[:, :, :] = 0.0
                arr_x_x3[:, :, :] = 0.0

                arr_y_x1[:, :, :] = 0.0
                arr_y_x1[:, :, :] = 0.0
                arr_y_x3[:, :, :] = 0.0

                arr_z_x1[:, :, :] = 0.0
                arr_z_x2[:, :, :] = 0.0
                arr_z_x3[:, :, :] = 0.0

                arr_weights[:, :, :] = 0.0

                arr_weights_x1[:, :, :] = 0.0
                arr_weights_x2[:, :, :] = 0.0
                arr_weights_x3[:, :, :] = 0.0

                arr_coeffs_x[:, :, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_y[:, :, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_z[:, :, :] = global_arr_coeff_z[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeff_weights[:, :, :] = global_arr_coeff_weigths[span_1 - f_p1:1 + span_1,
                                                                      span_2 - f_p2:1 + span_2,
                                                                      span_3 - f_p3:1 + span_3]

                for i_quad_1 in range(k1):
                    for i_quad_2 in range(k2):
                        for i_quad_3 in range(k3):
                            for i_basis_1 in range(1 + f_p1):
                                spline_1 = global_basis_1[i_cell_1, i_basis_1, 0, i_quad_1]
                                spline_x1 = global_basis_1[i_cell_1, i_basis_1, 1, i_quad_1]

                                for i_basis_2 in range(1 + f_p2):
                                    spline_2 = global_basis_2[i_cell_2, i_basis_2, 0, i_quad_2]
                                    spline_x2 = global_basis_2[i_cell_2, i_basis_2, 1, i_quad_2]

                                    for i_basis_3 in range(1 + f_p3):
                                        spline_3 = global_basis_3[i_cell_3, i_basis_3, 0, i_quad_3]
                                        spline_x3 = global_basis_3[i_cell_3, i_basis_3, 1, i_quad_3]

                                        mapping = spline_1 * spline_2 * spline_3
                                        mapping_x1 = spline_x1 * spline_2 * spline_3
                                        mapping_x2 = spline_1 * spline_x2 * spline_3
                                        mapping_x3 = spline_1 * spline_2 * spline_x3

                                        coeff_x = arr_coeffs_x[i_basis_1, i_basis_2, i_basis_3]
                                        coeff_y = arr_coeffs_y[i_basis_1, i_basis_2, i_basis_3]
                                        coeff_z = arr_coeffs_z[i_basis_1, i_basis_2, i_basis_3]

                                        coeff_weight = arr_coeff_weights[i_basis_1, i_basis_2, i_basis_3]

                                        arr_x[i_quad_1, i_quad_2, i_quad_3] += mapping * coeff_x * coeff_weight
                                        arr_y[i_quad_1, i_quad_2, i_quad_3] += mapping * coeff_y * coeff_weight
                                        arr_z[i_quad_1, i_quad_2, i_quad_3] += mapping * coeff_z * coeff_weight

                                        arr_x_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_x * coeff_weight
                                        arr_x_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_x * coeff_weight
                                        arr_x_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_x * coeff_weight

                                        arr_y_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_y * coeff_weight
                                        arr_y_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_y * coeff_weight
                                        arr_y_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_y * coeff_weight

                                        arr_z_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_z * coeff_weight
                                        arr_z_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_z * coeff_weight
                                        arr_z_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_z * coeff_weight

                                        arr_weights[i_quad_1, i_quad_2, i_quad_3] += mapping * coeff_weight

                                        arr_weights_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_weight
                                        arr_weights_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_weight
                                        arr_weights_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_weight

                            x = arr_x[i_quad_1, i_quad_2, i_quad_3]
                            y = arr_y[i_quad_1, i_quad_2, i_quad_3]
                            z = arr_z[i_quad_1, i_quad_2, i_quad_3]

                            x_x1 = arr_x_x1[i_quad_1, i_quad_2, i_quad_3]
                            x_x2 = arr_x_x2[i_quad_1, i_quad_2, i_quad_3]
                            x_x3 = arr_x_x3[i_quad_1, i_quad_2, i_quad_3]

                            y_x1 = arr_y_x1[i_quad_1, i_quad_2, i_quad_3]
                            y_x2 = arr_y_x2[i_quad_1, i_quad_2, i_quad_3]
                            y_x3 = arr_y_x3[i_quad_1, i_quad_2, i_quad_3]

                            z_x1 = arr_z_x1[i_quad_1, i_quad_2, i_quad_3]
                            z_x2 = arr_z_x2[i_quad_1, i_quad_2, i_quad_3]
                            z_x3 = arr_z_x3[i_quad_1, i_quad_2, i_quad_3]

                            weight = arr_weights[i_quad_1, i_quad_2, i_quad_3]

                            weight_x1 = arr_weights_x1[i_quad_1, i_quad_2, i_quad_3]
                            weight_x2 = arr_weights_x2[i_quad_1, i_quad_2, i_quad_3]
                            weight_x3 = arr_weights_x3[i_quad_1, i_quad_2, i_quad_3]

                            inv_weight = 1.0 / weight

                            x_x1 = (x_x1 - weight_x1 * x * inv_weight) * inv_weight
                            x_x2 = (x_x2 - weight_x2 * x * inv_weight) * inv_weight
                            x_x3 = (x_x3 - weight_x3 * x * inv_weight) * inv_weight

                            y_x1 = (y_x1 - weight_x1 * y * inv_weight) * inv_weight
                            y_x2 = (y_x2 - weight_x2 * y * inv_weight) * inv_weight
                            y_x3 = (y_x3 - weight_x3 * y * inv_weight) * inv_weight

                            z_x1 = (z_x1 - weight_x1 * z * inv_weight) * inv_weight
                            z_x2 = (z_x2 - weight_x2 * z * inv_weight) * inv_weight
                            z_x3 = (z_x3 - weight_x3 * z * inv_weight) * inv_weight

                            det = x_x1 * y_x2 * z_x3 + x_x2 * y_x3 * z_x1 + x_x3 * y_x1 * z_x2 \
                                  - x_x1 * y_x3 * z_x2 - x_x2 * y_x1 * z_x3 - x_x3 * y_x2 * z_x1

                            a_11 = y_x2 * z_x3 - y_x3 * z_x2
                            a_12 = - y_x1 * z_x3 + y_x3 * z_x1
                            a_13 = y_x1 * z_x2 - y_x2 * z_x1

                            a_21 = - x_x2 * z_x3 + x_x3 * z_x2
                            a_22 = x_x1 * z_x3 - x_x3 * z_x1
                            a_23 = - x_x1 * z_x2 + x_x2 * z_x1

                            a_31 = x_x2 * y_x3 - x_x3 * y_x2
                            a_32 = - x_x1 * y_x3 + x_x3 * y_x1
                            a_33 = x_x1 * y_x2 - x_x2 * y_x1

                            jacobians_inv[i_cell_1 * k1 + i_quad_1,
                                          i_cell_2 * k2 + i_quad_2,
                                          i_cell_3 * k3 + i_quad_3,
                                          :, :] = np.array([[a_11, a_21, a_31],
                                                            [a_12, a_22, a_32],
                                                            [a_13, a_23, a_33]]) / det


def eval_jacobians_inv_2d_weights(nc1: int, nc2: int,  f_p1: int, f_p2: int, k1: int, k2: int,
                                  global_basis_1: 'float[:,:,:,:]', global_basis_2: 'float[:,:,:,:]',
                                  global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_arr_coeff_x: 'T[:,:]',
                                  global_arr_coeff_y: 'T[:,:]', global_arr_coeff_weights: 'float[:,:]',
                                  jacobians_inv: 'T[:,:,:,:]'):
    """
    Parameters
    ----------
    nc1: int
        Number of cells in the X1 direction
    nc2: int
        Number of cells in the X2 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction

    k1: int
        Number of evaluation points in the X1 direction
    k2: int
        Number of evaluation points in the X2 direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X1 direction
    global_basis_2: ndarray of floats
        Basis functions values at each cell and quadrature points in the X2 direction

    global_spans_1: ndarray of ints
        Spans in the X1 direction
    global_spans_2: ndarray of ints
        Spans in the X2 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field

    global_arr_coeff_weights: ndarray of floats
        Coefficients of the weights field

    jacobians_inv: ndarray of floats
        Inverse of the Jacobian matrix at every point of the grid
       """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2))

    arr_x = np.zeros_like(global_arr_coeff_x, shape=(k1, k2))
    arr_y = np.zeros_like(global_arr_coeff_y, shape=(k1, k2))

    arr_x_x1 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2))
    arr_x_x2 = np.zeros_like(global_arr_coeff_x, shape=(k1, k2))

    arr_y_x1 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2))
    arr_y_x2 = np.zeros_like(global_arr_coeff_y, shape=(k1, k2))

    arr_coeff_weights = np.zeros((1 + f_p1, 1 + f_p2))
    arr_weights = np.zeros((k1, k2))

    arr_weights_x1 = np.zeros((k1, k2))
    arr_weights_x2 = np.zeros((k1, k2))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]

            arr_x[:, :] = 0.0
            arr_y[:, :] = 0.0

            arr_x_x1[:, :] = 0.0
            arr_x_x2[:, :] = 0.0

            arr_y_x1[:, :] = 0.0
            arr_y_x2[:, :] = 0.0

            arr_weights[:, :] = 0.0

            arr_weights_x1[:, :] = 0.0
            arr_weights_x2[:, :] = 0.0

            arr_coeffs_x[:, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                    span_2 - f_p2:1 + span_2]

            arr_coeffs_y[:, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                    span_2 - f_p2:1 + span_2]

            arr_coeff_weights[:, :] = global_arr_coeff_weights[span_1 - f_p1:1 + span_1,
                                                               span_2 - f_p2:1 + span_2]

            for i_quad_1 in range(k1):
                for i_quad_2 in range(k2):
                    for i_basis_1 in range(1 + f_p1):
                        spline_1 = global_basis_1[i_cell_1, i_basis_1, 0, i_quad_1]
                        spline_x1 = global_basis_1[i_cell_1, i_basis_1, 1, i_quad_1]

                        for i_basis_2 in range(1 + f_p2):
                            spline_2 = global_basis_2[i_cell_2, i_basis_2, 0, i_quad_2]
                            spline_x2 = global_basis_2[i_cell_2, i_basis_2, 1, i_quad_2]

                            mapping = spline_1 * spline_2
                            mapping_x1 = spline_x1 * spline_2
                            mapping_x2 = spline_1 * spline_x2

                            coeff_x = arr_coeffs_x[i_basis_1, i_basis_2]
                            coeff_y = arr_coeffs_y[i_basis_1, i_basis_2]

                            coeff_weight = arr_coeff_weights[i_basis_1, i_basis_2]

                            arr_x[i_quad_1, i_quad_2] += mapping * coeff_x * coeff_weight
                            arr_y[i_quad_1, i_quad_2] += mapping * coeff_y * coeff_weight

                            arr_x_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_x * coeff_weight
                            arr_x_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_x * coeff_weight

                            arr_y_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_y * coeff_weight
                            arr_y_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_y * coeff_weight

                            arr_weights[i_quad_1, i_quad_2] += mapping * coeff_weight

                            arr_weights_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_weight
                            arr_weights_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_weight

                    x = arr_x[i_quad_1, i_quad_2]
                    y = arr_y[i_quad_1, i_quad_2]

                    x_x1 = arr_x_x1[i_quad_1, i_quad_2]
                    x_x2 = arr_x_x2[i_quad_1, i_quad_2]

                    y_x1 = arr_y_x1[i_quad_1, i_quad_2]
                    y_x2 = arr_y_x2[i_quad_1, i_quad_2]

                    weight = arr_weights[i_quad_1, i_quad_2]

                    weight_x1 = arr_weights_x1[i_quad_1, i_quad_2]
                    weight_x2 = arr_weights_x2[i_quad_1, i_quad_2]

                    inv_weight = 1.0 / weight

                    x_x1 = (x_x1 - weight_x1 * x * inv_weight) * inv_weight
                    x_x2 = (x_x2 - weight_x2 * x * inv_weight) * inv_weight

                    y_x1 = (y_x1 - weight_x1 * y * inv_weight) * inv_weight
                    y_x2 = (y_x2 - weight_x2 * y * inv_weight) * inv_weight

                    det = x_x1 * y_x2 - x_x2 * y_x1

                    jacobians_inv[i_cell_1 * k1 + i_quad_1,
                                  i_cell_2 * k2 + i_quad_2,
                                  :, :] = np.array([[y_x2, - x_x2],
                                                    [- y_x1, x_x1]]) / det


# -----------------------------------------------------------------------------
# 4: Irregular tensor grid with weights
# -----------------------------------------------------------------------------
def eval_jacobians_inv_irregular_3d_weights(np1: int, np2: int, np3: int, f_p1: int, f_p2: int,
                                            f_p3: int, cell_index_1: 'int[:]', cell_index_2: 'int[:]', cell_index_3 : 'int[:]',
                                            global_basis_1: 'float[:,:,:]', global_basis_2: 'float[:,:,:]',
                                            global_basis_3: 'float[:,:,:]', global_spans_1: 'int[:]', global_spans_2: 'int[:]',
                                            global_spans_3: 'int[:]', global_arr_coeff_x: 'T[:,:,:]',
                                            global_arr_coeff_y: 'T[:,:,:]', global_arr_coeff_z: 'T[:,:,:]',
                                            global_arr_coeff_weights: 'float[:,:,:]', jacobians_inv: 'T[:,:,:,:,:]'):
    """
    Parameters
    ----------
    np1 : int
        Number of points in the X1 direction
    np2 : int
        Number of points in the X2 direction
    np3 : int
        Number of points in the X3 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction
    f_p3: int
        Degree in the X3 direction

    cell_index_1 : ndarray of ints
        Index of the cells in the X1 direction
    cell_index_2 : ndarray of ints
        Index of the cells in the X2 direction
    cell_index_3 : ndarray of ints
        Index of the cells in the X3 direction

    global_basis_1 : ndarray of floats
        Basis functions values at each point in the X1 direction
    global_basis_2 : ndarray of floats
        Basis functions values at each point in the X2 direction
    global_basis_3 : ndarray of floats
        Basis functions values at each point in the X3 direction

    global_spans_1 : ndarray of ints
        Spans in the X1 direction
    global_spans_2 : ndarray of ints
        Spans in the X2 direction
    global_spans_3 : ndarray of ints
        Spans in the X3 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field
    global_arr_coeff_z: ndarray of floats
        Coefficients of the X3 field

    global_arr_coeff_weights: ndarray of floats
        Coefficients of the weight field

    jacobians_inv: ndarray of floats
        Inverse of the Jacobian matrix at every point of the grid
    """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_z = np.zeros_like(global_arr_coeff_z, shape=(1 + f_p1, 1 + f_p2, 1 + f_p3))

    arr_coeffs_weights = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))

    temp_x = arr_coeffs_x[0,0,0]-arr_coeffs_x[0,0,0]
    temp_y = arr_coeffs_y[0,0,0]-arr_coeffs_y[0,0,0]
    temp_z = arr_coeffs_z[0,0,0]-arr_coeffs_z[0,0,0]

    temp_x_x1 = arr_coeffs_x[0,0,0]-arr_coeffs_x[0,0,0]
    temp_x_x2 = arr_coeffs_y[0,0,0]-arr_coeffs_y[0,0,0]
    temp_x_x3 = arr_coeffs_z[0,0,0]-arr_coeffs_z[0,0,0]

    temp_y_x1 = arr_coeffs_x[0,0,0]-arr_coeffs_x[0,0,0]
    temp_y_x2 = arr_coeffs_y[0,0,0]-arr_coeffs_y[0,0,0]
    temp_y_x3 = arr_coeffs_z[0,0,0]-arr_coeffs_z[0,0,0]

    temp_z_x1 = arr_coeffs_x[0,0,0]-arr_coeffs_x[0,0,0]
    temp_z_x2 = arr_coeffs_y[0,0,0]-arr_coeffs_y[0,0,0]
    temp_z_x3 = arr_coeffs_z[0,0,0]-arr_coeffs_z[0,0,0]

    for i_p_1 in range(np1):
        i_cell_1 = cell_index_1[i_p_1]
        span_1 = global_spans_1[i_cell_1]

        for i_p_2 in range(np2):
            i_cell_2 = cell_index_2[i_p_2]
            span_2 = global_spans_2[i_cell_2]

            for i_p_3 in range(np3):
                i_cell_3 = cell_index_3[i_p_3]
                span_3 = global_spans_3[i_cell_3]

                arr_coeffs_x[:, :, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_y[:, :, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_z[:, :, :] = global_arr_coeff_z[span_1 - f_p1:1 + span_1,
                                                           span_2 - f_p2:1 + span_2,
                                                           span_3 - f_p3:1 + span_3]

                arr_coeffs_weights[:, :, :] = global_arr_coeff_weights[span_1 - f_p1:1 + span_1,
                                                                       span_2 - f_p2:1 + span_2,
                                                                       span_3 - f_p3:1 + span_3]

                temp_x *= 0.0
                temp_y *= 0.0
                temp_z *= 0.0

                temp_weight= 0.0

                temp_x_x1 *= 0.0
                temp_x_x2 *= 0.0
                temp_x_x3 *= 0.0

                temp_y_x1 *= 0.0
                temp_y_x2 *= 0.0
                temp_y_x3 *= 0.0

                temp_z_x1 *= 0.0
                temp_z_x2 *= 0.0
                temp_z_x3 *= 0.0

                temp_weight_x1 = 0.0
                temp_weight_x2 = 0.0
                temp_weight_x3 = 0.0

                for i_basis_1 in range(1 + f_p1):
                    spline_1 = global_basis_1[i_p_1, i_basis_1, 0]
                    spline_x1 = global_basis_1[i_p_1, i_basis_1, 1]

                    for i_basis_2 in range(1 + f_p2):
                        spline_2 = global_basis_2[i_p_2, i_basis_2, 0]
                        spline_x2 = global_basis_2[i_p_2, i_basis_2, 1]

                        for i_basis_3 in range(1 + f_p3):
                            spline_3 = global_basis_3[i_p_3, i_basis_3, 0]
                            spline_x3 = global_basis_3[i_p_3, i_basis_3, 1]

                            mapping = spline_1 * spline_2 * spline_3

                            mapping_x1 = spline_x1 * spline_2 * spline_3
                            mapping_x2 = spline_1 * spline_x2 * spline_3
                            mapping_x3 = spline_1 * spline_2 * spline_x3

                            coeff_x = arr_coeffs_x[i_basis_1, i_basis_2, i_basis_3]
                            coeff_y = arr_coeffs_y[i_basis_1, i_basis_2, i_basis_3]
                            coeff_z = arr_coeffs_z[i_basis_1, i_basis_2, i_basis_3]

                            coeff_weight = arr_coeffs_weights[i_basis_1, i_basis_2, i_basis_3]

                            temp_x += mapping * coeff_x * coeff_weight
                            temp_y += mapping * coeff_y * coeff_weight
                            temp_z += mapping * coeff_z * coeff_weight

                            temp_weight += mapping * coeff_weight

                            temp_x_x1 += mapping_x1 * coeff_x * coeff_weight
                            temp_x_x2 += mapping_x2 * coeff_x * coeff_weight
                            temp_x_x3 += mapping_x3 * coeff_x * coeff_weight

                            temp_y_x1 += mapping_x1 * coeff_y * coeff_weight
                            temp_y_x2 += mapping_x2 * coeff_y * coeff_weight
                            temp_y_x3 += mapping_x3 * coeff_y * coeff_weight

                            temp_z_x1 += mapping_x1 * coeff_z * coeff_weight
                            temp_z_x2 += mapping_x2 * coeff_z * coeff_weight
                            temp_z_x3 += mapping_x3 * coeff_z * coeff_weight

                            temp_weight_x1 += mapping_x1 * coeff_weight
                            temp_weight_x2 += mapping_x2 * coeff_weight
                            temp_weight_x3 += mapping_x3 * coeff_weight

                inv_weight = 1.0 / temp_weight

                x_x1 = (temp_x_x1 - temp_weight_x1 * temp_x * inv_weight) * inv_weight
                x_x2 = (temp_x_x2 - temp_weight_x2 * temp_x * inv_weight) * inv_weight
                x_x3 = (temp_x_x3 - temp_weight_x3 * temp_x * inv_weight) * inv_weight

                y_x1 = (temp_y_x1 - temp_weight_x1 * temp_y * inv_weight) * inv_weight
                y_x2 = (temp_y_x2 - temp_weight_x2 * temp_y * inv_weight) * inv_weight
                y_x3 = (temp_y_x3 - temp_weight_x3 * temp_y * inv_weight) * inv_weight

                z_x1 = (temp_z_x1 - temp_weight_x1 * temp_z * inv_weight) * inv_weight
                z_x2 = (temp_z_x2 - temp_weight_x2 * temp_z * inv_weight) * inv_weight
                z_x3 = (temp_z_x3 - temp_weight_x3 * temp_z * inv_weight) * inv_weight

                det = x_x1 * y_x2 * z_x3 + x_x2 * y_x3 * z_x1 + x_x3 * y_x1 * z_x2 \
                        - x_x1 * y_x3 * z_x2 - x_x2 * y_x1 * z_x3 - x_x3 * y_x2 * z_x1

                a_11 = y_x2 * z_x3 - y_x3 * z_x2
                a_12 = - y_x1 * z_x3 + y_x3 * z_x1
                a_13 = y_x1 * z_x2 - y_x2 * z_x1

                a_21 = - x_x2 * z_x3 + x_x3 * z_x2
                a_22 = x_x1 * z_x3 - x_x3 * z_x1
                a_23 = - x_x1 * z_x2 + x_x2 * z_x1

                a_31 = x_x2 * y_x3 - x_x3 * y_x2
                a_32 = - x_x1 * y_x3 + x_x3 * y_x1
                a_33 = x_x1 * y_x2 - x_x2 * y_x1

                jacobians_inv[i_p_1,
                              i_p_2,
                              i_p_3,
                              :, :] = np.array([[a_11, a_21, a_31],
                                                [a_12, a_22, a_32],
                                                [a_13, a_23, a_33]]) / det


def eval_jacobians_inv_irregular_2d_weights(np1: int, np2: int, f_p1: int, f_p2: int,
                                            cell_index_1: 'int[:]', cell_index_2: 'int[:]',
                                            global_basis_1: 'float[:,:,:]', global_basis_2: 'float[:,:,:]',
                                            global_spans_1: 'int[:]', global_spans_2: 'int[:]',
                                            global_arr_coeff_x: 'T[:,:]', global_arr_coeff_y: 'T[:,:]',
                                            global_arr_coeff_weights: 'float[:,:]',
                                            jacobians_inv: 'T[:,:,:,:]'):
    """
    Parameters
    ----------
    np1 : int
        Number of points in the X1 direction
    np2 : int
        Number of points in the X2 direction

    f_p1: int
        Degree in the X1 direction
    f_p2: int
        Degree in the X2 direction

    cell_index_1 : ndarray of ints
        Index of the cells in the X1 direction
    cell_index_2 : ndarray of ints
        Index of the cells in the X2 direction

    global_basis_1 : ndarray of floats
        Basis functions values at each point in the X1 direction
    global_basis_2 : ndarray of floats
        Basis functions values at each point in the X2 direction

    global_spans_1 : ndarray of ints
        Spans in the X1 direction
    global_spans_2 : ndarray of ints
        Spans in the X2 direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X1 field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the X2 field

    global_arr_coeff_weights: ndarray of floats
        Coefficients of the weight field

    jacobians_inv: ndarray of floats
        Inverse of the Jacobian matrix at every point of the grid
    """

    arr_coeffs_x = np.zeros_like(global_arr_coeff_x, shape=(1 + f_p1, 1 + f_p2))
    arr_coeffs_y = np.zeros_like(global_arr_coeff_y, shape=(1 + f_p1, 1 + f_p2))

    arr_coeffs_weights = np.zeros((1 + f_p1, 1 + f_p2))

    temp_x = arr_coeffs_x[0,0]-arr_coeffs_x[0,0]
    temp_y = arr_coeffs_y[0,0]-arr_coeffs_y[0,0]

    temp_x_x1 = arr_coeffs_x[0,0]-arr_coeffs_x[0,0]
    temp_x_x2 = arr_coeffs_y[0,0]-arr_coeffs_y[0,0]

    temp_y_x1 = arr_coeffs_x[0,0]-arr_coeffs_x[0,0]
    temp_y_x2 = arr_coeffs_y[0,0]-arr_coeffs_y[0,0]

    for i_p_1 in range(np1):
        i_cell_1 = cell_index_1[i_p_1]
        span_1 = global_spans_1[i_cell_1]

        for i_p_2 in range(np2):
            i_cell_2 = cell_index_2[i_p_2]
            span_2 = global_spans_2[i_cell_2]

            arr_coeffs_x[:, :] = global_arr_coeff_x[span_1 - f_p1:1 + span_1,
                                                    span_2 - f_p2:1 + span_2]
            arr_coeffs_y[:, :] = global_arr_coeff_y[span_1 - f_p1:1 + span_1,
                                                    span_2 - f_p2:1 + span_2]

            arr_coeffs_weights[:, :] = global_arr_coeff_weights[span_1 - f_p1:1 + span_1,
                                                                span_2 - f_p2:1 + span_2]
            temp_x *=0.0
            temp_y *=0.0

            temp_weight= 0.0

            temp_x_x1 *= 0.0
            temp_x_x2 *= 0.0

            temp_y_x1 *= 0.0
            temp_y_x2 *= 0.0

            temp_weight_x1 = 0.0
            temp_weight_x2 = 0.0

            for i_basis_1 in range(1 + f_p1):
                spline_1 = global_basis_1[i_p_1, i_basis_1, 0]
                spline_x1 = global_basis_1[i_p_1, i_basis_1, 1]

                for i_basis_2 in range(1 + f_p2):
                    spline_2 = global_basis_2[i_p_2, i_basis_2, 0]
                    spline_x2 = global_basis_2[i_p_2, i_basis_2, 1]

                    mapping = spline_1 * spline_2

                    mapping_x1 = spline_x1 * spline_2
                    mapping_x2 = spline_1 * spline_x2

                    coeff_x = arr_coeffs_x[i_basis_1, i_basis_2]
                    coeff_y = arr_coeffs_y[i_basis_1, i_basis_2]

                    coeff_weight = arr_coeffs_weights[i_basis_1, i_basis_2]

                    temp_x += mapping * coeff_x * coeff_weight
                    temp_y += mapping * coeff_y * coeff_weight

                    temp_weight += mapping * coeff_weight

                    temp_x_x1 += mapping_x1 * coeff_x * coeff_weight
                    temp_x_x2 += mapping_x2 * coeff_x * coeff_weight

                    temp_y_x1 += mapping_x1 * coeff_y * coeff_weight
                    temp_y_x2 += mapping_x2 * coeff_y * coeff_weight

                    temp_weight_x1 += mapping_x1 * coeff_weight
                    temp_weight_x2 += mapping_x2 * coeff_weight

            inv_weight = 1.0 / temp_weight

            x_x1 = (temp_x_x1 - temp_weight_x1 * temp_x * inv_weight) * inv_weight
            x_x2 = (temp_x_x2 - temp_weight_x2 * temp_x * inv_weight) * inv_weight

            y_x1 = (temp_y_x1 - temp_weight_x1 * temp_y * inv_weight) * inv_weight
            y_x2 = (temp_y_x2 - temp_weight_x2 * temp_y * inv_weight) * inv_weight

            det = x_x1 * y_x2 - x_x2 * y_x1

            jacobians_inv[i_p_1, i_p_2, :, :] = np.array([[y_x2, - x_x2],
                                                          [- y_x1, x_x1]]) / det


# ==========================================================================
# Push forwards
# ==========================================================================
# --------------------------------------------------------------------------
# 1: L2 Push-forward
# --------------------------------------------------------------------------
def pushforward_2d_l2(fields_to_push: 'T[:,:,:]', sqrt_met_dets: 'float[:,:]',
                       pushed_fields: 'T[:,:,:]'):
    """
    Parameters
    ----------
    fields_to_push: ndarray
        Field values to push forward on the mapping
        This array as shape (n_x1, n_x2, n_f) where:
        * n_x1 is the number of points in direction 1 of the implicit grid.
        * n_x2 is the number of points in direction 2 of the implicit grid.
        * n_f is the number of fields to push-forward in the L2 space.

    sqrt_met_dets: ndarray
        Square root of the metric determinant of the mapping

    pushed_fields: ndarray
        Push forwarded fields
    """
    for i_f in range(pushed_fields.shape[2]):
        pushed_fields[:, :, i_f] = fields_to_push[:, :, i_f] / sqrt_met_dets[:, :]


def pushforward_3d_l2(fields_to_push: 'T[:,:,:,:]', sqrt_met_dets: 'float[:,:,:]', pushed_fields: 'T[:,:,:,:]'):
    """
    Parameters
    ----------
    fields_to_push: ndarray
        Field values to push forward on the mapping
        This array as shape (n_x1, n_x2, n_x3, n_f) where:
        * n_x1 is the number of points in direction 1 of the implicit grid.
        * n_x2 is the number of points in direction 2 of the implicit grid.
        * n_x3 is the number of points in direction 3 of the implicit grid.
        * n_f is the number of fields to push-forward in the L2 space.

    sqrt_met_dets: ndarray
        Square root of the metric determinant of the mapping

    pushed_fields: ndarray
        Push forwarded fields
    """
    for i_f in range(pushed_fields.shape[3]):
        pushed_fields[:, :, :, i_f] = fields_to_push[:, :, :, i_f] / sqrt_met_dets[:, :, :]


# --------------------------------------------------------------------------
# 2: Hcurl Push-forward
# --------------------------------------------------------------------------
def pushforward_2d_hcurl(fields_to_push: 'T[:,:,:,:]', inv_jac_mats: 'float[:,:,:,:]',
                          pushed_fields: 'T[:,:,:,:]'):
    """
    Parameters
    ----------
    fields_to_push: ndarray
        Field values to push forward on the mapping
        This array as shape (2, n_x1, n_x2, n_f) where:
        * 2 is the logical dimension of the problem (2 here)
        * n_x1 is the number of points in direction 1 of the implicit grid.
        * n_x2 is the number of points in direction 2 of the implicit grid.
        * n_f is the number of fields to push-forward in the Hcurl space.

    inv_jac_mats: ndarray
        Inverses of the Jacobian matrix of the mapping

    pushed_fields: ndarray
        Push forwarded fields
    """
    for i_f in range(pushed_fields.shape[0]):
        pushed_fields[i_f, 0, :, :] = (+ inv_jac_mats[:, :, 0, 0] * fields_to_push[0, :, :, i_f]
                                       + inv_jac_mats[:, :, 1, 0] * fields_to_push[1, :, :, i_f])

        pushed_fields[i_f, 1, :, :] = (+ inv_jac_mats[:, :, 0, 1] * fields_to_push[0, :, :, i_f]
                                       + inv_jac_mats[:, :, 1, 1] * fields_to_push[1, :, :, i_f])


def pushforward_3d_hcurl(fields_to_push: 'T[:,:,:,:,:]', inv_jac_mats: 'float[:,:,:,:,:]',
                          pushed_fields: 'T[:,:,:,:,:]'):
    """
    Parameters
    ----------
    fields_to_push: ndarray
        Field values to push forward on the mapping
        This array as shape (3, n_x1, n_x2, n_x3, n_f) where:
        * 3 is the logical dimension of the problem
        * n_x1 is the number of points in direction 1 of the implicit grid.
        * n_x2 is the number of points in direction 2 of the implicit grid.
        * n_x3 is the number of points in direction 3 of the implicit grid
        * n_f is the number of fields to push-forward in the Hcurl space.

    inv_jac_mats: ndarray
        Inverses of the Jacobian matrix of the mapping

    pushed_fields: ndarray
        Push forwarded fields
    """
    for i_f in range(pushed_fields.shape[0]):
        x = fields_to_push[0, :, :, :, i_f]
        y = fields_to_push[1, :, :, :, i_f]
        z = fields_to_push[2, :, :, :, i_f]

        pushed_fields[i_f, 0, :, :, :] = + inv_jac_mats[:, :, :, 0, 0] * x \
                                         + inv_jac_mats[:, :, :, 1, 0] * y \
                                         + inv_jac_mats[:, :, :, 2, 0] * z

        pushed_fields[i_f, 1, :, :, :] = + inv_jac_mats[:, :, :, 0, 1] * x \
                                         + inv_jac_mats[:, :, :, 1, 1] * y \
                                         + inv_jac_mats[:, :, :, 2, 1] * z

        pushed_fields[i_f, 2, :, :, :] = + inv_jac_mats[:, :, :, 0, 2] * x \
                                         + inv_jac_mats[:, :, :, 1, 2] * y \
                                         + inv_jac_mats[:, :, :, 2, 2] * z


# --------------------------------------------------------------------------
# 1: Hdiv Push-forward
# --------------------------------------------------------------------------
def pushforward_2d_hdiv(fields_to_push: 'T[:,:,:,:]', jac_mats: 'float[:,:,:,:]',
                        sqrt_met_dets: 'float[:,:]', pushed_fields: 'T[:,:,:,:]'):
    """
    Parameters
    ----------
    fields_to_push: ndarray
        Field values to push forward on the mapping
        This array as shape (2, n_x1, n_x2, n_f) where:
        * 2 is the logical dimension of the problem (2 here)
        * n_x1 is the number of points in direction 1 of the implicit grid.
        * n_x2 is the number of points in direction 2 of the implicit grid.
        * n_f is the number of fields to push-forward in the Hdiv space.

    jac_mats: ndarray
        Jacobian matrix of the mapping

    sqrt_met_dets: ndarray
        Square root of the metric determinant of the mapping

    pushed_fields: ndarray
        Push forwarded fields
    """
    for i_f in range(pushed_fields.shape[0]):

        pushed_fields[i_f, 0, :, :] = (+ jac_mats[:, :, 0, 0] * fields_to_push[0, :, :, i_f]
                                       + jac_mats[:, :, 0, 1] * fields_to_push[1, :, :, i_f]) / sqrt_met_dets[:, :]

        pushed_fields[i_f, 1, :, :] = (+ jac_mats[:, :, 1, 0] * fields_to_push[0, :, :, i_f]
                                       + jac_mats[:, :, 1, 1] * fields_to_push[1, :, :, i_f]) / sqrt_met_dets[:, :]


def pushforward_3d_hdiv(fields_to_push: 'T[:,:,:,:,:]', jac_mats: 'float[:,:,:,:,:]',
                        sqrt_met_dets: 'float[:,:,:]', pushed_fields: 'T[:,:,:,:,:]'):
    """
    Parameters
    ----------
    fields_to_push: ndarray
        Field values to push forward on the mapping
        This array as shape (3, n_x1, n_x2, n_x3, n_f) where:
        * 3 is the logical dimension of the problem
        * n_x1 is the number of points in direction 1 of the implicit grid.
        * n_x2 is the number of points in direction 2 of the implicit grid.
        * n_x3 is the number of points in direction 3 of the implicit grid
        * n_f is the number of fields to push-forward in the Hdiv space.

    jac_mats: ndarray
        Jacobian matrix of the mapping

    sqrt_met_dets: ndarray
        Square root of the metric determinant of the mapping

    pushed_fields: ndarray
        Push forwarded fields
    """
    for i_f in range(pushed_fields.shape[0]):

        x = fields_to_push[0, :, :, :, i_f]
        y = fields_to_push[1, :, :, :, i_f]
        z = fields_to_push[2, :, :, :, i_f]

        pushed_fields[i_f, 0, :, :, :] = (+ jac_mats[:, :, :, 0, 0] * x
                                          + jac_mats[:, :, :, 0, 1] * y
                                          + jac_mats[:, :, :, 0, 2] * z) / sqrt_met_dets[:, :, :]

        pushed_fields[i_f, 1, :, :, :] = (+ jac_mats[:, :, :, 1, 0] * x
                                          + jac_mats[:, :, :, 1, 1] * y
                                          + jac_mats[:, :, :, 1, 2] * z) / sqrt_met_dets[:, :, :]

        pushed_fields[i_f, 2 ,:, :, :] = (+ jac_mats[:, :, :, 2, 0] * x
                                          + jac_mats[:, :, :, 2, 1] * y
                                          + jac_mats[:, :, :, 2, 2] * z) / sqrt_met_dets[:, :, :]
