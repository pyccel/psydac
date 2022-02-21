import numpy as np


def eval_fields_3d_no_weights(nc1: int, nc2: int, nc3: int, pad1: int, pad2: int, pad3: int, f_p1: int, f_p2: int,
                              f_p3: int, k1: int, k2: int, k3: int, global_basis_1: 'float[:,:,:,:]',
                              global_basis_2: 'float[:,:,:,:]', global_basis_3: 'float[:,:,:,:]',
                              global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_spans_3: 'int[:]',
                              glob_arr_coeff: 'float[:,:,:,:]', out_fields: 'float[:,:,:,:]'):
    """
    Parameters
    ----------
    nc1: int
        Number of cells in the X direction
    nc2: int
        Number of cells in the Y direction
    nc3: int
        Number of cells in the Z direction

    pad1: int
        Padding in the X direction
    pad2: int
        Padding in the Y direction
    pad3: int
        Padding in the Z direction

    f_p1: int
        Degree in the X direction
    f_p2: int
        Degree in the Y direction
    f_p3: int
        Degree in the Z direction

    k1: int
        Quadrature order in the X direction
    k2: int
        Quadrature order in the Y direction
    k3: int
        Quadrature order in the Z direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X direction
    global_basis_2: ndarray of floats
        Basis functions values at each cell and quadrature points in the Y direction
    global_basis_3: ndarray of floats
        Basis functions values at each cell and quadrature points in the Z direction

    global_spans_1: ndarray of ints
        Spans in the X direction
    global_spans_2: ndarray of ints
        Spans in the Y direction
    global_spans_3: ndarray of ints
        Spans in the Z direction

    glob_arr_coeff: ndarray of floats
        Coefficients of the fields in the X,Y and Z directions

    out_fields: ndarray of floats
        Evaluated fields, filled with the correct values by the function
    """

    arr_coeff_fields = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3, out_fields.shape[3]))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]

            for i_cell_3 in range(nc3):
                span_3 = global_spans_3[i_cell_3]

                arr_coeff_fields[:, :, :, :] = glob_arr_coeff[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                              pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                              pad3 + span_3 - f_p3:1 + pad3 + span_3,
                                                              :]

                for i_basis_1 in range(1 + f_p1):
                    for i_basis_2 in range(1 + f_p2):
                        for i_basis_3 in range(1 + f_p3):
                            coeff_fields = arr_coeff_fields[i_basis_1, i_basis_2, i_basis_3, :]

                            for i_quad_1 in range(k1 + 1):
                                spline_1 = global_basis_1[i_cell_1, i_basis_1, 0, i_quad_1]

                                for i_quad_2 in range(k2 + 1):
                                    spline_2 = global_basis_2[i_cell_2, i_basis_2, 0, i_quad_2]

                                    for i_quad_3 in range(k3 + 1):
                                        spline_3 = global_basis_3[i_cell_3, i_basis_3, 0, i_quad_3]

                                        spline = spline_1 * spline_2 * spline_3

                                        out_fields[i_cell_1 * (k1 + 1) + i_quad_1,
                                                   i_cell_2 * (k2 + 1) + i_quad_2,
                                                   i_cell_3 * (k3 + 1) + i_quad_3,
                                                   :] += spline * coeff_fields


def eval_fields_2d_no_weights(nc1: int, nc2: int, pad1: int, pad2: int, f_p1: int, f_p2: int, k1: int, k2: int,
                              global_basis_1: 'float[:,:,:,:]', global_basis_2: 'float[:,:,:,:]',
                              global_spans_1: 'int[:]', global_spans_2: 'int[:]', glob_arr_coeff: 'float[:,:,:]',
                              out_fields: 'float[:,:,:]'):
    """
    Parameters
    ----------
    nc1: int
        Number of cells in the X direction
    nc2: int
        Number of cells in the Y direction

    pad1: int
        Padding in the X direction
    pad2: int
        Padding in the Y direction

    f_p1: int
        Degree in the X direction
    f_p2: int
        Degree in the Y direction

    k1: int
        Quadrature order in the X direction
    k2: int
        Quadrature order in the Y direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X direction
    global_basis_2: ndarray of floats
        Basis functions values at each cell and quadrature points in the Y direction

    global_spans_1: ndarray of ints
        Spans in the X direction
    global_spans_2: ndarray of ints
        Spans in the Y direction


    glob_arr_coeff: ndarray of floats
        Coefficients of the fields in the X and Y directions

    out_fields: ndarray of floats
        Evaluated fields, filled with the correct values by the function
    """

    arr_coeff_fields = np.zeros((1 + f_p1, 1 + f_p2, out_fields.shape[2]))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]
            arr_coeff_fields[:, :, :] = glob_arr_coeff[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                       pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                       :]
            for i_basis_1 in range(1 + f_p1):
                for i_basis_2 in range(1 + f_p2):
                    coeff_fields = arr_coeff_fields[i_basis_1, i_basis_2]

                    for i_quad_1 in range(k1+1):
                        spline_1 = global_basis_1[i_cell_1, i_basis_1, 0, i_quad_1]

                        for i_quad_2 in range(k2+1):
                            spline_2 = global_basis_2[i_cell_2, i_basis_2, 0, i_quad_2]

                            spline = spline_1 * spline_2

                            out_fields[i_cell_1 * (k1 + 1) + i_quad_1,
                                       i_cell_2 * (k2 + 1) + i_quad_2,
                                       :] += spline * coeff_fields


def eval_fields_3d_weighted(nc1: int, nc2: int, nc3: int, pad1: int, pad2: int, pad3: int, f_p1: int, f_p2: int,
                            f_p3: int, k1: int, k2: int, k3: int, global_basis_1: 'float[:,:,:,:]',
                            global_basis_2: 'float[:,:,:,:]', global_basis_3: 'float[:,:,:,:]',
                            global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_spans_3: 'int[:]',
                            glob_arr_coeff: 'float[:,:,:,:]', global_arr_weight: 'float[:,:,:]',
                            out_fields: 'float[:,:,:,:]'):
    """
    Parameters
    ----------
    nc1: int
        Number of cells in the X direction
    nc2: int
        Number of cells in the Y direction
    nc3: int
        Number of cells in the Z direction

    pad1: int
        Padding in the X direction
    pad2: int
        Padding in the Y direction
    pad3: int
        Padding in the Z direction

    f_p1: int
        Degree in the X direction
    f_p2: int
        Degree in the Y direction
    f_p3: int
        Degree in the Z direction

    k1: int
        Quadrature order in the X direction
    k2: int
        Quadrature order in the Y direction
    k3: int
        Quadrature order in the Z direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X direction
    global_basis_2: ndarray of floats
        Basis functions values at each cell and quadrature points in the Y direction
    global_basis_3: ndarray of floats
        Basis functions values at each cell and quadrature points in the Z direction

    global_spans_1: ndarray of ints
        Spans in the X direction
    global_spans_2: ndarray of ints
        Spans in the Y direction
    global_spans_3: ndarray of ints
        Spans in the Z direction

    glob_arr_coeff: ndarray of floats
        Coefficients of the fields in the X,Y and Z directions

    global_arr_weight: ndarray of floats

    out_fields: ndarray of floats
        Evaluated fields, filled with the correct values by the function
    """

    arr_coeff_fields = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3, out_fields.shape[3]))
    arr_coeff_weights = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))

    arr_fields = np.zeros((1 + k1, 1 + k2, 1 + k3, out_fields.shape[3]))
    arr_weights = np.zeros((1 + k1, 1 + k2, 1 + k3))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]

            for i_cell_3 in range(nc3):
                span_3 = global_spans_3[i_cell_3]

                arr_coeff_fields[:, :, :, :] = glob_arr_coeff[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                              pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                              pad3 + span_3 - f_p3:1 + pad3 + span_3,
                                                              :]

                arr_coeff_weights[:, :, :] = global_arr_weight[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                               pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                               pad3 + span_3 - f_p3:1 + pad3 + span_3]

                arr_fields[:, :, :, :] = 0.0
                arr_weights[:, :, :] = 0.0

                for i_quad_1 in range(k1 + 1):
                    for i_quad_2 in range(k2 + 1):
                        for i_quad_3 in range(k3 + 1):
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

                            out_fields[i_cell_1 * (k1 + 1) + i_quad_1,
                                       i_cell_2 * (k2 + 1) + i_quad_2,
                                       i_cell_3 * (k3 + 1) + i_quad_3,
                                       :] += fields / weight


def eval_fields_2d_weighted(nc1: int, nc2: int, pad1: int, pad2: int, f_p1: int, f_p2: int, k1: int, k2: int,
                            global_basis_1: 'float[:,:,:,:]', global_basis_2: 'float[:,:,:,:]',
                            global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_arr_coeff: 'float[:,:,:]',
                            global_arr_weight: 'float[:,:]', out_fields: 'float[:,:,:]'):
    """
    Parameters
    ----------
    nc1: int
        Number of cells in the X direction
    nc2: int
        Number of cells in the Y direction

    pad1: int
        Padding in the X direction
    pad2: int
        Padding in the Y direction

    f_p1: int
        Degree in the X direction
    f_p2: int
        Degree in the Y direction

    k1: int
        Quadrature order in the X direction
    k2: int
        Quadrature order in the Y direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X direction
    global_basis_2: ndarray of floats
        Basis functions values at each cell and quadrature points in the Y direction

    global_spans_1: ndarray of ints
        Spans in the X direction
    global_spans_2: ndarray of ints
        Spans in the Y direction

    global_arr_coeff: ndarray of floats
        Coefficients of the fields in the X,Y and Z directions

    global_arr_weight: ndarray of floats

    out_fields: ndarray of floats
        Evaluated fields, filled with the correct values by the function
    """

    arr_coeff_fields = np.zeros((1 + f_p1, 1 + f_p2, out_fields.shape[2]))
    arr_coeff_weights = np.zeros((1 + f_p1, 1 + f_p2))

    arr_fields = np.zeros((1 + k1, 1 + k2, out_fields.shape[2]))
    arr_weights = np.zeros((1 + k1, 1 + k2))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]

            arr_coeff_fields[:, :, :] = global_arr_coeff[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                         pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                         :]

            arr_coeff_weights[:, :] = global_arr_weight[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                        pad2 + span_2 - f_p2:1 + pad2 + span_2]

            arr_fields[:, :, :] = 0.0
            arr_weights[:, :] = 0.0

            for i_quad_1 in range(k1 + 1):
                for i_quad_2 in range(k2 + 1):
                    for i_basis_1 in range(1 + f_p1):
                        spline_1 = global_basis_1[i_cell_1, i_basis_1, 0, i_quad_1]

                        for i_basis_2 in range(1 + f_p2):
                            spline_2 = global_basis_2[i_cell_2, i_basis_2, 0, i_quad_2]

                            splines = spline_1 * spline_2

                            coeff_fields = arr_coeff_fields[i_basis_1, i_basis_2, :]
                            coeff_weight = arr_coeff_weights[i_basis_1, i_basis_2]

                            arr_fields[i_quad_1, i_quad_2, :] += splines * coeff_fields * coeff_weight

                            arr_weights[i_quad_1, i_quad_2] += splines * coeff_weight

                    fields = arr_fields[i_quad_1, i_quad_2, :]
                    weight = arr_weights[i_quad_1, i_quad_2]

                    out_fields[i_cell_1 * (k1 + 1) + i_quad_1,
                               i_cell_2 * (k2 + 1) + i_quad_2,
                               :] += fields / weight
