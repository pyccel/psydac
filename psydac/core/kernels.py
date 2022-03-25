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
        Number of evaluation points in the X direction
    k2: int
        Number of evaluation points in the Y direction
    k3: int
        Number of evaluation points in the Z direction

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


def eval_fields_3d_irregular_no_weights(np1: int, np2: int, np3: int, pad1: int, pad2: int, pad3: int, f_p1: int, f_p2: int,
                                        f_p3: int, cell_index_1: 'int[:]', cell_index_2: 'int[:]', cell_index_3 : 'int[:]', 
                                        global_basis_1: 'float[:,:,:]', global_basis_2: 'float[:,:,:]', global_basis_3: 'float[:,:,:]',
                                        global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_spans_3: 'int[:]',
                                        glob_arr_coeff: 'float[:,:,:,:]', out_fields: 'float[:,:,:,:]'):
    """
    Parameters
    ----------
    np1 : int
        Number of points in the X direction
    np2 : int
        Number of points in the Y direction
    np3 : int
        Number of points in the Z direction

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

    cell_index_1 : ndarray of ints
        Index of the cells in the X direction
    cell_index_2 : ndarray of ints
        Index of the cells in the Y direction
    cell_index_3 : ndarray of ints
        Index of the cells in the Z direction

    global_basis_1 : ndarray of floats
        Basis functions values at each point in the X direction
    global_basis_2 : ndarray of floats
        Basis functions values at each point in the Y direction
    global_basis_3 : ndarray of floats
        Basis functions values at each point in the Z direction

    global_spans_1 : ndarray of ints
        Spans in the X direction
    global_spans_2 : ndarray of ints
        Spans in the Y direction
    global_spans_3 : ndarray of ints
        Spans in the Z direction

    glob_arr_coeff : ndarray of floats
        Coefficients of the fields in the X and Y directions

    out_fields : ndarray of floats
        Evaluated fields, filled with the correct values by the function
    """    
    arr_coeff_fields = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3, out_fields.shape[3]))

    for i_p_1 in range(np1):
        i_cell_1 = cell_index_1[i_p_1]
        span_1 = global_spans_1[i_cell_1]

        for i_p_2 in range(np2):
            i_cell_2 = cell_index_2[i_p_2]
            span_2 = global_spans_2[i_cell_2]

            for i_p_3 in range(np3):
                i_cell_3 = cell_index_3[i_p_3]
                span_3 = global_spans_3[i_cell_3]

                arr_coeff_fields[:, :, :, :] = glob_arr_coeff[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                              pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                              pad3 + span_3 - f_p3:1 + pad3 + span_3,
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
        Number of evaluation points in the X direction
    k2: int
        Number of evaluation points in the Y direction

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
                    coeff_fields = arr_coeff_fields[i_basis_1, i_basis_2, :]

                    for i_quad_1 in range(k1):
                        spline_1 = global_basis_1[i_cell_1, i_basis_1, 0, i_quad_1]

                        for i_quad_2 in range(k2):
                            spline_2 = global_basis_2[i_cell_2, i_basis_2, 0, i_quad_2]

                            spline = spline_1 * spline_2

                            out_fields[i_cell_1 * k1 + i_quad_1,
                                       i_cell_2 * k2 + i_quad_2,
                                       :] += spline * coeff_fields


def eval_fields_2d_irregular_no_weights(np1: int, np2: int, pad1: int, pad2: int, f_p1: int, f_p2: int,
                                        cell_index_1: 'int[:]', cell_index_2: 'int[:]',
                                        global_basis_1: 'float[:,:,:]', global_basis_2: 'float[:,:,:]',
                                        global_spans_1: 'int[:]', global_spans_2: 'int[:]', glob_arr_coeff: 'float[:,:,:]',
                                        out_fields: 'float[:,:,:]'):
    """
    Parameters
    ----------
    np1: int
        Number of points in the X direction
    np2: int
        Number of points in the Y direction

    pad1: int
        Padding in the X direction
    pad2: int
        Padding in the Y direction

    f_p1: int
        Degree in the X direction
    f_p2: int
        Degree in the Y direction

    cell_index_1 : ndarray of ints
        Index of the cells in the X direction
    cell_index_2 : ndarray of ints
        Index of the cells in the Y direction

    global_basis_1: ndarray of floats
        Basis functions values at each point in the X direction
    global_basis_2: ndarray of floats
        Basis functions values at each point in the Y direction

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

    for i_p_1 in range(np1):
        i_cell_1 = cell_index_1[i_p_1]
        span_1 = global_spans_1[i_cell_1]

        for i_p_2 in range(np2):
            i_cell_2 = cell_index_2[i_p_2]
            span_2 = global_spans_2[i_cell_2]

            arr_coeff_fields[:, :, :] = glob_arr_coeff[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                       pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                       :]
            
            for i_basis_1 in range(1 + f_p1):
                spline_1 = global_basis_1[i_p_1, i_basis_1, 0]

                for i_basis_2 in range(1 + f_p2):
                    spline_2 = global_basis_2[i_p_2, i_basis_2, 0]

                    spline = spline_1 * spline_2

                    coeff_fields = arr_coeff_fields[i_basis_1, i_basis_2, :]

                    out_fields[i_p_1, i_p_2, :] += spline * coeff_fields


def eval_fields_3d_weighted(nc1: int, nc2: int, nc3: int, pad1: int, pad2: int, pad3: int, f_p1: int, f_p2: int,
                            f_p3: int, k1: int, k2: int, k3: int, global_basis_1: 'float[:,:,:,:]',
                            global_basis_2: 'float[:,:,:,:]', global_basis_3: 'float[:,:,:,:]',
                            global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_spans_3: 'int[:]',
                            glob_arr_coeff: 'float[:,:,:,:]', global_arr_weights: 'float[:,:,:]',
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
        Number of evaluation points in the X direction
    k2: int
        Number of evaluation points in the Y direction
    k3: int
        Number of evaluation points in the Z direction

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

    global_arr_weights: ndarray of float
        Coefficients of the weight field in the X, Y and Z directions

    out_fields: ndarray of floats
        Evaluated fields, filled with the correct values by the function
    """

    arr_coeff_fields = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3, out_fields.shape[3]))
    arr_coeff_weights = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))

    arr_fields = np.zeros((k1, k2, k3, out_fields.shape[3]))
    arr_weights = np.zeros((k1, k2, k3))

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

                arr_coeff_weights[:, :, :] = global_arr_weights[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                                pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                                pad3 + span_3 - f_p3:1 + pad3 + span_3]

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


def eval_fields_3d_irregular_weighted(np1: int, np2: int, np3: int, pad1: int, pad2: int, pad3: int, f_p1: int, f_p2: int,
                                      f_p3: int, cell_index_1: 'int[:]', cell_index_2: 'int[:]', cell_index_3 : 'int[:]', 
                                      global_basis_1: 'float[:,:,:]', global_basis_2: 'float[:,:,:]', global_basis_3: 'float[:,:,:]',
                                      global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_spans_3: 'int[:]',
                                      glob_arr_coeff: 'float[:,:,:,:]', global_arr_weights: 'float[:,:,:]',
                                      out_fields: 'float[:,:,:,:]'):
    """
    Parameters
    ----------
    np1 : int
        Number of points in the X direction
    np2 : int
        Number of points in the Y direction
    np3 : int
        Number of points in the Z direction

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

    cell_index_1 : ndarray of ints
        Index of the cells in the X direction
    cell_index_2 : ndarray of ints
        Index of the cells in the Y direction
    cell_index_3 : ndarray of ints
        Index of the cells in the Z direction

    global_basis_1 : ndarray of floats
        Basis functions values at each point in the X direction
    global_basis_2 : ndarray of floats
        Basis functions values at each point in the Y direction
    global_basis_3 : ndarray of floats
        Basis functions values at each point in the Z direction

    global_spans_1 : ndarray of ints
        Spans in the X direction
    global_spans_2 : ndarray of ints
        Spans in the Y direction
    global_spans_3 : ndarray of ints
        Spans in the Z direction

    glob_arr_coeff : ndarray of floats
        Coefficients of the fields in the X, Y and Z directions

    global_arr_weights: ndarray of float
        Coefficients of the weight field in the X, Y and Z directions

    out_fields : ndarray of floats
        Evaluated fields, filled with the correct values by the function
    """
    arr_coeff_fields = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3, out_fields.shape[3]))
    arr_coeff_weights = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))

    temp_fields = np.zeros(out_fields.shape[3])

    for i_p_1 in range(np1):
        i_cell_1 = cell_index_1[i_p_1]
        span_1 = global_spans_1[i_cell_1]

        for i_p_2 in range(np2):
            i_cell_2 = cell_index_2[i_p_2]
            span_2 = global_spans_2[i_cell_2]

            for i_p_3 in range(np3):
                i_cell_3 = cell_index_3[i_p_3]
                span_3 = global_spans_3[i_cell_3]

                arr_coeff_fields[:, :, :, :] = glob_arr_coeff[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                              pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                              pad3 + span_3 - f_p3:1 + pad3 + span_3,
                                                              :]

                arr_coeff_weights[:, :, :] = global_arr_weights[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                                pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                                pad3 + span_3 - f_p3:1 + pad3 + span_3]

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

                            temp_weight+= spline * coeff_weight

                out_fields[i_p_1, i_p_2, i_p_3, :] += temp_fields / temp_weight


def eval_fields_2d_weighted(nc1: int, nc2: int, pad1: int, pad2: int, f_p1: int, f_p2: int, k1: int, k2: int,
                            global_basis_1: 'float[:,:,:,:]', global_basis_2: 'float[:,:,:,:]',
                            global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_arr_coeff: 'float[:,:,:]',
                            global_arr_weights: 'float[:,:]', out_fields: 'float[:,:,:]'):
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
        Number of evaluation points in the X direction
    k2: int
        Number of evaluation points in the Y direction

    global_basis_1: ndarray of float
        Basis functions values at each cell and quadrature points in the X direction
    global_basis_2: ndarray of float
        Basis functions values at each cell and quadrature points in the Y direction

    global_spans_1: ndarray of int
        Spans in the X direction
    global_spans_2: ndarray of int
        Spans in the Y direction

    global_arr_coeff: ndarray of float
        Coefficients of the fields in the X,Y and Z directions

    global_arr_weights: ndarray of float
        Coefficients of the weight field in the X,Y and Z directions

    out_fields: ndarray of float
        Evaluated fields, filled with the correct values by the function
    """

    arr_coeff_fields = np.zeros((1 + f_p1, 1 + f_p2, out_fields.shape[2]))
    arr_coeff_weights = np.zeros((1 + f_p1, 1 + f_p2))

    arr_fields = np.zeros((k1, k2, out_fields.shape[2]))
    arr_weights = np.zeros((k1, k2))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]

            arr_coeff_fields[:, :, :] = global_arr_coeff[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                         pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                         :]

            arr_coeff_weights[:, :] = global_arr_weights[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                         pad2 + span_2 - f_p2:1 + pad2 + span_2]

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


def eval_fields_2d_irregular_weighted(np1: int, np2: int, pad1: int, pad2: int, f_p1: int, f_p2: int,
                                      cell_index_1: 'int[:]', cell_index_2: 'int[:]', global_basis_1: 'float[:,:,:]', 
                                      global_basis_2: 'float[:,:,:]', global_spans_1: 'int[:]', global_spans_2: 'int[:]', 
                                      glob_arr_coeff: 'float[:,:,:]', global_arr_weights: 'float[:,:]',
                                      out_fields: 'float[:,:,:]'):
    """
    Parameters
    ----------
    np1 : int
        Number of points in the X direction
    np2 : int
        Number of points in the Y direction

    pad1: int
        Padding in the X direction
    pad2: int
        Padding in the Y direction

    f_p1: int
        Degree in the X direction
    f_p2: int
        Degree in the Y direction

    cell_index_1 : ndarray of ints
        Index of the cells in the X direction
    cell_index_2 : ndarray of ints
        Index of the cells in the Y direction

    global_basis_1 : ndarray of floats
        Basis functions values at each point in the X direction
    global_basis_2 : ndarray of floats
        Basis functions values at each point in the Y direction

    global_spans_1 : ndarray of ints
        Spans in the X direction
    global_spans_2 : ndarray of ints
        Spans in the Y direction

    glob_arr_coeff : ndarray of floats
        Coefficients of the fields in the X and Y directions

    global_arr_weights: ndarray of float
        Coefficients of the weight field in the X and Y directions

    out_fields : ndarray of floats
        Evaluated fields, filled with the correct values by the function
    """
    arr_coeff_fields = np.zeros((1 + f_p1, 1 + f_p2, out_fields.shape[2]))
    arr_coeff_weights = np.zeros((1 + f_p1, 1 + f_p2))

    temp_fields = np.zeros(out_fields.shape[2])

    for i_p_1 in range(np1):
        i_cell_1 = cell_index_1[i_p_1]
        span_1 = global_spans_1[i_cell_1]

        for i_p_2 in range(np2):
            i_cell_2 = cell_index_2[i_p_2]
            span_2 = global_spans_2[i_cell_2]


            arr_coeff_fields[:, :, :] = glob_arr_coeff[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                       pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                       :]

            arr_coeff_weights[:, :] = global_arr_weights[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                         pad2 + span_2 - f_p2:1 + pad2 + span_2]

            temp_fields[:] = 0.0
            temp_weight = 0.0

            for i_basis_1 in range(1 + f_p1):
                spline_1 = global_basis_1[i_p_1, i_basis_1, 0]

                for i_basis_2 in range(1 + f_p2):
                    spline_2 = global_basis_2[i_p_2, i_basis_2, 0]

                    spline = spline_1 * spline_2

                    coeff_fields = arr_coeff_fields[i_basis_1, i_basis_2, :]

                    coeff_weight = arr_coeff_weights[i_basis_1, i_basis_2]

                    temp_fields[:] += spline * coeff_fields * coeff_weight

                    temp_weight+= spline * coeff_weight

            out_fields[i_p_1, i_p_2, :] += temp_fields / temp_weight


def eval_jac_det_3d(nc1: int, nc2: int, nc3: int, pad1: int, pad2: int, pad3: int, f_p1: int, f_p2: int, f_p3: int,
                    k1: int, k2: int, k3: int, global_basis_1: 'float[:,:,:,:]', global_basis_2: 'float[:,:,:,:]',
                    global_basis_3: 'float[:,:,:,:]', global_spans_1: 'int[:]', global_spans_2: 'int[:]',
                    global_spans_3: 'int[:]', global_arr_coeff_x: 'float[:,:,:]', global_arr_coeff_y: 'float[:,:,:]',
                    global_arr_coeff_z: 'float[:,:,:]', jac_det: 'float[:,:,:]'):

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
        Number of evaluation points in the X direction
    k2: int
        Number of evaluation points in the Y direction
    k3: int
        Number of evaluation points in the Z direction

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

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the Y field
    global_arr_coeff_z: ndarray of floats
        Coefficients of the Z field

    jac_det: ndarray of floats
        Jacobian determinant on the grid.
    """

    arr_coeffs_x = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_y = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_z = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))

    arr_x_x1 = np.zeros((k1, k2, k3))
    arr_x_x2 = np.zeros((k1, k2, k3))
    arr_x_x3 = np.zeros((k1, k2, k3))

    arr_y_x1 = np.zeros((k1, k2, k3))
    arr_y_x2 = np.zeros((k1, k2, k3))
    arr_y_x3 = np.zeros((k1, k2, k3))

    arr_z_x1 = np.zeros((k1, k2, k3))
    arr_z_x2 = np.zeros((k1, k2, k3))
    arr_z_x3 = np.zeros((k1, k2, k3))

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

                arr_coeffs_x[:, :, :] = global_arr_coeff_x[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                           pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                           pad3 + span_3 - f_p3:1 + pad3 + span_3]

                arr_coeffs_y[:, :, :] = global_arr_coeff_y[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                           pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                           pad3 + span_3 - f_p3:1 + pad3 + span_3]

                arr_coeffs_z[:, :, :] = global_arr_coeff_z[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                           pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                           pad3 + span_3 - f_p3:1 + pad3 + span_3]
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


def eval_jac_det_2d(nc1: int, nc2: int, pad1: int, pad2: int, f_p1: int, f_p2: int, k1: int, k2: int,
                    global_basis_1: 'float[:,:,:,:]', global_basis_2: 'float[:,:,:,:]', global_spans_1: 'int[:]',
                    global_spans_2: 'int[:]', global_arr_coeff_x: 'float[:,:]', global_arr_coeff_y: 'float[:,:]',
                    jac_det: 'float[:,:]'):
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
        Number of evaluation points in the X direction
    k2: int
        Number of evaluation points in the Y direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X direction
    global_basis_2: ndarray of floats
        Basis functions values at each cell and quadrature points in the Y direction

    global_spans_1: ndarray of ints
        Spans in the X direction
    global_spans_2: ndarray of ints
        Spans in the Y direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the Y field

    jac_det: ndarray of floats
        Jacobian determinant on the grid.
    """

    arr_coeffs_x = np.zeros((1 + f_p1, 1 + f_p2))
    arr_coeffs_y = np.zeros((1 + f_p1, 1 + f_p2))

    arr_x_x1 = np.zeros((k1, k2))
    arr_x_x2 = np.zeros((k1, k2))

    arr_y_x1 = np.zeros((k1, k2))
    arr_y_x2 = np.zeros((k1, k2))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]

            arr_x_x1[:, :] = 0.0
            arr_x_x2[:, :] = 0.0

            arr_y_x1[:, :] = 0.0
            arr_y_x2[:, :] = 0.0

            arr_coeffs_x[:, :] = global_arr_coeff_x[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                    pad2 + span_2 - f_p2:1 + pad2 + span_2]

            arr_coeffs_y[:, :] = global_arr_coeff_y[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                    pad2 + span_2 - f_p2:1 + pad2 + span_2]

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


def eval_jac_det_3d_weights(nc1: int, nc2: int, nc3: int, pad1: int, pad2: int, pad3: int, f_p1: int, f_p2: int,
                            f_p3: int, k1: int, k2: int, k3: int, global_basis_1: 'float[:,:,:,:]',
                            global_basis_2: 'float[:,:,:,:]', global_basis_3: 'float[:,:,:,:]',
                            global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_spans_3: 'int[:]',
                            global_arr_coeff_x: 'float[:,:,:]', global_arr_coeff_y: 'float[:,:,:]',
                            global_arr_coeff_z: 'float[:,:,:]', global_arr_coeff_weights: 'float[:,:,:]',
                            jac_det: 'float[:,:,:]'):

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
        Number of evaluation points in the X direction
    k2: int
        Number of evaluation points in the Y direction
    k3: int
        Number of evaluation points in the Z direction

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

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the Y field
    global_arr_coeff_z: ndarray of floats
        Coefficients of the Z field

    global_arr_coeff_weights: ndarray of floats
        Coefficients of the weight field

    jac_det: ndarray of floats
        Jacobian determinant on the grid
    """

    arr_coeffs_x = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_y = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_z = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeff_weights = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))

    arr_x = np.zeros((k1, k2, k3))
    arr_y = np.zeros((k1, k2, k3))
    arr_z = np.zeros((k1, k2, k3))

    arr_x_x1 = np.zeros((k1, k2, k3))
    arr_x_x2 = np.zeros((k1, k2, k3))
    arr_x_x3 = np.zeros((k1, k2, k3))

    arr_y_x1 = np.zeros((k1, k2, k3))
    arr_y_x2 = np.zeros((k1, k2, k3))
    arr_y_x3 = np.zeros((k1, k2, k3))

    arr_z_x1 = np.zeros((k1, k2, k3))
    arr_z_x2 = np.zeros((k1, k2, k3))
    arr_z_x3 = np.zeros((k1, k2, k3))

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

                arr_coeffs_x[:, :, :] = global_arr_coeff_x[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                           pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                           pad3 + span_3 - f_p3:1 + pad3 + span_3]

                arr_coeffs_y[:, :, :] = global_arr_coeff_y[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                           pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                           pad3 + span_3 - f_p3:1 + pad3 + span_3]

                arr_coeffs_z[:, :, :] = global_arr_coeff_z[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                           pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                           pad3 + span_3 - f_p3:1 + pad3 + span_3]

                arr_coeff_weights[:, :, :] = global_arr_coeff_weights[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                                      pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                                      pad3 + span_3 - f_p3:1 + pad3 + span_3]

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

                                        arr_x[i_quad_1, i_quad_2, i_quad_3] += mapping * coeff_x
                                        arr_y[i_quad_1, i_quad_2, i_quad_3] += mapping * coeff_y
                                        arr_z[i_quad_1, i_quad_2, i_quad_3] += mapping * coeff_z

                                        arr_x_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_x
                                        arr_x_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_x
                                        arr_x_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_x

                                        arr_y_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_y
                                        arr_y_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_y
                                        arr_y_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_y

                                        arr_z_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_z
                                        arr_z_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_z
                                        arr_z_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_z

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


def eval_jac_det_2d_weights(nc1: int, nc2: int, pad1: int, pad2: int, f_p1: int, f_p2: int, k1: int, k2: int,
                            global_basis_1: 'float[:,:,:,:]', global_basis_2: 'float[:,:,:,:]',
                            global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_arr_coeff_x: 'float[:,:]',
                            global_arr_coeff_y: 'float[:,:]', global_arr_coeff_weights: 'float[:,:]',
                            jac_det: 'float[:,:]'):
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
           Number of evaluation points in the X direction
       k2: int
           Number of evaluation points in the Y direction

       global_basis_1: ndarray of floats
           Basis functions values at each cell and quadrature points in the X direction
       global_basis_2: ndarray of floats
           Basis functions values at each cell and quadrature points in the Y direction

       global_spans_1: ndarray of ints
           Spans in the X direction
       global_spans_2: ndarray of ints
           Spans in the Y direction

       global_arr_coeff_x: ndarray of floats
           Coefficients of the X field
       global_arr_coeff_y: ndarray of floats
           Coefficients of the Y field

       global_arr_coeff_weights: ndarray of floats
           Coefficients of the weights field

       jac_det: ndarray of floats
           Jacobian determinant on the grid
       """
    arr_coeffs_x = np.zeros((1 + f_p1, 1 + f_p2))
    arr_coeffs_y = np.zeros((1 + f_p1, 1 + f_p2))
    arr_coeff_weights = np.zeros((1 + f_p1, 1 + f_p2))

    arr_x = np.zeros((k1, k2))
    arr_y = np.zeros((k1, k2))

    arr_x_x1 = np.zeros((k1, k2))
    arr_x_x2 = np.zeros((k1, k2))

    arr_y_x1 = np.zeros((k1, k2))
    arr_y_x2 = np.zeros((k1, k2))

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

            arr_coeffs_x[:, :] = global_arr_coeff_x[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                    pad2 + span_2 - f_p2:1 + pad2 + span_2]

            arr_coeffs_y[:, :] = global_arr_coeff_y[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                    pad2 + span_2 - f_p2:1 + pad2 + span_2]

            arr_coeff_weights[:, :] = global_arr_coeff_weights[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                               pad2 + span_2 - f_p2:1 + pad2 + span_2]

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

                            coeff_weights = arr_coeff_weights[i_basis_1, i_basis_2]

                            arr_x[i_quad_1, i_quad_2] += mapping * coeff_x
                            arr_y[i_quad_1, i_quad_2] += mapping * coeff_y

                            arr_x_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_x
                            arr_x_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_x

                            arr_y_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_y
                            arr_y_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_y

                            arr_weights[i_quad_1, i_quad_2] += mapping * coeff_weights

                            arr_weights_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_weights
                            arr_weights_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_weights

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


def eval_jacobians_3d(nc1: int, nc2: int, nc3: int, pad1: int, pad2: int, pad3: int, f_p1: int, f_p2: int, f_p3: int,
                      k1: int, k2: int, k3: int, global_basis_1: 'float[:,:,:,:]', global_basis_2: 'float[:,:,:,:]',
                      global_basis_3: 'float[:,:,:,:]', global_spans_1: 'int[:]', global_spans_2: 'int[:]',
                      global_spans_3: 'int[:]', global_arr_coeff_x: 'float[:,:,:]', global_arr_coeff_y: 'float[:,:,:]',
                      global_arr_coeff_z: 'float[:,:,:]', jacobians: 'float[:,:,:,:,:]'):

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
        Number of evaluation points in the X direction
    k2: int
        Number of evaluation points in the Y direction
    k3: int
        Number of evaluation points in the Z direction

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

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the Y field
    global_arr_coeff_z: ndarray of floats
        Coefficients of the Z field

    jacobians: ndarray of floats
        Jacobian matrix on the grid
    """

    arr_coeffs_x = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_y = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_z = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))

    arr_x_x1 = np.zeros((k1, k2, k3))
    arr_x_x2 = np.zeros((k1, k2, k3))
    arr_x_x3 = np.zeros((k1, k2, k3))

    arr_y_x1 = np.zeros((k1, k2, k3))
    arr_y_x2 = np.zeros((k1, k2, k3))
    arr_y_x3 = np.zeros((k1, k2, k3))

    arr_z_x1 = np.zeros((k1, k2, k3))
    arr_z_x2 = np.zeros((k1, k2, k3))
    arr_z_x3 = np.zeros((k1, k2, k3))

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

                arr_coeffs_x[:, :, :] = global_arr_coeff_x[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                           pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                           pad3 + span_3 - f_p3:1 + pad3 + span_3]

                arr_coeffs_y[:, :, :] = global_arr_coeff_y[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                           pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                           pad3 + span_3 - f_p3:1 + pad3 + span_3]

                arr_coeffs_z[:, :, :] = global_arr_coeff_z[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                           pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                           pad3 + span_3 - f_p3:1 + pad3 + span_3]
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


def eval_jacobians_2d(nc1: int, nc2: int, pad1: int, pad2: int, f_p1: int, f_p2: int, k1: int, k2: int,
                      global_basis_1: 'float[:,:,:,:]', global_basis_2: 'float[:,:,:,:]', global_spans_1: 'int[:]',
                      global_spans_2: 'int[:]', global_arr_coeff_x: 'float[:,:]', global_arr_coeff_y: 'float[:,:]',
                      jacobians: 'float[:,:,:,:]'):
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
        Number of evaluation points in the X direction
    k2: int
        Number of evaluation points in the Y direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X direction
    global_basis_2: ndarray of floats
        Basis functions values at each cell and quadrature points in the Y direction

    global_spans_1: ndarray of ints
        Spans in the X direction
    global_spans_2: ndarray of ints
        Spans in the Y direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the Y field

    jacobians: ndarray of floats
        Jacobian matrix at every point of the grid
    """

    arr_coeffs_x = np.zeros((1 + f_p1, 1 + f_p2))
    arr_coeffs_y = np.zeros((1 + f_p1, 1 + f_p2))

    arr_x_x1 = np.zeros((k1, k2))
    arr_x_x2 = np.zeros((k1, k2))

    arr_y_x1 = np.zeros((k1, k2))
    arr_y_x2 = np.zeros((k1, k2))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]

            arr_x_x1[:, :] = 0.0
            arr_x_x2[:, :] = 0.0

            arr_y_x1[:, :] = 0.0
            arr_y_x2[:, :] = 0.0

            arr_coeffs_x[:, :] = global_arr_coeff_x[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                    pad2 + span_2 - f_p2:1 + pad2 + span_2]

            arr_coeffs_y[:, :] = global_arr_coeff_y[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                    pad2 + span_2 - f_p2:1 + pad2 + span_2]

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


def eval_jacobians_3d_weights(nc1: int, nc2: int, nc3: int, pad1: int, pad2: int, pad3: int, f_p1: int, f_p2: int,
                              f_p3: int, k1: int, k2: int, k3: int, global_basis_1: 'float[:,:,:,:]',
                              global_basis_2: 'float[:,:,:,:]', global_basis_3: 'float[:,:,:,:]',
                              global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_spans_3: 'int[:]',
                              global_arr_coeff_x: 'float[:,:,:]', global_arr_coeff_y: 'float[:,:,:]',
                              global_arr_coeff_z: 'float[:,:,:]', global_arr_coeff_weights: 'float[:,:,:]',
                              jacobians: 'float[:,:,:,:,:]'):

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
        Number of evaluation points in the X direction
    k2: int
        Number of evaluation points in the Y direction
    k3: int
        Number of evaluation points in the Z direction

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

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the Y field
    global_arr_coeff_z: ndarray of floats
        Coefficients of the Z field

    global_arr_coeff_weights: ndarray of floats
        Coefficients of the weight field

    jacobians: ndarray of floats
        Jacobian matrix on the grid
    """

    arr_coeffs_x = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_y = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_z = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeff_weights = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))

    arr_x = np.zeros((k1, k2, k3))
    arr_y = np.zeros((k1, k2, k3))
    arr_z = np.zeros((k1, k2, k3))

    arr_x_x1 = np.zeros((k1, k2, k3))
    arr_x_x2 = np.zeros((k1, k2, k3))
    arr_x_x3 = np.zeros((k1, k2, k3))

    arr_y_x1 = np.zeros((k1, k2, k3))
    arr_y_x2 = np.zeros((k1, k2, k3))
    arr_y_x3 = np.zeros((k1, k2, k3))

    arr_z_x1 = np.zeros((k1, k2, k3))
    arr_z_x2 = np.zeros((k1, k2, k3))
    arr_z_x3 = np.zeros((k1, k2, k3))

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

                arr_coeffs_x[:, :, :] = global_arr_coeff_x[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                           pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                           pad3 + span_3 - f_p3:1 + pad3 + span_3]

                arr_coeffs_y[:, :, :] = global_arr_coeff_y[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                           pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                           pad3 + span_3 - f_p3:1 + pad3 + span_3]

                arr_coeffs_z[:, :, :] = global_arr_coeff_z[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                           pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                           pad3 + span_3 - f_p3:1 + pad3 + span_3]

                arr_coeff_weights[:, :, :] = global_arr_coeff_weights[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                                      pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                                      pad3 + span_3 - f_p3:1 + pad3 + span_3]

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

                                        arr_x[i_quad_1, i_quad_2, i_quad_3] += mapping * coeff_x
                                        arr_y[i_quad_1, i_quad_2, i_quad_3] += mapping * coeff_y
                                        arr_z[i_quad_1, i_quad_2, i_quad_3] += mapping * coeff_z

                                        arr_x_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_x
                                        arr_x_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_x
                                        arr_x_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_x

                                        arr_y_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_y
                                        arr_y_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_y
                                        arr_y_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_y

                                        arr_z_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_z
                                        arr_z_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_z
                                        arr_z_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_z

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


def eval_jacobians_2d_weights(nc1: int, nc2: int, pad1: int, pad2: int, f_p1: int, f_p2: int, k1: int, k2: int,
                              global_basis_1: 'float[:,:,:,:]', global_basis_2: 'float[:,:,:,:]',
                              global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_arr_coeff_x: 'float[:,:]',
                              global_arr_coeff_y: 'float[:,:]', global_arr_coeff_weights: 'float[:,:]',
                              jacobians: 'float[:,:,:,:]'):
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
           Number of evaluation points in the X direction
       k2: int
           Number of evaluation points in the Y direction

       global_basis_1: ndarray of floats
           Basis functions values at each cell and quadrature points in the X direction
       global_basis_2: ndarray of floats
           Basis functions values at each cell and quadrature points in the Y direction

       global_spans_1: ndarray of ints
           Spans in the X direction
       global_spans_2: ndarray of ints
           Spans in the Y direction

       global_arr_coeff_x: ndarray of floats
           Coefficients of the X field
       global_arr_coeff_y: ndarray of floats
           Coefficients of the Y field

       global_arr_coeff_weights: ndarray of floats
           Coefficients of the weights field

       jacobians: ndarray of floats
           Jacobian matrix at every point of the grid
       """
    arr_coeffs_x = np.zeros((1 + f_p1, 1 + f_p2))
    arr_coeffs_y = np.zeros((1 + f_p1, 1 + f_p2))
    arr_coeff_weights = np.zeros((1 + f_p1, 1 + f_p2))

    arr_x = np.zeros((k1, k2))
    arr_y = np.zeros((k1, k2))

    arr_x_x1 = np.zeros((k1, k2))
    arr_x_x2 = np.zeros((k1, k2))

    arr_y_x1 = np.zeros((k1, k2))
    arr_y_x2 = np.zeros((k1, k2))

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

            arr_coeffs_x[:, :] = global_arr_coeff_x[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                    pad2 + span_2 - f_p2:1 + pad2 + span_2]

            arr_coeffs_y[:, :] = global_arr_coeff_y[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                    pad2 + span_2 - f_p2:1 + pad2 + span_2]

            arr_coeff_weights[:, :] = global_arr_coeff_weights[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                               pad2 + span_2 - f_p2:1 + pad2 + span_2]

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

                            coeff_weights = arr_coeff_weights[i_basis_1, i_basis_2]

                            arr_x[i_quad_1, i_quad_2] += mapping * coeff_x
                            arr_y[i_quad_1, i_quad_2] += mapping * coeff_y

                            arr_x_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_x
                            arr_x_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_x

                            arr_y_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_y
                            arr_y_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_y

                            arr_weights[i_quad_1, i_quad_2] += mapping * coeff_weights

                            arr_weights_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_weights
                            arr_weights_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_weights

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


def eval_jacobians_inv_3d(nc1: int, nc2: int, nc3: int, pad1: int, pad2: int, pad3: int, f_p1: int, f_p2: int,
                          f_p3: int, k1: int, k2: int, k3: int, global_basis_1: 'float[:,:,:,:]',
                          global_basis_2: 'float[:,:,:,:]', global_basis_3: 'float[:,:,:,:]', global_spans_1: 'int[:]',
                          global_spans_2: 'int[:]', global_spans_3: 'int[:]', global_arr_coeff_x: 'float[:,:,:]',
                          global_arr_coeff_y: 'float[:,:,:]', global_arr_coeff_z: 'float[:,:,:]',
                          jacobians_inv: 'float[:,:,:,:,:]'):

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
        Number of evaluation points in the X direction
    k2: int
        Number of evaluation points in the Y direction
    k3: int
        Number of evaluation points in the Z direction

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

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the Y field
    global_arr_coeff_z: ndarray of floats
        Coefficients of the Z field

    jacobians_inv: ndarray of floats
        Inverse of the Jacobian matrix on the grid
    """

    arr_coeffs_x = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_y = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_z = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))

    arr_x_x1 = np.zeros((k1, k2, k3))
    arr_x_x2 = np.zeros((k1, k2, k3))
    arr_x_x3 = np.zeros((k1, k2, k3))

    arr_y_x1 = np.zeros((k1, k2, k3))
    arr_y_x2 = np.zeros((k1, k2, k3))
    arr_y_x3 = np.zeros((k1, k2, k3))

    arr_z_x1 = np.zeros((k1, k2, k3))
    arr_z_x2 = np.zeros((k1, k2, k3))
    arr_z_x3 = np.zeros((k1, k2, k3))

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

                arr_coeffs_x[:, :, :] = global_arr_coeff_x[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                           pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                           pad3 + span_3 - f_p3:1 + pad3 + span_3]

                arr_coeffs_y[:, :, :] = global_arr_coeff_y[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                           pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                           pad3 + span_3 - f_p3:1 + pad3 + span_3]

                arr_coeffs_z[:, :, :] = global_arr_coeff_z[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                           pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                           pad3 + span_3 - f_p3:1 + pad3 + span_3]
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


def eval_jacobians_inv_2d(nc1: int, nc2: int, pad1: int, pad2: int, f_p1: int, f_p2: int, k1: int, k2: int,
                          global_basis_1: 'float[:,:,:,:]', global_basis_2: 'float[:,:,:,:]', global_spans_1: 'int[:]',
                          global_spans_2: 'int[:]', global_arr_coeff_x: 'float[:,:]', global_arr_coeff_y: 'float[:,:]',
                          jacobians_inv: 'float[:,:,:,:]'):
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
        Number of evaluation points in the X direction
    k2: int
        Number of evaluation points in the Y direction

    global_basis_1: ndarray of floats
        Basis functions values at each cell and quadrature points in the X direction
    global_basis_2: ndarray of floats
        Basis functions values at each cell and quadrature points in the Y direction

    global_spans_1: ndarray of ints
        Spans in the X direction
    global_spans_2: ndarray of ints
        Spans in the Y direction

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the Y field

    jacobians_inv: ndarray of floats
        Inverse of the Jacobian matrix at every point of the grid
    """

    arr_coeffs_x = np.zeros((1 + f_p1, 1 + f_p2))
    arr_coeffs_y = np.zeros((1 + f_p1, 1 + f_p2))

    arr_x_x1 = np.zeros((k1, k2))
    arr_x_x2 = np.zeros((k1, k2))

    arr_y_x1 = np.zeros((k1, k2))
    arr_y_x2 = np.zeros((k1, k2))

    for i_cell_1 in range(nc1):
        span_1 = global_spans_1[i_cell_1]

        for i_cell_2 in range(nc2):
            span_2 = global_spans_2[i_cell_2]

            arr_x_x1[:, :] = 0.0
            arr_x_x2[:, :] = 0.0

            arr_y_x1[:, :] = 0.0
            arr_y_x2[:, :] = 0.0

            arr_coeffs_x[:, :] = global_arr_coeff_x[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                    pad2 + span_2 - f_p2:1 + pad2 + span_2]

            arr_coeffs_y[:, :] = global_arr_coeff_y[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                    pad2 + span_2 - f_p2:1 + pad2 + span_2]

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


def eval_jacobians_inv_3d_weights(nc1: int, nc2: int, nc3: int, pad1: int, pad2: int, pad3: int, f_p1: int, f_p2: int,
                                  f_p3: int, k1: int, k2: int, k3: int, global_basis_1: 'float[:,:,:,:]',
                                  global_basis_2: 'float[:,:,:,:]', global_basis_3: 'float[:,:,:,:]',
                                  global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_spans_3: 'int[:]',
                                  global_arr_coeff_x: 'float[:,:,:]', global_arr_coeff_y: 'float[:,:,:]',
                                  global_arr_coeff_z: 'float[:,:,:]', global_arr_coeff_weigths: 'float[:,:,:]',
                                  jacobians_inv: 'float[:,:,:,:,:]'):

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
        Number of evaluation points in the X direction
    k2: int
        Number of evaluation points in the Y direction
    k3: int
        Number of evaluation points in the Z direction

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

    global_arr_coeff_x: ndarray of floats
        Coefficients of the X field
    global_arr_coeff_y: ndarray of floats
        Coefficients of the Y field
    global_arr_coeff_z: ndarray of floats
        Coefficients of the Z field

    global_arr_coeff_weigths: ndarray of floats
        Coefficients of the weight field

    jacobians_inv: ndarray of floats
        Inverse of the Jacobian matrix on the grid
    """

    arr_coeffs_x = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_y = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeffs_z = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))
    arr_coeff_weights = np.zeros((1 + f_p1, 1 + f_p2, 1 + f_p3))

    arr_x = np.zeros((k1, k2, k3))
    arr_y = np.zeros((k1, k2, k3))
    arr_z = np.zeros((k1, k2, k3))

    arr_x_x1 = np.zeros((k1, k2, k3))
    arr_x_x2 = np.zeros((k1, k2, k3))
    arr_x_x3 = np.zeros((k1, k2, k3))

    arr_y_x1 = np.zeros((k1, k2, k3))
    arr_y_x2 = np.zeros((k1, k2, k3))
    arr_y_x3 = np.zeros((k1, k2, k3))

    arr_z_x1 = np.zeros((k1, k2, k3))
    arr_z_x2 = np.zeros((k1, k2, k3))
    arr_z_x3 = np.zeros((k1, k2, k3))

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

                arr_coeffs_x[:, :, :] = global_arr_coeff_x[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                           pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                           pad3 + span_3 - f_p3:1 + pad3 + span_3]

                arr_coeffs_y[:, :, :] = global_arr_coeff_y[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                           pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                           pad3 + span_3 - f_p3:1 + pad3 + span_3]

                arr_coeffs_z[:, :, :] = global_arr_coeff_z[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                           pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                           pad3 + span_3 - f_p3:1 + pad3 + span_3]

                arr_coeff_weights[:, :, :] = global_arr_coeff_weigths[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                                      pad2 + span_2 - f_p2:1 + pad2 + span_2,
                                                                      pad3 + span_3 - f_p3:1 + pad3 + span_3]

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

                                        arr_x[i_quad_1, i_quad_2, i_quad_3] += mapping * coeff_x
                                        arr_y[i_quad_1, i_quad_2, i_quad_3] += mapping * coeff_y
                                        arr_z[i_quad_1, i_quad_2, i_quad_3] += mapping * coeff_z

                                        arr_x_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_x
                                        arr_x_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_x
                                        arr_x_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_x

                                        arr_y_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_y
                                        arr_y_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_y
                                        arr_y_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_y

                                        arr_z_x1[i_quad_1, i_quad_2, i_quad_3] += mapping_x1 * coeff_z
                                        arr_z_x2[i_quad_1, i_quad_2, i_quad_3] += mapping_x2 * coeff_z
                                        arr_z_x3[i_quad_1, i_quad_2, i_quad_3] += mapping_x3 * coeff_z

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


def eval_jacobians_inv_2d_weights(nc1: int, nc2: int, pad1: int, pad2: int, f_p1: int, f_p2: int, k1: int, k2: int,
                                  global_basis_1: 'float[:,:,:,:]', global_basis_2: 'float[:,:,:,:]',
                                  global_spans_1: 'int[:]', global_spans_2: 'int[:]', global_arr_coeff_x: 'float[:,:]',
                                  global_arr_coeff_y: 'float[:,:]', global_arr_coeff_weights: 'float[:,:]',
                                  jacobians_inv: 'float[:,:,:,:]'):
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
           Number of evaluation points in the X direction
       k2: int
           Number of evaluation points in the Y direction

       global_basis_1: ndarray of floats
           Basis functions values at each cell and quadrature points in the X direction
       global_basis_2: ndarray of floats
           Basis functions values at each cell and quadrature points in the Y direction

       global_spans_1: ndarray of ints
           Spans in the X direction
       global_spans_2: ndarray of ints
           Spans in the Y direction

       global_arr_coeff_x: ndarray of floats
           Coefficients of the X field
       global_arr_coeff_y: ndarray of floats
           Coefficients of the Y field

       global_arr_coeff_weights: ndarray of floats
           Coefficients of the weights field

       jacobians_inv: ndarray of floats
           Inverse of the Jacobian matrix at every point of the grid
       """
    arr_coeffs_x = np.zeros((1 + f_p1, 1 + f_p2))
    arr_coeffs_y = np.zeros((1 + f_p1, 1 + f_p2))
    arr_coeff_weights = np.zeros((1 + f_p1, 1 + f_p2))

    arr_x = np.zeros((k1, k2))
    arr_y = np.zeros((k1, k2))

    arr_x_x1 = np.zeros((k1, k2))
    arr_x_x2 = np.zeros((k1, k2))

    arr_y_x1 = np.zeros((k1, k2))
    arr_y_x2 = np.zeros((k1, k2))

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

            arr_coeffs_x[:, :] = global_arr_coeff_x[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                    pad2 + span_2 - f_p2:1 + pad2 + span_2]

            arr_coeffs_y[:, :] = global_arr_coeff_y[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                    pad2 + span_2 - f_p2:1 + pad2 + span_2]

            arr_coeff_weights[:, :] = global_arr_coeff_weights[pad1 + span_1 - f_p1:1 + pad1 + span_1,
                                                               pad2 + span_2 - f_p2:1 + pad2 + span_2]

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

                            coeff_weights = arr_coeff_weights[i_basis_1, i_basis_2]

                            arr_x[i_quad_1, i_quad_2] += mapping * coeff_x
                            arr_y[i_quad_1, i_quad_2] += mapping * coeff_y

                            arr_x_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_x
                            arr_x_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_x

                            arr_y_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_y
                            arr_y_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_y

                            arr_weights[i_quad_1, i_quad_2] += mapping * coeff_weights

                            arr_weights_x1[i_quad_1, i_quad_2] += mapping_x1 * coeff_weights
                            arr_weights_x2[i_quad_1, i_quad_2] += mapping_x2 * coeff_weights

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


# ==========================================================================
# Push forwards
# ==========================================================================

# ...
def pushforward_2d_l2(fields_to_push: 'float[:,:,:]', jac_dets: 'float[:,:]', pushed_fields: 'float[:,:,:]'):
    """

    Parameters
    ----------
    fields_to_push: ndarray
        Field values to push forward on the mapping
        This array as shape (n_x1, n_x2, n_f) where:
        * n_x1 is the number of points in direction 1 of the implicit grid.
        * n_x2 is the number of points in direction 2 of the implicit grid.
        * n_f is the number of fields to push-forward in the L2 space.

    jac_dets: ndarray
        Values of the Jacobian determinant of the Mapping

    pushed_fields: ndarray
        Push forwarded fields
    """
    for i_f in range(pushed_fields.shape[2]):
        pushed_fields[:, :, i_f] = fields_to_push[:, :, i_f] / jac_dets[:, :]


# ...
def pushforward_3d_l2(fields_to_push: 'float[:,:,:,:]', jac_dets: 'float[:,:,:]', pushed_fields: 'float[:,:,:,:]'):
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

    jac_dets: ndarray
        Values of the Jacobian determinant of the Mapping

    pushed_fields: ndarray
        Push forwarded fields
    """
    for i_f in range(pushed_fields.shape[3]):
        pushed_fields[:, :, :, i_f] = fields_to_push[:, :, :, i_f] / jac_dets[:, :, :]


# ...
def pushforward_2d_hcurl(fields_to_push: 'float[:,:,:,:]', inv_jac_mats: 'float[:,:,:,:]',
                         pushed_fields: 'float[:,:,:,:]'):
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
    for i_f in range(pushed_fields.shape[3]):
        pushed_fields[:, :, 0, i_f] = (+ inv_jac_mats[:, :, 0, 0] * fields_to_push[0, :, :, i_f]
                                       + inv_jac_mats[:, :, 1, 0] * fields_to_push[1, :, :, i_f])

        pushed_fields[:, :, 1, i_f] = (+ inv_jac_mats[:, :, 0, 1] * fields_to_push[0, :, :, i_f] 
                                       + inv_jac_mats[:, :, 1, 1] * fields_to_push[1, :, :, i_f])


# ...
def pushforward_3d_hcurl(fields_to_push: 'float[:,:,:,:,:]', inv_jac_mats: 'float[:,:,:,:,:]',
                         pushed_fields: 'float[:,:,:,:,:]'):
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
    for i_f in range(pushed_fields.shape[4]):
        x = fields_to_push[0, :, :, :, i_f]
        y = fields_to_push[1, :, :, :, i_f]
        z = fields_to_push[2, :, :, :, i_f]

        pushed_fields[:, :, :, 0, i_f] = + inv_jac_mats[:, :, :, 0, 0] * x \
                                         + inv_jac_mats[:, :, :, 1, 0] * y \
                                         + inv_jac_mats[:, :, :, 2, 0] * z

        pushed_fields[:, :, :, 1, i_f] = + inv_jac_mats[:, :, :, 0, 1] * x \
                                         + inv_jac_mats[:, :, :, 1, 1] * y \
                                         + inv_jac_mats[:, :, :, 2, 1] * z

        pushed_fields[:, :, :, 2, i_f] = + inv_jac_mats[:, :, :, 0, 2] * x \
                                         + inv_jac_mats[:, :, :, 1, 2] * y \
                                         + inv_jac_mats[:, :, :, 2, 2] * z


# ...
def pushforward_2d_hdiv(fields_to_push: 'float[:,:,:,:]', jac_mats: 'float[:,:,:,:]', pushed_fields: 'float[:,:,:,:]'):
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

    pushed_fields: ndarray
        Push forwarded fields
    """
    for i_f in range(pushed_fields.shape[3]):

        pushed_fields[:, :, 0, i_f] = (+ jac_mats[:, :, 0, 0] * fields_to_push[0, :, :, i_f] 
                                       + jac_mats[:, :, 0, 1] * fields_to_push[1, :, :, i_f])
                                    
        pushed_fields[:, :, 1, i_f] = (+ jac_mats[:, :, 1, 0] * fields_to_push[0, :, :, i_f] 
                                       + jac_mats[:, :, 1, 1] * fields_to_push[1, :, :, i_f])


# ...
def pushforward_3d_hdiv(fields_to_push: 'float[:,:,:,:,:]', jac_mats: 'float[:,:,:,:,:]',
                        pushed_fields: 'float[:,:,:,:,:]'):
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

    pushed_fields: ndarray
        Push forwarded fields
    """
    for i_f in range(pushed_fields.shape[4]):

        x = fields_to_push[0, :, :, :, i_f]
        y = fields_to_push[1, :, :, :, i_f]
        z = fields_to_push[2, :, :, :, i_f]

        pushed_fields[:, :, :, 0, i_f] = + jac_mats[:, :, :, 0, 0] * x \
                                         + jac_mats[:, :, :, 0, 1] * y \
                                         + jac_mats[:, :, :, 0, 2] * z

        pushed_fields[:, :, :, 1, i_f] = + jac_mats[:, :, :, 1, 0] * x \
                                         + jac_mats[:, :, :, 1, 1] * y \
                                         + jac_mats[:, :, :, 1, 2] * z

        pushed_fields[:, :, :, 2, i_f] = + jac_mats[:, :, :, 2, 0] * x \
                                         + jac_mats[:, :, :, 2, 1] * y \
                                         + jac_mats[:, :, :, 2, 2] * z
