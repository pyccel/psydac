def assemble_matrix_q_4_global(global_basis_v1_0_1 : "float64[:,:,:,:]", global_basis_v1_0_2 : "float64[:,:,:,:]", global_basis_v1_0_3 : "float64[:,:,:,:]", global_basis_v1_1_1 : "float64[:,:,:,:]", global_basis_v1_1_2 : "float64[:,:,:,:]", global_basis_v1_1_3 : "float64[:,:,:,:]", global_basis_v1_2_1 : "float64[:,:,:,:]", global_basis_v1_2_2 : "float64[:,:,:,:]", global_basis_v1_2_3 : "float64[:,:,:,:]", global_basis_u1_0_1 : "float64[:,:,:,:]", global_basis_u1_0_2 : "float64[:,:,:,:]", global_basis_u1_0_3 : "float64[:,:,:,:]", global_basis_u1_1_1 : "float64[:,:,:,:]", global_basis_u1_1_2 : "float64[:,:,:,:]", global_basis_u1_1_3 : "float64[:,:,:,:]", global_basis_u1_2_1 : "float64[:,:,:,:]", global_basis_u1_2_2 : "float64[:,:,:,:]", global_basis_u1_2_3 : "float64[:,:,:,:]", global_span_v1_0_1 : "int64[:]", global_span_v1_0_2 : "int64[:]", global_span_v1_0_3 : "int64[:]", global_span_v1_1_1 : "int64[:]", global_span_v1_1_2 : "int64[:]", global_span_v1_1_3 : "int64[:]", global_span_v1_2_1 : "int64[:]", global_span_v1_2_2 : "int64[:]", global_span_v1_2_3 : "int64[:]", global_x1 : "float64[:,:]", global_x2 : "float64[:,:]", global_x3 : "float64[:,:]", test_v1_0_p1 : "int64", test_v1_0_p2 : "int64", test_v1_0_p3 : "int64", test_v1_1_p1 : "int64", test_v1_1_p2 : "int64", test_v1_1_p3 : "int64", test_v1_2_p1 : "int64", test_v1_2_p2 : "int64", test_v1_2_p3 : "int64", trial_u1_0_p1 : "int64", trial_u1_0_p2 : "int64", trial_u1_0_p3 : "int64", trial_u1_1_p1 : "int64", trial_u1_1_p2 : "int64", trial_u1_1_p3 : "int64", trial_u1_2_p1 : "int64", trial_u1_2_p2 : "int64", trial_u1_2_p3 : "int64", n_element_1 : "int64", n_element_2 : "int64", n_element_3 : "int64", k1 : "int64", k2 : "int64", k3 : "int64", pad1 : "int64", pad2 : "int64", pad3 : "int64", g_mat_u1_0_v1_0 : "float64[:,:,:,:,:,:]", g_mat_u1_1_v1_0 : "float64[:,:,:,:,:,:]", g_mat_u1_2_v1_0 : "float64[:,:,:,:,:,:]", g_mat_u1_0_v1_1 : "float64[:,:,:,:,:,:]", g_mat_u1_1_v1_1 : "float64[:,:,:,:,:,:]", g_mat_u1_2_v1_1 : "float64[:,:,:,:,:,:]", g_mat_u1_0_v1_2 : "float64[:,:,:,:,:,:]", g_mat_u1_1_v1_2 : "float64[:,:,:,:,:,:]", g_mat_u1_2_v1_2 : "float64[:,:,:,:,:,:]", global_basis_h_0_1 : "float64[:,:,:,:]", global_basis_h_0_2 : "float64[:,:,:,:]", global_basis_h_0_3 : "float64[:,:,:,:]", global_basis_h_1_1 : "float64[:,:,:,:]", global_basis_h_1_2 : "float64[:,:,:,:]", global_basis_h_1_3 : "float64[:,:,:,:]", global_basis_h_2_1 : "float64[:,:,:,:]", global_basis_h_2_2 : "float64[:,:,:,:]", global_basis_h_2_3 : "float64[:,:,:,:]", global_span_h_0_1 : "int64[:]", global_span_h_0_2 : "int64[:]", global_span_h_0_3 : "int64[:]", global_span_h_1_1 : "int64[:]", global_span_h_1_2 : "int64[:]", global_span_h_1_3 : "int64[:]", global_span_h_2_1 : "int64[:]", global_span_h_2_2 : "int64[:]", global_span_h_2_3 : "int64[:]", test_h_0_p1 : "int64", test_h_0_p2 : "int64", test_h_0_p3 : "int64", test_h_1_p1 : "int64", test_h_1_p2 : "int64", test_h_1_p3 : "int64", test_h_2_p1 : "int64", test_h_2_p2 : "int64", test_h_2_p3 : "int64", pad_h_0_1 : "int64", pad_h_0_2 : "int64", pad_h_0_3 : "int64", pad_h_1_1 : "int64", pad_h_1_2 : "int64", pad_h_1_3 : "int64", pad_h_2_1 : "int64", pad_h_2_2 : "int64", pad_h_2_3 : "int64", global_arr_coeffs_h_0 : "float64[:,:,:]", global_arr_coeffs_h_1 : "float64[:,:,:]", global_arr_coeffs_h_2 : "float64[:,:,:]"):

    from numpy import array, zeros, zeros_like, floor
    arr_coeffs_h_0 = zeros((1 + test_h_0_p1, 1 + test_h_0_p2, 1 + test_h_0_p3), dtype='float64')
    arr_coeffs_h_1 = zeros((1 + test_h_1_p1, 1 + test_h_1_p2, 1 + test_h_1_p3), dtype='float64')
    arr_coeffs_h_2 = zeros((1 + test_h_2_p1, 1 + test_h_2_p2, 1 + test_h_2_p3), dtype='float64')


    a3_u1_0_v1_0 = zeros((n_element_3 + 4, 9), dtype='float64')
    a2_u1_0_v1_0 = zeros((n_element_2 + 4, 9, n_element_3 + 4, 9), dtype='float64')
    a1_u1_0_v1_0 = zeros((n_element_1 + 3, 7, n_element_2 + 4, 9, n_element_3 + 4, 9), dtype='float64')

    a3_u1_1_v1_0 = zeros((n_element_3 + 4, 9), dtype='float64')
    a2_u1_1_v1_0 = zeros((n_element_2 + 4, 9, n_element_3 + 4, 9), dtype='float64')
    a1_u1_1_v1_0 = zeros((n_element_1 + 3, 9, n_element_2 + 4, 9, n_element_3 + 4, 9), dtype='float64')

    a3_u1_2_v1_0 = zeros((n_element_3 + 4, 9), dtype='float64')
    a2_u1_2_v1_0 = zeros((n_element_2 + 4, 9, n_element_3 + 4, 9), dtype='float64')
    a1_u1_2_v1_0 = zeros((n_element_1 + 3, 9, n_element_2 + 4, 9, n_element_3 + 4, 9), dtype='float64')

    a3_u1_0_v1_1 = zeros((n_element_3 + 4, 9), dtype='float64')
    a2_u1_0_v1_1 = zeros((n_element_2 + 3, 9, n_element_3 + 4, 9), dtype='float64')
    a1_u1_0_v1_1 = zeros((n_element_1 + 4, 9, n_element_2 + 3, 9, n_element_3 + 4, 9), dtype='float64')

    a3_u1_1_v1_1 = zeros((n_element_3 + 4, 9), dtype='float64')
    a2_u1_1_v1_1 = zeros((n_element_2 + 3, 7, n_element_3 + 4, 9), dtype='float64')
    a1_u1_1_v1_1 = zeros((n_element_1 + 4, 9, n_element_2 + 3, 7, n_element_3 + 4, 9), dtype='float64')

    a3_u1_2_v1_1 = zeros((n_element_3 + 4, 9), dtype='float64')
    a2_u1_2_v1_1 = zeros((n_element_2 + 3, 9, n_element_3 + 4, 9), dtype='float64')
    a1_u1_2_v1_1 = zeros((n_element_1 + 4, 9, n_element_2 + 3, 9, n_element_3 + 4, 9), dtype='float64')

    a3_u1_0_v1_2 = zeros((n_element_3 + 3, 9), dtype='float64')
    a2_u1_0_v1_2 = zeros((n_element_2 + 4, 9, n_element_3 + 3, 9), dtype='float64')
    a1_u1_0_v1_2 = zeros((n_element_1 + 4, 9, n_element_2 + 4, 9, n_element_3 + 3, 9), dtype='float64')

    a3_u1_1_v1_2 = zeros((n_element_3 + 3, 9), dtype='float64')
    a2_u1_1_v1_2 = zeros((n_element_2 + 4, 9, n_element_3 + 3, 9), dtype='float64')
    a1_u1_1_v1_2 = zeros((n_element_1 + 4, 9, n_element_2 + 4, 9, n_element_3 + 3, 9), dtype='float64')

    a3_u1_2_v1_2 = zeros((n_element_3 + 3, 7), dtype='float64')
    a2_u1_2_v1_2 = zeros((n_element_2 + 4, 9, n_element_3 + 3, 7), dtype='float64')
    a1_u1_2_v1_2 = zeros((n_element_1 + 4, 9, n_element_2 + 4, 9, n_element_3 + 3, 7), dtype='float64')

    h_0 = 0.0
    h_1 = 0.0
    h_2 = 0.0

    for k_1 in range(0, n_element_1, 1):
        span_v1_0_1 = global_span_v1_0_1[k_1]
        span_v1_1_1 = global_span_v1_1_1[k_1]
        span_v1_2_1 = global_span_v1_2_1[k_1]
        span_h_0_1 = global_span_h_0_1[k_1]
        span_h_1_1 = global_span_h_1_1[k_1]
        span_h_2_1 = global_span_h_2_1[k_1]
        for q_1 in range(0, 5, 1):
            a2_u1_0_v1_0 *= 0.0
            for k_2 in range(0, n_element_2, 1):
                span_v1_0_2 = global_span_v1_0_2[k_2]
                span_v1_1_2 = global_span_v1_1_2[k_2]
                span_v1_2_2 = global_span_v1_2_2[k_2]
                span_h_0_2 = global_span_h_0_2[k_2]
                span_h_1_2 = global_span_h_1_2[k_2]
                span_h_2_2 = global_span_h_2_2[k_2]
                for q_2 in range(0, 5, 1):
                    a3_u1_0_v1_0 *= 0.0
                    for k_3 in range(0, n_element_3, 1):
                        span_v1_0_3 = global_span_v1_0_3[k_3]
                        span_v1_1_3 = global_span_v1_1_3[k_3]
                        span_v1_2_3 = global_span_v1_2_3[k_3]
                        span_h_0_3 = global_span_h_0_3[k_3]
                        span_h_1_3 = global_span_h_1_3[k_3]
                        span_h_2_3 = global_span_h_2_3[k_3]
                        for q_3 in range(0, 5, 1):
                            arr_coeffs_h_0[:,:,:] = global_arr_coeffs_h_0[pad_h_0_1 + span_h_0_1 - test_h_0_p1:1 + pad_h_0_1 + span_h_0_1,pad_h_0_2 + span_h_0_2 - test_h_0_p2:1 + pad_h_0_2 + span_h_0_2,pad_h_0_3 + span_h_0_3 - test_h_0_p3:1 + pad_h_0_3 + span_h_0_3]
                            arr_coeffs_h_1[:,:,:] = global_arr_coeffs_h_1[pad_h_1_1 + span_h_1_1 - test_h_1_p1:1 + pad_h_1_1 + span_h_1_1,pad_h_1_2 + span_h_1_2 - test_h_1_p2:1 + pad_h_1_2 + span_h_1_2,pad_h_1_3 + span_h_1_3 - test_h_1_p3:1 + pad_h_1_3 + span_h_1_3]
                            arr_coeffs_h_2[:,:,:] = global_arr_coeffs_h_2[pad_h_2_1 + span_h_2_1 - test_h_2_p1:1 + pad_h_2_1 + span_h_2_1,pad_h_2_2 + span_h_2_2 - test_h_2_p2:1 + pad_h_2_2 + span_h_2_2,pad_h_2_3 + span_h_2_3 - test_h_2_p3:1 + pad_h_2_3 + span_h_2_3]
                            
                            h_0 *= 0.0
                            for i_1 in range(0, 1 + test_h_0_p1, 1):
                                for i_2 in range(0, 1 + test_h_0_p2, 1):
                                    for i_3 in range(0, 1 + test_h_0_p3, 1):
                                        coeff_h_0 = arr_coeffs_h_0[i_1,i_2,i_3]
                                        h_0_1 = global_basis_h_0_1[k_1,i_1,0,q_1]
                                        h_0_2 = global_basis_h_0_2[k_2,i_2,0,q_2]
                                        h_0_3 = global_basis_h_0_3[k_3,i_3,0,q_3]
                                        h_0 += h_0_1*h_0_2*h_0_3*coeff_h_0
                            h_1 *= 0.0
                            for i_1 in range(0, 1 + test_h_1_p1, 1):
                                for i_2 in range(0, 1 + test_h_1_p2, 1):
                                    for i_3 in range(0, 1 + test_h_1_p3, 1):
                                        coeff_h_1 = arr_coeffs_h_1[i_1,i_2,i_3]
                                        h_1_1 = global_basis_h_1_1[k_1,i_1,0,q_1]
                                        h_1_2 = global_basis_h_1_2[k_2,i_2,0,q_2]
                                        h_1_3 = global_basis_h_1_3[k_3,i_3,0,q_3]
                                        h_1 += h_1_1*h_1_2*h_1_3*coeff_h_1
                            h_2 *= 0.0
                            for i_1 in range(0, 1 + test_h_2_p1, 1):
                                for i_2 in range(0, 1 + test_h_2_p2, 1):
                                    for i_3 in range(0, 1 + test_h_2_p3, 1):
                                        coeff_h_2 = arr_coeffs_h_2[i_1,i_2,i_3]
                                        h_2_1 = global_basis_h_2_1[k_1,i_1,0,q_1]
                                        h_2_2 = global_basis_h_2_2[k_2,i_2,0,q_2]
                                        h_2_3 = global_basis_h_2_3[k_3,i_3,0,q_3]
                                        h_2 += h_2_1*h_2_2*h_2_3*coeff_h_2
                            a4 = h_1 ** 2 + h_2 ** 2
                            for i_3 in range(0, 5, 1):
                                for j_3 in range(0, 5, 1):
                                    v1_0_3 = global_basis_v1_0_3[k_3, i_3, 0, q_3]
                                    u1_0_3 = global_basis_u1_0_3[k_3, j_3, 0, q_3]
                                    a3_u1_0_v1_0[span_v1_0_3 - 4 + i_3, 4 - i_3 + j_3] += v1_0_3 * u1_0_3 * a4
                    for i_2 in range(0, 5, 1):
                        for j_2 in range(0, 5, 1):
                            v1_0_2 = global_basis_v1_0_2[k_2, i_2, 0, q_2]
                            u1_0_2 = global_basis_u1_0_2[k_2, j_2, 0, q_2]
                            a2_u1_0_v1_0[span_v1_0_2 - 4 + i_2, 4 - i_2 + j_2, :, :] += v1_0_2 * u1_0_2 * a3_u1_0_v1_0[:, :]
            for i_1 in range(0, 4, 1):
                for j_1 in range(0, 4, 1):
                    v1_0_1 = global_basis_v1_0_1[k_1, i_1, 0, q_1]
                    u1_0_1 = global_basis_u1_0_1[k_1, j_1, 0, q_1]
                    a1_u1_0_v1_0[span_v1_0_1 - 3 + i_1, 3 - i_1 + j_1, :, :, :, :] += v1_0_1 * u1_0_1 * a2_u1_0_v1_0[:, :, :, :]
    for i_1 in range(0, n_element_1 + 3, 1):
        for i_2 in range(0, n_element_2 + 4, 1):
            for i_3 in range(0, n_element_3 + 4, 1):
                for j_1 in range(0, 5, 1):
                    for j_2 in range(0, 9, 1):
                        g_mat_u1_0_v1_0[pad1 + i_1, pad2 + i_2, pad3 + i_3, j_1, j_2, :] = a1_u1_0_v1_0[i_1, j_1, i_2, j_2, i_3, :]

    for k_1 in range(0, n_element_1, 1):
        span_v1_0_1 = global_span_v1_0_1[k_1]
        span_v1_1_1 = global_span_v1_1_1[k_1]
        span_v1_2_1 = global_span_v1_2_1[k_1]
        span_h_0_1 = global_span_h_0_1[k_1]
        span_h_1_1 = global_span_h_1_1[k_1]
        span_h_2_1 = global_span_h_2_1[k_1]
        for q_1 in range(0, 5, 1):
            a2_u1_1_v1_0 *= 0.0
            for k_2 in range(0, n_element_2, 1):
                span_v1_0_2 = global_span_v1_0_2[k_2]
                span_v1_1_2 = global_span_v1_1_2[k_2]
                span_v1_2_2 = global_span_v1_2_2[k_2]
                span_h_0_2 = global_span_h_0_2[k_2]
                span_h_1_2 = global_span_h_1_2[k_2]
                span_h_2_2 = global_span_h_2_2[k_2]
                for q_2 in range(0, 5, 1):
                    a3_u1_1_v1_0 *= 0.0
                    for k_3 in range(0, n_element_3, 1):
                        span_v1_0_3 = global_span_v1_0_3[k_3]
                        span_v1_1_3 = global_span_v1_1_3[k_3]
                        span_v1_2_3 = global_span_v1_2_3[k_3]
                        span_h_0_3 = global_span_h_0_3[k_3]
                        span_h_1_3 = global_span_h_1_3[k_3]
                        span_h_2_3 = global_span_h_2_3[k_3]
                        for q_3 in range(0, 5, 1):
                            arr_coeffs_h_0[:,:,:] = global_arr_coeffs_h_0[pad_h_0_1 + span_h_0_1 - test_h_0_p1:1 + pad_h_0_1 + span_h_0_1,pad_h_0_2 + span_h_0_2 - test_h_0_p2:1 + pad_h_0_2 + span_h_0_2,pad_h_0_3 + span_h_0_3 - test_h_0_p3:1 + pad_h_0_3 + span_h_0_3]
                            arr_coeffs_h_1[:,:,:] = global_arr_coeffs_h_1[pad_h_1_1 + span_h_1_1 - test_h_1_p1:1 + pad_h_1_1 + span_h_1_1,pad_h_1_2 + span_h_1_2 - test_h_1_p2:1 + pad_h_1_2 + span_h_1_2,pad_h_1_3 + span_h_1_3 - test_h_1_p3:1 + pad_h_1_3 + span_h_1_3]
                            arr_coeffs_h_2[:,:,:] = global_arr_coeffs_h_2[pad_h_2_1 + span_h_2_1 - test_h_2_p1:1 + pad_h_2_1 + span_h_2_1,pad_h_2_2 + span_h_2_2 - test_h_2_p2:1 + pad_h_2_2 + span_h_2_2,pad_h_2_3 + span_h_2_3 - test_h_2_p3:1 + pad_h_2_3 + span_h_2_3]
                            
                            h_0 *= 0.0
                            for i_1 in range(0, 1 + test_h_0_p1, 1):
                                for i_2 in range(0, 1 + test_h_0_p2, 1):
                                    for i_3 in range(0, 1 + test_h_0_p3, 1):
                                        coeff_h_0 = arr_coeffs_h_0[i_1,i_2,i_3]
                                        h_0_1 = global_basis_h_0_1[k_1,i_1,0,q_1]
                                        h_0_2 = global_basis_h_0_2[k_2,i_2,0,q_2]
                                        h_0_3 = global_basis_h_0_3[k_3,i_3,0,q_3]
                                        h_0 += h_0_1*h_0_2*h_0_3*coeff_h_0
                            h_1 *= 0.0
                            for i_1 in range(0, 1 + test_h_1_p1, 1):
                                for i_2 in range(0, 1 + test_h_1_p2, 1):
                                    for i_3 in range(0, 1 + test_h_1_p3, 1):
                                        coeff_h_1 = arr_coeffs_h_1[i_1,i_2,i_3]
                                        h_1_1 = global_basis_h_1_1[k_1,i_1,0,q_1]
                                        h_1_2 = global_basis_h_1_2[k_2,i_2,0,q_2]
                                        h_1_3 = global_basis_h_1_3[k_3,i_3,0,q_3]
                                        h_1 += h_1_1*h_1_2*h_1_3*coeff_h_1
                            h_2 *= 0.0
                            for i_1 in range(0, 1 + test_h_2_p1, 1):
                                for i_2 in range(0, 1 + test_h_2_p2, 1):
                                    for i_3 in range(0, 1 + test_h_2_p3, 1):
                                        coeff_h_2 = arr_coeffs_h_2[i_1,i_2,i_3]
                                        h_2_1 = global_basis_h_2_1[k_1,i_1,0,q_1]
                                        h_2_2 = global_basis_h_2_2[k_2,i_2,0,q_2]
                                        h_2_3 = global_basis_h_2_3[k_3,i_3,0,q_3]
                                        h_2 += h_2_1*h_2_2*h_2_3*coeff_h_2
                            a4 = - h_0 * h_1
                            for i_3 in range(0, 5, 1):
                                for j_3 in range(0, 5, 1):
                                    v1_0_3 = global_basis_v1_0_3[k_3, i_3, 0, q_3]
                                    u1_1_3 = global_basis_u1_1_3[k_3, j_3, 0, q_3]
                                    a3_u1_1_v1_0[span_v1_0_3 - 4 + i_3, 4 - i_3 + j_3] += v1_0_3 * u1_1_3 * a4
                    for i_2 in range(0, 5, 1):
                        for j_2 in range(0, 4, 1):
                            v1_0_2 = global_basis_v1_0_2[k_2, i_2, 0, q_2]
                            u1_1_2 = global_basis_u1_1_2[k_2, j_2, 0, q_2]
                            a2_u1_1_v1_0[span_v1_0_2 - 4 + i_2, 4 - i_2 + j_2, :, :] += v1_0_2 * u1_1_2 * a3_u1_1_v1_0[:, :]
            for i_1 in range(0, 4, 1):
                for j_1 in range(0, 5, 1):
                    v1_0_1 = global_basis_v1_0_1[k_1, i_1, 0, q_1]
                    u1_1_1 = global_basis_u1_1_1[k_1, j_1, 0, q_1]
                    a1_u1_1_v1_0[span_v1_0_1 - 3 + i_1, 4 - i_1 + j_1, :, :, :, :] += v1_0_1 * u1_1_1 * a2_u1_1_v1_0[:, :, :, :]
    for i_1 in range(0, n_element_1 + 3, 1):
        for i_2 in range(0, n_element_2 + 4, 1):
            for i_3 in range(0, n_element_3 + 4, 1):
                for j_1 in range(0, 9, 1):
                    for j_2 in range(0, 9, 1):
                        g_mat_u1_1_v1_0[pad1 + i_1, pad2 + i_2, pad3 + i_3, j_1, j_2, :] = a1_u1_1_v1_0[i_1, j_1, i_2, j_2, i_3, :]

    for k_1 in range(0, n_element_1, 1):
        span_v1_0_1 = global_span_v1_0_1[k_1]
        span_v1_1_1 = global_span_v1_1_1[k_1]
        span_v1_2_1 = global_span_v1_2_1[k_1]
        span_h_0_1 = global_span_h_0_1[k_1]
        span_h_1_1 = global_span_h_1_1[k_1]
        span_h_2_1 = global_span_h_2_1[k_1]
        for q_1 in range(0, 5, 1):
            a2_u1_2_v1_0 *= 0.0
            for k_2 in range(0, n_element_2, 1):
                span_v1_0_2 = global_span_v1_0_2[k_2]
                span_v1_1_2 = global_span_v1_1_2[k_2]
                span_v1_2_2 = global_span_v1_2_2[k_2]
                span_h_0_2 = global_span_h_0_2[k_2]
                span_h_1_2 = global_span_h_1_2[k_2]
                span_h_2_2 = global_span_h_2_2[k_2]
                for q_2 in range(0, 5, 1):
                    a3_u1_2_v1_0 *= 0.0
                    for k_3 in range(0, n_element_3, 1):
                        span_v1_0_3 = global_span_v1_0_3[k_3]
                        span_v1_1_3 = global_span_v1_1_3[k_3]
                        span_v1_2_3 = global_span_v1_2_3[k_3]
                        span_h_0_3 = global_span_h_0_3[k_3]
                        span_h_1_3 = global_span_h_1_3[k_3]
                        span_h_2_3 = global_span_h_2_3[k_3]
                        for q_3 in range(0, 5, 1):
                            arr_coeffs_h_0[:,:,:] = global_arr_coeffs_h_0[pad_h_0_1 + span_h_0_1 - test_h_0_p1:1 + pad_h_0_1 + span_h_0_1,pad_h_0_2 + span_h_0_2 - test_h_0_p2:1 + pad_h_0_2 + span_h_0_2,pad_h_0_3 + span_h_0_3 - test_h_0_p3:1 + pad_h_0_3 + span_h_0_3]
                            arr_coeffs_h_1[:,:,:] = global_arr_coeffs_h_1[pad_h_1_1 + span_h_1_1 - test_h_1_p1:1 + pad_h_1_1 + span_h_1_1,pad_h_1_2 + span_h_1_2 - test_h_1_p2:1 + pad_h_1_2 + span_h_1_2,pad_h_1_3 + span_h_1_3 - test_h_1_p3:1 + pad_h_1_3 + span_h_1_3]
                            arr_coeffs_h_2[:,:,:] = global_arr_coeffs_h_2[pad_h_2_1 + span_h_2_1 - test_h_2_p1:1 + pad_h_2_1 + span_h_2_1,pad_h_2_2 + span_h_2_2 - test_h_2_p2:1 + pad_h_2_2 + span_h_2_2,pad_h_2_3 + span_h_2_3 - test_h_2_p3:1 + pad_h_2_3 + span_h_2_3]
                            
                            h_0 *= 0.0
                            for i_1 in range(0, 1 + test_h_0_p1, 1):
                                for i_2 in range(0, 1 + test_h_0_p2, 1):
                                    for i_3 in range(0, 1 + test_h_0_p3, 1):
                                        coeff_h_0 = arr_coeffs_h_0[i_1,i_2,i_3]
                                        h_0_1 = global_basis_h_0_1[k_1,i_1,0,q_1]
                                        h_0_2 = global_basis_h_0_2[k_2,i_2,0,q_2]
                                        h_0_3 = global_basis_h_0_3[k_3,i_3,0,q_3]
                                        h_0 += h_0_1*h_0_2*h_0_3*coeff_h_0
                            h_1 *= 0.0
                            for i_1 in range(0, 1 + test_h_1_p1, 1):
                                for i_2 in range(0, 1 + test_h_1_p2, 1):
                                    for i_3 in range(0, 1 + test_h_1_p3, 1):
                                        coeff_h_1 = arr_coeffs_h_1[i_1,i_2,i_3]
                                        h_1_1 = global_basis_h_1_1[k_1,i_1,0,q_1]
                                        h_1_2 = global_basis_h_1_2[k_2,i_2,0,q_2]
                                        h_1_3 = global_basis_h_1_3[k_3,i_3,0,q_3]
                                        h_1 += h_1_1*h_1_2*h_1_3*coeff_h_1
                            h_2 *= 0.0
                            for i_1 in range(0, 1 + test_h_2_p1, 1):
                                for i_2 in range(0, 1 + test_h_2_p2, 1):
                                    for i_3 in range(0, 1 + test_h_2_p3, 1):
                                        coeff_h_2 = arr_coeffs_h_2[i_1,i_2,i_3]
                                        h_2_1 = global_basis_h_2_1[k_1,i_1,0,q_1]
                                        h_2_2 = global_basis_h_2_2[k_2,i_2,0,q_2]
                                        h_2_3 = global_basis_h_2_3[k_3,i_3,0,q_3]
                                        h_2 += h_2_1*h_2_2*h_2_3*coeff_h_2
                            a4 = - h_0 * h_2
                            for i_3 in range(0, 5, 1):
                                for j_3 in range(0, 4, 1):
                                    v1_0_3 = global_basis_v1_0_3[k_3, i_3, 0, q_3]
                                    u1_2_3 = global_basis_u1_2_3[k_3, j_3, 0, q_3]
                                    a3_u1_2_v1_0[span_v1_0_3 - 4 + i_3, 4 - i_3 + j_3] += v1_0_3 * u1_2_3 * a4
                    for i_2 in range(0, 5, 1):
                        for j_2 in range(0, 5, 1):
                            v1_0_2 = global_basis_v1_0_2[k_2, i_2, 0, q_2]
                            u1_2_2 = global_basis_u1_2_2[k_2, j_2, 0, q_2]
                            a2_u1_2_v1_0[span_v1_0_2 - 4 + i_2, 4 - i_2 + j_2, :, :] += v1_0_2 * u1_2_2 * a3_u1_2_v1_0[:, :]
            for i_1 in range(0, 4, 1):
                for j_1 in range(0, 5, 1):
                    v1_0_1 = global_basis_v1_0_1[k_1, i_1, 0, q_1]
                    u1_2_1 = global_basis_u1_2_1[k_1, j_1, 0, q_1]
                    a1_u1_2_v1_0[span_v1_0_1 - 3 + i_1, 4 - i_1 + j_1, :, :, :, :] += v1_0_1 * u1_2_1 * a2_u1_2_v1_0[:, :, :, :]
    for i_1 in range(0, n_element_1 + 3, 1):
        for i_2 in range(0, n_element_2 + 4, 1):
            for i_3 in range(0, n_element_3 + 4, 1):
                for j_1 in range(0, 9, 1):
                    for j_2 in range(0, 9, 1):
                        g_mat_u1_2_v1_0[pad1 + i_1, pad2 + i_2, pad3 + i_3, j_1, j_2, :] = a1_u1_2_v1_0[i_1, j_1, i_2, j_2, i_3, :]

    for k_1 in range(0, n_element_1, 1):
        span_v1_0_1 = global_span_v1_0_1[k_1]
        span_v1_1_1 = global_span_v1_1_1[k_1]
        span_v1_2_1 = global_span_v1_2_1[k_1]
        span_h_0_1 = global_span_h_0_1[k_1]
        span_h_1_1 = global_span_h_1_1[k_1]
        span_h_2_1 = global_span_h_2_1[k_1]
        for q_1 in range(0, 5, 1):
            a2_u1_0_v1_1 *= 0.0
            for k_2 in range(0, n_element_2, 1):
                span_v1_0_2 = global_span_v1_0_2[k_2]
                span_v1_1_2 = global_span_v1_1_2[k_2]
                span_v1_2_2 = global_span_v1_2_2[k_2]
                span_h_0_2 = global_span_h_0_2[k_2]
                span_h_1_2 = global_span_h_1_2[k_2]
                span_h_2_2 = global_span_h_2_2[k_2]
                for q_2 in range(0, 5, 1):
                    a3_u1_0_v1_1 *= 0.0
                    for k_3 in range(0, n_element_3, 1):
                        span_v1_0_3 = global_span_v1_0_3[k_3]
                        span_v1_1_3 = global_span_v1_1_3[k_3]
                        span_v1_2_3 = global_span_v1_2_3[k_3]
                        span_h_0_3 = global_span_h_0_3[k_3]
                        span_h_1_3 = global_span_h_1_3[k_3]
                        span_h_2_3 = global_span_h_2_3[k_3]
                        for q_3 in range(0, 5, 1):
                            arr_coeffs_h_0[:,:,:] = global_arr_coeffs_h_0[pad_h_0_1 + span_h_0_1 - test_h_0_p1:1 + pad_h_0_1 + span_h_0_1,pad_h_0_2 + span_h_0_2 - test_h_0_p2:1 + pad_h_0_2 + span_h_0_2,pad_h_0_3 + span_h_0_3 - test_h_0_p3:1 + pad_h_0_3 + span_h_0_3]
                            arr_coeffs_h_1[:,:,:] = global_arr_coeffs_h_1[pad_h_1_1 + span_h_1_1 - test_h_1_p1:1 + pad_h_1_1 + span_h_1_1,pad_h_1_2 + span_h_1_2 - test_h_1_p2:1 + pad_h_1_2 + span_h_1_2,pad_h_1_3 + span_h_1_3 - test_h_1_p3:1 + pad_h_1_3 + span_h_1_3]
                            arr_coeffs_h_2[:,:,:] = global_arr_coeffs_h_2[pad_h_2_1 + span_h_2_1 - test_h_2_p1:1 + pad_h_2_1 + span_h_2_1,pad_h_2_2 + span_h_2_2 - test_h_2_p2:1 + pad_h_2_2 + span_h_2_2,pad_h_2_3 + span_h_2_3 - test_h_2_p3:1 + pad_h_2_3 + span_h_2_3]
                            
                            h_0 *= 0.0
                            for i_1 in range(0, 1 + test_h_0_p1, 1):
                                for i_2 in range(0, 1 + test_h_0_p2, 1):
                                    for i_3 in range(0, 1 + test_h_0_p3, 1):
                                        coeff_h_0 = arr_coeffs_h_0[i_1,i_2,i_3]
                                        h_0_1 = global_basis_h_0_1[k_1,i_1,0,q_1]
                                        h_0_2 = global_basis_h_0_2[k_2,i_2,0,q_2]
                                        h_0_3 = global_basis_h_0_3[k_3,i_3,0,q_3]
                                        h_0 += h_0_1*h_0_2*h_0_3*coeff_h_0
                            h_1 *= 0.0
                            for i_1 in range(0, 1 + test_h_1_p1, 1):
                                for i_2 in range(0, 1 + test_h_1_p2, 1):
                                    for i_3 in range(0, 1 + test_h_1_p3, 1):
                                        coeff_h_1 = arr_coeffs_h_1[i_1,i_2,i_3]
                                        h_1_1 = global_basis_h_1_1[k_1,i_1,0,q_1]
                                        h_1_2 = global_basis_h_1_2[k_2,i_2,0,q_2]
                                        h_1_3 = global_basis_h_1_3[k_3,i_3,0,q_3]
                                        h_1 += h_1_1*h_1_2*h_1_3*coeff_h_1
                            h_2 *= 0.0
                            for i_1 in range(0, 1 + test_h_2_p1, 1):
                                for i_2 in range(0, 1 + test_h_2_p2, 1):
                                    for i_3 in range(0, 1 + test_h_2_p3, 1):
                                        coeff_h_2 = arr_coeffs_h_2[i_1,i_2,i_3]
                                        h_2_1 = global_basis_h_2_1[k_1,i_1,0,q_1]
                                        h_2_2 = global_basis_h_2_2[k_2,i_2,0,q_2]
                                        h_2_3 = global_basis_h_2_3[k_3,i_3,0,q_3]
                                        h_2 += h_2_1*h_2_2*h_2_3*coeff_h_2
                            a4 = - h_0 * h_1
                            for i_3 in range(0, 5, 1):
                                for j_3 in range(0, 5, 1):
                                    v1_1_3 = global_basis_v1_1_3[k_3, i_3, 0, q_3]
                                    u1_0_3 = global_basis_u1_0_3[k_3, j_3, 0, q_3]
                                    a3_u1_0_v1_1[span_v1_1_3 - 4 + i_3, 4 - i_3 + j_3] += v1_1_3 * u1_0_3 * a4
                    for i_2 in range(0, 4, 1):
                        for j_2 in range(0, 5, 1):
                            v1_1_2 = global_basis_v1_1_2[k_2, i_2, 0, q_2]
                            u1_0_2 = global_basis_u1_0_2[k_2, j_2, 0, q_2]
                            a2_u1_0_v1_1[span_v1_1_2 - 3 + i_2, 4 - i_2 + j_2, :, :] += v1_1_2 * u1_0_2 * a3_u1_0_v1_1[:, :]
            for i_1 in range(0, 5, 1):
                for j_1 in range(0, 4, 1):
                    v1_1_1 = global_basis_v1_1_1[k_1, i_1, 0, q_1]
                    u1_0_1 = global_basis_u1_0_1[k_1, j_1, 0, q_1]
                    a1_u1_0_v1_1[span_v1_1_1 - 4 + i_1, 4 - i_1 + j_1, :, :, :, :] += v1_1_1 * u1_0_1 * a2_u1_0_v1_1[:, :, :, :]
    for i_1 in range(0, n_element_1 + 4, 1):
        for i_2 in range(0, n_element_2 + 3, 1):
            for i_3 in range(0, n_element_3 + 4, 1):
                for j_1 in range(0, 9, 1):
                    for j_2 in range(0, 9, 1):
                        g_mat_u1_0_v1_1[pad1 + i_1, pad2 + i_2, pad3 + i_3, j_1, j_2, :] = a1_u1_0_v1_1[i_1, j_1, i_2, j_2, i_3, :]

    for k_1 in range(0, n_element_1, 1):
        span_v1_0_1 = global_span_v1_0_1[k_1]
        span_v1_1_1 = global_span_v1_1_1[k_1]
        span_v1_2_1 = global_span_v1_2_1[k_1]
        span_h_0_1 = global_span_h_0_1[k_1]
        span_h_1_1 = global_span_h_1_1[k_1]
        span_h_2_1 = global_span_h_2_1[k_1]
        for q_1 in range(0, 5, 1):
            a2_u1_1_v1_1 *= 0.0
            for k_2 in range(0, n_element_2, 1):
                span_v1_0_2 = global_span_v1_0_2[k_2]
                span_v1_1_2 = global_span_v1_1_2[k_2]
                span_v1_2_2 = global_span_v1_2_2[k_2]
                span_h_0_2 = global_span_h_0_2[k_2]
                span_h_1_2 = global_span_h_1_2[k_2]
                span_h_2_2 = global_span_h_2_2[k_2]
                for q_2 in range(0, 5, 1):
                    a3_u1_1_v1_1 *= 0.0
                    for k_3 in range(0, n_element_3, 1):
                        span_v1_0_3 = global_span_v1_0_3[k_3]
                        span_v1_1_3 = global_span_v1_1_3[k_3]
                        span_v1_2_3 = global_span_v1_2_3[k_3]
                        span_h_0_3 = global_span_h_0_3[k_3]
                        span_h_1_3 = global_span_h_1_3[k_3]
                        span_h_2_3 = global_span_h_2_3[k_3]
                        for q_3 in range(0, 5, 1):
                            arr_coeffs_h_0[:,:,:] = global_arr_coeffs_h_0[pad_h_0_1 + span_h_0_1 - test_h_0_p1:1 + pad_h_0_1 + span_h_0_1,pad_h_0_2 + span_h_0_2 - test_h_0_p2:1 + pad_h_0_2 + span_h_0_2,pad_h_0_3 + span_h_0_3 - test_h_0_p3:1 + pad_h_0_3 + span_h_0_3]
                            arr_coeffs_h_1[:,:,:] = global_arr_coeffs_h_1[pad_h_1_1 + span_h_1_1 - test_h_1_p1:1 + pad_h_1_1 + span_h_1_1,pad_h_1_2 + span_h_1_2 - test_h_1_p2:1 + pad_h_1_2 + span_h_1_2,pad_h_1_3 + span_h_1_3 - test_h_1_p3:1 + pad_h_1_3 + span_h_1_3]
                            arr_coeffs_h_2[:,:,:] = global_arr_coeffs_h_2[pad_h_2_1 + span_h_2_1 - test_h_2_p1:1 + pad_h_2_1 + span_h_2_1,pad_h_2_2 + span_h_2_2 - test_h_2_p2:1 + pad_h_2_2 + span_h_2_2,pad_h_2_3 + span_h_2_3 - test_h_2_p3:1 + pad_h_2_3 + span_h_2_3]
                            
                            h_0 *= 0.0
                            for i_1 in range(0, 1 + test_h_0_p1, 1):
                                for i_2 in range(0, 1 + test_h_0_p2, 1):
                                    for i_3 in range(0, 1 + test_h_0_p3, 1):
                                        coeff_h_0 = arr_coeffs_h_0[i_1,i_2,i_3]
                                        h_0_1 = global_basis_h_0_1[k_1,i_1,0,q_1]
                                        h_0_2 = global_basis_h_0_2[k_2,i_2,0,q_2]
                                        h_0_3 = global_basis_h_0_3[k_3,i_3,0,q_3]
                                        h_0 += h_0_1*h_0_2*h_0_3*coeff_h_0
                            h_1 *= 0.0
                            for i_1 in range(0, 1 + test_h_1_p1, 1):
                                for i_2 in range(0, 1 + test_h_1_p2, 1):
                                    for i_3 in range(0, 1 + test_h_1_p3, 1):
                                        coeff_h_1 = arr_coeffs_h_1[i_1,i_2,i_3]
                                        h_1_1 = global_basis_h_1_1[k_1,i_1,0,q_1]
                                        h_1_2 = global_basis_h_1_2[k_2,i_2,0,q_2]
                                        h_1_3 = global_basis_h_1_3[k_3,i_3,0,q_3]
                                        h_1 += h_1_1*h_1_2*h_1_3*coeff_h_1
                            h_2 *= 0.0
                            for i_1 in range(0, 1 + test_h_2_p1, 1):
                                for i_2 in range(0, 1 + test_h_2_p2, 1):
                                    for i_3 in range(0, 1 + test_h_2_p3, 1):
                                        coeff_h_2 = arr_coeffs_h_2[i_1,i_2,i_3]
                                        h_2_1 = global_basis_h_2_1[k_1,i_1,0,q_1]
                                        h_2_2 = global_basis_h_2_2[k_2,i_2,0,q_2]
                                        h_2_3 = global_basis_h_2_3[k_3,i_3,0,q_3]
                                        h_2 += h_2_1*h_2_2*h_2_3*coeff_h_2
                            a4 = h_0 ** 2 +  h_2 ** 2
                            for i_3 in range(0, 5, 1):
                                for j_3 in range(0, 5, 1):
                                    v1_1_3 = global_basis_v1_1_3[k_3, i_3, 0, q_3]
                                    u1_1_3 = global_basis_u1_1_3[k_3, j_3, 0, q_3]
                                    a3_u1_1_v1_1[span_v1_1_3 - 4 + i_3, 4 - i_3 + j_3] += v1_1_3 * u1_1_3 * a4
                    for i_2 in range(0, 4, 1):
                        for j_2 in range(0, 4, 1):
                            v1_1_2 = global_basis_v1_1_2[k_2, i_2, 0, q_2]
                            u1_1_2 = global_basis_u1_1_2[k_2, j_2, 0, q_2]
                            a2_u1_1_v1_1[span_v1_1_2 - 3 + i_2, 3 - i_2 + j_2, :, :] += v1_1_2 * u1_1_2 * a3_u1_1_v1_1[:, :]
            for i_1 in range(0, 5, 1):
                for j_1 in range(0, 5, 1):
                    v1_1_1 = global_basis_v1_1_1[k_1, i_1, 0, q_1]
                    u1_1_1 = global_basis_u1_1_1[k_1, j_1, 0, q_1]
                    a1_u1_1_v1_1[span_v1_1_1 - 4 + i_1, 4 - i_1 + j_1, :, :, :, :] += v1_1_1 * u1_1_1 * a2_u1_1_v1_1[:, :, :, :]
    for i_1 in range(0, n_element_1 + 4, 1):
        for i_2 in range(0, n_element_2 + 3, 1):
            for i_3 in range(0, n_element_3 + 4, 1):
                for j_1 in range(0, 9, 1):
                    for j_2 in range(0, 5, 1):
                        g_mat_u1_1_v1_1[pad1 + i_1, pad2 + i_2, pad3 + i_3, j_1, j_2, :] = a1_u1_1_v1_1[i_1, j_1, i_2, j_2, i_3, :]

    for k_1 in range(0, n_element_1, 1):
        span_v1_0_1 = global_span_v1_0_1[k_1]
        span_v1_1_1 = global_span_v1_1_1[k_1]
        span_v1_2_1 = global_span_v1_2_1[k_1]
        span_h_0_1 = global_span_h_0_1[k_1]
        span_h_1_1 = global_span_h_1_1[k_1]
        span_h_2_1 = global_span_h_2_1[k_1]
        for q_1 in range(0, 5, 1):
            a2_u1_2_v1_1 *= 0.0
            for k_2 in range(0, n_element_2, 1):
                span_v1_0_2 = global_span_v1_0_2[k_2]
                span_v1_1_2 = global_span_v1_1_2[k_2]
                span_v1_2_2 = global_span_v1_2_2[k_2]
                span_h_0_2 = global_span_h_0_2[k_2]
                span_h_1_2 = global_span_h_1_2[k_2]
                span_h_2_2 = global_span_h_2_2[k_2]
                for q_2 in range(0, 5, 1):
                    a3_u1_2_v1_1 *= 0.0
                    for k_3 in range(0, n_element_3, 1):
                        span_v1_0_3 = global_span_v1_0_3[k_3]
                        span_v1_1_3 = global_span_v1_1_3[k_3]
                        span_v1_2_3 = global_span_v1_2_3[k_3]
                        span_h_0_3 = global_span_h_0_3[k_3]
                        span_h_1_3 = global_span_h_1_3[k_3]
                        span_h_2_3 = global_span_h_2_3[k_3]
                        for q_3 in range(0, 5, 1):
                            arr_coeffs_h_0[:,:,:] = global_arr_coeffs_h_0[pad_h_0_1 + span_h_0_1 - test_h_0_p1:1 + pad_h_0_1 + span_h_0_1,pad_h_0_2 + span_h_0_2 - test_h_0_p2:1 + pad_h_0_2 + span_h_0_2,pad_h_0_3 + span_h_0_3 - test_h_0_p3:1 + pad_h_0_3 + span_h_0_3]
                            arr_coeffs_h_1[:,:,:] = global_arr_coeffs_h_1[pad_h_1_1 + span_h_1_1 - test_h_1_p1:1 + pad_h_1_1 + span_h_1_1,pad_h_1_2 + span_h_1_2 - test_h_1_p2:1 + pad_h_1_2 + span_h_1_2,pad_h_1_3 + span_h_1_3 - test_h_1_p3:1 + pad_h_1_3 + span_h_1_3]
                            arr_coeffs_h_2[:,:,:] = global_arr_coeffs_h_2[pad_h_2_1 + span_h_2_1 - test_h_2_p1:1 + pad_h_2_1 + span_h_2_1,pad_h_2_2 + span_h_2_2 - test_h_2_p2:1 + pad_h_2_2 + span_h_2_2,pad_h_2_3 + span_h_2_3 - test_h_2_p3:1 + pad_h_2_3 + span_h_2_3]
                            
                            h_0 *= 0.0
                            for i_1 in range(0, 1 + test_h_0_p1, 1):
                                for i_2 in range(0, 1 + test_h_0_p2, 1):
                                    for i_3 in range(0, 1 + test_h_0_p3, 1):
                                        coeff_h_0 = arr_coeffs_h_0[i_1,i_2,i_3]
                                        h_0_1 = global_basis_h_0_1[k_1,i_1,0,q_1]
                                        h_0_2 = global_basis_h_0_2[k_2,i_2,0,q_2]
                                        h_0_3 = global_basis_h_0_3[k_3,i_3,0,q_3]
                                        h_0 += h_0_1*h_0_2*h_0_3*coeff_h_0
                            h_1 *= 0.0
                            for i_1 in range(0, 1 + test_h_1_p1, 1):
                                for i_2 in range(0, 1 + test_h_1_p2, 1):
                                    for i_3 in range(0, 1 + test_h_1_p3, 1):
                                        coeff_h_1 = arr_coeffs_h_1[i_1,i_2,i_3]
                                        h_1_1 = global_basis_h_1_1[k_1,i_1,0,q_1]
                                        h_1_2 = global_basis_h_1_2[k_2,i_2,0,q_2]
                                        h_1_3 = global_basis_h_1_3[k_3,i_3,0,q_3]
                                        h_1 += h_1_1*h_1_2*h_1_3*coeff_h_1
                            h_2 *= 0.0
                            for i_1 in range(0, 1 + test_h_2_p1, 1):
                                for i_2 in range(0, 1 + test_h_2_p2, 1):
                                    for i_3 in range(0, 1 + test_h_2_p3, 1):
                                        coeff_h_2 = arr_coeffs_h_2[i_1,i_2,i_3]
                                        h_2_1 = global_basis_h_2_1[k_1,i_1,0,q_1]
                                        h_2_2 = global_basis_h_2_2[k_2,i_2,0,q_2]
                                        h_2_3 = global_basis_h_2_3[k_3,i_3,0,q_3]
                                        h_2 += h_2_1*h_2_2*h_2_3*coeff_h_2
                            a4 = - h_1 * h_2
                            for i_3 in range(0, 5, 1):
                                for j_3 in range(0, 4, 1):
                                    v1_1_3 = global_basis_v1_1_3[k_3, i_3, 0, q_3]
                                    u1_2_3 = global_basis_u1_2_3[k_3, j_3, 0, q_3]
                                    a3_u1_2_v1_1[span_v1_1_3 - 4 + i_3, 4 - i_3 + j_3] += v1_1_3 * u1_2_3 * a4
                    for i_2 in range(0, 4, 1):
                        for j_2 in range(0, 5, 1):
                            v1_1_2 = global_basis_v1_1_2[k_2, i_2, 0, q_2]
                            u1_2_2 = global_basis_u1_2_2[k_2, j_2, 0, q_2]
                            a2_u1_2_v1_1[span_v1_1_2 - 3 + i_2, 4 - i_2 + j_2, :, :] += v1_1_2 * u1_2_2 * a3_u1_2_v1_1[:, :]
            for i_1 in range(0, 5, 1):
                for j_1 in range(0, 5, 1):
                    v1_1_1 = global_basis_v1_1_1[k_1, i_1, 0, q_1]
                    u1_2_1 = global_basis_u1_2_1[k_1, j_1, 0, q_1]
                    a1_u1_2_v1_1[span_v1_1_1 - 4 + i_1, 4 - i_1 + j_1, :, :, :, :] += v1_1_1 * u1_2_1 * a2_u1_2_v1_1[:, :, :, :]
    for i_1 in range(0, n_element_1 + 4, 1):
        for i_2 in range(0, n_element_2 + 3, 1):
            for i_3 in range(0, n_element_3 + 4, 1):
                for j_1 in range(0, 9, 1):
                    for j_2 in range(0, 9, 1):
                        g_mat_u1_2_v1_1[pad1 + i_1, pad2 + i_2, pad3 + i_3, j_1, j_2, :] = a1_u1_2_v1_1[i_1, j_1, i_2, j_2, i_3, :]

    for k_1 in range(0, n_element_1, 1):
        span_v1_0_1 = global_span_v1_0_1[k_1]
        span_v1_1_1 = global_span_v1_1_1[k_1]
        span_v1_2_1 = global_span_v1_2_1[k_1]
        span_h_0_1 = global_span_h_0_1[k_1]
        span_h_1_1 = global_span_h_1_1[k_1]
        span_h_2_1 = global_span_h_2_1[k_1]
        for q_1 in range(0, 5, 1):
            a2_u1_0_v1_2 *= 0.0
            for k_2 in range(0, n_element_2, 1):
                span_v1_0_2 = global_span_v1_0_2[k_2]
                span_v1_1_2 = global_span_v1_1_2[k_2]
                span_v1_2_2 = global_span_v1_2_2[k_2]
                span_h_0_2 = global_span_h_0_2[k_2]
                span_h_1_2 = global_span_h_1_2[k_2]
                span_h_2_2 = global_span_h_2_2[k_2]
                for q_2 in range(0, 5, 1):
                    a3_u1_0_v1_2 *= 0.0
                    for k_3 in range(0, n_element_3, 1):
                        span_v1_0_3 = global_span_v1_0_3[k_3]
                        span_v1_1_3 = global_span_v1_1_3[k_3]
                        span_v1_2_3 = global_span_v1_2_3[k_3]
                        span_h_0_3 = global_span_h_0_3[k_3]
                        span_h_1_3 = global_span_h_1_3[k_3]
                        span_h_2_3 = global_span_h_2_3[k_3]
                        for q_3 in range(0, 5, 1):
                            arr_coeffs_h_0[:,:,:] = global_arr_coeffs_h_0[pad_h_0_1 + span_h_0_1 - test_h_0_p1:1 + pad_h_0_1 + span_h_0_1,pad_h_0_2 + span_h_0_2 - test_h_0_p2:1 + pad_h_0_2 + span_h_0_2,pad_h_0_3 + span_h_0_3 - test_h_0_p3:1 + pad_h_0_3 + span_h_0_3]
                            arr_coeffs_h_1[:,:,:] = global_arr_coeffs_h_1[pad_h_1_1 + span_h_1_1 - test_h_1_p1:1 + pad_h_1_1 + span_h_1_1,pad_h_1_2 + span_h_1_2 - test_h_1_p2:1 + pad_h_1_2 + span_h_1_2,pad_h_1_3 + span_h_1_3 - test_h_1_p3:1 + pad_h_1_3 + span_h_1_3]
                            arr_coeffs_h_2[:,:,:] = global_arr_coeffs_h_2[pad_h_2_1 + span_h_2_1 - test_h_2_p1:1 + pad_h_2_1 + span_h_2_1,pad_h_2_2 + span_h_2_2 - test_h_2_p2:1 + pad_h_2_2 + span_h_2_2,pad_h_2_3 + span_h_2_3 - test_h_2_p3:1 + pad_h_2_3 + span_h_2_3]
                            
                            h_0 *= 0.0
                            for i_1 in range(0, 1 + test_h_0_p1, 1):
                                for i_2 in range(0, 1 + test_h_0_p2, 1):
                                    for i_3 in range(0, 1 + test_h_0_p3, 1):
                                        coeff_h_0 = arr_coeffs_h_0[i_1,i_2,i_3]
                                        h_0_1 = global_basis_h_0_1[k_1,i_1,0,q_1]
                                        h_0_2 = global_basis_h_0_2[k_2,i_2,0,q_2]
                                        h_0_3 = global_basis_h_0_3[k_3,i_3,0,q_3]
                                        h_0 += h_0_1*h_0_2*h_0_3*coeff_h_0
                            h_1 *= 0.0
                            for i_1 in range(0, 1 + test_h_1_p1, 1):
                                for i_2 in range(0, 1 + test_h_1_p2, 1):
                                    for i_3 in range(0, 1 + test_h_1_p3, 1):
                                        coeff_h_1 = arr_coeffs_h_1[i_1,i_2,i_3]
                                        h_1_1 = global_basis_h_1_1[k_1,i_1,0,q_1]
                                        h_1_2 = global_basis_h_1_2[k_2,i_2,0,q_2]
                                        h_1_3 = global_basis_h_1_3[k_3,i_3,0,q_3]
                                        h_1 += h_1_1*h_1_2*h_1_3*coeff_h_1
                            h_2 *= 0.0
                            for i_1 in range(0, 1 + test_h_2_p1, 1):
                                for i_2 in range(0, 1 + test_h_2_p2, 1):
                                    for i_3 in range(0, 1 + test_h_2_p3, 1):
                                        coeff_h_2 = arr_coeffs_h_2[i_1,i_2,i_3]
                                        h_2_1 = global_basis_h_2_1[k_1,i_1,0,q_1]
                                        h_2_2 = global_basis_h_2_2[k_2,i_2,0,q_2]
                                        h_2_3 = global_basis_h_2_3[k_3,i_3,0,q_3]
                                        h_2 += h_2_1*h_2_2*h_2_3*coeff_h_2
                            a4 = - h_0 * h_2
                            for i_3 in range(0, 4, 1):
                                for j_3 in range(0, 5, 1):
                                    v1_2_3 = global_basis_v1_2_3[k_3, i_3, 0, q_3]
                                    u1_0_3 = global_basis_u1_0_3[k_3, j_3, 0, q_3]
                                    a3_u1_0_v1_2[span_v1_2_3 - 3 + i_3, 4 - i_3 + j_3] += v1_2_3 * u1_0_3 * a4
                    for i_2 in range(0, 5, 1):
                        for j_2 in range(0, 5, 1):
                            v1_2_2 = global_basis_v1_2_2[k_2, i_2, 0, q_2]
                            u1_0_2 = global_basis_u1_0_2[k_2, j_2, 0, q_2]
                            a2_u1_0_v1_2[span_v1_2_2 - 4 + i_2, 4 - i_2 + j_2, :, :] += v1_2_2 * u1_0_2 * a3_u1_0_v1_2[:, :]
            for i_1 in range(0, 5, 1):
                for j_1 in range(0, 4, 1):
                    v1_2_1 = global_basis_v1_2_1[k_1, i_1, 0, q_1]
                    u1_0_1 = global_basis_u1_0_1[k_1, j_1, 0, q_1]
                    a1_u1_0_v1_2[span_v1_2_1 - 4 + i_1, 4 - i_1 + j_1, :, :, :, :] += v1_2_1 * u1_0_1 * a2_u1_0_v1_2[:, :, :, :]
    for i_1 in range(0, n_element_1 + 4, 1):
        for i_2 in range(0, n_element_2 + 4, 1):
            for i_3 in range(0, n_element_3 + 3, 1):
                for j_1 in range(0, 9, 1):
                    for j_2 in range(0, 9, 1):
                        g_mat_u1_0_v1_2[pad1 + i_1, pad2 + i_2, pad3 + i_3, j_1, j_2, :] = a1_u1_0_v1_2[i_1, j_1, i_2, j_2, i_3, :]

    for k_1 in range(0, n_element_1, 1):
        span_v1_0_1 = global_span_v1_0_1[k_1]
        span_v1_1_1 = global_span_v1_1_1[k_1]
        span_v1_2_1 = global_span_v1_2_1[k_1]
        span_h_0_1 = global_span_h_0_1[k_1]
        span_h_1_1 = global_span_h_1_1[k_1]
        span_h_2_1 = global_span_h_2_1[k_1]
        for q_1 in range(0, 5, 1):
            a2_u1_1_v1_2 *= 0.0
            for k_2 in range(0, n_element_2, 1):
                span_v1_0_2 = global_span_v1_0_2[k_2]
                span_v1_1_2 = global_span_v1_1_2[k_2]
                span_v1_2_2 = global_span_v1_2_2[k_2]
                span_h_0_2 = global_span_h_0_2[k_2]
                span_h_1_2 = global_span_h_1_2[k_2]
                span_h_2_2 = global_span_h_2_2[k_2]
                for q_2 in range(0, 5, 1):
                    a3_u1_1_v1_2 *= 0.0
                    for k_3 in range(0, n_element_3, 1):
                        span_v1_0_3 = global_span_v1_0_3[k_3]
                        span_v1_1_3 = global_span_v1_1_3[k_3]
                        span_v1_2_3 = global_span_v1_2_3[k_3]
                        span_h_0_3 = global_span_h_0_3[k_3]
                        span_h_1_3 = global_span_h_1_3[k_3]
                        span_h_2_3 = global_span_h_2_3[k_3]
                        for q_3 in range(0, 5, 1):
                            arr_coeffs_h_0[:,:,:] = global_arr_coeffs_h_0[pad_h_0_1 + span_h_0_1 - test_h_0_p1:1 + pad_h_0_1 + span_h_0_1,pad_h_0_2 + span_h_0_2 - test_h_0_p2:1 + pad_h_0_2 + span_h_0_2,pad_h_0_3 + span_h_0_3 - test_h_0_p3:1 + pad_h_0_3 + span_h_0_3]
                            arr_coeffs_h_1[:,:,:] = global_arr_coeffs_h_1[pad_h_1_1 + span_h_1_1 - test_h_1_p1:1 + pad_h_1_1 + span_h_1_1,pad_h_1_2 + span_h_1_2 - test_h_1_p2:1 + pad_h_1_2 + span_h_1_2,pad_h_1_3 + span_h_1_3 - test_h_1_p3:1 + pad_h_1_3 + span_h_1_3]
                            arr_coeffs_h_2[:,:,:] = global_arr_coeffs_h_2[pad_h_2_1 + span_h_2_1 - test_h_2_p1:1 + pad_h_2_1 + span_h_2_1,pad_h_2_2 + span_h_2_2 - test_h_2_p2:1 + pad_h_2_2 + span_h_2_2,pad_h_2_3 + span_h_2_3 - test_h_2_p3:1 + pad_h_2_3 + span_h_2_3]
                            
                            h_0 *= 0.0
                            for i_1 in range(0, 1 + test_h_0_p1, 1):
                                for i_2 in range(0, 1 + test_h_0_p2, 1):
                                    for i_3 in range(0, 1 + test_h_0_p3, 1):
                                        coeff_h_0 = arr_coeffs_h_0[i_1,i_2,i_3]
                                        h_0_1 = global_basis_h_0_1[k_1,i_1,0,q_1]
                                        h_0_2 = global_basis_h_0_2[k_2,i_2,0,q_2]
                                        h_0_3 = global_basis_h_0_3[k_3,i_3,0,q_3]
                                        h_0 += h_0_1*h_0_2*h_0_3*coeff_h_0
                            h_1 *= 0.0
                            for i_1 in range(0, 1 + test_h_1_p1, 1):
                                for i_2 in range(0, 1 + test_h_1_p2, 1):
                                    for i_3 in range(0, 1 + test_h_1_p3, 1):
                                        coeff_h_1 = arr_coeffs_h_1[i_1,i_2,i_3]
                                        h_1_1 = global_basis_h_1_1[k_1,i_1,0,q_1]
                                        h_1_2 = global_basis_h_1_2[k_2,i_2,0,q_2]
                                        h_1_3 = global_basis_h_1_3[k_3,i_3,0,q_3]
                                        h_1 += h_1_1*h_1_2*h_1_3*coeff_h_1
                            h_2 *= 0.0
                            for i_1 in range(0, 1 + test_h_2_p1, 1):
                                for i_2 in range(0, 1 + test_h_2_p2, 1):
                                    for i_3 in range(0, 1 + test_h_2_p3, 1):
                                        coeff_h_2 = arr_coeffs_h_2[i_1,i_2,i_3]
                                        h_2_1 = global_basis_h_2_1[k_1,i_1,0,q_1]
                                        h_2_2 = global_basis_h_2_2[k_2,i_2,0,q_2]
                                        h_2_3 = global_basis_h_2_3[k_3,i_3,0,q_3]
                                        h_2 += h_2_1*h_2_2*h_2_3*coeff_h_2
                            a4 = - h_1 * h_2
                            for i_3 in range(0, 4, 1):
                                for j_3 in range(0, 5, 1):
                                    v1_2_3 = global_basis_v1_2_3[k_3, i_3, 0, q_3]
                                    u1_1_3 = global_basis_u1_1_3[k_3, j_3, 0, q_3]
                                    a3_u1_1_v1_2[span_v1_2_3 - 3 + i_3, 4 - i_3 + j_3] += v1_2_3 * u1_1_3 * a4
                    for i_2 in range(0, 5, 1):
                        for j_2 in range(0, 4, 1):
                            v1_2_2 = global_basis_v1_2_2[k_2, i_2, 0, q_2]
                            u1_1_2 = global_basis_u1_1_2[k_2, j_2, 0, q_2]
                            a2_u1_1_v1_2[span_v1_2_2 - 4 + i_2, 4 - i_2 + j_2, :, :] += v1_2_2 * u1_1_2 * a3_u1_1_v1_2[:, :]
            for i_1 in range(0, 5, 1):
                for j_1 in range(0, 5, 1):
                    v1_2_1 = global_basis_v1_2_1[k_1, i_1, 0, q_1]
                    u1_1_1 = global_basis_u1_1_1[k_1, j_1, 0, q_1]
                    a1_u1_1_v1_2[span_v1_2_1 - 4 + i_1, 4 - i_1 + j_1, :, :, :, :] += v1_2_1 * u1_1_1 * a2_u1_1_v1_2[:, :, :, :]
    for i_1 in range(0, n_element_1 + 4, 1):
        for i_2 in range(0, n_element_2 + 4, 1):
            for i_3 in range(0, n_element_3 + 3, 1):
                for j_1 in range(0, 9, 1):
                    for j_2 in range(0, 9, 1):
                        g_mat_u1_1_v1_2[pad1 + i_1, pad2 + i_2, pad3 + i_3, j_1, j_2, :] = a1_u1_1_v1_2[i_1, j_1, i_2, j_2, i_3, :]

    for k_1 in range(0, n_element_1, 1):
        span_v1_0_1 = global_span_v1_0_1[k_1]
        span_v1_1_1 = global_span_v1_1_1[k_1]
        span_v1_2_1 = global_span_v1_2_1[k_1]
        span_h_0_1 = global_span_h_0_1[k_1]
        span_h_1_1 = global_span_h_1_1[k_1]
        span_h_2_1 = global_span_h_2_1[k_1]
        for q_1 in range(0, 5, 1):
            a2_u1_2_v1_2 *= 0.0
            for k_2 in range(0, n_element_2, 1):
                span_v1_0_2 = global_span_v1_0_2[k_2]
                span_v1_1_2 = global_span_v1_1_2[k_2]
                span_v1_2_2 = global_span_v1_2_2[k_2]
                span_h_0_2 = global_span_h_0_2[k_2]
                span_h_1_2 = global_span_h_1_2[k_2]
                span_h_2_2 = global_span_h_2_2[k_2]
                for q_2 in range(0, 5, 1):
                    a3_u1_2_v1_2 *= 0.0
                    for k_3 in range(0, n_element_3, 1):
                        span_v1_0_3 = global_span_v1_0_3[k_3]
                        span_v1_1_3 = global_span_v1_1_3[k_3]
                        span_v1_2_3 = global_span_v1_2_3[k_3]
                        span_h_0_3 = global_span_h_0_3[k_3]
                        span_h_1_3 = global_span_h_1_3[k_3]
                        span_h_2_3 = global_span_h_2_3[k_3]
                        for q_3 in range(0, 5, 1):
                            arr_coeffs_h_0[:,:,:] = global_arr_coeffs_h_0[pad_h_0_1 + span_h_0_1 - test_h_0_p1:1 + pad_h_0_1 + span_h_0_1,pad_h_0_2 + span_h_0_2 - test_h_0_p2:1 + pad_h_0_2 + span_h_0_2,pad_h_0_3 + span_h_0_3 - test_h_0_p3:1 + pad_h_0_3 + span_h_0_3]
                            arr_coeffs_h_1[:,:,:] = global_arr_coeffs_h_1[pad_h_1_1 + span_h_1_1 - test_h_1_p1:1 + pad_h_1_1 + span_h_1_1,pad_h_1_2 + span_h_1_2 - test_h_1_p2:1 + pad_h_1_2 + span_h_1_2,pad_h_1_3 + span_h_1_3 - test_h_1_p3:1 + pad_h_1_3 + span_h_1_3]
                            arr_coeffs_h_2[:,:,:] = global_arr_coeffs_h_2[pad_h_2_1 + span_h_2_1 - test_h_2_p1:1 + pad_h_2_1 + span_h_2_1,pad_h_2_2 + span_h_2_2 - test_h_2_p2:1 + pad_h_2_2 + span_h_2_2,pad_h_2_3 + span_h_2_3 - test_h_2_p3:1 + pad_h_2_3 + span_h_2_3]
                            
                            h_0 *= 0.0
                            for i_1 in range(0, 1 + test_h_0_p1, 1):
                                for i_2 in range(0, 1 + test_h_0_p2, 1):
                                    for i_3 in range(0, 1 + test_h_0_p3, 1):
                                        coeff_h_0 = arr_coeffs_h_0[i_1,i_2,i_3]
                                        h_0_1 = global_basis_h_0_1[k_1,i_1,0,q_1]
                                        h_0_2 = global_basis_h_0_2[k_2,i_2,0,q_2]
                                        h_0_3 = global_basis_h_0_3[k_3,i_3,0,q_3]
                                        h_0 += h_0_1*h_0_2*h_0_3*coeff_h_0
                            h_1 *= 0.0
                            for i_1 in range(0, 1 + test_h_1_p1, 1):
                                for i_2 in range(0, 1 + test_h_1_p2, 1):
                                    for i_3 in range(0, 1 + test_h_1_p3, 1):
                                        coeff_h_1 = arr_coeffs_h_1[i_1,i_2,i_3]
                                        h_1_1 = global_basis_h_1_1[k_1,i_1,0,q_1]
                                        h_1_2 = global_basis_h_1_2[k_2,i_2,0,q_2]
                                        h_1_3 = global_basis_h_1_3[k_3,i_3,0,q_3]
                                        h_1 += h_1_1*h_1_2*h_1_3*coeff_h_1
                            h_2 *= 0.0
                            for i_1 in range(0, 1 + test_h_2_p1, 1):
                                for i_2 in range(0, 1 + test_h_2_p2, 1):
                                    for i_3 in range(0, 1 + test_h_2_p3, 1):
                                        coeff_h_2 = arr_coeffs_h_2[i_1,i_2,i_3]
                                        h_2_1 = global_basis_h_2_1[k_1,i_1,0,q_1]
                                        h_2_2 = global_basis_h_2_2[k_2,i_2,0,q_2]
                                        h_2_3 = global_basis_h_2_3[k_3,i_3,0,q_3]
                                        h_2 += h_2_1*h_2_2*h_2_3*coeff_h_2
                            a4 = h_0 ** 2 + h_1 ** 2
                            for i_3 in range(0, 4, 1):
                                for j_3 in range(0, 4, 1):
                                    v1_2_3 = global_basis_v1_2_3[k_3, i_3, 0, q_3]
                                    u1_2_3 = global_basis_u1_2_3[k_3, j_3, 0, q_3]
                                    a3_u1_2_v1_2[span_v1_2_3 - 3 + i_3, 3 - i_3 + j_3] += v1_2_3 * u1_2_3 * a4
                    for i_2 in range(0, 5, 1):
                        for j_2 in range(0, 5, 1):
                            v1_2_2 = global_basis_v1_2_2[k_2, i_2, 0, q_2]
                            u1_2_2 = global_basis_u1_2_2[k_2, j_2, 0, q_2]
                            a2_u1_2_v1_2[span_v1_2_2 - 4 + i_2, 4 - i_2 + j_2, :, :] += v1_2_2 * u1_2_2 * a3_u1_2_v1_2[:, :]
            for i_1 in range(0, 5, 1):
                for j_1 in range(0, 5, 1):
                    v1_2_1 = global_basis_v1_2_1[k_1, i_1, 0, q_1]
                    u1_2_1 = global_basis_u1_2_1[k_1, j_1, 0, q_1]
                    a1_u1_2_v1_2[span_v1_2_1 - 4 + i_1, 4 - i_1 + j_1, :, :, :, :] += v1_2_1 * u1_2_1 * a2_u1_2_v1_2[:, :, :, :]
    for i_1 in range(0, n_element_1 + 4, 1):
        for i_2 in range(0, n_element_2 + 4, 1):
            for i_3 in range(0, n_element_3 + 3, 1):
                for j_1 in range(0, 9, 1):
                    for j_2 in range(0, 9, 1):
                        g_mat_u1_2_v1_2[pad1 + i_1, pad2 + i_2, pad3 + i_3, j_1, j_2, :] = a1_u1_2_v1_2[i_1, j_1, i_2, j_2, i_3, :]

    return