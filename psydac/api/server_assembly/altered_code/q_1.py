def assemble_matrix_q_1(global_test_basis_v1_0_1 : "float64[:,:,:,:]", global_test_basis_v1_0_2 : "float64[:,:,:,:]", global_test_basis_v1_0_3 : "float64[:,:,:,:]", global_test_basis_v1_1_1 : "float64[:,:,:,:]", global_test_basis_v1_1_2 : "float64[:,:,:,:]", global_test_basis_v1_1_3 : "float64[:,:,:,:]", global_test_basis_v1_2_1 : "float64[:,:,:,:]", global_test_basis_v1_2_2 : "float64[:,:,:,:]", global_test_basis_v1_2_3 : "float64[:,:,:,:]", global_trial_basis_u1_0_1 : "float64[:,:,:,:]", global_trial_basis_u1_0_2 : "float64[:,:,:,:]", global_trial_basis_u1_0_3 : "float64[:,:,:,:]", global_trial_basis_u1_1_1 : "float64[:,:,:,:]", global_trial_basis_u1_1_2 : "float64[:,:,:,:]", global_trial_basis_u1_1_3 : "float64[:,:,:,:]", global_trial_basis_u1_2_1 : "float64[:,:,:,:]", global_trial_basis_u1_2_2 : "float64[:,:,:,:]", global_trial_basis_u1_2_3 : "float64[:,:,:,:]", global_span_v1_0_1 : "int64[:]", global_span_v1_0_2 : "int64[:]", global_span_v1_0_3 : "int64[:]", global_span_v1_1_1 : "int64[:]", global_span_v1_1_2 : "int64[:]", global_span_v1_1_3 : "int64[:]", global_span_v1_2_1 : "int64[:]", global_span_v1_2_2 : "int64[:]", global_span_v1_2_3 : "int64[:]", global_x1 : "float64[:,:]", global_x2 : "float64[:,:]", global_x3 : "float64[:,:]", test_v1_0_p1 : "int64", test_v1_0_p2 : "int64", test_v1_0_p3 : "int64", test_v1_1_p1 : "int64", test_v1_1_p2 : "int64", test_v1_1_p3 : "int64", test_v1_2_p1 : "int64", test_v1_2_p2 : "int64", test_v1_2_p3 : "int64", trial_u1_0_p1 : "int64", trial_u1_0_p2 : "int64", trial_u1_0_p3 : "int64", trial_u1_1_p1 : "int64", trial_u1_1_p2 : "int64", trial_u1_1_p3 : "int64", trial_u1_2_p1 : "int64", trial_u1_2_p2 : "int64", trial_u1_2_p3 : "int64", n_element_1 : "int64", n_element_2 : "int64", n_element_3 : "int64", k1 : "int64", k2 : "int64", k3 : "int64", pad1 : "int64", pad2 : "int64", pad3 : "int64", g_mat_u1_0_v1_0_g2tmrhvb : "float64[:,:,:,:,:,:]", g_mat_u1_1_v1_0_g2tmrhvb : "float64[:,:,:,:,:,:]", g_mat_u1_2_v1_0_g2tmrhvb : "float64[:,:,:,:,:,:]", g_mat_u1_0_v1_1_g2tmrhvb : "float64[:,:,:,:,:,:]", g_mat_u1_1_v1_1_g2tmrhvb : "float64[:,:,:,:,:,:]", g_mat_u1_2_v1_1_g2tmrhvb : "float64[:,:,:,:,:,:]", g_mat_u1_0_v1_2_g2tmrhvb : "float64[:,:,:,:,:,:]", g_mat_u1_1_v1_2_g2tmrhvb : "float64[:,:,:,:,:,:]", g_mat_u1_2_v1_2_g2tmrhvb : "float64[:,:,:,:,:,:]", global_test_basis_h_0_1 : "float64[:,:,:,:]", global_test_basis_h_0_2 : "float64[:,:,:,:]", global_test_basis_h_0_3 : "float64[:,:,:,:]", global_test_basis_h_1_1 : "float64[:,:,:,:]", global_test_basis_h_1_2 : "float64[:,:,:,:]", global_test_basis_h_1_3 : "float64[:,:,:,:]", global_test_basis_h_2_1 : "float64[:,:,:,:]", global_test_basis_h_2_2 : "float64[:,:,:,:]", global_test_basis_h_2_3 : "float64[:,:,:,:]", global_span_h_0_1 : "int64[:]", global_span_h_0_2 : "int64[:]", global_span_h_0_3 : "int64[:]", global_span_h_1_1 : "int64[:]", global_span_h_1_2 : "int64[:]", global_span_h_1_3 : "int64[:]", global_span_h_2_1 : "int64[:]", global_span_h_2_2 : "int64[:]", global_span_h_2_3 : "int64[:]", test_h_0_p1 : "int64", test_h_0_p2 : "int64", test_h_0_p3 : "int64", test_h_1_p1 : "int64", test_h_1_p2 : "int64", test_h_1_p3 : "int64", test_h_2_p1 : "int64", test_h_2_p2 : "int64", test_h_2_p3 : "int64", pad_h_0_1 : "int64", pad_h_0_2 : "int64", pad_h_0_3 : "int64", pad_h_1_1 : "int64", pad_h_1_2 : "int64", pad_h_1_3 : "int64", pad_h_2_1 : "int64", pad_h_2_2 : "int64", pad_h_2_3 : "int64", global_arr_coeffs_h_0 : "float64[:,:,:]", global_arr_coeffs_h_1 : "float64[:,:,:]", global_arr_coeffs_h_2 : "float64[:,:,:]"):

    from numpy import array, zeros, zeros_like, floor
    local_x1 = zeros_like(global_x1[0,:])
    local_x2 = zeros_like(global_x2[0,:])
    local_x3 = zeros_like(global_x3[0,:])
    arr_coeffs_h_0 = zeros((1 + test_h_0_p1, 1 + test_h_0_p2, 1 + test_h_0_p3), dtype='float64')
    arr_h_0 = zeros((2, 2, 2), dtype='float64')
    arr_coeffs_h_1 = zeros((1 + test_h_1_p1, 1 + test_h_1_p2, 1 + test_h_1_p3), dtype='float64')
    arr_h_1 = zeros((2, 2, 2), dtype='float64')
    arr_coeffs_h_2 = zeros((1 + test_h_2_p1, 1 + test_h_2_p2, 1 + test_h_2_p3), dtype='float64')
    arr_h_2 = zeros((2, 2, 2), dtype='float64')
    
    l_mat_u1_0_v1_0_g2tmrhvb = zeros((1, 2, 2, 1, 3, 3), dtype='float64')
    l_mat_u1_0_v1_1_g2tmrhvb = zeros((2, 1, 2, 3, 3, 3), dtype='float64')
    l_mat_u1_0_v1_2_g2tmrhvb = zeros((2, 2, 1, 3, 3, 3), dtype='float64')
    l_mat_u1_1_v1_0_g2tmrhvb = zeros((1, 2, 2, 3, 3, 3), dtype='float64')
    l_mat_u1_1_v1_1_g2tmrhvb = zeros((2, 1, 2, 3, 1, 3), dtype='float64')
    l_mat_u1_1_v1_2_g2tmrhvb = zeros((2, 2, 1, 3, 3, 3), dtype='float64')
    l_mat_u1_2_v1_0_g2tmrhvb = zeros((1, 2, 2, 3, 3, 3), dtype='float64')
    l_mat_u1_2_v1_1_g2tmrhvb = zeros((2, 1, 2, 3, 3, 3), dtype='float64')
    l_mat_u1_2_v1_2_g2tmrhvb = zeros((2, 2, 1, 3, 3, 1), dtype='float64')
    c1_1      = zeros((2, 2, 2, 2), dtype='float64')
    c1_2      = zeros((2, 2, 2, 2), dtype='float64')
    c1_3      = zeros((2, 1, 2, 2), dtype='float64')
    c1_4      = zeros((2, 2, 2, 2), dtype='float64')
    c1_5      = zeros((2, 2, 2, 2), dtype='float64')
    c1_6      = zeros((2, 1, 2, 2), dtype='float64')
    c1_7      = zeros((1, 2, 2, 2), dtype='float64')
    c1_8      = zeros((1, 2, 2, 2), dtype='float64')
    c1_9      = zeros((2, 2, 3, 3), dtype='float64')
    c2_1      = zeros((2, 2, 2, 2, 2), dtype='float64')
    c2_2      = zeros((2, 1, 2, 2, 2), dtype='float64')
    c2_3      = zeros((2, 2, 2, 1, 2), dtype='float64')
    c2_4      = zeros((1, 2, 2, 2, 2), dtype='float64')
    c2_5      = zeros((1, 1, 2, 2, 2), dtype='float64')
    c2_6      = zeros((1, 2, 2, 1, 2), dtype='float64')
    c2_7      = zeros((2, 2, 1, 2, 2), dtype='float64')
    c2_8      = zeros((2, 1, 1, 2, 2), dtype='float64')
    c2_9      = zeros((2, 2, 1, 1, 2), dtype='float64')
    for i_element_1 in range(0, n_element_1, 1):
        local_x1[:] = global_x1[i_element_1,:]
        span_v1_0_1 = global_span_v1_0_1[i_element_1]
        span_v1_1_1 = global_span_v1_1_1[i_element_1]
        span_v1_2_1 = global_span_v1_2_1[i_element_1]
        span_h_0_1 = global_span_h_0_1[i_element_1]
        span_h_1_1 = global_span_h_1_1[i_element_1]
        span_h_2_1 = global_span_h_2_1[i_element_1]
        for i_element_2 in range(0, n_element_2, 1):
            local_x2[:] = global_x2[i_element_2,:]
            span_v1_0_2 = global_span_v1_0_2[i_element_2]
            span_v1_1_2 = global_span_v1_1_2[i_element_2]
            span_v1_2_2 = global_span_v1_2_2[i_element_2]
            span_h_0_2 = global_span_h_0_2[i_element_2]
            span_h_1_2 = global_span_h_1_2[i_element_2]
            span_h_2_2 = global_span_h_2_2[i_element_2]
            for i_element_3 in range(0, n_element_3, 1):
                local_x3[:] = global_x3[i_element_3,:]
                span_v1_0_3 = global_span_v1_0_3[i_element_3]
                span_v1_1_3 = global_span_v1_1_3[i_element_3]
                span_v1_2_3 = global_span_v1_2_3[i_element_3]
                span_h_0_3 = global_span_h_0_3[i_element_3]
                span_h_1_3 = global_span_h_1_3[i_element_3]
                span_h_2_3 = global_span_h_2_3[i_element_3]
                arr_h_0[:,:,:] = 0.0
                arr_coeffs_h_0[:,:,:] = global_arr_coeffs_h_0[pad_h_0_1 + span_h_0_1 - test_h_0_p1:1 + pad_h_0_1 + span_h_0_1,pad_h_0_2 + span_h_0_2 - test_h_0_p2:1 + pad_h_0_2 + span_h_0_2,pad_h_0_3 + span_h_0_3 - test_h_0_p3:1 + pad_h_0_3 + span_h_0_3]
                for i_basis_1 in range(0, 1 + test_h_0_p1, 1):
                    for i_basis_2 in range(0, 1 + test_h_0_p2, 1):
                        for i_basis_3 in range(0, 1 + test_h_0_p3, 1):
                            coeff_h_0 = arr_coeffs_h_0[i_basis_1,i_basis_2,i_basis_3]
                            for i_quad_1 in range(0, 2, 1):
                                h_0_1 = global_test_basis_h_0_1[i_element_1,i_basis_1,0,i_quad_1]
                                for i_quad_2 in range(0, 2, 1):
                                    h_0_2 = global_test_basis_h_0_2[i_element_2,i_basis_2,0,i_quad_2]
                                    for i_quad_3 in range(0, 2, 1):
                                        h_0_3 = global_test_basis_h_0_3[i_element_3,i_basis_3,0,i_quad_3]
                                        h_field_0 = h_0_1*h_0_2*h_0_3
                                        arr_h_0[i_quad_1,i_quad_2,i_quad_3] += h_field_0*coeff_h_0



                arr_h_1[:,:,:] = 0.0
                arr_coeffs_h_1[:,:,:] = global_arr_coeffs_h_1[pad_h_1_1 + span_h_1_1 - test_h_1_p1:1 + pad_h_1_1 + span_h_1_1,pad_h_1_2 + span_h_1_2 - test_h_1_p2:1 + pad_h_1_2 + span_h_1_2,pad_h_1_3 + span_h_1_3 - test_h_1_p3:1 + pad_h_1_3 + span_h_1_3]
                for i_basis_1 in range(0, 1 + test_h_1_p1, 1):
                    for i_basis_2 in range(0, 1 + test_h_1_p2, 1):
                        for i_basis_3 in range(0, 1 + test_h_1_p3, 1):
                            coeff_h_1 = arr_coeffs_h_1[i_basis_1,i_basis_2,i_basis_3]
                            for i_quad_1 in range(0, 2, 1):
                                h_1_1 = global_test_basis_h_1_1[i_element_1,i_basis_1,0,i_quad_1]
                                for i_quad_2 in range(0, 2, 1):
                                    h_1_2 = global_test_basis_h_1_2[i_element_2,i_basis_2,0,i_quad_2]
                                    for i_quad_3 in range(0, 2, 1):
                                        h_1_3 = global_test_basis_h_1_3[i_element_3,i_basis_3,0,i_quad_3]
                                        h_field_1 = h_1_1*h_1_2*h_1_3
                                        arr_h_1[i_quad_1,i_quad_2,i_quad_3] += h_field_1*coeff_h_1

 

                arr_h_2[:,:,:] = 0.0
                arr_coeffs_h_2[:,:,:] = global_arr_coeffs_h_2[pad_h_2_1 + span_h_2_1 - test_h_2_p1:1 + pad_h_2_1 + span_h_2_1,pad_h_2_2 + span_h_2_2 - test_h_2_p2:1 + pad_h_2_2 + span_h_2_2,pad_h_2_3 + span_h_2_3 - test_h_2_p3:1 + pad_h_2_3 + span_h_2_3]
                for i_basis_1 in range(0, 1 + test_h_2_p1, 1):
                    for i_basis_2 in range(0, 1 + test_h_2_p2, 1):
                        for i_basis_3 in range(0, 1 + test_h_2_p3, 1):
                            coeff_h_2 = arr_coeffs_h_2[i_basis_1,i_basis_2,i_basis_3]
                            for i_quad_1 in range(0, 2, 1):
                                h_2_1 = global_test_basis_h_2_1[i_element_1,i_basis_1,0,i_quad_1]
                                for i_quad_2 in range(0, 2, 1):
                                    h_2_2 = global_test_basis_h_2_2[i_element_2,i_basis_2,0,i_quad_2]
                                    for i_quad_3 in range(0, 2, 1):
                                        h_2_3 = global_test_basis_h_2_3[i_element_3,i_basis_3,0,i_quad_3]
                                        h_field_2 = h_2_1*h_2_2*h_2_3
                                        arr_h_2[i_quad_1,i_quad_2,i_quad_3] += h_field_2*coeff_h_2



                for i_basis_3 in range(0, 2, 1):
                    for j_basis_3 in range(0, 2, 1):
                        for i_quad_1 in range(0, 2, 1):
                            for i_quad_2 in range(0, 2, 1):
                                c = 0.0
                                for i_quad_3 in range(0, 2, 1):
                                    v1_0_3 = global_test_basis_v1_0_3[i_element_3,i_basis_3,0,i_quad_3]
                                    u1_0_3 = global_trial_basis_u1_0_3[i_element_3,j_basis_3,0,i_quad_3]
                                    h_0 = arr_h_0[i_quad_1,i_quad_2,i_quad_3]
                                    h_1 = arr_h_1[i_quad_1,i_quad_2,i_quad_3]
                                    h_2 = arr_h_2[i_quad_1,i_quad_2,i_quad_3]
                                    temp_v1_0_u1_0_3 = u1_0_3*v1_0_3
                                    c += h_1**2*temp_v1_0_u1_0_3 + h_2**2*temp_v1_0_u1_0_3
                                c1_1[i_basis_3, j_basis_3, i_quad_1, i_quad_2] = c
                for i_basis_2 in range(0, 2, 1):
                    for j_basis_2 in range(0, 2, 1):
                        for i_basis_3 in range(0, 2, 1):
                            for j_basis_3 in range(0, 2, 1):
                                for i_quad_1 in range(0, 2, 1):
                                    c = 0.0
                                    for i_quad_2 in range(0, 2, 1):
                                        v1_0_2 = global_test_basis_v1_0_2[i_element_2,i_basis_2,0,i_quad_2]
                                        u1_0_2 = global_trial_basis_u1_0_2[i_element_2,j_basis_2,0,i_quad_2]
                                        c += v1_0_2 * u1_0_2 * c1_1[i_basis_3, j_basis_3, i_quad_1, i_quad_2]
                                    c2_1[i_basis_2, j_basis_2, i_basis_3, j_basis_3, i_quad_1] = c
                for i_basis_1 in range(0, 1, 1):
                    for j_basis_1 in range(0, 1, 1):
                        for i_basis_2 in range(0, 2, 1):
                            for j_basis_2 in range(0, 2, 1):
                                for i_basis_3 in range(0, 2, 1):
                                    for j_basis_3 in range(0, 2, 1):
                                        c = 0.0
                                        for i_quad_1 in range(0, 2, 1):
                                            v1_0_1 = global_test_basis_v1_0_1[i_element_1,i_basis_1,0,i_quad_1]
                                            u1_0_1 = global_trial_basis_u1_0_1[i_element_1,j_basis_1,0,i_quad_1]
                                            c += v1_0_1 * u1_0_1 * c2_1[i_basis_2, j_basis_2, i_basis_3, j_basis_3, i_quad_1]
                                        l_mat_u1_0_v1_0_g2tmrhvb[i_basis_1,i_basis_2,i_basis_3,0 - i_basis_1 + j_basis_1,1 - i_basis_2 + j_basis_2,1 - i_basis_3 + j_basis_3] = c

                for i_basis_3 in range(0, 2, 1):
                    for j_basis_3 in range(0, 2, 1):
                        for i_quad_1 in range(0, 2, 1):
                            for i_quad_2 in range(0, 2, 1):
                                c = 0.0
                                for i_quad_3 in range(0, 2, 1):
                                    v1_0_3 = global_test_basis_v1_0_3[i_element_3,i_basis_3,0,i_quad_3]
                                    u1_1_3 = global_trial_basis_u1_1_3[i_element_3,j_basis_3,0,i_quad_3]
                                    h_0 = arr_h_0[i_quad_1,i_quad_2,i_quad_3]
                                    h_1 = arr_h_1[i_quad_1,i_quad_2,i_quad_3]
                                    h_2 = arr_h_2[i_quad_1,i_quad_2,i_quad_3]
                                    c += -h_0*h_1*u1_1_3*v1_0_3
                                c1_2[i_basis_3, j_basis_3, i_quad_1, i_quad_2] = c
                for i_basis_2 in range(0, 2, 1):
                    for j_basis_2 in range(0, 1, 1):
                        for i_basis_3 in range(0, 2, 1):
                            for j_basis_3 in range(0, 2, 1):
                                for i_quad_1 in range(0, 2, 1):
                                    c = 0.0
                                    for i_quad_2 in range(0, 2, 1):
                                        v1_0_2 = global_test_basis_v1_0_2[i_element_2,i_basis_2,0,i_quad_2]
                                        u1_1_2 = global_trial_basis_u1_1_2[i_element_2,j_basis_2,0,i_quad_2]
                                        c += v1_0_2 * u1_1_2 * c1_2[i_basis_3, j_basis_3, i_quad_1, i_quad_2]
                                    c2_2[i_basis_2, j_basis_2, i_basis_3, j_basis_3, i_quad_1] = c
                for i_basis_1 in range(0, 1, 1):
                    for j_basis_1 in range(0, 2, 1):
                        for i_basis_2 in range(0, 2, 1):
                            for j_basis_2 in range(0, 1, 1):
                                for i_basis_3 in range(0, 2, 1):
                                    for j_basis_3 in range(0, 2, 1):
                                        c = 0.0
                                        for i_quad_1 in range(0, 2, 1):
                                            v1_0_1 = global_test_basis_v1_0_1[i_element_1,i_basis_1,0,i_quad_1]
                                            u1_1_1 = global_trial_basis_u1_1_1[i_element_1,j_basis_1,0,i_quad_1]
                                            c += v1_0_1 * u1_1_1 * c2_2[i_basis_2, j_basis_2, i_basis_3, j_basis_3, i_quad_1]
                                        l_mat_u1_1_v1_0_g2tmrhvb[i_basis_1,i_basis_2,i_basis_3,1 - i_basis_1 + j_basis_1,1 - i_basis_2 + j_basis_2,1 - i_basis_3 + j_basis_3] = c

                for i_basis_3 in range(0, 2, 1):
                    for j_basis_3 in range(0, 1, 1):
                        for i_quad_1 in range(0, 2, 1):
                            for i_quad_2 in range(0, 2, 1):
                                c = 0.0
                                for i_quad_3 in range(0, 2, 1):
                                    v1_0_3 = global_test_basis_v1_0_3[i_element_3,i_basis_3,0,i_quad_3]
                                    u1_2_3 = global_trial_basis_u1_2_3[i_element_3,j_basis_3,0,i_quad_3]
                                    h_0 = arr_h_0[i_quad_1,i_quad_2,i_quad_3]
                                    h_1 = arr_h_1[i_quad_1,i_quad_2,i_quad_3]
                                    h_2 = arr_h_2[i_quad_1,i_quad_2,i_quad_3]
                                    c += -h_0*h_2*u1_2_3*v1_0_3
                                c1_3[i_basis_3, j_basis_3, i_quad_1, i_quad_2] = c
                for i_basis_2 in range(0, 2, 1):
                    for j_basis_2 in range(0, 2, 1):
                        for i_basis_3 in range(0, 2, 1):
                            for j_basis_3 in range(0, 1, 1):
                                for i_quad_1 in range(0, 2, 1):
                                    c = 0.0
                                    for i_quad_2 in range(0, 2, 1):
                                        v1_0_2 = global_test_basis_v1_0_2[i_element_2,i_basis_2,0,i_quad_2]
                                        u1_2_2 = global_trial_basis_u1_2_2[i_element_2,j_basis_2,0,i_quad_2]
                                        c += v1_0_2 * u1_2_2 * c1_3[i_basis_3, j_basis_3, i_quad_1, i_quad_2]
                                    c2_3[i_basis_2, j_basis_2, i_basis_3, j_basis_3, i_quad_1] = c
                for i_basis_1 in range(0, 1, 1):
                    for j_basis_1 in range(0, 2, 1):
                        for i_basis_2 in range(0, 2, 1):
                            for j_basis_2 in range(0, 2, 1):
                                for i_basis_3 in range(0, 2, 1):
                                    for j_basis_3 in range(0, 1, 1):
                                        c = 0.0
                                        for i_quad_1 in range(0, 2, 1):
                                            v1_0_1 = global_test_basis_v1_0_1[i_element_1,i_basis_1,0,i_quad_1]
                                            u1_2_1 = global_trial_basis_u1_2_1[i_element_1,j_basis_1,0,i_quad_1]
                                            c += v1_0_1 * u1_2_1 * c2_3[i_basis_2, j_basis_2, i_basis_3, j_basis_3, i_quad_1]
                                        l_mat_u1_2_v1_0_g2tmrhvb[i_basis_1,i_basis_2,i_basis_3,1 - i_basis_1 + j_basis_1,1 - i_basis_2 + j_basis_2,1 - i_basis_3 + j_basis_3] = c

                for i_basis_3 in range(0, 2, 1):
                    for j_basis_3 in range(0, 2, 1):
                        for i_quad_1 in range(0, 2, 1):
                            for i_quad_2 in range(0, 2, 1):
                                c = 0.0
                                for i_quad_3 in range(0, 2, 1):
                                    v1_1_3 = global_test_basis_v1_1_3[i_element_3,i_basis_3,0,i_quad_3]
                                    u1_0_3 = global_trial_basis_u1_0_3[i_element_3,j_basis_3,0,i_quad_3]
                                    h_0 = arr_h_0[i_quad_1,i_quad_2,i_quad_3]
                                    h_1 = arr_h_1[i_quad_1,i_quad_2,i_quad_3]
                                    h_2 = arr_h_2[i_quad_1,i_quad_2,i_quad_3]
                                    c += -h_0*h_1*u1_0_3*v1_1_3
                                c1_4[i_basis_3, j_basis_3, i_quad_1, i_quad_2] = c
                for i_basis_2 in range(0, 1, 1):
                    for j_basis_2 in range(0, 2, 1):
                        for i_basis_3 in range(0, 2, 1):
                            for j_basis_3 in range(0, 2, 1):
                                for i_quad_1 in range(0, 2, 1):
                                    c = 0.0
                                    for i_quad_2 in range(0, 2, 1):
                                        v1_1_2 = global_test_basis_v1_1_2[i_element_2,i_basis_2,0,i_quad_2]
                                        u1_0_2 = global_trial_basis_u1_0_2[i_element_2,j_basis_2,0,i_quad_2]
                                        c += v1_1_2 * u1_0_2 * c1_4[i_basis_3, j_basis_3, i_quad_1, i_quad_2]
                                    c2_4[i_basis_2, j_basis_2, i_basis_3, j_basis_3, i_quad_1] = c
                for i_basis_1 in range(0, 2, 1):
                    for j_basis_1 in range(0, 1, 1):
                        for i_basis_2 in range(0, 1, 1):
                            for j_basis_2 in range(0, 2, 1):
                                for i_basis_3 in range(0, 2, 1):
                                    for j_basis_3 in range(0, 2, 1):
                                        c = 0.0
                                        for i_quad_1 in range(0, 2, 1):
                                            v1_1_1 = global_test_basis_v1_1_1[i_element_1,i_basis_1,0,i_quad_1]
                                            u1_0_1 = global_trial_basis_u1_0_1[i_element_1,j_basis_1,0,i_quad_1]
                                            c += v1_1_1 * u1_0_1 * c2_4[i_basis_2, j_basis_2, i_basis_3, j_basis_3, i_quad_1]
                                        l_mat_u1_0_v1_1_g2tmrhvb[i_basis_1,i_basis_2,i_basis_3,1 - i_basis_1 + j_basis_1,1 - i_basis_2 + j_basis_2,1 - i_basis_3 + j_basis_3] = c

                for i_basis_3 in range(0, 2, 1):
                    for j_basis_3 in range(0, 2, 1):
                        for i_quad_1 in range(0, 2, 1):
                            for i_quad_2 in range(0, 2, 1):
                                c = 0.0
                                for i_quad_3 in range(0, 2, 1):
                                    v1_1_3 = global_test_basis_v1_1_3[i_element_3,i_basis_3,0,i_quad_3]
                                    u1_1_3 = global_trial_basis_u1_1_3[i_element_3,j_basis_3,0,i_quad_3]
                                    h_0 = arr_h_0[i_quad_1,i_quad_2,i_quad_3]
                                    h_1 = arr_h_1[i_quad_1,i_quad_2,i_quad_3]
                                    h_2 = arr_h_2[i_quad_1,i_quad_2,i_quad_3]
                                    temp_v1_1_u1_1_3 = u1_1_3*v1_1_3
                                    c += h_0**2*temp_v1_1_u1_1_3 + h_2**2*temp_v1_1_u1_1_3
                                c1_5[i_basis_3, j_basis_3, i_quad_1, i_quad_2] = c
                for i_basis_2 in range(0, 1, 1):
                    for j_basis_2 in range(0, 1, 1):
                        for i_basis_3 in range(0, 2, 1):
                            for j_basis_3 in range(0, 2, 1):
                                for i_quad_1 in range(0, 2, 1):
                                    c = 0.0
                                    for i_quad_2 in range(0, 2, 1):
                                        v1_1_2 = global_test_basis_v1_1_2[i_element_2,i_basis_2,0,i_quad_2]
                                        u1_1_2 = global_trial_basis_u1_1_2[i_element_2,j_basis_2,0,i_quad_2]
                                        c += v1_1_2 * u1_1_2 * c1_5[i_basis_3, j_basis_3, i_quad_1, i_quad_2]
                                    c2_5[i_basis_2, j_basis_2, i_basis_3, j_basis_3, i_quad_1] = c
                for i_basis_1 in range(0, 2, 1):
                    for j_basis_1 in range(0, 2, 1):
                        for i_basis_2 in range(0, 1, 1):
                            for j_basis_2 in range(0, 1, 1):
                                for i_basis_3 in range(0, 2, 1):
                                    for j_basis_3 in range(0, 2, 1):
                                        c = 0.0
                                        for i_quad_1 in range(0, 2, 1):
                                            v1_1_1 = global_test_basis_v1_1_1[i_element_1,i_basis_1,0,i_quad_1]
                                            u1_1_1 = global_trial_basis_u1_1_1[i_element_1,j_basis_1,0,i_quad_1]
                                            c += v1_1_1 * u1_1_1 * c2_5[i_basis_2, j_basis_2, i_basis_3, j_basis_3, i_quad_1]
                                        l_mat_u1_1_v1_1_g2tmrhvb[i_basis_1,i_basis_2,i_basis_3,1 - i_basis_1 + j_basis_1,0 - i_basis_2 + j_basis_2,1 - i_basis_3 + j_basis_3] = c

                for i_basis_3 in range(0, 2, 1):
                    for j_basis_3 in range(0, 1, 1):
                        for i_quad_1 in range(0, 2, 1):
                            for i_quad_2 in range(0, 2, 1):
                                c = 0.0
                                for i_quad_3 in range(0, 2, 1):
                                    v1_1_3 = global_test_basis_v1_1_3[i_element_3,i_basis_3,0,i_quad_3]
                                    u1_2_3 = global_trial_basis_u1_2_3[i_element_3,j_basis_3,0,i_quad_3]
                                    h_0 = arr_h_0[i_quad_1,i_quad_2,i_quad_3]
                                    h_1 = arr_h_1[i_quad_1,i_quad_2,i_quad_3]
                                    h_2 = arr_h_2[i_quad_1,i_quad_2,i_quad_3]
                                    c += -h_1*h_2*u1_2_3*v1_1_3
                                c1_6[i_basis_3, j_basis_3, i_quad_1, i_quad_2] = c
                for i_basis_2 in range(0, 1, 1):
                    for j_basis_2 in range(0, 2, 1):
                        for i_basis_3 in range(0, 2, 1):
                            for j_basis_3 in range(0, 1, 1):
                                for i_quad_1 in range(0, 2, 1):
                                    c = 0.0
                                    for i_quad_2 in range(0, 2, 1):
                                        v1_1_2 = global_test_basis_v1_1_2[i_element_2,i_basis_2,0,i_quad_2]
                                        u1_2_2 = global_trial_basis_u1_2_2[i_element_2,j_basis_2,0,i_quad_2]
                                        c += v1_1_2 * u1_2_2 * c1_6[i_basis_3, j_basis_3, i_quad_1, i_quad_2]
                                    c2_6[i_basis_2, j_basis_2, i_basis_3, j_basis_3, i_quad_1] = c
                for i_basis_1 in range(0, 2, 1):
                    for j_basis_1 in range(0, 2, 1):
                        for i_basis_2 in range(0, 1, 1):
                            for j_basis_2 in range(0, 2, 1):
                                for i_basis_3 in range(0, 2, 1):
                                    for j_basis_3 in range(0, 1, 1):
                                        c = 0.0
                                        for i_quad_1 in range(0, 2, 1):
                                            v1_1_1 = global_test_basis_v1_1_1[i_element_1,i_basis_1,0,i_quad_1]
                                            u1_2_1 = global_trial_basis_u1_2_1[i_element_1,j_basis_1,0,i_quad_1]
                                            c += v1_1_1 * u1_2_1 * c2_6[i_basis_2, j_basis_2, i_basis_3, j_basis_3, i_quad_1]
                                        l_mat_u1_2_v1_1_g2tmrhvb[i_basis_1,i_basis_2,i_basis_3,1 - i_basis_1 + j_basis_1,1 - i_basis_2 + j_basis_2,1 - i_basis_3 + j_basis_3] = c

                for i_basis_3 in range(0, 1, 1):
                    for j_basis_3 in range(0, 2, 1):
                        for i_quad_1 in range(0, 2, 1):
                            for i_quad_2 in range(0, 2, 1):
                                c = 0.0
                                for i_quad_3 in range(0, 2, 1):
                                    v1_2_3 = global_test_basis_v1_2_3[i_element_3,i_basis_3,0,i_quad_3]
                                    u1_0_3 = global_trial_basis_u1_0_3[i_element_3,j_basis_3,0,i_quad_3]
                                    h_0 = arr_h_0[i_quad_1,i_quad_2,i_quad_3]
                                    h_1 = arr_h_1[i_quad_1,i_quad_2,i_quad_3]
                                    h_2 = arr_h_2[i_quad_1,i_quad_2,i_quad_3]
                                    c += -h_0*h_2*u1_0_3*v1_2_3
                                c1_7[i_basis_3, j_basis_3, i_quad_1, i_quad_2] = c
                for i_basis_2 in range(0, 2, 1):
                    for j_basis_2 in range(0, 2, 1):
                        for i_basis_3 in range(0, 1, 1):
                            for j_basis_3 in range(0, 2, 1):
                                for i_quad_1 in range(0, 2, 1):
                                    c = 0.0
                                    for i_quad_2 in range(0, 2, 1):
                                        v1_2_2 = global_test_basis_v1_2_2[i_element_2,i_basis_2,0,i_quad_2]
                                        u1_0_2 = global_trial_basis_u1_0_2[i_element_2,j_basis_2,0,i_quad_2]
                                        c += v1_2_2 * u1_0_2 * c1_7[i_basis_3, j_basis_3, i_quad_1, i_quad_2]
                                    c2_7[i_basis_2, j_basis_2, i_basis_3, j_basis_3, i_quad_1] = c
                for i_basis_1 in range(0, 2, 1):
                    for j_basis_1 in range(0, 1, 1):
                        for i_basis_2 in range(0, 2, 1):
                            for j_basis_2 in range(0, 2, 1):
                                for i_basis_3 in range(0, 1, 1):
                                    for j_basis_3 in range(0, 2, 1):
                                        c = 0.0
                                        for i_quad_1 in range(0, 2, 1):
                                            v1_2_1 = global_test_basis_v1_2_1[i_element_1,i_basis_1,0,i_quad_1]
                                            u1_0_1 = global_trial_basis_u1_0_1[i_element_1,j_basis_1,0,i_quad_1]
                                            c += v1_2_1 * u1_0_1 * c2_7[i_basis_2, j_basis_2, i_basis_3, j_basis_3, i_quad_1]
                                        l_mat_u1_0_v1_2_g2tmrhvb[i_basis_1,i_basis_2,i_basis_3,1 - i_basis_1 + j_basis_1,1 - i_basis_2 + j_basis_2,1 - i_basis_3 + j_basis_3] = c

                for i_basis_3 in range(0, 1, 1):
                    for j_basis_3 in range(0, 2, 1):
                        for i_quad_1 in range(0, 2, 1):
                            for i_quad_2 in range(0, 2, 1):
                                c = 0.0
                                for i_quad_3 in range(0, 2, 1):
                                    v1_2_3 = global_test_basis_v1_2_3[i_element_3,i_basis_3,0,i_quad_3]
                                    u1_1_3 = global_trial_basis_u1_1_3[i_element_3,j_basis_3,0,i_quad_3]
                                    h_0 = arr_h_0[i_quad_1,i_quad_2,i_quad_3]
                                    h_1 = arr_h_1[i_quad_1,i_quad_2,i_quad_3]
                                    h_2 = arr_h_2[i_quad_1,i_quad_2,i_quad_3]
                                    c += -h_1*h_2*u1_1_3*v1_2_3
                                c1_8[i_basis_3, j_basis_3, i_quad_1, i_quad_2] = c
                for i_basis_2 in range(0, 2, 1):
                    for j_basis_2 in range(0, 1, 1):
                        for i_basis_3 in range(0, 1, 1):
                            for j_basis_3 in range(0, 2, 1):
                                for i_quad_1 in range(0, 2, 1):
                                    c = 0.0
                                    for i_quad_2 in range(0, 2, 1):
                                        v1_2_2 = global_test_basis_v1_2_2[i_element_2,i_basis_2,0,i_quad_2]
                                        u1_1_2 = global_trial_basis_u1_1_2[i_element_2,j_basis_2,0,i_quad_2]
                                        c += v1_2_2 * u1_1_2 * c1_8[i_basis_3, j_basis_3, i_quad_1, i_quad_2]
                                    c2_8[i_basis_2, j_basis_2, i_basis_3, j_basis_3, i_quad_1] = c
                for i_basis_1 in range(0, 2, 1):
                    for j_basis_1 in range(0, 2, 1):
                        for i_basis_2 in range(0, 2, 1):
                            for j_basis_2 in range(0, 1, 1):
                                for i_basis_3 in range(0, 1, 1):
                                    for j_basis_3 in range(0, 2, 1):
                                        c = 0.0
                                        for i_quad_1 in range(0, 2, 1):
                                            v1_2_1 = global_test_basis_v1_2_1[i_element_1,i_basis_1,0,i_quad_1]
                                            u1_1_1 = global_trial_basis_u1_1_1[i_element_1,j_basis_1,0,i_quad_1]
                                            c += v1_2_1 * u1_1_1 * c2_8[i_basis_2, j_basis_2, i_basis_3, j_basis_3, i_quad_1]
                                        l_mat_u1_1_v1_2_g2tmrhvb[i_basis_1,i_basis_2,i_basis_3,1 - i_basis_1 + j_basis_1,1 - i_basis_2 + j_basis_2,1 - i_basis_3 + j_basis_3] = c

                for i_basis_3 in range(0, 1, 1):
                    for j_basis_3 in range(0, 1, 1):
                        for i_quad_1 in range(0, 2, 1):
                            for i_quad_2 in range(0, 2, 1):
                                c = 0.0
                                for i_quad_3 in range(0, 2, 1):
                                    v1_2_3 = global_test_basis_v1_2_3[i_element_3,i_basis_3,0,i_quad_3]
                                    u1_2_3 = global_trial_basis_u1_2_3[i_element_3,j_basis_3,0,i_quad_3]
                                    h_0 = arr_h_0[i_quad_1,i_quad_2,i_quad_3]
                                    h_1 = arr_h_1[i_quad_1,i_quad_2,i_quad_3]
                                    h_2 = arr_h_2[i_quad_1,i_quad_2,i_quad_3]
                                    temp_v1_2_u1_2_3 = u1_2_3*v1_2_3
                                    c += h_0**2*temp_v1_2_u1_2_3 + h_1**2*temp_v1_2_u1_2_3
                                c1_9[i_basis_3, j_basis_3, i_quad_1, i_quad_2] = c
                for i_basis_2 in range(0, 2, 1):
                    for j_basis_2 in range(0, 2, 1):
                        for i_basis_3 in range(0, 1, 1):
                            for j_basis_3 in range(0, 1, 1):
                                for i_quad_1 in range(0, 2, 1):
                                    c = 0.0
                                    for i_quad_2 in range(0, 2, 1):
                                        v1_2_2 = global_test_basis_v1_2_2[i_element_2,i_basis_2,0,i_quad_2]
                                        u1_2_2 = global_trial_basis_u1_2_2[i_element_2,j_basis_2,0,i_quad_2]
                                        c += v1_2_2 * u1_2_2 * c1_9[i_basis_3, j_basis_3, i_quad_1, i_quad_2]
                                    c2_9[i_basis_2, j_basis_2, i_basis_3, j_basis_3, i_quad_1] = c
                for i_basis_1 in range(0, 2, 1):
                    for j_basis_1 in range(0, 2, 1):
                        for i_basis_2 in range(0, 2, 1):
                            for j_basis_2 in range(0, 2, 1):
                                for i_basis_3 in range(0, 1, 1):
                                    for j_basis_3 in range(0, 1, 1):
                                        c = 0.0
                                        for i_quad_1 in range(0, 2, 1):
                                            v1_2_1 = global_test_basis_v1_2_1[i_element_1,i_basis_1,0,i_quad_1]
                                            u1_2_1 = global_trial_basis_u1_2_1[i_element_1,j_basis_1,0,i_quad_1]
                                            c += v1_2_1 * u1_2_1 * c2_9[i_basis_2, j_basis_2, i_basis_3, j_basis_3, i_quad_1]
                                        l_mat_u1_2_v1_2_g2tmrhvb[i_basis_1,i_basis_2,i_basis_3,1 - i_basis_1 + j_basis_1,1 - i_basis_2 + j_basis_2,0 - i_basis_3 + j_basis_3] = c                                                                                                                                                                        


                g_mat_u1_0_v1_0_g2tmrhvb[pad1 + span_v1_0_1 - test_v1_0_p1:1 + pad1 + span_v1_0_1,pad2 + span_v1_0_2 - test_v1_0_p2:1 + pad2 + span_v1_0_2,pad3 + span_v1_0_3 - test_v1_0_p3:1 + pad3 + span_v1_0_3,:,:,:] += l_mat_u1_0_v1_0_g2tmrhvb[:,:,:,:,:,:]
                g_mat_u1_1_v1_0_g2tmrhvb[pad1 + span_v1_0_1 - test_v1_0_p1:1 + pad1 + span_v1_0_1,pad2 + span_v1_0_2 - test_v1_0_p2:1 + pad2 + span_v1_0_2,pad3 + span_v1_0_3 - test_v1_0_p3:1 + pad3 + span_v1_0_3,:,:,:] += l_mat_u1_1_v1_0_g2tmrhvb[:,:,:,:,:,:]
                g_mat_u1_2_v1_0_g2tmrhvb[pad1 + span_v1_0_1 - test_v1_0_p1:1 + pad1 + span_v1_0_1,pad2 + span_v1_0_2 - test_v1_0_p2:1 + pad2 + span_v1_0_2,pad3 + span_v1_0_3 - test_v1_0_p3:1 + pad3 + span_v1_0_3,:,:,:] += l_mat_u1_2_v1_0_g2tmrhvb[:,:,:,:,:,:]
                g_mat_u1_0_v1_1_g2tmrhvb[pad1 + span_v1_1_1 - test_v1_1_p1:1 + pad1 + span_v1_1_1,pad2 + span_v1_1_2 - test_v1_1_p2:1 + pad2 + span_v1_1_2,pad3 + span_v1_1_3 - test_v1_1_p3:1 + pad3 + span_v1_1_3,:,:,:] += l_mat_u1_0_v1_1_g2tmrhvb[:,:,:,:,:,:]
                g_mat_u1_1_v1_1_g2tmrhvb[pad1 + span_v1_1_1 - test_v1_1_p1:1 + pad1 + span_v1_1_1,pad2 + span_v1_1_2 - test_v1_1_p2:1 + pad2 + span_v1_1_2,pad3 + span_v1_1_3 - test_v1_1_p3:1 + pad3 + span_v1_1_3,:,:,:] += l_mat_u1_1_v1_1_g2tmrhvb[:,:,:,:,:,:]
                g_mat_u1_2_v1_1_g2tmrhvb[pad1 + span_v1_1_1 - test_v1_1_p1:1 + pad1 + span_v1_1_1,pad2 + span_v1_1_2 - test_v1_1_p2:1 + pad2 + span_v1_1_2,pad3 + span_v1_1_3 - test_v1_1_p3:1 + pad3 + span_v1_1_3,:,:,:] += l_mat_u1_2_v1_1_g2tmrhvb[:,:,:,:,:,:]
                g_mat_u1_0_v1_2_g2tmrhvb[pad1 + span_v1_2_1 - test_v1_2_p1:1 + pad1 + span_v1_2_1,pad2 + span_v1_2_2 - test_v1_2_p2:1 + pad2 + span_v1_2_2,pad3 + span_v1_2_3 - test_v1_2_p3:1 + pad3 + span_v1_2_3,:,:,:] += l_mat_u1_0_v1_2_g2tmrhvb[:,:,:,:,:,:]
                g_mat_u1_1_v1_2_g2tmrhvb[pad1 + span_v1_2_1 - test_v1_2_p1:1 + pad1 + span_v1_2_1,pad2 + span_v1_2_2 - test_v1_2_p2:1 + pad2 + span_v1_2_2,pad3 + span_v1_2_3 - test_v1_2_p3:1 + pad3 + span_v1_2_3,:,:,:] += l_mat_u1_1_v1_2_g2tmrhvb[:,:,:,:,:,:]
                g_mat_u1_2_v1_2_g2tmrhvb[pad1 + span_v1_2_1 - test_v1_2_p1:1 + pad1 + span_v1_2_1,pad2 + span_v1_2_2 - test_v1_2_p2:1 + pad2 + span_v1_2_2,pad3 + span_v1_2_3 - test_v1_2_p3:1 + pad3 + span_v1_2_3,:,:,:] += l_mat_u1_2_v1_2_g2tmrhvb[:,:,:,:,:,:]
            
        
    
    return