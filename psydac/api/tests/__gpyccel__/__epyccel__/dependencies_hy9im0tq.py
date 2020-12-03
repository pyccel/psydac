from pyccel.decorators import types
@types("real[:,:,:,:]", "real[:,:,:,:]", "real[:,:,:,:]", "real[:,:,:,:]", "int[:]", "int[:]", "real[:,:]", "real[:,:]", "real[:,:]", "real[:,:]", "int", "int", "int", "int", "int", "int", "int", "int", "int", "int", "real[:,:,:,:]", "real[:,:,:,:]")
def assembly(global_test_basis_v_1, global_test_basis_v_2, global_trial_basis_u_1, global_trial_basis_u_2, global_span_v_1, global_span_v_2, global_x1, global_w1, global_x2, global_w2, test_v_p1, test_v_p2, trial_u_p1, trial_u_p2, n_element_1, n_element_2, k1, k2, pad1, pad2, l_mat_u_v_hy9im0tq, g_mat_u_v_hy9im0tq):

    from numpy import zeros, zeros_like
    pad_u_v_1 = max((test_v_p1, trial_u_p1))
    pad_u_v_2 = max((test_v_p2, trial_u_p2))
    local_x1 = zeros_like(global_x1[0, : ])
    local_w1 = zeros_like(global_w1[0, : ])
    local_x2 = zeros_like(global_x2[0, : ])
    local_w2 = zeros_like(global_w2[0, : ])
    
    for i_element_1 in range(0, n_element_1, 1):
        local_x1[ : ] = global_x1[i_element_1, : ]
        local_w1[ : ] = global_w1[i_element_1, : ]
        span_v_1 = global_span_v_1[i_element_1]
        for i_element_2 in range(0, n_element_2, 1):
            local_x2[ : ] = global_x2[i_element_2, : ]
            local_w2[ : ] = global_w2[i_element_2, : ]
            span_v_2 = global_span_v_2[i_element_2]
            
            l_mat_u_v_hy9im0tq[ : , : , : , : ] = 0.0
            for i_basis_1 in range(0, 1 + test_v_p1, 1):
                for i_basis_2 in range(0, 1 + test_v_p2, 1):
                    for j_basis_1 in range(0, 1 + trial_u_p1, 1):
                        for j_basis_2 in range(0, 1 + trial_u_p2, 1):
                            for i_quad_1 in range(0, 3, 1):
                                x1 = local_x1[i_quad_1]
                                w1 = local_w1[i_quad_1]
                                v_1 = global_test_basis_v_1[i_element_1,i_basis_1,0,i_quad_1]
                                v_1_x1 = global_test_basis_v_1[i_element_1,i_basis_1,1,i_quad_1]
                                u_1 = global_trial_basis_u_1[i_element_1,j_basis_1,0,i_quad_1]
                                u_1_x1 = global_trial_basis_u_1[i_element_1,j_basis_1,1,i_quad_1]
                                for i_quad_2 in range(0, 3, 1):
                                    x2 = local_x2[i_quad_2]
                                    w2 = local_w2[i_quad_2]
                                    v_2 = global_test_basis_v_2[i_element_2,i_basis_2,0,i_quad_2]
                                    v_2_x2 = global_test_basis_v_2[i_element_2,i_basis_2,1,i_quad_2]
                                    u_2 = global_trial_basis_u_2[i_element_2,j_basis_2,0,i_quad_2]
                                    u_2_x2 = global_trial_basis_u_2[i_element_2,j_basis_2,1,i_quad_2]
                                    wvol_M_Square = w1*w2
                                    v = v_1*v_2
                                    v_x2 = v_1*v_2_x2
                                    v_x1 = v_1_x1*v_2
                                    u = u_1*u_2
                                    u_x2 = u_1*u_2_x2
                                    u_x1 = u_1_x1*u_2
                                    l_mat_u_v_hy9im0tq[i_basis_1,i_basis_2,-i_basis_1 + j_basis_1 + pad_u_v_1,-i_basis_2 + j_basis_2 + pad_u_v_2] += wvol_M_Square*(u_x1*v_x1 + u_x2*v_x2)
                                
                            
                        
                    
                
            
            g_mat_u_v_hy9im0tq[pad1 + span_v_1 - test_v_p1 : 1 + pad1 + span_v_1,pad2 + span_v_2 - test_v_p2 : 1 + pad2 + span_v_2, : , : ] += l_mat_u_v_hy9im0tq[ : , : , : , : ]
        
    
