from pyccel.decorators import types
@types("real[:,:,:,:]", "real[:,:,:,:]", "int[:]", "int[:]", "real[:,:]", "real[:,:]", "real[:,:]", "real[:,:]", "int", "int", "int", "int", "int", "int", "int", "int", "real[:,:]", "real[:,:]")
def assembly(global_test_basis_v_1, global_test_basis_v_2, global_span_v_1, global_span_v_2, global_x1, global_w1, global_x2, global_w2, test_v_p1, test_v_p2, n_element_1, n_element_2, k1, k2, pad1, pad2, l_vec_v_vuqhlic1, g_vec_v_vuqhlic1):

    from numpy import zeros, zeros_like, sin, pi
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
            
            l_vec_v_vuqhlic1[ : , : ] = 0.0
            for i_basis_1 in range(0, 1 + test_v_p1, 1):
                for i_basis_2 in range(0, 1 + test_v_p2, 1):
                    for i_quad_1 in range(0, k1, 1):
                        x1 = local_x1[i_quad_1]
                        w1 = local_w1[i_quad_1]
                        v_1 = global_test_basis_v_1[i_element_1,i_basis_1,0,i_quad_1]
                        v_1_x1 = global_test_basis_v_1[i_element_1,i_basis_1,1,i_quad_1]
                        for i_quad_2 in range(0, k2, 1):
                            x2 = local_x2[i_quad_2]
                            w2 = local_w2[i_quad_2]
                            v_2 = global_test_basis_v_2[i_element_2,i_basis_2,0,i_quad_2]
                            v_2_x2 = global_test_basis_v_2[i_element_2,i_basis_2,1,i_quad_2]
                            wvol_M_Square = w1*w2
                            v = v_1*v_2
                            v_x2 = v_1*v_2_x2
                            v_x1 = v_1_x1*v_2
                            l_vec_v_vuqhlic1[i_basis_1,i_basis_2] += 2*pi**2*v*wvol_M_Square*sin(pi*x1)*sin(pi*x2)
                        
                    
                
            
            g_vec_v_vuqhlic1[pad1 + span_v_1 - test_v_p1 : 1 + pad1 + span_v_1,pad2 + span_v_2 - test_v_p2 : 1 + pad2 + span_v_2] += l_vec_v_vuqhlic1[ : , : ]
        
    
