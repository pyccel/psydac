from pyccel.decorators import types
@types("real[:,:,:,:]", "real[:,:,:,:]", "int[:]", "int[:]", "real[:,:]", "real[:,:]", "real[:,:]", "real[:,:]", "int", "int", "int", "int", "int", "int", "int", "int", "real[:]", "real[:]", "real[:,:]")
def assembly(global_test_basis_u_1, global_test_basis_u_2, global_span_u_1, global_span_u_2, global_x1, global_w1, global_x2, global_w2, test_u_p1, test_u_p2, n_element_1, n_element_2, k1, k2, pad1, pad2, l_el_j6duku, g_el_wzmbdw, global_arr_coeffs_u):

    from numpy import zeros, zeros_like, cos, sin, pi
    local_x1 = zeros_like(global_x1[0, : ])
    local_w1 = zeros_like(global_w1[0, : ])
    local_x2 = zeros_like(global_x2[0, : ])
    local_w2 = zeros_like(global_w2[0, : ])
    arr_u = zeros((k1, k2))
    arr_u_x2 = zeros((k1, k2))
    arr_u_x1 = zeros((k1, k2))
    arr_coeffs_u = zeros((1 + test_u_p1, 1 + test_u_p2))
    
    for i_element_1 in range(0, n_element_1, 1):
        local_x1[ : ] = global_x1[i_element_1, : ]
        local_w1[ : ] = global_w1[i_element_1, : ]
        span_u_1 = global_span_u_1[i_element_1]
        for i_element_2 in range(0, n_element_2, 1):
            local_x2[ : ] = global_x2[i_element_2, : ]
            local_w2[ : ] = global_w2[i_element_2, : ]
            span_u_2 = global_span_u_2[i_element_2]
            
            arr_u[ : , : ] = 0.0
            arr_u_x2[ : , : ] = 0.0
            arr_u_x1[ : , : ] = 0.0
            arr_coeffs_u[ : , : ] = global_arr_coeffs_u[pad1 + span_u_1 - test_u_p1 : 1 + pad1 + span_u_1,pad2 + span_u_2 - test_u_p2 : 1 + pad2 + span_u_2]
            for i_basis_1 in range(0, 1 + test_u_p1, 1):
                for i_basis_2 in range(0, 1 + test_u_p2, 1):
                    coeff_u = arr_coeffs_u[i_basis_1,i_basis_2]
                    for i_quad_1 in range(0, k1, 1):
                        u_1 = global_test_basis_u_1[i_element_1,i_basis_1,0,i_quad_1]
                        u_1_x1 = global_test_basis_u_1[i_element_1,i_basis_1,1,i_quad_1]
                        for i_quad_2 in range(0, k2, 1):
                            u_2 = global_test_basis_u_2[i_element_2,i_basis_2,0,i_quad_2]
                            u_2_x2 = global_test_basis_u_2[i_element_2,i_basis_2,1,i_quad_2]
                            u = u_1*u_2
                            u_x2 = u_1*u_2_x2
                            u_x1 = u_1_x1*u_2
                            arr_u[i_quad_1,i_quad_2] += u*coeff_u
                            arr_u_x2[i_quad_1,i_quad_2] += u_x2*coeff_u
                            arr_u_x1[i_quad_1,i_quad_2] += u_x1*coeff_u
                        
                    
                
            
            l_el_j6duku[0] = 0.0
            for i_quad_1 in range(0, k1, 1):
                x1 = local_x1[i_quad_1]
                w1 = local_w1[i_quad_1]
                for i_quad_2 in range(0, k2, 1):
                    x2 = local_x2[i_quad_2]
                    w2 = local_w2[i_quad_2]
                    wvol_M_Square = w1*w2
                    u = arr_u[i_quad_1,i_quad_2]
                    u_x2 = arr_u_x2[i_quad_1,i_quad_2]
                    u_x1 = arr_u_x1[i_quad_1,i_quad_2]
                    temp0 = pi*x1
                    temp1 = cos(temp0)
                    temp2 = pi*x2
                    temp3 = sin(temp2)
                    temp4 = 2*pi
                    temp5 = cos(temp2)
                    temp6 = sin(temp0)
                    temp7 = pi**2
                    l_el_j6duku[0] += wvol_M_Square*(temp1**2*temp3**2*temp7 - temp1*temp3*temp4*u_x1 - temp4*temp5*temp6*u_x2 + temp5**2*temp6**2*temp7 + u_x1**2 + u_x2**2)
                
            
            g_el_wzmbdw[0] += l_el_j6duku[0]
        
    
