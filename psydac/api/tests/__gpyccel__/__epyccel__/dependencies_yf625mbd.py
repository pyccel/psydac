from pyccel.decorators import types
@types("real[:,:,:,:]", "real[:,:,:,:]", "int[:]", "real[:,:]", "real[:,:]", "int", "int", "int", "int", "int", "real[:,:]", "real[:,:]")
def assembly(global_test_basis_v1_1, global_trial_basis_u1_1, global_span_v1_1, global_x1, global_w1, test_v1_p1, trial_u1_p1, n_element_1, k1, pad1, l_mat_u1_v1_yf625mbd, g_mat_u1_v1_yf625mbd):

    from numpy import zeros, zeros_like, pi, cos, sqrt
    pad_u1_v1_1 = max((test_v1_p1, trial_u1_p1))
    local_x1 = zeros_like(global_x1[0, : ])
    local_w1 = zeros_like(global_w1[0, : ])
    
    for i_element_1 in range(0, n_element_1, 1):
        local_x1[ : ] = global_x1[i_element_1, : ]
        local_w1[ : ] = global_w1[i_element_1, : ]
        span_v1_1 = global_span_v1_1[i_element_1]
        
        l_mat_u1_v1_yf625mbd[ : , : ] = 0.0
        for i_basis_1 in range(0, 1 + test_v1_p1, 1):
            for j_basis_1 in range(0, 1 + trial_u1_p1, 1):
                for i_quad_1 in range(0, 4, 1):
                    x1 = local_x1[i_quad_1]
                    w1 = local_w1[i_quad_1]
                    v1_1 = global_test_basis_v1_1[i_element_1,i_basis_1,0,i_quad_1]
                    v1_1_x1 = global_test_basis_v1_1[i_element_1,i_basis_1,1,i_quad_1]
                    u1_1 = global_trial_basis_u1_1[i_element_1,j_basis_1,0,i_quad_1]
                    u1_1_x1 = global_trial_basis_u1_1[i_element_1,j_basis_1,1,i_quad_1]
                    wvol_M = w1
                    v1 = v1_1
                    v1_x1 = v1_1_x1
                    u1 = u1_1
                    u1_x1 = u1_1_x1
                    temp0 = (0.25*cos(2*pi*x1) + 1.0)**2
                    l_mat_u1_v1_yf625mbd[i_basis_1,-i_basis_1 + j_basis_1 + pad_u1_v1_1] += 1.0*u1*v1*wvol_M/sqrt(temp0)
                
            
        
        g_mat_u1_v1_yf625mbd[pad1 + span_v1_1 - test_v1_p1 : 1 + pad1 + span_v1_1, : ] += l_mat_u1_v1_yf625mbd[ : , : ]
    
