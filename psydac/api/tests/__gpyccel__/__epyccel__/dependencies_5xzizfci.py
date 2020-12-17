from pyccel.decorators import types
@types("real[:,:,:,:]", "real[:,:,:,:]", "int[:]", "real[:,:]", "real[:,:]", "int", "int", "int", "int", "int", "real[:,:]", "real[:,:]")
def assembly(global_test_basis_v0_1, global_trial_basis_u0_1, global_span_v0_1, global_x1, global_w1, test_v0_p1, trial_u0_p1, n_element_1, k1, pad1, l_mat_u0_v0_5xzizfci, g_mat_u0_v0_5xzizfci):

    from numpy import zeros, zeros_like, pi, cos, sqrt
    pad_u0_v0_1 = max((test_v0_p1, trial_u0_p1))
    local_x1 = zeros_like(global_x1[0, : ])
    local_w1 = zeros_like(global_w1[0, : ])
    
    for i_element_1 in range(0, n_element_1, 1):
        local_x1[ : ] = global_x1[i_element_1, : ]
        local_w1[ : ] = global_w1[i_element_1, : ]
        span_v0_1 = global_span_v0_1[i_element_1]
        
        l_mat_u0_v0_5xzizfci[ : , : ] = 0.0
        for i_basis_1 in range(0, 1 + test_v0_p1, 1):
            for j_basis_1 in range(0, 1 + trial_u0_p1, 1):
                for i_quad_1 in range(0, 3, 1):
                    x1 = local_x1[i_quad_1]
                    w1 = local_w1[i_quad_1]
                    v0_1 = global_test_basis_v0_1[i_element_1,i_basis_1,0,i_quad_1]
                    v0_1_x1 = global_test_basis_v0_1[i_element_1,i_basis_1,1,i_quad_1]
                    u0_1 = global_trial_basis_u0_1[i_element_1,j_basis_1,0,i_quad_1]
                    u0_1_x1 = global_trial_basis_u0_1[i_element_1,j_basis_1,1,i_quad_1]
                    wvol_M = w1
                    v0 = v0_1
                    v0_x1 = v0_1_x1
                    u0 = u0_1
                    u0_x1 = u0_1_x1
                    l_mat_u0_v0_5xzizfci[i_basis_1,-i_basis_1 + j_basis_1 + pad_u0_v0_1] += 1.0*u0*v0*wvol_M*sqrt((0.25*cos(2*pi*x1) + 1.0)**2)
                
            
        
        g_mat_u0_v0_5xzizfci[pad1 + span_v0_1 - test_v0_p1 : 1 + pad1 + span_v0_1, : ] += l_mat_u0_v0_5xzizfci[ : , : ]
    
