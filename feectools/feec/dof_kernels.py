#==============================================================================
# 1D DEGREES OF FREEDOM
#==============================================================================
def evaluate_dofs_1d_0form(f : 'float[:]', f_pts : 'float[:]'):
    
    (n1,) = f.shape
    for i1 in range(n1):
        f[i1] = f_pts[i1]
        
#------------------------------------------------------------------------------
def evaluate_dofs_1d_1form(quad_w1 : 'float[:,:]', f : 'float[:]', f_pts : 'float[:]'):

    k1 = quad_w1.shape[1]

    (n1,) = f.shape
    for i1 in range(n1):
        f[i1] = 0.0
        for g1 in range(k1):
            f[i1] += quad_w1[i1, g1] * f_pts[i1*k1 + g1]

#==============================================================================
# 2D DEGREES OF FREEDOM
#==============================================================================
def evaluate_dofs_2d_0form(f : 'float[:,:]', f_pts : 'float[:,:]'):
    
    n1, n2 = f.shape
    for i1 in range(n1):
        for i2 in range(n2):
            f[i1, i2] = f_pts[i1, i2]

#------------------------------------------------------------------------------
def evaluate_dofs_2d_1form_hcurl(quad_w1 : 'float[:,:]', quad_w2 : 'float[:,:]', f1 : 'float[:,:]', f2 : 'float[:,:]', f1_pts : 'float[:,:]', f2_pts : 'float[:,:]'):

    k1 = quad_w1.shape[1]
    k2 = quad_w2.shape[1]

    n1, n2 = f1.shape
    for i1 in range(n1):
        for i2 in range(n2):
            f1[i1, i2] = 0.0
            for g1 in range(k1):
                f1[i1, i2] += quad_w1[i1, g1] * f1_pts[i1*k1 + g1, i2]

    n1, n2 = f2.shape
    for i1 in range(n1):
        for i2 in range(n2):
            f2[i1, i2] = 0.0
            for g2 in range(k2):
                f2[i1, i2] += quad_w2[i2, g2] * f2_pts[i1, i2*k2 + g2]

#------------------------------------------------------------------------------
def evaluate_dofs_2d_1form_hdiv(quad_w1 : 'float[:,:]', quad_w2 : 'float[:,:]', f1 : 'float[:,:]', f2 : 'float[:,:]', f1_pts : 'float[:,:]', f2_pts : 'float[:,:]'):

    k1 = quad_w1.shape[1]
    k2 = quad_w2.shape[1]

    n1, n2 = f1.shape
    for i1 in range(n1):
        for i2 in range(n2):
            f1[i1, i2] = 0.0
            for g2 in range(k2):
                f1[i1, i2] += quad_w2[i2, g2] * f1_pts[i1, i2*k2 + g2]

    n1, n2 = f2.shape
    for i1 in range(n1):
        for i2 in range(n2):
            f2[i1, i2] = 0.0
            for g1 in range(k1):
                f2[i1, i2] += quad_w1[i1, g1] * f2_pts[i1*k1 + g1, i2]

#------------------------------------------------------------------------------
def evaluate_dofs_2d_2form(quad_w1 : 'float[:,:]', quad_w2 : 'float[:,:]', f : 'float[:,:]', f_pts : 'float[:,:]'):

    k1 = quad_w1.shape[1]
    k2 = quad_w2.shape[1]

    n1, n2 = f.shape
    for i1 in range(n1):
        for i2 in range(n2):
            f[i1, i2] = 0.0
            for g1 in range(k1):
                for g2 in range(k2):
                    f[i1, i2] += quad_w1[i1, g1] * quad_w2[i2, g2] * f_pts[i1*k1 + g1, i2*k2 + g2]

#------------------------------------------------------------------------------
def evaluate_dofs_2d_vec(f1 : 'float[:,:]', f2 : 'float[:,:]', f1_pts : 'float[:,:]', f2_pts : 'float[:,:]'):
    
    n1, n2 = f1.shape
    for i1 in range(n1):
        for i2 in range(n2):
            f1[i1, i2] = f1_pts[i1, i2]
                
    n1, n2 = f2.shape
    for i1 in range(n1):
        for i2 in range(n2):
            f2[i1, i2] = f2_pts[i1, i2]

#==============================================================================
# 3D DEGREES OF FREEDOM
#==============================================================================
def evaluate_dofs_3d_0form(f : 'float[:,:,:]', f_pts : 'float[:,:,:]'):
    
    n1, n2, n3 = f.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                f[i1, i2, i3] = f_pts[i1, i2, i3]

#------------------------------------------------------------------------------
def evaluate_dofs_3d_1form(quad_w1 : 'float[:,:]', quad_w2 : 'float[:,:]', quad_w3 : 'float[:,:]', f1 : 'float[:,:,:]', f2 : 'float[:,:,:]', f3 : 'float[:,:,:]', f1_pts : 'float[:,:,:]', f2_pts : 'float[:,:,:]', f3_pts : 'float[:,:,:]'):

    k1 = quad_w1.shape[1]
    k2 = quad_w2.shape[1]
    k3 = quad_w3.shape[1]

    n1, n2, n3 = f1.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                f1[i1, i2, i3] = 0.0
                for g1 in range(k1):
                    f1[i1, i2, i3] += quad_w1[i1, g1] * f1_pts[i1*k1 + g1, i2, i3]

    n1, n2, n3 = f2.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                f2[i1, i2, i3] = 0.0
                for g2 in range(k2):
                    f2[i1, i2, i3] += quad_w2[i2, g2] * f2_pts[i1, i2*k2 + g2, i3]

    n1, n2, n3 = f3.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                f3[i1, i2, i3] = 0.0
                for g3 in range(k3):
                    f3[i1, i2, i3] += quad_w3[i3, g3] * f3_pts[i1, i2, i3*k3 + g3]

#------------------------------------------------------------------------------
def evaluate_dofs_3d_2form(quad_w1 : 'float[:,:]', quad_w2 : 'float[:,:]', quad_w3 : 'float[:,:]', f1 : 'float[:,:,:]', f2 : 'float[:,:,:]', f3 : 'float[:,:,:]', f1_pts : 'float[:,:,:]', f2_pts : 'float[:,:,:]', f3_pts : 'float[:,:,:]'):

    k1 = quad_w1.shape[1]
    k2 = quad_w2.shape[1]
    k3 = quad_w3.shape[1]

    n1, n2, n3 = f1.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                f1[i1, i2, i3] = 0.0
                for g2 in range(k2):
                    for g3 in range(k3):
                        f1[i1, i2, i3] += quad_w2[i2, g2] * quad_w3[i3, g3] * f1_pts[i1, i2*k2 + g2, i3*k3 + g3]

    n1, n2, n3 = f2.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                f2[i1, i2, i3] = 0.0
                for g1 in range(k1):
                    for g3 in range(k3):
                        f2[i1, i2, i3] += quad_w1[i1, g1] * quad_w3[i3, g3] * f2_pts[i1*k1 + g1, i2, i3*k3 + g3]

    n1, n2, n3 = f3.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                f3[i1, i2, i3] = 0.0
                for g1 in range(k1):
                    for g2 in range(k2):
                        f3[i1, i2, i3] += quad_w1[i1, g1] * quad_w2[i2, g2] * f3_pts[i1*k1 + g1, i2*k2 + g2, i3]

#------------------------------------------------------------------------------
def evaluate_dofs_3d_3form(quad_w1 : 'float[:,:]', quad_w2 : 'float[:,:]', quad_w3 : 'float[:,:]', f : 'float[:,:,:]', f_pts : 'float[:,:,:]'):

    k1 = quad_w1.shape[1]
    k2 = quad_w2.shape[1]
    k3 = quad_w3.shape[1]

    n1, n2, n3 = f.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                f[i1, i2, i3] = 0.0
                for g1 in range(k1):
                    for g2 in range(k2):
                        for g3 in range(k3):
                            f[i1, i2, i3] += quad_w1[i1, g1] * quad_w2[i2, g2] * quad_w3[i3, g3] * f_pts[i1*k1 + g1, i2*k2 + g2, i3*k3 + g3]

#------------------------------------------------------------------------------
def evaluate_dofs_3d_vec(f1 : 'float[:,:,:]', f2 : 'float[:,:,:]', f3 : 'float[:,:,:]', f1_pts : 'float[:,:,:]', f2_pts : 'float[:,:,:]', f3_pts : 'float[:,:,:]'):
    
    n1, n2, n3 = f1.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                f1[i1, i2, i3] = f1_pts[i1, i2, i3]
                
    n1, n2, n3 = f2.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                f2[i1, i2, i3] = f2_pts[i1, i2, i3]
                
    n1, n2, n3 = f3.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                f3[i1, i2, i3] = f3_pts[i1, i2, i3]