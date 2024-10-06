def transpose_1d_kernel(mat: 'float[:, :]',
                        matT: 'float[:, :]',
                        s_in: int,  # refers to matT
                        p_in: int,
                        add: int,
                        s_out: int,
                        e_out: int,
                        p_out: int):

    for i1 in range(s_out, e_out):  # global row index of matT = global column index of mat
        i1_loc = i1 - s_out  # local row index of matT
        for d1 in range(2*p_in + 1):
            j1 = i1 - p_in + d1  # global column index of matT
            j1_loc = j1 - s_in  # local column index of matT = local row index of mat

            matT[p_out + i1_loc, d1] = mat[p_in + j1_loc, p_out + i1 - j1]

    # last row treated separately
    i1 = e_out
    i1_loc = i1 - s_out  # local row index of matT
    for d1 in range(2*p_in + add):
        j1 = i1 - p_in + d1  # global column index of matT
        j1_loc = j1 - s_in  # local column index of matT = local row index of mat

        matT[p_out + i1_loc, d1] = mat[p_in + j1_loc, p_out + i1 - j1]


def transpose_3d_kernel(mat: 'float[:, :, :, :, :, :]',
                        matT: 'float[:, :, :, :, :, :]',
                        s_in: 'int[:]',  # refers to matT
                        p_in: 'int[:]',
                        add: 'int[:]',
                        s_out: 'int[:]',
                        e_out: 'int[:]',
                        p_out: 'int[:]'):

    #####################################
    #####################################
    # without last row in 1st direction #
    #####################################
    #####################################
    for i1 in range(s_out[0], e_out[0]):  # global row index of matT = global column index of mat
        i1_loc = i1 - s_out[0]  # local row index of matT
        
        #####################################
        # without last row in 2nd direction #
        #####################################
        for i2 in range(s_out[1], e_out[1]):  # global row index of matT = global column index of mat
            i2_loc = i2 - s_out[1]  # local row index of matT
            
            # without last row in 3rd direction
            for i3 in range(s_out[2], e_out[2]):  # global row index of matT = global column index of mat
                i3_loc = i3 - s_out[2]  # local row index of matT
        
                for d1 in range(2*p_in[0] + 1):
                    j1 = i1 - p_in[0] + d1  # global column index of matT
                    j1_loc = j1 - s_in[0]  # local column index of matT = local row index of mat
                    for d2 in range(2*p_in[1] + 1):
                        j2 = i2 - p_in[1] + d2  # global column index of matT
                        j2_loc = j2 - s_in[1]  # local column index of matT = local row index of mat
                        for d3 in range(2*p_in[2] + 1):
                            j3 = i3 - p_in[2] + d3  # global column index of matT
                            j3_loc = j3 - s_in[2]  # local column index of matT = local row index of mat
        
                            matT[p_out[0] + i1_loc,
                                 p_out[1] + i2_loc,
                                 p_out[2] + i3_loc,
                                 d1, d2, d3] = mat[p_in[0] + j1_loc,
                                                   p_in[1] + j2_loc,
                                                   p_in[2] + j3_loc,
                                                   p_out[0] + i1 - j1,
                                                   p_out[1] + i2 - j2,
                                                   p_out[2] + i3 - j3]
                                 
            # treat last row in 3rd direction separately
            i3 = e_out[2]
            i3_loc = i3 - s_out[2]  # local row index of matT
        
            for d1 in range(2*p_in[0] + 1):
                j1 = i1 - p_in[0] + d1  # global column index of matT
                j1_loc = j1 - s_in[0]  # local column index of matT = local row index of mat
                for d2 in range(2*p_in[1] + 1):
                    j2 = i2 - p_in[1] + d2  # global column index of matT
                    j2_loc = j2 - s_in[1]  # local column index of matT = local row index of mat
                    for d3 in range(2*p_in[2] + add[2]):
                        j3 = i3 - p_in[2] + d3  # global column index of matT
                        j3_loc = j3 - s_in[2]  # local column index of matT = local row index of mat
    
                        matT[p_out[0] + i1_loc,
                                p_out[1] + i2_loc,
                                p_out[2] + i3_loc,
                                d1, d2, d3] = mat[p_in[0] + j1_loc,
                                                p_in[1] + j2_loc,
                                                p_in[2] + j3_loc,
                                                p_out[0] + i1 - j1,
                                                p_out[1] + i2 - j2,
                                                p_out[2] + i3 - j3]
                                
        ##############################################    
        # treat last row in 2nd direction separately #
        ##############################################
        i2 = e_out[1]
        i2_loc = i2 - s_out[1]  # local row index of matT
            
        # without last row in 3rd direction
        for i3 in range(s_out[2], e_out[2]):  # global row index of matT = global column index of mat
            i3_loc = i3 - s_out[2]  # local row index of matT
    
            for d1 in range(2*p_in[0] + 1):
                j1 = i1 - p_in[0] + d1  # global column index of matT
                j1_loc = j1 - s_in[0]  # local column index of matT = local row index of mat
                for d2 in range(2*p_in[1] + add[1]):
                    j2 = i2 - p_in[1] + d2  # global column index of matT
                    j2_loc = j2 - s_in[1]  # local column index of matT = local row index of mat
                    for d3 in range(2*p_in[2] + 1):
                        j3 = i3 - p_in[2] + d3  # global column index of matT
                        j3_loc = j3 - s_in[2]  # local column index of matT = local row index of mat
    
                        matT[p_out[0] + i1_loc,
                                p_out[1] + i2_loc,
                                p_out[2] + i3_loc,
                                d1, d2, d3] = mat[p_in[0] + j1_loc,
                                                p_in[1] + j2_loc,
                                                p_in[2] + j3_loc,
                                                p_out[0] + i1 - j1,
                                                p_out[1] + i2 - j2,
                                                p_out[2] + i3 - j3]
                                
        # treat last row in 3rd direction separately
        i3 = e_out[2]
        i3_loc = i3 - s_out[2]  # local row index of matT
    
        for d1 in range(2*p_in[0] + 1):
            j1 = i1 - p_in[0] + d1  # global column index of matT
            j1_loc = j1 - s_in[0]  # local column index of matT = local row index of mat
            for d2 in range(2*p_in[1] + add[1]):
                j2 = i2 - p_in[1] + d2  # global column index of matT
                j2_loc = j2 - s_in[1]  # local column index of matT = local row index of mat
                for d3 in range(2*p_in[2] + add[2]):
                    j3 = i3 - p_in[2] + d3  # global column index of matT
                    j3_loc = j3 - s_in[2]  # local column index of matT = local row index of mat

                    matT[p_out[0] + i1_loc,
                            p_out[1] + i2_loc,
                            p_out[2] + i3_loc,
                            d1, d2, d3] = mat[p_in[0] + j1_loc,
                                            p_in[1] + j2_loc,
                                            p_in[2] + j3_loc,
                                            p_out[0] + i1 - j1,
                                            p_out[1] + i2 - j2,
                                            p_out[2] + i3 - j3]
                            
    ##############################################
    ##############################################    
    # treat last row in 1st direction separately #
    ##############################################
    ##############################################
    i1 = e_out[0]
    i1_loc = i1 - s_out[0]  # local row index of matT
        
    #####################################
    # without last row in 2nd direction #
    #####################################
    for i2 in range(s_out[1], e_out[1]):  # global row index of matT = global column index of mat
        i2_loc = i2 - s_out[1]  # local row index of matT
        
        # without last row in 3rd direction
        for i3 in range(s_out[2], e_out[2]):  # global row index of matT = global column index of mat
            i3_loc = i3 - s_out[2]  # local row index of matT
    
            for d1 in range(2*p_in[0] + add[0]):
                j1 = i1 - p_in[0] + d1  # global column index of matT
                j1_loc = j1 - s_in[0]  # local column index of matT = local row index of mat
                for d2 in range(2*p_in[1] + 1):
                    j2 = i2 - p_in[1] + d2  # global column index of matT
                    j2_loc = j2 - s_in[1]  # local column index of matT = local row index of mat
                    for d3 in range(2*p_in[2] + 1):
                        j3 = i3 - p_in[2] + d3  # global column index of matT
                        j3_loc = j3 - s_in[2]  # local column index of matT = local row index of mat
    
                        matT[p_out[0] + i1_loc,
                                p_out[1] + i2_loc,
                                p_out[2] + i3_loc,
                                d1, d2, d3] = mat[p_in[0] + j1_loc,
                                                p_in[1] + j2_loc,
                                                p_in[2] + j3_loc,
                                                p_out[0] + i1 - j1,
                                                p_out[1] + i2 - j2,
                                                p_out[2] + i3 - j3]
                                
        # treat last row in 3rd direction separately
        i3 = e_out[2]
        i3_loc = i3 - s_out[2]  # local row index of matT
    
        for d1 in range(2*p_in[0] + add[0]):
            j1 = i1 - p_in[0] + d1  # global column index of matT
            j1_loc = j1 - s_in[0]  # local column index of matT = local row index of mat
            for d2 in range(2*p_in[1] + 1):
                j2 = i2 - p_in[1] + d2  # global column index of matT
                j2_loc = j2 - s_in[1]  # local column index of matT = local row index of mat
                for d3 in range(2*p_in[2] + add[2]):
                    j3 = i3 - p_in[2] + d3  # global column index of matT
                    j3_loc = j3 - s_in[2]  # local column index of matT = local row index of mat

                    matT[p_out[0] + i1_loc,
                            p_out[1] + i2_loc,
                            p_out[2] + i3_loc,
                            d1, d2, d3] = mat[p_in[0] + j1_loc,
                                            p_in[1] + j2_loc,
                                            p_in[2] + j3_loc,
                                            p_out[0] + i1 - j1,
                                            p_out[1] + i2 - j2,
                                            p_out[2] + i3 - j3]
                            
    ##############################################    
    # treat last row in 2nd direction separately #
    ##############################################
    i2 = e_out[1]
    i2_loc = i2 - s_out[1]  # local row index of matT
        
    # without last row in 3rd direction
    for i3 in range(s_out[2], e_out[2]):  # global row index of matT = global column index of mat
        i3_loc = i3 - s_out[2]  # local row index of matT

        for d1 in range(2*p_in[0] + add[0]):
            j1 = i1 - p_in[0] + d1  # global column index of matT
            j1_loc = j1 - s_in[0]  # local column index of matT = local row index of mat
            for d2 in range(2*p_in[1] + add[1]):
                j2 = i2 - p_in[1] + d2  # global column index of matT
                j2_loc = j2 - s_in[1]  # local column index of matT = local row index of mat
                for d3 in range(2*p_in[2] + 1):
                    j3 = i3 - p_in[2] + d3  # global column index of matT
                    j3_loc = j3 - s_in[2]  # local column index of matT = local row index of mat

                    matT[p_out[0] + i1_loc,
                            p_out[1] + i2_loc,
                            p_out[2] + i3_loc,
                            d1, d2, d3] = mat[p_in[0] + j1_loc,
                                            p_in[1] + j2_loc,
                                            p_in[2] + j3_loc,
                                            p_out[0] + i1 - j1,
                                            p_out[1] + i2 - j2,
                                            p_out[2] + i3 - j3]
                            
    # treat last row in 3rd direction separately
    i3 = e_out[2]
    i3_loc = i3 - s_out[2]  # local row index of matT

    for d1 in range(2*p_in[0] + add[0]):
        j1 = i1 - p_in[0] + d1  # global column index of matT
        j1_loc = j1 - s_in[0]  # local column index of matT = local row index of mat
        for d2 in range(2*p_in[1] + add[1]):
            j2 = i2 - p_in[1] + d2  # global column index of matT
            j2_loc = j2 - s_in[1]  # local column index of matT = local row index of mat
            for d3 in range(2*p_in[2] + add[2]):
                j3 = i3 - p_in[2] + d3  # global column index of matT
                j3_loc = j3 - s_in[2]  # local column index of matT = local row index of mat

                matT[p_out[0] + i1_loc,
                        p_out[1] + i2_loc,
                        p_out[2] + i3_loc,
                        d1, d2, d3] = mat[p_in[0] + j1_loc,
                                        p_in[1] + j2_loc,
                                        p_in[2] + j3_loc,
                                        p_out[0] + i1 - j1,
                                        p_out[1] + i2 - j2,
                                        p_out[2] + i3 - j3]
