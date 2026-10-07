"""Transpose of a stencil matrix, ``matT = mat.T`` on the owned rows of ``matT`` (2D).

The host version of ``stencil_transpose_2d``.

Moved from ``feectools.linalg.stencil_transpose_kernels.transpose_2d_kernel``. The CUDA version in
``stencil_transpose_2d_cuda.cu`` takes the same arguments in the same order. ``e_in`` (the end of the row range of ``mat``) is not
needed here; the CUDA version needs it for the row extents of ``mat``, which it reads through a raw
pointer in 3D.
"""


def stencil_transpose_2d(mat: 'float[:, :, :, :]',
                         matT: 'float[:, :, :, :]',
                         s_in: 'int[:]',  # refers to matT
                         e_in: 'int[:]',  # refers to matT; used by the CUDA version only
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

            for d1 in range(2 * p_in[0] + 1):
                j1 = i1 - p_in[0] + d1  # global column index of matT
                j1_loc = j1 - s_in[0]  # local column index of matT = local row index of mat
                for d2 in range(2 * p_in[1] + 1):
                    j2 = i2 - p_in[1] + d2  # global column index of matT
                    j2_loc = j2 - s_in[1]  # local column index of matT = local row index of mat

                    matT[p_out[0] + i1_loc,
                         p_out[1] + i2_loc,
                         d1, d2] = mat[p_in[0] + j1_loc,
                                       p_in[1] + j2_loc,
                                       p_out[0] + i1 - j1,
                                       p_out[1] + i2 - j2]

        ##############################################
        # treat last row in 2nd direction separately #
        ##############################################
        i2 = e_out[1]
        i2_loc = i2 - s_out[1]  # local row index of matT

        for d1 in range(2 * p_in[0] + 1):
            j1 = i1 - p_in[0] + d1  # global column index of matT
            j1_loc = j1 - s_in[0]  # local column index of matT = local row index of mat
            for d2 in range(2 * p_in[1] + add[1]):
                j2 = i2 - p_in[1] + d2  # global column index of matT
                j2_loc = j2 - s_in[1]  # local column index of matT = local row index of mat

                matT[p_out[0] + i1_loc,
                     p_out[1] + i2_loc,
                     d1, d2] = mat[p_in[0] + j1_loc,
                                   p_in[1] + j2_loc,
                                   p_out[0] + i1 - j1,
                                   p_out[1] + i2 - j2]

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
    for i2 in range(s_out[1], e_out[1]):
        i2_loc = i2 - s_out[1]  # local row index of matT

        for d1 in range(2 * p_in[0] + add[0]):
            j1 = i1 - p_in[0] + d1  # global column index of matT
            j1_loc = j1 - s_in[0]  # local column index of matT = local row index of mat
            for d2 in range(2 * p_in[1] + 1):
                j2 = i2 - p_in[1] + d2  # global column index of matT
                j2_loc = j2 - s_in[1]  # local column index of matT = local row index of mat

                matT[p_out[0] + i1_loc,
                     p_out[1] + i2_loc,
                     d1, d2] = mat[p_in[0] + j1_loc,
                                   p_in[1] + j2_loc,
                                   p_out[0] + i1 - j1,
                                   p_out[1] + i2 - j2]

    ##############################################
    # treat last row in 2nd direction separately #
    ##############################################
    i2 = e_out[1]
    i2_loc = i2 - s_out[1]  # local row index of matT

    for d1 in range(2 * p_in[0] + add[0]):
        j1 = i1 - p_in[0] + d1  # global column index of matT
        j1_loc = j1 - s_in[0]  # local column index of matT = local row index of mat
        for d2 in range(2 * p_in[1] + add[1]):
            j2 = i2 - p_in[1] + d2  # global column index of matT
            j2_loc = j2 - s_in[1]  # local column index of matT = local row index of mat

            matT[p_out[0] + i1_loc,
                 p_out[1] + i2_loc,
                 d1, d2] = mat[p_in[0] + j1_loc,
                               p_in[1] + j2_loc,
                               p_out[0] + i1 - j1,
                               p_out[1] + i2 - j2]
