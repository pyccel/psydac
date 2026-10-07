"""Stencil matrix-vector product ``out = mat @ x`` on the owned rows (3D).

The host version of ``stencil_dot_3d``.

Moved from ``feectools.linalg.stencil_dot_kernels.matvec_3d_kernel``. The CUDA version in ``stencil_dot_3d_cuda.cu``
takes the same arguments in the same order.
"""


def stencil_dot_3d(mat: 'float[:, :, :, :, :, :]',
                   x: 'float[:, :, :]',
                   out: 'float[:, :, :]',
                   s_in: 'int[:]',
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
    for i1 in range(s_out[0], e_out[0]):
        i1_loc = i1 - s_out[0]

        #####################################
        # without last row in 2nd direction #
        #####################################
        for i2 in range(s_out[1], e_out[1]):
            i2_loc = i2 - s_out[1]

            # without last row in 3rd direction
            for i3 in range(s_out[2], e_out[2]):
                i3_loc = i3 - s_out[2]

                val = 0.
                for d1 in range(2*p_in[0] + 1):
                    for d2 in range(2*p_in[1] + 1):
                        for d3 in range(2*p_in[2] + 1):

                            val += mat[p_out[0] + i1_loc,
                                       p_out[1] + i2_loc,
                                       p_out[2] + i3_loc,
                                       d1, d2, d3] * x[i1 + d1 - s_in[0],
                                                       i2 + d2 - s_in[1],
                                                       i3 + d3 - s_in[2]]
                out[p_out[0] + i1_loc,
                    p_out[1] + i2_loc,
                    p_out[2] + i3_loc] = val

            # treat last row in 3rd direction separately
            i3 = e_out[2]
            i3_loc = i3 - s_out[2]
            val = 0.
            for d1 in range(2*p_in[0] + 1):
                for d2 in range(2*p_in[1] + 1):
                    for d3 in range(2*p_in[2] + add[2]):

                        val += mat[p_out[0] + i1_loc,
                                   p_out[1] + i2_loc,
                                   p_out[2] + i3_loc,
                                   d1, d2, d3] * x[i1 + d1 - s_in[0],
                                                   i2 + d2 - s_in[1],
                                                   i3 + d3 - s_in[2]]
            out[p_out[0] + i1_loc,
                p_out[1] + i2_loc,
                p_out[2] + i3_loc] = val

        ##############################################
        # treat last row in 2nd direction separately #
        ##############################################
        i2 = e_out[1]
        i2_loc = i2 - s_out[1]

        # without last row in 3rd direction
        for i3 in range(s_out[2], e_out[2]):
            i3_loc = i3 - s_out[2]

            val = 0.
            for d1 in range(2*p_in[0] + 1):
                for d2 in range(2*p_in[1] + add[1]):
                    for d3 in range(2*p_in[2] + 1):

                        val += mat[p_out[0] + i1_loc,
                                   p_out[1] + i2_loc,
                                   p_out[2] + i3_loc,
                                   d1, d2, d3] * x[i1 + d1 - s_in[0],
                                                   i2 + d2 - s_in[1],
                                                   i3 + d3 - s_in[2]]
            out[p_out[0] + i1_loc,
                p_out[1] + i2_loc,
                p_out[2] + i3_loc] = val

        # treat last row in 3rd direction separately
        i3 = e_out[2]
        i3_loc = i3 - s_out[2]
        val = 0.
        for d1 in range(2*p_in[0] + 1):
            for d2 in range(2*p_in[1] + add[1]):
                for d3 in range(2*p_in[2] + add[2]):

                    val += mat[p_out[0] + i1_loc,
                               p_out[1] + i2_loc,
                               p_out[2] + i3_loc,
                               d1, d2, d3] * x[i1 + d1 - s_in[0],
                                               i2 + d2 - s_in[1],
                                               i3 + d3 - s_in[2]]
        out[p_out[0] + i1_loc,
            p_out[1] + i2_loc,
            p_out[2] + i3_loc] = val

    ##############################################
    ##############################################
    # treat last row in 1st direction separately #
    ##############################################
    ##############################################
    i1 = e_out[0]
    i1_loc = i1 - s_out[0]

    #####################################
    # without last row in 2nd direction #
    #####################################
    for i2 in range(s_out[1], e_out[1]):
        i2_loc = i2 - s_out[1]

        # without last row in 3rd direction
        for i3 in range(s_out[2], e_out[2]):
            i3_loc = i3 - s_out[2]

            val = 0.
            for d1 in range(2*p_in[0] + add[0]):
                for d2 in range(2*p_in[1] + 1):
                    for d3 in range(2*p_in[2] + 1):

                        val += mat[p_out[0] + i1_loc,
                                   p_out[1] + i2_loc,
                                   p_out[2] + i3_loc,
                                   d1, d2, d3] * x[i1 + d1 - s_in[0],
                                                   i2 + d2 - s_in[1],
                                                   i3 + d3 - s_in[2]]
            out[p_out[0] + i1_loc,
                p_out[1] + i2_loc,
                p_out[2] + i3_loc] = val

        # treat last row in 3rd direction separately
        i3 = e_out[2]
        i3_loc = i3 - s_out[2]
        val = 0.
        for d1 in range(2*p_in[0] + add[0]):
            for d2 in range(2*p_in[1] + 1):
                for d3 in range(2*p_in[2] + add[2]):

                    val += mat[p_out[0] + i1_loc,
                               p_out[1] + i2_loc,
                               p_out[2] + i3_loc,
                               d1, d2, d3] * x[i1 + d1 - s_in[0],
                                               i2 + d2 - s_in[1],
                                               i3 + d3 - s_in[2]]
        out[p_out[0] + i1_loc,
            p_out[1] + i2_loc,
            p_out[2] + i3_loc] = val

    ##############################################
    # treat last row in 2nd direction separately #
    ##############################################
    i2 = e_out[1]
    i2_loc = i2 - s_out[1]

    # without last row in 3rd direction
    for i3 in range(s_out[2], e_out[2]):
        i3_loc = i3 - s_out[2]

        val = 0.
        for d1 in range(2*p_in[0] + add[0]):
            for d2 in range(2*p_in[1] + add[1]):
                for d3 in range(2*p_in[2] + 1):

                    val += mat[p_out[0] + i1_loc,
                               p_out[1] + i2_loc,
                               p_out[2] + i3_loc,
                               d1, d2, d3] * x[i1 + d1 - s_in[0],
                                               i2 + d2 - s_in[1],
                                               i3 + d3 - s_in[2]]
        out[p_out[0] + i1_loc,
            p_out[1] + i2_loc,
            p_out[2] + i3_loc] = val

    # treat last row in 3rd direction separately
    i3 = e_out[2]
    i3_loc = i3 - s_out[2]
    val = 0.
    for d1 in range(2*p_in[0] + add[0]):
        for d2 in range(2*p_in[1] + add[1]):
            for d3 in range(2*p_in[2] + add[2]):

                val += mat[p_out[0] + i1_loc,
                           p_out[1] + i2_loc,
                           p_out[2] + i3_loc,
                           d1, d2, d3] * x[i1 + d1 - s_in[0],
                                           i2 + d2 - s_in[1],
                                           i3 + d3 - s_in[2]]
    out[p_out[0] + i1_loc,
        p_out[1] + i2_loc,
        p_out[2] + i3_loc] = val
