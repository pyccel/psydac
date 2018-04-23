# coding: utf-8
import numpy as np
from spl.linalg.stencil     import VectorSpace, Vector, Matrix
import time
try:
    from pyccel.epyccel import epyccel
    WITH_PYCCEL = True
except:
    WITH_PYCCEL = False

# ... pure python assembly
def assembly_v0(V):

    # ... sizes
    [s1, s2] = V.vector_space.starts
    [e1, e2] = V.vector_space.ends
    [p1, p2] = V.vector_space.pads
    # ...

    # ... seetings
    [k1, k2] = [W.quad_order for W in V.spaces]
    [spans_1, spans_2] = [W.spans for W in V.spaces]
    [basis_1, basis_2] = [W.basis for W in V.spaces]
    [weights_1, weights_2] = [W.weights for W in V.spaces]
    [points_1, points_2] = [W.points for W in V.spaces]
    # ...

    # ... data structure
    M = Matrix(V.vector_space, V.vector_space)
    # ...

    # ... build matrices
    for ie1 in range(s1, e1+1-p1):
        for ie2 in range(s2, e2+1-p2):
            i_span_1 = spans_1[ie1]
            i_span_2 = spans_2[ie2]
            for il_1 in range(0, p1+1):
                for jl_1 in range(0, p1+1):
                    for il_2 in range(0, p2+1):
                        for jl_2 in range(0, p2+1):

                            i1 = i_span_1 - p1  - 1 + il_1
                            j1 = i_span_1 - p1  - 1 + jl_1

                            i2 = i_span_2 - p2  - 1 + il_2
                            j2 = i_span_2 - p2  - 1 + jl_2

                            v_s = 0.0
                            for g1 in range(0, k1):
                                for g2 in range(0, k2):
                                    bi_0 = basis_1[il_1, 0, g1, ie1] * basis_2[il_2, 0, g2, ie2]
                                    bi_x = basis_1[il_1, 1, g1, ie1] * basis_2[il_2, 0, g2, ie2]
                                    bi_y = basis_1[il_1, 0, g1, ie1] * basis_2[il_2, 1, g2, ie2]

                                    bj_0 = basis_1[jl_1, 0, g1, ie1] * basis_2[jl_2, 0, g2, ie2]
                                    bj_x = basis_1[jl_1, 1, g1, ie1] * basis_2[jl_2, 0, g2, ie2]
                                    bj_y = basis_1[jl_1, 0, g1, ie1] * basis_2[jl_2, 1, g2, ie2]

                                    wvol = weights_1[g1, ie1] * weights_2[g2, ie2]

                                    v_s += (bi_x * bj_x + bi_y * bj_y) * wvol

                            M[i1, i2, j1 - i1, j2 - i2]  += v_s
    # ...
# ...

# ...
header_v1 = '#$ header procedure kernel_v1(int, int, double [:,:], double [:,:], double [:,:], double [:,:], double [:], double [:])'
def kernel_v1(k1, k2, bi1, bi2, bj1, bj2, w1, w2):
    v = 0.0
    for g1 in range(0, k1):
        for g2 in range(0, k2):
            bi_0 = bi1[0, g1] * bi2[0, g2]
            bi_x = bi1[1, g1] * bi2[0, g2]
            bi_y = bi1[0, g1] * bi2[1, g2]

            bj_0 = bj1[0, g1] * bj2[0, g2]
            bj_x = bj1[1, g1] * bj2[0, g2]
            bj_y = bj1[0, g1] * bj2[1, g2]

            wvol = w1[g1] * w2[g2]

            v += (bi_x * bj_x + bi_y * bj_y) * wvol
    return v
# ...

# ... pyccel fine assembly
def assembly_v1(V, kernel):

    # ... sizes
    [s1, s2] = V.vector_space.starts
    [e1, e2] = V.vector_space.ends
    [p1, p2] = V.vector_space.pads
    # ...

    # ... seetings
    [k1, k2] = [W.quad_order for W in V.spaces]
    [spans_1, spans_2] = [W.spans for W in V.spaces]
    [basis_1, basis_2] = [W.basis for W in V.spaces]
    [weights_1, weights_2] = [W.weights for W in V.spaces]
    [points_1, points_2] = [W.points for W in V.spaces]
    # ...

    # ... data structure
    M = Matrix(V.vector_space, V.vector_space)
    # ...

    # ... build matrices
    for ie1 in range(s1, e1+1-p1):
        for ie2 in range(s2, e2+1-p2):
            i_span_1 = spans_1[ie1]
            i_span_2 = spans_2[ie2]
            for il_1 in range(0, p1+1):
                for jl_1 in range(0, p1+1):
                    for il_2 in range(0, p2+1):
                        for jl_2 in range(0, p2+1):

                            i1 = i_span_1 - p1  - 1 + il_1
                            j1 = i_span_1 - p1  - 1 + jl_1

                            i2 = i_span_2 - p2  - 1 + il_2
                            j2 = i_span_2 - p2  - 1 + jl_2

                            bi1 = basis_1[il_1, :, :, ie1]
                            bi2 = basis_2[il_2, :, :, ie2]

                            bj1 = basis_1[jl_1, :, :, ie1]
                            bj2 = basis_2[jl_2, :, :, ie2]

                            w1 = weights_1[:, ie1]
                            w2 = weights_2[:, ie2]

                            v_s = kernel(k1, k2, bi1, bi2, bj1, bj2, w1, w2)
                            M[i1, i2, j1 - i1, j2 - i2]  += v_s
    # ...
# ...

# ...
header_v2 = '#$ header procedure kernel_v2(int, int, int, int, double [:,:,:], double [:,:,:], double [:], double [:], double [:,:,:,:])'
def kernel_v2(p1, p2, k1, k2, bs1, bs2, w1, w2, mat):
    mat = 0.
    for il_1 in range(0, p1+1):
        for jl_1 in range(0, p1+1):
            for il_2 in range(0, p2+1):
                for jl_2 in range(0, p2+1):

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0 = bs1[il_1, 0, g1] * bs2[il_2, 0, g2]
                            bi_x = bs1[il_1, 1, g1] * bs2[il_2, 0, g2]
                            bi_y = bs1[il_1, 0, g1] * bs2[il_2, 1, g2]

                            bj_0 = bs1[jl_1, 0, g1] * bs2[jl_2, 0, g2]
                            bj_x = bs1[jl_1, 1, g1] * bs2[jl_2, 0, g2]
                            bj_y = bs1[jl_1, 0, g1] * bs2[jl_2, 1, g2]

                            wvol = w1[g1] * w2[g2]

                            v += (bi_x * bj_x + bi_y * bj_y) * wvol
                    mat[il_1, il_2, p1 + jl_1 - il_1, p2 + jl_2 - il_2] = v
# ...

# ... pure python assembly
def assembly_v2(V, kernel):

    # ... sizes
    [s1, s2] = V.vector_space.starts
    [e1, e2] = V.vector_space.ends
    [p1, p2] = V.vector_space.pads
    # ...

    # ... seetings
    [k1, k2] = [W.quad_order for W in V.spaces]
    [spans_1, spans_2] = [W.spans for W in V.spaces]
    [basis_1, basis_2] = [W.basis for W in V.spaces]
    [weights_1, weights_2] = [W.weights for W in V.spaces]
    [points_1, points_2] = [W.points for W in V.spaces]
    # ...

    # ... data structure
    M = Matrix(V.vector_space, V.vector_space)
    # ...

    # ... element matrix
    mat = np.zeros((p1+1, p2+1, 2*p1+1, 2*p2+1), order='F')
    # ...

    # ... build matrices
    for ie1 in range(s1, e1+1-p1):
        for ie2 in range(s2, e2+1-p2):
            i_span_1 = spans_1[ie1]
            i_span_2 = spans_2[ie2]

            bs1 = basis_1[:, :, :, ie1]
            bs2 = basis_2[:, :, :, ie2]
            w1 = weights_1[:, ie1]
            w2 = weights_2[:, ie2]
            kernel(p1, p2, k1, k2, bs1, bs2, w1, w2, mat)

            for il_1 in range(0, p1+1):
                for jl_1 in range(0, p1+1):
                    for il_2 in range(0, p2+1):
                        for jl_2 in range(0, p2+1):

                            i1 = i_span_1 - p1  - 1 + il_1
                            j1 = i_span_1 - p1  - 1 + jl_1

                            i2 = i_span_2 - p2  - 1 + il_2
                            j2 = i_span_2 - p2  - 1 + jl_2

                            ij1 = p1 + jl_1 - il_1
                            ij2 = p2 + jl_2 - il_2
                            M[i1, i2, j1 - i1, j2 - i2] += mat[il_1, il_2, ij1, ij2]
    # ...
# ...

# ... pure python assembly
def assembly_v3(V, kernel):

    # ... sizes
    [s1, s2] = V.vector_space.starts
    [e1, e2] = V.vector_space.ends
    [p1, p2] = V.vector_space.pads
    # ...

    # ... seetings
    [k1, k2] = [W.quad_order for W in V.spaces]
    [spans_1, spans_2] = [W.spans for W in V.spaces]
    [basis_1, basis_2] = [W.basis for W in V.spaces]
    [weights_1, weights_2] = [W.weights for W in V.spaces]
    [points_1, points_2] = [W.points for W in V.spaces]
    # ...

    # ... data structure
    M = Matrix(V.vector_space, V.vector_space)
    # ...

    # ... element matrix
    mat = np.zeros((p1+1, p2+1, 2*p1+1, 2*p2+1), order='F')
    # ...

    # ... build matrices
    for ie1 in range(s1, e1+1-p1):
        for ie2 in range(s2, e2+1-p2):
            i_span_1 = spans_1[ie1]
            i_span_2 = spans_2[ie2]

            bs1 = basis_1[:, :, :, ie1]
            bs2 = basis_2[:, :, :, ie2]
            w1 = weights_1[:, ie1]
            w2 = weights_2[:, ie2]
            kernel(p1, p2, k1, k2, bs1, bs2, w1, w2, mat)

            s1 = i_span_1 - p1 - 1
            s2 = i_span_2 - p2 - 1
            M._data[s1:s1+p1+1,s2:s2+p2+1,:,:] += mat[:,:,:,:]
    # ...
# ...


####################################################################################
if __name__ == '__main__':

    from spl.core.interface import make_open_knots
    from spl.fem.splines import SplineSpace
    from spl.fem.tensor  import TensorSpace

    # ... numbers of elements and degres
    p1  = 5 ; p2  = 5
    ne1 = 64 ; ne2 = 64
    n1 = p1 + ne1 ;  n2 = p2 + ne2
    # ...

    print('> Grid   :: [{ne1},{ne2}]'.format(ne1=ne1, ne2=ne2))
    print('> Degree :: [{p1},{p2}]'.format(p1=p1, p2=p2))

    knots_1 = make_open_knots(p1, n1)
    knots_2 = make_open_knots(p2, n2)

    V1 = SplineSpace(knots_1, p1)
    V2 = SplineSpace(knots_2, p2)

    V = TensorSpace(V1, V2)

#    # ... pure python assembly
#    tb = time.time()
#    assembly_v0(V)
#    te = time.time()
#    print('> elapsed time v0 : {} [pure Python]'.format(te-tb))
#    # ...

    # ... assembly using Pyccel
    if WITH_PYCCEL:

#        # ... using pyccel version 1
#        kernel = epyccel(kernel_v1, header_v1)
#        tb = time.time()
#        assembly_v1(V, kernel)
#        te = time.time()
#        print('> elapsed time v1 : {} [using Pyccel]'.format(te-tb))
#        # ...
#
#        # ... using pyccel version 2
#        kernel = epyccel(kernel_v2, header_v2)
#        tb = time.time()
#        assembly_v2(V, kernel)
#        te = time.time()
#        print('> elapsed time v2 : {} [using Pyccel]'.format(te-tb))
#        # ...

        # ... using pyccel version 3
        kernel = epyccel(kernel_v2, header_v2)
        tb = time.time()
        assembly_v3(V, kernel)
        te = time.time()
        print('> elapsed time v3 : {} [using Pyccel]'.format(te-tb))
        # ...
    # ...
