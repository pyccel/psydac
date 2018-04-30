# coding: utf-8
import numpy as np
import time

from spl.linalg.stencil import StencilMatrix
try:
    from pyccel.epyccel import epyccel
    WITH_PYCCEL = True
except:
    WITH_PYCCEL = False

# TODO check if we need to create basis (and other multi-dim array) in spl using Fortran ordering

# ... pure python assembly
def assembly_v0(V):

    # ... sizes
    [s1] = V.vector_space.starts
    [e1] = V.vector_space.ends
    [p1] = V.vector_space.pads

    k1 = V.quad_order
    spans_1 = V.spans
    basis_1 = V.basis
    weights_1 = V.weights
    # ...

    # ... data structure
    M = StencilMatrix(V.vector_space, V.vector_space)
    # ...

    # ... build matrices
    for ie1 in range(s1, e1+1-p1):
        i_span_1 = spans_1[ie1]
        for il_1 in range(0, p1+1):
            for jl_1 in range(0, p1+1):
                i1 = i_span_1 - p1  - 1 + il_1
                j1 = i_span_1 - p1  - 1 + jl_1

                v_s = 0.0
                for g1 in range(0, k1):
                    bi_0 = basis_1[il_1, 0, g1, ie1]
                    bi_x = basis_1[il_1, 1, g1, ie1]

                    bj_0 = basis_1[jl_1, 0, g1, ie1]
                    bj_x = basis_1[jl_1, 1, g1, ie1]

                    wvol = weights_1[g1, ie1]

                    v_s += (bi_x * bj_x) * wvol

                M[i1, j1 - i1]  += v_s
    # ...
# ...

# ...
header_v1 = '#$ header procedure kernel_v1(int, double [:,:], double [:,:], double [:])'
def kernel_v1(k1, bi, bj, w):
    v = 0.0
    for g1 in range(0, k1):
        bi_0 = bi[0, g1]
        bi_x = bi[1, g1]

        bj_0 = bj[0, g1]
        bj_x = bj[1, g1]

        wvol = w[g1]

        v += (bi_x * bj_x) * wvol
    return v
# ...

# ... pyccel fine assembly
def assembly_v1(V, kernel):

    # ... sizes
    [s1] = V.vector_space.starts
    [e1] = V.vector_space.ends
    [p1] = V.vector_space.pads

    k1 = V.quad_order
    spans_1 = V.spans
    basis_1 = V.basis
    weights_1 = V.weights
    # ...

    # ... data structure
    M = StencilMatrix( V.vector_space, V.vector_space )
    # ...

    # ... build matrices
    for ie1 in range(s1, e1+1-p1):
        i_span_1 = spans_1[ie1]
        for il_1 in range(0, p1+1):
            for jl_1 in range(0, p1+1):
                i1 = i_span_1 - p1  - 1 + il_1
                j1 = i_span_1 - p1  - 1 + jl_1

                bi = basis_1[il_1, :, :, ie1]
                bj = basis_1[jl_1, :, :, ie1]
                w = weights_1[:, ie1]

                v_s = kernel(k1, bi, bj, w)
                M[i1, j1 - i1] += v_s
    # ...
# ...

# ...
header_v2 = '#$ header procedure kernel_v2(int, int, double [:,:,:], double [:], double [:,:])'
def kernel_v2(p1, k1, basis, w, mat):
    mat = 0.
    for il_1 in range(0, p1+1):
        for jl_1 in range(0, p1+1):

            v = 0.0
            for g1 in range(0, k1):
                bi_0 = basis[il_1, 0, g1]
                bi_x = basis[il_1, 1, g1]

                bj_0 = basis[jl_1, 0, g1]
                bj_x = basis[jl_1, 1, g1]

                wvol = w[g1]

                v += (bi_x * bj_x) * wvol
            mat[il_1, jl_1] = v
# ...

# ... pyccel coarse assembly
def assembly_v2(V, kernel):

    # ... sizes
    [s1] = V.vector_space.starts
    [e1] = V.vector_space.ends
    [p1] = V.vector_space.pads

    k1 = V.quad_order
    spans_1 = V.spans
    basis_1 = V.basis
    weights_1 = V.weights
    # ...

    # ... data structure
    M = StencilMatrix(V.vector_space, V.vector_space)
    # ...

    # ... element matrix
    mat = np.zeros((p1+1,p1+1), order='F')
    # ...

    # ... build matrices
    for ie1 in range(s1, e1+1-p1):
        i_span_1 = spans_1[ie1]

        bs = basis_1[:, :, :, ie1]
        w = weights_1[:, ie1]
        kernel(p1, k1, bs, w, mat)
        for il_1 in range(0, p1+1):
            for jl_1 in range(0, p1+1):
                i1 = i_span_1 - p1 - 1 + il_1
                j1 = i_span_1 - p1 - 1 + jl_1

                M[i1, j1 - i1]  += mat[il_1, jl_1]
    # ...
# ...

####################################################################################
if __name__ == '__main__':

    from spl.core.interface import make_open_knots
    from spl.fem.splines import SplineSpace

    ne1 = 32
    p   = 7
    knots = make_open_knots(p, ne1)

    V = SplineSpace(p, knots=knots)

    # ... pure python assembly
    tb = time.time()
    assembly_v0(V)
    te = time.time()
    print('> elapsed time v0 : {} [pure Python]'.format(te-tb))
    # ...

    # ... assembly using Pyccel
    if WITH_PYCCEL:

        # ... using pyccel version 1
        kernel = epyccel(kernel_v1, header_v1)
        tb = time.time()
        assembly_v1(V, kernel)
        te = time.time()
        print('> elapsed time v1 : {} [using Pyccel]'.format(te-tb))
        # ...

        # ... using pyccel version 2
        kernel = epyccel(kernel_v2, header_v2)
        tb = time.time()
        assembly_v2(V, kernel)
        te = time.time()
        print('> elapsed time v2 : {} [using Pyccel]'.format(te-tb))
        # ...
    # ...
