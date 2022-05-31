from pyccel.decorators import types
from numpy import shape
from numpy import empty

@types("float64[:,:,:,:,:,:]", "float64[:,:,:,:,:,:]", "float64[:,:,:,:,:,:]",
       "float64[:,:,:]", "float64[:,:,:]", "float64[:,:,:]",
       "float64[:,:,:]", "float64[:,:,:]", "float64[:,:,:]",
       "int64", "int64", "int64",
       "int64", "int64", "int64",
       "int64", "int64", "int64",
       "int64", "int64", "int64",
       "int64", "int64", "int64",
       "int64", "int64", "int64", "int64", "int64", "int64")
def dot_1(mat00, mat11, mat22,
          x0, x1, x2,
          out0, out1, out2, 
          s00_1, s00_2, s00_3,
          s11_1, s11_2, s11_3,
          s22_1, s22_2, s22_3,
          n00_1, n00_2, n00_3,
          n11_1, n11_2, n11_3,
          n22_1, n22_2, n22_3, bls1, bls2, bls3):

    #$ omp parallel default(private) shared(mat00, mat11, mat22,x0, x1, x2,out0, out1, out2) firstprivate( n00_1, n00_2, n00_3, n11_1, n11_2, n11_3, n22_1, n22_2, n22_3, bls1, bls2, bls3)
    xx0     = empty((2+bls1,2+bls2,2+bls3))
    v0      = empty((bls1,bls2,bls3))

    #$ omp for schedule(static) collapse(3) nowait
    for i_outer_1 in range(0, n00_1//bls1, 1):
        for i_outer_2 in range(0, n00_2//bls2, 1):
            for i_outer_3 in range(0, n00_3//bls3, 1):
                xx0[:,:,:]  = x0[i_outer_1*bls1:i_outer_1*bls1+2+bls1,i_outer_2*bls2:i_outer_2*bls2+2+bls2,i_outer_3*bls3:i_outer_3*bls3+2+bls3]
                v0[:,:,:]   = 0.
                for i_inner_1 in range(0, bls1, 1):
                    i1 = i_outer_1*bls1 + i_inner_1
                    for i_inner_2 in range(0, bls2, 1):
                        i2 = i_outer_2*bls2 + i_inner_2
                        for i_inner_3 in range(0, bls3, 1):
                            i3 = i_outer_3*bls3 + i_inner_3
                            for k1 in range(0, 3, 1):
                                for k2 in range(0, 3, 1):
                                    for k3 in range(0, 3, 1):
                                        v0[i_inner_1, i_inner_2, i_inner_3] += mat00[1+i1,1+i2, 1+i3,k1,k2,k3]*xx0[i_inner_1+k1,i_inner_2+k2,i_inner_3+k3]

                out0[1 + i_outer_1*bls1:1+i_outer_1*bls1+bls1,1 + i_outer_2*bls2:1+i_outer_2*bls2+bls2,1 + i_outer_3*bls3:1+i_outer_3*bls3+bls3] = v0[:,:,:]


    xx1     = empty((2+bls1,2+bls2,2+bls3))
    v1      = empty((bls1,bls2,bls3))

    #$ omp for schedule(static) collapse(3) nowait
    for i_outer_1 in range(0, n11_1//bls1, 1):
        for i_outer_2 in range(0, n11_2//bls2, 1):
            for i_outer_3 in range(0, n11_3//bls3, 1):
                xx1[:,:,:]  = x1[i_outer_1*bls1:i_outer_1*bls1+2+bls1,i_outer_2*bls2:i_outer_2*bls2+2+bls2,i_outer_3*bls3:i_outer_3*bls3+2+bls3]
                v1[:,:,:]   = 0.
                for i_inner_1 in range(0, bls1, 1):
                    i1 = i_outer_1*bls1 + i_inner_1
                    for i_inner_2 in range(0, bls2, 1):
                        i2 = i_outer_2*bls2 + i_inner_2
                        for i_inner_3 in range(0, bls3, 1):
                            i3 = i_outer_3*bls3 + i_inner_3
                            for k1 in range(0, 3, 1):
                                for k2 in range(0, 3, 1):
                                    for k3 in range(0, 3, 1):
                                        v1[i_inner_1, i_inner_2, i_inner_3] += mat11[1+i1,1+i2, 1+i3,k1,k2,k3]*xx1[i_inner_1+k1,i_inner_2+k2,i_inner_3+k3]

                out1[1 + i_outer_1*bls1:1+i_outer_1*bls1+bls1,1 + i_outer_2*bls2:1+i_outer_2*bls2+bls2,1 + i_outer_3*bls3:1+i_outer_3*bls3+bls3] = v1[:,:,:]

    xx2     = empty((2+bls1,2+bls2,2+bls3))
    v2      = empty((bls1,bls2,bls3))

    #$ omp for schedule(static) collapse(3) nowait
    for i_outer_1 in range(0, n22_1//bls1, 1):
        for i_outer_2 in range(0, n22_2//bls2, 1):
            for i_outer_3 in range(0, n22_3//bls3, 1):
                xx2[:,:,:]  = x2[i_outer_1*bls1:i_outer_1*bls1+2+bls1,i_outer_2*bls2:i_outer_2*bls2+2+bls2,i_outer_3*bls3:i_outer_3*bls3+2+bls3]
                v2[:,:,:]   = 0.
                for i_inner_1 in range(0, bls1, 1):
                    i1 = i_outer_1*bls1 + i_inner_1
                    for i_inner_2 in range(0, bls2, 1):
                        i2 = i_outer_2*bls2 + i_inner_2
                        for i_inner_3 in range(0, bls3, 1):
                            i3 = i_outer_3*bls3 + i_inner_3
                            for k1 in range(0, 3, 1):
                                for k2 in range(0, 3, 1):
                                    for k3 in range(0, 3, 1):
                                        v2[i_inner_1, i_inner_2, i_inner_3] += mat22[1+i1,1+i2, 1+i3,k1,k2,k3]*xx2[i_inner_1+k1,i_inner_2+k2,i_inner_3+k3]

                out2[1 + i_outer_1*bls1:1+i_outer_1*bls1+bls1,1 + i_outer_2*bls2:1+i_outer_2*bls2+bls2,1 + i_outer_3*bls3:1+i_outer_3*bls3+bls3] = v2[:,:,:]
    #$ omp end parallel
    return

@types("float64[:,:,:,:,:,:]", "float64[:,:,:,:,:,:]", "float64[:,:,:,:,:,:]",
       "float64[:,:,:]", "float64[:,:,:]", "float64[:,:,:]",
       "float64[:,:,:]", "float64[:,:,:]", "float64[:,:,:]",
       "int64", "int64", "int64",
       "int64", "int64", "int64",
       "int64", "int64", "int64",
       "int64", "int64", "int64",
       "int64", "int64", "int64",
       "int64", "int64", "int64", "int64", "int64", "int64")
def dot_2(mat00, mat11, mat22,
                    x0, x1, x2,
                    out0, out1, out2, 
                    s00_1, s00_2, s00_3,
                    s11_1, s11_2, s11_3,
                    s22_1, s22_2, s22_3,
                    n00_1, n00_2, n00_3,
                    n11_1, n11_2, n11_3,
                    n22_1, n22_2, n22_3, bls1, bls2, bls3):

    #$ omp parallel default(private) shared(mat00, mat11, mat22,x0, x1, x2,out0, out1, out2) firstprivate( n00_1, n00_2, n00_3, n11_1, n11_2, n11_3, n22_1, n22_2, n22_3, bls1, bls2, bls3)
    bls     = 2
    xx0     = empty((4+bls1,4+bls2,4+bls3))
    v0      = empty((bls1, bls2, bls3))

    #$ omp for schedule(static) collapse(3) nowait
    for i_outer_1 in range(0, n00_1//bls1, 1):
        for i_outer_2 in range(0, n00_2//bls2, 1):
            for i_outer_3 in range(0, n00_3//bls3, 1):
                xx0[:,:,:]  = x0[i_outer_1*bls1:i_outer_1*bls1+4+bls1,i_outer_2*bls2:i_outer_2*bls2+4+bls2,i_outer_3*bls3:i_outer_3*bls3+4+bls3]
                v0[:,:,:]   = 0.
                for i_inner_1 in range(0, bls1, 1):
                    i1 = i_outer_1*bls1 + i_inner_1
                    for i_inner_2 in range(0, bls2, 1):
                        i2 = i_outer_2*bls2 + i_inner_2
                        for i_inner_3 in range(0, bls3, 1):
                            i3 = i_outer_3*bls3 + i_inner_3
                            for k1 in range(0, 5, 1):
                                for k2 in range(0, 5, 1):
                                    for k3 in range(0, 5, 1):
                                        v0[i_inner_1, i_inner_2, i_inner_3] += mat00[2+i1,2+i2,2+i3,k1,k2,k3]*xx0[i_inner_1+k1,i_inner_2+k2,i_inner_3+k3]

                out0[2 + i_outer_1*bls1:2+i_outer_1*bls1+bls1,2 + i_outer_2*bls2:2+i_outer_2*bls2+bls2,2 + i_outer_3*bls3:2+i_outer_3*bls3+bls3] = v0[:,:,:]

    xx1     = empty((4+bls1,4+bls2,4+bls3))
    v1      = empty((bls1,bls2,bls3))
    #$ omp for schedule(static) collapse(3) nowait
    for i_outer_1 in range(0, n11_1//bls1, 1):
        for i_outer_2 in range(0, n11_2//bls2, 1):
            for i_outer_3 in range(0, n11_3//bls3, 1):
                xx1[:,:,:]  = x1[i_outer_1*bls1:i_outer_1*bls1+4+bls1,i_outer_2*bls2:i_outer_2*bls2+4+bls2,i_outer_3*bls3:i_outer_3*bls3+4+bls3]
                v1[:,:,:]   = 0.
                for i_inner_1 in range(0, bls1, 1):
                    i1 = i_outer_1*bls1 + i_inner_1
                    for i_inner_2 in range(0, bls2, 1):
                        i2 = i_outer_2*bls2 + i_inner_2
                        for i_inner_3 in range(0, bls3, 1):
                            i3 = i_outer_3*bls3 + i_inner_3
                            for k1 in range(0, 5, 1):
                                for k2 in range(0, 5, 1):
                                    for k3 in range(0, 5, 1):
                                        v1[i_inner_1, i_inner_2, i_inner_3] += mat11[2+i1,2+i2,2+i3,k1,k2,k3]*xx1[i_inner_1+k1,i_inner_2+k2,i_inner_3+k3]

                out1[2 + i_outer_1*bls1:2+i_outer_1*bls1+bls1,2 + i_outer_2*bls2:2+i_outer_2*bls2+bls2,2 + i_outer_3*bls3:2+i_outer_3*bls3+bls3] = v1[:,:,:]

    xx2     = empty((4+bls1,4+bls2,4+bls3))
    v2      = empty((bls1,bls2,bls3))
    #$ omp for schedule(static) collapse(3) nowait
    for i_outer_1 in range(0, n22_1//bls1, 1):
        for i_outer_2 in range(0, n22_2//bls2, 1):
            for i_outer_3 in range(0, n22_3//bls3, 1):
                xx2[:,:,:]  = x2[i_outer_1*bls1:i_outer_1*bls1+4+bls1,i_outer_2*bls2:i_outer_2*bls2+4+bls2,i_outer_3*bls3:i_outer_3*bls3+4+bls3]
                v2[:,:,:]   = 0.
                for i_inner_1 in range(0, bls1, 1):
                    i1 = i_outer_1*bls1 + i_inner_1
                    for i_inner_2 in range(0, bls2, 1):
                        i2 = i_outer_2*bls2 + i_inner_2
                        for i_inner_3 in range(0, bls3, 1):
                            i3 = i_outer_3*bls3 + i_inner_3
                            for k1 in range(0, 5, 1):
                                for k2 in range(0, 5, 1):
                                    for k3 in range(0, 5, 1):
                                        v2[i_inner_1, i_inner_2, i_inner_3] += mat22[2+i1,2+i2,2+i3,k1,k2,k3]*xx2[i_inner_1+k1,i_inner_2+k2,i_inner_3+k3]

                out2[2 + i_outer_1*bls1:2+i_outer_1*bls1+bls1,2 + i_outer_2*bls2:2+i_outer_2*bls2+bls2,2 + i_outer_3*bls3:2+i_outer_3*bls3+bls3] = v2[:,:,:]
    #$ omp end parallel
    return

@types("float64[:,:,:,:,:,:]", "float64[:,:,:,:,:,:]", "float64[:,:,:,:,:,:]",
       "float64[:,:,:]", "float64[:,:,:]", "float64[:,:,:]",
       "float64[:,:,:]", "float64[:,:,:]", "float64[:,:,:]",
       "int64", "int64", "int64",
       "int64", "int64", "int64",
       "int64", "int64", "int64",
       "int64", "int64", "int64",
       "int64", "int64", "int64",
       "int64", "int64", "int64", "int64", "int64", "int64")
def dot_3(mat00, mat11, mat22,
                    x0, x1, x2,
                    out0, out1, out2, 
                    s00_1, s00_2, s00_3,
                    s11_1, s11_2, s11_3,
                    s22_1, s22_2, s22_3,
                    n00_1, n00_2, n00_3,
                    n11_1, n11_2, n11_3,
                    n22_1, n22_2, n22_3, bls1, bls2, bls3):


    #$ omp parallel default(private) shared(mat00, mat11, mat22,x0, x1, x2,out0, out1, out2) firstprivate( n00_1, n00_2, n00_3, n11_1, n11_2, n11_3, n22_1, n22_2, n22_3, bls1, bls2, bls3)
    xx0    = empty((6+bls1,6+bls2,6+bls3))
    v0     = empty((bls1, bls2, bls3))

    #$ omp for schedule(static) collapse(3) nowait
    for i_outer_1 in range(0, n00_1//bls1, 1):
        for i_outer_2 in range(0, n00_2//bls2, 1):
            for i_outer_3 in range(0, n00_3//bls3, 1):
                xx0[:,:,:]  = x0[i_outer_1*bls1:i_outer_1*bls1+6+bls1,i_outer_2*bls2:i_outer_2*bls2+6+bls2,i_outer_3*bls3:i_outer_3*bls3+6+bls3]
                v0[:,:,:]   = 0.
                for i_inner_1 in range(0, bls1, 1):
                    i1 = i_outer_1*bls1 + i_inner_1
                    for i_inner_2 in range(0, bls2, 1):
                        i2 = i_outer_2*bls2 + i_inner_2
                        for i_inner_3 in range(0, bls3, 1):
                            i3 = i_outer_3*bls3 + i_inner_3
                            for k1 in range(0, 7, 1):
                                for k2 in range(0, 7, 1):
                                    for k3 in range(0, 7, 1):
                                        v0[i_inner_1, i_inner_2, i_inner_3] += mat00[3+i1,3+i2,3+i3,k1,k2,k3]*xx0[i_inner_1+k1,i_inner_2+k2,i_inner_3+k3]

                out0[3 + i_outer_1*bls1:3+i_outer_1*bls1+bls1,3 + i_outer_2*bls2:3+i_outer_2*bls2+bls2,3 + i_outer_3*bls3:3+i_outer_3*bls3+bls3] = v0[:,:,:]

    xx1    = empty((6+bls1,6+bls2,6+bls3))
    v1     = empty((bls1, bls2, bls3))

    #$ omp for schedule(static) collapse(3) nowait
    for i_outer_1 in range(0, n11_1//bls1, 1):
        for i_outer_2 in range(0, n11_2//bls2, 1):
            for i_outer_3 in range(0, n11_3//bls3, 1):
                xx1[:,:,:]  = x1[i_outer_1*bls1:i_outer_1*bls1+6+bls1,i_outer_2*bls2:i_outer_2*bls2+6+bls2,i_outer_3*bls3:i_outer_3*bls3+6+bls3]
                v1[:,:,:]   = 0.
                for i_inner_1 in range(0, bls1, 1):
                    i1 = i_outer_1*bls1 + i_inner_1
                    for i_inner_2 in range(0, bls2, 1):
                        i2 = i_outer_2*bls2 + i_inner_2
                        for i_inner_3 in range(0, bls3, 1):
                            i3 = i_outer_3*bls3 + i_inner_3
                            for k1 in range(0, 7, 1):
                                for k2 in range(0, 7, 1):
                                    for k3 in range(0, 7, 1):
                                        v1[i_inner_1, i_inner_2, i_inner_3] += mat11[3+i1,3+i2,3+i3,k1,k2,k3]*xx1[i_inner_1+k1,i_inner_2+k2,i_inner_3+k3]

                out1[3 + i_outer_1*bls1:3+i_outer_1*bls1+bls1,3 + i_outer_2*bls2:3+i_outer_2*bls2+bls2,3 + i_outer_3*bls3:3+i_outer_3*bls3+bls3] = v1[:,:,:]

    xx2    = empty((6+bls1,6+bls2,6+bls3))
    v2     = empty((bls1, bls2, bls3))

    #$ omp for schedule(static) collapse(3) nowait
    for i_outer_1 in range(0, n22_1//bls1, 1):
        for i_outer_2 in range(0, n22_2//bls2, 1):
            for i_outer_3 in range(0, n22_3//bls3, 1):
                xx2[:,:,:]  = x2[i_outer_1*bls1:i_outer_1*bls1+6+bls1,i_outer_2*bls2:i_outer_2*bls2+6+bls2,i_outer_3*bls3:i_outer_3*bls3+6+bls3]
                v2[:,:,:]   = 0.
                for i_inner_1 in range(0, bls1, 1):
                    i1 = i_outer_1*bls1 + i_inner_1
                    for i_inner_2 in range(0, bls2, 1):
                        i2 = i_outer_2*bls2 + i_inner_2
                        for i_inner_3 in range(0, bls3, 1):
                            i3 = i_outer_3*bls3 + i_inner_3
                            for k1 in range(0, 7, 1):
                                for k2 in range(0, 7, 1):
                                    for k3 in range(0, 7, 1):
                                        v2[i_inner_1, i_inner_2, i_inner_3] += mat22[3+i1,3+i2,3+i3,k1,k2,k3]*xx2[i_inner_1+k1,i_inner_2+k2,i_inner_3+k3]

                out2[3 + i_outer_1*bls1:3+i_outer_1*bls1+bls1,3 + i_outer_2*bls2:3+i_outer_2*bls2+bls2,3 + i_outer_3*bls3:3+i_outer_3*bls3+bls3] = v2[:,:,:]
    #$ omp end parallel
    return

@types("float64[:,:,:,:,:,:]", "float64[:,:,:,:,:,:]", "float64[:,:,:,:,:,:]",
       "float64[:,:,:]", "float64[:,:,:]", "float64[:,:,:]",
       "float64[:,:,:]", "float64[:,:,:]", "float64[:,:,:]",
       "int64", "int64", "int64",
       "int64", "int64", "int64",
       "int64", "int64", "int64",
       "int64", "int64", "int64",
       "int64", "int64", "int64",
       "int64", "int64", "int64","int64", "int64", "int64")
def dot_4(mat00, mat11, mat22,
                    x0, x1, x2,
                    out0, out1, out2, 
                    s00_1, s00_2, s00_3,
                    s11_1, s11_2, s11_3,
                    s22_1, s22_2, s22_3,
                    n00_1, n00_2, n00_3,
                    n11_1, n11_2, n11_3,
                    n22_1, n22_2, n22_3, bls1, bls2, bls3):

    #$ omp parallel default(private) shared(mat00, mat11, mat22,x0, x1, x2,out0, out1, out2) firstprivate( n00_1, n00_2, n00_3, n11_1, n11_2, n11_3, n22_1, n22_2, n22_3, bls1, bls2, bls3)
    xx0     = empty((8+bls1,8+bls2,8+bls3))
    v0      = empty((bls1, bls2, bls3))

    #$ omp for schedule(static) collapse(3) nowait
    for i_outer_1 in range(0, n00_1//bls1, 1):
        for i_outer_2 in range(0, n00_2//bls2, 1):
            for i_outer_3 in range(0, n00_3//bls3, 1):
                xx0[:,:,:]  = x0[i_outer_1*bls1:i_outer_1*bls1+8+bls1,i_outer_2*bls2:i_outer_2*bls2+8+bls2,i_outer_3*bls3:i_outer_3*bls3+8+bls3]
                v0[:,:,:]   = 0.
                for i_inner_1 in range(0, bls1, 1):
                    i1 = i_outer_1*bls1 + i_inner_1
                    for i_inner_2 in range(0, bls2, 1):
                        i2 = i_outer_2*bls2 + i_inner_2
                        for i_inner_3 in range(0, bls3, 1):
                            i3 = i_outer_3*bls3 + i_inner_3
                            for k1 in range(0, 9, 1):
                                for k2 in range(0, 9, 1):
                                    for k3 in range(0, 9, 1):
                                        v0[i_inner_1, i_inner_2, i_inner_3] += mat00[4+i1,4+i2,4+i3,k1,k2,k3]*xx0[i_inner_1+k1,i_inner_2+k2,i_inner_3+k3]

                out0[4 + i_outer_1*bls1:4+i_outer_1*bls1+bls1,4 + i_outer_2*bls2:4+i_outer_2*bls2+bls2,4 + i_outer_3*bls3:4+i_outer_3*bls3+bls3] = v0[:,:,:]

    xx1     = empty((8+bls1,8+bls2,8+bls3))
    v1      = empty((bls1,bls2,bls3))

    #$ omp for schedule(static) collapse(3) nowait
    for i_outer_1 in range(0, n11_1//bls1, 1):
        for i_outer_2 in range(0, n11_2//bls2, 1):
            for i_outer_3 in range(0, n11_3//bls3, 1):
                xx1[:,:,:]  = x1[i_outer_1*bls1:i_outer_1*bls1+8+bls1,i_outer_2*bls2:i_outer_2*bls2+8+bls2,i_outer_3*bls3:i_outer_3*bls3+8+bls3]
                v1[:,:,:]   = 0.
                for i_inner_1 in range(0, bls1, 1):
                    i1 = i_outer_1*bls1 + i_inner_1
                    for i_inner_2 in range(0, bls2, 1):
                        i2 = i_outer_2*bls2 + i_inner_2
                        for i_inner_3 in range(0, bls3, 1):
                            i3 = i_outer_3*bls3 + i_inner_3
                            for k1 in range(0, 9, 1):
                                for k2 in range(0, 9, 1):
                                    for k3 in range(0, 9, 1):
                                        v1[i_inner_1, i_inner_2, i_inner_3] += mat11[4+i1,4+i2,4+i3,k1,k2,k3]*xx1[i_inner_1+k1,i_inner_2+k2,i_inner_3+k3]

                out1[4 + i_outer_1*bls1:4+i_outer_1*bls1+bls1,4 + i_outer_2*bls2:4+i_outer_2*bls2+bls2,4 + i_outer_3*bls3:4+i_outer_3*bls3+bls3] = v1[:,:,:]

    xx2     = empty((8+bls1,8+bls2,8+bls3))
    v2      = empty((bls1,bls2,bls3))

    #$ omp for schedule(static) collapse(3) nowait
    for i_outer_1 in range(0, n22_1//bls1, 1):
        for i_outer_2 in range(0, n22_2//bls2, 1):
            for i_outer_3 in range(0, n22_3//bls3, 1):
                xx2[:,:,:]  = x2[i_outer_1*bls1:i_outer_1*bls1+8+bls1,i_outer_2*bls2:i_outer_2*bls2+8+bls2,i_outer_3*bls3:i_outer_3*bls3+8+bls3]
                v2[:,:,:]   = 0.
                for i_inner_1 in range(0, bls1, 1):
                    i1 = i_outer_1*bls1 + i_inner_1
                    for i_inner_2 in range(0, bls2, 1):
                        i2 = i_outer_2*bls2 + i_inner_2
                        for i_inner_3 in range(0, bls3, 1):
                            i3 = i_outer_3*bls3 + i_inner_3
                            for k1 in range(0, 9, 1):
                                for k2 in range(0, 9, 1):
                                    for k3 in range(0, 9, 1):
                                        v2[i_inner_1, i_inner_2, i_inner_3] += mat22[4+i1,4+i2,4+i3,k1,k2,k3]*xx2[i_inner_1+k1,i_inner_2+k2,i_inner_3+k3]

                out2[4 + i_outer_1*bls1:4+i_outer_1*bls1+bls1,4 + i_outer_2*bls2:4+i_outer_2*bls2+bls2,4 + i_outer_3*bls3:4+i_outer_3*bls3+bls3] = v2[:,:,:]

    #$ omp end parallel
    return

@types("float64[:,:,:,:,:,:]", "float64[:,:,:,:,:,:]", "float64[:,:,:,:,:,:]",
       "float64[:,:,:]", "float64[:,:,:]", "float64[:,:,:]",
       "float64[:,:,:]", "float64[:,:,:]", "float64[:,:,:]",
       "int64", "int64", "int64",
       "int64", "int64", "int64",
       "int64", "int64", "int64",
       "int64", "int64", "int64",
       "int64", "int64", "int64",
       "int64", "int64", "int64", "int64", "int64", "int64")
def dot_5(mat00, mat11, mat22,
                    x0, x1, x2,
                    out0, out1, out2, 
                    s00_1, s00_2, s00_3,
                    s11_1, s11_2, s11_3,
                    s22_1, s22_2, s22_3,
                    n00_1, n00_2, n00_3,
                    n11_1, n11_2, n11_3,
                    n22_1, n22_2, n22_3, bls1, bls2, bls3):

    #$ omp parallel default(private) shared(mat00, mat11, mat22,x0, x1, x2,out0, out1, out2) firstprivate( n00_1, n00_2, n00_3, n11_1, n11_2, n11_3, n22_1, n22_2, n22_3, bls1, bls2, bls3)
    xx0     = empty((10+bls1,10+bls2,10+bls3))
    v0      = empty((bls1, bls2, bls3))

    #$ omp for schedule(static) collapse(3) nowait
    for i_outer_1 in range(0, n00_1//bls1, 1):
        for i_outer_2 in range(0, n00_2//bls2, 1):
            for i_outer_3 in range(0, n00_3//bls3, 1):
                xx0[:,:,:]  = x0[i_outer_1*bls1:i_outer_1*bls1+10+bls1,i_outer_2*bls2:i_outer_2*bls2+10+bls2,i_outer_3*bls3:i_outer_3*bls3+10+bls3]
                v0[:,:,:]   = 0.
                for i_inner_1 in range(0, bls1, 1):
                    i1 = i_outer_1*bls1 + i_inner_1
                    for i_inner_2 in range(0, bls2, 1):
                        i2 = i_outer_2*bls2 + i_inner_2
                        for i_inner_3 in range(0, bls3, 1):
                            i3 = i_outer_3*bls3 + i_inner_3
                            for k1 in range(0, 11, 1):
                                for k2 in range(0, 11, 1):
                                    for k3 in range(0, 11, 1):
                                        v0[i_inner_1, i_inner_2, i_inner_3] += mat00[5+i1,5+i2,5+i3,k1,k2,k3]*xx0[i_inner_1+k1,i_inner_2+k2,i_inner_3+k3]

                out0[5 + i_outer_1*bls1:5+i_outer_1*bls1+bls1,5 + i_outer_2*bls2:5+i_outer_2*bls2+bls2,5 + i_outer_3*bls3:5+i_outer_3*bls3+bls3] = v0[:,:,:]

    xx1     = empty((10+bls1,10+bls2,10+bls3))
    v1      = empty((bls1,bls2,bls3))

    #$ omp for schedule(static) collapse(3) nowait
    for i_outer_1 in range(0, n11_1//bls1, 1):
        for i_outer_2 in range(0, n11_2//bls2, 1):
            for i_outer_3 in range(0, n11_3//bls3, 1):
                xx1[:,:,:]  = x1[i_outer_1*bls1:i_outer_1*bls1+10+bls1,i_outer_2*bls2:i_outer_2*bls2+10+bls2,i_outer_3*bls3:i_outer_3*bls3+10+bls3]
                v1[:,:,:]   = 0.
                for i_inner_1 in range(0, bls1, 1):
                    i1 = i_outer_1*bls1 + i_inner_1
                    for i_inner_2 in range(0, bls2, 1):
                        i2 = i_outer_2*bls2 + i_inner_2
                        for i_inner_3 in range(0, bls3, 1):
                            i3 = i_outer_3*bls3 + i_inner_3
                            for k1 in range(0, 11, 1):
                                for k2 in range(0, 11, 1):
                                    for k3 in range(0, 11, 1):
                                        v1[i_inner_1, i_inner_2, i_inner_3] += mat11[5+i1,5+i2,5+i3,k1,k2,k3]*xx1[i_inner_1+k1,i_inner_2+k2,i_inner_3+k3]

                out1[5 + i_outer_1*bls1:5+i_outer_1*bls1+bls1,5 + i_outer_2*bls2:5+i_outer_2*bls2+bls2,5 + i_outer_3*bls3:5+i_outer_3*bls3+bls3] = v1[:,:,:]

    xx2     = empty((10+bls1,10+bls2,10+bls3))
    v2      = empty((bls1,bls2,bls3))

    #$ omp for schedule(static) collapse(3) nowait
    for i_outer_1 in range(0, n22_1//bls1, 1):
        for i_outer_2 in range(0, n22_2//bls2, 1):
            for i_outer_3 in range(0, n22_3//bls3, 1):
                xx2[:,:,:]  = x2[i_outer_1*bls1:i_outer_1*bls1+10+bls1,i_outer_2*bls2:i_outer_2*bls2+10+bls2,i_outer_3*bls3:i_outer_3*bls3+10+bls3]
                v2[:,:,:]   = 0.
                for i_inner_1 in range(0, bls1, 1):
                    i1 = i_outer_1*bls1 + i_inner_1
                    for i_inner_2 in range(0, bls2, 1):
                        i2 = i_outer_2*bls2 + i_inner_2
                        for i_inner_3 in range(0, bls3, 1):
                            i3 = i_outer_3*bls3 + i_inner_3
                            for k1 in range(0, 11, 1):
                                for k2 in range(0, 11, 1):
                                    for k3 in range(0, 11, 1):
                                        v2[i_inner_1, i_inner_2, i_inner_3] += mat22[5+i1,5+i2,5+i3,k1,k2,k3]*xx2[i_inner_1+k1,i_inner_2+k2,i_inner_3+k3]

                out2[5 + i_outer_1*bls1:5+i_outer_1*bls1+bls1,5 + i_outer_2*bls2:5+i_outer_2*bls2+bls2,5 + i_outer_3*bls3:5+i_outer_3*bls3+bls3] = v2[:,:,:]
    #$ omp end parallel
    return

