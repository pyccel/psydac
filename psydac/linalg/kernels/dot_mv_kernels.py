
from pyccel.decorators import template
@template(name='T', types=['float[:]', 'complex[:]'])
@template(name='Tarray', types=['float[:,:]', 'complex[:,:]'])
def Mv_product_1d(mat00:'Tarray', x0:'T', out0:'T', starts: 'int64[:]', nrows: 'int64[:]', nrows_extra: 'int64[:]',
                  dm:'int64[:]', cm:'int64[:]', diff:'int64[:]', bb:'int64[:]', ndiags:'int64[:]', gpads: 'int64[:]'):

    nrows1 = nrows[0]

    dstart1 = starts[0]

    dshift1 = dm[0]

    cshift1 = cm[0]

    diff1 = diff[0]

    bb1 = bb[0]

    ndiags1 = ndiags[0]

    dpads1 = gpads[0]

    pxm1 = dpads1*cshift1

    for i1 in range(0, nrows1, 1):
        v00 = mat00[0,0]-mat00[0,0]+x0[0]-x0[0]
        temp1 = bb1 - diff1 + (i1 + dstart1 % dshift1) // cshift1 * dshift1
        for k1 in range(0, ndiags1, 1):
            v00 += mat00[pxm1 + i1, k1] * x0[k1 + temp1]

        out0[pxm1 + i1] = v00

    if 0<nrows_extra[0]:
        for i1 in range(0, nrows_extra[0], 1):
            v00 = mat00[0,0]-mat00[0,0]+x0[0]-x0[0]
            temp1 = bb1 - diff1 + (i1 + nrows1 + dstart1 % dshift1) // cshift1 * dshift1
            for k1 in range(0, ndiags1-i1-1, 1):
                v00 += mat00[nrows1 + pxm1 + i1, k1] * x0[temp1 + k1]

            out0[pxm1 + nrows1 + i1] = v00

    return


@template(name='T', types=['float[:,:]', 'complex[:,:]'])
@template(name='Tarray', types=['float[:,:,:,:]', 'complex[:,:,:,:]'])
def Mv_product_2d(mat00:'Tarray', x0:'T', out0:'T', starts:'int64[:]', nrows:'int64[:]', nrows_extra:'int64[:]',
                  dm:'int64[:]', cm:'int64[:]', diff:'int64[:]', bb:'int64[:]', ndiags:'int64[:]', gpads: 'int64[:]'):

    nrows1  = nrows[0]
    nrows2  = nrows[1]

    dstart1 = starts[0]
    dstart2 = starts[1]


    dshift1 = dm[0]
    dshift2 = dm[1]

    cshift1 = cm[0]
    cshift2 = cm[1]

    diff1 = diff[0]
    diff2 = diff[1]

    bb1 = bb[0]
    bb2 = bb[1]

    ndiags1 = ndiags[0]
    ndiags2 = ndiags[1]

    dpads1 = gpads[0]
    dpads2 = gpads[1]

    pxm1 = dpads1*cshift1
    pxm2 = dpads2*cshift2

    for i1 in range(0, nrows1, 1):
        for i2 in range(0, nrows2, 1):
            v00 = mat00[0,0,0,0]-mat00[0,0,0,0]+x0[0,0]-x0[0,0]
            temp1 = bb1 - diff1 + (i1 + dstart1 % dshift1) // cshift1 * dshift1
            temp2 = bb2 - diff2 + (i2 + dstart2 % dshift2) // cshift2 * dshift2
            for k1 in range(0, ndiags1, 1):
                for k2 in range(0, ndiags2, 1):
                    v00 += mat00[pxm1 + i1, pxm2 + i2, k1, k2] * x0[k1 + temp1,k2 + temp2]

            out0[pxm1 + i1, pxm2 + i2] = v00

    if 0<nrows_extra[0]:
        for i1 in range(0, nrows_extra[0], 1):
            for i2 in range(0, nrows2, 1):
                v00 = mat00[0,0,0,0]-mat00[0,0,0,0]+x0[0,0]-x0[0,0]
                temp1 = bb1 - diff1 + (i1 + nrows1 + dstart1 % dshift1) // cshift1 * dshift1
                temp2 = bb2 - diff2 + (i2 +          dstart2 % dshift2) // cshift2 * dshift2
                for k1 in range(0, ndiags1-i1-1, 1):
                    for k2 in range(0, ndiags2, 1):
                        v00 += mat00[nrows1 + pxm1 + i1, pxm2 + i2, k1, k2] * x0[temp1 + k1, temp2 + k2]

                out0[pxm1 + nrows1 + i1,  pxm2 + i2] = v00

    if 0<nrows_extra[1]:
        for i1 in range(0, nrows1+nrows_extra[0], 1):
            for i2 in range(0, nrows_extra[1], 1):
                v00 = mat00[0, 0, 0, 0] - mat00[0, 0, 0, 0] + x0[0, 0] - x0[0, 0]
                temp1 = bb1 - diff1 + (i1 +          dstart1 % dshift1) // cshift1 * dshift1
                temp2 = bb2 - diff2 + (i2 + nrows2 + dstart2 % dshift2) // cshift2 * dshift2
                for k1 in range(0, ndiags1 - max(0, i1+1-nrows1+nrows_extra[0]), 1):
                    for k2 in range(0, ndiags2-i2-1, 1):
                        v00 +=mat00[pxm1 + i1, nrows2 + pxm2 + i2, k1, k2] * x0[temp1 + k1, temp2 + k2]

                out0[pxm1 + i1, pxm2 + nrows2 + i2] = v00

    return

@template(name='T', types=['float[:,:,:]', 'complex[:,:,:]'])
@template(name='Tarray', types=['float[:,:,:,:,:,:]', 'complex[:,:,:,:,:,:]'])
def Mv_product_3d(mat00:'Tarray', x0:'T', out0:'T', starts:'int64[:]', nrows:'int64[:]', nrows_extra:'int64[:]',
                  dm:'int64[:]', cm:'int64[:]', diff:'int64[:]', bb:'int64[:]', ndiags:'int64[:]', gpads: 'int64[:]'):

    nrows1  = nrows[0]
    nrows2  = nrows[1]
    nrows3  = nrows[2]

    dstart1 = starts[0]
    dstart2 = starts[1]
    dstart3 = starts[2]

    dshift1 = dm[0]
    dshift2 = dm[1]
    dshift3 = dm[2]

    cshift1 = cm[0]
    cshift2 = cm[1]
    cshift3 = cm[2]

    diff1 = diff[0]
    diff2 = diff[1]
    diff3 = diff[2]

    bb1 = bb[0]
    bb2 = bb[1]
    bb3 = bb[2]

    ndiags1 = ndiags[0]
    ndiags2 = ndiags[1]
    ndiags3 = ndiags[2]

    dpads1 = gpads[0]
    dpads2 = gpads[1]
    dpads3 = gpads[2]

    pxm1 = dpads1*cshift1
    pxm2 = dpads2*cshift2
    pxm3 = dpads3*cshift3

    for i1 in range(0, nrows1, 1):
        for i2 in range(0, nrows2, 1):
            for i3 in range(0, nrows3, 1):
                v00 = mat00[0,0,0,0,0,0]-mat00[0,0,0,0,0,0]+x0[0,0,0]-x0[0,0,0]
                temp1 = bb1 - diff1 + (i1 + dstart1 % dshift1) // cshift1 * dshift1
                temp2 = bb2 - diff2 + (i2 + dstart2 % dshift2) // cshift2 * dshift2
                temp3 = bb3 - diff3 + (i3 + dstart3 % dshift3) // cshift3 * dshift3
                for k1 in range(0, ndiags1, 1):
                    for k2 in range(0, ndiags2, 1):
                        for k3 in range(0, ndiags3, 1):
                            v00 += mat00[pxm1 + i1, pxm2 + i2, pxm3 + i3, k1, k2, k3] * x0[k1 + temp1, k2 + temp2, k3 + temp3]

                out0[pxm1 + i1, pxm2 + i2, pxm3 + i3] = v00

    if 0<nrows_extra[0]:
        for i1 in range(0, nrows_extra[0], 1):
            for i2 in range(0, nrows2, 1):
                for i3 in range(0, nrows3, 1):
                    v00 = mat00[0,0,0,0,0,0]-mat00[0,0,0,0,0,0]+x0[0,0,0]-x0[0,0,0]
                    temp1 = bb1 - diff1 + (i1 + nrows1 + dstart1 % dshift1) // cshift1 * dshift1
                    temp2 = bb2 - diff2 + (i2 +          dstart2 % dshift2) // cshift2 * dshift2
                    temp3 = bb3 - diff3 + (i3 +          dstart3 % dshift3) // cshift3 * dshift3
                    for k1 in range(0, ndiags1-i1-1, 1):
                        for k2 in range(0, ndiags2, 1):
                            for k3 in range(0, ndiags3, 1):
                                v00 += mat00[nrows1 + pxm1 + i1, pxm2 + i2, pxm3 + i3, k1, k2, k3] *  x0[temp1 + k1, temp2 + k2, temp3 + k3]

                    out0[pxm1 + nrows1 + i1,  pxm2 + i2,  pxm3 + i3] = v00

    if 0<nrows_extra[1]:
        for i1 in range(0, nrows1+nrows_extra[0], 1):
            for i2 in range(0, nrows_extra[1], 1):
                for i3 in range(0, nrows3, 1):
                    v00 = mat00[0,0,0,0,0,0]-mat00[0,0,0,0,0,0]+x0[0,0,0]-x0[0,0,0]
                    temp1 = bb1 - diff1 + (i1 +          dstart1 % dshift1) // cshift1 * dshift1
                    temp2 = bb2 - diff2 + (i2 + nrows2 + dstart2 % dshift2) // cshift2 * dshift2
                    temp3 = bb3 - diff3 + (i3 +          dstart3 % dshift3) // cshift3 * dshift3
                    for k1 in range(0, ndiags1 - max(0, i1+1-nrows1+nrows_extra[0]), 1):
                        for k2 in range(0, ndiags2-i2-1, 1):
                            for k3 in range(0, ndiags3, 1):
                                v00 +=mat00[pxm1 + i1, nrows2 + pxm2 + i2, pxm3 + i3, k1, k2, k3] * x0[temp1 + k1, temp2 + k2, temp3 + k3]

                    out0[pxm1 + i1, pxm2 + nrows2 + i2, pxm3 + i3] = v00

    if 0 < nrows_extra[2]:
        for i1 in range(0, nrows1 + nrows_extra[0], 1):
            for i2 in range(0, nrows2 + nrows_extra[1], 1):
                for i3 in range(0, nrows_extra[2], 1):
                    v00 = mat00[0,0,0,0,0,0]-mat00[0,0,0,0,0,0]+x0[0,0,0]-x0[0,0,0]
                    temp1 = bb1 - diff1 + (i1 +          dstart1 % dshift1) // cshift1 * dshift1
                    temp2 = bb2 - diff2 + (i2 +          dstart2 % dshift2) // cshift2 * dshift2
                    temp3 = bb3 - diff3 + (i3 + nrows2 + dstart3 % dshift3) // cshift3 * dshift3
                    for k1 in range(0, ndiags1 - max(0, i1 + 1 - nrows1 + nrows_extra[0]), 1):
                        for k2 in range(0, ndiags2 - max(0, i2 + 1 - nrows2 + nrows_extra[1]), 1):
                            for k3 in range(0, ndiags3 - i3 -1 , 1):
                                v00 +=mat00[pxm1 + i1, pxm2 + i2, nrows3 + pxm3 + i3, k1, k2, k3] * x0[temp1 + k1, temp2 + k2, temp3 + k3]

                    out0[pxm1 + i1, pxm2 + i2, nrows3 + pxm3 + i3] = v00

    return
