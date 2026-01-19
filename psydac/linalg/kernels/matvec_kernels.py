#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from typing import TypeVar

T = TypeVar('T', float, complex)

def matvec_1d(mat00:'T[:,:]', x0:'T[:]', out0:'T[:]', starts: 'int64[:]', nrows: 'int64[:]', nrows_extra: 'int64[:]',
                  dm:'int64[:]', cm:'int64[:]', pad_imp:'int64[:]', ndiags:'int64[:]', gpads: 'int64[:]'):

    nrows1   = nrows[0]
    dstart1  = starts[0]
    dshift1  = dm[0]
    cshift1  = cm[0]
    ndiags1  = ndiags[0]
    dpads1   = gpads[0]
    pad_imp1 = pad_imp[0]

    pxm1 = dpads1 * cshift1

    start_impact1 = dstart1 % dshift1

    v00 = mat00[0, 0] - mat00[0, 0] + x0[0] - x0[0]

    for i1 in range(nrows1):
        v00 *= 0
        x_min1 = pad_imp1 + (i1 + start_impact1) // cshift1 * dshift1
        for k1 in range(ndiags1):
            v00 += mat00[pxm1 + i1, k1] * x0[k1 + x_min1]
        out0[pxm1 + i1] = v00

    if 0 < nrows_extra[0]:
        pxm1          += nrows1
        start_impact1 += nrows1
        for i1 in range(nrows_extra[0]):
            v00 *= 0
            x_min1 = pad_imp1 + (i1 + start_impact1) // cshift1 * dshift1
            for k1 in range(ndiags1 - i1 - 1):
                v00 += mat00[pxm1 + i1, k1] * x0[x_min1 + k1]
            out0[pxm1 + i1] = v00



def matvec_2d(mat00:'T[:,:,:,:]', x0:'T[:,:]', out0:'T[:,:]', starts:'int64[:]', nrows:'int64[:]', nrows_extra:'int64[:]',
                  dm:'int64[:]', cm:'int64[:]', pad_imp:'int64[:]', ndiags:'int64[:]', gpads: 'int64[:]'):

    nrows1   = nrows[0]
    nrows2   = nrows[1]
    dstart1  = starts[0]
    dstart2  = starts[1]
    dshift1  = dm[0]
    dshift2  = dm[1]
    cshift1  = cm[0]
    cshift2  = cm[1]
    ndiags1  = ndiags[0]
    ndiags2  = ndiags[1]
    dpads1   = gpads[0]
    dpads2   = gpads[1]
    pad_imp1 = pad_imp[0]
    pad_imp2 = pad_imp[1]

    pxm1 = dpads1 * cshift1
    pxm2 = dpads2 * cshift2

    start_impact1 = dstart1 % dshift1
    start_impact2 = dstart2 % dshift2

    v00 = mat00[0, 0, 0, 0] - mat00[0, 0, 0, 0] + x0[0, 0] - x0[0, 0]

    for i1 in range(nrows1):
        for i2 in range(nrows2):
            v00 *= 0
            x_min1 = pad_imp1 + (i1 + start_impact1) // cshift1 * dshift1
            x_min2 = pad_imp2 + (i2 + start_impact2) // cshift2 * dshift2
            for k1 in range(ndiags1):
                for k2 in range(ndiags2):
                    v00 += mat00[pxm1 + i1, pxm2 + i2, k1, k2] * x0[k1 + x_min1, k2 + x_min2]
            out0[pxm1 + i1, pxm2 + i2] = v00

    if 0 < nrows_extra[0]:
        pxm1          += nrows1
        start_impact1 += nrows1
        for i1 in range(nrows_extra[0]):
            for i2 in range(nrows2):
                v00 *= 0
                x_min1 = pad_imp1 + (i1 + start_impact1) // cshift1 * dshift1
                x_min2 = pad_imp2 + (i2 + start_impact2) // cshift2 * dshift2
                for k1 in range(ndiags1 - i1 - 1):
                    for k2 in range(ndiags2):
                        v00 += mat00[pxm1 + i1, pxm2 + i2, k1, k2] * x0[x_min1 + k1, x_min2 + k2]
                out0[pxm1 + i1,  pxm2 + i2] = v00

    if 0 < nrows_extra[1]:
        pxm1           = dpads1  * cshift1
        start_impact1  = dstart1 % dshift1
        pxm2          += nrows2
        start_impact2 += nrows2
        for i1 in range(nrows1 + nrows_extra[0]):
            for i2 in range(nrows_extra[1]):
                v00 *= 0
                x_min1 = pad_imp1 + (i1 + start_impact1) // cshift1 * dshift1
                x_min2 = pad_imp2 + (i2 + start_impact2) // cshift2 * dshift2
                for k1 in range(ndiags1 - max(0, i1 + 1 - nrows1)):
                    for k2 in range(ndiags2 - i2 - 1):
                        v00 += mat00[pxm1 + i1, pxm2 + i2, k1, k2] * x0[x_min1 + k1, x_min2 + k2]
                out0[pxm1 + i1, pxm2 + i2] = v00


def matvec_3d(mat00:'T[:,:,:,:,:,:]', x0:'T[:,:,:]', out0:'T[:,:,:]', starts:'int64[:]', nrows:'int64[:]', nrows_extra:'int64[:]',
                  dm:'int64[:]', cm:'int64[:]', pad_imp:'int64[:]', ndiags:'int64[:]', gpads: 'int64[:]'):

    nrows1   = nrows[0]
    nrows2   = nrows[1]
    nrows3   = nrows[2]
    dstart1  = starts[0]
    dstart2  = starts[1]
    dstart3  = starts[2]
    dshift1  = dm[0]
    dshift2  = dm[1]
    dshift3  = dm[2]
    cshift1  = cm[0]
    cshift2  = cm[1]
    cshift3  = cm[2]
    ndiags1  = ndiags[0]
    ndiags2  = ndiags[1]
    ndiags3  = ndiags[2]
    dpads1   = gpads[0]
    dpads2   = gpads[1]
    dpads3   = gpads[2]
    pad_imp1 = pad_imp[0]
    pad_imp2 = pad_imp[1]
    pad_imp3 = pad_imp[2]

    pxm1 = dpads1 * cshift1
    pxm2 = dpads2 * cshift2
    pxm3 = dpads3 * cshift3

    start_impact1 = dstart1 % dshift1
    start_impact2 = dstart2 % dshift2
    start_impact3 = dstart3 % dshift3

    v00 = mat00[0, 0, 0, 0, 0, 0] - mat00[0, 0, 0, 0, 0, 0] + x0[0, 0, 0] - x0[0, 0, 0]

    for i1 in range(nrows1):
        for i2 in range(nrows2):
            for i3 in range(nrows3):
                v00 *= 0
                x_min1 = pad_imp1 + (i1 + start_impact1) // cshift1 * dshift1
                x_min2 = pad_imp2 + (i2 + start_impact2) // cshift2 * dshift2
                x_min3 = pad_imp3 + (i3 + start_impact3) // cshift3 * dshift3
                for k1 in range(ndiags1):
                    for k2 in range(ndiags2):
                        for k3 in range(ndiags3):
                            v00 += mat00[pxm1 + i1, pxm2 + i2, pxm3 + i3, k1, k2, k3] * x0[k1 + x_min1, k2 + x_min2, k3 + x_min3]
                out0[pxm1 + i1, pxm2 + i2, pxm3 + i3] = v00

    if 0 < nrows_extra[0]:
        pxm1 += nrows1
        start_impact1 += nrows1
        for i1 in range(nrows_extra[0]):
            for i2 in range(nrows2):
                for i3 in range(nrows3):
                    v00 *= 0
                    x_min1 = pad_imp1 + (i1 + start_impact1) // cshift1 * dshift1
                    x_min2 = pad_imp2 + (i2 + start_impact2) // cshift2 * dshift2
                    x_min3 = pad_imp3 + (i3 + start_impact3) // cshift3 * dshift3
                    for k1 in range(ndiags1 - i1 - 1):
                        for k2 in range(ndiags2):
                            for k3 in range(ndiags3):
                                v00 += mat00[pxm1 + i1, pxm2 + i2, pxm3 + i3, k1, k2, k3] *  x0[x_min1 + k1, x_min2 + k2, x_min3 + k3]
                    out0[pxm1 + i1,  pxm2 + i2,  pxm3 + i3] = v00

    if 0 < nrows_extra[1]:
        pxm1           = dpads1  * cshift1
        start_impact1  = dstart1 % dshift1
        pxm2          += nrows2
        start_impact2 += nrows2
        for i1 in range(nrows1 + nrows_extra[0]):
            for i2 in range(nrows_extra[1]):
                for i3 in range(nrows3):
                    v00 *= 0
                    x_min1 = pad_imp1 + (i1 + start_impact1) // cshift1 * dshift1
                    x_min2 = pad_imp2 + (i2 + start_impact2) // cshift2 * dshift2
                    x_min3 = pad_imp3 + (i3 + start_impact3) // cshift3 * dshift3
                    for k1 in range(ndiags1 - max(0, i1 + 1 - nrows1)):
                        for k2 in range(ndiags2 - i2 - 1):
                            for k3 in range(ndiags3):
                                v00 += mat00[pxm1 + i1, pxm2 + i2, pxm3 + i3, k1, k2, k3] * x0[x_min1 + k1, x_min2 + k2, x_min3 + k3]
                    out0[pxm1 + i1, pxm2 + i2, pxm3 + i3] = v00

    if 0 < nrows_extra[2]:
        pxm1           = dpads1  * cshift1
        pxm2           = dpads2  * cshift2
        start_impact1  = dstart1 % dshift1
        start_impact2  = dstart2 % dshift2
        pxm3          += nrows3
        start_impact3 += nrows3
        for i1 in range(nrows1 + nrows_extra[0]):
            for i2 in range(nrows2 + nrows_extra[1]):
                for i3 in range(nrows_extra[2]):
                    v00 *= 0
                    x_min1 = pad_imp1 + (i1 + start_impact1) // cshift1 * dshift1
                    x_min2 = pad_imp2 + (i2 + start_impact2) // cshift2 * dshift2
                    x_min3 = pad_imp3 + (i3 + start_impact3) // cshift3 * dshift3
                    for k1 in range(ndiags1 - max(0, i1 + 1 - nrows1)):
                        for k2 in range(ndiags2 - max(0, i2 + 1 - nrows2)):
                            for k3 in range(ndiags3 - i3 - 1):
                                v00 += mat00[pxm1 + i1, pxm2 + i2, pxm3 + i3, k1, k2, k3] * x0[x_min1 + k1, x_min2 + k2, x_min3 + k3]
                    out0[pxm1 + i1, pxm2 + i2, pxm3 + i3] = v00
