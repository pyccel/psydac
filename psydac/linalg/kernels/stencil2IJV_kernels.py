#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from typing import TypeVar

T = TypeVar('T', float, complex)

#========================================================================================================
def stencil2IJV_1d_C(A:'T[:,:]', Ib:'int64[:]', Jb:'int64[:]', Vb:'T[:]', rowmapb:'int64[:]',
                     cnl1:'int64', dng1:'int64', cs1:'int64', cp1:'int64', cm1:'int64',
                     dsh:'int64[:]', csh:'int64[:]', dgs1:'int64[:]', dge1:'int64[:]', 
                     cgs1:'int64[:]', cge1:'int64[:]', dnlb1:'int64[:]', cnlb1:'int64[:]'):

    nnz = 0
    nnz_rows = 0
    gr1 = cp1 * cm1

    for i1 in range(cnl1):
        nnz_in_row = 0
        i1_n = cs1 + i1

        pr_i1 = 0
        for k in range(cgs1.size):
            if i1_n < cgs1[k] or i1_n > cge1[k]:
                continue
            pr_i1 = k

        i_g = csh[pr_i1] + i1_n - cgs1[pr_i1]
        stencil_size1 = A[i1 + gr1].size

        for k1 in range(stencil_size1):

            j1_n = (i1_n + k1 - stencil_size1 // 2) % dng1 
            value = A[i1 + gr1, k1]

            if abs(value) == 0.0:
                continue

            pr_j1 = 0
            for k in range(dgs1.size):
                if j1_n < dgs1[k] or j1_n > dge1[k]:
                    continue
                pr_j1 = k            
                         
            j_g = dsh[pr_j1] + j1_n - dgs1[pr_j1]

            if nnz_in_row == 0:
                rowmapb[nnz_rows] = i_g  

            Jb[nnz] = j_g           
            Vb[nnz] = value  
            nnz += 1
            nnz_in_row += 1

        if nnz_in_row > 0:
            Ib[1 + nnz_rows] = Ib[nnz_rows] + nnz_in_row
            nnz_rows += 1

    return nnz_rows, nnz

#========================================================================================================
def stencil2IJV_2d_C(A:'T[:,:,:,:]', Ib:'int64[:]', Jb:'int64[:]', Vb:'T[:]', rowmapb:'int64[:]',
                     cnl1:'int64', cnl2:'int64', dng1:'int64', dng2:'int64', cs1:'int64', 
                     cs2:'int64', cp1:'int64', cp2:'int64', cm1:'int64', cm2:'int64',
                     dsh:'int64[:]', csh:'int64[:]', dgs1:'int64[:]', dgs2:'int64[:]', 
                     dge1:'int64[:]', dge2:'int64[:]', cgs1:'int64[:]', cgs2:'int64[:]', 
                     cge1:'int64[:]', cge2:'int64[:]', dnlb1:'int64[:]', dnlb2:'int64[:]', 
                     cnlb1:'int64[:]', cnlb2:'int64[:]'):

    nnz = 0
    nnz_rows = 0
    gr1 = cp1 * cm1
    gr2 = cp2 * cm2

    for i1 in range(cnl1):
        for i2 in range(cnl2):
            nnz_in_row = 0
            i1_n = cs1 + i1
            i2_n = cs2 + i2

            pr_i1 = 0
            for k in range(cgs1.size):
                if i1_n < cgs1[k] or i1_n > cge1[k]:
                    continue
                pr_i1 = k

            pr_i2 = 0
            for k in range(cgs2.size):
                if i2_n < cgs2[k] or i2_n > cge2[k]:
                    continue
                pr_i2 = k

            pr_i = pr_i2 + pr_i1 * cgs2.size
            i_g = csh[pr_i] + i2_n - cgs2[pr_i2] + (i1_n - cgs1[pr_i1]) * cnlb2[pr_i]
            stencil_size1, stencil_size2 = A.shape[2:]

            for k1 in range(stencil_size1):
                for k2 in range(stencil_size2):
                    j1_n = (i1_n + k1 - stencil_size1 // 2) % dng1 
                    j2_n = (i2_n + k2 - stencil_size2 // 2) % dng2

                    value = A[i1 + gr1, i2 + gr2, k1, k2]
                    if abs(value) == 0.0:
                        continue

                    pr_j1 = 0
                    for k in range(dgs1.size):
                        if j1_n < dgs1[k] or j1_n > dge1[k]:
                            continue
                        pr_j1 = k

                    pr_j2 = 0
                    for k in range(dgs2.size):
                        if j2_n < dgs2[k] or j2_n > dge2[k]:
                            continue
                        pr_j2 = k

                    pr_j = pr_j2 + pr_j1 * dgs2.size
                    j_g = dsh[pr_j] + j2_n - dgs2[pr_j2] + (j1_n - dgs1[pr_j1]) * dnlb2[pr_j]

                    if nnz_in_row == 0:
                        rowmapb[nnz_rows] = i_g  

                    Jb[nnz] = j_g           
                    Vb[nnz] = value  
                    nnz += 1
                    nnz_in_row += 1

            if nnz_in_row > 0:
                Ib[1 + nnz_rows] = Ib[nnz_rows] + nnz_in_row
                nnz_rows += 1

    return nnz_rows, nnz

#========================================================================================================
def stencil2IJV_3d_C(A:'T[:,:,:,:,:,:]', Ib:'int64[:]', Jb:'int64[:]', Vb:'T[:]', rowmapb:'int64[:]',
                     cnl1:'int64', cnl2:'int64', cnl3:'int64', dng1:'int64', dng2:'int64', dng3:'int64', 
                     cs1:'int64', cs2:'int64', cs3:'int64', cp1:'int64', cp2:'int64', cp3:'int64', 
                     cm1:'int64', cm2:'int64', cm3:'int64', dsh:'int64[:]', csh:'int64[:]',
                     dgs1:'int64[:]', dgs2:'int64[:]', dgs3:'int64[:]', dge1:'int64[:]', dge2:'int64[:]', 
                     dge3:'int64[:]', cgs1:'int64[:]', cgs2:'int64[:]', cgs3:'int64[:]', 
                     cge1:'int64[:]', cge2:'int64[:]', cge3:'int64[:]', dnlb1:'int64[:]', dnlb2:'int64[:]', 
                     dnlb3:'int64[:]', cnlb1:'int64[:]', cnlb2:'int64[:]', cnlb3:'int64[:]'):

    nnz = 0
    nnz_rows = 0
    gr1 = cp1*cm1
    gr2 = cp2*cm2
    gr3 = cp3*cm3

    for i1 in range(cnl1):
        for i2 in range(cnl2):
            for i3 in range(cnl3):
                nnz_in_row = 0
                i1_n = cs1 + i1
                i2_n = cs2 + i2
                i3_n = cs3 + i3

                pr_i1 = 0
                for k in range(cgs1.size):
                    if i1_n < cgs1[k] or i1_n > cge1[k]:
                        continue
                    pr_i1 = k

                pr_i2 = 0
                for k in range(cgs2.size):
                    if i2_n < cgs2[k] or i2_n > cge2[k]:
                        continue
                    pr_i2 = k

                pr_i3 = 0
                for k in range(cgs3.size):
                    if i3_n < cgs3[k] or i3_n > cge3[k]:
                        continue
                    pr_i3 = k                    

                pr_i = pr_i3 + pr_i2 * cgs3.size + pr_i1 * cgs2.size * cgs3.size
                i_g = csh[pr_i] + i3_n - cgs3[pr_i3] + (i2_n - cgs2[pr_i2]) * cnlb3[pr_i] + (i1_n - cgs1[pr_i1]) * cnlb2[pr_i] * cnlb3[pr_i]
                stencil_size1, stencil_size2, stencil_size3 = A.shape[3:]

                for k1 in range(stencil_size1):
                    for k2 in range(stencil_size2):
                        for k3 in range(stencil_size3):
                            j1_n = (i1_n + k1 - stencil_size1 // 2) % dng1 
                            j2_n = (i2_n + k2 - stencil_size2 // 2) % dng2
                            j3_n = (i3_n + k3 - stencil_size3 // 2) % dng3

                            value = A[i1 + gr1, i2 + gr2, i3 + gr3, k1, k2, k3]
                            if abs(value) == 0.0:
                                continue

                            pr_j1 = 0
                            for k in range(dgs1.size):
                                if j1_n < dgs1[k] or j1_n > dge1[k]:
                                    continue
                                pr_j1 = k

                            pr_j2 = 0
                            for k in range(dgs2.size):
                                if j2_n < dgs2[k] or j2_n > dge2[k]:
                                    continue
                                pr_j2 = k

                            pr_j3 = 0
                            for k in range(dgs3.size):
                                if j3_n < dgs3[k] or j3_n > dge3[k]:
                                    continue
                                pr_j3 = k  

                            pr_j = pr_j3 + pr_j2 * dgs3.size + pr_j1 * dgs2.size * dgs3.size
                            j_g = dsh[pr_j] + j3_n - dgs3[pr_j3] + (j2_n - dgs2[pr_j2]) * dnlb3[pr_j] + (j1_n - dgs1[pr_j1]) * dnlb2[pr_j] * dnlb3[pr_j]

                            if nnz_in_row == 0:
                                rowmapb[nnz_rows] = i_g  

                            Jb[nnz] = j_g           
                            Vb[nnz] = value  
                            nnz += 1
                            nnz_in_row += 1

                if nnz_in_row > 0:
                    Ib[1 + nnz_rows] = Ib[nnz_rows] + nnz_in_row
                    nnz_rows += 1

    return nnz_rows, nnz
