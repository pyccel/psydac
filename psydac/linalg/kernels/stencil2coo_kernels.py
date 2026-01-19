#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#

#!!!!!!!!!!!!!
#TODO avoid using The expensive modulo operator % in the non periodic case to make the methods faster
#!!!!!!!!!!!!!

from typing import TypeVar

T = TypeVar('T', float, complex)

#__all__ = ['stencil2coo_1d_C','stencil2coo_1d_F','stencil2coo_2d_C','stencil2coo_2d_F', 'stencil2coo_3d_C', 'stencil2coo_3d_F']

#========================================================================================================
def stencil2coo_1d_C(A:'T[:,:]', data:'T[:]', rows:'int64[:]', cols:'int64[:]', nrl1:'int64', ncl1:'int64',
                     s1:'int64', nr1:'int64', nc1:'int64', dm1:'int64', cm1:'int64', p1:'int64', dp1:'int64'):
    nnz = 0
    pp1 = cm1*p1
    for i1 in range(nrl1):
        I = s1+i1
        for j1 in range(ncl1):
            value = A[i1+pp1,j1]
            if abs(value) == 0.0:continue
            J = ((I*dm1//cm1)+j1-dp1)%nc1
            rows[nnz] = I
            cols[nnz] = J
            data[nnz] = value
            nnz += 1

    return nnz

#========================================================================================================
def stencil2coo_1d_F(A:'T[:,:]', data:'T[:]', rows:'int64[:]', cols:'int64[:]', nrl1:'int64', ncl1:'int64',
                     s1:'int64', nr1:'int64', nc1:'int64', dm1:'int64', cm1:'int64', p1:'int64', dp1:'int64'):
    nnz = 0
    pp1 = cm1*p1
    for i1 in range(nrl1):
        I = s1+i1
        for j1 in range(ncl1):
            value = A[i1+pp1,j1]
            if abs(value) == 0.0:continue
            J = ((I*dm1//cm1)+j1-dp1)%nc1
            rows[nnz] = I
            cols[nnz] = J
            data[nnz] = value
            nnz += 1

    return nnz

#========================================================================================================
def stencil2coo_2d_C(A:'T[:,:,:,:]', data:'T[:]', rows:'int64[:]', cols:'int64[:]',
                     nrl1:'int64', nrl2:'int64', ncl1:'int64', ncl2:'int64',
                     s1:'int64', s2:'int64', nr1:'int64', nr2:'int64',
                     nc1:'int64', nc2:'int64', dm1:'int64', dm2:'int64',
                     cm1:'int64', cm2:'int64', p1:'int64', p2:'int64',
                     dp1:'int64', dp2:'int64'):
    nnz = 0
    pp1 = cm1*p1
    pp2 = cm2*p2
    for i1 in range(nrl1):
        for i2 in range(nrl2):
            ii1 = s1+i1
            ii2 = s2+i2
            I   = ii1*nr2 + ii2
            for j1 in range(ncl1):
                for j2 in range(ncl2):
                    value = A[i1+pp1,i2+pp2,j1,j2]
                    if abs(value) == 0.0:continue
                    jj1 = ((ii1*dm1//cm1)+j1-dp1)%nc1
                    jj2 = ((ii2*dm2//cm2)+j2-dp2)%nc2

                    J   = jj1*nc2 + jj2

                    rows[nnz] = I
                    cols[nnz] = J
                    data[nnz] = value
                    nnz += 1
    return nnz

#========================================================================================================
def stencil2coo_2d_F(A:'T[:,:,:,:]', data:'T[:]', rows:'int64[:]', cols:'int64[:]',
                     nrl1:'int64', nrl2:'int64', ncl1:'int64', ncl2:'int64',
                     s1:'int64', s2:'int64', nr1:'int64', nr2:'int64',
                     nc1:'int64', nc2:'int64', dm1:'int64', dm2:'int64',
                     cm1:'int64', cm2:'int64', p1:'int64', p2:'int64',
                     dp1:'int64', dp2:'int64'):
    nnz = 0
    pp1 = cm1*p1
    pp2 = cm2*p2
    for i1 in range(nrl1):
        for i2 in range(nrl2):
            ii1 = s1+i1
            ii2 = s2+i2
            I   = ii2*nr1 + ii1
            for j1 in range(ncl1):
                for j2 in range(ncl2):
                    value = A[i1+pp1,i2+pp2,j1,j2]
                    if abs(value) == 0.0:continue
                    jj1 = ((ii1*dm1//cm1)+j1-dp1)%nc1
                    jj2 = ((ii2*dm2//cm2)+j2-dp2)%nc2

                    J   = jj2*nc1 + jj1

                    rows[nnz] = I
                    cols[nnz] = J
                    data[nnz] = value
                    nnz += 1
    return nnz

#========================================================================================================
def stencil2coo_3d_C(A:'T[:,:,:,:,:,:]', data:'T[:]', rows:'int64[:]', cols:'int64[:]',
                     nrl1:'int64', nrl2:'int64', nrl3:'int64', ncl1:'int64', ncl2:'int64', ncl3:'int64',
                     s1:'int64', s2:'int64', s3:'int64', nr1:'int64', nr2:'int64', nr3:'int64',
                     nc1:'int64', nc2:'int64', nc3:'int64', dm1:'int64', dm2:'int64', dm3:'int64',
                     cm1:'int64', cm2:'int64', cm3:'int64', p1:'int64', p2:'int64', p3:'int64',
                     dp1:'int64', dp2:'int64', dp3:'int64'):
    nnz = 0
    pp1 = cm1*p1
    pp2 = cm2*p2
    pp3 = cm3*p3
    for i1 in range(nrl1):
        for i2 in range(nrl2):
            for i3 in range(nrl3):
                ii1 = s1+i1
                ii2 = s2+i2
                ii3 = s3+i3
                I   = ii1*nr2*nr3 + ii2*nr3 + ii3
                for j1 in range(ncl1):
                    for j2 in range(ncl2):
                        for j3 in range(ncl3):
                            value = A[i1+pp1,i2+pp2,i3+pp3,j1,j2,j3]
                            if abs(value) == 0.0:continue
                            jj1 = ((ii1*dm1//cm1)+j1-dp1)%nc1
                            jj2 = ((ii2*dm2//cm2)+j2-dp2)%nc2
                            jj3 = ((ii3*dm3//cm3)+j3-dp3)%nc3

                            J   = jj1*nc2*nc3 + jj2*nc3 + jj3

                            rows[nnz] = I
                            cols[nnz] = J
                            data[nnz] = value
                            nnz += 1

    return nnz


#========================================================================================================
def stencil2coo_3d_F(A:'T[:,:,:,:,:,:]', data:'T[:]', rows:'int64[:]', cols:'int64[:]',
                     nrl1:'int64', nrl2:'int64', nrl3:'int64', ncl1:'int64', ncl2:'int64', ncl3:'int64',
                     s1:'int64', s2:'int64', s3:'int64', nr1:'int64', nr2:'int64', nr3:'int64',
                     nc1:'int64', nc2:'int64', nc3:'int64', dm1:'int64', dm2:'int64', dm3:'int64',
                     cm1:'int64', cm2:'int64', cm3:'int64', p1:'int64', p2:'int64', p3:'int64',
                     dp1:'int64', dp2:'int64', dp3:'int64'):
    nnz = 0
    pp1 = cm1*p1
    pp2 = cm2*p2
    pp3 = cm3*p3
    for i1 in range(nrl1):
        for i2 in range(nrl2):
            for i3 in range(nrl3):
                ii1 = s1+i1
                ii2 = s2+i2
                ii3 = s3+i3
                I   = ii3*nr1*nr2 + ii2*nr1 + ii1
                for j1 in range(ncl1):
                    for j2 in range(ncl2):
                        for j3 in range(ncl3):
                            value = A[i1+pp1,i2+pp2,i3+pp3,j1,j2,j3]
                            if abs(value) == 0.0:continue
                            jj1 = ((ii1*dm1//cm1)+j1-dp1)%nc1
                            jj2 = ((ii2*dm2//cm2)+j2-dp2)%nc2
                            jj3 = ((ii3*dm3//cm3)+j3-dp3)%nc3

                            J   = jj3*nc1*nc2 + jj2*nc1 + jj1

                            rows[nnz] = I
                            cols[nnz] = J
                            data[nnz] = value
                            nnz += 1
    return nnz
