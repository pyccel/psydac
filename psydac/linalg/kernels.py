# coding: utf-8
from pyccel.decorators import template
from numpy import conjugate
#========================================================================================================

#__all__ = ['stencil2coo_1d_C','stencil2coo_1d_F','stencil2coo_2d_C','stencil2coo_2d_F', 'stencil2coo_3d_C', 'stencil2coo_3d_F']

# If we can create an array of a pyccel template, we should only create one template
#       @template(name='T', types=['float', 'complex'])
@template(name='T', types=['float[:,:]', 'complex[:,:]'])
@template(name='data', types=['float[:]', 'complex[:]'])
#TODO avoid using The expensive modulo operator % in the non periodic case to make the methods faster
def stencil2coo_1d_C(A:'T', data:'data', rows:'int64[:]', cols:'int64[:]', nrl1:'int64', ncl1:'int64',
                     s1:'int64', nr1:'int64', nc1:'int64', dm1:'int64', cm1:'int64', p1:'int64', dp1:'int64'):
    nnz = 0
    pp1 = cm1*p1
    for i1 in range(nrl1):
        I = s1+i1
        for j1 in range(ncl1):
            value = A[i1+pp1,j1]
            if abs(value) == 0.0:continue
            J = ((I//cm1)*dm1+j1-dp1)%nc1
            rows[nnz] = I
            cols[nnz] = J
            data[nnz] = value
            nnz += 1

    return nnz


#========================================================================================================
@template(name='T', types=['float[:,:]', 'complex[:,:]'])
@template(name='data', types=['float[:]', 'complex[:]'])
def stencil2coo_1d_F(A:'T', data:'data', rows:'int64[:]', cols:'int64[:]', nrl1:'int64', ncl1:'int64', s1:'int64', nr1:'int64', nc1:'int64', dm1:'int64', cm1:'int64', p1:'int64', dp1:'int64'):
    nnz = 0
    pp1 = cm1*p1
    for i1 in range(nrl1):
        I = s1+i1
        for j1 in range(ncl1):
            value = A[i1+pp1,j1]
            if abs(value) == 0.0:continue
            J = ((I//cm1)*dm1+j1-dp1)%nc1
            rows[nnz] = I
            cols[nnz] = J
            data[nnz] = value
            nnz += 1

    return nnz


#========================================================================================================
@template(name='T', types=['float[:,:,:,:]', 'complex[:,:,:,:]'])
@template(name='data', types=['float[:]', 'complex[:]'])
def stencil2coo_2d_C(A:'T', data:'data', rows:'int64[:]', cols:'int64[:]',
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
                    jj1 = ((ii1//cm1)*dm1+j1-dp1)%nc1
                    jj2 = ((ii2//cm2)*dm2+j2-dp2)%nc2

                    J   = jj1*nc2 + jj2

                    rows[nnz] = I
                    cols[nnz] = J
                    data[nnz] = value
                    nnz += 1
    return nnz


#========================================================================================================
@template(name='T', types=['float[:,:,:,:]', 'complex[:,:,:,:]'])
@template(name='data', types=['float[:]', 'complex[:]'])
def stencil2coo_2d_F(A:'T', data:'data', rows:'int64[:]', cols:'int64[:]',
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
                    jj1 = ((ii1//cm1)*dm1+j1-dp1)%nc1
                    jj2 = ((ii2//cm2)*dm2+j2-dp2)%nc2

                    J   = jj2*nc1 + jj1

                    rows[nnz] = I
                    cols[nnz] = J
                    data[nnz] = value
                    nnz += 1
    return nnz


#========================================================================================================
@template(name='T', types=['float[:,:,:,:,:,:]', 'complex[:,:,:,:,:,:]'])
@template(name='data', types=['float[:]', 'complex[:]'])
def stencil2coo_3d_C(A:'T', data:'data', rows:'int64[:]', cols:'int64[:]',
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
                            jj1 = ((ii1//cm1)*dm1+j1-dp1)%nc1
                            jj2 = ((ii2//cm2)*dm2+j2-dp2)%nc2
                            jj3 = ((ii3//cm3)*dm3+j3-dp3)%nc3

                            J   = jj1*nc2*nc3 + jj2*nc3 + jj3

                            rows[nnz] = I
                            cols[nnz] = J
                            data[nnz] = value
                            nnz += 1

    return nnz


#========================================================================================================
@template(name='T', types=['float[:,:,:,:,:,:]', 'complex[:,:,:,:,:,:]'])
@template(name='data', types=['float[:]', 'complex[:]'])
def stencil2coo_3d_F(A:'T', data:'data', rows:'int64[:]', cols:'int64[:]',
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
                            jj1 = ((ii1//cm1)*dm1+j1-dp1)%nc1
                            jj2 = ((ii2//cm2)*dm2+j2-dp2)%nc2
                            jj3 = ((ii3//cm3)*dm3+j3-dp3)%nc3

                            J   = jj3*nc1*nc2 + jj2*nc1 + jj1

                            rows[nnz] = I
                            cols[nnz] = J
                            data[nnz] = value
                            nnz += 1
    return nnz


#========================================================================================================

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!#
#!!!!!!! Conjugate on the first argument !!!!!!!#
#!!!!!!!!!! This will need an update !!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
@template(name='T', types=['float[:]', 'complex[:]'])
def dot_product_1d(v1: 'T', v2: 'T', pads0: 'int64', shift0: 'int64'):
    """
    kernel for computing the inner product (case two complex 1d vectors)

    Parameters
    ----------
        v1, v2 : 1d complex array
            Data of the vectors from which we are computing the inner product.

        pads0, shift0, shape0 : int
            pads, shift and shape of the space of the vectors in the 0 direction.

    Returns
    -------
        res : complex containing the results
        """
    #$omp parallel firstprivate( pads0, shift0) shared(res, v1, v2) private(shape0)
    res = v1[0]-v1[0]
    shape0, = v1.shape
    #$omp for collapse(1) reduction(+ : res)
    for i0 in range(pads0*shift0, shape0-pads0*shift0):
        res += v1[i0].conjugate()*v2[i0]
    #$omp end parallel
    return res


#========================================================================================================
@template(name='T', types=['float[:,:]', 'complex[:,:]'])
def dot_product_2d(v1: 'T', v2: 'T', pads0: 'int64', pads1: 'int64', shift0: 'int64',
                              shift1: 'int64'):
    """
    kernel for computing the inner product (case two complex 2d vectors)

    Parameters
    ----------
        v1, v2 : 2d complex array
            Data of the vectors from which we are computing the inner product.

        pads0, shift0, shape0, pads1, shift1, shape1 : int
            pads, shift and shape of the space of the vectors in the 0/1 direction.

    Returns
    -------
        res : complex containing the results
    """
    #$omp parallel firstprivate( pads0, pads1, shift0, shift1) shared(res, v1, v2) private(shape0, shape1)
    res = v1[0, 0]-v1[0, 0]
    shape0, shape1 = v1.shape
    #$omp for collapse(2) reduction(+ : res)
    for i0 in range(pads0*shift0, shape0-pads0*shift0):
        for i1 in range(pads1*shift1, shape1-pads1*shift1):
            res += v1[i0, i1].conjugate()*v2[i0, i1]
    #$omp end parallel
    return res


#========================================================================================================
@template(name='T', types=['float[:,:,:]', 'complex[:,:,:]'])
def dot_product_3d(v1: 'T', v2: 'T', pads0: 'int64', pads1: 'int64', pads2: 'int64',
                              shift0: 'int64', shift1: 'int64', shift2: 'int64'):
    """
    kernel for computing the inner product (case two complex 3d vectors)

    Parameters
    ----------
        v1, v2 : 3d complex array
            Data of the vectors from which we are computing the inner product.

        pads0, shift0, shape0, pads1, shift1, shape1, pads2, shift2, shape2 : int
            pads, shift and shape of the space of the vectors in the 0/1/2 direction.

    Returns
    -------
        res : complex containing the results
    """
    #$omp parallel firstprivate( pads0, pads1, pads2, shift0, shift1, shift2) shared(res, v1, v2) private(shape0, shape1, shape2)
    res = v1[0, 0, 0] - v1[0, 0, 0]
    shape0, shape1, shape2 = v1.shape
    #$omp for collapse(3) reduction(+ : res)
    for i0 in range(pads0*shift0, shape0-pads0*shift0):
        for i1 in range(pads1*shift1, shape1-pads1*shift1):
            for i2 in range(pads2*shift2, shape2-pads2*shift2):
                res += v1[i0, i1, i2].conjugate()*v2[i0, i1, i2]
    #$omp end parallel
    return res


#========================================================================================================

#Implementation of the axpy kernels

@template(name='Tarray', types=['float[:]', 'complex[:]'])
@template(name='T', types=['float', 'complex'])
def axpy_1d(alpha: 'T', x: "Tarray", y: "Tarray"):
    """
        kernel for computing y=y+alpha*x

        Parameters
        ----------
            x, y : 1d array
                Data of the vectors from which we are computing the inner product.

            alpha : scalar
                Coefficient needed for the operation to multiply v2
    """
    n1, = x.shape
    #$ omp parallel for default(private) firstprivate(alpha, n1) shared(x, y) schedule(static)
    for i1 in range(n1):
        y[i1] += alpha*x[i1]
    return


#========================================================================================================

@template(name='Tarray', types=['float[:,:]', 'complex[:,:]'])
@template(name='T', types=['float', 'complex'])
def axpy_2d(alpha: 'T', x: "Tarray", y: "Tarray"):
    """
        kernel for computing y=y+alpha*x

        Parameters
        ----------
            x, y : 2d complex array
                Data of the vectors from which we are computing the inner product.

            alpha : scalar
                Coefficient needed for the operation to multiply v2
    """
    n1, n2 = x.shape
    #$ omp parallel for default(private) firstprivate(alpha, n1, n2) shared(x, y) collapse(2) schedule(static)
    for i1 in range(n1):
        for i2 in range(n2):
            y[i1, i2] += alpha * x[i1, i2]
    return

#========================================================================================================
@template(name='Tarray', types=['float[:,:,:]', 'complex[:,:,:]'])
@template(name='T', types=['float', 'complex'])
def axpy_3d(alpha: 'T', x: "Tarray", y: "Tarray"):
    """
        kernel for computing y=y+alpha*x

        Parameters
        ----------
            x, y : 3d complex array
                Data of the vectors from which we are computing the inner product.

            alpha : scalar
                Coefficient needed for the operation to multiply v2
    """
    n1, n2, n3 = x.shape
    #$ omp parallel for default(private) firstprivate(alpha, n1, n2, n3) shared(x, y) collapse(3) schedule(static)
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                y[i1, i2, i3] += alpha*x[i1, i2, i3]
    return


#========================================================================================================


