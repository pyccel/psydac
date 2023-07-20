
from pyccel.decorators import template

#========================================================================================================

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!#
#!!!!!!! Conjugate on the first argument !!!!!!!#
#!!!!!!!!!! This will need an update !!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
@template(name='T', types=['float[:]', 'complex[:]'])
def inner_dot_1d(v1: 'T', v2: 'T', pads0: 'int64', shift0: 'int64'):
    """
    kernel for computing the inner product (case of two 1d vectors)

    Parameters
    ----------
        v1, v2 : 1d array
            Data of the vectors from which we are computing the inner product.

        pads0, shift0, shape0 : int
            pads, shift and shape of the space of the vectors in the 0 direction.

    Returns
    -------
        res : scalar (real or complex) containing the results
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
def inner_dot_2d(v1: 'T', v2: 'T', pads0: 'int64', pads1: 'int64', shift0: 'int64',
                              shift1: 'int64'):
    """
    kernel for computing the inner product (case two 2d vectors)

    Parameters
    ----------
        v1, v2 : 2d array
            Data of the vectors from which we are computing the inner product.

        pads0, shift0, shape0, pads1, shift1, shape1 : int
            pads, shift and shape of the space of the vectors in the 0/1 direction.

    Returns
    -------
        res : scalar containing the results
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
def inner_dot_3d(v1: 'T', v2: 'T', pads0: 'int64', pads1: 'int64', pads2: 'int64',
                              shift0: 'int64', shift1: 'int64', shift2: 'int64'):
    """
    kernel for computing the inner product (case of two 3d vectors)

    Parameters
    ----------
        v1, v2 : 3d array
            Data of the vectors from which we are computing the inner product.

        pads0, shift0, shape0, pads1, shift1, shape1, pads2, shift2, shape2 : int
            pads, shift and shape of the space of the vectors in the 0/1/2 direction.

    Returns
    -------
        res : scalar (real or complex) containing the results
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

