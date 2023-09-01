from pyccel.decorators import template

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!#
#!!!!!!! Conjugate on the first argument !!!!!!!#
#!!!!!!!!!! This will need an update !!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

# TODO [YG, 01.09.2023]:
# Do not pass (pads, shift) separately, use their product instead!

#==============================================================================
@template(name='T', types=['float[:]', 'complex[:]'])
def inner_dot_1d(v1: 'T', v2: 'T', pads0: 'int64', shift0: 'int64'):
    """
    Kernel for computing the inner product (case of two 1D vectors).

    Parameters
    ----------
    v1, v2 : 1D NumPy array
        Data of the vectors from which we are computing the inner product.

    pads0 : int
        Array padding along the index 0.

    shift0: int
        Factor which multiplies the padding size (equal to 1 in most cases).

    Returns
    -------
    res : scalar
        Scalar (real or complex) containing the result of the inner product.
    """
    shape0, = v1.shape

    res = v1[0] - v1[0]
    for i0 in range(pads0 * shift0, shape0 - pads0 * shift0):
        res += v1[i0].conjugate() * v2[i0]

    return res

#==============================================================================
@template(name='T', types=['float[:,:]', 'complex[:,:]'])
def inner_dot_2d(v1: 'T', v2: 'T', pads0: 'int64', pads1: 'int64',
                              shift0: 'int64', shift1: 'int64'):
    """
    Kernel for computing the inner product (case two 2D vectors).

    Parameters
    ----------
    v1, v2 : 2D NumPy array
        Data of the vectors from which we are computing the inner product.

    pads0, pads1 : int
        Array padding along indices 0 and 1.

    shift0, shift1: int
        Factors which multiply the padding size (equal to 1 in most cases).

    Returns
    -------
    res : scalar
        Scalar (real or complex) containing the result of the inner product.
    """
    res = v1[0, 0]-v1[0, 0]
    shape0, shape1 = v1.shape
    for i0 in range(pads0 * shift0, shape0 - pads0 * shift0):
        for i1 in range(pads1 * shift1, shape1 - pads1 * shift1):
            res += v1[i0, i1].conjugate()*v2[i0, i1]
    return res

#==============================================================================
@template(name='T', types=['float[:,:,:]', 'complex[:,:,:]'])
def inner_dot_3d(v1: 'T', v2: 'T', pads0: 'int64', pads1: 'int64', pads2: 'int64',
                              shift0: 'int64', shift1: 'int64', shift2: 'int64'):
    """
    Kernel for computing the inner product (case two 3D vectors).

    Parameters
    ----------
    v1, v2 : 3D NumPy array
        Data of the vectors from which we are computing the inner product.

    pads0, pads1, pads2 : int
        Array padding along indices 0, 1, and 2.

    shift0, shift1, shift2: int
        Factors which multiply the padding size (equal to 1 in most cases).

    Returns
    -------
    res : scalar
        Scalar (real or complex) containing the result of the inner product.
    """
    res = v1[0, 0, 0] - v1[0, 0, 0]
    shape0, shape1, shape2 = v1.shape
    for i0 in range(pads0 * shift0, shape0 - pads0 * shift0):
        for i1 in range(pads1 * shift1, shape1 - pads1 * shift1):
            for i2 in range(pads2 * shift2, shape2 - pads2 * shift2):
                res += v1[i0, i1, i2].conjugate() * v2[i0, i1, i2]
    return res
