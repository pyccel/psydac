#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!#
#!!!!!!! Conjugate on the first argument !!!!!!!#
#!!!!!!!!!! This will need an update !!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

from typing import TypeVar

T = TypeVar('T', float, complex)

#==============================================================================
def inner_1d(v1: 'T[:]', v2: 'T[:]', nghost0: 'int64'):
    """
    Kernel for computing the inner product (case of two 1D vectors).

    Parameters
    ----------
    v1, v2 : 1D NumPy array
        Data of the vectors from which we are computing the inner product.

    nghost0 : int
        Number of ghost cells of the arrays along the index 0.

    Returns
    -------
    res : scalar
        Scalar (real or complex) containing the result of the inner product.
    """
    shape0, = v1.shape

    res = v1[0] - v1[0]
    for i0 in range(nghost0, shape0 - nghost0):
        res += v1[i0].conjugate() * v2[i0]

    return res

#==============================================================================
def inner_2d(v1: 'T[:,:]', v2: 'T[:,:]', nghost0: 'int64', nghost1: 'int64'):
    """
    Kernel for computing the inner product (case of two 2D vectors).

    Parameters
    ----------
    v1, v2 : 2D NumPy array
        Data of the vectors from which we are computing the inner product.

    nghost0 : int
        Number of ghost cells of the arrays along the index 0.

    nghost1 : int
        Number of ghost cells of the arrays along the index 1.

    Returns
    -------
    res : scalar
        Scalar (real or complex) containing the result of the inner product.
    """
    shape0, shape1 = v1.shape

    res = v1[0, 0] - v1[0, 0]
    for i0 in range(nghost0, shape0 - nghost0):
        for i1 in range(nghost1, shape1 - nghost1):
            res += v1[i0, i1].conjugate() * v2[i0, i1]

    return res

#==============================================================================
def inner_3d(v1: 'T[:,:,:]', v2: 'T[:,:,:]', nghost0: 'int64', nghost1: 'int64', nghost2: 'int64'):
    """
    Kernel for computing the inner product (case of two 3D vectors).

    Parameters
    ----------
    v1, v2 : 3D NumPy array
        Data of the vectors from which we are computing the inner product.

    nghost0 : int
        Number of ghost cells of the arrays along the index 0.

    nghost1 : int
        Number of ghost cells of the arrays along the index 1.

    nghost2 : int
        Number of ghost cells of the arrays along the index 2.

    Returns
    -------
    res : scalar
        Scalar (real or complex) containing the result of the inner product.
    """
    shape0, shape1, shape2 = v1.shape

    res = v1[0, 0, 0] - v1[0, 0, 0]
    for i0 in range(nghost0, shape0 - nghost0):
        for i1 in range(nghost1, shape1 - nghost1):
            for i2 in range(nghost2, shape2 - nghost2):
                res += v1[i0, i1, i2].conjugate() * v2[i0, i1, i2]

    return res
