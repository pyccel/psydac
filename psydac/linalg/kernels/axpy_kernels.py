
from pyccel.decorators import template

#========================================================================================================
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
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                y[i1, i2, i3] += alpha*x[i1, i2, i3]
    return
