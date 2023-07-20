
from pyccel.decorators import template

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