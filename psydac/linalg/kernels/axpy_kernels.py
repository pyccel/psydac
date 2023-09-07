from pyccel.decorators import template

#========================================================================================================
@template(name='Tarray', types=['float[:]', 'complex[:]'])
@template(name='T', types=['float', 'complex'])
def axpy_1d(alpha: 'T', x: "Tarray", y: "Tarray"):
    """
    Kernel for computing y = alpha * x + y.

    Parameters
    ----------
    alpha : float | complex
        Scaling coefficient.

    x, y : 1D Numpy arrays of (float | complex) data
        Data of the vectors.
    """
    n1, = x.shape
    for i1 in range(n1):
        y[i1] += alpha * x[i1]

#========================================================================================================
@template(name='Tarray', types=['float[:,:]', 'complex[:,:]'])
@template(name='T', types=['float', 'complex'])
def axpy_2d(alpha: 'T', x: "Tarray", y: "Tarray"):
    """
    Kernel for computing y = alpha * x + y.

    Parameters
    ----------
    alpha : float | complex
        Scaling coefficient.

    x, y : 2D Numpy arrays of (float | complex) data
        Data of the vectors.
    """
    n1, n2 = x.shape
    for i1 in range(n1):
        for i2 in range(n2):
            y[i1, i2] += alpha * x[i1, i2]

#========================================================================================================
@template(name='Tarray', types=['float[:,:,:]', 'complex[:,:,:]'])
@template(name='T', types=['float', 'complex'])
def axpy_3d(alpha: 'T', x: "Tarray", y: "Tarray"):
    """
    Kernel for computing y = alpha * x + y.

    Parameters
    ----------
    alpha : float | complex
        Scaling coefficient.

    x, y : 3D Numpy arrays of (float | complex) data
        Data of the vectors.
    """
    n1, n2, n3 = x.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                y[i1, i2, i3] += alpha * x[i1, i2, i3]
