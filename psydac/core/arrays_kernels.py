from pyccel.decorators import pure
from numpy import shape

@pure
def matmul(a: 'float[:,:]', b: 'float[:,:]', c: 'float[:,:]'):
    """
    Performs the matrix-matrix product a*b and writes the result into c.

    Parameters
    ----------
        a : array[float]
            The first input array (matrix).

        b : array[float]
            The second input array (matrix).

        c : array[float]
            The output array (matrix) which is the result of the matrix-matrix product a.dot(b).
    """

    sh_a = shape(a)
    sh_b = shape(b)
    
    if sh_a[0] == 0 or sh_a[1] == 0 or sh_b[0] == 0 or sh_b[1] == 0:
        c[:, :] = 0.
    else:
        c[:, :] = 0.
        for i in range(sh_a[0]):
            for j in range(sh_b[1]):
                for k in range(sh_a[1]):
                    c[i, j] += a[i, k] * b[k, j] 
                
                
@pure
def sum_vec(a: 'float[:]') -> float:
    """
    Sum the elements of a 1D vector.

    Parameters
    ----------
        a : array[float]
            The 1d vector.
    """

    out = 0.
    
    sh_a = shape(a)

    for i in range(sh_a[0]):
        out += a[i] 
        
    return out
        
        
@pure
def min_vec(a: 'float[:]') -> float:
    """
    Compute the minimum a 1D vector.

    Parameters
    ----------
        a : array[float]
            The 1D vector.
    """

    out = a[0]
    
    sh_a = shape(a)

    for i in range(sh_a[0]):
        if a[i] < out:
            out = a[i] 
            
    return out


@pure
def max_vec(a: 'float[:]') -> float:
    """
    Compute the maximum a 1D vector.

    Parameters
    ----------
        a : array[float]
            The 1D vector.
    """

    out = a[0]
    
    sh_a = shape(a)

    for i in range(sh_a[0]):
        if a[i] > out:
            out = a[i] 
            
    return out


@pure
def max_vec_int(a: 'int[:]') -> int:
    """
    Compute the maximum a 1D vector.

    Parameters
    ----------
        a : array[float]
            The 1D vector.
    """

    out = a[0]
    
    sh_a = shape(a)

    for i in range(sh_a[0]):
        if a[i] > out:
            out = a[i] 
            
    return out