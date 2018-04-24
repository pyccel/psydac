# -*- coding: UTF-8 -*-
from spl.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix

# ...
def test_1():
    nx = ny = 3
    px = py = 1

    V = StencilVectorSpace( [0,0], [nx,ny], [px,py] )
    M = StencilMatrix( V, V )

    print (">>> M shape: ", M._data.shape)
    M[:, :, 0, 0] = 4.
    M[:, :, 1, 0] = 1.
    M[:, :,-1, 0] = -1.
    M[:, :, 0, 1] = 2.
    M[:, :, 0,-1] = -2.

    print (">>> M = ")
    print (M.toarray())
# ....

# ...
def test_2():
    nx = ny = 2
    px = py = 1

    V = StencilVectorSpace([0, 0], [nx, ny], [px, py])
    x = StencilVector( V )
    M = StencilMatrix( V, V )

    for ix in range(nx+1):
        for iy in range(ny+1):
            M[ix, iy, 0, 0] =  4.
            M[ix, iy, 1, 0] =  1.
            M[ix, iy,-1, 0] = -1.
            M[ix, iy, 0, 1] =  2.
            M[ix, iy, 0,-1] = -2.

    x[:,:] = 1.

    y = M.dot(x)

    print (">>> M shape = ", M._data.shape)
    print (">>> M = ")
    print ( M.toarray() )

    print (">>> x shape = ", x._data.shape)
    print (">>> x = ", x.toarray() )

    print (">>> y (M dot ones) = ", y.toarray() )
# ....

if __name__ is "__main__":
    test_1()
    test_2()
