# -*- coding: UTF-8 -*-
from spl.linalg.stencil import Matrix, Vector

# ...
def test_1():
    nx = ny = 3
    px = py = 1

    M = Matrix([0, 0], [nx, ny], [px, py])

    print (">>> shape: ", M._data.shape)
    M[:, :, 0, 0] = 4.
    M[:, :, 1, 0] = 1.
    M[:, :,-1, 0] = -1.
    M[:, :, 0, 1] = 2.
    M[:, :, 0,-1] = -2.

    print M.tocoo().toarray()
# ....

# ...
def test_2():
    nx = ny = 2
    px = py = 1

    M = Matrix([0, 0], [nx, ny], [px, py])
    x = Vector([0, 0], [nx, ny], [px, py])

    print (">>> M shape: ", M._data.shape,  x._data.shape)

    for ix in range(nx+1):
        for iy in range(ny+1):
            M[ix, iy, 1, 0] = 1.
            M[ix, iy,-1, 0] = -1.
            M[ix, iy, 0,-1] = -1.
            M[ix, iy, 0, 1] = 1.
            M[ix, iy, 0, 0] = 4.

    print M.tocoo().toarray()

    x[:, :] = 1.

    print (">>> M shape = ", x._data.shape)
    print (">>> M = ")
    M.tocoo().toarray()

    y = M.dot(x)

    print (">>> y (M dot ones) = ", y)
    print (">>> y to array = ", y.toarray())
# ....

test_1()
test_2()
