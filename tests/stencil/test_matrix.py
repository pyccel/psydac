# -*- coding: UTF-8 -*-
from spl.stencil import Matrix, Vector

# ...
def test_1():
    nx = ny = 3
    px = py = 1

    x = Matrix([0, 0], [nx, ny], [px, py])

    print '>>> shape: ', x._data.shape

    for ix in range(nx+1):
        for iy in range(ny+1):
            x[ 1,  0, ix, iy] = 1.
            x[-1,  0, ix, iy] = -1.
            x[ 0, -1, ix, iy] = -1.
            x[ 0,  1, ix, iy] = 1.
            x[ 0,  0, ix, iy] = 4.

    print x.tocoo().toarray()
# ....

# ...
def test_2():
    nx = ny = 2
    px = py = 1

    M = Matrix([0, 0], [nx, ny], [px, py])
    x = Vector([0, 0], [nx, ny], [px, py])

    print '>>> M shape: ', M._data.shape,  x._data.shape

    for ix in range(nx+1):
        for iy in range(ny+1):
            M[ 1,  0, ix, iy] = 1.
            M[-1,  0, ix, iy] = -1.
            M[ 0, -1, ix, iy] = -1.
            M[ 0,  1, ix, iy] = 1.
            M[ 0,  0, ix, iy] = 2.

    print M.tocoo().toarray()

    x[:, :] = 1.

    print '>>> x shape = ', x._data.shape
    print x

    y = M.dot(x)

    print '>>> y = '
    a = y.toarray()
    print a
# ....

#test_1()
test_2()
