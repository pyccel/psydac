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

    mat = Matrix([0, 0], [nx, ny], [px, py])
    vec = Vector([0, 0], [nx, ny], [px, py])

    print '>>> shape: ', mat._data.shape,  vec._data.shape

    for ix in range(nx+1):
        for iy in range(ny+1):
            mat[ 1,  0, ix, iy] = 1.
            mat[-1,  0, ix, iy] = -1.
            mat[ 0, -1, ix, iy] = -1.
            mat[ 0,  1, ix, iy] = 1.
            mat[ 0,  0, ix, iy] = 2.

    print mat.tocoo().toarray()

    vec[:, :] = 1.
    y = mat.dot(vec)

    print '>>> y = ', y
# ....

#test_1()
test_2()
