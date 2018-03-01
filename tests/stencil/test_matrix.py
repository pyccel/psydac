# -*- coding: UTF-8 -*-
from spl.stencil import Matrix

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

test_1()
