# -*- coding: UTF-8 -*-
from spl.stencil import Vector
import numpy as np

# ...
def test_1():
    n1 = n2 = 3
    p1 = p2 = 1

    x = Vector([0, 0], [n1, n2], [p1, p2])

    print '>>> shape: ', x._data.shape
    for i1 in range(0, n1+1):
        for i2 in range(0, n2+1):
            for k1 in range(-p1, p1+1):
                for k2 in range(-p2, p2+1):
                    j1 = k1+i1
                    j2 = k2+i2
                    x[j1,j2] = j1+j2

    print x
    # ... TODO test get item
    # ....

test_1()
