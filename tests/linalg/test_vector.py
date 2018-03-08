# -*- coding: UTF-8 -*-
from spl.linalg.stencil import Vector
import numpy as np

# ...
def test_1():
    n1 = n2 = 3
    p1 = p2 = 1

    x = Vector([0, 0], [n1, n2], [p1, p2])

    print '>>> x shape: ', x._data.shape

    for i1 in range(0, n1+1):
        for i2 in range(0, n2+1):
            for k1 in range(-p1, p1+1):
                for k2 in range(-p2, p2+1):
                    j1 = k1+i1
                    j2 = k2+i2
                    x[j1,j2] = j1+j2

    print (">>> x = ")
    print  x
    print (">>> x.toarry() = ",  x.toarray())

    y = x.zeros_like()

    print (">>> y = x.zeros_lik:")
    print y

    z = x.copy()
    print (">>> z = x.copy() = ")
    print z

# ....

# ...
def test_2():
    n1 = n2 = 2
    p1 = p2 = 1

    x = Vector([0, 0], [n1, n2], [p1, p2])
    y = Vector([0, 0], [n1, n2], [p1, p2])

    x[:, :] = 42.
    y[:, :] = 10.

    # a = x + y
    a = x.copy()
    a.add(y)
    print (">>> x+y = ")
    print a

    # b = x- y
    b = x.copy()
    b.sub(y)
    print (">>> x-y = ")
    print b

    # c =2*x
    c = x.copy()
    c.mul(2.)
    print (">>> 2*x = ")
    print c
# ....

# ...
def test_3():
    n1 = n2 = 2
    p1 = p2 = 1

    x = Vector([0, 0], [n1, n2], [p1, p2])
    y = Vector([0, 0], [n1, n2], [p1, p2])

    x[:, :] = 2.
    y[:, :] = 5.

    # z = x*y
    z = x.dot(y)
    print (">>> x=2, y=5, x dot y = ")
    print z
# ....

test_1()
test_2()
test_3()
