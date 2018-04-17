# -*- coding: UTF-8 -*-
from __future__ import print_function
from spl.linalg.stencil import Vector, VectorSpace
import numpy as np

# ...
def test_1():
    print( "", "-"*40, "Test 1", "-"*40, "", sep="\n" )

    n1 = n2 = 3
    p1 = p2 = 1
    V = VectorSpace([0, 0], [n1, n2], [p1, p2])

    x = Vector( V )
    for i1 in range(0, n1+1):
        for i2 in range(0, n2+1):
            for k1 in range(-p1, p1+1):
                for k2 in range(-p2, p2+1):
                    j1 = k1+i1
                    j2 = k2+i2
                    x[j1,j2] = j1+j2
    z = x.copy()

    print('>>> x shape: ', x._data.shape)
    print(">>> x = ")
    print(x)
    print(">>> x.toarray() = ",  x.toarray())
    print(">>> z = x.copy() = ")
    print(z)

# ....

# ...
def test_2():
    print( "", "-"*40, "Test 2", "-"*40, "", sep="\n" )

    n1 = n2 = 2
    p1 = p2 = 1
    V = VectorSpace([0, 0], [n1, n2], [p1, p2])

    x = Vector( V )
    y = Vector( V )

    x[:,:] = 42.
    y[:,:] = 10.

    a = x + y
    b = x - y
    c = 2 * x

    print('>>> x shape: ', x._data.shape)
    print(">>> x = ")
    print(x)
    print('>>> y shape: ', y._data.shape)
    print(">>> y = ")
    print(y)
    print(">>> x+y = ")
    print(a)
    print(">>> x-y = ")
    print(b)
    print(">>> 2*x = ")
    print(c)
# ....

# ...
def test_3():
    print( "", "-"*40, "Test 3", "-"*40, "", sep="\n" )

    n1 = n2 = 2
    p1 = p2 = 1
    V = VectorSpace([0, 0], [n1, n2], [p1, p2])

    x = Vector( V )
    y = Vector( V )

    x[:, :] = 2.
    y[:, :] = 5.

    z = x.dot(y)

    print('>>> x shape: ', x._data.shape)
    print(">>> x.toarray() = ", x.toarray())
    print('>>> y shape: ', y._data.shape)
    print(">>> y.toarray() = ", y.toarray())
    print(">>> dot(x,y) = ")
    print(z)
# ....

if __name__ is "__main__":
    test_1()
    test_2()
    test_3()
