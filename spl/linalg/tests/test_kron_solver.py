# -*- coding: UTF-8 -*-

import pytest
import numpy as np
from mpi4py             import MPI
from spl.ddm.cart       import Cart

from spl.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix


def test_kron_solve_2d_sparses( n1, n2, p1, p2, P1=True, P2=False ):

    # ... 2D MPI cart
    cart = Cart(npts = [n1, n2], pads = [p1, p2], periods = [False, False],\
                reorder = True, comm = comm)

    # ... Vector Spaces
    V = StencilVectorSpace(cart)
    V1 = StencilVectorSpace([n1], [p1], [False])
    V2 = StencilVectorSpace([n2], [p2], [False])

    # ... Inputs
    Y = StencilVector(V)
    A1 = StencilMatrix(V1, V1)
    A2 = StencilMatrix(V2, V2)
    # ...

    # ... Build banded matrices
    A1[:,-p1:0  ] = -1
    A1[:, 0:1  ] = 2*p1
    A1[:, 1:p1+1] = -1
    A1.remove_spurious_entries()

    A2[:,-p2:0  ] = -1
    A2[:, 0:1  ] = 2*p2
    A2[:, 1:p2+1] = -1
    A2.remove_spurious_entries()




============================================================================


#===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================
#if __name__ == "__main__":
#    import sys
#    pytest.main( sys.argv )
