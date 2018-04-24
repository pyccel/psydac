# coding: utf-8
import numpy as np

from spl.utilities.quadratures import gauss_legendre
from spl.linalg.stencil        import StencilVector, StencilMatrix
from spl.linalg.solvers        import cg

# TODO make_open_knots: ne is not the number of elements!!

# ... assembly of mass and stiffness matrices using stencil forms
def assembly_matrices(V):

    # ... sizes
    [s1] = V.vector_space.starts
    [e1] = V.vector_space.ends
    [p1] = V.vector_space.pads

    k1 = V.quad_order
    spans_1 = V.spans
    basis_1 = V.basis
    weights_1 = V.weights
    # ...

    # ... data structure
    mass      = StencilMatrix( V.vector_space, V.vector_space )
    stiffness = StencilMatrix( V.vector_space, V.vector_space )
    # ...

    # ... build matrices
    for ie1 in range(s1, e1+1-p1):
        i_span_1 = spans_1[ie1]
        for il_1 in range(0, p1+1):
            for jl_1 in range(0, p1+1):
                i1 = i_span_1 - p1  - 1 + il_1
                j1 = i_span_1 - p1  - 1 + jl_1

                v_m = 0.0
                v_s = 0.0
                for g1 in range(0, k1):
                    bi_0 = basis_1[il_1, 0, g1, ie1]
                    bi_x = basis_1[il_1, 1, g1, ie1]

                    bj_0 = basis_1[jl_1, 0, g1, ie1]
                    bj_x = basis_1[jl_1, 1, g1, ie1]

                    wvol = weights_1[g1, ie1]

                    v_m += bi_0 * bj_0 * wvol
                    v_s += (bi_x * bj_x) * wvol

                mass[i1, j1 - i1] += v_m
                stiffness[i1, j1 - i1]  += v_s
    # ...

    # ...
    return mass , stiffness
    # ...
# ...

# ... example of assembly of the rhs: f(x1,x2) = x1*(1-x2)*x2*(1-x2)
def assembly_rhs(V):

    # ... sizes
    [s1] = V.vector_space.starts
    [e1] = V.vector_space.ends
    [p1] = V.vector_space.pads

    k1 = V.quad_order
    spans_1 = V.spans
    basis_1 = V.basis
    points_1 = V.points
    weights_1 = V.weights
    # ...

    # ... data structure
    rhs = StencilVector( V.vector_space )
    # ...

    # ... build rhs
    for ie1 in range(s1, e1+1-p1):
        i_span_1 = spans_1[ie1]
        for il_1 in range(0, p1+1):
            i1 = i_span_1 - p1  - 1 + il_1

            v = 0.0
            for g1 in range(0, k1):
                bi_0 = basis_1[il_1, 0, g1, ie1]
                bi_x = basis_1[il_1, 1, g1, ie1]

                x1    = points_1[g1, ie1]
                wvol  = weights_1[g1, ie1]

#                v += bi_0 * x1 * (1.0 - x1) * wvol
                v += bi_0 * 2. * wvol


            rhs[i1] += v
    # ...

    # ...
    return rhs
    # ...
# ...


####################################################################################
if __name__ == '__main__':

    from spl.core.interface import make_open_knots
    from spl.fem.splines import SplineSpace

    p  = 3
    ne = 32
    n  = p + ne

    print('> Grid   :: {ne}'.format(ne=ne))
    print('> Degree :: {p}'.format(p=p))

    knots = make_open_knots(p, n)

    V = SplineSpace(knots, p)

    # ... builds matrices and rhs
    mass, stiffness = assembly_matrices(V)

    rhs  = assembly_rhs(V)
    # ...

    # ... apply homogeneous dirichlet boundary conditions
    rhs[0] = 0.
    rhs[V.nbasis-1] = 0.
    # ...

    # ... solve the system
    x, info = cg( stiffness, rhs, tol=1e-9, maxiter=1000, verbose=False )
    # ...

    # ... check
    print ('> info: ',info)
    # ...
