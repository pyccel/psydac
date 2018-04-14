# coding: utf-8
import numpy as np
from spl.utilities.quadratures import gauss_legendre
from spl.linalg.stencil     import VectorSpace, Vector, Matrix
from spl.linalg.solvers     import cg


# ... assembly of mass and stiffness matrices using stencil forms
def assembly_matrices(V, spans_1, basis_1, weights_1):

    # ... sizes
    [s1] = V.starts
    [e1] = V.ends
    [p1] = V.pads
    # ...

    # ... quadrature points number
    k1 = len(weights_1)

    # ... data structure
    mass      = Matrix(V, V)
    stiffness = Matrix(V, V)
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
def assembly_rhs(V, spans_1, basis_1, weights_1, points_1):

    # ... sizes
    [s1] = V.starts
    [e1] = V.ends
    [p1] = V.pads
    # ...

    # ... data structure
    rhs = Vector(V)
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

                v += bi_0 * x1 * (1.0 - x1) * wvol

            rhs[i1] += v
    # ...

    # ...
    return rhs
    # ...
# ...


####################################################################################
if __name__ == '__main__':
    from spl.core.bsp    import bsp_utils as bu

    # ... numbers of elements and degres
    ne1 = 8
    p1  = 2
    # ...

    # ... number of control points
    n1 = ne1 + p1

    # ... number of derivatives
    d1 = 1

    # ... knot vectors
    T1 = bu.make_open_knots(p1, n1)
    # ...

    # ...
    u1, w1 = gauss_legendre(p1)

    k1 = len(u1)

    # ... construct the quadrature points grid
    grid_1 = bu.construct_grid_from_knots(p1, n1, T1)

    points_1, weights_1 = bu.construct_quadrature_grid(ne1, k1, u1, w1, grid_1)

    basis_1 = bu.eval_on_grid_splines_ders(p1, n1, k1, d1, T1, points_1)

    spans_1 = bu.compute_spans(p1, n1, T1)

    # ... starts and ends
    s1 = 0
    e1 = n1-1
    # ...

    # ... VectorSpace
    V = VectorSpace([s1], [e1], [p1])

    # ... builds matrices and rhs
    mass, stiffness = assembly_matrices(V, spans_1, basis_1, weights_1)

    rhs  = assembly_rhs(V, spans_1, basis_1, weights_1, points_1)
    # ...

    # ... solve the system
    x, info = cg( mass, rhs, tol=1e-12, verbose=True )
    # ...

    # ... check
    print ('> info: ',info)
    # ...
