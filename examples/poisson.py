# coding: utf-8
import numpy as np
from spl.utilities.quadratures import gauss_legendre
from spl.linalg.stencil     import VectorSpace, Vector, Matrix
from spl.linalg.solvers     import cg


# ... assembly of mass and stiffness matrices using stencil forms
def assembly_matrices(V, spans, basis, weights):

    # ... sizes

    [s1, s2] = V.starts
    [e1, e2] = V.ends
    [p1, p2] = V.pads
    # ...

    # ... seetings
    [spans_1, spans_2] = spans
    [basis_1, basis_2] = basis
    [weights_1, weights_2] = weights
    # ...

    # ... quadrature points number
    k1 = len(weights_1)
    k2 = len(weights_2)

    # ... data structure
    mass      = Matrix(V, V)
    stiffness = Matrix(V, V)
    # ...

    # ... build matrices
    for ie1 in range(s1, e1+1-p1):
        for ie2 in range(s2, e2+1-p2):
            i_span_1 = spans_1[ie1]
            i_span_2 = spans_2[ie2]
            for il_1 in range(0, p1+1):
                for jl_1 in range(0, p1+1):
                    for il_2 in range(0, p2+1):
                        for jl_2 in range(0, p2+1):

                            i1 = i_span_1 - p1  - 1 + il_1
                            j1 = i_span_1 - p1  - 1 + jl_1

                            i2 = i_span_2 - p2  - 1 + il_2
                            j2 = i_span_2 - p2  - 1 + jl_2

                            v_m = 0.0
                            v_s = 0.0
                            for g1 in range(0, k1):
                                for g2 in range(0, k2):
                                    bi_0 = basis_1[il_1, 0, g1, ie1] * basis_2[il_2, 0, g2, ie2]
                                    bi_x = basis_1[il_1, 1, g1, ie1] * basis_2[il_2, 0, g2, ie2]
                                    bi_y = basis_1[il_1, 0, g1, ie1] * basis_2[il_2, 1, g2, ie2]

                                    bj_0 = basis_1[jl_1, 0, g1, ie1] * basis_2[jl_2, 0, g2, ie2]
                                    bj_x = basis_1[jl_1, 1, g1, ie1] * basis_2[jl_2, 0, g2, ie2]
                                    bj_y = basis_1[jl_1, 0, g1, ie1] * basis_2[jl_2, 1, g2, ie2]

                                    wvol = weights_1[g1, ie1] * weights_2[g2, ie2]

                                    v_m += bi_0 * bj_0 * wvol
                                    v_s += (bi_x * bj_x + bi_y * bj_y) * wvol

                            mass[i1, i2, j1 - i1, j2 - i2] += v_m
                            stiffness[i1, i2, j1 - i1, j2 - i2]  += v_s
    # ...

    # ...
    return mass , stiffness
    # ...
# ...

# ... example of assembly of the rhs: f(x1,x2) = x1*(1-x2)*x2*(1-x2)
def assembly_rhs(V, spans, basis, weights, points):

    # ... sizes
    [s1, s2] = V.starts
    [e1, e2] = V.ends
    [p1, p2] = V.pads
    # ...

    # ... seetings
    [spans_1, spans_2] = spans
    [basis_1, basis_2] = basis
    [weights_1, weights_2] = weights
    [points_1, points_2] = points
    # ...

    # ... data structure
    rhs = Vector(V)
    # ...

    # ... build rhs
    for ie1 in range(s1, e1+1-p1):
        for ie2 in range(s2, e2+1-p2):
            i_span_1 = spans_1[ie1]
            i_span_2 = spans_2[ie2]
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1  - 1 + il_1
                    i2 = i_span_2 - p2  - 1 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0 = basis_1[il_1, 0, g1, ie1] * basis_2[il_2, 0, g2, ie2]
                            bi_x = basis_1[il_1, 1, g1, ie1] * basis_2[il_2, 0, g2, ie2]
                            bi_y = basis_1[il_1, 0, g1, ie1] * basis_2[il_2, 1, g2, ie2]

                            x1    = points_1[g1, ie1]
                            x2    = points_2[g2, ie2]
                            wvol  = weights_1[g1, ie1] * weights_2[g2, ie2]

                            v += bi_0 * x1 * (1.0 - x1) * x2 * (1.0 - x2) * wvol

                    rhs[i1, i2] += v
    # ...

    # ...
    return rhs
    # ...
# ...


####################################################################################
if __name__ == '__main__':
    from spl.core.bsp    import bsp_utils as bu

    # ... numbers of elements and degres
    ne1 = 8 ;  ne2 = 8
    p1  = 2 ;  p2  = 2
    # ...

    # ... number of control points
    n1 = ne1 + p1
    n2 = ne2 + p2

    # ... number of derivatives
    d1 = 1
    d2 = 1

    # ... knot vectors
    T1 = bu.make_open_knots(p1, n1)
    T2 = bu.make_open_knots(p2, n2)
    # ...

    # ...
    u1, w1 = gauss_legendre(p1)
    u2, w2 = gauss_legendre(p2)

    k1 = len(u1)
    k2 = len(u2)

    # ... construct the quadrature points grid
    grid_1 = bu.construct_grid_from_knots(p1, n1, T1)
    grid_2 = bu.construct_grid_from_knots(p2, n2, T2)

    points_1, weights_1 = bu.construct_quadrature_grid(ne1, k1, u1, w1, grid_1)
    points_2, weights_2 = bu.construct_quadrature_grid(ne2, k2, u2, w2, grid_2)

    basis_1 = bu.eval_on_grid_splines_ders(p1, n1, k1, d1, T1, points_1)
    basis_2 = bu.eval_on_grid_splines_ders(p2, n2, k2, d2, T2, points_2)

    spans_1 = bu.compute_spans(p1, n1, T1)
    spans_2 = bu.compute_spans(p2, n2, T2)

    # ... starts and ends
    s1 = 0
    e1 = n1-1

    s2 = 0
    e2 = n2-1
    # ...

    # ... VectorSpace
    V = VectorSpace((s1, s2), (e1, e2), (p1, p2))

    # ... builds matrices and rhs
    mass, stiffness = assembly_matrices(V, \
                                        [spans_1, spans_2], [basis_1, basis_2],\
                                        [weights_1, weights_2])

    rhs  = assembly_rhs(V, \
                        [spans_1, spans_2], [basis_1, basis_2],\
                        [weights_1, weights_2], [points_1, points_2])
    # ...

    # ... solve the system
    x, info = cg( mass, rhs, tol=1e-12, verbose=True )
    # ...

    # ... check
    print ('> info: ',info)
    # ...
#
#    Mc = mass.tocoo()
#    print ('>>> shape and nnz: ', np.shape(Mc), Mc.nnz)
#
#    # ... plot and print mass matrix
#    import matplotlib.pyplot as plt
#    import matplotlib.colors as colors
#    plt.matshow(Mc.todense(), norm=colors.LogNorm())
#    plt.title('Mass matrix')
#    plt.show()

