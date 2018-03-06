# coding: utf-8
import numpy as np
from spl.quadratures import gauss_legendre
from spl.stencil     import Matrix, Vector

# ... assembly of mass and stiffness matrices using stencil forms
def assembly_matrices(starts, ends, pads, spans, basis, weights):

    # ... sizes
    [s1, s2] = starts
    [e1, e2] = ends
    [p1, p2] = pads
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
    mass      = Matrix((s1, s2), (e1, e2), (p1, p2))
    stiffness = Matrix((s1, s2), (e1, e2), (p1, p2))
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

                            mass[j1 - i1, j2 - i2, i1, i2] += v_m
                            stiffness[j1 - i1, j2 - i2, i1, i2]  += v_s
    # ...

    # ...
    return mass , stiffness
    # ...
# ...

# ... example of assembly of the rhs: f(x1,x2) = x1*(1-x2)*x2*(1-x2)
def assembly_rhs(starts, ends, pads, spans, basis, weights, points):

    # ... sizes
    [s1, s2] = starts
    [e1, e2] = ends
    [p1, p2] = pads
    # ...

    # ... seetings
    [spans_1, spans_2] = spans
    [basis_1, basis_2] = basis
    [weights_1, weights_2] = weights
    [points_1, points_2] = points
    # ...

    # ... data structure
    rhs = Vector((s1, s2), (e1, e2), (p1, p2))
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

# ... Solver: CGL performs maxit CG iterations on the linear system Ax = b
#     starting from x = x0
def cgl(mat, b, x0, maxit, tol):
    xk = x0.zeros_like()
    mx = x0.zeros_like()
    p  = x0.zeros_like()
    q  = x0.zeros_like()
    r  = x0.zeros_like()

    # xk = x0
    xk = x0.copy()
    mx = mat.dot(x0)

    # r = b - mx
    r = b.copy()
    b.sub(mx)

    # p = r
    p = r.copy()

    rdr = r.dot(r)

    for i_iter in range(1, maxit+1):
        q = mat.dot(p)
        alpha = rdr / p.dot(q)

        # xk = xk + alpha * p
        ap = p.copy()
        ap.mul(alpha)
        xk.add(ap)

        # r  = r - alpha * q
        aq = q.copy()
        aq.mul(alpha)
        r.sub(aq)

        # ... TODO check why r.dot r can be < 0
        if r.dot(r) >= 0.:
            norm_err = np.sqrt(r.dot(r))
            print (i_iter, norm_err )

            if norm_err < tol:
                x0 = xk.copy()
                break

        rdrold = rdr
        rdr = r.dot(r)
        beta = rdr / rdrold

        #p = r + beta * p
        bp = p.copy()
        bp.mul(beta)
        p  = r.copy()
        p.add(bp)

    x0 = xk.copy()
    # ...

    return x0
# ....

####################################################################################
if __name__ == '__main__':
    from spl.core.bsp    import bsp_utils as bu

    # ... numbers of elements and degres
    ne1 = 32 ;  ne2 = 32
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

    # ... builds matrices and rhs
    mass, stiffness = assembly_matrices([s1, s2], [e1, e2], [p1, p2], \
                                        [spans_1, spans_2], [basis_1, basis_2],\
                                        [weights_1, weights_2])

    rhs  = assembly_rhs([s1, s2], [e1, e2], [p1, p2], \
                        [spans_1, spans_2], [basis_1, basis_2],\
                        [weights_1, weights_2], [points_1, points_2])
    # ...

    # ...
    x0 = Vector((s1, s2), (e1, e2), (p1, p2))
    xn = Vector((s1, s2), (e1, e2), (p1, p2))
    y  = Vector((s1, s2), (e1, e2), (p1, p2))
    # ...

    # ...
    n_maxiter = 100
    tol = 1.0e-7
    # ...

    # ... solve the system
    xn[:, :] = 0.0
    xn = cgl(mass, rhs, xn, n_maxiter, tol)
    # ...

    # ... check
    x0 = mass.dot(xn)

    e = x0.copy()
    e.sub(rhs)

    print ('> residual error = ', max(abs(e.toarray())))
    # ...

