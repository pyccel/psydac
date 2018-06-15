# coding: utf-8
import numpy as np

from spl.utilities.quadratures import gauss_legendre
from spl.linalg.stencil        import StencilVector, StencilMatrix
from spl.linalg.solvers        import cg

# ... assembly of mass and stiffness matrices using stencil forms
def assembly_matrices(V):

    # ... sizes
    [s1, s2] = V.vector_space.starts
    [e1, e2] = V.vector_space.ends
    [p1, p2] = V.vector_space.pads
    # ...

    # ... seetings
    [       k1,        k2] = [W.quad_order   for W in V.spaces]
    [  spans_1,   spans_2] = [W.spans        for W in V.spaces]
    [  basis_1,   basis_2] = [W.quad_basis   for W in V.spaces]
    [ points_1,  points_2] = [W.quad_points  for W in V.spaces]
    [weights_1, weights_2] = [W.quad_weights for W in V.spaces]
    # ...

    # ... data structure
    mass      = StencilMatrix( V.vector_space, V.vector_space )
    stiffness = StencilMatrix( V.vector_space, V.vector_space )
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
def assembly_rhs(V):

    # ... sizes
    [s1, s2] = V.vector_space.starts
    [e1, e2] = V.vector_space.ends
    [p1, p2] = V.vector_space.pads
    # ...

    # ... seetings
    [       k1,        k2] = [W.quad_order   for W in V.spaces]
    [  spans_1,   spans_2] = [W.spans        for W in V.spaces]
    [  basis_1,   basis_2] = [W.quad_basis   for W in V.spaces]
    [ points_1,  points_2] = [W.quad_points  for W in V.spaces]
    [weights_1, weights_2] = [W.quad_weights for W in V.spaces]
    # ...

    # ... data structure
    rhs = StencilVector( V.vector_space )
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

#                            v += bi_0 * x1 * (1.0 - x1) * x2 * (1.0 - x2) * wvol
                            v += bi_0 * 2. * (x1 * (1.0 - x1) + x2 * (1.0 - x2)) * wvol

                    rhs[i1, i2] += v
    # ...

    # ...
    return rhs
    # ...
# ...


####################################################################################
if __name__ == '__main__':

    from numpy import linspace
    from spl.fem.splines import SplineSpace
    from spl.fem.tensor  import TensorFemSpace

    # ... numbers of elements and degres
    p1  = 2 ; p2  = 2
    ne1 = 8 ; ne2 = 8
    # ...

    print('> Grid   :: [{ne1},{ne2}]'.format(ne1=ne1, ne2=ne2))
    print('> Degree :: [{p1},{p2}]'.format(p1=p1, p2=p2))

    grid_1 = linspace(0., 1., ne1+1)
    grid_2 = linspace(0., 1., ne2+1)

    V1 = SplineSpace(p1, grid=grid_1)
    V2 = SplineSpace(p2, grid=grid_2)

    V = TensorFemSpace(V1, V2)

    # ... builds matrices and rhs
    mass, stiffness = assembly_matrices(V)

    rhs  = assembly_rhs(V)
    # ...

    # ... apply homogeneous dirichlet boundary conditions
    # left  bc at x=0.
    for j in range(0, V2.nbasis):
        rhs[0, j] = 0.
    # right bc at x=1.
    for j in range(0, V2.nbasis):
        rhs[V1.nbasis-1, j] = 0.
    # lower bc at y=0.
    for i in range(0, V1.nbasis):
        rhs[i, 0] = 0.
    # upper bc at y=1.
    for i in range(0, V1.nbasis):
        rhs[i, V2.nbasis-1] = 0.
    # ...

    # ... solve the system
    x, info = cg( stiffness, rhs, tol=1e-9, maxiter=1000, verbose=False )
    # ...

    # ... check
    print ('> info: ',info)
    # ...
