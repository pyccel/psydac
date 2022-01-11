# -*- coding: UTF-8 -*-
from time import time
ttt1 = time()
from mpi4py import MPI
from sympy  import pi, sin, symbols
import pytest

from sympde.calculus import grad, dot
from sympde.calculus import laplace
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of
from sympde.topology import NormalVector
from sympde.topology import Square, Cube
from sympde.topology import Union
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm
from sympde.expr     import find, EssentialBC

from psydac.api.discretization import discretize
from psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL, PSYDAC_BACKEND_NUMBA
from pyccel.epyccel            import epyccel
from psydac.linalg.stencil import StencilVector

x,y,z = symbols('x1, x2, x3')

comm = MPI.COMM_WORLD

#==============================================================================
def get_boundaries(*args):

    if not args:
        return ()
    else:
        assert all(1 <= a <= 6 for a in args)
        assert len(set(args)) == len(args)

    boundaries = {1: {'axis': 0, 'ext': -1},
                  2: {'axis': 0, 'ext':  1},
                  3: {'axis': 1, 'ext': -1},
                  4: {'axis': 1, 'ext':  1},
                  5: {'axis': 2, 'ext': -1},
                  6: {'axis': 2, 'ext':  1}}

    return tuple(boundaries[i] for i in args)
ttt2 = time()

from pyccel.decorators import types
@types("real[:,:,:,:,:,:]", "real[:,:,:]", "real[:,:,:]", "int", "int", "int", "int", "int", "int")
def lo_dot(mat, x, out, s1, s2, s3, n1, n2, n3):
    from pyccel.stdlib.internal.blas import ddot
    import numpy as np
    #$ omp parallel default(private) shared(mat,x,out) firstprivate(s1,s2,s3,n1,n2,n3)
    #$ omp for schedule(static)  collapse(3)
    for i1 in range(0, n1, 1):
        for i2 in range(0, n2, 1):
            for i3 in range(0, n3, 1):
#                v = sum(mat[4 + i1,4 + i2,4 + i3,:,:,:]*x[i1:i1+9,i2:i2+9,i3:i3+9])
#                for k1 in range(0, 9, 1):
#                    for k2 in range(0, 9, 1):
#                        for k3 in range(0, 9, 1):
#                            v += mat[4 + i1,4 + i2,4 + i3,k1,k2,k3]*x[i1 + k1,i2 + k2,i3 + k3]



                out[4 + i1,4 + i2,4 + i3] = ddot(np.int32(729), mat[4 + i1,4 + i2,4 + i3,:,:,:],np.int32(1), x[i1:i1+9,i2:i2+9,i3:i3+9],np.int32(1))
    #$ omp end parallel
    return

@types("real[:,:,:,:]", "real[:,:,:,:]", "real[:,:,:,:]", "real[:,:,:,:]", "real[:,:,:,:]", "real[:,:,:,:]", "int[:]", "int[:]", "int[:]", "real[:,:]", "real[:,:]", "real[:,:]", "int", "int", "int", "int", "int", "int", "int", "int", "int", "int", "int", "int", "int", "int", "int", "real[:,:,:,:,:,:]", "real[:,:,:,:,:,:]", "int", "int", "int", "int", "int", "int")
def assembly(global_test_basis_v_1, global_test_basis_v_2, global_test_basis_v_3, global_trial_basis_u_1, global_trial_basis_u_2, global_trial_basis_u_3, global_span_v_1, global_span_v_2, global_span_v_3, global_x1, global_x2, global_x3, test_v_p1, test_v_p2, test_v_p3, trial_u_p1, trial_u_p2, trial_u_p3, n_element_1, n_element_2, n_element_3, k1, k2, k3, pad1, pad2, pad3, l_mat_u_v_9gl97406, g_mat_u_v_9gl97406, b01, b02, b03, e01, e02, e03):

    from pyccel.stdlib.internal.openmp import omp_get_num_threads,omp_get_thread_num
    from numpy import zeros, zeros_like

    next_thread_1_span_v1  = zeros_like(global_span_v_1)
    next_thread_2_span_v2  = zeros_like(global_span_v_2)
    next_thread_3_span_v3  = zeros_like(global_span_v_3)

    next_thread_1_span_v1[:]  = global_span_v_1[n_element_1-1] + 20
    next_thread_2_span_v2[:]  = global_span_v_2[n_element_2-1] + 20
    next_thread_3_span_v3[:]  = global_span_v_3[n_element_3-1] + 20

    #$ omp parallel default(private) shared(next_thread_1_span_v1, next_thread_2_span_v2, next_thread_3_span_v3, global_test_basis_v_1,global_test_basis_v_2,global_test_basis_v_3,global_trial_basis_u_1,global_trial_basis_u_2,global_trial_basis_u_3,global_span_v_1,global_span_v_2,global_span_v_3,global_x1,global_x2,global_x3,g_mat_u_v_9gl97406,l_mat_u_v_9gl97406)&
    #$ firstprivate(test_v_p1, test_v_p2, test_v_p3, trial_u_p1, trial_u_p2, trial_u_p3, n_element_1, n_element_2, n_element_3, k1, k2, k3, pad1, pad2, pad3,b01, b02, b03, e01, e02, e03)
    local_x1 = zeros_like(global_x1[0, : ])
    local_x2 = zeros_like(global_x2[0, : ])
    local_x3 = zeros_like(global_x3[0, : ])
    l_mat_u_v = zeros((5,5,5,11,11,11))
    nn = omp_get_thread_num()
    #$ omp for schedule(static)  collapse(3)
    for i_element_1 in range(0, n_element_1, 1):
        for i_element_2 in range(0, n_element_2, 1):
            for i_element_3 in range(0, n_element_3, 1):
                local_x1[ : ] = global_x1[i_element_1, : ]
                span_v_1 = global_span_v_1[i_element_1]
                b_v_1 = max((3 - i_element_1)*b01, 0)
                e_v_1 = max((4 + i_element_1 - n_element_1)*e01, 0)
                local_x2[ : ] = global_x2[i_element_2, : ]
                span_v_2 = global_span_v_2[i_element_2]
                b_v_2 = max((3 - i_element_2)*b02, 0)
                e_v_2 = max((4 + i_element_2 - n_element_2)*e02, 0)
                local_x3[ : ] = global_x3[i_element_3, : ]
                span_v_3 = global_span_v_3[i_element_3]
                b_v_3 = max((3 - i_element_3)*b03, 0)
                e_v_3 = max((4 + i_element_3 - n_element_3)*e03, 0)
                next_thread_1_span_v1[i_element_1] += 1
                next_thread_2_span_v2[i_element_2] += 1
                next_thread_3_span_v3[i_element_3] += 1

                l_mat_u_v[ : , : , : , : , : , : ] = 0.0
                for i_basis_1 in range(b_v_1, 5 - e_v_1, 1):
                    for i_basis_2 in range(b_v_2, 5 - e_v_2, 1):
                        for i_basis_3 in range(b_v_3, 5 - e_v_3, 1):
                            for j_basis_1 in range(0, 5, 1):
                                for j_basis_2 in range(0, 5, 1):
                                    for j_basis_3 in range(0, 5, 1):
                                        for i_quad_1 in range(0, 5, 1):
                                            x1 = local_x1[i_quad_1]
                                            v_1 = global_test_basis_v_1[i_element_1,i_basis_1,0,i_quad_1]
                                            v_1_x1 = global_test_basis_v_1[i_element_1,i_basis_1,1,i_quad_1]
                                            u_1 = global_trial_basis_u_1[i_element_1,j_basis_1,0,i_quad_1]
                                            u_1_x1 = global_trial_basis_u_1[i_element_1,j_basis_1,1,i_quad_1]
                                            for i_quad_2 in range(0, 5, 1):
                                                x2 = local_x2[i_quad_2]
                                                v_2 = global_test_basis_v_2[i_element_2,i_basis_2,0,i_quad_2]
                                                v_2_x2 = global_test_basis_v_2[i_element_2,i_basis_2,1,i_quad_2]
                                                u_2 = global_trial_basis_u_2[i_element_2,j_basis_2,0,i_quad_2]
                                                u_2_x2 = global_trial_basis_u_2[i_element_2,j_basis_2,1,i_quad_2]
                                                for i_quad_3 in range(0, 5, 1):
                                                    x3 = local_x3[i_quad_3]
                                                    v_3 = global_test_basis_v_3[i_element_3,i_basis_3,0,i_quad_3]
                                                    v_3_x3 = global_test_basis_v_3[i_element_3,i_basis_3,1,i_quad_3]
                                                    u_3 = global_trial_basis_u_3[i_element_3,j_basis_3,0,i_quad_3]
                                                    u_3_x3 = global_trial_basis_u_3[i_element_3,j_basis_3,1,i_quad_3]
                                                    v = v_1*v_2*v_3
                                                    v_x3 = v_1*v_2*v_3_x3
                                                    v_x2 = v_1*v_2_x2*v_3
                                                    v_x1 = v_1_x1*v_2*v_3
                                                    u = u_1*u_2*u_3
                                                    u_x3 = u_1*u_2*u_3_x3
                                                    u_x2 = u_1*u_2_x2*u_3
                                                    u_x1 = u_1_x1*u_2*u_3
                                                    l_mat_u_v[i_basis_1,i_basis_2,i_basis_3,4 - i_basis_1 + j_basis_1,4 - i_basis_2 + j_basis_2,4 - i_basis_3 + j_basis_3] += u_x1*v_x1 + u_x2*v_x2 + u_x3*v_x3
                while not (pad1 + span_v_1 < next_thread_1_span_v1[i_element_1] and pad2 + span_v_2 < next_thread_2_span_v2[i_element_2] and pad3 + span_v_3 < next_thread_3_span_v3[i_element_3]):
                    continue
                g_mat_u_v_9gl97406[pad1 + span_v_1 - test_v_p1 : 1 + pad1 + span_v_1,pad2 + span_v_2 - test_v_p2 : 1 + pad2 + span_v_2,pad3 + span_v_3 - test_v_p3 : 1 + pad3 + span_v_3, : , : , : ] += l_mat_u_v[ : , : , : , : , : , : ]
    #$ omp end parallel
    return
assembly = epyccel(assembly, 
                  accelerators=['openmp'],
                  verbose=True,
                  fflags ='-O3 -march=native -mtune=native  -mavx -ffast-math',
                  comm        = comm,
                  bcast       = True)
lo_dot  = epyccel(lo_dot,
                  accelerators=['openmp'],
                  verbose=True,
                  fflags ='-O3 -march=native -mtune=native -mavx -ffast-math',
                  comm        = comm,
                  bcast       = True)
#==============================================================================
def run_poisson_2d(solution, f, dir_zero_boundary, dir_nonzero_boundary,
        ncells, degree, backend=None, comm=None):
    
    comm.Barrier()

    assert isinstance(dir_zero_boundary   , (list, tuple))
    assert isinstance(dir_nonzero_boundary, (list, tuple))
    t1 = MPI.Wtime()
    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++
    domain = Cube()

    B_dirichlet_0 = Union(*[domain.get_boundary(**kw) for kw in dir_zero_boundary])
    B_dirichlet_i = Union(*[domain.get_boundary(**kw) for kw in dir_nonzero_boundary])
    B_dirichlet   = Union(B_dirichlet_0, B_dirichlet_i)
    B_neumann = domain.boundary.complement(B_dirichlet)

    V  = ScalarFunctionSpace('V', domain)
    u  = element_of(V, name='u')
    v  = element_of(V, name='v')
    nn = NormalVector('nn')

    # Bilinear form a: V x V --> R
    a = BilinearForm((u, v), integral(domain, dot(grad(u), grad(v))))

    # Linear form l: V --> R
    l0 = LinearForm(v, integral(domain, f * v))
    if B_neumann:
        l1 = LinearForm(v, integral(B_neumann, v * dot(grad(solution), nn)))
        l  = LinearForm(v, l0(v) + l1(v))
    else:
        l = l0

    # Dirichlet boundary conditions
    bc = []
    if B_dirichlet_0:  bc += [EssentialBC(u,        0, B_dirichlet_0)]
    if B_dirichlet_i:  bc += [EssentialBC(u, solution, B_dirichlet_i)]

    # Variational model
    equation = find(u, forall=v, lhs=a(u, v), rhs=l(v), bc=bc)

    # Error norms
    error  = u - solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    # Create computational domain from topological domain
    domain_h = discretize(domain, ncells=ncells, comm=comm)

    # Discrete spaces
    Vh = discretize(V, domain_h, degree=degree)

    # Discretize equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh], backend=backend)

    #+++++++++++++++++++++++++++++++
    # 3. Solution
    #+++++++++++++++++++++++++++++++
        
    # Solve linear system
    M = equation_h.lhs
    M._func = assembly
    M._matrix._func = lo_dot
    t2 = MPI.Wtime()
    comm.Barrier()
    a = MPI.Wtime()
    A = M.assemble()
    b = MPI.Wtime()
    print('The partition:',Vh.vector_space.cart._dims)
    print('The assembly execution time is:', b-a)
    print('The preparation time is:', t2-t1)
    print('The import  time is:', ttt2-ttt1)

    b = equation_h.rhs.assemble()
    out = b.copy()
    T = 0
    for i in range(10):
        comm.Barrier()
        t1 = MPI.Wtime()
        b.update_ghost_regions()
        A.dot(b, out=out)
        t2 = MPI.Wtime()
        T += t2-t1

    xx = A.dot(b)
    xx = xx.dot(xx)
    b  = b.dot(b)
    print('The dot execution time is :', T/10)
    print('The dot results:', xx, b)
    
#==============================================================================
# 2D Poisson's equation
#==============================================================================
def test_poisson_2d_dir0_1234_gpyccel(ncells, degree):

    solution = sin(pi*x)*sin(pi*y)*sin(pi*z)
    f        = 3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)

    dir_zero_boundary    = get_boundaries(1, 2, 3, 4, 5, 6)
    dir_nonzero_boundary = get_boundaries()

    run_poisson_2d(solution, f, dir_zero_boundary,
            dir_nonzero_boundary, ncells=ncells, degree=degree, backend=PSYDAC_BACKEND_GPYCCEL, comm=comm)


#==============================================================================
# Parser
#==============================================================================
def parse_input_arguments():

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        description     = "Solve Poisson's equation on a 3D domain."
    )

    parser.add_argument( '-d',
        type    = int,
        nargs   = 3,
        default = [2,2,2],
        metavar = ('P1','P2','P3'),
        dest    = 'degree',
        help    = 'Spline degree along each dimension'
    )

    parser.add_argument( '-n',
        type    = int,
        nargs   = 3,
        default = [10,10,10],
        metavar = ('N1','N2','N3'),
        dest    = 'ncells',
        help    = 'Number of grid cells (elements) along each dimension'
    )


    return parser.parse_args()

#==============================================================================
# Script functionality
#==============================================================================
if __name__ == '__main__':

    args = parse_input_arguments()
    test_poisson_2d_dir0_1234_gpyccel(**vars(args))                                                 
