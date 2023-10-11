# -*- coding: UTF-8 -*-

from psydac.feec.global_projectors import Projector_H1, Projector_L2, Projector_Hcurl, Projector_Hdiv
from psydac.fem.tensor       import TensorFemSpace, SplineSpace
from psydac.fem.vector       import VectorFemSpace
from psydac.linalg.block     import BlockVector
from psydac.core.bsplines    import make_knots
from psydac.feec.derivatives import Derivative_1D, Gradient_2D, Gradient_3D
from psydac.feec.derivatives import ScalarCurl_2D, VectorCurl_2D, Curl_3D
from psydac.feec.derivatives import Divergence_2D, Divergence_3D
from psydac.ddm.cart         import DomainDecomposition

from mpi4py import MPI
import numpy as np
import pytest

#==============================================================================
# Run test
#==============================================================================
@pytest.mark.parametrize('Nel', [8, 12])
@pytest.mark.parametrize('Nq', [5])
@pytest.mark.parametrize('p', [2,3])
@pytest.mark.parametrize('bc', [True, False])
def test_3d_commuting_pro_1(Nel, Nq, p, bc):

    fun1    = lambda xi1, xi2, xi3 : np.sin(xi1)*np.sin(xi2)*np.sin(xi3)
    D1fun1  = lambda xi1, xi2, xi3 : np.cos(xi1)*np.sin(xi2)*np.sin(xi3)
    D2fun1  = lambda xi1, xi2, xi3 : np.sin(xi1)*np.cos(xi2)*np.sin(xi3)
    D3fun1  = lambda xi1, xi2, xi3 : np.sin(xi1)*np.sin(xi2)*np.cos(xi3)

    Nel = [Nel]*3
    Nq  = [Nq]*3
    p   = [p]*3
    bc  = [bc]*3

    # Side lengths of logical cube [0, L]^3
    L = [2*np.pi, 2*np.pi , 2*np.pi]

    # element boundaries
    el_b = [np.linspace(0., L_i, Nel_i + 1) for L_i, Nel_i in zip(L, Nel)] 

    # knot sequences
    knots = [make_knots(el_b_i, p_i, bc_i) for el_b_i, p_i, bc_i in zip(el_b, p, bc)]

    Vs     = [SplineSpace(pi, knots=Ti, periodic=periodic, basis='B') for pi, Ti, periodic in zip(p, knots, bc)]

    domain_decomposition = DomainDecomposition(Nel, bc, comm=MPI.COMM_WORLD)
    
    H1     = TensorFemSpace(domain_decomposition, *Vs)

    spaces = [H1.reduce_degree(axes=[0], basis='M'),
              H1.reduce_degree(axes=[1], basis='M'),
              H1.reduce_degree(axes=[2], basis='M')]

    Hcurl  = VectorFemSpace(*spaces)

    # create an instance of the H1 projector class
    P0 = Projector_H1(H1)

    # Build linear operators on stencil arrays
    grad = Gradient_3D(H1, Hcurl)

    # create an instance of the projector class
    P1 = Projector_Hcurl(Hcurl, Nq)
    #-------------------------------------
    # Projections and discrete derivatives
    #-------------------------------------

    u0        = P0(fun1)
    u1        = P1((D1fun1, D2fun1, D3fun1))
    Dfun_h    = grad(u0)
    Dfun_proj = u1

    error = (Dfun_proj.coeffs-Dfun_h.coeffs).toarray()
    assert abs(error).max()<1e-9

#==============================================================================
@pytest.mark.parametrize('Nel', [8, 12])
@pytest.mark.parametrize('Nq', [8])
@pytest.mark.parametrize('p', [2,3])
@pytest.mark.parametrize('bc', [True, False])
def test_3d_commuting_pro_2(Nel, Nq, p, bc):

    fun1    = lambda xi1, xi2, xi3 : np.sin(xi1)*np.sin(xi2)*np.sin(xi3)
    D1fun1  = lambda xi1, xi2, xi3 : np.cos(xi1)*np.sin(xi2)*np.sin(xi3)
    D2fun1  = lambda xi1, xi2, xi3 : np.sin(xi1)*np.cos(xi2)*np.sin(xi3)
    D3fun1  = lambda xi1, xi2, xi3 : np.sin(xi1)*np.sin(xi2)*np.cos(xi3)

    fun2    = lambda xi1, xi2, xi3 :   np.sin(2*xi1)*np.sin(2*xi2)*np.sin(2*xi3)
    D1fun2  = lambda xi1, xi2, xi3 : 2*np.cos(2*xi1)*np.sin(2*xi2)*np.sin(2*xi3)
    D2fun2  = lambda xi1, xi2, xi3 : 2*np.sin(2*xi1)*np.cos(2*xi2)*np.sin(2*xi3)
    D3fun2  = lambda xi1, xi2, xi3 : 2*np.sin(2*xi1)*np.sin(2*xi2)*np.cos(2*xi3)

    fun3    = lambda xi1, xi2, xi3 :   np.sin(3*xi1)*np.sin(3*xi2)*np.sin(3*xi3)
    D1fun3  = lambda xi1, xi2, xi3 : 3*np.cos(3*xi1)*np.sin(3*xi2)*np.sin(3*xi3)
    D2fun3  = lambda xi1, xi2, xi3 : 3*np.sin(3*xi1)*np.cos(3*xi2)*np.sin(3*xi3)
    D3fun3  = lambda xi1, xi2, xi3 : 3*np.sin(3*xi1)*np.sin(3*xi2)*np.cos(3*xi3)

    cf1 = lambda xi1, xi2, xi3 : D2fun3(xi1, xi2, xi3) - D3fun2(xi1, xi2, xi3)
    cf2 = lambda xi1, xi2, xi3 : D3fun1(xi1, xi2, xi3) - D1fun3(xi1, xi2, xi3)
    cf3 = lambda xi1, xi2, xi3 : D1fun2(xi1, xi2, xi3) - D2fun1(xi1, xi2, xi3)

    Nel = [Nel]*3
    Nq  = [Nq]*3
    p   = [p]*3
    bc  = [bc]*3

    # Side lengths of logical cube [0, L]^3
    L = [2*np.pi, 2*np.pi , 2*np.pi]

    # element boundaries
    el_b = [np.linspace(0., L_i, Nel_i + 1) for L_i, Nel_i in zip(L, Nel)] 

    # knot sequences
    knots = [make_knots(el_b_i, p_i, bc_i) for el_b_i, p_i, bc_i in zip(el_b, p, bc)]

    Vs     = [SplineSpace(pi, knots=Ti, periodic=periodic, basis='B') for pi, Ti, periodic in zip(p, knots, bc)]

    domain_decomposition = DomainDecomposition(Nel, bc)
    H1       = TensorFemSpace(domain_decomposition, *Vs)

    spaces = [H1.reduce_degree(axes=[0], basis='M'),
              H1.reduce_degree(axes=[1], basis='M'),
              H1.reduce_degree(axes=[2], basis='M')]

    Hcurl  = VectorFemSpace(*spaces)

    spaces = [H1.reduce_degree(axes=[1,2], basis='M'),
              H1.reduce_degree(axes=[0,2], basis='M'),
              H1.reduce_degree(axes=[0,1], basis='M')]

    Hdiv  = VectorFemSpace(*spaces)

    # Build linear operators on stencil arrays
    curl = Curl_3D(Hcurl, Hdiv)

    # create an instance of the projector class
    P1 = Projector_Hcurl(Hcurl, Nq)
    P2 = Projector_Hdiv(Hdiv, Nq)

    #-------------------------------------
    # Projections and discrete derivatives
    #-------------------------------------
    u1        = P1((fun1, fun2, fun3))
    u2        = P2((cf1, cf2, cf3))
    Dfun_h    = curl(u1)
    Dfun_proj = u2

    error = (Dfun_proj.coeffs-Dfun_h.coeffs).toarray()
    assert abs(error).max()<1e-9

#==============================================================================
@pytest.mark.parametrize('Nel', [8, 12])
@pytest.mark.parametrize('Nq', [8])
@pytest.mark.parametrize('p', [2,3])
@pytest.mark.parametrize('bc', [True, False])
def test_3d_commuting_pro_3(Nel, Nq, p, bc):

    fun1    = lambda xi1, xi2, xi3 : np.sin(xi1)*np.sin(xi2)*np.sin(xi3)
    D1fun1  = lambda xi1, xi2, xi3 : np.cos(xi1)*np.sin(xi2)*np.sin(xi3)

    fun2    = lambda xi1, xi2, xi3 :   np.sin(2*xi1)*np.sin(2*xi2)*np.sin(2*xi3)
    D2fun2  = lambda xi1, xi2, xi3 : 2*np.sin(2*xi1)*np.cos(2*xi2)*np.sin(2*xi3)

    fun3    = lambda xi1, xi2, xi3 :   np.sin(3*xi1)*np.sin(3*xi2)*np.sin(3*xi3)
    D3fun3  = lambda xi1, xi2, xi3 : 3*np.sin(3*xi1)*np.sin(3*xi2)*np.cos(3*xi3)

    difun = lambda xi1, xi2, xi3 : D1fun1(xi1, xi2, xi3)+ D2fun2(xi1, xi2, xi3) + D3fun3(xi1, xi2, xi3)

    Nel = [Nel]*3
    Nq  = [Nq]*3
    p   = [p]*3
    bc  = [bc]*3

    # Side lengths of logical cube [0, L]^3
    L = [2*np.pi, 2*np.pi , 2*np.pi]

    # element boundaries
    el_b = [np.linspace(0., L_i, Nel_i + 1) for L_i, Nel_i in zip(L, Nel)] 

    # knot sequences
    knots = [make_knots(el_b_i, p_i, bc_i) for el_b_i, p_i, bc_i in zip(el_b, p, bc)]

    Vs     = [SplineSpace(pi, knots=Ti, periodic=periodic, basis='B') for pi, Ti, periodic in zip(p, knots, bc)]

    domain_decomposition = DomainDecomposition(Nel, bc, comm=MPI.COMM_WORLD)
    H1       = TensorFemSpace(domain_decomposition, *Vs)

    spaces = [H1.reduce_degree(axes=[1,2], basis='M'),
              H1.reduce_degree(axes=[0,2], basis='M'),
              H1.reduce_degree(axes=[0,1], basis='M')]

    Hdiv  = VectorFemSpace(*spaces)

    L2  = H1.reduce_degree(axes=[0,1,2], basis='M')

    # create an instance of the H1 projector class

    # Build linear operators on stencil arrays
    div  = Divergence_3D(Hdiv, L2)

    # create an instance of the projector class
    P2 = Projector_Hdiv(Hdiv, Nq)
    P3 = Projector_L2(L2, Nq)

    #-------------------------------------
    # Projections and discrete derivatives
    #-------------------------------------

    u2        = P2((fun1, fun2, fun3))
    u3        = P3(difun)
    Dfun_h    = div(u2)
    Dfun_proj = u3

    error = (Dfun_proj.coeffs-Dfun_h.coeffs).toarray()
    assert abs(error).max()<1e-9

@pytest.mark.parametrize('Nel', [8, 12])
@pytest.mark.parametrize('Nq', [5])
@pytest.mark.parametrize('p', [2,3])
@pytest.mark.parametrize('bc', [True, False])
def test_2d_commuting_pro_1(Nel, Nq, p, bc):

    fun1    = lambda xi1, xi2 : np.sin(xi1)*np.sin(xi2)
    D1fun1  = lambda xi1, xi2 : np.cos(xi1)*np.sin(xi2)
    D2fun1  = lambda xi1, xi2 : np.sin(xi1)*np.cos(xi2)

    Nel = [Nel]*2
    Nq  = [Nq]*2
    p   = [p]*2
    bc  = [bc]*2

    # Side lengths of logical cube [0, L]^2
    L = [2*np.pi, 2*np.pi]

    # element boundaries
    el_b = [np.linspace(0., L_i, Nel_i + 1) for L_i, Nel_i in zip(L, Nel)]

    # knot sequences
    knots = [make_knots(el_b_i, p_i, bc_i) for el_b_i, p_i, bc_i in zip(el_b, p, bc)]

    Vs     = [SplineSpace(pi, knots=Ti, periodic=periodic, basis='B') for pi, Ti, periodic in zip(p, knots, bc)]

    domain_decomposition = DomainDecomposition(Nel, bc, comm=MPI.COMM_WORLD)
    H1       = TensorFemSpace(domain_decomposition, *Vs)

    spaces = [H1.reduce_degree(axes=[0], basis='M'),
              H1.reduce_degree(axes=[1], basis='M')]

    Hcurl  = VectorFemSpace(*spaces)

    # create an instance of the H1 projector class
    P0 = Projector_H1(H1)

    # Build linear operators on stencil arrays
    grad = Gradient_2D(H1, Hcurl)

    # create an instance of the projector class
    P1 = Projector_Hcurl(Hcurl, Nq)
    #-------------------------------------
    # Projections and discrete derivatives
    #-------------------------------------

    u0        = P0(fun1)
    u1        = P1((D1fun1, D2fun1))
    Dfun_h    = grad(u0)
    Dfun_proj = u1

    error = (Dfun_proj.coeffs-Dfun_h.coeffs).toarray()
    assert abs(error).max()<1e-9

@pytest.mark.parametrize('Nel', [8, 12])
@pytest.mark.parametrize('Nq', [5])
@pytest.mark.parametrize('p', [2,3])
@pytest.mark.parametrize('bc', [True, False])
def test_2d_commuting_pro_2(Nel, Nq, p, bc):

    fun1    = lambda xi1, xi2 : np.sin(xi1)*np.sin(xi2)
    D2fun1  = lambda xi1, xi2 : np.sin(xi1)*np.cos(xi2)
    D1fun1  = lambda xi1, xi2 : -np.cos(xi1)*np.sin(xi2)

    Nel = [Nel]*2
    Nq  = [Nq]*2
    p   = [p]*2
    bc  = [bc]*2

    # Side lengths of logical cube [0, L]^2
    L = [2*np.pi, 2*np.pi]

    # element boundaries
    el_b = [np.linspace(0., L_i, Nel_i + 1) for L_i, Nel_i in zip(L, Nel)]

    # knot sequences
    knots = [make_knots(el_b_i, p_i, bc_i) for el_b_i, p_i, bc_i in zip(el_b, p, bc)]

    Vs     = [SplineSpace(pi, knots=Ti, periodic=periodic, basis='B') for pi, Ti, periodic in zip(p, knots, bc)]

    domain_decomposition = DomainDecomposition(Nel, bc, comm=MPI.COMM_WORLD)
    H1       = TensorFemSpace(domain_decomposition, *Vs)

    spaces = [H1.reduce_degree(axes=[1], basis='M'),
              H1.reduce_degree(axes=[0], basis='M')]

    Hdiv  = VectorFemSpace(*spaces)

    # create an instance of the H1 projector class
    P0 = Projector_H1(H1)

    # Linear operator: 2D vector curl
    curl = VectorCurl_2D(H1, Hdiv)

    # create an instance of the projector class
    P1 = Projector_Hdiv(Hdiv, Nq)
    #-------------------------------------
    # Projections and discrete derivatives
    #-------------------------------------

    u0        = P0(fun1)
    u1        = P1((D2fun1, D1fun1))
    Dfun_h    = curl(u0)
    Dfun_proj = u1

    error = (Dfun_proj.coeffs-Dfun_h.coeffs).toarray()
    assert abs(error).max()<1e-9

@pytest.mark.parametrize('Nel', [8, 12])
@pytest.mark.parametrize('Nq', [8])
@pytest.mark.parametrize('p', [2,3])
@pytest.mark.parametrize('bc', [True, False])
def test_2d_commuting_pro_3(Nel, Nq, p, bc):

    fun1    = lambda xi1, xi2 : np.sin(xi1)*np.sin(xi2)
    D1fun1  = lambda xi1, xi2 : np.cos(xi1)*np.sin(xi2)

    fun2    = lambda xi1, xi2 :   np.sin(2*xi1)*np.sin(2*xi2)
    D2fun2  = lambda xi1, xi2 : 2*np.sin(2*xi1)*np.cos(2*xi2)

    difun = lambda xi1, xi2 : D1fun1(xi1, xi2)+ D2fun2(xi1, xi2)

    Nel = [Nel]*2
    Nq  = [Nq]*2
    p   = [p]*2
    bc  = [bc]*2

    # Side lengths of logical cube [0, L]^2
    L = [2*np.pi, 2*np.pi]

    # element boundaries
    el_b = [np.linspace(0., L_i, Nel_i + 1) for L_i, Nel_i in zip(L, Nel)]

    # knot sequences
    knots = [make_knots(el_b_i, p_i, bc_i) for el_b_i, p_i, bc_i in zip(el_b, p, bc)]

    Vs     = [SplineSpace(pi, knots=Ti, periodic=periodic, basis='B') for pi, Ti, periodic in zip(p, knots, bc)]

    domain_decomposition = DomainDecomposition(Nel, bc, comm=MPI.COMM_WORLD)
    H1       = TensorFemSpace(domain_decomposition, *Vs)

    spaces = [H1.reduce_degree(axes=[1], basis='M'),
              H1.reduce_degree(axes=[0], basis='M')]

    Hdiv  = VectorFemSpace(*spaces)

    L2  = H1.reduce_degree(axes=[0,1], basis='M')

    # create an instance of the H1 projector class

    # Build linear operators on stencil arrays
    div  = Divergence_2D(Hdiv, L2)

    # create an instance of the projector class
    P2 = Projector_Hdiv(Hdiv, Nq)
    P3 = Projector_L2(L2, Nq)

    #-------------------------------------
    # Projections and discrete derivatives
    #-------------------------------------

    u2        = P2((fun1, fun2))
    u3        = P3(difun)
    Dfun_h    = div(u2)
    Dfun_proj = u3

    error = (Dfun_proj.coeffs-Dfun_h.coeffs).toarray()
    assert abs(error).max()<1e-9

@pytest.mark.parametrize('Nel', [8, 12])
@pytest.mark.parametrize('Nq', [8])
@pytest.mark.parametrize('p', [2,3])
@pytest.mark.parametrize('bc', [True, False])
def test_2d_commuting_pro_4(Nel, Nq, p, bc):

    fun1    = lambda xi1, xi2 : np.sin(xi1)*np.sin(xi2)
    D2fun1  = lambda xi1, xi2 : np.sin(xi1)*np.cos(xi2)

    fun2    = lambda xi1, xi2 :   np.sin(2*xi1)*np.sin(2*xi2)
    D1fun2  = lambda xi1, xi2 : 2*np.cos(2*xi1)*np.sin(2*xi2)

    difun = lambda xi1, xi2 : D1fun2(xi1, xi2) - D2fun1(xi1, xi2)

    Nel = [Nel]*2
    Nq  = [Nq]*2
    p   = [p]*2
    bc  = [bc]*2

    # Side lengths of logical cube [0, L]^2
    L = [2*np.pi, 2*np.pi]

    # element boundaries
    el_b = [np.linspace(0., L_i, Nel_i + 1) for L_i, Nel_i in zip(L, Nel)]

    # knot sequences
    knots = [make_knots(el_b_i, p_i, bc_i) for el_b_i, p_i, bc_i in zip(el_b, p, bc)]

    Vs     = [SplineSpace(pi, knots=Ti, periodic=periodic, basis='B') for pi, Ti, periodic in zip(p, knots, bc)]

    domain_decomposition = DomainDecomposition(Nel, bc, comm=MPI.COMM_WORLD)
    H1       = TensorFemSpace(domain_decomposition, *Vs)

    spaces = [H1.reduce_degree(axes=[0], basis='M'),
              H1.reduce_degree(axes=[1], basis='M')]

    Hcurl  = VectorFemSpace(*spaces)

    L2  = H1.reduce_degree(axes=[0,1], basis='M')

    # create an instance of the H1 projector class

    # Build linear operators on stencil arrays
    curl  = ScalarCurl_2D(Hcurl, L2)

    # create an instance of the projector class
    P1 = Projector_Hcurl(Hcurl, Nq)
    P2 = Projector_L2(L2, Nq)

    #-------------------------------------
    # Projections and discrete derivatives
    #-------------------------------------

    u2        = P1((fun1, fun2))
    u3        = P2(difun)
    Dfun_h    = curl(u2)
    Dfun_proj = u3

    error = (Dfun_proj.coeffs-Dfun_h.coeffs).toarray()
    assert abs(error).max()<1e-9

@pytest.mark.parametrize('Nel', [16, 20])
@pytest.mark.parametrize('Nq', [5])
@pytest.mark.parametrize('p', [2,3])
@pytest.mark.parametrize('bc', [True, False])
def test_1d_commuting_pro_1(Nel, Nq, p, bc):

    fun1    = lambda xi1 : np.sin(xi1)
    Dfun1   = lambda xi1 : np.cos(xi1)

    Nel = [Nel]
    Nq  = [Nq]
    p   = [p]
    bc  = [bc]

    # Side lengths of logical cube [0, L]
    L = [2*np.pi]

    # element boundaries
    el_b = [np.linspace(0., L_i, Nel_i + 1) for L_i, Nel_i in zip(L, Nel)]

    # knot sequences
    knots = [make_knots(el_b_i, p_i, bc_i) for el_b_i, p_i, bc_i in zip(el_b, p, bc)]

    Vs     = [SplineSpace(pi, knots=Ti, periodic=periodic, basis='B') for pi, Ti, periodic in zip(p, knots, bc)]

    domain_decomposition = DomainDecomposition(Nel, bc, comm=MPI.COMM_WORLD)
    H1       = TensorFemSpace(domain_decomposition, *Vs)
    L2       = H1.reduce_degree(axes=[0], basis='M')

    # create an instance of the H1 projector class
    P0 = Projector_H1(H1)

    # Build linear operators on stencil arrays
    grad = Derivative_1D(H1, L2)

    # create an instance of the projector class
    P1 = Projector_L2(L2, Nq)
    #-------------------------------------
    # Projections and discrete derivatives
    #-------------------------------------

    u0        = P0(fun1)
    u1        = P1(Dfun1)
    Dfun_h    = grad(u0)
    Dfun_proj = u1

    error = (Dfun_proj.coeffs-Dfun_h.coeffs).toarray()
    assert abs(error).max()<1e-9
#==============================================================================
if __name__ == '__main__':

    Nel = 8
    Nq  = 8
    p   = 3
    bc  = True

    test_3d_commuting_pro_1(Nel, Nq, p, bc)
    test_3d_commuting_pro_2(Nel, Nq, p, bc)
    test_3d_commuting_pro_3(Nel, Nq, p, bc)

