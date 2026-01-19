#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from mpi4py import MPI
import numpy as np
import pytest

from psydac.feec.global_geometric_projectors import GlobalGeometricProjectorH1
from psydac.feec.global_geometric_projectors import GlobalGeometricProjectorL2
from psydac.feec.global_geometric_projectors import GlobalGeometricProjectorHcurl
from psydac.feec.global_geometric_projectors import GlobalGeometricProjectorHdiv

from psydac.fem.tensor       import TensorFemSpace, SplineSpace
from psydac.fem.vector       import VectorFemSpace
from psydac.core.bsplines    import make_knots
from psydac.feec.derivatives import Derivative1D, Gradient2D, Gradient3D
from psydac.feec.derivatives import ScalarCurl2D, VectorCurl2D, Curl3D
from psydac.feec.derivatives import Divergence2D, Divergence3D
from psydac.ddm.cart         import DomainDecomposition
from psydac.linalg.solvers   import inverse
from psydac.linalg.basic     import IdentityOperator

#==============================================================================
# 3D tests
#==============================================================================
@pytest.mark.parametrize('m', [1, 2])
@pytest.mark.parametrize('bc', [True, False])
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('Nq', [5])
@pytest.mark.parametrize('Nel', [5, 6])
def test_3d_commuting_pro_1(Nel, Nq, p, bc, m):

    fun1    = lambda xi1, xi2, xi3 : np.sin(xi1)*np.sin(xi2)*np.sin(xi3)
    D1fun1  = lambda xi1, xi2, xi3 : np.cos(xi1)*np.sin(xi2)*np.sin(xi3)
    D2fun1  = lambda xi1, xi2, xi3 : np.sin(xi1)*np.cos(xi2)*np.sin(xi3)
    D3fun1  = lambda xi1, xi2, xi3 : np.sin(xi1)*np.sin(xi2)*np.cos(xi3)

    Nel = [Nel]*3
    Nq  = [Nq]*3
    p   = [p]*3
    bc  = [bc]*3
    m   = [m]*3

    # Side lengths of logical cube [0, L]^3
    L = [2*np.pi, 2*np.pi , 2*np.pi]

    # element boundaries
    el_b = [np.linspace(0., L_i, Nel_i + 1) for L_i, Nel_i in zip(L, Nel)] 

    # knot sequences
    knots = [make_knots(el_b_i, p_i, bc_i, m_i) for el_b_i, p_i, bc_i, m_i in zip(el_b, p, bc, m)]

    Vs     = [SplineSpace(pi, knots=Ti, periodic=periodic, basis='B') for pi, Ti, periodic in zip(p, knots, bc)]

    domain_decomposition = DomainDecomposition(Nel, bc, comm=MPI.COMM_WORLD)
    
    H1     = TensorFemSpace(domain_decomposition, *Vs)

    spaces = [H1.reduce_degree(axes=[0], basis='M'),
              H1.reduce_degree(axes=[1], basis='M'),
              H1.reduce_degree(axes=[2], basis='M')]

    Hcurl  = VectorFemSpace(*spaces)

    # create an instance of the H1 projector class
    P0 = GlobalGeometricProjectorH1(H1)

    # Build linear operators on stencil arrays
    grad = Gradient3D(H1, Hcurl)

    # create an instance of the projector class
    P1 = GlobalGeometricProjectorHcurl(Hcurl, Nq)
    #-------------------------------------
    # Projections and discrete derivatives
    #-------------------------------------

    u0        = P0(fun1)
    u1        = P1((D1fun1, D2fun1, D3fun1))
    Dfun_h    = grad(u0)
    Dfun_proj = u1

    error = abs((Dfun_proj.coeffs-Dfun_h.coeffs).toarray()).max()
    assert error < 1e-9

    #--------------------------
    # check BlockLinearOperator
    #--------------------------
    Id_0 = IdentityOperator(H1.coeff_space)
    Err_0 = P0.solver @ P0.imat_kronecker - Id_0
    e0 = Err_0 @ u0.coeffs  # random vector could be used as well
    norm2_e0 = np.sqrt(e0.inner(e0))
    assert norm2_e0 < 1e-12

    Id_1 = IdentityOperator(Hcurl.coeff_space)
    Err_1 = P1.solver @ P1.imat_kronecker - Id_1
    e1 = Err_1 @ u1.coeffs  # random vector could be used as well
    norm2_e1 = np.sqrt(e1.inner(e1))
    assert norm2_e1 < 1e-12

@pytest.mark.parametrize('m', [1, 2])
@pytest.mark.parametrize('bc', [True, False])
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('Nq', [7])
@pytest.mark.parametrize('Nel', [5, 6])
def test_3d_commuting_pro_2(Nel, Nq, p, bc, m):

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
    m   = [m]*3

    # Side lengths of logical cube [0, L]^3
    L = [2*np.pi, 2*np.pi , 2*np.pi]

    # element boundaries
    el_b = [np.linspace(0., L_i, Nel_i + 1) for L_i, Nel_i in zip(L, Nel)] 

    # knot sequences
    knots = [make_knots(el_b_i, p_i, bc_i, m_i) for el_b_i, p_i, bc_i, m_i in zip(el_b, p, bc, m)]

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
    curl = Curl3D(Hcurl, Hdiv)

    # create an instance of the projector class
    P1 = GlobalGeometricProjectorHcurl(Hcurl, Nq)
    P2 = GlobalGeometricProjectorHdiv(Hdiv, Nq)

    #-------------------------------------
    # Projections and discrete derivatives
    #-------------------------------------
    u1        = P1((fun1, fun2, fun3))
    u2        = P2((cf1, cf2, cf3))
    Dfun_h    = curl(u1)
    Dfun_proj = u2

    error = abs((Dfun_proj.coeffs-Dfun_h.coeffs).toarray()).max()
    assert error < 1e-9

    #--------------------------
    # check BlockLinearOperator
    #--------------------------

    Id_1 = IdentityOperator(Hcurl.coeff_space)
    Err_1 = P1.solver @ P1.imat_kronecker - Id_1
    e1 = Err_1 @ u1.coeffs  # random vector could be used as well
    norm2_e1 = np.sqrt(e1.inner(e1))
    assert norm2_e1 < 1e-12

    Id_2 = IdentityOperator(Hdiv.coeff_space)
    Err_2 = P2.solver @ P2.imat_kronecker - Id_2
    e2 = Err_2 @ u2.coeffs  # random vector could be used as well
    norm2_e2 = np.sqrt(e2.inner(e2))
    assert norm2_e2 < 1e-12

@pytest.mark.parametrize('m', [1, 2])
@pytest.mark.parametrize('bc', [True, False])
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('Nq', [7])
@pytest.mark.parametrize('Nel', [5, 6])
def test_3d_commuting_pro_3(Nel, Nq, p, bc, m):

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
    m   = [m]*3

    # Side lengths of logical cube [0, L]^3
    L = [2*np.pi, 2*np.pi , 2*np.pi]

    # element boundaries
    el_b = [np.linspace(0., L_i, Nel_i + 1) for L_i, Nel_i in zip(L, Nel)] 

    # knot sequences
    knots = [make_knots(el_b_i, p_i, bc_i, m_i) for el_b_i, p_i, bc_i, m_i in zip(el_b, p, bc, m)]

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
    div  = Divergence3D(Hdiv, L2)

    # create an instance of the projector class
    P2 = GlobalGeometricProjectorHdiv(Hdiv, Nq)
    P3 = GlobalGeometricProjectorL2(L2, Nq)

    #-------------------------------------
    # Projections and discrete derivatives
    #-------------------------------------

    u2        = P2((fun1, fun2, fun3))
    u3        = P3(difun)
    Dfun_h    = div(u2)
    Dfun_proj = u3

    error = abs((Dfun_proj.coeffs-Dfun_h.coeffs).toarray()).max()
    assert error < 1e-9

    #--------------------------
    # check BlockLinearOperator
    #--------------------------

    Id_2 = IdentityOperator(Hdiv.coeff_space)
    Err_2 = P2.solver @ P2.imat_kronecker - Id_2
    e2 = Err_2 @ u2.coeffs  # random vector could be used as well
    norm2_e2 = np.sqrt(e2.inner(e2))
    assert norm2_e2 < 1e-12

    Id_3 = IdentityOperator(L2.coeff_space)
    Err_3 = P3.solver @ P3.imat_kronecker - Id_3
    e3 = Err_3 @ u3.coeffs  # random vector could be used as well
    norm2_e3 = np.sqrt(e3.inner(e3))
    assert norm2_e3 < 1e-12

#==============================================================================
# 2D tests
#==============================================================================
@pytest.mark.mpi
@pytest.mark.parametrize('Nel', [8, 12])
@pytest.mark.parametrize('Nq', [5])
@pytest.mark.parametrize('p', [2,3])
@pytest.mark.parametrize('bc', [True, False])
@pytest.mark.parametrize('m', [1,2])
def test_2d_commuting_pro_1(Nel, Nq, p, bc, m):

    fun1    = lambda xi1, xi2 : np.sin(xi1)*np.sin(xi2)
    D1fun1  = lambda xi1, xi2 : np.cos(xi1)*np.sin(xi2)
    D2fun1  = lambda xi1, xi2 : np.sin(xi1)*np.cos(xi2)

    Nel = [Nel]*2
    Nq  = [Nq]*2
    p   = [p]*2
    bc  = [bc]*2
    m   = [m]*2

    # Side lengths of logical cube [0, L]^2
    L = [2*np.pi, 2*np.pi]

    # element boundaries
    el_b = [np.linspace(0., L_i, Nel_i + 1) for L_i, Nel_i in zip(L, Nel)]

    # knot sequences
    knots = [make_knots(el_b_i, p_i, bc_i, m_i) for el_b_i, p_i, bc_i, m_i in zip(el_b, p, bc, m)]

    Vs     = [SplineSpace(pi, knots=Ti, periodic=periodic, basis='B') for pi, Ti, periodic in zip(p, knots, bc)]

    domain_decomposition = DomainDecomposition(Nel, bc, comm=MPI.COMM_WORLD)
    H1       = TensorFemSpace(domain_decomposition, *Vs)

    spaces = [H1.reduce_degree(axes=[0], basis='M'),
              H1.reduce_degree(axes=[1], basis='M')]

    Hcurl  = VectorFemSpace(*spaces)

    # create an instance of the H1 projector class
    P0 = GlobalGeometricProjectorH1(H1)

    # Build linear operators on stencil arrays
    grad = Gradient2D(H1, Hcurl)

    # create an instance of the projector class
    P1 = GlobalGeometricProjectorHcurl(Hcurl, Nq)
    #-------------------------------------
    # Projections and discrete derivatives
    #-------------------------------------

    u0        = P0(fun1)
    u1        = P1((D1fun1, D2fun1))
    Dfun_h    = grad(u0)
    Dfun_proj = u1

    error = abs((Dfun_proj.coeffs-Dfun_h.coeffs).toarray()).max()
    assert error < 1e-9

    #--------------------------
    # check BlockLinearOperator
    #--------------------------

    Id_0 = IdentityOperator(H1.coeff_space)
    Err_0 = P0.solver @ P0.imat_kronecker - Id_0
    e0 = Err_0 @ u0.coeffs  # random vector could be used as well
    norm2_e0 = np.sqrt(e0.inner(e0))
    assert norm2_e0 < 1e-12

    Id_1 = IdentityOperator(Hcurl.coeff_space)
    Err_1 = P1.solver @ P1.imat_kronecker - Id_1
    e1 = Err_1 @ u1.coeffs  # random vector could be used as well
    norm2_e1 = np.sqrt(e1.inner(e1))
    assert norm2_e1 < 1e-12

@pytest.mark.mpi
@pytest.mark.parametrize('Nel', [8, 12])
@pytest.mark.parametrize('Nq', [5])
@pytest.mark.parametrize('p', [2,3])
@pytest.mark.parametrize('bc', [True, False])
@pytest.mark.parametrize('m', [1,2])
def test_2d_commuting_pro_2(Nel, Nq, p, bc, m):

    fun1    = lambda xi1, xi2 : np.sin(xi1)*np.sin(xi2)
    D2fun1  = lambda xi1, xi2 : np.sin(xi1)*np.cos(xi2)
    D1fun1  = lambda xi1, xi2 : -np.cos(xi1)*np.sin(xi2)

    Nel = [Nel]*2
    Nq  = [Nq]*2
    p   = [p]*2
    bc  = [bc]*2
    m   = [m]*2

    # Side lengths of logical cube [0, L]^2
    L = [2*np.pi, 2*np.pi]

    # element boundaries
    el_b = [np.linspace(0., L_i, Nel_i + 1) for L_i, Nel_i in zip(L, Nel)]

    # knot sequences
    knots = [make_knots(el_b_i, p_i, bc_i, m_i) for el_b_i, p_i, bc_i, m_i in zip(el_b, p, bc, m)]

    Vs     = [SplineSpace(pi, knots=Ti, periodic=periodic, basis='B') for pi, Ti, periodic in zip(p, knots, bc)]

    domain_decomposition = DomainDecomposition(Nel, bc, comm=MPI.COMM_WORLD)
    H1       = TensorFemSpace(domain_decomposition, *Vs)

    spaces = [H1.reduce_degree(axes=[1], basis='M'),
              H1.reduce_degree(axes=[0], basis='M')]

    Hdiv  = VectorFemSpace(*spaces)

    # create an instance of the H1 projector class
    P0 = GlobalGeometricProjectorH1(H1)

    # Linear operator: 2D vector curl
    curl = VectorCurl2D(H1, Hdiv)

    # create an instance of the projector class
    P1 = GlobalGeometricProjectorHdiv(Hdiv, Nq)
    #-------------------------------------
    # Projections and discrete derivatives
    #-------------------------------------

    u0        = P0(fun1)
    u1        = P1((D2fun1, D1fun1))
    Dfun_h    = curl(u0)
    Dfun_proj = u1

    error = abs((Dfun_proj.coeffs-Dfun_h.coeffs).toarray()).max()
    assert error < 1e-9

    #--------------------------
    # check BlockLinearOperator
    #--------------------------

    Id_0 = IdentityOperator(H1.coeff_space)
    Err_0 = P0.solver @ P0.imat_kronecker - Id_0
    e0 = Err_0 @ u0.coeffs  # random vector could be used as well
    norm2_e0 = np.sqrt(e0.inner(e0))
    assert norm2_e0 < 1e-12

    Id_1 = IdentityOperator(Hdiv.coeff_space)
    Err_1 = P1.solver @ P1.imat_kronecker - Id_1
    e1 = Err_1 @ u1.coeffs  # random vector could be used as well
    norm2_e1 = np.sqrt(e1.inner(e1))
    assert norm2_e0 < 1e-12

@pytest.mark.mpi
@pytest.mark.parametrize('Nel', [8, 12])
@pytest.mark.parametrize('Nq', [8])
@pytest.mark.parametrize('p', [2,3])
@pytest.mark.parametrize('bc', [True, False])
@pytest.mark.parametrize('m', [1,2])
def test_2d_commuting_pro_3(Nel, Nq, p, bc, m):

    fun1    = lambda xi1, xi2 : np.sin(xi1)*np.sin(xi2)
    D1fun1  = lambda xi1, xi2 : np.cos(xi1)*np.sin(xi2)

    fun2    = lambda xi1, xi2 :   np.sin(2*xi1)*np.sin(2*xi2)
    D2fun2  = lambda xi1, xi2 : 2*np.sin(2*xi1)*np.cos(2*xi2)

    difun = lambda xi1, xi2 : D1fun1(xi1, xi2)+ D2fun2(xi1, xi2)

    Nel = [Nel]*2
    Nq  = [Nq]*2
    p   = [p]*2
    bc  = [bc]*2
    m   = [m]*2

    # Side lengths of logical cube [0, L]^2
    L = [2*np.pi, 2*np.pi]

    # element boundaries
    el_b = [np.linspace(0., L_i, Nel_i + 1) for L_i, Nel_i in zip(L, Nel)]

    # knot sequences
    knots = [make_knots(el_b_i, p_i, bc_i, m_i) for el_b_i, p_i, bc_i, m_i in zip(el_b, p, bc, m)]

    Vs     = [SplineSpace(pi, knots=Ti, periodic=periodic, basis='B') for pi, Ti, periodic in zip(p, knots, bc)]

    domain_decomposition = DomainDecomposition(Nel, bc, comm=MPI.COMM_WORLD)
    H1       = TensorFemSpace(domain_decomposition, *Vs)

    spaces = [H1.reduce_degree(axes=[1], basis='M'),
              H1.reduce_degree(axes=[0], basis='M')]

    Hdiv  = VectorFemSpace(*spaces)

    L2  = H1.reduce_degree(axes=[0,1], basis='M')

    # create an instance of the H1 projector class

    # Build linear operators on stencil arrays
    div  = Divergence2D(Hdiv, L2)

    # create an instance of the projector class
    P2 = GlobalGeometricProjectorHdiv(Hdiv, Nq)
    P3 = GlobalGeometricProjectorL2(L2, Nq)

    #-------------------------------------
    # Projections and discrete derivatives
    #-------------------------------------

    u2        = P2((fun1, fun2))
    u3        = P3(difun)
    Dfun_h    = div(u2)
    Dfun_proj = u3

    error = abs((Dfun_proj.coeffs-Dfun_h.coeffs).toarray()).max()
    assert error < 1e-9

    #--------------------------
    # check BlockLinearOperator
    #--------------------------

    Id_2 = IdentityOperator(Hdiv.coeff_space)
    Err_2 = P2.solver @ P2.imat_kronecker - Id_2
    e2 = Err_2 @ u2.coeffs
    norm2_e2 = np.sqrt(e2.inner(e2))
    assert norm2_e2 < 1e-12

    Id_3 = IdentityOperator(L2.coeff_space)
    Err_3 = P3.solver @ P3.imat_kronecker - Id_3
    e3 = Err_3 @ u3.coeffs
    norm2_e3 = np.sqrt(e3.inner(e3))
    assert norm2_e3 < 1e-12

@pytest.mark.mpi
@pytest.mark.parametrize('Nel', [8, 12])
@pytest.mark.parametrize('Nq', [8])
@pytest.mark.parametrize('p', [2,3])
@pytest.mark.parametrize('bc', [True, False])
@pytest.mark.parametrize('m', [1,2])
def test_2d_commuting_pro_4(Nel, Nq, p, bc, m):

    fun1    = lambda xi1, xi2 : np.sin(xi1)*np.sin(xi2)
    D2fun1  = lambda xi1, xi2 : np.sin(xi1)*np.cos(xi2)

    fun2    = lambda xi1, xi2 :   np.sin(2*xi1)*np.sin(2*xi2)
    D1fun2  = lambda xi1, xi2 : 2*np.cos(2*xi1)*np.sin(2*xi2)

    difun = lambda xi1, xi2 : D1fun2(xi1, xi2) - D2fun1(xi1, xi2)

    Nel = [Nel]*2
    Nq  = [Nq]*2
    p   = [p]*2
    bc  = [bc]*2
    m   = [m]*2

    # Side lengths of logical cube [0, L]^2
    L = [2*np.pi, 2*np.pi]

    # element boundaries
    el_b = [np.linspace(0., L_i, Nel_i + 1) for L_i, Nel_i in zip(L, Nel)]

    # knot sequences
    knots = [make_knots(el_b_i, p_i, bc_i, m_i) for el_b_i, p_i, bc_i, m_i in zip(el_b, p, bc, m)]

    Vs     = [SplineSpace(pi, knots=Ti, periodic=periodic, basis='B') for pi, Ti, periodic in zip(p, knots, bc)]

    domain_decomposition = DomainDecomposition(Nel, bc, comm=MPI.COMM_WORLD)
    H1       = TensorFemSpace(domain_decomposition, *Vs)

    spaces = [H1.reduce_degree(axes=[0], basis='M'),
              H1.reduce_degree(axes=[1], basis='M')]

    Hcurl  = VectorFemSpace(*spaces)

    L2  = H1.reduce_degree(axes=[0,1], basis='M')

    # create an instance of the H1 projector class

    # Build linear operators on stencil arrays
    curl  = ScalarCurl2D(Hcurl, L2)

    # create an instance of the projector class
    P1 = GlobalGeometricProjectorHcurl(Hcurl, Nq)
    P2 = GlobalGeometricProjectorL2(L2, Nq)

    #-------------------------------------
    # Projections and discrete derivatives
    #-------------------------------------

    u1        = P1((fun1, fun2))
    u2        = P2(difun)
    Dfun_h    = curl(u1)
    Dfun_proj = u2

    error = abs((Dfun_proj.coeffs-Dfun_h.coeffs).toarray()).max()
    assert error < 1e-9

    #--------------------------
    # check BlockLinearOperator
    #--------------------------

    Id_1 = IdentityOperator(Hcurl.coeff_space)
    Err_1 = P1.solver @ P1.imat_kronecker - Id_1
    e1 = Err_1 @ u1.coeffs
    norm2_e1 = np.sqrt(e1.inner(e1))
    assert norm2_e1 < 1e-12

    Id_2 = IdentityOperator(L2.coeff_space)
    Err_2 = P2.solver @ P2.imat_kronecker - Id_2
    e2 = Err_2 @ u2.coeffs
    norm2_e2 = np.sqrt(e2.inner(e2))
    assert norm2_e2 < 1e-12

#==============================================================================
# 1D tests
#==============================================================================
@pytest.mark.parametrize('Nel', [16, 20])
@pytest.mark.parametrize('Nq', [5])
@pytest.mark.parametrize('p', [2,3])
@pytest.mark.parametrize('bc', [True, False])
@pytest.mark.parametrize('m', [1,2])
def test_1d_commuting_pro_1(Nel, Nq, p, bc, m):

    fun1    = lambda xi1 : np.sin(xi1)
    Dfun1   = lambda xi1 : np.cos(xi1)

    Nel = [Nel]
    Nq  = [Nq]
    p   = [p]
    bc  = [bc]
    m   = [m]

    # Side lengths of logical cube [0, L]
    L = [2*np.pi]

    # element boundaries
    el_b = [np.linspace(0., L_i, Nel_i + 1) for L_i, Nel_i in zip(L, Nel)]

    # knot sequences
    knots = [make_knots(el_b_i, p_i, bc_i, m_i) for el_b_i, p_i, bc_i, m_i in zip(el_b, p, bc, m)]

    Vs     = [SplineSpace(pi, knots=Ti, periodic=periodic, basis='B') for pi, Ti, periodic in zip(p, knots, bc)]

    domain_decomposition = DomainDecomposition(Nel, bc, comm=MPI.COMM_WORLD)
    H1       = TensorFemSpace(domain_decomposition, *Vs)
    L2       = H1.reduce_degree(axes=[0], basis='M')

    # create an instance of the H1 projector class
    P0 = GlobalGeometricProjectorH1(H1)

    # Build linear operators on stencil arrays
    grad = Derivative1D(H1, L2)

    # create an instance of the projector class
    P1 = GlobalGeometricProjectorL2(L2, Nq)
    #-------------------------------------
    # Projections and discrete derivatives
    #-------------------------------------

    u0        = P0(fun1)
    u1        = P1(Dfun1)
    Dfun_h    = grad(u0)
    Dfun_proj = u1

    error = abs((Dfun_proj.coeffs-Dfun_h.coeffs).toarray()).max()
    assert error < 1e-9

    #--------------------------
    # check BlockLinearOperator
    #--------------------------

    Id_0 = IdentityOperator(H1.coeff_space)
    Err_0 = P0.solver @ P0.imat_kronecker - Id_0
    e0 = Err_0 @ u0.coeffs
    norm2_e0 = np.sqrt(e0.inner(e0))
    assert norm2_e0 < 1e-12

    Id_1 = IdentityOperator(L2.coeff_space)
    Err_1 = P1.solver @ P1.imat_kronecker - Id_1
    e1 = Err_1 @ u1.coeffs
    norm2_e1 = np.sqrt(e1.inner(e1))
    assert norm2_e1 < 1e-12
    
#==============================================================================
if __name__ == '__main__':

    Nel = 8
    Nq  = 8
    p   = 2
    bc  = True
    m   = 2

    test_3d_commuting_pro_1(Nel, Nq, p, bc, m)
    test_3d_commuting_pro_2(Nel, Nq, p, bc, m)
    test_3d_commuting_pro_3(Nel, Nq, p, bc, m)
    test_2d_commuting_pro_1(Nel, Nq, p, bc, m)
    test_2d_commuting_pro_2(Nel, Nq, p, bc, m)
    test_2d_commuting_pro_3(Nel, Nq, p, bc, m)
    test_2d_commuting_pro_4(Nel, Nq, p, bc, m)
    test_1d_commuting_pro_1(Nel, Nq, p, bc, m)
