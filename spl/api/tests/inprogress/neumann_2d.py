# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin
from sympy import S
from sympy import Tuple
from sympy import Matrix

from sympde.core import dx, dy, dz
from sympde.core import Mapping
from sympde.core import Constant
from sympde.core import Field
from sympde.core import VectorField
from sympde.core import grad, dot, inner, cross, rot, curl, div
from sympde.core import FunctionSpace, VectorFunctionSpace
from sympde.core import TestFunction
from sympde.core import VectorTestFunction
from sympde.core import BilinearForm, LinearForm, Integral
from sympde.core import Norm
from sympde.core import Equation, DirichletBC
from sympde.core import Domain
from sympde.core import Boundary, trace_0, trace_1
from sympde.core import ComplementBoundary
from sympde.gallery import Poisson, Stokes

from spl.fem.context import fem_context
from spl.fem.basic   import FemField
from spl.fem.splines import SplineSpace
from spl.fem.tensor  import TensorFemSpace
from spl.fem.vector  import ProductFemSpace, VectorFemField
from spl.api.discretization import discretize
from spl.api.boundary_condition import DiscreteBoundary
from spl.api.boundary_condition import DiscreteComplementBoundary
from spl.api.boundary_condition import DiscreteDirichletBC

from spl.mapping.discrete import SplineMapping

from numpy import linspace, zeros, allclose

import os

base_dir = os.path.dirname(os.path.realpath(__file__))
mesh_dir = os.path.join(base_dir, '..', 'mesh')

domain = Domain('\Omega', dim=2)


#==============================================================================
def create_discrete_space(p=(2,2), ne=(2**3,2**3)):
    # ... discrete spaces
    # Input data: degree, number of elements
    p1,p2 = p
    ne1,ne2 = ne

    # Create uniform grid
    grid_1 = linspace( 0., 1., num=ne1+1 )
    grid_2 = linspace( 0., 1., num=ne2+1 )

    # Create 1D finite element spaces and precompute quadrature data
    V1 = SplineSpace( p1, grid=grid_1 ); V1.init_fem()
    V2 = SplineSpace( p2, grid=grid_2 ); V2.init_fem()

    # Create 2D tensor product finite element space
    V = TensorFemSpace( V1, V2 )
    # ...

    return V

#==============================================================================
def assert_identical_coo(A, B):

    if isinstance(A, (list, tuple)) and isinstance(B, (list, tuple)):
        assert len(A) == len(B)

        for a,b in zip(A, B): assert_identical_coo(a, b)

    elif not(isinstance(A, (list, tuple))) and not(isinstance(B, (list, tuple))):
        A = A.tocoo()
        B = B.tocoo()

        assert(A.shape == B.shape)
        assert(A.nnz == B.nnz)

        assert(allclose(A.row,  B.row))
        assert(allclose(A.col,  B.col))
        assert(allclose(A.data, B.data))

    else:
        raise TypeError('Wrong types for entries')

#==============================================================================
def test_api_poisson_2d_neumann_():

    # ... abstract model
    mapping = Mapping('M', rdim=2, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain)
    B2 = Boundary(r'\Gamma_2', domain)
    B3 = Boundary(r'\Gamma_3', domain)
    B4 = Boundary(r'\Gamma_4', domain)

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u)) + 0.2*u*v
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = x*y #sin(pi*x)*sin(pi*y)

#    expr = (2.*pi**2 + 0.2)*solution*v
    expr = (x*y)*v
    l0 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B1)
    l_B1 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B3)
    l_B3 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B4)
    l_B4 = LinearForm(v, expr, mapping=mapping)

    expr = l0(v) + l_B1(v) + l_B2(v) + l_B3(v) + l_B4(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    equation = Equation(a(v,u), l(v))
    # ...

    # ... discrete spaces
#    Vh, mapping = fem_context(os.path.join(mesh_dir, 'collela_2d.h5'))
    Vh, mapping = fem_context('domain_1.h5')
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)

    equation_h = discretize(equation, [Vh, Vh], mapping, boundary=[B1, B2, B3, B4])
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, mapping)
    h1norm_h = discretize(h1norm, Vh, mapping)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)
    print('> l2 norm = ', l2_error)
    print('> h1 norm = ', h1_error)

#    expected_l2_error =  0.039447502116924604
#    expected_h1_error =  0.5887019756700849
#
#    assert( abs(l2_error - expected_l2_error) < 1.e-7)
#    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...


    ##########
    import numpy as np
    import matplotlib.pyplot as plt
    from spl.utilities.utils            import refine_array_1d

    N = 5
    ##########

    # Compute numerical solution (and error) on refined logical grid
    V1, V2 = Vh.spaces
    eta1 = refine_array_1d( V1.breaks, N )
    eta2 = refine_array_1d( V2.breaks, N )
    num = np.array( [[      phi( e1,e2 ) for e2 in eta2] for e1 in eta1] )

    # Recompute physical coordinates of logical grid using spline mapping
    pcoords = np.array( [[mapping( [e1,e2] ) for e2 in eta2] for e1 in eta1] )
    xx = pcoords[:,:,0]
    yy = pcoords[:,:,1]

    # Plot numerical solution
    fig, ax = plt.subplots( 1, 1 )
    im = ax.contourf( xx, yy, num, 40, cmap='jet' )
    fig.colorbar( im )
    ax.set_xlabel( r'$x$', rotation='horizontal' )
    ax.set_ylabel( r'$y$', rotation='horizontal' )
    ax.set_title ( r'$\phi(x,y)$' )
    ax.plot( xx[:,::N]  , yy[:,::N]  , 'k' )
    ax.plot( xx[::N,:].T, yy[::N,:].T, 'k' )
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.show()
    plt.show()





#==============================================================================
def test_api_poisson_2d_dirneu_square_mod_3():

    # ... abstract model
    mapping = Mapping('M', rdim=2, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain) # Neumann bc will be applied on B1
    B2 = Boundary(r'\Gamma_2', domain) # Neumann bc will be applied on B2
    B3 = Boundary(r'\Gamma_3', domain) # Neumann bc will be applied on B3
    B4 = Boundary(r'\Gamma_4', domain) # Dirichlet bc will be applied on B4

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = sin(pi*x)*sin(pi*y)

    expr = 2.*pi**2*solution*v
    l0 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B1)
    l_B1 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B3)
    l_B3 = LinearForm(v, expr, mapping=mapping)

    expr = l0(v) + l_B1(v) + l_B2(v) + l_B3(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(B4)]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
#    Vh, mapping = fem_context(os.path.join(mesh_dir, 'square_mod_3.h5'))
    Vh, mapping = fem_context('square_mod_3.h5')
    # ...

#    # ...
#    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
#    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
#    B3 = DiscreteBoundary(B3, axis=1, ext=-1)
#    B4 = DiscreteBoundary(B4, axis=1, ext= 1)
#
#    lh = discretize(l, Vh, mapping, boundary=[B1,B2,B3])
#    lh.assemble()
#    import sys; sys.exit(0)
#    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)

    bc = [DiscreteDirichletBC(B4)]
    equation_h = discretize(equation, [Vh, Vh], mapping, boundary=[B1,B2,B3], bc=bc)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, mapping)
    h1norm_h = discretize(h1norm, Vh, mapping)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()

    lhs = equation_h.linear_system.lhs
    rhs = equation_h.linear_system.rhs
    lhs = lhs.tocoo().tocsr()
    rhs = rhs.toarray()

    from scipy.sparse.linalg import gmres as sc_gmres
    from scipy.sparse.linalg import cg as sc_cg

    x_gmres,status = sc_gmres(lhs, rhs, tol=1.e-10, maxiter=1000)
    x_cg,status    = sc_cg(lhs, rhs, tol=1.e-10, maxiter=1000)
    print(allclose(x_cg, x_gmres))
    print(allclose(x.toarray(), x_gmres))
#    import sys; sys.exit(0)
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)
    print('> l2 norm = ', l2_error)
    print('> h1 norm = ', h1_error)

#    expected_l2_error =  0.039447502116924604
#    expected_h1_error =  0.5887019756700849
#
#    assert( abs(l2_error - expected_l2_error) < 1.e-7)
#    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...
    import sys; sys.exit(0)


    ##########
    import numpy as np
    import matplotlib.pyplot as plt
    from spl.utilities.utils            import refine_array_1d

    N = 5
    ##########

    # Compute numerical solution (and error) on refined logical grid
    V1, V2 = Vh.spaces
    eta1 = refine_array_1d( V1.breaks, N )
    eta2 = refine_array_1d( V2.breaks, N )
    num = np.array( [[      phi( e1,e2 ) for e2 in eta2] for e1 in eta1] )

    # Recompute physical coordinates of logical grid using spline mapping
    pcoords = np.array( [[mapping( [e1,e2] ) for e2 in eta2] for e1 in eta1] )
    xx = pcoords[:,:,0]
    yy = pcoords[:,:,1]

    # Plot numerical solution
    fig, ax = plt.subplots( 1, 1 )
    im = ax.contourf( xx, yy, num, 40, cmap='jet' )
    fig.colorbar( im )
    ax.set_xlabel( r'$x$', rotation='horizontal' )
    ax.set_ylabel( r'$y$', rotation='horizontal' )
    ax.set_title ( r'$\phi(x,y)$' )
    ax.plot( xx[:,::N]  , yy[:,::N]  , 'k' )
    ax.plot( xx[::N,:].T, yy[::N,:].T, 'k' )
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.show()
    plt.show()


#==============================================================================
def test_api_poisson_2d_neu_identity_1():

    # ... abstract model
    mapping = Mapping('M', rdim=2, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain) # Neumann bc will be applied on B1
    B2 = Boundary(r'\Gamma_2', domain) # Neumann bc will be applied on B2
    B3 = Boundary(r'\Gamma_3', domain) # Neumann bc will be applied on B3
    B4 = Boundary(r'\Gamma_4', domain) # Neumann bc will be applied on B4

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u)) + v*u
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = cos(0.25*pi*x)*cos(0.25*pi*y)

    expr = ((1./8.)*pi**2 + 1.)*solution*v
    l0 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B1)
    l_B1 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B3)
    l_B3 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B4)
    l_B4 = LinearForm(v, expr)

    expr = l0(v) + l_B1(v) + l_B2(v) + l_B3(v) + l_B4(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    equation = Equation(a(v,u), l(v))
    # ...

    # ... discrete spaces
#    Vh, mapping = fem_context(os.path.join(mesh_dir, 'identity_2d.h5'))
    Vh, mapping = fem_context('identity_2d.h5')
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)

    equation_h = discretize(equation, [Vh, Vh], mapping, boundary=[B1,B2,B3,B4])
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, mapping)
    h1norm_h = discretize(h1norm, Vh, mapping)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # TODO to remove
    lhs = equation_h.linear_system.lhs.tocoo()
    rhs = equation_h.linear_system.rhs.toarray()
    from scipy.io import mmwrite
    import numpy as np
    mmwrite('lhs_identity.mtx', lhs)
    np.savetxt('rhs_identity.txt', rhs)
    #

    return lhs, rhs

#    # ...
#    phi = FemField( Vh, 'phi' )
#    phi.coeffs[:,:] = x[:,:]
#    # ...
#
#    # ... compute norms
#    l2_error = l2norm_h.assemble(F=phi)
#    h1_error = h1norm_h.assemble(F=phi)
#    print('> l2 norm = ', l2_error)
#    print('> h1 norm = ', h1_error)

#    expected_l2_error =  0.0005265958470026676
#    expected_h1_error =  0.027894350363093987
#
#    assert( abs(l2_error - expected_l2_error) < 1.e-7)
#    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#    ##########
#    import numpy as np
#    import matplotlib.pyplot as plt
#    from spl.utilities.utils            import refine_array_1d
#
#    N = 5
#    ##########
#
#    # Compute numerical solution (and error) on refined logical grid
#    V1, V2 = Vh.spaces
#    eta1 = refine_array_1d( V1.breaks, N )
#    eta2 = refine_array_1d( V2.breaks, N )
#    num = np.array( [[      phi( e1,e2 ) for e2 in eta2] for e1 in eta1] )
#
#    # Recompute physical coordinates of logical grid using spline mapping
#    pcoords = np.array( [[mapping( [e1,e2] ) for e2 in eta2] for e1 in eta1] )
#    xx = pcoords[:,:,0]
#    yy = pcoords[:,:,1]
#
#    # Plot numerical solution
#    fig, ax = plt.subplots( 1, 1 )
#    im = ax.contourf( xx, yy, num, 40, cmap='jet' )
#    fig.colorbar( im )
#    ax.set_xlabel( r'$x$', rotation='horizontal' )
#    ax.set_ylabel( r'$y$', rotation='horizontal' )
#    ax.set_title ( r'$\phi(x,y)$' )
#    ax.plot( xx[:,::N]  , yy[:,::N]  , 'k' )
#    ax.plot( xx[::N,:].T, yy[::N,:].T, 'k' )
#    ax.set_aspect('equal')
#    fig.tight_layout()
#    fig.show()
#    plt.show()

#==============================================================================
def test_api_poisson_2d_neu_1():

    # ... abstract model
    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain) # Neumann bc will be applied on B1
    B2 = Boundary(r'\Gamma_2', domain) # Neumann bc will be applied on B2
    B3 = Boundary(r'\Gamma_3', domain) # Neumann bc will be applied on B3
    B4 = Boundary(r'\Gamma_4', domain) # Neumann bc will be applied on B4

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u)) + v*u
    a = BilinearForm((v,u), expr)

    solution = cos(0.25*pi*x)*cos(0.25*pi*y)

    expr = ((1./8.)*pi**2 + 1.)*solution*v
    l0 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B1)
    l_B1 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B3)
    l_B3 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B4)
    l_B4 = LinearForm(v, expr)

    expr = l0(v) + l_B1(v) + l_B2(v) + l_B3(v) + l_B4(v)
    l = LinearForm(v, expr)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u')
    h1norm = Norm(error, domain, kind='h1', name='u')

    equation = Equation(a(v,u), l(v))
    # ...

    # ... discrete spaces
#    Vh = create_discrete_space()
#    Vh = create_discrete_space(ne=(2,2))
    Vh = create_discrete_space(ne=(2,3))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)

    equation_h = discretize(equation, [Vh, Vh], boundary=[B1,B2,B3,B4])
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh)
    h1norm_h = discretize(h1norm, Vh)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # TODO to remove
    lhs = equation_h.linear_system.lhs.tocoo()
    rhs = equation_h.linear_system.rhs.toarray()
    from scipy.io import mmwrite
    import numpy as np
    mmwrite('lhs.mtx', lhs)
    np.savetxt('rhs.txt', rhs)
    #
    return lhs, rhs

#    # ...
#    phi = FemField( Vh, 'phi' )
#    phi.coeffs[:,:] = x[:,:]
#    # ...
#
#    # ... compute norms
#    l2_error = l2norm_h.assemble(F=phi)
#    h1_error = h1norm_h.assemble(F=phi)
#
#    expected_l2_error =  2.7510665198168697e-06
#    expected_h1_error =  0.00015490443857562876
#
#    assert( abs(l2_error - expected_l2_error) < 1.e-7)
#    assert( abs(h1_error - expected_h1_error) < 1.e-7)
#    # ...

###############################################
if __name__ == '__main__':
#    test_api_poisson_2d_neumann_()
#    test_api_poisson_2d_dirneu_square_mod_3()
    lhs_id, rhs_id = test_api_poisson_2d_neu_identity_1()
    from sympy import cache
    cache.clear_cache()

#    lhs, rhs = test_api_poisson_2d_neu_1()
#    from sympy import cache
#    cache.clear_cache()
#
#    assert_identical_coo(lhs, lhs_id)
#    assert(allclose(rhs, rhs_id))
