# coding: utf-8
import numpy as np

from spl.utilities.quadratures import gauss_legendre
from spl.linalg.stencil        import StencilVector, StencilMatrix
from spl.linalg.solvers        import cg

#==============================================================================
class Poisson2D:
    """
    Exact solution to the 1D Poisson equation, to be employed for the method
    of manufactured solutions.

    :code
    $(\partial^2_{xx} + \partial^2_{yy}) \phi(x,y) = -\rho(x,y)$

    """
    def __init__( self ):
        from sympy import symbols, sin, cos, pi, lambdify
        x,y = symbols('x y')
        phi_e = sin( 2*pi*x ) * sin( 2*pi*y )
        rho_e = -phi_e.diff(x,2)-phi_e.diff(y,2)
        self._phi = lambdify( [x,y], phi_e )
        self._rho = lambdify( [x,y], rho_e )

    def phi( self, x, y ):
        return self._phi( x, y )

    def rho( self, x, y ):
        return self._rho( x, y )

    @property
    def domain( self ):
        return ((0,1), (0,1))

    @property
    def periodic( self ):
        return (False, False)

#==============================================================================
def kernel( p1, p2, k1, k2, bs1, bs2, w1, w2, mat_m, mat_s ):
    """
    Kernel for computing the mass/stiffness element matrices.

    Parameters
    ----------
    p1 : int
        Spline degree along x1 direction.

    p2 : int
        Spline degree along x2 direction.

    k1 : int
        Number of quadrature points along x1 (same in each element).

    k2 : int
        Number of quadrature points along x2 (same in each element).

    bs1 : 3D array_like (p1+1, nderiv, k1)
        Values (and derivatives) of non-zero basis functions along x1
        at each quadrature point.

    bs2 : 3D array_like (p1+1, nderiv, k1)
        Values (and derivatives) of non-zero basis functions along x2
        at each quadrature point.

    w1 : 1D array_like (k1,)
        Quadrature weights at each quadrature point.

    w2 : 1D array_like (k1,)
        Quadrature weights at each quadrature point.

    mat_m : 2D array_like (p1+1, p2+1, 2*p1+1, 2*p2+1)
        Element mass matrix (in/out argument).

    mat_s : 2D array_like (p1+1, p2+1, 2*p1+1, 2*p2+1)
        Element stiffness matrix (in/out argument).

    """
    # Reset element matrices
    mat_m[:,:,:,:] = 0.
    mat_s[:,:,:,:] = 0.

    # Cycle over non-zero test functions in element
    for il_1 in range(p1+1):
        for il_2 in range(p2+1):

            # Cycle over non-zero trial functions in element
            for jl_1 in range(p1+1):
                for jl_2 in range(p2+1):

                    # Reset integrals over element
                    v_m = 0.0
                    v_s = 0.0

                    # Cycle over quadrature points
                    for g1 in range(k1):
                        for g2 in range(k2):

                            # Get test function's value and derivatives
                            bi_0 = bs1[il_1, 0, g1] * bs2[il_2, 0, g2]
                            bi_x = bs1[il_1, 1, g1] * bs2[il_2, 0, g2]
                            bi_y = bs1[il_1, 0, g1] * bs2[il_2, 1, g2]

                            # Get trial function's value and derivatives
                            bj_0 = bs1[jl_1, 0, g1] * bs2[jl_2, 0, g2]
                            bj_x = bs1[jl_1, 1, g1] * bs2[jl_2, 0, g2]
                            bj_y = bs1[jl_1, 0, g1] * bs2[jl_2, 1, g2]

                            # Get quadrature weight
                            wvol = w1[g1] * w2[g2]

                            # Add contribution to integrals
                            v_m += bi_0 * bj_0 * wvol
                            v_s += (bi_x * bj_x + bi_y * bj_y) * wvol

                    # Update element matrices
                    mat_m[il_1, il_2, p1+jl_1-il_1, p2+jl_2-il_2 ] = v_m
                    mat_s[il_1, il_2, p1+jl_1-il_1, p2+jl_2-il_2 ] = v_s

#==============================================================================
def assemble_matrices( V, kernel ):
    """
    Assemble mass and stiffness matrices using 2D stencil format.

    Parameters
    ----------
    V : TensorFemSpace
        Finite element space where the Galerkin method is applied.

    kernel : callable
        Function that performs the assembly process on small element matrices.

    Returns
    -------
    mass : StencilMatrix
        Mass matrix in 2D stencil format.

    stiffness : StencilMatrix
        Stiffness matrix in 2D stencil format.

    """
    # Sizes
    [s1, s2] = V.vector_space.starts
    [e1, e2] = V.vector_space.ends
    [p1, p2] = V.vector_space.pads

    # Quadrature data
    [       k1,        k2] = [W.quad_order   for W in V.spaces]
    [  spans_1,   spans_2] = [W.spans        for W in V.spaces]
    [  basis_1,   basis_2] = [W.quad_basis   for W in V.spaces]
    [ points_1,  points_2] = [W.quad_points  for W in V.spaces]
    [weights_1, weights_2] = [W.quad_weights for W in V.spaces]

    # Create global matrices
    mass      = StencilMatrix( V.vector_space, V.vector_space )
    stiffness = StencilMatrix( V.vector_space, V.vector_space )

    # Create element matrices
    mat_m = np.zeros( (p1+1, p2+1, 2*p1+1, 2*p2+1) ) # mass
    mat_s = np.zeros( (p1+1, p2+1, 2*p1+1, 2*p2+1) ) # stiffness

    # Build global matrices: cycle over elements
    for ie1 in range(s1, e1+1-p1):
        for ie2 in range(s2, e2+1-p2):

            # Get spline index, B-splines' values and quadrature weights
            is1 =   spans_1[ie1]
            bs1 =   basis_1[ie1,:,:,:]
            w1  = weights_1[ie1,:]

            is2 =   spans_2[ie2]
            bs2 =   basis_2[ie2,:,:,:]
            w2  = weights_2[ie2,:]

            # Compute element matrices
            kernel( p1, p2, k1, k2, bs1, bs2, w1, w2, mat_m, mat_s )

            # Update global matrices
            mass     [is1-p1:is1+1, is2-p2:is2+1, :, :] += mat_m[:,:,:,:]
            stiffness[is1-p1:is1+1, is2-p2:is2+1, :, :] += mat_s[:,:,:,:]

    return mass , stiffness

#==============================================================================
def assemble_rhs( V, f ):
    """
    Assemble right-hand-side vector.

    Parameters
    ----------
    V : TensorFemSpace
        Finite element space where the Galerkin method is applied.

    f : callable
        Right-hand side function rho(x,y) (charge density).

    Returns
    -------
    rhs : StencilVector
        Vector b of coefficients, in linear system Ax=b.

    """
    # Sizes
    [s1, s2] = V.vector_space.starts
    [e1, e2] = V.vector_space.ends
    [p1, p2] = V.vector_space.pads

    # Quadrature data
    [       k1,        k2] = [W.quad_order   for W in V.spaces]
    [  spans_1,   spans_2] = [W.spans        for W in V.spaces]
    [  basis_1,   basis_2] = [W.quad_basis   for W in V.spaces]
    [ points_1,  points_2] = [W.quad_points  for W in V.spaces]
    [weights_1, weights_2] = [W.quad_weights for W in V.spaces]

    # Data structure
    rhs = StencilVector( V.vector_space )

    # Build RHS
    for ie1 in range(s1, e1+1-p1):
        for ie2 in range(s2, e2+1-p2):

            # Get spline index, B-splines' values and quadrature weights
            is1 =   spans_1[ie1]
            bs1 =   basis_1[ie1,:,:,:]
            w1  = weights_1[ie1,:]
            x1  =  points_1[ie1,:]

            is2 =   spans_2[ie2]
            bs2 =   basis_2[ie2,:,:,:]
            w2  = weights_2[ie2,:]
            x2  =  points_2[ie2,:]

            # Evaluate function at all quadrature points
            f_quad = f( *np.meshgrid( x1, x2, indexing='ij' ) )

            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0 = bs1[il_1, 0, g1] * bs2[il_2, 0, g2]
                            wvol = w1[g1] * w2[g2]
                            v   += bi_0 * f_quad[g1,g2] * wvol

                    i1 = is1 - p1 + il_1
                    i2 = is2 - p2 + il_2

                    rhs[i1, i2] += v

    return rhs

#===================================================================================
def integral( V, f ):
    """
    Compute integral over domain of $f(x1,x2)$ using Gaussian quadrature.

    Parameters
    ----------
    V : TensorFemSpace
        Finite element space that defines the quadrature rule.
        (normally the quadrature is exact for any element of this space).

    f : callable
        Scalar function of location $(x1,x2)$.

    Returns
    -------
    c : float
        Integral of $f$ over domain.

    """
    # Sizes
    [s1, s2] = V.vector_space.starts
    [e1, e2] = V.vector_space.ends
    [p1, p2] = V.vector_space.pads

    # Quadrature data
    [       k1,        k2] = [W.quad_order   for W in V.spaces]
    [ points_1,  points_2] = [W.quad_points  for W in V.spaces]
    [weights_1, weights_2] = [W.quad_weights for W in V.spaces]

    c = 0.0
    for ie1 in range(s1, e1+1-p1):
        for ie2 in range(s2, e2+1-p2):

            x1 =  points_1[ie1,:]
            w1 = weights_1[ie1,:]

            x2 =  points_2[ie2,:]
            w2 = weights_2[ie2,:]

            for g1 in range(k1):
                for g2 in range(k2):
                    c += f( x1[g1], x2[g2] ) * w1[g1] * w2[g2]

    return c

#===================================================================================
def error_norm( V, phi, phi_ex, order=2 ):
    """
    Compute Lp norm of error using Gaussian quadrature.

    Parameters
    ----------
    V : TensorFemSpace
        Finite element space to which the numerical solution belongs.

    phi : FemField
        Numerical solution; 2D Spline that can be evaluated at location $(x1,x2)$.

    phi_ex : callable
        Exact solution; scalar function of location $(x1,x2)$.

    order : int
        Order of the norm (default: 2).

    Returns
    -------
    norm : float
        Lp norm of error.

    """
    f = lambda x,y: abs(phi(x,y)-phi_ex(x,y))**order

    norm = integral( V, f )**(1/order)

    return norm

####################################################################################
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from time import time

    from spl.fem.splines import SplineSpace
    from spl.fem.tensor  import TensorFemSpace
    from spl.fem.basic   import FemField

    timing = {}

    # Input data: degree, number of elements
    p1  = 3  ; p2  = 3
    ne1 = 16 ; ne2 = 16

    # Method of manufactured solution
    model = Poisson2D()

    # Create uniform grid
    grid_1 = np.linspace( *model.domain[0], num=ne1+1 )
    grid_2 = np.linspace( *model.domain[1], num=ne2+1 )

    # Create 1D finite element spaces and precompute quadrature data
    V1 = SplineSpace( p1, grid=grid_1 ); V1.init_fem()
    V2 = SplineSpace( p2, grid=grid_2 ); V2.init_fem()

    # Create 2D tensor product finite element space
    V = TensorFemSpace( V1, V2 )

    # Build mass and stiffness matrices
    t0 = time()
    mass, stiffness = assemble_matrices( V, kernel )
    t1 = time()
    timing['assembly'] = t1-t0

    # Build right-hand side vector
    rhs = assemble_rhs( V, model.rho )

    # Apply homogeneous dirichlet boundary conditions
    # left  bc at x=0.
    stiffness[0,:,:,:] = 0.
    rhs      [0,:]     = 0.
    # right bc at x=1.
    stiffness[V1.nbasis-1,:,:,:] = 0.
    rhs      [V1.nbasis-1,:]     = 0.
    # lower bc at y=0.
    stiffness[:,0,:,:] = 0.
    rhs      [:,0]     = 0.
    # upper bc at y=1.
    stiffness[:,V2.nbasis-1,:,:] = 0.
    rhs      [:,V2.nbasis-1]     = 0.

    # Solve linear system
    t0 = time()
    x, info = cg( stiffness, rhs, tol=1e-9, maxiter=1000, verbose=False )
    t1 = time()
    timing['solution'] = t1-t0

    # Create potential field
    phi = FemField( V, 'phi' )
    phi.coeffs[:] = x[:]
    phi.coeffs.update_ghost_regions()

    # Compute L2 norm of error
    t0 = time()
    e2 = error_norm( V, phi, model.phi, order=2 )
    t1 = time()
    timing['diagnostics'] = t1-t0

    # Print some information to terminal
    print( '> Grid          :: [{ne1},{ne2}]'.format( ne1=ne1, ne2=ne2) )
    print( '> Degree        :: [{p1},{p2}]'  .format( p1=p1, p2=p2 ) )
    print( '> CG info       :: ',info )
    print( '> L2 error      :: {:.2e}'.format( e2 ) )
    print( '' )
    print( '> Assembly time :: {:.2e}'.format( timing['assembly'] ) )
    print( '> Solution time :: {:.2e}'.format( timing['solution'] ) )
    print( '> Evaluat. time :: {:.2e}'.format( timing['diagnostics'] ) )

    # Plot solution on refined grid
    xx = np.linspace( *model.domain[0], num=101 )
    yy = np.linspace( *model.domain[1], num=101 )
    zz = np.array( [[phi( xi,yi ) for yi in yy] for xi in xx] )
    fig, ax = plt.subplots( 1, 1 )
    im = ax.contourf( xx, yy, zz.transpose(), 40, cmap='jet' )
    fig.colorbar( im )
    ax.set_xlabel( r'$x$', rotation='horizontal' )
    ax.set_ylabel( r'$y$', rotation='horizontal' )
    ax.set_title ( r'$\phi(x,y)$' )
    ax.grid()

    # Show figure and keep it open if necessary
    fig.tight_layout()
    fig.show()

    import __main__ as main
    if hasattr( main, '__file__' ):
        try:
           __IPYTHON__
        except NameError:
            plt.show()
