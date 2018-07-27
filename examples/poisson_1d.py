# coding: utf-8
import numpy as np

from spl.utilities.quadratures      import gauss_legendre
from spl.linalg.stencil             import StencilVector, StencilMatrix
from spl.linalg.iterative_solvers   import cg

#==============================================================================
class Poisson1D:
    """
    Exact solution to the 1D Poisson equation, to be employed for the method
    of manufactured solutions.

    :code
    $\frac{d^2}{dx^2}\phi(x) = -\rho(x)$

    """
    def __init__( self ):
        from sympy import symbols, sin, pi, lambdify
        x = symbols('x')
        phi_e = sin( 2*pi*x )
        rho_e = -phi_e.diff(x,2)
        self._phi = lambdify( x, phi_e )
        self._rho = lambdify( x, rho_e )

    def phi( self, x ):
        return self._phi( x )

    def rho( self, x ):
        return self._rho( x )

    @property
    def domain( self ):
        return (0, 1)

    @property
    def periodic( self ):
        return False

#==============================================================================
def kernel( p1, k1, bs1, w1, mat_m, mat_s ):
    """
    Kernel for computing the mass/stiffness element matrices.

    Parameters
    ----------
    p1 : int
        Spline degree.

    k1 : int
        Number of quadrature points in each element.

    bs1 : 3D array_like (p1+1, nderiv, k1)
        Values (and derivatives) of non-zero basis functions at each
        quadrature point.

    w1 : 1D array_like (k1,)
        Quadrature weights at each quadrature point.

    mat_m : 2D array_like (p1+1, 2*p1+1)
        Element mass matrix (in/out argument).

    mat_s : 2D array_like (p1+1, 2*p1+1)
        Element stiffness matrix (in/out argument).

    """
    # Reset element matrices
    mat_m[:,:] = 0.
    mat_s[:,:] = 0.

    # Cycle over non-zero test functions in element
    for il_1 in range(p1+1):

        # Cycle over non-zero trial functions in element
        for jl_1 in range(p1+1):

            # Reset integrals over element
            v_m = 0.0
            v_s = 0.0

            # Cycle over quadrature points
            for g1 in range(k1):

                # Get test function's value and derivative
                bi_0 = bs1[il_1, 0, g1]
                bi_x = bs1[il_1, 1, g1]

                # Get trial function's value and derivative
                bj_0 = bs1[jl_1, 0, g1]
                bj_x = bs1[jl_1, 1, g1]

                # Get quadrature weight
                wvol = w1[g1]

                # Add contribution to integrals
                v_m += bi_0 * bj_0 * wvol
                v_s += bi_x * bj_x * wvol

            # Update element matrices
            mat_m[il_1, p1+jl_1-il_1] = v_m
            mat_s[il_1, p1+jl_1-il_1] = v_s

#==============================================================================
def assemble_matrices( V, kernel ):
    """
    Assemble mass and stiffness matrices using 1D stencil format.

    Parameters
    ----------
    V : SplineSpace
        Finite element space where the Galerkin method is applied.

    kernel : callable
        Function that performs the assembly process on small element matrices.

    Returns
    -------
    mass : StencilMatrix
        Mass matrix in 1D stencil format.

    stiffness : StencilMatrix
        Stiffness matrix in 1D stencil format.

    """
    # Sizes
    [s1] = V.vector_space.starts
    [e1] = V.vector_space.ends
    [p1] = V.vector_space.pads

    # Quadrature data
    k1        = V.quad_order
    spans_1   = V.spans
    basis_1   = V.quad_basis
    weights_1 = V.quad_weights

    # Create global matrices
    mass      = StencilMatrix( V.vector_space, V.vector_space )
    stiffness = StencilMatrix( V.vector_space, V.vector_space )

    # Create element matrices
    mat_m = np.zeros( (p1+1,2*p1+1) ) # mass
    mat_s = np.zeros( (p1+1,2*p1+1) ) # stiffness

    # Build global matrices: cycle over elements
    for ie1 in range(s1, e1+1-p1):

        # Get spline index, B-splines' values and quadrature weights
        is1 =   spans_1[ie1]
        bs1 =   basis_1[ie1,:,:,:]
        w1  = weights_1[ie1,:]

        # Compute element matrices
        kernel( p1, k1, bs1, w1, mat_m, mat_s )

        # Update global matrices
        mass     [is1-p1:is1+1,:] += mat_m[:,:]
        stiffness[is1-p1:is1+1,:] += mat_s[:,:]

    return mass, stiffness

#==============================================================================
def assemble_rhs( V, f ):
    """
    Assemble right-hand-side vector.

    Parameters
    ----------
    V : SplineSpace
        Finite element space where the Galerkin method is applied.

    f : callable
        Right-hand side function (charge density).

    Returns
    -------
    rhs : StencilVector
        Vector b of coefficients, in linear system Ax=b.

    """
    # Sizes
    [s1] = V.vector_space.starts
    [e1] = V.vector_space.ends
    [p1] = V.vector_space.pads

    # Quadrature data
    spans_1   = V.spans
    basis_1   = V.quad_basis
    points_1  = V.quad_points
    weights_1 = V.quad_weights

    # Data structure
    rhs = StencilVector( V.vector_space )

    # Build RHS
    for ie1 in range(s1, e1+1-p1):

        i_span_1 =   spans_1[ie1]
        x1       =  points_1[ie1, :]
        wvol     = weights_1[ie1, :]
        f_quad   = f( x1 )

        for il_1 in range(0, p1+1):

            bi_0 = basis_1[ie1, il_1, 0, :]
            i1   = i_span_1 - p1 + il_1
            v    = bi_0 * f_quad * wvol

            rhs[i1] += v.sum()

    return rhs

#===================================================================================
def integral( V, f ):
    """
    Compute integral over domain of $f(x)$ using Gaussian quadrature.

    Parameters
    ----------
    V : SplineSpace
        Finite element space that defines the quadrature rule.
        (normally the quadrature is exact for any element of this space).

    f : callable
        Scalar function of location $x$.

    Returns
    -------
    c : float
        Integral of $f$ over domain.

    """
    # Sizes
    [s1] = V.vector_space.starts
    [e1] = V.vector_space.ends
    [p1] = V.vector_space.pads

    # Quadrature data
    k1        = V.quad_order
    points_1  = V.quad_points
    weights_1 = V.quad_weights

    c = 0.0
    for ie1 in range(s1, e1+1-p1):

        x1 =  points_1[ie1,:]
        w1 = weights_1[ie1,:]

        for g1 in range(k1):
            c+= f( x1[g1] ) * w1[g1]

    return c

#===================================================================================
def error_norm( V, phi, phi_ex, order=2 ):
    """
    Compute Lp norm of error using Gaussian quadrature.

    Parameters
    ----------
    V : SplineSpace
        Finite element space to which the numerical solution belongs.

    phi : FemField
        Numerical solution; 1D Spline that can be evaluated at location $x$.

    phi_ex : callable
        Exact solution; scalar function of location $x$.

    order : int
        Order of the norm (default: 2).

    Returns
    -------
    norm : float
        Lp norm of error.

    """
    f = lambda x: abs(phi(x)-phi_ex(x))**order

    norm = integral( V, f )**(1/order)

    return norm

####################################################################################
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from time import time

    from spl.fem.splines import SplineSpace
    from spl.fem.basic   import FemField

    timing = {}

    # Input data: degree, number of elements
    p  = 3
    ne = 2**4

    # Method of manufactured solution
    model = Poisson1D()

    # Create uniform grid
    grid = np.linspace( *model.domain, num=ne+1 )

    # Create finite element space and precompute quadrature data
    V = SplineSpace( p, grid=grid, periodic=model.periodic )
    V.init_fem()

    # Build mass and stiffness matrices
    t0 = time()
    mass, stiffness = assemble_matrices( V, kernel )
    t1 = time()
    timing['assembly'] = t1-t0

    # Build right-hand side vector
    rhs = assemble_rhs( V, model.rho )

    # Apply homogeneous dirichlet boundary conditions
    stiffness[ 0,:] = 0.
    stiffness[-1,:] = 0.
    rhs[0] = 0.
    rhs[V.nbasis-1] = 0.

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
    print( '> Grid          :: {ne}'.format(ne=ne) )
    print( '> Degree        :: {p}'.format(p=p) )
    print( '> CG info       :: ',info )
    print( '> L2 error      :: {:.2e}'.format( e2 ) )
    print( '' )
    print( '> Assembly time :: {:.2e}'.format( timing['assembly'] ) )
    print( '> Solution time :: {:.2e}'.format( timing['solution'] ) )
    print( '> Evaluat. time :: {:.2e}'.format( timing['diagnostics'] ) )

    # Plot solution on refined grid
    y      = np.linspace( grid[0], grid[-1], 101 )
    phi_y  = np.array( [phi(yj) for yj in y] )
    fig,ax = plt.subplots( 1, 1 )
    ax.plot( y, phi_y )
    ax.set_xlabel( 'x' )
    ax.set_ylabel( 'y' )
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
