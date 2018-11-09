# coding: utf-8
from mpi4py import MPI
from time   import time

import numpy as np
import matplotlib.pyplot as plt

from spl.linalg.stencil             import StencilVector, StencilMatrix
from spl.linalg.iterative_solvers   import cg
from spl.fem.splines                import SplineSpace
from spl.fem.tensor                 import TensorFemSpace
from spl.fem.basic                  import FemField
from spl.mapping.analytical_gallery import Annulus
from spl.mapping.analytical         import IdentityMapping
from spl.mapping.discrete           import SplineMapping
from spl.utilities.utils            import refine_array_1d

#==============================================================================
class Poisson2D:
    """
    Exact solution to the 2D Poisson equation with Dirichlet boundary
    conditions, to be employed for the method of manufactured solutions.

    :code
    $(\partial^2_{xx} + \partial^2_{yy}) \phi(x,y) = -\rho(x,y)$

    """
    def __init__( self, domain, periodic, mapping, phi, rho ):

        self._domain   = domain
        self._periodic = periodic
        self._mapping  = mapping
        self._phi      = phi
        self._rho      = rho

    # ...
    @staticmethod
    def new_square( mx=1, my=1 ):
        """
        Solve Poisson's equation on the unit square.

        : code
        $\phi(x,y) = sin( mx*pi*x ) + sin( my*pi*y )$

        with $mx$ and $my$ user-defined integer numbers.

        """
        domain   = ((0,1), (0,1))
        periodic = (False, False)
        mapping  = IdentityMapping( ndim=2 )

        from sympy import symbols, sin, cos, pi, lambdify
        x,y   = symbols('x y')
        phi_e = sin( mx*pi*x ) * sin( my*pi*y )
        rho_e = -phi_e.diff(x,2)-phi_e.diff(y,2)

        phi = lambdify( [x,y], phi_e )
        rho = lambdify( [x,y], rho_e )

        return Poisson2D( domain, periodic, mapping, phi, rho )

    # ...
    @staticmethod
    def new_annulus( rmin=0.5, rmax=1.0 ):
        """
        Solve Poisson's equation on an annulus centered at (x,y)=(0,0),
        with logical coordinates (r,theta):

        - The radial coordinate r belongs to the interval [rmin,rmax];
        - The angular coordinate theta belongs to the interval [0,2*pi).

        : code
        $\phi(x,y) = (r-rmin)^2 (rmax-r)^2 \sin( mx*pi*x ) \sin( my*pi*y )$

        with $mx$ and $my$ user-defined integer numbers.

        """
        domain   = ((rmin,rmax),(0,2*np.pi))
        periodic = (False, True)
        mapping  = Annulus()

        from sympy import symbols, sin, cos, pi, sqrt, lambdify

        # Manufactured solutions in physical coordinates
        x,y   = symbols('x y', real=True )
        r     = sqrt( x**2 + y**2 )
#        phi_e = (r-rmin)**2 * (rmax-r)**2 * sin( 2*pi*x ) * sin( 2*pi*y )
        phi_e = (rmax-r**2) * cos( 2*pi*x ) * sin( 2*pi*y )
        rho_e = -phi_e.diff(x,2)-phi_e.diff(y,2)

        # Change to logical coordinates
        s,t   = Annulus.symbolic.eta
        X,Y   = (Xd.subs( mapping.params ) for Xd in Annulus.symbolic.map)
        phi_e = phi_e.subs( {x:X, y:Y} )
        rho_e = rho_e.subs( {x:X, y:Y} )

        # For further simplifications, assume that (s,t) are positive and real
        S,T   = symbols( 's t', real=True, positive=True )
        phi_e = phi_e.subs( {s:S, t:T} ).simplify()
        rho_e = rho_e.subs( {s:S, t:T} ).simplify()

        # Callable functions
        phi = lambdify( [S,T], phi_e )
        rho = lambdify( [S,T], rho_e )

        return Poisson2D( domain, periodic, mapping, phi, rho )

    # ...
    def new_annulus_separable( rmin=0.5, rmax=1.0 ):

        domain   = ((rmin,rmax),(0,2*np.pi))
        periodic = (False, True)
        mapping  = Annulus()

        from sympy import symbols, sin, cos, pi, sqrt, lambdify

        r,t   = symbols( 'r t', real=True, positive=True )
        R     = 4 * (r-rmin) * (rmax-r) / (rmax-rmin)**2
        T     = cos( t )
        phi_e = R * T
        rho_e = -(R.diff(r,r)+R.diff(r)/r)*T -(R/r**2)*T.diff(t,t)

        # Simplify expressions
        phi_e = phi_e.simplify()
        rho_e = rho_e.simplify()

        # Callable functions
        phi = lambdify( [r,t], phi_e )
        rho = lambdify( [r,t], rho_e )

        return Poisson2D( domain, periodic, mapping, phi, rho )

    # ...
    def new_circle_separable():

        domain   = ((0,1),(0,2*np.pi))
        periodic = (False, True)
        mapping  = Annulus()

        from sympy import symbols, sin, cos, pi, sqrt, lambdify

        r,t   = symbols( 'r t', real=True, positive=True )
        phi_e = r**2 * (1-r**2)
        rho_e = 16*r**2 - 4

        # Callable functions
        phi = lambdify( [r,t], phi_e )
        rho = lambdify( [r,t], rho_e )

        return Poisson2D( domain, periodic, mapping, phi, rho )

    # ...
    @property
    def domain( self ):
        return self._domain

    @property
    def periodic( self ):
        return self._periodic

    @property
    def mapping( self ):
        return self._mapping

    @property
    def phi( self ):
        return self._phi

    @property
    def rho( self ):
        return self._rho

#==============================================================================
def kernel( p1, p2, nq1, nq2, bs1, bs2, w1, w2, jac_mat, mat_m, mat_s ):
    """
    Kernel for computing the mass/stiffness element matrices.

    Parameters
    ----------
    p1 : int
        Spline degree along x1 direction.

    p2 : int
        Spline degree along x2 direction.

    nq1 : int
        Number of quadrature points along x1 (same in each element).

    nq2 : int
        Number of quadrature points along x2 (same in each element).

    bs1 : 3D array_like (p1+1, nderiv, nq1)
        Values (and derivatives) of non-zero basis functions along x1
        at each quadrature point.

    bs2 : 3D array_like (p2+1, nderiv, nq2)
        Values (and derivatives) of non-zero basis functions along x2
        at each quadrature point.

    w1 : 1D array_like (nq1,)
        Quadrature weights at each quadrature point.

    w2 : 1D array_like (nq2,)
        Quadrature weights at each quadrature point.

    jac_mat : 4D array_like (nq1, nq2, 2, 2)
        Jacobian matrix of the mapping F(x1,x2)=(x,y) at each quadrature point.

    mat_m : 4D array_like (p1+1, p2+1, 2*p1+1, 2*p2+1)
        Element mass matrix (in/out argument).

    mat_s : 4D array_like (p1+1, p2+1, 2*p1+1, 2*p2+1)
        Element stiffness matrix (in/out argument).

    """
    # Reset element matrices
    mat_m[:,:,:,:] = 0.
    mat_s[:,:,:,:] = 0.

    # Cycle over non-zero test functions in element
    for il1 in range(p1+1):
        for il2 in range(p2+1):

            # Cycle over non-zero trial functions in element
            for jl1 in range(p1+1):
                for jl2 in range(p2+1):

                    # Reset integrals over element
                    v_m = 0.0
                    v_s = 0.0

                    # Cycle over quadrature points
                    for q1 in range(nq1):
                        for q2 in range(nq2):

                            # Get test function's value and derivatives
                            bi_0  = bs1[il1, 0, q1] * bs2[il2, 0, q2]
                            bi_x1 = bs1[il1, 1, q1] * bs2[il2, 0, q2]
                            bi_x2 = bs1[il1, 0, q1] * bs2[il2, 1, q2]

                            # Get trial function's value and derivatives
                            bj_0  = bs1[jl1, 0, q1] * bs2[jl2, 0, q2]
                            bj_x1 = bs1[jl1, 1, q1] * bs2[jl2, 0, q2]
                            bj_x2 = bs1[jl1, 0, q1] * bs2[jl2, 1, q2]

                            # Mapping:
                            #  - from logical coordinates (x1,x2)
                            #  - to Cartesian coordinates (x,y)
                            [[x_x1, x_x2],
                             [y_x1, y_x2]] = jac_mat[q1,q2,:,:]

                            jac_det = x_x1*y_x2 - x_x2*y_x1
                            inv_jac_det = 1./jac_det

                            # Convert basis functions' derivatives:
                            #  - from logical coordinates (x1,x2)
                            #  - to Cartesian coordinates (x,y)
                            bi_x = inv_jac_det * ( y_x2*bi_x1 - y_x1*bi_x2)
                            bi_y = inv_jac_det * (-x_x2*bi_x1 + x_x1*bi_x2)

                            bj_x = inv_jac_det * ( y_x2*bj_x1 - y_x1*bj_x2)
                            bj_y = inv_jac_det * (-x_x2*bj_x1 + x_x1*bj_x2)

                            # Get volume associated to quadrature point
                            wvol = w1[q1] * w2[q2] * abs( jac_det )

                            # Add contribution to integrals
                            v_m +=  bi_0 * bj_0 * wvol
                            v_s += (bi_x * bj_x + bi_y * bj_y) * wvol

                    # Update element matrices
                    mat_m[il1, il2, p1+jl1-il1, p2+jl2-il2 ] = v_m
                    mat_s[il1, il2, p1+jl1-il1, p2+jl2-il2 ] = v_s

#==============================================================================
def assemble_matrices( V, mapping, kernel ):
    """
    Assemble mass and stiffness matrices using 2D stencil format.

    Parameters
    ----------
    V : TensorFemSpace
        Finite element space where the Galerkin method is applied.

    mapping : spl.mapping.basic.Mapping
        Mapping (analytical or discrete) from logical to physical coordinates.

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
    [n1, n2] = V.vector_space.npts

    # Quadrature data
    [      nq1,       nq2] = [W.quad_order   for W in V.spaces]
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

    # Element range
    support1, support2 = V.local_support

    # Build global matrices: cycle over elements
    for k1 in support1:
        for k2 in support2:

            # Get spline index, B-splines' values and quadrature weights
            is1 =   spans_1[k1]
            bs1 =   basis_1[k1,:,:,:]
            w1  = weights_1[k1,:]

            is2 =   spans_2[k2]
            bs2 =   basis_2[k2,:,:,:]
            w2  = weights_2[k2,:]

            # Compute Jacobian matrix at all quadrature points
            jac_mat = np.empty( (nq1,nq2, 2, 2) )
            for q1 in range( nq1 ):
                for q2 in range( nq2 ):
                    x1 = points_1[k1,q1]
                    x2 = points_2[k2,q2]
                    jac_mat[q1,q2,:,:] = mapping.jac_mat( [x1,x2] )

            # Compute element matrices
            kernel( p1, p2, nq1, nq2, bs1, bs2, w1, w2, jac_mat, mat_m, mat_s )

            # Update global matrices
            for il1 in range( p1+1 ):
                for il2 in range( p2+1 ):

                    # Global index of test basis
                    i1 = (is1-p1+il1) % n1
                    i2 = (is2-p2+il2) % n2

                    # If basis belongs to process,
                    # update one row of the global matrices
                    if s1 <= i1 <= e1 and s2 <= i2 <= e2:
                        mass     [i1,i2,:,:] += mat_m[il1,il2,:,:]
                        stiffness[i1,i2,:,:] += mat_s[il1,il2,:,:]

    # Make sure that periodic corners are zero in non-periodic case
    mass     .remove_spurious_entries()
    stiffness.remove_spurious_entries()

    return mass, stiffness

#==============================================================================
def assemble_rhs( V, mapping, f ):
    """
    Assemble right-hand-side vector.

    Parameters
    ----------
    V : TensorFemSpace
        Finite element space where the Galerkin method is applied.

    mapping : spl.mapping.basic.Mapping
        Mapping (analytical or discrete) from logical to physical coordinates.

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
    [n1, n2] = V.vector_space.npts

    # Quadrature data
    [      nq1,       nq2] = [W.quad_order   for W in V.spaces]
    [  spans_1,   spans_2] = [W.spans        for W in V.spaces]
    [  basis_1,   basis_2] = [W.quad_basis   for W in V.spaces]
    [ points_1,  points_2] = [W.quad_points  for W in V.spaces]
    [weights_1, weights_2] = [W.quad_weights for W in V.spaces]

    # Data structure
    rhs = StencilVector( V.vector_space )

    # Element range
    support1, support2 = V.local_support

    # Build RHS
    for k1 in support1:
        for k2 in support2:

            # Get spline index, B-splines' values and quadrature weights
            is1 =   spans_1[k1]
            bs1 =   basis_1[k1,:,:,:]
            w1  = weights_1[k1,:]
            x1  =  points_1[k1,:]

            is2 =   spans_2[k2]
            bs2 =   basis_2[k2,:,:,:]
            w2  = weights_2[k2,:]
            x2  =  points_2[k2,:]

            # Evaluate function at all quadrature points
            f_quad = f( *np.meshgrid( x1, x2, indexing='ij' ) )

            # Compute Jacobian determinant at all quadrature points
            metric_det = np.empty( (nq1,nq2) )
            for q1 in range( nq1 ):
                for q2 in range( nq2 ):
                    metric_det[q1,q2] = mapping.metric_det( [x1[q1],x2[q2]] )
            jac_det = np.sqrt( metric_det )

            for il1 in range( p1+1 ):
                for il2 in range( p2+1 ):

                    v = 0.0
                    for q1 in range( nq1 ):
                        for q2 in range( nq2 ):
                            bi_0 = bs1[il1, 0, q1] * bs2[il2, 0, q2]
                            wvol = w1[q1] * w2[q2] * jac_det[q1,q2]
                            v   += bi_0 * f_quad[q1,q2] * wvol

                    # Global index of test basis
                    i1 = (is1-p1+il1) % n1
                    i2 = (is2-p2+il2) % n2

                    rhs[i1, i2] += v

    # IMPORTANT: ghost regions must be up-to-date
    rhs.update_ghost_regions()

    return rhs

####################################################################################
if __name__ == '__main__':

    timing = {}

    # Input data: degree, number of elements
    p1  = 3  ; p2  = 3
    ne1 = 8 ; ne2 = 16

    # Method of manufactured solution
#    model = Poisson2D.new_square( mx=1, my=1 )
#    model = Poisson2D.new_annulus( rmin=0.05, rmax=1 )
#    model = Poisson2D.new_annulus_separable( rmin=0.3, rmax=1.2 )
    model = Poisson2D.new_circle_separable()

    per1, per2 = model.periodic

    # Create uniform grid
    grid_1 = np.linspace( *model.domain[0], num=ne1+1 )
    grid_2 = np.linspace( *model.domain[1], num=ne2+1 )

    # Create 1D finite element spaces and precompute quadrature data
    V1 = SplineSpace( p1, grid=grid_1, periodic=per1 ); V1.init_fem()
    V2 = SplineSpace( p2, grid=grid_2, periodic=per2 ); V2.init_fem()

    # Create 2D tensor product finite element space
    V = TensorFemSpace( V1, V2 )
#    V = TensorFemSpace( V1, V2, comm=MPI.COMM_WORLD )

    # Analytical and spline mappings
    map_analytic = model.mapping
#    map_discrete = SplineMapping.from_mapping( V, map_analytic )

    # Build mass and stiffness matrices
    t0 = time()
#    mass, stiffness = assemble_matrices( V, map_discrete, kernel )
    mass, stiffness = assemble_matrices( V, map_analytic, kernel )
    t1 = time()
    timing['assembly'] = t1-t0

    # Build right-hand side vector
#    rhs = assemble_rhs( V, map_discrete, model.rho )
    rhs = assemble_rhs( V, map_analytic, model.rho )

    # Apply homogeneous dirichlet boundary conditions
    if not V1.periodic:
        # left  bc at x=0.
        stiffness[0,:,:,:] = 0.
        rhs      [0,:]     = 0.
        # right bc at x=1.
        stiffness[V1.nbasis-1,:,:,:] = 0.
        rhs      [V1.nbasis-1,:]     = 0.

    if not V2.periodic:
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
#    jac_det   = lambda *x: np.sqrt( map_discrete.metric_det( x ) )
    jac_det   = lambda *x: np.sqrt( map_analytic.metric_det( x ) )
    integrand = lambda *x: (phi(*x)-model.phi(*x))**2 * jac_det(*x)
    e2 = np.sqrt( V.integral( integrand ) )
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

    ##########
    N = 10
    ##########

    # Compute numerical solution (and error) on refined logical grid
    eta1 = refine_array_1d( V1.breaks, N )
    eta2 = refine_array_1d( V2.breaks, N )
    num = np.array( [[      phi( e1,e2 ) for e2 in eta2] for e1 in eta1] )
    ex  = np.array( [[model.phi( e1,e2 ) for e2 in eta2] for e1 in eta1] )
    err = num - ex

    # Compute physical coordinates of logical grid
    pcoords = np.array( [[model.mapping( [e1,e2] ) for e2 in eta2] for e1 in eta1] )
    xx = pcoords[:,:,0]
    yy = pcoords[:,:,1]

    # Plot exact solution
    fig, ax = plt.subplots( 1, 1 )
    im = ax.contourf( xx, yy, ex, 40, cmap='jet' )
    fig.colorbar( im )
    ax.set_xlabel( r'$x$', rotation='horizontal' )
    ax.set_ylabel( r'$y$', rotation='horizontal' )
    ax.set_title ( r'$\phi_{ex}(x,y)$' )
    ax.plot( xx[:,::N]  , yy[:,::N]  , 'k' )
    ax.plot( xx[::N,:].T, yy[::N,:].T, 'k' )
    fig.tight_layout()
    fig.show()

    # Plot numerical solution
    fig, ax = plt.subplots( 1, 1 )
    im = ax.contourf( xx, yy, num, 40, cmap='jet' )
    fig.colorbar( im )
    ax.set_xlabel( r'$x$', rotation='horizontal' )
    ax.set_ylabel( r'$y$', rotation='horizontal' )
    ax.set_title ( r'$\phi(x,y)$' )
    ax.plot( xx[:,::N]  , yy[:,::N]  , 'k' )
    ax.plot( xx[::N,:].T, yy[::N,:].T, 'k' )
    fig.tight_layout()
    fig.show()

    # Plot numerical error
    fig, ax = plt.subplots( 1, 1 )
    im = ax.contourf( xx, yy, err, 40, cmap='jet' )
    fig.colorbar( im )
    ax.set_xlabel( r'$x$', rotation='horizontal' )
    ax.set_ylabel( r'$y$', rotation='horizontal' )
    ax.set_title ( r'$\phi(x,y) - \phi_{ex}(x,y)$' )
    ax.plot( xx[:,::N]  , yy[:,::N]  , 'k' )
    ax.plot( xx[::N,:].T, yy[::N,:].T, 'k' )
    fig.tight_layout()
    fig.show()

    # Keep figures open if necessary
    import __main__ as main
    if hasattr( main, '__file__' ):
        try:
           __IPYTHON__
        except NameError:
            plt.show()
