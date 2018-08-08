# coding: utf-8
import numpy as np

from spl.utilities.quadratures      import gauss_legendre
from spl.linalg.stencil             import StencilVector, StencilMatrix
from spl.linalg.iterative_solvers   import cg

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
#        phi_e = sin( 2*pi*x ) * sin( 2*pi*y )
        phi_e = cos( 2*pi*(x+0.1) ) * sin( pi*y ) * y
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
#        return (False, False)
        return (True, False)

#==============================================================================
def kernel( p1, p2, nq1, nq2, bs1, bs2, w1, w2, mat_m, mat_s ):
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

    mat_m : 2D array_like (p1+1, p2+1, 2*p1+1, 2*p2+1)
        Element mass matrix (in/out argument).

    mat_s : 2D array_like (p1+1, p2+1, 2*p1+1, 2*p2+1)
        Element stiffness matrix (in/out argument).

    """
    # Reset element matrices
    mat_m[:,:,:,:] = 0.
    mat_s[:,:,:,:] = 0.

    # Cycle over non-zero test functions in element
    for il1 in range( p1+1 ):
        for il2 in range( p2+1 ):

            # Cycle over non-zero trial functions in element
            for jl1 in range( p1+1 ):
                for jl2 in range( p2+1 ):

                    # Reset integrals over element
                    v_m = 0.0
                    v_s = 0.0

                    # Cycle over quadrature points
                    for q1 in range( nq1 ):
                        for q2 in range( nq2 ):

                            # Get test function's value and derivatives
                            bi_0 = bs1[il1, 0, q1] * bs2[il2, 0, q2]
                            bi_x = bs1[il1, 1, q1] * bs2[il2, 0, q2]
                            bi_y = bs1[il1, 0, q1] * bs2[il2, 1, q2]

                            # Get trial function's value and derivatives
                            bj_0 = bs1[jl1, 0, q1] * bs2[jl2, 0, q2]
                            bj_x = bs1[jl1, 1, q1] * bs2[jl2, 0, q2]
                            bj_y = bs1[jl1, 0, q1] * bs2[jl2, 1, q2]

                            # Get quadrature weight
                            wvol = w1[q1] * w2[q2]

                            # Add contribution to integrals
                            v_m += bi_0 * bj_0 * wvol
                            v_s += (bi_x * bj_x + bi_y * bj_y) * wvol

                    # Update element matrices
                    mat_m[il1, il2, p1+jl1-il1, p2+jl2-il2 ] = v_m
                    mat_s[il1, il2, p1+jl1-il1, p2+jl2-il2 ] = v_s

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

    # Cycle over elements
    for k1 in support1:
        for k2 in support2:

            # Get spline index, B-splines' values and quadrature weights
            is1 =   spans_1[k1]
            bs1 =   basis_1[k1,:,:,:]
            w1  = weights_1[k1,:]

            is2 =   spans_2[k2]
            bs2 =   basis_2[k2,:,:,:]
            w2  = weights_2[k2,:]

            # Compute element matrices
            kernel( p1, p2, nq1, nq2, bs1, bs2, w1, w2, mat_m, mat_s )

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

            for il1 in range( p1+1 ):
                for il2 in range( p2+1 ):

                    v = 0.0
                    for q1 in range( nq1 ):
                        for q2 in range( nq2 ):
                            bi_0 = bs1[il1, 0, q1] * bs2[il2, 0, q2]
                            wvol = w1[q1] * w2[q2]
                            v   += bi_0 * f_quad[q1,q2] * wvol

                    i1 = (is1-p1+il1) % n1
                    i2 = (is2-p2+il2) % n2

                    if s1<=i1<=e1 and s2<=i2<=e2:
                        rhs[i1, i2] += v

    # IMPORTANT: ghost regions must be up-to-date
    rhs.update_ghost_regions()

    return rhs

####################################################################################
if __name__ == '__main__':

    from mpi4py import MPI
    from time import time, sleep
    import matplotlib.pyplot as plt

    from spl.fem.splines import SplineSpace
    from spl.fem.tensor  import TensorFemSpace
    from spl.fem.basic   import FemField

    timing = {}

    # Communicator, size, rank
    mpi_comm = MPI.COMM_WORLD
    mpi_size = mpi_comm.Get_size()
    mpi_rank = mpi_comm.Get_rank()

    # Input data: degree, number of elements
    p1  = 3 ; p2  = 3
    nk1 = 16; nk2 = 16

    # Method of manufactured solution
    model = Poisson2D()
    per1, per2 = model.periodic

    # Create uniform grid
    grid_1 = np.linspace( *model.domain[0], num=nk1+1 )
    grid_2 = np.linspace( *model.domain[1], num=nk2+1 )

    # Create 1D finite element spaces and precompute quadrature data
    V1 = SplineSpace( p1, grid=grid_1, periodic=per1 ); V1.init_fem()
    V2 = SplineSpace( p2, grid=grid_2, periodic=per2 ); V2.init_fem()

    # Create 2D tensor product finite element space
    V = TensorFemSpace( V1, V2, comm=mpi_comm )
    s1, s2 = V.vector_space.starts
    e1, e2 = V.vector_space.ends

    # Build mass and stiffness matrices
    t0 = time()
    mass, stiffness = assemble_matrices( V, kernel )
    t1 = time()
    timing['assembly'] = t1-t0

    # Build right-hand side vector
    rhs = assemble_rhs( V, model.rho )

    # Apply homogeneous dirichlet boundary conditions
    if not V1.periodic:
        # left  bc at x=0.
        if s1 == 0:
            stiffness[0,:,:,:] = 0.
            rhs      [0,:]     = 0.
        # right bc at x=1.
        if e1 == V1.nbasis-1:
            stiffness[e1,:,:,:] = 0.
            rhs      [e1,:]     = 0.

    if not V2.periodic:
        # lower bc at y=0.
        if s2 == 0:
            stiffness[:,0,:,:] = 0.
            rhs      [:,0]     = 0.
        # upper bc at y=1.
        if e2 == V2.nbasis-1:
            stiffness[:,e2,:,:] = 0.
            rhs      [:,e2]     = 0.

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
    err2 = np.sqrt( V.integral( lambda *x: (phi(*x)-model.phi(*x))**2 ) )
    t1 = time()
    timing['diagnostics'] = t1-t0

    # Print some information to terminal
    for i in range( mpi_size ):
        if i == mpi_rank:
            print( '--------------------------------------------------' )
            print( ' RANK = {}'.format( mpi_rank ) )
            print( '--------------------------------------------------' )
            print( '> Grid          :: [{nk1},{nk2}]'.format( nk1=nk1, nk2=nk2) )
            print( '> Degree        :: [{p1},{p2}]'  .format( p1=p1, p2=p2 ) )
            print( '> CG info       :: ',info )
            print( '> L2 error      :: {:.2e}'.format( err2 ) )
            print( '' )
            print( '> Assembly time :: {:.2e}'.format( timing['assembly'] ) )
            print( '> Solution time :: {:.2e}'.format( timing['solution'] ) )
            print( '> Evaluat. time :: {:.2e}'.format( timing['diagnostics'] ) )
            print( '', flush=True )
            sleep( 0.001 )
        mpi_comm.Barrier()

#    # Plot solution on refined grid
#    xx = np.linspace( *model.domain[0], num=101 )
#    yy = np.linspace( *model.domain[1], num=101 )
#    zz = np.array( [[phi( xi,yi ) for yi in yy] for xi in xx] )
#    fig, ax = plt.subplots( 1, 1 )
#    im = ax.contourf( xx, yy, zz.transpose(), 40, cmap='jet' )
#    fig.colorbar( im )
#    ax.set_xlabel( r'$x$', rotation='horizontal' )
#    ax.set_ylabel( r'$y$', rotation='horizontal' )
#    ax.set_title ( r'$\phi(x,y)$' )
#    ax.grid()
#
#    # Show figure and keep it open if necessary
#    fig.tight_layout()
#    fig.show()
#
#    import __main__ as main
#    if hasattr( main, '__file__' ):
#        try:
#           __IPYTHON__
#        except NameError:
#            plt.show()
