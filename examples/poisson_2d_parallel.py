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
#        return (True, False)

#==============================================================================
def compute_basis_support( V ):
    """
    Compute the support of all the basis functions local to the process.

    Thanks to the presence of ghost values, this is also equivalent to the
    region over which the coefficients of all non-zero basis functions are
    available and hence a field can be evaluated.

    This function takes into account:
        1. periodic boundary conditions
        2. repeated internal knots

    """
    starts  = V.vector_space.starts
    ends    = V.vector_space.ends
    nbasis  = V.vector_space.npts

    degrees = V.degree
    ncells  = V.ncells
    spans   = [W.spans for W in V.spaces]

    supports = [[k for k in range( nc )
        if any( s <= i%nb <= e for i in range( span[k]-p, span[k]+1 ) )]
        for (s,e,p,nb,nc,span) in zip( starts, ends, degrees, nbasis, ncells, spans )]

    return tuple( tuple( np.unique( sup ) ) for sup in supports )

#==============================================================================
def compute_domain_decomposition( V ):
    """
    Determine logical domain local to the process, assuming the global domain
    is decomposed across processes without any overlapping.

    This information is fundamental for avoiding double-counting when computing
    integrals over the global domain.

    """
    # TODO: 1) take into account periodic boundary conditions
    # TODO: 2) take into account repeated internal knots

    v = V.vector_space

    if v.parallel: 
        coords = v.cart.coords
        nprocs = v.cart.nprocs
    else:
        coords = tuple( [0]*v.ndim )
        nprocs = tuple( [1]*v.ndim )

    iterator = lambda: zip( v.starts, v.ends, v.pads, coords, nprocs ) 

    element_starts = [(s   if c == 0    else s-p+1) for s,e,p,c,np in iterator()]
    element_ends   = [(e-p if c == np-1 else e-p+1) for s,e,p,c,np in iterator()]
    
    return element_starts, element_ends

#==============================================================================
def assemble_matrices( V, kernel=None, debug=False ):
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
    [      nq1,       nq2] = [W.quad_order   for W in V.spaces]
    [  spans_1,   spans_2] = [W.spans        for W in V.spaces]
    [  basis_1,   basis_2] = [W.quad_basis   for W in V.spaces]
    [ points_1,  points_2] = [W.quad_points  for W in V.spaces]
    [weights_1, weights_2] = [W.quad_weights for W in V.spaces]

    # Create global matrices
    mass      = StencilMatrix( V.vector_space, V.vector_space )
    stiffness = StencilMatrix( V.vector_space, V.vector_space )

    # Element range
    support1, support2 = compute_basis_support( V )

    # Build global matrices: cycle over elements
    for k1 in support1:

        # Get spline index, B-splines' values and quadrature weights
        is1 =   spans_1[k1]
        bs1 =   basis_1[k1,:,:,:]
        w1  = weights_1[k1,:]

        # Get local start/end index of basis functions
        sl1 = max(  0, s1-is1+p1 )
        el1 = min( p1, e1-is1+p1 )

        for k2 in support2:

            # Get spline index, B-splines' values and quadrature weights
            is2 =   spans_2[k2]
            bs2 =   basis_2[k2,:,:,:]
            w2  = weights_2[k2,:]

            # Get local start/end index of basis functions
            sl2 = max(  0, s2-is2+p2 )
            el2 = min( p2, e2-is2+p2 )

            # Reset element matrices
            mat_m = np.zeros( (el1-sl1+1, el2-sl2+1, 2*p1+1, 2*p2+1) ) # mass
            mat_s = np.zeros( (el1-sl1+1, el2-sl2+1, 2*p1+1, 2*p2+1) ) # stiffness

            # Cycle over non-zero test functions in element
            for il1 in range( sl1, el1+1 ):
                for il2 in range( sl2, el2+1 ):

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
                            mat_m[il1-sl1, il2-sl2, p1+jl1-il1, p2+jl2-il2] = v_m
                            mat_s[il1-sl1, il2-sl2, p1+jl1-il1, p2+jl2-il2] = v_s

            # Update global matrices (NOTE: COMPACT VERSION)
            i1_slice = slice( is1-p1+sl1, is1-p1+el1+1 )
            i2_slice = slice( is2-p2+sl2, is2-p2+el2+1 )

            mass     [i1_slice, i2_slice, :, :] += mat_m[:,:,:,:]
            stiffness[i1_slice, i2_slice, :, :] += mat_s[:,:,:,:]

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

    # Quadrature data
    [      nq1,       nq2] = [W.quad_order   for W in V.spaces]
    [  spans_1,   spans_2] = [W.spans        for W in V.spaces]
    [  basis_1,   basis_2] = [W.quad_basis   for W in V.spaces]
    [ points_1,  points_2] = [W.quad_points  for W in V.spaces]
    [weights_1, weights_2] = [W.quad_weights for W in V.spaces]

    # Data structure
    rhs = StencilVector( V.vector_space )

    # Element range
    support1, support2 = compute_basis_support( V )

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

                    i1 = is1 - p1 + il1
                    i2 = is2 - p2 + il2

                    rhs[i1, i2] += v

    # IMPORTANT: ghost regions must be up-to-date
    rhs.update_ghost_regions()

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
    [      nq1,       nq2] = [W.quad_order   for W in V.spaces]
    [ points_1,  points_2] = [W.quad_points  for W in V.spaces]
    [weights_1, weights_2] = [W.quad_weights for W in V.spaces]

    # Element range
    (sk1,sk2), (ek1,ek2) = compute_domain_decomposition( V )

    c = 0.0
    for k1 in range(sk1, ek1+1):
        for k2 in range(sk2, ek2+1):

            x1 =  points_1[k1,:]
            w1 = weights_1[k1,:]

            x2 =  points_2[k2,:]
            w2 = weights_2[k2,:]

            for q1 in range( nq1 ):
                for q2 in range( nq2 ):
                    c += f( x1[q1], x2[q2] ) * w1[q1] * w2[q2]

    # All reduce (MPI_SUM)
    mpi_comm = V.vector_space.cart._comm
    c = mpi_comm.allreduce( c )

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
    mass, stiffness = assemble_matrices( V, debug=False )
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
    err2 = error_norm( V, phi, model.phi, order=2 )
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
