#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np

from psydac.utilities.quadratures      import gauss_legendre
from psydac.linalg.stencil             import StencilVector, StencilMatrix
from psydac.linalg.solvers             import inverse
from psydac.ddm.cart                   import DomainDecomposition

#==============================================================================
class Poisson1D:
    """
    Exact solution to the 1D Poisson equation, to be employed for the method
    of manufactured solutions.

    :code
    $\frac{d^2}{dx^2}\phi(x) = -\rho(x)$

    """
    def __init__(self):
        from sympy import symbols, sin, pi, lambdify
        x = symbols('x')
        phi_e = sin(2*pi*x)
        rho_e = -phi_e.diff(x, 2)
        self._phi = lambdify(x, phi_e)
        self._rho = lambdify(x, rho_e)

    def phi(self, x):
        return self._phi(x)

    def rho(self, x):
        return self._rho(x)

    @property
    def domain(self):
        return (0, 1)

    @property
    def periodic(self):
        return False

#==============================================================================
def kernel(p1, k1, bs1, w1, mat_m, mat_s):
    """
    Kernel for computing the mass/stiffness element matrices.

    Parameters
    ----------
    p1 : int
        Spline degree.

    k1 : int
        Number of quadrature points in each element.

    bs1 : 3D array_like (p1+1, 1+nderiv, k1)
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
    mat_m[:, :] = 0.
    mat_s[:, :] = 0.

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
def assemble_matrices(V, kernel, *, nquads):
    """
    Assemble mass and stiffness matrices using 1D stencil format.

    Parameters
    ----------
    V : SplineSpace
        Finite element space where the Galerkin method is applied.

    kernel : callable
        Function that performs the assembly process on small element matrices.

    nquads : list or tuple of int
        Number of quadrature points in each direction (here only one).

    Returns
    -------
    mass : StencilMatrix
        Mass matrix in 1D stencil format.

    stiffness : StencilMatrix
        Stiffness matrix in 1D stencil format.

    """
    # Sizes
    [s1] = V.coeff_space.starts
    [e1] = V.coeff_space.ends
    [p1] = V.coeff_space.pads

    # Quadrature data
    quad_grid = V.get_quadrature_grids(*nquads)[0]
    nk1       = quad_grid.num_elements
    nq1       = quad_grid.num_quad_pts
    spans_1   = quad_grid.spans
    basis_1   = quad_grid.basis
    weights_1 = quad_grid.weights

    # Create global matrices
    mass      = StencilMatrix(V.coeff_space, V.coeff_space)
    stiffness = StencilMatrix(V.coeff_space, V.coeff_space)

    # Create element matrices
    mat_m = np.zeros((p1+1, 2*p1+1)) # mass
    mat_s = np.zeros((p1+1, 2*p1+1)) # stiffness

    # Build global matrices: cycle over elements
    for k1 in range(nk1):

        # Get spline index, B-splines' values and quadrature weights
        is1 =   spans_1[k1]
        bs1 =   basis_1[k1, :, :, :]
        w1  = weights_1[k1, :]

        # Compute element matrices
        kernel(p1, nq1, bs1, w1, mat_m, mat_s)

        # Update global matrices
        mass     [is1-p1:is1+1, :] += mat_m[:, :]
        stiffness[is1-p1:is1+1, :] += mat_s[:, :]

    return mass, stiffness

#==============================================================================
def assemble_rhs(V, f, *, nquads):
    """
    Assemble right-hand-side vector.

    Parameters
    ----------
    V : SplineSpace
        Finite element space where the Galerkin method is applied.

    f : callable
        Right-hand side function (charge density).

    nquads : list or tuple of int
        Number of quadrature points in each direction (here only one).

    Returns
    -------
    rhs : StencilVector
        Vector b of coefficients, in linear system Ax=b.

    """
    # Sizes
    [s1] = V.coeff_space.starts
    [e1] = V.coeff_space.ends
    [p1] = V.coeff_space.pads

    # Quadrature data
    quad_grid = V.get_quadrature_grids(*nquads)[0]
    nk1       = quad_grid.num_elements
    nq1       = quad_grid.num_quad_pts
    spans_1   = quad_grid.spans
    basis_1   = quad_grid.basis
    points_1  = quad_grid.points
    weights_1 = quad_grid.weights

    # Data structure
    rhs = StencilVector(V.coeff_space)

    # Build RHS
    for k1 in range(nk1):

        is1    =   spans_1[k1]
        bs1    =   basis_1[k1, :, :, :]
        x1     =  points_1[k1, :]
        wvol   = weights_1[k1, :]
        f_quad = f(x1)

        for il1 in range(p1+1):

            bi_0 = bs1[il1, 0, :]
            v    = bi_0 * f_quad * wvol

            rhs[is1-p1+il1] += v.sum()

    return rhs

####################################################################################
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from time import time

    from psydac.fem.splines import SplineSpace
    from psydac.fem.tensor  import TensorFemSpace
    from psydac.fem.basic   import FemField

    timing = {}

    # Input data: degree, number of elements
    p  = 3
    ne = 2**4

    # Number of quadrature points for assembling of matrices and vectors
    nquads = [p + 1]

    # Method of manufactured solution
    model = Poisson1D()

    # Create uniform grid
    grid = np.linspace(*model.domain, num=ne+1)

    # Create finite element space
    space = SplineSpace(p, grid=grid, periodic=model.periodic)
    dd = DomainDecomposition(ncells=[space.ncells], periods=[model.periodic])
    V = TensorFemSpace(dd, space)

    # Build mass and stiffness matrices
    t0 = time()
    mass, stiffness = assemble_matrices(V, kernel, nquads=nquads)
    t1 = time()
    timing['assembly'] = t1-t0

    # Build right-hand side vector
    rhs = assemble_rhs(V, model.rho, nquads=nquads)

    # Apply homogeneous Dirichlet boundary conditions
    s1, = V.coeff_space.starts
    e1, = V.coeff_space.ends

    stiffness[s1, :] = 0.
    stiffness[e1, :] = 0.
    rhs[s1] = 0.
    rhs[e1] = 0.

    # Solve linear system
    t0 = time()
    stiffness_inv = inverse(stiffness, 'cg', tol=1e-9, maxiter=1000, verbose=False)
    x = stiffness_inv @ rhs
    info = stiffness_inv.get_info()
    t1 = time()
    timing['solution'] = t1-t0

    # Create potential field
    phi = FemField(V, coeffs=x)
    phi.coeffs.update_ghost_regions()

    # Compute L2 norm of error
    t0 = time()
    e2 = np.sqrt(V.integral(lambda x: (phi(x)-model.phi(x))**2, nquads=[8]))
    t1 = time()
    timing['diagnostics'] = t1-t0

    # Print some information to terminal
    print('> Grid          :: {ne}'.format(ne=ne))
    print('> Degree        :: {p}'.format(p=p))
    print('> CG info       :: ', info)
    print('> L2 error      :: {:.2e}'.format(e2))
    print()
    print('> Assembly time :: {:.2e}'.format(timing['assembly']))
    print('> Solution time :: {:.2e}'.format(timing['solution']))
    print('> Evaluat. time :: {:.2e}'.format(timing['diagnostics']))

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
