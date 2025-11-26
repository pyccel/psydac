#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from time import time, sleep

from   mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from   mpl_toolkits.axes_grid1 import make_axes_locatable

from sympde.topology.callable_mapping   import CallableMapping
from sympde.topology.analytical_mapping import IdentityMapping, PolarMapping
from sympde.topology.analytical_mapping import TargetMapping, CzarnyMapping

from psydac.ddm.cart             import DomainDecomposition
from psydac.linalg.stencil       import StencilVector, StencilMatrix
from psydac.linalg.solvers       import inverse
from psydac.fem.splines          import SplineSpace
from psydac.fem.tensor           import TensorFemSpace
from psydac.fem.basic            import FemField
from psydac.mapping.discrete     import SplineMapping
from psydac.utilities.utils      import refine_array_1d
from psydac.cad.geometry         import Geometry
from psydac.ddm.cart             import DomainDecomposition
from psydac.polar.c1_projections import C1Projector

#==============================================================================
class Laplacian:

    def __init__(self, mapping):

        assert isinstance(mapping, CallableMapping)

        sym = mapping.symbolic_mapping

        self._eta        = sym.logical_coordinates
        self._metric     = sym.metric_expr
        self._metric_det = sym.metric_det_expr

    # ...
    def __call__(self, phi):

        from sympy import sqrt, Matrix

        u      = self._eta
        G      = self._metric
        sqrt_g = sqrt(self._metric_det)

        # Store column vector of partial derivatives of phi w.r.t. uj
        dphi_du = Matrix([phi.diff(uj) for uj in u])

        # Compute gradient of phi in tangent basis: A = G^(-1) dphi_du
        A = G.LUsolve(dphi_du)

        # Compute Laplacian of phi using formula for divergence of vector A
        lapl = sum((sqrt_g * Ai).diff(ui) for ui, Ai in zip(u, A)) / sqrt_g

        return lapl

#==============================================================================
class Poisson2D:
    r"""
    Exact solution to the 2D Poisson equation with Dirichlet boundary
    conditions, to be employed for the method of manufactured solutions.

    :code
    $(\partial^2_{xx} + \partial^2_{yy}) \phi(x,y) = -\rho(x,y)$

    """
    def __init__(self, domain, periodic, mapping, phi, rho, O_point=False):

        self._domain   = domain
        self._periodic = periodic
        self._mapping  = mapping
        self._phi      = phi
        self._rho      = rho
        self._O_point  = O_point

    # ...
    @staticmethod
    def new_square(mx=1, my=1):
        r"""
        Solve Poisson's equation on the unit square.

        : code
        $\phi(x,y) = sin( mx*pi*x ) + sin( my*pi*y )$

        with $mx$ and $my$ user-defined integer numbers.

        """
        domain   = ((0,1), (0,1))
        periodic = (False, False)
        mapping  = IdentityMapping('F', dim=2).get_callable_mapping()

        from sympy import symbols, sin, cos, pi, lambdify
        x,y   = symbols('x y')
        phi_e = sin(mx * pi * x) * sin(my * pi * y)
        rho_e = -phi_e.diff(x, 2) - phi_e.diff(y, 2)

        phi = lambdify([x, y], phi_e)
        rho = lambdify([x, y], rho_e)

        return Poisson2D(domain, periodic, mapping, phi, rho)

    # ...
    @staticmethod
    def new_annulus(rmin=0.5, rmax=1.0):
        r"""
        Solve Poisson's equation on an annulus centered at (x,y)=(0,0),
        with logical coordinates (r,theta):

        - The radial coordinate r belongs to the interval [rmin,rmax];
        - The angular coordinate theta belongs to the interval [0,2*pi).

        : code
        $\phi(x,y) = 4(r-rmin)(rmax-r)/(rmax-rmin)^2 \sin(2\pi x) \sin(2\pi y)$.

        """
        domain   = ((0, 1), (0, 2*np.pi))
        periodic = (False, True)
        mapping  = PolarMapping('F', c1=0, c2=0, rmin=rmin, rmax=rmax).get_callable_mapping()

        from sympy import symbols, sin, cos, pi, lambdify

        lapl  = Laplacian(mapping)
        r, t  = mapping.symbolic_mapping.logical_coordinates
        x, y  = mapping.symbolic_mapping.expressions

        # Manufactured solutions in logical coordinates
        parab = (r-rmin) * (rmax-r) * 4 / (rmax-rmin)**2
        phi_e = parab * sin(2*pi*x) * sin(2*pi*y)
        rho_e = -lapl(phi_e)

        # For further simplifications, assume that (r,t) are positive and real
        R,T   = symbols('R T', real=True, positive=True)
        phi_e = phi_e.subs({r:R, t:T}).simplify()
        rho_e = rho_e.subs({r:R, t:T}).simplify()

        # Callable functions
        phi = lambdify([R, T], phi_e)
        rho = lambdify([R, T], rho_e)

        return Poisson2D( domain, periodic, mapping, phi, rho, O_point=(rmin==0) )

    # ...
    @staticmethod
    def new_circle():
        r"""
        Solve Poisson's equation on a unit circle centered at (x,y)=(0,0),
        with logical coordinates (r,theta):

        - The radial coordinate r belongs to the interval [0,1];
        - The angular coordinate theta belongs to the interval [0,2*pi).

        : code
        $\phi(x,y) = 1-r**2$.

        """
        domain   = ((0, 1), (0, 2*np.pi))
        periodic = (False, True)
        mapping  = PolarMapping('F', c1=0, c2=0, rmin=0, rmax=2*np.pi).get_callable_mapping()

        from sympy import lambdify

        lapl  = Laplacian(mapping)
        r, t  = mapping.symbolic_mapping.logical_coordinates

        # Manufactured solutions in logical coordinates
        phi_e = 1-r**2
        rho_e = -lapl(phi_e)

        # Callable functions
        phi = lambdify([r, t], phi_e)
        rho = lambdify([r, t], rho_e)

        rho = np.vectorize(rho)

        return Poisson2D(domain, periodic, mapping, phi, rho, O_point=True)

    # ...
    @staticmethod
    def new_target():

        domain   = ((0, 1), (0, 2*np.pi))
        periodic = (False, True)
        params   = dict(c1=0, c2=0, k=0.3, D=0.2)
        mapping  = TargetMapping('F', **params).get_callable_mapping()

        from sympy import symbols, sin, cos, pi, lambdify

        lapl  = Laplacian(mapping)
        s, t  = mapping.symbolic_mapping.logical_coordinates
        x, y  = mapping.symbolic_mapping.expressions

        # Manufactured solution in logical coordinates
        k     = params['k']
        D     = params['D']
        kx    = 2*pi/(1-k+D)
        ky    = 2*pi/(1+k)
        phi_e = (1-s**8) * sin(kx*(x-0.5)) * cos(ky*y)
        rho_e = -lapl(phi_e)

        # Callable functions
        phi = lambdify([s, t], phi_e)
        rho = lambdify([s, t], rho_e)

        return Poisson2D( domain, periodic, mapping, phi, rho, O_point=True )

    # ...
    @staticmethod
    def new_czarny():

        domain   = ((0, 1), (0, 2*np.pi))
        periodic = (False, True)
        params   = dict(c1=0, c2=0, eps=0.2, b=1.4)
        mapping  = CzarnyMapping('F', **params).get_callable_mapping()

        from sympy import symbols, sin, cos, pi, lambdify

        lapl = Laplacian(mapping)
        s, t = mapping.symbolic_mapping.logical_coordinates
        x, y = mapping.symbolic_mapping.expressions

        # Manufactured solution in logical coordinates
        phi_e = (1-s**8) * sin(pi*x) * cos(pi*y)
        rho_e = -lapl(phi_e)

        # Callable functions
        phi = lambdify([s, t], phi_e)
        rho = lambdify([s, t], rho_e)

        return Poisson2D(domain, periodic, mapping, phi, rho, O_point=True)

    # ...
    @property
    def domain(self):
        return self._domain

    @property
    def periodic(self):
        return self._periodic

    @property
    def mapping(self):
        return self._mapping

    @property
    def phi(self):
        return self._phi

    @property
    def rho(self):
        return self._rho

    @property
    def O_point(self):
        return self._O_point

#==============================================================================
def mpi_print(string, *args, comm=None, **kwargs):
    if comm is not None:
        assert isinstance(comm, MPI.Comm)
        if comm.rank == 0:
            kwargs['flush'] = True
            print(string, *args, **kwargs)
        comm.Barrier()
    else:
        print(string, *args, **kwargs)

#==============================================================================
def kernel(p1, p2, nq1, nq2, bs1, bs2, w1, w2, jac_mat, mat_m, mat_s):
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

    bs1 : 3D array_like (p1+1, 1+nderiv, nq1)
        Values (and derivatives) of non-zero basis functions along x1
        at each quadrature point.

    bs2 : 3D array_like (p2+1, 1+nderiv, nq2)
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
    mat_m[:, :, :, :] = 0.
    mat_s[:, :, :, :] = 0.

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
                            wvol = w1[q1] * w2[q2] * abs(jac_det)

                            # Add contribution to integrals
                            v_m +=  bi_0 * bj_0 * wvol
                            v_s += (bi_x * bj_x + bi_y * bj_y) * wvol

                    # Update element matrices
                    mat_m[il1, il2, p1+jl1-il1, p2+jl2-il2] = v_m
                    mat_s[il1, il2, p1+jl1-il1, p2+jl2-il2] = v_s

#==============================================================================
def assemble_matrices(V, mapping, kernel, *, nquads):
    """
    Assemble mass and stiffness matrices using 2D stencil format.

    Parameters
    ----------
    V : TensorFemSpace
        Finite element space where the Galerkin method is applied.

    mapping : psydac.mapping.basic.Mapping
        Mapping (analytical or discrete) from logical to physical coordinates.

    kernel : callable
        Function that performs the assembly process on small element matrices.

    nquads : list or tuple of int
        Number of quadrature points in each direction (here two).

    Returns
    -------
    mass : StencilMatrix
        Mass matrix in 2D stencil format.

    stiffness : StencilMatrix
        Stiffness matrix in 2D stencil format.

    """
    # Sizes
    [s1, s2] = V.coeff_space.starts
    [e1, e2] = V.coeff_space.ends
    [p1, p2] = V.coeff_space.pads

    # Quadrature data
    quad_grids = V.get_assembly_grids(*nquads)
    [      nk1,       nk2] = [g.num_elements for g in quad_grids]
    [      nq1,       nq2] = [g.num_quad_pts for g in quad_grids]
    [  spans_1,   spans_2] = [g.spans        for g in quad_grids]
    [  basis_1,   basis_2] = [g.basis        for g in quad_grids]
    [ points_1,  points_2] = [g.points       for g in quad_grids]
    [weights_1, weights_2] = [g.weights      for g in quad_grids]

    # Create global matrices
    mass      = StencilMatrix(V.coeff_space, V.coeff_space)
    stiffness = StencilMatrix(V.coeff_space, V.coeff_space)

    # Create element matrices
    mat_m = np.zeros((p1+1, p2+1, 2*p1+1, 2*p2+1)) # mass
    mat_s = np.zeros((p1+1, p2+1, 2*p1+1, 2*p2+1)) # stiffness

    # Build global matrices: cycle over elements
    for k1 in range(nk1):
        for k2 in range(nk2):

            # Get spline index, B-splines' values and quadrature weights
            is1 =   spans_1[k1]
            bs1 =   basis_1[k1, :, :, :]
            w1  = weights_1[k1, :]

            is2 =   spans_2[k2]
            bs2 =   basis_2[k2, :, :, :]
            w2  = weights_2[k2, :]

            # Compute Jacobian matrix at all quadrature points
            jac_mat = np.empty((nq1, nq2, 2, 2))
            for q1 in range(nq1):
                for q2 in range(nq2):
                    x1 = points_1[k1, q1]
                    x2 = points_2[k2, q2]
                    jac_mat[q1, q2, :, :] = mapping.jacobian(x1, x2)

            # Compute element matrices
            kernel(p1, p2, nq1, nq2, bs1, bs2, w1, w2, jac_mat, mat_m, mat_s)

            # Update global matrices
            mass     [is1-p1:is1+1, is2-p2:is2+1, :, :] += mat_m[:, :, :, :]
            stiffness[is1-p1:is1+1, is2-p2:is2+1, :, :] += mat_s[:, :, :, :]

    # IMPORTANT: new assembly procedure requires dedicated data exchange
    mass     .exchange_assembly_data()
    stiffness.exchange_assembly_data()

    # Make sure that periodic corners are zero in non-periodic case
    mass     .remove_spurious_entries()
    stiffness.remove_spurious_entries()

    return mass, stiffness

#==============================================================================
def assemble_rhs(V, mapping, f, *, nquads):
    """
    Assemble right-hand-side vector.

    Parameters
    ----------
    V : TensorFemSpace
        Finite element space where the Galerkin method is applied.

    mapping : psydac.mapping.basic.Mapping
        Mapping (analytical or discrete) from logical to physical coordinates.

    f : callable
        Right-hand side function rho(x,y) (charge density).

    Returns
    -------
    rhs : StencilVector
        Vector b of coefficients, in linear system Ax=b.

    """
    # Sizes
    [s1, s2] = V.coeff_space.starts
    [e1, e2] = V.coeff_space.ends
    [p1, p2] = V.coeff_space.pads

    # Quadrature data
    quad_grids = V.get_assembly_grids(*nquads)
    [      nk1,       nk2] = [g.num_elements for g in quad_grids]
    [      nq1,       nq2] = [g.num_quad_pts for g in quad_grids]
    [  spans_1,   spans_2] = [g.spans        for g in quad_grids]
    [  basis_1,   basis_2] = [g.basis        for g in quad_grids]
    [ points_1,  points_2] = [g.points       for g in quad_grids]
    [weights_1, weights_2] = [g.weights      for g in quad_grids]

    # Data structure
    rhs = StencilVector(V.coeff_space)

    # Build RHS
    for k1 in range(nk1):
        for k2 in range(nk2):

            # Get spline index, B-splines' values and quadrature weights
            is1 =   spans_1[k1]
            bs1 =   basis_1[k1, :, :, :]
            w1  = weights_1[k1, :]
            x1  =  points_1[k1, :]

            is2 =   spans_2[k2]
            bs2 =   basis_2[k2, :, :, :]
            w2  = weights_2[k2, :]
            x2  =  points_2[k2, :]

            # Evaluate function at all quadrature points
            f_quad = f(*np.meshgrid(x1, x2, indexing='ij'))

            # Compute Jacobian determinant at all quadrature points
            metric_det = np.empty((nq1, nq2))
            for q1 in range(nq1):
                for q2 in range(nq2):
                    metric_det[q1, q2] = mapping.metric_det(x1[q1], x2[q2])
            jac_det = np.sqrt(metric_det)

            for il1 in range(p1+1):
                for il2 in range(p2+1):

                    v = 0.0
                    for q1 in range(nq1):
                        for q2 in range(nq2):
                            bi_0 = bs1[il1, 0, q1] * bs2[il2, 0, q2]
                            wvol = w1[q1] * w2[q2] * jac_det[q1, q2]
                            v   += bi_0 * f_quad[q1, q2] * wvol

                    # Global index of test basis
                    i1 = is1 - p1 + il1
                    i2 = is2 - p2 + il2

                    # Update one element of the rhs vector
                    rhs[i1, i2] += v

    # IMPORTANT: new assembly procedure requires dedicated data exchange
    rhs.exchange_assembly_data()

    # IMPORTANT: ghost regions must be up-to-date
    rhs.update_ghost_regions()

    return rhs

####################################################################################

def main(*, test_case, ncells, degree, nquads,
         use_spline_mapping, c1_correction, distribute_viz):

    timing = {}
    timing['assembly'   ] = 0.0
    timing['projection' ] = 0.0
    timing['solution'   ] = 0.0
    timing['diagnostics'] = 0.0
    timing['export'     ] = 0.0

    # Method of manufactured solution
    if test_case == 'square':
        model = Poisson2D.new_square(mx=1, my=1)
    elif test_case == 'annulus':
        model = Poisson2D.new_annulus(rmin=0.1, rmax=1.0)
    elif test_case == 'circle':
        model = Poisson2D.new_circle()
    elif test_case == 'target':
        model = Poisson2D.new_target()
    elif test_case == 'czarny':
        model = Poisson2D.new_czarny()
    else:
        raise ValueError("Only available test-cases are 'square', 'annulus', "
                         "'circle', 'target' and 'czarny'")

    # Communicator, size, rank
    mpi_comm = MPI.COMM_WORLD
    mpi_size = mpi_comm.Get_size()
    mpi_rank = mpi_comm.Get_rank()

    # If not explicitly provided, set number quadrature points to default value
    if nquads is None:
        nquads = [d + 1 for d in degree]
        mpi_print(f'NOTE: Setting number of quadrature points to {nquads}', comm=mpi_comm)

    if c1_correction and (not model.O_point):
        mpi_print("WARNING: cannot use C1 correction in geometry without polar singularity!\n"
                  "WARNING: setting 'c1_correction' flag to False...\n",
                  comm = mpi_comm)
        c1_correction = False

    if c1_correction and (not use_spline_mapping):
        mpi_print("WARNING: cannot use C1 correction without spline mapping!\n"
                  "WARNING: setting 'c1_correction' flag to False...\n",
                  comm = mpi_comm)
        c1_correction = False

    # Number of elements and spline degree
    ne1, ne2 = ncells
    p1 , p2  = degree

    # Is solution periodic?
    per1, per2 = model.periodic

    # Create uniform grid
    grid_1 = np.linspace(*model.domain[0], num=ne1+1)
    grid_2 = np.linspace(*model.domain[1], num=ne2+1)

    # Decompose 2D domain across MPI processes
    dd = DomainDecomposition(ncells, model.periodic, comm=mpi_comm)

    # Create 1D finite element spaces
    V1 = SplineSpace(p1, grid=grid_1, periodic=per1)
    V2 = SplineSpace(p2, grid=grid_2, periodic=per2)

    # Create 2D tensor product finite element space
    V = TensorFemSpace(dd, V1, V2)

    s1, s2 = V.coeff_space.starts
    e1, e2 = V.coeff_space.ends

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Print decomposition information to terminal
    if mpi_rank == 0:
        print('--------------------------------------------------')
        print(' CARTESIAN DECOMPOSITION' )
        print('--------------------------------------------------')
        int_array_to_str = lambda array: ','.join('{:3d}'.format(i) for i in array)
        int_tuples_to_str = lambda tuples:  ',  '.join(
                '[{:d}, {:d}]'.format(a,b) for a,b in tuples)

        cart = V.coeff_space.cart

        block_sizes_i1     = [e1-s1+1 for s1, e1 in zip(cart.global_starts[0], cart.global_ends[0])]
        block_sizes_i2     = [e2-s2+1 for s2, e2 in zip(cart.global_starts[1], cart.global_ends[1])]

        block_intervals_i1 = [(s1, e1) for s1, e1 in zip(cart.global_starts[0], cart.global_ends[0])]
        block_intervals_i2 = [(s2, e2) for s2, e2 in zip(cart.global_starts[1], cart.global_ends[1])]

        print('> No. of points along eta1 :: {:d}'.format(cart.npts[0]))
        print('> No. of points along eta2 :: {:d}'.format(cart.npts[1]))
        print('')
        print('> No. of blocks along eta1 :: {:d}'.format(cart.nprocs[0]))
        print('> No. of blocks along eta2 :: {:d}'.format(cart.nprocs[1]))
        print('')
        print('> Block sizes along eta1 :: ' + int_array_to_str(block_sizes_i1))
        print('> Block sizes along eta2 :: ' + int_array_to_str(block_sizes_i2))
        print('')
        print('> Intervals along eta1   :: ' + int_tuples_to_str(block_intervals_i1))
        print('> Intervals along eta2   :: ' + int_tuples_to_str(block_intervals_i2))
        print('', flush=True)
        sleep(0.001)

    mpi_comm.Barrier()
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Analytical and spline mappings
    map_analytic = model.mapping

    if use_spline_mapping:
        map_discrete = SplineMapping.from_mapping(V, map_analytic)
        # Write discrete geometry to HDF5 file
        t0 = time()
        geometry = Geometry.from_discrete_mapping(map_discrete, comm=mpi_comm)
        geometry.export('geo.h5')
        t1 = time()
        timing['export'] += t1-t0
        mapping = map_discrete
    else:
        mapping = map_analytic

    # Build mass and stiffness matrices, and right-hand side vector
    t0 = time()
    M, S = assemble_matrices(V, mapping, kernel, nquads=nquads)
    b  = assemble_rhs(V, mapping, model.rho, nquads=nquads)
    t1 = time()
    timing['assembly'] = t1-t0

    # If required by user, create C1 projector and then restrict
    # stiffness/mass matrices and right-hand-side vector to C1 space
    if c1_correction:
        t0 = time()
        proj = C1Projector(mapping)
        Sp   = proj.change_matrix_basis(S)
        Mp   = proj.change_matrix_basis(M)
        bp   = proj.change_rhs_basis(b)
        t1 = time()
        timing['projection'] = t1-t0

    # Apply homogeneous Dirichlet boundary conditions where appropriate
    # NOTE: this does not effect ghost regions
    if not V1.periodic:
        # left  bc at x=0.
        if not model.O_point and s1 == 0:
            S[s1, :, :, :] = 0.
            S[s1, :, 0, 0] = 1.
            b[s1, :]       = 0.
        # right bc at x=1.
        if e1 == V1.nbasis-1:
            S[e1, :, :, :] = 0.
            S[e1, :, 0, 0] = 1.
            b[e1, :]       = 0.

    if not V2.periodic:
        # lower bc at y=0.
        if s2 == 0:
            S[:, s2, :, :] = 0.
            S[:, s2, 0, 0] = 1.
            b[:, s2]       = 0.
        # upper bc at y=1.
        if e2 == V2.nbasis-1:
            S[:, e2, :, :] = 0.
            S[:, e2, 0, 0] = 1.
            b[:, e2]       = 0.

    if c1_correction and e1 == V1.nbasis-1:
        # only bc is at s=1
        last = bp[1].space.npts[0] - 1
        Sp[1,1][last, :, :, :] = 0.
        Sp[1,1][last, :, 0, 0] = 1.
        bp[1]  [last, :]       = 0.

    # Solve linear system
    t0 = time()
    if c1_correction:
        Sp_inv = inverse(Sp, 'cg', tol=1e-7, maxiter=1000, verbose=False)
        xp     = Sp_inv @ bp
        info   = Sp_inv.get_info()
        x      = proj.convert_to_tensor_basis(xp)
    else:
        S_inv = inverse(S, 'cg', tol=1e-7, maxiter=1000, verbose=False)
        x     = S_inv @ b
        info  = S_inv.get_info()
    t1 = time()
    timing['solution'] = t1-t0

    # Create potential field
    phi = FemField(V, coeffs=x)
    phi.coeffs.update_ghost_regions()

    # Compute L2 norm of error
    t0 = time()
    sqrt_g    = lambda *x: np.sqrt(mapping.metric_det(*x))
    integrand = lambda *x: (phi(*x) - model.phi(*x))**2 * sqrt_g(*x)
    err2 = np.sqrt(V.integral(integrand, nquads=nquads))
    t1 = time()
    timing['diagnostics'] = t1-t0

    # Write solution to HDF5 file
    t0 = time()
    V.export_fields('fields.h5', phi=phi)
    t1 = time()
    timing['export'] += t1-t0

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Print some information to terminal
    for i in range(mpi_size):
        if i == mpi_rank:
            print('--------------------------------------------------' )
            print(' RANK = {}'.format(mpi_rank))
            print('--------------------------------------------------' )
            print('> Grid          :: [{ne1},{ne2}]'.format(ne1=ne1, ne2=ne2))
            print('> Degree        :: [{p1},{p2}]'  .format(p1=p1, p2=p2))
            print('> CG info       :: ', info)
            print('> L2 error      :: {:.2e}'.format(err2))
            print('' )
            print('> Assembly time :: {:.2e}'.format(timing['assembly']))
            if c1_correction:
                print('> Project. time :: {:.2e}'.format( timing['projection']))
            print('> Solution time :: {:.2e}'.format(timing['solution']))
            print('> Evaluat. time :: {:.2e}'.format(timing['diagnostics']))
            print('> Export   time :: {:.2e}'.format(timing['export']))
            print('', flush=True)
            sleep(0.001)
        mpi_comm.Barrier()

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # VISUALIZATION
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    ##########
    N = 10
    ##########

    # Plot domain decomposition (collective call, master creates a figure)
    fig = V.plot_2d_decomposition(mapping, refine=N)
    if fig:
        fig.show()

    # Perform other visualization using master or all processes
    if not distribute_viz:

        # Non-master processes stop here
        if mpi_rank != 0:
            return

        # Create new serial FEM space and mapping (if needed)
        if use_spline_mapping:
            geometry = Geometry(filename='geo.h5', comm=MPI.COMM_SELF)
            map_discrete = [*geometry.mappings.values()].pop()
            Vnew = map_discrete.space
            mapping = map_discrete
        else:
            dd = DomainDecomposition(ncells, model.periodic, comm=MPI.COMM_SELF)
            Vnew = TensorFemSpace(dd, V1, V2)

        # Import solution vector into new serial field
        phi, = Vnew.import_fields( 'fields.h5', 'phi' )

    else:
        Vnew = V

    # Compute numerical solution (and error) on refined logical grid
    [sk1, sk2], [ek1, ek2] = Vnew.local_domain

    eta1 = refine_array_1d(V1.breaks[sk1:ek1+2], N)
    eta2 = refine_array_1d(V2.breaks[sk2:ek2+2], N)
    num = np.array([[      phi(e1, e2) for e2 in eta2] for e1 in eta1])
    ex  = np.array([[model.phi(e1, e2) for e2 in eta2] for e1 in eta1])
    err = num - ex

    # Compute physical coordinates of logical grid
    pcoords = np.array([[model.mapping(e1, e2) for e2 in eta2] for e1 in eta1])
    xx = pcoords[:, :, 0]
    yy = pcoords[:, :, 1]

    # Create figure with 3 subplots:
    #  1. exact solution on exact domain
    #  2. numerical solution on mapped domain (analytical or spline)
    #  3. numerical error    on mapped domain (analytical or spline)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.8))

    def add_colorbar(im, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.2)
        cbar = ax.get_figure().colorbar(im, cax=cax)
        return cbar

    # Plot exact solution
    ax = axes[0]
    im = ax.contourf(xx, yy, ex, 40, cmap='jet')
    add_colorbar(im, ax)
    ax.set_xlabel(r'$x$', rotation='horizontal')
    ax.set_ylabel(r'$y$', rotation='horizontal')
    ax.set_title (r'$\phi_{ex}(x,y)$')
    ax.plot(xx[:, ::N]  , yy[:, ::N]  , 'k')
    ax.plot(xx[::N, :].T, yy[::N, :].T, 'k')
    ax.set_aspect('equal')

    if use_spline_mapping:
        # Recompute physical coordinates of logical grid using spline mapping
        pcoords = np.array([[map_discrete(e1, e2) for e2 in eta2] for e1 in eta1])
        xx = pcoords[:, :, 0]
        yy = pcoords[:, :, 1]

    # Plot numerical solution
    ax = axes[1]
    im = ax.contourf(xx, yy, num, 40, cmap='jet')
    add_colorbar(im, ax)
    ax.set_xlabel(r'$x$', rotation='horizontal')
    ax.set_ylabel(r'$y$', rotation='horizontal')
    ax.set_title (r'$\phi(x,y)$')
    ax.plot(xx[:, ::N]  , yy[:, ::N]  , 'k')
    ax.plot(xx[::N, :].T, yy[::N, :].T, 'k')
    ax.set_aspect('equal')

    # Plot numerical error
    ax = axes[2]
    im = ax.contourf(xx, yy, err, 40, cmap='jet')
    add_colorbar(im, ax)
    ax.set_xlabel(r'$x$', rotation='horizontal')
    ax.set_ylabel(r'$y$', rotation='horizontal')
    ax.set_title (r'$\phi(x,y) - \phi_{ex}(x,y)$')
    ax.plot(xx[:, ::N]  , yy[:, ::N]  , 'k')
    ax.plot(xx[::N, :].T, yy[::N, :].T, 'k')
    ax.set_aspect('equal')

    # Show figure
    fig.show()

    return locals()

#==============================================================================
# Parser
#==============================================================================
def parse_input_arguments():

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class = argparse.HelpFormatter,
        description     = "Solve Poisson's equation on a 2D domain."
    )

    parser.add_argument( '-t',
        type    = str,
        choices =('square', 'annulus', 'circle', 'target', 'czarny'),
        default = 'square',
        dest    = 'test_case',
        help    = 'Test case (default: square)'
    )

    parser.add_argument( '-n',
        type    = int,
        nargs   = 2,
        default = [10, 10],
        metavar = ('N1','N2'),
        dest    = 'ncells',
        help    = 'Number of grid cells (elements) along each dimension (default: [10, 10])'
    )

    parser.add_argument( '-d',
        type    = int,
        nargs   = 2,
        default = [2, 2],
        metavar = ('P1','P2'),
        dest    = 'degree',
        help    = 'Spline degree along each dimension (default: [2, 2])'
    )

    parser.add_argument( '-q',
        type    = int,
        nargs   = 2,
        default = None,
        metavar = ('Q1','Q2'),
        dest    = 'nquads',
        help    = 'Number of quadrature points along each dimension (default: [P1+1, P2+1])'
    )

    parser.add_argument( '-s',
        action  = 'store_true',
        dest    = 'use_spline_mapping',
        help    = 'Use spline mapping in finite element calculations (default: False)'
    )

    parser.add_argument( '-c',
        action  = 'store_true',
        dest    = 'c1_correction',
        help    = 'Apply C1 correction at polar singularity (O point) (default: False)'
    )

    parser.add_argument( '--distribute_viz',
        action  = 'store_true',
        dest    = 'distribute_viz',
        help    = 'Create separate plots for each subdomain (default: False)'
    )

    return parser.parse_args()

#==============================================================================
# Script functionality
#==============================================================================
if __name__ == '__main__':

    args = parse_input_arguments()
    namespace = main(**vars(args))

    import __main__
    if hasattr(__main__, '__file__'):
        try:
           __IPYTHON__
        except NameError:
            import matplotlib.pyplot as plt
            plt.show()
