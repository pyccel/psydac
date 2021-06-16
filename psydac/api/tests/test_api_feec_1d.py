# coding: utf-8
# Copyright 2020 Yaman Güçlü

#==============================================================================
# TIME STEPPING METHOD
#==============================================================================
def step_faraday_1d(dt, e, b, M0, M1, D0, D0_T, **kwargs):
    """
    Exactly integrate the semi-discrete Faraday equation over one time-step:

    b_new = b - ∆t D0 e

    """
    b -= dt * D0.dot(e)
  # e += 0

def step_ampere_1d(dt, e, b, M0, M1, D0, D0_T, *, pc=None, tol=1e-7, verbose=False):
    """
    Exactly integrate the semi-discrete Amperè equation over one time-step:

    e_new = e + ∆t (M0^{-1} D0^T M1) b

    """
    options = dict(tol=tol, verbose=verbose)
    if pc:
        from psydac.linalg.iterative_solvers import pcg as isolve
        options['pc'] = pc
    else:
        from psydac.linalg.iterative_solvers import cg as isolve

  # b += 0
    e += dt * isolve(M0, D0_T.dot((M1.dot(b))), **options)[0]

#==============================================================================
# VISUALIZATION
#==============================================================================
def make_plot(ax, t, sol_ex, sol_num, x, xlim, label):
    ax.plot(x, sol_ex , '--', label='exact')
    ax.plot(x, sol_num, '-' , label='numerical')
    ax.legend(loc='upper right')
    ax.grid()
    ax.set_title('Time t = {:10.3e}'.format(t))
    ax.set_xlabel('x')
    ax.set_ylabel(label, rotation='horizontal')
    ax.set_xlim(xlim)

def update_plot(ax, t, sol_ex, sol_num):
    ax.set_title('Time t = {:10.3e}'.format(t))
    ax.lines[0].set_ydata(sol_ex )
    ax.lines[1].set_ydata(sol_num)
    ax.get_figure().canvas.draw()

#==============================================================================
# SIMULATION
#==============================================================================
def run_maxwell_1d(*, L, eps, ncells, degree, periodic, Cp, nsteps, tend,
        splitting_order, plot_interval, diagnostics_interval,
        bc_mode, tol, verbose):

    import numpy as np
    import matplotlib.pyplot as plt
    from mpi4py          import MPI
    from scipy.integrate import quad

    from sympde.topology import Mapping
    from sympde.topology import Line
    from sympde.topology import Derham
    from sympde.topology import elements_of
    from sympde.expr     import integral
    from sympde.expr     import BilinearForm

    from psydac.api.discretization import discretize
    from psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL
    from psydac.feec.pull_push     import push_1d_l2

    #--------------------------------------------------------------------------
    # Analytical objects: SymPDE
    #--------------------------------------------------------------------------

    # Logical domain: interval (0, 1)
    logical_domain = Line('Omega', bounds=(0, 1))

    #... Mapping and physical domain
    class CollelaMapping1D(Mapping):

        _expressions = {'x': 'k * (x1 + eps / (2*pi) * sin(2*pi*x1))'}
        _ldim = 1
        _pdim = 1

    mapping = CollelaMapping1D('M', k=L, eps=eps)
    domain  = mapping(logical_domain)
    #...

    # Exact solution
    if periodic:
        g    = lambda w: np.exp(-(w/0.1)**2)        # Gaussian waveform
        wr   = lambda t, x: (x-t) % L - L/2         # Right-traveling wave, L-periodic
        E_ex = lambda t, x: g(wr(t,x))              # Exact solution in periodic domain
        B_ex = lambda t, x: g(wr(t,x))              # Exact solution in periodic domain
    else:
        g    = lambda w: np.exp(-(w/0.1)**2)        # Gaussian waveform
        wr   = lambda t, x: (x-t+L/2) % (2*L) - L   # Right-traveling wave, (2L)-periodic
        wl   = lambda t, x: (x+t-L/2) % (2*L) - L   #  Left-traveling wave, (2L)-periodic
        E_ex = lambda t, x: g(wr(t,x)) - g(wl(t,x)) # Exact solution in bounded domain
        B_ex = lambda t, x: g(wr(t,x)) + g(wl(t,x)) # Exact solution in bounded domain

    # DeRham sequence
    derham = Derham(domain)

    # Trial and test functions
    u0, v0 = elements_of(derham.V0, names='u0, v0')
    u1, v1 = elements_of(derham.V1, names='u1, v1')

    # Bilinear forms that correspond to mass matrices for spaces V0 and V1
    a0 = BilinearForm((u0, v0), integral(domain, u0 * v0))
    a1 = BilinearForm((u1, v1), integral(domain, u1 * v1))

    # ...
    # If needed, apply homogeneous Dirichlet BCs
    if not periodic:

        # Option 1: Apply essential BCs to elements of V0 space
        if bc_mode == 'strong':
            from sympde.expr import EssentialBC
            bcs = [EssentialBC(u0, 0, side) for side in domain.boundary]

        # Option 2: Penalize L2 projection to V0 space
        elif bc_mode == 'penalization':
            a0_bc = BilinearForm((u0, v0), integral(domain.boundary, 1e30 * u0 * v0))

        else:
            NotImplementedError('bc_mode = {}'.format(bc_mode))
    # ...

    #--------------------------------------------------------------------------
    # Discrete objects: Psydac
    #--------------------------------------------------------------------------

    # Discrete physical domain and discrete DeRham sequence
    domain_h = discretize(domain, ncells=[ncells], comm=MPI.COMM_WORLD)
    derham_h = discretize(derham, domain_h, degree=[degree], periodic=[periodic])

    # Discrete bilinear forms
    a0_h = discretize(a0, domain_h, (derham_h.V0, derham_h.V0), backend=PSYDAC_BACKEND_GPYCCEL)
    a1_h = discretize(a1, domain_h, (derham_h.V1, derham_h.V1), backend=PSYDAC_BACKEND_GPYCCEL)

    # Mass matrices (StencilMatrix objects)
    M0 = a0_h.assemble()
    M1 = a1_h.assemble()

    # Differential operators
    D0, = derham_h.derivatives_as_matrices

    # Transpose of derivative matrix
    D0_T = D0.T

    # Boundary conditions
    if not periodic:

        # Option 1: Modify operators to V0h space: mass matrix M0, differentiation matrix D0^T
        if bc_mode == 'strong':
            from psydac.api.essential_bc import apply_essential_bc
            M0_dir   = M0.copy()
            D0_T_dir = D0_T.tokronstencil().tostencil().copy()
            apply_essential_bc(  M0_dir, *bcs)
            apply_essential_bc(D0_T_dir, *bcs)

            # Make sure that we have ones on the diagonal of the mass matrix,
            # in order to use a Jacobi preconditioner
            s, = M0.codomain.starts
            e, = M0.codomain.ends
            n, = M0.codomain.npts
            if s == 0:
                M0_dir[s, 0] = 1.0
            if e + 1 == n:
                M0_dir[e, 0] = 1.0

        # Option 2: Discretize and assemble penalization matrix
        elif bc_mode == 'penalization':
            a0_bc_h = discretize(a0_bc, domain_h, (derham_h.V0, derham_h.V0), backend=PSYDAC_BACKEND_GPYCCEL)
            M0_bc   = a0_bc_h.assemble()

    # Projectors
    P0, P1 = derham_h.projectors(nquads=[degree+2])

    # Logical and physical grids
    F = mapping.get_callable_mapping()
    grid_x1 = derham_h.V0.breaks[0]
    grid_x  = F(grid_x1)[0]

    xmin = grid_x[ 0]
    xmax = grid_x[-1]

    #--------------------------------------------------------------------------
    # Time integration setup
    #--------------------------------------------------------------------------

    t = 0

    # Initial conditions, discrete fields
    E = P0(lambda x: E_ex(0, x))
    B = P1(lambda x: B_ex(0, x))

    # Initial conditions, spline coefficients
    e = E.coeffs
    b = B.coeffs

    # Time step size
    dx_min = min(np.diff(grid_x))
    dt = Cp * dx_min

    # If final time is given, compute number of time steps
    if tend is not None:
        nsteps = int(np.ceil(tend / dt))

    #--------------------------------------------------------------------------
    # Scalar diagnostics setup
    #--------------------------------------------------------------------------

    # Energy of exact solution
    def exact_energies(t, E_ex, B_ex):
        """ Compute electric & magnetic energies of exact solution.
        """
        We = 0.5 * quad(lambda x: E_ex(t, x)**2, xmin, xmax)[0]
        Wb = 0.5 * quad(lambda x: B_ex(t, x)**2, xmin, xmax)[0]
        return (We, Wb)

    # Energy of numerical solution
    def discrete_energies(e, b):
        """ Compute electric & magnetic energies of numerical solution.
        """
        We = 0.5 * M0.dot(e).dot(e)
        Wb = 0.5 * M1.dot(b).dot(b)
        return (We, Wb)

    # Scalar diagnostics:
    diagnostics_ex  = {'time': [], 'electric_energy': [], 'magnetic_energy': []}
    diagnostics_num = {'time': [], 'electric_energy': [], 'magnetic_energy': []}

    #--------------------------------------------------------------------------
    # Visualization and diagnostics setup
    #--------------------------------------------------------------------------

    # Very fine grids for evaluation of solution
    x1 = np.linspace(grid_x1[0], grid_x1[-1], 101)
    x  = F(x1)[0]

    # Prepare plots
    if plot_interval:

        # Plot physical grid
        fig1, ax1 = plt.subplots(2, 1, figsize=(6, 6))
        ax1[0].set_ylim(-1, 1)
        ax1[0].plot([xmin, xmax], [0, 0], 'orange')
        ax1[0].plot(grid_x, np.zeros_like(grid_x), 'o')
        ax1[0].set_title('Mapped grid obtained from uniform logical grid of {} cells'.format(ncells))
        ax1[0].set_xlabel('x', fontsize=14)
        ax1[0].yaxis.set_visible(False)

        # Plot derivative of mapping
        ax1[1].plot(x1, F.jacobian(x1)[0, 0], '-')
        ax1[1].grid()
        ax1[1].set_title(r'Derivative of mapping $F$ w.r.t. logical coordinate $x_1$')
        ax1[1].set_xlabel(r'$x_1$', size=14)
        ax1[1].set_ylabel(r"$F'\left(x_1\right)$", size=14)

        fig1.tight_layout()
        fig1.canvas.draw()
        fig1.show()

        # ...
        # Prepare animations
        E_values = [E(xi) for xi in x1]
        B_values = push_1d_l2(lambda x1: np.array([B(xi) for xi in x1]), x1, mapping)

        fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))
        make_plot(ax2[0], t, E_ex(0, x), E_values, x, [xmin, xmax], label='E')
        make_plot(ax2[1], t, B_ex(0, x), B_values, x, [xmin, xmax], label='B')

        ylim = (-0.2, 1.2) if periodic else (-1, 2)
        for ax in ax2:
            ax.set_ylim(*ylim)
            ax.plot(grid_x, np.zeros_like(grid_x), 'ok', mfc='None', ms=5)

        fig2.tight_layout()
        fig2.canvas.draw()
        fig2.show()
        # ...

        input('\nSimulation setup done... press any key to start')

    # Prepare diagnostics
    if diagnostics_interval:

        # Exact energy at t=0
        We_ex, Wb_ex = exact_energies(t, E_ex, B_ex)
        diagnostics_ex['time'].append(t)
        diagnostics_ex['electric_energy'].append(We_ex)
        diagnostics_ex['magnetic_energy'].append(Wb_ex)

        # Discrete energy at t=0
        We_num, Wb_num = discrete_energies(e, b)
        diagnostics_num['time'].append(t)
        diagnostics_num['electric_energy'].append(We_num)
        diagnostics_num['magnetic_energy'].append(Wb_num)

        print('\nTotal energy in domain:')
        print('t = {:8.4f},  exact = {Wt_ex:.13e},  discrete = {Wt_num:.13e}'.format(t,
            Wt_ex  = We_ex  + Wb_ex,
            Wt_num = We_num + Wb_num)
        )

    #--------------------------------------------------------------------------
    # Solution
    #--------------------------------------------------------------------------

    # TODO: add option to convert to scipy sparse format

    # ... Arguments for time stepping
    kwargs = {'verbose': verbose, 'tol': tol}

    if periodic:
        args = (e, b, M0, M1, D0, D0_T)

    elif bc_mode == 'strong':
        args = (e, b, M0_dir, M1, D0, D0_T_dir)

    elif bc_mode == 'penalization':
        args = (e, b, M0 + M0_bc, M1, D0, D0_T)
        kwargs['pc'] = 'jacobi'
    # ...

    # Time loop
    for i in range(nsteps):

        # TODO: allow for high-order splitting

        # Strang splitting, 2nd order
        step_faraday_1d(0.5*dt, *args, **kwargs)
        step_ampere_1d (    dt, *args, **kwargs)
        step_faraday_1d(0.5*dt, *args, **kwargs)

        t += dt

        # Animation
        if plot_interval and (i % plot_interval == 0 or i == nsteps-1):

            E_values = [E(xi) for xi in x1]
            B_values = push_1d_l2(lambda x1: np.array([B(xi) for xi in x1]), x1, mapping)

            # Update plot
            update_plot(ax2[0], t, E_ex(t, x), E_values)
            update_plot(ax2[1], t, B_ex(t, x), B_values)
            plt.pause(0.01)

        # Scalar diagnostics
        if diagnostics_interval and i % diagnostics_interval == 0:

            # Update exact diagnostics
            We_ex, Wb_ex = exact_energies(t, E_ex, B_ex)
            diagnostics_ex['time'].append(t)
            diagnostics_ex['electric_energy'].append(We_ex)
            diagnostics_ex['magnetic_energy'].append(Wb_ex)

            # Update numerical diagnostics
            We_num, Wb_num = discrete_energies(e, b)
            diagnostics_num['time'].append(t)
            diagnostics_num['electric_energy'].append(We_num)
            diagnostics_num['magnetic_energy'].append(Wb_num)

            # Print total energy to terminal
            print('t = {:8.4f},  exact = {Wt_ex:.13e},  discrete = {Wt_num:.13e}'.format(t,
                Wt_ex  = We_ex  + Wb_ex,
                Wt_num = We_num + Wb_num)
            )

    #--------------------------------------------------------------------------
    # Post-processing
    #--------------------------------------------------------------------------

    # Error at final time
    E_values = np.array([E(xi) for xi in x1])
    B_values = push_1d_l2(lambda x1: np.array([B(xi) for xi in x1]), x1, mapping)

    error_E = max(abs(E_ex(t, x) - E_values))
    error_B = max(abs(B_ex(t, x) - B_values))
    print()
    print('Max-norm of error on E(t,x) at final time: {:.2e}'.format(error_E))
    print('Max-norm of error on B(t,x) at final time: {:.2e}'.format(error_B))

    if diagnostics_interval:

        # Extract exact diagnostics
        t_ex  = np.asarray(diagnostics_ex['time'])
        We_ex = np.asarray(diagnostics_ex['electric_energy'])
        Wb_ex = np.asarray(diagnostics_ex['magnetic_energy'])
        Wt_ex = We_ex + Wb_ex

        # Extract numerical diagnostics
        t_num  = np.asarray(diagnostics_num['time'])
        We_num = np.asarray(diagnostics_num['electric_energy'])
        Wb_num = np.asarray(diagnostics_num['magnetic_energy'])
        Wt_num = We_num + Wb_num

        # Energy plots
        fig3, (ax31, ax32, ax33) = plt.subplots(3, 1, figsize=(12, 10))
        #
        ax31.set_title('Energy of exact solution')
        ax31.plot(t_ex, We_ex, label='electric')
        ax31.plot(t_ex, Wb_ex, label='magnetic')
        ax31.plot(t_ex, Wt_ex, label='total'   )
        ax31.legend()
        ax31.set_xlabel('t')
        ax31.set_ylabel('W', rotation='horizontal')
        ax31.grid()
        #
        ax32.set_title('Energy of numerical solution')
        ax32.plot(t_num, We_num, label='electric')
        ax32.plot(t_num, Wb_num, label='magnetic')
        ax32.plot(t_num, Wt_num, label='total'   )
        ax32.legend()
        ax32.set_xlabel('t')
        ax32.set_ylabel('W', rotation='horizontal')
        ax32.grid()
        #
        ax33.set_title('Relative error in total energy')
        ax33.plot(t_ex , (Wt_ex  - Wt_ex) / Wt_ex[0], '--', label='exact')
        ax33.plot(t_num, (Wt_num - Wt_ex) / Wt_ex[0], '-' , label='numerical')
        ax33.legend()
        ax33.set_xlabel('t')
        ax33.set_ylabel('(W - W_ex) / W_ex(t=0)')
        ax33.grid()
        #
        fig3.tight_layout()
        fig3.show()

    # Return whole namespace as dictionary
    return locals()

#==============================================================================
# UNIT TESTS
#==============================================================================

def test_maxwell_1d_periodic():

    namespace = run_maxwell_1d(
        L        = 1.0,
        eps      = 0.5,
        ncells   = 30,
        degree   = 3,
        periodic = True,
        Cp       = 0.5,
        nsteps   = 1,
        tend     = None,
        splitting_order      = 2,
        plot_interval        = 0,
        diagnostics_interval = 0,
        tol = 1e-6,
        bc_mode = None,
        verbose = False
    )

    TOL = 1e-6
    ref = dict(error_E = 4.191954319623381e-04,
               error_B = 4.447074070748624e-04)

    assert abs(namespace['error_E'] - ref['error_E']) / ref['error_E'] <= TOL
    assert abs(namespace['error_B'] - ref['error_B']) / ref['error_B'] <= TOL


def test_maxwell_1d_dirichlet_strong():

    namespace = run_maxwell_1d(
        L        = 1.0,
        eps      = 0.5,
        ncells   = 20,
        degree   = 5,
        periodic = False,
        Cp       = 0.5,
        nsteps   = 1,
        tend     = None,
        splitting_order      = 2,
        plot_interval        = 0,
        diagnostics_interval = 0,
        tol = 1e-6,
        bc_mode = 'strong',
        verbose = False
    )

    TOL = 1e-6
    ref = dict(error_E = 1.320471502738063e-03,
               error_B = 7.453774187340390e-04)

    assert abs(namespace['error_E'] - ref['error_E']) / ref['error_E'] <= TOL
    assert abs(namespace['error_B'] - ref['error_B']) / ref['error_B'] <= TOL


def test_maxwell_1d_dirichlet_penalization():

    namespace = run_maxwell_1d(
        L        = 1.0,
        eps      = 0.5,
        ncells   = 20,
        degree   = 5,
        periodic = False,
        Cp       = 0.5,
        nsteps   = 1,
        tend     = None,
        splitting_order      = 2,
        plot_interval        = 0,
        diagnostics_interval = 0,
        tol = 1e-6,
        bc_mode = 'penalization',
        verbose = False
    )

    TOL = 1e-6
    ref = dict(error_E = 1.320290052669426e-03,
               error_B = 7.453277842247585e-04)

    assert abs(namespace['error_E'] - ref['error_E']) / ref['error_E'] <= TOL
    assert abs(namespace['error_B'] - ref['error_B']) / ref['error_B'] <= TOL


#==============================================================================
# SCRIPT CAPABILITIES
#==============================================================================
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        description     = "Solve 1D Maxwell equations with spline FEEC method."
    )

    parser.add_argument('ncells',
        type = int,
        help = 'Number of cells in domain'
    )

    parser.add_argument('degree',
        type = int,
        help = 'Polynomial spline degree'
    )

    parser.add_argument( '-P', '--periodic',
        action  = 'store_true',
        help    = 'Use periodic boundary conditions'
    )

    parser.add_argument('-o', '--splitting_order',
        type    = int,
        default = 2,
        choices = [2, 4, 6],
        help    = 'Order of accuracy of operator splitting'
    )

    parser.add_argument( '-l',
        type    = float,
        default = 1.0,
        dest    = 'L',
        metavar = 'L',
        help    = 'Length of domain [0, L]'
    )

    parser.add_argument( '-e',
        type    = float,
        default = 0.25,
        dest    = 'eps',
        metavar = 'EPS',
        help    = 'Deformation level (0 <= EPS < 1)'
    )

    parser.add_argument( '-c',
        type    = float,
        default = 0.5,
        dest    = 'Cp',
        metavar = 'Cp',
        help    = 'Courant parameter on uniform grid'
    )

    # ...
    time_opts = parser.add_mutually_exclusive_group()
    time_opts.add_argument( '-t',
        type    = int,
        default = 1,
        dest    = 'nsteps',
        metavar = 'NSTEPS',
        help    = 'Number of time-steps to be taken'
    )
    time_opts.add_argument( '-T',
        type    = float,
        dest    = 'tend',
        metavar = 'END_TIME',
        help    = 'Run simulation until given final time'
    )
    # ...

    parser.add_argument( '-p',
        type    = int,
        default = 4,
        metavar = 'I',
        dest    = 'plot_interval',
        help    = 'No. of time steps between successive plots of solution, if I=0 no plots are made'
    )

    parser.add_argument( '-d',
        type    = int,
        default = 1,
        metavar = 'I',
        dest    = 'diagnostics_interval',
        help    = 'No. of time steps between successive calculations of scalar diagnostics, if I=0 no diagnostics are computed'
    )

    parser.add_argument( '-v', '--verbose',
        action  = 'store_true',
        help    = 'Print convergence information of iterative solver'
    )

    parser.add_argument( '--tol',
        type    = float,
        default = 1e-7,
        help    = 'Tolerance for iterative solver (L2-norm of residual)'
    )

    parser.add_argument( '--bc_mode',
        choices = ['strong', 'penalization'],
        default = 'strong',
        help    = 'Strategy for imposing Dirichlet BCs'
    )

    # Read input arguments
    args = parser.parse_args()

    # Run simulation
    namespace = run_maxwell_1d(**vars(args))

    # Keep matplotlib windows open
    import matplotlib.pyplot as plt
    plt.show()
