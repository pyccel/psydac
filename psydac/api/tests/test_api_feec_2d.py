# coding: utf-8
# Copyright 2021 Yaman Güçlü

#==============================================================================
# TIME STEPPING METHOD
#==============================================================================
def step_faraday_2d(dt, e, b, M1, M2, D1, D1_T, **kwargs):
    """
    Exactly integrate the semi-discrete Faraday equation over one time-step:

    b_new = b - ∆t D1 e

    """
    b -= dt * D1.dot(e)
  # e += 0

def step_ampere_2d(dt, e, b, M1, M2, D1, D1_T, *, pc=None, tol=1e-7, verbose=False):
    """
    Exactly integrate the semi-discrete Amperè equation over one time-step:

    e_new = e - ∆t (M1^{-1} D1^T M2) b

    """
    options = dict(tol=tol, verbose=verbose)
    if pc:
        from psydac.linalg.iterative_solvers import pcg as isolve
        options['pc'] = pc
    else:
        from psydac.linalg.iterative_solvers import cg as isolve

  # b += 0
    e += dt * isolve(M1, D1_T.dot(M2.dot(b)), **options)[0]

#==============================================================================
# ANALYTICAL SOLUTION
#==============================================================================
class CavitySolution:
    """
    Time-harmonic solution of Maxwell's equations in a rectangular cavity with
    perfectly conducting walls. This is a "transverse electric" solution, with
    E = (Ex, Ey) and B = Bz. Domain is [0, a] x [0, b].

    Parameters
    ----------
    a : float
        Size of cavity along x direction.

    b : float
        Size of cavity along y direction.

    c : float
        Speed of light in arbitrary units.

    nx : int
        Number of half wavelengths along x direction.

    ny : int
        Number of half wavelengths along y direction.

    """
    def __init__(self, *, a, b, c, nx, ny):

        from sympy import symbols
        from sympy import lambdify

        sym_params, sym_fields, sym_energy = self.symbolic()

        params = {'a': a, 'b': b, 'c': c, 'nx': nx, 'ny': ny}
        repl = [(sym_params[k], params[k]) for k in sym_params.keys()]
        args = symbols('t, x, y')

        # Callable functions
        fields = {k: lambdify(args   , v.subs(repl), 'numpy') for k, v in sym_fields.items()}
        energy = {k: lambdify(args[0], v.subs(repl), 'numpy') for k, v in sym_energy.items()}

        # Store private attributes
        self._sym_params = sym_params
        self._sym_fields = sym_fields
        self._sym_energy = sym_energy

        self._params = params
        self._fields = fields
        self._energy = energy

    #--------------------------------------------------------------------------
    @staticmethod
    def symbolic():

        from sympy import symbols
        from sympy import cos, sin, pi, sqrt
        from sympy.integrals import integrate

        t, x, y = symbols('t x y', real=True)
        a, b, c = symbols('a b c', positive=True)
        nx, ny  = symbols('nx ny', positive=True, integer=True)

        kx = pi * nx / a
        ky = pi * ny / b
        omega = c * sqrt(kx**2 + ky**2)

        # Exact solutions for electric and magnetic field
        Ex =  cos(kx * x) * sin(ky * y) * cos(omega * t)
        Ey = -sin(kx * x) * cos(ky * y) * cos(omega * t)
        Bz =  cos(kx * x) * cos(ky * y) * sin(omega * t) * (kx + ky) / omega

        # Electric and magnetic energy in domain
        We = integrate(integrate((Ex**2 + Ey**2)/ 2, (x, 0, a)), (y, 0, b)).simplify()
        Wb = integrate(integrate(         Bz**2 / 2, (x, 0, a)), (y, 0, b)).simplify()

        params = {'a': a, 'b': b, 'c': c, 'nx': nx, 'ny': ny}
        fields = {'Ex': Ex, 'Ey': Ey, 'Bz': Bz}
        energy = {'We': We, 'Wb': Wb}

        return params, fields, energy

    #--------------------------------------------------------------------------
    @property
    def params(self):
        return self._params

    @property
    def fields(self):
        return self._fields

    @property
    def energy(self):
        return self._energy

    @property
    def derived_params(self):
        from numpy import pi, sqrt
        kx    = pi * self.params['nx'] / self.params['a']
        ky    = pi * self.params['ny'] / self.params['b']
        omega = self.params['c'] * sqrt(kx**2 + ky**2)
        return {'kx': kx, 'ky' : ky, 'omega': omega}

    @property
    def sym_params(self):
        return self._sym_params

    @property
    def sym_fields(self):
        return self._sym_fields

    @property
    def sym_energy(self):
        return self._sym_energy

#==============================================================================
# VISUALIZATION
#==============================================================================

def add_colorbar(im, ax, **kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.2, pad=0.3)
    cbar = ax.get_figure().colorbar(im, cax=cax, **kwargs)
    return cbar

def plot_field_and_error(name, x, y, field_h, field_ex, *gridlines):
    import matplotlib.pyplot as plt
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 6))
    im0 = ax0.contourf(x, y, field_h)
    im1 = ax1.contourf(x, y, field_ex - field_h)
    ax0.set_title(r'${0}_h$'.format(name))
    ax1.set_title(r'${0} - {0}_h$'.format(name))
    for ax in (ax0, ax1):
        ax.plot(*gridlines[0], color='k')
        ax.plot(*gridlines[1], color='k')
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14, rotation='horizontal')
        ax.set_aspect('equal')
    add_colorbar(im0, ax0)
    add_colorbar(im1, ax1)
    fig.suptitle('Time t = {:10.3e}'.format(0))
    fig.tight_layout()
    return fig

def update_plot(fig, t, x, y, field_h, field_ex):
    ax0, ax1, cax0, cax1 = fig.axes
    ax0.collections.clear(); cax0.clear()
    ax1.collections.clear(); cax1.clear()
    im0 = ax0.contourf(x, y, field_h)
    im1 = ax1.contourf(x, y, field_ex - field_h)
    fig.colorbar(im0, cax=cax0)
    fig.colorbar(im1, cax=cax1)
    fig.suptitle('Time t = {:10.3e}'.format(t))
    fig.canvas.draw()

#==============================================================================
# SIMULATION
#==============================================================================
def run_maxwell_2d_TE(*, eps, ncells, degree, periodic, Cp, nsteps, tend,
        splitting_order, plot_interval, diagnostics_interval, tol, verbose):

    import numpy as np
    import matplotlib.pyplot as plt
    from mpi4py          import MPI
    from scipy.integrate import dblquad

    from sympde.topology import Square
    from sympde.topology import Mapping
#    from sympde.topology import CollelaMapping2D
    from sympde.topology import Derham
    from sympde.topology import elements_of
    from sympde.topology import NormalVector
    from sympde.calculus import dot, cross
    from sympde.expr     import integral
    from sympde.expr     import BilinearForm

    from psydac.api.discretization import discretize
    from psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL
    from psydac.feec.pull_push     import push_2d_hcurl, push_2d_l2
    from psydac.utilities.utils    import refine_array_1d

    #--------------------------------------------------------------------------
    # Problem setup
    #--------------------------------------------------------------------------

    # Physical domain is rectangle [0, a] x [0, b]
    a = 2.0
    b = 2.0

    # Speed of light is 1
    c = 1.0

    # Mode number
    (nx, ny) = (2, 2)

    # Exact solution
    exact_solution = CavitySolution(a=a, b=b, c=c, nx=nx, ny=ny)

    # Exact fields, as callable functions of (t, x, y)
    Ex_ex = exact_solution.fields['Ex']
    Ey_ex = exact_solution.fields['Ey']
    Bz_ex = exact_solution.fields['Bz']

    #...

    #--------------------------------------------------------------------------
    # Analytical objects: SymPDE
    #--------------------------------------------------------------------------

    # Logical domain is unit square [0, 1] x [0, 1]
    logical_domain = Square('Omega')

    # Mapping and physical domain
    class CollelaMapping2D(Mapping):

        _ldim = 2
        _pdim = 2
        _expressions = {'x': 'a * (x1 + eps / (2*pi) * sin(2*pi*x1) * sin(2*pi*x2))',
                        'y': 'b * (x2 + eps / (2*pi) * sin(2*pi*x1) * sin(2*pi*x2))'}

#    mapping = CollelaMapping2D('M', k1=1, k2=1, eps=eps)
    mapping = CollelaMapping2D('M', a=a, b=b, eps=eps)
    domain  = mapping(logical_domain)

    # DeRham sequence
    derham = Derham(domain, sequence=['h1', 'hcurl', 'l2'])

    # Trial and test functions
    u1, v1 = elements_of(derham.V1, names='u1, v1')  # electric field E = (Ex, Ey)
    u2, v2 = elements_of(derham.V2, names='u2, v2')  # magnetic field Bz

    # Bilinear forms that correspond to mass matrices for spaces V1 and V2
    a1 = BilinearForm((u1, v1), integral(domain, dot(u1, v1)))
    a2 = BilinearForm((u2, v2), integral(domain, u2 * v2))

    # If needed, use penalization to apply homogeneous Dirichlet BCs
    if not periodic:
        nn = NormalVector('nn')
        a1_bc = BilinearForm((u1, v1),
                integral(domain.boundary, 1e30 * cross(u1, nn) * cross(v1, nn)))

    #--------------------------------------------------------------------------
    # Discrete objects: Psydac
    #--------------------------------------------------------------------------

    # Discrete physical domain and discrete DeRham sequence
    domain_h = discretize(domain, ncells=[ncells, ncells], comm=MPI.COMM_WORLD)
    derham_h = discretize(derham, domain_h, degree=[degree, degree], periodic=[periodic, periodic])

    # Discrete bilinear forms
    a1_h = discretize(a1, domain_h, (derham_h.V1, derham_h.V1), backend=PSYDAC_BACKEND_GPYCCEL)
    a2_h = discretize(a2, domain_h, (derham_h.V2, derham_h.V2), backend=PSYDAC_BACKEND_GPYCCEL)

    # Mass matrices (StencilMatrix objects)
    M1 = a1_h.assemble()
    M2 = a2_h.assemble()

    # Differential operators
    D0, D1 = derham_h.derivatives_as_matrices

    # Discretize and assemble penalization matrix
    if not periodic:
        a1_bc_h = discretize(a1_bc, domain_h, (derham_h.V1, derham_h.V1), backend=PSYDAC_BACKEND_GPYCCEL)
        M1_bc   = a1_bc_h.assemble()

    # Transpose of derivative matrix
    D1_T = D1.T

    # Projectors
    P0, P1, P2 = derham_h.projectors(nquads=[degree+2, degree+2])

    # Logical and physical grids
    F = mapping.get_callable_mapping()
    grid_x1 = derham_h.V0.breaks[0]
    grid_x2 = derham_h.V0.breaks[1]

    grid_x, grid_y = F(*np.meshgrid(grid_x1, grid_x2, indexing='ij'))

    #--------------------------------------------------------------------------
    # Time integration setup
    #--------------------------------------------------------------------------

    t = 0

    # Initial conditions, discrete fields
    E = P1((lambda x, y: Ex_ex(0, x, y), lambda x, y: Ey_ex(0, x, y)))
    B = P2(lambda x, y: Bz_ex(0, x, y))

    # Initial conditions, spline coefficients
    e = E.coeffs
    b = B.coeffs

    # Time step size
    dx_min_1 = np.sqrt(np.diff(grid_x, axis=0)**2 + np.diff(grid_y, axis=0)**2).min()
    dx_min_2 = np.sqrt(np.diff(grid_x, axis=1)**2 + np.diff(grid_y, axis=1)**2).min()

    dx_min = min(dx_min_1, dx_min_2)
    dt = Cp * dx_min / c

    # If final time is given, compute number of time steps
    if tend is not None:
        nsteps = int(np.ceil(tend / dt))

    #--------------------------------------------------------------------------
    # Scalar diagnostics setup
    #--------------------------------------------------------------------------

    # Energy of exact solution
    def exact_energies(t):
        """ Compute electric & magnetic energies of exact solution.
        """
        We = exact_solution.energy['We'](t)
        Wb = exact_solution.energy['Wb'](t)
        return (We, Wb)

    # Energy of numerical solution
    def discrete_energies(e, b):
        """ Compute electric & magnetic energies of numerical solution.
        """
        We = 0.5 * M1.dot(e).dot(e)
        Wb = 0.5 * M2.dot(b).dot(b)
        return (We, Wb)

    # Scalar diagnostics:
    diagnostics_ex  = {'time': [], 'electric_energy': [], 'magnetic_energy': []}
    diagnostics_num = {'time': [], 'electric_energy': [], 'magnetic_energy': []}

    #--------------------------------------------------------------------------
    # Visualization and diagnostics setup
    #--------------------------------------------------------------------------

    # Very fine grids for evaluation of solution
    N = 5
    x1 = refine_array_1d(grid_x1, N)
    x2 = refine_array_1d(grid_x2, N)

    x1, x2 = np.meshgrid(x1, x2, indexing='ij')
    x, y = F(x1, x2)

    gridlines_x1 = (x[:, ::N],   y[:, ::N]  )
    gridlines_x2 = (x[::N, :].T, y[::N, :].T)
    gridlines = (gridlines_x1, gridlines_x2)

    Ex_values = np.empty_like(x1)
    Ey_values = np.empty_like(x1)
    Bz_values = np.empty_like(x1)

    # Prepare plots
    if plot_interval:

        # Plot physical grid and mapping's metric determinant
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        im = ax1.contourf(x, y, np.sqrt(F.metric_det(x1, x2)))
        add_colorbar(im, ax1, label=r'Metric determinant $\sqrt{g}$ of mapping $F$')
        ax1.plot(*gridlines_x1, color='k')
        ax1.plot(*gridlines_x2, color='k')
        ax1.set_title('Mapped grid of {} x {} cells'.format(ncells, ncells))
        ax1.set_xlabel('x', fontsize=14)
        ax1.set_ylabel('y', fontsize=14)
        ax1.set_aspect('equal')
        fig1.tight_layout()
        fig1.show()

        # ...
        # Plot initial conditions
        # TODO: improve
        for i, x1i in enumerate(x1[:, 0]):
            for j, x2j in enumerate(x2[0, :]):

                Ex_values[i, j], Ey_values[i, j] = \
                        push_2d_hcurl(E.fields[0], E.fields[1], x1i, x2j, mapping)

                Bz_values[i, j] = push_2d_l2(B, x1i, x2j, mapping)

        # Electric field, x component
        fig2 = plot_field_and_error(r'E^x', x, y, Ex_values, Ex_ex(0, x, y), *gridlines)
        fig2.show()                                             
                                                                
        # Electric field, y component                           
        fig3 = plot_field_and_error(r'E^y', x, y, Ey_values, Ey_ex(0, x, y), *gridlines)
        fig3.show()                                             
                                                                
        # Magnetic field, z component                           
        fig4 = plot_field_and_error(r'B^z', x, y, Bz_values, Bz_ex(0, x, y), *gridlines)
        fig4.show()
        # ...

        input('\nSimulation setup done... press any key to start')

    # Prepare diagnostics
    if diagnostics_interval:

        # Exact energy at t=0
        We_ex, Wb_ex = exact_energies(t)
        diagnostics_ex['time'].append(t)
        diagnostics_ex['electric_energy'].append(We_ex)
        diagnostics_ex['magnetic_energy'].append(Wb_ex)

        # Discrete energy at t=0
        We_num, Wb_num = discrete_energies(e, b)
        diagnostics_num['time'].append(t)
        diagnostics_num['electric_energy'].append(We_num)
        diagnostics_num['magnetic_energy'].append(Wb_num)

        print('\nTotal energy in domain:')
        print('ts = {:4d},  t = {:8.4f},  exact = {Wt_ex:.13e},  discrete = {Wt_num:.13e}'.format(0,
            t,
            Wt_ex  = We_ex  + Wb_ex,
            Wt_num = We_num + Wb_num)
        )
    else:
        print('ts = {:4d},  t = {:8.4f}'.format(0, t))

    #--------------------------------------------------------------------------
    # Solution
    #--------------------------------------------------------------------------

    # TODO: add option to convert to scipy sparse format

    # ... Arguments for time stepping
    kwargs = {'verbose': verbose, 'tol': tol}

    if periodic:
        args = (e, b, M1, M2, D1, D1_T)
    else:
        args = (e, b, M1 + M1_bc, M2, D1, D1_T)
        kwargs['pc'] = 'jacobi'
    # ...

    # Time loop
    for ts in range(1, nsteps+1):

        # TODO: allow for high-order splitting

        # Strang splitting, 2nd order
        step_faraday_2d(0.5*dt, *args, **kwargs)
        step_ampere_2d (    dt, *args, **kwargs)
        step_faraday_2d(0.5*dt, *args, **kwargs)

        t += dt

        # Animation
        if plot_interval and (ts % plot_interval == 0 or ts == nsteps):

            # ...
            # TODO: improve
            for i, x1i in enumerate(x1[:, 0]):
                for j, x2j in enumerate(x2[0, :]):

                    Ex_values[i, j], Ey_values[i, j] = \
                            push_2d_hcurl(E.fields[0], E.fields[1], x1i, x2j, mapping)

                    Bz_values[i, j] = push_2d_l2(B, x1i, x2j, mapping)
            # ...

            # Update plot
            update_plot(fig2, t, x, y, Ex_values, Ex_ex(t, x, y))
            update_plot(fig3, t, x, y, Ey_values, Ey_ex(t, x, y))
            update_plot(fig4, t, x, y, Bz_values, Bz_ex(t, x, y))
            plt.pause(0.1)

        # Scalar diagnostics
        if diagnostics_interval and ts % diagnostics_interval == 0:

            # Update exact diagnostics
            We_ex, Wb_ex = exact_energies(t)
            diagnostics_ex['time'].append(t)
            diagnostics_ex['electric_energy'].append(We_ex)
            diagnostics_ex['magnetic_energy'].append(Wb_ex)

            # Update numerical diagnostics
            We_num, Wb_num = discrete_energies(e, b)
            diagnostics_num['time'].append(t)
            diagnostics_num['electric_energy'].append(We_num)
            diagnostics_num['magnetic_energy'].append(Wb_num)

            # Print total energy to terminal
            print('ts = {:4d},  t = {:8.4f},  exact = {Wt_ex:.13e},  discrete = {Wt_num:.13e}'.format(ts,
                t,
                Wt_ex  = We_ex  + Wb_ex,
                Wt_num = We_num + Wb_num)
            )
        else:
            print('ts = {:4d},  t = {:8.4f}'.format(ts, t))

    #--------------------------------------------------------------------------
    # Post-processing
    #--------------------------------------------------------------------------

    # ...
    # TODO: improve
    for i, x1i in enumerate(x1[:, 0]):
        for j, x2j in enumerate(x2[0, :]):

            Ex_values[i, j], Ey_values[i, j] = \
                    push_2d_hcurl(E.fields[0], E.fields[1], x1i, x2j, mapping)

            Bz_values[i, j] = push_2d_l2(B, x1i, x2j, mapping)
    # ...

    # Error at final time
    error_Ex = abs(Ex_ex(t, x, y) - Ex_values).max()
    error_Ey = abs(Ey_ex(t, x, y) - Ey_values).max()
    error_Bz = abs(Bz_ex(t, x, y) - Bz_values).max()
    print()
    print('Max-norm of error on Ex(t,x) at final time: {:.2e}'.format(error_Ex))
    print('Max-norm of error on Ey(t,x) at final time: {:.2e}'.format(error_Ey))
    print('Max-norm of error on Bz(t,x) at final time: {:.2e}'.format(error_Bz))

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

def test_maxwell_2d_periodic():

    namespace = run_maxwell_2d_TE(
        eps      = 0.5,
        ncells   = 12,
        degree   = 3,
        periodic = True,
        Cp       = 0.5,
        nsteps   = 1,
        tend     = None,
        splitting_order      = 2,
        plot_interval        = 0,
        diagnostics_interval = 0,
        tol = 1e-6,
        verbose = False
    )

    TOL = 1e-6
    ref = dict(error_Ex = 6.870389e-03,
               error_Ey = 6.870389e-03,
               error_Bz = 4.443822e-03)

    assert abs(namespace['error_Ex'] - ref['error_Ex']) / ref['error_Ex'] <= TOL
    assert abs(namespace['error_Ey'] - ref['error_Ey']) / ref['error_Ey'] <= TOL
    assert abs(namespace['error_Bz'] - ref['error_Bz']) / ref['error_Bz'] <= TOL


def test_maxwell_2d_dirichlet():

    namespace = run_maxwell_2d_TE(
        eps      = 0.5,
        ncells   = 10,
        degree   = 5,
        periodic = False,
        Cp       = 0.5,
        nsteps   = 1,
        tend     = None,
        splitting_order      = 2,
        plot_interval        = 0,
        diagnostics_interval = 0,
        tol = 1e-6,
        verbose = False
    )

    TOL = 1e-6
    ref = dict(error_Ex = 3.597840e-03,
               error_Ey = 3.597840e-03,
               error_Bz = 4.366314e-03)

    assert abs(namespace['error_Ex'] - ref['error_Ex']) / ref['error_Ex'] <= TOL
    assert abs(namespace['error_Ey'] - ref['error_Ey']) / ref['error_Ey'] <= TOL
    assert abs(namespace['error_Bz'] - ref['error_Bz']) / ref['error_Bz'] <= TOL

#==============================================================================
# SCRIPT CAPABILITIES
#==============================================================================
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        description = "Solve 2D Maxwell's equations in rectangular cavity with spline FEEC method."
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

    parser.add_argument( '--tol',
        type    = float,
        default = 1e-7,
        help    = 'Tolerance for iterative solver (L2-norm of residual)'
    )

    parser.add_argument( '-v', '--verbose',
        action  = 'store_true',
        help    = 'Print convergence information of iterative solver'
    )

    # Read input arguments
    args = parser.parse_args()

    # Run simulation
    namespace = run_maxwell_2d_TE(**vars(args))

    # Keep matplotlib windows open
    import matplotlib.pyplot as plt
    plt.show()
