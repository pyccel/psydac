import  os
import  time
import  numpy               as np
import  matplotlib.pyplot   as plt
from    sympy               import Matrix, lambdify, sqrt, exp, sin, cos, atan2

from    sympde.topology     import Square, Derham, elements_of, Mapping
from    sympde.calculus     import inner
from    sympde.expr         import BilinearForm, integral

from    psydac.api.discretization       import discretize
from    psydac.api.settings             import PSYDAC_BACKEND_GPYCCEL
from    psydac.linalg.basic             import IdentityOperator
from    psydac.linalg.solvers           import inverse
from    psydac.fem.basic                import FemField
from    psydac.api.postprocessing       import OutputManager

#from    code.wedge_duality_preconditioner  import construct_wd_preconditioner
#from    code.domains                       import get_mapping
#from    code.tensors                       import get_smoothly_rotating_tensor_2d, get_radial_angular_tensor_2d, get_wiggly_radial_angular_tensor_2d
#from    code.kronecker_preconditioner      import construct_Kronecker_preconditioners

class StarPiece(Mapping):
    _expressions = {'x': 'a1*x1*x2 + b1*x1 + c1*x2 + d1',
                    'y': 'a2*x1*x2 + b2*x1 + c2*x2 + d2'}
    _ldim = 2
    _pdim = 2

phi = 2*np.pi/5
R   = 2.5

bottom_left_piece = {'a1': R*np.cos(3*np.pi/2 - phi/2) - np.cos(3*np.pi/2 - phi),
                     'a2': R*np.sin(3*np.pi/2 - phi/2) - np.sin(3*np.pi/2 - phi) + 1,
                     'b1': np.cos(3*np.pi/2 - phi),
                     'b2': np.sin(3*np.pi/2 - phi),
                     'c1': 0, 'c2': -1, 'd1': 0, 'd2': 0}
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
class HalfAnnulus(Mapping):
    _expressions = {'x': '(R0 + (R1-R0)*x1) * cos(pi*x2)',
                    'y': '(R0 + (R1-R0)*x1) * sin(pi*x2)'}
    _ldim = 2
    _pdim = 2
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
class WigglyHalfAnnulus(Mapping):
    _expressions = {'x': '(R0 + (R1-R0)*x1 + (R0/2)*cos(4*pi*x2)) * cos(pi*x2)',
                    'y': '(R0 + (R1-R0)*x1 + (R0/2)*cos(4*pi*x2)) * sin(pi*x2)'}
    _ldim = 2
    _pdim = 2
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
class RightDiscPiece(Mapping):
    _expressions = {'x': '(sqrt(2)/2) * (x1 + 1) * (1 + x1*(sqrt(4*x2*(1-x2)+1)-1))',
                    'y': '(sqrt(2)/2) * (x1 + 1) * (2*x2 - 1)'}
    _ldim = 2
    _pdim = 2
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
class HoledPlatePiece(Mapping):
    _expressions = {'x': '-4 * (1 - x1) + cos(pi+pi/4*x2)*x1',
                    'y': '4*x2*(1-x1) - sin(pi+pi/4*x2)*x1'}
    _ldim = 2
    _pdim = 2
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
class ColellaMapping2D(Mapping):
    _expressions = {'x': 'x1 + (a/2)*sin(2.*pi*(x1-0.5))*sin(2.*pi*(x2-0.5))',
                    'y': 'x2 + (a/2)*sin(2.*pi*(x1-0.5))*sin(2.*pi*(x2-0.5))'}
    _ldim        = 2
    _pdim        = 2

a2 = 0.2
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
class ColellaMapping3D(Mapping):
    _expressions = {'x': 'x1 + (a/2)*sin(2.*pi*(x1-0.5))*sin(2.*pi*(x2-0.5))',
                    'y': 'x2 + (a/2)*sin(2.*pi*(x1-0.5))*sin(2.*pi*(x2-0.5))',
                    'z': 'x3'}
    _ldim        = 3
    _pdim        = 3

a3 = 0.2
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
class Blade(Mapping):
    _expressions = {'x': 'c0*(1-tau*x3) * (x1-0.5) + A*x3*(2-x3)',
                    'y': 'c0*(1-tau*x3) * (kappa*4*x1*(1-x1) + (2*x2-1)*(T*4*x1*(1-x1) + tmin)) + (s0+(s1-s0)*x3) * c0*(1-tau*x3) * (x1-0.5) + B*x3**2',
                    'z': 'H*x3'}
    _ldim = 3
    _pdim = 3

blade_kwargs = {'c0':1.0, 'tau':0.55, 'kappa':0.06, 'T':0.16, 'tmin':0.02, 
                'A':0.35, 'B':0.15, 'H':2.2, 's0': 0., 's1': 1.6} 
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
class Kite(Mapping):
    _expressions = {'x': '4*x1*x2 + x1',
                    'y': '4*x1*x2 + x2'}
    _ldim        = 2
    _pdim        = 2
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
class Identity3D(Mapping):
    _expressions = {'x': 'x1',
                    'y': 'x2',
                    'z': 'x3'}
    _ldim = 3
    _pdim = 3
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
class Identity2D(Mapping):
    _expressions = {'x': 'x1',
                    'y': 'x2'}
    _ldim = 2
    _pdim = 2
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------

def get_mapping(domain_name, **kwargs):
    if domain_name == 'StarPiece':
        kwargs = bottom_left_piece
        mapping = StarPiece('SP', **kwargs)
    elif domain_name == 'HalfAnnulus':
        R0 = kwargs.pop('R0', 1)
        R1 = kwargs.pop('R1', 2)
        mapping = HalfAnnulus('HA', R0=R0, R1=R1)
    elif domain_name == 'WigglyHalfAnnulus':
        R0 = kwargs.pop('R0', 1)
        R1 = kwargs.pop('R1', 2)
        mapping = WigglyHalfAnnulus('WHA', R0=R0, R1=R1)
    elif domain_name == 'RightDiscPiece':
        mapping = RightDiscPiece('RDP')
    elif domain_name == 'HoledPlatePiece':
        mapping = HoledPlatePiece('HPP')
    elif domain_name == 'Colella':
        mapping = ColellaMapping2D('C', a=a2)
    elif domain_name == 'Colella3D':
        mapping = ColellaMapping3D('C', a=a3)
    elif domain_name == 'Blade':
        kwargs = blade_kwargs
        mapping = Blade('B', **kwargs)
    elif domain_name == 'Identity3D':
        mapping = Identity3D('I')
    elif domain_name == 'Kite':
        mapping = Kite('K')
    elif domain_name == 'Identity2D':
        mapping = Identity2D('I')
    else:
        raise ValueError(f'{domain_name =} not understood.')
    return mapping

def get_wiggly_radial_angular_tensor_2d(x, y, R0, eps_r, eps_theta):
    r     = sqrt(x**2 + y**2)
    theta = atan2(y, x)

    e_theta_not_normalized = Matrix([-y, x]) - (2*R0*sin(4*theta))/r * Matrix([x, y])
    norm = sqrt(e_theta_not_normalized[0]**2 + e_theta_not_normalized[1]**2)

    e_theta = (1/norm) * e_theta_not_normalized
    e_r     = Matrix([e_theta[1], -e_theta[0]])

    a1 = e_r     # e_r     for beta = 0
    a2 = e_theta # e_theta for beta = 0

    A1 = Matrix([[a1[0]*a1[0], a1[0]*a1[1]], [a1[1]*a1[0], a1[1]*a1[1]]])
    A2 = Matrix([[a2[0]*a2[0], a2[0]*a2[1]], [a2[1]*a2[0], a2[1]*a2[1]]])

    tensor         =    eps_r  * A1 +    eps_theta  * A2
    inverse_tensor = (1/eps_r) * A1 + (1/eps_theta) * A2

    return tensor, inverse_tensor

# -----------------------------------------------------------------------------------------------------------------

def get_new_initial_condition(x, y, R0, R1, eps_r, eps_theta, radial_angular_tensor, rotating_radial_angular_tensor, wiggly_radial_angular_tensor):
    """
    Obtain initial conditions for the electric and magnetic field in a 2d transverse electric Maxwell simulation.
    This initial condition should be a so called wave packet.

    Parameters
    ----------
    x, y : sympy.symbol
        The coordinates of the physical domain. The physical domain is a half-annulus bound in radius by R0 < r < R1 and angle by 0 < theta < pi.

    R0, R1 : float
        The radial bounds of the half-annulus domain.

    eps_r, eps_theta : float
        Material parameters defining the electric permittivity tensor.

    Returns
    -------
    E0, B0 : sympy.expr
        Sympy expressions corresponding to the initial electric and magnetic field.
    
    """
    assert not (radial_angular_tensor and rotating_radial_angular_tensor)
    if wiggly_radial_angular_tensor:
        assert not (radial_angular_tensor or rotating_radial_angular_tensor)
    # Center of the wave packet (x_c, y_c)
    if not wiggly_radial_angular_tensor:
        R01      = (R1 + R0)/2
    else:
        R01      = (R1 + R0)/2 + R0/2
    #x_c, y_c = float(1/np.sqrt(2)) * (-R01), float(1/np.sqrt(2)) * R01
    x_c, y_c = 0, R01

    # Gaussian function with center (x_c, y_c)
    sigma   = .4 # .5
    xy_c_2  = (x-x_c)**2 + (y-y_c)**2
    g       = exp(-xy_c_2/(2*sigma**2))

    # (Clockwise) Rotation operator
    def J(vec):
        return Matrix([vec[1], -vec[0]])

    # We want to excite the wave in the direction s = (1, 0). 
    # We obtain the inverse electric permittivity tensor, evaluate it at the center, and apply it to s, then prescribe the resulting vector a norm, to obtain k_0
    if rotating_radial_angular_tensor:
        _, inverse_tensor = 0 #get_smoothly_rotating_tensor_2d(x, y, R0, R1, eps_r, eps_theta)
    elif radial_angular_tensor:
        _, inverse_tensor = 0 #get_radial_angular_tensor_2d(x, y, eps_r, eps_theta)
    elif wiggly_radial_angular_tensor:
        _, inverse_tensor = get_wiggly_radial_angular_tensor_2d(x, y, R0, eps_r, eps_theta)
    else:
        inverse_tensor = Matrix([[1, 0], [0, 1]])
    inverse_tensor_c  = inverse_tensor.subs([(x, x_c), (y, y_c)])
    s                 = Matrix([1, 0])
    k_not_normalized  = inverse_tensor_c @ s
    k_norm            = float(7.5 * np.pi)
    k                 = k_norm * (1/sqrt(k_not_normalized.dot(k_not_normalized))) * k_not_normalized

    # We obtain the initial electric field E0
    # psi      = g(x) * sin(phi(x))
    # grad psi = g(x) * [(-1/sigma**2) * sin(phi(x)) * xy_c + cos(phi(x)) * k]
    xy_c     = Matrix([x-x_c, y-y_c])
    phi      = k.dot(xy_c)
    grad_psi = g * ((-1/sigma**2)*sin(phi)*xy_c + cos(phi)*k)
    D0       = J(grad_psi)
    E0       = inverse_tensor @ D0

    # We compute the wave frequency
    w0_2 = J(k).dot(inverse_tensor_c @ J(k))
    w0   = sqrt(w0_2)

    # We compute the initial magnetic field B0
    B0 = - w0 * g * cos(phi)

    return E0, B0

def main(foldername, 
         ncells, degree, 
         R0=1, R1=2, 
         T=None, n_steps=100, dt=1e-2, 
         pc='wedge', maxiter=1000, tol=1e-10, 
         isotropic=False, eps_r=1, eps_theta=3, 
         radial_angular_tensor = False, rotating_radial_angular_tensor=True, 
         wiggly_radial_angular_tensor=False, 
         save=False, save_every=10,
         measure_timings=False):

    assert (T is not None) or (n_steps is not None), f'Require either a max. simulation time or a max. amount of simulation steps'

    backend = PSYDAC_BACKEND_GPYCCEL

    # ----- SYMBOLIC OBJECTS -----

    logical_domain = Square('S', bounds1=(0, 1), bounds2=(0, 1))
    if wiggly_radial_angular_tensor:
        mapping    = get_mapping('WigglyHalfAnnulus', R0=R0, R1=R1)
    else:
        mapping    = get_mapping('HalfAnnulus', R0=R0, R1=R1)
    domain         = mapping(logical_domain)
    sequence       = ['h1', 'hcurl', 'l2']
    derham         = Derham(domain, sequence=sequence)
    x, y           = domain.coordinates

    V0, V1, V2 = derham.spaces
    u0, v0    = elements_of(V0, names='u0, v0')
    u1, v1    = elements_of(V1, names='u1, v1')
    u2, v2    = elements_of(V2, names='u2, v2')

    # -----

    if rotating_radial_angular_tensor:
        tensor, inverse_tensor = 0 #get_smoothly_rotating_tensor_2d(x, y, R0, R1, eps_r, eps_theta)
    elif wiggly_radial_angular_tensor:
        tensor, inverse_tensor = get_wiggly_radial_angular_tensor_2d(x, y, R0, eps_r, eps_theta)
    else:
        assert isotropic or radial_angular_tensor # if isotropic, tensor won't be used and inverse_tensor will be changed to None
        tensor, inverse_tensor = 0 #get_radial_angular_tensor_2d(x, y, eps_r, eps_theta)

    if isotropic:
        inverse_tensor = None
    
    tensor_u1_dot_v1 = (tensor[0,0]*v1[0] + tensor[0,1]*v1[1])*u1[0] + \
                       (tensor[1,0]*v1[0] + tensor[1,1]*v1[1])*u1[1]

    m0     = BilinearForm((u0, v0), integral(domain, u0*v0))
    if isotropic:
        m1 = BilinearForm((u1, v1), integral(domain, inner(u1, v1)))
    else:
        m1 = BilinearForm((u1, v1), integral(domain, tensor_u1_dot_v1))
    m2     = BilinearForm((u2, v2), integral(domain, u2*v2))

    # ----- DISCRETIZATION -----

    domain_h = discretize(domain, ncells=ncells)
    derham_h = discretize(derham, domain_h, degree=degree)

    V0h, V1h, V2h = derham_h.spaces

    V0cs = V0h.coeff_space
    I0   = IdentityOperator(V0cs)
    V1cs = V1h.coeff_space
    I1   = IdentityOperator(V1cs)

    G, C      = derham_h.derivatives(kind='linop')
    DP0, DP1, _ = derham_h.dirichlet_projectors(kind='linop')
    _, P1, P2 = derham_h.projectors()

    M0   = discretize(m0, domain_h, (V0h, V0h), backend=backend).assemble()
    M1   = discretize(m1, domain_h, (V1h, V1h), backend=backend).assemble()
    M2   = discretize(m2, domain_h, (V2h, V2h), backend=backend).assemble()

    M0_0 = DP0 @ M0 @ DP0 + (I0 - DP0)
    M1_0 = DP1 @ M1 @ DP1 + (I1 - DP1)

    if pc == 'wedge':
        t0 = time.time()
        M1_pc    = 0 #construct_wd_preconditioner(domain_h, V1h, inverse_tensor=inverse_tensor, backend=backend)
        t1 = time.time()
        M1_0_pc  = DP1 @ M1_pc @ DP1
    elif pc =='lst':
        t0 = time.time()
        M1_0_pc, = derham_h.LST_preconditioners(M1=M1, hom_bc=True)
        t1 = time.time()
    else:
        M1_0_pc  = None

    t_pc = t1-t0 if pc in ('wedge', 'lst') else 0

    M1_0_inv = inverse(M1_0, 'CG', pc=M1_0_pc, maxiter=maxiter, tol=tol)

    # ----- Measure Timings -----
    if measure_timings:
        reps         = 100
        slow_reps    = 10
        test_maxiter = 10000
        test_tol     = 1e-10

        E, B        = get_new_initial_condition(x, y, R0, R1, eps_r, eps_theta, radial_angular_tensor, rotating_radial_angular_tensor, wiggly_radial_angular_tensor)
        Ex_callable = lambdify((x, y), E[0])
        Ey_callable = lambdify((x, y), E[1])
        E_callable  = (Ex_callable, Ey_callable)
        B_callable  = lambdify((x, y), B)

        e0          = DP1 @ P1(E_callable).coeffs
        b0          = P2(B_callable).coeffs

        b0_half     = b0 - (dt/2) * C @ e0
        rhs         = M1_0 @ e0 + dt * DP1 @ C.T @ M2 @ b0_half

        M1_wedge_pc    = 0 #construct_wd_preconditioner(domain_h, V1h, inverse_tensor=inverse_tensor, backend=backend)
        #M1_wedge_pc    = DP1 @ M1_wedge_pc @ DP1
        M1_lst_pc,     = derham_h.LST_preconditioners(M1=M1) # , hom_bc=True)
        M1_kron_pc     = 0 #construct_Kronecker_preconditioners(derham_h)[1]
        M1_jacobi_pc   = M1.diagonal(inverse=True, sqrt=False)
        pcs            = [M1_wedge_pc, M1_lst_pc, M1_kron_pc, M1_jacobi_pc, None]
        pc_reps        = [reps, reps, reps, slow_reps, slow_reps]
        #Ms             = [M1_0, M1_0, M1, M1, M1]
        timings        = []

        #def test(M1, M1_pc, reps, rhs, test_maxiter, test_tol):
        def test(M1_pc, reps, rhs, test_maxiter, test_tol):
            M1_inv = inverse(M1, 'CG', pc=M1_pc, maxiter=test_maxiter, tol=test_tol)
            T = 0.
            for _ in range(reps):
                t0 = time.time()
                y = M1_inv @ rhs
                t1 = time.time()
                T += t1-t0
            print(M1_inv.get_info(), f'{reps = }')
            T /= reps

            return T

        #timings = [test(M, pc, pc_rep, rhs, test_maxiter, test_tol) for (M, pc, pc_rep) in zip(Ms, pcs, pc_reps)]
        timings = [test(pc, pc_rep, rhs, test_maxiter, test_tol) for (pc, pc_rep) in zip(pcs, pc_reps)]

        return timings

    # ----- INITIAL CONDITION & SETUP -----

    #E, B        = get_initial_condition(x, y, R0, R1, eps_r, eps_theta, radial_angular_tensor, rotating_radial_angular_tensor)
    E, B        = get_new_initial_condition(x, y, R0, R1, eps_r, eps_theta, radial_angular_tensor, rotating_radial_angular_tensor, wiggly_radial_angular_tensor)
    Ex_callable = lambdify((x, y), E[0])
    Ey_callable = lambdify((x, y), E[1])
    E_callable  = (Ex_callable, Ey_callable)
    B_callable  = lambdify((x, y), B)

    e0      = DP1 @ P1(E_callable).coeffs
    b0      = P2(B_callable).coeffs
    energy0 = 0.5 * (M1.dot_inner(e0, e0) + M2.dot_inner(b0, b0))

    # Testing whether the continuous initial condition already satisfies hom. Dirichlet BCs
    e0_alt = P1(E_callable).coeffs
    diff = e0 - e0_alt
    err = np.sqrt(diff.inner(diff))
    print(f' | P1(E) - DP(P1(E)) |_l2 = {err:.3g}')

    # Computing the weak-divergence of the initial condition (doesn't make a lot of sense because D0, not E0, must be divergence free.)
    #M0_pc    = construct_wedge_pc(domain_h, V0h, epsilon_inverse=None, backend=backend)
    #M0_0_pc  = DP0 @ M0_pc @ DP0
    #M0_0_inv = inverse(M0_0, 'CG', pc=M0_0_pc, maxiter=1000, tol=1e-15)
    #weak_div = - M0_0_inv @ DP0 @ G.T @ M1

    #weak_div_e0 = weak_div @ e0
    #weak_div_e0_norm = np.sqrt(M0.dot_inner(weak_div_e0, weak_div_e0))

    e_history           = [e0, ]
    b_history           = [b0, ]
    time_history        = [0, ]
    energy_history      = [energy0, ]
    niter_history       = []
    inverse_t_history   = []

    t            = 0
    ts           = 0
    t_global     = 0
    t_inv_global = 0
    t_out_global = 0

    if n_steps is not None:
        # T beats n_steps
        n_steps = int(np.ceil(T/dt)) if T is not None else n_steps
    else:
        n_steps = int(np.ceil(T/dt))

    if save:
        space_file = f'{foldername}/space_info.yml'
        coeff_file = f'{foldername}/field_info.h5'

        Om = OutputManager(space_file, coeff_file)
        Om.add_spaces(V1=V1h)
        Om.add_spaces(V2=V2h)
        Om.export_space_info()

        Om.add_snapshot(t=t, ts=ts)
        Om.export_fields(E=FemField(V1h, e_history[0]))
        Om.export_fields(B=FemField(V2h, b_history[0]))

    # ----- TIME STEPPING -----

    t0_global = time.time()

    for i in range(1, n_steps+1):

        e_n = e_history[-1]
        b_n = b_history[-1]

        b_n_half     = b_n - (dt/2) * C @ e_n
        rhs          = M1_0 @ e_n + dt * DP1 @ C.T @ M2 @ b_n_half
        t0           = time.time()
        e_n1         = M1_0_inv @ rhs
        t1           = time.time()
        solve_time   = t1-t0
        t_inv_global += solve_time
        niter        = M1_0_inv.get_info()['niter']
        b_n1         = b_n_half - (dt/2) * C @ e_n1

        e_history.append(e_n1)
        b_history.append(b_n1)

        t  += dt
        ts += 1

        energy = 0.5 * (M1.dot_inner(e_n1, e_n1) + M2.dot_inner(b_n1, b_n1))

        energy_history.append(energy)
        inverse_t_history.append(solve_time)
        niter_history.append(niter)
        time_history.append(t)

        if i % save_every == 0:
            t0_out = time.time()
            if save:
                Om.add_snapshot(t=t, ts=ts)
                Om.export_fields(E=FemField(V1h, e_history[-1]))
                Om.export_fields(B=FemField(V2h, b_history[-1]))
            t1_out = time.time()
            t_out_global += t1_out - t0_out
            print(f'{t = }, Step {i} completed.\nM1_0_inv convergence information: \n{M1_0_inv.get_info()} in {solve_time:.3g} sec.')

        if i == n_steps:
            #weak_div_eT = weak_div @ e_history[-1]
            #weak_div_eT_norm = np.sqrt(M0.dot_inner(weak_div_eT, weak_div_eT))
            if i % save_every != 0:
                t0_out = time.time()
                if save:
                    Om.add_snapshot(t=t, ts=ts)
                    Om.export_fields(E=FemField(V1h, e_history[-1]))
                    Om.export_fields(B=FemField(V2h, b_history[-1]))
                t1_out = time.time()
                t_out_global += t1_out - t0_out
            print(f'Simulation completed, {t = }, Step {i}.')

    t1_global = time.time()
    t_global += t1_global - t0_global

    plt.plot(time_history, energy_history)
    plt.title('Energy vs. Simulated Time')
    plt.savefig(f'{foldername}/energy.png')
    plt.clf()

    plt.plot(time_history[1:], inverse_t_history)
    plt.title('M1_eps_inv Application Time vs. Simulated Time')
    plt.savefig(f'{foldername}/inv_time.png')
    plt.clf()

    plt.plot(time_history[1:], niter_history)
    plt.title('M1_eps_inv Iterations vs. Simulated Time')
    plt.savefig(f'{foldername}/niter.png')
    plt.clf()

    runtime_info  = f'\n'
    runtime_info += f' ----- \n'
    runtime_info += f' Entire Loop Time:                       {t_global}\n'
    runtime_info += f' - of which spent to invert M1_eps_0:    {t_inv_global}\n'
    runtime_info += f' - of which spent to store coefficients: {t_out_global}\n'
    runtime_info += f' Time spent to obtain the pc:            {t_pc}\n'
    runtime_info += f' ----- \n'
    #runtime_info += f' Norm of weak divergence at t=0 and t=T: {weak_div_e0_norm} and {weak_div_eT_norm}\n'
    #runtime_info += f' ----- \n'

    if save:
        Om.close()

    return runtime_info

# ------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # spacial discretization paramaters
    ncells = [32, 256]
    degree = [3, 3]

    # domain boundary
    R0 = 2
    R1 = 4

    # simulation parameters
    T       = 10
    n_steps = None
    dt      = 5e-4

    # M1_0_inv parameters
    maxiter = 1000
    tol     = 1e-10
    pc      = 'lst'

    # material parameters
    isotropic                      = False
    radial_angular_tensor          = False
    rotating_radial_angular_tensor = False
    wiggly_radial_angular_tensor   = True
    eps_r                          = .5     # mostly e_r        # speeds up  angular transport as it decreases
    eps_theta                      = 100    # mostly e_theta    # slows down radial  transport as it increases
    assert not (radial_angular_tensor and rotating_radial_angular_tensor)
    if wiggly_radial_angular_tensor:
        assert not (radial_angular_tensor or rotating_radial_angular_tensor)

    measure_timings = False # True

    save       = True
    # amount of .vtu files to generate (a.k.a. amount of coefficients saved):
    n_vtu      = 100
    save_every = int((T/dt)/n_vtu) # 100(+1) coeffs saved, T/dt = 10/5e-4=2e4=20000 timesteps -> save coeffs every 200 time steps

    foldername   = 'generate_plot_data_xxOkt'

    test_purpose = 'Data to be used for plotting utils development.'

    assert not os.path.isdir(foldername)
    os.makedirs(foldername)

    runtime_info = main(foldername, 
                        ncells, degree, 
                        R0=R0, R1=R1, 
                        T=T, n_steps=n_steps, dt=dt, 
                        pc=pc, maxiter=maxiter, tol=tol, 
                        isotropic=isotropic, eps_r=eps_r, eps_theta=eps_theta, 
                        radial_angular_tensor=radial_angular_tensor, rotating_radial_angular_tensor=rotating_radial_angular_tensor, 
                        wiggly_radial_angular_tensor=wiggly_radial_angular_tensor, 
                        save=save, save_every=save_every,
                        measure_timings=measure_timings)

    if measure_timings:
        print(runtime_info)

    if test_purpose is not None:
        txt = test_purpose + '\n'
    else:
        txt = ''
    txt += f'Simulation using {ncells} B-spline cells and B-splines of degree {degree}\n'
    txt += f'inner radius {R0 = }, outer radius {R1 = }\n'
    txt += f'max time {T = }, max steps {n_steps = }, stepsize {dt = }\n'
    txt += f'M1_0_inv {pc = }, max M1_0_inv iterations = {maxiter} and tolerance {tol}\n'
    txt += f'material parameters: {isotropic = }, {eps_r = }, {eps_theta = }, {radial_angular_tensor = }, {rotating_radial_angular_tensor = }\n'
    if not measure_timings:
        txt += runtime_info
    else:
        txt += '\t'.join(str(timing) for timing in runtime_info)

    f = open(f'{foldername}/info.txt', 'w')
    f.writelines(txt)
    f.close()
