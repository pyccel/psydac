import  time
import  numpy as np

from    sympde.calculus                     import inner
from    sympde.expr                         import integral, BilinearForm
from    sympde.topology                     import elements_of, Derham, Mapping, Cube

from    psydac.api.discretization           import discretize
from    psydac.api.settings                 import PSYDAC_BACKEND_GPYCCEL
from    psydac.linalg.basic                 import IdentityOperator
from    psydac.linalg.solvers               import inverse

def compute_vector_potential_3d(b, derham_h):
    """
    Computes a weak-divergence-free vector potential A in a subspace of H_0(curl) of a divergence-free function B in a subspace of H_0(div).
    This example highlights the usage of LST preconditioners and Dirichlet projectors introduced in recent PRs.

    Parameters
    ----------
    b : psydac.linalg.block.BlockVector
        Coefficient vector of a divergence-free function belonging to the third space (H_0(div))
        of the discrete de Rham sequence derham_h.

    derham_h : psydac.api.feec.DiscreteDeRham
        b belongs to the second third space V2_h of derham_h.
    
    Returns
    -------
    a : psydac.linalg.block.BlockVector
        Coefficient vector of the weak-divergence-free vector potential of the function corresponding
        to the coefficient vector b.

    Notes
    -----
    Denoting the hom. DBC satisfying function spaces of a 3D de Rham sequence by V0h, V1h, V2h and V3h,
    the problem

        Given B in V2h s.th. div(B)=0, find A in V1h s.th.
                curl(A) = B,
            weak-div(A) = 0,

    can be equivalently stated as the Hodge-Laplace problem

        Given B in V2h s.th. div(B)=0, find A in V1h s.th.
            weak-curl(curl(A)) - grad(weak-div(A)) = weak-curl(B).

    The corresponding variational formulation reads

        Given B in V2h s.th. div(B)=0, find A in V1h s.th.
            (curl(v), curl(A)) + (weak-div(v), weak-div(A)) = (curl(v), B)      for all v in V1h.

    The weak-divergence is defined implicitly by the set of equations

            (grad(u), v) = - (u, weak-div(v))       for all u in V0h, v in V1h.

    Hence, in terms of matrices (G for grad, wD for weak-divergence)

            u^T G^T M1 v = - u^T M0 wD v        <=>         wD = - (M0)^-1 G^T M1.

    Thus, not taking into account boundary conditions, the full system of equations reads

            v^T C^T M2 C A + v^T M1 G (M0)^-1 M0 (M0)^-1 G^T M1   A = v^T C^T M2 B   for all v in V1h

        <=>   ( C^T M2 C   +     M1 G            (M0)^-1 G^T M1 ) A =     C^T M2 B.
    
    Note: (M0)^-1 here is the inverse "mass matrix of functions satisfying hom. DBCs", 
    not the inverse of the entire M0 "mass matrix of all functions".

    This example highlights the importance of choosing the correct inverse,
    choosing the correct preconditioner for this correct inverse,
    and applying the projection method using `DirichletProjector`s to solve this discrete problem.
    """

    assert b.space is derham_h.spaces[2].coeff_space

    # ----- Obtain standard objects: domain, domain_h, derham, function spaces, mass matrices, derivative operators, ...

    domain_h = derham_h.domain_h
    domain   = domain_h.domain
    derham   = Derham(domain)

    V0, V1, V2, V3          = derham.spaces
    V0h, V1h, V2h, V3h      = derham_h.spaces
    V0cs, V1cs, V2cs, V3cs  = [Vh.coeff_space for Vh in derham_h.spaces]

    u0, v0 = elements_of(V0, names='u0, v0')
    u1, v1 = elements_of(V1, names='u1, v1')
    u2, v2 = elements_of(V2, names='u2, v2')
    u3, v3 = elements_of(V3, names='u3, v3')

    G, C, D = derham_h.derivatives(kind='linop')

    a0 = BilinearForm((u0, v0), integral(domain, u0*v0))
    a1 = BilinearForm((u1, v1), integral(domain, inner(u1, v1)))
    a2 = BilinearForm((u2, v2), integral(domain, inner(u2, v2)))
    a3 = BilinearForm((u3, v3), integral(domain, u3*v3))

    t0  = time.time()
    a0h = discretize(a0, domain_h, (V0h, V0h), backend=backend)
    a1h = discretize(a1, domain_h, (V1h, V1h), backend=backend)
    a2h = discretize(a2, domain_h, (V2h, V2h), backend=backend)
    a3h = discretize(a3, domain_h, (V3h, V3h), backend=backend)
    t1 = time.time()
    print()
    print(f'Mass matrix bilinear forms discretized in {t1-t0:.3g}')

    t0 = time.time()
    M0 = a0h.assemble()
    M1 = a1h.assemble()
    M2 = a2h.assemble()
    M3 = a3h.assemble()
    t1 = time.time()
    print(f'Mass matrices assembled in {t1-t0:.3g}')

    # ----- Sanity check: b should be divergence-free. The method will convergence even if it is not,
    #       but b = curl(a) won't hold.
    divB        = D @ b
    l2norm_divB = np.sqrt(M3.dot_inner(divB, divB))
    assert l2norm_divB < 1e-10, 'The coefficient vector passed to this function should belong to a divergence-free function.'

    # ----- We need to obtain an inverse M0 object in order to assemble the system matrix
    DP0, DP1, _, _   = derham_h.dirichlet_projectors(kind='linop') # <- Dirichlet boundary projectors for the projection method
    I0               = IdentityOperator(V0cs)

    # Option1: Modified mass matrix of functions satisfying hom. DBCs
    M0_0     = DP0 @ M0 @ DP0 + (I0 - DP0)
    t0       = time.time()
    M0_0_pc, = derham_h.LST_preconditioners(M0=M0, hom_bc=True)
    t1       = time.time()
    M0_0_inv = inverse(M0_0, 'pcg', pc=M0_0_pc, maxiter=1000, tol=1e-15)

    # Option2: Inverse of entire mass matrix
    t2       = time.time()
    M0_pc,   = derham_h.LST_preconditioners(M0=M0, hom_bc=False)
    t3       = time.time()
    M0_inv   = inverse(M0,   'pcg', pc=M0_pc,   maxiter=1000, tol=1e-15)

    print(f'M0 and M0_0 preconditioner obtained in {((t3-t2)+(t1-t0)):.3g}')
    print()

    # ----- There are now (at least) three ways to assemble the system matrix

    # Option1: The correct option. Using M0_0_inv and a projection operator immediately before M0_0_inv
    S1  = C.T @ M2 @ C  +  M1 @ G @ M0_0_inv @ DP0 @ G.T @ M1
    # Note that not including the projector here, i.e., using the following stiffness matrix
    #S1 = C.T @ M2 @ C  +  M1 @ G @ M0_0_inv @        G.T @ M1
    # results in a wrong solution (and is super slow)

    # Option2: The entirely wrong option. Using M0_inv and no additional projector
    S2  = C.T @ M2 @ C  +  M1 @ G @ M0_inv         @ G.T @ M1

    # Option3: The better of the wrong options. Using M0_inv, but an additional projector
    S3  = C.T @ M2 @ C  +  M1 @ G @ M0_inv   @ DP0 @ G.T @ M1

    # ----- We assemble the rhs vector
    rhs = C.T @ M2 @ b

    # ----- We can now solve the three resulting systems
    I1      = IdentityOperator(V1cs)

    maxiter = 5000
    tol     = 1e-10

    vector_potential_list = []
    timings_list          = []
    info_list             = []

    for S in [S1, S2, S3]:
        # Apply the projection method
        S_bc    = DP1 @ S @ DP1 + (I1 - DP1)
        rhs_bc  = DP1 @ rhs

        # Obtain a solver for the system matrix
        S_bc_inv = inverse(S_bc, 'cg', maxiter=maxiter, tol=tol)

        # Solve the system
        t0   = time.time()
        a    = S_bc_inv @ rhs_bc
        t1   = time.time()
        info = S_bc_inv.get_info()

        vector_potential_list.append(a)
        timings_list.append(t1-t0)
        info_list.append(info)

    # ----- Analyse the results

    # Option1 is the fastest, and accurate.
    a1 = vector_potential_list[0]
    info1 = info_list[0]
    diff = b - C @ a1
    l2norm_diff = np.sqrt(M2.dot_inner(diff, diff))
    print(f' ----- Option1: C.T @ M2 @ C  +  M1 @ G @ M0_0_inv @ DP0 @ G.T @ M1 -----')
    print()
    print(f' || B - curl(A) || = {l2norm_diff:.3g}')
    print(f' Time              : {timings_list[0]:.3g}')
    print(f' Niter             : {info1["niter"]}')
    print(f' Convergence       : {info1["success"]}')
    print(f' Res. Norm         : {info1["res_norm"]:.3g}')
    print()

    # Option2 is not much slower, but delivers an entirely wrong solution
    a2 = vector_potential_list[1]
    info2 = info_list[1]
    diff = b - C @ a2
    l2norm_diff = np.sqrt(M2.dot_inner(diff, diff))
    print(f' ----- Option2: C.T @ M2 @ C  +  M1 @ G @ M0_inv         @ G.T @ M1 -----')
    print()
    print(f' || B - curl(A) || = {l2norm_diff:.3g}')
    print(f' Time              : {timings_list[1]:.3g}')
    print(f' Niter             : {info2["niter"]}')
    print(f' Convergence       : {info2["success"]}')
    print(f' Res. Norm         : {info2["res_norm"]:.3g}')
    print()

    # Option3 is takes forever to converge, but delivers an accurate solution
    a3 = vector_potential_list[2]
    info3 = info_list[2]
    diff = b - C @ a3
    l2norm_diff = np.sqrt(M2.dot_inner(diff, diff))
    print(f' ----- Option3: C.T @ M2 @ C  +  M1 @ G @ M0_inv   @ DP0 @ G.T @ M1 -----')
    print()
    print(f' || B - curl(A) || = {l2norm_diff:.3g}')
    print(f' Time              : {timings_list[2]:.3g}')
    print(f' Niter             : {info3["niter"]}')
    print(f' Convergence       : {info3["success"]}')
    print(f' Res. Norm         : {info3["res_norm"]:.3g}')
    print()
    print(f'Note that increasing the problem size, or decreasing the tolerance, renders Option3 unafforadable.')

    return a1

#==============================================================================
if __name__ == '__main__':

    ncells   = [16, 16, 16]
    degree   = [3, 3, 3]
    periodic = [False, False, False]

    comm     = None
    backend  = PSYDAC_BACKEND_GPYCCEL

    logical_domain = Cube('C', bounds1=(0,1), bounds2=(0,1), bounds3=(0,1))

    class CollelaMap3D(Mapping):
        _expressions = {'x': 'x1 + (a/2)*sin(2.*pi*(x1-0.5))*sin(2.*pi*(x2-0.5))',
                        'y': 'x2 + (a/2)*sin(2.*pi*(x1-0.5))*sin(2.*pi*(x2-0.5))',
                        'z': 'x3'}

        _ldim = 3
        _pdim = 3

    mapping     = CollelaMap3D('C', a=0.1)

    domain      = mapping(logical_domain)
    derham      = Derham(domain)

    domain_h    = discretize(domain, ncells=ncells, periodic=periodic, comm=comm)
    derham_h    = discretize(derham, domain_h, degree=degree)

    G, C, D        = derham_h.derivatives(kind='linop')
    P0, P1, P2, P3 = derham_h.projectors()

    from sympde.utilities.utils import plot_domain
    #plot_domain(domain, draw=True, isolines=True)

    A1 = lambda x, y, z: np.sin(2*np.pi*y) * np.sin(2*np.pi*z)
    A2 = lambda x, y, z: np.sin(2*np.pi*z) * np.sin(2*np.pi*x)
    A3 = lambda x, y, z: np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
    A  = (A1, A2, A3)

    a_ex    = P1(A).coeffs
    # a should already satisfy hom. DBCs (n \times a = 0 on the boundary)
    # But we can make sure that this holds exactly by projecting it
    _, DP1, _, _ = derham_h.dirichlet_projectors(kind='linop')
    a_ex    = DP1 @ a_ex

    # Now b belongs to H_0(div0) as required
    b       = C @ a_ex

    a = compute_vector_potential_3d(b, derham_h)
