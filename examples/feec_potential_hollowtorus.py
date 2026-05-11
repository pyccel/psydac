from    mpi4py  import MPI
import  numpy   as np
import time

from    sympde.topology import Cube, Mapping, Derham

from    psydac.api.discretization import discretize
from    sympde.topology import Union, NormalVector

from    sympde.calculus         import inner, cross
from    sympde.expr             import integral, BilinearForm
from    sympde.topology         import elements_of

from    psydac.api.settings     import PSYDAC_BACKEND_GPYCCEL
from    psydac.linalg.basic     import IdentityOperator, MatrixFreeLinearOperator
from    psydac.linalg.solvers   import inverse

from scipy.sparse               import bmat, csc_matrix
from scipy.sparse.linalg        import inv
from scipy.sparse.linalg        import spsolve, eigsh

from psydac.api.postprocessing  import OutputManager, PostProcessManager

from psydac.fem.basic           import FemField
from psydac.linalg.block        import BlockLinearOperator, BlockVectorSpace
from psydac.linalg.utilities    import array_to_psydac
import os

def get_eigenvalues(nb_eigs, sigma, A_m, M_m):
    """
    Compute the eigenvalues of the matrix A close to sigma and right-hand-side M
    Function seen and adapted from >>> psydac_dev/psydac/feec/multipatch/examples/hcurl_eigen_pbms_conga_2d.py <<< 
    (Commit a748a4d8c1569a8765f6688d228f65ea6073c252)

    Parameters
    ----------
    nb_eigs : int
        Number of eigenvalues to compute
    sigma : float
        Value close to which the eigenvalues are computed
    A_m : sparse matrix
        Matrix A
    M_m : sparse matrix
        Matrix M
    """

    #print()
    print('-----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  ----- ')
    print(
        'computing {0} eigenvalues (and eigenvectors) close to sigma={1} with scipy.sparse.eigsh...'.format(nb_eigs, sigma))
    mode = 'normal'
    which = 'LM'
    ncv = 4 * nb_eigs
    max_shape_splu = 24000
    if A_m.shape[0] >= max_shape_splu:
        raise ValueError(f'Matrix too large.')
        
    eigenvalues, eigenvectors = eigsh(
        A_m, k=nb_eigs, M=M_m, sigma=sigma, mode=mode, which=which, ncv=ncv)

    print("done: eigenvalues found: " + repr(eigenvalues))
    print('-----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----')
    print()

    return eigenvalues, eigenvectors

# ----- Parameters & Settings -----
# Domain Parameters
eps      = 0.15 # inner radius of the toroidal cavity
r        = 0.75 # inner radius of the torus
R        = 1.   # outer radius of the torus

# Discretization Parameters
ncells   = [4, 4, 4]         # number of cells in each direction
# ncells   = [8, 8, 8]         # number of cells in each direction
degree   = [2, 2, 2]            # B-spline degree in each direction
periodic = [False, True, True] # periodicity of the domain

# Correction Potential Tolerance - defines the "accuracy" of the harmonic vector potential
cor_tol = 1e-12

# Save results (in particular also as .vtu)?
save      = True
file_name = f'visu_eps{eps}_n{ncells[0]}_d{degree[0]}_tol{cor_tol}'
assert not os.path.exists(f'hollow_torus_output/{file_name}.static.vtu')

# Print additional timing results
verbose               = True

# Tests
plot_domain           = True
convergence_info      = True
test_normal_harmonic  = True
test_tangent_harmonic = True
# ---------------------------------


# ----- Performance Settings -----
comm    = MPI.COMM_WORLD # not used
backend = PSYDAC_BACKEND_GPYCCEL
# --------------------------------


# ----- Mapping & Domain Definition -----
logical_domain = Cube('C', bounds1=(r+eps, R), bounds2=(0, 2*np.pi), bounds3=(0, 2*np.pi))
class HollowTorus(Mapping):
    _expressions = {'x': '( r + (x1 - r) * cos(x3) ) * cos(x2)',
                    'y': '( r + (x1 - r) * cos(x3) ) * sin(x2)',
                    'z': '(x1 - r) * sin(x3)'}
    _ldim        = 3
    _pdim        = 3

mapping = HollowTorus('HT', r=r)
domain  = mapping(logical_domain)
if plot_domain:
    from sympde.utilities.utils import plot_domain as plot_domain_sympde
    plot_domain_sympde(domain, draw=True, isolines=True)
dim_normal_harmonic_space = 1 # for HollowTorus domain
# ---------------------------------------


# ----- Symbolic Objects -----
boundary = Union(domain.get_boundary(0,-1), domain.get_boundary(0,1))
nn       = NormalVector('nn')

derham   = Derham(domain)

V0, V1, V2, V3 = derham.spaces

u0, v0 = elements_of(V0, names='u0, v0')
u1, v1 = elements_of(V1, names='u1, v1')
u2, v2 = elements_of(V2, names='u2, v2')
u3, v3 = elements_of(V3, names='u3, v3')

m0 = BilinearForm((u0, v0), integral(domain, u0*v0))
m1 = BilinearForm((u1, v1), integral(domain, inner(u1, v1)))
m2 = BilinearForm((u2, v2), integral(domain, inner(u2, v2)))
m3 = BilinearForm((u3, v3), integral(domain, u3*v3))
# ----------------------------


# ----- Psydac Discrete Objects -----
domain_h = discretize(domain, ncells=ncells, periodic=periodic)#, comm=comm) <- causes problems when calling .tosparse()
derham_h = discretize(derham, domain_h, degree=degree)

V0h,  V1h,  V2h,  V3h  = derham_h.spaces
V0cs, V1cs, V2cs, V3cs = [Vh.coeff_space for Vh in derham_h.spaces]

t0 = time.time()
m0h = discretize(m0, domain_h, (V0h, V0h), backend=backend)
m1h = discretize(m1, domain_h, (V1h, V1h), backend=backend)
m2h = discretize(m2, domain_h, (V2h, V2h), backend=backend)
m3h = discretize(m3, domain_h, (V3h, V3h), backend=backend)
t_discretize = time.time() - t0

t0 = time.time()
M0 = m0h.assemble()
M1 = m1h.assemble()
M2 = m2h.assemble()
M3 = m3h.assemble()
t_assembly = time.time() - t0

G, C, D = derham_h.derivatives(kind='linop')

DP0, DP1, DP2, _ = derham_h.dirichlet_projectors(kind='linop')
I0, I1, I2       = [IdentityOperator(Vcs) for Vcs in [V0cs, V1cs, V2cs]]

M0_0     = DP0 @ M0 @ DP0 + (I0 - DP0)
M1_0     = DP1 @ M1 @ DP1 + (I1 - DP1)

t0 = time.time()
M0_0_pc, = derham_h.LST_preconditioners(M0=M0, hom_bc=True)
M1_0_pc, = derham_h.LST_preconditioners(M1=M1, hom_bc=True)
t_pc = time.time() - t0

inv_M0_0 = inverse(M0_0, 'CG', pc=M0_0_pc, maxiter=1000, tol=1e-15)
inv_M1_0 = inverse(M1_0, 'CG', pc=M1_0_pc, maxiter=1000, tol=1e-15)
# -----------------------------------


# ----- Scipy Discrete Objects -----
t0 = time.time()
M0_s = M0.tosparse()
M1_s = M1.tosparse()
M2_s = M2.tosparse()

G_s = G.tosparse()
C_s = C.tosparse()

DP0_s = DP0.tosparse()
DP1_s = DP1.tosparse()

I0_s = I0.tosparse()
I1_s = I1.tosparse()
t_sparse = time.time() - t0

M0_0_s     = DP0_s @ M0_s @ DP0_s + (I0_s - DP0_s)
t0 = time.time()
inv_M0_0_s = inv(M0_0_s.tocsc())
inv_M0_0_s.eliminate_zeros()
t_inv = time.time() - t0
# ----------------------------------


# ----- Psydac & Scipy 1-Hodge-Laplacian -----
# Psydac
S     = C.T @ M2 @ C + M1 @ G @ inv_M0_0 @ DP0 @ G.T @ M1
S_0   = DP1 @ S @ DP1 + (I1 - DP1)
# Scipy
S_s   = C_s.transpose() @ M2_s @ C_s + M1_s @ G_s @ inv_M0_0_s @ DP0_s @ G_s.transpose() @ M1_s
S_0_s = DP1_s @ S_s @ DP1_s + (I1_s - DP1_s)
# --------------------------------------------


# ----- Compute 0 Eigenvector of 1-Hodge-Laplacian -----
t0 = time.time()
eigenvalues, eigenvectors = get_eigenvalues(dim_normal_harmonic_space + 2, 1e-6, S_0_s, None)
t_evs = time.time() - t0
normal_harmonic_field = array_to_psydac(eigenvectors[:,0], V1cs)
Normal_harmonic_field = FemField(V1h, normal_harmonic_field)
# ------------------------------------------------------


# ----- Obtain Lifting Fields 1 & 2 corresponding to tunnels 1 & 2 -----
def get_lifting_field_1():
    dim = V1cs.dimension
    array = np.zeros(dim)

    nx, ny, nz = ncells
    px, py, pz = degree

    Nx = nx+px
    Ny = ny
    Nz = nz

    start = (Nx-1) * Ny * Nz
    for iz in range(Nz):
        index = iz
        array[start+index] = 1

    coeffs = array_to_psydac(array, V1cs)
    return coeffs
def get_lifting_field_2():
    dim = V1cs.dimension
    array = np.zeros(dim)

    nx, ny, nz = ncells
    px, py, pz = degree

    Nx = nx+px
    Ny = ny
    Nz = nz

    iz = int(np.floor(Nz/2))

    start = (Nx-1) * Ny * Nz + Nx*Ny*Nz
    for iy in range(Ny):
        index = (Nx-1)*Ny*Nz + iy*Nz + iz
        array[start+index] = 1

    coeffs = array_to_psydac(array, V1cs)
    return coeffs

lifting_field_1 = get_lifting_field_1()
lifting_field_2 = get_lifting_field_2()

Lifting_field_1 = FemField(V1h, lifting_field_1)
Lifting_field_2 = FemField(V1h, lifting_field_2)
# ----------------------------------------------------------------------


# ----- Projection onto the space orthogonal to the normal harmonic field -----
norm_squared_nhf = normal_harmonic_field.inner(normal_harmonic_field)
dot = lambda x : x - (x.inner(normal_harmonic_field) / norm_squared_nhf) * normal_harmonic_field
P_H1 = MatrixFreeLinearOperator(domain=V1cs, codomain=V1cs, dot=dot, dot_transpose=dot)
# -----------------------------------------------------------------------------


# ----- Remove normal harmonic kernel of 1-Hodge-Laplacian -----
S_0_kernel_free = P_H1 @ S_0 @ P_H1 + (I1 - P_H1)
# --------------------------------------------------------------


# ----- Compute Correction Potentials 1 & 2 -----
#
# Solution of 1-Hodge-Laplace Problem with r.h.s. -curl curl lifting_field_i, i=1, 2
#
maxiter = 20000
inv_S_0_kernel_free = inverse(S_0_kernel_free, 'CG', maxiter=maxiter, tol=cor_tol)

rhs1 = -C.T @ M2 @ C @ lifting_field_1
rhs2 = -C.T @ M2 @ C @ lifting_field_2

rhs1_0 = P_H1 @ DP1 @ rhs1
rhs2_0 = P_H1 @ DP1 @ rhs2

t0 = time.time()
correction_field1 = inv_S_0_kernel_free @ rhs1_0
t1 = time.time()
info1 = inv_S_0_kernel_free.get_info()
t2 = time.time()
correction_field2 = inv_S_0_kernel_free @ rhs2_0
t3 = time.time()
info2 = inv_S_0_kernel_free.get_info()
if convergence_info:
    print(f' Correction Field Convergence Stats')
    print(f' Maxiter = {maxiter} - Tol = {cor_tol}')
    print(f' Correction Field 1 obtained in {t1-t0:.3g}s')
    print(f' {info1}')
    print(f' Correction Field 2 obtained in {t3-t2:.3g}s')
    print(f' {info2}')
    print()
# -----------------------------------------------


# ----- Obtain Harmonic Vector Potentials 1 & 2 -----
harvepo1 = lifting_field_1 + correction_field1
harvepo2 = lifting_field_2 + correction_field2

Harvepo1 = FemField(V1h, harvepo1)
Harvepo2 = FemField(V1h, harvepo2)
# ---------------------------------------------------


# ----- Obtain & Normalize Tangent Harmonic Fields 1 & 2 -----
tangent_harmonic_field1 = C @ harvepo1
tangent_harmonic_field2 = C @ harvepo2

tangent_harmonic_field1 /= np.sqrt(M2.dot_inner(tangent_harmonic_field1, tangent_harmonic_field1))
tangent_harmonic_field2 /= np.sqrt(M2.dot_inner(tangent_harmonic_field2, tangent_harmonic_field2))

Tangent_harmonic_field1 = FemField(V2h, tangent_harmonic_field1)
Tangent_harmonic_field2 = FemField(V2h, tangent_harmonic_field2)
# ------------------------------------------------------------


# ----- Verify that normal_harmonic_field is in fact normal harmonic -----
if test_normal_harmonic:
    weak_D = - inv_M0_0 @ DP0 @ G.T @ M1

    normal_boundary_int = BilinearForm((u1, v1), integral(boundary, inner(cross(nn, u1), cross(nn, v1))))
    normal_boundary_int_h = discretize(normal_boundary_int, domain_h, (V1h, V1h), backend=backend)
    Normal_boundary_int = normal_boundary_int_h.assemble()

    print(f' Test normal harmonic property')
    a = normal_harmonic_field
    l2_norm_normal_trace = np.sqrt(Normal_boundary_int.dot_inner(a, a))

    diff               = a - DP1 @ a
    l2_norm_projection = np.sqrt(M1.dot_inner(diff, diff))

    curl_a        = C @ a
    weak_div_a    = weak_D @ a

    norm_curl     = np.sqrt(M2.dot_inner(curl_a, curl_a))
    norm_weak_div = np.sqrt(M0.dot_inner(weak_div_a, weak_div_a))
    print(f' || A \\times n ||_L2(boundary)      = {l2_norm_normal_trace:.3g}')
    print(f' || A - DP1(A) ||_L2(domain)        = {l2_norm_projection:.3g}')
    print(f' || curl A     ||_L2(domain)        = {norm_curl:.3g}')
    print(f' || weak-div A ||_L2(domain)        = {norm_weak_div:.3g}')
    print()
# ------------------------------------------------------------------------


# ----- Verify that both fields are in fact tangent harmonic -----
if test_tangent_harmonic:
    weak_C = inv_M1_0 @ DP1 @ C.T @ M2

    tangent_boundary_int = BilinearForm((u2, v2), integral(boundary, inner(u2, nn) * inner(v2, nn)))
    tangent_boundary_int_h = discretize(tangent_boundary_int, domain_h, (V2h, V2h), backend=backend)
    Tangent_boundary_int = tangent_boundary_int_h.assemble()

    print(f' Test tangent harmonic property')
    for b in (tangent_harmonic_field1, tangent_harmonic_field2):
        l2_norm_tangent_trace = np.sqrt(Tangent_boundary_int.dot_inner(b, b))

        diff               = b - DP2 @ b
        l2_norm_projection = np.sqrt(M2.dot_inner(diff, diff))

        div_b          = D @ b
        weak_curl_b    = weak_C @ b

        norm_div       = np.sqrt(M3.dot_inner(div_b, div_b))
        norm_weak_curl = np.sqrt(M1.dot_inner(weak_curl_b, weak_curl_b))
        print(f' || B \cdot n   ||_L2(boundary)     = {l2_norm_tangent_trace:.3g}')
        print(f' || B - DP2(B)  ||_L2(domain)       = {l2_norm_projection:.3g}')
        print(f' || div B       ||_L2(domain)       = {norm_div:.3g}')
        print(f' || weak-curl B ||_L2(domain)       = {norm_weak_curl:.3g}')
        print()
# ----------------------------------------------------------------


# ----- Export Fields and prepare for Paraview Visualization -----
if save:
    save_fields = {
        'Normal_harmonic_field'   : Normal_harmonic_field,
        'Tangent_harmonic_field1' : Tangent_harmonic_field1,
        'Tangent_harmonic_field2' : Tangent_harmonic_field2,
        'Lifting_field_1'         : Lifting_field_1,
        'Lifting_field_2'         : Lifting_field_2,
        'Harvepo1'                : Harvepo1,
        'Harvepo2'                : Harvepo2
    }

    os.makedirs('hollow_torus_output', exist_ok=True)
    Om = OutputManager(
        'hollow_torus_output/space_info.yml',
        'hollow_torus_output/field_info.h5',
        comm=comm,
        save_mpi_rank=True, 
        mode='w' 
    )
    Om.add_spaces(V1=V1h)
    Om.add_spaces(V2=V2h)
    Om.export_space_info()
    Om.set_static()
    Om.export_fields(**save_fields)
    Om.close()

    Pm = PostProcessManager(
        domain=domain,
        space_file=f'hollow_torus_output/space_info.yml',
        fields_file=f'hollow_torus_output/field_info.h5',
        comm=comm
    )
    Pm.export_to_vtk(
        f'hollow_torus_output/{file_name}',
        grid=None,
        npts_per_cell=[6,6,6],
        fields=save_fields.keys()
    )
    Pm.close()
# ----------------------------------------------------------------


# ----- Print Timings -----
if verbose:
    txt  = f' Discretization : {t_discretize:.3g}\n'
    txt += f' Assembly       : {t_assembly:.3g}\n'
    txt += f' Preconditioner : {t_pc:.3g}\n'
    txt += f' .tosparse()    : {t_sparse:.3g}\n'
    txt += f' sparse inverse : {t_inv:.3g}\n'
    txt += f' Eigenvectors   : {t_evs:.3g}'
    print(txt)
    print()
# -------------------------
