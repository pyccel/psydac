import  time
import  numpy   as      np
from    mpi4py  import  MPI

from    sympde.calculus import dot, cross
from    sympde.topology import Domain, elements_of, Derham, Cube
from    sympde.expr     import BilinearForm, integral

from    psydac.api.discretization import discretize
from    psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL
from    psydac.fem.basic          import FemField
from    psydac.linalg.basic       import IdentityOperator
from    psydac.linalg.solvers     import inverse
from    psydac.linalg.utilities   import array_to_psydac

from    psydac.api.assembly_speed_analysis.generate_mapping import make_quarter_torus_geometry
from    psydac.api.assembly_speed_analysis.utils            import HcurlBoundaryProjector_3D, HdivBoundaryProjector_3D, get_LST_pcs

"""
Measure the assembly speed of the Q matrix appearing in MHD relaxation.

"""

def run_s(filename, degree, ncells, backend=None, comm=None, rep=3):

    prin        = True if comm is None or comm.rank==0 else False
    maxiter_M1  = 1000
    tol_M1      = 1e-12

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    domain  = Domain.from_file(filename)
    derham  = Derham(domain)

    V0          = derham.V0
    V1          = derham.V1
    V2          = derham.V2
    V3          = derham.V3
    Vs = [V0, V1, V2, V3]

    u1, v1, h   = elements_of(V1, names='u1, v1, h')
    u2, v2      = elements_of(V2, names='u2, v2')

    a1      = BilinearForm((u1, v1), integral(domain, dot(u1, v1)))
    a2      = BilinearForm((u2, v2), integral(domain, dot(u2, v2)))
    q       = BilinearForm((u1, v1), integral(domain, dot( cross(h, u1), cross(h, v1) )))

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    domain_h    = discretize(domain, filename=filename, comm=comm)
    derham_h    = discretize(derham, domain_h, degree=degree)

    V0h         = derham_h.V0
    V1h         = derham_h.V1
    V2h         = derham_h.V2
    V3h         = derham_h.V3
    Vhs = [V0h, V1h, V2h, V3h]

    V0cs        = V0h.coeff_space
    V1cs        = V1h.coeff_space
    V2cs        = V2h.coeff_space
    V3cs        = V3h.coeff_space
    Vcs = [V0cs, V1cs, V2cs, V3cs]

    t0 = time.time()
    a1_h        = discretize(a1, domain_h, (V1h, V1h), backend=backend)
    t1 = time.time()
    if prin:
        print(f'a1 discretized in {t1-t0:.3g}')
    t0 = time.time()
    a2_h        = discretize(a2, domain_h, (V2h, V2h), backend=backend)
    t1 = time.time()
    if prin:
        print(f'a2 discretized in {t1-t0:.3g}')

    t0 = time.time()
    M1      = a1_h.assemble()
    t1 = time.time()
    if prin:
        print(f'M1 assembled in {t1-t0:.3g}')
    t0 = time.time()
    M2      = a2_h.assemble()
    t1 = time.time()
    if prin:
        print(f'M2 assembled in {t1-t0:.3g}')

    _, C, _ = derham_h.derivatives_as_matrices
    Ct      = C.T

    I1      = IdentityOperator(V1cs)
    I2      = IdentityOperator(V2cs)
    Pcurl   = HcurlBoundaryProjector_3D(V1cs, V1, periodic=[False, False, False])
    Pdiv    = HdivBoundaryProjector_3D (V2cs, V2, periodic=[False, False, False])
    IPcurl  = I1 - Pcurl
    IPdiv   = I2 - Pdiv

    M1_0 = Pcurl @ M1 @ Pcurl + IPcurl

    logical_domain = Cube('C', bounds1=(0.5, 1), bounds2=(0, np.pi/2), bounds3=(0, 1))
    t0 = time.time()
    _, M1_pc, _ = get_LST_pcs(None, M1_0, None, logical_domain, Vs, Vhs, Vcs, ncells, degree, [False, False, False], comm, backend)
    t1 = time.time()
    if prin:
        print(f'pc obtained in {t1-t0:.3g}')

    M1_inv = inverse(M1_0, 'pcg', pc=M1_pc, maxiter=maxiter_M1, tol=tol_M1)

    C_0 = C @ Pcurl
    Ct_0 = Pcurl @ Ct

    rng = np.random.default_rng(seed=2)
    v1 = np.zeros(V1cs.dimension)
    v2 = np.zeros(V2cs.dimension)
    rng.random(out=v1)
    rng.random(out=v2)
    vector1 = array_to_psydac(v1, V1cs)
    vector2 = array_to_psydac(v2, V2cs)
    vector1 = Pcurl @ vector1
    vector2 = Pdiv @ vector2
    vector1 /= np.linalg.norm(vector1.toarray())
    vector2 /= np.linalg.norm(vector2.toarray())
    vector2_out = vector2.copy()
    vector2_out *= 0.0
    H = FemField(V1h, vector1)


    t0 = time.time()
    q_h         = discretize(q , domain_h, (V1h, V1h), backend=backend)
    t1 = time.time()
    if prin:
        print(f'q  discretized in {t1-t0:.3g}')
    t0 = time.time()
    Qp = q_h.assemble(h=H)
    t1 = time.time()
    if prin:
        print(f'Q assembled in {t1-t0:.3g}')


    dt = 1.5e-8
    S = Pdiv @ (M2 + (dt/2) * M2 @ C_0 @ M1_inv @ Qp @ M1_inv @ Ct_0 @ M2) @ Pdiv + IPdiv

    avr_matvec_speed = 0.0
    t_arr = []

    if prin:
        print()
        print('Start measuring the Matrix Vector Product Speed')
        print('No more temporaries should be created from here on')

    for i in range(rep):

        t0 = time.time()
        S.dot(vector2, out=vector2_out)
        t1 = time.time()

        dt = t1-t0

        avr_matvec_speed += dt
        t_arr.append(dt)

    if prin:
        print()
        print('Done.')
        print()
    
    avr_matvec_speed /= rep

    if prin:
        size = 1 if ((comm is None) or (comm is not None and comm.size==1)) else comm.size 
        print(f'Out of {rep} using {size} process(es): Max. speed {max(t_arr):.3g} - Min. speed {min(t_arr):.3g} - Avr. speed {avr_matvec_speed:.3g}')
        print(t_arr)
        print()

    return avr_matvec_speed

#==============================================================================

def run(ncells):
    comm        = MPI.COMM_WORLD
    backend     = PSYDAC_BACKEND_GPYCCEL

    degree_list = [[2, 2, 2], [3, 3, 3], [4, 4, 4]] # , [5, 5, 5]]
    rep_list    = [20, 20, 10]#, 2]
    unique_name = 's_matvec_weak_data'

    avr_times = []

    for degree, rep in zip(degree_list, rep_list):

        filename    = make_quarter_torus_geometry(ncells, degree, comm=comm)

        avr_matvec_speed = run_s(filename, degree, ncells, backend=backend, comm=comm, rep=rep)

        avr_times.append(avr_matvec_speed)

    txt = f'S matvec speed: MPI size {comm.size}; degree_list {degree_list}; ncells {ncells}; rep_list {rep_list}; Quarter Torus\n'
    for t in avr_times:
        txt += f'{t}\n'

    if comm.rank == 0:
        f = open(f'{unique_name}_n{comm.size}.txt', 'w')
        f.writelines(txt)
        f.close()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        description = "Measure S Matrix Vector Product speed"
    )

    parser.add_argument('-n', '--ncells',
        type    = int,
        nargs   = '+',
        default = None,
        dest    = 'ncells',
        help    = 'Bspline cells in each direction'
    )

    args = parser.parse_args()

    # Run simulation
    namespace = run(**vars(args))
