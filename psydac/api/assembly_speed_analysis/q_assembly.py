import  time
import  numpy   as      np
from    mpi4py  import  MPI

from    sympde.calculus import dot, cross
from    sympde.topology import Domain, elements_of, Derham
from    sympde.expr     import BilinearForm, integral

from    psydac.api.discretization import discretize
from    psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL
from    psydac.fem.basic          import FemField

from    generate_mapping import make_quarter_torus_geometry

"""
Measure the assembly speed of the Q matrix appearing in MHD relaxation.

"""

def run_q(filename, degree, backend=None, comm=None, rep=3):

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    prin = True if comm is None or comm.rank==0 else False

    domain  = Domain.from_file(filename)
    derham  = Derham(domain)

    V1          = derham.V1
    u1, v1, h   = elements_of(V1, names='u1, v1, h')

    q       = BilinearForm((u1, v1), integral(domain, dot( cross(h, u1), cross(h, v1) )))

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    domain_h    = discretize(domain, filename=filename, comm=comm)
    derham_h    = discretize(derham, domain_h, degree=degree)
    V1h         = derham_h.V1

    t0 = time.time()
    q_h         = discretize(q, domain_h, (V1h, V1h), backend=backend)
    t1 = time.time()
    if prin:
        print(f'q discretized in {t1-t0:.3g}')

    rng = np.random.default_rng(seed=2)
    v = np.zeros(V1h.coeff_space.dimension)
    rng.random(out=v)
    n = np.linalg.norm(v)
    v /= n
    from psydac.linalg.utilities import array_to_psydac
    vector = array_to_psydac(v, V1h.coeff_space)
    H = FemField(V1h, vector)

    avr_assembly_speed = 0.0
    t_arr = []

    for i in range(rep+1):

        t0 = time.time()
        q_h.assemble(h=H)
        t1 = time.time()

        dt = t1-t0

        # Assumption: For some reason the first assembly is slightly slower than the following assemblies.
        if i != 0:
            avr_assembly_speed += dt
            t_arr.append(dt)
    
    avr_assembly_speed /= rep

    if prin:
        size = 1 if ((comm is None) or (comm is not None and comm.size==1)) else comm.size 
        print(f'Out of {rep} using {size} process(es): Max. speed {max(t_arr):.3g} - Min. speed {min(t_arr):.3g} - Avr. speed {avr_assembly_speed:.3g}')
        print(t_arr)
        print()

    return avr_assembly_speed

#==============================================================================

def run(ncells):
    comm        = MPI.COMM_WORLD
    backend     = PSYDAC_BACKEND_GPYCCEL

    #ncells      = [32, 32, 32]
    degree_list = [[2, 2, 2], [3, 3, 3]]#, [4, 4, 4], [5, 5, 5]]
    rep_list    = [20, 20]#, 5, 2]
    unique_name = 'q_assembly_data'

    avr_times = []

    for degree, rep in zip(degree_list, rep_list):

        filename    = make_quarter_torus_geometry(ncells, degree, comm=comm)

        avr_assembly_speed = run_q(filename, degree, backend=backend, comm=comm, rep=rep)

        avr_times.append(avr_assembly_speed)

    txt = f'Q assembly speed: MPI size {comm.size}; degree_list {degree_list}; rep_list {rep_list}; Quarter Torus\n'
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
        description = "Measure Q assembly speed"
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
