from mpi4py     import MPI
import numpy    as np
import os
import time
import datetime

from sympde.calculus            import dot, cross
from sympde.expr                import BilinearForm, integral
from sympde.topology            import elements_of, Derham, Cube

from psydac.api.discretization      import discretize
from psydac.api.settings            import PSYDAC_BACKEND_GPYCCEL
from psydac.fem.basic               import FemField

#from altered_q_assembly_code.q_1 import assemble_matrix_q_1
from altered_code.q_2 import assemble_matrix_q_2
from altered_code.q_3 import assemble_matrix_q_3
#from altered_q_assembly_code.q_4 import assemble_matrix_q_4

#from altered_q_assembly_code.q_1_global import assemble_matrix_q_1_global
from altered_code.q_2_global import assemble_matrix_q_2_global
from altered_code.q_3_global import assemble_matrix_q_3_global
#from altered_q_assembly_code.q_4_global import assemble_matrix_q_4_global

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

funcs_sf = [assemble_matrix_q_2, assemble_matrix_q_3]#, assemble_matrix_q_4] [assemble_matrix_q_1 , 
funcs_sfg = [assemble_matrix_q_2_global, assemble_matrix_q_3_global]#, assemble_matrix_q_4_global] [assemble_matrix_q_1_global, 
labels = ['old', 'sf', 'sfg']

degree_list = [[2, 2, 2], [3, 3, 3]]#, [4, 4, 4]] [1, 1, 1], 
ncells_list = [[16, 16, 16], [32, 32, 32]]

new_funcs = [[funcs_sf[i], funcs_sfg[i]] for i in range(len(degree_list))]

periodic = [False, False, False]

backend = PSYDAC_BACKEND_GPYCCEL

domain = Cube('C', bounds1=(0, 1), bounds2=(0, 1), bounds3=(0, 1))
derham = Derham(domain)

def get_A_fun():
    """Get the tuple A = (A1, A2, A3), with each element of A being a function taking x,y,z as input."""

    n = 1
    m = 1
    A0 = 1e04
    mu_tilde = np.sqrt(m**2 + n**2)  

    eta = lambda x, y, z: x**2 * (1-x)**2 * y**2 * (1-y)**2 * z**2 * (1-z)**2

    u1  = lambda x, y, z:  A0 * (n/mu_tilde) * np.sin(np.pi * m * x) * np.cos(np.pi * n * y)
    u2  = lambda x, y, z: -A0 * (m/mu_tilde) * np.cos(np.pi * m * x) * np.sin(np.pi * n * y)
    u3  = lambda x, y, z:  A0 * np.sin(np.pi * m * x) * np.sin(np.pi * n * y)

    A1 = lambda x, y, z: eta(x, y, z) * u1(x, y, z)
    A2 = lambda x, y, z: eta(x, y, z) * u2(x, y, z)
    A3 = lambda x, y, z: eta(x, y, z) * u3(x, y, z)

    A = (A1, A2, A3)
    return A

if mpi_rank == 0:
    if not os.path.isdir('data'):
        os.makedirs('data')
currtime = datetime.datetime.now().strftime("%Y-%m-%d_-_%H%M%S")
f = open(f'data/q_data_{currtime}.txt', 'w')
f.writelines([f'{mpi_size} \t']+[f'{str(labels[i])} \t' for i in range(len(labels))]+['\n'])
f.writelines('degree list:\n')
f.writelines([f'{str(degree[0])} \t' for degree in degree_list]+['\n'])
f.writelines([f'{str(degree[1])} \t' for degree in degree_list]+['\n'])
f.writelines([f'{str(degree[2])} \t' for degree in degree_list]+['\n'])
f.writelines('ncells list:\n')
f.writelines([f'{str(ncells[0])} \t' for ncells in ncells_list]+['\n'])
f.writelines([f'{str(ncells[1])} \t' for ncells in ncells_list]+['\n'])
f.writelines([f'{str(ncells[2])} \t' for ncells in ncells_list]+['\n'])
f.close()

A = get_A_fun()
u1, v1, h = elements_of(derham.V1, names='u1, v1, h')
q = BilinearForm((u1, v1), integral(domain, dot( cross(h, u1), cross(h, v1) )))

# contains old and new matrices, to be compared for almost-equality after the timings are stored
matrices = [[] for label in labels]

for i, degree in enumerate(degree_list):
    timings_degree = [[] for label in labels] # old, sf, sfg
    new_funcs_degree = new_funcs[i] # sf_i, sfg_i
    for j, ncells in enumerate(ncells_list):

        domain_h = discretize(domain, ncells=ncells, periodic=periodic, comm=comm)
        derham_h = discretize(derham, domain_h, degree=degree)

        P0, P1, P2, P3  = derham_h.projectors()
        a = P1(A).coeffs
        A_field = FemField(derham_h.V1, a)

        if mpi_rank == 0:
            print(f'Degree {degree[0]}x{degree[1]}x{degree[2]} ncells {ncells[0]}x{ncells[1]}x{ncells[2]}')
            print('Discretizing q ...')
        q_h = discretize(q, domain_h, (derham_h.V1, derham_h.V1), backend=backend)

        for k, label in enumerate(labels):
            if mpi_rank == 0:
                print(f'Start {label} assembly of A ...')
            if k != 0:
                q_h._func = new_funcs_degree[k-1]
            time.sleep(1)
            start = time.time()
            Q = q_h.assemble(h=A_field)
            stop = time.time()
            
            time_measured = stop-start
            if mpi_rank == 0:
                print(f'{label} time: {time_measured}')
            matrices[k].append((Q.copy(), f'{label}_d_{degree[0]}{degree[1]}{degree[2]}_nc_{ncells[0]}{ncells[1]}{ncells[2]}'))
            del Q
            timings_degree[k].append(time_measured)

    if mpi_rank == 0:
        f = open(f'data/q_data_{currtime}.txt', 'a')
        for k in range(len(labels)):
            f.writelines([f'{str(timings_degree[k][i])} \t' for i in range(len(timings_degree[k]))]+['\n'])
        f.close()
        print()

if mpi_rank == 0:
    print('Testing for equality of old and new matrices')
l = int( (len(labels)-1) * len(matrices[0]))
for i, k in enumerate(labels[1:]):
    for j in range(len(matrices[0])):
        old = matrices[0][j]
        new = matrices[i+1][j]
        diff = old[0] - new[0]
        diffs = diff.tosparse()
        diffs.eliminate_zeros()
        abs_diffs = np.absolute(diffs)
        max_diff = np.max(abs_diffs)
        if mpi_rank == 0:
            print(f'{len(matrices[0])*i+j+1}/{l}: max. abs. diff.: {max_diff} | {old[1]} vs. {new[1]}')
