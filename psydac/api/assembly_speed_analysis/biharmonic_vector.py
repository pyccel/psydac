import  time
from    mpi4py  import  MPI
import  numpy   as      np

from    sympy           import pi, symbols, sin, atan2, sqrt

from    sympde.calculus import grad, dot, laplace
from    sympde.topology import ScalarFunctionSpace, Domain, NormalVector, elements_of
from    sympde.expr import BilinearForm, LinearForm, Norm, SemiNorm, EssentialBC, integral, find

from    psydac.api.discretization import discretize
from    psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL

from    generate_mapping import make_quarter_torus_geometry

"""
Measure the matrix-vector-speed for the biharmonic problem matrix.

"""

def run_biharmonic_3d_dir(filename, solution, f, 
                          backend=None, comm=None, 
                          rep=3):

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    domain   = Domain.from_file(filename)
    boundary = domain.boundary

    V    = ScalarFunctionSpace('V', domain)
    u, v = elements_of(V, names='u, v')
    nn   = NormalVector('nn')

    a    = BilinearForm((u, v), integral(domain, laplace(u) * laplace(v)))
    l    = LinearForm(v, integral(domain, f * v))
    bc   = [EssentialBC(   u , 0, boundary)] + [EssentialBC(dot(grad(u), nn), 0, boundary)]

    equation = find(u, forall=v, lhs=a(u, v), rhs=l(v), bc=bc)

    error  = u - solution
    l2norm =     Norm(error, domain, kind='l2')
    h1norm = SemiNorm(error, domain, kind='h1')
    h2norm = SemiNorm(error, domain, kind='h2')

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    domain_h    = discretize(domain, filename=filename, comm=comm)
    Vh          = discretize(V, domain_h)
    equation_h  = discretize(equation, domain_h, [Vh, Vh], backend=backend) # , fast_assembly=False)

    avr_matvec_speed = 0.0
    t_arr = []

    matrix = equation_h.lhs.assemble()
    rng = np.random.default_rng(seed=2)
    v = np.zeros(Vh.coeff_space.dimension)
    rng.random(out=v)
    n = np.linalg.norm(v)
    v /= n
    norm_vector = np.linalg.norm(v)
    print(norm_vector)
    from psydac.linalg.utilities import array_to_psydac
    vector = array_to_psydac(v, Vh.coeff_space)
    
    for _ in range(rep):

        t0 = time.time()
        matrix @ vector
        t1 = time.time()

        dt = t1 - t0

        avr_matvec_speed += dt
        t_arr.append(dt)

    avr_matvec_speed /= rep

    if ((comm is not None and comm.rank==0) or (comm is None)):
        size = 1 if ((comm is None) or (comm is not None and comm.size==1)) else comm.size 
        print(f'Out of {rep} using {size} process(es): Max. speed {max(t_arr):.3g} - Min. speed {min(t_arr):.3g} - Avr. speed {avr_matvec_speed:.3g}')

    if False:
        l2norm_h = discretize(l2norm, domain_h, Vh, backend=backend)
        h1norm_h = discretize(h1norm, domain_h, Vh, backend=backend)
        h2norm_h = discretize(h2norm, domain_h, Vh, backend=backend)

        uh = equation_h.solve()

        l2_error = l2norm_h.assemble(u=uh)
        h1_error = h1norm_h.assemble(u=uh)
        h2_error = h2norm_h.assemble(u=uh)

        print(l2_error)
        print(h1_error)
        print(h2_error)

        return l2_error, h1_error, h2_error, avr_assembly_speed
    else:
        return avr_matvec_speed

#==============================================================================

comm        = MPI.COMM_WORLD
backend     = PSYDAC_BACKEND_GPYCCEL

x,y,z       = symbols('x,y,z', real=True)
solution    = (sin(2*pi*sqrt(x**2+y**2)) * sin(2*atan2(y, x)) * sin(pi*z))**2
f           = laplace(laplace(solution))

ncells      = [32, 32, 32]
degree_list = [[2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]
rep_list    = [20, 20, 20, 20]
unique_name = 'biharmonic_matvec1_Raven'

avr_times = []

for degree, rep in zip(degree_list, rep_list):

    filename    = make_quarter_torus_geometry(ncells, degree, comm=comm)

    if False:
        l2_error, h1_error, h2_error, avr_assembly_speed = run_biharmonic_3d_dir(filename, solution, f, 
                                                                                 backend=backend, comm=comm, 
                                                                                 rep=rep)
    else:
        avr_matvec_speed = run_biharmonic_3d_dir(filename, solution, f, 
                                                   backend=backend, comm=comm, 
                                                   rep=rep)

    avr_times.append(avr_matvec_speed)

txt = f'MPI size {comm.size}; degree_list {degree_list}; rep_list {rep_list}; Quarter Torus\n'
for t in avr_times:
    txt += f'{t}\n'

if comm.rank == 0:
    f = open(f'{unique_name}_n{comm.size}.txt', 'w')
    f.writelines(txt)
    f.close()
