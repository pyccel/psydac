import  os
import  time
from    mpi4py  import  MPI
import  numpy   as      np

from    sympy   import  sin

from sympde.calculus    import dot, grad, curl, div # , laplace
from sympde.expr        import BilinearForm, integral
from sympde.topology    import elements_of, Cube, Mapping, ScalarFunctionSpace, Domain, Derham

from psydac.api.discretization  import discretize
from psydac.api.settings        import PSYDAC_BACKEND_GPYCCEL
from psydac.cad.geometry        import Geometry
from psydac.mapping.discrete    import SplineMapping

def make_half_hollow_torus_geometry_3d(ncells, degree, comm=None):

    if comm is not None:
        mpi_rank = comm.Get_rank()
    else:
        mpi_rank = 0
    
    class HalfHollowTorusMapping3D(Mapping):

        _expressions = {'x': '(R + r * x1 * cos(2*pi*x3)) * cos(pi*x2)',
                        'y': '(R + r * x1 * cos(2*pi*x3)) * sin(pi*x2)',
                        'z': 'r * x1 * sin(2*pi*x3)'}

        _ldim        = 3
        _pdim        = 3

    if (ncells[0] == ncells[1]) and (ncells[0] == ncells[2]):
        nc = f'{ncells[0]}'
    else:
        nc = f'{ncells[0]}_{ncells[1]}_{ncells[2]}'
    
    if (degree[0] == degree[1]) and (degree[0] == degree[2]):
        de = f'{degree[0]}'
    else:
        de = f'{degree[0]}_{degree[1]}_{degree[2]}'

    name = f'hht_3d_nc_{nc}_d_{de}'

    domain = Cube('S', bounds1=(0,1), bounds2=(0,1), bounds3=(0,1))
    domain_h = discretize(domain, ncells=ncells, comm=comm)

    V = ScalarFunctionSpace('V', domain)
    V_h = discretize(V, domain_h, degree=degree)

    M = HalfHollowTorusMapping3D('M', R=2, r=1)
    map_discrete = SplineMapping.from_mapping(V_h, M.get_callable_mapping())

    geometry = Geometry.from_discrete_mapping(map_discrete, comm=comm)

    if mpi_rank == 0:
        if not os.path.isdir('geometry'):
            os.makedirs('geometry')
            os.makedirs('geometry/files')
        else:
            if not os.path.isdir('geometry/files'):
                os.makedirs('geometry/files')

    geometry.export(f'geometry/files/{name}.h5')

    return f'geometry/files/{name}.h5'

def make_collela_geometry_3d(ncells, degree, eps, comm=None):

    if comm is not None:
        mpi_rank = comm.Get_rank()
    else:
        mpi_rank = 0
    
    class CollelaMapping3D(Mapping):

        _expressions = {'x': 'k1*(x1 + eps*sin(2.*pi*x1)*sin(2.*pi*x2)) - 1',
                        'y': 'k2*(x2 + eps*sin(2.*pi*x1)*sin(2.*pi*x2)) - 1',
                        'z': 'k3*x3 - 1'}

        _ldim        = 3
        _pdim        = 3

    if (ncells[0] == ncells[1]) and (ncells[0] == ncells[2]):
        nc = f'{ncells[0]}'
    else:
        nc = f'{ncells[0]}_{ncells[1]}_{ncells[2]}'
    
    if (degree[0] == degree[1]) and (degree[0] == degree[2]):
        de = f'{degree[0]}'
    else:
        de = f'{degree[0]}_{degree[1]}_{degree[2]}'

    name = f'collela_3d_eps_{eps}_nc_{nc}_d_{de}'

    domain = Cube('S', bounds1=(0,1), bounds2=(0,1), bounds3=(0,1))
    domain_h = discretize(domain, ncells=ncells, comm=comm)

    V = ScalarFunctionSpace('V', domain)
    V_h = discretize(V, domain_h, degree=degree)

    M = CollelaMapping3D('M', k1=2, k2=2, k3=2, eps=eps)
    map_discrete = SplineMapping.from_mapping(V_h, M.get_callable_mapping())

    geometry = Geometry.from_discrete_mapping(map_discrete, comm=comm)

    if mpi_rank == 0:
        if not os.path.isdir('geometry'):
            os.makedirs('geometry')
            os.makedirs('geometry/files')
        else:
            if not os.path.isdir('geometry/files'):
                os.makedirs('geometry/files')

    geometry.export(f'geometry/files/{name}.h5')

    return f'geometry/files/{name}.h5'

def build_matrices(mapping_option, verbose, backend, comm):

    ncells      = [12, 8, 16]
    degree      = [2, 4, 3]
    periodic    = [False, True, False]
    eps         = 0.1

    mpi_rank = comm.Get_rank() if comm is not None else 0

    if mapping_option is None:

        domain = Cube('C', bounds1=(0,1), bounds2=(0,1), bounds3=(0,1))
        derham = Derham(domain)

        domain_h = discretize(domain, ncells=ncells, periodic=periodic, comm=comm)
        derham_h = discretize(derham, domain_h, degree=degree)

    elif mapping_option == 'Bspline':

        filename = make_collela_geometry_3d(ncells, degree, eps, comm=comm)

        domain = Domain.from_file(filename)
        derham = Derham(domain)

        domain_h = discretize(domain, filename=filename, comm=comm)
        derham_h = discretize(derham, domain_h, degree=domain.mapping.get_callable_mapping().space.degree)

    elif mapping_option == 'analytical':

        class HalfHollowTorusMapping3D(Mapping):
            _expressions = {'x': '(R + r * x1 * cos(2*pi*x3)) * cos(pi*x2)',
                            'y': '(R + r * x1 * cos(2*pi*x3)) * sin(pi*x2)',
                            'z': 'r * x1 * sin(2*pi*x3)'}
            _ldim        = 3
            _pdim        = 3
        mapping = HalfHollowTorusMapping3D('M', R=2, r=1)
        logical_domain = Cube('C', bounds1=(0,1), bounds2=(0,1), bounds3=(0,1))

        domain = mapping(logical_domain)
        derham = Derham(domain)

        domain_h = discretize(domain, ncells=ncells, periodic=periodic, comm=comm)
        derham_h = discretize(derham, domain_h, degree=degree)
    
    x, y, z         = domain.coordinates
    gamma           = (x*y*z + sin(x*y+z)**2)

    int_0           = lambda expr: integral(domain, expr)

    V0  = derham.V0
    V0h = derham_h.V0
    V1  = derham.V1
    V1h = derham_h.V1
    V2  = derham.V2
    V2h = derham_h.V2
    V3  = derham.V3
    V3h = derham_h.V3

    u1, u2 = elements_of(V0, names='u1, u2')
    v1, v2 = elements_of(V1, names='v1, v2')
    w1, w2 = elements_of(V2, names='w1, w2')
    f1, f2 = elements_of(V3, names='f1, f2')

    spaces = {'V0':{'Vh':V0h, 'funcs':[u1, u2]},
              'V1':{'Vh':V1h, 'funcs':[v1, v2]},
              'V2':{'Vh':V2h, 'funcs':[w1, w2]},
              'V3':{'Vh':V3h, 'funcs':[f1, f2]}}

    bilinear_forms = {  'gradgrad':{'trial':'V0',
                                    'test':'V0',
                                    'expr':dot(grad(u1), grad(u2))}, 
                        'curlcurl':{'trial':'V1',
                                    'test':'V1',
                                    'expr':dot(curl(v1), curl(v2))},
                        'weighted_hdiv_mass':{'trial':'V2',
                                              'test':'V2',
                                              'expr':dot(w1, w2)*gamma},
                        'dot(v1,v2)':{'trial':'V1',
                                      'test':'V1',
                                      'expr':dot(v1, v2)}
                     }
    
    bilinear_forms_to_test = bilinear_forms # ('curlcurl', )
    
    for bf in bilinear_forms_to_test:
        value = bilinear_forms[bf]
        trial   = value['trial']
        test    = value['test']
        expr    = value['expr']

        Vh      = spaces[trial]['Vh']
        Wh      = spaces[test] ['Vh']

        u       = spaces[trial]['funcs'][0]
        v       = spaces[test] ['funcs'][1]

        a = BilinearForm((u, v), int_0(expr))

        t0 = time.time()
        a_h = discretize(a, domain_h, (Vh, Wh), backend=backend, new_assembly='test')
        t1 = time.time()
        disc_time_old = t1-t0

        t0 = time.time()
        A_old = a_h.assemble()
        t1 = time.time()
        time_old = t1-t0

        t0 = time.time()
        a_h = discretize(a, domain_h, (Vh, Wh), backend=backend)
        t1 = time.time()
        disc_time_new = t1-t0

        t0 = time.time()
        A_new = a_h.assemble()
        t1 = time.time()
        time_new = t1-t0

        #A_old_norm = np.linalg.norm(A_old.toarray())
        #err = np.linalg.norm((A_old-A_new).toarray())
        #rel_err = err / A_old_norm

        if verbose and (mpi_rank == 0):
            print(f' >>> Mapping Option : {mapping_option} ')
            print(f' >>> BF {bf} ')
            print(f' >>> Discretization in: Old {disc_time_old:.3g}\t\t|| New {disc_time_new:.3g} ')
            print(f' >>> Assembly in:       Old {time_old:.3g}\t\t|| New {time_new:.3g}\t\t|| Old/New {time_old/time_new:.3g} ')
            #print(f' >>>      Error:            {err:.3g} ')
            #print(f' >>> Rel. Error:            {rel_err:.3g} ')
            print()
        
        #assert rel_err < 1e-9 # arbitrary rel. error bound

#==============================================================================

verbose = True

mapping_options = [None, 'analytical', 'Bspline']

comm    = MPI.COMM_WORLD
backend = PSYDAC_BACKEND_GPYCCEL

for option in mapping_options:

    build_matrices(mapping_option=option, verbose=verbose, backend=backend, comm=comm)
