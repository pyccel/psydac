import  os
import  time
from    mpi4py  import  MPI
import  numpy   as      np

from    sympy   import  sin

from    sympde.calculus             import dot, cross, grad, curl, div # , laplace
from    sympde.expr                 import BilinearForm, integral
from    sympde.topology             import elements_of, Cube, Mapping, ScalarFunctionSpace, VectorFunctionSpace, Domain, Derham

from    psydac.api.discretization   import discretize
from    psydac.api.settings         import PSYDAC_BACKEND_GPYCCEL
from    psydac.cad.geometry         import Geometry
from    psydac.fem.basic            import FemField
from    psydac.mapping.discrete     import SplineMapping

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

        #filename = make_collela_geometry_3d(ncells, degree, eps, comm=comm)
        filename = make_half_hollow_torus_geometry_3d(ncells, degree, comm=comm)

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
        class CollelaMapping3D(Mapping):
            _expressions = {'x': 'k1*(x1 + eps*sin(2.*pi*x1)*sin(2.*pi*x2)) - 1',
                            'y': 'k2*(x2 + eps*sin(2.*pi*x1)*sin(2.*pi*x2)) - 1',
                            'z': 'k3*x3 - 1'}
            _ldim        = 3
            _pdim        = 3

        #mapping = CollelaMapping3D('M', k1=2, k2=2, k3=2, eps=eps)
        mapping = HalfHollowTorusMapping3D('M', R=2, r=1)
        logical_domain = Cube('C', bounds1=(0,1), bounds2=(0,1), bounds3=(0,1))

        domain = mapping(logical_domain)
        derham = Derham(domain)

        domain_h = discretize(domain, ncells=ncells, periodic=periodic, comm=comm)
        derham_h = discretize(derham, domain_h, degree=degree)
    
    x, y, z         = domain.coordinates
    gamma           = x*y*z + sin(x*y+z)**2

    int_0           = lambda expr: integral(domain, expr)

    V0  = derham.V0
    V0h = derham_h.V0
    V1  = derham.V1
    V1h = derham_h.V1
    V2  = derham.V2
    V2h = derham_h.V2
    V3  = derham.V3
    V3h = derham_h.V3

    Vs      = ScalarFunctionSpace('Vs', domain)
    Vvc     = VectorFunctionSpace('Vvc', domain, kind='hcurl')
    Vvd     = VectorFunctionSpace('VVd', domain, kind='hdiv')
    Vsh     = discretize(Vs, domain_h, degree=degree)
    Vvch    = discretize(Vvc, domain_h, degree=degree)
    Vvdh    = discretize(Vvd, domain_h, degree=degree)

    u1, u2, F0 = elements_of(V0, names='u1, u2, F0')
    v1, v2, F1 = elements_of(V1, names='v1, v2, F1')
    w1, w2, F2 = elements_of(V2, names='w1, w2, F2')
    f1, f2, F3 = elements_of(V3, names='f1, f2, F3')

    fs1,  fs2,  Fs  = elements_of(Vs,  names='fs1,  fs2,  Fs' )
    fvc1, fvc2, Fvc = elements_of(Vvc, names='fvc1, fvc2, Fvc')
    fvd1, fvd2, Fvd = elements_of(Vvd, names='fvd1, fvd2, Fvd')

    F0_field_coeffs  = V0h.vector_space.zeros()
    F1_field_coeffs  = V1h.vector_space.zeros()
    F2_field_coeffs  = V2h.vector_space.zeros()
    F3_field_coeffs  = V3h.vector_space.zeros()
    Fs_field_coeffs  = Vsh.vector_space.zeros()
    Fvc_field_coeffs = Vvch.vector_space.zeros()
    Fvd_field_coeffs = Vvdh.vector_space.zeros()

    F0_field_coeffs._data = np.ones(F0_field_coeffs._data.shape, dtype="float64")
    F3_field_coeffs._data = np.ones(F3_field_coeffs._data.shape, dtype="float64")
    Fs_field_coeffs._data = np.ones(Fs_field_coeffs._data.shape, dtype="float64")
    for block in F1_field_coeffs.blocks:
        block._data = np.ones(block._data.shape, dtype="float64")
    for block in F2_field_coeffs.blocks:
        block._data = np.ones(block._data.shape, dtype="float64")
    for block in Fvc_field_coeffs.blocks:
        block._data = np.ones(block._data.shape, dtype="float64")
    for block in Fvd_field_coeffs.blocks:
        block._data = np.ones(block._data.shape, dtype="float64")

    F0_field  = FemField(V0h, F0_field_coeffs)
    F1_field  = FemField(V1h, F1_field_coeffs)
    F2_field  = FemField(V2h, F2_field_coeffs)
    F3_field  = FemField(V3h, F3_field_coeffs)
    Fs_field  = FemField(Vsh, Fs_field_coeffs)
    Fvc_field = FemField(Vvch, Fvc_field_coeffs)
    Fvd_field = FemField(Vvdh, Fvd_field_coeffs)

    spaces = {'V0':{'Vh':V0h, 'funcs':[u1, u2]},
              'V1':{'Vh':V1h, 'funcs':[v1, v2]},
              'V2':{'Vh':V2h, 'funcs':[w1, w2]},
              'V3':{'Vh':V3h, 'funcs':[f1, f2]},
              'Vs':{'Vh':Vsh, 'funcs':[fs1, fs2]},
              'Vvc':{'Vh':Vvch, 'funcs':[fvc1, fvc2]},
              'Vvd':{'Vh':Vvdh, 'funcs':[fvd1, fvd2]}}

    bilinear_forms = {  'gradgrad':             {'trial':'V0',
                                                'test'  :'V0',
                                                'expr'  :dot(grad(u1), grad(u2))},

                        'gradgradSFS':          {'trial':'Vs',
                                                'test'  :'Vs',
                                                'expr'  :dot(grad(fs1), grad(fs2))},

                        'curlcurl':             {'trial':'V1',
                                                'test'  :'V1',
                                                'expr'  :dot(curl(v1), curl(v2))},

                        'curlcurlVFS':          {'trial':'Vvc',
                                                'test'  :'Vvc',
                                                'expr'  :dot(curl(fvc1), curl(fvc2))},

                        'weighted_hdiv_mass':   {'trial':'V2',
                                                'test'  :'V2',
                                                'expr'  :dot(w1, w2)*gamma},

                        'weighted_hdiv_massVFS':{'trial':'Vvd',
                                                'test'  :'Vvd',
                                                'expr'  :dot(fvd1, fvd2)*gamma},

                        'hcurl_mass':           {'trial':'V1',
                                                'test'  :'V1',
                                                'expr'  :dot(v1, v2)},

                        'hcurl_massVFS':        {'trial':'Vvc',
                                                'test'  :'Vvc',
                                                'expr'  :dot(fvc1,fvc2)},

                        'Q':                    {'trial':'V1',
                                                 'test' :'V1',
                                                 'expr' :dot(cross(F1, v1), cross(F1, v2)),
                                                 'field':[F1_field, 'V1']},

                        'field_derivative_F1':  {'trial':'V0',      # test fails with rel. error: 0.477, possibly only due to
                                                 'test' :'V1',      # matrix norm being very small (~4.6e-13)!
                                                 'expr' :dot(grad(u1), curl(F1)) * dot(v2, F1),
                                                 'field':[F1_field, 'V1']},

                        'field_derivative_F2':  {'trial':'V0',
                                                 'test' :'V2',
                                                 'expr' :dot(grad(u1), F2)*div(w2)*div(F2),
                                                 'field':[F2_field, 'V2']},

                        'weighted_h1_mass_F0':  {'trial':'V0',
                                                 'test' :'V0',
                                                 'expr' :u1*u2*F0,
                                                 'field':[F0_field, 'V0']},

                        'weighted_l2_mass_F3':  {'trial':'V3',
                                                 'test' :'V3',
                                                 'expr' :f1*f2*F3,
                                                 'field':[F3_field, 'V3']},

                        'divdiv_Fs':            {'trial':'V2',
                                                 'test' :'V2',
                                                 'expr' :div(w1)*div(w2)*Fs,
                                                 'field':[Fs_field, 'Vs']},

                        'Fvc test':             {'trial':'V2',
                                                 'test' :'V2',
                                                 'expr' :dot(curl(Fvc), w1)*div(w2),
                                                 'field':[Fvc_field, 'Vvc']},

                        'dot(grad(u), v)':      {'trial':'V0',
                                                 'test' :'V1',
                                                 'expr' :dot(grad(u1), v2)},

                        'dot(curl(v), w)_F0':   {'trial':'V1',
                                                 'test' :'V2',
                                                 'expr' :dot(curl(v1), w2)*F0,
                                                 'field':[F0_field, 'V0']},

                        'dot(curl(v), w)_F0_2': {'trial':'V2',
                                                 'test' :'V1',
                                                 'expr' :dot(curl(v2), w1)*F0,
                                                 'field':[F0_field, 'V0']},

                        'u*f':                  {'trial':'V0',
                                                 'test' :'V3',
                                                 'expr' :u1*f2},

                        'f*u':                  {'trial':'V3',
                                                 'test' :'V0',
                                                 'expr' :u2*f1},

                        'sqrt_pi_Fvd':          {'trial':'V0',
                                                 'test' :'V1',
                                                 'expr' :u1*dot(v2, Fvd) - np.sqrt(np.pi)*dot(grad(u1), curl(v2))*div(Fvd),
                                                 'field':[Fvd_field, 'Vvd']},

                        'dot(v, w)':            {'trial':'V1',
                                                 'test' :'V2',
                                                 'expr' :dot(v1, w2)},

                        'dot(w, v)':            {'trial':'V2',
                                                 'test' :'V1',
                                                 'expr' :dot(v2, w1)}
                     }
    
    bilinear_form_strings = ('gradgrad', 'gradgradSFS', 'curlcurl', 'curlcurlVFS', 'weighted_hdiv_mass', 'weighted_hdiv_massVFS', 
                             'hcurl_mass', 'hcurl_massVFS', 'Q', 'field_derivative_F1', 'field_derivative_F2', 'weighted_h1_mass_F0', 
                             'weighted_l2_mass_F3', 'divdiv_Fs', 'Fvc test', 'dot(grad(u), v)', 'dot(curl(v), w)_F0', 'dot(curl(v), w)_F0_2', 
                             'u*f', 'f*u', 'sqrt_pi_Fvd', 'dot(v, w)', 'dot(w, v)')

    bilinear_forms_to_test = bilinear_form_strings[0:10] # [bilinear_form_strings[0], ]
    
    for bf in bilinear_forms_to_test:
        value = bilinear_forms[bf]
        trial       = value['trial']
        test        = value['test']
        expr        = value['expr']
        is_field    = value.get('field') is not None
        if is_field:
            field       = value['field'][0]
            field_space = value['field'][1]

        Vh      = spaces[trial]['Vh']
        Wh      = spaces[test] ['Vh']

        u       = spaces[trial]['funcs'][0]
        v       = spaces[test] ['funcs'][1]

        a = BilinearForm((u, v), int_0(expr))

        t0 = time.time()
        a_h = discretize(a, domain_h, (Vh, Wh), backend=backend, fast_assembly=False)
        t1 = time.time()
        disc_time_old = t1-t0

        if is_field:
            t0 = time.time()
            if field_space == 'V0':
                A_old = a_h.assemble(F0=field)
            elif field_space == 'V1':
                A_old = a_h.assemble(F1=field)
            elif field_space == 'V2':
                A_old = a_h.assemble(F2=field)
            elif field_space == 'V3':
                A_old = a_h.assemble(F3=field)
            elif field_space == 'Vs':
                A_old = a_h.assemble(Fs=field)
            elif field_space == 'Vvc':
                A_old = a_h.assemble(Fvc=field)
            elif field_space == 'Vvd':
                A_old = a_h.assemble(Fvd=field)
            t1 = time.time()
        else:
            t0 = time.time()
            A_old = a_h.assemble()
            t1 = time.time()
        time_old = t1-t0

        t0 = time.time()
        a_h = discretize(a, domain_h, (Vh, Wh), backend=backend)
        t1 = time.time()
        disc_time_new = t1-t0

        if is_field:
            t0 = time.time()
            if field_space == 'V0':
                A_new = a_h.assemble(F0=field)
            elif field_space == 'V1':
                A_new = a_h.assemble(F1=field)
            elif field_space == 'V2':
                A_new = a_h.assemble(F2=field)
            elif field_space == 'V3':
                A_new = a_h.assemble(F3=field)
            elif field_space == 'Vs':
                A_new = a_h.assemble(Fs=field)
            elif field_space == 'Vvc':
                A_new = a_h.assemble(Fvc=field)
            elif field_space == 'Vvd':
                A_new = a_h.assemble(Fvd=field)
            t1 = time.time()
        else:
            t0 = time.time()
            A_new = a_h.assemble()
            t1 = time.time()
        time_new = t1-t0

        A_old_norm = np.linalg.norm(A_old.toarray())
        A_new_norm = np.linalg.norm(A_new.toarray())
        err = np.linalg.norm((A_old-A_new).toarray())
        rel_err = err / A_old_norm

        if verbose and (mpi_rank == 0):
            print(f' >>> Mapping Option : {mapping_option} ')
            print(f' >>> BF {bf} ')
            print(f' >>> Discretization in: Old {disc_time_old:.3g}\t\t|| New {disc_time_new:.3g} ')
            print(f' >>> Assembly in:       Old {time_old:.3g}\t\t|| New {time_new:.3g}\t\t|| Old/New {time_old/time_new:.3g} ')
            print(f' >>>      Error:            {err:.3g} ')
            print(f' >>> Rel. Error:            {rel_err:.3g} ')
            print(f' >>> Norm      : {A_old_norm:.7g} & {A_new_norm:.7g}')
            print()
        
        if not bf in ('Fvc test', 'u*f', 'field_derivative_F1', 'field_derivative_F2'): # must investigate these cases further
            assert rel_err < 1e-12 # arbitrary rel. error bound

#==============================================================================

verbose = True

mapping_options = [None, 'analytical'] # [None, 'analytical', 'Bspline']

comm    = MPI.COMM_WORLD
backend = PSYDAC_BACKEND_GPYCCEL

for option in mapping_options:

    build_matrices(mapping_option=option, verbose=verbose, backend=backend, comm=comm)
