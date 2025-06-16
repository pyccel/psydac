import  os
import  time
from    mpi4py  import  MPI
import  numpy   as      np

from    sympy   import  sin

from    sympde.calculus             import dot, cross, grad, curl, div, laplace
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

    ncells      = [5, 2, 4] # [12, 8, 16]
    degree      = [2, 1, 3] # [2, 4, 3]
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
    v1, v2, F1, F12 = elements_of(V1, names='v1, v2, F1, F12')
    w1, w2, F2 = elements_of(V2, names='w1, w2, F2')
    f1, f2, F3 = elements_of(V3, names='f1, f2, F3')

    fs1,  fs2,  Fs  = elements_of(Vs,  names='fs1,  fs2,  Fs' )
    fvc1, fvc2, Fvc = elements_of(Vvc, names='fvc1, fvc2, Fvc')
    fvd1, fvd2, Fvd = elements_of(Vvd, names='fvd1, fvd2, Fvd')

    F0_field_coeffs  = V0h.coeff_space.zeros()
    F1_field_coeffs  = V1h.coeff_space.zeros()
    F1_field_coeffs2 = F1_field_coeffs.copy()
    F2_field_coeffs  = V2h.coeff_space.zeros()
    F3_field_coeffs  = V3h.coeff_space.zeros()
    Fs_field_coeffs  = Vsh.coeff_space.zeros()
    Fvc_field_coeffs = Vvch.coeff_space.zeros()
    Fvd_field_coeffs = Vvdh.coeff_space.zeros()

    F0_field_coeffs._data = np.ones(F0_field_coeffs._data.shape, dtype="float64")
    F3_field_coeffs._data = np.ones(F3_field_coeffs._data.shape, dtype="float64")
    Fs_field_coeffs._data = np.ones(Fs_field_coeffs._data.shape, dtype="float64")
    for block in F1_field_coeffs.blocks:
        block._data = np.ones(block._data.shape, dtype="float64")
    rng = np.random.default_rng(seed=42)
    for block in F1_field_coeffs2.blocks:
        rng.random(size=block._data.shape, dtype="float64", out=block._data)
    for block in F2_field_coeffs.blocks:
        block._data = np.ones(block._data.shape, dtype="float64")
    for block in Fvc_field_coeffs.blocks:
        block._data = np.ones(block._data.shape, dtype="float64")
    for block in Fvd_field_coeffs.blocks:
        block._data = np.ones(block._data.shape, dtype="float64")

    F0_field  = FemField(V0h, F0_field_coeffs)
    F1_field  = FemField(V1h, F1_field_coeffs)
    F1_field2 = FemField(V1h, F1_field_coeffs2)
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

    bilinear_forms = {  # 'gradgrad' and 'gradgradSFS':
                        # no particular difficulty expected, should produce similar timings and identical norm and error values
                        # fast, can always be included in tests
                        'gradgrad':             {'trial':'V0',
                                                'test'  :'V0',
                                                'expr'  :dot(grad(u1), grad(u2))},

                        'gradgradSFS':          {'trial':'Vs',
                                                'test'  :'Vs',
                                                'expr'  :dot(grad(fs1), grad(fs2))},
                        # 'curlcurl' and 'curlcurlVFS':
                        # see 'gradgrad' and 'gradgradSFS'
                        # interestingly, both matrices are completely different (analytical mapping)! Is that supposed to be the case?
                        # takes rather long, should possibly be excluded
                        'curlcurl':             {'trial':'V1',
                                                'test'  :'V1',
                                                'expr'  :dot(curl(v1), curl(v2))},

                        'curlcurlVFS':          {'trial':'Vvc',
                                                'test'  :'Vvc',
                                                'expr'  :dot(curl(fvc1), curl(fvc2))},
                        # 'weighted_hdiv_mass' and '...VFS':
                        # introduces an analytical weight function (new difficulty)
                        # as in the curlcurl case: both matrices are completely different (analytical mapping case).
                        # neither fast nor slow, can remain included
                        # if no mapping: Much(!) faster due to extremely naive implementation
                        'weighted_hdiv_mass':   {'trial':'V2',
                                                'test'  :'V2',
                                                'expr'  :dot(w1, w2)*gamma},

                        'weighted_hdiv_massVFS':{'trial':'Vvd',
                                                'test'  :'Vvd',                     # Note: Often time discretization in case of VFS slower! Check!
                                                'expr'  :dot(fvd1, fvd2)*gamma},
                        # 'hcurl_mass' and '...VFS':
                        # introduces no new difficulty, can be removed
                        'hcurl_mass':           {'trial':'V1',
                                                'test'  :'V1',
                                                'expr'  :dot(v1, v2)},

                        'hcurl_massVFS':        {'trial':'Vvc',
                                                'test'  :'Vvc',
                                                'expr'  :dot(fvc1,fvc2)},
                        # 'Q':
                        # relevant, as I use this matrix in my simulation.
                        # Also rather interesting, given no derivatives involved.
                        # New difficulty: One free FemField
                        # rather slow, but should remain included in future tests
                        'Q':                    {'trial':'V1',
                                                 'test' :'V1',
                                                 'expr' :dot(cross(F1, v1), cross(F1, v2)),
                                                 'field':[F1_field, 'V1']},
                        # 'field_derivative_F1':
                        # relevant, as it used to fail previosuly (does not fail when analytical mapping, other cases not tested yet)
                        # unfortunately, can't tell which change caused this test to pass (analytical test case only)
                        # new_difficulty: derivatives on free FemFields
                        # keep!
                        # Update: Still fails if no mapping! Must investigate!
                        'field_derivative_F1':  {'trial':'V0',      # test fails with rel. error: 0.477, possibly only due to
                                                 'test' :'V1',      # matrix norm being very small (~4.6e-13)!
                                                 'expr' :dot(grad(u1), curl(F1)) * dot(v2, F1),
                                                 'field':[F1_field, 'V1']},
                        # 'field_derivative_F2':
                        # never caused problems, not new difficulty, but also unsure why the '..._F1' test case caused problems, but not this one
                        # not too slow, keep!
                        # Update: Other code snipped suggests that this test case caused problems in the past: Keep!
                        # Update: Still fails if no mapping! Must investigate!
                        'field_derivative_F2':  {'trial':'V0',
                                                 'test' :'V2',
                                                 'expr' :dot(grad(u1), F2)*div(w2)*div(F2),
                                                 'field':[F2_field, 'V2']},
                        # 'weighted_h1_mass_F0' and 'weighted_l2_mass_F3':
                        # semi-new difficulty: free FemField as weight function
                        # fast, so can remain included
                        'weighted_h1_mass_F0':  {'trial':'V0',
                                                 'test' :'V0',
                                                 'expr' :u1*u2*F0,
                                                 'field':[F0_field, 'V0']},

                        'weighted_l2_mass_F3':  {'trial':'V3',
                                                 'test' :'V3',
                                                 'expr' :f1*f2*F3,
                                                 'field':[F3_field, 'V3']},
                        # 'divdiv_Fs':
                        # idea: free FemField acting as weight function belongs to a ScalarFunctionSpace (not deRham)
                        # yet: not really a new difficulty, no reason for why this should fail
                        # also: slow! Remove from tests
                        'divdiv_Fs':            {'trial':'V2',
                                                 'test' :'V2',
                                                 'expr' :div(w1)*div(w2)*Fs,
                                                 'field':[Fs_field, 'Vs']},
                        # 'Fvc test':
                        # used to fail previosuly! Does not fail now when analytical mapping (others not tested yet)!
                        # Rather slow, but due to the above relevant, thus keep!
                        # new difficulty: derivative on non deRham VectorFunctionSpace free FemField!
                        # Update: Still fails if no mapping! Must investigate!
                        'Fvc test':             {'trial':'V2',
                                                 'test' :'V2',
                                                 'expr' :dot(curl(Fvc), w1)*div(w2),
                                                 'field':[Fvc_field, 'Vvc']},
                        # 'dot(grad(u), v)', 'dot(curl(v), w)_F0' and ''dot(curl(v), w)_F0_2':
                        # no new difficulty, no reason for why it should fail: Remove!
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
                        # 'u*f' and 'f*u':
                        # super interesting (and fast), always keep!
                        # 'u*f', althought it should just be the transpose of 'f*u', fails! Error in the size of the allocated global matrix
                        # must solve properly!
                        'u*f':                  {'trial':'V0',
                                                 'test' :'V3',
                                                 'expr' :u1*f2},

                        'f*u':                  {'trial':'V3',
                                                 'test' :'V0',
                                                 'expr' :u2*f1},
                        # 'sqrt_pi_Fvd'
                        # new difficulty: test whether we properly import sqrt and pi
                        # Keep!
                        'sqrt_pi_Fvd':          {'trial':'V0',
                                                 'test' :'V1',
                                                 'expr' :u1*dot(v2, Fvd) - np.sqrt(np.pi)*dot(grad(u1), curl(v2))*div(Fvd),
                                                 'field':[Fvd_field, 'Vvd']},
                        # 'dot(v, w)' and 'dot(w, v)':
                        # the counter part of 'f*u' and 'u*f': But, no bugs here, rather slow, remove!
                        'dot(v, w)':            {'trial':'V1',
                                                 'test' :'V2',
                                                 'expr' :dot(v1, w2)},

                        'dot(w, v)':            {'trial':'V2',
                                                 'test' :'V1',
                                                 'expr' :dot(v2, w1)},

                        'equilibrium':          {'trial':'V1',
                                                 'test' :'V1',
                                                 'expr' :dot( cross(F1, v1), cross(v2, F12) ),
                                                 'field':[[F1_field, 'V1'], [F1_field2, 'V1']]},

                        'bilaplace':            {'trial':'Vs',
                                                 'test' :'Vs',
                                                 'expr' : laplace(fs1) * laplace(fs2)}
                     }
    #                           0               1           2           3                   4                       5
    bilinear_form_strings = ('gradgrad', 'gradgradSFS', 'curlcurl', 'curlcurlVFS', 'weighted_hdiv_mass', 'weighted_hdiv_massVFS',
    #                           6               7           8           9                       10                      11 
                             'hcurl_mass', 'hcurl_massVFS', 'Q', 'field_derivative_F1', 'field_derivative_F2', 'weighted_h1_mass_F0', 
    #                                   12              13          14              15                  16                      17
                             'weighted_l2_mass_F3', 'divdiv_Fs', 'Fvc test', 'dot(grad(u), v)', 'dot(curl(v), w)_F0', 'dot(curl(v), w)_F0_2', 
    #                           18      19      20          21              22              23          24
                             'u*f', 'f*u', 'sqrt_pi_Fvd', 'dot(v, w)', 'dot(w, v)', 'equilibrium', 'bilaplace')


    standard_test_indices = [0, 1, 4, 5, 8, 9, 10, 11, 12, 14, 18, 19, 20, 23, 24]

    test_indices = standard_test_indices
    # test_indices = [9, ]

    bilinear_forms_to_test = [bilinear_form_strings[i] for i in test_indices]

    # bilinear_forms_to_test = bilinear_form_strings
    
    for bf in bilinear_forms_to_test:
        value = bilinear_forms[bf]
        trial       = value['trial']
        test        = value['test']
        expr        = value['expr']
        is_field    = value.get('field') is not None
        if is_field:
            if not isinstance(value['field'][0], list): 
                field        = value['field'][0]
                field_space  = value['field'][1]
                field_spaces = None
            else:
                fields       = [value['field'][i][0] for i in range(len(value['field']))]
                field_spaces = [value['field'][i][1] for i in range(len(value['field']))]
                field_space  = None

        Vh      = spaces[trial]['Vh']
        Wh      = spaces[test] ['Vh']

        u       = spaces[trial]['funcs'][0]
        v       = spaces[test] ['funcs'][1]

        a = BilinearForm((u, v), int_0(expr))

        t0 = time.time()
        a_h = discretize(a, domain_h, (Vh, Wh), fast_assembly=False, backend=backend)
        t1 = time.time()
        disc_time_old = t1-t0

        if is_field:
            
            if field_spaces is None:
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
                if field_spaces == ['V1', 'V1']:
                    t0 = time.time()
                    A_old = a_h.assemble(F1=fields[0], F12=fields[1])
                    t1 = time.time()
                    c1 = fields[0].coeffs
                    c2 = fields[1].coeffs
                    #print(f'c1 norm: {np.linalg.norm(c1.toarray())}')
                    #print(f'c2 norm: {np.linalg.norm(c2.toarray())}')
                    #equil = fields[0].coeffs.inner(A_old @ fields[1].coeffs)
                    equil = c1.inner(A_old @ c2)
                    #equil_T = c2.inner(A_old @ c1)
                    print(f'old equil: {equil}')
                    #print(f'old equil_T: {equil_T}')
                else:
                    raise NotImplementedError('This special case must still be taken care of.')
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
            if field_spaces is None:
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
                if field_spaces == ['V1', 'V1']:
                    #from psydac.api.tests.assemble_4srahtxb import assemble_matrix
                    #a_h._func = assemble_matrix
                    t0 = time.time()
                    A_new = a_h.assemble(F1=fields[0], F12=fields[1])
                    t1 = time.time()
                    c1 = fields[0].coeffs
                    c2 = fields[1].coeffs
                    #print(f'c1 norm: {np.linalg.norm(c1.toarray())}')
                    #print(f'c2 norm: {np.linalg.norm(c2.toarray())}')
                    #equil = fields[0].coeffs.inner(A_new @ fields[1].coeffs)
                    equil = c1.inner(A_new @ c2)
                    #equil_T = c2.inner(A_new @ c1)
                    print(f'new equil: {equil}')
                    #print(f'new equil_T: {equil_T}')
                    print()
                else:
                    raise NotImplementedError('This special case must still be taken care of.')
        else:
            t0 = time.time()
            A_new = a_h.assemble()
            t1 = time.time()
        time_new = t1-t0

        A_old_arr = A_old.toarray()
        A_new_arr = A_new.toarray()
        #A_new_T_arr = A_new.T.toarray()
        A_old_norm = np.linalg.norm(A_old_arr)
        A_new_norm = np.linalg.norm(A_new_arr)
        #A_new_T_norm = np.linalg.norm(A_new_T_arr)
        #A_old_norm = np.linalg.norm(A_old.toarray())
        #A_new_norm = np.linalg.norm(A_new.toarray())
        err = np.linalg.norm(A_old_arr - A_new_arr)
        #err_T = np.linalg.norm(A_old_arr - A_new_T_arr)
        #err = np.linalg.norm((A_old-A_new).toarray())
        rel_err = err / A_old_norm

        if verbose: #  and (mpi_rank == 0):
            print(f' >>> MPI rank       : {comm.Get_rank()} ')
            print(f' >>> Mapping Option : {mapping_option} ')
            print(f' >>> BF {bf} ')
            print(f' >>> Discretization in: Old {disc_time_old:.3g}\t\t|| New {disc_time_new:.3g} ')
            print(f' >>> Assembly in:       Old {time_old:.3g}\t\t|| New {time_new:.3g}\t\t|| Old/New {time_old/time_new:.3g} ')
            print(f' >>>      Error:            {err:.3g} ')
            #print(f' >>> Err. T.   :            {err_T:.3g}')
            print(f' >>> Rel. Error:            {rel_err:.3g} ')
            print(f' >>> Norm      : {A_old_norm:.7g} & {A_new_norm:.7g}')
            print()

        #import sys
        #np.set_printoptions(threshold=sys.maxsize)
        #print(A_old_arr)
        #print()
        #print()
        #print()
        #print(A_new_arr)
        
        # must investigate these cases further
        if not bf in ('Fvc test', 'u*f', 'field_derivative_F1', 'field_derivative_F2', 'equilibrium'):
            assert rel_err < 1e-12 # arbitrary rel. error bound

#==============================================================================

verbose = True

mapping_options = [None, 'analytical', 'Bspline']

comm    = MPI.COMM_WORLD
backend = PSYDAC_BACKEND_GPYCCEL

for option in mapping_options:

    build_matrices(mapping_option=option, verbose=verbose, backend=backend, comm=comm)
