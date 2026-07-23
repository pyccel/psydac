#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import  os
from    pathlib import  Path

import  pytest
import  time
import  numpy   as      np
from    sympy   import  sin, sqrt, pi, Abs, cos, tan

from    sympde.calculus             import dot, cross, grad, curl, div, laplace
from    sympde.expr                 import BilinearForm, integral
from    sympde.topology             import element_of, elements_of, Cube, Mapping, ScalarFunctionSpace, VectorFunctionSpace, Domain, Derham

from    psydac.api.discretization   import discretize
from    psydac.api.settings         import PSYDAC_BACKEND_GPYCCEL
from    psydac.linalg.block         import BlockVectorSpace
from    psydac.fem.basic            import FemField

# Get the mesh directory
import psydac.cad.mesh as mesh_mod
mesh_dir = Path(mesh_mod.__file__).parent

# With PR #448, matrices corresponding to bilinear forms on 3D domains are being assembled using a so called sum factorization algorithm.
# Unless explicitely using the old algorithm, this happens automatically, and hence all old tests passing should indicate that the implementation of the sum factorization algorithm has been successful.
# Nonetheless, there are various difficulties in the implementation, and possibly not all of them are accounted for in the existing tests.

# This file is designed to test such "difficult" edge cases - for mapped (Bspline & analytical) and parametric domains:

# Such "difficult" edge cases are:

# 1. bilinear forms on different spaces
# 2. (FemField / analytical / ...) weight functions
# 3. high derivatives (>=2)
# 4. (multiple) free FemFields
# 5. complicated expressions
# 6. uncommen (numpy) functions that need be imported correctly in the assembly file (here in the weight function)

# These tests also return old and new discretization and assembly times.

# Most of the time, being close to the "old matrix" (generated using the old assembly algorithm) will be the requirement to pass a test, 
# as the old implementation has not caused problems in a long time and is considered to function properly.

# Update: Instead of testing all mapping options all of the time, we now rather randomly test one of the three options!
#@pytest.mark.parametrize('mapping', ('None', 'Analytical', 'Bspline'))
def test_assembly(): # mapping):

    rng = np.random.default_rng() # (seed=42)
    mapping_options = ['None', 'Analytical', 'Bspline']
    mapping = mapping_options[int(np.floor(rng.random()*3))]

    ncells      = [7, 5, 6]
    degree      = [2, 4, 3]
    periodic    = [int(np.floor(rng.random()*2))==True for _ in range(3)]
    print(f'Random periodicity: {periodic}')

    trial_multiplicity = [1, 3, 2]
    test_multiplicity  = [2, 2, 3]

    backend = PSYDAC_BACKEND_GPYCCEL

    if mapping == 'None':

        domain = Cube('C', bounds1=(0,1), bounds2=(0,1), bounds3=(0,1))
        derham = Derham(domain)

        domain_h = discretize(domain, ncells=ncells, periodic=periodic)
        derham_h = discretize(derham, domain_h, degree=degree, multiplicity=trial_multiplicity)
        derham_test_h = discretize(derham, domain_h, degree=degree, multiplicity=test_multiplicity)

    elif mapping == 'Bspline':

        filename = os.path.join(mesh_dir, 'identity_3d.h5')

        domain = Domain.from_file(filename=filename)
        derham = Derham(domain)

        domain_h = discretize(domain, filename=filename)
        derham_h = discretize(derham, domain_h, degree=domain.mapping.get_callable_mapping().space.degree, multiplicity=trial_multiplicity)
        derham_test_h = discretize(derham, domain_h, degree=domain.mapping.get_callable_mapping().space.degree, multiplicity=test_multiplicity)

    elif mapping == 'Analytical':

        class HalfSquareTorusMapping3D(Mapping):
            _expressions = {'x': 'x1 * cos(x2)',
                            'y': 'x1 * sin(x2)',
                            'z': 'x3'}

            _ldim        = 3
            _pdim        = 3

        M = HalfSquareTorusMapping3D('M')
        logical_domain = Cube('C', bounds1=(0.3,1), bounds2=(0,np.pi), bounds3=(0,1))

        domain = M(logical_domain)
        derham = Derham(domain)

        domain_h = discretize(domain, ncells=ncells, periodic=periodic)
        derham_h = discretize(derham, domain_h, degree=degree, multiplicity=trial_multiplicity)
        derham_test_h = discretize(derham, domain_h, degree=degree, multiplicity=test_multiplicity)

    x, y, z = domain.coordinates
    weight  = 1 + sqrt(Abs(x*y**2 + z)) + abs(sin(x-2*y + pi + np.pi))

    V0  = derham.V0
    V0h = derham_h.V0
    V1  = derham.V1
    V1h = derham_h.V1
    V2  = derham.V2
    V2h = derham_h.V2
    V3  = derham.V3
    V3h = derham_h.V3

    V0h_test = derham_test_h.V0
    V1h_test = derham_test_h.V1
    V2h_test = derham_test_h.V2
    V3h_test = derham_test_h.V3

    Vs      = ScalarFunctionSpace('Vsh', domain)
    Vsh     = discretize(Vs, domain_h, degree=degree)
    Vvc     = VectorFunctionSpace('Vvc', domain, kind='hcurl')
    Vvch    = discretize(Vvc, domain_h, degree=degree)
    Vvd     = VectorFunctionSpace('Vvd', domain, kind='hdiv')
    Vvdh    = discretize(Vvd, domain_h, degree=degree)

    Vsh_test      = discretize(Vs, domain_h, degree=degree, multiplicity=test_multiplicity)
    Vvch_test     = discretize(Vs, domain_h, degree=degree, multiplicity=test_multiplicity)
    Vvdh_test     = discretize(Vs, domain_h, degree=degree, multiplicity=test_multiplicity)

    u1, u2, F01, F02, F03 = elements_of(V0, names='u1, u2, F01, F02, F03')
    v1, v2, F11, F12, F13 = elements_of(V1, names='v1, v2, F11, F12, F13')
    w1, w2, F21, F22, F23 = elements_of(V2, names='w1, w2, F21, F22, F23')
    f1, f2, F31, F32, F33 = elements_of(V3, names='f1, f2, F31, F32, F33')

    fs1,  fs2,  Fs1,  Fs2,  Fs3  = elements_of(Vs,  names='fs1,  fs2,  Fs1,  Fs2,  Fs3')
    fvc1, fvc2, Fvc1, Fvc2, Fvc3 = elements_of(Vvc, names='fvc1, fvc2, Fvc1, Fvc2, Fvc3')
    fvd1, fvd2, Fvd1, Fvd2, Fvd3 = elements_of(Vvd, names='fvd1, fvd2, Fvd1, Fvd2, Fvd3')

    trial_spaces = {'V0': {'Vh':V0h,  'funcs':[u1, u2]},
                    'V1': {'Vh':V1h,  'funcs':[v1, v2]},
                    'V2': {'Vh':V2h,  'funcs':[w1, w2]},
                    'V3': {'Vh':V3h,  'funcs':[f1, f2]},
                    'Vs': {'Vh':Vsh,  'funcs':[fs1, fs2]},
                    'Vvc':{'Vh':Vvch, 'funcs':[fvc1, fvc2]},
                    'Vvd':{'Vh':Vvdh, 'funcs':[fvd1, fvd2]}}
    
    test_spaces  = {'V0': {'Vh':V0h_test,  'funcs':[u1, u2]},
                    'V1': {'Vh':V1h_test,  'funcs':[v1, v2]},
                    'V2': {'Vh':V2h_test,  'funcs':[w1, w2]},
                    'V3': {'Vh':V3h_test,  'funcs':[f1, f2]},
                    'Vs': {'Vh':Vsh_test,  'funcs':[fs1, fs2]},
                    'Vvc':{'Vh':Vvch_test, 'funcs':[fvc1, fvc2]},
                    'Vvd':{'Vh':Vvdh_test, 'funcs':[fvd1, fvd2]}}

    F01_coeffs = V0h.coeff_space.zeros()
    rng.random(size=F01_coeffs._data.shape, dtype='float64', out=F01_coeffs._data)
    F11_coeffs = V1h.coeff_space.zeros()
    for block in F11_coeffs.blocks:
        rng.random(size=block._data.shape, dtype='float64', out=block._data)
    F12_coeffs = V1h.coeff_space.zeros()
    for block in F12_coeffs.blocks:
        rng.random(size=block._data.shape, dtype='float64', out=block._data)
    F21_coeffs = V2h.coeff_space.zeros()
    for block in F21_coeffs.blocks:
        rng.random(size=block._data.shape, dtype='float64', out=block._data)
    Fvc1_coeffs = Vvch.coeff_space.zeros()
    for block in Fvc1_coeffs.blocks:
        rng.random(size=block._data.shape, dtype='float64', out=block._data)
    Fs1_coeffs = Vsh.coeff_space.zeros()
    rng.random(size=Fs1_coeffs._data.shape, dtype='float64', out=Fs1_coeffs._data)
    Fs2_coeffs = Vsh.coeff_space.zeros()
    rng.random(size=Fs2_coeffs._data.shape, dtype='float64', out=Fs2_coeffs._data)
    Fvd1_coeffs = Vvdh.coeff_space.zeros()
    for block in Fvd1_coeffs.blocks:
        rng.random(size=block._data.shape, dtype='float64', out=block._data)

    F01_field   = FemField(V0h, F01_coeffs)
    F11_field   = FemField(V1h, F11_coeffs)
    F12_field   = FemField(V1h, F12_coeffs)
    F21_field   = FemField(V2h, F21_coeffs)
    Fvc1_field  = FemField(Vvch, Fvc1_coeffs)
    Fs1_field   = FemField(Vsh, Fs1_coeffs)
    Fs2_field   = FemField(Vsh, Fs2_coeffs)
    Fvd1_field  = FemField(Vvdh, Fvd1_coeffs)

    bilinear_forms = {  # one and two free FemFields without derivatives (with derivatives in seperate test)
                        # complicated expressions
                        'Q'             :{'trial' :'V1', 'test':'V1',
                                          'expr'  :dot(cross(F11, v1), cross(F11, v2)),
                                          'fields':[F11_field, ]},
                        'equilibrium'   :{'trial' :'V1', 'test':'V1',
                                          'expr'  :dot(cross(F11, v1), cross(v2, F12)),
                                          'fields':[F11_field, F12_field]},
                        'Elena'         :{'trial' :'V1', 'test':'V1',
                                          'expr'  :dot(F01*v1, v2),
                                          'fields':[F01_field, ]},
                        # weight function, free FemField, different spaces
                        'dot(grad(u),v)':{'trial' :'V0', 'test':'V1',
                                          'expr'  :dot(grad(u1), v2)*F01*weight,
                                          'fields':[F01_field, ]},
                        'dot(curl(v),w)':{'trial' :'V1', 'test':'V2',
                                          'expr'  :dot(curl(v1), F21)*div(w2)*weight,
                                          'fields':[F21_field, ]},
                        # among other difficulties: multiple scalar FemFields
                        'ScalarFields'  :{'trial' :'V0', 'test':'V1',
                                          'expr'  :dot(grad(u1), curl(Fvc1))*dot(grad(Fs1), curl(v2))*Fs2*div(Fvd1),
                                          'fields':[Fvc1_field, Fs1_field, Fs2_field, Fvd1_field]},
                        # high derivatives, not FEEC
                        'bilaplace'     :{'trial' :'Vs', 'test':'Vs',
                                          'expr'  :laplace(fs1)*laplace(fs2)}
                     }
    
    # test all BFs
    # bilinear_form_strings_to_test   = list(bilinear_forms.keys())

    # or only a subset
    bilinear_form_strings_to_test   = list(bilinear_forms.keys())[:-1] # exclude expensive bilaplace test

    bilinear_forms_to_test          = {}
    for name in bilinear_form_strings_to_test:
        bilinear_forms_to_test[name] = bilinear_forms[name]
    
    int_0 = lambda expr: integral(domain, expr)
    print()
    
    for bf_name, bf_data in bilinear_forms_to_test.items():

        trial_space = trial_spaces[bf_data['trial']]
        Vh          = trial_space['Vh']
        u           = trial_space['funcs'][0]
        test_space  = test_spaces[bf_data['test']]
        Wh          = test_space ['Vh']
        v           = test_space['funcs'][1]
        expr        = bf_data['expr']
        if 'fields' in bf_data.keys():
            fields      = bf_data['fields']

        a = BilinearForm((u, v), int_0(expr))

        t0 = time.time()
        ah_old = discretize(a, domain_h, (Vh, Wh), backend=backend, sum_factorization=False)
        t1 = time.time()
        discretization_time_old = t1 - t0

        t0 = time.time()
        ah = discretize(a, domain_h, (Vh, Wh), backend=backend)
        t1 = time.time()
        discretization_time     = t1 - t0

        if bf_name == 'Q':
            t0_old  = time.time()
            A_old   = ah_old.assemble(F11=fields[0])
            t1_old  = time.time()
            
            t0      = time.time()
            A       = ah.assemble(F11=fields[0])
            t1      = time.time()
        elif bf_name == 'equilibrium':
            t0_old  = time.time()
            A_old   = ah_old.assemble(F11=fields[0], F12=fields[1])
            t1_old  = time.time()
            
            t0      = time.time()
            A       = ah.assemble(F11=fields[0], F12=fields[1])
            t1      = time.time()
        elif bf_name in ('Elena', 'dot(grad(u),v)'):
            t0_old  = time.time()
            A_old   = ah_old.assemble(F01=fields[0])
            t1_old  = time.time()
            
            t0      = time.time()
            A       = ah.assemble(F01=fields[0])
            t1      = time.time()
        elif bf_name == 'dot(curl(v),w)':
            t0_old  = time.time()
            A_old   = ah_old.assemble(F21=fields[0])
            t1_old  = time.time()
            
            t0      = time.time()
            A       = ah.assemble(F21=fields[0])
            t1      = time.time()
        elif bf_name == 'ScalarFields':
            t0_old  = time.time()
            A_old   = ah_old.assemble(Fvc1=fields[0], Fs1=fields[1], Fs2=fields[2], Fvd1=fields[3])
            t1_old  = time.time()

            t0      = time.time()
            A       = ah.assemble(Fvc1=fields[0], Fs1=fields[1], Fs2=fields[2], Fvd1=fields[3])
            t1      = time.time()
        else:
            t0_old  = time.time()
            A_old   = ah_old.assemble()
            t1_old  = time.time()
            
            t0      = time.time()
            A       = ah.assemble()
            t1      = time.time()
            
        assembly_time_old   = t1_old - t0_old
        assembly_time       = t1     - t0

        # Testing whether two linear operators are identical by comparing their arrays is quite expensive.
        # Thus we instead test whether three random domain vectors applied to both
        # the old and the new matrix produce the same codomain vector.
        
        domain_vector1 = Vh.coeff_space.zeros()
        domain_vector2 = Vh.coeff_space.zeros()
        domain_vector3 = Vh.coeff_space.zeros()
        domain_vectors = [domain_vector1, domain_vector2, domain_vector3]

        if isinstance(Vh.coeff_space, BlockVectorSpace):
            for domain_vector in domain_vectors:
                for block in domain_vector.blocks:
                    rng.random(size=block._data.shape, dtype='float64', out=block._data)
        else:
            for domain_vector in domain_vectors:
                rng.random(size=domain_vector._data.shape, dtype='float64', out=domain_vector._data)
        
        err     = []
        rel_err = []

        for domain_vector in domain_vectors:
            codomain_vector     = A @ domain_vector
            codomain_vector_old = A_old @ domain_vector

            norm_old            = np.sqrt(codomain_vector_old.inner(codomain_vector_old))

            diff                = codomain_vector - codomain_vector_old

            err.append(np.sqrt(diff.inner(diff)))
            rel_err.append(err[-1] / norm_old)
        
        print(f' >>> Mapping: {mapping}')
        print(f' >>> BilinearForm: {bf_name}')
        print(f' >>> Discretization in: Old {discretization_time_old:.3g} \t\t || New {discretization_time:.3g} \t\t || Old/New {discretization_time_old/discretization_time:.3g}')
        print(f' >>> Assembly in: Old {assembly_time_old:.3g} \t \t || New {assembly_time:.3g} \t\t || Old/New {assembly_time_old/assembly_time:.3g}')
        print(f' >>>      Error: {max(err):.3g}')
        print(f' >>> Rel. Error: {max(rel_err):.3g}')
        print()

        assert max(rel_err) < 1e-12 # arbitrary rel. error bound (How to test better?)


# fixed by PR #507
#@pytest.mark.xfail
def test_allocate_matrix_bug():
    """
    This test is related to Issue #504.

    The bilinear form 
    (V0 x V3) ni (u, f) mapsto int_{Omega} u*f
    should be the transpose of the bilinear form
    (V3 x V0) ni (f, u) mapsto int_{Omega} u*f
    but is not.
    """

    ncells      = [15, 16, 17]
    degree      = [4, 3, 2]
    periodic    = [False, True, False]

    backend = PSYDAC_BACKEND_GPYCCEL

    domain = Cube('C', bounds1=(0,1), bounds2=(0,1), bounds3=(0,1))
    derham = Derham(domain)

    domain_h = discretize(domain, ncells=ncells, periodic=periodic)
    derham_h = discretize(derham, domain_h, degree=degree)

    P0, _, _, P3 = derham_h.projectors()

    V0  = derham.V0
    V0h = derham_h.V0
    V3  = derham.V3
    V3h = derham_h.V3

    u = element_of(V0, name='u')
    f = element_of(V3, name='f')

    fun = lambda x, y, z : 1
    u_coeffs = P0(fun).coeffs
    f_coeffs = P3(fun).coeffs

    a0 = BilinearForm((u, f), integral(domain, u*f))
    a1 = BilinearForm((f, u), integral(domain, u*f))

    a0h = discretize(a0, domain_h, (V0h, V3h), backend=backend, sum_factorization=False)
    a1h = discretize(a1, domain_h, (V3h, V0h), backend=backend, sum_factorization=False)

    A0  = a0h.assemble()
    A1  = a1h.assemble()
    A1T = A1.T

    # Clearly, it should hold A1T = A0, and further ||A0|| = ||A1||.
    A0arr  = A0.toarray()
    A1arr  = A1.toarray()
    A1Tarr = A1T.toarray()

    diff1 = np.linalg.norm(A0arr - A1Tarr)
    diff2 = np.linalg.norm(A0arr) - np.linalg.norm(A1arr)

    print(f' || A0 - A1.T || = {diff1:.3g}')
    print(f' ||A0|| - ||A1|| = {diff2:.3g}')

    # Further, the following integral should evaluate to 1. 
    # This however is only the case for the second integral, 
    # independent on whether one uses the new or old assembly algorithm.

    print(f' 1 =? {A0.dot_inner(u_coeffs, f_coeffs)}')
    print(f' 1 =? {A1.dot_inner(f_coeffs, u_coeffs)}')

    assert diff1 <= 1e-12 # arbitrary error bound
    assert diff2 <= 1e-12 # arbitrary error bound

#@pytest.mark.xfail
def test_free_FemField_derivatives():
    """
    These particular bilinear forms, when using a constant 1-vector coefficient vector for the free FemFields, 
    causes problems in a different test file of mine.
    In particular, the assembled matrices used to have really small norms (~e-13).
    That is probably due to the constant 1-vector corresponding to a constant function,
    which means that all appearing derivatives of free FemFields are 0.

    When using meaningful free FemFields, these dubious observations disappeared.
    
    """

    ncells      = [5, 2, 4]
    degree      = [2, 1, 3]
    periodic    = [False, False, False]

    backend = PSYDAC_BACKEND_GPYCCEL

    domain = Cube('C', bounds1=(0,1), bounds2=(0,1), bounds3=(0,1))
    derham = Derham(domain)

    domain_h = discretize(domain, ncells=ncells, periodic=periodic)
    derham_h = discretize(derham, domain_h, degree=degree)

    P0, P1, P2, P3 = derham_h.projectors()

    V0      = derham.V0
    V0h     = derham_h.V0
    V1      = derham.V1
    V1h     = derham_h.V1
    V2      = derham.V2
    V2h     = derham_h.V2

    u           = element_of (V0, name= 'u')
    v, F1       = elements_of(V1, names='v, F1')
    w1, w2, F2  = elements_of(V2, names='w1, w2, F2')

    u_func = lambda x, y, z: x + y + z
    u_coeffs = P0(u_func).coeffs

    v_1 = lambda x, y, z: 1
    v_2 = lambda x, y, z: 1
    v_3 = lambda x, y, z: 1
    v_func = (v_1, v_2, v_3)
    v_coeffs = P1(v_func).coeffs

    w_1 = lambda x, y, z: x
    w_2 = lambda x, y, z: y
    w_3 = lambda x, y, z: z
    w_func = (w_1, w_2, w_3)
    w_coeffs = P2(w_func).coeffs

    dubious_observations = False

    if dubious_observations:
        F1_coeffs = V1h.coeff_space.zeros()
        F2_coeffs = V2h.coeff_space.zeros()
        for block in F1_coeffs.blocks:
            block._data = np.ones(block._data.shape, dtype='float64')
        for block in F2_coeffs.blocks:
            block._data = np.ones(block._data.shape, dtype='float64')
        F1_FF = FemField(V1h, F1_coeffs)
        F2_FF = FemField(V2h, F2_coeffs)
    else:
        F1_1 = lambda x, y, z: z
        F1_2 = lambda x, y, z: x
        F1_3 = lambda x, y, z: y
        F1_func = (F1_1, F1_2, F1_3)
        F1_FF = P1(F1_func)

        F2_FF = FemField(V2h, w_coeffs.copy())

    # with the above choices (dubious_observation = False): 
    # a0 reduces to 3*int_{Omega}x+y+z with Omega being the unit square. Expected value: 4.5
    a0 = BilinearForm((u, v), integral(domain, dot(grad(u), curl(F1)) * dot(v, F1)))
    # a1 reduces to 9* ----------------------------------------- " -------------------- 13.5
    a1 = BilinearForm((u, w2), integral(domain, dot(grad(u), F2)*div(w2)*div(F2)))
    # a2 reduces to 3* ----------------------------------------- " --------------------- 4.5
    a2 = BilinearForm((w1, w2), integral(domain, dot(curl(F1), w1)*div(w2)))

    a0h_old = discretize(a0, domain_h, (V0h, V1h), backend=backend, sum_factorization=False)
    a1h_old = discretize(a1, domain_h, (V0h, V2h), backend=backend, sum_factorization=False)
    a2h_old = discretize(a2, domain_h, (V2h, V2h), backend=backend, sum_factorization=False)

    a0h = discretize(a0, domain_h, (V0h, V1h), backend=backend)
    a1h = discretize(a1, domain_h, (V0h, V2h), backend=backend)
    a2h = discretize(a2, domain_h, (V2h, V2h), backend=backend)

    bfs = [(a0h_old, a0h), (a1h_old, a1h), (a2h_old, a2h)]
    print()

    for i, (ah_old, ah) in enumerate(bfs):

        if i in (0, 2):
            A_old   = ah_old.assemble(F1=F1_FF)
            A       = ah.assemble(F1=F1_FF)
        else:
            A_old   = ah_old.assemble(F2=F2_FF)
            A       = ah.assemble(F2=F2_FF)

        if i == 0:
            value_old   = A_old.dot_inner(u_coeffs, v_coeffs)
            value       = A.dot_inner(u_coeffs, v_coeffs)
        elif i == 1:
            value_old   = A_old.dot_inner(u_coeffs, w_coeffs)
            value       = A.dot_inner(u_coeffs, w_coeffs)
        else:
            value_old   = A_old.dot_inner(w_coeffs, w_coeffs)
            value       = A.dot_inner(w_coeffs, w_coeffs)

        A_old_arr   = A_old.toarray()
        A_arr       = A.toarray()
        A_old_norm  = np.linalg.norm(A_old_arr)
        A_norm      = np.linalg.norm(A_arr)

        err         = np.linalg.norm(A_old_arr - A_arr)
        rel_err     = err / A_old_norm

        print(f' i = {i}')
        print(f' >>>      Error: {err:.3g}')
        print(f' >>> Rel. Error: {rel_err:.3g}')
        print(f' >>>      Norms: ||A_old|| = {A_old_norm:.3g} \t\t ||A|| = {A_norm:.3g}')
        print()

        # arbitrary tolerance
        tol = 1e-12
        if not dubious_observations:
            assert abs(value-value_old) < tol
            assert rel_err              < tol

def test_assembly_free_FemFields():

    backend = PSYDAC_BACKEND_GPYCCEL

    domain = Cube(bounds1=(2626,3179), bounds2=(-138, 138), bounds3=(-760.3, 69))
    derham = Derham(domain)
    V0 = derham.V0
    V1 = derham.V1

    u = element_of(V1, name='u')
    v = element_of(V1, name='v')
    p = element_of(V0, name='p')

    p_call = lambda xi,yi,zi: np.cos(2*np.pi*(xi-2626)/553) + np.tan(2*np.pi*(zi+760.3)/(5*829.3))
    x,y,z = domain.coordinates
    p_sym = cos(2*np.pi*(x-2626)/553) + tan(2*np.pi*(z+760.3)/(5*829.3))

    a_sym = BilinearForm((u, v), integral(domain, p_sym*dot(u, v)))
    a_fem = BilinearForm((u, v), integral(domain, p*dot(u, v)))

    ncells = (11, 1, 17)
    degree = (3, 1, 4)
    domain_h = discretize(domain, ncells=ncells, periodic=(False, True, False))
    derham_h = discretize(derham, domain_h, degree=degree)
    V1_h = derham_h.V1
    P0 = derham_h.projectors()[0]

    p_fem = P0(p_call)
    
    a_sym_h = discretize(a_sym, domain_h, (V1_h, V1_h), backend=backend)
    a_fem_h = discretize(a_fem, domain_h, (V1_h, V1_h), backend=backend)

    A_sym = a_sym_h.assemble()
    A_fem = a_fem_h.assemble(p=p_fem)

    A_sym_sp = A_sym.tosparse()
    A_fem_sp = A_fem.tosparse()
    diff = A_sym_sp - A_fem_sp
    norm_sym = np.linalg.norm(A_sym_sp.data)
    norm_fem = np.linalg.norm(A_fem_sp.data)
    error    = np.linalg.norm(diff.data)
    rel_err  = error / norm_sym
    
    #print(norm_sym, norm_fem, error, rel_err)

    assert rel_err < 1e-4
