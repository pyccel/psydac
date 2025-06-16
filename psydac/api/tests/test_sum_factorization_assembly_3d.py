import  os
import  pytest
import  time
import  numpy   as      np

from    sympy   import  sin

from    sympde.calculus             import dot, cross, grad, curl, div, laplace
from    sympde.expr                 import BilinearForm, integral
from    sympde.topology             import elements_of, Cube, Mapping, ScalarFunctionSpace, VectorFunctionSpace, Domain, Derham

from    psydac.api.discretization   import discretize
from    psydac.api.settings         import PSYDAC_BACKEND_GPYCCEL
from    psydac.fem.basic            import FemField

try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']

except:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')

"""
With PR #448, matrices corresponding to bilinear forms on 3D domains are being assembled using a so called sum factorization algorithm.
Unless explicitely using the old algorithm, this happens automatically, and hence all old tests passing should indicate that the implementation of the sum factorization algorithm has been successful.
Nonetheless, there are various difficulties in the implementation, and possibly not all of them are accounted for in the existing tests.

This file is designed to test such "difficult" edge cases - for mapped (Bspline & analytical) and parametric domains:

Such "difficult" edge cases are:

1. bilinear forms between different spaces
2. (FemField / analytical / ...) weight functions
3. high derivatives (>=2)
4. (multiple) free FemFields
5. complicated expressions
6. uncommen (numpy) functions that need be imported correctly in the assembly file

These tests also return old and new discretization and assembly times.

Most of the time, being close to the "old matrix" (generated using the old assembly algorithm) will be the requirement to pass the test, as the old implementation has not caused problems in a long time and is considered to function properly.

"""

@pytest.mark.parametrize('mapping', ('None', 'Analytical', 'Bspline'))
def test_free_FemFields(mapping):

    ncells      = [5, 2, 4]
    degree      = [2, 1, 3]
    periodic    = [False, True, False]

    backend = PSYDAC_BACKEND_GPYCCEL

    if mapping == 'None':

        domain = Cube('C', bounds1=(0,1), bounds2=(0,1), bounds3=(0,1))
        derham = Derham(domain)

        domain_h = discretize(domain, ncells=ncells, periodic=periodic)
        derham_h = discretize(derham, domain_h, degree=degree)

    elif mapping == 'Bspline':

        filename = os.path.join(mesh_dir, 'identity_3d.h5')

        domain = Domain.from_file(filename=filename)
        derham = Derham(domain)

        domain_h = discretize(domain, filename=filename)
        derham_h = discretize(derham, domain_h, degree=domain.mapping.get_callable_mapping().space.degree)

    elif mapping == 'Analytical':

        class HalfSquareTorusMapping3D(Mapping):
            _expressions = {'x': 'x1 * cos(x2)',
                            'y': 'x1 * sin(x2)',
                            'z': 'x3'}

            _ldim        = 3
            _pdim        = 3

        map = HalfSquareTorusMapping3D('M')
        logical_domain = Cube('C', bounds1=(0.3,1), bounds2=(0,np.pi), bounds3=(0,1))

        domain = map(logical_domain)
        derham = Derham(domain)

        domain_h = discretize(domain, ncells=ncells, periodic=periodic)
        derham_h = discretize(derham, domain_h, degree=degree)

    V0  = derham.V0
    V0h = derham_h.V0
    V1  = derham.V1
    V1h = derham_h.V1
    V2  = derham.V2
    V2h = derham_h.V2
    V3  = derham.V3
    V3h = derham_h.V3

    Vsh     = ScalarFunctionSpace('Vsh', domain, kind='h1')
    Vshh    = discretize(Vsh, domain_h, degree=degree)
    Vvc     = VectorFunctionSpace('Vvc', domain, kind='hcurl')
    Vvch    = discretize(Vvc, domain_h, degree=degree)
    Vvd     = VectorFunctionSpace('Vvd', domain, kind='hdiv')
    Vvdh    = discretize(Vvd, domain_h, degree=degree)
    Vsl     = ScalarFunctionSpace('Vsl', domain, kind='l2')
    Vslh    = discretize(Vsl, domain_h, degree=degree)

    u1, u2, F01, F02, F03 = elements_of(V0, names='u1, u2, F01, F02, F03')
    v1, v2, F11, F12, F13 = elements_of(V1, names='v1, v2, F11, F12, F13')
    w1, w2, F21, F22, F23 = elements_of(V2, names='w1, w2, F21, F22, F23')
    f1, f2, F31, F32, F33 = elements_of(V3, names='f1, f2, F31, F32, F33')

    fsh1, fsh2, Fsh1, Fsh2, Fsh3 = elements_of(Vsh, names='fsh1, fsh2, Fsh1, Fsh2, Fsh3')
    fvc1, fvc2, Fvc1, Fvc2, Fvc3 = elements_of(Vvc, names='fvc1, fvc2, Fvc1, Fvc2, Fvc3')
    fvd1, fvd2, Fvd1, Fvd2, Fvd3 = elements_of(Vvd, names='fvd1, fvd2, Fvd1, Fvd2, Fvd3')
    fsl1, fsl2, Fsl1, Fsl2, Fsl3 = elements_of(Vsl, names='fsl1, fsl2, Fsl1, Fsl2, Fsl3')

    spaces = {'V0': {'Vh':V0h,  'funcs':[u1, u2]},
              'V1': {'Vh':V1h,  'funcs':[v1, v2]},
              'V2': {'Vh':V2h,  'funcs':[w1, w2]},
              'V3': {'Vh':V3h,  'funcs':[f1, f2]},
              'Vsh':{'Vh':Vshh, 'funcs':[fsh1, fsh2]},
              'Vvc':{'Vh':Vvch, 'funcs':[fvc1, fvc2]},
              'Vvd':{'Vh':Vvdh, 'funcs':[fvd1, fvd2]},
              'Vsl':{'Vh':Vslh, 'funcs':[fsl1, fsl2]}}
    
    rng = np.random.default_rng(seed=42)

    F01_coeffs = V0h.coeff_space.zeros()
    rng.random(size=F01_coeffs._data.shape, dtype='float64', out=F01_coeffs._data)
    F11_coeffs = V1h.coeff_space.zeros()
    for block in F11_coeffs.blocks:
        rng.random(size=block._data.shape, dtype='float64', out=block._data)
    F12_coeffs = V1h.coeff_space.zeros()
    for block in F12_coeffs.blocks:
        rng.random(size=block._data.shape, dtype='float64', out=block._data)

    F01_field   = FemField(V0h, F01_coeffs)
    F11_field   = FemField(V1h, F11_coeffs)
    F12_field   = FemField(V1h, F12_coeffs)

    bilinear_forms = {  'Q':            {'trial' :'V1', 'test':'V1',
                                         'expr'  :dot(cross(F11, v1), cross(F11, v2)),
                                         'fields':[F11_field, ]},
                            
                        'equilibrium':  {'trial' :'V1', 'test':'V1',
                                         'expr'  :dot(cross(F11, v1), cross(v2, F12)),
                                         'fields':[F11_field, F12_field]},

                        'Elena':        {'trial' :'V1', 'test':'V1',
                                         'expr'  :dot(F01*v1, v2),
                                         'fields':[F01_field, ]}
                     }
    
    int_0 = lambda expr: integral(domain, expr)
    
    for bf_name, bf_data in bilinear_forms.items():

        trial_space = spaces[bf_data['trial']]
        Vh          = trial_space['Vh']
        u           = trial_space['funcs'][0]
        test_space  = spaces[bf_data['test']]
        Wh          = test_space ['Vh']
        v           = test_space['funcs'][1]
        expr        = bf_data['expr']
        fields      = bf_data['fields']

        a = BilinearForm((u, v), int_0(expr))

        t0 = time.time()
        ah_old = discretize(a, domain_h, (Vh, Wh), backend=backend, fast_assembly=False)
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
        elif bf_name == 'Elena':
            t0_old  = time.time()
            A_old   = ah_old.assemble(F01=fields[0])
            t1_old  = time.time()
            
            t0      = time.time()
            A       = ah.assemble(F01=fields[0])
            t1      = time.time()
            
        assembly_time_old   = t1_old - t0_old
        assembly_time       = t1     - t0

        A_old_arr   = A_old.toarray()
        A_arr       = A.toarray()
        A_old_norm  = np.linalg.norm(A_old_arr)
        A_norm      = np.linalg.norm(A_arr)

        err         = np.linalg.norm(A_old_arr - A_arr)
        rel_err     = err / A_old_norm

        assert rel_err < 1e-12 # arbitrary rel. error bound (How to test better?)

        print(f' >>> Mapping: {mapping}')
        print(f' >>> BilinearForm: {bf_name}')
        print(f' >>> Discretization in: Old {discretization_time_old:.3g} \t\t || New {discretization_time:.3g} \t\t || Old/New {discretization_time_old/discretization_time:.3g}')
        print(f' >>> Assembly in: Old {assembly_time_old:.3g} \t \t || New {assembly_time:.3g} \t\t || Old/New {assembly_time_old/assembly_time:.3g}')
        print(f' >>>      Error: {err:.3g}')
        print(f' >>> Rel. Error: {rel_err:.3g}')
        print(f' >>>      Norms: ||A_old|| = {A_old_norm:.3g} \t\t || ||A|| = {A_norm:.3g}')
        print()

'''
@pytest.mark.parametrize('geometry', ('collela_3d.h5', 'identity_3d.h5'))
def _test_bspline_mapping(geometry):
    comm    = MPI.COMM_WORLD
    backend = PSYDAC_BACKEND_GPYCCEL

    filename = os.path.join(mesh_dir, geometry)

    domain = Domain.from_file(filename=filename)
    derham = Derham(domain)

    domain_h = discretize(domain, filename=filename, comm=comm)
    derham_h = discretize(derham, domain_h, degree=domain.mapping.get_callable_mapping().space.degree)

    fs = lambda x, y, z: 1
    fv = (fs, fs, fs)

    P0, P1, P2, P3 = derham_h.projectors()

    fs0 = P0(fs).coeffs
    fv1 = P1(fv).coeffs
    fv2 = P2(fv).coeffs
    fs3 = P3(fs).coeffs

    V0 = derham.V0
    V1 = derham.V1
    V2 = derham.V2
    V3 = derham.V3

    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2
    V3h = derham_h.V3

    u0, v0 = elements_of(V0, names='u0, v0')
    u1, v1 = elements_of(V1, names='u1, v1')
    u2, v2 = elements_of(V2, names='u2, v2')
    u3, v3 = elements_of(V3, names='u3, v3')

    a0 = BilinearForm((u0, v0), integral(domain, u0*v0))
    a1 = BilinearForm((u1, v1), integral(domain, dot(u1, v1)))
    a2 = BilinearForm((u2, v2), integral(domain, dot(u2, v2)))
    a3 = BilinearForm((u3, v3), integral(domain, u3*v3))

    t0 = time.time()
    a0h = discretize(a0, domain_h, (V0h, V0h), backend=backend)
    t1 = time.time()
    print(f'a0 discretized in {t1-t0:.3g}s')
    t0 = time.time()
    a1h = discretize(a1, domain_h, (V1h, V1h), backend=backend)
    t1 = time.time()
    print(f'a1 discretized in {t1-t0:.3g}s')
    t0 = time.time()
    a2h = discretize(a2, domain_h, (V2h, V2h), backend=backend)
    t1 = time.time()
    print(f'a2 discretized in {t1-t0:.3g}s')
    t0 = time.time()
    a3h = discretize(a3, domain_h, (V3h, V3h), backend=backend)
    t1 = time.time()
    print(f'a3 discretized in {t1-t0:.3g}s')

    t0 = time.time()
    M0 = a0h.assemble()
    M1 = a1h.assemble()
    M2 = a2h.assemble()
    M3 = a3h.assemble()
    t1 = time.time()
    print(f'Matrices assembled in {t1-t0:.3g}s')

    v0 = M0.dot_inner(fs0, fs0)
    v1 = M1.dot_inner(fv1, fv1)
    v2 = M2.dot_inner(fv2, fv2)
    v3 = M3.dot_inner(fs3, fs3)

    print(v0, v1, v2, v3)
'''
