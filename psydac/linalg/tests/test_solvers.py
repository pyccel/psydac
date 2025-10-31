import  time
import  numpy as np
import  pytest

from    sympy import sin, pi, sqrt, Tuple

from    scipy.sparse                import dia_matrix

from    sympde.calculus             import inner, cross
from    sympde.expr                 import integral, LinearForm, BilinearForm, EssentialBC
from    sympde.topology             import element_of, elements_of, Derham, Mapping, Line, Square, Cube, Union, NormalVector, ScalarFunctionSpace, VectorFunctionSpace
from    sympde.topology.datatype    import SpaceType, H1Space, HcurlSpace

from    psydac.api.discretization   import discretize
from    psydac.api.essential_bc     import apply_essential_bc
from    psydac.api.settings         import PSYDAC_BACKEND_GPYCCEL
from    psydac.ddm.cart             import DomainDecomposition, CartDecomposition
from    psydac.fem.basic            import FemSpace
from    psydac.linalg.basic         import LinearOperator, Vector, IdentityOperator
from    psydac.linalg.block         import BlockVectorSpace, BlockLinearOperator
from    psydac.linalg.kron          import KroneckerLinearSolver, KroneckerStencilMatrix
from    psydac.linalg.solvers       import inverse
from    psydac.linalg.stencil       import StencilVectorSpace, StencilMatrix, StencilVector
from    psydac.linalg.tests.test_kron_direct_solver import matrix_to_bandsolver
from    psydac.linalg.direct_solvers                import BandedSolver

def define_data_hermitian(n, p, dtype=float):
    domain_decomposition = DomainDecomposition([n - p], [False])
    cart = CartDecomposition(domain_decomposition, [n], [np.array([0])], [np.array([n - 1])], [p], [1])
    # ... Vector Spaces
    V = StencilVectorSpace(cart,dtype=dtype)
    e = V.ends[0]
    s = V.starts[0]

    # Build banded matrix with 2p+1 diagonals: must be symmetric and positive definite
    # Here we assign value 2*p on main diagonal and -1 on other diagonals
    if dtype==complex:
        factor=1+1j
    else:
        factor=1
    A = StencilMatrix(V, V)
    A[:, -p:0] = 1-1*factor
    A[:, 0:1] = 2 * p
    A[:, 1:p + 1] = 1-1*factor.conjugate()
    A.remove_spurious_entries()

    # Build exact solution
    xe = StencilVector(V)
    xe[s:e + 1] = factor*np.random.random(e + 1 - s)
    return(V, A, xe)

def define_data(n, p, matrix_data, dtype=float):
    domain_decomposition = DomainDecomposition([n - p], [False])
    cart = CartDecomposition(domain_decomposition, [n], [np.array([0])], [np.array([n - 1])], [p], [1])
    # ... Vector Spaces
    V = StencilVectorSpace(cart, dtype=dtype)
    e = V.ends[0]
    s = V.starts[0]

    # Build banded matrix with 2p+1 diagonals: must be symmetric and positive definite
    # Here we assign value 2*p on main diagonal and -1 on other diagonals

    A = StencilMatrix(V, V)
    A[:, -p:0] = -matrix_data[0]
    A[:, 0:1] = matrix_data[1]
    A[:, 1:p + 1] = matrix_data[2]
    A.remove_spurious_entries()

    # Build exact solution
    xe = StencilVector(V)
    xe[s:e + 1] = np.random.random(e + 1 - s)
    return(V, A, xe)

class SquareTorus(Mapping):

    _expressions = {'x': 'x1 * cos(x2)',
                    'y': 'x1 * sin(x2)',
                    'z': 'x3'}
    
    _ldim        = 3
    _pdim        = 3

class Annulus(Mapping):

    _expressions = {'x': 'x1 * cos(x2)',
                    'y': 'x1 * sin(x2)'}
    
    _ldim        = 2
    _pdim        = 2

class SinMapping1D(Mapping):

    _expressions = {'x': 'sin((pi/2)*x1)'}
    
    _ldim        = 1
    _pdim        = 1

def _test_LO_equality_using_rng(A, B):
    """
    A simple tool to check with almost certainty that two linear operators are identical, 
    by applying them repeatedly to random vectors.
    
    """

    assert isinstance(A, LinearOperator)
    assert isinstance(B, LinearOperator)
    assert A.domain is B.domain
    assert A.codomain is B.codomain

    rng = np.random.default_rng(42)

    x   = A.domain.zeros()
    y1  = A.codomain.zeros()
    y2  = y1.copy()

    n   = 10

    for _ in range(n):

        x *= 0.

        if isinstance(A.domain, BlockVectorSpace):
            for block in x.blocks:
                rng.random(size=block._data.shape, dtype="float64", out=block._data)
        else:
            rng.random(size=x._data.shape, dtype="float64", out=x._data)

        A.dot(x, out=y1)
        B.dot(x, out=y2)

        diff = y1 - y2
        err  = A.codomain.inner(diff, diff)
        
        assert err < 1e-15

class DirichletBoundaryProjector(LinearOperator):

    def __init__(self, fem_space, bcs=None, space_kind=None):

        assert isinstance(fem_space, FemSpace)

        coeff_space    = fem_space.coeff_space
        self._domain   = coeff_space
        self._codomain = coeff_space

        if bcs is not None:
            self._bcs = bcs
        else:
            self._bcs = self._get_bcs(fem_space, space_kind=space_kind)

    @property
    def domain(self):
        return self._domain
    
    @property
    def codomain(self):
        return self._domain
    
    @property
    def dtype(self):
        return None
    
    @property
    def bcs(self):
        return self._bcs
    
    def tosparse(self):
        raise NotImplementedError
    
    def toarray(self):
        raise NotImplementedError
    
    def transpose(self, conjugate=False):
        return self
    
    def _get_bcs(self, fem_space, space_kind=None):
        """Returns the correct Dirichlet boundary conditions for the passed fem_space."""
        space    = fem_space.symbolic_space
        periodic = fem_space.periodic

        space_kind_str = space.kind.name
        if space_kind is not None:
            # Check whether kind is a valid input
            if isinstance(space_kind, str):
                kind_str = space_kind.lower()
                assert(kind_str in ['h1', 'hcurl', 'hdiv', 'l2', 'undefined'])
            elif isinstance(space_kind, SpaceType):
                kind_str = space_kind.name
            else:
                raise TypeError(f'Expecting space_kind {space_kind} to be a str or of SpaceType')
            
            # If fem_space has a kind, it must be compatible with kind
            if space_kind_str != 'undefined':
                assert space_kind_str == kind_str, f'fem_space and space_kind are not compatible.'
            else:
                # If space_kind_str = 'undefined': Update the variable using kind
                space_kind_str = kind_str


        kind = space_kind_str
        dim  = space.domain.dim
        
        if kind == 'l2':
            return None
        
        u = element_of(space, name="u")
        ebcs = [EssentialBC(u, 0, side, position=0) for side in space.domain.boundary]

        if kind == "h1":
            bcs = [ebcs[0], ebcs[1]] if periodic[0] == False else []
            if dim >= 2:
                bcs += [ebcs[2], ebcs[3]] if periodic[1] == False else []
            if dim == 3:
                bcs += [ebcs[4], ebcs[5]] if periodic[2] == False else []

        elif kind == 'hcurl':
            assert dim in (2, 3)
            bcs_x = [ebcs[2], ebcs[3]] if periodic[1] == False else []
            if dim == 3:
                bcs_x += [ebcs[4], ebcs[5]] if periodic[2] == False else []
            bcs_y = [ebcs[0], ebcs[1]] if periodic[0] == False else []
            if dim == 3:
                bcs_y += [ebcs[4], ebcs[5]] if periodic[2] == False else []
            if dim == 3:
                bcs_z = [ebcs[0], ebcs[1]] if periodic[0] == False else []
                bcs_z += [ebcs[2], ebcs[3]] if periodic[1] == False else []
            bcs = [bcs_x, bcs_y]
            if dim == 3:
                bcs.append(bcs_z)

        elif kind == 'hdiv':
            assert dim in (2, 3)
            bcs_x = [ebcs[0], ebcs[1]] if periodic[0] == False else []
            bcs_y = [ebcs[2], ebcs[3]] if periodic[1] == False else []
            if dim == 3:
                bcs_z = [ebcs[4], ebcs[5]] if periodic[2] == False else []
            bcs = [bcs_x, bcs_y]
            if dim == 3:
                bcs.append(bcs_z)

        else:
            raise ValueError(f'{kind} must be either "h1", "hcurl" or "hdiv"')
        
        return bcs

    def dot(self, v, out=None):
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space is self.codomain
        else:
            out = self.codomain.zeros()

        v.copy(out=out)
        if isinstance(self.domain, StencilVectorSpace):
            apply_essential_bc(out, *self._bcs)
        else:
            for block, block_bcs in zip(out, self._bcs):
                apply_essential_bc(block, *block_bcs)

        return out

#===============================================================================
@pytest.mark.parametrize( 'n', [5, 10, 13] )
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('solver', ['cg', 'pcg', 'bicg', 'bicgstab', 'pbicgstab', 'minres', 'lsmr', 'gmres'])

def test_solver_tridiagonal(n, p, dtype, solver, verbose=False):

    #---------------------------------------------------------------------------
    # PARAMETERS
    #---------------------------------------------------------------------------

    if solver in ['bicg', 'bicgstab', 'pbicgstab', 'lsmr']:
        if dtype==complex:
            diagonals = [1-10j,6+9j,3+5j]
        else:
            diagonals = [1,6,3]
            
        if solver == 'pbicgstab' and dtype == complex:
            # pbicgstab only works for real matrices
            return
    elif solver == 'gmres':
        if dtype==complex:
            diagonals = [-7-2j,-6-2j,-1-10j]
        else:
            diagonals = [-7,-1,-3]

    if solver in ['cg', 'pcg', 'minres']:
        # pcg runs with Jacobi preconditioner
        V, A, xe = define_data_hermitian(n, p, dtype=dtype)
        if solver == 'minres' and dtype == complex:
            # minres only works for real matrices
            return
    else:
        V, A, xe = define_data(n, p, diagonals, dtype=dtype)

    # Tolerance for success: 2-norm of error in solution
    tol = 1e-8

    #---------------------------------------------------------------------------
    # TEST
    #---------------------------------------------------------------------------
    if verbose:
        # Title
        print()
        print( "="*80 )
        print( f"SERIAL TEST: solve linear system A*x = b using {solver}")
        print( "="*80 )
        print()

    #Create the solvers
    if solver in ['pcg', 'pbicgstab']:
        pc = A.diagonal(inverse=True)
        solv = inverse(A, solver, pc=pc, tol=1e-13, verbose=verbose, recycle=True)
    else:
        solv = inverse(A, solver, tol=1e-13, verbose=verbose, recycle=True)
    solvt = solv.transpose()
    solvh = solv.H
    solv2 = inverse(A@A, solver, tol=1e-13, verbose=verbose, recycle=True) # Test solver of composition of operators

    # Manufacture right-hand-side vector from exact solution
    be  = A @ xe
    be2 = A @ be # Test solver with consecutive solves
    bet = A.T @ xe
    beh = A.H @ xe

    # Solve linear system
    # Assert x0 got updated correctly and is not the same object as the previous solution, but just a copy
    x = solv @ be
    info = solv.get_info()
    solv_x0 = solv._options["x0"]
    assert np.array_equal(x.toarray(), solv_x0.toarray())
    assert x is not solv_x0

    x2 = solv @ be2
    solv_x0 = solv._options["x0"]
    assert np.array_equal(x2.toarray(), solv_x0.toarray())
    assert x2 is not solv_x0

    xt = solvt.solve(bet)
    solvt_x0 = solvt._options["x0"]
    assert np.array_equal(xt.toarray(), solvt_x0.toarray())
    assert xt is not solvt_x0

    xh = solvh.dot(beh)
    solvh_x0 = solvh._options["x0"]
    assert np.array_equal(xh.toarray(), solvh_x0.toarray())
    assert xh is not solvh_x0

    if solver != 'pcg':
        # PCG only works with operators with diagonal
        xc = solv2 @ be2
        solv2_x0 = solv2._options["x0"]
        assert np.array_equal(xc.toarray(), solv2_x0.toarray())
        assert xc is not solv2_x0


    # Verify correctness of calculation: 2-norm of error
    b = A @ x
    b2 = A @ x2
    bt = A.T @ xt
    bh = A.H @ xh
    if solver != 'pcg':
        bc = A @ A @ xc

    err = b - be
    err_norm = np.linalg.norm( err.toarray() )
    err2 = b2 - be2
    err2_norm = np.linalg.norm( err2.toarray() )
    errt = bt - bet
    errt_norm = np.linalg.norm( errt.toarray() )
    errh = bh - beh
    errh_norm = np.linalg.norm( errh.toarray() )

    if solver != 'pcg': 
        errc = bc - be2
        errc_norm = np.linalg.norm( errc.toarray() )

    #---------------------------------------------------------------------------
    # TERMINAL OUTPUT
    #---------------------------------------------------------------------------
    if verbose:
        print()
        print( 'A  =', A, sep='\n' )
        print( 'b  =', b )
        print( 'x  =', x )
        print( 'xe =', xe )
        print( 'info =', info )
        print()

        print( "-"*40 )
        print( f"2-norm of error in solution = {err_norm:.2e}" )
        if err_norm < tol:
            print( "PASSED" )
        else:
            print( "FAIL" )
        print( "-"*40 )

    #---------------------------------------------------------------------------
    # PYTEST
    #---------------------------------------------------------------------------
    # The lsmr solver does not consistently produce outputs x whose error ||Ax - b|| is less than tol.
    if solver != 'lsmr':
        assert err_norm < tol
        assert err2_norm < tol
        assert errt_norm < tol
        assert errh_norm < tol
        assert solver == 'pcg' or errc_norm < tol

#===============================================================================
def test_function_space_dirichlet_projector():

    ncells_3d   = [8, 8, 8]
    degree_3d   = [2, 2, 2]
    periodic_3d = [False, True, False]

    comm     = None
    backend  = PSYDAC_BACKEND_GPYCCEL

    logical_domain_1d = Line  ('L', bounds= (0,   1))
    logical_domain_2d = Square('S', bounds1=(0.5, 1), bounds2=(0, 2*np.pi))
    logical_domain_3d = Cube  ('C', bounds1=(0.5, 1), bounds2=(0, 2*np.pi), bounds3=(0, 1))
    logical_domains   = [logical_domain_1d, logical_domain_2d, logical_domain_3d]

    mapping_1d = SinMapping1D('LM')
    mapping_2d = Annulus     ('A' )
    mapping_3d = SquareTorus ('ST')
    mappings   = [mapping_1d, mapping_2d, mapping_3d]

    dims = [1, 2, 3]
    rng  = np.random.default_rng(42)

    print()
    for dim in dims:
        print(f' ----- Test projectors in dimension {dim} -----')
        print()

        domain        = mappings[dim-1](logical_domains[dim-1])
        from sympde.utilities.utils import plot_domain
        #plot_domain(domain, draw=True, isolines=True)

        # Obtain "true" boundary, i.e., remove periodic y-direction boundary
        if dim == 1:
            boundary  = domain.boundary
        elif dim == 2:
            boundary  = Union(domain.get_boundary(axis=0, ext=-1), domain.get_boundary(axis=0, ext=1))
        else:
            boundary  = Union(domain.get_boundary(axis=0, ext=-1), domain.get_boundary(axis=0, ext=1),
                              domain.get_boundary(axis=2, ext=-1), domain.get_boundary(axis=2, ext=1))
            
        ncells    = [ncells_3d[0], ]   if dim == 1 else ncells_3d  [0:dim]
        degree    = [degree_3d[0], ]   if dim == 1 else degree_3d  [0:dim]
        periodic  = [periodic_3d[0], ] if dim == 1 else periodic_3d[0:dim]

        domain_h = discretize(domain, ncells=ncells, periodic=periodic, comm=comm)

        nn            = NormalVector('nn')

        for i in range(dim):
            print(f'      - Test DBP{i}')

            # The function defined here satisfy the corresponding homogeneous Dirichlet BCs
            if dim == 1:
                x = domain.coordinates
                V = ScalarFunctionSpace('V', domain, kind='H1')
                f = sin(2*pi*x)
            if dim == 2:
                x, y = domain.coordinates
                if i == 0:
                    V  = ScalarFunctionSpace('V', domain, kind=H1Space)
                    f  = (sqrt(x**2 + y**2)-0.5) * (sqrt(x**2 + y**2)-1)
                else:
                    V  = VectorFunctionSpace('V', domain, kind='hCuRl')
                    f1 = x
                    f2 = y
                    f  = Tuple(f1, f2)
            if dim == 3:
                x, y, z = domain.coordinates
                if i == 0:
                    V  = ScalarFunctionSpace('V', domain, kind='h1')
                    f  = (sqrt(x**2 + y**2)-0.5) * (sqrt(x**2 + y**2)-1) * z * (z-1)
                elif i == 1:
                    V  = VectorFunctionSpace('V', domain, kind=HcurlSpace)
                    f1 = z * (z - 1) * x
                    f2 = z * (z - 1) * y
                    f3 = (sqrt(x**2 + y**2)-0.5) * (sqrt(x**2 + y**2)-1)
                    f  = Tuple(f1, f2, f3)
                else:
                    V  = VectorFunctionSpace('V', domain, kind='Hdiv')
                    f1 = (sqrt(x**2 + y**2)-0.5) * (sqrt(x**2 + y**2)-1)
                    f2 = (sqrt(x**2 + y**2)-0.5) * (sqrt(x**2 + y**2)-1)
                    f3 = z * (z-1) * sin(x*y)
                    f  = Tuple(f1, f2, f3)

            u, v = elements_of(V, names='u, v')
            if i == 0:
                boundary_expr = u*v
            if (i == 1) and (dim == 2):
                boundary_expr = cross(nn, u) * cross(nn, v)
            if (i == 1) and (dim == 3):
                boundary_expr = inner(cross(nn, u), cross(nn, v))
            if i == 2:
                boundary_expr = inner(nn, u) * inner(nn, v)

            Vh   = discretize(V, domain_h, degree=degree)
            expr = inner(u, v) if isinstance(Vh.coeff_space, BlockVectorSpace) else u*v

            a   = BilinearForm((u, v), integral(domain,            expr))            
            ab  = BilinearForm((u, v), integral(boundary, boundary_expr))

            ah  = discretize(a,  domain_h, (Vh, Vh), backend=backend)
            abh = discretize(ab, domain_h, (Vh, Vh), backend=backend, sum_factorization=False)

            I   = IdentityOperator(Vh.coeff_space)
            DBP = DirichletBoundaryProjector(Vh)

            M   = ah.assemble()
            M_0 = DBP @ M @ DBP + (I - DBP)
            Mb  = abh.assemble()

            # We project f into the conforming discrete space using a penalization method. It's coefficients are stored in fc
            lexpr = inner(v, f) if isinstance(Vh.coeff_space, BlockVectorSpace) else v*f
            l = LinearForm(v, integral(domain, lexpr))
            lh = discretize(l, domain_h, Vh, backend=backend)
            rhs = lh.assemble()
            A = M + 1e30*Mb
            A_inv = inverse(A, 'cg', maxiter=1000, tol=1e-10)
            fc = A_inv @ rhs

            # 1.
            # In 1D, 2D, 3D, the coefficients of functions satisfying homogeneous Dirichlet 
            # boundary conditions should not change under application of the corresponding projector
            fc2  = DBP @ fc
            diff = fc - fc2
            err  = np.linalg.norm(diff.toarray())
            print(f' | f - P @ f |          = {err}')
            assert err < 1e-15

            # 2.
            # After applying a projector to a random vector, we want to verify that the 
            # corresponding boundary integral vanishes
            rdm_coeffs = Vh.coeff_space.zeros()
            print(' Random boundary integrals:')
            for _ in range(3):
                if isinstance(rdm_coeffs.space, BlockVectorSpace):
                    for block in rdm_coeffs.blocks:
                        rng.random(size=block._data.shape, dtype="float64", out=block._data)
                else:
                    rng.random(size=rdm_coeffs._data.shape, dtype="float64", out=rdm_coeffs._data)
                rdm_coeffs2 = DBP @ rdm_coeffs
                boundary_int_rdm = Mb.dot_inner(rdm_coeffs, rdm_coeffs)
                boundary_int_proj_rdm = Mb.dot_inner(rdm_coeffs2, rdm_coeffs2)
                print(f'  rdm: {boundary_int_rdm}    proj. rdm: {boundary_int_proj_rdm}')
                assert boundary_int_proj_rdm < 1e-15

            # 3.
            # We want to verify that applying a projector twice does not change the vector twice
            fc3  = DBP @ fc2
            diff = fc2 - fc3
            err  = np.linalg.norm(diff.toarray())
            print(f' | P @ f - P @ P @ f |  = {err}')
            assert err == 0.

            # 4.
            # Finally, the modified mass matrix should still compute inner products correctly
            l2_norm_squared  = M.dot_inner  (fc, fc)
            l2_norm_squared2 = M_0.dot_inner(fc, fc)
            diff             = l2_norm_squared - l2_norm_squared2
            print(f' ||   f   ||^2          = {l2_norm_squared} should be equal to')
            print(f' || P @ f ||^2          = {l2_norm_squared2}')
            assert diff < 1e-15

            print()

        print()

#===============================================================================
def test_discrete_derham_dirichlet_projector():

    ncells   = [8, 8, 8]
    degree   = [2, 2, 2]
    periodic = [False, True, False]

    comm     = None
    backend  = PSYDAC_BACKEND_GPYCCEL

    logical_domain_1d = Line  ('L', bounds= (0,   1))
    logical_domain_2d = Square('S', bounds1=(0.5, 1), bounds2=(0, 2*np.pi))
    logical_domain_3d = Cube  ('C', bounds1=(0.5, 1), bounds2=(0, 2*np.pi), bounds3=(0, 1))
    logical_domains   = [logical_domain_1d, logical_domain_2d, logical_domain_3d]

    mapping_1d = SinMapping1D('LM')
    mapping_2d = Annulus     ('A' )
    mapping_3d = SquareTorus ('ST')
    mappings   = [mapping_1d, mapping_2d, mapping_3d]

    dims = [1, 2, 3]
    rng  = np.random.default_rng(42)

    # The following are functions (1D, 2D & 3D) satisfying homogeneous Dirichlet BCs

    f11     = lambda x : np.sin(2*np.pi*x)

    r2      = lambda x, y : np.sqrt(x**2 + y**2)
    f21     = lambda x, y : (r2(x, y) - 0.5) * (r2(x, y) - 1)
    f22_1   = lambda x, y : x
    f22_2   = lambda x, y : y
    f22     = (f22_1, f22_2)

    f31     = lambda x, y, z : (r2(x, y) - 0.5) * (r2(x, y) - 1) * z * (z - 1)
    f32_1   = lambda x, y, z : z * (z - 1) * x
    f32_2   = lambda x, y, z : z * (z - 1) * y
    f32_3   = lambda x, y, z : (r2(x, y) - 0.5) * (r2(x, y) - 1)
    f32     = (f32_1, f32_2, f32_3)
    f33_1   = lambda x, y, z : (r2(x, y) - 0.5) * (r2(x, y) - 1)
    f33_2   = lambda x, y, z : (r2(x, y) - 0.5) * (r2(x, y) - 1)
    f33_3   = lambda x, y, z : z * (z - 1) * np.sin(x*y)
    f33     = (f33_1, f33_2, f33_3)

    funs    = [[f11], [f21, f22], [f31, f32, f33]]

    print()
    for dim in dims:
        print(f' ----- Test projectors in dimension {dim} -----')
        print()

        domain        = mappings[dim-1](logical_domains[dim-1])
        from sympde.utilities.utils import plot_domain
        #plot_domain(domain, draw=True, isolines=True)

        # Obtain "true" boundary, i.e., remove periodic y-direction boundary
        if dim == 1:
            boundary  = domain.boundary
        elif dim == 2:
            boundary  = Union(domain.get_boundary(axis=0, ext=-1), domain.get_boundary(axis=0, ext=1))
        else:
            boundary  = Union(domain.get_boundary(axis=0, ext=-1), domain.get_boundary(axis=0, ext=1),
                              domain.get_boundary(axis=2, ext=-1), domain.get_boundary(axis=2, ext=1))

        derham        = Derham(domain) if dim in (1, 3) else Derham(domain, sequence=['h1', 'hcurl', 'l2'])

        ncells_dim    = [ncells[0], ] if dim == 1 else ncells[0:dim]
        degree_dim    = [degree[0], ] if dim == 1 else degree[0:dim]
        periodic_dim  = [periodic[0], ] if dim == 1 else periodic[0:dim]

        domain_h      = discretize(domain, ncells=ncells_dim, periodic=periodic_dim, comm=comm)
        derham_h      = discretize(derham, domain_h, degree=degree_dim)

        db_projectors = derham_h.dirichlet_projectors(kind='linop')

        if dim == 2: 
            conf_projectors = derham_h.conforming_projectors(kind='linop', hom_bc=True)

        nn            = NormalVector('nn')

        for i in range(dim):
            print(f'      - Test DBP{i}')

            u, v = elements_of(derham.spaces[i], names='u, v')

            if i == 0:
                boundary_expr = u*v
            if (i == 1) and (dim == 2):
                boundary_expr = cross(nn, u) * cross(nn, v)
            if (i == 1) and (dim == 3):
                boundary_expr = inner(cross(nn, u), cross(nn, v))
            if i == 2:
                boundary_expr = inner(nn, u) * inner(nn, v)

            expr = inner(u, v) if isinstance(derham_h.spaces[i].coeff_space, BlockVectorSpace) else u*v

            a   = BilinearForm((u, v), integral(domain,            expr))            
            ab  = BilinearForm((u, v), integral(boundary, boundary_expr))

            ah  = discretize(a,  domain_h, (derham_h.spaces[i], derham_h.spaces[i]), backend=backend)
            abh = discretize(ab, domain_h, (derham_h.spaces[i], derham_h.spaces[i]), backend=backend, sum_factorization=False)

            I   = IdentityOperator(derham_h.spaces[i].coeff_space)
            DBP = db_projectors[i]

            if dim == 2: 
                CP = conf_projectors[i]
                _test_LO_equality_using_rng(DBP, CP)

            M   = ah.assemble()
            M_0 = DBP @ M @ DBP + (I - DBP)
            Mb  = abh.assemble()

            f   = funs[dim-1][i]
            fc  = derham_h.projectors()[i](f).coeffs

            # 1.
            # In 1D, 2D, 3D, the coefficients of functions satisfying homogeneous Dirichlet 
            # boundary conditions should not change under application of the corresponding projector
            fc2  = DBP @ fc
            diff = fc - fc2
            err  = np.linalg.norm(diff.toarray())
            print(f' | f - P @ f |          = {err}')
            assert err < 1e-15

            # 2.
            # After applying a projector to a random vector, we want to verify that the 
            # corresponding boundary integral vanishes
            rdm_coeffs = derham_h.spaces[i].coeff_space.zeros()
            print(' Random boundary integrals:')
            for _ in range(3):
                if isinstance(rdm_coeffs.space, BlockVectorSpace):
                    for block in rdm_coeffs.blocks:
                        rng.random(size=block._data.shape, dtype="float64", out=block._data)
                else:
                    rng.random(size=rdm_coeffs._data.shape, dtype="float64", out=rdm_coeffs._data)
                rdm_coeffs2 = DBP @ rdm_coeffs
                boundary_int_rdm = Mb.dot_inner(rdm_coeffs, rdm_coeffs)
                boundary_int_proj_rdm = Mb.dot_inner(rdm_coeffs2, rdm_coeffs2)
                print(f'  rdm: {boundary_int_rdm}    proj. rdm: {boundary_int_proj_rdm}')
                assert boundary_int_proj_rdm < 1e-15

            # 3.
            # We want to verify that applying a projector twice does not change the vector twice
            fc3  = DBP @ fc2
            diff = fc2 - fc3
            err  = np.linalg.norm(diff.toarray())
            print(f' | P @ f - P @ P @ f |  = {err}')
            assert err == 0.

            # 4.
            # Finally, the modified mass matrix should still compute inner products correctly
            l2_norm_squared  = M.dot_inner  (fc, fc)
            l2_norm_squared2 = M_0.dot_inner(fc, fc)
            diff             = l2_norm_squared - l2_norm_squared2
            print(f' ||   f   ||^2          = {l2_norm_squared} should be equal to')
            print(f' || P @ f ||^2          = {l2_norm_squared2}')
            assert diff < 1e-15

            print()

        print()

#===============================================================================
class DirichletMultipatchBoundaryProjector(LinearOperator):

    def __init__(self, fem_space, bcs=None, space_kind=None):

        assert isinstance(fem_space, FemSpace)
        assert fem_space.is_multipatch

        coeff_space    = fem_space.coeff_space
        self._domain   = coeff_space
        self._codomain = coeff_space

        if bcs is not None:
            self._bcs = bcs
        else:
            self._bcs = self._get_bcs(fem_space, space_kind=space_kind)

    @property
    def domain(self):
        return self._domain
    
    @property
    def codomain(self):
        return self._domain
    
    @property
    def dtype(self):
        return None
    
    @property
    def bcs(self):
        return self._bcs
    
    def tosparse(self):
        raise NotImplementedError
    
    def toarray(self):
        raise NotImplementedError
    
    def transpose(self, conjugate=False):
        return self
    
    def _get_bcs(self, fem_space, space_kind=None):
        """Returns the correct Dirichlet boundary conditions for the passed fem_space."""
        space    = fem_space.symbolic_space
        periodic = fem_space.periodic

        space_kind_str = space.kind.name
        if space_kind is not None:
            # Check whether kind is a valid input
            if isinstance(space_kind, str):
                kind_str = space_kind.lower()
                assert(kind_str in ['h1', 'hcurl', 'hdiv', 'l2', 'undefined'])
            elif isinstance(space_kind, SpaceType):
                kind_str = space_kind.name
            else:
                raise TypeError(f'Expecting space_kind {space_kind} to be a str or of SpaceType')
            
            # If fem_space has a kind, it must be compatible with kind
            if space_kind_str != 'undefined':
                assert space_kind_str == kind_str, f'fem_space and space_kind are not compatible.'
            else:
                # If space_kind_str = 'undefined': Update the variable using kind
                space_kind_str = kind_str

        kind = space_kind_str
        dim  = space.domain.dim
        assert dim==2 
        
        if kind == 'l2':
            return None
        
        u = element_of(space, name="u")

        if kind == "h1":
            bcs = [EssentialBC(u, 0, side, position=0) for side in space.domain.boundary]


        elif kind == 'hcurl':
            bcs_x = []
            bcs_y = []

            for bn in space.domain.boundary:
                if bn.axis == 0:
                    bcs_y.append(EssentialBC(u, 0, bn, position=0))
                elif bn.axis == 1:
                    bcs_x.append(EssentialBC(u, 0, bn, position=0))

            bcs = [bcs_x, bcs_y]


        elif kind == 'hdiv':
            bcs_x = []
            bcs_y = []

            for bn in space.domain.boundary:
                if bn.axis == 1:
                    bcs_y.append(EssentialBC(u, 0, bn, position=0))
                elif bn.axis == 0:
                    bcs_x.append(EssentialBC(u, 0, bn, position=0))

            bcs = [bcs_x, bcs_y]

        else:
            raise ValueError(f'{kind} must be either "h1", "hcurl" or "hdiv"')
        
        return bcs

    def dot(self, v, out=None):
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space is self.codomain
        else:
            out = self.codomain.zeros()

        v.copy(out=out)

        # apply bc on each patch
        for p in out.blocks:

            if isinstance(p, StencilVector):
                apply_essential_bc(p, *self._bcs)
            else:
                for block, block_bcs in zip(p, self._bcs):
                    apply_essential_bc(block, *block_bcs)

        return out

#===============================================================================
def test_discrete_derham_dirichlet_projector_multipatch():

    ncells   = [8, 8]
    degree   = [2, 2]

    comm     = None
    backend  = PSYDAC_BACKEND_GPYCCEL

    from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain
    domain = build_multipatch_domain(domain_name='annulus_3')

    rng = np.random.default_rng(42)

    # The following are functions satisfying homogeneous Dirichlet BCs
    r      = lambda x, y : np.sqrt(x**2 + y**2)
    f1     = lambda x, y : (r(x, y) - 0.5) * (r(x, y) - 1)
    f2_1   = lambda x, y : x
    f2_2   = lambda x, y : y
    f2     = (f2_1, f2_2)
    funs   = [f1, f2]
    print()

    boundary = domain.boundary

    derham = Derham(domain, sequence=['h1', 'hcurl', 'l2'])
    
    ncells_h = {}
    for k, D in enumerate(domain.interior):
        ncells_h[D.name] = ncells

    domain_h = discretize(domain, ncells=ncells_h, comm=comm)
    derham_h = discretize(derham, domain_h, degree=degree)

    projectors = derham_h.projectors(nquads=[(d + 1) for d in degree])

    db_projectors = derham_h.dirichlet_projectors(kind='linop')

    conf_projectors = derham_h.conforming_projectors(kind='linop', hom_bc=True)

    nn = NormalVector('nn')

    for i in range(2):
        print(f'      - Test DBP{i}')

        u, v = elements_of(derham.spaces[i], names='u, v')

        if i == 0:
            boundary_expr = u*v
            expr = u*v
        if (i == 1):
            boundary_expr = cross(nn, u) * cross(nn, v)
            expr = inner(u,v)

        a   = BilinearForm((u, v), integral(domain,            expr))            
        ab  = BilinearForm((u, v), integral(boundary, boundary_expr))

        ah  = discretize(a,  domain_h, (derham_h.spaces[i], derham_h.spaces[i]), backend=backend)
        abh = discretize(ab, domain_h, (derham_h.spaces[i], derham_h.spaces[i]), backend=backend, sum_factorization=False)

        I   = IdentityOperator(derham_h.spaces[i].coeff_space)
        DBP = db_projectors[i]

        M   = ah.assemble()
        M_0 = DBP @ M @ DBP + (I - DBP)
        Mb  = abh.assemble()

        f   = funs[i]
        fc  = projectors[i](f).coeffs

        # 1.
        # The coefficients of functions satisfying homogeneous Dirichlet 
        # boundary conditions should not change under application of the corresponding projector
        fc2  = DBP @ fc
        diff = fc - fc2
        err  = np.linalg.norm(diff.toarray())
        print(f' | f - P @ f |          = {err}')
        assert err < 1e-15

        # 2.
        # After applying a projector to a random vector, we want to verify that the 
        # corresponding boundary integral vanishes
        rdm_coeffs = derham_h.spaces[i].coeff_space.zeros()
        print(' Random boundary integrals:')
        for _ in range(3):
            for patch in rdm_coeffs.blocks:

                if isinstance(patch.space, BlockVectorSpace):
                    for block in patch.blocks:
                        rng.random(size=block._data.shape, dtype="float64", out=block._data)
                else:
                    rng.random(size=patch._data.shape, dtype="float64", out=patch._data)

            rdm_coeffs2 = DBP @ rdm_coeffs
            boundary_int_rdm = Mb.dot_inner(rdm_coeffs, rdm_coeffs)
            boundary_int_proj_rdm = Mb.dot_inner(rdm_coeffs2, rdm_coeffs2)
            print(f'  rdm: {boundary_int_rdm}    proj. rdm: {boundary_int_proj_rdm}')
            assert boundary_int_proj_rdm < 1e-15

        # 3.
        # We want to verify that applying a projector twice does not change the vector twice
        fc3  = DBP @ fc2
        diff = fc2 - fc3
        err  = np.linalg.norm(diff.toarray())
        print(f' | P @ f - P @ P @ f |  = {err}')
        assert err == 0.

        # 4.
        # Finally, the modified mass matrix should still compute inner products correctly
        l2_norm_squared  = M.dot_inner  (fc, fc)
        l2_norm_squared2 = M_0.dot_inner(fc, fc)
        diff             = l2_norm_squared - l2_norm_squared2
        print(f' ||   f   ||^2          = {l2_norm_squared} should be equal to')
        print(f' || P @ f ||^2          = {l2_norm_squared2}')
        assert diff < 1e-15

        print()

# ===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================

if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
