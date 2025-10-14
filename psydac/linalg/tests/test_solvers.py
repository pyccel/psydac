import  time
import  numpy as np
import  pytest

from    scipy.sparse        import dia_matrix

from    sympde.calculus     import inner, cross
from    sympde.expr         import integral, BilinearForm, EssentialBC
from    sympde.topology     import element_of, elements_of, Derham, Mapping, Line, Square, Cube, BasicFunctionSpace, Union, NormalVector

from    psydac.api.discretization   import discretize
from    psydac.api.essential_bc     import apply_essential_bc
from    psydac.api.settings         import PSYDAC_BACKEND_GPYCCEL
from    psydac.ddm.cart             import DomainDecomposition, CartDecomposition
from    psydac.linalg.basic         import LinearOperator, Vector, VectorSpace, IdentityOperator
from    psydac.linalg.block         import BlockVectorSpace, BlockLinearOperator
from    psydac.linalg.kron          import KroneckerLinearSolver
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

def get_LST_preconditioner(derham_h, M0=None, M1=None, M2=None, M3=None, backend=None, bcs=False):
    """
    LST (Loli, Sangalli, Tani) preconditioners are mass matrix preconditioners of the form

    pc = D_inv_sqrt @ D_log_sqrt @ M_log_kron_solver @ D_log_sqrt @ D_inv_sqrt,

    where

    D_inv_sqrt          is the diagonal matrix of the square roots of the inverse diagonal entries of the mass matrix M,
    D_log_sqrt          is the diagonal matrix of the square roots of the diagonal entries of the mass matrix on the logical domain,
    M_log_kron_solver   is the Kronecker Solver of the mass matrix on the logical domain.

    These preconditioners work very well even on complex domains as numerical experiments have shown.

    Upon choosing bcs=True, preconditioner for the modified mass matrices M0_0, M1_0 and M2_0 are being returned.
    The preconditioner for M3 remains identical as there are no BCs to take care of.
    M{i}_0, i=0,1,2, is a mass matrix of the form
    M{i}_0 = DBP @ M{i} @ DBP + (I - DBP)
    where DBP and I are the corresponding DirichletBoundaryProjector and IdentityOperator.
    See examples/vector_potential_3d.
    
    """

    assert derham_h is not None

    dim = derham_h.dim
    assert dim in (2, 3)

    if bcs == True:
        def toarray_1d(A):
            """
            Obtain a numpy array representation of a (1D) LinearOperator (which has not implemented toarray()).
            
            We fill an empty numpy array row by row by repeatedly applying unit vectors
            to the transpose of A. In order to obtain those unit vectors in Stencil format,
            we make use of an auxiliary function that takes periodicity into account.
            """
            
            assert isinstance(A, LinearOperator)
            W = A.codomain
            assert isinstance(W, StencilVectorSpace)

            def get_unit_vector_1d(v, periodic, n1, npts1, pads1):

                v *= 0.0
                v._data[pads1+n1] = 1.

                if periodic:
                    if n1 < pads1:
                        v._data[-pads1+n1] = 1.
                    if n1 >= npts1-pads1:
                        v._data[n1-npts1+pads1] = 1.
                
                return v

            periods  = W.periods
            periodic = periods[0]

            w = W.zeros()
            At = A.T

            A_arr = np.zeros(A.shape, dtype=A.dtype)

            npts1,  = W.npts
            pads1,  = W.pads
            for n1 in range(npts1):
                e_n1 = get_unit_vector_1d(w, periodic, n1, npts1, pads1)
                A_n1 = At @ e_n1
                A_arr[n1, :] = A_n1.toarray()

            return A_arr

        def M0_0_1d_to_bandsolver(A):
            """
            Converts the M0_0_1d StencilMatrix to a BandedSolver.

            Closely resembles a combination of the two functions
            matrix_to_bandsolver & to_bnd
            found in test_kron_direct_solver,
            the difference being that M0_0_1d neither has a 
            remove_spurious_entries()
            nor a 
            toarray()
            function.
            
            """

            dmat = dia_matrix(toarray_1d(A), dtype=A.dtype)
            la   = abs(dmat.offsets.min())
            ua   = dmat.offsets.max()
            cmat = dmat.tocsr()

            A_bnd = np.zeros((1+ua+2*la, cmat.shape[1]), A.dtype)

            for i,j in zip(*cmat.nonzero()):
                A_bnd[la+ua+i-j, j] = cmat[i,j]

            return BandedSolver(ua, la, A_bnd)

    domain_h    = derham_h.domain_h

    domain      = domain_h.domain
    if dim == 2:
        derham  = Derham(domain, derham_h.sequence)
    else:
        derham  = Derham(domain)

    comm        = domain_h.comm
    backend     = backend

    ncells,     = domain_h.ncells.values()
    degree      = derham_h.V0.degree
    periodic,   = domain_h.periodic.values()
    
    logical_domain      = domain.logical_domain
    logical_domain_h    = discretize(logical_domain, ncells=ncells, periodic=periodic, comm=comm)

    if dim == 2:
        Ms = [M0, M1, M2]
    else:
        Ms = [M0, M1, M2, M3]

    # -----

    D_inv_sqrt_arr = []

    for M in Ms:
        if M is not None:
            D_inv_sqrt_arr.append(M.diagonal(inverse=True, sqrt=True))
        else:
            D_inv_sqrt_arr.append(None)

    # -----

    D_log_sqrt_arr = []

    for M, V, Vh in zip(Ms, derham.spaces, derham_h.spaces):
        if M is not None:
            u, v  = elements_of(V, names='u, v')
            expr  = inner(u, v) if isinstance(M.domain, BlockVectorSpace) else u*v
            a     = BilinearForm((u, v), integral(logical_domain, expr))
            ah    = discretize(a, logical_domain_h, (Vh, Vh), backend=backend)
            M_log = ah.assemble()
            D_log_sqrt_arr.append(M_log.diagonal(inverse=False, sqrt=True))
        else:
            D_log_sqrt_arr.append(None)

    # -----

    M_log_kron_solver_arr = []

    logical_domain_1d_x = Line('L', bounds=logical_domain.bounds1)
    logical_domain_1d_y = Line('L', bounds=logical_domain.bounds2)
    if dim == 3:
        logical_domain_1d_z = Line('L', bounds=logical_domain.bounds3)

    logical_domain_1d_list = [logical_domain_1d_x, logical_domain_1d_y]
    if dim == 3:
        logical_domain_1d_list += [logical_domain_1d_z]

    M0_1d_solvers = []
    M1_1d_solvers = []

    for ncells_1d, degree_1d, periodic_1d, logical_domain_1d in zip(ncells, degree, periodic, logical_domain_1d_list):

        derham_1d = Derham(logical_domain_1d)

        logical_domain_1d_h = discretize(logical_domain_1d, ncells=[ncells_1d, ], periodic=[periodic_1d, ])
        derham_1d_h = discretize(derham_1d, logical_domain_1d_h, degree=[degree_1d, ])

        V0_1d,  V1_1d  = derham_1d.spaces
        V0h_1d, V1h_1d = derham_1d_h.spaces

        u0, v0 = elements_of(V0_1d, names='u0, v0')
        u1, v1 = elements_of(V1_1d, names='u1, v1')

        a0_1d = BilinearForm((u0, v0), integral(logical_domain_1d, u0*v0))
        a1_1d = BilinearForm((u1, v1), integral(logical_domain_1d, u1*v1))

        a0h_1d = discretize(a0_1d, logical_domain_1d_h, (V0h_1d, V0h_1d))
        a1h_1d = discretize(a1_1d, logical_domain_1d_h, (V1h_1d, V1h_1d))

        M0_1d = a0h_1d.assemble()
        M1_1d = a1h_1d.assemble()

        if bcs == True:
            DBP0,   = get_DirichletBoundaryProjector(derham_1d_h)
            
            if DBP0 is not None:
                I0      = IdentityOperator(V0h_1d.coeff_space)
                M0_0_1d = DBP0 @ M0_1d @ DBP0 + (I0 - DBP0)

                M0_0_1d_solver = M0_0_1d_to_bandsolver(M0_0_1d)
                M0_1d_solvers.append(M0_0_1d_solver)
            else:
                M0_1d_solver = matrix_to_bandsolver(M0_1d)
                M0_1d_solvers.append(M0_1d_solver)
        else:
            M0_1d_solver = matrix_to_bandsolver(M0_1d)
            M0_1d_solvers.append(M0_1d_solver)

        M1_1d_solver = matrix_to_bandsolver(M1_1d)
        M1_1d_solvers.append(M1_1d_solver)

    if dim == 2:
        V0_cs = derham_h.V0.coeff_space
        V1_cs = derham_h.V1.coeff_space
        V2_cs = derham_h.V2.coeff_space

        if M0 is not None:
            M0_log_kron_solver = KroneckerLinearSolver(V0_cs, V0_cs, (M0_1d_solvers[0], M0_1d_solvers[1]))
            M_log_kron_solver_arr.append(M0_log_kron_solver)
        else:
            M_log_kron_solver_arr.append(None)

        if M1 is not None:
            if derham_h.sequence[1] == 'hcurl':
                M1_0_log_kron_solver = KroneckerLinearSolver(V1_cs[0], V1_cs[0], (M1_1d_solvers[0], M0_1d_solvers[1]))
                M1_1_log_kron_solver = KroneckerLinearSolver(V1_cs[1], V1_cs[1], (M0_1d_solvers[0], M1_1d_solvers[1]))
                M1_log_kron_solver = BlockLinearOperator(V1_cs, V1_cs, [[M1_0_log_kron_solver, None],
                                                                        [None, M1_1_log_kron_solver]])
            elif derham_h.sequence[1] == 'hdiv':
                M1_0_log_kron_solver = KroneckerLinearSolver(V1_cs[0], V1_cs[0], (M0_1d_solvers[0], M1_1d_solvers[1]))
                M1_1_log_kron_solver = KroneckerLinearSolver(V1_cs[1], V1_cs[1], (M1_1d_solvers[0], M0_1d_solvers[1]))
                M1_log_kron_solver = BlockLinearOperator(V1_cs, V1_cs, [[M1_0_log_kron_solver, None],
                                                                        [None, M1_1_log_kron_solver]])
            else:
                raise ValueError(f'The second space in the sequence {derham_h.sequence} must be either "hcurl" or "hdiv".')
            M_log_kron_solver_arr.append(M1_log_kron_solver)
        else:
            M_log_kron_solver_arr.append(None)

        if M2 is not None:
            M2_log_kron_solver = KroneckerLinearSolver(V2_cs, V2_cs, (M1_1d_solvers[0], M1_1d_solvers[1]))
            M_log_kron_solver_arr.append(M2_log_kron_solver)
        else:
            M_log_kron_solver_arr.append(None)
    else:
        V0_cs = derham_h.V0.coeff_space
        V1_cs = derham_h.V1.coeff_space
        V2_cs = derham_h.V2.coeff_space
        V3_cs = derham_h.V3.coeff_space

        if M0 is not None:
            M0_log_kron_solver = KroneckerLinearSolver(V0_cs, V0_cs, (M0_1d_solvers[0], M0_1d_solvers[1], M0_1d_solvers[2]))
            M_log_kron_solver_arr.append(M0_log_kron_solver)
        else:
            M_log_kron_solver_arr.append(None)

        if M1 is not None:
            M1_0_log_kron_solver = KroneckerLinearSolver(V1_cs[0], V1_cs[0], (M1_1d_solvers[0], M0_1d_solvers[1], M0_1d_solvers[2]))
            M1_1_log_kron_solver = KroneckerLinearSolver(V1_cs[1], V1_cs[1], (M0_1d_solvers[0], M1_1d_solvers[1], M0_1d_solvers[2]))
            M1_2_log_kron_solver = KroneckerLinearSolver(V1_cs[2], V1_cs[2], (M0_1d_solvers[0], M0_1d_solvers[1], M1_1d_solvers[2]))
            M1_log_kron_solver = BlockLinearOperator(V1_cs, V1_cs, [[M1_0_log_kron_solver, None, None],
                                                                    [None, M1_1_log_kron_solver, None],
                                                                    [None, None, M1_2_log_kron_solver]])
            M_log_kron_solver_arr.append(M1_log_kron_solver)
        else:
            M_log_kron_solver_arr.append(None)
        
        if M2 is not None:
            M2_0_log_kron_solver = KroneckerLinearSolver(V2_cs[0], V2_cs[0], (M0_1d_solvers[0], M1_1d_solvers[1], M1_1d_solvers[2]))
            M2_1_log_kron_solver = KroneckerLinearSolver(V2_cs[1], V2_cs[1], (M1_1d_solvers[0], M0_1d_solvers[1], M1_1d_solvers[2]))
            M2_2_log_kron_solver = KroneckerLinearSolver(V2_cs[2], V2_cs[2], (M1_1d_solvers[0], M1_1d_solvers[1], M0_1d_solvers[2]))
            M2_log_kron_solver = BlockLinearOperator(V2_cs, V2_cs, [[M2_0_log_kron_solver, None, None],
                                                                    [None, M2_1_log_kron_solver, None],
                                                                    [None, None, M2_2_log_kron_solver]])
            M_log_kron_solver_arr.append(M2_log_kron_solver)
        else:
            M_log_kron_solver_arr.append(None)

        if M3 is not None:
            M3_log_kron_solver = KroneckerLinearSolver(V3_cs, V3_cs, (M1_1d_solvers[0], M1_1d_solvers[1], M1_1d_solvers[2]))
            M_log_kron_solver_arr.append(M3_log_kron_solver)
        else:
            M_log_kron_solver_arr.append(None)

    # --------------------------------

    M_pc_arr = []

    for M, D_inv_sqrt, D_log_sqrt, M_log_kron_solver in zip(Ms, D_inv_sqrt_arr, D_log_sqrt_arr, M_log_kron_solver_arr):
        if M is not None:
            M_pc = D_inv_sqrt @ D_log_sqrt @ M_log_kron_solver @ D_log_sqrt @ D_inv_sqrt
            M_pc_arr.append(M_pc)

    return M_pc_arr

def get_DirichletBoundaryProjector(derham_h):

    dim = derham_h.dim

    domain_h    = derham_h.domain_h
    domain      = domain_h.domain

    if dim == 2:
        derham  = Derham(domain, derham_h.sequence)
    else:
        derham  = Derham(domain)

    periodic,   = domain_h.periodic.values()

    db_projectors = []

    for V, Vh in zip(derham.spaces[:-1], derham_h.spaces[:-1]):
        Vcs = Vh.coeff_space
        bcs = get_bcs(V, periodic)
        if bcs is not None:
            DBP = DirichletBoundaryProjector(domain=Vcs, codomain=Vcs, bcs=bcs)
        else:
            DBP = None
        db_projectors.append(DBP)

    return db_projectors

def get_bcs(space, periodic):
    
    # does not work if periodic is a bool instead of a list
    if all([p == True for p in periodic]):
        return None
    
    kind = space.kind.name
    dim = space.domain.dim
    
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

class DirichletBoundaryProjector(LinearOperator):

    def __init__(self, domain, codomain, bcs=None, space=None, periodic=None):

        assert domain is codomain
        assert isinstance(domain, VectorSpace)

        self._domain = domain
        self._codomain = codomain
        if bcs is not None:
            self._bcs = bcs
        else:
            assert (space is not None) and (periodic is not None)
            assert all([isinstance(p, bool) for p in periodic])
            assert isinstance(space, BasicFunctionSpace)
            assert len(periodic) == space.shape
            self._bcs = get_bcs(space, periodic)

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
def test_LST_preconditioner():

    ncells_3d      = [16, 7, 11]
    degree_3d      = [1, 4, 2]
    periodic_3d    = [False, True, False]

    comm    = None
    backend = PSYDAC_BACKEND_GPYCCEL

    dimensions = [2, 3]

    maxiter = 20000
    tol     = 1e-13

    # Test both in 2D and 3D
    for dim in dimensions:
        print(f' ----- Start {dim}D test -----')

        ncells      = ncells_3d  [0:2] if dim == 2 else ncells_3d
        degree      = degree_3d  [0:2] if dim == 2 else degree_3d
        periodic    = periodic_3d[0:2] if dim == 2 else periodic_3d

        if dim == 2:
            logical_domain = Square('S', bounds1=(0.5, 1), bounds2=(0, 2*np.pi))
            mapping = Annulus('A')
            sequence = ['h1', 'hcurl', 'l2']
        else:
            logical_domain = Cube  ('C', bounds1=(0.5, 1), bounds2=(0, 2*np.pi), bounds3=(0, 1))
            mapping = SquareTorus('ST')

        domain  = mapping(logical_domain)

        derham = Derham(domain, sequence=sequence) if dim == 2 else Derham(domain)

        domain_h = discretize(domain, ncells=ncells, periodic=periodic, comm=comm)
        derham_h = discretize(derham, domain_h, degree=degree)

        Vs                      = derham.spaces
        Vhs                     = derham_h.spaces

        db_projectors = get_DirichletBoundaryProjector(derham_h)

        mass_matrices = []
        mass_0_matrices = []

        for i, (V, Vh) in enumerate(zip(Vs, Vhs)):
            u, v = elements_of(V, names='u, v')
            expr = inner(u, v) if isinstance(Vh.coeff_space, BlockVectorSpace) else u*v
            a    = BilinearForm((u, v), integral(domain, expr))
            ah   = discretize(a, domain_h, (Vh, Vh), backend=backend)
            M    = ah.assemble()
            mass_matrices.append(M)
            if i < dim:
                DBP = db_projectors[i]
                I   = IdentityOperator(Vhs[i].coeff_space)
                M_0 = DBP @ M @ DBP + (I - DBP)
                mass_0_matrices.append(M_0)

        if dim == 2:
            M0, M1, M2 = mass_matrices
        else:
            M0, M1, M2, M3   = mass_matrices

        if dim == 2:
            mass_matrix_preconditioners   = get_LST_preconditioner(derham_h, M0=M0, M1=M1, M2=M2,           backend=backend)
            mass_0_matrix_preconditioners = get_LST_preconditioner(derham_h, M0=M0, M1=M1,        bcs=True, backend=backend)
        else:
            mass_matrix_preconditioners   = get_LST_preconditioner(derham_h, M0=M0, M1=M1, M2=M2, M3=M3,           backend=backend)
            mass_0_matrix_preconditioners = get_LST_preconditioner(derham_h, M0=M0, M1=M1, M2=M2,        bcs=True, backend=backend)

        # Prepare testing whether obtaining only a subset of preconditioners works
        mass_matrix_preconditioners_1,  = get_LST_preconditioner(derham_h, M1=M1)
        mass_matrix_preconditioners_2 = get_LST_preconditioner(derham_h, M0=M0, M2=M2)
        if dim == 3:
            mass_matrix_preconditioners_3,  = get_LST_preconditioner(derham_h, M3=M3)

        test_pcs = [mass_matrix_preconditioners_2[0], mass_matrix_preconditioners_1, mass_matrix_preconditioners_2[1]]
        if dim == 3:
            test_pcs += [mass_matrix_preconditioners_3, ]

        # Test 1: Test whether obtaining only a subset of all possible preconditioners works
        for pc, test_pc in zip(mass_matrix_preconditioners, test_pcs):
            _test_LO_equality_using_rng(pc, test_pc)

        print(f' Accessing a subset of all possible preconditioners works.')

        rng = np.random.default_rng(42)

        # For comparison and testing: Number of iterations required, not using and using a preconditioner
        # More information via "-s" when running the test
        #                           dim 2                           dim 3
        #                  M0   M1  M2  M0_0  M1_0     M0    M1    M2   M3  M0_0  M1_0  M2_0
        true_cg_niter  = [[90, 681, 62,   77,  600], [486, 7970, 5292, 147,  356, 5892, 4510]]
        true_pcg_niter = [[ 6,   6,  2,    5,    5], [  6,    7,    6,   2,    5,    5,    5]]
        # M{i}_0 matrices preconditioned with a LST preconditioner designed for M{i} instead:
        #                               M0_0  M1_0                          M0_0  M1_0  M2_0
        true_pcg_niter2= [[               23,   24], [                       367, 2867,  220]]

        mass_matrices               += mass_0_matrices
        mass_matrix_preconditioners += mass_0_matrix_preconditioners

        for i, (M, Mpc) in enumerate(zip(mass_matrices, mass_matrix_preconditioners)):

            cg = False

            if cg:
                M_inv_cg  = inverse(M, 'cg',          maxiter=maxiter, tol=tol)
            M_inv_pcg = inverse(M, 'pcg', pc=Mpc, maxiter=maxiter, tol=tol)

            y = M.codomain.zeros()
            if isinstance(M.codomain, BlockVectorSpace):
                for block in y.blocks:
                    rng.random(size=block._data.shape, dtype="float64", out=block._data)
            else:
                rng.random(size=y._data.shape, dtype="float64", out=y._data)

            if (i > dim):
                print(f' Projecting rhs vector into space of functions satisfying hom. DBCs')
                DBP = db_projectors[i-(dim+1)]
                y   = DBP @ y

            if cg:
                t0 = time.time()
                x_cg = M_inv_cg @ y
                t1 = time.time()

                y_cg     = M @ x_cg
                diff_cg  = y - y_cg
                err_cg   = np.sqrt(M.codomain.inner(diff_cg, diff_cg))
                time_cg  = t1 - t0
                info_cg  = M_inv_cg.get_info()

            t0 = time.time()
            x_pcg = M_inv_pcg @ y
            t1 = time.time()

            y_pcg    = M @ x_pcg
            diff_pcg = y - y_pcg
            err_pcg  = np.sqrt(M.codomain.inner(diff_pcg, diff_pcg))
            time_pcg = t1 - t0
            info_pcg = M_inv_pcg.get_info()

            if dim == 2:
                mat_txt = f'M{i}' if i <= 2 else f'M{i-3}_0'
            else:
                mat_txt = f'M{i}' if i <= 3 else f'M{i-4}_0'

            print(f' - {mat_txt} test -')
            if cg:
                print(f' CG : {info_cg} in {time_cg:.3g}s       - err.: {err_cg:.3g}')
            print(f' PCG: {info_pcg} in {time_pcg:.3g}s     - err.: {err_pcg:.3g}')

            if dim == 2:
                assert info_pcg['niter'] == true_pcg_niter[0][i]
            else:
                assert info_pcg['niter'] == true_pcg_niter[1][i]

        print()

#===============================================================================
def test_DirichletBoundaryProjector():

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
    mapping_2d = Annulus        ('A' )
    mapping_3d = SquareTorus    ('ST')
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

        db_projectors = get_DirichletBoundaryProjector(derham_h)

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

# ===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================

if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
