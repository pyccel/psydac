import  time
import  numpy as np
import  pytest

from    scipy.sparse                               import dia_matrix

from    sympde.topology             import elements_of, Mapping, Derham, Square, Cube, Line, ScalarFunctionSpace, VectorFunctionSpace
from    sympde.topology.datatype    import SpaceType
from    sympde.calculus             import inner
from    sympde.expr                 import integral, BilinearForm

from    psydac.api.discretization   import discretize
from    psydac.api.settings         import PSYDAC_BACKEND_GPYCCEL
from    psydac.ddm.cart             import DomainDecomposition, CartDecomposition
from    psydac.fem.projectors       import DirichletProjector
from    psydac.fem.tests.test_dirichlet_projectors import _test_LO_equality_using_rng, Annulus, SquareTorus
from    psydac.linalg.basic         import IdentityOperator, LinearOperator
from    psydac.linalg.block         import BlockVectorSpace, BlockLinearOperator
from    psydac.linalg.solvers       import inverse
from    psydac.linalg.stencil       import StencilVectorSpace, StencilMatrix, StencilVector

from    psydac.linalg.direct_solvers                import BandedSolver
from    psydac.linalg.kron                          import KroneckerLinearSolver, KroneckerStencilMatrix
from    psydac.linalg.tests.test_kron_direct_solver import matrix_to_bandsolver


class SinMapping1D(Mapping):

    _expressions = {'x': 'sin((pi/2)*x1)'}
    
    _ldim        = 1
    _pdim        = 1

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

def construct_LST_preconditioner(M, domain_h, fem_space, hom_bc=False, kind=None):
    """
    LST (Loli, Sangalli, Tani) preconditioners are mass matrix preconditioners of the form
    pc = D_inv_sqrt @ D_log_sqrt @ M_log_kron_solver @ D_log_sqrt @ D_inv_sqrt, where

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

    Parameters
    ----------
    M : psydac.linalg.stencil.StencilMatrix | psydac.linalg.block.BlockLinearOperator
        Mass matrix corresponding to fem_space

    domain_h : psydac.cad.geometry.Geometry
        discretized physical domain used to discretize fem_space

    fem_space : psydac.fem.basic.FemSpace
        discretized Scalar- or VectorFunctionSpace. M is the corresponding mass matrix

    hom_bc : bool
        If True, return LST preconditioner for modified M_0 = DBP @ M @ DBP + (I - DBP) mass matrix.
        The argument M in that case remains the same (M, not M_0). DBP and I are DirichletBoundaryProjector and IdentityOperator.
        Default False

    kind : str | None
        Optional. Must be passed if fem_space has no kind. Must match the kind of fem_space if fem_space has a kind.
        Relevant as we must know whether M is a H1, Hcurl, Hdiv or L2 mass matrix.

    Returns
    -------
    psydac.linalg.stencil.StencilMatrix | psydac.linalg.block.BlockLinearOperator
        LST preconditioner for M (hom_bc=False) or M_0 (hom_b=True).
    
    """

    dim = fem_space.ldim
    #! dim=1 should also be allowed
    assert dim in (2, 3)

    if hom_bc == True:
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

        def M_0_1d_to_bandsolver(A):
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

    domain      = domain_h.domain

    ncells,     = domain_h.ncells.values()
    degree      = fem_space.degree
    periodic,   = domain_h.periodic.values()

    V_cs        = fem_space.coeff_space
    
    logical_domain      = domain.logical_domain

    # ----- Compute D_inv_sqrt

    D_inv_sqrt = M.diagonal(inverse=True, sqrt=True)

    # ----- Compute M_log_kron_solver

    logical_domain_1d_x = Line('L', bounds=logical_domain.bounds1)
    logical_domain_1d_y = Line('L', bounds=logical_domain.bounds2)
    if dim == 3:
        logical_domain_1d_z = Line('L', bounds=logical_domain.bounds3)

    logical_domain_1d_list = [logical_domain_1d_x, logical_domain_1d_y]
    if dim == 3:
        logical_domain_1d_list += [logical_domain_1d_z]

    # We gather the 1D mass matrices.
    # Those will be used to obtain D_log_sqrt using the new
    # diagonal function for KroneckerStencilMatrices.
    M_1d_solvers = [[],[]]
    Ms_1d        = [[],[]]
    if dim == 3:
        M_1d_solvers += [[]]
        Ms_1d        += [[]]

    # Mark 1D 'h1' spaces built using B-splines. 
    # 1D spaces for which (i, j) \notin keys are 'l2' spaces built using M-splines.
    fem_space_kind = fem_space.symbolic_space.kind.name

    if kind is not None:
        if isinstance(kind, str):
            kind = kind.lower()
            assert(kind in ['h1', 'hcurl', 'hdiv', 'l2'])
        elif isinstance(kind, SpaceType):
            kind = kind.name
        else:
            raise TypeError(f'Expecting kind {kind} to be a str or of SpaceType')
        
        # If fem_space has a kind, it must be compatible with kind
        if fem_space_kind != 'undefined':
            assert fem_space_kind == kind, f'fem_space and space_kind are not compatible.'
    else:
        kind = fem_space_kind

    if kind == 'h1':
        keys = ((0, 0), (1, 0), (2, 0))
    elif kind == 'hcurl':
        keys = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
    elif kind == 'hdiv':
        keys = ((0, 0), (1, 1), (2, 2))
    elif kind == 'l2':
        keys = ()
    else:
        raise ValueError(f'kind {kind} must be either h1, hcurl, hdiv or l2.')

    for i, (ncells_1d, periodic_1d, logical_domain_1d) in enumerate(zip(ncells, periodic, logical_domain_1d_list)):

        logical_domain_1d_h = discretize(logical_domain_1d, ncells=[ncells_1d, ], periodic=[periodic_1d, ])

        degrees_1d = [degree_dir[i] for degree_dir in degree] if isinstance(fem_space.coeff_space, BlockVectorSpace) else [degree[i], ]

        for j, d in enumerate(degrees_1d):

            kind_1d = 'h1' if (i, j) in keys else 'l2'
            basis   = 'B'  if (i, j) in keys else 'M'

            if basis == 'M':
                d += 1

            V_1d    = ScalarFunctionSpace('V', logical_domain_1d,        kind =kind_1d)
            Vh_1d   = discretize(V_1d, logical_domain_1d_h, degree=[d,], basis=basis)

            u, v    = elements_of(V_1d, names='u, v')
            a_1d    = BilinearForm((u, v), integral(logical_domain_1d, u*v))
            ah_1d   = discretize(a_1d, logical_domain_1d_h, (Vh_1d, Vh_1d))
            M_1d    = ah_1d.assemble()
            Ms_1d[j].append(M_1d)

            if (hom_bc == True) and ((i, j) in keys):
                DP = DirichletProjector(Vh_1d, space_kind='h1')
                if DP.bcs != ():#DP is not None:
                    I      = IdentityOperator(Vh_1d.coeff_space)
                    M_0_1d = DP @ M_1d @ DP + (I - DP)

                    M_0_1d_solver = M_0_1d_to_bandsolver(M_0_1d)
                    M_1d_solvers[j].append(M_0_1d_solver)
                else:
                    M_1d_solver = matrix_to_bandsolver(M_1d)
                    M_1d_solvers[j].append(M_1d_solver)
            else:
                M_1d_solver = matrix_to_bandsolver(M_1d)
                M_1d_solvers[j].append(M_1d_solver)

    if isinstance(V_cs, StencilVectorSpace):
        M_log_kron_solver       = KroneckerLinearSolver(V_cs, V_cs, M_1d_solvers[0])

    else:
        M_0_log_kron_solver     = KroneckerLinearSolver(V_cs[0], V_cs[0], M_1d_solvers[0])
        M_1_log_kron_solver     = KroneckerLinearSolver(V_cs[1], V_cs[1], M_1d_solvers[1])
        if dim == 3:
            M_2_log_kron_solver = KroneckerLinearSolver(V_cs[2], V_cs[2], M_1d_solvers[2])

        if dim == 2:
            blocks = [[M_0_log_kron_solver, None],
                      [None, M_1_log_kron_solver]]
        else:
            blocks = [[M_0_log_kron_solver, None, None],
                      [None, M_1_log_kron_solver, None],
                      [None, None, M_2_log_kron_solver]]
        
        M_log_kron_solver       = BlockLinearOperator  (V_cs, V_cs, blocks)

    # ----- Compute D_log_sqrt

    if isinstance(V_cs, StencilVectorSpace):
        M_log            = KroneckerStencilMatrix(V_cs, V_cs, *Ms_1d[0])

        D_log_sqrt       = M_log.diagonal  (inverse=False, sqrt=True)

    else:
        M_0_log          = KroneckerStencilMatrix(V_cs[0], V_cs[0], *Ms_1d[0])
        M_1_log          = KroneckerStencilMatrix(V_cs[1], V_cs[1], *Ms_1d[1])
        if dim == 3:
            M_2_log      = KroneckerStencilMatrix(V_cs[2], V_cs[2], *Ms_1d[2])

        if dim == 2:
            blocks  = [[M_0_log, None],
                      [None, M_1_log]]
        else:
            blocks = [[M_0_log, None, None],
                      [None, M_1_log, None],
                      [None, None, M_2_log]]
            
        M_log = BlockLinearOperator(V_cs, V_cs, blocks=blocks)
        D_log_sqrt = M_log.diagonal(inverse=False, sqrt=True)

    # --------------------------------

    M_pc = D_inv_sqrt @ D_log_sqrt @ M_log_kron_solver @ D_log_sqrt @ D_inv_sqrt

    return M_pc

#===============================================================================
@pytest.mark.parametrize('n', [5, 10, 13] )
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

    print()
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

        db_projectors = derham_h.dirichlet_projectors(kind='linop')

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
            mass_matrix_preconditioners   = derham_h.LST_preconditioner(M0=M0, M1=M1, M2=M2             )
            mass_0_matrix_preconditioners = derham_h.LST_preconditioner(M0=M0, M1=M1,        hom_bc=True)
        else:
            mass_matrix_preconditioners   = derham_h.LST_preconditioner(M0=M0, M1=M1, M2=M2, M3=M3,            )
            mass_0_matrix_preconditioners = derham_h.LST_preconditioner(M0=M0, M1=M1, M2=M2,        hom_bc=True)

        # Prepare testing whether obtaining only a subset of preconditioners works
        M1_pc,       = derham_h.LST_preconditioner(M1=M1       )
        M0_pc, M2_pc = derham_h.LST_preconditioner(M0=M0, M2=M2)
        if dim == 3:
            M3_pc,   = derham_h.LST_preconditioner(M3=M3       )

        test_pcs = [M0_pc, M1_pc, M2_pc]
        if dim == 3:
            test_pcs += [M3_pc]

        # Test whether obtaining only a subset of all possible preconditioners works
        for pc, test_pc in zip(mass_matrix_preconditioners, test_pcs):
            _test_LO_equality_using_rng(pc, test_pc)#, tol=1e-13)

        print(f' Accessing a subset of all possible preconditioners works.')

        rng = np.random.default_rng(42)

        # For comparison and testing: Number of iterations required, not using and using a preconditioner
        # More information via " -s" when running the test
        #                           dim 2                           dim 3
        #                  M0   M1  M2  M0_0  M1_0     M0    M1    M2   M3  M0_0  M1_0  M2_0
        true_cg_niter  = [[90, 681, 62,   77,  600], [486, 7970, 5292, 147,  356, 5892, 4510]]
        true_pcg_niter = [[ 6,   6,  2,    5,    5], [  6,    7,    6,   2,    5,    5,    5]]
        # M{i}_0 matrices preconditioned with a LST preconditioner designed for M{i} instead:
        #                               M0_0  M1_0                          M0_0  M1_0  M2_0
        true_pcg_niter2= [[               23,   24], [                       367, 2867,  220]]

        mass_matrices               += mass_0_matrices
        mass_matrix_preconditioners += mass_0_matrix_preconditioners
        extended_fem_spaces         = Vhs + Vhs[:-1]

        for i, (M, Mpc, Vh) in enumerate(zip(mass_matrices, mass_matrix_preconditioners, extended_fem_spaces)):

            cg = False # Set to True to compare iterations and time with not-preconditioned Conjugate Gradient solver

            # hom_bc = False for M0 M1 M2 (M3), then hom_bc = True for M0_0 M1_0 (M2_0)
            hom_bc = True if i > dim else False

            # In order to obtain an LST for M{i}_0, we still have to pass M{i} to `construct_LST_preconditioner`.`
            # M2 = M{i} if M = M{i}_0 and hence can be used to obtain the pc for M{i}_0
            M2 = mass_matrices[i-dim-1] if i > dim else M
            Mpc2 = construct_LST_preconditioner(M2, domain_h, Vh, hom_bc=hom_bc)
            _test_LO_equality_using_rng(Mpc, Mpc2, tol=1e-12)
            print(' The LST pc obtained using derham_h.LST_preconditioner is the same as the one obtained from construct_LST_preconditioner.')

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


# ===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================

if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
