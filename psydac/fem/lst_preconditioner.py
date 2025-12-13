#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from   functools                    import lru_cache
from   scipy.sparse                 import dia_matrix
import numpy as np

from   sympde.topology              import elements_of, Line, ScalarFunctionSpace
from   sympde.topology.datatype     import SpaceType
from   sympde.expr                  import integral, BilinearForm

from   psydac.linalg.basic          import IdentityOperator, LinearOperator
from   psydac.linalg.block          import BlockVectorSpace, BlockLinearOperator
from   psydac.linalg.direct_solvers import BandedSolver
from   psydac.linalg.kron           import KroneckerLinearSolver, KroneckerStencilMatrix
from   psydac.linalg.stencil        import StencilVectorSpace
from   psydac.fem.projectors        import DirichletProjector


@lru_cache
def construct_LST_preconditioner(M, domain_h, fem_space, hom_bc=False, kind=None):
    """
    LST (Loli, Sangalli, Tani) preconditioners [1] are mass matrix preconditioners of the form
    pc = D_inv_sqrt @ D_log_sqrt @ M_log_kron_solver @ D_log_sqrt @ D_inv_sqrt, where

    D_inv_sqrt          is the diagonal matrix of the square roots of the inverse diagonal entries of the mass matrix M,
    D_log_sqrt          is the diagonal matrix of the square roots of the diagonal entries of the mass matrix on the logical domain,
    M_log_kron_solver   is the Kronecker Solver of the mass matrix on the logical domain.

    These preconditioners work very well even on complex domains as numerical experiments have shown.
    Upon choosing hom_bc=True, a preconditioner for the modified mass matrix M_0 is returned.
    M_0 is a mass matrix of the form
    M_0 = DP @ M @ DP + (I - DP)
    where DP and I are the corresponding DirichletProjector and IdentityOperator.
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
        If True, return LST preconditioner for modified M_0 = DP @ M @ DP + (I - DP) mass matrix.
        The argument M in that case remains the same (M, not M_0). DP and I are DirichletProjector and IdentityOperator.
        Default: False.

    kind : str | None
        Optional. Must be passed if fem_space has no kind. Must match the kind of fem_space if fem_space has a kind.
        Relevant as we must know whether M is a H1, Hcurl, Hdiv or L2 mass matrix.

    Returns
    -------
    psydac.linalg.stencil.StencilMatrix | psydac.linalg.block.BlockLinearOperator
        LST preconditioner for M (hom_bc=False) or M_0 (hom_b=True).

    References
    ----------
    [1] Gabriele Loli, Giancarlo Sangalli, Mattia Tani. “Easy and efficient preconditioning of the isogeometric mass 
        matrix”. In: Computers & Mathematics with Applications 116 (2022), pp. 245–264
    """

    # to avoid circular import
    from psydac.api.discretization                   import discretize

    dim = fem_space.ldim
    # In 1D one can solve the linear system directly (instead of using this preconditioner)
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

            Closely resembles BandedSolver.from_stencil_mat_1d,
            the difference being that M0_0_1d neither has a 
            remove_spurious_entries() nor a toarray() function.
            
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
                if DP.bcs != ():
                    I      = IdentityOperator(Vh_1d.coeff_space)
                    M_0_1d = DP @ M_1d @ DP + (I - DP)

                    M_0_1d_solver = M_0_1d_to_bandsolver(M_0_1d)
                    M_1d_solvers[j].append(M_0_1d_solver)
                else:
                    M_1d_solver = BandedSolver.from_stencil_mat_1d(M_1d)
                    M_1d_solvers[j].append(M_1d_solver)
            else:
                M_1d_solver = BandedSolver.from_stencil_mat_1d(M_1d)
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
