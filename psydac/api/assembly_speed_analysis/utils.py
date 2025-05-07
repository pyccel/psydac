import  h5py
import  numpy as np

from    scipy.sparse                    import dia_matrix

from    sympde.calculus                 import dot
from    sympde.expr                     import EssentialBC, BilinearForm, integral
from    sympde.topology                 import element_of, elements_of, Mapping, Derham, Line
from    sympde.topology.datatype        import H1Space, HcurlSpace, HdivSpace

from    psydac.api.discretization       import discretize
from    psydac.api.essential_bc         import apply_essential_bc
from    psydac.fem.basic                import FemField
from    psydac.linalg.basic             import LinearOperator, Vector, IdentityOperator
from    psydac.linalg.block             import BlockVectorSpace, BlockVector, BlockLinearOperator
from    psydac.linalg.direct_solvers    import BandedSolver
from    psydac.linalg.kron              import KroneckerLinearSolver
from    psydac.linalg.stencil           import StencilVectorSpace, StencilMatrix, StencilDiagonalMatrix
from    psydac.linalg.utilities         import array_to_psydac


# ---------- Domains ----------
mapping_name_list = ('Cube', 'HalfHollowTorus3D', 'SquareTorus', 'Annulus')

def get_mapping(domain_name):
    assert domain_name in mapping_name_list

    if domain_name == 'HalfHollowTorus3D':
        mapping = HalfHollowTorus3D('M', R=2, r=1)
    elif domain_name == 'SquareTorus':
        mapping = SquareTorus('M')
    elif domain_name =='Annulus':
        mapping = Annulus('A')

    else:
        raise ValueError(f'domain_name {domain_name} not understood.')
    
    return mapping

class HalfHollowTorus3D(Mapping):

    _expressions = {'x': '(R + r * x1 * cos(2*pi*x3)) * cos(pi*x2)',
                    'y': '(R + r * x1 * cos(2*pi*x3)) * sin(pi*x2)',
                    'z': 'r * x1 * sin(2*pi*x3)'}

    _ldim        = 3
    _pdim        = 3

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
# -----------------------------


# ---------- Solver Utilities ----------
def get_diagonal_parallel_3d(A, Vhserial, periodic, inv=False, sqrt=False):
    assert isinstance(inv, bool)
    assert isinstance(sqrt, bool)
    assert isinstance(A, LinearOperator)
    assert A.domain is A.codomain

    V = A.domain
    is_block = True if isinstance(V, BlockVectorSpace) else False
    if not is_block:
        assert isinstance(V, StencilVectorSpace)

    if is_block:
        mat = BlockLinearOperator(V, V)
        v = V.zeros()
        out = V.zeros()
        
        localslice0 = tuple([slice(s, e+1) for s, e in zip(V[0].starts, V[0].ends)])
        localslice1 = tuple([slice(s, e+1) for s, e in zip(V[1].starts, V[1].ends)])
        localslice2 = tuple([slice(s, e+1) for s, e in zip(V[2].starts, V[2].ends)])
        slices = [localslice0, localslice1, localslice2]

        for block_index in range(V.n_blocks):
            v *= 0.0

            V_block = V[block_index]

            npts1, npts2, npts3 = V_block.npts
            v_serial = Vhserial[block_index].zeros()
            e = V_block.zeros()
            
            diag = []
            for i in range(npts1):
                j_arr = []
                for j in range(npts2):
                    k_arr = []
                    for k in range(npts3):

                        e_serial = get_unit_vector(v_serial, [i, j, k], V_block.pads, V_block.npts, periodic)
                        e[slices[block_index]] = e_serial[slices[block_index]]
                        v[block_index] = e

                        A.dot(v, out=out)
                        d = v.dot(out)

                        if inv:
                            assert d != 0.0
                            d = 1/d
                        if sqrt:
                            assert d >= 0.0
                            d = np.sqrt(d)
                        
                        k_arr.append(d)
                    j_arr.append(k_arr)
                diag.append(j_arr)
            diag_data = np.array(diag)

            diag_data_local = diag_data[slices[block_index]]

            block_mat = StencilDiagonalMatrix(V_block, V_block, diag_data_local)

            mat[block_index, block_index] = block_mat
    else:
        npts1, npts2, npts3 = V.npts
        v_serial = Vhserial.zeros()
        e = V.zeros()
        localslice = tuple([slice(s, e+1) for s, e in zip(V.starts, V.ends)])

        diag = []
        for i in range(npts1):
            j_arr = []
            for j in range(npts2):
                k_arr = []
                for k in range(npts3):

                    e_serial = get_unit_vector(v_serial, [i, j, k], V.pads, V.npts, periodic)
                    e[localslice] = e_serial[localslice]
                    d = e.dot(A@e)

                    if inv:
                        assert d != 0.0
                        d = 1/d
                    if sqrt:
                        assert d >= 0.0
                        d = np.sqrt(d)
                    
                    k_arr.append(d)
                j_arr.append(k_arr)
            diag.append(j_arr)
        diag_data = np.array(diag)

        local_slice = tuple([slice(s, e+1) for s, e in zip(V.starts, V.ends)])
        diag_data_local = diag_data[local_slice]

        mat = StencilDiagonalMatrix(V, V, diag_data_local)

    return mat

def get_diagonal(A, inv=True, sqrt=False, tol=1e-13):
    """
    Takes a square LinearOperator A and returns a similar diagonal LinearOperator D with A's diagonal entries.

    If inv=True, returns a diagonal LinearOperator with A's inverse diagonal entries.
    In that case, a diagonal entry < tol will result in a ValueError.
    If sqrt = True, replaces the diagonal entries by their square roots (D->D^{1/2}; D^{-1}->D^{-1/2}).
    
    """
    assert isinstance(inv, bool)
    assert isinstance(sqrt, bool)
    assert isinstance(A, LinearOperator)
    assert A.domain is A.codomain

    V   = A.domain
    v   = V.zeros()
    is_block = True if isinstance(V, BlockVectorSpace) else False
    if not is_block:
        assert isinstance(V, StencilVectorSpace)

    if is_block:
        D = BlockLinearOperator(V, V)
    else:
        D = StencilMatrix(V, V)

    if is_block:
        for block_index in range(V.n_blocks):
            diag_values = []
            V_block = V[block_index]
            npts1, npts2, npts3 = V_block.npts
            pads1, pads2, pads3 = V_block.pads

            for n1 in range(npts1):
                diag_values_block2 = []
                for n2 in range(npts2):
                    diag_values_block = []
                    for n3 in range(npts3):
                        v *= 0.0
                        v[block_index]._data[pads1+n1, pads2+n2] = 1.
                        w = A @ v
                        d = w[block_index]._data[pads1+n1, pads2+n2, pads3+n3]
                        if sqrt:
                            assert d >= 0, f'Diagonal entry {d} must be non-negative in order to apply the sqrt.'
                            d = np.sqrt(d)
                        if inv:
                            assert d != 0, f'Diagonal entry {d} must be non-zero to be invertible.'
                            d = 1 / d
                        diag_values_block.append(d)
                    diag_values_block2.append(diag_values_block)
                diag_values.append(diag_values_block2)
            diag_values = np.array(diag_values)
            D_block = StencilMatrix(V_block, V_block)
            D_block._data[D_block._get_diagonal_indices()] = diag_values
            D[block_index, block_index] = D_block
    else:
        diag_values = []
        npts1, npts2, npts3 = V.npts
        pads1, pads2, pads3 = V.pads

        for n1 in range(npts1):
            diag_values_block2 = []
            for n2 in range(npts2):
                diag_values_block = []
                for n3 in range(npts3):
                    v *= 0.0
                    v._data[pads1+n1, pads2+n2, pads3+n3] = 1.
                    w = A @ v
                    d = w._data[pads1+n1, pads2+n2, pads3+n3]
                    if sqrt:
                        assert d >= 0, f'Diagonal entry {d} must be positive in order to apply the sqrt.'
                        d = np.sqrt(d)
                    if inv:
                        if abs(d) < tol:
                            d = 1 / tol # raise ValueError(f'Diagonal value d with abs(d) < tol = {tol} encountered.')
                        else:
                            d = 1 / d
                    diag_values_block.append(d)
                diag_values_block2.append(diag_values_block)
            diag_values.append(diag_values_block2)
        diag_values = np.array(diag_values)
        D._data[D._get_diagonal_indices()] = diag_values

    return D

def get_unit_vector(v, ns, pads, npts, periodic):
    dim = len(pads)
    assert dim in [1, 2, 3]
    assert len(ns) == dim
    assert len(npts) == dim
    assert len(periodic) == dim

    if dim == 1:
        n1,    = ns
        npts1, = npts
        pads1, = pads

        periodic1, = periodic
    elif dim == 2:
        n1, n2       = ns
        npts1, npts2 = npts
        pads1, pads2 = pads

        periodic1, periodic2 = periodic
    else:
        n1, n2, n3          = ns
        npts1, npts2, npts3 = npts
        pads1, pads2, pads3 = pads

        periodic1, periodic2, periodic3 = periodic

    v *= 0.0
    if dim == 1:
        v._data[pads1+n1] = 1.
    elif dim == 2:
        v._data[pads1+n1, pads2+n2] = 1.
    else:
        v._data[pads1+n1, pads2+n2, pads3+n3] = 1.

    if dim == 1:
        if periodic1:
            if n1 < pads1:
                v._data[-pads1+n1] = 1.
            if n1 >= npts1-pads1:
                v._data[n1-npts1+pads1] = 1.

    elif dim == 2:
        if periodic1:
            if n1 < pads1:
                v._data[-pads1+n1, pads2+n2] = 1.
            if n1 >= npts1-pads1:
                v._data[n1-npts1+pads1, pads2+n2] = 1.
        
        if periodic2:
            if n2 < pads2:
                v._data[pads1+n1, -pads2+n2] = 1.
            if n2 >= npts2-pads2:
                v._data[pads1+n1, n2-npts2+pads2] = 1.

    else:
        if periodic1:
            if n1 < pads1:
                v._data[-pads1+n1, pads2+n2, pads3+n3] = 1.
            if n1 >= npts1-pads1:
                v._data[n1-npts1+pads1, pads2+n2, pads3+n3] = 1.

        if periodic2:
            if n2 < pads2:
                v._data[pads1+n1, -pads2+n2, pads3+n3] = 1.
            if n2 >= npts2-pads2:
                v._data[pads1+n1, n2-npts2+pads2, pads3+n3] = 1.

        if periodic3:
            if n3 < pads3:
                v._data[pads1+n1, pads2+n2, -pads3+n3] = 1.
            if n3 >= npts3-pads3:
                v._data[pads1+n1, pads2+n2, n3-npts3+pads3] = 1.

    return v

def toarray(A):
    """Obtain a numpy array representation of a LinearOperator (which has not implemented toarray())."""
    assert isinstance(A, LinearOperator)

    At = A.T
    W  = A.codomain
    w  = W.zeros()

    W_is_block = True if isinstance(W, BlockVectorSpace) else False
    if not W_is_block:
        assert isinstance(W, StencilVectorSpace)

    A_arr = np.zeros(A.shape, dtype="float64")

    if W_is_block:
        codomain_blocks = [W[i] for i in range(W.n_blocks)]
    else:
        codomain_blocks = [W, ]

    start_index = 0
    for k, Wk in enumerate(codomain_blocks):
        w *= 0.
        v = Wk.zeros()
        pads = Wk.pads
        npts = Wk.npts
        periodic = Wk.periods

        dim = len(npts)
        assert dim in [1, 2, 3]

        if dim == 1:
            npts1, = npts
            for n1 in range(npts1):
                e = get_unit_vector(v, [n1, ], pads, npts, periodic)
                row = At @ e
                A_arr[start_index + n1, :] = row.toarray()

        if dim == 2:
            npts1, npts2 = npts
            for n1 in range(npts1):
                for n2 in range(npts2):
                    e = get_unit_vector(v, [n1, n2], pads, npts, periodic)
                    if W_is_block:
                        w[k] = e
                        e = w
                    row = At @ e
                    A_arr[start_index + n1*npts2 + n2, :] = row.toarray()

        if dim == 3:
            npts1, npts2, npts3 = npts
            for n1 in range(npts1):
                for n2 in range(npts2):
                    for n3 in range(npts3):
                        e = get_unit_vector(v, [n1, n2, n3], pads, npts, periodic)
                        if W_is_block:
                            w[k] = e
                            e = w
                        row = At @ e
                        A_arr[start_index + n1*npts2*npts3 + n2*npts3 + n3, :] = row.toarray()
                    
        start_index += Wk.dimension

    return A_arr

# check first if matrix has toarray() method implemented before using custom slow toarray
def to_bnd(A):

    #dmat = dia_matrix(toarray_1d(A), dtype=A.dtype)
    dmat = dia_matrix(toarray(A), dtype=A.dtype)
    la   = abs(dmat.offsets.min())
    ua   = dmat.offsets.max()
    cmat = dmat.tocsr()

    A_bnd = np.zeros((1+ua+2*la, cmat.shape[1]), A.dtype)

    for i,j in zip(*cmat.nonzero()):
        A_bnd[la+ua+i-j, j] = cmat[i,j]

    return A_bnd, la, ua

def matrix_to_bandsolver(A):
    A_bnd, la, ua = to_bnd(A)
    return BandedSolver(ua, la, A_bnd)
# --------------------------------------


# ---------- LST preconditioners ----------
def get_LST_pcs(M0, M1, M2, logical_domain, Vs, Vhs, Vcs, ncells, degree, periodic, comm, backend):
    """
    LST (Loli, Sangalli, Tani) preconditioners are mass matrix preconditioners of the form

    pc = D_inv_sqrt @ D_log_sqrt @ M_log_kron_solver @ D_log_sqrt @ D_inv_sqrt,

    where

    D_inv_sqrt          is the diagonal matrix of the square roots of the inverse diagonal entries of the mass matrix M,
    D_log_sqrt          is the diagonal matrix of the square roots of the diagonal entries of the mass matrix on the logical domain,
    M_log_kron_solver   is the Kronecker Solver of the mass matrix on the logical domain.

    These preconditioners work very well even on complex domains as numerical experiments have shown.
    
    """

    logical_domain_serial_h = discretize(logical_domain, ncells=ncells, periodic=periodic)
    logical_derham = Derham(logical_domain)
    logical_derham_serial_h = discretize(logical_derham, logical_domain_serial_h, degree=degree)
    V0h_serial = logical_derham_serial_h.V0.coeff_space
    V1h_serial = logical_derham_serial_h.V1.coeff_space
    V2h_serial = logical_derham_serial_h.V2.coeff_space

    # ---------- D_inv_sqrt ----------
    D0_inv_sqrt      = get_diagonal_parallel_3d(M0, Vhserial=V0h_serial, periodic=periodic, inv=True, sqrt=True) if M0 is not None else None
    D1_inv_sqrt      = get_diagonal_parallel_3d(M1, Vhserial=V1h_serial, periodic=periodic, inv=True, sqrt=True) if M1 is not None else None
    D2_inv_sqrt      = get_diagonal_parallel_3d(M2, Vhserial=V2h_serial, periodic=periodic, inv=True, sqrt=True) if M2 is not None else None
    # --------------------------------

    # ---------- D_log_sqrt ----------
    logical_domain_h = discretize(logical_domain, ncells=ncells, periodic=periodic, comm=comm)

    V0_log, V1_log, V2_log, V3_log = Vs

    V0h_log, V1h_log, V2h_log, V3h_log = Vhs

    V0_cs, V1_cs, V2_cs, V3_cs = Vcs

    u0, v0  = elements_of(V0_log, names='u0, v0')
    u1, v1  = elements_of(V1_log, names='u1, v1')
    u2, v2  = elements_of(V2_log, names='u2, v2')

    a0      = BilinearForm((u0, v0), integral(logical_domain, u0*v0))
    a1      = BilinearForm((u1, v1), integral(logical_domain, dot(u1, v1)))
    a2      = BilinearForm((u2, v2), integral(logical_domain, dot(u2, v2)))

    if M0 is not None:
        a0h     = discretize(a0, logical_domain_h, (V0h_log, V0h_log), backend=backend)
        M0_log  = a0h.assemble()
        D0_log_sqrt = get_diagonal_parallel_3d(M0_log, Vhserial=V0h_serial, periodic=periodic, inv=False, sqrt=True)

    if M1 is not None:
        a1h     = discretize(a1, logical_domain_h, (V1h_log, V1h_log), backend=backend)
        M1_log  = a1h.assemble()
        D1_log_sqrt = get_diagonal_parallel_3d(M1_log, Vhserial=V1h_serial, periodic=periodic, inv=False, sqrt=True)

    if M2 is not None:
        a2h     = discretize(a2, logical_domain_h, (V2h_log, V2h_log), backend=backend)
        M2_log  = a2h.assemble()
        D2_log_sqrt = get_diagonal_parallel_3d(M2_log, Vhserial=V2h_serial, periodic=periodic, inv=False, sqrt=True)
    # --------------------------------
    
    # ---------- M_log_kron_solver ----------
    ncells_x, ncells_y, ncells_z        = ncells
    degree_x, degree_y, degree_z        = degree
    periodic_x, periodic_y, periodic_z  = periodic

    bounds1 = logical_domain.bounds1
    bounds2 = logical_domain.bounds2
    bounds3 = logical_domain.bounds3
    logical_domain_1d_x = Line('L', bounds=bounds1)
    logical_domain_1d_y = Line('L', bounds=bounds2)
    logical_domain_1d_z = Line('L', bounds=bounds3)
    derham_1d_x = Derham(logical_domain_1d_x)
    derham_1d_y = Derham(logical_domain_1d_y)
    derham_1d_z = Derham(logical_domain_1d_z)
    #logical_domain_1d = Line('L', bounds=(0,1))
    #derham_1d   = Derham(logical_domain_1d)

    domain_xh   = discretize(logical_domain_1d_x, ncells=[ncells_x, ], periodic=[periodic_x, ])
    domain_yh   = discretize(logical_domain_1d_y, ncells=[ncells_y, ], periodic=[periodic_y, ])
    domain_zh   = discretize(logical_domain_1d_z, ncells=[ncells_z, ], periodic=[periodic_z, ])

    derham_xh   = discretize(derham_1d_x, domain_xh, degree=[degree_x, ])
    derham_yh   = discretize(derham_1d_y, domain_yh, degree=[degree_y, ])
    derham_zh   = discretize(derham_1d_z, domain_zh, degree=[degree_z, ])

    V0_1d_x     = derham_1d_x.V0
    V0_1d_y     = derham_1d_y.V0
    V0_1d_z     = derham_1d_z.V0
    V1_1d_x     = derham_1d_x.V1
    V1_1d_y     = derham_1d_y.V1
    V1_1d_z     = derham_1d_z.V1

    V0_xh       = derham_xh.V0
    V1_xh       = derham_xh.V1

    V0_yh       = derham_yh.V0
    V1_yh       = derham_yh.V1

    V0_zh       = derham_zh.V0
    V1_zh       = derham_zh.V1

    V0_x_cs = V0_xh.coeff_space

    V0_y_cs = V0_yh.coeff_space

    V0_z_cs = V0_zh.coeff_space

    Px = H1BoundaryProjector_1D(V0_x_cs, V0_1d_x, periodic_x)
    Py = H1BoundaryProjector_1D(V0_y_cs, V0_1d_y, periodic_y)
    Pz = H1BoundaryProjector_1D(V0_z_cs, V0_1d_z, periodic_z)

    IPx = IdentityOperator(V0_x_cs) - Px
    IPy = IdentityOperator(V0_y_cs) - Py
    IPz = IdentityOperator(V0_z_cs) - Pz

    u0_1d_x, v0_1d_x = elements_of(V0_1d_x, names='u0_1d_x, v0_1d_x')
    u1_1d_x, v1_1d_x = elements_of(V1_1d_x, names='u1_1d_x, v1_1d_x')
    u0_1d_y, v0_1d_y = elements_of(V0_1d_y, names='u0_1d_y, v0_1d_y')
    u1_1d_y, v1_1d_y = elements_of(V1_1d_y, names='u1_1d_y, v1_1d_y')
    u0_1d_z, v0_1d_z = elements_of(V0_1d_z, names='u0_1d_z, v0_1d_z')
    u1_1d_z, v1_1d_z = elements_of(V1_1d_z, names='u1_1d_z, v1_1d_z')

    m0_1d_x       = BilinearForm((u0_1d_x, v0_1d_x), integral(logical_domain_1d_x, u0_1d_x*v0_1d_x))
    m1_1d_x       = BilinearForm((u1_1d_x, v1_1d_x), integral(logical_domain_1d_x, u1_1d_x*v1_1d_x))
    m0_1d_y       = BilinearForm((u0_1d_y, v0_1d_y), integral(logical_domain_1d_y, u0_1d_y*v0_1d_y))
    m1_1d_y       = BilinearForm((u1_1d_y, v1_1d_y), integral(logical_domain_1d_y, u1_1d_y*v1_1d_y))
    m0_1d_z       = BilinearForm((u0_1d_z, v0_1d_z), integral(logical_domain_1d_z, u0_1d_z*v0_1d_z))
    m1_1d_z       = BilinearForm((u1_1d_z, v1_1d_z), integral(logical_domain_1d_z, u1_1d_z*v1_1d_z))

    m0_xh       = discretize(m0_1d_x, domain_xh, (V0_xh, V0_xh), backend=backend)
    m0_yh       = discretize(m0_1d_y, domain_yh, (V0_yh, V0_yh), backend=backend)
    m0_zh       = discretize(m0_1d_z, domain_zh, (V0_zh, V0_zh), backend=backend)

    m1_xh       = discretize(m1_1d_x, domain_xh, (V1_xh, V1_xh), backend=backend)
    m1_yh       = discretize(m1_1d_y, domain_yh, (V1_yh, V1_yh), backend=backend)
    m1_zh       = discretize(m1_1d_z, domain_zh, (V1_zh, V1_zh), backend=backend)

    M0_x        = m0_xh.assemble()
    M0_y        = m0_yh.assemble()
    M0_z        = m0_zh.assemble()

    M0_x        = Px @ M0_x @ Px + IPx
    M0_y        = Py @ M0_y @ Py + IPy
    M0_z        = Pz @ M0_z @ Pz + IPz

    M1_x        = m1_xh.assemble()
    M1_y        = m1_yh.assemble()
    M1_z        = m1_zh.assemble()

    M0_solvers  = [matrix_to_bandsolver(M) for M in [M0_x, M0_y, M0_z]]
    M1_solvers  = [matrix_to_bandsolver(M) for M in [M1_x, M1_y, M1_z]]

    M0_log_kron_solver = KroneckerLinearSolver(V0_cs, V0_cs, (M0_solvers[0], M0_solvers[1], M0_solvers[2]))

    M1_0_log_kron_solver = KroneckerLinearSolver(V1_cs[0], V1_cs[0], (M1_solvers[0], M0_solvers[1], M0_solvers[2]))
    M1_1_log_kron_solver = KroneckerLinearSolver(V1_cs[1], V1_cs[1], (M0_solvers[0], M1_solvers[1], M0_solvers[2]))
    M1_2_log_kron_solver = KroneckerLinearSolver(V1_cs[2], V1_cs[2], (M0_solvers[0], M0_solvers[1], M1_solvers[2]))
    M1_log_kron_solver = BlockLinearOperator(V1_cs, V1_cs, [[M1_0_log_kron_solver, None, None],
                                                            [None, M1_1_log_kron_solver, None],
                                                            [None, None, M1_2_log_kron_solver]])
    
    M2_0_log_kron_solver = KroneckerLinearSolver(V2_cs[0], V2_cs[0], (M0_solvers[0], M1_solvers[1], M1_solvers[2]))
    M2_1_log_kron_solver = KroneckerLinearSolver(V2_cs[1], V2_cs[1], (M1_solvers[0], M0_solvers[1], M1_solvers[2]))
    M2_2_log_kron_solver = KroneckerLinearSolver(V2_cs[2], V2_cs[2], (M1_solvers[0], M1_solvers[1], M0_solvers[2]))
    M2_log_kron_solver = BlockLinearOperator(V2_cs, V2_cs, [[M2_0_log_kron_solver, None, None],
                                                            [None, M2_1_log_kron_solver, None],
                                                            [None, None, M2_2_log_kron_solver]])

    M0_pc = D0_inv_sqrt @ D0_log_sqrt @ M0_log_kron_solver @ D0_log_sqrt @ D0_inv_sqrt if M0 is not None else None
    M1_pc = D1_inv_sqrt @ D1_log_sqrt @ M1_log_kron_solver @ D1_log_sqrt @ D1_inv_sqrt if M1 is not None else None
    M2_pc = D2_inv_sqrt @ D2_log_sqrt @ M2_log_kron_solver @ D2_log_sqrt @ D2_inv_sqrt if M2 is not None else None
    return M0_pc, M1_pc, M2_pc
# -----------------------------------------


# ---------- beltrami.py script utilities ----------
def read_folder(simulation_folder_name):

    # ---------- Gather paths (coeff paths to be added later) ----------
    simulation_folder_path  = 'simulations/' + simulation_folder_name
    coefficient_folder_path = simulation_folder_path + '/coefficients'
    diagnostics_file        = simulation_folder_path + '/diagnostics.txt'
    about_file              = simulation_folder_path + '/about.txt'
    space_file              = simulation_folder_path + '/spaces.yml'

    paths = {'simulation_folder_path':simulation_folder_path, 
             'diagnostics_file':diagnostics_file,
             'about_file':about_file,
             'space_file':space_file}
    # ------------------------------------------------------------------

    # ---------- Read from about.txt ----------
    f           = open(about_file, 'r')
    about_lines = f.readlines()
    f.close()

    ncells              = [int(n) for n in about_lines[1].split()[1:]]
    degree              = [int(d) for d in about_lines[2].split()[1:]]
    periodic            = [b == 'True' for b in about_lines[3].split()[1:]]
    domain_name         = about_lines[4].split()[1]
    intended_MPI_size   = int(about_lines[5].split()[1])

    store_hamiltonian_at    = int(about_lines[7].split()[1])
    store_coefficients_at   = float(about_lines[8].split()[1])          # this used to be int() previously
    f_update            = float(about_lines[9].split()[1])
    N_min               = int(about_lines[10].split()[1])
    N_max               = int(about_lines[11].split()[1])
    max_N_p             = int(about_lines[12].split()[1])
    dt0                 = float(about_lines[13].split()[1])
    dt_max              = float(about_lines[14].split()[1])
    tol_p               = float(about_lines[15].split()[1])
    tol_cg              = float(about_lines[16].split()[1])
    tol_vp              = float(about_lines[17].split()[1])

    params = {'ncells':ncells, 'degree':degree, 'periodic':periodic, 'domain_name':domain_name, 'intended_MPI_size':intended_MPI_size,
              'store_hamiltonian_at':store_hamiltonian_at, 'store_coefficients_at':store_coefficients_at,
              'f_update':f_update, 'N_min':N_min, 'N_max':N_max, 'max_N_p':max_N_p,
              'dt0':dt0, 'dt_max':dt_max,
              'tol_p':tol_p, 'tol_cg':tol_cg, 'tol_vp':tol_vp}
    # -----------------------------------------
    
    

    # ---------- Gather restart data from diagnostics.txt and about.txt (if applicable) ----------
    if len(about_lines) > 19:
        # -----
        last_about_line         = about_lines[-1].split()
        previous_simulation_nr  = int(last_about_line[1])
        previous_T              = float(last_about_line[-1])
        params['previous_simulation_nr'] = previous_simulation_nr
        params['previous_T']    = previous_T
        # -----
        f                       = open(diagnostics_file, 'r')
        diagnostics_lines       = f.readlines()
        f.close()

        last_line               = diagnostics_lines[-1].split()
        p_timestep, p_time, p_stepsize, p_hamil, p_entropy, p_div, p_equil, p_runtime, p_psteps, p_cgits, p_QTtime, p_QAtime = last_line
        restart_data            = {'p_timestep':p_timestep, 'p_time':p_time, 'p_stepsize':p_stepsize,
                                   'p_hamil':p_hamil, 'p_entropy':p_entropy, 'p_div':p_div, 'p_equil':p_equil,
                                   'p_runtime':p_runtime, 'p_psteps':p_psteps, 'p_cgits':p_cgits,
                                   'p_QTtime':p_QTtime, 'p_QAtime':p_QAtime}
        # -----
        previous_coefficient_file           = coefficient_folder_path + f'/coeffs_simulation_{previous_simulation_nr}.h5'
        current_coefficient_file            = coefficient_folder_path + f'/coeffs_simulation_{previous_simulation_nr + 1}.h5'
        paths['previous_coefficient_file']  = previous_coefficient_file
        paths['current_coefficient_file']   = current_coefficient_file
    else:
        paths['current_coefficient_file']   = coefficient_folder_path + '/coeffs_simulation_1.h5'
    # --------------------------------------------------------------------------------------------

    data = {'params':params, 
            'paths':paths}
    if len(about_lines) > 19:
        data['restart_data'] = restart_data
    
    return data

def store_coefficients(Om, b, Vh, t, ts):
    B = FemField(Vh, b)
    Om.add_snapshot(t=t, ts=ts)
    Om.export_fields(u2=B)

def get_coefficients_from_data(V_vs, data):
    V0 = V_vs.spaces[0]
    V1 = V_vs.spaces[1]
    V2 = V_vs.spaces[2]

    v0 = V0.zeros()
    v1 = V1.zeros()
    v2 = V2.zeros()

    pads0_0, pads0_1, pads0_2 = V0.pads
    pads1_0, pads1_1, pads1_2 = V1.pads
    pads2_0, pads2_1, pads2_2 = V2.pads

    v0._data[pads0_0:-pads0_0, pads0_1:-pads0_1, pads0_2:-pads0_2] = data[0]
    v1._data[pads1_0:-pads1_0, pads1_1:-pads1_1, pads1_2:-pads1_2] = data[1]
    v2._data[pads2_0:-pads2_0, pads2_1:-pads2_1, pads2_2:-pads2_2] = data[2]

    coeffs = BlockVector(V_vs, (v0, v1, v2))

    return coeffs

def get_coefficients(V_vs, filename, step):

    with h5py.File(filename, "r") as f:

        snapshot = list(f.keys())[step]
        domain_name = list(f[snapshot].keys())[0]

        data_0 = f[snapshot][domain_name]['V2[0]']['u2[0]'][()]
        data_1 = f[snapshot][domain_name]['V2[1]']['u2[1]'][()]
        data_2 = f[snapshot][domain_name]['V2[2]']['u2[2]'][()]

        data = [data_0, data_1, data_2]

    coeffs = get_coefficients_from_data(V_vs, data)

    return coeffs

def store_diagnostics(diagnostics_file, step, time, stepsize, hamil, entro, diver, equil, ttime, picar, cgite, qatot, qaavg):
    txt = f'{step}\t{time}\t{stepsize}\t{hamil}\t{entro}\t{diver}\t{equil}\t{ttime}\t{picar}\t{cgite}\t{qatot}\t{qaavg}\n'
    f = open(diagnostics_file, 'a')
    f.writelines(txt)
    f.close()

def read_coefficient_files(simulation_folder_name, V_vs):

    simulation_folder_path = 'simulations/' + simulation_folder_name
    coefficient_folder_path = simulation_folder_path + '/coefficients'
    about_file_path = simulation_folder_path + '/about.txt'

    f               = open(about_file_path, 'r')
    about_lines     = f.readlines()
    last_line       = about_lines[-1].split()
    mpi_size        = int(about_lines[5].split()[1])
    n_simulations   = int(last_line[1])

    coefficient_file_paths = [coefficient_folder_path + f'/coeffs_simulation_{n+1}.h5' for n in range(n_simulations)]

    data = {}

    for n, cfp in enumerate(coefficient_file_paths):
        file_key = str(int(n+1))
        file_data = {}
        with h5py.File(cfp, 'r') as f:
            snapshots = list(f.keys())
            if mpi_size > 1:
                snapshots = snapshots[1:]
            for s, snapshot in enumerate(snapshots):
                snapshot_key = str(int(s+1))
                t = f[snapshot].attrs['t']
                ts = f[snapshot].attrs['ts']
                s_key = list(f[snapshot].keys())[0]
                data_0 = f[snapshot][s_key]['V2[0]']['u2[0]'][()]
                data_1 = f[snapshot][s_key]['V2[1]']['u2[1]'][()]
                data_2 = f[snapshot][s_key]['V2[2]']['u2[2]'][()]
                coeff_data = [data_0, data_1, data_2]
                coeffs = get_coefficients_from_data(V_vs, coeff_data)
                file_data[snapshot_key] = {'t':t, 'ts':ts, 'coeffs':coeffs}
        data[file_key] = file_data
    return data

def parallel_data_transfer(b, V):
    v = V.zeros()
    localslice0 = tuple([slice(s, e+1) for s, e in zip(V[0].starts, V[0].ends)])
    v[0][localslice0] = b[0][localslice0]
    localslice1 = tuple([slice(s, e+1) for s, e in zip(V[1].starts, V[1].ends)])
    v[1][localslice1] = b[1][localslice1]
    localslice2 = tuple([slice(s, e+1) for s, e in zip(V[2].starts, V[2].ends)])
    v[2][localslice2] = b[2][localslice2]
    return v
# --------------------------------------------------


# ---------- Boundary Projectors ----------
# also here (and everywhere else!): check first always if toarray() method is implemented before using the slow alternative!
class H1BoundaryProjector_1D(LinearOperator):
    def __init__(self, domain, space, periodic):
        
        self._domain = domain
        self._space = space
        self._periodic = periodic
        assert isinstance(periodic, bool)
        assert space.kind == H1Space
        self._BC = self._get_BC()

    def _get_BC(self):
        periodic    = self._periodic
        if periodic:
            return None
        space       = self._space

        u   = element_of(space, name='u')
        bcs = [EssentialBC(u, 0, side, position=0) for side in space.domain.boundary]

        return bcs

    @property
    def domain(self):
        return self._domain
    
    @property
    def codomain(self):
        return self._domain
    
    @property
    def dtype(self):
        return None
    
    def tosparse(self):
        raise NotImplementedError
    
    def toarray(self):
        #return toarray_1d(self)
        return toarray(self)
    
    def transpose(self, conjugate=False):
        return self
    
    def dot(self, v, out=None):
        BC = self._BC
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space is self.codomain
        else:
            out = self.codomain.zeros()

        v.copy(out=out)
        if BC is not None:
            apply_essential_bc(out, *BC)
        return out


class H1BoundaryProjector_3D(LinearOperator):
    def __init__(self, domain, space, periodic):

        self._domain = domain
        self._space = space
        self._periodic = periodic
        self._BC = self._get_BC()
        assert all([isinstance(P, bool) for P in periodic])
        assert space.kind == H1Space

    def _get_BC(self):
        periodic    = self._periodic
        if all([P == True for P in periodic]):
            return None
        space       = self._space
        
        u   = element_of(space, name='u')
        bcs = [EssentialBC(u, 0, side, position=0) for side in space.domain.boundary]

        bcs_x = [bcs[0], bcs[1]] if periodic[0] == False else []
        bcs_y = [bcs[2], bcs[3]] if periodic[1] == False else []
        bcs_z = [bcs[4], bcs[5]] if periodic[2] == False else []

        BC  = bcs_x + bcs_y + bcs_z

        return BC

    @property
    def domain(self):
        return self._domain
    
    @property
    def codomain(self):
        return self._domain
    
    @property
    def dtype(self):
        return None
    
    def tosparse(self):
        raise NotImplementedError
    
    def toarray(self):
        return toarray(self)
    
    def transpose(self, conjugate=False):
        return self

    def dot(self, v, out=None):
        BC = self._BC
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space is self.codomain
        else:
            out = self.codomain.zeros()

        v.copy(out=out)
        apply_essential_bc(out, *BC)
        return out


class HcurlBoundaryProjector_3D(LinearOperator):
    def __init__(self, domain, space, periodic):

        self._domain = domain
        self._space = space
        self._periodic = periodic
        self._BC = self._get_BC()
        assert all([isinstance(P, bool) for P in periodic])
        assert space.kind == HcurlSpace

    def _get_BC(self):
        periodic    = self._periodic
        if all([P == True for P in periodic]):
            return None
        space       = self._space
        
        u   = element_of(space, name='u')
        bcs = [EssentialBC(u, 0, side, position=0) for side in space.domain.boundary]

        if periodic[1] == False:
            bcs_x = [bcs[2], bcs[3], bcs[4], bcs[5]] if periodic[2] == False else [bcs[2], bcs[3]]
        else:
            bcs_x = [bcs[4], bcs[5]] if periodic[2] == False else []

        if periodic[0] == False:
            bcs_y = [bcs[0], bcs[1], bcs[4], bcs[5]] if periodic[2] == False else [bcs[0], bcs[1]]
        else:
            bcs_y = [bcs[4], bcs[5]] if periodic[2] == False else []

        if periodic[0] == False:
            bcs_z = [bcs[0], bcs[1], bcs[2], bcs[3]] if periodic[1] == False else [bcs[0], bcs[1]]
        else:
            bcs_z = [bcs[2], bcs[3]] if periodic[1] == False else []

        BC  = [bcs_x, bcs_y, bcs_z]

        return BC

    @property
    def domain(self):
        return self._domain
    
    @property
    def codomain(self):
        return self._domain
    
    @property
    def dtype(self):
        return None
    
    def tosparse(self):
        raise NotImplementedError
    
    def toarray(self):
        return toarray(self)
    
    def transpose(self, conjugate=False):
        return self

    def dot(self, v, out=None):
        BC = self._BC
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space is self.codomain
        else:
            out = self.codomain.zeros()

        v.copy(out=out)
        for outi, BCi in zip(out, BC):
            apply_essential_bc(outi, *BCi)
        return out


class HdivBoundaryProjector_3D(LinearOperator):
    def __init__(self, domain, space, periodic):

        self._domain = domain
        self._space = space
        self._periodic = periodic
        self._BC = self._get_BC()
        assert all([isinstance(P, bool) for P in periodic])
        assert space.kind == HdivSpace

    def _get_BC(self):
        periodic    = self._periodic
        if all([P == True for P in periodic]):
            return None
        space       = self._space
        
        u   = element_of(space, name='u')
        bcs = [EssentialBC(u, 0, side, position=0) for side in space.domain.boundary]

        bcs_x = [bcs[0], bcs[1]] if periodic[0] == False else []
        bcs_y = [bcs[2], bcs[3]] if periodic[1] == False else []
        bcs_z = [bcs[4], bcs[5]] if periodic[2] == False else []

        BC  = [bcs_x, bcs_y, bcs_z]

        return BC

    @property
    def domain(self):
        return self._domain
    
    @property
    def codomain(self):
        return self._domain
    
    @property
    def dtype(self):
        return None
    
    def tosparse(self):
        raise NotImplementedError
    
    def toarray(self):
        return toarray(self)
    
    def transpose(self, conjugate=False):
        return self

    def dot(self, v, out=None):
        BC = self._BC
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space is self.codomain
        else:
            out = self.codomain.zeros()

        v.copy(out=out)
        for outi, BCi in zip(out, BC):
            apply_essential_bc(outi, *BCi)
        return out
# -----------------------------------------


# ---------- Initial Condition ----------
def get_A_fun(n=1, m=1, A0=1e04):
    mu_tilde = np.sqrt(m**2 + n**2)  

    eta = lambda x, y, z: x**2 * (1-x)**2 * y**2 * (1-y)**2 * z**2 * (1-z)**2

    u1  = lambda x, y, z:  A0 * (n/mu_tilde) * np.sin(np.pi * m * x) * np.cos(np.pi * n * y)
    u2  = lambda x, y, z: -A0 * (m/mu_tilde) * np.cos(np.pi * m * x) * np.sin(np.pi * n * y)
    u3  = lambda x, y, z:  A0 * np.sin(np.pi * m * x) * np.sin(np.pi * n * y)

    A1 = lambda x, y, z: eta(x, y, z) * u1(x, y, z)
    A2 = lambda x, y, z: eta(x, y, z) * u2(x, y, z)
    A3 = lambda x, y, z: eta(x, y, z) * u3(x, y, z)

    A = (A1, A2, A3)
    return A

def get_B_SquareTorus_fun(r, R, C0=-25, C1=50):

    rad = lambda x, y : np.sqrt(x**2 + y**2)

    B1 = lambda x, y, z : (1/rad(x, y)) * ( (-C0 * y) + C1 * ( (x/rad(x, y))*(rad(x, y) - r)*(rad(x, y) - R)*(2*z-1) ) )
    B2 = lambda x, y, z : (1/rad(x, y)) * ( (C0 * x) + C1 * ( (y/rad(x, y))*(rad(x, y) - r)*(rad(x, y) - R)*(2*z-1) ) )
    B3 = lambda x, y, z : (1/rad(x, y)) * ( C1 * ((r+R)-2*rad(x, y))*z*(z-1) )

    B = (B1, B2, B3)
    return B

def get_B_SquareTorus_fun2(r, R, C0=1e5, C1=5e1):
    rad = lambda x, y : np.sqrt(x**2 + y**2)

    B1 = lambda x, y, z : (1/rad(x, y)**2) * ( (-C0 * y) + C1 * ( x * (rad(x, y) - r)*(rad(x, y) - R)*(2*z-1) ) )
    B2 = lambda x, y, z : (1/rad(x, y)**2) * ( (C0 * x) + C1 * ( y *(rad(x, y) - r)*(rad(x, y) - R)*(2*z-1) ) )
    B3 = lambda x, y, z : (1/rad(x, y)**2) * ( C1 * rad(x, y) * ((r+R)-2*rad(x, y))*z*(z-1) )

    B = (B1, B2, B3)
    return B
# ---------------------------------------


# ---------- PCs on the logical domain ----------
def get_M0_kron_solver(V0, ncells, degree, periodic):
    """
    Given a 3D DeRham sequenece (V0 = H(grad) --grad--> V1 = H(curl) --curl--> V2 = H(div) --div--> V3 = L2)
    discreticed using ncells, degree and periodic,

        domain = Cube('C', bounds1=(0, 1), bounds2=(0, 1), bounds3=(0, 1))
        derham = Derham(domain)
        domain_h = discretize(domain, ncells=ncells, periodic=periodic, comm=comm)
        derham_h = discretize(derham, domain_h, degree=degree),

    returns the inverse of the mass matrix M0 as a KroneckerLinearSolver.
    """
    raise NotImplementedError('Must include bounds of the logical domain!')
    # assert 3D
    assert len(ncells) == 3
    assert len(degree) == 3
    assert len(periodic) == 3

    # 1D domain to be discreticed using the respective values of ncells, degree, periodic
    domain_1d = Line('L', bounds=(0,1))
    derham_1d = Derham(domain_1d)

    # storage for the 1D mass matrices
    M0_matrices = []
    M1_matrices = []

    # assembly of the 1D mass matrices
    for (n, p, P) in zip(ncells, degree, periodic):

        domain_1d_h = discretize(domain_1d, ncells=[n], periodic=[P])
        derham_1d_h = discretize(derham_1d, domain_1d_h, degree=[p])

        V0_1d       = derham_1d.V0
        V1_1d       = derham_1d.V1
        V0_1d_h       = derham_1d_h.V0
        V1_1d_h       = derham_1d_h.V1
        V0_1d_cs = V0_1d_h.coeff_space
        P_1d = H1BoundaryProjector_1D(V0_1d_cs, V0_1d, P)
        IP_1d = IdentityOperator(V0_1d_cs) - P_1d

        u_1d_0, v_1d_0 = elements_of(V0_1d, names='u_1d_0, v_1d_0')
        u_1d_1, v_1d_1 = elements_of(V1_1d, names='u_1d_1, v_1d_1')

        a_1d_0 = BilinearForm((u_1d_0, v_1d_0), integral(domain_1d, u_1d_0 * v_1d_0))
        a_1d_1 = BilinearForm((u_1d_1, v_1d_1), integral(domain_1d, u_1d_1 * v_1d_1))

        a_1d_0_h = discretize(a_1d_0, domain_1d_h, (V0_1d_h, V0_1d_h))
        a_1d_1_h = discretize(a_1d_1, domain_1d_h, (V1_1d_h, V1_1d_h))

        M_1d_0 = a_1d_0_h.assemble()
        M_1d_0_0 = P_1d @ M_1d_0 @ P_1d + IP_1d
        M_1d_1 = a_1d_1_h.assemble()

        #M0_matrices.append(M_1d_0)
        M0_matrices.append(M_1d_0_0)
        M1_matrices.append(M_1d_1)

    M0_solvers = [matrix_to_bandsolver(M) for M in M0_matrices]

    M0_kron_solver = KroneckerLinearSolver(V0, V0, M0_solvers)

    return M0_kron_solver

def get_M1_block_kron_solver(V1, ncells, degree, periodic):
    """
    Given a 3D DeRham sequenece (V0 = H(grad) --grad--> V1 = H(curl) --curl--> V2 = H(div) --div--> V3 = L2)
    discreticed using ncells, degree and periodic,

        domain = Cube('C', bounds1=(0, 1), bounds2=(0, 1), bounds3=(0, 1))
        derham = Derham(domain)
        domain_h = discretize(domain, ncells=ncells, periodic=periodic, comm=comm)
        derham_h = discretize(derham, domain_h, degree=degree),

    returns the inverse of the mass matrix M1 as a BlockLinearOperator consisting of three KroneckerLinearSolvers on the diagonal.
    """
    raise NotImplementedError('Must include bounds of the logical domain!')
    # assert 3D
    assert len(ncells) == 3
    assert len(degree) == 3
    assert len(periodic) == 3

    # 1D domain to be discreticed using the respective values of ncells, degree, periodic
    domain_1d = Line('L', bounds=(0,1))
    derham_1d = Derham(domain_1d)

    # storage for the 1D mass matrices
    M0_matrices = []
    M1_matrices = []

    # assembly of the 1D mass matrices
    for (n, p, P) in zip(ncells, degree, periodic):

        domain_1d_h = discretize(domain_1d, ncells=[n], periodic=[P])
        derham_1d_h = discretize(derham_1d, domain_1d_h, degree=[p])

        V0_1d       = derham_1d.V0
        V1_1d       = derham_1d.V1
        V0_1d_h       = derham_1d_h.V0
        V1_1d_h       = derham_1d_h.V1
        V0_1d_cs = V0_1d_h.coeff_space
        P_1d = H1BoundaryProjector_1D(V0_1d_cs, V0_1d, P)
        IP_1d = IdentityOperator(V0_1d_cs) - P_1d

        u_1d_0, v_1d_0 = elements_of(V0_1d, names='u_1d_0, v_1d_0')
        u_1d_1, v_1d_1 = elements_of(V1_1d, names='u_1d_1, v_1d_1')

        a_1d_0 = BilinearForm((u_1d_0, v_1d_0), integral(domain_1d, u_1d_0 * v_1d_0))
        a_1d_1 = BilinearForm((u_1d_1, v_1d_1), integral(domain_1d, u_1d_1 * v_1d_1))

        a_1d_0_h = discretize(a_1d_0, domain_1d_h, (V0_1d_h, V0_1d_h))
        a_1d_1_h = discretize(a_1d_1, domain_1d_h, (V1_1d_h, V1_1d_h))

        M_1d_0 = a_1d_0_h.assemble()
        M_1d_0_0 = P_1d @ M_1d_0 @ P_1d + IP_1d
        M_1d_1 = a_1d_1_h.assemble()

        #M0_matrices.append(M_1d_0)
        M0_matrices.append(M_1d_0_0)
        M1_matrices.append(M_1d_1)

    V1_1 = V1[0]
    V1_2 = V1[1]
    V1_3 = V1[2]

    B1_mat = [M1_matrices[0], M0_matrices[1], M0_matrices[2]]
    B2_mat = [M0_matrices[0], M1_matrices[1], M0_matrices[2]]
    B3_mat = [M0_matrices[0], M0_matrices[1], M1_matrices[2]]

    B1_solvers = [matrix_to_bandsolver(Ai) for Ai in B1_mat]
    B2_solvers = [matrix_to_bandsolver(Ai) for Ai in B2_mat]
    B3_solvers = [matrix_to_bandsolver(Ai) for Ai in B3_mat]

    B1_kron_inv = KroneckerLinearSolver(V1_1, V1_1, B1_solvers)
    B2_kron_inv = KroneckerLinearSolver(V1_2, V1_2, B2_solvers)
    B3_kron_inv = KroneckerLinearSolver(V1_3, V1_3, B3_solvers)

    M1_block_kron_solver = BlockLinearOperator(V1, V1, ((B1_kron_inv, None, None), 
                                                              (None, B2_kron_inv, None), 
                                                              (None, None, B3_kron_inv)))

    return M1_block_kron_solver

def get_M2_block_kron_solver(V2, ncells, degree, periodic):
    """
    Given a 3D DeRham sequenece (V0 = H(grad) --grad--> V1 = H(curl) --curl--> V2 = H(div) --div--> V3 = L2)
    discreticed using ncells, degree and periodic,

        domain      = Cube('C', bounds1=(0, 1), bounds2=(0, 1), bounds3=(0, 1))
        derham      = Derham(domain)
        domain_h    = discretize(domain, ncells=ncells, periodic=periodic, comm=comm)
        derham_h    = discretize(derham, domain_h, degree=degree),

    returns the inverse of the mass matrix M2 as a BlockLinearOperator consisting of three KroneckerLinearSolvers on the diagonal.
    """
    raise NotImplementedError('Must include bounds of the logical domain!')
    # assert 3D
    assert len(ncells) == 3
    assert len(degree) == 3
    assert len(periodic) == 3

    # 1D domain to be discreticed using the respective values of ncells, degree, periodic
    domain_1d = Line('L', bounds=(0,1))
    derham_1d = Derham(domain_1d)

    # storage for the 1D mass matrices
    M0_matrices = []
    M1_matrices = []

    # assembly of the 1D mass matrices
    for (n, p, P) in zip(ncells, degree, periodic):

        domain_1d_h = discretize(domain_1d, ncells=[n], periodic=[P])
        derham_1d_h = discretize(derham_1d, domain_1d_h, degree=[p])

        V0_1d       = derham_1d.V0
        V1_1d       = derham_1d.V1
        V0_1d_h       = derham_1d_h.V0
        V1_1d_h       = derham_1d_h.V1
        V0_1d_cs = V0_1d_h.coeff_space
        P_1d = H1BoundaryProjector_1D(V0_1d_cs, V0_1d, P)
        IP_1d = IdentityOperator(V0_1d_cs) - P_1d

        u_1d_0, v_1d_0 = elements_of(V0_1d, names='u_1d_0, v_1d_0')
        u_1d_1, v_1d_1 = elements_of(V1_1d, names='u_1d_1, v_1d_1')

        a_1d_0 = BilinearForm((u_1d_0, v_1d_0), integral(domain_1d, u_1d_0 * v_1d_0))
        a_1d_1 = BilinearForm((u_1d_1, v_1d_1), integral(domain_1d, u_1d_1 * v_1d_1))

        a_1d_0_h = discretize(a_1d_0, domain_1d_h, (V0_1d_h, V0_1d_h))
        a_1d_1_h = discretize(a_1d_1, domain_1d_h, (V1_1d_h, V1_1d_h))

        M_1d_0 = a_1d_0_h.assemble()
        M_1d_0_0 = P_1d @ M_1d_0 @ P_1d + IP_1d
        M_1d_1 = a_1d_1_h.assemble()

        #M0_matrices.append(M_1d_0)
        M0_matrices.append(M_1d_0_0)
        M1_matrices.append(M_1d_1)

    V2_1 = V2[0]
    V2_2 = V2[1]
    V2_3 = V2[2]

    B1_mat = [M0_matrices[0], M1_matrices[1], M1_matrices[2]]
    B2_mat = [M1_matrices[0], M0_matrices[1], M1_matrices[2]]
    B3_mat = [M1_matrices[0], M1_matrices[1], M0_matrices[2]]

    B1_solvers = [matrix_to_bandsolver(Ai) for Ai in B1_mat]
    B2_solvers = [matrix_to_bandsolver(Ai) for Ai in B2_mat]
    B3_solvers = [matrix_to_bandsolver(Ai) for Ai in B3_mat]

    B1_kron_inv = KroneckerLinearSolver(V2_1, V2_1, B1_solvers)
    B2_kron_inv = KroneckerLinearSolver(V2_2, V2_2, B2_solvers)
    B3_kron_inv = KroneckerLinearSolver(V2_3, V2_3, B3_solvers)

    M2_block_kron_solver = BlockLinearOperator(V2, V2, ((B1_kron_inv, None, None), 
                                                              (None, B2_kron_inv, None), 
                                                              (None, None, B3_kron_inv)))

    return M2_block_kron_solver
# --------------------------------------------------


# ---------- 2D code ----------
class H1BoundaryProjector2D(LinearOperator):
    def __init__(self, V0, V0_vs, periodic=[False, False]):

        assert all([isinstance(P, bool) for P in periodic])

        self._domain    = V0_vs
        self._codomain  = V0_vs
        self._space     = V0
        self._periodic  = periodic

        self._BC        = self._get_BC()
        
    def copy(self):
        return H1BoundaryProjector2D(self._space, self._domain, self._periodic)

    def _get_BC(self):

        periodic = self._periodic
        if all([P == True for P in periodic]):
            return None
        
        space   = self._space
        u       = element_of(space, name='u')
        bcs     = [EssentialBC(u, 0, side, position=0) for side in space.domain.boundary]

        bcs_x   = [bcs[0], bcs[1]] if periodic[0] == False else []
        bcs_y   = [bcs[2], bcs[3]] if periodic[1] == False else []

        BC      = bcs_x + bcs_y

        return BC

    @property
    def domain(self):
        return self._domain
    
    @property
    def codomain(self):
        return self._codomain
    
    @property
    def dtype(self):
        return None
    
    def tosparse(self):
        raise NotImplementedError
    
    def toarray(self):
        return toarray(self)
    
    def transpose(self, conjugate=False):
        return self

    def dot(self, v, out=None):
        BC = self._BC
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space is self.codomain
        else:
            out = self.codomain.zeros()

        v.copy(out=out)
        apply_essential_bc(out, *BC)
        return out


class HcurlBoundaryProjector2D(LinearOperator):
    def __init__(self, V1, V1_vs, periodic=[False, False]):

        assert all([isinstance(P, bool) for P in periodic])

        self._domain    = V1_vs
        self._codomain  = V1_vs
        self._space     = V1
        self._periodic  = periodic

        self._BC        = self._get_BC()
        
    def copy(self):
        return HcurlBoundaryProjector2D(self._space, self._domain, self._periodic)

    def _get_BC(self):

        periodic = self._periodic
        if all([P == True for P in periodic]):
            return None
        
        space   = self._space
        u       = element_of(space, name='u')
        bcs     = [EssentialBC(u, 0, side, position=0) for side in space.domain.boundary]

        bcs_x   = [bcs[0], bcs[1]] if periodic[0] == False else []
        bcs_y   = [bcs[2], bcs[3]] if periodic[1] == False else []

        BC      = bcs_x + bcs_y

        bcs_x = [bcs[2], bcs[3]] if periodic[1] == False else []
        bcs_y = [bcs[0], bcs[1]] if periodic[0] == False else []

        BC  = [bcs_x, bcs_y]

        return BC

    @property
    def domain(self):
        return self._domain
    
    @property
    def codomain(self):
        return self._codomain
    
    @property
    def dtype(self):
        return None
    
    def tosparse(self):
        raise NotImplementedError
    
    def toarray(self):
        return toarray(self)
    
    def transpose(self, conjugate=False):
        return self

    def dot(self, v, out=None):
        BC = self._BC
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space is self.codomain
        else:
            out = self.codomain.zeros()

        v.copy(out=out)
        for outi, BCi in zip(out, BC):
            apply_essential_bc(outi, *BCi)
        return out


def get_diagonal_2d(A, inv=True, sqrt=False, tol=1e-13):
    """
    Takes a square LinearOperator A and returns a similar diagonal LinearOperator D with A's diagonal entries.

    If inv=True, returns a diagonal LinearOperator with A's inverse diagonal entries.
    In that case, a diagonal entry < tol will result in a ValueError.
    If sqrt = True, replaces the diagonal entries by their square roots (D->D^{1/2}; D^{-1}->D^{-1/2}).
    
    """
    assert isinstance(inv, bool)
    assert isinstance(sqrt, bool)
    assert isinstance(A, LinearOperator)
    assert A.domain is A.codomain

    V   = A.domain
    v   = V.zeros()
    is_block = True if isinstance(V, BlockVectorSpace) else False
    if not is_block:
        assert isinstance(V, StencilVectorSpace)

    if is_block:
        D = BlockLinearOperator(V, V)
    else:
        D = StencilMatrix(V, V)

    if is_block:
        for block_index in range(V.n_blocks):
            diag_values = []
            D_block = D[block_index, block_index]
            V_block = V[block_index]
            npts1, npts2 = V_block.npts
            pads1, pads2 = V_block.pads

            for n1 in range(npts1):
                diag_values_block = []
                for n2 in range(npts2):
                    v *= 0.0
                    v[block_index]._data[pads1+n1, pads2+n2] = 1.
                    w = A @ v
                    d = w[block_index]._data[pads1+n1, pads2+n2]
                    if sqrt:
                        assert d >= 0, f'Diagonal entry {d} must be positive in order to apply the sqrt.'
                        d = np.sqrt(d)
                    if inv:
                        if abs(d) < tol:
                            raise ValueError(f'Diagonal value d with abs(d) < tol = {tol} encountered.')
                        else:
                            d = 1 / d
                    diag_values_block.append(d)
                diag_values.append(diag_values_block)
            diag_values = np.array(diag_values)
            D_block = StencilMatrix(V_block, V_block)
            D_block._data[D_block._get_diagonal_indices()] = diag_values
            D[block_index, block_index] = D_block
    else:
        diag_values = []
        npts1, npts2 = V.npts
        pads1, pads2 = V.pads

        for n1 in range(npts1):
            diag_values_block = []
            for n2 in range(npts2):
                v *= 0.0
                v._data[pads1+n1, pads2+n2] = 1.
                w = A @ v
                d = w._data[pads1+n1, pads2+n2]
                if sqrt:
                    assert d >= 0, f'Diagonal entry {d} must be positive in order to apply the sqrt.'
                    d = np.sqrt(d)
                if inv:
                    if abs(d) < tol:
                        raise ValueError(f'Diagonal value d with abs(d) < tol = {tol} encountered.')
                    else:
                        d = 1 / d
                diag_values_block.append(d)
            diag_values.append(diag_values_block)
        diag_values = np.array(diag_values)
        D._data[D._get_diagonal_indices()] = diag_values

    return D

def get_LST_pcs_2d(M0, M1, logical_domain, Vs, Vhs, Vcs, ncells, degree, periodic, comm, backend):
    """
    LST (Loli, Sangalli, Tani) preconditioners are mass matrix preconditioners of the form

    pc = D_inv_sqrt @ D_log_sqrt @ M_log_kron_solver @ D_log_sqrt @ D_inv_sqrt,

    where

    D_inv_sqrt          is the diagonal matrix of the square roots of the inverse diagonal entries of the mass matrix M,
    D_log_sqrt          is the diagonal matrix of the square roots of the diagonal entries of the mass matrix on the logical domain,
    M_log_kron_solver   is the Kronecker Solver of the mass matrix on the logical domain.

    These preconditioners work very well even on complex domains as numerical experiments have shown.
    
    """
    ###
    ### What about BCs???
    ###

    # ---------- D_inv_sqrt ----------
    D0_inv_sqrt      = get_diagonal_2d(M0, inv=True, sqrt=True)
    D1_inv_sqrt      = get_diagonal_2d(M1, inv=True, sqrt=True)
    # --------------------------------

    # ---------- D_log_sqrt ----------
    #logical_derham = Derham(logical_domain)

    logical_domain_h = discretize(logical_domain, ncells=ncells, periodic=periodic, comm=comm)
    #logical_derham_h = discretize(logical_derham, logical_domain_h, degree=degree)

    V0_log, V1_log, V2_log = Vs
    #V0_log = derham.V0 # logical_derham.V0
    #V1_log = derham.V1 # logical_derham.V1
    #V2_log = derham.V2 # logical_derham.V2

    V0h_log, V1h_log, V2h_log = Vhs
    #V0h_log = derham_h.V0 # logical_derham_h.V0
    #V1h_log = derham_h.V1 # logical_derham_h.V1
    #V2h_log = derham_h.V2 # logical_derham_h.V2

    V0_cs, V1_cs, V2_cs = Vcs
    #V0_cs   = V0h_log.coeff_space
    #V1_cs   = V1h_log.coeff_space
    #V2_cs   = V2h_log.coeff_space

    u0, v0  = elements_of(V0_log, names='u0, v0')
    u1, v1  = elements_of(V1_log, names='u1, v1')

    a0      = BilinearForm((u0, v0), integral(logical_domain, u0*v0))
    a1      = BilinearForm((u1, v1), integral(logical_domain, dot(u1, v1)))

    a0h     = discretize(a0, logical_domain_h, (V0h_log, V0h_log), backend=backend)
    a1h     = discretize(a1, logical_domain_h, (V1h_log, V1h_log), backend=backend)

    M0_log  = a0h.assemble()
    M1_log  = a1h.assemble()
    
    D0_log_sqrt = get_diagonal_2d(M0_log, inv=False, sqrt=True)
    D1_log_sqrt = get_diagonal_2d(M1_log, inv=False, sqrt=True)
    # --------------------------------
    
    # ---------- M_log_kron_solver ----------
    ncells_x, ncells_y        = ncells
    degree_x, degree_y        = degree
    periodic_x, periodic_y    = periodic

    logical_domain_1d = Line('L', bounds=(0,1))
    derham_1d   = Derham(logical_domain_1d)

    domain_xh   = discretize(logical_domain_1d, ncells=[ncells_x, ], periodic=[periodic_x, ])
    domain_yh   = discretize(logical_domain_1d, ncells=[ncells_y, ], periodic=[periodic_y, ])

    derham_xh   = discretize(derham_1d, domain_xh, degree=[degree_x, ])
    derham_yh   = discretize(derham_1d, domain_yh, degree=[degree_y, ])

    V0_1d       = derham_1d.V0
    V1_1d       = derham_1d.V1

    V0_xh       = derham_xh.V0
    V1_xh       = derham_xh.V1

    V0_yh       = derham_yh.V0
    V1_yh       = derham_yh.V1

    V0_x_cs = V0_xh.coeff_space

    V0_y_cs = V0_yh.coeff_space

    Px = H1BoundaryProjector_1D(V0_x_cs, V0_1d, periodic_x)
    Py = H1BoundaryProjector_1D(V0_y_cs, V0_1d, periodic_y)

    IPx = IdentityOperator(V0_x_cs) - Px
    IPy = IdentityOperator(V0_y_cs) - Py

    u0_1d, v0_1d = elements_of(V0_1d, names='u0_1d, v0_1d')
    u1_1d, v1_1d = elements_of(V1_1d, names='u1_1d, v1_1d')

    m0_1d       = BilinearForm((u0_1d, v0_1d), integral(logical_domain_1d, u0_1d*v0_1d))
    m1_1d       = BilinearForm((u1_1d, v1_1d), integral(logical_domain_1d, u1_1d*v1_1d))

    m0_xh       = discretize(m0_1d, domain_xh, (V0_xh, V0_xh), backend=backend)
    m0_yh       = discretize(m0_1d, domain_yh, (V0_yh, V0_yh), backend=backend)

    m1_xh       = discretize(m1_1d, domain_xh, (V1_xh, V1_xh), backend=backend)
    m1_yh       = discretize(m1_1d, domain_yh, (V1_yh, V1_yh), backend=backend)

    M0_x        = m0_xh.assemble()
    M0_y        = m0_yh.assemble()

    M0_x        = Px @ M0_x @ Px + IPx
    M0_y        = Py @ M0_y @ Py + IPy

    M1_x        = m1_xh.assemble()
    M1_y        = m1_yh.assemble()

    M0_solvers  = [matrix_to_bandsolver(M) for M in [M0_x, M0_y]]
    M1_solvers  = [matrix_to_bandsolver(M) for M in [M1_x, M1_y]]

    M0_log_kron_solver = KroneckerLinearSolver(V0_cs, V0_cs, (M0_solvers[0], M0_solvers[1]))

    M1_0_log_kron_solver = KroneckerLinearSolver(V1_cs[0], V1_cs[0], (M1_solvers[0], M0_solvers[1]))
    M1_1_log_kron_solver = KroneckerLinearSolver(V1_cs[1], V1_cs[1], (M0_solvers[0], M1_solvers[1]))
    M1_log_kron_solver = BlockLinearOperator(V1_cs, V1_cs, [[M1_0_log_kron_solver, None],
                                                            [None, M1_1_log_kron_solver]])


    M0_pc = D0_inv_sqrt @ D0_log_sqrt @ M0_log_kron_solver @ D0_log_sqrt @ D0_inv_sqrt
    M1_pc = D1_inv_sqrt @ D1_log_sqrt @ M1_log_kron_solver @ D1_log_sqrt @ D1_inv_sqrt

    return M0_pc, M1_pc
# -----------------------------












# ---------- old code ----------

'''
def get_unit_vector(v, n1, n2, n3, pads1, pads2, pads3):

    v *= 0.0
    if n3 is None:
        assert pads3 is None
        if n2 is None:
            raise NotImplementedError('This get_unit_vector method is only implemented in 2D.')
        else:
            v._data[pads1+n1, pads2+n2] = 1.
    else:
        raise NotImplementedError('This get_unit_vector method is only implemented in 2D.')
    
    return v

def toarray(A):
    """Obtain a numpy array representation of a LinearOperator (which has not implemented toarray())."""
    assert isinstance(A, LinearOperator)

    W = A.codomain

    W_is_block = True if isinstance(W, BlockVectorSpace) else False
    if not W_is_block:
        assert isinstance(W, StencilVectorSpace)

    A_arr = np.zeros(A.shape, dtype="float64")
    w = W.zeros()
    At = A.T

    if W_is_block:
        codomain_blocks = [W[i] for i in range(W.n_blocks)]
    else:
        codomain_blocks = (W, )

    start_index = 0
    for k, Wk in enumerate(codomain_blocks):
        w *= 0.
        v = Wk.zeros()
        if len(Wk.npts) == 2:
            npts1, npts2 = Wk.npts
            pads1, pads2 = Wk.pads
            for n1 in range(npts1):
                for n2 in range(npts2):
                    e_n1_n2 = get_unit_vector(v, n1, n2, None, pads1, pads2, None)
                    if W_is_block:
                        w[k] = e_n1_n2
                        e_n1_n2 = w
                    A_n1_n2 = At @ e_n1_n2
                    A_arr[start_index + n1*npts2+n2, :] = A_n1_n2.toarray()
        else:
            raise NotImplementedError('This toarray method is currently only implemented in 2D.')
        start_index += Wk.dimension

    return A_arr
'''

'''
def get_unit_vector_1d(v, periodic, n1, npts1, pads1):

    v *= 0.0
    v._data[pads1+n1] = 1.
    if periodic:
        if n1 < pads1:
            v._data[-pads1+n1] = 1.
        if n1 >= npts1-pads1:
            v._data[n1-npts1+pads1] = 1.
    
    return v

def toarray_1d(A):
    """Obtain a numpy array representation of a LinearOperator (which has not implemented toarray())."""
    assert isinstance(A, LinearOperator)
    W = A.codomain
    periods = W.periods
    periodic = periods[0]
    assert isinstance(W, StencilVectorSpace)

    A_arr = np.zeros(A.shape, dtype="float64")
    w = W.zeros()
    At = A.T

    npts1,  = W.npts
    pads1,  = W.pads
    for n1 in range(npts1):
            e_n1_n2 = get_unit_vector_1d(w, periodic, n1, npts1, pads1)
            A_n1_n2 = At @ e_n1_n2
            A_arr[n1, :] = A_n1_n2.toarray()

    return A_arr
'''

# ------------------------------