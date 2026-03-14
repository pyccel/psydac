#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np
import pytest

from psydac.core.bsplines    import make_knots
from psydac.fem.basic        import FemField
from psydac.fem.splines      import SplineSpace
from psydac.fem.tensor       import TensorFemSpace
from psydac.fem.vector       import VectorFemSpace

from psydac.feec.derivatives import DirectionalDerivativeOperator
from psydac.feec.derivatives import Derivative1D, Gradient2D, Gradient3D
from psydac.feec.derivatives import ScalarCurl2D, VectorCurl2D, Curl3D
from psydac.feec.derivatives import Divergence2D, Divergence3D

from psydac.ddm.cart         import DomainDecomposition
from mpi4py                  import MPI

#==============================================================================

# these tests test the DirectionalDerivativeOperator structurally.
# They do not check, if it really computes the derivatives
# (this is done in the gradient etc. tests below already)
def run_directional_derivative_operator(comm, domain, ncells, degree, periodic, direction, negative, transposed, seed, matrix_assembly=False):

    if not all([not periodic[i] or (ncells[i] >= degree[i]) for i in range(len(degree)) ]):
       return
    
    # assemble matrix when 1 cell in each direction
    matrix_assembly = (ncells == (1,1,1))
    
    # determinize tests
    np.random.seed(seed)

    breaks = [np.linspace(*lims, num=n+1) for lims, n in zip(domain, ncells)]

    Ns = [SplineSpace(degree=d, grid=g, periodic=p, basis='B') \
                                  for d, g, p in zip(degree, breaks, periodic)]

    domain_decomposition = DomainDecomposition(ncells, periodic, comm=comm)
    # original space
    V0 = TensorFemSpace(domain_decomposition, *Ns)

    # reduced space
    V1 = V0.reduce_degree(axes=[direction], basis='M')

    diffop = DirectionalDerivativeOperator(V0.coeff_space, V1.coeff_space, direction, negative=negative, transposed=transposed)

    # some boundary, and transposed handling
    vpads = np.array(V0.coeff_space.pads, dtype=int)
    pads = np.array(V1.coeff_space.pads, dtype=int)

    # transpose, if needed
    if transposed:
        V0, V1 = V1, V0
    
    counts = np.array(V1.coeff_space.ends, dtype=int) - np.array(V1.coeff_space.starts, dtype=int) + 1
    diffadd = np.zeros((len(ncells),), dtype=int)
    diffadd[direction] = 1

    localslice = tuple([slice(p,-p) for p in V1.coeff_space.pads])

    # random vector, scaled-up data (with fixed seed)
    v = V0.coeff_space.zeros()
    v._data[:] = np.random.random(v._data.shape) * 100
    v.update_ghost_regions()

    # compute reference solution (do it element-wise for now...)
    # (but we only test small domains here)
    ref = V1.coeff_space.zeros()

    outslice = tuple([slice(s, s+c) for s,c in zip(pads, counts)])
    idslice = tuple([slice(s, s+c) for s,c in zip(vpads, counts)])
    diffslice = tuple([slice(s+d, s+c+d) for s,c,d in zip(vpads, counts, diffadd)])
    if transposed:
        ref._data[idslice] -= v._data[outslice]
        ref._data[diffslice] += v._data[outslice]

        # we need to account for the ghost region write which diffslice does,
        # i.e. the part which might be sent to another process, or even swapped to the other side
        # since the ghost layers of v are updated, we update the data on the other side
        # (also update_ghost_regions won't preserve the data we wrote there)
        ref_restslice = [c for c in idslice]
        ref_restslice[direction] = slice(vpads[direction], vpads[direction] + 1)
        v_restslice = [c for c in outslice]
        v_restslice[direction] = slice(pads[direction] - 1, pads[direction])
        ref._data[tuple(ref_restslice)] += v._data[tuple(v_restslice)]
    else:
        ref._data[outslice] = v._data[diffslice] - v._data[idslice]
    if negative:
        ref._data[localslice] = -ref._data[localslice]
    ref.update_ghost_regions()

    # compute and compare

    # case one: dot(v, out=None)
    res1 = diffop.dot(v)
    assert np.allclose(ref._data[localslice], res1._data[localslice])

    # case two: dot(v, out=w)
    out = V1.coeff_space.zeros()
    res2 = diffop.dot(v, out=out)

    assert res2 is out
    assert np.allclose(ref._data[localslice], res2._data[localslice])
    
    # flag to skip matrix assembly if it takes too long or fails
    if matrix_assembly:
        # case three: tokronstencil().tostencil().dot(v)
        # (i.e. test matrix conversion)
        matrix = diffop.tokronstencil().tostencil()
        res3 = matrix.dot(v)
        assert np.allclose(ref._data[localslice], res3._data[localslice])

        # compare matrix assembly (in non-parallel case at least)
        if not diffop.domain.parallel:
            assert np.array_equal(diffop.toarray(), matrix.toarray())

    # case four: tosparse().dot(v._data)
    res4 = diffop.tosparse(with_pads=True).dot(v._data.flatten())
    assert np.allclose(ref._data[localslice], res4.reshape(ref._data.shape)[localslice])

def compare_diff_operators_by_matrixassembly(lo1, lo2):
    m1 = lo1.tokronstencil().tostencil()
    m2 = lo2.tokronstencil().tostencil()
    m1.update_ghost_regions()
    m2.update_ghost_regions()
    assert np.allclose(m1._data, m2._data)

def test_directional_derivative_operator_invalid_wrongsized1():
    # test if we detect incorrectly-sized spaces
    # i.e. V0.coeff_space.npts != V1.coeff_space.npts
    # (NOTE: if periodic was [True,True], this test would most likely pass)

    periodic = [False, False]
    domain = [(0,1),(0,1)]
    ncells = [8, 8]
    degree = [3, 3]
    direction = 0
    negative = False

    breaks = [np.linspace(*lims, num=n+1) for lims, n in zip(domain, ncells)]

    Ns = [SplineSpace(degree=d, grid=g, periodic=p, basis='B') \
                                  for d, g, p in zip(degree, breaks, periodic)]

    domain_decomposition = DomainDecomposition(ncells, periodic)
    # original space
    V0 = TensorFemSpace(domain_decomposition, *Ns)

    # reduced space
    V1 = V0.reduce_degree(axes=[1], basis='M')

    with pytest.raises(AssertionError):
        _ = DirectionalDerivativeOperator(V0.coeff_space, V1.coeff_space, direction, negative=negative)

def test_directional_derivative_operator_invalid_wrongspace2():
    # test, if it fails when the pads are not the same
    periodic = [False, False]
    domain = [(0,1),(0,1)]
    ncells = [8, 8]
    degree = [3, 3]
    direction = 0
    negative = False

    breaks = [np.linspace(*lims, num=n+1) for lims, n in zip(domain, ncells)]

    Ns = [SplineSpace(degree=d, grid=g, periodic=p, basis='B') \
                                  for d, g, p in zip(degree, breaks, periodic)]
    Ms = [SplineSpace(degree=d-1, grid=g, periodic=p, basis='B') \
                                  for d, g, p in zip(degree, breaks, periodic)]

    domain_decomposition = DomainDecomposition(ncells, periodic)
    # original space
    V0 = TensorFemSpace(domain_decomposition, *Ns)

    # reduced space
    V1 = TensorFemSpace(domain_decomposition, *Ms)

    with pytest.raises(AssertionError):
        _ = DirectionalDerivativeOperator(V0.coeff_space, V1.coeff_space, direction, negative=negative)

def test_directional_derivative_operator_transposition_correctness():
    # interface tests, to see if negation and transposition work as their methods suggest

    periodic = [False, False]
    domain = [(0,1),(0,1)]
    ncells = [8, 8]
    degree = [3, 3]
    direction = 0

    breaks = [np.linspace(*lims, num=n+1) for lims, n in zip(domain, ncells)]

    Ns = [SplineSpace(degree=d, grid=g, periodic=p, basis='B') \
                                  for d, g, p in zip(degree, breaks, periodic)]

    domain_decomposition = DomainDecomposition(ncells, periodic)
    # original space
    V0 = TensorFemSpace(domain_decomposition, *Ns)

    # reduced space
    V1 = V0.reduce_degree(axes=[0], basis='M')

    diff = DirectionalDerivativeOperator(V0.coeff_space, V1.coeff_space, direction, negative=False, transposed=False)

    # compare, if the transpose is actually correct
    M = diff.tokronstencil().tostencil()
    MT = diff.T.tokronstencil().tostencil()
    assert np.allclose(M.T._data, MT._data)
    assert np.allclose(M._data, MT.T._data)

    sparseM = diff.tosparse().tocoo()
    sparseMT = diff.T.tosparse().tocoo()

    sparseM_T = sparseM.T.tocoo()
    sparseMT_T = sparseMT.T.tocoo()

    assert np.array_equal( sparseMT.col , sparseM_T.col  )
    assert np.array_equal( sparseMT.row , sparseM_T.row  )
    assert np.array_equal( sparseMT.data, sparseM_T.data )
    assert np.array_equal( sparseM.col , sparseMT_T.col  )
    assert np.array_equal( sparseM.row , sparseMT_T.row  )
    assert np.array_equal( sparseM.data, sparseMT_T.data )

def test_directional_derivative_operator_interface():
    # interface tests, to see if negation and transposition work as their methods suggest

    periodic = [False, False]
    domain = [(0,1),(0,1)]
    ncells = [8, 8]
    degree = [3, 3]
    direction = 0

    breaks = [np.linspace(*lims, num=n+1) for lims, n in zip(domain, ncells)]

    Ns = [SplineSpace(degree=d, grid=g, periodic=p, basis='B') \
                                  for d, g, p in zip(degree, breaks, periodic)]

    domain_decomposition = DomainDecomposition(ncells, periodic)

    # original space
    V0 = TensorFemSpace(domain_decomposition, *Ns)

    # reduced space
    V1 = V0.reduce_degree(axes=[0], basis='M')

    diff = DirectionalDerivativeOperator(V0.coeff_space, V1.coeff_space, direction, negative=False, transposed=False)
    diffT = DirectionalDerivativeOperator(V0.coeff_space, V1.coeff_space, direction, negative=False, transposed=True)
    diffN = DirectionalDerivativeOperator(V0.coeff_space, V1.coeff_space, direction, negative=True, transposed=False)
    diffNT = DirectionalDerivativeOperator(V0.coeff_space, V1.coeff_space, direction, negative=True, transposed=True)

    # compare all with all by assembling matrices
    compare_diff_operators_by_matrixassembly(diff.T, diffT)
    compare_diff_operators_by_matrixassembly(-diff, diffN)
    compare_diff_operators_by_matrixassembly(-diff.T, diffNT)

    compare_diff_operators_by_matrixassembly(diffT.T, diff)
    compare_diff_operators_by_matrixassembly(-diffT, diffNT)
    compare_diff_operators_by_matrixassembly(-diffT.T, diffN)

    compare_diff_operators_by_matrixassembly(diffN.T, diffNT)
    compare_diff_operators_by_matrixassembly(-diffN, diff)
    compare_diff_operators_by_matrixassembly(-diffN.T, diffT)

    compare_diff_operators_by_matrixassembly(diffNT.T, diffN)
    compare_diff_operators_by_matrixassembly(-diffNT, diffT)
    compare_diff_operators_by_matrixassembly(-diffNT.T, diff)

@pytest.mark.parametrize('domain', [(0, 1), (-2, 3)])
@pytest.mark.parametrize('ncells', [11, 37])
@pytest.mark.parametrize('degree', [2, 3, 4, 5])
@pytest.mark.parametrize('periodic', [True, False])
@pytest.mark.parametrize('direction', [0])
@pytest.mark.parametrize('negative', [True, False])
@pytest.mark.parametrize('transposed', [True, False])
@pytest.mark.parametrize('seed', [1,3])
def test_directional_derivative_operator_1d_ser(domain, ncells, degree, periodic, direction, negative, transposed, seed):
    run_directional_derivative_operator(None, [domain], [ncells], [degree], [periodic], direction, negative, transposed, seed, True)

@pytest.mark.parametrize('domain', [([-2, 3], [6, 8])])
@pytest.mark.parametrize('ncells', [(10, 9), (27, 15)])
@pytest.mark.parametrize('degree', [(3, 2), (4, 5)])
@pytest.mark.parametrize('periodic', [(True, False), (False, True)])
@pytest.mark.parametrize('direction', [0,1])
@pytest.mark.parametrize('negative', [True, False])
@pytest.mark.parametrize('transposed', [True, False])
@pytest.mark.parametrize('seed', [1,3])
def test_directional_derivative_operator_2d_ser(domain, ncells, degree, periodic, direction, negative, transposed, seed):
    run_directional_derivative_operator(None, domain, ncells, degree, periodic, direction, negative, transposed, seed, True) 

@pytest.mark.parametrize('domain', [([-2, 3], [6, 8], [-0.5, 0.5])])
@pytest.mark.parametrize('ncells', [(4, 5, 7), (1, 1, 1)])
@pytest.mark.parametrize('degree', [(3, 2, 5), (2, 4, 7), (1, 1, 1)])
@pytest.mark.parametrize('periodic', [( True, False, False),
                                      (False,  True, False),
                                      (False, False,  True)])
@pytest.mark.parametrize('direction', [0,1,2])
@pytest.mark.parametrize('negative', [True, False])
@pytest.mark.parametrize('transposed', [True, False])
@pytest.mark.parametrize('seed', [1,3])
def test_directional_derivative_operator_3d_ser(domain, ncells, degree, periodic, direction, negative, transposed, seed):
    run_directional_derivative_operator(None, domain, ncells, degree, periodic, direction, negative, transposed, seed) 

@pytest.mark.parametrize('domain', [(0, 1), (-2, 3)])
@pytest.mark.parametrize('ncells', [29, 37])
@pytest.mark.parametrize('degree', [2, 3, 4, 5])
@pytest.mark.parametrize('periodic', [True, False])
@pytest.mark.parametrize('direction', [0])
@pytest.mark.parametrize('negative', [True, False])
@pytest.mark.parametrize('transposed', [True, False])
@pytest.mark.parametrize('seed', [1,3])
@pytest.mark.mpi
def test_directional_derivative_operator_1d_par(domain, ncells, degree, periodic, direction, negative, transposed, seed):
    # TODO: re-enable KroneckerStencilMatrix assembly here (fails right now sometimes when transposing)
    run_directional_derivative_operator(MPI.COMM_WORLD, [domain], [ncells], [degree], [periodic], direction, negative, transposed, seed)

@pytest.mark.parametrize('domain', [([-2, 3], [6, 8])])
@pytest.mark.parametrize('ncells', [(17, 25), (27, 39)])
@pytest.mark.parametrize('degree', [(3, 2), (4, 5)])
@pytest.mark.parametrize('periodic', [(True, False), (False, True)])
@pytest.mark.parametrize('direction', [0,1])
@pytest.mark.parametrize('negative', [True, False])
@pytest.mark.parametrize('transposed', [True, False])
@pytest.mark.parametrize('seed', [1,3])
@pytest.mark.mpi
def test_directional_derivative_operator_2d_par(domain, ncells, degree, periodic, direction, negative, transposed, seed):
    # TODO: re-enable KroneckerStencilMatrix assembly here (fails right now sometimes when transposing)
    run_directional_derivative_operator(MPI.COMM_WORLD, domain, ncells, degree, periodic, direction, negative, transposed, seed) 

@pytest.mark.parametrize('domain', [([-2, 3], [6, 8], [-0.5, 0.5])])
@pytest.mark.parametrize('ncells', [(10, 10, 13)])
@pytest.mark.parametrize('degree', [(2, 2, 3)])
@pytest.mark.parametrize('periodic', [( True, False, False),          
                                      (False,  True, False),
                                      (False, False,  True)])
@pytest.mark.parametrize('direction', [0,1,2])
@pytest.mark.parametrize('negative', [True, False])
@pytest.mark.parametrize('transposed', [True, False])
@pytest.mark.parametrize('seed', [3])
@pytest.mark.mpi
def test_directional_derivative_operator_3d_par(domain, ncells, degree, periodic, direction, negative, transposed, seed):
    run_directional_derivative_operator(MPI.COMM_WORLD, domain, ncells, degree, periodic, direction, negative, transposed, seed) 

# (higher dimensions are not tested here for now)

#==============================================================================
@pytest.mark.parametrize('domain', [(0, 1), (-2, 3)])
@pytest.mark.parametrize('ncells', [11, 37])
@pytest.mark.parametrize('degree', [2, 3, 4, 5])
@pytest.mark.parametrize('periodic', [True, False])
@pytest.mark.parametrize('seed', [1,3])
@pytest.mark.parametrize('multiplicity', [1,2])

def test_Derivative1D(domain, ncells, degree, periodic, seed, multiplicity):
    # determinize tests
    np.random.seed(seed)

    breaks = np.linspace(*domain, num=ncells+1)
    knots  = make_knots(breaks, degree, periodic, multiplicity=multiplicity)

    # H1 space (0-forms)
    N  = SplineSpace(degree=degree, knots=knots, periodic=periodic, basis='B')

    domain_decomposition = DomainDecomposition([ncells], [periodic])
    V0 = TensorFemSpace(domain_decomposition, N)

    # L2 space (1-forms)
    V1 = V0.reduce_degree(axes=[0], basis='M')

    # Create random field in V0
    u0 = FemField(V0)

    # Linear operator: 1D derivative
    grad = Derivative1D(V0, V1)

    # Create random field in V0
    s, = V0.coeff_space.starts
    e, = V0.coeff_space.ends

    u0.coeffs[s:e+1] = np.random.random(e-s+1)

    # Compute gradient (=derivative) of u0
    u1 = grad(u0)

    # Create evaluation grid, and check if ∂/∂x u0(x) == u1(x)
    xgrid = np.linspace(*N.domain, num=11)
    vals_grad_u0 = np.array([u0.gradient(x)[0] for x in xgrid])
    vals_u1 = np.array([u1(x) for x in xgrid])

    # Test if relative max-norm of error is <= TOL
    maxnorm_field = abs(vals_u1).max()
    maxnorm_error = abs(vals_u1 - vals_grad_u0).max()
    assert maxnorm_error / maxnorm_field <= 1e-14

#==============================================================================
@pytest.mark.parametrize('domain', [([-2, 3], [6, 8])])              # 1 case
@pytest.mark.parametrize('ncells', [(10, 9), (27, 15)])              # 2 cases
@pytest.mark.parametrize('degree', [(3, 2), (4, 5)])                 # 2 cases
@pytest.mark.parametrize('periodic', [(True, False), (False, True)]) # 2 cases
@pytest.mark.parametrize('seed', [1,3])
@pytest.mark.parametrize('multiplicity', [(1, 1), (1, 2), (2, 2)])

def test_Gradient2D(domain, ncells, degree, periodic, seed, multiplicity):
    # determinize tests
    np.random.seed(seed)

    # Compute breakpoints along each direction
    breaks = [np.linspace(*lims, num=n+1) for lims, n in zip(domain, ncells)]

    # H1 space (0-forms)
    Nx, Ny = [SplineSpace(degree=d, grid=g, periodic=p, basis='B', multiplicity=m) \
                                  for d, g, p, m in zip(degree, breaks, periodic, multiplicity)]

    domain_decomposition = DomainDecomposition(ncells, periodic)
    V0 = TensorFemSpace(domain_decomposition, Nx, Ny)

    # H-curl space (1-forms)
    DxNy = V0.reduce_degree(axes=[0], basis='M')
    NxDy = V0.reduce_degree(axes=[1], basis='M')
    V1 = VectorFemSpace(DxNy, NxDy)

    # Linear operator: 2D gradient
    grad = Gradient2D(V0, V1)

    # Create random field in V0
    u0 = FemField(V0)

    s1, s2 = V0.coeff_space.starts
    e1, e2 = V0.coeff_space.ends

    u0.coeffs[s1:e1+1, s2:e2+1] = np.random.random((e1-s1+1, e2-s2+1))

    # Compute gradient of u0
    u1 = grad(u0)

    # x and y components of u1 vector field
    u1x = u1.fields[0]
    u1y = u1.fields[1]

    # Create evaluation grid, and check if
    # ∂/∂x u0(x, y) == u1x(x, y)
    # ∂/∂y u0(x, y) == u1y(x, y)

    xgrid = np.linspace(*domain[0], num=11)
    ygrid = np.linspace(*domain[1], num=11)

    vals_grad_u0 = np.array([[u0.gradient(x, y) for x in xgrid] for y in ygrid])
    vals_u1 = np.array([[[u1x(x, y), u1y(x, y)] for x in xgrid] for y in ygrid])

    # Test if relative max-norm of error is <= TOL
    maxnorm_field = abs(vals_u1).max()
    maxnorm_error = abs(vals_u1 - vals_grad_u0).max()
    assert maxnorm_error / maxnorm_field <= 1e-14

#==============================================================================
@pytest.mark.parametrize('domain', [([-2, 3], [6, 8], [-0.5, 0.5])])             # 1 case
@pytest.mark.parametrize('ncells', [(1, 8, 3), (7, 1, 2), (2, 2, 1), (4, 5, 7)]) # 4 cases
@pytest.mark.parametrize('degree', [(1, 3, 1), (3, 1, 5), (2, 4, 7)])            # 3 cases
@pytest.mark.parametrize('periodic', [( True, False, False),                     # 3 cases
                                      (False,  True, False),
                                      (False, False,  True)])
@pytest.mark.parametrize('seed', [1,3])
@pytest.mark.parametrize('multiplicity', [(1, 1, 1), (1, 2, 2), (2, 2, 2)])

def test_Gradient3D(domain, ncells, degree, periodic, seed, multiplicity):
    if any([ncells[d] <= degree[d] and periodic[d] for d in range(3)]):
        return
    
    # determinize tests
    np.random.seed(seed)

    # Compute breakpoints along each direction
    breaks = [np.linspace(*lims, num=n+1) for lims, n in zip(domain, ncells)]

    #change multiplicity if higher than degree to avoid problems (case p<m doesn't work)
    multiplicity = [min(m,p) for p, m in zip (degree, multiplicity)]
    # H1 space (0-forms)
    Nx, Ny, Nz = [SplineSpace(degree=d, grid=g, periodic=p, basis='B', multiplicity=m) \
                                  for d, g, p, m in zip(degree, breaks, periodic, multiplicity)]

    domain_decomposition = DomainDecomposition(ncells, periodic)
    V0 = TensorFemSpace(domain_decomposition, Nx, Ny, Nz)

    # H-curl space (1-forms)
    DxNyNz = V0.reduce_degree(axes=[0], basis='M')
    NxDyNz = V0.reduce_degree(axes=[1], basis='M')
    NxNyDz = V0.reduce_degree(axes=[2], basis='M')
    V1 = VectorFemSpace(DxNyNz, NxDyNz, NxNyDz)

    # Linear operator: 3D gradient
    grad = Gradient3D(V0, V1)

    # Create random field in V0
    u0 = FemField(V0)
    s1, s2, s3 = V0.coeff_space.starts
    e1, e2, e3 = V0.coeff_space.ends
    u0.coeffs[s1:e1+1, s2:e2+1, s3:e3+1] = np.random.random((e1-s1+1, e2-s2+1, e3-s3+1))

    # Compute gradient of u0
    u1 = grad(u0)

    # Components of u1 vector field
    u1x, u1y, u1z = u1.fields

    # Create random evaluation points (x, y, z) for evaluating fields
    npts = 1000
    xyz_pts = [[lims[0]+s*(lims[1]-lims[0]) for s, lims in zip(np.random.random(3), domain)]
        for i in range(npts)]

    # Check if
    #   ∂/∂x u0(x, y, z) == u1x(x, y, z)
    #   ∂/∂y u0(x, y, z) == u1y(x, y, z)
    #   ∂/∂z u0(x, y, z) == u1z(x, y, z)

    vals_grad_u0 = np.array([u0.gradient(*xyz) for xyz in xyz_pts])
    vals_u1 = np.array([[u1x(*xyz), u1y(*xyz), u1z(*xyz)] for xyz in xyz_pts])

    # Test if relative max-norm of error is <= TOL
    maxnorm_field = abs(vals_u1).max()
    maxnorm_error = abs(vals_u1 - vals_grad_u0).max()
    assert maxnorm_error / maxnorm_field <= 1e-14

#==============================================================================
@pytest.mark.parametrize('domain', [([-2, 3], [6, 8])])              # 1 case
@pytest.mark.parametrize('ncells', [(10, 9), (27, 15)])              # 2 cases
@pytest.mark.parametrize('degree', [(3, 2), (4, 5)])                 # 2 cases
@pytest.mark.parametrize('periodic', [(True, False), (False, True)]) # 2 cases
@pytest.mark.parametrize('seed', [1,3])
@pytest.mark.parametrize('multiplicity', [(1, 1), (1, 2), (2, 2)])


def test_ScalarCurl2D(domain, ncells, degree, periodic, seed, multiplicity):
    # determinize tests
    np.random.seed(seed)

    # Compute breakpoints along each direction
    breaks = [np.linspace(*lims, num=n+1) for lims, n in zip(domain, ncells)]
    #change mulitplicity if higher than degree to avoid problems (case p<m doesn't work)
    multiplicity = [min(m,p) for p, m in zip (degree, multiplicity)]
    # H1 space (0-forms)
    Nx, Ny = [SplineSpace(degree=d, grid=g, periodic=p, basis='B', multiplicity=m) \
                                  for d, g, p, m in zip(degree, breaks, periodic, multiplicity)]

    domain_decomposition = DomainDecomposition(ncells, periodic)
    V0 = TensorFemSpace(domain_decomposition, Nx, Ny)

    # H-curl space (1-forms)
    DxNy = V0.reduce_degree(axes=[0], basis='M')
    NxDy = V0.reduce_degree(axes=[1], basis='M')
    V1 = VectorFemSpace(DxNy, NxDy)

    # L2 space (2-forms)
    DxDy = V0.reduce_degree(axes=[0, 1], basis='M')
    V2 = DxDy

    # Linear operator: curl
    curl = ScalarCurl2D(V1, V2)

    # ...
    # Create random field in V1
    u1 = FemField(V1)

    s1, s2 = V1.coeff_space[0].starts
    e1, e2 = V1.coeff_space[0].ends
    u1.coeffs[0][s1:e1+1, s2:e2+1] = np.random.random((e1-s1+1, e2-s2+1))

    s1, s2 = V1.coeff_space[1].starts
    e1, e2 = V1.coeff_space[1].ends
    u1.coeffs[1][s1:e1+1, s2:e2+1] = np.random.random((e1-s1+1, e2-s2+1))
    # ...

    # Compute curl of u1
    u2 = curl(u1)

    # Components of vector field u1
    u1x, u1y = u1.fields

    # Create random evaluation points (x, y, z) for evaluating fields
    npts = 1000
    xyz_pts = [[lims[0]+s*(lims[1]-lims[0]) for s, lims in zip(np.random.random(2), domain)]
        for i in range(npts)]

    # Check if
    #   ∂/∂y u1x(x, y) - ∂/∂x u1y(x, y) == u2(x, y)

    def eval_curl(fx, fy, *eta):
        dfx_dx, dfx_dy = fx.gradient(*eta)
        dfy_dx, dfy_dy = fy.gradient(*eta)
        return dfy_dx - dfx_dy

    vals_curl_u1 = np.array([eval_curl(u1x, u1y, *xyz) for xyz in xyz_pts])
    vals_u2 = np.array([u2(*xyz) for xyz in xyz_pts])

    # Test if relative max-norm of error is <= TOL
    maxnorm_field = abs(vals_u2).max()
    maxnorm_error = abs(vals_u2 - vals_curl_u1).max()
    assert maxnorm_error / maxnorm_field <= 1e-14

#==============================================================================
@pytest.mark.parametrize('domain', [([-2, 3], [6, 8])])              # 1 case
@pytest.mark.parametrize('ncells', [(10, 9), (27, 15)])              # 2 cases
@pytest.mark.parametrize('degree', [(3, 2), (4, 5)])                 # 2 cases
@pytest.mark.parametrize('periodic', [(True, False), (False, True)]) # 2 cases
@pytest.mark.parametrize('seed', [1,3])
@pytest.mark.parametrize('multiplicity', [(1, 1), (1, 2), (2, 2)])

def test_VectorCurl2D(domain, ncells, degree, periodic, seed, multiplicity):
    # determinize tests
    np.random.seed(seed)

    # Compute breakpoints along each direction
    breaks = [np.linspace(*lims, num=n+1) for lims, n in zip(domain, ncells)]
    #change mulitplicity if higher than degree to avoid problems (case p<m doesn't work)
    multiplicity = [min(m,p) for p, m in zip (degree, multiplicity)]
    # H1 space (0-forms)
    Nx, Ny = [SplineSpace(degree=d, grid=g, periodic=p, basis='B', multiplicity=m) \
                                  for d, g, p, m in zip(degree, breaks, periodic, multiplicity)]
    domain_decomposition = DomainDecomposition(ncells, periodic)
    V0 = TensorFemSpace(domain_decomposition, Nx, Ny)

    # Hdiv space (1-forms)
    NxDy = V0.reduce_degree(axes=[1], basis='M')
    DxNy = V0.reduce_degree(axes=[0], basis='M')
    V1 = VectorFemSpace(NxDy, DxNy)

    # Linear operator: 2D vector curl
    curl = VectorCurl2D(V0, V1)

    # Create random field in V0
    u0 = FemField(V0)

    s1, s2 = V0.coeff_space.starts
    e1, e2 = V0.coeff_space.ends

    u0.coeffs[s1:e1+1, s2:e2+1] = np.random.random((e1-s1+1, e2-s2+1))

    # Compute curl of u0
    u1 = curl(u0)

    # x and y components of u1 vector field
    u1x = u1.fields[0]
    u1y = u1.fields[1]

    # Create evaluation grid, and check if
    #  ∂/∂y u0(x, y) == u1x(x, y)
    # -∂/∂x u0(x, y) == u1y(x, y)

    def eval_curl(f, *eta):
        df_dx, df_dy = f.gradient(*eta)
        return [df_dy, -df_dx]

    xgrid = np.linspace(*domain[0], num=11)
    ygrid = np.linspace(*domain[1], num=11)

    vals_curl_u0 = np.array([[eval_curl(u0, x, y) for x in xgrid] for y in ygrid])
    vals_u1 = np.array([[[u1x(x, y), u1y(x, y)] for x in xgrid] for y in ygrid])

    # Test if relative max-norm of error is <= TOL
    maxnorm_field = abs(vals_u1).max()
    maxnorm_error = abs(vals_u1 - vals_curl_u0).max()
    assert maxnorm_error / maxnorm_field <= 1e-14

#==============================================================================
@pytest.mark.parametrize('domain', [([-2, 3], [6, 8], [-0.5, 0.5])])             # 1 case
@pytest.mark.parametrize('ncells', [(1, 8, 3), (5, 1, 2), (2, 2, 1), (4, 5, 7)]) # 3 cases
@pytest.mark.parametrize('degree', [(1, 3, 5), (3, 1, 2), (2, 4, 7)])            # 3 cases
@pytest.mark.parametrize('periodic', [( True, False, False),                     # 3 cases
                                      (False,  True, False),
                                      (False, False, True)])
@pytest.mark.parametrize('seed', [1,3])
@pytest.mark.parametrize('multiplicity', [(1, 1, 1), (1, 2, 2), (2, 2, 2)])

def test_Curl3D(domain, ncells, degree, periodic, seed, multiplicity):
    if any([ncells[d] <= degree[d] and periodic[d] for d in range(3)]):
        return

    # determinize tests
    np.random.seed(seed)

    # Compute breakpoints along each direction
    breaks = [np.linspace(*lims, num=n+1) for lims, n in zip(domain, ncells)]

    #change mulitplicity if higher than degree to avoid problems (case p<m doesn't work)
    multiplicity = [min(m,p) for p, m in zip (degree, multiplicity)]
    # H1 space (0-forms)
    Nx, Ny, Nz = [SplineSpace(degree=d, grid=g, periodic=p, basis='B', multiplicity=m) \
                                  for d, g, p, m in zip(degree, breaks, periodic, multiplicity)]

    domain_decomposition = DomainDecomposition(ncells, periodic)
    V0 = TensorFemSpace(domain_decomposition, Nx, Ny, Nz)

    # H-curl space (1-forms)
    DxNyNz = V0.reduce_degree(axes=[0], basis='M')
    NxDyNz = V0.reduce_degree(axes=[1], basis='M')
    NxNyDz = V0.reduce_degree(axes=[2], basis='M')
    V1 = VectorFemSpace(DxNyNz, NxDyNz, NxNyDz)

    # H-div space (2-forms)
    NxDyDz = V0.reduce_degree(axes=[1, 2], basis='M')
    DxNyDz = V0.reduce_degree(axes=[2, 0], basis='M')
    DxDyNz = V0.reduce_degree(axes=[0, 1], basis='M')
    V2 = VectorFemSpace(NxDyDz, DxNyDz, DxDyNz)

    # Linear operator: curl
    curl = Curl3D(V1, V2)

    # ...
    # Create random field in V1
    u1 = FemField(V1)

    s1, s2, s3 = V1.coeff_space[0].starts
    e1, e2, e3 = V1.coeff_space[0].ends
    u1.coeffs[0][s1:e1+1, s2:e2+1, s3:e3+1] = np.random.random((e1-s1+1, e2-s2+1, e3-s3+1))

    s1, s2, s3 = V1.coeff_space[1].starts
    e1, e2, e3 = V1.coeff_space[1].ends
    u1.coeffs[1][s1:e1+1, s2:e2+1, s3:e3+1] = np.random.random((e1-s1+1, e2-s2+1, e3-s3+1))

    s1, s2, s3 = V1.coeff_space[2].starts
    e1, e2, e3 = V1.coeff_space[2].ends
    u1.coeffs[2][s1:e1+1, s2:e2+1, s3:e3+1] = np.random.random((e1-s1+1, e2-s2+1, e3-s3+1))
    # ...

    # Compute curl of u1
    u2 = curl(u1)

    # Components of vector fields u1 and u2
    u1x, u1y, u1z = u1.fields
    u2x, u2y, u2z = u2.fields

    # Create random evaluation points (x, y, z) for evaluating fields
    npts = 1000
    xyz_pts = [[lims[0]+s*(lims[1]-lims[0]) for s, lims in zip(np.random.random(3), domain)]
        for i in range(npts)]

    # Check if
    #   ∂/∂y u1z(x, y, z) - ∂/∂z u1y(x, y, z) == u2x(x, y, z)
    #   ∂/∂z u1x(x, y, z) - ∂/∂x u1z(x, y, z) == u2y(x, y, z)
    #   ∂/∂x u1y(x, y, z) - ∂/∂y u1x(x, y, z) == u2z(x, y, z)

    def eval_curl(fx, fy, fz, *eta):
        dfx_dx, dfx_dy, dfx_dz = fx.gradient(*eta)
        dfy_dx, dfy_dy, dfy_dz = fy.gradient(*eta)
        dfz_dx, dfz_dy, dfz_dz = fz.gradient(*eta)
        return [dfz_dy - dfy_dz,
                dfx_dz - dfz_dx,
                dfy_dx - dfx_dy]

    vals_curl_u1 = np.array([eval_curl(u1x, u1y, u1z, *xyz) for xyz in xyz_pts])
    vals_u2 = np.array([[u2x(*xyz), u2y(*xyz), u2z(*xyz)] for xyz in xyz_pts])

    # Test if relative max-norm of error is <= TOL
    maxnorm_field = abs(vals_u2).max()
    maxnorm_error = abs(vals_u2 - vals_curl_u1).max()
    assert maxnorm_error / maxnorm_field <= 1e-14

#==============================================================================
@pytest.mark.parametrize('domain', [([-2, 3], [6, 8])])              # 1 case
@pytest.mark.parametrize('ncells', [(10, 9), (27, 15)])              # 2 cases
@pytest.mark.parametrize('degree', [(3, 2), (4, 5)])                 # 2 cases
@pytest.mark.parametrize('periodic', [(True, False), (False, True)]) # 2 cases
@pytest.mark.parametrize('seed', [1,3])
@pytest.mark.parametrize('multiplicity', [(1, 1), (1, 2), (2, 2)])

def test_Divergence2D(domain, ncells, degree, periodic, seed, multiplicity):
    # determinize tests
    np.random.seed(seed)

    # Compute breakpoints along each direction
    breaks = [np.linspace(*lims, num=n+1) for lims, n in zip(domain, ncells)]
    #change mulitplicity if higher than degree to avoid problems (case p<m doesn't work)
    multiplicity = [min(m,p) for p, m in zip (degree, multiplicity)]
    # H1 space (0-forms)
    Nx, Ny = [SplineSpace(degree=d, grid=g, periodic=p, basis='B', multiplicity=m) \
                                  for d, g, p, m in zip(degree, breaks, periodic, multiplicity)]

    domain_decomposition = DomainDecomposition(ncells, periodic)
    V0 = TensorFemSpace(domain_decomposition, Nx, Ny)

    # H-div space (1-forms)
    NxDy = V0.reduce_degree(axes=[1], basis='M')
    DxNy = V0.reduce_degree(axes=[0], basis='M')
    V1 = VectorFemSpace(NxDy, DxNy)

    # L2 space (2-forms)
    V2 = V0.reduce_degree(axes=[0, 1], basis='M')

    # Linear operator: divergence
    div = Divergence2D(V1, V2)

    # ...
    # Create random field in V1
    u1 = FemField(V1)

    s1, s2 = V1.coeff_space[0].starts
    e1, e2 = V1.coeff_space[0].ends
    u1.coeffs[0][s1:e1+1, s2:e2+1] = np.random.random((e1-s1+1, e2-s2+1))

    s1, s2 = V1.coeff_space[1].starts
    e1, e2 = V1.coeff_space[1].ends
    u1.coeffs[1][s1:e1+1, s2:e2+1] = np.random.random((e1-s1+1, e2-s2+1))
    # ...

    # Compute divergence of u1
    u2 = div(u1)

    # Components of vector field u1
    u1x, u1y = u1.fields

    # Create random evaluation points (x, y, z) for evaluating fields
    npts = 1000
    xyz_pts = [[lims[0]+s*(lims[1]-lims[0]) for s, lims in zip(np.random.random(3), domain)]
        for i in range(npts)]

    # Check if
    #   ∂/∂x u1x(x, y) + ∂/∂y u1y(x, y) == u2(x, y)

    def eval_div(fx, fy, *eta):
        dfx_dx, dfx_dy = fx.gradient(*eta)
        dfy_dx, dfy_dy = fy.gradient(*eta)
        return dfx_dx + dfy_dy

    vals_div_u1 = np.array([eval_div(u1x, u1y, *xyz) for xyz in xyz_pts])
    vals_u2 = np.array([u2(*xyz) for xyz in xyz_pts])

    # Test if relative max-norm of error is <= TOL
    maxnorm_field = abs(vals_u2).max()
    maxnorm_error = abs(vals_u2 - vals_div_u1).max()
    assert maxnorm_error / maxnorm_field <= 1e-14

#==============================================================================
@pytest.mark.parametrize('domain', [([-2, 3], [6, 8], [-0.5, 0.5])])  # 1 case
@pytest.mark.parametrize('ncells', [(4, 5, 7)])                       # 1 case
@pytest.mark.parametrize('degree', [(3, 2, 5), (2, 4, 7)])            # 2 cases
@pytest.mark.parametrize('periodic', [( True, False, False),          # 3 cases
                                      (False,  True, False),
                                      (False, False,  True)])
@pytest.mark.parametrize('seed', [1,3])
@pytest.mark.parametrize('multiplicity', [(1, 1, 1), (1, 2, 2), (2, 2, 2)])

def test_Divergence3D(domain, ncells, degree, periodic, seed, multiplicity):
    # determinize tests
    np.random.seed(seed)

    # Compute breakpoints along each direction
    breaks = [np.linspace(*lims, num=n+1) for lims, n in zip(domain, ncells)]

    #change mulitplicity if higher than degree to avoid problems (case p<m doesn't work)
    multiplicity = [min(m,p) for p, m in zip (degree, multiplicity)]
    # H1 space (0-forms)
    Nx, Ny, Nz = [SplineSpace(degree=d, grid=g, periodic=p, basis='B', multiplicity=m) \
                                  for d, g, p, m in zip(degree, breaks, periodic, multiplicity)]

    domain_decomposition = DomainDecomposition(ncells, periodic)
    V0 = TensorFemSpace(domain_decomposition, Nx, Ny, Nz)

    # H-div space (2-forms)
    NxDyDz = V0.reduce_degree(axes=[1, 2], basis='M')
    DxNyDz = V0.reduce_degree(axes=[2, 0], basis='M')
    DxDyNz = V0.reduce_degree(axes=[0, 1], basis='M')
    V2 = VectorFemSpace(NxDyDz, DxNyDz, DxDyNz)

    # L2 space (3-forms)
    V3 = V0.reduce_degree(axes=[0, 1, 2], basis='M')

    # Linear operator: divergence
    div = Divergence3D(V2, V3)

    # ...
    # Create random field in V2
    u2 = FemField(V2)

    s1, s2, s3 = V2.coeff_space[0].starts
    e1, e2, e3 = V2.coeff_space[0].ends
    u2.coeffs[0][s1:e1+1, s2:e2+1, s3:e3+1] = np.random.random((e1-s1+1, e2-s2+1, e3-s3+1))

    s1, s2, s3 = V2.coeff_space[1].starts
    e1, e2, e3 = V2.coeff_space[1].ends
    u2.coeffs[1][s1:e1+1, s2:e2+1, s3:e3+1] = np.random.random((e1-s1+1, e2-s2+1, e3-s3+1))

    s1, s2, s3 = V2.coeff_space[2].starts
    e1, e2, e3 = V2.coeff_space[2].ends
    u2.coeffs[2][s1:e1+1, s2:e2+1, s3:e3+1] = np.random.random((e1-s1+1, e2-s2+1, e3-s3+1))
    # ...

    # Compute divergence of u2
    u3 = div(u2)

    # Components of vector field u2
    u2x, u2y, u2z = u2.fields

    # Create random evaluation points (x, y, z) for evaluating fields
    npts = 1000
    xyz_pts = [[lims[0]+s*(lims[1]-lims[0]) for s, lims in zip(np.random.random(3), domain)]
        for i in range(npts)]

    # Check if
    #   ∂/∂x u2x(x, y, z) + ∂/∂y u2y(x, y, z) + ∂/∂z u2z(x, y, z) == u3(x, y, z)

    def eval_div(fx, fy, fz, *eta):
        dfx_dx, dfx_dy, dfx_dz = fx.gradient(*eta)
        dfy_dx, dfy_dy, dfy_dz = fy.gradient(*eta)
        dfz_dx, dfz_dy, dfz_dz = fz.gradient(*eta)
        return dfx_dx + dfy_dy + dfz_dz

    vals_div_u2 = np.array([eval_div(u2x, u2y, u2z, *xyz) for xyz in xyz_pts])
    vals_u3 = np.array([u3(*xyz) for xyz in xyz_pts])

    # Test if relative max-norm of error is <= TOL
    maxnorm_field = abs(vals_u3).max()
    maxnorm_error = abs(vals_u3 - vals_div_u2).max()
    assert maxnorm_error / maxnorm_field <= 1e-14

#==============================================================================
if __name__ == '__main__':
    
    test_Derivative1D(domain=[0, 1], ncells=3, degree=3, periodic=False, seed=1, multiplicity=1)
    test_Derivative1D(domain=[0, 1], ncells=12, degree=3, periodic=True, seed=1, multiplicity=1)


    test_Gradient2D(
        domain   = ([0, 1], [0, 1]),
        ncells   = (10, 15),
        degree   = (3, 2),
        periodic = (False, True),
        seed     = 1
    )

    test_Gradient3D(
        domain   = ([0, 1], [0, 1], [0, 1]),
        ncells   = (5, 8, 4),
        degree   = (3, 2, 3),
        periodic = (False, True, True),
        seed     = 1
    )

    test_ScalarCurl2D(
        domain   = ([0, 1], [0, 1]),
        ncells   = (10, 15),
        degree   = (3, 2),
        periodic = (False, True),
        seed     = 1
    )

    test_VectorCurl2D(
        domain   = ([0, 1], [0, 1]),
        ncells   = (10, 15),
        degree   = (3, 2),
        periodic = (False, True),
        seed     = 1
    )

    test_Curl3D(
        domain   = ([0, 1], [0, 1], [0, 1]),
        ncells   = (5, 8, 4),
        degree   = (3, 2, 3),
        periodic = (False, True, True),
        seed     = 1
    )

    test_Divergence2D(
        domain   = ([0, 1], [0, 1]),
        ncells   = (10, 15),
        degree   = (3, 2),
        periodic = (False, True),
        seed     = 1
    )

    test_Divergence3D(
        domain   = ([0, 1], [0, 1], [0, 1]),
        ncells   = (5, 8, 4),
        degree   = (3, 2, 3),
        periodic = (False, True, True),
        seed     = 1
    )
