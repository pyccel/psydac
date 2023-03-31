
import numpy as np
import pytest
from psydac.linalg.solvers import inverse
from psydac.linalg.stencil import StencilVectorSpace, StencilMatrix, StencilVector
from psydac.linalg.basic import LinearSolver
from psydac.ddm.cart import DomainDecomposition, CartDecomposition


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

def define_data(n, p, matrix_data, dtype):
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

#===============================================================================
@pytest.mark.parametrize( 'n', [5, 10, 13] )
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('dtype', [float,complex])
def test_bicgstab_tridiagonal(n, p, dtype, verbose=False):

    #---------------------------------------------------------------------------
    # PARAMETERS
    #---------------------------------------------------------------------------
    if dtype==complex:
        V, A, xe = define_data(n, p, [1-10j,6+9j,3+5j], dtype=dtype)
    else:
        V, A, xe = define_data(n, p, [1,6,3], dtype=dtype)
    # Tolerance for success: L2-norm of error in solution
    tol = 1e-10

    #---------------------------------------------------------------------------
    # TEST
    #---------------------------------------------------------------------------
    if verbose:
        # Title
        print()
        print( "="*80 )
        print( "SERIAL TEST: solve linear system A*x = b using biconjugate gradient" )
        print( "="*80 )
        print()

    # Manufacture right-hand-side vector from exact solution
    b = A.dot( xe )
    bt = (A.T).dot( xe )
    bh = (A.H).dot( xe )

    #Create the solvers
    solv = inverse(A, 'bicgstab', tol=1e-13, verbose=True)
    solvt = solv.transpose()
    solvh = solv.H()

    # Solve linear system using BiCGSTAB
    x = solv @ b
    xt = solvt.solve(bt)
    xh = solvh.dot(bh)
    info = solv.get_info()

    # Verify correctness of calculation: L2-norm of error
    err = x-xe
    err_norm = np.linalg.norm( err.toarray() )

    errt = xt-xe
    errt_norm = np.linalg.norm( errt.toarray() )

    errh = xh-xe
    errh_norm = np.linalg.norm( errh.toarray() )

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
        print( "L2-norm of error in solution = {:.2e}".format( err_norm ) )
        if err_norm < tol:
            print( "PASSED" )
        else:
            print( "FAIL" )
        print( "-"*40 )

    #---------------------------------------------------------------------------
    # PYTEST
    #---------------------------------------------------------------------------
    assert err_norm < tol
    assert errt_norm < tol
    assert errh_norm < tol

#===============================================================================
@pytest.mark.parametrize( 'n', [5, 10, 13] )
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('dtype', [float,complex])
def test_bicg_tridiagonal(n, p, dtype, verbose=False):

    #---------------------------------------------------------------------------
    # PARAMETERS
    #---------------------------------------------------------------------------
    if dtype==complex:
        V, A, xe = define_data(n, p, [1-10j,6+9j,3+5j], dtype=dtype)
    else:
        V, A, xe = define_data(n, p, [1,6,3], dtype=dtype)
    # Tolerance for success: L2-norm of error in solution
    tol = 1e-10

    #---------------------------------------------------------------------------
    # TEST
    #---------------------------------------------------------------------------
    if verbose:
        # Title
        print()
        print( "="*80 )
        print( "SERIAL TEST: solve linear system A*x = b using biconjugate gradient" )
        print( "="*80 )
        print()

    # Manufacture right-hand-side vector from exact solution
    b  = A.dot( xe )
    bt = A.T.dot( xe )
    bh = A.H.dot( xe )

    #Create the solvers
    solv  = inverse(A, 'bicg', tol=1e-13, verbose=True)
    solvt = solv.transpose()
    solvh = solv.H()

    # Solve linear system using BiCG
    x = solv @ b
    xt = solvt.solve(bt)
    xh = solvh.dot(bh)
    info = solv.get_info()

    # Verify correctness of calculation: L2-norm of error
    err = x-xe
    err_norm = np.linalg.norm( err.toarray() )
    errt = xt-xe
    errt_norm = np.linalg.norm( errt.toarray() )
    errh = xh-xe
    errh_norm = np.linalg.norm( errh.toarray() )

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
        print( "L2-norm of error in solution = {:.2e}".format( err_norm ) )
        if err_norm < tol:
            print( "PASSED" )
        else:
            print( "FAIL" )
        print( "-"*40 )

    #---------------------------------------------------------------------------
    # PYTEST
    #---------------------------------------------------------------------------
    assert err_norm < tol
    assert errt_norm < tol
    assert errh_norm < tol

#===============================================================================
@pytest.mark.parametrize( 'n', [5, 10, 13] )
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('dtype', [float, complex])
def test_lsmr_tridiagonal(n, p, dtype, verbose=False):

    #---------------------------------------------------------------------------
    # PARAMETERS
    #---------------------------------------------------------------------------

    if dtype==complex:
        V, A, xe = define_data(n, p, [1+2j,6+2j,3+2j],dtype=dtype)
    else:
        V, A, xe = define_data(n, p, [1,6,3],dtype=dtype)

    # Tolerance for success: L2-norm of error in solution
    tol = 1e-10

    #---------------------------------------------------------------------------
    # TEST
    #---------------------------------------------------------------------------
    if verbose:
        # Title
        print()
        print( "="*80 )
        print( "SERIAL TEST: solve linear system A*x = b using lsmr" )
        print( "="*80 )
        print()

    # Manufacture right-hand-side vector from exact solution
    b  = A.dot( xe )
    bt = A.T.dot( xe )
    bh = A.H.dot( xe )

    #Create the solvers
    solv  = inverse(A, 'lsmr', tol=1e-13, verbose=True)
    solvt = solv.transpose()
    solvh = solv.H()

    # Solve linear system using lsmr
    x = solv @ b
    info = solv.get_info()
    xt = solvt.solve(bt)
    xh = solvh.dot(bh)

    # Verify correctness of calculation: L2-norm of error
    res = A.dot(x)-b
    res_norm = np.linalg.norm( res.toarray() )
    rest = A.T.dot(xt)-bt
    rest_norm = np.linalg.norm( rest.toarray() )
    resh = A.H.dot(xh)-bh
    resh_norm = np.linalg.norm( resh.toarray() )

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
        print( "L2-norm of error in solution = {:.2e}".format( res_norm ) )
        if res_norm < tol:
            print( "PASSED" )
        else:
            print( "FAIL" )
        print( "-"*40 )

    #---------------------------------------------------------------------------
    # PYTEST
    #---------------------------------------------------------------------------
    assert res_norm < tol
    assert rest_norm < tol
    assert resh_norm < tol

#===============================================================================
@pytest.mark.parametrize( 'n', [5, 10, 13] )
@pytest.mark.parametrize('p', [2, 3])
def test_minres_tridiagonal(n, p, verbose=False):

    #---------------------------------------------------------------------------
    # PARAMETERS
    #---------------------------------------------------------------------------

    V, A, xe = define_data_hermitian(n, p)

    # Tolerance for success: L2-norm of error in solution
    tol = 1e-12

    #---------------------------------------------------------------------------
    # TEST
    #---------------------------------------------------------------------------
    if verbose:
        # Title
        print()
        print( "="*80 )
        print( "SERIAL TEST: solve linear system A*x = b using minres" )
        print( "="*80 )
        print()

    # Manufacture right-hand-side vector from exact solution
    b = A.dot( xe )
    bt = A.T.dot( xe )
    bh = A.H.dot( xe )


    #Create the solvers
    solv  = inverse(A, 'minres', tol=1e-13, verbose=True)
    solvt = solv.transpose()
    solvh = solv.H()

    # Solve linear system using minres
    x = solv @ b
    info = solv.get_info()
    xt = solvt.solve(bt)
    xh = solvh.dot(b)

    # Verify correctness of calculation: L2-norm of error
    res = A.dot(x)-b
    res_norm = np.linalg.norm( res.toarray() )
    rest = A.T.dot(xt)-bt
    rest_norm = np.linalg.norm( rest.toarray() )
    resh = A.H.dot(x)-bh
    resh_norm = np.linalg.norm( resh.toarray() )

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
        print( "L2-norm of error in solution = {:.2e}".format( res_norm ) )
        if res_norm < tol:
            print( "PASSED" )
        else:
            print( "FAIL" )
        print( "-"*40 )

    #---------------------------------------------------------------------------
    # PYTEST
    #---------------------------------------------------------------------------
    assert res_norm < tol
    assert rest_norm < tol
    assert resh_norm < tol

# ===============================================================================
@pytest.mark.parametrize('n', [8, 16])
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('dtype', [float, complex])
def test_pcg_tridiagonal(n, p, dtype, verbose=False):
    # ---------------------------------------------------------------------------
    # PARAMETERS
    # ---------------------------------------------------------------------------

    V, A, xe = define_data_hermitian(n, p, dtype=dtype)

    # Tolerance for success: L2-norm of error in solution
    tol = 1e-10

    # ---------------------------------------------------------------------------
    # TEST
    # ---------------------------------------------------------------------------
    if verbose:
        # Title
        print()
        print("=" * 80)
        print("SERIAL TEST: solve linear system A*x = b using preconditioned conjugate gradient")
        print("=" * 80)
        print()

    # Manufacture right-hand-side vector from exact solution
    b  = A.dot(xe)
    bt = A.T.dot(xe)
    bh = A.H.dot(xe)

    # class LocallyOnlyJacobiSolver(LinearSolver):
    #     @property
    #     def space(self):
    #         return V
    #
    #     def solve(self, rhs, out=None, transposed=False):
    #         # (don't care about out or any other parameter here; it's only used locally)
    #         return jacobi(A, rhs)

    # Solve linear system using PCG (and CG)
    # also does an interface test for the Jacobi preconditioner


    #Create the solvers with different preconditioner
    solv0  = inverse(A, 'pcg', tol=1e-13, verbose=True)
    solv0t = solv0.transpose()
    solv0h = solv0.H()
    solv1  = inverse(A, 'pcg', tol=1e-13, verbose=True, pc="jacobi")
    # solv1b = inverse(A, 'pcg', tol=1e-13, verbose=True, pc=jacobi)
    # solv1c = inverse(A, 'pcg', tol=1e-13, verbose=True, pc=LocallyOnlyJacobiSolver())
    # solv2 = inverse(A, 'pcg', tol=1e-13, verbose=True, pc="weighted_jacobi")

    # Solve linear system using PCG with different preconditioner
    x0 = solv0 @ b
    info0 = solv0.get_info()
    x0t = solv0t.solve(bt)
    x0h = solv0h.dot(bh)
    x1 = solv1 @ b
    info1 = solv1.get_info()
    # x1b = solv1b @ b
    # info1b = solv1b.get_info()
    # x1c = solv1c @ b
    # info1c = solv1c.get_info()
    # x2 = solv2 @ b
    # info2 = solv2.get_info()

    # Verify correctness of calculation: L2-norm of error
    err0 = x0 - xe
    err_norm0 = np.linalg.norm(err0.toarray())
    err0t = x0t - xe
    errt_norm0 = np.linalg.norm(err0t.toarray())
    err0h = x0h - xe
    errh_norm0 = np.linalg.norm(err0h.toarray())

    err1 = x1 - xe
    err_norm1 = np.linalg.norm(err1.toarray())

    # err2 = x2 - xe
    # err_norm2 = np.linalg.norm(err2.toarray())

    # ---------------------------------------------------------------------------
    # TERMINAL OUTPUT
    # ---------------------------------------------------------------------------
    if verbose:
        print()
        print('A  =', A.toarray(), sep='\n')
        print('b  =', b.toarray())
        print('x1 =', x1.toarray())
        # print('x2 =', x2.toarray())
        print('xe =', xe.toarray())
        print('info1 (Jac)  =', info1)
        # print('info2 (w-Jac)=', info2)
        print()

        print("-" * 40)
        print("L2-norm of error in (PCG + Jacobi) solution = {:.2e}".format(err_norm1))
        # print("L2-norm of error in solution (PCG + weighted Jacobi) solution = {:.2e}".format(err_norm2))
        if err_norm0 < tol and err_norm1 < tol: #and err_norm2 < tol:
            print("PASSED")
        else:
            print("FAIL")
        print("-" * 40)

    # ---------------------------------------------------------------------------
    # PYTEST
    # ---------------------------------------------------------------------------
    assert err_norm0 < tol and err_norm1 < tol # and err_norm2 < tol
    assert errt_norm0 < tol and errh_norm0 < tol
    # assert info1 == info1b and info1 == info1c

#===============================================================================
@pytest.mark.parametrize( 'n', [5, 10, 13] )
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('dtype', [float, complex])
def test_cg_tridiagonal( n, p, dtype, verbose=False):

    # ---------------------------------------------------------------------------
    # PARAMETERS
    # ---------------------------------------------------------------------------

    V, A, xe = define_data_hermitian(n, p, dtype=dtype)

    # Tolerance for success: L2-norm of error in solution
    tol = 1e-10

    #---------------------------------------------------------------------------
    # TEST
    #---------------------------------------------------------------------------
    if verbose:
        # Title
        print()
        print( "="*80 )
        print( "SERIAL TEST: solve linear system A*x = b using conjugate gradient" )
        print( "="*80 )
        print()

    # Manufacture right-hand-side vector from exact solution
    b = A.dot( xe )
    bt = A.T.dot( xe )
    bh = A.H.dot( xe )

    #Create the solvers
    solv  = inverse(A, 'cg', tol=1e-13, verbose=True)
    solvt = solv.transpose()
    solvh = solv.H()

    # Solve linear system using lsmr
    x = solv @ b
    xt = solvt.solve(bt)
    xh = solvh.dot(bh)
    info = solv.get_info()

    # Verify correctness of calculation: L2-norm of error
    err = x-xe
    err_norm = np.linalg.norm( err.toarray() )
    errt = xt-xe
    errt_norm = np.linalg.norm( errt.toarray() )
    errh = xh-xe
    errh_norm = np.linalg.norm( errh.toarray() )

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
        print( "L2-norm of error in solution = {:.2e}".format( err_norm ) )
        if err_norm < tol:
            print( "PASSED" )
        else:
            print( "FAIL" )
        print( "-"*40 )

    #---------------------------------------------------------------------------
    # PYTEST
    #---------------------------------------------------------------------------
    assert err_norm < tol
    assert errt_norm < tol
    assert errh_norm < tol

# ===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================

if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
