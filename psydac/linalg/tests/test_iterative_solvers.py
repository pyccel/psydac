
import numpy as np
import pytest
from psydac.linalg.iterative_solvers import bicg, bicgstab, lsmr, minres, pcg, jacobi, cg
from psydac.linalg.stencil import StencilVectorSpace, StencilMatrix, StencilVector
from psydac.linalg.basic import LinearSolver
from psydac.ddm.cart import DomainDecomposition, CartDecomposition


def define_data_hermitien(n, p, dtype=float):
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
def test_bicgstab_tridiagonal( n, p, dtype ):

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

    # Title
    print()
    print( "="*80 )
    print( "SERIAL TEST: solve linear system A*x = b using biconjugate gradient" )
    print( "="*80 )
    print()

    # Manufacture right-hand-side vector from exact solution
    b = A.dot( xe )

    # Solve linear system using BiCG
    x, info = bicgstab( A, b, tol=1e-13, verbose=True )

    # Verify correctness of calculation: L2-norm of error
    err = x-xe
    err_norm = np.linalg.norm( err.toarray() )

    #---------------------------------------------------------------------------
    # TERMINAL OUTPUT
    #---------------------------------------------------------------------------

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

#===============================================================================
@pytest.mark.parametrize( 'n', [5, 10, 13] )
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('dtype', [float,complex])
def test_bicg_tridiagonal( n, p, dtype ):

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

    # Title
    print()
    print( "="*80 )
    print( "SERIAL TEST: solve linear system A*x = b using biconjugate gradient" )
    print( "="*80 )
    print()

    # Manufacture right-hand-side vector from exact solution
    b = A.dot( xe )

    # Solve linear system using BiCG
    x, info = bicg( A, A.H, b, tol=1e-13, verbose=True )

    # Verify correctness of calculation: L2-norm of error
    err = x-xe
    err_norm = np.linalg.norm( err.toarray() )

    #---------------------------------------------------------------------------
    # TERMINAL OUTPUT
    #---------------------------------------------------------------------------

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

#===============================================================================
@pytest.mark.parametrize( 'n', [5, 10, 13] )
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('dtype', [float, complex])
def test_lsmr_tridiagonal( n, p, dtype ):

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

    # Title
    print()
    print( "="*80 )
    print( "SERIAL TEST: solve linear system A*x = b using lsmr" )
    print( "="*80 )
    print()

    # Manufacture right-hand-side vector from exact solution
    b = A.dot( xe )

    # Solve linear system using BiCG
    x, info = lsmr( A, A.H, b, tol=1e-13, verbose=True )

    # Verify correctness of calculation: L2-norm of error
    res = A.dot(x)-b
    res_norm = np.linalg.norm( res.toarray() )

    #---------------------------------------------------------------------------
    # TERMINAL OUTPUT
    #---------------------------------------------------------------------------

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

#===============================================================================
@pytest.mark.parametrize( 'n', [5, 10, 13] )
@pytest.mark.parametrize('p', [2, 3])
def test_minres_tridiagonal( n, p ):

    #---------------------------------------------------------------------------
    # PARAMETERS
    #---------------------------------------------------------------------------

    V, A, xe = define_data_hermitien(n, p)

    # Tolerance for success: L2-norm of error in solution
    tol = 1e-12

    #---------------------------------------------------------------------------
    # TEST
    #---------------------------------------------------------------------------

    # Title
    print()
    print( "="*80 )
    print( "SERIAL TEST: solve linear system A*x = b using minres" )
    print( "="*80 )
    print()

    # Manufacture right-hand-side vector from exact solution
    b = A.dot( xe )

    # Solve linear system using BiCG
    x, info = minres( A, b, tol=1e-13, verbose=True )

    # Verify correctness of calculation: L2-norm of error
    res = A.dot(x)-b
    res_norm = np.linalg.norm( res.toarray() )

    #---------------------------------------------------------------------------
    # TERMINAL OUTPUT
    #---------------------------------------------------------------------------

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

# ===============================================================================
@pytest.mark.parametrize('n', [8, 16])
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('dtype', [float, complex])
def test_pcg_tridiagonal(n, p, dtype):
    # ---------------------------------------------------------------------------
    # PARAMETERS
    # ---------------------------------------------------------------------------

    V, A, xe = define_data_hermitien(n, p, dtype=dtype)

    # Tolerance for success: L2-norm of error in solution
    tol = 1e-10

    # ---------------------------------------------------------------------------
    # TEST
    # ---------------------------------------------------------------------------

    # Title
    print()
    print("=" * 80)
    print("SERIAL TEST: solve linear system A*x = b using preconditioned conjugate gradient")
    print("=" * 80)
    print()

    # Manufacture right-hand-side vector from exact solution
    b = A.dot(xe)

    class LocallyOnlyJacobiSolver(LinearSolver):
        @property
        def space(self):
            return V

        def solve(self, rhs, out=None, transposed=False):
            # (don't care about out or any other parameter here; it's only used locally)
            return jacobi(A, rhs)

    # Solve linear system using PCG (and CG)
    # also does an interface test for the Jacobi preconditioner
    x0, info0 = pcg(A, b, pc=None, tol=1e-12)
    x1, info1 = pcg(A, b, pc="jacobi", tol=1e-12)
    x1b, info1b = pcg(A, b, pc=jacobi, tol=1e-12)
    x1c, info1c = pcg(A, b, pc=LocallyOnlyJacobiSolver(), tol=1e-12)
    x2, info2 = pcg(A, b, pc="weighted_jacobi", tol=1e-12)

    # Verify correctness of calculation: L2-norm of error
    err0 = x0 - xe
    err_norm0 = np.linalg.norm(err0.toarray())

    err1 = x1 - xe
    err_norm1 = np.linalg.norm(err1.toarray())

    err2 = x2 - xe
    err_norm2 = np.linalg.norm(err2.toarray())

    # ---------------------------------------------------------------------------
    # TERMINAL OUTPUT
    # ---------------------------------------------------------------------------

    print()
    print('A  =', A.toarray(), sep='\n')
    print('b  =', b.toarray())
    print('x1 =', x1.toarray())
    print('x2 =', x2.toarray())
    print('xe =', xe.toarray())
    print('info1 (Jac)  =', info1)
    print('info2 (w-Jac)=', info2)
    print()

    print("-" * 40)
    print("L2-norm of error in (PCG + Jacobi) solution = {:.2e}".format(err_norm1))
    print("L2-norm of error in solution (PCG + weighted Jacobi) solution = {:.2e}".format(err_norm2))
    if err_norm0 < tol and err_norm1 < tol and err_norm2 < tol:
        print("PASSED")
    else:
        print("FAIL")
    print("-" * 40)

    # ---------------------------------------------------------------------------
    # PYTEST
    # ---------------------------------------------------------------------------
    assert err_norm0 < tol and err_norm1 < tol and err_norm2 < tol
    assert info1 == info1b and info1 == info1c

#===============================================================================
@pytest.mark.parametrize( 'n', [5, 10, 13] )
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('dtype', [float, complex])
def test_cg_tridiagonal( n, p, dtype ):

    # ---------------------------------------------------------------------------
    # PARAMETERS
    # ---------------------------------------------------------------------------

    V, A, xe = define_data_hermitien(n, p, dtype=dtype)

    # Tolerance for success: L2-norm of error in solution
    tol = 1e-10

    #---------------------------------------------------------------------------
    # TEST
    #---------------------------------------------------------------------------

    # Title
    print()
    print( "="*80 )
    print( "SERIAL TEST: solve linear system A*x = b using conjugate gradient" )
    print( "="*80 )
    print()

    # Manufacture right-hand-side vector from exact solution
    b = A.dot( xe )

    # Solve linear system using CG
    x, info = cg( A, b, tol=1e-13, verbose=True )

    # Verify correctness of calculation: L2-norm of error
    err = x-xe
    err_norm = np.linalg.norm( err.toarray() )

    #---------------------------------------------------------------------------
    # TERMINAL OUTPUT
    #---------------------------------------------------------------------------

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

# ===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================

if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
