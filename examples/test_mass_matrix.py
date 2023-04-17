from psydac.linalg.solvers      import inverse

f1 = lambda s,t: 1.0+s**3

rhs = assemble_rhs( V, mapping, f1 )

M_inv = inverse(M, 'cg', tol=1e-10, maxiter=100, verbose=True)
sol = M_inv @ rhs
info = M_inv.get_info()

f2 = FemField( V, 'f2' )
f2.coeffs[:] = sol[:]
f2.coeffs.update_ghost_regions()

# Compute L2 norm of error
sqrt_g    = lambda *x: np.sqrt( mapping.metric_det( x ) )
integrand = lambda *x: (f1(*x)-f2(*x))**2 * sqrt_g(*x)
e2 = np.sqrt( V.integral( integrand ) )

del f2
print( 'L2 error :: {:.2e}'.format( e2 ) )
