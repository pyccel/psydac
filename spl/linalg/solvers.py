# coding: utf-8

import numpy as np

# ... Solver: CGL performs maxit CG iterations on the linear system Ax = b
#     starting from x = x0
def cgl(mat, b, x0, maxit, tol):
    xk = x0.zeros_like()
    mx = x0.zeros_like()
    p  = x0.zeros_like()
    q  = x0.zeros_like()
    r  = x0.zeros_like()

    # xk = x0
    xk = x0.copy()
    mx = mat.dot(x0)

    # r = b - mx
    r = b.copy()
    b.sub(mx)

    # p = r
    p = r.copy()

    rdr = r.dot(r)

    for i_iter in range(1, maxit+1):
        q = mat.dot(p)
        alpha = rdr / p.dot(q)

        # xk = xk + alpha * p
        ap = p.copy()
        ap.mul(alpha)
        xk.add(ap)

        # r  = r - alpha * q
        aq = q.copy()
        aq.mul(alpha)
        r.sub(aq)

        # ...
        if r.dot(r) >= 0.:
            norm_err = np.sqrt(r.dot(r))
            print (i_iter, norm_err )

            if norm_err < tol:
                x0 = xk.copy()
                break

        rdrold = rdr
        rdr = r.dot(r)
        beta = rdr / rdrold

        #p = r + beta * p
        bp = p.copy()
        bp.mul(beta)
        p  = r.copy()
        p.add(bp)

    x0 = xk.copy()
    # ...

    return x0
# ....
