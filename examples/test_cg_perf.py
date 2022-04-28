import time
import numpy as np

from mpi4py       import MPI
from psydac.ddm.cart       import CartDecomposition
from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.api.settings   import PSYDAC_BACKEND_GPYCCEL
from psydac.linalg.iterative_solvers import cg

n1 = 10000
p1 = 1
P1 = False

comm = MPI.COMM_WORLD
cart = CartDecomposition(
    npts    = [n1],
    pads    = [p1],
    periods = [P1],
    reorder = False,
    comm    = comm,
    reverse_axis = False,
)

V = StencilVectorSpace( cart )
s1, = V.starts
e1, = V.ends

A = StencilMatrix( V, V , backend=PSYDAC_BACKEND_GPYCCEL)
x0 = StencilVector( V )

A[:, 0]  = 2
A[:,-1] = -1
A[:, 1] = -1

A.remove_spurious_entries()

for i1 in range(s1,e1+1):
    x0[i1] = 2.0*np.random.random() - 1.0

b = A.dot(x0)

t0 = time.time()
x, info = cg( A, b, tol=1e-13, maxiter=30000 )
t1 = time.time()
T  = t1-t0
T  = comm.reduce(T, op=MPI.MAX)

if comm.rank == 0:
    print(info)
    print("Elapsed time ::", t1-t0)



