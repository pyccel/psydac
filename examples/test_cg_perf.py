#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import time
import numpy as np

from mpi4py       import MPI
from psydac.ddm.cart       import CartDecomposition, DomainDecomposition
from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.api.settings   import PSYDAC_BACKEND_GPYCCEL
from psydac.linalg.solvers import inverse

def compute_global_starts_ends(domain_decomposition, npts):
    ndims         = len(npts)
    global_starts = [None]*ndims
    global_ends   = [None]*ndims

    for axis in range(ndims):
        es = domain_decomposition.global_element_starts[axis]
        ee = domain_decomposition.global_element_ends  [axis]

        global_ends  [axis]     = ee.copy()
        global_ends  [axis][-1] = npts[axis]-1
        global_starts[axis]     = np.array([0] + (global_ends[axis][:-1]+1).tolist())

    return global_starts, global_ends

n1 = 10000
p1 = 1
P1 = False

npts = [n1]
pads = [p1]
periods = [P1]
comm = MPI.COMM_WORLD
D = DomainDecomposition(npts, periods=periods, comm=comm)
global_starts, global_ends = compute_global_starts_ends(D, npts)
C = CartDecomposition(D, npts, global_starts, global_ends, pads=pads, shifts=[1])

V = StencilVectorSpace(C)
s1, = V.starts
e1, = V.ends

A = StencilMatrix( V, V , backend=PSYDAC_BACKEND_GPYCCEL)
x0 = StencilVector( V )

A[:, 0]  = 2
A[:,-1] = -1
A[:, 1] = -1
A = A.T
A.remove_spurious_entries()

for i1 in range(s1,e1+1):
    x0[i1] = 1

b = A.dot(x0)

t0 = time.time()
A_inv = inverse(A, 'cg', tol=1e-13, maxiter=30000)
x = A_inv @ b
info = A_inv.get_info()
t1 = time.time()
T  = t1-t0
T  = comm.reduce(T, op=MPI.MAX)

if comm.rank == 0:
    print("CG info      ::", info)
    print("Elapsed time ::", t1-t0)
