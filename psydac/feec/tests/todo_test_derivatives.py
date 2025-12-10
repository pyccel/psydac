from psydac.feec.derivatives import Grad, Curl, Div
from psydac.fem.tensor       import TensorFemSpace
from psydac.fem.splines      import SplineSpace
from psydac.linalg.stencil   import StencilVector
from psydac.linalg.block     import BlockVector,ProductSpace
from psydac.fem.vector       import ProductFemSpace
from psydac.feec.utilities   import Interpolation, interpolation_matrices
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

p = 3
grid_1 = np.linspace(0., 1., 6)
grid_2 = np.linspace(0., 1., 6)

# ... first component
V1 = SplineSpace(p, grid=grid_2)
V2 = SplineSpace(p, grid=grid_2)

H1    = TensorFemSpace(V1, V2, comm=comm)
Hcurl = ProductFemSpace(H1.reduce_degree([0]), H1.reduce_degree([1]))
L2    = H1.reduce_degree([0,1])


Int = Interpolation(H1=H1, Hcurl=Hcurl, L2=L2)
M0, M1, M2 = interpolation_matrices(H1)
# ... H1
f = lambda x,y: x*(1.-x)*y*(1.-y)
F = Int('H1', f)

# ... Hcurl
g0 = lambda x,y: (1.-2.*x)*y*(1.-y)
g1 = lambda x,y: x*(1.-x)*(1.-2.*y)
G = Int('Hcurl', [g0, g1])

F =  M0.solve(F)
G =  M1.solve(G)

grad = Grad(H1, Hcurl.vector_space)
curl = Curl(H1, Hcurl.vector_space, L2.vector_space)

x    = BlockVector(ProductSpace(H1.vector_space))
y    = BlockVector(Hcurl.vector_space)

x[0]._data = F._data
x = grad(x).toarray()
y = np.concatenate(tuple(g.toarray() for g in G))
print(np.abs(x-y).max())

