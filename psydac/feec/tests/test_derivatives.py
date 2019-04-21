from psydac.feec.derivatives import Grad, Curl, Div
from psydac.fem.tensor       import TensorFemSpace
from psydac.fem.splines      import SplineSpace
from psydac.linalg.stencil   import StencilVector
from psydac.linalg.block     import BlockVector,ProductSpace
from psydac.fem.vector       import ProductFemSpace
import numpy as np

p = 2
grid_1 = np.linspace(0., 1., 3)
grid_2 = np.linspace(0., 1., 5)

# ... first component
V1 = SplineSpace(p, grid=grid_2)
V2 = SplineSpace(p, grid=grid_2)

Vh      = TensorFemSpace(V1, V2)
Grad_Vh = ProductFemSpace(Vh.reduce_degree([0]), Vh.reduce_degree([1]))
Curl_Vh = Vh.reduce_degree([0,1])

grad = Grad(Vh, Grad_Vh.vector_space)
curl = Curl(Vh, Grad_Vh.vector_space, Curl_Vh.vector_space)

x    = BlockVector(ProductSpace(Vh.vector_space))
y    = BlockVector(Grad_Vh.vector_space)

grad(x)
curl(y)
grad._matrix.toarray()
x.toarray()


