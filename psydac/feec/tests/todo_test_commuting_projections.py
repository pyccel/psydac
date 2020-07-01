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


# ... de Rham sequence
drs = DeRhamSequence([p,p], grid=[grid_1,grid_2], kind='grad-curl')  # BCs ?

proj = drs.projectors()
diff = drs.differentials()

# or from the spaces ?
V0 = drs.space('V0')
V1 = drs.space('V1')


# ... phi in V^0
phi = lambda x, y: np.sin(x)*np.sin(y)

# ... F = grad phi in V^1
dx_phi = lambda x, y: np.cos(x)*np.sin(y)
dy_phi = lambda x, y: np.sin(x)*np.cos(y)
F = [dx_phi, dy_phi]

# ... projections in V^0_h and V^1_h

# from operator sequences:
phi_h = proj('V0', phi)
grad_phi_h = diff('V0', phi_h)
F_h = proj('V1',F)

# or:
phi_h = V0.proj(phi)
grad_phi_h = V0.diff(phi_h)
F_h = V1.proj(F)

# ... checking commutation property
assert grad_phi_h == F_h

