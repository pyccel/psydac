#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np

from psydac.mapping.discrete import SplineMapping
from psydac.linalg.stencil   import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.block     import BlockVector, BlockLinearOperator
from psydac.polar .dense     import DenseVector, DenseMatrix
from psydac.polar.c1_spaces  import new_c1_vector_space
from psydac.polar.c1_linops  import LinearOperator_StencilToDense
from psydac.polar.c1_linops  import LinearOperator_DenseToStencil

__all__ = ('C1Projector',)

#===============================================================================
class C1Projector: 
    #     +---+---------+
    #     |   |         |
    #     | L |    0    |
    #     |   |         |
    #     +---+---------+
    #E =  |   |         |
    #     |   |         |
    #     | 0 |    I    |
    #     |   |         |
    #     |   |         |
    #     +---+---------+
    
    def __init__(self, mapping):

        assert isinstance(mapping, SplineMapping)

        S = mapping.space.coeff_space

        assert isinstance(S, StencilVectorSpace)

        # Vector spaces
        self._S = S
        self._P = new_c1_vector_space(S, radial_dim=0)

        # Store matrix L with 3 indices
        self._L = self.compute_lambda(mapping)

    # ...
    def compute_lambda(self, mapping):

        s1, s2 = mapping.space.coeff_space.starts
        e1, e2 = mapping.space.coeff_space.ends
        p1, p2 = mapping.space.coeff_space.pads

        if s1 == 0:

            # Number of coefficients
            n0 = 3

            # Extract control points, including ghost regions
            x_ext = mapping.control_points[0:2, :, 0]
            y_ext = mapping.control_points[0:2, :, 1]

            # Exclude ghost regions for calculations

            x = x_ext[:, p2:-p2]
            y = y_ext[:, p2:-p2]

            SQRT3     = np.sqrt(3.0)
            ONE_THIRD = 1.0 / 3.0
            (x0, y0)  = (x[0, 0], y[0, 0])

            # Define equilateral triangle enclosing first row of control points
            tau = max(np.max(-2 * (x[1,:] - x0)),
                      np.max(x[1, :] - x0 - SQRT3 * (y[1, :] - y0)),
                      np.max(x[1, :] - x0 + SQRT3 * (y[1, :] - y0)))

            # Obtain maximum from all processes at center
            if self.c1_space[0].parallel:
                from mpi4py import MPI
                comm = self.c1_space[0].angle_comm
                tau  = comm.allreduce(tau, op=MPI.MAX)

            # Coordinates of vertices of equilateral triangle
            vrtx = [(x0 - tau/2, y0 - SQRT3 * tau/2),
                    (x0 + tau  , y0                ),
                    (x0 - tau/2, y0 + SQRT3 * tau/2)]

            # Define barycentric coordinates with respect to smallest circle
            # enclosing first row of control points
            # [extended domain]
            lambda_0  = ONE_THIRD * (1.0 + 2.0 * (x_ext - x0)                         / tau)
            lambda_1  = ONE_THIRD * (1.0 -      ((x_ext - x0) - SQRT3 * (y_ext - y0)) / tau)
            lambda_2  = ONE_THIRD * (1.0 -      ((x_ext - x0) + SQRT3 * (y_ext - y0)) / tau)
            lamb = (lambda_0, lambda_1, lambda_2)

        else:

            n0   = 0
            tau  = None
            vrtx = None
            lamb = ()

        # STORE INFO
        # TODO: remove this?
        self._tau  = tau
        self._vrtx = vrtx
        self._lamb = lamb

        L = np.array(lamb, dtype=float).reshape(n0, 2, e2-s2+1 + 2*p2)
        return L

    # ...
    @property
    def L(self):
        L = self._L[:, :]
        L.setflags(write=False)
        return L

    # ...
    @property
    def tensor_space(self):
        return self._S

    # ...
    @property
    def c1_space(self):
        return self._P

    #---------------------------------------------------------------------------
    def change_matrix_basis(self, G):
        """
        Compute G' = E^t * G * E.

        """

        assert isinstance(G, StencilMatrix)
        assert G.domain   == self.tensor_space
        assert G.codomain == self.tensor_space

        L = self._L
        P = self.c1_space


        n0 = P[0].ncoeff

        n1, n2 = P[1].npts
        s1, s2 = P[1].starts
        e1, e2 = P[1].ends
        p1, p2 = P[1].pads

        #****************************************
        # Compute A' = L^T A L
        #****************************************
        Ap = np.zeros((n0, n0))

        # Compute product (A L) and store it
        AL = np.zeros((n0, 2, e2-s2+1))
        for v in range(n0):
            for i1 in [0, 1]:
                for i2 in range(s2, e2+1):

                    # Sum over j1 and j2
                    temp = 0
                    for j1 in [0,1]:
                        for k2 in range(-p2, p2+1):
                            k1 = j1 - i1
                            j2 = i2 + k2
                            temp += G[i1, i2, k1, k2] * L[v, j1, p2+j2-s2]

                    AL[v, i1, i2-s2] = temp

        # Compute product A' = L^T (A L)
        for u in range(n0):
            for v in range(n0):
                Ap[u, v] = np.dot(L[u, :, p2:-p2].flat, AL[v, :, :].flat)

        # Merge all contributions at different angles
        if P[0].parallel:
            from mpi4py import MPI
            U = P[0]
            if U.radial_comm.rank == U.radial_root:
                U.angle_comm.Allreduce(MPI.IN_PLACE, Ap, op=MPI.SUM)
            del U, MPI

        # Convert Numpy 2D array to our own linear operator
        Ap = DenseMatrix(P[0], P[0], Ap)

        #****************************************
        # Compute B' = L^T B
        #****************************************
        Bp = np.zeros((n0, p1, e2-s2+1 + 2*p2))

        for u in range(n0):
            for i1 in [0, 1]:
                for i2 in range(s2, e2+1):
                    for j1 in range(2, i1+p1+1):
                        k1 = j1 - i1
                        for k2 in range(-p2, p2+1):
                            j2 = i2 + k2
                            Bp[u, j1-2, p2+j2-s2] += L[u, i1, p2+i2-s2] * G[i1, i2, k1, k2]

        # Create linear operator
        Bp = LinearOperator_StencilToDense(P[1], P[0], Bp)

        #****************************************
        # Compute C' = C L
        #****************************************
        Cp = np.zeros((p1, e2-s2+1, n0))

        for i1 in range(2, 2+p1):
            for i2 in range(s2, e2+1):
                for v in range(n0):

                    # Sum over j1 and j2
                    temp = 0
                    for j1 in range(max(0, i1-p1), 2):
                        for k2 in range(-p2, p2+1):
                            k1 = j1 - i1
                            j2 = i2 + k2
                            temp += G[i1, i2, k1, k2] * L[v, j1, p2+j2-s2]

                    Cp[i1-2, i2-s2, v] = temp

        # Create linear operator
        Cp = LinearOperator_DenseToStencil(P[0], P[1], Cp)

        #****************************************
        # Store D' = D
        #****************************************

        # Create linear operator
        Dp = StencilMatrix(P[1], P[1])

        # Copy all data from G, but skip values for i1 = 0, 1
        Dp[s1:e1+1, :, :, :] = G[2+s1:2+e1+1, :, :, :]

        # Remove values with negative j1 which may have polluted ghost region
        Dp.remove_spurious_entries()

        #****************************************
        # Block linear operator G' = E^T G E
        #****************************************

        return BlockLinearOperator(P, P, blocks = [[Ap, Bp], [Cp, Dp]])

    #---------------------------------------------------------------------------
    def change_rhs_basis(self, b):
        """
        Compute b' = E^t * b.

        We assume that b was obtained by L^2-projection of a callable function
        onto the tensor-product spline space.

        """
        assert isinstance(b, StencilVector)
        assert b.space == self.tensor_space

        L = self._L
        P = self.c1_space
        n0 = P[0].ncoeff

        s1, s2 = P[1].starts
        e1, e2 = P[1].ends
        p1, p2 = P[1].pads

        bp0 = np.zeros(n0)
        for u in range(n0):
            for i1 in [0, 1]:
                for i2 in range(s2, e2+1):
                    bp0[u] += L[u, i1, p2+i2-s2] * b[i1, i2]

        # Merge all contributions at different angles
        if P[0].parallel:
            from mpi4py import MPI
            U = P[0]
            if U.radial_comm.rank == U.radial_root:
                U.angle_comm.Allreduce(MPI.IN_PLACE, bp0, op=MPI.SUM)
            del U, MPI

        bp0 =   DenseVector(P[0], bp0)
        bp1 = StencilVector(P[1])

        bp1[s1:e1+1, :] = b[2+s1:2+e1+1, :]
        bp1.update_ghost_regions()

        return BlockVector(P, blocks = [bp0, bp1])

    #---------------------------------------------------------------------------
    def convert_to_tensor_basis(self, vp):
        """
        Compute v = E * v'

        """
        assert isinstance(vp, BlockVector)
        assert vp.space == self.c1_space

        L = self._L
        P = self.c1_space

        n0 = P[0].ncoeff

        s1, s2 = P[1].starts
        e1, e2 = P[1].ends
        p1, p2 = P[1].pads

        v = StencilVector(self.tensor_space)
        v[2+s1:2+e1+1, :] = vp[1][s1:e1+1, :]

        vp0 = vp[0].toarray()
        for i1 in [0, 1]:
            for i2 in range(s2, e2+1):
                for u in range(n0):
                    v[i1, i2] = np.dot(L[:, i1, p2+i2-s2], vp0)

        v.update_ghost_regions()

        return v
