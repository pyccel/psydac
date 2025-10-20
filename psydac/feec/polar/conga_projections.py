import numpy as np

from scipy.sparse import coo_matrix

from psydac.linalg.basic     import LinearOperator, Vector
from psydac.linalg.stencil   import StencilVector
from psydac.fem.tensor       import TensorFemSpace


class C0PolarProjection_V0(LinearOperator):
    """
    CONGA Projector P0 from the full spline space S^{p1, p2} on logical domain
    to V0 the pre-polar 0-forms splines. The associate matrix is square as in
    the CONGA approach we keep using the tensor B-spline basis, instead of the
    polar basis of Toshniwal. P0 enforces coefficient relations to be in V0.

    Parameters:
    -----------

    W0 : TensorFemSpace
         The full tensor product spline space S^{p1,p2}

    transposed : Boolean
         switch between P0 and P0 transposed (defalut is False)

    hbc : Boolean
         switch on and off the imposition of homogeneous Dirichlet boundary
         conditions (default is False)
    """

    def __init__(self, W0, *, transposed=False, hbc=False):
        assert isinstance(W0, TensorFemSpace)

        self.W0 = W0
        self.transposed = transposed
        self.hbc = hbc

        # Radial and angle sub-communicators (1D)
        self._cart = W0.coeff_space.cart
        self._radial_comm = self._cart.subcomm[0]
        self._angle_comm = self._cart.subcomm[1]

    @property
    def domain(self):
        return self.W0.coeff_space

    @property
    def codomain(self):
        return self.W0.coeff_space

    @property
    def dtype(self):
        return float

    def dot(self, x, out=None):
        assert isinstance(x, StencilVector)
        if not x.ghost_regions_in_sync:
            x.update_ghost_regions()

        [s1, s2] = self.W0.coeff_space.starts
        [e1, e2] = self.W0.coeff_space.ends
        [n1, n2] = self.W0.coeff_space.npts
        rank_at_polar_edge = (s1 == 0)
        rank_at_outer_edge = (e1 == n1 - 1)

        if out is None:
            y = self.W0.coeff_space.zeros()
        else:
            assert isinstance(out, StencilVector)
            assert out.space is self.W0.coeff_space
            y = out

        if rank_at_polar_edge:
            local_avg = np.average(x[0, s2:e2 + 1])
            print("local avg:", local_avg, self._angle_comm.rank)

            if self._cart.is_parallel:
                from mpi4py import MPI
                local_avg = self._angle_comm.allreduce(local_avg, op=MPI.SUM)
                print("sum after allreduce:", local_avg, self._angle_comm.rank)

            y[0, s2:e2 + 1] = local_avg / self._angle_comm.size
            y[1:e1 + 1, s2:e2 + 1] = x[1:e1 + 1, s2:e2 + 1]
        else:
            y[s1:e1 + 1, s2:e2 + 1] = x[s1:e1 + 1, s2:e2 + 1]


        if self.hbc:
            if rank_at_outer_edge:
                y[e1, :] = 0.

        y.update_ghost_regions()
        return y

    def transpose(self, conjugate=False):
        #should just return self since it's symmetric?
        return C0PolarProjection_V0(self.W0, transposed=not self.transposed, hbc=self.hbc)

    def tosparse(self):

        [n1, n2] = self.W0.coeff_space.npts

        data = np.tile((1 / n2) * np.ones(n2), n2)
        cols = np.repeat(np.arange(n2), n2)
        rows = np.tile(np.arange(n2), n2)
        if self.hbc:
            data = np.concatenate((data, np.ones(n2 * (n1 - 2))))
            cols = np.concatenate((cols, np.arange(n2, (n1 - 1) * n2)))
            rows = np.concatenate((rows, np.arange(n2, (n1 - 1) * n2)))
        else:
            data = np.concatenate((data, np.ones(n2 * (n1 - 1))))
            cols = np.concatenate((cols, np.arange(n2, n1 * n2)))
            rows = np.concatenate((rows, np.arange(n2, n1 * n2)))

        P = coo_matrix((data, (rows, cols)), shape=[n1 * n2, n1 * n2], dtype=self.W0.coeff_space.dtype)
        P.eliminate_zeros()

        # P is symmetric so we always return P no matter self.transposed
        return P

    def toarray(self):
        return self.tosparse().toarray()

