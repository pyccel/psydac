import numpy as np
from numpy import pi

from scipy.linalg import toeplitz, matmul_toeplitz
from scipy.sparse import coo_matrix, lil_matrix, spmatrix, eye as sp_eye
from scipy.sparse.linalg import inv as sp_inv

from psydac.fem.vector import VectorFemSpace
from psydac.linalg.basic     import LinearOperator, Vector
from psydac.linalg.block import BlockLinearOperator
from psydac.linalg.stencil   import StencilVector
from psydac.fem.tensor       import TensorFemSpace
from psydac.linalg.utilities import array_to_psydac


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
            out *= 0
            y = out

        if rank_at_polar_edge:
            # compute sum of points with s = 0 for the process
            local_sum = np.sum(x[0, s2:e2 + 1])
            if x.space.parallel:
                from mpi4py import MPI
                angle_comm = x.space.cart.subcomm[1]
                local_sum = angle_comm.allreduce(local_sum, op=MPI.SUM)
            #compute average of all points with s = 0
            y[0, s2:e2 + 1] = local_sum / n2
            y[1:e1 + 1, s2:e2 + 1] = x[1:e1 + 1, s2:e2 + 1]
        else:
            y[s1:e1 + 1, s2:e2 + 1] = x[s1:e1 + 1, s2:e2 + 1]


        if self.hbc:
            if rank_at_outer_edge:
                y[e1, :] = 0.

        y.update_ghost_regions()
        return y

    def transpose(self, conjugate=False):
        return C0PolarProjection_V0(self.W0, transposed=not self.transposed, hbc=self.hbc)

    def tosparse(self):

        # Matrix size is n1*n2. Rows with global indices i*n2 + j (C-style flattening) with
        # s1 <= i <= e1, s2 <= j <= e2 are filled for the current process

        [n1, n2] = self.W0.coeff_space.npts
        [s1, s2] = self.W0.coeff_space.starts
        [e1, e2] = self.W0.coeff_space.ends
        rank_at_polar_edge = (s1 == 0)
        rank_at_outer_edge = (e1 == n1 - 1)
        data, cols, rows = [], [], []

        # Assemble the matrix entries which contribute to the polar edge (s1 = 0)
        len_theta = e2 - s2 + 1
        if rank_at_polar_edge:
            data = np.tile((1 / n2) * np.ones(n2), len_theta)
            rows = np.repeat(np.arange(s2, e2 + 1), n2)
            cols = np.tile(np.arange(n2), len_theta)

        # Assemble the rest of the matrix (identity block)
        # We do not need entries with s1 = 0 (already accounted for)
        start_s = s1 + 1 if rank_at_polar_edge else s1
        end_s = e1 if (rank_at_outer_edge and self.hbc) else e1 + 1

        i = np.arange(start_s, end_s)[:, None]
        j = np.arange(s2, e2 + 1)[None, :]
        # rows of identity owned by the process
        local_rows = (i * n2 + j).ravel()

        data = np.concatenate((data, np.ones(len_theta * (end_s - start_s))))
        cols = np.concatenate((cols, local_rows))
        rows = np.concatenate((rows, local_rows))

        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)

        P = coo_matrix((data, (rows, cols)), shape=[n1 * n2, n1 * n2], dtype=self.W0.coeff_space.dtype)
        P.eliminate_zeros()

        # P is symmetric so we always return P no matter self.transposed
        return P

    def toarray(self):
        return self.tosparse().toarray()


# --------- 1-FORMS CONGA PROJECTOR P1 ----------#
# It is a BlockLinearOperator with 4 blocks Upper-Left (0, 0), Upper-Right (0, 1)
# Lower-Left (1, 0) and Lower-Right (1, 1). (0, 1) and (1, 0) are identical.
#
#                ______________________
#               |           |          |
#               |   (0,0)   |   (1,0)  |
#               |___________|__________|
#               |           |          |
#               |   (1,0)   |   (1,1)  |
#               |___________|__________|

class C0PolarProjection_V1_00(LinearOperator):
    """
    Upper Left block of P1.

    Parameters:
    ----------

    W1 : VectorFemSpace
         Full tensor product spline space of the 1-forms S^{p1-1, p2} x S^{p1, p2-1}

    transposed : Boolean
         Switch between P1 and P1 transposed (default is False)
    """

    def __init__(self, W1):
        # assert isinstance(W1, ProductFemSpace)
        assert isinstance(W1, VectorFemSpace)

        self.W1 = W1

    @property
    def domain(self):
        return self.W1.coeff_space[0]

    @property
    def codomain(self):
        return self.W1.coeff_space[0]

    @property
    def dtype(self):
        return float

    # The upper-left block of P1 is an identity block
    def dot(self, x, out=None):
        assert isinstance(x, StencilVector)

        if not x.ghost_regions_in_sync:
            x.update_ghost_regions()

        [s1, s2] = self.W1.coeff_space[0].starts
        [e1, e2] = self.W1.coeff_space[0].ends

        if out is None:
            y = self.W1.coeff_space[0].zeros()
        else:
            assert isinstance(out, StencilVector)
            assert out.space is self.W1.coeff_space[0]
            out *= 0
            y = out

        y[s1:e1+1, s2:e2+1] = x[s1:e1+1, s2:e2+1]

        y.update_ghost_regions()
        return y

    def transpose(self, conjugate=False):
        return self

    @property
    def T(self):
        return self.transpose()

    def tosparse(self):

        [n01, n02] = self.domain.npts
        [s1, s2] = self.domain.starts
        [e1, e2] = self.domain.ends

        i = np.arange(s1, e1 + 1)[:, None]
        j = np.arange(s2, e2 + 1)[None, :]
        local_rows = (i * n02 + j).ravel()
        data = np.ones((e1 - s1 + 1) * (e2 - s2 + 1))

        local_rows = np.asarray(local_rows, dtype=np.int64)

        P = coo_matrix((data, (local_rows, local_rows)), shape=[n01 * n02, n01 * n02], dtype=self.domain.dtype)
        P.eliminate_zeros()

        return P

    def toarray(self):
        return self.tosparse().toarray()



class C0PolarProjection_V1_10(LinearOperator):
    """
    Lower left block of P1.

    Parameters:
    -----------

    W1 : VectorFemSpace (former ProductFemSpace)
         Full tensor product spline space of the 1-forms S^{p1-1, p2} x S^{p1, p2-1}

    transposed : Boolean
         Switch between P1 and P1 transposed (default is False)
    """

    def __init__(self, W1, transposed=False):
        # assert isinstance(W1, ProductFemSpace)
        assert isinstance(W1, VectorFemSpace)

        self.W1 = W1
        self.transposed = transposed

    @property
    def domain(self):
        idx = 1 if self.transposed else 0
        return self.W1.coeff_space[idx]

    @property
    def codomain(self):
        idx = 0 if self.transposed else 1
        return self.W1.coeff_space[idx]

    @property
    def dtype(self):
        return float

    def dot(self, x, out=None):
        assert isinstance(x, StencilVector)

        if not x.ghost_regions_in_sync:
            x.update_ghost_regions()

        # The number of radial basis functions is one less than the number on the angular basis functions along dir x1
        [s1, s2] = self.domain.starts
        [e1, e2] = self.domain.ends
        rank_at_polar_edge = (s1 == 0)

        if out is None:
            y = self.codomain.zeros()
        else:
            assert isinstance(out, StencilVector)
            assert out.space is self.codomain
            out *= 0
            y = out

        y[s1:e1 + 1, s2:e2 + 1] = 0

        if self.transposed:
            if rank_at_polar_edge:
                y[0, s2:e2 + 1] = x[1, s2 - 1:e2] - x[1, s2:e2 + 1]
        else:
            if rank_at_polar_edge:
                # the cells are periodic in angular dim, so we use ghost regions
                y[1, s2:e2 + 1] = x[0, s2 + 1:e2 + 2] - x[0, s2:e2 + 1]

        y.update_ghost_regions()
        return y

    def transpose(self, conjugate=False):
        return C0PolarProjection_V1_10(self.W1, transposed=not self.transposed)

    @property
    def T(self):
        return self.transpose()

    def tosparse(self):

        [n01, n02] = self.domain.npts
        [n11, n12] = self.codomain.npts
        s1, s2 = self.domain.starts
        e1, e2 = self.codomain.ends
        rank_at_polar_edge = (s1 == 0)

        data, cols, rows = [], [], []

        # matrix d (derivatives of periodic splines)
        # for example: n_theta = 3, single rank at polar edge, not transposed. Then:
        # data = [-1 1 -1 1 -1 1], rows = [3 3 4 4 5 5], cols = [0 1 1 2 2 0]
        if rank_at_polar_edge:
            len_theta = e2 - s2 + 1
            data = np.tile([-1, 1], len_theta)
            k = np.arange(s2, e2 + 1)
            if self.transposed:
                rows = np.repeat(np.arange(s2, e2 + 1), 2)
                cols = np.column_stack((k, (k - 1) % n12)).ravel() + n12
            else:
                rows = np.repeat(np.arange(s2, e2 + 1), 2) + n12
                cols = np.column_stack((k, (k + 1) % n12 )).ravel()

        dtype = self.domain.dtype
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)

        P = coo_matrix((data, (rows, cols)), shape=[n11 * n12, n01 * n02], dtype=dtype)
        P.eliminate_zeros()
        return P

    def toarray(self):
        return self.tosparse().toarray()


class C0PolarProjection_V1_11(LinearOperator):
    """
    Lower right block of P1.

    Parameters:
    -----------

    W1 : VectorFemSpace (former ProductFemSpace)
         Full tensor product spline space of the 1-forms S^{p1-1, p2} x S^{p1, p2-1}

    transposed : Boolean
         Switch between P1 and P1 transposed (default is False)

    hbc : Boolean
         Switch on and off the imposition of homogeneous Dirichlet boundary
         conditions on the tangential (angular) direction (default is False)
    """

    def __init__(self, W1, transposed=False, hbc=False):
        # assert isinstance(W1, ProductFemSpace)
        assert isinstance(W1, VectorFemSpace)

        self.W1 = W1
        self.transposed = transposed
        self.hbc = hbc

    @property
    def domain(self):
        return self.W1.coeff_space[1]

    @property
    def codomain(self):
        return self.W1.coeff_space[1]

    @property
    def dtype(self):
        return float

    # It is diagonal block with a zero block of size (2n2)x(2n2) and an identity block
    def dot(self, x, out=None):
        assert isinstance(x, StencilVector)

        if not x.ghost_regions_in_sync:
            x.update_ghost_regions()

        [s1, s2] = self.codomain.starts
        [e1, e2] = self.codomain.ends
        [n1, n2] = self.codomain.npts
        rank_at_polar_edge = (s1 == 0)
        rank_at_outer_edge = (e1 == n1 - 1)

        if out is None:
            y = self.W1.coeff_space[1].zeros()
        else:
            assert isinstance(out, StencilVector)
            assert out.space is self.codomain
            out *= 0
            y = out
        if rank_at_polar_edge:
            y[0, s2:e2 + 1] = 0
            y[1, s2:e2 + 1] = 0
            y[2:e1 + 1, s2:e2 + 1] = x[2:e1 + 1, s2:e2 + 1]  # Identity block
        else:
            y[s1:e1 + 1, s2:e2 + 1] = x[s1:e1 + 1, s2:e2 + 1]

        if self.hbc and rank_at_outer_edge:
            y[e1, :] = 0.

        y.update_ghost_regions()
        return y

    def transpose(self, conjugate=False):
        return C0PolarProjection_V1_11(self.W1, transposed=not self.transposed, hbc=self.hbc)

    @property
    def T(self):
        return self.transpose()

    def tosparse(self):

        [s1, s2] = self.codomain.starts
        [e1, e2] = self.codomain.ends
        [n01, n02] = self.domain.npts
        [n11, n12] = self.codomain.npts
        rank_at_outer_edge = (e1 == n01 - 1)
        dtype = self.domain.dtype

        # identity block of size n12(n12 - 2)
        start_s = max(2, s1)
        end_s = e1 if (rank_at_outer_edge and self.hbc) else e1 + 1
        len_theta = e2 - s2 + 1
        data = np.ones(len_theta * (end_s - start_s))

        i = np.arange(start_s, end_s)[:, None]
        j = np.arange(s2, e2 + 1)[None, :]
        local_rows = (i * n02 + j).ravel()
        local_rows = np.asarray(local_rows, dtype=np.int64)

        P = coo_matrix((data, (local_rows, local_rows)), shape=[n11 * n12, n01 * n02], dtype=dtype)
        P.eliminate_zeros()
        return P


    def toarray(self):
        return self.tosparse().toarray()


class C0PolarProjection_V1(BlockLinearOperator):
    """
    CONGA Projector P1 from the full spline space S^{p1-1, p2} x S^{p1, p2-1}
    on logical domain to V1 the pre-polar 1-forms splines. The associate matrix
    is square as in the CONGA approach we keep using the tensor B-spline basis,
    instead of the polar basis of Toshniwal. P1 enforces coefficient relations
    to be in V1.

    Parameters:
    -----------

    W1 : VectorFemSpace (former ProductFemSpace)
         Full tensor product spline space of the 1-forms S^{p1-1, p2} x S^{p1, p2-1}

    transposed : Boolean
         Switch between P1 and P1 transposed (default is False)

    hbc : Boolean
         Switch on and off the imposition of homogeneous Dirichlet boundary
         conditions on the tangential (angular) direction (default is False)
    """

    def __init__(self, W1, transposed=False, hbc=False):
        assert isinstance(W1, VectorFemSpace)
        assert W1.symbolic_space.kind.name == 'hcurl'

        T1 = W1.coeff_space

        super().__init__(T1, T1)

        self[0, 0] = C0PolarProjection_V1_00(W1)
        # self[0, 1] = C0CongaProjector1_01(W1, transposed = transposed)
        self[1, 0] = C0PolarProjection_V1_10(W1, transposed=transposed)
        self[1, 1] = C0PolarProjection_V1_11(W1, transposed=transposed, hbc=hbc)

        self.W1 = W1
        self.transposed = transposed
        self.hbc = hbc


    @property
    def T(self):
        return self.transpose()


# ---------------- 2-FORMS CONGA PROJECTOR P2 ----------------#
class C0PolarProjection_V2(LinearOperator):
    """
    CONGA Projector P2 from the full spline space S^{p1-1, p2-1} on logical
    domain to V2, the pre-polar 2-forms splines. The associate matrix
    is square, as in the CONGA approach we keep using the tensor B-spline basis,
    instead of the polar basis of Toshniwal. P2 enforces coefficient relations
    to be in V2.

    Parameters:
    -----------

    W2 : TensorFemSpace
         Full tensor product spline space of the 2-forms S^{p1-1, p2-1}

    transposed : Boolean
         Switch between P2 and P2 transposed (default is False)
    """

    def __init__(self, W2, transposed=False):
        assert isinstance(W2, TensorFemSpace)

        self.W2 = W2
        self.transposed = transposed

    @property
    def domain(self):
        return self.W2.coeff_space

    @property
    def codomain(self):
        return self.W2.coeff_space

    @property
    def dtype(self):
        return float

    def dot(self, x, out=None):
        assert isinstance(x, StencilVector)

        if not x.ghost_regions_in_sync:
            x.update_ghost_regions()

        [s1, s2] = self.W2.coeff_space.starts
        [e1, e2] = self.W2.coeff_space.ends
        [n1, n2] = self.W2.coeff_space.npts
        rank_at_polar_edge = (s1 == 0)

        if out is None:
            y = self.W2.coeff_space.zeros()
        else:
            assert isinstance(out, StencilVector)
            assert out.space is self.W2.coeff_space
            out *= 0
            y = out

        if self.transposed:
            if rank_at_polar_edge:
                y[0, s2:e2 + 1] = x[1, s2:e2 + 1]
                y[1, s2:e2 + 1] = x[1, s2:e2 + 1]
                y[2:e1 + 1, s2:e2 + 1] = x[2:e1 + 1, s2:e2 + 1]
            else:
                y[s1:e1 + 1, s2:e2 + 1] = x[s1:e1 + 1, s2:e2 + 1]

        else:
            if rank_at_polar_edge:
                y[0, s2:e2 + 1] = 0
                y[1, s2:e2 + 1] = x[0, s2:e2 + 1] + x[1, s2:e2 + 1]
                y[2:e1 + 1, s2:e2 + 1] = x[2:e1 + 1, s2:e2 + 1]
            else:
                y[s1:e1 + 1, s2:e2 + 1] = x[s1:e1 + 1, s2:e2 + 1]


        y.update_ghost_regions()
        return y

    def transpose(self, conjugate=False):
        return C0PolarProjection_V2(self.W2, transposed=not self.transposed)

    @property
    def T(self):
        return self.transpose()

    def tosparse(self):

        [s1, s2] = self.W2.coeff_space.starts
        [e1, e2] = self.W2.coeff_space.ends
        [n1, n2] = self.W2.coeff_space.npts

        data = np.ones((e1 - s1 + 1) * (e2 - s2 + 1))

        i = np.arange(s1, e1 + 1)[:, None]
        j = np.arange(s2, e2 + 1)[None, :]
        local_rows = (i * n2 + j).ravel()
        if self.transposed:
            rows = local_rows
            cols = local_rows[local_rows < n2] + n2
        else:
            rows_to_repeat = local_rows[(local_rows >= n2) & (local_rows < 2 * n2)]
            rows = np.tile(rows_to_repeat, 2)
            cols = rows_to_repeat - n2
            rows = np.concatenate((rows, local_rows[local_rows >= 2 * n2]))
        cols = np.concatenate((cols, local_rows[local_rows >= n2]))

        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)

        P = coo_matrix((data, (rows, cols)), shape=(n1 * n2, n1 * n2), dtype=self.W2.coeff_space.dtype)
        P.eliminate_zeros()

        return P

    def toarray(self):
        return self.tosparse().toarray()




# =============================================================================#
#                        C1 POLAR SPLINE COMPLEX (C1P)                        #
# =============================================================================#

def cos_sin_avg(theta, x, angle_comm, s2, e2, n2, i):
    """
    Compute weighted averages  in parallel:
    sum (2/n2) * cos(theta_k) * x_ik
    sum (2/n2) * sin(theta_k) * x_ik
    over all k belonging to the process, for fixed i (0 or 1)
    then sum over all processes
    """
    from mpi4py import MPI
    sum_cos = (2 / n2) * np.dot(np.cos(theta), x[i, s2:e2 + 1])
    sum_sin = (2 / n2) * np.dot(np.sin(theta), x[i, s2:e2 + 1])
    buf = np.array([sum_cos, sum_sin])

    if angle_comm is not None:
        buf = angle_comm.allreduce(buf, op=MPI.SUM)
    return buf


# --------- 0-FORMS CONGA PROJECTOR P0 ----------#

# matrix p in the paper notation
def toeplitz_columns_sym(t, s2, e2, n2):

    i = np.arange(n2)[:, None]  # (n2, 1)
    j = np.arange(s2, e2 + 1)[None, :]  # (1, m)
    idx = np.abs(i - j).ravel('F')  # (n2, m)
    return np.cos(t[idx])


class C1PolarProjection_V0(LinearOperator):
    """
    CONGA Projector P0 from the full spline space S^{p1, p2} on logical domain
    to U0 the pre-polar 0-forms splines. The associate matrix is square as in
    the CONGA approach we keep using the tensor B-spline basis, instead of the
    polar basis of Toshniwal. P0 enforces coefficient relations to be in U0.

    Parameters:
    -----------

    W0 : TensorFemSpace
         The full tensor product spline space S^{p1,p2}

    gamma : float
         free parameter in the entries of P0. Any values provides a valid CONGA
         prjector in U0. However, in order to have the commuting property
                             grad P0 u = P1 grad u
         for u in Im(Pi0) and Pi0 the geometric projector on W0, we should set
         gamma = 1 (default)

    transposed : Boolean
         switch between P0 and P0 transposed (defalut is False)

    hbc : Boolean
         switch on and off the imposition of homogeneous Dirichlet boundary
         conditions (default is False)
    """

    def __init__(self, W0, *, gamma=1, transposed=False, hbc=False):
        assert isinstance(W0, TensorFemSpace)

        self.gamma = gamma
        self.W0 = W0
        self.transposed = transposed
        self.hbc = hbc

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
        theta_local = np.linspace(s2 * 2 * pi / n2, e2 * 2 * pi / n2, e2  - s2 + 1)
        rank_at_polar_edge = (s1 == 0)
        rank_at_outer_edge = (e1 == n1 - 1)

        if out is None:
            y = self.W0.coeff_space.zeros()
        else:
            assert isinstance(out, StencilVector)
            assert out.space is self.W0.coeff_space
            out *= 0
            y = out

        if rank_at_polar_edge:
            # compute sums of points with s = 0, 1 for the process
            x0 = np.sum(x[0, s2:e2 + 1]) / n2
            x1 = np.sum(x[1, s2:e2 + 1]) / n2

            angle_comm = x.space.cart.subcomm[1] if x.space.parallel else None
            sum_cos, sum_sin = cos_sin_avg(theta_local, x, angle_comm, s2, e2, n2, 1)

            if x.space.parallel:

                from mpi4py import MPI
                # global averages
                x0 = angle_comm.allreduce(x0, op=MPI.SUM)
                x1 = angle_comm.allreduce(x1, op=MPI.SUM)

            if self.transposed:
                y[0, s2:e2 + 1] = self.gamma * x0 + self.gamma * x1
                y[1, s2:e2 + 1] = (1 - self.gamma) * x0 + (1 - self.gamma) * x1 + \
                        np.cos(theta_local) * sum_cos + np.sin(theta_local) * sum_sin
            else:
                y[0, s2:e2 + 1] = self.gamma * x0 + (1 - self.gamma) * x1
                y[1, s2:e2 + 1] = self.gamma * x0 + (1 - self.gamma) * x1 + \
                        np.cos(theta_local) * sum_cos + np.sin(theta_local) * sum_sin
            y[2:e1 + 1, s2:e2 + 1] = x[2:e1 + 1, s2:e2 + 1]
        else:
            y[s1:e1 + 1, s2:e2 + 1] = x[s1:e1 + 1, s2:e2 + 1]

        if self.hbc and rank_at_outer_edge:
            y[e1, :] = 0.

        y.update_ghost_regions()
        return y

    def transpose(self, conjugate=False):
        return C1PolarProjection_V0(self.W0, gamma=self.gamma, transposed=not self.transposed, hbc=self.hbc)

    @property
    def T(self):
        return self.transpose()

    def tosparse(self):

        [n1, n2] = self.W0.coeff_space.npts
        [s1, s2] = self.W0.coeff_space.starts
        [e1, e2] = self.W0.coeff_space.ends
        theta = np.linspace(0, 2 * pi, n2, endpoint=False)

        rank_at_polar_edge = (s1 == 0)
        rank_at_outer_edge = (e1 == n1 - 1)
        data, cols, rows = [], [], []

        if rank_at_polar_edge:
            # matrix of size n2*n2 with all entries equal to gamma / n2
            data = (self.gamma / n2) * np.ones(n2 * (e2 - s2 + 1))
            # matrix of size n2*n2 with all entries equal to (1 - gamma) / n2
            data = np.concatenate((data, (1 - self.gamma) / n2 * np.ones((e2 - s2 + 1) * n2)))
            rows = np.tile(np.repeat(np.arange(s2, e2 + 1), n2), 2)
            cols = np.tile(np.arange(n2), e2 - s2 + 1)
            cols = np.concatenate((cols, np.tile(np.arange(n2, 2 * n2), e2 - s2 + 1)))
        if e1 > 1 > s1:
            # matrix of size n2*n2 with all entries equal to gamma / n2
            d_block2 = (self.gamma / n2) * np.ones((e2 - s2 + 1) * n2)
            data = np.concatenate((data, d_block2))
            rows = np.concatenate((rows, np.repeat(np.arange(n2 + s2, n2 + e2 + 1), n2)))
            cols = np.concatenate((cols, np.tile(np.arange(n2), e2 - s2 + 1)))

            # matrix p
            d_block2 = (1 - self.gamma) / n2 * np.ones((e2 - s2 + 1) * n2) + \
                       2 / n2 * toeplitz_columns_sym(theta, s2, e2, n2)
            data = np.concatenate((data, d_block2))
            rows = np.concatenate((rows, np.repeat(np.arange(n2 + s2, n2 + e2 + 1), n2)))
            cols = np.concatenate((cols, np.tile(np.arange(n2, 2 * n2), e2 - s2 + 1)))

        # Assemble the rest of the matrix (identity block)
        start_s = max(s1, 2)
        end_s = e1 if (rank_at_outer_edge and self.hbc) else e1 + 1
        i = np.arange(start_s, end_s)[:, None]
        j = np.arange(s2, e2 + 1)[None, :]
        local_rows = (i * n2 + j).ravel()

        data = np.concatenate((data, np.ones((e2 - s2 + 1) * (end_s - start_s))))
        cols = np.concatenate((cols, local_rows))
        rows = np.concatenate((rows, local_rows))

        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)

        P = coo_matrix((data, (rows, cols)), shape=[n1 * n2, n1 * n2], dtype=self.W0.coeff_space.dtype)
        P.eliminate_zeros()

        return P.T if self.transposed else P

    def toarray(self):
        return self.tosparse().toarray()


# --------- 1-FORMS CONGA PROJECTOR P1 ----------#
# It is a BlockLinearOperator with 4 blocks Upper-Left (0, 0), Upper-Right (0, 1)
# Lower-Left (1, 0) and Lower-Right (1, 1).
#
#                ______________________
#               |           |          |
#               |   (0,0)   |   (1,0)  |
#               |___________|__________|
#               |           |          |
#               |   (1,0)   |   (1,1)  |
#               |___________|__________|

class C1PolarProjection_V1_00(LinearOperator):
    """
    Upper Left block of P1.

    Parameters:
    ----------

    W1 : VectorFemSpace (former ProductFemSpace)
         Full tensor product spline space of the 1-forms S^{p1-1, p2} x S^{p1, p2-1}

    transposed : Boolean
         Switch between P1 and P1 transposed (default is False)
    """

    def __init__(self, W1, transposed=False):
        assert isinstance(W1, VectorFemSpace)

        self.W1 = W1
        self.transposed = transposed

    @property
    def domain(self):
        return self.W1.coeff_space[0]

    @property
    def codomain(self):
        return self.W1.coeff_space[0]

    @property
    def dtype(self):
        return float

    def dot(self, x, out=None):
        assert isinstance(x, StencilVector)

        [s1, s2] = self.domain.starts
        [e1, e2] = self.domain.ends
        [n1, n2] = self.domain.npts
        theta_local = np.linspace(s2 * 2 * pi / n2, e2 * 2 * pi / n2, e2  - s2 + 1)
        rank_at_polar_edge = (s1 == 0)

        if out is None:
            y = self.codomain.zeros()
        else:
            assert isinstance(out, StencilVector)
            assert out.space is self.codomain
            out *= 0
            y = out

        if rank_at_polar_edge:
            angle_comm = x.space.cart.subcomm[1] if x.space.parallel else None
            sum_cos0, sum_sin0 = cos_sin_avg(theta_local, x, angle_comm, s2, e2, n2, 0)

            if self.transposed:
                sum_cos1, sum_sin1 = cos_sin_avg(theta_local, x, angle_comm, s2, e2, n2, 1)
                y[0, s2:e2 + 1] = sum_cos0 * np.cos(theta_local) + sum_sin0 * np.sin(theta_local) +\
                    x[1, s2:e2 + 1] - sum_cos1 * np.cos(theta_local) - sum_sin1 * np.sin(theta_local)
                y[1, s2:e2 + 1] = x[1, s2:e2 + 1]
            else:
                y[0, s2:e2 + 1] = sum_cos0 * np.cos(theta_local) + sum_sin0 * np.sin(theta_local)
                y[1, s2:e2 + 1] = x[0, s2:e2 + 1] + x[1, s2:e2 + 1] - y[0, s2:e2 + 1]

            y[2:e1 + 1, s2:e2 + 1] = x[2:e1 + 1, s2:e2 + 1]
        else:
            y[s1:e1 + 1, s2:e2 + 1] = x[s1:e1 + 1, s2:e2 + 1]

        y.update_ghost_regions()
        return y

    def transpose(self, conjugate=False):
        return C1PolarProjection_V1_00(self.W1, transposed=not self.transposed)

    @property
    def T(self):
        return self.transpose()

    def tosparse(self):

        [n01, n02] = self.domain.npts
        theta = np.linspace(0, 2 * pi, n02, endpoint=False)  # Warning not mpi!
        [s1, s2] = self.domain.starts
        [e1, e2] = self.domain.ends

        data, cols, rows = [], [], []
        rank_at_polar_edge = (s1 == 0)

        if rank_at_polar_edge:
            #block p
            data = (2 / n02) * toeplitz_columns_sym(theta, s2, e2, n02)
            # block I - p
            i_minus_p = -1 * data.copy()
            diag_js = np.arange(s2, e2 + 1)
            diag_pos = (diag_js - s2) * n02 + diag_js
            i_minus_p[diag_pos] += 1.0

            data = np.concatenate((data, i_minus_p))

            cols_theta = np.tile(np.arange(n02), e2 - s2 + 1)
            rows_theta = np.repeat(np.arange(s2, e2 + 1), n02)
            rows = np.concatenate((rows, rows_theta))
            cols = np.concatenate((cols, cols_theta))
            rows = np.concatenate((rows, rows_theta + 1 * n02))
            cols = np.concatenate((cols, cols_theta))

        # Assemble the rest of the matrix (identity block)
        start_s = max(s1, 1)
        end_s = e1 + 1
        i = np.arange(start_s, end_s)[:, None]
        j = np.arange(s2, e2 + 1)[None, :]
        local_rows = (i * n02 + j).ravel()

        data = np.concatenate((data, np.ones((e2 - s2 + 1) * (end_s - start_s))))
        cols = np.concatenate((cols, local_rows))
        rows = np.concatenate((rows, local_rows))

        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)

        P = coo_matrix((data, (rows, cols)), shape=[n01 * n02, n01 * n02], dtype=self.domain.dtype)
        P.eliminate_zeros()

        return P.T if self.transposed else P

    def toarray(self):
        return self.tosparse().toarray()


class C1PolarProjection_V1_10(LinearOperator):
    """
    Lower left block of P1.

    Parameters:
    -----------

    W1 : VectorFemSpace (former ProductFemSpace)
         Full tensor product spline space of the 1-forms S^{p1-1, p2} x S^{p1, p2-1}

    transposed : Boolean
         Switch between P1 and P1 transposed (default is False)
    """

    def __init__(self, W1, transposed=False):
        assert isinstance(W1, VectorFemSpace)

        self.W1 = W1
        self.transposed = transposed

    @property
    def domain(self):
        if self.transposed:
            return self.W1.coeff_space[1]
        return self.W1.coeff_space[0]

    @property
    def codomain(self):
        if self.transposed:
            return self.W1.coeff_space[0]
        return self.W1.coeff_space[1]

    @property
    def dtype(self):
        return float

    def forward_theta_diff_local(self, z, angle_comm):

        # Compute result[j] = z[j+1] - z[j] for distributed array z (periodic)
        result = np.empty_like(z)

        if angle_comm is None or angle_comm.size == 1:
            result[:] = np.roll(z, -1) - z
            return result

        result[:-1] = z[1:] - z[:-1]

        rank = angle_comm.rank
        size = angle_comm.size
        right_rank = (rank + 1) % size
        left_rank = (rank - 1) % size

        recvbuf = np.empty(1, dtype=z.dtype)
        angle_comm.Sendrecv(sendbuf=z[0], dest=left_rank, recvbuf=recvbuf, source=right_rank)

        result[-1] = recvbuf[0] - z[-1]
        return result

    def dot(self, x, out=None):
        assert isinstance(x, StencilVector)
        if not x.ghost_regions_in_sync:
            x.update_ghost_regions()

        # The number of radial basis functions is one less than
        # the number on the angular basis functions
        [s1, s2] = self.domain.starts
        [e1, e2] = self.domain.ends
        [n1, n2] = self.domain.npts
        theta_local = np.linspace(s2 * 2 * pi / n2, e2 * 2 * pi / n2, e2  - s2 + 1)
        rank_at_polar_edge = (s1 == 0)

        if out is None:
            y = self.codomain.zeros()
        else:
            assert isinstance(out, StencilVector)
            assert out.space is self.codomain
            out *= 0
            y = out

        if rank_at_polar_edge:
            angle_comm = x.space.cart.subcomm[1] if x.space.parallel else None

            if self.transposed:
                y[0, s2:e2 + 1] = x[1, s2 - 1:e2] - x[1, s2:e2 + 1]
                sum_cos, sum_sin = cos_sin_avg(theta_local, y, angle_comm, s2, e2, n2, 0)
                y[0, s2:e2 + 1] = np.cos(theta_local) * sum_cos + np.sin(theta_local) * sum_sin
                y[1:e1 + 1, s2:e2 + 1] = 0
            else:
                y[0, s2:e2 + 1] = 0
                sum_cos, sum_sin = cos_sin_avg(theta_local, x, angle_comm, s2, e2, n2, 0)
                row = np.cos(theta_local) * sum_cos + np.sin(theta_local) * sum_sin
                y[1, s2:e2 + 1] = self.forward_theta_diff_local(row, angle_comm)
                y[2:e1 + 1, s2:e2 + 1] = 0
        else:
            y[s1:e1 + 1, s2:e2 + 1] = 0


        y.update_ghost_regions()
        return y

    def transpose(self, conjugate=False):
        return C1PolarProjection_V1_10(self.W1, transposed=not self.transposed)

    @property
    def T(self):
        return self.transpose()

    def tosparse(self):

        if self.transposed:
            domain_P1_10 = self.codomain
            codomain_P1_10 = self.domain
        else:
            domain_P1_10 = self.domain
            codomain_P1_10 = self.codomain

        [n01, n02] = domain_P1_10.npts  # radial grid
        [n11, n12] = codomain_P1_10.npts  # angular grid
        [s1, s2] = domain_P1_10.starts
        [e1, e2] = domain_P1_10.ends

        rank_at_polar_edge = (s1 == 0)
        data, cols, rows = [], [], []
        if rank_at_polar_edge:
            local_j = np.arange(s2, e2 + 1)  # rows
            all_k = np.arange(n02)  # cols

            rows = np.repeat(local_j + n02, n02)
            cols = np.tile(all_k, len(local_j))

            theta = 2.0 * pi * np.arange(n02) / n02

            j = local_j[:, None]
            i = all_k[None, :]

            p = np.cos(theta[j - i])
            p_next = np.cos(theta[(j + 1) % n02 - i])
            q = (2 / n02) * (p_next - p)
            data = q.ravel(order='C')

        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)

        P = coo_matrix((data, (rows, cols)), shape=[n11 * n12, n01 * n02], dtype=self.domain.dtype)

        return P.T if self.transposed else P

    def toarray(self):
        return self.tosparse().toarray()



class C1PolarProjection_V1(BlockLinearOperator):
    """
    CONGA Projector P1 from the full spline space S^{p1-1, p2} x S^{p1, p2-1}
    on logical domain to U1 the pre-polar 1-forms splines. The associate matrix
    is square as in the CONGA approach we keep using the tensor B-spline basis,
    instead of the polar basis of Toshniwal. P1 enforces coefficient relations
    to be in U1.

    Parameters:
    -----------

    W1 : VectorFemSpace (ProductFemSpace)
         Full tensor product spline space of the 1-forms S^{p1-1, p2} x S^{p1, p2-1}

    transposed : Boolean
         Switch between P1 and P1 transposed (default is False)

    hbc : Boolean
         Switch on and off the imposition of homogeneous Dirichlet boundary
         conditions on the tangential (angular) direction (default is False)
    """

    def __init__(self, W1, transposed=False, hbc=False):
        assert isinstance(W1, VectorFemSpace)
        assert W1.symbolic_space.kind.name == 'hcurl'

        T1 = W1.coeff_space

        super().__init__(T1, T1)

        self[0, 0] = C1PolarProjection_V1_00(W1, transposed=transposed)
        self[1, 0] = C1PolarProjection_V1_10(W1, transposed=transposed)
        # self[0, 1] is 0 block
        # The lower right block is the same for C0 and C1 projections
        self[1, 1] = C0PolarProjection_V1_11(W1, transposed=transposed, hbc=hbc)

        self.W1 = W1
        self.transposed = transposed
        self.hbc = hbc


# -------------- 2-FORMS CONGA PROJECTOR P2 ----------------#
# Equals C0PolarProjection_V2

# ------------ strong and weak curl -----------------

from scipy.sparse.linalg import spilu, cg
from scipy.sparse.linalg import LinearOperator as SparseLinearOperator


class SparseCurlAsOperator(LinearOperator):

    def __init__(self, W1, W2, strong_curl_sp, M1=None, M2=None, strong=False, store_M1inv=False):
        assert isinstance(W1, VectorFemSpace)
        assert isinstance(W2, TensorFemSpace)
        assert isinstance(strong_curl_sp, spmatrix)

        self.W1 = W1
        self.W2 = W2

        self.strong = strong
        self._store_M1inv = store_M1inv

        if strong:
            self.curl_sp = strong_curl_sp

        else:
            self.curl_sp = strong_curl_sp.T  # dual curl (in the dual bases)
            assert isinstance(M1, LinearOperator)
            assert isinstance(M2, LinearOperator)
            self.M1_sp = M1.tosparse()
            self.M2_sp = M2.tosparse()
            if store_M1inv:
                self._M1inv_sp = sp_inv(self.M1_sp)
            else:
                self._M1_spilu = spilu(self.M1_sp)
                self._precond_M = SparseLinearOperator(self.M1_sp.shape, self._M1_spilu.solve)

    @property
    def domain(self):
        if self.strong:
            return self.W1.coeff_space
        else:
            return self.W2.coeff_space

    @property
    def codomain(self):
        if self.strong:
            return self.W2.coeff_space
        else:
            return self.W1.coeff_space

    @property
    def dtype(self):
        return float

    def toarray(self):
        # return self
        raise NotImplementedError('toarray() is not defined for this class.')

    def tosparse(self):
        # return self
        raise NotImplementedError('tosparse() is not defined for this class.')

    def transpose(self, conjugate=False):
        raise NotImplementedError('transpose() is not defined for this class.')

    # Warning: this dot method has to be revised for mpi!
    # the toeplitz multiplication requires all processes along the theta-dir to communicate.
    def dot(self, x, out=None):
        assert isinstance(x, Vector)
        if self.strong:
            Cx_arr = self.curl_sp @ x.toarray()
        else:
            tx_arr = self.M2_sp @ x.toarray()
            tCx_arr = self.curl_sp @ tx_arr
            if self._store_M1inv:
                Cx_arr = self._M1inv_sp.dot(tCx_arr)
            else:
                # Cx_arr = spsolve(self.M1_sp, tCx_arr)
                Cx_arr, exit_code = cg(self.M1_sp, tCx_arr, M=self._precond_M, rtol=1e-7)
                # Cx_arr = self._M1_spilu.solve(tCx_arr)
        if out is None:
            y = self.codomain.zeros()
        else:
            assert isinstance(out, Vector)
            assert out.space is self.codomain
            y = out

        y1 = array_to_psydac(Cx_arr, self.codomain)
        y1.copy(out=y)
        return y



