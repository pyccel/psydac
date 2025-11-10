import numpy as np

from scipy.linalg import toeplitz
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

        # Radial and angle sub-communicators (1D)
        self._cart = W0.coeff_space.cart
        if self._cart.is_parallel:
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
            # compute sum of points with s = 0 for the process
            local_sum = np.sum(x[0, s2:e2 + 1])

            if self._cart.is_parallel:
                from mpi4py import MPI
                local_sum = self._angle_comm.allreduce(local_sum, op=MPI.SUM)
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

        [s1, s2] = self.W1.coeff_space[0].starts
        [e1, e2] = self.W1.coeff_space[0].ends

        if out is None:
            y = self.W1.coeff_space[0].zeros()
        else:
            assert isinstance(out, StencilVector)
            assert out.space is self.W1.coeff_space[0]
            y = out

        y[:, s2:e2 + 1] = x[:, s2:e2 + 1]

        y.update_ghost_regions()
        return y

    def transpose(self, conjugate=False):
        return self

    @property
    def T(self):
        return self.transpose()

    def tosparse(self):

        [n01, n02] = self.domain.npts
        return sp_eye(n01 * n02)

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

        # Warning: this dot method has to be revised for mpi!
        # the toeplitz multiplication requires all processes along the theta dir. to communicate.

    def dot(self, x, out=None):
        assert isinstance(x, StencilVector)

        # The number of radial basis functions is one less than the number on the angular basis functions along dir x1
        [s1, s2] = self.domain.starts
        [e1, e2] = self.domain.ends
        [n1, n2] = self.domain.npts

        if out is None:
            y = self.codomain.zeros()
        else:
            assert isinstance(out, StencilVector)
            assert out.space is self.codomain
            y = out

        if self.transposed:
            y[0, s2:e2 + 1] = np.subtract(np.roll(x[1, s2:e2 + 1], 1), x[1, s2:e2 + 1])
            y[1:, s2:e2 + 1] = 0
        else:
            y[0, s2:e2 + 1] = 0
            y[1, s2:e2 + 1] = np.subtract(np.roll(x[0, s2:e2 + 1], -1), x[0, s2:e2 + 1])
            y[2:, s2:e2 + 1] = 0

        y.update_ghost_regions()
        return y

    def transpose(self, conjugate=False):
        return C0PolarProjection_V1_10(self.W1, transposed=not self.transposed)

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

        [n01, n02] = domain_P1_10.npts
        [n11, n12] = codomain_P1_10.npts

        dtype = domain_P1_10.dtype

        r = np.zeros(n02)
        r[:2] = [-1, 1]
        c = np.zeros(n02)
        c[::n02 - 1] = [-1, 1]

        d_block = toeplitz(c, r)
        P = lil_matrix((n11 * n12, n01 * n02), dtype=dtype)
        P[n12:2 * n12, :n02] = d_block

        # print(f'in C0CongaProjector1_10.tosparse : P.T.shape = {P.T.shape}, P.shape = {P.shape}, self.transposed = {self.transposed}')
        return P.T if self.transposed else P

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

        [s1, s2] = self.codomain.starts
        [e1, e2] = self.codomain.ends
        [n1, n2] = self.codomain.npts

        if out is None:
            y = self.W1.coeff_space[1].zeros()
        else:
            assert isinstance(out, StencilVector)
            assert out.space is self.codomain
            y = out

        y[0, s2:e2 + 1] = 0
        y[1, s2:e2 + 1] = 0
        y[2:, s2:e2 + 1] = x[2:, s2:e2 + 1]  # Identity block

        if self.hbc:
            if e1 == n1 - 1:
                y[e1, :] = 0.

        y.update_ghost_regions()
        return y

    def transpose(self, conjugate=False):
        return C0PolarProjection_V1_11(self.W1, transposed=not self.transposed, hbc=self.hbc)

    @property
    def T(self):
        return self.transpose()

    def tosparse(self):

        [n01, n02] = self.domain.npts
        [n11, n12] = self.codomain.npts
        dtype = self.domain.dtype

        P = lil_matrix((n11 * n12, n01 * n02), dtype=dtype)
        P[2 * n12:, 2 * n02:] = sp_eye((n11 - 2) * n12)
        if self.hbc:
            P[-n12:, -n02:] = 0
        return P.T if self.transposed else P

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

        if out is None:
            y = self.W2.coeff_space.zeros()
        else:
            assert isinstance(out, StencilVector)
            assert out.space is self.W2.coeff_space
            y = out

        if self.transposed:

            y[0, s2:e2 + 1] = x[1, s2:e2 + 1]
            y[1, s2:e2 + 1] = x[1, s2:e2 + 1]

        else:
            y[0, s2:e2 + 1] = 0
            y[1, s2:e2 + 1] = x[0, s2:e2 + 1] + x[1, s2:e2 + 1]

        y[2:, s2:e2 + 1] = x[2:, s2:e2 + 1]

        y.update_ghost_regions()
        return y

    def transpose(self, conjugate=False):
        return C0PolarProjection_V2(self.W2, transposed=not self.transposed)

    @property
    def T(self):
        return self.transpose()

    def tosparse(self):

        [n1, n2] = self.W2.coeff_space.npts

        data = np.ones(n1 * n2)

        cols = np.arange(n1 * n2)
        rows = np.tile(np.arange(n2, 2 * n2), 2)
        rows = np.concatenate((rows, np.arange(2 * n2, n1 * n2)))

        P = coo_matrix((data, (rows, cols)), shape=(n1 * n2, n1 * n2), dtype=self.W2.coeff_space.dtype)
        P.eliminate_zeros()

        return P.T if self.transposed else P

    def toarray(self):
        return self.tosparse().toarray()


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


