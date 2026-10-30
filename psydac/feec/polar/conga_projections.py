#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
r"""
CONGA projectors (as matrix-free `LinearOperator` objects) which allow using
standard tensor-product B-splines in 2D mapped domains with a polar singularity.

The domain is disk-like and parametrized by polar-like coordinates $(s, \theta)$.
Usually, the pole is at $s = 0$, while the boundary is at $s = L$. The $\theta$
coordinate is $2\pi$-periodic. We assume that the logical (a.k.a. parametric)
domain is the strip

$$\hat{\Omega} := [0, L] \times (\mathbb{R}/2\pi\mathbb{Z})$$

where tensor-product B-splines are defined. The physical (a.k.a. computational)
domain is $\Omega = F(\hat{\Omega})$, where $F: (s, \theta) \mapsto (x_1, x_2)$ is
the singular mapping which collapses the side $s = 0$ to a single point which
we call the "pole".

The application of these projectors to a vector of B-spline coefficients
forces the corresponding spline function to belong to the proper continuous
space in the de Rham complex, after the push-forward. Two different families
of projectors are available: `C0PolarProjection_V[0|1|2]` enforce $C^0$
continuity at the pole, while `C1PolarProjection_U[0|1|2]` enforce $C^1$.

Additionally, these operator can enforce homogeneous Dirichlet boundary
conditions at $s = L$. The exact meaning of these boundary conditions depends
on the space in the de Rham complex.

References
----------
The CONGA projectors are part of the "broken-FEEC" approach detailed here:

.. [1] Y. Güçlü, F. Patrizi, M. Campos Pinto, "A broken-FEEC framework for
   structure-preserving discretizations of polar domains with tensor-product
   splines", https://doi.org/10.48550/arXiv.2505.15996.
"""

import numpy as np
from numpy import pi
from scipy.sparse import coo_matrix

from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import VectorFemSpace
from psydac.linalg.basic import LinearOperator, Vector
from psydac.linalg.block import BlockLinearOperator
from psydac.linalg.stencil import StencilVector

__all__ = (
    "C0PolarProjection_V0",
    "C0PolarProjection_V1",
    "C0PolarProjection_V2",
    "C1PolarProjection_U0",
    "C1PolarProjection_U1",
    "C1PolarProjection_U2",
)


def polar_index(radial_dim, radial_index, angular_index):
    """Index tuple for (radial, angular) indices in the correct array ordering."""
    if radial_dim == 0:
        return (radial_index, angular_index)
    else:
        return (angular_index, radial_index)

def transpose_flat_index(k, nrows, ncols):
    """
    Map C-order flat indices of an (nrows, ncols) array to those of its
    (ncols, nrows) transpose.
    """
    row = k // ncols
    col = k % ncols
    return col * nrows + row

class C0PolarProjection_V0(LinearOperator):
    r"""
    Matrix-free representation of the CONGA projection matrix $\mathbb{P}_V^0$.

    This matrix acts on coefficient vectors in the full tensor-product spline
    basis of $\mathbb{S}_{p_1, p_2}(\hat{\Omega})$. It enforces the coefficient
    relations required for the corresponding spline, after push-forward, to
    be in the conforming spline space $V_h^0$.

    Parameters
    ----------
    W0 : TensorFemSpace
        The full tensor product spline space $\mathbb{S}_{p_1, p_2}(\hat{\Omega})$.

    transposed : bool, default=False
        If `True`, create the transposed projection matrix $(\mathbb{P}_V^0)^T$.

    hbc : bool, default=False
        If `True`, impose homogeneous Dirichlet boundary conditions.

    radial_dim : int, default=0
        Index of the logical dimension corresponding to the radial coordinate
        $s$. The other dimension is the (periodic) angular coordinate $\theta$.
        Use `radial_dim=1` for spaces built on $(\theta, s)$.
    """
    def __init__(self, W0, *, transposed=False, hbc=False, radial_dim=0):
        assert isinstance(W0, TensorFemSpace)
        assert radial_dim in (0, 1)
        assert W0.coeff_space.periods[radial_dim] == False
        assert W0.coeff_space.periods[1 - radial_dim] == True

        self.W0 = W0
        self.transposed = transposed
        self.hbc = hbc
        self.radial_dim = radial_dim
        self.angle_dim = 1 - radial_dim

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

        rdim, adim = self.radial_dim, self.angle_dim
        V = self.W0.coeff_space
        sr, sa = V.starts[rdim], V.starts[adim]
        er, ea = V.ends[rdim], V.ends[adim]
        nr, na = V.npts[rdim], V.npts[adim]
        rank_at_polar_edge = sr == 0
        rank_at_outer_edge = er == nr - 1

        if out is None:
            y = V.zeros()
        else:
            assert isinstance(out, StencilVector)
            assert out.space is V
            out *= 0
            y = out

        ang = slice(sa, ea + 1)

        if rank_at_polar_edge:
            # regions of the local coefficient array near the pole,
            # in the correct array ordering
            pole  = polar_index(rdim, 0, ang)                  # s = 0
            inner = polar_index(rdim, slice(1, er + 1), ang)   # s > 0

            # compute sum of points with s = 0 for the process
            local_sum = np.sum(x[pole])
            if x.space.parallel:
                from mpi4py import MPI

                angle_comm = x.space.cart.subcomm[adim]
                local_sum = angle_comm.allreduce(local_sum, op=MPI.SUM)
            # compute average of all points with s = 0
            y[pole] = local_sum / na
            y[inner] = x[inner]
        else:
            # all radial indices of the rank
            owned = polar_index(rdim, slice(sr, er + 1), ang)
            y[owned] = x[owned]

        if self.hbc:
            if rank_at_outer_edge:
                outer = polar_index(rdim, er, slice(None))  # s = L
                y[outer] = 0.0

        y.update_ghost_regions()
        return y

    def transpose(self, conjugate=False):
        return C0PolarProjection_V0(
            self.W0, transposed=not self.transposed, hbc=self.hbc,
            radial_dim=self.radial_dim
        )

    def tosparse(self):

        # Matrix size is nr*na. Rows with radial index sr <= i <= er and angular
        # index sa <= j <= ea are filled for the current process. The matrix is
        # assembled with (radial, angular) C-style flat indices i*na + j, which
        # are remapped to j*nr + i if the radial dimension is the second one.

        rdim, adim = self.radial_dim, self.angle_dim
        V = self.W0.coeff_space
        sr, sa = V.starts[rdim], V.starts[adim]
        er, ea = V.ends[rdim], V.ends[adim]
        nr, na = V.npts[rdim], V.npts[adim]
        rank_at_polar_edge = sr == 0
        rank_at_outer_edge = er == nr - 1
        data, cols, rows = [], [], []

        # Assemble the matrix entries which contribute to the polar edge (s = 0)
        len_theta = ea - sa + 1
        if rank_at_polar_edge:
            data = np.tile((1 / na) * np.ones(na), len_theta)
            rows = np.repeat(np.arange(sa, ea + 1), na)
            cols = np.tile(np.arange(na), len_theta)

        # Assemble the rest of the matrix (identity block)
        # We do not need entries with s1 = 0 (already accounted for)
        start_s = sr + 1 if rank_at_polar_edge else sr
        end_s = er if (rank_at_outer_edge and self.hbc) else er + 1

        i = np.arange(start_s, end_s)[:, None]
        j = np.arange(sa, ea + 1)[None, :]
        # rows of identity owned by the process
        local_rows = (i * na + j).ravel()

        data = np.concatenate((data, np.ones(len_theta * (end_s - start_s))))
        cols = np.concatenate((cols, local_rows))
        rows = np.concatenate((rows, local_rows))

        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)

        if rdim == 1:
            rows = transpose_flat_index(rows, nr, na)
            cols = transpose_flat_index(cols, nr, na)

        P = coo_matrix(
            (data, (rows, cols)),
            shape=[nr * na, nr * na],
            dtype=self.W0.coeff_space.dtype,
        )
        P.eliminate_zeros()

        # P is symmetric so we always return P no matter self.transposed
        return P

    def toarray(self):
        return self.tosparse().toarray()


# --------- 1-FORMS CONGA PROJECTOR P1 ----------#
# It is a BlockLinearOperator acting on the s- and theta-components of a
# 1-form. The s-component has index radial_dim in W1 and the theta-component
# has index 1 - radial_dim. P1 has three non-zero blocks:
#
#   s_s         : s-component     -> s-component,     identity
#   theta_s     : s-component     -> theta-component, coupling at the pole
#   theta_theta : theta-component -> theta-component, diagonal
#
#   radial_dim = 0, (s, theta):              radial_dim = 1, (theta, s):
#    _____________________________            _____________________________
#   |              |              |          |              |              |
#   |     s_s      |      0       |          | theta_theta  |   theta_s    |
#   |______________|______________|          |______________|______________|
#   |              |              |          |              |              |
#   |   theta_s    | theta_theta  |          |      0       |     s_s      |
#   |______________|______________|          |______________|______________|
#
# In the transposed matrix, the transpose of theta_s is in the other
# off-diagonal position.


class C0PolarProjection_V1_s_s(LinearOperator):
    r"""
    Matrix-free representation of the block of the CONGA projection matrix
    $\mathbb{P}_V^1$ which maps the $s$-component of a 1-form to itself.
    This block is the identity.

    Parameters
    ----------
    W1 : VectorFemSpace
        The full tensor product spline space of 1-forms
        $\mathbb{S}_{p_1-1, p_2}(\hat{\Omega})
        \times \mathbb{S}_{p_1, p_2-1}(\hat{\Omega})$.

    radial_dim : int, default=0
        Index of the logical dimension corresponding to the radial coordinate. 
        With `radial_dim=0` this is the upper-left block of $\mathbb{P}_V^1$, with
        `radial_dim=1` it is the lower-right block.
    """
    def __init__(self, W1, radial_dim=0):
        assert isinstance(W1, VectorFemSpace)
        assert radial_dim in (0, 1)

        self.W1 = W1
        self.radial_dim = radial_dim

    @property
    def domain(self):
        return self.W1.coeff_space[self.radial_dim]

    @property
    def codomain(self):
        return self.W1.coeff_space[self.radial_dim]

    @property
    def dtype(self):
        return float

    # This block is the identity, so it does not depend on the ordering of
    # the logical coordinates.
    def dot(self, x, out=None):
        assert isinstance(x, StencilVector)

        if not x.ghost_regions_in_sync:
            x.update_ghost_regions()

        [s1, s2] = self.domain.starts
        [e1, e2] = self.domain.ends

        if out is None:
            y = self.codomain.zeros()
        else:
            assert isinstance(out, StencilVector)
            assert out.space is self.codomain
            out *= 0
            y = out

        y[s1 : e1 + 1, s2 : e2 + 1] = x[s1 : e1 + 1, s2 : e2 + 1]

        y.update_ghost_regions()
        return y

    def transpose(self, conjugate=False):
        return self

    def tosparse(self):

        [n1, n2] = self.domain.npts
        [s1, s2] = self.domain.starts
        [e1, e2] = self.domain.ends

        i = np.arange(s1, e1 + 1)[:, None]
        j = np.arange(s2, e2 + 1)[None, :]
        local_rows = (i * n2 + j).ravel()
        data = np.ones((e1 - s1 + 1) * (e2 - s2 + 1))

        local_rows = np.asarray(local_rows, dtype=np.int64)

        P = coo_matrix(
            (data, (local_rows, local_rows)),
            shape=[n1 * n2, n1 * n2],
            dtype=self.domain.dtype,
        )
        P.eliminate_zeros()

        return P

    def toarray(self):
        return self.tosparse().toarray()


class C0PolarProjection_V1_theta_s(LinearOperator):
    r"""
    Matrix-free representation of the block of the CONGA projection matrix
    $\mathbb{P}_V^1$ which maps the $s$-component of a 1-form to its
    $\theta$-component. 

    Parameters
    ----------
    W1 : VectorFemSpace
        The full tensor product spline space of 1-forms
        $\mathbb{S}_{p_1-1, p_2}(\hat{\Omega})
        \times \mathbb{S}_{p_1, p_2-1}(\hat{\Omega})$.

    transposed : bool, default=False
        If `True`, create block of transposed projection matrix
        $(\mathbb{P}_V^1)^T$, which maps the $\theta$-component of a 1-form
        to its $s$-component.

    radial_dim : int, default=0
        Index of the logical dimension corresponding to the radial coordinate
        $s$. With `radial_dim=0` this is the lower-left block of $\mathbb{P}_V^1$
        (upper-right if transposed), with `radial_dim=1` it is the upper-right
        block (lower-left if transposed).
    """
    def __init__(self, W1, transposed=False, radial_dim=0):
        assert isinstance(W1, VectorFemSpace)
        assert radial_dim in (0, 1)

        self.W1 = W1
        self.transposed = transposed
        self.radial_dim = radial_dim
        self.angle_dim = 1 - radial_dim

    @property
    def domain(self):
        # s-component, or theta-component if transposed
        idx = self.angle_dim if self.transposed else self.radial_dim
        return self.W1.coeff_space[idx]

    @property
    def codomain(self):
        # theta-component, or s-component if transposed
        idx = self.radial_dim if self.transposed else self.angle_dim
        return self.W1.coeff_space[idx]

    @property
    def dtype(self):
        return float

    def dot(self, x, out=None):
        assert isinstance(x, StencilVector)

        if not x.ghost_regions_in_sync:
            x.update_ghost_regions()

        rdim, adim = self.radial_dim, self.angle_dim
        V = self.domain
        sr, sa = V.starts[rdim], V.starts[adim]
        ea = V.ends[adim]
        rank_at_polar_edge = sr == 0

        if out is None:
            y = self.codomain.zeros()
        else:
            assert isinstance(out, StencilVector)
            assert out.space is self.codomain
            out *= 0
            y = out

        if rank_at_polar_edge:
            # the cells are periodic in angular dim, so we use ghost regions            
            ang      = slice(sa, ea + 1)
            ang_next = slice(sa + 1, ea + 2)
            ang_prev = slice(sa - 1, ea)

            if self.transposed:
                # (s-component)[0, j] = (theta-component)[1, j-1] - (theta-component)[1, j]
                s_ring0          = polar_index(rdim, 0, ang)
                theta_ring1      = polar_index(rdim, 1, ang)
                theta_ring1_prev = polar_index(rdim, 1, ang_prev)
                y[s_ring0] = x[theta_ring1_prev] - x[theta_ring1]
            else:
                # (theta-component)[1, j] = (s-component)[0, j+1] - (s-component)[0, j]
                theta_ring1  = polar_index(rdim, 1, ang)
                s_ring0      = polar_index(rdim, 0, ang)
                s_ring0_next = polar_index(rdim, 0, ang_next)
                y[theta_ring1] = x[s_ring0_next] - x[s_ring0]

        y.update_ghost_regions()
        return y

    def transpose(self, conjugate=False):
        return C0PolarProjection_V1_theta_s(
            self.W1, transposed=not self.transposed, radial_dim=self.radial_dim
        )

    def tosparse(self):

        rdim, adim = self.radial_dim, self.angle_dim
        nr_dom, na = self.domain.npts[rdim], self.domain.npts[adim]
        nr_cod = self.codomain.npts[rdim]
        sr, sa = self.domain.starts[rdim], self.domain.starts[adim]
        ea = self.domain.ends[adim]
        rank_at_polar_edge = sr == 0

        data, cols, rows = [], [], []

        # matrix d (derivatives of periodic splines)
        # for example: n_theta = 3, single rank at polar edge, not transposed.
        # Then, in (radial, angular) flat indices:
        # data = [-1 1 -1 1 -1 1], rows = [3 3 4 4 5 5], cols = [0 1 1 2 2 0]
        if rank_at_polar_edge:
            len_theta = ea - sa + 1
            data = np.tile([-1, 1], len_theta)
            k = np.arange(sa, ea + 1)
            if self.transposed:
                rows = np.repeat(k, 2)
                cols = np.column_stack((k, (k - 1) % na)).ravel() + na
            else:
                rows = np.repeat(k, 2) + na
                cols = np.column_stack((k, (k + 1) % na)).ravel()

        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)

        if rdim == 1:
            rows = transpose_flat_index(rows, nr_cod, na)
            cols = transpose_flat_index(cols, nr_dom, na)

        P = coo_matrix(
            (data, (rows, cols)),
            shape=[nr_cod * na, nr_dom * na],
            dtype=self.domain.dtype,
        )
        P.eliminate_zeros()
        return P

    def toarray(self):
        return self.tosparse().toarray()


class C0PolarProjection_V1_theta_theta(LinearOperator):
    r"""
    Matrix-free representation of the block of the CONGA projection matrix
    $\mathbb{P}_V^1$ which maps the $\theta$-component of a 1-form to itself.

    Parameters
    ----------
    W1 : VectorFemSpace
        The full tensor product spline space of 1-forms
        $\mathbb{S}_{p_1-1, p_2}(\hat{\Omega})
        \times \mathbb{S}_{p_1, p_2-1}(\hat{\Omega})$.

    transposed : bool, default=False
        If `True`, create block of transposed projection matrix
        $(\mathbb{P}_V^1)^T$.

    hbc : bool, default=False
        If `True`, impose homogeneous Dirichlet boundary conditions on the
        tangential (angular) component of the field.

    radial_dim : int, default=0
        Index of the logical dimension corresponding to the radial coordinate
        $s$. The $\theta$-component is the other component of `W1`. With
        `radial_dim=0` this is the lower-right block of $\mathbb{P}_V^1$, with
        `radial_dim=1` it is the upper-left block.
    """
    def __init__(self, W1, transposed=False, hbc=False, radial_dim=0):
        assert isinstance(W1, VectorFemSpace)
        assert radial_dim in (0, 1)

        self.W1 = W1
        self.transposed = transposed
        self.hbc = hbc
        self.radial_dim = radial_dim
        self.angle_dim = 1 - radial_dim

    @property
    def domain(self):
        return self.W1.coeff_space[self.angle_dim]

    @property
    def codomain(self):
        return self.W1.coeff_space[self.angle_dim]

    @property
    def dtype(self):
        return float

    def dot(self, x, out=None):
        assert isinstance(x, StencilVector)

        if not x.ghost_regions_in_sync:
            x.update_ghost_regions()

        rdim, adim = self.radial_dim, self.angle_dim
        V = self.domain
        sr, sa = V.starts[rdim], V.starts[adim]
        er, ea = V.ends[rdim], V.ends[adim]
        nr = V.npts[rdim]
        rank_at_polar_edge = sr == 0
        rank_at_outer_edge = er == nr - 1

        if out is None:
            y = V.zeros()
        else:
            assert isinstance(out, StencilVector)
            assert out.space is V
            out *= 0
            y = out

        ang = slice(sa, ea + 1)
        if rank_at_polar_edge:
            # radial indices 0 and 1 stay zero, identity on the others
            inner = polar_index(rdim, slice(2, er + 1), ang)
            y[inner] = x[inner]
        else:
            # all radial indices of the rank
            owned = polar_index(rdim, slice(sr, er + 1), ang)
            y[owned] = x[owned]

        if self.hbc and rank_at_outer_edge:
            outer = polar_index(rdim, er, slice(None)) # s = L
            y[outer] = 0.0

        y.update_ghost_regions()
        return y

    def transpose(self, conjugate=False):
        return C0PolarProjection_V1_theta_theta(
            self.W1, transposed=not self.transposed, hbc=self.hbc,
            radial_dim=self.radial_dim
        )

    def tosparse(self):

        rdim, adim = self.radial_dim, self.angle_dim
        V = self.domain
        sr, sa = V.starts[rdim], V.starts[adim]
        er, ea = V.ends[rdim], V.ends[adim]
        nr, na = V.npts[rdim], V.npts[adim]
        rank_at_outer_edge = er == nr - 1

        # identity on radial indices 2 <= i < nr (i < nr - 1 with hbc),
        # which is a block of size na*(nr - 2) (na*(nr - 3) with hbc)
        start_s = max(2, sr)
        end_s = er if (rank_at_outer_edge and self.hbc) else er + 1
        len_theta = ea - sa + 1
        data = np.ones(len_theta * (end_s - start_s))

        i = np.arange(start_s, end_s)[:, None]
        j = np.arange(sa, ea + 1)[None, :]
        local_rows = (i * na + j).ravel()
        local_rows = np.asarray(local_rows, dtype=np.int64)

        if rdim == 1:
            local_rows = transpose_flat_index(local_rows, nr, na)

        P = coo_matrix(
            (data, (local_rows, local_rows)),
            shape=[nr * na, nr * na],
            dtype=V.dtype,
        )
        P.eliminate_zeros()
        return P

    def toarray(self):
        return self.tosparse().toarray()


class C0PolarProjection_V1(BlockLinearOperator):
    r"""
    Matrix-free representation of the CONGA projection matrix $\mathbb{P}_V^1$.

    This matrix acts on coefficient vectors in the full tensor-product spline
    basis of $\mathbb{S}_{p_1-1, p_2}(\hat{\Omega})
    \times \mathbb{S}_{p_1, p_2-1}(\hat{\Omega})$. It enforces the coefficient
    relations required for the corresponding spline, after push-forward, to
    be in the conforming spline space $V_h^1$.

    Parameters
    ----------
    W1 : VectorFemSpace
        The full tensor product spline space of 1-forms
        $\mathbb{S}_{p_1-1, p_2}(\hat{\Omega})
        \times \mathbb{S}_{p_1, p_2-1}(\hat{\Omega})$.

    transposed : bool, default=False
        If `True`, create the transposed projection matrix $(\mathbb{P}_V^1)^T$.

    hbc : bool, default=False
        If `True`, impose homogeneous Dirichlet boundary conditions on the
        tangential (angular) component of the field.
    
    radial_dim : int, default=0
        Index of the logical dimension corresponding to the radial coordinate
        $s$, which is also the index of the $s$-component in `W1`. With
        `radial_dim=0` the matrix is block lower triangular, with
        `radial_dim=1` it is block upper triangular (and vice versa if
        transposed).
    """
    def __init__(self, W1, transposed=False, hbc=False, radial_dim=0):
        assert isinstance(W1, VectorFemSpace)
        assert W1.symbolic_space.kind.name == "hcurl"
        assert radial_dim in (0, 1)

        T1 = W1.coeff_space

        super().__init__(T1, T1)

        # indices of the s- and theta-components in W1
        rdim, adim = radial_dim, 1 - radial_dim

        self[rdim, rdim] = C0PolarProjection_V1_s_s(W1, radial_dim=radial_dim)
        self[adim, adim] = C0PolarProjection_V1_theta_theta(
            W1, transposed=transposed, hbc=hbc, radial_dim=radial_dim
        )

        P_theta_s = C0PolarProjection_V1_theta_s(
            W1, transposed=transposed, radial_dim=radial_dim
        )

        if transposed:
            self[rdim, adim] = P_theta_s

        else:
            self[adim, rdim] = P_theta_s

        self.W1 = W1
        self.transposed = transposed
        self.hbc = hbc
        self.radial_dim = radial_dim

# ---------------- 2-FORMS CONGA PROJECTOR P2 ----------------#
class C0PolarProjection_V2(LinearOperator):
    r"""
    Matrix-free representation of the CONGA projection matrix $\mathbb{P}_V^2$.

    This matrix acts on coefficient vectors in the full tensor-product spline
    basis of $\mathbb{S}_{p_1-1, p_2-1}(\hat{\Omega})$. It enforces the coefficient
    relations required for the corresponding spline, after push-forward, to
    be in the conforming spline space $V_h^2$.

    Parameters
    ----------
    W2 : TensorFemSpace
        The full tensor product spline space of 2-forms
        $\mathbb{S}_{p_1-1, p_2-1}(\hat{\Omega})$.

    transposed : bool, default=False
        If `True`, create the transposed projection matrix $(\mathbb{P}_V^2)^T$.
    
    radial_dim : int, default=0
        Index of the logical dimension corresponding to the radial coordinate
        $s$. The other dimension is the (periodic) angular coordinate $\theta$.
        Use `radial_dim=1` for spaces built on $(\theta, s)$.
    """
    def __init__(self, W2, transposed=False, radial_dim=0):
        assert isinstance(W2, TensorFemSpace)
        assert radial_dim in (0, 1)
        assert W2.coeff_space.periods[radial_dim] == False
        assert W2.coeff_space.periods[1 - radial_dim] == True

        self.W2 = W2
        self.transposed = transposed
        self.radial_dim = radial_dim
        self.angle_dim = 1 - radial_dim

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

        rdim, adim = self.radial_dim, self.angle_dim
        V = self.W2.coeff_space
        sr, sa = V.starts[rdim], V.starts[adim]
        er, ea = V.ends[rdim], V.ends[adim]
        rank_at_polar_edge = sr == 0

        if out is None:
            y = V.zeros()
        else:
            assert isinstance(out, StencilVector)
            assert out.space is V
            out *= 0
            y = out

        ang = slice(sa, ea + 1)

        if rank_at_polar_edge:
            # regions of the local coefficient array near the pole, in the correct array ordering
            ring0 = polar_index(rdim, 0, ang)                  # first radial index, touching the pole
            ring1 = polar_index(rdim, 1, ang)                  # second radial index
            inner = polar_index(rdim, slice(2, er + 1), ang)   # radial indices > 1 owned by the rank

            if self.transposed:
                y[ring0] = x[ring1]
                y[ring1] = x[ring1]
            else:
                y[ring0] = 0
                y[ring1] = x[ring0] + x[ring1]
            y[inner] = x[inner]
        else:
            # all radial indices of the rank
            owned = polar_index(rdim, slice(sr, er + 1), ang)
            y[owned] = x[owned]

        y.update_ghost_regions()
        return y

    def transpose(self, conjugate=False):
        return type(self)(
            self.W2, transposed=not self.transposed, radial_dim=self.radial_dim
        )

    def tosparse(self):

        # The matrix is assembled with (radial, angular) C-style flat indices
        # i*na + j for radial index i and angular index j, so that radial
        # index 0 has flat indices k < na and radial index 1 has na <= k < 2*na.

        rdim, adim = self.radial_dim, self.angle_dim
        V = self.W2.coeff_space
        sr, sa = V.starts[rdim], V.starts[adim]
        er, ea = V.ends[rdim], V.ends[adim]
        nr, na = V.npts[rdim], V.npts[adim]

        data = np.ones((er - sr + 1) * (ea - sa + 1))

        i = np.arange(sr, er + 1)[:, None]
        j = np.arange(sa, ea + 1)[None, :]
        local_rows = (i * na + j).ravel()
        if self.transposed:
            rows = local_rows
            cols = local_rows[local_rows < na] + na
        else:
            rows_to_repeat = local_rows[(local_rows >= na) & (local_rows < 2 * na)]
            rows = np.tile(rows_to_repeat, 2)
            cols = rows_to_repeat - na
            rows = np.concatenate((rows, local_rows[local_rows >= 2 * na]))
        cols = np.concatenate((cols, local_rows[local_rows >= na]))

        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)

        # If the radial dimension is the second one, the coefficient array is
        # ordered (angular, radial): remap flat indices i*na + j to j*nr + i
        if rdim == 1:
            rows = transpose_flat_index(rows, nr, na)
            cols = transpose_flat_index(cols, nr, na)

        P = coo_matrix(
            (data, (rows, cols)),
            shape=(nr * na, nr * na),
            dtype=V.dtype,
        )
        P.eliminate_zeros()

        return P

    def toarray(self):
        return self.tosparse().toarray()


# =============================================================================#
#                        C1 POLAR SPLINE COMPLEX (C1P)                        #
# =============================================================================#


def cos_sin_avg(theta, x, angle_comm, s2, e2, n2, i, radial_dim=0):
    """
    Compute weighted averages  in parallel:
    sum (2/n2) * cos(theta_k) * x_ik
    sum (2/n2) * sin(theta_k) * x_ik
    over all k belonging to the process, for fixed i (0 or 1)
    then sum over all processes
    """
    from mpi4py import MPI

    ring = x[polar_index(radial_dim, i, slice(s2, e2 + 1))]
    sum_cos = (2 / n2) * np.dot(np.cos(theta), ring)
    sum_sin = (2 / n2) * np.dot(np.sin(theta), ring)
    buf = np.array([sum_cos, sum_sin])

    if angle_comm is not None:
        buf = angle_comm.allreduce(buf, op=MPI.SUM)
    return buf


# --------- 0-FORMS CONGA PROJECTOR P0 ----------#


# matrix p in the paper notation
def toeplitz_columns_sym(t, s2, e2, n2):

    i = np.arange(n2)[:, None]  # (n2, 1)
    j = np.arange(s2, e2 + 1)[None, :]  # (1, m)
    idx = np.abs(i - j).ravel("F")  # (n2, m)
    return np.cos(t[idx])


class C1PolarProjection_U0(LinearOperator):
    r"""
    Matrix-free representation of the CONGA projection matrix $\mathbb{P}_U^0$.

    This matrix acts on coefficient vectors in the full tensor-product spline
    basis of $\mathbb{S}_{p_1, p_2}(\hat{\Omega})$. It enforces the coefficient
    relations required for the corresponding spline, after push-forward, to
    be in the conforming spline space $U_h^0$.

    Parameters
    ----------
    W0 : TensorFemSpace
        The full tensor product spline space $\mathbb{S}_{p_1, p_2}(\hat{\Omega})$.

    transposed : bool, default=False
        If `True`, create the transposed projection matrix $(\mathbb{P}_U^0)^T$.

    hbc : bool, default=False
        If `True`, impose homogeneous Dirichlet boundary conditions.
    
    radial_dim : int, default=0
        Index of the logical dimension corresponding to the radial coordinate
        $s$. The other dimension is the (periodic) angular coordinate $\theta$.
        Use `radial_dim=1` for spaces built on $(\theta, s)$.
    """
    def __init__(self, W0, *, transposed=False, hbc=False, radial_dim=0):
        assert isinstance(W0, TensorFemSpace)
        assert radial_dim in (0, 1)
        assert W0.coeff_space.periods[radial_dim] == False
        assert W0.coeff_space.periods[1 - radial_dim] == True

        self.W0 = W0
        self.transposed = transposed
        self.hbc = hbc
        self.radial_dim = radial_dim
        self.angle_dim = 1 - radial_dim

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

        rdim, adim = self.radial_dim, self.angle_dim
        V = self.W0.coeff_space
        sr, sa = V.starts[rdim], V.starts[adim]
        er, ea = V.ends[rdim], V.ends[adim]
        nr, na = V.npts[rdim], V.npts[adim]
        theta_local = np.linspace(sa * 2 * pi / na, ea * 2 * pi / na, ea - sa + 1)
        rank_at_polar_edge = sr == 0
        rank_at_outer_edge = er == nr - 1

        if out is None:
            y = V.zeros()
        else:
            assert isinstance(out, StencilVector)
            assert out.space is V
            out *= 0
            y = out

        ang = slice(sa, ea + 1)

        if rank_at_polar_edge:
            # regions of the coefficient array local to the process, in the correct array ordering
            pole  = polar_index(rdim, 0, ang)                  # s = 0
            ring1 = polar_index(rdim, 1, ang)                  # first ring around the pole
            inner = polar_index(rdim, slice(2, er + 1), ang)   # s > 1, rank at polar edge

            # compute sums of points with s = 0, 1 for the process
            x0 = np.sum(x[pole]) / na
            x1 = np.sum(x[ring1]) / na

            angle_comm = x.space.cart.subcomm[adim] if x.space.parallel else None
            sum_cos, sum_sin = cos_sin_avg(
                theta_local, x, angle_comm, sa, ea, na, 1, radial_dim=rdim
            )

            if x.space.parallel:

                from mpi4py import MPI

                # global averages
                x0 = angle_comm.allreduce(x0, op=MPI.SUM)
                x1 = angle_comm.allreduce(x1, op=MPI.SUM)

            if self.transposed:
                y[pole] = x0 + x1
                y[ring1] = (
                    + np.cos(theta_local) * sum_cos
                    + np.sin(theta_local) * sum_sin
                )
            else:
                y[pole] = x0
                y[ring1] = (
                    x0 + np.cos(theta_local) * sum_cos
                    + np.sin(theta_local) * sum_sin
                )
            y[inner] = x[inner]
        else:
            owned = polar_index(rdim, slice(sr, er + 1), ang)  # all radial indices of the rank
            y[owned] = x[owned]

        if self.hbc and rank_at_outer_edge:
            outer = polar_index(rdim, er, slice(None))  # s = L
            y[outer] = 0.0

        y.update_ghost_regions()
        return y

    def transpose(self, conjugate=False):
        return C1PolarProjection_U0(
            self.W0, transposed=not self.transposed, hbc=self.hbc,
            radial_dim=self.radial_dim
        )

    def tosparse(self):

        # The matrix is assembled with (radial, angular) C-style flat indices
        # i*na + j for radial index i and angular index j.

        rdim, adim = self.radial_dim, self.angle_dim
        V = self.W0.coeff_space
        sr, sa = V.starts[rdim], V.starts[adim]
        er, ea = V.ends[rdim], V.ends[adim]
        nr, na = V.npts[rdim], V.npts[adim]
        theta = np.linspace(0, 2 * pi, na, endpoint=False)

        rank_at_polar_edge = sr == 0
        rank_at_outer_edge = er == nr - 1
        data, cols, rows = [], [], []

        if rank_at_polar_edge:
            # matrix of size na*na with all entries equal to 1 / na
            data = (1 / na) * np.ones(na * (ea - sa + 1))
            rows = np.tile(np.repeat(np.arange(sa, ea + 1), na), 1)
            cols = np.tile(np.arange(na), ea - sa + 1)
        if er > 1 > sr:
            # matrix of size na*na with all entries equal to 1 / na
            d_block2 = (1 / na) * np.ones((ea - sa + 1) * na)
            data = np.concatenate((data, d_block2))
            rows = np.concatenate(
                (rows, np.repeat(np.arange(na + sa, na + ea + 1), na))
            )
            cols = np.concatenate((cols, np.tile(np.arange(na), ea - sa + 1)))

            # matrix p
            d_block2 = 2 / na * toeplitz_columns_sym(theta, sa, ea, na)
            data = np.concatenate((data, d_block2))
            rows = np.concatenate(
                (rows, np.repeat(np.arange(na + sa, na + ea + 1), na))
            )
            cols = np.concatenate((cols, np.tile(np.arange(na, 2 * na), ea - sa + 1)))

        # Assemble the rest of the matrix (identity block)
        start_s = max(sr, 2)
        end_s = er if (rank_at_outer_edge and self.hbc) else er + 1
        i = np.arange(start_s, end_s)[:, None]
        j = np.arange(sa, ea + 1)[None, :]
        local_rows = (i * na + j).ravel()

        data = np.concatenate((data, np.ones((ea - sa + 1) * (end_s - start_s))))
        cols = np.concatenate((cols, local_rows))
        rows = np.concatenate((rows, local_rows))

        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)

        # If the radial dimension is the second one, the coefficient array is
        # ordered (angular, radial): remap flat indices i*na + j to j*nr + i
        if rdim == 1:
            rows = transpose_flat_index(rows, nr, na)
            cols = transpose_flat_index(cols, nr, na)

        P = coo_matrix(
            (data, (rows, cols)),
            shape=[nr * na, nr * na],
            dtype=V.dtype,
        )
        P.eliminate_zeros()

        return P.T if self.transposed else P

    def toarray(self):
        return self.tosparse().toarray()


# --------- 1-FORMS CONGA PROJECTOR P1 ----------#
# Same block structure as the C0 projector P_V^1 (see above)
#
#   s_s         : s-component     -> s-component,     p at the pole
#   theta_s     : s-component     -> theta-component, q = d p at the pole
#   theta_theta : theta-component -> theta-component, same block as for P_V^1


class C1PolarProjection_U1_s_s(LinearOperator):
    r"""
    Matrix-free representation of the block of the CONGA projection matrix
    $\mathbb{P}_U^1$ which maps the $s$-component of a 1-form to itself.

    Parameters
    ----------
    W1 : VectorFemSpace
        The full tensor product spline space of 1-forms
        $\mathbb{S}_{p_1-1, p_2}(\hat{\Omega})
        \times \mathbb{S}_{p_1, p_2-1}(\hat{\Omega})$.

    transposed : bool, default=False
        If `True`, create block of transposed projection matrix
        $(\mathbb{P}_U^1)^T$.

    radial_dim : int, default=0
        Index of the logical dimension corresponding to the radial coordinate
        $s$, which is also the index of the $s$-component in `W1`. With
        `radial_dim=0` this is the upper-left block of $\mathbb{P}_U^1$, with
        `radial_dim=1` it is the lower-right block.
    """
    def __init__(self, W1, transposed=False, radial_dim=0):
        assert isinstance(W1, VectorFemSpace)
        assert radial_dim in (0, 1)

        self.W1 = W1
        self.transposed = transposed
        self.radial_dim = radial_dim
        self.angle_dim = 1 - radial_dim

    @property
    def domain(self):
        return self.W1.coeff_space[self.radial_dim]

    @property
    def codomain(self):
        return self.W1.coeff_space[self.radial_dim]

    @property
    def dtype(self):
        return float

    def dot(self, x, out=None):
        assert isinstance(x, StencilVector)

        rdim, adim = self.radial_dim, self.angle_dim
        V = self.domain
        sr, sa = V.starts[rdim], V.starts[adim]
        er, ea = V.ends[rdim], V.ends[adim]
        na = V.npts[adim]
        theta_local = np.linspace(sa * 2 * pi / na, ea * 2 * pi / na, ea - sa + 1)
        rank_at_polar_edge = sr == 0

        if out is None:
            y = self.codomain.zeros()
        else:
            assert isinstance(out, StencilVector)
            assert out.space is self.codomain
            out *= 0
            y = out

        ang = slice(sa, ea + 1)

        if rank_at_polar_edge:
            ring0 = polar_index(rdim, 0, ang)
            ring1 = polar_index(rdim, 1, ang)
            inner = polar_index(rdim, slice(2, er + 1), ang)

            angle_comm = x.space.cart.subcomm[adim] if x.space.parallel else None
            sum_cos0, sum_sin0 = cos_sin_avg(
                theta_local, x, angle_comm, sa, ea, na, 0, radial_dim=rdim
            )
            p_x0 = sum_cos0 * np.cos(theta_local) + sum_sin0 * np.sin(theta_local)

            if self.transposed:
                sum_cos1, sum_sin1 = cos_sin_avg(
                    theta_local, x, angle_comm, sa, ea, na, 1, radial_dim=rdim
                )
                p_x1 = sum_cos1 * np.cos(theta_local) + sum_sin1 * np.sin(theta_local)
                y[ring0] = p_x0 + x[ring1] - p_x1
                y[ring1] = x[ring1]
            else:
                y[ring0] = p_x0
                y[ring1] = x[ring0] + x[ring1] - p_x0

            y[inner] = x[inner]
        else:
            owned = polar_index(rdim, slice(sr, er + 1), ang)  # all radial indices of the rank
            y[owned] = x[owned]

        y.update_ghost_regions()
        return y

    def transpose(self, conjugate=False):
        return C1PolarProjection_U1_s_s(
            self.W1, transposed=not self.transposed, radial_dim=self.radial_dim
        )

    def tosparse(self):

        # The matrix is assembled with (radial, angular) C-style flat indices
        # i*na + j for radial index i and angular index j.

        rdim, adim = self.radial_dim, self.angle_dim
        V = self.domain
        sr, sa = V.starts[rdim], V.starts[adim]
        er, ea = V.ends[rdim], V.ends[adim]
        nr, na = V.npts[rdim], V.npts[adim]
        theta = np.linspace(0, 2 * pi, na, endpoint=False)
        rank_at_polar_edge = sr == 0

        data, cols, rows = [], [], []

        if rank_at_polar_edge:
            # block p (ring 0 -> ring 0) and block I - p (ring 0 -> ring 1)
            p = (2 / na) * toeplitz_columns_sym(theta, sa, ea, na)
            i_minus_p = -p
            diag_js = np.arange(sa, ea + 1)
            i_minus_p[(diag_js - sa) * na + diag_js] += 1.0
            data = np.concatenate((p, i_minus_p))

            rows_theta = np.repeat(np.arange(sa, ea + 1), na)
            cols_theta = np.tile(np.arange(na), ea - sa + 1)
            rows = np.concatenate((rows_theta, rows_theta + na))
            cols = np.concatenate((cols_theta, cols_theta))

        # identity on radial indices i >= 1
        i = np.arange(max(sr, 1), er + 1)[:, None]
        j = np.arange(sa, ea + 1)[None, :]
        local_rows = (i * na + j).ravel()

        data = np.concatenate((data, np.ones(local_rows.size)))
        cols = np.concatenate((cols, local_rows))
        rows = np.concatenate((rows, local_rows))

        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)

        if rdim == 1:
            rows = transpose_flat_index(rows, nr, na)
            cols = transpose_flat_index(cols, nr, na)

        P = coo_matrix(
            (data, (rows, cols)),
            shape=[nr * na, nr * na],
            dtype=V.dtype,
        )
        P.eliminate_zeros()

        return P.T if self.transposed else P

    def toarray(self):
        return self.tosparse().toarray()


class C1PolarProjection_U1_theta_s(LinearOperator):
    r"""
    Matrix-free representation of the block of the CONGA projection matrix
    $\mathbb{P}_U^1$ which maps the $s$-component of a 1-form to its
    $\theta$-component.

    Parameters
    ----------
    W1 : VectorFemSpace
        The full tensor product spline space of 1-forms
        $\mathbb{S}_{p_1-1, p_2}(\hat{\Omega})
        \times \mathbb{S}_{p_1, p_2-1}(\hat{\Omega})$.

    transposed : bool, default=False
        If `True`, create block of transposed projection matrix
        $(\mathbb{P}_U^1)^T$, which maps the $\theta$-component of a 1-form
        to its $s$-component.

    radial_dim : int, default=0
        Index of the logical dimension corresponding to the radial coordinate
        $s$. With `radial_dim=0` this is the lower-left block of $\mathbb{P}_U^1$
        (upper-right if transposed), with `radial_dim=1` it is the upper-right
        block (lower-left if transposed).
    """
    def __init__(self, W1, transposed=False, radial_dim=0):
        assert isinstance(W1, VectorFemSpace)
        assert radial_dim in (0, 1)

        self.W1 = W1
        self.transposed = transposed
        self.radial_dim = radial_dim
        self.angle_dim = 1 - radial_dim

    @property
    def domain(self):
        # s-component, or theta-component if transposed
        idx = self.angle_dim if self.transposed else self.radial_dim
        return self.W1.coeff_space[idx]

    @property
    def codomain(self):
        # theta-component, or s-component if transposed
        idx = self.radial_dim if self.transposed else self.angle_dim
        return self.W1.coeff_space[idx]

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
        angle_comm.Sendrecv(
            sendbuf=z[0], dest=left_rank, recvbuf=recvbuf, source=right_rank
        )

        result[-1] = recvbuf[0] - z[-1]
        return result

    def dot(self, x, out=None):
        assert isinstance(x, StencilVector)

        if not x.ghost_regions_in_sync:
            x.update_ghost_regions()

        rdim, adim = self.radial_dim, self.angle_dim
        V = self.domain
        sr, sa = V.starts[rdim], V.starts[adim]
        ea = V.ends[adim]
        na = V.npts[adim]
        theta_local = np.linspace(sa * 2 * pi / na, ea * 2 * pi / na, ea - sa + 1)
        rank_at_polar_edge = sr == 0

        if out is None:
            y = self.codomain.zeros()
        else:
            assert isinstance(out, StencilVector)
            assert out.space is self.codomain
            out *= 0
            y = out

        if rank_at_polar_edge:
            angle_comm = x.space.cart.subcomm[adim] if x.space.parallel else None
            ang = slice(sa, ea + 1)

            if self.transposed:
                s_ring0          = polar_index(rdim, 0, ang)
                theta_ring1      = polar_index(rdim, 1, ang)
                theta_ring1_prev = polar_index(rdim, 1, slice(sa - 1, ea))
                y[s_ring0] = x[theta_ring1_prev] - x[theta_ring1]
                sum_cos, sum_sin = cos_sin_avg(
                    theta_local, y, angle_comm, sa, ea, na, 0, radial_dim=rdim
                )
                y[s_ring0] = np.cos(theta_local) * sum_cos + np.sin(theta_local) * sum_sin
            else:
                theta_ring1 = polar_index(rdim, 1, ang)
                sum_cos, sum_sin = cos_sin_avg(
                    theta_local, x, angle_comm, sa, ea, na, 0, radial_dim=rdim
                )
                p_x0 = np.cos(theta_local) * sum_cos + np.sin(theta_local) * sum_sin
                y[theta_ring1] = self.forward_theta_diff_local(p_x0, angle_comm)

        y.update_ghost_regions()
        return y

    def transpose(self, conjugate=False):
        return C1PolarProjection_U1_theta_s(
            self.W1, transposed=not self.transposed, radial_dim=self.radial_dim
        )

    def tosparse(self):

        # The non-transposed block is assembled, which maps the s-component
        # to the theta-component, with (radial, angular) C-style flat indices
        # i*na + j for radial index i and angular index j.

        rdim, adim = self.radial_dim, self.angle_dim
        V_s = self.W1.coeff_space[rdim]
        V_theta = self.W1.coeff_space[adim]
        nr_s, na = V_s.npts[rdim], V_s.npts[adim]
        nr_theta = V_theta.npts[rdim]
        sr, sa = V_s.starts[rdim], V_s.starts[adim]
        ea = V_s.ends[adim]
        rank_at_polar_edge = sr == 0

        data, cols, rows = [], [], []
        if rank_at_polar_edge:
            local_j = np.arange(sa, ea + 1)
            all_k = np.arange(na)
            rows = np.repeat(local_j + na, na)
            cols = np.tile(all_k, len(local_j))

            theta = 2 * pi * np.arange(na) / na
            j = local_j[:, None]
            k = all_k[None, :]
            p = np.cos(theta[(j - k) % na])
            p_next = np.cos(theta[(j + 1 - k) % na])
            data = ((2 / na) * (p_next - p)).ravel()

        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)

        if rdim == 1:
            rows = transpose_flat_index(rows, nr_theta, na)
            cols = transpose_flat_index(cols, nr_s, na)

        P = coo_matrix(
            (data, (rows, cols)),
            shape=[nr_theta * na, nr_s * na],
            dtype=V_s.dtype,
        )

        return P.T if self.transposed else P

    def toarray(self):
        return self.tosparse().toarray()


class C1PolarProjection_U1(BlockLinearOperator):
    r"""
    Matrix-free representation of the CONGA projection matrix $\mathbb{P}_U^1$.

    This matrix acts on coefficient vectors in the full tensor-product spline
    basis of $\mathbb{S}_{p_1-1, p_2}(\hat{\Omega})
    \times \mathbb{S}_{p_1, p_2-1}(\hat{\Omega})$. It enforces the coefficient
    relations required for the corresponding spline, after push-forward, to
    be in the conforming spline space $U_h^1$.

    Parameters
    ----------
    W1 : VectorFemSpace
        The full tensor product spline space of 1-forms
        $\mathbb{S}_{p_1-1, p_2}(\hat{\Omega})
        \times \mathbb{S}_{p_1, p_2-1}(\hat{\Omega})$.

    transposed : bool, default=False
        If `True`, create the transposed projection matrix $(\mathbb{P}_U^1)^T$.

    hbc : bool, default=False
        If `True`, impose homogeneous Dirichlet boundary conditions on the
        tangential (angular) component of the field.
    radial_dim : int, default=0
        Index of the logical dimension corresponding to the radial coordinate
        $s$, which is also the index of the $s$-component in `W1`.
    """
    def __init__(self, W1, transposed=False, hbc=False, radial_dim=0):
        assert isinstance(W1, VectorFemSpace)
        assert W1.symbolic_space.kind.name == "hcurl"
        assert radial_dim in (0, 1)

        T1 = W1.coeff_space

        super().__init__(T1, T1)

        # indices of the s- and theta-components in W1
        rdim, adim = radial_dim, 1 - radial_dim

        self[rdim, rdim] = C1PolarProjection_U1_s_s(
            W1, transposed=transposed, radial_dim=radial_dim
        )
        # The theta_theta block is the same for C0 and C1 projections
        self[adim, adim] = C0PolarProjection_V1_theta_theta(
            W1, transposed=transposed, hbc=hbc, radial_dim=radial_dim
        )

        P_theta_s = C1PolarProjection_U1_theta_s(
            W1, transposed=transposed, radial_dim=radial_dim
        )

        if transposed:
            self[rdim, adim] = P_theta_s
        else:
            self[adim, rdim] = P_theta_s

        self.W1 = W1
        self.transposed = transposed
        self.hbc = hbc
        self.radial_dim = radial_dim


# -------------- 2-FORMS CONGA PROJECTOR P2 ----------------#
class C1PolarProjection_U2(C0PolarProjection_V2):
    r"""
    Matrix-free representation of the CONGA projection matrix $\mathbb{P}_U^2$.

    This matrix acts on coefficient vectors in the full tensor-product spline
    basis of $\mathbb{S}_{p_1-1, p_2-1}(\hat{\Omega})$. It enforces the coefficient
    relations required for the corresponding spline, after push-forward, to
    be in the conforming spline space $U_h^2$.

    Parameters
    ----------
    W2 : TensorFemSpace
        The full tensor product spline space of 2-forms
        $\mathbb{S}_{p_1-1, p_2-1}(\hat{\Omega})$.

    transposed : bool, default=False
        If `True`, create the transposed projection matrix $(\mathbb{P}_U^2)^T$.
    
    radial_dim : int, default=0
        Index of the logical dimension corresponding to the radial coordinate
        $s$. The other dimension is the (periodic) angular coordinate $\theta$.
        Use `radial_dim=1` for spaces built on $(\theta, s)$.

    """
