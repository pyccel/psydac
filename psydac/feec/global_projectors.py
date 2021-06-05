# -*- coding: UTF-8 -*-

import numpy as np

from psydac.linalg.utilities      import array_to_stencil
from psydac.linalg.kron           import KroneckerLinearSolver
from psydac.linalg.stencil        import StencilVector
from psydac.linalg.block          import BlockDiagonalSolver, BlockVector
from psydac.core.bsplines         import quadrature_grid
from psydac.utilities.quadratures import gauss_legendre
from psydac.fem.basic             import FemField

from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import ProductFemSpace

#==============================================================================
class GlobalProjector:
    def __init__(self, space, nquads = None):
        self.space = space
        self.rhs = space.vector_space.zeros()

        if isinstance(space, TensorFemSpace):
            tensorspaces = [space]
            rhsblocks = [self.rhs]
        elif isinstance(space, ProductFemSpace):
            tensorspaces = space.spaces
            rhsblocks = self.rhs.blocks
        else:
            # no SplineSpace support for now
            raise NotImplementedError()
        
        # from now on, we only deal with tensorspaces, i.e. a list of TensorFemSpace instances
        
        self.dim    = tensorspaces[0].ldim
        self.blockcount = len(tensorspaces)

        # set up quadrature weights
        if nquads:
            assert len(nquads) == self.dim
            uw = [gauss_legendre( k-1 ) for k in nquads]
            uw = [(u[::-1], w[::-1]) for u,w in uw]
        else:
            uw = [(V.quad_grids[i].quad_rule_x,V.quad_grids[i].quad_rule_w) for i,V in enumerate(tensorspaces)]
        
        # retrieve projection space structure
        # this is a 2D Python array (first level: block, second level: tensor direction)
        # which indicates where to use histopolation and where interpolation
        structure = self._structure(self.dim)

        # check for the existence of interpolation/histopolation...
        has_i = False
        has_h = False
        for block in structure:
            for cell in block:
                if cell == 'I': has_i = True
                if cell == 'H': has_h = True

        # ... and activate them.
        for space in tensorspaces:
            if has_i: space.init_interpolation()
            if has_h: space.init_histopolation()

        # retrieve the restriction function
        self.func = self._function(self.dim)

        # construct arguments for the function
        # TODO: verify that histopolation and interpolation behave EXACTLY the same for all blocks, including their distribution in MPI
        # (this is currently not checked anywhere)
        intp_x = [None] * self.dim if has_i else []
        quad_x = [None] * self.dim if has_h else []
        quad_w = [None] * self.dim if has_h else []
        dofs = [None] * self.blockcount

        # in the meanwhile, also store all grids in a canonical format
        # (and fetch the interpolation/histopolation solvers)
        self.grid_x = []
        self.grid_w = []
        solverblocks = []
        for i,block in enumerate(structure):
            # do for each block (i.e. each TensorFemSpace):
             
            block_x = []
            block_w = []
            solvercells = []
            for j, cell in enumerate(block):
                # for each direction in the tensor space (i.e. each SplineSpace):

                V = tensorspaces[i].spaces[j]
                s = tensorspaces[i].vector_space.starts[j]
                e = tensorspaces[i].vector_space.ends[j]

                if cell == 'I':
                    # interpolation case
                    if intp_x[j] is None:
                        intp_x[j] = V.greville[s:e+1]
                    local_intp_x = intp_x[j]

                    # for the grids, make interpolation appear like quadrature
                    local_x = local_intp_x[:, np.newaxis]
                    local_w = np.ones_like(local_x)
                    solvercells += [V._interpolator]
                elif cell == 'H':
                    # histopolation case
                    if quad_x[j] is None:
                        u, w = uw[j]
                        quad_x[j], quad_w[j] = quadrature_grid(V.histopolation_grid, u, w)[s:e+1]
                    local_x, local_w = quad_x[j], quad_w[j]
                    solvercells += [V._histopolator]
                else:
                    raise NotImplementedError('Invalid entry in structure array.')
                
                block_x += [local_x]
                block_w += [local_w]

            # finish block, build solvers, get dataslice to project to
            self.grid_x += [block_x]
            self.grid_w += [block_w]

            solverblocks += [KroneckerLinearSolver(tensorspaces[i].vector_space, solvercells)]

            dataslice = tuple(slice(p, -p) for p in tensorspaces[i].vector_space.pads)
            dofs[i] = rhsblocks[i]._data[dataslice]
        
        self.args = (*intp_x, *quad_x, *quad_w, *dofs)

        # build a BlockDiagonalSolver, if necessary
        if len(solverblocks) == 1:
            self.solver = solverblocks[0]
        else:
            self.solver = BlockDiagonalSolver(self.space.vector_space, blocks=solverblocks)
    
    def _structure(self, ndim):
        raise NotImplementedError()
    
    def _function(self, ndim):
        raise NotImplementedError()
    
    def __call__(self, fun):
        """
        Project vector function onto the given finite element
        space by the instance of this class. This happens in the logical domain $\hat{\Omega}$.

        Parameters
        ----------
        fun : callable or list/tuple of callables
            Scalar components of the real-valued vector function to be
            projected, with arguments the coordinates (x_1, ..., x_N) of a
            point in the logical domain.

            $fun_i : \hat{\Omega} \mapsto \mathbb{R}$ with i = 1, ..., N.

        Returns
        -------
        field : FemField
            Field obtained by projection (element of the target space-conforming
            finite element space). This is also a real-valued scalar/vector function
            in the logical domain.
        """
        # build the rhs
        if self.blockcount > 1 or isinstance(fun, list) or isinstance(fun, tuple):
            # (we also support 1-tuples as argument for scalar spaces)
            self.func(*self.args, *fun)
        else:
            self.func(*self.args, fun)

        coeffs = self.solver.solve(self.rhs)

        return FemField(self.space, coeffs=coeffs)

#==============================================================================
class Projector_H1(GlobalProjector):
    """
    Projector from H1 to an H1-conforming finite element space (i.e. a finite
    dimensional subspace of H1) constructed with tensor-product B-splines in 1,
    2 or 3 dimensions.

    This is a global projector based on interpolation over a tensor-product
    grid in the logical domain. The interpolation grid is the tensor product of
    the 1D splines' Greville points along each direction.

    Parameters
    ----------
    H1 : SplineSpace or TensorFemSpace
        H1-conforming finite element space, codomain of the projection operator
    """
    def _structure(self, dim):
        return [['I'] * dim]
    
    def _function(self, dim):
        if   dim == 1:  return evaluate_dofs_1d_0form
        elif dim == 2:  return evaluate_dofs_2d_0form
        elif dim == 3:  return evaluate_dofs_3d_0form
        else:
            raise ValueError('H1 projector of dimension {} not available'.format(dim)) 

    #--------------------------------------------------------------------------
    def __call__(self, fun):
        r"""
        Project scalar function onto the H1-conforming finite element space.
        This happens in the logical domain $\hat{\Omega}$.

        Parameters
        ----------
        fun : callable
            Real-valued scalar function to be projected, with arguments the
            coordinates (x_1, ..., x_N) of a point in the logical domain. This
            corresponds to the coefficient of a 0-form.

            $fun : \hat{\Omega} \mapsto \mathbb{R}$.

        Returns
        -------
        field : FemField
            Field obtained by projection (element of the H1-conforming finite
            element space). This is also a real-valued scalar function in the
            logical domain.
        """
        return super().__call__(fun)

#==============================================================================
class Projector_Hcurl(GlobalProjector):
    """
    Projector from H(curl) to an H(curl)-conforming finite element space, i.e.
    a finite dimensional subspace of H(curl), constructed with tensor-product
    B- and M-splines in 2 or 3 dimensions.

    This is a global projector constructed over a tensor-product grid in the
    logical domain. The vertices of this grid are obtained as the tensor
    product of the 1D splines' Greville points along each direction.

    The H(curl) projector matches the "geometric" degrees of freedom of
    discrete 1-forms, which are the line integrals of a vector field along cell
    edges. To achieve this, each component of the vector field is projected
    independently, by combining 1D histopolation along the direction of the
    edges with 1D interpolation along the other directions.

    Parameters
    ----------
    Hcurl : ProductFemSpace
        H(curl)-conforming finite element space, codomain of the projection
        operator.

    nquads : list(int) | tuple(int)
        Number of quadrature points along each direction, to be used in Gauss
        quadrature rule for computing the (approximated) degrees of freedom.
    """
    def _structure(self, dim):
        if dim == 3:
            return [
                ['H', 'I', 'I'],
                ['I', 'H', 'I'],
                ['I', 'I', 'H']
            ]
        elif dim == 2:
            return [
                ['H', 'I'],
                ['I', 'H']
            ]
        else:
            raise NotImplementedError('The Hcurl projector is only available in 2D or 3D.')
    
    def _function(self, dim):
        if dim == 3: return evaluate_dofs_3d_1form
        elif dim == 2: return evaluate_dofs_2d_1form_hcurl
        else:
            raise NotImplementedError('The Hcurl projector is only available in 2D or 3D.')

    #--------------------------------------------------------------------------
    def __call__(self, fun):
        r"""
        Project vector function onto the H(curl)-conforming finite element
        space. This happens in the logical domain $\hat{\Omega}$.

        Parameters
        ----------
        fun : list/tuple of callables
            Scalar components of the real-valued vector function to be
            projected, with arguments the coordinates (x_1, ..., x_N) of a
            point in the logical domain. These correspond to the coefficients
            of a 1-form in the canonical basis (dx_1, ..., dx_N).

            $fun_i : \hat{\Omega} \mapsto \mathbb{R}$ with i = 1, ..., N.

        Returns
        -------
        field : FemField
            Field obtained by projection (element of the H(curl)-conforming
            finite element space). This is also a real-valued vector function
            in the logical domain.
        """
        return super().__call__(fun)

#==============================================================================
class Projector_Hdiv(GlobalProjector):
    """
    Projector from H(div) to an H(div)-conforming finite element space, i.e. a
    finite dimensional subspace of H(div), constructed with tensor-product
    B- and M-splines in 2 or 3 dimensions.

    This is a global projector constructed over a tensor-product grid in the
    logical domain. The vertices of this grid are obtained as the tensor
    product of the 1D splines' Greville points along each direction.

    The H(div) projector matches the "geometric" degrees of freedom of discrete
    (N-1)-forms in N dimensions, which are the integrated flux of a vector
    field through cell faces (in 3D) or cell edges (in 2D).

    To achieve this, each component of the vector field is projected
    independently, by combining histopolation along the direction(s) tangential
    to the face (in 3D) or edge (in 2D), with 1D interpolation along the normal
    direction.

    Parameters
    ----------
    Hdiv : ProductFemSpace
        H(div)-conforming finite element space, codomain of the projection
        operator.

    nquads : list(int) | tuple(int)
        Number of quadrature points along each direction, to be used in Gauss
        quadrature rule for computing the (approximated) degrees of freedom.
    """
    def _structure(self, dim):
        if dim == 3:
            return [
                ['I', 'H', 'H'],
                ['H', 'I', 'H'],
                ['H', 'H', 'I']
            ]
        elif dim == 2:
            return [
                ['I', 'H'],
                ['H', 'I']
            ]
        else:
            raise NotImplementedError('The Hdiv projector is only available in 2D or 3D.')
    
    def _function(self, dim):
        if dim == 3: return evaluate_dofs_3d_2form
        elif dim == 2: return evaluate_dofs_2d_1form_hdiv
        else:
            raise NotImplementedError('The Hdiv projector is only available in 2D or 3D.')

    #--------------------------------------------------------------------------
    def __call__(self, fun):
        r"""
        Project vector function onto the H(div)-conforming finite element
        space. This happens in the logical domain $\hat{\Omega}$.

        Parameters
        ----------
        fun : list/tuples of callable
            Scalar components of the real-valued vector function to be
            projected, with arguments the coordinates (x_1, ..., x_N) of a
            point in the logical domain. In 3D these correspond to the
            coefficients of a 2-form in the canonical basis (dx_1 ∧ dx_2,
            dx_2 ∧ dx_3, dx_3 ∧ dx_1).

            $fun_i : \hat{\Omega} \mapsto \mathbb{R}$ with i = 1, ..., N.

        Returns
        -------
        field : FemField
            Field obtained by projection (element of the H(div)-conforming
            finite element space). This is also a real-valued vector function
            in the logical domain.
        """
        return super().__call__(fun)

#==============================================================================
class Projector_L2(GlobalProjector):
    """
    Projector from L2 to an L2-conforming finite element space (i.e. a finite
    dimensional subspace of L2) constructed with tensor-product M-splines in 1,
    2 or 3 dimensions.

    This is a global projector constructed over a tensor-product grid in the
    logical domain. The vertices of this grid are obtained as the tensor
    product of the 1D splines' Greville points along each direction.

    The L2 projector matches the "geometric" degrees of freedom of discrete
    N-forms in N dimensions, which are line/surface/volume integrals of a
    scalar field over an edge/face/cell in 1/2/3 dimension(s). To this end
    histopolation is used along each direction.

    Parameters
    ----------
    L2 : SplineSpace
        L2-conforming finite element space, codomain of the projection operator

    nquads : list(int) | tuple(int)
        Number of quadrature points along each direction, to be used in Gauss
        quadrature rule for computing the (approximated) degrees of freedom.
    """
    def _structure(self, dim):
        return [['H'] * dim]
    
    def _function(self, dim):
        if   dim == 1:  return evaluate_dofs_1d_1form
        elif dim == 2:  return evaluate_dofs_2d_2form
        elif dim == 3:  return evaluate_dofs_3d_3form
        else:
            raise ValueError('L2 projector of dimension {} not available'.format(dim))

    #--------------------------------------------------------------------------
    def __call__(self, fun):
        r"""
        Project scalar function onto the L2-conforming finite element space.
        This happens in the logical domain $\hat{\Omega}$.

        Parameters
        ----------
        fun : callable
            Real-valued scalar function to be projected, with arguments the
            coordinates (x_1, ..., x_N) of a point in the logical domain. This
            corresponds to the coefficient of an N-form in N dimensions, in
            the canonical basis dx_1 ∧ ... ∧ dx_N.

            $fun : \hat{\Omega} \mapsto \mathbb{R}$.

        Returns
        -------
        field : FemField
            Field obtained by projection (element of the L2-conforming finite
            element space). This is also a real-valued scalar function in the
            logical domain.
        """
        return super().__call__(fun)

#==============================================================================
# 1D DEGREES OF FREEDOM
#==============================================================================

def evaluate_dofs_1d_0form(intp_x1, F, f):
    (n1,) = F.shape

    for i1 in range(n1):
        F[i1] = f(intp_x1[i1])
        
#------------------------------------------------------------------------------
def evaluate_dofs_1d_1form(
        quad_x1, # quadrature points
        quad_w1, # quadrature weights
        F,       # array of degrees of freedom (intent out)
        f        # input scalar function (callable)
        ):

    k1 = quad_x1.shape[1]

    n1, = F.shape
    for i1 in range(n1):
        F[i1] = 0.0
        for g1 in range(k1):
            F[i1] += quad_w1[i1, g1] * f(quad_x1[i1, g1])

#==============================================================================
# 2D DEGREES OF FREEDOM
#==============================================================================

def evaluate_dofs_2d_0form(intp_x1, intp_x2, F, f):
    n1, n2 = F.shape

    for i1 in range(n1):
        for i2 in range(n2):
            F[i1, i2] = f(intp_x1[i1], intp_x2[i2])

#------------------------------------------------------------------------------
def evaluate_dofs_2d_1form_hcurl(
        intp_x1, intp_x2, # interpolation points
        quad_x1, quad_x2, # quadrature points
        quad_w1, quad_w2, # quadrature weights
        F1, F2,           # arrays of degrees of freedom (intent out)
        f1, f2            # input scalar functions (callable)
        ):

    k1 = quad_x1.shape[1]
    k2 = quad_x2.shape[1]

    n1, n2 = F1.shape
    for i1 in range(n1):
        for i2 in range(n2):
            F1[i1, i2] = 0.0
            for g1 in range(k1):
                F1[i1, i2] += quad_w1[i1, g1] * f1(quad_x1[i1, g1], intp_x2[i2])

    n1, n2 = F2.shape
    for i1 in range(n1):
        for i2 in range(n2):
            F2[i1, i2] = 0.0
            for g2 in range(k2):
                F2[i1, i2] += quad_w2[i2, g2] * f2(intp_x1[i1], quad_x2[i2, g2])

#------------------------------------------------------------------------------
def evaluate_dofs_2d_1form_hdiv(
        intp_x1, intp_x2, # interpolation points
        quad_x1, quad_x2, # quadrature points
        quad_w1, quad_w2, # quadrature weights
        F1, F2,           # arrays of degrees of freedom (intent out)
        f1, f2            # input scalar functions (callable)
        ):

    k1 = quad_x1.shape[1]
    k2 = quad_x2.shape[1]

    n1, n2 = F1.shape
    for i1 in range(n1):
        for i2 in range(n2):
            F1[i1, i2] = 0.0
            for g2 in range(k2):
                F1[i1, i2] += quad_w2[i2, g2] * f1(intp_x1[i1], quad_x2[i2, g2])

    n1, n2 = F2.shape
    for i1 in range(n1):
        for i2 in range(n2):
            F2[i1, i2] = 0.0
            for g1 in range(k1):
                F2[i1, i2] += quad_w1[i1, g1] * f2(quad_x1[i1, g1], intp_x2[i2])

#------------------------------------------------------------------------------
def evaluate_dofs_2d_2form(
        quad_x1, quad_x2, # quadrature points
        quad_w1, quad_w2, # quadrature weights
        F,                # array of degrees of freedom (intent out)
        f,                # input scalar function (callable)
        ):

    k1 = quad_x1.shape[1]
    k2 = quad_x2.shape[1]

    n1, n2 = F.shape
    for i1 in range(n1):
        for i2 in range(n2):
            F[i1, i2] = 0.0
            for g1 in range(k1):
                for g2 in range(k2):
                    F[i1, i2] += quad_w1[i1, g1] * quad_w2[i2, g2] * \
                            f(quad_x1[i1, g1], quad_x2[i2, g2])

#==============================================================================
# 3D DEGREES OF FREEDOM
#==============================================================================

def evaluate_dofs_3d_0form(intp_x1, intp_x2, intp_x3, F, f):
    n1, n2, n3 = F.shape

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F[i1, i2, i3] = f(intp_x1[i1], intp_x2[i2], intp_x3[i3])

#------------------------------------------------------------------------------
def evaluate_dofs_3d_1form(
        intp_x1, intp_x2, intp_x3, # interpolation points
        quad_x1, quad_x2, quad_x3, # quadrature points
        quad_w1, quad_w2, quad_w3, # quadrature weights
        F1, F2, F3,                # arrays of degrees of freedom (intent out)
        f1, f2, f3                 # input scalar functions (callable)
        ):

    k1 = quad_x1.shape[1]
    k2 = quad_x2.shape[1]
    k3 = quad_x3.shape[1]

    n1, n2, n3 = F1.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F1[i1, i2, i3] = 0.0
                for g1 in range(k1):
                    F1[i1, i2, i3] += quad_w1[i1, g1] * \
                            f1(quad_x1[i1, g1], intp_x2[i2], intp_x3[i3])

    n1, n2, n3 = F2.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F2[i1, i2, i3] = 0.0
                for g2 in range(k2):
                    F2[i1, i2, i3] += quad_w2[i2, g2] * \
                            f2(intp_x1[i1], quad_x2[i2, g2], intp_x3[i3])

    n1, n2, n3 = F3.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F3[i1, i2, i3] = 0.0
                for g3 in range(k3):
                    F3[i1, i2, i3] += quad_w3[i3, g3] * \
                            f3(intp_x1[i1], intp_x2[i2], quad_x3[i3, g3])

#------------------------------------------------------------------------------
def evaluate_dofs_3d_2form(
        intp_x1, intp_x2, intp_x3, # interpolation points
        quad_x1, quad_x2, quad_x3, # quadrature points
        quad_w1, quad_w2, quad_w3, # quadrature weights
        F1, F2, F3,                # arrays of degrees of freedom (intent out)
        f1, f2, f3                 # input scalar functions (callable)
        ):

    k1 = quad_x1.shape[1]
    k2 = quad_x2.shape[1]
    k3 = quad_x3.shape[1]

    n1, n2, n3 = F1.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F1[i1, i2, i3] = 0.0
                for g2 in range(k2):
                    for g3 in range(k3):
                        F1[i1, i2, i3] += quad_w2[i2, g2] * quad_w3[i3, g3] * \
                            f1(intp_x1[i1], quad_x2[i2, g2], quad_x3[i3, g3])

    n1, n2, n3 = F2.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F2[i1, i2, i3] = 0.0
                for g1 in range(k1):
                    for g3 in range(k3):
                        F2[i1, i2, i3] += quad_w1[i1, g1] * quad_w3[i3, g3] * \
                            f2(quad_x1[i1, g1], intp_x2[i2], quad_x3[i3, g3])

    n1, n2, n3 = F3.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F3[i1, i2, i3] = 0.0
                for g1 in range(k1):
                    for g2 in range(k2):
                        F3[i1, i2, i3] += quad_w1[i1, g1] * quad_w2[i2, g2] * \
                            f3(quad_x1[i1, g1], quad_x2[i2, g2], intp_x3[i3])

#------------------------------------------------------------------------------
def evaluate_dofs_3d_3form(
        quad_x1, quad_x2, quad_x3, # quadrature points
        quad_w1, quad_w2, quad_w3, # quadrature weights
        F,                         # array of degrees of freedom (intent out)
        f,                         # input scalar function (callable)
        ):

    k1 = quad_x1.shape[1]
    k2 = quad_x2.shape[1]
    k3 = quad_x3.shape[1]

    n1, n2, n3 = F.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F[i1, i2, i3] = 0.0
                for g1 in range(k1):
                    for g2 in range(k2):
                        for g3 in range(k3):
                            F[i1, i2, i3] += \
                                    quad_w1[i1, g1] * quad_w2[i2, g2] * quad_w3[i3, g3] * \
                                    f(quad_x1[i1, g1], quad_x2[i2, g2], quad_x3[i3, g3])
