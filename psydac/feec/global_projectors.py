# -*- coding: UTF-8 -*-

import numpy as np

from psydac.linalg.kron           import KroneckerLinearSolver, KroneckerStencilMatrix
from psydac.linalg.stencil        import StencilMatrix, StencilVectorSpace
from psydac.linalg.block          import BlockLinearOperator
from psydac.core.bsplines         import quadrature_grid
from psydac.utilities.quadratures import gauss_legendre
from psydac.fem.basic             import FemField
from psydac.feec                  import dof_kernels

from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import VectorFemSpace

from psydac.ddm.cart import DomainDecomposition, CartDecomposition
from psydac.utilities.utils import roll_edges

from abc import ABCMeta, abstractmethod

__all__ = ('GlobalProjector', 'Projector_H1', 'Projector_Hcurl', 'Projector_Hdiv', 'Projector_L2',
           'evaluate_dofs_1d_0form', 'evaluate_dofs_1d_1form',
           'evaluate_dofs_2d_0form', 'evaluate_dofs_2d_1form_hcurl', 'evaluate_dofs_2d_1form_hdiv', 'evaluate_dofs_2d_2form',
           'evaluate_dofs_3d_0form', 'evaluate_dofs_3d_1form', 'evaluate_dofs_3d_2form', 'evaluate_dofs_3d_3form')

#==============================================================================
class GlobalProjector(metaclass=ABCMeta):
    """
    Projects callable functions to some scalar or vector FEM space.

    A global projector is constructed over a tensor-product grid in the logical
    domain. The vertices of this grid are obtained as the tensor product of the
    1D splines' Greville points along each direction.

    This projector matches the "geometric" degrees of freedom of discrete
    n-forms (where n depends on the underlying space).
    This is done by projecting each component of the vector field independently,
    by combining 1D histopolation with 1D interpolation.

    This class cannot be instantiated directly (use a subclass instead).

    Parameters
    ----------
    space : VectorFemSpace | TensorFemSpace
        Some finite element space, codomain of the projection
        operator. The exact structure where to use histopolation and where interpolation
        has to be given by a subclass of the GlobalProjector class.
        As of now, it is implicitly assumed for a VectorFemSpace, that for each direction
        that all spaces with interpolation are the same, and all spaces with histopolation are the same
        (i.e. yield the same quadrature/interpolation points etc.); so use with care on an arbitrary VectorFemSpace.
        It is right now only intended to be used with VectorFemSpaces or TensorFemSpaces from DeRham complex objects.

    nquads : list(int) | tuple(int)
        Number of quadrature points along each direction, to be used in Gauss
        quadrature rule for computing the (approximated) degrees of freedom.
        This parameter is ignored, if the projector only uses interpolation (and no histopolation).
    """

    def __init__(self, space, nquads = None):
        self._space = space
        self._rhs = space.vector_space.zeros()

        if isinstance(space, TensorFemSpace):
            tensorspaces = [space]
            rhsblocks = [self._rhs]
        elif isinstance(space, VectorFemSpace):
            tensorspaces = space.spaces
            rhsblocks = self._rhs.blocks
        else:
            # no SplineSpace support for now
            raise NotImplementedError()
        
        # from now on, we only deal with tensorspaces, i.e. a list of TensorFemSpace instances
        
        self._dim    = tensorspaces[0].ldim
        assert all([self._dim == tspace.ldim for tspace in tensorspaces])

        self._blockcount = len(tensorspaces)

        # set up quadrature weights
        if nquads:
            assert len(nquads) == self._dim
            uw = [gauss_legendre(k) for k in nquads]

        # retrieve projection space structure
        # this is a 2D Python array (first level: block, second level: tensor direction)
        # which indicates where to use histopolation and where interpolation
        structure = self._structure(self._dim)

        # check the layout of the array, and for the existence of interpolation/histopolation...
        assert len(structure) == self._blockcount

        has_i = False
        has_h = False
        for block in structure:
            assert len(block) == self._dim
            for cell in block:
                if cell == 'I': has_i = True
                if cell == 'H': has_h = True

        if has_h and nquads is None:
            raise ValueError('The number of quadrature points `nquads` must be provided for performing histopolation')

        # ... and activate them.
        for space in tensorspaces:
            if has_i: space.init_interpolation()
            if has_h: space.init_histopolation()

        # retrieve the restriction function
        func = self._function(self._dim)

        # construct arguments for the function
        # TODO: verify that histopolation and interpolation behave EXACTLY the same for all blocks, including their distribution in MPI
        # (this is currently not checked anywhere)
        intp_x = [None] * self._dim if has_i else []
        quad_x = [None] * self._dim if has_h else []
        quad_w = [None] * self._dim if has_h else []
        dofs = [None] * self._blockcount

        # in the meanwhile, also store all grids in a canonical format
        # (and fetch the interpolation/histopolation solvers)
        self._grid_x = []
        self._grid_w = []
        solverblocks = []
        matrixblocks = []
        blocks = [] # for BlockLinearOperator
        for i,block in enumerate(structure):
            # do for each block (i.e. each TensorFemSpace):
            
            block_x = []
            block_w = []
            solvercells = []
            matrixcells = []
            blocks += [[]] 
            for j, cell in enumerate(block):
                # for each direction in the tensor space (i.e. each SplineSpace):

                V = tensorspaces[i].spaces[j]
                s = tensorspaces[i].vector_space.starts[j]
                e = tensorspaces[i].vector_space.ends[j]
                p = tensorspaces[i].vector_space.pads[j]
                n = tensorspaces[i].vector_space.npts[j]
                m = tensorspaces[i].multiplicity[j]
                periodic = tensorspaces[i].vector_space.periods[j]
                ncells = tensorspaces[i].ncells[j]
                blocks[-1] += [None] # fill blocks with None, fill the diagonals later

                # create a distributed matrix for the current 1d SplineSpace
                domain_decomp = DomainDecomposition([ncells], [periodic])
                cart_decomp = CartDecomposition(domain_decomp, [n], [[s]], [[e]], [p], [m])
                V_cart = StencilVectorSpace(cart_decomp)
                M = StencilMatrix(V_cart, V_cart)

                if cell == 'I':
                    # interpolation case
                    if intp_x[j] is None:
                        intp_x[j] = V.greville[s:e+1]
                    local_intp_x = intp_x[j]

                    # for the grids, make interpolation appear like quadrature
                    local_x = local_intp_x[:, np.newaxis]
                    local_w = np.ones_like(local_x)
                    solvercells += [V._interpolator]
                    
                    # make 1D collocation matrix in stencil format
                    row_indices, col_indices = np.nonzero(V.imat)

                    for row_i, col_i in zip(row_indices, col_indices):

                        # only consider row indices on process
                        if row_i in range(V_cart.starts[0], V_cart.ends[0] + 1):
                            row_i_loc = row_i - s

                            M._data[row_i_loc + m*p, (col_i + p - row_i)%V.imat.shape[1]] = V.imat[row_i, col_i]

                    # check if stencil matrix was built correctly
                    assert np.allclose(M.toarray()[s:e + 1], V.imat[s:e + 1])
                    # TODO Fix toarray() for multiplicity m > 1
                    matrixcells += [M.copy()]
                    
                elif cell == 'H':
                    # histopolation case
                    if quad_x[j] is None:
                        u, w = uw[j]
                        global_quad_x, global_quad_w = quadrature_grid(V.histopolation_grid, u, w)

                        # "roll" back points to the interval to ensure that the quadrature points are
                        # in the domain. Only useful in the periodic case (else do nothing)
                        # if not used then you will have quadrature points outside of the domain which
                        # might cause problem when your function is only defined inside the domain.
                        roll_edges(V.domain, global_quad_x) 
                        quad_x[j] = global_quad_x[s:e+1]
                        quad_w[j] = global_quad_w[s:e+1]

                    local_x, local_w = quad_x[j], quad_w[j]
                    solvercells += [V._histopolator]
                    
                    # make 1D collocation matrix in stencil format
                    row_indices, col_indices = np.nonzero(V.hmat)

                    for row_i, col_i in zip(row_indices, col_indices):

                        # only consider row indices on process
                        if row_i in range(V_cart.starts[0], V_cart.ends[0] + 1):
                            row_i_loc = row_i - s

                            M._data[row_i_loc + m*p, (col_i + p - row_i)%V.hmat.shape[1]] = V.hmat[row_i, col_i]

                    # check if stencil matrix was built correctly
                    assert np.allclose(M.toarray()[s:e + 1], V.hmat[s:e + 1])

                    matrixcells += [M.copy()]
                    
                else:
                    raise NotImplementedError('Invalid entry in structure array.')
                
                block_x += [local_x]
                block_w += [local_w]
            
            # build Kronecker out of single directions    
            if isinstance(self.space, TensorFemSpace):
                matrixblocks += [KroneckerStencilMatrix(self.space.vector_space, self.space.vector_space, *matrixcells)]
            else:
                matrixblocks += [KroneckerStencilMatrix(self.space.vector_space[i], self.space.vector_space[i], *matrixcells)]

            # fill the diagonals for BlockLinearOperator
            blocks[i][i] = matrixblocks[-1]

            # finish block, build solvers, get dataslice to project to
            self._grid_x += [block_x]
            self._grid_w += [block_w]

            solverblocks += [KroneckerLinearSolver(tensorspaces[i].vector_space, tensorspaces[i].vector_space, solvercells)]

            dataslice = tuple(slice(p*m, -p*m) for p, m in zip(tensorspaces[i].vector_space.pads,tensorspaces[i].vector_space.shifts))
            dofs[i] = rhsblocks[i]._data[dataslice]
            
        # build final Inter-/Histopolation matrix (distributed)        
        if isinstance(self.space, TensorFemSpace):
            self._imat_kronecker = matrixblocks[0]
        else:
            # self._imat_kronecker = BlockLinearOperator(self.space.vector_space, self.space.vector_space, 
            #                                            blocks=blocks)
            self._imat_kronecker = BlockLinearOperator(self.space.vector_space, self.space.vector_space, 
                                               blocks=[[matrixblocks[0], None, None], 
                                                       [None, matrixblocks[1], None], 
                                                       [None, None, matrixblocks[2]]])
        
        # finish arguments and create a lambda
        args = (*intp_x, *quad_x, *quad_w, *dofs)
        self._func = lambda *fun: func(*args, *fun)

        # build a BlockLinearOperator, if necessary
        if len(solverblocks) == 1:
            self._solver = solverblocks[0]
        else:
            domain = codomain = self._space.vector_space
            blocks = {(i, i): B_i for i, B_i in enumerate(solverblocks)}
            self._solver = BlockLinearOperator(domain, codomain, blocks)
    
    @property
    def space(self):
        """
        The space to which this Projector projects.
        """
        return self._space
    
    @property
    def dim(self):
        """
        The dimension of the underlying TensorFemSpaces.
        """
        return self._dim
    
    @property
    def blockcount(self):
        """
        The number of blocks. In case that self.space is a TensorFemSpace, this is 1,
        otherwise it denotes the number of blocks in the VectorFemSpace.
        """
        return self._blockcount
    
    @property
    def grid_x(self):
        """
        The local interpolation/histopolation grids which are used; it denotes the position of the interpolation/quadrature points.
        All the grids are stored inside a two-dimensional array; the outer dimension denotes the block, the inner the tensor space direction.
        """
        return self._grid_x
    
    @property
    def grid_w(self):
        """
        The local interpolation/histopolation grids which are used; it denotes the weights of the quadrature points (in the case of interpolation, this will return the weight 1 for the given positions).
        All the grids are stored inside a two-dimensional array; the outer dimension denotes the block, the inner the tensor space direction.
        """
        return self._grid_w
    
    @property
    def func(self):
        """
        The function which is used for projecting a given callable (or list thereof) to the DOFs in the target space.
        """
        return self._func
    
    @property
    def solver(self):
        """
        The solver used for transforming the DOFs in the target space into spline coefficients.
        """
        return self._solver
    
    @property
    def imat_kronecker(self):
        """
        Inter-/Histopolation matrix in distributed format.
        """
        return self._imat_kronecker
    
    @abstractmethod
    def _structure(self, dim):
        """
        Has to be implemented by a subclass. Returns a 2-dimensional array
        which contains strings which either say 'I' or 'H', e.g.
        [['H', 'I', 'I'], ['I', 'H', 'I'], ['I', 'I', 'H']] for the 3-dimensional Hcurl space.

        The inner array dimension has to conform to the dim parameter,
        the outer with the number of blocks of the target space.

        Parameters
        ----------
        dim : int
            The dimension of the underlying TensorFemSpaces.
        
        Returns
        -------
        structure : array
            The described structure matrix.
        """
        pass
    
    @abstractmethod
    def _function(self, dim):
        """
        Has to be implemented by a subclass. Returns a function which accepts the arguments
        in the order (*intp_x, *quad_x, *quad_w, *dofs, *f) (see __init__ function) and then
        evaluates a given callable using these arguments. Note that the dofs array is modified by the function.

        (this can and most likely will be replaced, if we construct the functions somewhere else, e.g. with code generation)

        Parameters
        ----------
        dim : int
            The dimension of the underlying TensorFemSpaces.
        
        Returns
        -------
        func : callable
            The described function.
        """
        pass
    
    def __call__(self, fun, dofs_only = False):
        """
        Project vector function onto the given finite element
        space by the instance of this class. This happens in the logical domain $\hat{\Omega}$.

        Parameters
        ----------
        fun : callable or list/tuple of callables
            Scalar components of the real- or complex-valued vector function to be
            projected, with arguments the coordinates (x_1, ..., x_N) of a
            point in the logical domain.

            $fun_i : \hat{\Omega} \mapsto \mathbb{R}$ with i = 1, ..., N.

        dofs_only : bool
            Whether to just compute and return the DOFs 
            (i.e. no inversion of the inter-/histopolation matrix needed to get the FEM coefficiens)
        
        Returns
        -------
        field : FemField
            Field obtained by projection (element of the target space-conforming
            finite element space). This is also a real- or complex-valued scalar/vector function
            in the logical domain.
        """
        # build the rhs (degrees of freedom - DOFs)
        if self._blockcount > 1 or isinstance(fun, list) or isinstance(fun, tuple):
            # (we also support 1-tuples as argument for scalar spaces)
            assert self._blockcount == len(fun)
            self._func(*fun)
        else:
            self._func(fun)
        if dofs_only:
            return self._rhs.copy()
        else:
            # solver for FEM coefficients
            coeffs = self._solver.dot(self._rhs)

            return FemField(self._space, coeffs=coeffs)

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
    def __call__(self, fun, dofs_only = False):
        r"""
        Project scalar function onto the H1-conforming finite element space.
        This happens in the logical domain $\hat{\Omega}$.

        Parameters
        ----------
        fun : callable
            Real- or complex-valued scalar function to be projected, with arguments the
            coordinates (x_1, ..., x_N) of a point in the logical domain. This
            corresponds to the coefficient of a 0-form.

            $fun : \hat{\Omega} \mapsto \mathbb{R}$.

        Returns
        -------
        field : FemField
            Field obtained by projection (element of the H1-conforming finite
            element space). This is also a real- or complex-valued scalar function in the
            logical domain.
        """
        return super().__call__(fun, dofs_only = dofs_only)

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
    Hcurl : VectorFemSpace
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
    def __call__(self, fun, dofs_only = False):
        r"""
        Project vector function onto the H(curl)-conforming finite element
        space. This happens in the logical domain $\hat{\Omega}$.

        Parameters
        ----------
        fun : list/tuple of callables
            Scalar components of the real- or complex-valued vector function to be
            projected, with arguments the coordinates (x_1, ..., x_N) of a
            point in the logical domain. These correspond to the coefficients
            of a 1-form in the canonical basis (dx_1, ..., dx_N).

            $fun_i : \hat{\Omega} \mapsto \mathbb{R}$ with i = 1, ..., N.

        Returns
        -------
        field : FemField
            Field obtained by projection (element of the H(curl)-conforming
            finite element space). This is also a real- or complex-valued vector function
            in the logical domain.
        """
        return super().__call__(fun, dofs_only = dofs_only)

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
    Hdiv : VectorFemSpace
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
    def __call__(self, fun, dofs_only = False):
        r"""
        Project vector function onto the H(div)-conforming finite element
        space. This happens in the logical domain $\hat{\Omega}$.

        Parameters
        ----------
        fun : list/tuples of callable
            Scalar components of the real- or complex-valued vector function to be
            projected, with arguments the coordinates (x_1, ..., x_N) of a
            point in the logical domain. In 3D these correspond to the
            coefficients of a 2-form in the canonical basis (dx_1 ∧ dx_2,
            dx_2 ∧ dx_3, dx_3 ∧ dx_1).

            $fun_i : \hat{\Omega} \mapsto \mathbb{R}$ with i = 1, ..., N.

        Returns
        -------
        field : FemField
            Field obtained by projection (element of the H(div)-conforming
            finite element space). This is also a real- or complex-valued vector function
            in the logical domain.
        """
        return super().__call__(fun, dofs_only = dofs_only)

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
    def __call__(self, fun, dofs_only = False):
        r"""
        Project scalar function onto the L2-conforming finite element space.
        This happens in the logical domain $\hat{\Omega}$.

        Parameters
        ----------
        fun : callable
            Real- or complex-valued scalar function to be projected, with arguments the
            coordinates (x_1, ..., x_N) of a point in the logical domain. This
            corresponds to the coefficient of an N-form in N dimensions, in
            the canonical basis dx_1 ∧ ... ∧ dx_N.

            $fun : \hat{\Omega} \mapsto \mathbb{R}$.

        Returns
        -------
        field : FemField
            Field obtained by projection (element of the L2-conforming finite
            element space). This is also a real- or complex-valued scalar function in the
            logical domain.
        """
        return super().__call__(fun, dofs_only = dofs_only)

class Projector_H1vec(GlobalProjector):
    """
    Projector from H1^3 = H1 x H1 x H1 to a conforming finite element space, i.e.
    a finite dimensional subspace of H1^3, constructed with tensor-product
    B-splines in 2 or 3 dimensions.
    This is a global projector constructed over a tensor-product grid in the
    logical domain. The vertices of this grid are obtained as the tensor
    product of the 1D splines' Greville points along each direction.
    
    Parameters
    ----------
    H1vec : ProductFemSpace
        H1 x H1 x H1-conforming finite element space, codomain of the projection
        operator.
        
    nquads : list(int) | tuple(int)
        Number of quadrature points along each direction, to be used in Gauss
        quadrature rule for computing the (approximated) degrees of freedom.
    """
    def _structure(self, dim):
        if dim == 3:
            return [
                ['I', 'I', 'I'],
                ['I', 'I', 'I'],
                ['I', 'I', 'I']
            ]
        elif dim == 2:
            return [
                ['I', 'I'],
                ['I', 'I']
            ]
        else:
            raise NotImplementedError('The H1vec projector is only available in 2D or 3D.')
    
    def _function(self, dim):
        if dim == 3: return evaluate_dofs_3d_vec
        elif dim == 2: return evaluate_dofs_2d_vec
        else:
            raise NotImplementedError('The H1vec projector is only available in 2/3D.')

    #--------------------------------------------------------------------------
    def __call__(self, fun, dofs_only = False):
        r"""
        Project vector function onto the H1 x H1 x H1-conforming finite element
        space. This happens in the logical domain $\hat{\Omega}$.

        Parameters
        ----------
        fun : list/tuple of callables
            Scalar components of the real-valued vector function to be
            projected, with arguments the coordinates (x_1, ..., x_N) of a
            point in the logical domain. These correspond to the coefficients
            of a vector-field.
            $fun_i : \hat{\Omega} \mapsto \mathbb{R}$ with i = 1, ..., N.

        Returns
        -------
        field : FemField
            Field obtained by projection (element of the H1^3-conforming
            finite element space). This is also a real-valued vector function
            in the logical domain.
        """
        return super().__call__(fun, dofs_only = dofs_only)

#==============================================================================
# 1D DEGREES OF FREEDOM
#==============================================================================

def evaluate_dofs_1d_0form(
        intp_x1,     # interpolation points
        F,           # array of degrees of freedom (intent out)
        f,           # input scalar function (callable)
        ):
    
    # evaluate input functions at interpolation points (make sure that points are in [0, 1])
    assert np.all(np.logical_and(intp_x1 >= 0., intp_x1 <= 1.))
    
    E1, = np.meshgrid(intp_x1, indexing='ij')
    f_pts = f(E1)
    
    F_temp = np.zeros_like(F, order='C')
    
    dof_kernels.evaluate_dofs_1d_0form(F_temp, f_pts)
    
    F[:] = F_temp
        
#------------------------------------------------------------------------------
def evaluate_dofs_1d_1form(
        quad_x1,       # quadrature points
        quad_w1,       # quadrature weights
        F,             # array of degrees of freedom (intent out)
        f,             # input scalar function (callable)
        ):

    # evaluate input functions at quadrature points (make sure that points are in [0, 1])
    E1, = np.meshgrid(quad_x1.flatten()%1., indexing='ij')
    f_pts = f(E1)
    
    # call kernel
    F_temp = np.zeros_like(F, order='C')
    
    dof_kernels.evaluate_dofs_1d_1form(quad_w1, F_temp, f_pts)
    
    F[:] = F_temp

#==============================================================================
# 2D DEGREES OF FREEDOM
#==============================================================================

def evaluate_dofs_2d_0form(
        intp_x1, intp_x2,     # interpolation points
        F,                    # array of degrees of freedom (intent out)
        f,                    # input scalar function (callable)
        ):
    
    # evaluate input functions at interpolation points (make sure that points are in [0, 1])
    assert np.all(np.logical_and(intp_x1 >= 0., intp_x1 <= 1.))
    assert np.all(np.logical_and(intp_x2 >= 0., intp_x2 <= 1.))
    
    E1, E2 = np.meshgrid(intp_x1, intp_x2, indexing='ij')
    f_pts = f(E1, E2)
    
    F_temp = np.zeros_like(F, order='C')
    
    dof_kernels.evaluate_dofs_2d_0form(F_temp, f_pts)
    
    F[:, :] = F_temp

#------------------------------------------------------------------------------
def evaluate_dofs_2d_1form_hcurl(
        intp_x1, intp_x2,      # interpolation points
        quad_x1, quad_x2,      # quadrature points
        quad_w1, quad_w2,      # quadrature weights
        F1, F2,                # arrays of degrees of freedom (intent out)
        f1, f2,                # input scalar functions (callable)
        ):

    # evaluate input functions at quadrature/interpolation points (make sure that points are in [0, 1])
    assert np.all(np.logical_and(intp_x1 >= 0., intp_x1 <= 1.))
    assert np.all(np.logical_and(intp_x2 >= 0., intp_x2 <= 1.))
    
    E1, E2 = np.meshgrid(quad_x1.flatten()%1., intp_x2, indexing='ij')
    f1_pts = f1(E1, E2)
    
    E1, E2 = np.meshgrid(intp_x1, quad_x2.flatten()%1., indexing='ij')
    f2_pts = f2(E1, E2)
    
    # call kernel
    F1_temp = np.zeros_like(F1, order='C')
    F2_temp = np.zeros_like(F2, order='C')
    
    dof_kernels.evaluate_dofs_2d_1form_hcurl(quad_w1, quad_w2, F1_temp, F2_temp, f1_pts, f2_pts)
    
    F1[:, :] = F1_temp
    F2[:, :] = F2_temp

#------------------------------------------------------------------------------
def evaluate_dofs_2d_1form_hdiv(
        intp_x1, intp_x2,       # interpolation points
        quad_x1, quad_x2,       # quadrature points
        quad_w1, quad_w2,       # quadrature weights
        F1, F2,                 # arrays of degrees of freedom (intent out)
        f1, f2,                 # input scalar functions (callable)
        ):

    # evaluate input functions at quadrature/interpolation points (make sure that points are in [0, 1])
    assert np.all(np.logical_and(intp_x1 >= 0., intp_x1 <= 1.))
    assert np.all(np.logical_and(intp_x2 >= 0., intp_x2 <= 1.))
    
    E1, E2 = np.meshgrid(intp_x1, quad_x2.flatten()%1., indexing='ij')
    f1_pts = f1(E1, E2)
    
    E1, E2 = np.meshgrid(quad_x1.flatten()%1., intp_x2, indexing='ij')
    f2_pts = f2(E1, E2)
    
    # call kernel
    F1_temp = np.zeros_like(F1, order='C')
    F2_temp = np.zeros_like(F2, order='C')
    
    dof_kernels.evaluate_dofs_2d_1form_hdiv(quad_w1, quad_w2, F1_temp, F2_temp, f1_pts, f2_pts)
    
    F1[:, :, :] = F1_temp
    F2[:, :, :] = F2_temp

#------------------------------------------------------------------------------
def evaluate_dofs_2d_2form(
        quad_x1, quad_x2,       # quadrature points
        quad_w1, quad_w2,       # quadrature weights
        F,                      # array of degrees of freedom (intent out)
        f,                      # input scalar function (callable)
        ):

    # evaluate input functions at quadrature points (make sure that points are in [0, 1])
    E1, E2 = np.meshgrid(quad_x1.flatten()%1., quad_x2.flatten()%1., indexing='ij')
    f_pts = f(E1, E2)
    
    # call kernel
    F_temp = np.zeros_like(F, order='C')
    
    dof_kernels.evaluate_dofs_2d_2form(quad_w1, quad_w2, F_temp, f_pts)
    
    F[:, :] = F_temp

#------------------------------------------------------------------------------    
def evaluate_dofs_2d_vec(
        intp_x1, intp_x2,      # interpolation points
        F1, F2,                # array of degrees of freedom (intent out)
        f1, f2,                # input scalar function (callable)
        ):
    
    # evaluate input functions at interpolation points (make sure that points are in [0, 1])
    assert np.all(np.logical_and(intp_x1 >= 0., intp_x1 <= 1.))
    assert np.all(np.logical_and(intp_x2 >= 0., intp_x2 <= 1.))
    
    E1, E2 = np.meshgrid(intp_x1, intp_x2, indexing='ij')
    f1_pts = f1(E1, E2)
    f2_pts = f2(E1, E2)
    
    # call kernel
    F1_temp = np.zeros_like(F1, order='C')
    F2_temp = np.zeros_like(F2, order='C')
    
    dof_kernels.evaluate_dofs_2d_vec(F1_temp, F2_temp, f1_pts, f2_pts)
    
    F1[:, :, :] = F1_temp
    F2[:, :, :] = F2_temp
    
#==============================================================================
# 3D DEGREES OF FREEDOM
#==============================================================================

def evaluate_dofs_3d_0form(
        intp_x1, intp_x2, intp_x3, # interpolation points
        F,                         # array of degrees of freedom (intent out)
        f,                         # input scalar function (callable)
        ):
    
    # evaluate input functions at interpolation points (make sure that points are in [0, 1])
    assert np.all(np.logical_and(intp_x1 >= 0., intp_x1 <= 1.))
    assert np.all(np.logical_and(intp_x2 >= 0., intp_x2 <= 1.))
    assert np.all(np.logical_and(intp_x3 >= 0., intp_x3 <= 1.))
    
    E1, E2, E3 = np.meshgrid(intp_x1, intp_x2, intp_x3, indexing='ij')
    f_pts = f(E1, E2, E3)
    
    F_temp = np.zeros_like(F, order='C')
    
    dof_kernels.evaluate_dofs_3d_0form(F_temp, f_pts)
    
    F[:, :, :] = F_temp

#------------------------------------------------------------------------------
def evaluate_dofs_3d_1form(
        intp_x1, intp_x2, intp_x3, # interpolation points
        quad_x1, quad_x2, quad_x3, # quadrature points
        quad_w1, quad_w2, quad_w3, # quadrature weights
        F1, F2, F3,                # arrays of degrees of freedom (intent out)
        f1, f2, f3                 # input scalar functions (callable)
        ):

    # evaluate input functions at quadrature/interpolation points (make sure that points are in [0, 1])
    assert np.all(np.logical_and(intp_x1 >= 0., intp_x1 <= 1.))
    assert np.all(np.logical_and(intp_x2 >= 0., intp_x2 <= 1.))
    assert np.all(np.logical_and(intp_x3 >= 0., intp_x3 <= 1.))
    
    E1, E2, E3 = np.meshgrid(quad_x1.flatten()%1., intp_x2, intp_x3, indexing='ij')
    f1_pts = f1(E1, E2, E3)
    
    E1, E2, E3 = np.meshgrid(intp_x1, quad_x2.flatten()%1., intp_x3, indexing='ij')
    f2_pts = f2(E1, E2, E3)
    
    E1, E2, E3 = np.meshgrid(intp_x1, intp_x2, quad_x3.flatten()%1., indexing='ij')
    f3_pts = f3(E1, E2, E3)
    
    # call kernel
    F1_temp = np.zeros_like(F1, order='C')
    F2_temp = np.zeros_like(F2, order='C')
    F3_temp = np.zeros_like(F3, order='C')
    
    dof_kernels.evaluate_dofs_3d_1form(quad_w1, quad_w2, quad_w3, F1_temp, F2_temp, F3_temp, f1_pts, f2_pts, f3_pts)
    
    F1[:, :, :] = F1_temp
    F2[:, :, :] = F2_temp
    F3[:, :, :] = F3_temp

#------------------------------------------------------------------------------
def evaluate_dofs_3d_2form(
        intp_x1, intp_x2, intp_x3, # interpolation points
        quad_x1, quad_x2, quad_x3, # quadrature points
        quad_w1, quad_w2, quad_w3, # quadrature weights
        F1, F2, F3,                # arrays of degrees of freedom (intent out)
        f1, f2, f3                 # input scalar functions (callable)
        ):

    # evaluate input functions at quadrature/interpolation points (make sure that points are in [0, 1])
    assert np.all(np.logical_and(intp_x1 >= 0., intp_x1 <= 1.))
    assert np.all(np.logical_and(intp_x2 >= 0., intp_x2 <= 1.))
    assert np.all(np.logical_and(intp_x3 >= 0., intp_x3 <= 1.))
    
    E1, E2, E3 = np.meshgrid(intp_x1, quad_x2.flatten()%1., quad_x3.flatten()%1., indexing='ij')
    f1_pts = f1(E1, E2, E3)
    
    E1, E2, E3 = np.meshgrid(quad_x1.flatten()%1., intp_x2, quad_x3.flatten()%1., indexing='ij')
    f2_pts = f2(E1, E2, E3)
    
    E1, E2, E3 = np.meshgrid(quad_x1.flatten()%1., quad_x2.flatten()%1., intp_x3, indexing='ij')
    f3_pts = f3(E1, E2, E3)
    
    # call kernel
    F1_temp = np.zeros_like(F1, order='C')
    F2_temp = np.zeros_like(F2, order='C')
    F3_temp = np.zeros_like(F3, order='C')
    
    dof_kernels.evaluate_dofs_3d_2form(quad_w1, quad_w2, quad_w3, F1_temp, F2_temp, F3_temp, f1_pts, f2_pts, f3_pts)
    
    F1[:, :, :] = F1_temp
    F2[:, :, :] = F2_temp
    F3[:, :, :] = F3_temp

#------------------------------------------------------------------------------
def evaluate_dofs_3d_3form(
        quad_x1, quad_x2, quad_x3, # quadrature points
        quad_w1, quad_w2, quad_w3, # quadrature weights
        F,                         # array of degrees of freedom (intent out)
        f,                         # input scalar function (callable)
        ):

    # evaluate input functions at quadrature points (make sure that points are in [0, 1])
    E1, E2, E3 = np.meshgrid(quad_x1.flatten()%1., quad_x2.flatten()%1., quad_x3.flatten()%1., indexing='ij')
    f_pts = f(E1, E2, E3)
    
    # call kernel
    F_temp = np.zeros_like(F, order='C')
    
    dof_kernels.evaluate_dofs_3d_3form(quad_w1, quad_w2, quad_w3, F_temp, f_pts)
    
    F[:, :, :] = F_temp

#------------------------------------------------------------------------------
def evaluate_dofs_3d_vec(
        intp_x1, intp_x2, intp_x3, # interpolation points
        F1, F2, F3,                # array of degrees of freedom (intent out)
        f1, f2, f3,                # input scalar function (callable)
        ):
    
    # evaluate input functions at interpolation points (make sure that points are in [0, 1])
    assert np.all(np.logical_and(intp_x1 >= 0., intp_x1 <= 1.))
    assert np.all(np.logical_and(intp_x2 >= 0., intp_x2 <= 1.))
    assert np.all(np.logical_and(intp_x3 >= 0., intp_x3 <= 1.))
    
    E1, E2, E3 = np.meshgrid(intp_x1, intp_x2, intp_x3, indexing='ij')
    f1_pts = f1(E1, E2, E3)
    f2_pts = f2(E1, E2, E3)
    f3_pts = f3(E1, E2, E3)
    
    # call kernel
    F1_temp = np.zeros_like(F1, order='C')
    F2_temp = np.zeros_like(F2, order='C')
    F3_temp = np.zeros_like(F3, order='C')
    
    dof_kernels.evaluate_dofs_3d_vec(F1_temp, F2_temp, F3_temp, f1_pts, f2_pts, f3_pts)
    
    F1[:, :, :] = F1_temp
    F2[:, :, :] = F2_temp
    F3[:, :, :] = F3_temp