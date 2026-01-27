#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import sys
import os
import importlib

import numpy as np

from sympy                  import ImmutableDenseMatrix, Matrix, Symbol, sympify
from sympy.tensor.indexed   import Indexed, IndexedBase
from sympy.simplify         import cse_main
from sympy.printing.pycode  import pycode

from pyccel import epyccel

from sympde.topology.basic       import Boundary, Interface
from sympde.topology.mapping     import Mapping, SymbolicExpr
from sympde.topology.space       import ScalarFunction, VectorFunction, IndexedVectorFunction
from sympde.topology.derivatives import get_atom_logical_derivatives
from sympde.topology.derivatives import _logical_partial_derivatives
from sympde.topology.derivatives import get_index_logical_derivatives
from sympde.topology.derivatives import get_max_logical_partial_derivatives # NOTE [YG 31.07.2025]: Maybe use the one in ast.utilities
from sympde.expr.expr            import BilinearForm
from sympde.expr.evaluation      import KernelExpression, TerminalExpr
from sympde.calculus.core        import PlusInterfaceOperator

from psydac.cad.geometry      import Geometry
from psydac.mapping.discrete  import SplineMapping, NurbsMapping
from psydac.fem.basic         import FemSpace, FemField
from psydac.fem.vector        import VectorFemSpace
from psydac.linalg.stencil    import StencilMatrix
from psydac.linalg.block      import BlockVectorSpace, BlockLinearOperator
from psydac.api.grid          import QuadratureGrid, BasisValues
from psydac.api.settings      import PSYDAC_BACKENDS
from psydac.api.utilities     import flatten, random_string
from psydac.api.fem_common    import (
    compute_imports,
    compute_max_nderiv,
    compute_free_arguments,
    construct_test_space_arguments,
    construct_trial_space_arguments,
    construct_quad_grids_arguments,
    reset_arrays,
    do_nothing,
    extract_stencil_mats,
)

# TODO [YG 01.08.2025]: Avoid importing anything from psydac.pyccel
from psydac.pyccel.ast.core import _atomic, Assign

__all__ = ('DiscreteBilinearForm',)

NoneType = type(None)

#==============================================================================
class DiscreteBilinearForm:
    """
    Discrete bilinear form ready to be assembled into a matrix.

    This class represents the concept of a discrete bilinear form in PSYDAC.
    Instances of this class generate an appropriate matrix assembly kernel,
    allocate the matrix if not provided, and prepare a list of arguments for
    the kernel.

    An implementation of the sum factorization algorithm is used to assemble
    the matrix.

    Parameters
    ----------

    expr : sympde.expr.expr.BilinearForm
        The symbolic bilinear form.

    kernel_expr : list or tuple of sympde.expr.evaluation.KernelExpression
        The atomic representation of the bilinear form.

    domain_h : psydac.cad.geometry.Geometry
        The discretized domain.

    spaces : list of psydac.fem.basic.FemSpace
        The discrete trial and test spaces.

    nquads : list or tuple of int
        The number of quadrature points used in the assembly kernel along each
        direction.

    matrix : psydac.linalg.stencil.StencilMatrix or psydac.linalg.block.BlockLinearOperator, optional
        The matrix that we assemble into. If not provided, a new matrix is
        created with the appropriate domain and codomain (default: None).

    update_ghost_regions : bool, default=True
        Accumulate the contributions of the neighbouring processes.

    backend : dict, optional
        The backend used to accelerate the computing kernels.
        The backend dictionaries are defined in the file psydac/api/settings.py

    assembly_backend : dict, optional
        The backend used to accelerate the assembly kernel.
        The backend dictionaries are defined in the file psydac/api/settings.py

    linalg_backend : dict, optional
        The backend used to accelerate the computing kernels of the linear operator.
        The backend dictionaries are defined in the file psydac/api/settings.py

    symbolic_mapping : sympde.topology.mapping.Mapping, optional
        The symbolic mapping which defines the physical domain of the bilinear form.

    See Also
    --------
    DiscreteLinearForm
    DiscreteFunctional
    DiscreteSumForm

    """
    def __init__(self, expr, kernel_expr, domain_h, spaces, *, nquads,
                 matrix=None, update_ghost_regions=True, backend=None,
                 linalg_backend=None, assembly_backend=None,
                 symbolic_mapping=None):

        #... Sanity checks
        assert isinstance(expr, BilinearForm)
        assert isinstance(domain_h, Geometry)
        for space in spaces:
            assert isinstance(space, FemSpace)
        for nquad in nquads:
            assert isinstance(nquad, int)
            assert nquad > 0
        assert isinstance(matrix, (NoneType, StencilMatrix, BlockLinearOperator))
        assert isinstance(update_ghost_regions, bool)
        assert isinstance(         backend, (NoneType, dict))
        assert isinstance(  linalg_backend, (NoneType, dict))
        assert isinstance(assembly_backend, (NoneType, dict))
        assert isinstance(symbolic_mapping, (NoneType, Mapping))
        #...

        if isinstance(kernel_expr, (tuple, list)):
            if len(kernel_expr) == 1:
                kernel_expr = kernel_expr[0]
            else:
                raise ValueError('> Expecting only one kernel')
        assert isinstance(kernel_expr, KernelExpression)

        self._kernel_expr = kernel_expr
        self._expr   = expr
        self._target = kernel_expr.target
        self._domain = domain_h.domain
        self._spaces = spaces
        self._matrix = matrix
        self._func   = None    # The assembly function will be generated later

        domain = self.domain
        target = self.target

        # ...
        if len(domain) > 1:
            i, j = self.get_space_indices_from_target(domain, target)
            test_space  = self.spaces[1].spaces[i]
            trial_space = self.spaces[0].spaces[j]
            if isinstance(target, Interface):
                m,_       = self.get_space_indices_from_target(domain, target.minus)
                p,_       = self.get_space_indices_from_target(domain, target.plus)
                mapping_m = list(domain_h.mappings.values())[m]
                mapping_p = list(domain_h.mappings.values())[p]
                mapping   = (mapping_m, mapping_p) if mapping_m else None
            else:
                mapping = list(domain_h.mappings.values())[i]
        else:
            trial_space = self.spaces[0]
            test_space  = self.spaces[1]
            mapping     = list(domain_h.mappings.values())[0]

        self._mapping = mapping

        is_rational_mapping = False
        mapping_space       = None
        if (mapping is not None) and not isinstance(target, Interface):
            is_rational_mapping = isinstance(mapping, NurbsMapping)
            mapping_space = mapping.space
        elif (mapping is not None) and isinstance(target, Interface):
            is_rational_mapping = (isinstance(mapping[0], NurbsMapping), isinstance(mapping[1], NurbsMapping))
            mapping_space = (mapping[0].space, mapping[1].space)

        self._is_rational_mapping = is_rational_mapping
        # ...

        if isinstance(test_space.coeff_space, BlockVectorSpace):
            coeff_space = test_space.coeff_space.spaces[0]
        else:
            coeff_space = test_space.coeff_space

        self._coeff_space = coeff_space
        self._num_threads = 1
        if coeff_space.parallel and coeff_space.cart.num_threads > 1:
            self._num_threads = coeff_space.cart.num_threads

        self._update_ghost_regions = update_ghost_regions

        # In case of multiple patches, if the communicator is MPI_COMM_NULL, we do not generate the assembly code
        # because the patch is not owned by the MPI rank.
        if coeff_space.parallel and coeff_space.cart.is_comm_null:
            self._free_args = ()
            self._func      = do_nothing
            self._args      = ()
            self._threads_args        = ()
            self._global_matrices     = ()
            self._update_ghost_regions = False
            return

        # ...
        test_ext  = None
        trial_ext = None
        if isinstance(target, Boundary):
            axis      = target.axis
            test_ext  = target.ext
            trial_ext = target.ext
        elif isinstance(target, Interface):
            # this part treats the cases of:
            # integral(v_minus * u_plus)
            # integral(v_plus  * u_minus)
            # the other cases, integral(v_minus * u_minus) and integral(v_plus * u_plus)
            # are converted to boundary integrals by Sympde
            axis         = target.axis
            test         = self.kernel_expr.test
            trial        = self.kernel_expr.trial
            test_target  = target.plus if isinstance( test, PlusInterfaceOperator) else target.minus
            trial_target = target.plus if isinstance(trial, PlusInterfaceOperator) else target.minus
            test_ext     = test_target.ext
            trial_ext    = trial_target.ext
            ncells       = tuple(max(i, j) for i, j in zip(test_space.ncells, trial_space.ncells))
            if isinstance(trial_space, VectorFemSpace):
                spaces = []
                for sp in trial_space.spaces:
                    if (trial_target.axis, trial_target.ext) in sp.interfaces:
                        spaces.append(sp.get_refined_space(ncells).interfaces[trial_target.axis, trial_target.ext])

                if len(spaces) == len(trial_space.spaces):
                    sym_space   = trial_space.symbolic_space
                    trial_space = VectorFemSpace(*spaces)
                    trial_space.symbolic_space = sym_space

            elif (trial_target.axis, trial_target.ext) in trial_space.interfaces:
                sym_space   = trial_space.symbolic_space
                trial_space = trial_space.get_refined_space(ncells).interfaces[trial_target.axis, trial_target.ext]
                trial_space.symbolic_space = sym_space

            test_space      = test_space.get_refined_space(ncells)
            self._test_ext  = test_target.ext
            self._trial_ext = trial_target.ext

        #...

        # Determine the type of scalar quantities to be managed in the code
        dtypes = [getattr(V.symbolic_space, 'codomain_type', 'real') for V in (trial_space, test_space)]
        assert all(t in ['complex', 'real'] for t in dtypes)
        self._dtype = 'complex128' if 'complex' in dtypes else 'float64'

        # Assuming that all vector spaces (and their Cartesian decomposition,
        # if any) are compatible with each other, extract the first available
        # vector space from which (starts, ends, npts) will be read:
        starts = coeff_space.starts
        ends   = coeff_space.ends
        npts   = coeff_space.npts

        # MPI communicator
        comm = coeff_space.cart.comm if coeff_space.parallel else None

        # Store the MPI communicator (or None)
        self._comm = comm

        #...
        # Get default backend from environment, or use 'python'.
        default_backend = PSYDAC_BACKENDS.get(os.environ.get('PSYDAC_BACKEND'))\
                       or PSYDAC_BACKENDS['python']

        # Backends for code generation
        assembly_backend = backend or assembly_backend
        linalg_backend   = backend or linalg_backend

        # Store backend dictionary
        self._backend = assembly_backend or default_backend
        #...

        # TODO: remove
        # BasicDiscrete generates the assembly code and sets the following attributes that are used afterwards:
        # self._func, self._free_args, self._max_nderiv and self._backend
#        BasicDiscrete.__init__(self, expr, kernel_expr, comm=comm, root=0, discrete_space=discrete_space,
#                       nquads=nquads, is_rational_mapping=is_rational_mapping, mapping=symbolic_mapping,
#                       mapping_space=mapping_space, num_threads=self._num_threads, backend=assembly_backend)


        #... Compute the string with all the imports
        texpr = kernel_expr
        sym_expr = SymbolicExpr(texpr.expr)
        imports = compute_imports(sym_expr, spaces=(trial_space, test_space), openmp=False)
        indent = 4
        glue = '\n' + ' '* indent
        imports_str = glue.join([f"from {m} import {', '.join(vars)}"
                                 for m, vars in imports.items()])

        # Broadcast the import information (sqrt, sin, pi, ...) to all processes
        if (comm is not None) and (comm.size > 1):
            imports_str = comm.bcast(imports_str, root=0)

        # Store the imports string as it will be used by make_file()
        self._imports_string = imports_str
        #...

        # Compute the highest order of derivation in the kernel expression
        self._max_nderiv = compute_max_nderiv(kernel_expr)

        # TODO [YG 31.07.2025]: Implement this
        self._free_args = compute_free_arguments(expr, kernel_expr)

        #... Handle the special case where the current MPI process does not need to do anything
        if isinstance(target, (Boundary, Interface)):

            # If process does not own the boundary or interface, do not assemble anything
            if test_ext == -1:
                if starts[axis] != 0:
                    self._func = do_nothing

            elif test_ext == 1:
                if ends[axis] != npts[axis]-1:
                    self._func = do_nothing

            # In case of target==Interface, we only use the MPI ranks that are on the interface to assemble the BilinearForm
            if self._func == do_nothing and isinstance(target, Interface):
                self._free_args = ()
                self._args      = ()
                self._global_matrices = ()
                self._threads_args    = ()
                return
        #...

        #... Build the quadrature grids
        if isinstance(target, Boundary):
            test_grid  = QuadratureGrid( test_space, axis=axis, ext= test_ext, nquads=nquads)
            trial_grid = QuadratureGrid(trial_space, axis=axis, ext=trial_ext, nquads=nquads)
            self._grid = (test_grid,)
        elif isinstance(target, Interface):
            # this part treats the cases of:
            # integral(v_minus * u_plus)
            # integral(v_plus  * u_minus)
            # the other cases, integral(v_minus * u_minus) and integral(v_plus * u_plus)
            # are converted to boundary integrals by Sympde
            test_grid  = QuadratureGrid( test_space, axis=axis, ext= test_ext, nquads=nquads)
            trial_grid = QuadratureGrid(trial_space, axis=axis, ext=trial_ext, nquads=nquads)
            self._grid = (test_grid, trial_grid) if test_target == target.minus else (trial_grid, test_grid)
            self._test_ext  =  test_target.ext
            self._trial_ext = trial_target.ext
        else:
            test_grid  = QuadratureGrid( test_space, nquads=nquads)
            trial_grid = QuadratureGrid(trial_space, nquads=nquads)
            self._grid = (test_grid,)
        #...

        # Extract the basis function values on the quadrature grids
        self._test_basis  = BasisValues(
            test_space,
            nderiv = self.max_nderiv,
            nquads = nquads,
            trial  = False,
            grid   = test_grid
        )
        self._trial_basis = BasisValues(
            trial_space,
            nderiv = self.max_nderiv,
            nquads = nquads,
            trial  = True ,
            grid   = trial_grid
        )

        # Allocate the output matrix, if needed
        self.allocate_matrices(linalg_backend)

        # Determine whether OpenMP instructions were generated
        self._with_openmp = (assembly_backend['name'] == 'pyccel' and assembly_backend['openmp']) if assembly_backend else False

        # Construct the arguments to be passed to the assemble() function, which is stored in self._func
        # First we generate the assembly file

        # pyccelize process of computing the test_trial arrays
        # currently set to False, as a Python 3.9 test fails, and due to the "speed up" not being significant
        self._pyccelize_test_trial_computation = False

        # no openmp support yet: with_openmp is not passed
        self._args, self._threads_args = self.construct_arguments_generate_assembly_file()

    #--------------------------------------------------------------------------
    @property
    def comm(self):
        return self._comm

    @property
    def expr(self):
        return self._expr

    @property
    def kernel_expr(self):
        return self._kernel_expr

    @property
    def domain(self):
        return self._domain

    @property
    def mapping(self):
        return self._mapping

    @property
    def is_rational_mapping(self):
        return self._is_rational_mapping

    @property
    def target(self):
        return self._target

    @property
    def spaces(self):
        return self._spaces

    @property
    def test_basis(self):
        return self._test_basis

    @property
    def trial_basis(self):
        return self._trial_basis

    @property
    def grid(self):
        return self._grid

    @property
    def nquads(self):
        return self._grid[0].nquads

    @property
    def free_args(self):
        return self._free_args

    @property
    def max_nderiv(self):
        # TODO: compute with read_BilinearForm and store
        return self._max_nderiv

    @property
    def backend(self):
        return self._backend

    @property
    def args(self):
        return self._args

    @property
    def global_matrices(self):
        return self._global_matrices

    #--------------------------------------------------------------------------
    def allocate_matrices(self, backend=None):
        """
        Allocate the global matrices used in the assembly method.
        In this method we allocate only the matrices that are computed in the self._target domain,
        we also avoid double allocation if we have many DiscreteLinearForm that are defined on the same self._target domain.

        Parameters
        ----------
        backend : dict
         The backend used to accelerate the computing kernels.

        """
        global_mats     = {}

        expr            = self.kernel_expr.expr
        target          = self.kernel_expr.target
        test_degree     = np.array(self.test_basis.space.degree)
        trial_degree    = np.array(self.trial_basis.space.degree)
        test_space      = self.spaces[1].coeff_space
        trial_space     = self.spaces[0].coeff_space
        test_fem_space  = self.spaces[1]
        trial_fem_space = self.spaces[0]
        domain          = self.domain
        is_broken       = len(domain) > 1
        is_conformal    = True

        if isinstance(expr, (ImmutableDenseMatrix, Matrix)):
            if not isinstance(test_degree[0],(list, tuple, np.ndarray)):
                test_degree = [test_degree]

            if not isinstance(trial_degree[0],(list, tuple, np.ndarray)):
                trial_degree = [trial_degree]

            pads = np.empty((len(test_degree),len(trial_degree),len(test_degree[0])), dtype=int)
            for i in range(len(test_degree)):
                for j in range(len(trial_degree)):
                    td  = test_degree[i]
                    trd = trial_degree[j]
                    pads[i,j][:] = np.array([td, trd]).max(axis=0)
        else:
            pads = np.maximum(test_degree, trial_degree)

        if self._matrix is None and (is_broken or isinstance(expr, (ImmutableDenseMatrix, Matrix))):
            self._matrix = BlockLinearOperator(trial_space, test_space)

        if is_broken:
            i, j = self.get_space_indices_from_target(domain, target)
            test_fem_space   = self.spaces[1].spaces[i]
            trial_fem_space  = self.spaces[0].spaces[j]
            test_space  = test_space.spaces[i]
            trial_space = trial_space.spaces[j]
            ncells = tuple(max(i,j) for i,j in zip(test_fem_space.ncells, trial_fem_space.ncells))
            is_conformal = tuple(test_fem_space.ncells) == ncells and tuple(trial_fem_space.ncells) == ncells
            if is_broken and not is_conformal and not i==j:
                use_restriction = all(trn>=tn for trn,tn in zip(trial_fem_space.ncells, test_fem_space.ncells))
                use_prolongation = not use_restriction

        else:
            ncells = tuple(max(i,j) for i,j in zip(test_fem_space.ncells, trial_fem_space.ncells))
            i=0
            j=0
            #else so initialisation causing bug on line 682

        if isinstance(expr, (ImmutableDenseMatrix, Matrix)): # case of system of equations

            if is_broken: #multi patch
                if not self._matrix[i,j]:
                    mat = BlockLinearOperator(trial_fem_space.get_refined_space(ncells).coeff_space, test_fem_space.get_refined_space(ncells).coeff_space)
                    if not is_conformal and not i==j:
                        if use_restriction:
                            Ps  = [knot_insertion_projection_operator(ts.get_refined_space(ncells), ts) for ts in test_fem_space.spaces]
                            P   = BlockLinearOperator(test_fem_space.get_refined_space(ncells).coeff_space, test_fem_space.coeff_space)
                            for ni,Pi in enumerate(Ps):
                                P[ni,ni] = Pi

                            mat = ComposedLinearOperator(trial_space, test_space, P, mat)

                        elif use_prolongation:
                            Ps  = [knot_insertion_projection_operator(trs, trs.get_refined_space(ncells)) for trs in trial_fem_space.spaces]
                            P   = BlockLinearOperator(trial_fem_space.coeff_space, trial_fem_space.get_refined_space(ncells).coeff_space)
                            for ni,Pi in enumerate(Ps):
                                P[ni,ni] = Pi

                            mat = ComposedLinearOperator(trial_space, test_space, mat, P)

                    self._matrix[i,j] = mat

                matrix = self._matrix[i,j]
            else: # single patch
                matrix = self._matrix

            shape = expr.shape
            for k1 in range(shape[0]):
                for k2 in range(shape[1]):
                    if expr[k1,k2].is_zero:
                        continue

                    if isinstance(test_fem_space, VectorFemSpace):
                        ts_space = test_fem_space.get_refined_space(ncells).coeff_space.spaces[k1]
                    else:
                        ts_space = test_fem_space.get_refined_space(ncells).coeff_space

                    if isinstance(trial_fem_space, VectorFemSpace):
                        tr_space = trial_fem_space.get_refined_space(ncells).coeff_space.spaces[k2]
                    else:
                        tr_space = trial_fem_space.get_refined_space(ncells).coeff_space

                    if is_conformal and matrix[k1, k2]:
                        global_mats[k1, k2] = matrix[k1, k2]
                    elif not i == j: # assembling in an interface (type(target) == Interface)
                        axis    = target.axis
                        ext_d   = self._trial_ext
                        ext_c   = self._test_ext
                        test_n  = self. test_basis.space.spaces[k1].spaces[axis].nbasis
                        test_s  = self. test_basis.space.spaces[k1].coeff_space.starts[axis]
                        trial_n = self.trial_basis.space.spaces[k2].spaces[axis].nbasis
                        cart    = self.trial_basis.space.spaces[k2].coeff_space.cart
                        trial_s = cart.global_starts[axis][cart._coords[axis]]

                        s_d = trial_n - trial_s - trial_degree[k2][axis] - 1 if ext_d == 1 else 0
                        s_c =  test_n - trial_s -  test_degree[k1][axis] - 1 if ext_c == 1 else 0

                        # We only handle the case where direction = 1
                        direction = target.ornt
                        if domain.dim == 2:
                            assert direction == 1
                        elif domain.dim == 3:
                            assert all(d==1 for d in direction)

                        direction = 1
                        flip = [direction]*domain.dim
                        flip[axis] = 1
                        if self._func != do_nothing:
                            global_mats[k1, k2] = StencilInterfaceMatrix(tr_space, ts_space,
                                                                         s_d, s_c,
                                                                         axis, axis,
                                                                         ext_d, ext_c,
                                                                         pads=tuple(pads[k1, k2]),
                                                                         flip=flip)
                    else:
                        global_mats[k1, k2] = StencilMatrix(tr_space, ts_space, pads = tuple(pads[k1, k2]))

                    if is_conformal:
                        matrix[k1, k2] = global_mats[k1, k2]
                    elif use_restriction:
                        matrix.multiplicants[-1][k1, k2] = global_mats[k1, k2]
                    elif use_prolongation:
                        matrix.multiplicants[0][k1, k2] = global_mats[k1, k2]

        else: # case of scalar equation
            if is_broken: # multi-patch
                if self._matrix[i, j]:
                    global_mats[i, j] = self._matrix[i, j]

                elif not i == j: # assembling in an interface (type(target) == Interface)
                    axis   = target.axis
                    ext_d  = self._trial_ext
                    ext_c  = self._test_ext
                    test_n  = self.test_basis.space.spaces[axis].nbasis
                    test_s  = self.test_basis.space.coeff_space.starts[axis]
                    trial_n = self.trial_basis.space.spaces[axis].nbasis
                    cart    = self.trial_basis.space.coeff_space.cart
                    trial_s = cart.global_starts[axis][cart._coords[axis]]

                    s_d = trial_n - trial_s - trial_degree[axis] - 1 if ext_d == 1 else 0
                    s_c =  test_n - trial_s -  test_degree[axis] - 1 if ext_c == 1 else 0

                    # We only handle the case where direction = 1
                    direction = target.ornt
                    if domain.dim == 2:
                        assert direction == 1
                    elif domain.dim == 3:
                        assert all(d==1 for d in direction)

                    direction = 1
                    flip = [direction]*domain.dim
                    flip[axis] = 1

                    if self._func != do_nothing:
                        mat = StencilInterfaceMatrix(trial_fem_space.get_refined_space(ncells).coeff_space,
                                                     test_fem_space.get_refined_space(ncells).coeff_space,
                                                     s_d, s_c,
                                                     axis, axis,
                                                     ext_d, ext_c,
                                                     flip=flip)
                        if not is_conformal:
                            if use_restriction:
                                P   = knot_insertion_projection_operator(test_fem_space.get_refined_space(ncells), test_fem_space)
                                mat = ComposedLinearOperator(trial_space, test_space, P, mat)
                            elif use_prolongation:
                                P   = knot_insertion_projection_operator(trial_fem_space, trial_fem_space.get_refined_space(ncells))
                                mat = ComposedLinearOperator(trial_space, test_space, mat, P)

                        global_mats[i, j] = mat

                # define part of the global matrix as a StencilMatrix
                else:
                    global_mats[i, j] = StencilMatrix(trial_space, test_space, pads=tuple(pads))

                if (i, j) in global_mats:
                    self._matrix[i, j] = global_mats[i, j]


            # in single patch case, we define the matrices needed for the patch
            else:
                if self._matrix:
                    global_mats[0, 0] = self._matrix
                else:
                    global_mats[0, 0] = StencilMatrix(trial_space, test_space, pads=tuple(pads))

                self._matrix = global_mats[0, 0]

        # Set the backend of our matrices if given
        if backend is not None and is_broken:
            for mat in global_mats.values():
                mat.set_backend(backend)
        elif backend is not None:
            self._matrix.set_backend(backend)

        self._global_matrices = [M._data for M in extract_stencil_mats(global_mats.values())]

    #--------------------------------------------------------------------------
    def assemble(self, *, reset=True, **kwargs):
        """
        This method assembles the left hand side Matrix by calling the private method `self._func` with proper arguments.

        In the complex case, this function returns the matrix conjugate. This comes from the fact that the
        problem `a(u,v)=b(v)` is discretized as `A @ conj(U) = B` due to the antilinearity of `a` in the first variable.
        Thus, to obtain `U`, the assemble function returns `conj(A)`.

        TODO: remove these lines when the dot product is changed for complex.
        For now, since the dot product does not compute the conjugate in the complex case. We do not use the conjugate in the assemble function.
        It should work if the complex only comes from the `rhs` in the linear form.
        """

        if self._free_args:
            basis   = []
            spans   = []
            degrees = []
            pads    = []
            coeffs  = []
            consts  = []

            for key in self._free_args:
                v = kwargs[key]

                if len(self.domain) > 1 and isinstance(v, FemField) and (v.space.is_multipatch or v.space.is_vector_valued):
                    assert v.space.is_multipatch ## [MCP 27.03.2025] should hold since len(domain) > 1. If Ok we can simplify above if
                    i, j = self.get_space_indices_from_target(self.domain, self.target)
                    assert i == j
                    v = v[i]
                if isinstance(v, FemField):
                    assert len(self.grid) == 1
                    if not v.coeffs.ghost_regions_in_sync:
                        v.coeffs.update_ghost_regions()
                    basis_v = BasisValues(
                        v.space,
                        nderiv = self.max_nderiv,
                        nquads = self.nquads,
                        trial  = True,
                        grid   = self.grid[0]
                    )
                    bs, d, s, p, mult = construct_test_space_arguments(basis_v)
                    basis   += bs
                    spans   += s
                    degrees += [np.int64(a) for a in d]
                    pads    += [np.int64(a) for a in p]
                    if v.space.is_multipatch or v.space.is_vector_valued:
                        coeffs += (e._data for e in v.coeffs)
                    else:
                        coeffs += (v.coeffs._data, )
                else:
                    consts += (v, )

            args = (*self.args, *basis, *spans, *degrees, *pads, *coeffs, *consts)

        else:
            args = self._args

        if reset:
            reset_arrays(*self.global_matrices)

        self._func(*args, *self._threads_args)
        if self._matrix and self._update_ghost_regions:
            self._matrix.exchange_assembly_data()

        # TODO : uncomment this line when the conjugate is applied on the dot product in the complex case
        #self._matrix.conjugate(out=self._matrix)

        if self._matrix:
            self._matrix.ghost_regions_in_sync = False

        return self._matrix

    #--------------------------------------------------------------------------
    @property
    def _assembly_template_head(self):
        """A template for the 'head' of the assembly function. Only used with the sum factorization algorithm."""
        code = '''def assemble_matrix_{FILE_ID}({MAPPING_PART_1}
{SPAN}                    {MAPPING_PART_2}
                    global_x1 : "float64[:,:]", global_x2 : "float64[:,:]", global_x3 : "float64[:,:]", 
                    {MAPPING_PART_3}
                    n_element_1 : "int64", n_element_2 : "int64", n_element_3 : "int64", 
                    nq1 : "int64", nq2 : "int64", nq3 : "int64", 
                    pad1 : "int64", pad2 : "int64", pad3 : "int64", 
                    {MAPPING_PART_4}
{G_MAT}{NEW_ARGS}{FIELD_ARGS}):

    from numpy import abs as Abs
    {imports}
'''
        return code

    #--------------------------------------------------------------------------
    @property
    def _assembly_template_body_bspline(self):
        """A template for the 'body' of the assembly function (when using a spline mapping). Only used with the sum factorization algorithm."""
        code = '''
    arr_coeffs_x = zeros((1 + test_mapping_p1, 1 + test_mapping_p2, 1 + test_mapping_p3), dtype='float64')
    arr_coeffs_y = zeros((1 + test_mapping_p1, 1 + test_mapping_p2, 1 + test_mapping_p3), dtype='float64')
    arr_coeffs_z = zeros((1 + test_mapping_p1, 1 + test_mapping_p2, 1 + test_mapping_p3), dtype='float64')

{F_COEFFS_ZEROS}
    
{KEYS}
    for k_1 in range(n_element_1):
        span_mapping_1 = global_span_mapping_1[k_1]
{LOCAL_SPAN}{F_SPAN_1}{A1}
        for q_1 in range(nq1):
            for k_2 in range(n_element_2):
                span_mapping_2 = global_span_mapping_2[k_2]
{F_SPAN_2}
                for q_2 in range(nq2):
                    for k_3 in range(n_element_3):
                        span_mapping_3 = global_span_mapping_3[k_3]
{F_SPAN_3}{F_COEFFS}
                        arr_coeffs_x[:,:,:] = global_arr_coeffs_x[test_mapping_p1 + span_mapping_1 - test_mapping_p1:test_mapping_p1 + 1 + span_mapping_1,test_mapping_p2 + span_mapping_2 - test_mapping_p2:test_mapping_p2 + 1 + span_mapping_2,test_mapping_p3 + span_mapping_3 - test_mapping_p3:test_mapping_p3 + 1 + span_mapping_3]
                        arr_coeffs_y[:,:,:] = global_arr_coeffs_y[test_mapping_p1 + span_mapping_1 - test_mapping_p1:test_mapping_p1 + 1 + span_mapping_1,test_mapping_p2 + span_mapping_2 - test_mapping_p2:test_mapping_p2 + 1 + span_mapping_2,test_mapping_p3 + span_mapping_3 - test_mapping_p3:test_mapping_p3 + 1 + span_mapping_3]
                        arr_coeffs_z[:,:,:] = global_arr_coeffs_z[test_mapping_p1 + span_mapping_1 - test_mapping_p1:test_mapping_p1 + 1 + span_mapping_1,test_mapping_p2 + span_mapping_2 - test_mapping_p2:test_mapping_p2 + 1 + span_mapping_2,test_mapping_p3 + span_mapping_3 - test_mapping_p3:test_mapping_p3 + 1 + span_mapping_3]
                        for q_3 in range(nq3):
                            x = 0.0
                            y = 0.0
                            z = 0.0

                            x_x1 = 0.0
                            x_x2 = 0.0
                            x_x3 = 0.0
                            y_x1 = 0.0
                            y_x2 = 0.0
                            y_x3 = 0.0
                            z_x1 = 0.0
                            z_x2 = 0.0
                            z_x3 = 0.0
{D2_1}

{F_INIT}

{F_ASSIGN_LOOP}

                            for i_1 in range(test_mapping_p1+1):
                                mapping_1       = global_basis_mapping_1[k_1, i_1, 0, q_1]
                                mapping_1_x1    = global_basis_mapping_1[k_1, i_1, 1, q_1]
                                {D2_2}
                                for i_2 in range(test_mapping_p2+1):
                                    mapping_2       = global_basis_mapping_2[k_2, i_2, 0, q_2]
                                    mapping_2_x2    = global_basis_mapping_2[k_2, i_2, 1, q_2]
                                    {D2_3}
                                    for i_3 in range(test_mapping_p3+1):
                                        mapping_3       = global_basis_mapping_3[k_3, i_3, 0, q_3]
                                        mapping_3_x3    = global_basis_mapping_3[k_3, i_3, 1, q_3]
                                        {D2_4}

                                        coeff_x = arr_coeffs_x[i_1,i_2,i_3]
                                        coeff_y = arr_coeffs_y[i_1,i_2,i_3]
                                        coeff_z = arr_coeffs_z[i_1,i_2,i_3]

                                        mapping = mapping_1*mapping_2*mapping_3
                                        mapping_x1 = mapping_1_x1*mapping_2*mapping_3
                                        mapping_x2 = mapping_1*mapping_2_x2*mapping_3
                                        mapping_x3 = mapping_1*mapping_2*mapping_3_x3

{D2_5}

                                        x += mapping*coeff_x
                                        y += mapping*coeff_y
                                        z += mapping*coeff_z

                                        x_x1 += mapping_x1*coeff_x
                                        x_x2 += mapping_x2*coeff_x
                                        x_x3 += mapping_x3*coeff_x
                                        y_x1 += mapping_x1*coeff_y
                                        y_x2 += mapping_x2*coeff_y
                                        y_x3 += mapping_x3*coeff_y
                                        z_x1 += mapping_x1*coeff_z
                                        z_x2 += mapping_x2*coeff_z
                                        z_x3 += mapping_x3*coeff_z

{D2_6}
                                        
{TEMPS}
{COUPLING_TERMS}
'''
        return code

    #--------------------------------------------------------------------------
    @property 
    def _assembly_template_body_analytic(self):
        """A template for the 'body' of the assembly function (when using an analytic or no mapping). Only used with the sum factorization algorithm."""
        code = '''
    local_x1 = zeros_like(global_x1[0,:])
    local_x2 = zeros_like(global_x2[0,:])
    local_x3 = zeros_like(global_x3[0,:])

{F_COEFFS_ZEROS}

{KEYS}
    for k_1 in range(n_element_1):
        local_x1[:] = global_x1[k_1,:]
{LOCAL_SPAN}{F_SPAN_1}{A1}
        for q_1 in range(nq1):
            x1 = local_x1[q_1]
            for k_2 in range(n_element_2):
                local_x2[:] = global_x2[k_2,:]
{F_SPAN_2}
                for q_2 in range(nq2):
                    x2 = local_x2[q_2]
                    for k_3 in range(n_element_3):
                        local_x3[:] = global_x3[k_3,:]
{F_SPAN_3}{F_COEFFS}
                        for q_3 in range(nq3):
                            x3 = local_x3[q_3]

{F_INIT}

{F_ASSIGN_LOOP}

{TEMPS}
{COUPLING_TERMS}
'''
        return code
    
    #--------------------------------------------------------------------------
    @property
    def _assembly_template_loop(self):
        """A template for the 'loop' of the assembly function. Only used with the sum factorization algorithm."""
        code = '''
            {A2}[:] = 0.0
            for k_2 in range(n_element_2):
                {SPAN_2} = {GLOBAL_SPAN_2}[k_2]
                for q_2 in range(nq2):
                    {A3}[:] = 0.0
                    for k_3 in range(n_element_3):
                        {SPAN_3} = {GLOBAL_SPAN_3}[k_3]
                        for q_3 in range(nq3):
                            a4 = {COUPLING_TERMS}[k_2, q_2, k_3, q_3, :]
                            for i_3 in range({TEST_V_P3} + 1):
                                for j_3 in range({TRIAL_U_P3} + 1):
                                    for e in range({NEXPR}):
                                        {A3}[e, {SPAN_3} - {TEST_V_P3} + i_3, {MAX_P3} - {I_3} + j_3] += {TEST_TRIAL_3}[k_3, q_3, i_3, j_3, {KEYS_3}[2*e], {KEYS_3}[2*e+1]] * a4[e]
                    for i_2 in range({TEST_V_P2} + 1):
                        for j_2 in range({TRIAL_U_P2} + 1):
                            for e in range({NEXPR}):
                                {A2}[e, {SPAN_2} - {TEST_V_P2} + i_2, :, {MAX_P2} - {I_2} + j_2, :] += {TEST_TRIAL_2}[k_2, q_2, i_2, j_2, {KEYS_2}[2*e], {KEYS_2}[2*e+1]] * {A3}[e,:,:]
            for i_1 in range({TEST_V_P1} + 1):
                for j_1 in range({TRIAL_U_P1} + 1):
                    {A1}[i_1, :, :, {MAX_P1} - {I_1} + j_1, :, :] += {A2_TEMP}
'''
        return code

    #--------------------------------------------------------------------------
    def make_file(self, temps, ordered_stmts, field_derivatives, max_logical_derivative, test_mult, trial_mult, test_v_p, trial_u_p, keys_1, keys_2, keys_3, mapping_option):
        """
        Part of the sum factorization algorithm implementation.
        Generates the correct assembly file.
        Used at the end of construct_arguments_generate_assembly_file, before eventually pyccelizing that file.

        Parameters
        ----------
        temps : tuple
            Tuple of Assign statements defining temporary values. 
            Arithmetic combinations of these make up the coupling terms.

        ordered_stmts : dict
            Dictionary defining the coupling terms. Keys are combinations of
            test and trial function components, values are Assign statements
            in terms of temporaries appearing in temps.

        field_derivatives : dict
            Dictionary containing information on the derivatives of free FemFields.
            Keys are components of free FemFields. Values are dictionaries again.
            Their keys are names, as appearing in the assembly file, of partial derivatives of the 
            corresponding FemField component, and their values are dictionaries again.
            Example: {F1_0_x3 : {'x1': 0, 'x2': 0, 'x3': 1}, F1_0_x2 : ...}
            Meaning: There exists a free FemField named F1. Among other, the partial derivative w.r.t. x3
            of its first component F1_0 appears.

        max_logical_derivative : int
            The largest appearing derivative order.

        test_mult : list
            List of length 3(scalar test function) or 9(vector test function) including multiplicity information.

        trial_mult : list
            List of length 3(scalar trial function) or 9(vector trial function) including multiplicity information.

        test_v_p : dict
            Dictionary of length 1(scalar test function) or length 3(vector test function).
            Each key corresponds to a component of the funciton (space), and each corresponding value
            is a list of Bspline degrees of length 3. Example: Discretizing a de de Rham sequence using 
            a degree vector [2, 3, 4] means that test_v_p for a test function belonging to H(curl) will be
            {0: [1, 3, 4], 1: [2, 2, 4], 2: [2, 3, 3]}

        trial_u_p : dict
            Dictionary of length 1(scalar trial function) or length 3(vector trial function).
            Each key corresponds to a component of the funciton (space), and each corresponding value
            is a list of Bspline degrees of length 3. Example: Discretizing a de de Rham sequence using 
            a degree vector [2, 3, 4] means that trial_u_p for a trial function belonging to H^1 will be
            {0: [2, 3, 4]}

        keys_1 : dict
            Dictionary relating subexpressions to x1-derivative combinations.
            Keys are combinations of test and trial function components.
            Values are lists, each entry corresponding to one appearing partial derivative
            combination of these components. 
            Example: keys_1[(u[0], v[1])][3] = [1,0] means that the fourth ([3]) 
            sub-expression (partial derivative combination) corresponding to the trial-test-function-component-product
            u[0] * v[1] involves a first derivative in x1 direction of the trial function 
            and no derivative in x1 direction of the test function.
            Information on appearing partial derivatives in x2 and x3 direction is stored in keys_2 and keys_3.

        keys_2 : dict
            See keys_1.

        keys_3 : dict
            See keys_1.

        mapping_option : None | 'Bspline'
            None in case of no mapping or an analytical mapping, 'Bspline' in case of a Bspline mapping.

        Returns
        -------

        file_id : str
            random string of length 8, corresponding to the assembly file name located in __psydac__/
        
        """

        #------------------------- FILE_ID -------------------------
        comm = self.comm

        # Root process generates a random string to be used as file_id
        if comm is None or comm.rank == 0:
            file_id = random_string(size=8)
        else:
            file_id = None

        # Parallel case: root process broadcasts file_id to all processes
        if comm is not None and comm.size > 1:
            file_id = comm.bcast(file_id, root=0)

        # ----- free FemField related strings -----

        # used as {FIELD_ARGS} in _assembly_template_head
        # adding the right arguments for free FemFields to the assembly function header
        basis_args_block = [f'global_test_basis_'+'{field}'+f'_{i+1} : "float64[:,:,:,:]"' for i in range(3)]
        basis_args_block = ", ".join(basis_args_block) + ","
        basis_args_block = [basis_args_block.format(field=field) for field in field_derivatives]
        basis_args = "                    " + "\n                    ".join(basis_args_block) + "\n"
        span_args_block = [f'global_span_'+'{field}'+f'_{i+1} : "int64[:]"' for i in range(3)]
        span_args_block = ", ".join(span_args_block) + ","
        span_args_block = [span_args_block.format(field=field) for field in field_derivatives]
        span_args = "                    " + "\n                    ".join(span_args_block) + "\n"
        degree_args_block = [f'test_'+'{field}'+f'_p{i+1} : "int64"' for i in range(3)]
        degree_args_block = ", ".join(degree_args_block) + ","
        degree_args_block = [degree_args_block.format(field=field) for field in field_derivatives]
        degree_args = "                    " + "\n                    ".join(degree_args_block) + "\n"
        pad_args_block = [f'pad_'+'{field}'+f'_{i+1} : "int64"' for i in range(3)]
        pad_args_block = ", ".join(pad_args_block) + ","
        pad_args_block = [pad_args_block.format(field=field) for field in field_derivatives]
        pad_args = "                    " + "\n                    ".join(pad_args_block) + "\n"
        coeff_args_block = [f'global_arr_coeffs_{field} : "float64[:,:,:]"' for field in field_derivatives]
        coeff_args = "                    " + ", ".join(coeff_args_block)
        FIELD_ARGS = basis_args+span_args+degree_args+pad_args+coeff_args

        # {F_COEFFS_ZEROS} in both _assembly_template_body_bspline & _analytic
        F_COEFFS_ZEROS = "\n".join([f"    arr_coeffs_{field} = zeros((1 + test_{field}_p1, 1 + test_{field}_p2, 1 + test_{field}_p3), dtype='float64')" for field in field_derivatives])

        # {F_SPAN_1}, {F_SPAN_2}, {F_SPAN_3} in both _assembly_template_body_bspline & _analytic
        F_SPAN_1 = "\n".join([f"        span_{field}_1 = global_span_{field}_1[k_1]" for field in field_derivatives]) + "\n"
        F_SPAN_2 = "\n".join([f"                span_{field}_2 = global_span_{field}_2[k_2]" for field in field_derivatives]) + "\n"
        F_SPAN_3 = "\n".join([f"                        span_{field}_3 = global_span_{field}_3[k_3]" for field in field_derivatives]) + "\n"

        # {F_COEFFS} in both _assembly_template_body_bspline & _analytic
        coeff_ranges = ", ".join([f"pad_"+"{field}"+f"_{i+1} + span_"+"{field}"+f"_{i+1} - test_"+"{field}"+f"_p{i+1}:1 + pad_"+"{field}"+f"_{i+1} + span_"+"{field}"+f"_{i+1}" for i in range(3)])
        F_COEFFS = "\n".join([f"                        arr_coeffs_{field}[:,:,:] = global_arr_coeffs_{field}[{coeff_ranges.format(field=field)}]" for i, field in enumerate(field_derivatives)])

        # {F_INIT}
        F_INIT = "\n".join([f"                            {derivative}     = 0.0" for field in field_derivatives for derivative in field_derivatives[field]])

        #
        # field_init assigns 0 to appearing free FemField derivatives (F_x1 = 0.0 \n F_x2 = 0.0 \n ...)
        # In the following, we assemble loops that correctly compute those free FemField derivatives at 
        # a specific quadrature point (q_1, q_2, q_3). Those values will then be used in the computation 
        # of the temps or directly in the computation of the coupling terms
        #
        assign_loop_contents = {'1':{}, '2':{}, '3':{}}
        multiplication_info = {}

        for field, derivatives in field_derivatives.items():
            multiplication_info[field] = {}
            assign_statements = {'1':[], '2':[], '3':[]}
            for derivative, dxs in derivatives.items():
                multiplication_info[field][derivative] = []
                dx1 = dxs['x1']
                dx2 = dxs['x2']
                dx3 = dxs['x3']
                for i, dx in enumerate([dx1, dx2, dx3]):
                    name = f"{field}_{i+1}" if dx == 0 else f"{field}_{i+1}_{dx*f'x{i+1}'}"
                    multiplication_info[field][derivative].append(name)
                    if dx == 0:
                        assign_statement = f"{name}     = global_test_basis_{field}_{i+1}[k_{i+1}, i_{i+1}, 0, q_{i+1}]"
                    else:
                        assign_statement = f"{name} = global_test_basis_{field}_{i+1}[k_{i+1}, i_{i+1}, {dx}, q_{i+1}]"
                    if assign_statement not in assign_statements[f"{i+1}"]:
                        assign_statements[f"{i+1}"].append(assign_statement)
            for i in range(3):
                content = ("\n"+(8+i)*"    ").join(assign_statements[f"{i+1}"])
                assign_loop_contents[f"{i+1}"][field] = content
        tab = 7*"    "
        assign = []
        for field in field_derivatives:
            txt =   f"{tab}for i_1 in range(1 + test_{field}_p1):\n" + \
                    f"{tab}    {assign_loop_contents['1'][field]}\n" + \
                    f"{tab}    for i_2 in range(1 + test_{field}_p2):\n" + \
                    f"{tab}        {assign_loop_contents['2'][field]}\n" + \
                    f"{tab}        for i_3 in range(1 + test_{field}_p3):\n" + \
                    f"{tab}            {assign_loop_contents['3'][field]}\n" + \
                    f"{tab}            coeff_{field} = arr_coeffs_{field}[i_1, i_2, i_3]\n"
            for derivative in multiplication_info[field]:
                factors = " * ".join(multiplication_info[field][derivative])
                txt += f"{tab}            {derivative} += {factors} * coeff_{field}\n"
            txt += "\n"
            assign.append(txt)

        # {F_ASSIGN_LOOP} in both _assembly_template_body_bspline & _analytic
        F_ASSIGN_LOOP = "\n".join(assign)

        # -----------------------------------------

        # ----- load the templates -----
        #
        # head for the function header and imports
        # body for the computation of coupling terms
        # loop (part of function body): one loop per block ( e.g. (u[0], v[1]) ), each loop effectively 
        # assembles one StencilMatrix per sub expression ( e.g. (dx1(u[0]), dx3(v[1])) )
        code_head = self._assembly_template_head
        code_loop = self._assembly_template_loop
        if mapping_option == 'Bspline':
            code_body = self._assembly_template_body_bspline
        else:
            code_body = self._assembly_template_body_analytic
        # ------------------------------
        
        # ---- obtain basic information not explicitely passed in the args -----
        blocks              = ordered_stmts.keys()
        block_list          = list(blocks)
        trial_components    = [block[0] for block in block_list]
        test_components     = [block[1] for block in block_list]
        nu                  = len(set(trial_components))
        nv                  = len(set(test_components))
        d = 3
        assert d == 3
        # ----------------------------------------------------------------------

        # Prepare strings and string templates depending on whether the trial and test function are vector-valued or not (nu, nv > 1 or == 1)

        # ------------------------- STRINGS HEAD -------------------------
        
        global_span_v_str   = 'global_span_v_{v_j}_'  if nv > 1 else 'global_span_v_'

        if mapping_option == 'Bspline':
            MAPPING_PART_1 = 'global_basis_mapping_1 : "float64[:,:,:,:]", global_basis_mapping_2 : "float64[:,:,:,:]", global_basis_mapping_3 : "float64[:,:,:,:]", '
            MAPPING_PART_2 = 'global_span_mapping_1 : "int64[:]", global_span_mapping_2 : "int64[:]", global_span_mapping_3 : "int64[:]", '
            MAPPING_PART_3 = 'test_mapping_p1 : "int64", test_mapping_p2 : "int64", test_mapping_p3 : "int64", '
            MAPPING_PART_4 = 'global_arr_coeffs_x : "float64[:,:,:]", global_arr_coeffs_y : "float64[:,:,:]", global_arr_coeffs_z : "float64[:,:,:]", '
        else:
            MAPPING_PART_1 = ''
            MAPPING_PART_2 = ''
            MAPPING_PART_3 = ''
            MAPPING_PART_4 = ''

        if nv > 1:
            tt1_str     = 'test_trial_1_u_{u_i}_v_{v_j}'    if nu > 1 else 'test_trial_1_u_v_{v_j}'
            tt2_str     = 'test_trial_2_u_{u_i}_v_{v_j}'    if nu > 1 else 'test_trial_2_u_v_{v_j}'
            tt3_str     = 'test_trial_3_u_{u_i}_v_{v_j}'    if nu > 1 else 'test_trial_3_u_v_{v_j}'
            a3_str      = 'a3_u_{u_i}_v_{v_j}'              if nu > 1 else 'a3_u_v_{v_j}'
            a2_str      = 'a2_u_{u_i}_v_{v_j}'              if nu > 1 else 'a2_u_v_{v_j}'
            ct_str      = 'coupling_terms_u_{u_i}_v_{v_j}'  if nu > 1 else 'coupling_terms_u_v_{v_j}'
            g_mat_str   = 'g_mat_u_{u_i}_v_{v_j}'           if nu > 1 else 'g_mat_u_v_{v_j}'
        else:
            tt1_str     = 'test_trial_1_u_{u_i}_v'      if nu > 1 else 'test_trial_1_u_v'
            tt2_str     = 'test_trial_2_u_{u_i}_v'      if nu > 1 else 'test_trial_2_u_v'
            tt3_str     = 'test_trial_3_u_{u_i}_v'      if nu > 1 else 'test_trial_3_u_v'
            a3_str      = 'a3_u_{u_i}_v'                if nu > 1 else 'a3_u_v'
            a2_str      = 'a2_u_{u_i}_v'                if nu > 1 else 'a2_u_v'
            ct_str      = 'coupling_terms_u_{u_i}_v'    if nu > 1 else 'coupling_terms_u_v'
            g_mat_str   = 'g_mat_u_{u_i}_v'             if nu > 1 else 'g_mat_u_v'

        # ----------------------------------------------------------------

        # ------------------------- STRINGS BODY -------------------------
        
        span_v_1_str    = 'span_v_{v_j}_1'  if nv > 1 else 'span_v_1'
        test_v_p1_str   = 'test_v_{v_j}_p1' if nv > 1 else 'test_v_p1'

        if nv > 1:
            keys_2_str  = 'keys_2_u_{u_i}_v_{v_j}'  if nu > 1 else 'keys_2_u_v_{v_j}'
            keys_3_str  = 'keys_3_u_{u_i}_v_{v_j}'  if nu > 1 else 'keys_3_u_v_{v_j}'
            a1_str      = 'a1_u_{u_i}_v_{v_j}'      if nu > 1 else 'a1_u_v_{v_j}'
        else:
            keys_2_str  = 'keys_2_u_{u_i}_v'    if nu > 1 else 'keys_2_u_v'
            keys_3_str  = 'keys_3_u_{u_i}_v'    if nu > 1 else 'keys_3_u_v'
            a1_str      = 'a1_u_{u_i}_v'        if nu > 1 else 'a1_u_v'

        # ----------------------------------------------------------------

        # ------------------------- STRINGS LOOP -------------------------

        span_2_str          = 'span_v_{v_j}_2'          if nv > 1 else 'span_v_2'
        span_3_str          = 'span_v_{v_j}_3'          if nv > 1 else 'span_v_3'
        global_span_2_str   = 'global_span_v_{v_j}_2'   if nv > 1 else 'global_span_v_2'
        global_span_3_str   = 'global_span_v_{v_j}_3'   if nv > 1 else 'global_span_v_3'

        # ----------------------------------------------------------------

        #------------------------- MAKE HEAD -------------------------
        SPAN            = ''
        G_MAT           = ''

        TT1             = '                    '
        TT2             = '                    '
        TT3             = '                    '
        A3              = '                    '
        A2              = '                    '
        CT              = '                    '

        for v_j in range(nv):
            global_span_v = global_span_v_str.format(v_j=v_j)
            SPAN += '                    '
            for di in range(d):
                SPAN += f'{global_span_v}{di+1} : "int64[:]", '
            SPAN = SPAN[:-1] + '\n'

        for block in blocks:
            u_i = block[0].indices[0] if nu > 1 else 0
            v_j = block[1].indices[0] if nv > 1 else 0

            # reverse order intended
            if ((nu > 1) and (nv > 1)):
                g_mat = g_mat_str.format(u_i=v_j, v_j=u_i)
            else:
                g_mat = g_mat_str.format(u_i=u_i, v_j=v_j)
            G_MAT += f'                    {g_mat} : f"{self._dtype}[:,:,:,:,:,:]",\n'

            TT1 += tt1_str.format(u_i=u_i, v_j=v_j) + ' : "float64[:,:,:,:,:,:]", '
            TT2 += tt2_str.format(u_i=u_i, v_j=v_j) + ' : "float64[:,:,:,:,:,:]", '
            TT3 += tt3_str.format(u_i=u_i, v_j=v_j) + ' : "float64[:,:,:,:,:,:]", '
            A3  += a3_str.format(u_i=u_i, v_j=v_j)  + ' : "float64[:,:,:]", '
            A2  += a2_str.format(u_i=u_i, v_j=v_j)  + ' : "float64[:,:,:,:,:]", '
            CT  += ct_str.format(u_i=u_i, v_j=v_j)  + ' : "float64[:,:,:,:,:]", '

        TT1 += '\n'
        TT2 += '\n'
        TT3 += '\n'
        A3  += '\n'
        A2  += '\n'
        CT  += '\n'
        NEW_ARGS = TT1 + TT2 + TT3 + A3 + A2 + CT
        IMPORTS = self._imports_string

        head = code_head.format(FILE_ID         = file_id,
                                SPAN            = SPAN,
                                G_MAT           = G_MAT, 
                                NEW_ARGS        = NEW_ARGS,
                                MAPPING_PART_1  = MAPPING_PART_1,
                                MAPPING_PART_2  = MAPPING_PART_2,
                                MAPPING_PART_3  = MAPPING_PART_3,
                                MAPPING_PART_4  = MAPPING_PART_4,
                                FIELD_ARGS      = FIELD_ARGS,
                                imports         = IMPORTS)
        
        #------------------------- MAKE BODY -------------------------
        A1              = ''
        KEYS_2          = ''
        KEYS_3          = ''
        LOCAL_SPAN      = ''
        TEMPS           = ''
        COUPLING_TERMS  = ''

        for block in blocks:
            u_i = block[0].indices[0] if nu > 1 else 0
            v_j = block[1].indices[0] if nv > 1 else 0
            
            keys2   = keys_2[block].copy()
            keys3   = keys_3[block].copy()
            keys2   = ','.join(str(i) for i in keys2.flatten())
            keys3   = ','.join(str(i) for i in keys3.flatten())
            KEYS2   = keys_2_str.format(u_i=u_i, v_j=v_j)
            KEYS3   = keys_3_str.format(u_i=u_i, v_j=v_j)
            KEYS_2  += f'    {KEYS2} = array([{keys2}])\n'
            KEYS_3  += f'    {KEYS3} = array([{keys3}])\n'

            test_v_p1, test_v_p2, test_v_p3 = test_v_p[v_j]
            a1          = a1_str.format(u_i=u_i, v_j=v_j)
            g_mat       = g_mat_str.format(u_i=u_i, v_j=v_j)
            TEST_V_P1   = test_v_p1_str.format(v_j=v_j)
            SPAN_V_1    = span_v_1_str.format(v_j=v_j)

            A1_1 = f'{test_mult[0]}*pad1 + {SPAN_V_1} - {test_v_p1} : {test_mult[0]}*pad1 + {SPAN_V_1} + 1'                   if test_mult[0] > 1 else f'pad1 + {SPAN_V_1} - {test_v_p1} : pad1 + {SPAN_V_1} + 1'
            A1_2 = f'{test_mult[1]}*pad2 : {test_mult[1]}*pad2 + n_element_2 + {test_v_p2} + ({test_mult[1]}-1)*(n_element_2-1)'   if test_mult[1] > 1 else f'pad2 : pad2 + n_element_2 + {test_v_p2}'
            A1_3 = f'{test_mult[2]}*pad3 : {test_mult[2]}*pad3 + n_element_3 + {test_v_p3} + ({test_mult[2]}-1)*(n_element_3-1)'   if test_mult[2] > 1 else f'pad3 : pad3 + n_element_3 + {test_v_p3}'
            A1          += f'        {a1} = {g_mat}[{A1_1}, {A1_2}, {A1_3}, :, :, :]\n'

        for v_j in range(nv):
            local_span_v_1 = span_v_1_str.format(v_j=v_j)
            global_span_v = global_span_v_str.format(v_j=v_j)
            LOCAL_SPAN += f'        {local_span_v_1} = {global_span_v}1[k_1]\n'

        # Print expressions using SymPy's Python code printer
        pyc = lambda expr: pycode(expr, fully_qualified_modules=False)

        for temp in temps:
            TEMPS += f'                            {temp.lhs} = {pyc(temp.rhs)}\n'
        for block in blocks:
            for stmt in ordered_stmts[block]:
                COUPLING_TERMS += f'                            {stmt.lhs} = {pyc(stmt.rhs)}\n'
        
        KEYS = KEYS_2 + KEYS_3

        # This part is interesting. Right now, below you find hardcoded rules regarding lines of code
        # that need to be included when max_logical_derivative == 2 ( and mapping_option == 'Bspline').
        # E.g., the bilinear form corresponding to a bilaplacian problem ( laplace(laplace(u)) = f ) satisfies this assumption.
        # This hardcoded set of rules could be generalized to n-th max derivatives - if needed!
        # But for now, higher than second order derivatives on either trial or test function are not supported.
        #
        # Additional note: Given a Bspline mapping, the code computing the first order derivatives of mapping related terms is always required!
        # Even in the case of a trivial bilinear form without derivatives. But only when there are second order partial derivatives involved
        # do we need to compute second derivatives of the mapping (chain rule).
        if (mapping_option == 'Bspline') and (max_logical_derivative == 2):
            D2_1 = '\n'
            spaces1 = '                            '
            spaces2 = spaces1 + '            '
            for symbol in ('x', 'y', 'z'):
                for d1 in range(1, 4):
                    for d2 in range(1, 4):
                        if d2 >= d1:
                            D2_1 += spaces1 + f'{symbol}_x{d1}x{d2} = 0.0\n'
                D2_1 += '\n'
            D2_2 = 'mapping_1_x1x1  = global_basis_mapping_1[k_1, i_1, 2, q_1]'
            D2_3 = 'mapping_2_x2x2  = global_basis_mapping_2[k_2, i_2, 2, q_2]'
            D2_4 = 'mapping_3_x3x3  = global_basis_mapping_3[k_3, i_3, 2, q_3]'
            D2_5 = spaces2+'mapping_x1x1 = mapping_1_x1x1 * mapping_2      * mapping_3\n'+spaces2
            D2_5 += 'mapping_x1x2 = mapping_1_x1   * mapping_2_x2   * mapping_3\n'+spaces2
            D2_5 += 'mapping_x1x3 = mapping_1_x1   * mapping_2      * mapping_3_x3\n'+spaces2
            D2_5 += 'mapping_x2x2 = mapping_1      * mapping_2_x2x2 * mapping_3\n'+spaces2
            D2_5 += 'mapping_x2x3 = mapping_1      * mapping_2_x2   * mapping_3_x3\n'+spaces2
            D2_5 += 'mapping_x3x3 = mapping_1      * mapping_2      * mapping_3_x3x3\n'
            D2_6 = ''
            for symbol in ('x', 'y', 'z'):
                for d1 in range(1, 4):
                    for d2 in range(1, 4):
                        if d2 >= d1:
                            D2_6 += f'{spaces2}{symbol}_x{d1}x{d2} += mapping_x{d1}x{d2} * coeff_{symbol}\n'
                D2_6 += '\n'
        else:
            D2_1 = ''
            D2_2 = ''
            D2_3 = ''
            D2_4 = ''
            D2_5 = ''
            D2_6 = ''

        body = code_body.format(LOCAL_SPAN      = LOCAL_SPAN, 
                                KEYS            = KEYS, 
                                A1              = A1, 
                                TEMPS           = TEMPS, 
                                COUPLING_TERMS  = COUPLING_TERMS,
                                F_COEFFS_ZEROS  = F_COEFFS_ZEROS,
                                F_SPAN_1        = F_SPAN_1,
                                F_SPAN_2        = F_SPAN_2,
                                F_SPAN_3        = F_SPAN_3,
                                F_COEFFS        = F_COEFFS,
                                F_INIT          = F_INIT,
                                F_ASSIGN_LOOP   = F_ASSIGN_LOOP,
                                D2_1            = D2_1,
                                D2_2            = D2_2,
                                D2_3            = D2_3,
                                D2_4            = D2_4,
                                D2_5            = D2_5,
                                D2_6            = D2_6)
        
        #------------------------- MAKE LOOP -------------------------
        assembly_code = head + body
        loop_str = ''

        for block in blocks:
            u_i = block[0].indices[0] if nu > 1 else 0
            v_j = block[1].indices[0] if nv > 1 else 0

            A1              = a1_str.format(u_i=u_i, v_j=v_j)
            A2              = a2_str.format(u_i=u_i, v_j=v_j)
            A3              = a3_str.format(u_i=u_i, v_j=v_j)
            TEST_TRIAL_2    = tt2_str.format(u_i=u_i, v_j=v_j)
            TEST_TRIAL_3    = tt3_str.format(u_i=u_i, v_j=v_j)
            SPAN_2          = span_2_str.format(u_i=u_i, v_j=v_j)
            SPAN_3          = span_3_str.format(u_i=u_i, v_j=v_j)
            GLOBAL_SPAN_2   = global_span_2_str.format(u_i=u_i, v_j=v_j)
            GLOBAL_SPAN_3   = global_span_3_str.format(u_i=u_i, v_j=v_j)  
            KEYS_3          = keys_3_str.format(u_i=u_i, v_j=v_j)
            KEYS_2          = keys_2_str.format(u_i=u_i, v_j=v_j)
            COUPLING_TERMS  = ct_str.format(u_i=u_i, v_j=v_j)
            
            TEST_V_P1, TEST_V_P2, TEST_V_P3     = test_v_p[v_j]
            TRIAL_U_P1, TRIAL_U_P2, TRIAL_U_P3  = trial_u_p[u_i]
            MAX_P1 = max(TEST_V_P1, TRIAL_U_P1)
            MAX_P2 = max(TEST_V_P2, TRIAL_U_P2)
            MAX_P3 = max(TEST_V_P3, TRIAL_U_P3)
            NEXPR  = len(ordered_stmts[block])

            keys1 = keys_1[block]
            TEST_TRIAL_1    = tt1_str.format(u_i=u_i, v_j=v_j)
            A2_TEMP = " + ".join([f"{TEST_TRIAL_1}[k_1, q_1, i_1, j_1, {keys1[e][0]}, {keys1[e][1]}] * {A2}[{e},:,:,:,:]" for e in range(NEXPR)])

            I_1 = f'int(floor(i_1/{test_mult[0]})*{trial_mult[0]})' if max(test_mult[0], trial_mult[0]) > 1 else 'i_1'
            I_2 = f'int(floor(i_2/{test_mult[1]})*{trial_mult[1]})' if max(test_mult[1], trial_mult[1]) > 1 else 'i_2'
            I_3 = f'int(floor(i_3/{test_mult[2]})*{trial_mult[2]})' if max(test_mult[2], trial_mult[2]) > 1 else 'i_3'
            #MAX_P1 = max(int( ( MAX_P1 + np.floor(MAX_P1 / test_mult[0]) * trial_mult[0] ) / 2 ), MAX_P1) if max(test_mult[0], trial_mult[0]) > 1 else MAX_P1
            #MAX_P2 = max(int( ( MAX_P2 + np.floor(MAX_P2 / test_mult[1]) * trial_mult[1] ) / 2 ), MAX_P2) if max(test_mult[1], trial_mult[1]) > 1 else MAX_P2
            #MAX_P3 = max(int( ( MAX_P3 + np.floor(MAX_P3 / test_mult[2]) * trial_mult[2] ) / 2 ), MAX_P3) if max(test_mult[2], trial_mult[2]) > 1 else MAX_P3
            n_cols_x1 = max( int(MAX_P1 + 1 + np.floor(MAX_P1 / test_mult[0]) * trial_mult[0]), 2*MAX_P1+1 )
            n_cols_x2 = max( int(MAX_P2 + 1 + np.floor(MAX_P2 / test_mult[1]) * trial_mult[1]), 2*MAX_P2+1 )
            n_cols_x3 = max( int(MAX_P3 + 1 + np.floor(MAX_P3 / test_mult[2]) * trial_mult[2]), 2*MAX_P3+1 )
            MAX_P1 = n_cols_x1 - MAX_P1 - 1
            MAX_P2 = n_cols_x2 - MAX_P2 - 1
            MAX_P3 = n_cols_x3 - MAX_P3 - 1

            loop = code_loop.format(A1              = A1,
                                    A2              = A2,
                                    A3              = A3,
                                    TEST_TRIAL_2    = TEST_TRIAL_2,
                                    TEST_TRIAL_3    = TEST_TRIAL_3,
                                    SPAN_2          = SPAN_2,
                                    SPAN_3          = SPAN_3,
                                    GLOBAL_SPAN_2   = GLOBAL_SPAN_2,
                                    GLOBAL_SPAN_3   = GLOBAL_SPAN_3,
                                    KEYS_2          = KEYS_2,
                                    KEYS_3          = KEYS_3,
                                    COUPLING_TERMS  = COUPLING_TERMS,
                                    TEST_V_P1       = TEST_V_P1,
                                    TEST_V_P2       = TEST_V_P2,
                                    TEST_V_P3       = TEST_V_P3,
                                    TRIAL_U_P1      = TRIAL_U_P1,
                                    TRIAL_U_P2      = TRIAL_U_P2,
                                    TRIAL_U_P3      = TRIAL_U_P3,
                                    MAX_P1          = MAX_P1,
                                    MAX_P2          = MAX_P2,
                                    MAX_P3          = MAX_P3,
                                    NEXPR           = NEXPR,
                                    A2_TEMP         = A2_TEMP,
                                    I_1             = I_1,
                                    I_2             = I_2,
                                    I_3             = I_3)
            
            loop_str += loop

        assembly_code += loop_str
        assembly_code += '\n    return\n'
        
        #------------------------- MAKE FILE -------------------------
        import os
        if not os.path.isdir('__psydac__'):
            os.makedirs('__psydac__')

        # Root process writes the assembly code to a file
        if comm is None or comm.rank == 0:
            filename = f'__psydac__/assemble_{file_id}.py'
            f = open(filename, 'w')
            f.writelines(assembly_code)
            f.close()

        # Parallel case: wait for the file to be closed before proceeding
        if comm is not None and comm.size > 1:
            _ = comm.bcast(None, root=0)

        return file_id

    #--------------------------------------------------------------------------
    def read_BilinearForm(self):
        """
        Part of the sum factorization algorithm implementation.
        Used at the beginning of construct_arguments_generate_assembly_file().
        It's output determines both the design of the assembly function, and the arguments passed to it.

        Returns
        -------

        temps : tuple
            tuple of Assign objects. Often times usable building blocks of complicated coupling terms.
        
        ordered_stmts : dict
            assigns each block (trial&test component combination) a list of coupling term assignment
        
        ordered_sub_exprs_keys : dict
            relates each coupling term assignment of ordered_stmts a partial derivative combination
        
        mapping_option : str | None
            'Bspline' if a spline mapping is involved, None if an analytical or no mapping is involved
        
        field_derivatives : dict
            contains information regarding appearing free FemFields and appearing partial derivatives of those
        
        g_mat_information_false : list
            possibly wrong list of non-zero blocks

        g_mat_information_true : list
            correct list of non-zero blocks
        
        max_logical_derivative : int
            maximum appearing partial derivative (in any fixed direction)
        
        """

        a       = self.expr
        domain  = a.domain

        # Because an analytical mapping only changes the expression, only the case of a Bspline mapping has to be treated 
        # entirely different
        mapping_option = 'Bspline' if isinstance(self._mapping, SplineMapping) else None

        # The following are tuples consisting of test, trial and free FemField functions appearing, e.g.
        # u, v, F1, F2 = elements_of(V, names='u, v, F1, F2)
        # a = BilinearForm((u, v), integral(domain, dot(u, F1) * dot(v, F2)))
        # tests = (v, ), trials = (u, ) fields = (F1, F2) - Note: The order of F1 & F2 is apparently random and changes from time to time!
        # tuple entries are either sympde.topology.space.ScalarFunction or sympde.topology.space.VectorFunction objects
        tests  = a.test_functions
        trials = a.trial_functions
        fields = a.fields

        # A sympde.expr.evaluation.DomainExpression object
        # TODO [YG 31.07.2025]: Why not using self.terminal_expr[0] instead?
        texpr  = TerminalExpr(a, domain)[0]

        # We extract all appearing components of test, trial and free FemFields, as well as appearing partial derivatives of these.
        # e.g. atoms = [F1[1], F2[1], v[0], u[0], F1[2], F2[2], F1[0], v[1], v[2], F2[0], u[1], u[2]]
        # for a bilinear form, without derivatives, involving two vector valued Fem fields F1 & F2 and vector valued test & trial functions v and u 
        atoms_types = (ScalarFunction, VectorFunction, IndexedVectorFunction)
        atoms       = _atomic(texpr, cls=atoms_types+_logical_partial_derivatives)

        # Preparing to sort all atoms into test_, trial_ and field_atoms
        test_atoms  = {}
        for v in tests:
            if isinstance(v, VectorFunction):
                for i in range(domain.dim):
                    test_atoms[v[i]] = []
            else:
                test_atoms[v] = []

        trial_atoms = {}
        for u in trials:
            if isinstance(u, VectorFunction):
                for i in range(domain.dim):
                    trial_atoms[u[i]] = []
            else:
                trial_atoms[u] = []

        field_atoms = {}
        for f in fields:
            if isinstance(f, VectorFunction):
                for i in range(domain.dim):
                    field_atoms[f[i]] = []
            else:
                field_atoms[f] = []

        # atoms can consist of scalar functions (u, v), partial derivatives of scalar functions (dx1(u), dx3(v), ...),
        # components of vector valued functions (u[0], v[1], ...), partial derivatives of components of vector valued functions
        # (dx1(u[0]), dx3(v[1]), ...), and the same thing but for free FemFields.
        # With 
        # get_atom_logical_derivatives(atom)
        # we obtain the component without partial derivatives (u -> u ; dx1(u) -> u ; dx2(v[2]) -> v[2] ; ...)
        # This way we can gather subexpressions belonging to the same block
        for atom in atoms:
            a = get_atom_logical_derivatives(atom)
            # IF: NOT Indexed Mapping AND NOT VectorFunction
            # I guess: <=> IF ScalarFunction
            if not ((isinstance(a, Indexed) and isinstance(a.base, Mapping)) or (isinstance(a, IndexedVectorFunction))):
                if a in tests:
                    # tests is a tuple, e.g. (v, ), hence tests[0] = v
                    test_atoms[tests[0]].append(atom)
                elif a in trials:
                    trial_atoms[trials[0]].append(atom)
                elif a in fields:
                    # while there can only be one trial and one test function, there can be multiple free FemFields.
                    for f in field_atoms:
                        if f == a:
                            field_atoms[f].append(atom)
                else:
                    raise NotImplementedError(f"atoms of type {str(atom)} are not supported")
            # IF VectorFunction
            elif isinstance(a, IndexedVectorFunction):
                # .base returns ... the base of a VectorFunction! E.g., u[2] -> u, v[0] -> v
                if a.base in tests:
                    for vi in test_atoms:
                        if vi == a:
                            test_atoms[vi].append(atom)
                            break
                elif a.base in trials:
                    for ui in trial_atoms:
                        if ui == a:
                            trial_atoms[ui].append(atom)
                            break
                elif a.base in fields:
                    for fi in field_atoms:
                        if fi == a:
                            field_atoms[fi].append(atom)
                            break
                else:
                    raise NotImplementedError(f"atoms of type {str(atom)} are not supported")

        # ----- Julian O. 11.06.25 -----
        # Regarding the code that follows:
        # When dealing with a DiscreteBilinearForm depending on two or more free FemFields,
        # the order of the dictionary `field_derivatives` must be the same as the order
        # of the free FemFields in `self._free_args`.
        # For some reason, the order of all appearing "atoms" in a BilinearForm (trial function, test function, free fields, .?.)
        # as obtained in the __init__ of AST
        #       atoms               = terminal_expr.expr.atoms(ScalarFunction, VectorFunction)
        # is random and changes from code execution to code execution.
        # This order of atoms however determines the order of the free FemFields appearing in `self._free_args`.
        # In particular, this order only sometimes matches the order of `field_derivatives`, which results in wrong matrices.
        #
        # Below is the old version of the code that follows:
        #field_derivatives = {}
        #for key in field_atoms:
        #    sym_key = SymbolicExpr(key)
        #    field_derivatives[sym_key] = {}
        #    for f in field_atoms[key]:
        #        field_derivatives[sym_key][SymbolicExpr(f)] = get_index_logical_derivatives(f)
        # ------------------------------

        # For the computation of the coupling terms, among other we need to organize information 
        # related to free FemFields. For now, we have the dictionary field_atoms, whose keys are 
        # components of appearing fields, and whose values are appearing partial derivatives of these, e.g.,
        # field_atoms = {'F1[0]':[dx1(F1[0]), ], 'F1[1]':[dx2(F1[1]), ], 'F1[2]':[dx3(F1[2]), ], 'F2':[F2, ]}
        #
        # We now create the dictionary field_derivatives. 
        # It's keys are SymbolicExpr of the previous keys (F1[0] -> F1_0, F1[1] -> F1_1, F1[2] -> F1_2, F2 -> F2)
        # and its values are again dictionaries, whose keys are symbolic expressions of the appearing partial derivatives, e.g.
        # dx1(F1[0]) -> F1_0_x1, dx2(F1[1]) -> F1_1_x2, dx3(F1[2]) -> F1_2_x3, F2 -> F2,
        # and whose values are dictionaries that store the respective derivative information.
        # Consider for example the BilinearForm (u, v) \mapsto integral(domain, dot(u, grad(Fs)) * dot(v, grad(Fs2)):
        # The corresponding field_derivatives dict will be 
        # {Fs: {Fs_x3: {'x1': 0, 'x2': 0, 'x3': 1}, Fs_x2: {'x1': 0, 'x2': 1, 'x3': 0}, Fs_x1: {'x1': 1, 'x2': 0, 'x3': 0}}, Fs2: {Fs2_x3: {'x1': 0, 'x2': 0, 'x3': 1}, Fs2_x2: {'x1': 0, 'x2': 1, 'x3': 0}, Fs2_x1: {'x1': 1, 'x2': 0, 'x3': 0}}}

        # Amount of free FemFields (NOT counting each component individually)
        n_free_fields = len(self._free_args)
        field_derivatives = {}
        # The keys in field_derivatives will be in the same order as the fields appearing in self._free_args
        for n in range(n_free_fields):
            # The key might be F1[0], but we want to check whether F1 == self._free_args[0], and ...
            for key in field_atoms:
                # ... field_name does exactly that
                field_name = str(key.base) if hasattr(key, 'base') else str(key)
                if field_name == self._free_args[n]:
                    # SymbolicExpr transforms something like F1[0] into F1_0 (part of the name of a variable in the assembly code later)
                    sym_key = SymbolicExpr(key)
                    field_derivatives[sym_key] = {}
                    for f in field_atoms[key]:
                        # And similarly f, which might look like dx1(F1[0]), will be transformed to F1_0_x2
                        # while get_index_logical_derivatives(dx1(F1[0])) = {'x1': 1, 'x2': 0, 'x3': 0}
                        field_derivatives[sym_key][SymbolicExpr(f)] = get_index_logical_derivatives(f)

        # This part was proposed by Said at some point
        #syme = False
        #if syme:
        #    from symengine import sympify as syme_sympify
        #    sym_test_atoms  = {k:[syme_sympify(SymbolicExpr(ai)) for ai in a] for k,a in test_atoms.items()}
        #    sym_trial_atoms = {k:[syme_sympify(SymbolicExpr(ai)) for ai in a] for k,a in trial_atoms.items()}
        #    sym_expr        = syme_sympify(SymbolicExpr(texpr.expr))
        #else:
        #    sym_test_atoms  = {k:[SymbolicExpr(ai) for ai in a] for k,a in test_atoms.items()}
        #    sym_trial_atoms = {k:[SymbolicExpr(ai) for ai in a] for k,a in trial_atoms.items()}
        #    sym_expr        = SymbolicExpr(texpr.expr)

        # test_atoms is a dict whose values are components of the test function and whose values
        # are arrays with appearing partial derivatives of those components.
        # sym_test_atoms has the same structure, but replaces the appearing partial derivatives with
        # symbolic expressions of those partial derivatives. E.g., 
        # test_atoms:     {v2[0]: [dx3(v2[0]), dx2(v2[0])], v2[1]: [dx1(v2[1]), dx3(v2[1])], v2[2]: [dx2(v2[2]), dx1(v2[2])]}
        # sym_test_atoms: {v2[0]: [v2_0_x3, v2_0_x2], v2[1]: [v2_1_x1, v2_1_x3], v2[2]: [v2_2_x2, v2_2_x1]}
        # In the following, we will gather all (coupling) terms of a specific combination of a sym_test_atom with a sym_trial_atom in sym_expr
        sym_test_atoms  = {k:[SymbolicExpr(ai) for ai in a] for k,a in test_atoms.items()}
        sym_trial_atoms = {k:[SymbolicExpr(ai) for ai in a] for k,a in trial_atoms.items()}
        sym_expr        = SymbolicExpr(texpr.expr)

        # ----- temps, rhs -----

        trials_subs = {ui:0 for u in sym_trial_atoms for ui in sym_trial_atoms[u]}
        tests_subs  = {vi:0 for v in sym_test_atoms  for vi in sym_test_atoms[v]}
        sub_exprs   = {}
        
        # This is where the real magic happens: The at times extremely long and complicated SymbolicExpr sym_expr
        # 0. is brought into a more readable form (sub_exprs) &
        # 1. gets split into many small parts (temps), that often times appear in multiple sub_exprs,
        #    but now only have to be computed once, e.g. (temp_0, -F2_1*F1_1) &
        # 2. those temporaries get assigned to coupling terms (rhs), i.e.: 
        #    The coupling term corresponding to the sub-expr dx1(u[0])*dx3(v[1]) might be -temp_7*(temp_22*temp_27 + temp_33*temp_35 + temp_36*temp_37)
        for u in sym_trial_atoms:
            for v in sym_test_atoms:
                if isinstance(u, IndexedVectorFunction) and isinstance(v, IndexedVectorFunction):
                    sub_expr = sym_expr[v.indices[0], u.indices[0]]
                elif isinstance(u, ScalarFunction) and isinstance(v, ScalarFunction):
                    sub_expr = sym_expr
                elif isinstance(u, ScalarFunction) and isinstance(v, IndexedVectorFunction):
                    sub_expr = sym_expr[v.indices[0]]
                elif isinstance(u, IndexedVectorFunction) and isinstance(v, ScalarFunction):
                    sub_expr = sym_expr[u.indices[0]]
                for ui,sui in zip(trial_atoms[u], sym_trial_atoms[u]):
                    trcp = trials_subs.copy()
                    trcp[sui] = 1
                    newsub_expr = sub_expr.subs(trcp)
                    for vi,svi in zip(test_atoms[v],sym_test_atoms[v]):
                        tcp = tests_subs.copy()
                        tcp[svi] = 1
                        expr = newsub_expr.subs(tcp)
                        if not expr.is_zero:
                            sub_exprs[ui,vi] = sympify(expr)

        temps, rhs = cse_main.cse(sub_exprs.values(), symbols=cse_main.numbered_symbols(prefix=f'temp_'))

        # ----------------------

        # Finally, temps and rhs must be brought into a form that can be included in the assembly code, e.g.
        #                    temp_0 = x_x1*y_x2
        #                    temp_1 = x_x2*z_x1
        #                    temp_2 = y_x1*z_x2
        #                    ...
        #                    coupling_terms_u_v[k_2, q_2, k_3, q_3, 0] = temp_7*(temp_10**2*temp_9 + temp_11**2*temp_9 + temp_8**2*temp_9)
        #                    coupling_terms_u_v[k_2, q_2, k_3, q_3, 1] = temp_18
        #                    coupling_terms_u_v[k_2, q_2, k_3, q_3, 2] = temp_22
        #                    ...

        # See above example: In our implementation of the sum factorization algorithm, we precompute arrays
        # for each quadrature point in x1 direction, meaning that those arrays contain values depending on 
        # elements and quadrature points in x2 and x3 direction (k_2, k_3 & q_2 & q_3)
        element_indices    = [Symbol('k_{}'.format(i)) for i in range(2,4)]
        quadrature_indices = [Symbol('q_{}'.format(i)) for i in range(2,4)]
        # indices = (k_2, q_2, k_3, q_3)
        indices = tuple(j for i in zip(element_indices, quadrature_indices) for j in i)

        # From the sub_exprs dictionary, we read all the appearing trial and test component combinations (blocks) that
        # add a non-zero contribution to the matrix
        ordered_stmts = {}
        ordered_sub_exprs_keys = {}
        for key in sub_exprs.keys():
            u_i, v_j = [get_atom_logical_derivatives(atom) for atom in key]
            ordered_stmts[u_i, v_j] = []
            ordered_sub_exprs_keys[u_i, v_j] = []
        blocks = ordered_stmts.keys()

        block_list          = list(blocks)
        trial_components    = [block[0] for block in block_list]
        test_components     = [block[1] for block in block_list]
        nu                  = len(set(trial_components))
        nv                  = len(set(test_components))

        expr = self.kernel_expr.expr

        # We store the maximum partial derivative (for a fixed direction), not including pertial derivatives
        # appearing in mapping related terms (i.e., a BilinearForm on a mapped domain will have max_logical_derivative = 0
        # even though derivatives of the (spline) mapping appear in the coupling terms).
        if isinstance(expr, (ImmutableDenseMatrix, Matrix)):
            shape = expr.shape
            logical_max_derivatives = []
            for k1 in range(shape[0]):
                for k2 in range(shape[1]):
                    logical_max_derivatives.append(get_max_logical_partial_derivatives(expr[k1,k2]))
            max_logical_derivative = max([max([value for value in dic.values()]) for dic in logical_max_derivatives])
        else:
            max_logical_derivative = max([value for value in get_max_logical_partial_derivatives(expr).values()])

        # See comment underneath this code block for more details.
        # There was a test case, in which the amount of generated StencilMatrices (one for each appearing block,
        # i.e., one for each trial&test component combination for which a non-zero coupling term exists)
        # was larger than the amount true amount of needed StencilMatrices.
        # That discrepancy appears when expr[block].is_zero wrongly does not detect that a block is zero,
        # whereas the corresponding block does rightfully not appear in block_list!
        if isinstance(expr, (ImmutableDenseMatrix, Matrix)): # only relevenat if either trial or test function is vector valued
            g_mat_information_false = []
            shape = expr.shape
            for k1 in range(shape[0]):
                for k2 in range(shape[1]):
                    if not expr[k1,k2].is_zero: # although it might actually be zero!
                        if (nu == 1) and (nv > 1):
                            g_mat_information_false.append((k2,k1))
                        else:
                            g_mat_information_false.append((k1,k2))
            if nu == 1:
                g_mat_information_true = [(0, get_atom_logical_derivatives(block[1]).indices[0]) for block in block_list]
            elif nv == 1:
                g_mat_information_true = [(get_atom_logical_derivatives(block[0]).indices[0], 0) for block in block_list]
            else:
                g_mat_information_true = [(get_atom_logical_derivatives(block[0]).indices[0], get_atom_logical_derivatives(block[1]).indices[0]) for block in block_list]
        else:
            g_mat_information_false = []
            g_mat_information_true = []

        # Julian O. 17.06.25: Back when I added this unreadable comment below I forgot to write a test for this problem.
        #                     Eventually it might be interesting to remove everything related to `g_mat_information_false/true`
        #                     and see where errors occur.
        #
        #1, 1: expr[1,1] = F0*sqrt(x1**2*(x1*cos(2*pi*x3) + 2)**2*(sin(pi*x2)**2 + cos(pi*x2)**2)**2*(sin(2*pi*x3)**2 + cos(2*pi*x3)**2)**2)*(pi*(x1*cos(2*pi*x3) + 2)*
        # (-2*pi*x1*sin(pi*x2)*sin(2*pi*x3)*dx1(v1[1]) - sin(pi*x2)*cos(2*pi*x3)*dx3(v1[1]))*cos(pi*x2)*w2[1] - pi*(x1*cos(2*pi*x3) + 2)*(-2*pi*x1*sin(2*pi*x3)*cos(pi*x2)*dx1(v1[1]) - 
        # cos(pi*x2)*cos(2*pi*x3)*dx3(v1[1]))*sin(pi*x2)*w2[1])/(2*pi**2*x1**2*(x1*cos(2*pi*x3) + 2)**2*(sin(pi*x2)**2 + cos(pi*x2)**2)**2*(sin(2*pi*x3)**2 + cos(2*pi*x3)**2)**2)
        # = 0 - but is not yet detected as 0! Hence a matrix is generated, that later is not required!
        #

        # Here we create a template for the names of the coupling terms arrays, 
        # depending on whether or not trial and test function are scalar or vector valued
        if nv > 1:
            ct_str = 'coupling_terms_u_{u_i}_v_{v_j}' if nu > 1 else 'coupling_terms_u_v_{v_j}'
        else:
            ct_str = 'coupling_terms_u_{u_i}_v' if nu > 1 else 'coupling_terms_u_v'

        # Now we format this template based on the appearing blocks (combinations of trial and test function components)
        # and transform those formatted strings into IndexedBase objects
        lhs = {}
        for block in blocks:
            u_i = get_atom_logical_derivatives(block[0]).indices[0] if nu > 1 else 0
            v_j = get_atom_logical_derivatives(block[1]).indices[0] if nv > 1 else 0
            ct = ct_str.format(u_i=u_i, v_j=v_j)
            lhs[block] = IndexedBase(f'{ct}')
        
        # lhs[block] will look w.g. like this coupling_terms_u_v (u, v scalar).
        # Now, we add to that [k_2, q_2, k_3, q_3, count], where count enumerates the sub expressions belonging to the same block
        # sub expressions corresponding to the block (u[0], v[1]) might be: (u[0], v[1]), (dx1(u[0]), v[1]), (dx2(u[0]), v[1]), ...
        # and then assign the corresponding rhs, e.g. temp_7*(temp_10**2*temp_9 + temp_11**2*temp_9 + temp_8**2*temp_9), to obtain:
        # coupling_terms_u_v[k_2, q_2, k_3, q_3, 4] = temp_7*(temp_10**2*temp_9 + temp_11**2*temp_9 + temp_8**2*temp_9)
        counts = {block:0 for block in blocks}
        for r,key in zip(rhs, sub_exprs.keys()):
            u_i, v_j = [get_atom_logical_derivatives(atom) for atom in key]
            count = counts[u_i, v_j]
            counts[u_i, v_j] += 1
            ordered_stmts[u_i, v_j].append(Assign(lhs[u_i, v_j][(*indices, count)], r))
            ordered_sub_exprs_keys[u_i, v_j].append(key)
        # ordered_stmts is a dict whose keys are combinations of trial and test functions components (e.g. u[0], v[1]),
        # and whose values are a list of coupling term assignments corresponding to this block, e.g.
        # (v1[0], v2[0]): [coupling_terms_u_0_v_0[k_2, q_2, k_3, q_3, 0] := -1, coupling_terms_u_0_v_0[k_2, q_2, k_3, q_3, 1] := 1]
        # 
        # The information regarding which partial derivative combination belongs to which coupling term is stored in ordered_sub_exprs_keys.
        # This dict has the same keys, but instead of coupling term assignments as values, list of tuples of partial derivative combinations are stored.

        # temps, which previously consisted of tuples like this one: (temp_0, -F2_1*F1_1),
        # will now be a tuple consisting of assignments, e.g. (temp_0 := -F2_1*F1_1, ...)
        temps = tuple(Assign(a,b) for a,b in temps)

        return temps, ordered_stmts, ordered_sub_exprs_keys, mapping_option, field_derivatives, g_mat_information_false, g_mat_information_true, max_logical_derivative

    #--------------------------------------------------------------------------
    def construct_arguments_generate_assembly_file(self):
        """
        Collect the arguments used in the assembly method, and generate and possibly pyccelize the assembly function.

        Used only when sum factorization is enabled, else the method construct_arguments is called.

        Returns
        -------
        args: tuple
            The arguments passed to the assembly method.

        threads_args: None
            None as openMP parallelization is not supported by this implementation.

        """
        temps, ordered_stmts, ordered_sub_exprs_keys, mapping_option, field_derivatives, g_mat_information_false, g_mat_information_true, max_logical_derivative = self.read_BilinearForm()

        # Each block corresponds to a combination of trial and test function components, and thus indeed to a "block" in the matrix.
        # Not all possible combination have to exist, e.g.,
        # given a function space of vector valued functions V (3d) and a bilinear form a: VxV -> R, a(u, v) = (u, v)_L^2(Omega)
        # there will be only 3 blocks on a logical domain (u[0]&v[0], u[1]&v[1], u[2]&v[2]),
        # but up to 9 blocks on a mapped domain (e.g. u[0]&v[1], ...)
        blocks              = ordered_stmts.keys()
        block_list          = list(blocks)
        trial_components    = [block[0] for block in block_list]
        test_components     = [block[1] for block in block_list]
        # dim = 1 corresponds to a scalar valued function, dim = 3 to a vector valued function
        trial_dim           = len(set(trial_components))
        test_dim            = len(set(test_components))

        # A reminder that this implementation only supports bilinear forms on 3d domains.
        d = 3
        assert d == 3

        # Rename - also: establish that throughout "u" corresponds to the trial function, whereas "v" corresponds to the test function
        nu = trial_dim # dim of trial function; 1 (scalar) or 3 (vector)
        nv = test_dim  # dim of test function ; 1 (scalar) or 3 (vector)

        # Obtain the most basic information: function values, degrees, spans, ...
        test_basis, test_degrees, spans, pads, test_mult = construct_test_space_arguments(self.test_basis)
        trial_basis, trial_degrees, pads, trial_mult      = construct_trial_space_arguments(self.trial_basis)
        n_elements, quads, quad_degrees             = construct_quad_grids_arguments(self.grid[0], use_weights=False)

        #! pads is being overwritten. That is because already somewhere else (__init__ of StencilMatrix via self.allocate_matrices)
        # do we assert that domain and codomain (trial and test) pads coincide!
        # That is not strictly necessary as Valentin at some point proved in one of his branches, but currently not implemented as
        # not required.

        #! the above pads variable is multiplied by the multiplicity vector! For the remaining implementation, we need
        # the pads vector un-multiplied, as obtained by :
        pads = self.test_basis.space.coeff_space.pads

        # quad_degrees is the amount of quadrature points per element in each direction
        # Clearly, this amount must coincide with the amount of basis function values stored per element in test_basis and trial_basis
        n_element_1, n_element_2, n_element_3   = n_elements
        k1, k2, k3                              = quad_degrees

        # We store component wise degree and function values for trial and test function in the dictionaries
        # trial_u_p, global_basis_u, test_v_p, global_basis_v
        if (nu == 3) and (len(trial_basis) == 3):
            # Edge Case: If the trial function space V is a VectorFunctionSpace
            # but neither an Hdiv nor an Hcurl space, i.e.,
            # V = VectorFunctionSpace('V', domain) and not +, kind='hcurl') or +, kind='hdiv')
            # then the function values in each of the three directions are identical for each of the three components.
            # Hence len(trial_basis) == 3 instead of 9. 
            # 
            # global_basis_u is a dict whose values are arrays of function values of one particular trial function component,
            # hence for this edge case we simply assign the same array trial_basis to each component
            # Same function degree in each direction for each component -> do the same thing with trial_u_p
            trial_u_p   = {u:trial_degrees for u in range(nu)}
            global_basis_u  = {u:trial_basis    for u in range(nu)}
        else:
            trial_u_p       = {u:trial_degrees[d*u:d*(u+1)] for u in range(nu)}
            global_basis_u  = {u:trial_basis[d*u:d*(u+1)]    for u in range(nu)}
        if (nv == 3) and (len(test_basis) == 3):
            # See above explanation, which also applies for the spans variable
            test_v_p   = {v:test_degrees for v in range(nv)}
            global_basis_v  = {v:test_basis   for v in range(nv)}
            spans = [*spans, *spans, *spans]
        else:
            test_v_p       = {v:test_degrees[d*v:d*(v+1)] for v in range(nv)}
            global_basis_v  = {v:test_basis[d*v:d*(v+1)]   for v in range(nv)}

        # See other method construct_arguments:
        # When self._target is an Interface domain len(self._grid) == 2
        # where grid contains the QuadratureGrid of both sides of the interface
        assert len(self.grid) == 1
        if self.mapping:
            # We gather mapping related information in the case of a Bspline mapping
            # self.mapping == False if either no or an analytical mapping

            map_coeffs = [[e._coeffs._data for e in self.mapping._fields]]
            spaces     = [self.mapping._fields[0].space]
            map_degree = [sp.degree for sp in spaces]
            map_span   = [[q.spans - s for q,s in zip(sp.get_assembly_grids(*self.nquads), sp.coeff_space.starts)] for sp in spaces]
            map_basis  = [[q.basis for q in sp.get_assembly_grids(*self.nquads)] for sp in spaces]
            points     = [g.points for g in self.grid]
            weights    = [self.mapping.weights_field.coeffs._data] if self.is_rational_mapping else []

            for i in range(len(self.grid)):
                axis   = self.grid[i].axis
                # See construct_arguments - have not come across an example of when axis was not None!
                assert axis is None

            map_degree = flatten(map_degree)
            map_span   = flatten(map_span)
            map_basis  = flatten(map_basis)
            points     = flatten(points)
            mapping = [*map_coeffs[0], *weights]
        else:

            mapping    = []
            map_degree = []
            map_span   = []
            map_basis  = []

        #---------- The following part is entirely different from the old construct_arguments method ----------

        # Each block, say u[0]&v[1], 
        # consists of possibly many derivative combinations (sub-expressions) of these two components, e.g.
        # dx1(u[0])&dx1(v[1]) or dx1(u[0])&dx2(v[1]) (dx1, dx2, dx3 representing respective partial derivatives).
        #
        # For each block, here still e.g. u[0]&v[1], 
        # and for each sub-expression, we store corresponding derivative information:
        # get_index_logical_derivatives(dx1(u[0])) = {'x1': 1, 'x2': 0, 'x3': 0}
        # get_index_logical_derivatives(dx2(v[1])) = {'x1': 0, 'x2': 1, 'x3': 0}
        # Each of these 6 dicts has for each block an array of length #sub-expressions (appearing derivative combination) stored
        # x2_test_keys[(u[0], v[1])][3] = 2 means, that the fourth sub-expression of block (u[0], v[1]) 
        # involves a second partial derivative of the test function in x2 direction
        x1_trial_keys = {block:[] for block in blocks}
        x1_test_keys  = {block:[] for block in blocks}
        x2_trial_keys = {block:[] for block in blocks}
        x2_test_keys  = {block:[] for block in blocks}
        x3_trial_keys = {block:[] for block in blocks}
        x3_test_keys  = {block:[] for block in blocks}

        for block in blocks:
            # alpha, beta for example being dx1(u[0]), dx2(v[1])
            for alpha, beta in ordered_sub_exprs_keys[block]:
                x1_trial_keys[block].append(get_index_logical_derivatives(alpha)['x1'])
                x1_test_keys [block].append(get_index_logical_derivatives(beta) ['x1'])
                x2_trial_keys[block].append(get_index_logical_derivatives(alpha)['x2'])
                x2_test_keys [block].append(get_index_logical_derivatives(beta) ['x2'])
                x3_trial_keys[block].append(get_index_logical_derivatives(alpha)['x3'])
                x3_test_keys [block].append(get_index_logical_derivatives(beta) ['x3'])

        # See sum factorization paper by Bressan & Takacs:
        # coupling_terms, a3 and a2 correspond to A^{>=4}_{x1,x2,x3}, A^{>=3}_{x1,x2} and A^{>=2}_{x1}
        # Here, for each block we assign a zero-array of the correct size.
        coupling_terms = {}
        a3 = {}
        a2 = {}

        # For each block, we precompute ~enough~ products of partial derivatives of trial and basis functions in each direction
        # These precomputed values will then be read rather than computed in the assembly
        test_trial_1s = {}
        test_trial_2s = {}
        test_trial_3s = {}

        # keys_1/2/3 is a restructuring of the 6 dictionaries created above
        keys_1 = {}
        keys_2 = {}
        keys_3 = {}

        assembly_backend = self.backend
        if self._pyccelize_test_trial_computation and assembly_backend['name'] == 'pyccel':

            import os
            if not os.path.isdir('__psydac__'):
                os.makedirs('__psydac__')

            comm = self.comm

            if comm is not None and comm.size > 1:
                if comm.rank == 0:
                    filename = '__psydac__/test_trial_computation.py'
                    code = self.test_trial_template
                    f = open(filename, 'w')
                    f.writelines(code)
                    f.close()
            else:
                filename = '__psydac__/test_trial_computation.py'
                code = self.test_trial_template
                f = open(filename, 'w')
                f.writelines(code)
                f.close()

            base_dirpath = os.getcwd()
            sys.path.insert(0, base_dirpath)

            package = importlib.import_module(f'__psydac__.test_trial_computation')
            kwargs = {
                'language'          : 'fortran',
                'compiler_family'   : assembly_backend['compiler_family'],
                'flags'             : assembly_backend['flags'],
                'openmp'            : True if assembly_backend['openmp'] else False,
                'verbose'           : False,
                'comm'              : self.comm,
            }

            test_trial_func = epyccel(package.test_trial_array, **kwargs)

        for block in blocks:
            # We translate a block, e.g. (u[0], v[1]) into two integers u_i=0, v_j=1.
            # In the case of a scalar function (u, v instead of u[0], u[1], u[2], v[0], v[1], v[2]), store 0.
            u_i = block[0].indices[0] if nu > 1 else 0
            v_j = block[1].indices[0] if nv > 1 else 0

            # keys_2[(u[0], v[1])][3] = (1,2) means that the fourth sub-expression corresponding to the trial-test-function-component-product
            # u[0] * v[1] involves a first derivative in x2 direction of the trial function and a second derivative in x2 direction of the test function            
            keys_1[block] = np.array([(alpha_1, beta_1) for alpha_1, beta_1 in zip(x1_trial_keys[block], x1_test_keys[block])])
            keys_2[block] = np.array([(alpha_2, beta_2) for alpha_2, beta_2 in zip(x2_trial_keys[block], x2_test_keys[block])])
            keys_3[block] = np.array([(alpha_3, beta_3) for alpha_3, beta_3 in zip(x3_trial_keys[block], x3_test_keys[block])])

            # Those are the function values in each direction of a particular component of the trial/test function
            global_basis_u_1, global_basis_u_2, global_basis_u_3 = global_basis_u[u_i]
            global_basis_v_1, global_basis_v_2, global_basis_v_3 = global_basis_v[v_j]

            # Those are the Bspline degrees in each direction of a particular component of the trial/test function
            trial_u_p1, trial_u_p2, trial_u_p3 = trial_u_p[u_i]
            test_v_p1,  test_v_p2,  test_v_p3  = test_v_p [v_j]
            
            max_p_2 = max(test_v_p2, trial_u_p2)
            max_p_3 = max(test_v_p3, trial_u_p3)

            # That's the amount of subexpressions, i.e., combinations of partial derivatives appearing for a specific combination of 
            # trial and test function components
            n_expr = len(ordered_stmts[block])

            # To compute enough (possibly too many, but never too few) products of trial and test functions, we read the maximum
            # appearing partial derivative (for this specific block, in each direction, for both trial and test function)
            max_block_trial_x1_derivative = max(x1_trial_keys[block])
            max_block_trial_x2_derivative = max(x2_trial_keys[block])
            max_block_trial_x3_derivative = max(x3_trial_keys[block])
            max_block_test_x1_derivative = max(x1_test_keys[block])
            max_block_test_x2_derivative = max(x2_test_keys[block])
            max_block_test_x3_derivative = max(x3_test_keys[block])

            # On each Bspline cell (element / subdomain), there are (test_degree+1)*(trial_degree+1) test & trial function pairs
            # of non-zero product.
            # Hence, we assign zeros for each element, each quadrature point on the element, each test and trial function combination,
            # and each (or even more than required) appearing partial derivative combination of these functions - in each direction
            test_trial_1 = np.zeros((n_element_1, k1, test_v_p1 + 1, trial_u_p1 + 1, max_block_trial_x1_derivative+1, max_block_test_x1_derivative+1), dtype='float64')
            test_trial_2 = np.zeros((n_element_2, k2, test_v_p2 + 1, trial_u_p2 + 1, max_block_trial_x2_derivative+1, max_block_test_x2_derivative+1), dtype='float64')
            test_trial_3 = np.zeros((n_element_3, k3, test_v_p3 + 1, trial_u_p3 + 1, max_block_trial_x3_derivative+1, max_block_test_x3_derivative+1), dtype='float64')

            # And that's how we fill the test_trial arrays
            if self._pyccelize_test_trial_computation and assembly_backend['name'] == 'pyccel':
                for args in zip(n_elements, 
                                quad_degrees, [test_v_p1,  test_v_p2,  test_v_p3], [trial_u_p1, trial_u_p2, trial_u_p3],
                                [global_basis_u_1, global_basis_u_2, global_basis_u_3], [global_basis_v_1, global_basis_v_2, global_basis_v_3], 
                                [max_block_trial_x1_derivative, max_block_trial_x2_derivative, max_block_trial_x3_derivative], [max_block_test_x1_derivative, max_block_test_x2_derivative, max_block_test_x3_derivative], 
                                [test_trial_1, test_trial_2, test_trial_3]):
                    
                    args = tuple(np.int64(a) if isinstance(a, int) else a for a in args)

                    test_trial_func(*args)
            else:
                for k_1 in range(n_element_1):
                    for q_1 in range(k1):
                        for i_1 in range(test_v_p1 + 1):
                            for j_1 in range(trial_u_p1 + 1):
                                trial   = global_basis_u_1[k_1, j_1, :, q_1]
                                test    = global_basis_v_1[k_1, i_1, :, q_1]
                                for alpha_1 in range(max_block_trial_x1_derivative+1):
                                    for beta_1 in range(max_block_test_x1_derivative+1):
                                        test_trial_1[k_1, q_1, i_1, j_1, alpha_1, beta_1] = trial[alpha_1] * test[beta_1]

                for k_2 in range(n_element_2):
                    for q_2 in range(k2):
                        for i_2 in range(test_v_p2 + 1):
                            for j_2 in range(trial_u_p2 + 1):
                                trial   = global_basis_u_2[k_2, j_2, :, q_2]
                                test    = global_basis_v_2[k_2, i_2, :, q_2]
                                for alpha_2 in range(max_block_trial_x2_derivative+1):
                                    for beta_2 in range(max_block_test_x2_derivative+1):
                                        test_trial_2[k_2, q_2, i_2, j_2, alpha_2, beta_2] = trial[alpha_2] * test[beta_2]

                for k_3 in range(n_element_3):
                    for q_3 in range(k3):
                        for i_3 in range(test_v_p3 + 1):
                            for j_3 in range(trial_u_p3 + 1):
                                trial   = global_basis_u_3[k_3, j_3, :, q_3]
                                test    = global_basis_v_3[k_3, i_3, :, q_3]
                                for alpha_3 in range(max_block_trial_x3_derivative+1):
                                    for beta_3 in range(max_block_test_x3_derivative+1):
                                        test_trial_3[k_3, q_3, i_3, j_3, alpha_3, beta_3] = trial[alpha_3] * test[beta_3]

            test_trial_1s[block] = test_trial_1
            test_trial_2s[block] = test_trial_2
            test_trial_3s[block] = test_trial_3

            # Instead of having a different a3, a2 & coupling term array for each sub-expression, we choose to have only one
            # such array per block.
            # a3 will store line integral values for all combinations of test and trial functions in x3 direction, hence the dimension
            # (n_element_3 + test_v_p3 + (mult[2]-1)*(n_element_3-1), 2 * max_p_3 + 1)
            # a2 will store surface integral values for all combinations of test and trial functions in x2 and x3 direction, hence the dimension ...
            # coupling_terms stores point values of the coupling terms at all quadrature points 
            # but only in x2 and x3 direction, because we only "precompute" this array for a fixed quadrature point in x1 direction

            # a3[block] size explained: #sub expressions ; #test functions depending on x3 ; #complicated expression for the minimum columns needed
            # to store local information correctly. 2*degree+1 in the simplest case.
            n_funs_x2 = n_element_2 + test_v_p2 + (test_mult[1]-1)*(n_element_2-1)
            n_funs_x3 = n_element_3 + test_v_p3 + (test_mult[2]-1)*(n_element_3-1)
            n_cols_x2 = max( int(max_p_2 + 1 + np.floor(max_p_2 / test_mult[1]) * trial_mult[1]), 2*max_p_2+1 )
            n_cols_x3 = max( int(max_p_3 + 1 + np.floor(max_p_3 / test_mult[2]) * trial_mult[2]), 2*max_p_3+1 )
            
            a3[block] = np.zeros((n_expr, n_funs_x3, n_cols_x3), dtype='float64')
            a2[block] = np.zeros((n_expr, n_funs_x2, n_funs_x3, n_cols_x2, n_cols_x3), dtype='float64')

            coupling_terms[block] = np.zeros((n_element_2, k2, n_element_3, k3, n_expr), dtype='float64')

        # We gather the socalled new args - all other args are being obtained in a similar way using the old assembly implementation
        new_args = (*list(test_trial_1s.values()), 
                    *list(test_trial_2s.values()), 
                    *list(test_trial_3s.values()), 
                    *list(a3.values()),
                    *list(a2.values()),
                    *list(coupling_terms.values()))
        
        # This part is a bit shady.
        # There has been a case, where my code wasn't running, because one instance of deep-(PSYDAC/Sympde/Sympy)-code
        # correctly understood that a possibly complicated expression (corresponding to a block) in fact evaluates to 0,
        # and hence no StencilMatrix for that particular block ever needs to be created - but a different part of
        # deep-(PSYDAC/SymPDE/SymPy)-code did not get that simplification right (yet?), and decided that the assembly code
        # needs a StencilMatrix as input for this particular block.
        # See readBilinearForm for additional information.
        # This part of the code filters out unnecessary StencilMatrices, such that only the relevant StencilMatrices
        # are being passed to the assembly function
        expr = self.kernel_expr.expr
        if isinstance(expr, (ImmutableDenseMatrix, Matrix)):
            matrices = []
            for i, block in enumerate(g_mat_information_false):
                if block in g_mat_information_true:
                    matrices.append(self._global_matrices[i])
        else:
            matrices = self._global_matrices

        # We have gathered all args!
        args = (*map_basis, *spans, *map_span, *quads, *map_degree, *n_elements, *quad_degrees, *pads, *mapping, *matrices,
                *new_args)
        
        threads_args = ()

        args = tuple(np.int64(a) if isinstance(a, int) else a for a in args)
        threads_args = tuple(np.int64(a) if isinstance(a, int) else a for a in threads_args)

        #---------- We now generate the assembly file ----------

        # file_id is a random string that has been used to name the assembly file
        file_id = self.make_file(temps, ordered_stmts, field_derivatives, max_logical_derivative, test_mult, trial_mult, test_v_p, trial_u_p, keys_1, keys_2, keys_3, mapping_option)

        # Store the current directory and add it to the variable `sys.path`
        # to imitate Python's import behavior
        import os
        base_dirpath = os.getcwd()
        sys.path.insert(0, base_dirpath)

        # Import the generated assembly function
        package = importlib.import_module(f'__psydac__.assemble_{file_id}')

        # The assembly function is the one that has been generated in the make_file method
        assembly_function_name = f'assemble_matrix_{file_id}'
        assembly_function = getattr(package, assembly_function_name)

        # If the backend is pyccel, we compile the new assembly function
        assembly_backend = self.backend
        if assembly_backend['name'] == 'pyccel':
            kwargs = {
                'language'          : 'fortran',  # hardcoded for now
                'compiler_family'   : assembly_backend['compiler_family'],
                'flags'             : assembly_backend['flags'],
                'openmp'            : True if assembly_backend['openmp'] else False,
                'verbose'           : False,
                # 'folder': assembly_backend['folder'],
                'comm'              : self.comm,
                # 'time_execution': verbose,
                # 'verbose': verbose
            }
            new_func = epyccel(assembly_function, **kwargs)
        else:
            new_func = assembly_function

        # Use the new assembly function (either compiled or not)
        self._func = new_func

        return args, threads_args

    #--------------------------------------------------------------------------
    @property
    def test_trial_template(self):
        code = '''def test_trial_array(n_element : "int64", 
                     quad_degree : "int64", test_degree : "int64", trial_degree : "int64", 
                     trial_basis : "float64[:,:,:,:]", test_basis : "float64[:,:,:,:]", 
                     max_trial_derivative : "int64", max_test_derivative : "int64", 
                     test_trial : "float64[:,:,:,:,:,:]"):

    for k in range(n_element):
        for q in range(quad_degree):
            for i in range(test_degree + 1):
                for j in range(trial_degree + 1):
                    trial   = trial_basis[k, j, :, q]
                    test    = test_basis [k, i, :, q]
                    for alpha in range(max_trial_derivative + 1):
                        for beta in range(max_test_derivative + 1):
                            test_trial[k, q, i, j, alpha, beta] = trial[alpha] * test[beta]

    return
'''
        return code
