# coding: utf-8

# TODO: - init_fem is called whenever we call discretize. we should check that
#         nderiv has not been changed. shall we add nquads too?

import numpy as np

import string
import random
import importlib
#import inspect

from sympy                  import ImmutableDenseMatrix, Matrix, Symbol, sympify
from sympy.tensor.indexed   import Indexed, IndexedBase
from sympy.simplify         import cse_main

from pyccel.epyccel import epyccel

from sympde.expr          import BilinearForm as sym_BilinearForm
from sympde.expr          import LinearForm as sym_LinearForm
from sympde.expr          import Functional as sym_Functional
from sympde.expr          import Norm as sym_Norm
from sympde.expr          import SemiNorm as sym_SemiNorm
from sympde.expr          import TerminalExpr
from sympde.topology      import Boundary, Interface
from sympde.topology      import LogicalExpr
from sympde.topology      import VectorFunctionSpace
from sympde.topology      import ProductSpace
from sympde.topology      import H1SpaceType, L2SpaceType, UndefinedSpaceType
from sympde.topology             import SymbolicExpr, Mapping
from sympde.topology.space       import ScalarFunction, VectorFunction, IndexedVectorFunction
from sympde.topology.derivatives import get_atom_logical_derivatives
from sympde.topology.derivatives import _logical_partial_derivatives
from sympde.topology.derivatives import get_index_logical_derivatives
from sympde.calculus.core        import PlusInterfaceOperator

from psydac.api.basic        import BasicDiscrete
from psydac.api.basic        import random_string
from psydac.api.grid         import QuadratureGrid, BasisValues
from psydac.api.utilities    import flatten
from psydac.linalg.stencil   import StencilVector, StencilMatrix, StencilInterfaceMatrix
from psydac.linalg.basic     import ComposedLinearOperator
from psydac.linalg.block     import BlockVectorSpace, BlockVector, BlockLinearOperator
from psydac.cad.geometry     import Geometry
from psydac.mapping.discrete import NurbsMapping, SplineMapping
from psydac.fem.vector       import ProductFemSpace, VectorFemSpace
from psydac.fem.basic        import FemField
from psydac.fem.projectors   import knot_insertion_projection_operator
from psydac.core.bsplines    import find_span, basis_funs_all_ders
from psydac.ddm.cart         import InterfaceCartDecomposition
from psydac.pyccel.ast.core  import _atomic, Assign

__all__ = (
    'collect_spaces',
    'compute_diag_len',
    'construct_test_space_arguments',
    'construct_trial_space_arguments',
    'construct_quad_grids_arguments',
    'reset_arrays',
    'do_nothing',
    'extract_stencil_mats',
    'DiscreteBilinearForm',
    'DiscreteFunctional',
    'DiscreteLinearForm',
    'DiscreteSumForm',
)

#==============================================================================
def collect_spaces(space, *args):
    """
    This function collect the arguments used in the assembly function

    Parameters
    ----------
    space: <FunctionSpace>
        the symbolic space

    args : <list>
        list of discrete space components like basis values, spans, ...

    Returns
    -------
    args : <list>
        list of discrete space components elements used in the asembly

    """

    if isinstance(space, ProductSpace):
        spaces = space.spaces
        indices = []
        i = 0
        for space in spaces:
            if isinstance(space, VectorFunctionSpace):
                if isinstance(space.kind, (H1SpaceType, L2SpaceType, UndefinedSpaceType)):
                    indices.append(i)
                else:
                    indices += [i+j for j in range(space.ldim)]
                i = i + space.ldim
            else:
                indices.append(i)
                i = i + 1
        args = [[e[i] for i in indices] for e in args]

    elif isinstance(space, VectorFunctionSpace):
        if isinstance(space.kind, (H1SpaceType, L2SpaceType, UndefinedSpaceType)):
            args = [[e[0]] for e in args]

    return args

#==============================================================================
def compute_diag_len(p, md, mc):
    n = ((np.ceil((p+1)/mc)-1)*md).astype('int')
    n = n-np.minimum(0, n-p)+p+1
    return n.astype('int')

#==============================================================================
def construct_test_space_arguments(basis_values):
    space          = basis_values.space
    test_basis     = basis_values.basis
    spans          = basis_values.spans
    test_degrees   = space.degree
    pads           = space.pads
    multiplicity   = space.multiplicity

    test_basis, test_degrees, spans = collect_spaces(space.symbolic_space, test_basis, test_degrees, spans)

    test_basis    = flatten(test_basis)
    test_degrees  = flatten(test_degrees)
    spans         = flatten(spans)
    pads          = flatten(pads)
    multiplicity  = flatten(multiplicity)
    pads          = [p*m for p,m in zip(pads, multiplicity)]
    return test_basis, test_degrees, spans, pads, multiplicity

def construct_trial_space_arguments(basis_values):
    space          = basis_values.space
    trial_basis    = basis_values.basis
    trial_degrees  = space.degree
    pads           = space.pads
    multiplicity   = space.multiplicity
    trial_basis, trial_degrees = collect_spaces(space.symbolic_space, trial_basis, trial_degrees)

    trial_basis    = flatten(trial_basis)
    trial_degrees  = flatten(trial_degrees)
    pads           = flatten(pads)
    multiplicity   = flatten(multiplicity)
    pads           = [p*m for p,m in zip(pads, multiplicity)]
    return trial_basis, trial_degrees, pads, multiplicity

#==============================================================================
def construct_quad_grids_arguments(grid, use_weights=True):
    points         = grid.points
    if use_weights:
        weights        = grid.weights
        quads          = flatten(list(zip(points, weights)))
    else:
        quads = flatten(list(zip(points)))

    nquads        = flatten(grid.nquads)
    n_elements    = grid.n_elements
    return n_elements, quads, nquads

def reset_arrays(*args):
    for a in args:
        a[:]= 0.j if a.dtype==complex else 0.

def do_nothing(*args): return 0

def extract_stencil_mats(mats):
    new_mats = []
    for M in mats:
        if isinstance(M, (StencilInterfaceMatrix, StencilMatrix)):
            new_mats.append(M)
        elif isinstance(M, ComposedLinearOperator):
            new_mats += [i for i in M.multiplicants if isinstance(i, (StencilInterfaceMatrix, StencilMatrix))]
    return new_mats

def id_generator(size=8, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
#==============================================================================
class DiscreteBilinearForm(BasicDiscrete):
    """
    Discrete bilinear form ready to be assembled into a matrix.

    This class represents the concept of a discrete bilinear form in Psydac.
    Instances of this class generate an appropriate matrix assembly kernel,
    allocate the matrix if not provided, and prepare a list of arguments for
    the kernel.

    Parameters
    ----------

    expr : sympde.expr.expr.BilinearForm
        The symbolic bilinear form.

    kernel_expr : sympde.expr.evaluation.KernelExpression
        The atomic representation of the bilinear form.

    domain_h : Geometry
        The discretized domain.

    spaces : list of FemSpace
        The discrete trial and test spaces.

    nquads : list or tuple of int
        The number of quadrature points used in the assembly kernel along each
        direction.

    matrix : StencilMatrix or BlockLinearOperator, optional
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

    symbolic_mapping : Sympde.topology.Mapping, optional
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
                 symbolic_mapping=None,
                 fast_assembly=False):

        if not isinstance(expr, sym_BilinearForm):
            raise TypeError('> Expecting a symbolic BilinearForm')

        assert isinstance(domain_h, Geometry)

        self._spaces = spaces

        if isinstance(kernel_expr, (tuple, list)):
            if len(kernel_expr) == 1:
                kernel_expr = kernel_expr[0]
            else:
                raise ValueError('> Expecting only one kernel')

        self._kernel_expr = kernel_expr
        self._target = kernel_expr.target
        self._domain = domain_h.domain
        self._matrix = matrix

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

        if isinstance(test_space.vector_space, BlockVectorSpace):
            vector_space = test_space.vector_space.spaces[0]
        else:
            vector_space = test_space.vector_space

        self._vector_space = vector_space
        self._num_threads  = 1
        if vector_space.parallel and vector_space.cart.num_threads>1:
            self._num_threads = vector_space.cart.num_threads

        self._update_ghost_regions = update_ghost_regions

        # In case of multiple patches, if the communicator is MPI_COMM_NULL, we do not generate the assembly code
        # because the patch is not owned by the MPI rank.
        if vector_space.parallel and vector_space.cart.is_comm_null:
            self._free_args = ()
            self._func      = do_nothing
            self._args      = ()
            self._element_loop_starts = ()
            self._element_loop_ends   = ()
            self._global_matrices     = ()
            self._threads_args        = ()
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
        discrete_space = (trial_space, test_space)

        # Assuming that all vector spaces (and their Cartesian decomposition,
        # if any) are compatible with each other, extract the first available
        # vector space from which (starts, ends, npts) will be read:
        starts = vector_space.starts
        ends   = vector_space.ends
        npts   = vector_space.npts

        # MPI communicator
        comm = vector_space.cart.comm if vector_space.parallel else None

        # Backends for code generation
        assembly_backend = backend or assembly_backend
        linalg_backend   = backend or linalg_backend

        # BasicDiscrete generates the assembly code and sets the following attributes that are used afterwards:
        # self._func, self._free_args, self._max_nderiv and self._backend
        BasicDiscrete.__init__(self, expr, kernel_expr, comm=comm, root=0, discrete_space=discrete_space,
                       nquads=nquads, is_rational_mapping=is_rational_mapping, mapping=symbolic_mapping,
                       mapping_space=mapping_space, num_threads=self._num_threads, backend=assembly_backend,
                       fast_assembly=fast_assembly)

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
                self._element_loop_starts = ()
                self._element_loop_ends   = ()
                self._global_matrices     = ()
                self._threads_args        = ()
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

        # temporary feature that allows to choose either old or new assembly to verify whether the new assembly works
        assert isinstance(fast_assembly, bool)
        self._fast_assembly = fast_assembly

        # Allocate the output matrix, if needed
        self.allocate_matrices(linalg_backend)

        #from psydac.api.tests.allocate_matrix_bug import fix_bug
        #self._fix_bug = fix_bug
        # ----- Uncomment only for the u*f // f*u test case -----
        #if fix_bug:
        #    mat = StencilMatrix(self._matrix.domain, self._matrix.codomain)
        #    self._matrix = mat
        #    self._global_matrices = [mat._data, ]

        #print(self._global_matrices[0].shape)
        # -------------------------------------------------------

        # Determine whether OpenMP instructions were generated
        with_openmp = (assembly_backend['name'] == 'pyccel' and assembly_backend['openmp']) if assembly_backend else False

        # Construct the arguments to be passed to the assemble() function, which is stored in self._func
        if self._fast_assembly == True:
            # no openmp support yet
            self._args, self._threads_args = self.construct_arguments_generate_assembly_file()
        else:
            self._args, self._threads_args = self.construct_arguments(with_openmp)

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def spaces(self):
        return self._spaces

    @property
    def grid(self):
        return self._grid

    @property
    def nquads(self):
        return self._grid[0].nquads

    @property
    def test_basis(self):
        return self._test_basis

    @property
    def trial_basis(self):
        return self._trial_basis

    @property
    def global_matrices(self):
        return self._global_matrices

    @property
    def args(self):
        return self._args

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

                if len(self.domain) > 1 and isinstance(v, FemField) and v.space.is_product:
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
                    if v.space.is_product:
                        coeffs += (e._data for e in v.coeffs)
                    else:
                        coeffs += (v.coeffs._data, )
                else:
                    consts += (v, )

            args = (*self.args, *basis, *spans, *degrees, *pads, *coeffs, *consts)

        else:
            args = self._args
        # ----- Uncomment only for the u*f // f*u test case -----
        #if self._fix_bug:
        #    if self._fast_assembly == True:
        #        if self._mapping_option == 'Bspline':
        #            args = (*args[0:28], *self._global_matrices, *args[29:])
        #        else:
        #            args = (*args[0:15], *self._global_matrices, *args[16:])
        #    else:
        #        args = (*args[:-1], *self._global_matrices)
        # -------------------------------------------------------
#        args = args + self._element_loop_starts + self._element_loop_ends

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

    def get_space_indices_from_target(self, domain, target):
        if domain.mapping:
            domain = domain.logical_domain
        if target.mapping:
            target = target.logical_domain
        domains = domain.interior.args
        if isinstance(target, Interface):
            test  = self.kernel_expr.test
            trial = self.kernel_expr.trial
            test_target  = target.plus if isinstance( test, PlusInterfaceOperator) else target.minus
            trial_target = target.plus if isinstance(trial, PlusInterfaceOperator) else target.minus
            i, j = [domains.index(test_target.domain), domains.index(trial_target.domain)]
        else:
            if isinstance(target, Boundary):
                i = domains.index(target.domain)
                j = i
            else:
                i = domains.index(target)
                j = i
        return i, j

    @property
    def _assembly_template_head(self):
        code = '''def assemble_matrix({MAPPING_PART_1}
{SPAN}                    {MAPPING_PART_2}
                    global_x1 : "float64[:,:]", global_x2 : "float64[:,:]", global_x3 : "float64[:,:]", 
                    {MAPPING_PART_3}
                    n_element_1 : "int64", n_element_2 : "int64", n_element_3 : "int64", 
                    nq1 : "int64", nq2 : "int64", nq3 : "int64", 
                    pad1 : "int64", pad2 : "int64", pad3 : "int64", 
                    {MAPPING_PART_4}
{G_MAT}{NEW_ARGS}{FIELD_ARGS}):

    from numpy import array, zeros, zeros_like, floor
    from numpy import abs as Abs
    from math import sqrt, sin, pi, cos
'''
        return code
    
    @property
    def _assembly_template_body_bspline(self):
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

{F_INIT}

{F_ASSIGN_LOOP}

                            for i_1 in range(test_mapping_p1+1):
                                mapping_1       = global_basis_mapping_1[k_1, i_1, 0, q_1]
                                mapping_1_x1    = global_basis_mapping_1[k_1, i_1, 1, q_1]
                                for i_2 in range(test_mapping_p2+1):
                                    mapping_2       = global_basis_mapping_2[k_2, i_2, 0, q_2]
                                    mapping_2_x2    = global_basis_mapping_2[k_2, i_2, 1, q_2]
                                    for i_3 in range(test_mapping_p3+1):
                                        mapping_3       = global_basis_mapping_3[k_3, i_3, 0, q_3]
                                        mapping_3_x3    = global_basis_mapping_3[k_3, i_3, 1, q_3]

                                        coeff_x = arr_coeffs_x[i_1,i_2,i_3]
                                        coeff_y = arr_coeffs_y[i_1,i_2,i_3]
                                        coeff_z = arr_coeffs_z[i_1,i_2,i_3]

                                        mapping = mapping_1*mapping_2*mapping_3
                                        mapping_x1 = mapping_1_x1*mapping_2*mapping_3
                                        mapping_x2 = mapping_1*mapping_2_x2*mapping_3
                                        mapping_x3 = mapping_1*mapping_2*mapping_3_x3

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
                                        
{TEMPS}
{COUPLING_TERMS}
'''
        return code
    
    @property 
    def _assembly_template_body_analytic(self):
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
    
    @property
    def _assembly_template_loop(self):
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

    def make_file(self, temps, ordered_stmts, field_derivatives, mult, *args, mapping_option=None):

        # ----- field strings -----
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
        field_args = basis_args+span_args+degree_args+pad_args+coeff_args

        arr_coeffs_txt = "\n".join([f"    arr_coeffs_{field} = zeros((1 + test_{field}_p1, 1 + test_{field}_p2, 1 + test_{field}_p3), dtype='float64')" for field in field_derivatives])
        span_1_txt = "\n".join([f"        span_{field}_1 = global_span_{field}_1[k_1]" for field in field_derivatives]) + "\n"
        span_2_txt = "\n".join([f"                span_{field}_2 = global_span_{field}_2[k_2]" for field in field_derivatives]) + "\n"
        span_3_txt = "\n".join([f"                        span_{field}_3 = global_span_{field}_3[k_3]" for field in field_derivatives]) + "\n"
        coeff_ranges = ", ".join([f"pad_"+"{field}"+f"_{i+1} + span_"+"{field}"+f"_{i+1} - test_"+"{field}"+f"_p{i+1}:1 + pad_"+"{field}"+f"_{i+1} + span_"+"{field}"+f"_{i+1}" for i in range(3)])
        arr_coeffs_assign_txt = "\n".join([f"                        arr_coeffs_{field}[:,:,:] = global_arr_coeffs_{field}[{coeff_ranges.format(field=field)}]" for i, field in enumerate(field_derivatives)])
        field_init = "\n".join([f"                            {derivative}     = 0.0" for field in field_derivatives for derivative in field_derivatives[field]])
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
        assign = "\n".join(assign)
        # -------------------------

        verbose = False
        if verbose: print(mapping_option)

        code_head = self._assembly_template_head
        code_loop = self._assembly_template_loop

        if mapping_option == 'Bspline':
            code_body = self._assembly_template_body_bspline
        else:
            code_body = self._assembly_template_body_analytic
        self._mapping_option = mapping_option
        
        test_v_p, trial_u_p, keys_1, keys_2, keys_3 = args

        blocks              = ordered_stmts.keys()
        block_list          = list(blocks)
        trial_components    = [block[0] for block in block_list]
        test_components     = [block[1] for block in block_list]
        nu                  = len(set(trial_components))
        nv                  = len(set(test_components))
        d = 3
        assert d == 3

        # Prepare strings depending on whether the trial and test function are vector-valued or not (nu, nv > 1 or == 1)

        #------------------------- STRINGS HEAD -------------------------
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
        #------------------------- STRINGS BODY -------------------------
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
        #------------------------- STRINGS LOOP -------------------------
        span_2_str          = 'span_v_{v_j}_2'          if nv > 1 else 'span_v_2'
        span_3_str          = 'span_v_{v_j}_3'          if nv > 1 else 'span_v_3'
        global_span_2_str   = 'global_span_v_{v_j}_2'   if nv > 1 else 'global_span_v_2'
        global_span_3_str   = 'global_span_v_{v_j}_3'   if nv > 1 else 'global_span_v_3'

        #-------------------------------------------------------------

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
            G_MAT += f'                    {g_mat} : "float64[:,:,:,:,:,:]",\n'

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

        head = code_head.format(SPAN=SPAN, 
                                G_MAT=G_MAT, 
                                NEW_ARGS=NEW_ARGS,
                                MAPPING_PART_1=MAPPING_PART_1,
                                MAPPING_PART_2=MAPPING_PART_2,
                                MAPPING_PART_3=MAPPING_PART_3,
                                MAPPING_PART_4=MAPPING_PART_4,
                                FIELD_ARGS=field_args)
        
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
            A1_1 = f'{mult[0]}*pad1 + {SPAN_V_1} - {test_v_p1} : {mult[0]}*pad1 + {SPAN_V_1} + 1'                   if mult[0] > 1 else f'pad1 + {SPAN_V_1} - {test_v_p1} : pad1 + {SPAN_V_1} + 1'
            A1_2 = f'{mult[1]}*pad2 : {mult[1]}*pad2 + n_element_2 + {test_v_p2} + ({mult[1]}-1)*(n_element_2-1)'   if mult[1] > 1 else f'pad2 : pad2 + n_element_2 + {test_v_p2}'
            A1_3 = f'{mult[2]}*pad3 : {mult[2]}*pad3 + n_element_3 + {test_v_p3} + ({mult[2]}-1)*(n_element_3-1)'   if mult[2] > 1 else f'pad3 : pad3 + n_element_3 + {test_v_p3}'
            A1          += f'        {a1} = {g_mat}[{A1_1}, {A1_2}, {A1_3}, :, :, :]\n'

        for v_j in range(nv):
            local_span_v_1 = span_v_1_str.format(v_j=v_j)
            global_span_v = global_span_v_str.format(v_j=v_j)
            LOCAL_SPAN += f'        {local_span_v_1} = {global_span_v}1[k_1]\n'

        for temp in temps:
            TEMPS += f'                            {temp.lhs} = {temp.rhs}\n'
        for block in blocks:
            for stmt in ordered_stmts[block]:
                COUPLING_TERMS += f'                            {stmt.lhs} = {stmt.rhs}\n'
        
        KEYS = KEYS_2 + KEYS_3

        body = code_body.format(LOCAL_SPAN=LOCAL_SPAN, 
                                KEYS=KEYS, 
                                A1=A1, 
                                TEMPS=TEMPS, 
                                COUPLING_TERMS=COUPLING_TERMS,
                                F_COEFFS_ZEROS=arr_coeffs_txt,
                                F_SPAN_1=span_1_txt,
                                F_SPAN_2=span_2_txt,
                                F_SPAN_3=span_3_txt,
                                F_COEFFS=arr_coeffs_assign_txt,
                                F_INIT=field_init,
                                F_ASSIGN_LOOP=assign)
        
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

            I_1 = f'int(floor(i_1/{mult[0]})*{mult[0]})' if mult[0] > 1 else 'i_1'
            I_2 = f'int(floor(i_2/{mult[1]})*{mult[1]})' if mult[1] > 1 else 'i_2'
            I_3 = f'int(floor(i_3/{mult[2]})*{mult[2]})' if mult[2] > 1 else 'i_3'

            loop = code_loop.format(A1=A1,
                                    A2=A2,
                                    A3=A3,
                                    TEST_TRIAL_2=TEST_TRIAL_2,
                                    TEST_TRIAL_3=TEST_TRIAL_3,
                                    SPAN_2=SPAN_2,
                                    SPAN_3=SPAN_3,
                                    GLOBAL_SPAN_2=GLOBAL_SPAN_2,
                                    GLOBAL_SPAN_3=GLOBAL_SPAN_3,
                                    KEYS_2=KEYS_2,
                                    KEYS_3=KEYS_3,
                                    COUPLING_TERMS=COUPLING_TERMS,
                                    TEST_V_P1=TEST_V_P1,
                                    TEST_V_P2=TEST_V_P2,
                                    TEST_V_P3=TEST_V_P3,
                                    TRIAL_U_P1=TRIAL_U_P1,
                                    TRIAL_U_P2=TRIAL_U_P2,
                                    TRIAL_U_P3=TRIAL_U_P3,
                                    MAX_P1=MAX_P1,
                                    MAX_P2=MAX_P2,
                                    MAX_P3=MAX_P3,
                                    NEXPR=NEXPR,
                                    A2_TEMP=A2_TEMP,
                                    I_1=I_1,
                                    I_2=I_2,
                                    I_3=I_3)
            
            loop_str += loop

        assembly_code += loop_str
        assembly_code += '\n    return\n'
        
        #------------------------- MAKE FILE -------------------------
        import os
        if not os.path.isdir('__psydac__'):
            os.makedirs('__psydac__')
        id = id_generator()
        filename = f'__psydac__/assemble_{id}.py'
        f = open(filename, 'w')
        f.writelines(assembly_code)
        f.close()
        return id

    def read_BilinearForm(self):
        
        a = self.expr
        verbose = False

        domain = a.domain
        mapping_option = 'Bspline' if isinstance(self._mapping, SplineMapping) else None

        tests  = a.test_functions
        trials = a.trial_functions
        fields = a.fields
        if verbose: print(f'In readBF: tests: {tests}')
        if verbose: print(f'In readBF: trials: {trials}')
        if verbose: print(f'In readBF: fields: {fields}')

        texpr  = TerminalExpr(a, domain)[0]
        if verbose: print(f'In readBF: texpr: {texpr}')

        atoms_types = (ScalarFunction, VectorFunction, IndexedVectorFunction)
        atoms       = _atomic(texpr, cls=atoms_types+_logical_partial_derivatives)
        if verbose: print(f'In readBF: atoms: {atoms}')

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

        if verbose: print(f'In readBF: test_atoms: {test_atoms}')
        if verbose: print(f'In readBF: trial_atoms: {trial_atoms}')
        if verbose: print(f'In readBF: field_atoms: {field_atoms}')

        # u in trials is not get_atom_logical_derivative(a) for a in atoms and atom in trials

        for a in atoms:
            atom = get_atom_logical_derivatives(a)
            if hasattr(atom, 'base'):
                if verbose: print(f'In readBF: atom.__class__: {atom.__class__} --- atom.base.__class__: {atom.base.__class__}')
            else:
                if verbose: print(f'In readBF: atom.__class__: {atom.__class__}')
            if not ((isinstance(atom, Indexed) and isinstance(atom.base, Mapping)) or (isinstance(atom, IndexedVectorFunction))):
                if atom in tests:
                    test_atoms[tests[0]].append(a) # apparently previously atom instead of tests[0]
                elif atom in trials:
                    trial_atoms[trials[0]].append(a) # apparently previously atom instead of trials[0]
                elif atom in fields:
                    field_atoms[fields[0]].append(a) # this 0 here can't always work! At least not for >1 free fields
                else:
                    raise NotImplementedError(f"atoms of type {str(a)} are not supported")
            elif isinstance(atom, IndexedVectorFunction):
                if atom.base in tests:
                    for vi in test_atoms:
                        if vi == atom:
                            test_atoms[vi].append(a)
                            break
                elif atom.base in trials:
                    for ui in trial_atoms:
                        if ui == atom:
                            trial_atoms[ui].append(a)
                            break
                elif atom.base in fields:
                    for fi in field_atoms:
                        if fi == atom:
                            field_atoms[fi].append(a)
                            break
                else:
                    raise NotImplementedError(f"atoms of type {str(a)} are not supported")
            if verbose: print(f'In readBF: test_atoms: {test_atoms}')
            if verbose: print(f'In readBF: trial_atoms: {trial_atoms}')
            if verbose: print(f'In readBF: field_atoms: {field_atoms}')

        field_derivatives = {}
        for key in field_atoms:
            sym_key = SymbolicExpr(key)
            field_derivatives[sym_key] = {}
            for f in field_atoms[key]:
                field_derivatives[sym_key][SymbolicExpr(f)] = get_index_logical_derivatives(f)
        
        if verbose: print(f'In readBF: field_derivatives: {field_derivatives}')

        syme = False
        if syme:
            from symengine import sympify as syme_sympify
            sym_test_atoms  = {k:[syme_sympify(SymbolicExpr(ai)) for ai in a] for k,a in test_atoms.items()}
            sym_trial_atoms = {k:[syme_sympify(SymbolicExpr(ai)) for ai in a] for k,a in trial_atoms.items()}
            sym_expr        = syme_sympify(SymbolicExpr(texpr.expr))
        else:
            sym_test_atoms  = {k:[SymbolicExpr(ai) for ai in a] for k,a in test_atoms.items()}
            sym_trial_atoms = {k:[SymbolicExpr(ai) for ai in a] for k,a in trial_atoms.items()}
            sym_expr        = SymbolicExpr(texpr.expr)
        if verbose: print(f'In readBF: sym_test_atoms: {sym_test_atoms}')
        if verbose: print(f'In readBF: sym_trial_atoms: {sym_trial_atoms}')

        trials_subs = {ui:0 for u in sym_trial_atoms for ui in sym_trial_atoms[u]}
        tests_subs  = {vi:0 for v in sym_test_atoms  for vi in sym_test_atoms[v]}
        sub_exprs   = {}

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

        element_indices    = [Symbol('k_{}'.format(i)) for i in range(2,4)]
        quadrature_indices = [Symbol('q_{}'.format(i)) for i in range(2,4)]
        indices = tuple(j for i in zip(element_indices, quadrature_indices) for j in i)

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
        if isinstance(expr, (ImmutableDenseMatrix, Matrix)):
            g_mat_information_false = []
            shape = expr.shape
            for k1 in range(shape[0]):
                for k2 in range(shape[1]):
                    if not expr[k1,k2].is_zero:
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

        '''
        1, 1: expr[1,1] = F0*sqrt(x1**2*(x1*cos(2*pi*x3) + 2)**2*(sin(pi*x2)**2 + cos(pi*x2)**2)**2*(sin(2*pi*x3)**2 + cos(2*pi*x3)**2)**2)*(pi*(x1*cos(2*pi*x3) + 2)*
        (-2*pi*x1*sin(pi*x2)*sin(2*pi*x3)*dx1(v1[1]) - sin(pi*x2)*cos(2*pi*x3)*dx3(v1[1]))*cos(pi*x2)*w2[1] - pi*(x1*cos(2*pi*x3) + 2)*(-2*pi*x1*sin(2*pi*x3)*cos(pi*x2)*dx1(v1[1]) - 
        cos(pi*x2)*cos(2*pi*x3)*dx3(v1[1]))*sin(pi*x2)*w2[1])/(2*pi**2*x1**2*(x1*cos(2*pi*x3) + 2)**2*(sin(pi*x2)**2 + cos(pi*x2)**2)**2*(sin(2*pi*x3)**2 + cos(2*pi*x3)**2)**2)
        = 0 - but is not yet detected as 0! Hence a matrix is generated, that later is not required!
        '''

        if nv > 1:
            ct_str = 'coupling_terms_u_{u_i}_v_{v_j}' if nu > 1 else 'coupling_terms_u_v_{v_j}'
        else:
            ct_str = 'coupling_terms_u_{u_i}_v' if nu > 1 else 'coupling_terms_u_v'

        lhs = {}
        for block in blocks:
            u_i = get_atom_logical_derivatives(block[0]).indices[0] if nu > 1 else 0
            v_j = get_atom_logical_derivatives(block[1]).indices[0] if nv > 1 else 0
            ct = ct_str.format(u_i=u_i, v_j=v_j)
            lhs[block] = IndexedBase(f'{ct}')
        
        counts = {block:0 for block in blocks}

        for r,key in zip(rhs, sub_exprs.keys()):
            u_i, v_j = [get_atom_logical_derivatives(atom) for atom in key]
            count = counts[u_i, v_j]
            counts[u_i, v_j] += 1
            ordered_stmts[u_i, v_j].append(Assign(lhs[u_i, v_j][(*indices, count)], r))
            ordered_sub_exprs_keys[u_i, v_j].append(key)

        temps = tuple(Assign(a,b) for a,b in temps)

        return temps, ordered_stmts, ordered_sub_exprs_keys, mapping_option, field_derivatives, g_mat_information_false, g_mat_information_true

    def construct_arguments_generate_assembly_file(self):
        """
        Collect the arguments used in the assembly method.

        Parameters # no openmp support for now
        ----------
        #with_openmp : bool
        # If set to True we collect some extra arguments used in the assembly method

        Returns
        -------
        
        args: tuple
         The arguments passed to the assembly method.

        threads_args: None # for now, used to be tuple
          #Extra arguments used in the assembly method in case with_openmp=True.

        """
        verbose = False

        temps, ordered_stmts, ordered_sub_exprs_keys, mapping_option, field_derivatives, g_mat_information_false, g_mat_information_true = self.read_BilinearForm()

        blocks              = ordered_stmts.keys()
        block_list          = list(blocks)
        trial_components    = [block[0] for block in block_list]
        test_components     = [block[1] for block in block_list]
        trial_dim           = len(set(trial_components))
        test_dim            = len(set(test_components))
        
        if verbose: print(f'blocks: {blocks}')
        if verbose: print(f'block_list: {block_list}')
        if verbose: print(f'trial_components: {trial_components}')
        if verbose: print(f'test_components: {test_components}')
        if verbose: print(f'trial_dim: {trial_dim}')
        if verbose: print(f'test_dim: {test_dim}')

        d = 3
        assert d == 3 # dim 3 assembly method
        nu = trial_dim # dim of trial function; 1 (scalar) or 3 (vector)
        nv = test_dim  # dim of trial function; 1 (scalar) or 3 (vector)

        test_basis, test_degrees, spans, pads, mult = construct_test_space_arguments(self.test_basis)
        trial_basis, trial_degrees, pads, mult      = construct_trial_space_arguments(self.trial_basis)
        n_elements, quads, quad_degrees             = construct_quad_grids_arguments(self.grid[0], use_weights=False)

        pads = self.test_basis.space.vector_space.pads

        n_element_1, n_element_2, n_element_3   = n_elements
        k1, k2, k3                              = quad_degrees

        if (nu == 3) and (len(trial_basis) == 3):
            # VectorFunction not belonging to a de Rham sequence - 3 instead of 9 variables in trial/test_degrees and trial/test_basis
            trial_u_p   = {u:trial_degrees for u in range(nu)}
            global_basis_u  = {u:trial_basis    for u in range(nu)}
        else:
            trial_u_p       = {u:trial_degrees[d*u:d*(u+1)] for u in range(nu)}
            global_basis_u  = {u:trial_basis[d*u:d*(u+1)]    for u in range(nu)}
        if (nv == 3) and (len(test_basis) == 3):
            test_v_p   = {v:test_degrees for v in range(nv)}
            global_basis_v  = {v:test_basis   for v in range(nv)}
            spans = [*spans, *spans, *spans]
        else:
            test_v_p       = {v:test_degrees[d*v:d*(v+1)] for v in range(nv)}
            global_basis_v  = {v:test_basis[d*v:d*(v+1)]   for v in range(nv)}

        assert len(self.grid) == 1, f'len(self.grid) is supposed to be 1 for now'

        # When self._target is an Interface domain len(self._grid) == 2
        # where grid contains the QuadratureGrid of both sides of the interface
        if self.mapping:

            map_coeffs = [[e._coeffs._data for e in self.mapping._fields]]
            spaces     = [self.mapping._fields[0].space]
            map_degree = [sp.degree for sp in spaces]
            map_span   = [[q.spans - s for q,s in zip(sp.get_assembly_grids(*self.nquads), sp.vector_space.starts)] for sp in spaces]
            map_basis  = [[q.basis for q in sp.get_assembly_grids(*self.nquads)] for sp in spaces]
            points     = [g.points for g in self.grid]
            weights    = [self.mapping.weights_field.coeffs._data] if self.is_rational_mapping else []

            for i in range(len(self.grid)):
                axis   = self.grid[i].axis
                if axis is not None:
                    raise ValueError(f'axis is supposed to be None for now!')

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

        #--------------------

        x1_trial_keys = {block:[] for block in blocks}
        x1_test_keys  = {block:[] for block in blocks}
        x2_trial_keys = {block:[] for block in blocks}
        x2_test_keys  = {block:[] for block in blocks}
        x3_trial_keys = {block:[] for block in blocks}
        x3_test_keys  = {block:[] for block in blocks}

        for block in blocks:
            for alpha, beta in ordered_sub_exprs_keys[block]:
                x1_trial_keys[block].append(get_index_logical_derivatives(alpha)['x1'])
                x1_test_keys [block].append(get_index_logical_derivatives(beta) ['x1'])
                x2_trial_keys[block].append(get_index_logical_derivatives(alpha)['x2'])
                x2_test_keys [block].append(get_index_logical_derivatives(beta) ['x2'])
                x3_trial_keys[block].append(get_index_logical_derivatives(alpha)['x3'])
                x3_test_keys [block].append(get_index_logical_derivatives(beta) ['x3'])

        a3 = {}
        a2 = {}
        coupling_terms = {}
        test_trial_1s = {}
        test_trial_2s = {}
        test_trial_3s = {}
        keys_1 = {}
        keys_2 = {}
        keys_3 = {}

        for block in blocks:
            u_i = block[0].indices[0] if nu > 1 else 0
            v_j = block[1].indices[0] if nv > 1 else 0
            
            keys_1[block] = np.array([(alpha_1, beta_1) for alpha_1, beta_1 in zip(x1_trial_keys[block], x1_test_keys[block])])
            keys_2[block] = np.array([(alpha_2, beta_2) for alpha_2, beta_2 in zip(x2_trial_keys[block], x2_test_keys[block])])
            keys_3[block] = np.array([(alpha_3, beta_3) for alpha_3, beta_3 in zip(x3_trial_keys[block], x3_test_keys[block])])

            global_basis_u_1, global_basis_u_2, global_basis_u_3 = global_basis_u[u_i]
            global_basis_v_1, global_basis_v_2, global_basis_v_3 = global_basis_v[v_j]

            trial_u_p1, trial_u_p2, trial_u_p3 = trial_u_p[u_i]
            test_v_p1,  test_v_p2,  test_v_p3  = test_v_p [v_j]
            
            max_p_2 = max(test_v_p2, trial_u_p2)
            max_p_3 = max(test_v_p3, trial_u_p3)

            n_expr = len(ordered_stmts[block])

            test_trial_1 = np.zeros((n_element_1, k1, test_v_p1 + 1, trial_u_p1 + 1, 2, 2), dtype='float64')
            test_trial_2 = np.zeros((n_element_2, k2, test_v_p2 + 1, trial_u_p2 + 1, 2, 2), dtype='float64')
            test_trial_3 = np.zeros((n_element_3, k3, test_v_p3 + 1, trial_u_p3 + 1, 2, 2), dtype='float64')

            for k_1 in range(n_element_1):
                for q_1 in range(k1):
                    for i_1 in range(test_v_p1 + 1):
                        for j_1 in range(trial_u_p1 + 1):
                            trial   = global_basis_u_1[k_1, j_1, :, q_1]
                            test    = global_basis_v_1[k_1, i_1, :, q_1]
                            for alpha_1 in range(2):
                                for beta_1 in range(2):
                                    test_trial_1[k_1, q_1, i_1, j_1, alpha_1, beta_1] = trial[alpha_1] * test[beta_1]

            for k_2 in range(n_element_2):
                for q_2 in range(k2):
                    for i_2 in range(test_v_p2 + 1):
                        for j_2 in range(trial_u_p2 + 1):
                            trial   = global_basis_u_2[k_2, j_2, :, q_2]
                            test    = global_basis_v_2[k_2, i_2, :, q_2]
                            for alpha_2 in range(2):
                                for beta_2 in range(2):
                                    test_trial_2[k_2, q_2, i_2, j_2, alpha_2, beta_2] = trial[alpha_2] * test[beta_2]

            for k_3 in range(n_element_3):
                for q_3 in range(k3):
                    for i_3 in range(test_v_p3 + 1):
                        for j_3 in range(trial_u_p3 + 1):
                            trial   = global_basis_u_3[k_3, j_3, :, q_3]
                            test    = global_basis_v_3[k_3, i_3, :, q_3]
                            for alpha_3 in range(2):
                                for beta_3 in range(2):
                                    test_trial_3[k_3, q_3, i_3, j_3, alpha_3, beta_3] = trial[alpha_3] * test[beta_3]

            test_trial_1s[block] = test_trial_1
            test_trial_2s[block] = test_trial_2
            test_trial_3s[block] = test_trial_3

            a3[block] = np.zeros((n_expr, n_element_3 + test_v_p3 + (mult[2]-1)*(n_element_3-1), 2 * max_p_3 + 1), dtype='float64')
            a2[block] = np.zeros((n_expr, n_element_2 + test_v_p2 + (mult[1]-1)*(n_element_2-1), n_element_3 + test_v_p3 + (mult[2]-1)*(n_element_3-1), 2 * max_p_2 + 1, 2 * max_p_3 + 1), dtype='float64')
            coupling_terms[block] = np.zeros((n_element_2, k2, n_element_3, k3, n_expr), dtype='float64')

        new_args = (*list(test_trial_1s.values()), 
                    *list(test_trial_2s.values()), 
                    *list(test_trial_3s.values()), 
                    *list(a3.values()),
                    *list(a2.values()),
                    *list(coupling_terms.values()))
        
        expr = self.kernel_expr.expr
        if isinstance(expr, (ImmutableDenseMatrix, Matrix)):
            matrices = []
            for i, block in enumerate(g_mat_information_false):
                if block in g_mat_information_true:
                    matrices.append(self._global_matrices[i])
        else:
            matrices = self._global_matrices

        args = (*map_basis, *spans, *map_span, *quads, *map_degree, *n_elements, *quad_degrees, *pads, *mapping, *matrices,
                *new_args)
        
        threads_args = ()

        args = tuple(np.int64(a) if isinstance(a, int) else a for a in args)
        threads_args = tuple(np.int64(a) if isinstance(a, int) else a for a in threads_args)

        id = self.make_file(temps, ordered_stmts, field_derivatives, mult, test_v_p, trial_u_p, keys_1, keys_2, keys_3, mapping_option=mapping_option)
        #from psydac.api.tests.multiplicity_issue_copy import turn_off_pyccel
        #if turn_off_pyccel:
        #    from __psydac__.assemble import assemble_matrix
        #    self._func = assemble_matrix
        #else:
        #    from __psydac__.assemble import assemble_matrix
        #    from pyccel.epyccel import epyccel
        #    new_func = epyccel(assemble_matrix, language='fortran')
        #    self._func = new_func

        package = importlib.import_module(f'__psydac__.assemble_{id}')

        # print("\n\n" + "#" + "-"*78)
        # print("Source code of the generated assembly method:")
        # print("#" + "-"*78 + "\n")
        # print(inspect.getsource(package.assemble_matrix))

        new_func = epyccel(package.assemble_matrix, language='fortran')
        self._func = new_func

        return args, threads_args
        
    def construct_arguments(self, with_openmp=False):
        """
        Collect the arguments used in the assembly method.

        Parameters
        ----------
        with_openmp : bool
         If set to True we collect some extra arguments used in the assembly method

        Returns
        -------
        
        args: tuple
         The arguments passed to the assembly method.

        threads_args: tuple
          Extra arguments used in the assembly method in case with_openmp=True.

        """
        test_basis, test_degrees, spans, pads, mult = construct_test_space_arguments(self.test_basis)
        trial_basis, trial_degrees, pads, mult      = construct_trial_space_arguments(self.trial_basis)
        n_elements, quads, quad_degrees             = construct_quad_grids_arguments(self.grid[0], use_weights=False)
        if len(self.grid)>1:
            quads  = [*quads, *self.grid[1].points]

        pads = self.test_basis.space.vector_space.pads

        # When self._target is an Interface domain len(self._grid) == 2
        # where grid contains the QuadratureGrid of both sides of the interface
        if self.mapping:

            if len(self.grid) == 1:
                map_coeffs = [[e._coeffs._data for e in self.mapping._fields]]
                spaces     = [self.mapping._fields[0].space]
                map_degree = [sp.degree for sp in spaces]
                map_span   = [[q.spans - s for q,s in zip(sp.get_assembly_grids(*self.nquads), sp.vector_space.starts)] for sp in spaces]
                map_basis  = [[q.basis for q in sp.get_assembly_grids(*self.nquads)] for sp in spaces]
                points     = [g.points for g in self.grid]
                weights    = [self.mapping.weights_field.coeffs._data] if self.is_rational_mapping else []

            elif len(self.grid) == 2:
                target = self.kernel_expr.target
                assert isinstance(target, Interface)
                mappings = list(self.mapping)
                i, j = self.get_space_indices_from_target(self.domain, target)
                m, _ = self.get_space_indices_from_target(self.domain, target.minus)
                p,_ = self.get_space_indices_from_target(self.domain, target.plus)

                map_coeffs = [[e._coeffs for e in mapping._fields] for mapping in self.mapping]
                spaces     = [mapping._fields[0].space for mapping in self.mapping]
                weights_m  = [mappings[0].weights_field.coeffs] if self.is_rational_mapping[0] else []
                weights_p  = [mappings[1].weights_field.coeffs] if self.is_rational_mapping[1] else []
                if m == j:
                    axis = target.minus.axis
                    ext  = target.minus.ext

                    spaces[0]     = spaces[0].interfaces[axis, ext]
                    map_coeffs[0] = [coeff._interface_data[axis, ext] for coeff in map_coeffs[0]]
                    map_coeffs[1] = [coeff._data for coeff in map_coeffs[1]]
                    if weights_m:
                        weights_m[0] = weights_m[0]._interface_data[axis, ext]
                    if weights_p:
                        weights_p[0] = weights_p[0]._data
                elif p == j:
                    axis = target.plus.axis
                    ext  = target.plus.ext

                    spaces[1]     = spaces[1].interfaces[axis, ext]
                    map_coeffs[0] = [coeff._data for coeff in map_coeffs[0]]
                    map_coeffs[1] = [coeff._interface_data[axis, ext] for coeff in map_coeffs[1]]
                    if weights_m:
                        weights_m[0] = weights_m[0]._data
                    if weights_p:
                        weights_p[0] = weights_p[0]._interface_data[axis, ext]

                map_degree = [sp.degree for sp in spaces]
                map_span   = [[q.spans - s for q, s in zip(sp.get_assembly_grids(*self.nquads), sp.vector_space.starts)] for sp in spaces]
                map_basis  = [[q.basis for q in sp.get_assembly_grids(*self.nquads)] for sp in spaces]
                points     = [g.points for g in self.grid]

            nderiv = self.max_nderiv
            for i in range(len(self.grid)):
                axis   = self.grid[i].axis
                ext    = self.grid[i].ext
                if axis is None:continue
                space  = spaces[i].spaces[axis]
                points_i = points[i][axis]
                local_span = find_span(space.knots, space.degree, points_i[0, 0])
                boundary_basis = basis_funs_all_ders(space.knots, space.degree,
                                                     points_i[0, 0], local_span, nderiv, space.basis)
                map_basis[i][axis] = map_basis[i][axis].copy()
                map_basis[i][axis][0, :, 0:nderiv+1, 0] = np.transpose(boundary_basis)
                if ext == 1:
                    map_span[i][axis]    = map_span[i][axis].copy()
                    map_span[i][axis][0] = map_span[i][axis][-1]

            map_degree = flatten(map_degree)
            map_span   = flatten(map_span)
            map_basis  = flatten(map_basis)
            points     = flatten(points)
            if len(self.grid) == 1:
                mapping = [*map_coeffs[0], *weights]
            elif len(self.grid)==2:
                mapping   = [*map_coeffs[0], *weights_m, *map_coeffs[1], *weights_p]
        else:
            mapping    = []
            map_degree = []
            map_span   = []
            map_basis  = []

        args = (*test_basis, *trial_basis, *map_basis, *spans, *map_span, *quads, *test_degrees, *trial_degrees, *map_degree, 
                *n_elements, *quad_degrees, *pads, *mapping, *self._global_matrices)

        with_openmp  = with_openmp and self._num_threads>1

        threads_args = ()
        if with_openmp:
            threads_args = self._vector_space.cart.get_shared_memory_subdivision(n_elements)
            threads_args = (threads_args[0], threads_args[1], *threads_args[2], *threads_args[3], threads_args[4])

        args = tuple(np.int64(a) if isinstance(a, int) else a for a in args)
        threads_args = tuple(np.int64(a) if isinstance(a, int) else a for a in threads_args)

        return args, threads_args

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
        test_space      = self.spaces[1].vector_space
        trial_space     = self.spaces[0].vector_space
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
            pads = test_degree

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
                    mat = BlockLinearOperator(trial_fem_space.get_refined_space(ncells).vector_space, test_fem_space.get_refined_space(ncells).vector_space)
                    if not is_conformal and not i==j:
                        if use_restriction:
                            Ps  = [knot_insertion_projection_operator(ts.get_refined_space(ncells), ts) for ts in test_fem_space.spaces]
                            P   = BlockLinearOperator(test_fem_space.get_refined_space(ncells).vector_space, test_fem_space.vector_space)
                            for ni,Pi in enumerate(Ps):
                                P[ni,ni] = Pi

                            mat = ComposedLinearOperator(trial_space, test_space, P, mat)

                        elif use_prolongation:
                            Ps  = [knot_insertion_projection_operator(trs, trs.get_refined_space(ncells)) for trs in trial_fem_space.spaces]
                            P   = BlockLinearOperator(trial_fem_space.vector_space, trial_fem_space.get_refined_space(ncells).vector_space)
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
                        ts_space = test_fem_space.get_refined_space(ncells).vector_space.spaces[k1]
                    else:
                        ts_space = test_fem_space.get_refined_space(ncells).vector_space

                    if isinstance(trial_fem_space, VectorFemSpace):
                        tr_space = trial_fem_space.get_refined_space(ncells).vector_space.spaces[k2]
                    else:
                        tr_space = trial_fem_space.get_refined_space(ncells).vector_space

                    if is_conformal and matrix[k1, k2]:
                        global_mats[k1, k2] = matrix[k1, k2]
                    elif not i == j: # assembling in an interface (type(target) == Interface)
                        axis    = target.axis
                        ext_d   = self._trial_ext
                        ext_c   = self._test_ext
                        test_n  = self. test_basis.space.spaces[k1].spaces[axis].nbasis
                        test_s  = self. test_basis.space.spaces[k1].vector_space.starts[axis]
                        trial_n = self.trial_basis.space.spaces[k2].spaces[axis].nbasis
                        cart    = self.trial_basis.space.spaces[k2].vector_space.cart
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
                    test_s  = self.test_basis.space.vector_space.starts[axis]
                    trial_n = self.trial_basis.space.spaces[axis].nbasis
                    cart    = self.trial_basis.space.vector_space.cart
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
                        mat = StencilInterfaceMatrix(trial_fem_space.get_refined_space(ncells).vector_space,
                                                     test_fem_space.get_refined_space(ncells).vector_space,
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


# ==============================================================================
class DiscreteSesquilinearForm(DiscreteBilinearForm):
    """ Class that represents the concept of a discrete sesqui-linear form with the antilinearity on the first variable.
        This class allocates the matrix and generates the matrix assembly method.

    Parameters
    ----------

    expr : sympde.expr.expr.SesquilinearForm
        The symbolic sesqui-linear form.

    kernel_expr : sympde.expr.evaluation.KernelExpression
        The atomic representation of the sesqui-linear form.

    domain_h : Geometry
        The discretized domain

    spaces: list of FemSpace
        The trial and test discrete spaces.

    nquads : list or tuple of int
        The number of quadrature points used in the low-level assembly function
        along each direction.

    matrix: Matrix
        The matrix that we assemble into it.
        If not provided, it will create a new Matrix of the appropriate space.

    update_ghost_regions: bool
        Accumulate the contributions of the neighbouring processes.

    backend: dict
        The backend used to accelerate the computing kernels.
        The backend dictionaries are defined in the file psydac/api/settings.py

    assembly_backend: dict
        The backend used to accelerate the assembly method.
        The backend dictionaries are defined in the file psydac/api/settings.py

    linalg_backend: dict
        The backend used to accelerate the computing kernels of the linear operator.
        The backend dictionaries are defined in the file psydac/api/settings.py

    symbolic_mapping: Sympde.topology.Mapping
        The symbolic mapping which defines the physical domain of the sesqui-linear form.

    """


#==============================================================================
class DiscreteLinearForm(BasicDiscrete):
    """
    Discrete linear form ready to be assembled into a vector.

    This class represents the concept of a discrete linear form in Psydac.
    Instances of this class generate an appropriate vector assembly kernel,
    allocate the vector if not provided, and prepare a list of arguments for
    the kernel.

    Parameters
    ----------

    expr : sympde.expr.expr.LinearForm
        The symbolic linear form.

    kernel_expr : sympde.expr.evaluation.KernelExpression
        The atomic representation of the linear form.

    domain_h : Geometry
        The discretized domain.

    space : FemSpace
        The discrete test space.

    nquads : list or tuple of int
        The number of quadrature points used in the assembly kernel along each
        direction.

    vector : StencilVector or BlockVector, optional
        The vector that we assemble into. If not provided, a new vector of the
        appropriate space is created.

    update_ghost_regions : bool, default=False
        Accumulate the contributions of the neighbouring processes.

    backend : dict, optional
        The backend used to accelerate the computing kernels.
        The backend dictionaries are defined in the file psydac/api/settings.py

    symbolic_mapping : Sympde.topology.Mapping, optional
        The symbolic mapping which defines the physical domain of the linear form.

    See Also
    --------
    DiscreteBilinearForm
    DiscreteFunctional
    DiscreteSumForm

    """
    def __init__(self, expr, kernel_expr, domain_h, space, *, nquads,
                 vector=None, update_ghost_regions=True, backend=None,
                 symbolic_mapping=None):

        if not isinstance(expr, sym_LinearForm):
            raise TypeError('> Expecting a symbolic LinearForm')

        assert isinstance(domain_h, Geometry)

        self._space = space

        if isinstance(kernel_expr, (tuple, list)):
            if len(kernel_expr) == 1:
                kernel_expr = kernel_expr[0]
            else:
                raise ValueError('> Expecting only one kernel')

        # ...
        self._kernel_expr = kernel_expr
        self._target      = kernel_expr.target
        self._domain      = domain_h.domain
        self._vector      = vector

        domain = self.domain
        target = self.target

        if len(domain) > 1:
            i = self.get_space_indices_from_target(domain, target)
            test_space  = self._space.spaces[i]
            mapping = list(domain_h.mappings.values())[i]
        else:
            test_space  = self._space
            mapping = list(domain_h.mappings.values())[0]

        if isinstance(test_space.vector_space, BlockVectorSpace):
            vector_space = test_space.vector_space.spaces[0]
            if isinstance(vector_space, BlockVectorSpace):
                vector_space = vector_space.spaces[0]
        else:
            vector_space = test_space.vector_space

        self._mapping      = mapping
        self._vector_space = vector_space
        self._num_threads  = 1
        if vector_space.parallel and vector_space.cart.num_threads>1:
            self._num_threads = vector_space.cart._num_threads

        self._update_ghost_regions = update_ghost_regions

        # In case of multiple patches, if the communicator is MPI_COMM_NULL or the cart is an Interface cart,
        # we do not generate the assembly code, because the patch is not owned by the MPI rank.
        if vector_space.parallel and (vector_space.cart.is_comm_null or isinstance(vector_space.cart, InterfaceCartDecomposition)):
            self._free_args = ()
            self._func      = do_nothing
            self._args      = ()
            self._threads_args     = ()
            self._global_matrices  = ()
            self._update_ghost_regions = False
            return

        if mapping is not None:
            is_rational_mapping = isinstance(mapping, NurbsMapping)
            mapping_space = mapping.space
        else:
            is_rational_mapping = False
            mapping_space = None

        self._is_rational_mapping = is_rational_mapping
        discrete_space            = test_space

        # MPI communicator
        comm = vector_space.cart.comm if vector_space.parallel else None

        # BasicDiscrete generates the assembly code and sets the following attributes that are used afterwards:
        # self._func, self._free_args, self._max_nderiv and self._backend
        BasicDiscrete.__init__(self, expr, kernel_expr, comm=comm, root=0, discrete_space=discrete_space,
                              nquads=nquads, is_rational_mapping=is_rational_mapping, mapping=symbolic_mapping,
                              mapping_space=mapping_space, num_threads=self._num_threads, backend=backend)

        #... Handle the special case where the current MPI process does not need to do anything
        if not isinstance(target, Boundary):
            ext  = None
            axis = None
        else:
            ext  = target.ext
            axis = target.axis

            # Assuming that all vector spaces (and their Cartesian decomposition,
            # if any) are compatible with each other, extract the first available
            # vector space from which (starts, ends, pads) will be read:
            # If process does not own the boundary or interface, do not assemble anything
            if ext == -1:
                start = vector_space.starts[axis]
                if start != 0:
                    self._func = do_nothing

            elif ext == 1:
                end  = vector_space.ends[axis]
                npts = vector_space.npts[axis]
                if end + 1 != npts:
                    self._func = do_nothing
        #...

        # Build the quadrature grids
        test_grid  = QuadratureGrid(test_space, axis=axis, ext=ext, nquads=nquads)
        self._grid = test_grid

        # Extract the basis function values on the quadrature grid
        self._test_basis = BasisValues(
            test_space,
            nderiv = self.max_nderiv,
            nquads = nquads,
            grid   = test_grid
        )

        # Allocate the output vector, if needed
        self.allocate_matrices()

        # Determine whether OpenMP instructions were generated
        with_openmp = (backend['name'] == 'pyccel' and backend['openmp']) if backend else False

        # Construct the arguments to be passed to the assemble() function, which is stored in self._func
        self._args, self._threads_args = self.construct_arguments(with_openmp=with_openmp)

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def space(self):
        return self._space

    @property
    def grid(self):
        return self._grid

    @property
    def nquads(self):
        return self._grid.nquads

    @property
    def test_basis(self):
        return self._test_basis

    @property
    def global_matrices(self):
        return self._global_matrices

    @property
    def args(self):
        return self._args

    def assemble(self, *, reset=True, **kwargs):
        """
        This method assembles the right-hand side Vector by calling the private method `self._func` with proper arguments.

        In the complex case, this function returns the vector conjugate. This comes from the fact that the
        problem `a(u,v)=b(v)` is discretize as `A @ conj(U) = B` due to the antilinearity of `a` in the first variable.
        Thus, to obtain `U`, the assemble function for the LinearForm return `conj(B)`.

        TODO: remove these lines when the dot product is changed for complex in sympde.
        For now, since the dot product does not do the conjugate in the complex case, we do not use the conjugate in the assemble function.
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
                if len(self.domain) > 1 and isinstance(v, FemField) and v.space.is_product:
                    i = self.get_space_indices_from_target(self.domain, self.target)
                    v = v[i]
                if isinstance(v, FemField):
                    if not v.coeffs.ghost_regions_in_sync:
                        v.coeffs.update_ghost_regions()
                    basis_v  = BasisValues(
                        v.space,
                        nderiv = self.max_nderiv,
                        nquads = self.nquads,
                        trial  = True,
                        grid   = self.grid
                    )
                    bs, d, s, p, m = construct_test_space_arguments(basis_v)
                    basis   += bs
                    spans   += s
                    degrees += [np.int64(a) for a in d]
                    pads    += [np.int64(a) for a in p]
                    if v.space.is_product:
                        coeffs += (e._data for e in v.coeffs)
                    else:
                        coeffs += (v.coeffs._data,)
                else:
                    consts += (v,)

            args = (*self.args, *basis, *spans, *degrees, *pads, *coeffs, *consts)

        else:
            args = self._args

        if reset:
            reset_arrays(*self.global_matrices)

        self._func(*args, *self._threads_args)
        if self._vector and self._update_ghost_regions:
            self._vector.exchange_assembly_data()

        # TODO : uncomment this line when the conjugate is applied on the dot product in the complex case
        # self._vector.conjugate(out=self._vector)

        if self._vector:
            self._vector.ghost_regions_in_sync = False

        return self._vector

    def get_space_indices_from_target(self, domain, target):
        if domain.mapping:
            domain = domain.logical_domain
        if target.mapping:
            target = target.logical_domain

        domains = domain.interior.args

        if isinstance(target, Interface):
            raise NotImplementedError("Index of an interface is not defined for the LinearForm")
        elif isinstance(target, Boundary):
            i = domains.index(target.domain)
        else:
            i = domains.index(target)
        return i

    def construct_arguments(self, with_openmp=False):
        """
        Collect the arguments used in the assembly method.

        Parameters
        ----------
        with_openmp : bool
         If set to True we collect some extra arguments used in the assembly method

        Returns
        -------
        
        args: tuple
         The arguments passed to the assembly method.

        threads_args: tuple
          Extra arguments used in the assembly method in case with_openmp=True.

        """
        tests_basis, tests_degrees, spans, pads, mult = construct_test_space_arguments(self.test_basis)
        n_elements, quads, nquads               = construct_quad_grids_arguments(self.grid, use_weights=False)

        global_pads   = self.space.vector_space.pads

        if self.mapping:
            mapping    = [e._coeffs._data for e in self.mapping._fields]
            space      = self.mapping._fields[0].space
            map_degree = space.degree
            map_span   = [q.spans - s for q, s in zip(space.get_assembly_grids(*self.nquads), space.vector_space.starts)]
            map_basis  = [q.basis for q in space.get_assembly_grids(*self.nquads)]
            axis       = self.grid.axis
            ext        = self.grid.ext
            points     = self.grid.points
            if axis is not None:
                nderiv = self.max_nderiv
                space  = space.spaces[axis]
                points = points[axis]
                local_span = find_span(space.knots, space.degree, points[0, 0])
                boundary_basis = basis_funs_all_ders(space.knots, space.degree,
                                                     points[0, 0], local_span, nderiv, space.basis)
                map_basis[axis] = map_basis[axis].copy()
                map_basis[axis][0, :, 0:nderiv+1, 0] = np.transpose(boundary_basis)
                if ext == 1:
                    map_span[axis]    = map_span[axis].copy()
                    map_span[axis][0] = map_span[axis][-1]
            if self.is_rational_mapping:
                mapping = [*mapping, self.mapping.weights_field.coeffs._data]
        else:
            mapping    = []
            map_degree = []
            map_span   = []
            map_basis  = []

        args = (*tests_basis, *map_basis, *spans, *map_span, *quads, *tests_degrees, *map_degree, *n_elements, *nquads, *global_pads, *mapping, *self._global_matrices)

        with_openmp  = with_openmp and self._num_threads>1

        threads_args = ()
        if with_openmp:
            threads_args = self._vector_space.cart.get_shared_memory_subdivision(n_elements)
            threads_args = (threads_args[0], threads_args[1], *threads_args[2], *threads_args[3], threads_args[4])

        args = tuple(np.int64(a) if isinstance(a, int) else a for a in args)
        threads_args = tuple(np.int64(a) if isinstance(a, int) else a for a in threads_args)

        return args, threads_args

    def allocate_matrices(self):
        """
        Allocate the global matrices used in the assembly method.
        In this method we allocate only the matrices that are computed in the self._target domain,
        we also avoid double allocation if we have many DiscreteLinearForm that are defined on the same self._target domain.
        """
        global_mats   = {}

        test_space  = self.test_basis.space.vector_space
        test_degree = np.array(self.test_basis.space.degree)

        expr        = self.kernel_expr.expr
        target      = self.kernel_expr.target
        domain      = self.domain
        is_broken   = len(domain) > 1

        if self._vector is None and (is_broken or isinstance(expr, (ImmutableDenseMatrix, Matrix))):
            self._vector = BlockVector(self.space.vector_space)

        if isinstance(expr, (ImmutableDenseMatrix, Matrix)): # case system of equations

            if is_broken: #multi patch
                i = self.get_space_indices_from_target(domain, target)
                if not self._vector[i]:
                    self._vector[i] = BlockVector(test_space)
                vector = self._vector[i]
            else: # single patch
                vector = self._vector

            expr = expr[:]
            for i in range(len(expr)):
                if expr[i].is_zero:
                    continue
                else:
                    if vector[i]:
                        global_mats[i] = vector[i]
                    else:
                        global_mats[i] = StencilVector(test_space.spaces[i])

                vector[i] = global_mats[i]
        else:
            if is_broken:
                i = self.get_space_indices_from_target(domain, target)
                if self._vector[i]:
                    global_mats[i] = self._vector[i]
                else:
                    global_mats[i] = StencilVector(test_space)

                self._vector[i] = global_mats[i]
            else:
                if self._vector:
                    global_mats[0] = self._vector
                else:
                    global_mats[0] = StencilVector(test_space)
                    self._vector   = global_mats[0]

        self._global_matrices = [M._data for M in global_mats.values()]


#==============================================================================
# NOTE: why do we pass a FemSpace to the constructor?
class DiscreteFunctional(BasicDiscrete):
    """
    Discrete functional ready to be assembled into a scalar (real or complex).

    This class represents the concept of a discrete functional in Psydac.
    Instances of this class generate an appropriate functional assembly kernel,
    and prepare a list of arguments for the kernel.

    Parameters
    ----------

    expr : sympde.expr.expr.Functional
        The symbolic functional form.

    kernel_expr : sympde.expr.evaluation.KernelExpression
        The atomic representation of the functional form.

    domain_h : Geometry
        The discretized domain.

    space : FemSpace
        The discrete space.

    nquads : list or tuple of int
        The number of quadrature points used in the assembly kernel along each
        direction.

    update_ghost_regions : bool, default=True
        Accumulate the contributions of the neighbouring processes.

    backend : dict
        The backend used to accelerate the computing kernels.
        The backend dictionaries are defined in the file psydac/api/settings.py

    symbolic_mapping : Sympde.topology.Mapping
        The symbolic mapping which defines the physical domain of the functional.

    See Also
    --------
    DiscreteBilinearForm
    DiscreteLinearForm
    DiscreteSumForm

    """
    def __init__(self, expr, kernel_expr, domain_h, space, *, nquads,
                 backend=None, symbolic_mapping=None):

        if not isinstance(expr, sym_Functional):
            raise TypeError('> Expecting a symbolic Functional')

        # ...
        assert isinstance(domain_h, Geometry)

        self._space = space

        if isinstance(kernel_expr, (tuple, list)):
            if len(kernel_expr) == 1:
                kernel_expr = kernel_expr[0]
            else:
                raise ValueError('> Expecting only one kernel')

        # ...
        self._kernel_expr     = kernel_expr
        self._target          = kernel_expr.target
        self._symbolic_space  = self._space.symbolic_space
        self._domain          = domain_h.domain
        # ...

        domain = self.domain
        target = self.target

        if len(domain) > 1:
            i = self.get_space_indices_from_target(domain, target)
            self._space = self._space.spaces[i]
            mapping = list(domain_h.mappings.values())[i]
        else:
            mapping = list(domain_h.mappings.values())[0]

        if isinstance(self.space.vector_space, BlockVectorSpace):
            vector_space = self.space.vector_space.spaces[0]
            if isinstance(vector_space, BlockVectorSpace):
                vector_space = vector_space.spaces[0]
        else:
            vector_space = self.space.vector_space

        num_threads = 1
        if vector_space.parallel and vector_space.cart.num_threads > 1:
            num_threads = vector_space.cart._num_threads

        # In case of multiple patches, if the communicator is MPI_COMM_NULL, we do not generate the assembly code
        # because the patch is not owned by the MPI rank.
        if vector_space.parallel and vector_space.cart.is_comm_null:
            self._free_args = ()
            self._func      = do_nothing
            self._args      = ()
            self._expr      = expr
            self._comm      = domain_h.comm
            return

        if isinstance(target, Boundary):
            ext  = target.ext
            axis = target.axis
        else:
            ext  = None
            axis = None

        if mapping is not None:
            is_rational_mapping = isinstance( mapping, NurbsMapping )
            mapping_space = mapping.space
        else:
            is_rational_mapping = False
            mapping_space = None

        self._mapping             = mapping
        self._is_rational_mapping = is_rational_mapping
        discrete_space            = self._space

        # MPI communicator
        comm = vector_space.cart.comm if vector_space.parallel else None

        # BasicDiscrete generates the assembly code and sets the following attributes that are used afterwards:
        # self._func, self._free_args, self._max_nderiv and self._backend
        BasicDiscrete.__init__(self, expr, kernel_expr, comm=comm, root=0, discrete_space=discrete_space,
                              nquads=nquads, is_rational_mapping=is_rational_mapping, mapping=symbolic_mapping,
                              mapping_space=mapping_space, num_threads=num_threads, backend=backend)

        # Build the quadrature grid
        grid       = QuadratureGrid(self.space,  axis=axis, ext=ext, nquads=nquads)
        self._grid = grid

        # Extract the basis function values on the quadrature grid
        self._test_basis = BasisValues(
            self.space,
            nderiv = self.max_nderiv,
            nquads = nquads,
            trial  = True,
            grid   = grid
        )

        # Store MPI communicator
        # NOTE [YG 18.04.2024]: this is not equal to the variable `comm` when we have multiple patches
        self._comm = domain_h.comm

        # Construct the arguments to be passed to the assemble() function, which is stored in self._func
        self._args = self.construct_arguments()

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def space(self):
        return self._space

    @property
    def grid(self):
        return self._grid

    @property
    def nquads(self):
        return self._grid.nquads

    @property
    def test_basis(self):
        return self._test_basis

    def get_space_indices_from_target(self, domain, target):
        if domain.mapping:
            domain = domain.logical_domain
        if target.mapping:
            target = target.logical_domain

        domains = domain.interior.args
        if isinstance(target, Interface):
            raise NotImplementedError("Index of an interface is not defined for the FunctionalForm")
        elif isinstance(target, Boundary):
            i = domains.index(target.domain)
        else:
            i = domains.index(target)
        return i

    def construct_arguments(self):
        """
        Collect the arguments used in the assembly method.

        Returns
        -------
        args: tuple
         The arguments passed to the assembly method.
        """

        n_elements  = [e-s+1 for s,e in zip(self.grid.local_element_start,self.grid.local_element_end)]

        points        = self.grid.points
        weights       = self.grid.weights
        tests_basis   = self.test_basis.basis
        spans         = self.test_basis.spans
        tests_degrees = self.space.degree

        tests_basis, tests_degrees, spans = collect_spaces(self.space.symbolic_space, tests_basis, tests_degrees, spans)

        global_pads   = flatten(self.test_basis.space.pads)
        multiplicity  = flatten(self.test_basis.space.multiplicity)
        global_pads   = [p*m for p,m in zip(global_pads, multiplicity)]

        tests_basis   = flatten(tests_basis)
        tests_degrees = flatten(tests_degrees)
        spans         = flatten(spans)
        quads         = flatten(list(zip(points, weights)))
        nquads        = flatten(self.grid.nquads)

        if self.mapping:
            mapping    = [e._coeffs._data for e in self.mapping._fields]
            space      = self.mapping._fields[0].space
            map_degree = space.degree
            map_span   = [q.spans-s for q,s in zip(space.get_assembly_grids(*self.nquads), space.vector_space.starts)]
            map_basis  = [q.basis for q in space.get_assembly_grids(*self.nquads)]

            if self.is_rational_mapping:
                mapping = [*mapping, self.mapping._weights_field._coeffs._data]
        else:
            mapping    = []
            map_degree = []
            map_span   = []
            map_basis  = []

        args = (*tests_basis, *map_basis, *spans, *map_span, *quads, *tests_degrees, *map_degree, *n_elements, *nquads, *global_pads, *mapping)
        args = tuple(np.int64(a) if isinstance(a, int) else a for a in args)

        return args

    def assemble(self, **kwargs):
        """
        This method assembles the square of the functional expression with the given arguments and then compute
        the square root of the absolute value of the result.

        Examples
        --------
        >>> n = SemiNorm(1.0j*v, domain, kind='l2')
        >>> nh = discretize(n, domain_h,      Vh , **kwargs)
        >>> fh = FemField(Vh)
        >>> fh.coeffs[:] = 1
        >>> n_value = nh.assemble(v=fh)

        In n_value we have the value of np.sqrt(abs(sum((1.0jv)**2)))
        """
        args = [*self._args]
        for key in self._free_args:
            v = kwargs[key]
            if isinstance(v, FemField):
                if not v.coeffs.ghost_regions_in_sync:
                    v.coeffs.update_ghost_regions()
                if v.space.is_product:
                    coeffs = v.coeffs
                    if self._symbolic_space.is_broken:
                        index = self.get_space_indices_from_target(self._domain, self._target)
                        coeffs = coeffs[index]

                    if isinstance(coeffs, StencilVector):
                        args += (coeffs._data, )
                    else:
                        args += (e._data for e in coeffs)
                else:
                    args += (v.coeffs._data, )
            else:
                args += (v, )

        v = self._func(*args)
        if isinstance(self.expr, (sym_Norm, sym_SemiNorm)):
            if not( self.comm is None ):
                v = self.comm.allreduce(sendobj=v)

            if self.expr.exponent == 2:
                # add abs because of 0 machine
                v = np.sqrt(np.abs(v))
            else:
                raise NotImplementedError('TODO')
        return v

#==============================================================================
class DiscreteSumForm(BasicDiscrete):

    def __init__(self, a, kernel_expr, *args, **kwargs):
        # TODO Uncomment when the SesquilinearForm exist in SymPDE
        #if not isinstance(a, (sym_BilinearForm, sym_SesquilinearForm, sym_LinearForm, sym_Functional)):
            # raise TypeError('> Expecting a symbolic BilinearForm, SesquilinearForm, LinearForm, Functional')
        if not isinstance(a, (sym_BilinearForm, sym_LinearForm, sym_Functional)):
            raise TypeError('> Expecting a symbolic BilinearForm, LinearForm, Functional')

        self._expr = a
        backend = kwargs.pop('backend', None)
        self._backend = backend

        folder = kwargs.get('folder', None)
        self._folder = self._initialize_folder(folder)

        # create a module name if not given
        tag = random_string(8)

        # ...
        forms = []
        free_args = []
        self._kernel_expr = kernel_expr
        operator = None
        for e in kernel_expr:
            if isinstance(a, sym_LinearForm):
                kwargs['update_ghost_regions'] = False
                ah = DiscreteLinearForm(a, e, *args, backend=backend, **kwargs)
                kwargs['vector'] = ah._vector
                operator = ah._vector

            # TODO Uncomment when the SesquilinearForm exist in SymPDE
            # elif isinstance(a, sym_SesquilinearForm):
            #     kwargs['update_ghost_regions'] = False
            #     ah = DiscreteSesquilinearForm(a, e, *args, assembly_backend=backend, **kwargs)
            #     kwargs['matrix'] = ah._matrix
            #     operator = ah._matrix

            elif isinstance(a, sym_BilinearForm):
                kwargs['update_ghost_regions'] = False
                ah = DiscreteBilinearForm(a, e, *args, assembly_backend=backend, **kwargs)
                kwargs['matrix'] = ah._matrix
                operator = ah._matrix

            elif isinstance(a, sym_Functional):
                ah = DiscreteFunctional(a, e, *args, backend=backend, **kwargs)

            forms.append(ah)
            free_args.extend(ah.free_args)

        if isinstance(a, sym_BilinearForm):
            is_broken   = len(args[0].domain)>1
            if self._backend is not None and is_broken:
                for mat in kwargs['matrix']._blocks.values():
                    mat.set_backend(backend)
            elif self._backend is not None:
                kwargs['matrix'].set_backend(backend)

        self._forms         = forms
        self._operator      = operator
        self._free_args     = tuple(set(free_args))
        self._is_functional = isinstance(a, sym_Functional)
        # ...

    @property
    def forms(self):
        return self._forms

    @property
    def free_args(self):
        return self._free_args

    @property
    def is_functional(self):
        return self._is_functional

    def assemble(self, *, reset=True, **kwargs):
        if not self.is_functional:
            if reset :
                reset_arrays(*[i for M in self.forms for i in M.global_matrices])

            for form in self.forms:
                form.assemble(reset=False, **kwargs)
            self._operator.exchange_assembly_data()
            return self._operator
        else:
            M = [form.assemble(**kwargs) for form in self.forms]
            M = np.sum(M)
            return M

