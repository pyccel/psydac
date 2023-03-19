# coding: utf-8

# TODO: - init_fem is called whenever we call discretize. we should check that
#         nderiv has not been changed. shall we add quad_order too?

import numpy as np
from sympy import ImmutableDenseMatrix, Matrix

from sympde.expr          import BilinearForm as sym_BilinearForm
from sympde.expr          import LinearForm as sym_LinearForm
from sympde.expr          import Functional as sym_Functional
from sympde.expr          import Norm as sym_Norm
from sympde.topology      import Boundary, Interface
from sympde.topology      import VectorFunctionSpace
from sympde.topology      import ProductSpace
from sympde.topology      import H1SpaceType, L2SpaceType, UndefinedSpaceType
from sympde.calculus.core import PlusInterfaceOperator

from psydac.api.basic        import BasicDiscrete
from psydac.api.basic        import random_string
from psydac.api.grid         import QuadratureGrid, BasisValues
from psydac.api.utilities    import flatten
from psydac.linalg.stencil   import StencilVector, StencilMatrix, StencilInterfaceMatrix
from psydac.linalg.basic     import ComposedLinearOperator
from psydac.linalg.block     import BlockVectorSpace, BlockVector, BlockLinearOperator
from psydac.cad.geometry     import Geometry
from psydac.mapping.discrete import NurbsMapping
from psydac.fem.vector       import ProductFemSpace, VectorFemSpace
from psydac.fem.basic        import FemField
from psydac.fem.projectors   import construct_projection_operator
from psydac.core.bsplines    import find_span, basis_funs_all_ders
from psydac.ddm.cart         import InterfaceCartDecomposition

__all__ = (
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

def get_quad_order(Vh):
    if isinstance(Vh, (ProductFemSpace, VectorFemSpace)):
        return get_quad_order(Vh.spaces[0])
    return tuple([g.weights.shape[1] for g in Vh.quad_grids])

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
    return test_basis, test_degrees, spans, pads

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
    return trial_basis, trial_degrees, pads

#==============================================================================
def construct_quad_grids_arguments(grid, use_weights=True):
    points         = grid.points
    if use_weights:
        weights        = grid.weights
        quads          = flatten(list(zip(points, weights)))
    else:
        quads = flatten(list(zip(points)))

    quads_order   = flatten(grid.quad_order)
    n_elements    = grid.n_elements
    return n_elements, quads, quads_order

def reset_arrays(*args):
    for a in args: a[:] = 0.

def do_nothing(*args): return 0

def extract_stencil_mats(mats):
    new_mats = []
    for M in mats:
        if isinstance(M, (StencilInterfaceMatrix, StencilMatrix)):
            new_mats.append(M)
        elif isinstance(M, ComposedLinearOperator):
            new_mats += [i for i in M.multiplicants if isinstance(i, (StencilInterfaceMatrix, StencilMatrix))]
    return new_mats
#==============================================================================
class DiscreteBilinearForm(BasicDiscrete):
    """ Class that represents the concept of a discrete bi-linear form.
        This class allocates the matrix and generates the matrix assembly method.

    Parameters
    ----------

    expr : sympde.expr.expr.BilinearForm
        The symbolic bi-linear form.

    kernel_expr : sympde.expr.evaluation.KernelExpression
        The atomic representation of the bi-linear form.

    domain_h : Geometry
        The discretized domain

    spaces: list of FemSpace
        The trial and test discrete spaces.

    matrix: Matrix
        The matrix that we assemble into it.
        If not provided, it will create a new Matrix of the appropriate space.

    update_ghost_regions: bool
        Accumulate the contributions of the neighbouring processes.

    quad_order: list of tuple
        The number of quadrature points used in the assembly method.

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
        The symbolic mapping which defines the physical domain of the bi-linear form.

    """
    def __init__(self, expr, kernel_expr, domain_h, spaces, *, matrix=None, update_ghost_regions=True,
                       quad_order=None, backend=None, linalg_backend=None, assembly_backend=None,
                       symbolic_mapping=None):

        if not isinstance(expr, sym_BilinearForm):
            raise TypeError('> Expecting a symbolic BilinearForm')

        assert( isinstance(domain_h, Geometry) )

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
        if len(domain)>1:
            i,j = self.get_space_indices_from_target(domain, target )
            test_space   = self.spaces[1].spaces[i]
            trial_space  = self.spaces[0].spaces[j]
            if isinstance(target, Interface):
                m,_       = self.get_space_indices_from_target(domain, target.minus )
                p,_       = self.get_space_indices_from_target(domain, target.plus )
                mapping_m = list(domain_h.mappings.values())[m]
                mapping_p = list(domain_h.mappings.values())[p]
                mapping   = (mapping_m, mapping_p) if mapping_m else None
            else:
                mapping = list(domain_h.mappings.values())[i]
        else:
            trial_space  = self.spaces[0]
            test_space   = self.spaces[1]
            mapping      = list(domain_h.mappings.values())[0]

        self._mapping = mapping

        is_rational_mapping = False
        mapping_space       = None
        if not( mapping is None ) and not isinstance(target, Interface):
            is_rational_mapping = isinstance( mapping, NurbsMapping )
            mapping_space = mapping.space
        elif not( mapping is None ) and isinstance(target, Interface):
            is_rational_mapping = (isinstance( mapping[0], NurbsMapping ), isinstance( mapping[1], NurbsMapping ))
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
            axis        = target.axis
            test_ext    = target.ext
            trial_ext   = target.ext
        elif isinstance(target, Interface):
            # this part treats the cases of:
            # integral(v_minus * u_plus)
            # integral(v_plus  * u_minus)
            # the other cases, integral(v_minus * u_minus) and integral(v_plus * u_plus)
            # are converted to boundary integrals by Sympde
            axis         = target.axis
            test         = self.kernel_expr.test
            trial        = self.kernel_expr.trial
            test_target  = target.plus if isinstance(test, PlusInterfaceOperator) else target.minus
            trial_target = target.plus if isinstance(trial, PlusInterfaceOperator) else target.minus
            test_ext     = test_target.ext
            trial_ext    = trial_target.ext
            ncells       = tuple(max(i,j) for i,j in zip(test_space.ncells, trial_space.ncells))
            if isinstance(trial_space, VectorFemSpace):
                spaces = []
                for sp in trial_space.spaces:
                    if (trial_target.axis, trial_target.ext) in sp.interfaces:
                        spaces.append(sp._refined_space[ncells].interfaces[trial_target.axis, trial_target.ext])

                if len(spaces) == len(trial_space.spaces):
                    sym_space   = trial_space.symbolic_space
                    trial_space = VectorFemSpace(*spaces)
                    trial_space.symbolic_space = sym_space

            elif (trial_target.axis, trial_target.ext) in trial_space.interfaces:
                sym_space   = trial_space.symbolic_space
                trial_space = trial_space._refined_space[ncells].interfaces[trial_target.axis, trial_target.ext]
                trial_space.symbolic_space = sym_space

            test_space   = test_space._refined_space[ncells]
            self._test_ext  = test_target.ext
            self._trial_ext = trial_target.ext

        #...
        discrete_space   = (trial_space, test_space)
        space_quad_order = [qo - 1 for qo in get_quad_order(test_space)]
        quad_order       = [qo + 1 for qo in (quad_order or space_quad_order)]

        # this doesn't work right now otherwise. TODO: fix this and remove this assertion
        assert np.array_equal(quad_order, get_quad_order(test_space))

        # Assuming that all vector spaces (and their Cartesian decomposition,
        # if any) are compatible with each other, extract the first available
        # vector space from which (starts, ends, npts) will be read:
        starts = vector_space.starts
        ends   = vector_space.ends
        npts   = vector_space.npts

        comm = None
        if vector_space.parallel:
            comm = vector_space.cart.comm

        assembly_backend = backend or assembly_backend
        linalg_backend   = backend or linalg_backend

        # BasicDiscrete generates the assembly code and sets the following attributes that are used afterwards:
        # self._func, self._free_args, self._max_nderiv and self._backend
        BasicDiscrete.__init__(self, expr, kernel_expr, comm=comm, root=0, discrete_space=discrete_space,
                       quad_order=quad_order, is_rational_mapping=is_rational_mapping, mapping=symbolic_mapping,
                       mapping_space=mapping_space, num_threads=self._num_threads,backend=assembly_backend)

        #...
        if isinstance(target, (Boundary, Interface)):
            #...
            # If process does not own the boundary or interface, do not assemble anything
            if test_ext == -1:
                if starts[axis] != 0:
                    self._func = do_nothing

            elif test_ext == 1:
                if ends[axis] != npts[axis]-1:
                    self._func = do_nothing

            # In case of target==Interface, we only use the mpi ranks that are on the interface to assemble the BilinearForm
            if self._func == do_nothing and isinstance(target, Interface):
                self._free_args = ()
                self._args      = ()
                self._element_loop_starts = ()
                self._element_loop_ends   = ()
                self._global_matrices     = ()
                self._threads_args        = ()
                return

        if isinstance(target, Boundary):
            test_grid   = QuadratureGrid( test_space, axis, test_ext)
            trial_grid  = QuadratureGrid( trial_space, axis, trial_ext)
            self._grid  = (test_grid,)
        elif isinstance(target, Interface):
            # this part treats the cases of:
            # integral(v_minus * u_plus)
            # integral(v_plus  * u_minus)
            # the other cases, integral(v_minus * u_minus) and integral(v_plus * u_plus)
            # are converted to boundary integrals by Sympde
            test_grid    = QuadratureGrid( test_space, axis, test_ext)
            trial_grid   = QuadratureGrid( trial_space, axis, trial_ext)
            self._grid   = (test_grid, trial_grid) if test_target==target.minus else (trial_grid, test_grid)
            self._test_ext  = test_target.ext
            self._trial_ext = trial_target.ext
        else:
            test_grid   = QuadratureGrid( test_space)
            trial_grid  = QuadratureGrid( trial_space)
            self._grid  = (test_grid,)
        #...
        self._test_basis  = BasisValues( test_space,  nderiv = self.max_nderiv , trial=False, grid=test_grid)
        self._trial_basis = BasisValues( trial_space, nderiv = self.max_nderiv , trial=True, grid=trial_grid)

        self.allocate_matrices(linalg_backend)
        with_openmp  = (assembly_backend['name'] == 'pyccel' and assembly_backend['openmp']) if assembly_backend else False
        self._args , self._threads_args = self.construct_arguments(with_openmp=with_openmp)

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

        if self._free_args:
            basis   = []
            spans   = []
            degrees = []
            pads    = []
            coeffs  = []
            consts  = []

            for key in self._free_args:
                v = kwargs[key]

                if len(self.domain)>1 and isinstance(v, FemField) and v.space.is_product:
                    i,j = self.get_space_indices_from_target(self.domain, self.target)
                    assert i==j
                    v = v[i]
                if isinstance(v, FemField):
                    assert len(self.grid) == 1
                    if not v.coeffs.ghost_regions_in_sync:
                        v.coeffs.update_ghost_regions()
                    basis_v  = BasisValues(v.space, nderiv = self.max_nderiv, trial=True, grid=self.grid[0])
                    bs, d, s, p = construct_test_space_arguments(basis_v)
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

#        args = args + self._element_loop_starts + self._element_loop_ends

        if reset:
            reset_arrays(*self.global_matrices)

        self._func(*args, *self._threads_args)
        if self._matrix and self._update_ghost_regions:
            self._matrix.exchange_assembly_data()

        if self._matrix: self._matrix.ghost_regions_in_sync = False
        return self._matrix

    def get_space_indices_from_target(self, domain, target):
        if domain.mapping:
            domain = domain.logical_domain
        if target.mapping:
            target = target.logical_domain
        domains = domain.interior.args
        if isinstance(target, Interface):
            test       = self.kernel_expr.test
            trial      = self.kernel_expr.trial
            test_target  =  target.plus if isinstance(test, PlusInterfaceOperator) else target.minus
            trial_target = target.plus if isinstance(trial, PlusInterfaceOperator) else target.minus
            i,j = [domains.index(test_target.domain), domains.index(trial_target.domain)]
        else:
            if isinstance(target, Boundary):
                i = domains.index(target.domain)
                j = i
            else:
                i = domains.index(target)
                j = i
        return i,j

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
        test_basis, test_degrees, spans, pads = construct_test_space_arguments(self.test_basis)
        trial_basis, trial_degrees, pads      = construct_trial_space_arguments(self.trial_basis)
        n_elements, quads, quad_degrees       = construct_quad_grids_arguments(self.grid[0], use_weights=False)
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
                map_span   = [[q.spans-s for q,s in zip(sp.quad_grids, sp.vector_space.starts)] for sp in spaces]
                map_basis  = [[q.basis for q in sp.quad_grids] for sp in spaces]
                points     = [g.points for g in self.grid]
                weights    = [self.mapping.weights_field.coeffs._data] if self.is_rational_mapping else []
            elif len(self.grid) == 2:
                target   = self.kernel_expr.target
                assert isinstance(target, Interface)
                mappings = list(self.mapping)
                i,j = self.get_space_indices_from_target(self.domain, target)
                m,_ = self.get_space_indices_from_target(self.domain, target.minus)
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
                map_span   = [[q.spans-s for q,s in zip(sp.quad_grids, sp.vector_space.starts)] for sp in spaces]
                map_basis  = [[q.basis for q in sp.quad_grids] for sp in spaces]
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
        Allocate the global matrices used in the assmebly method.
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
        test_fem_space   = self.spaces[1]
        trial_fem_space  = self.spaces[0]
        domain          = self.domain
        is_broken       = len(domain)>1
        is_conformal    = True

        if isinstance(expr, (ImmutableDenseMatrix, Matrix)):
            if not isinstance(test_degree[0],(list, tuple, np.ndarray)):
                test_degree = [test_degree]

            if not isinstance(trial_degree[0],(list, tuple, np.ndarray)):
                trial_degree = [trial_degree]

            pads         = np.empty((len(test_degree),len(trial_degree),len(test_degree[0])), dtype=int)
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
            i,j = self.get_space_indices_from_target(domain, target )
            test_fem_space   = self.spaces[1].spaces[i]
            trial_fem_space  = self.spaces[0].spaces[j]
            test_space  = test_space.spaces[i]
            trial_space = trial_space.spaces[j]
            ncells = tuple(max(i,j) for i,j in zip(test_fem_space.ncells, trial_fem_space.ncells))
            is_conformal = tuple(test_fem_space.ncells) == ncells and tuple(trial_fem_space.ncells) == ncells

        else:
            ncells = tuple(max(i,j) for i,j in zip(test_fem_space.ncells, trial_fem_space.ncells))
            i=0
            j=0
            #else so initialisation causing bug on line 682

        if isinstance(expr, (ImmutableDenseMatrix, Matrix)): # case of system of equations

            if is_broken: #multi patch
                if not self._matrix[i,j]:
                    mat = BlockLinearOperator(trial_fem_space._refined_space[ncells].vector_space, test_fem_space._refined_space[ncells].vector_space)
                    if not is_conformal and not i==j:
                        if all(trn>=tn for trn,tn in zip(trial_fem_space.ncells, test_fem_space.ncells)):
                            Ps  = [construct_projection_operator(ts._refined_space[ncells], ts) for ts in test_fem_space.spaces]
                            P   = BlockLinearOperator(test_fem_space._refined_space[ncells].vector_space, test_fem_space.vector_space)

                            for ni,Pi in enumerate(Ps):
                                P[ni,ni] = Pi

                            mat = ComposedLinearOperator(trial_space, test_space, P, mat)
                        else:
                            Ps  = [construct_projection_operator(trs, trs._refined_space[ncells]) for trs in trial_fem_space.spaces]
                            P   = BlockLinearOperator(trial_fem_space.vector_space, trial_fem_space._refined_space[ncells].vector_space)
                            for ni,Pi in enumerate(Ps):P[ni,ni] = Pi
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
                        ts_space = test_fem_space._refined_space[ncells].vector_space.spaces[k1]
                    else:
                        ts_space = test_fem_space._refined_space[ncells].vector_space

                    if isinstance(trial_fem_space, VectorFemSpace):
                        tr_space = trial_fem_space._refined_space[ncells].vector_space.spaces[k2]
                    else:
                        tr_space = trial_fem_space._refined_space[ncells].vector_space

                    if is_conformal and matrix[k1,k2]:
                        global_mats[k1,k2] = matrix[k1,k2]
                    elif not i == j: # assembling in an interface (type(target) == Interface)
                        axis   = target.axis
                        ext_d  = self._trial_ext
                        ext_c  = self._test_ext
                        test_n  = self.test_basis.space.spaces[k1].spaces[axis].nbasis
                        test_s  = self.test_basis.space.spaces[k1].vector_space.starts[axis]
                        trial_n = self.trial_basis.space.spaces[k2].spaces[axis].nbasis
                        cart    = self.trial_basis.space.spaces[k2].vector_space.cart
                        trial_s = cart.global_starts[axis][cart._coords[axis]]

                        s_d = trial_n-trial_s - trial_degree[k2][axis]-1 if ext_d == 1 else 0
                        s_c = test_n-trial_s  - test_degree[k1][axis] -1 if ext_c == 1 else 0

                        direction = target.direction
                        direction = 1 if direction is None else direction
                        flip = [direction]*domain.dim
                        flip[axis] = 1
                        if self._func != do_nothing:
                            global_mats[k1,k2] = StencilInterfaceMatrix(tr_space, ts_space,
                                                                        s_d, s_c,
                                                                        axis, axis,
                                                                        ext_d, ext_c,
                                                                        pads=tuple(pads[k1,k2]), 
                                                                        flip=flip)
                    else:
                        global_mats[k1,k2] = StencilMatrix(tr_space, ts_space, pads = tuple(pads[k1,k2]))

                    if is_conformal:
                        matrix[k1,k2] = global_mats[k1,k2]
                    elif all(trn>=tn for trn,tn in zip(trial_fem_space.ncells, test_fem_space.ncells)):
                        matrix.operators[-1][k1,k2] = global_mats[k1,k2]
                    else:
                        matrix.operators[0][k1,k2] = global_mats[k1,k2]

        else: # case of scalar equation
            if is_broken: # multi-patch
                if self._matrix[i,j]:
                    global_mats[i,j] = self._matrix[i,j]
                elif not i == j: # assembling in an interface (type(target) == Interface)
                    axis   = target.axis
                    ext_d  = self._trial_ext
                    ext_c  = self._test_ext
                    test_n  = self.test_basis.space.spaces[axis].nbasis
                    test_s  = self.test_basis.space.vector_space.starts[axis]
                    trial_n = self.trial_basis.space.spaces[axis].nbasis
                    cart    = self.trial_basis.space.vector_space.cart
                    trial_s = cart.global_starts[axis][cart._coords[axis]]

                    s_d = trial_n-trial_s - trial_degree[axis]-1 if ext_d == 1 else 0
                    s_c = test_n-trial_s  - test_degree[axis] -1 if ext_c == 1 else 0

                    direction = target.direction
                    direction = 1 if direction is None else direction
                    flip = [direction]*domain.dim
                    flip[axis] = 1

                    if self._func != do_nothing:
                        mat = StencilInterfaceMatrix(trial_fem_space._refined_space[ncells].vector_space,
                                                     test_fem_space._refined_space[ncells].vector_space,
                                                     s_d, s_c,
                                                     axis, axis,
                                                     ext_d, ext_c,
                                                     flip=flip)
                        if not is_conformal:
                            if all(trn>=tn for trn,tn in zip(trial_fem_space.ncells, test_fem_space.ncells)):
                                P   = construct_projection_operator(test_fem_space._refined_space[ncells], test_fem_space)
                                mat = ComposedLinearOperator(trial_space, test_space, P, mat)
                            else:
                                P   = construct_projection_operator(trial_fem_space, trial_fem_space._refined_space[ncells])
                                mat = ComposedLinearOperator(trial_space, test_space, mat, P)
                        global_mats[i,j] = mat
                else:
                    global_mats[i,j] = StencilMatrix(trial_space, test_space, pads=tuple(pads))

                if (i,j) in global_mats:
                    self._matrix[i,j] = global_mats[i,j]

            else: # single patch
                if self._matrix:
                    global_mats[0,0] = self._matrix
                else:
                    global_mats[0,0] = StencilMatrix(trial_space, test_space, pads=tuple(pads))

                self._matrix = global_mats[0,0]

        if backend is not None and is_broken:
            for mat in global_mats.values():
                mat.set_backend(backend)
        elif backend is not None:
            self._matrix.set_backend(backend)

        self._global_matrices     = [M._data for M in extract_stencil_mats(global_mats.values())]

#==============================================================================
class DiscreteLinearForm(BasicDiscrete):
    """ Class that represents the concept of a discrete linear form.
        This class allocates the vector and generates the vector assembly method.

    Parameters
    ----------

    expr : sympde.expr.expr.LinearForm
        The symbolic linear form.

    kernel_expr : sympde.expr.evaluation.KernelExpression
        The atomic representation of the linear form.

    domain_h : Geometry
        The discretized domain

    space : FemSpace
        The discrete space.

    vector : Vector
        The vector that we assemble into it.
        If not provided, it will create a new Vector of the appropriate space.

    update_ghost_regions : bool
        Accumulate the contributions of the neighbouring processes.

    quad_order : list or tuple
        The number of quadrature points used in the assembly method.

    backend : dict
        The backend used to accelerate the computing kernels.
        The backend dictionaries are defined in the file psydac/api/settings.py

    symbolic_mapping : Sympde.topology.Mapping
        The symbolic mapping which defines the physical domain of the linear form.

    """
    def __init__(self, expr, kernel_expr, domain_h, space, *, vector=None,
                       update_ghost_regions=True, quad_order=None, backend=None,
                       symbolic_mapping=None):

        if not isinstance(expr, sym_LinearForm):
            raise TypeError('> Expecting a symbolic LinearForm')

        assert( isinstance(domain_h, Geometry) )

        self._space  = space

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

        if len(domain)>1:
            i = self.get_space_indices_from_target(domain, target )
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

        is_rational_mapping = False
        mapping_space       = None
        if not( mapping is None ):
            is_rational_mapping = isinstance( mapping, NurbsMapping )
            mapping_space = mapping.space

        self._is_rational_mapping = is_rational_mapping
        discrete_space            = test_space


        space_quad_order = [qo - 1 for qo in get_quad_order(test_space)]
        quad_order       = [qo + 1 for qo in (quad_order or space_quad_order)]

        # this doesn't work right now otherwise. TODO: fix this and remove this assertion
        assert np.array_equal(quad_order, get_quad_order(test_space))

        comm        = None
        if vector_space.parallel:
            comm = vector_space.cart.comm

        # BasicDiscrete generates the assembly code and sets the following attributes that are used afterwards:
        # self._func, self._free_args, self._max_nderiv and self._backend
        BasicDiscrete.__init__(self, expr, kernel_expr, comm=comm, root=0, discrete_space=discrete_space,
                              quad_order=quad_order, is_rational_mapping=is_rational_mapping, mapping=symbolic_mapping,
                              mapping_space=mapping_space, num_threads=self._num_threads, backend=backend)

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

        grid             = QuadratureGrid( test_space, axis=axis, ext=ext )
        self._grid       = grid
        self._test_basis = BasisValues( test_space, nderiv = self.max_nderiv, grid=grid)

        self.allocate_matrices()

        with_openmp  = (backend['name'] == 'pyccel' and backend['openmp']) if backend else False
        self._args , self._threads_args = self.construct_arguments(with_openmp=with_openmp)

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
    def test_basis(self):
        return self._test_basis

    @property
    def global_matrices(self):
        return self._global_matrices

    @property
    def args(self):
        return self._args

    def assemble(self, *, reset=True, **kwargs):
        if self._free_args:
            basis   = []
            spans   = []
            degrees = []
            pads    = []
            coeffs  = []
            consts  = []
            for key in self._free_args:
                v = kwargs[key]
                if len(self.domain)>1 and isinstance(v, FemField) and v.space.is_product:
                    i = self.get_space_indices_from_target(self.domain, self.target)
                    v = v[i]
                if isinstance(v, FemField):
                    if not v.coeffs.ghost_regions_in_sync:
                        v.coeffs.update_ghost_regions()
                    basis_v  = BasisValues(v.space, nderiv = self.max_nderiv, trial=True, grid=self.grid)
                    bs, d, s, p = construct_test_space_arguments(basis_v)
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

        if reset:
            reset_arrays(*self.global_matrices)

        self._func(*args, *self._threads_args)
        if self._vector and self._update_ghost_regions:
            self._vector.exchange_assembly_data()

        if self._vector: self._vector.ghost_regions_in_sync = False
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
        tests_basis, tests_degrees, spans, pads = construct_test_space_arguments(self.test_basis)
        n_elements, quads, quads_degree         = construct_quad_grids_arguments(self.grid, use_weights=False)

        global_pads   = self.space.vector_space.pads

        if self.mapping:
            mapping    = [e._coeffs._data for e in self.mapping._fields]
            space      = self.mapping._fields[0].space
            map_degree = space.degree
            map_span   = [q.spans-s for q,s in zip(space.quad_grids, space.vector_space.starts)]
            map_basis  = [q.basis for q in space.quad_grids]
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

        args = (*tests_basis, *map_basis, *spans, *map_span, *quads, *tests_degrees, *map_degree, *n_elements, *quads_degree, *global_pads, *mapping, *self._global_matrices)

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
        Allocate the global matrices used in the assmebly method.
        In this method we allocate only the matrices that are computed in the self._target domain,
        we also avoid double allocation if we have many DiscreteLinearForm that are defined on the same self._target domain.
        """
        global_mats   = {}

        test_space  = self.test_basis.space.vector_space
        test_degree = np.array(self.test_basis.space.degree)

        expr        = self.kernel_expr.expr
        target      = self.kernel_expr.target
        domain      = self.domain
        is_broken   = len(domain)>1

        if self._vector is None and (is_broken or isinstance( expr, (ImmutableDenseMatrix, Matrix))):
            self._vector = BlockVector(self.space.vector_space)

        if isinstance(expr, (ImmutableDenseMatrix, Matrix)): # case system of equations

            if is_broken: #multi patch
                i = self.get_space_indices_from_target(domain, target )
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
                    if  vector[i]:
                        global_mats[i] = vector[i]
                    else:
                        global_mats[i] = StencilVector(test_space.spaces[i])

                vector[i] = global_mats[i]
        else:
            if is_broken:
                i = self.get_space_indices_from_target(domain, target )
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
class DiscreteFunctional(BasicDiscrete):
    """ Class that represents the concept of a discrete functional form.
        This class generates the functiona form assembly method.

    Parameters
    ----------

    expr : sympde.expr.expr.LinearForm
        The symbolic functional form.

    kernel_expr : sympde.expr.evaluation.KernelExpression
        The atomic representation of the functional form.

    domain_h : Geometry
        The discretized domain

    space : FemSpace
        The discrete space.

    update_ghost_regions : bool
        Accumulate the contributions of the neighbouring processes.

    quad_order : list or tuple
        The number of quadrature points used in the assembly method.

    backend : dict
        The backend used to accelerate the computing kernels.
        The backend dictionaries are defined in the file psydac/api/settings.py

    symbolic_mapping : Sympde.topology.Mapping
        The symbolic mapping which defines the physical domain of the functional form.

    """
    def __init__(self, expr, kernel_expr, domain_h, space, *, quad_order=None,
                       backend=None, symbolic_mapping=None):

        if not isinstance(expr, sym_Functional):
            raise TypeError('> Expecting a symbolic Functional')

        # ...
        assert( isinstance(domain_h, Geometry) )

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

        if len(domain)>1:
            i = self.get_space_indices_from_target(domain, target)
            self._space  = self._space.spaces[i]
            mapping = list(domain_h.mappings.values())[i]
        else:
            mapping = list(domain_h.mappings.values())[0]

        if isinstance(self.space.vector_space, BlockVectorSpace):
            vector_space = self.space.vector_space.spaces[0]
            if isinstance(vector_space, BlockVectorSpace):
                vector_space = vector_space.spaces[0]
        else:
            vector_space = self.space.vector_space

        num_threads  = 1
        if vector_space.parallel and vector_space.cart.num_threads>1:
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
            ext        = target.ext
            axis       = target.axis
        else:
            ext        = None
            axis       = None

        is_rational_mapping = False
        mapping_space       = None
        if not( mapping is None ):
            is_rational_mapping = isinstance( mapping, NurbsMapping )
            mapping_space = mapping.space

        self._mapping             = mapping
        self._is_rational_mapping = is_rational_mapping
        discrete_space            = self._space

        comm = None
        if vector_space.parallel:
            comm = vector_space.cart.comm

        space_quad_order = [qo - 1 for qo in get_quad_order(self._space)]
        quad_order       = [qo + 1 for qo in (quad_order or space_quad_order)]

        # this doesn't work right now otherwise. TODO: fix this and remove this assertion
        assert np.array_equal(quad_order, get_quad_order(self.space))

        # BasicDiscrete generates the assembly code and sets the following attributes that are used afterwards:
        # self._func, self._free_args, self._max_nderiv and self._backend
        BasicDiscrete.__init__(self, expr, kernel_expr, comm=comm, root=0, discrete_space=discrete_space,
                              quad_order=quad_order, is_rational_mapping=is_rational_mapping, mapping=symbolic_mapping,
                              mapping_space=mapping_space, num_threads=num_threads, backend=backend)

        self._comm       = domain_h.comm
        grid             = QuadratureGrid( self.space,  axis=axis, ext=ext)
        self._grid       = grid
        self._test_basis = BasisValues( self.space, nderiv = self.max_nderiv, trial=True, grid=grid)

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
        quads_degree  = flatten(self.grid.quad_order)

        if self.mapping:
            mapping    = [e._coeffs._data for e in self.mapping._fields]
            space      = self.mapping._fields[0].space
            map_degree = space.degree
            map_span   = [q.spans-s for q,s in zip(space.quad_grids, space.vector_space.starts)]
            map_basis  = [q.basis for q in space.quad_grids]

            if self.is_rational_mapping:
                mapping = [*mapping, self.mapping._weights_field._coeffs._data]
        else:
            mapping    = []
            map_degree = []
            map_span   = []
            map_basis  = []

        args = (*tests_basis, *map_basis, *spans, *map_span, *quads, *tests_degrees, *map_degree, *n_elements, *quads_degree, *global_pads, *mapping)
        args = tuple(np.int64(a) if isinstance(a, int) else a for a in args)

        return args

    def assemble(self, **kwargs):
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
        if isinstance(self.expr, sym_Norm):
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
        if not isinstance(a, (sym_BilinearForm, sym_LinearForm, sym_Functional)):
            raise TypeError('> Expecting a symbolic BilinearForm, LinearFormn Functional')

        self._expr = a
        backend = kwargs.pop('backend', None)
        self._backend = backend

        folder = kwargs.get('folder', None)
        self._folder = self._initialize_folder(folder)

        # create a module name if not given
        tag = random_string( 8 )

        # ...
        forms = []
        free_args = []
        self._kernel_expr = kernel_expr
        operator = None
        for e in kernel_expr:
            if isinstance(a, sym_BilinearForm):
                kwargs['update_ghost_regions'] = False
                ah = DiscreteBilinearForm(a, e, *args, assembly_backend=backend, **kwargs)
                kwargs['matrix'] = ah._matrix
                operator = ah._matrix

            elif isinstance(a, sym_LinearForm):
                kwargs['update_ghost_regions'] = False
                ah = DiscreteLinearForm(a, e, *args, backend=backend, **kwargs)
                kwargs['vector'] = ah._vector
                operator = ah._vector

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
