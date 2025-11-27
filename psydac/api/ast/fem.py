#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np
from itertools   import groupby, product

from sympy import Basic, S, Function, Integer, Symbol
from sympy import Matrix, ImmutableDenseMatrix, true
from sympy.core.containers import Tuple

from sympde.expr                 import LinearForm, BilinearForm, Functional
from sympde.topology.basic       import Boundary, Interface
from sympde.topology             import H1SpaceType, HcurlSpaceType, HdivSpaceType, L2SpaceType, UndefinedSpaceType, IdentityMapping
from sympde.topology.space       import ScalarFunction, VectorFunction, IndexedVectorFunction
from sympde.topology.derivatives import _logical_partial_derivatives, get_atom_logical_derivatives
from sympde.topology.mapping     import InterfaceMapping
from sympde.calculus.core        import is_zero, PlusInterfaceOperator

from psydac.pyccel.ast.core import _atomic, Assign, Import, Return, Comment, Continue, Slice

from .nodes import GlobalTensorQuadratureGrid, PlusGlobalTensorQuadratureGrid, LocalTensorQuadratureGrid, PlusLocalTensorQuadratureGrid
from .nodes import GlobalTensorQuadratureTestBasis, LocalTensorQuadratureTestBasis, GlobalTensorQuadratureTrialBasis, LocalTensorQuadratureTrialBasis
from .nodes import LengthElement, LengthQuadrature
from .nodes import LengthDofTrial, LengthDofTest
from .nodes import Reset, ProductGenerator
from .nodes import BlockStencilMatrixLocalBasis, StencilMatrixLocalBasis, BlockStencilMatrixGlobalBasis, BlockScalarLocalBasis
from .nodes import BlockStencilVectorLocalBasis, StencilVectorLocalBasis, BlockStencilVectorGlobalBasis
from .nodes import GlobalElementBasis, LocalElementBasis
from .nodes import GlobalSpanArray, LocalSpanArray, GlobalThreadSpanArray, CoefficientBasis
from .nodes import MatrixLocalBasis, MatrixGlobalBasis, MatrixRankFromCoords, MatrixCoordsFromRank
from .nodes import GeometryExpressions
from .nodes import Loop, VectorAssign
from .nodes import EvalMapping, EvalField
from .nodes import ComputeKernelExpr
from .nodes import ElementOf, Reduce, Reduction
from .nodes import construct_logical_expressions
from .nodes import Pads, Mask
from .nodes import index_quad, index_element, index_dof_test, index_dof_trial, index_outer_dof_test, index_inner_dof_test
from .nodes import thread_coords, local_index_element, thread_id, neighbour_threads
from .nodes import TensorAssignExpr, TensorInteger, TensorAdd, TensorMul, TensorMax
from .nodes import IntDivNode, AddNode, MulNode, EqNode, IfNode
from .nodes import GlobalThreadStarts, GlobalThreadEnds, GlobalThreadSizes, LocalThreadStarts, LocalThreadEnds
from .nodes import Allocate, Array
from .nodes import AndNode, StrictLessThanNode, WhileLoop, NotNode
from .nodes import GlobalThreadStarts, GlobalThreadEnds, GlobalThreadSizes
from .nodes import Allocate, Array
from .nodes import Block, ParallelBlock
from .utilities  import get_max_partial_derivatives

from psydac.api.ast.utilities import variables
from psydac.api.utilities     import flatten
from psydac.linalg.block      import BlockVectorSpace
from psydac.fem.vector        import VectorFemSpace

#==============================================================================
def toInteger(a):
    if isinstance(a,(int, np.int64)):
        return Integer(int(a))
    return a

#==============================================================================
def convert(dtype):
    """
    This function returns the index of a Function Space in a 3D DeRham sequence

    """
    if isinstance(dtype, (H1SpaceType, UndefinedSpaceType)):
        return 0
    elif isinstance(dtype, HcurlSpaceType):
        return 1
    elif isinstance(dtype, HdivSpaceType):
        return 2
    elif isinstance(dtype, L2SpaceType):
        return 3

#==============================================================================
def compute_diag_len(p, md, mc):
    n = ((np.ceil((p+1)/mc)-1)*md).astype('int')
    n = n-np.minimum(0, n-p)+p+1
    return n.astype('int')

#==============================================================================
def regroup(tests):
    """
    This function regourps the test/trial functions by their Function Space

    """
    tests  = [i.base if isinstance(i, IndexedVectorFunction) else i for i in tests]
    new_tests = []
    for i in tests:
        if i not in new_tests:
            new_tests.append(i)
    tests = new_tests

    spaces = [i.space for i in tests]
    kinds  = [i.kind for i in spaces]
    funcs  = dict(zip(tests, kinds))
    funcs  = sorted(funcs.items(), key=lambda x:convert(x[1]))
    grs = [dict(g) for k,g in groupby(funcs,key=lambda x:convert(x[1]))]
    grs = [(list(g.values())[0],tuple(g.keys())) for g in grs]
    groups = []
    for d,g in grs:
        if isinstance(d, (HcurlSpaceType, HdivSpaceType)) and isinstance(g[0], VectorFunction):
            dim = g[0].space.ldim
            for i in range(dim):
                s = [u[i] for u in g]
                groups += [(d,tuple(s))]
        else:
            groups += [(d,g)]
    return groups

#==============================================================================
def expand(args):
    """
    This function expands vector functions into indexed functions

    """
    new_args = []
    for i in args:
        if isinstance(i, (ScalarFunction, IndexedVectorFunction)):
            new_args += [i]
        elif isinstance(i, VectorFunction):
            new_args += [i[k] for k in  range(i.space.ldim)]
        else:
            raise NotImplementedError("TODO")
    return tuple(new_args)

#==============================================================================
class DefNode(Basic):
    """
    DefNode represents a function definition where it contains the arguments and the body

    """
    def __new__(cls, name, arguments, local_variables, body, imports, results, kind, domain_dtype='real'):
        obj = Basic.__new__(cls, name, arguments, local_variables, body, imports, results, kind)
        obj._domain_dtype=domain_dtype
        return obj

    @property
    def name(self):
        return self._args[0]

    @property
    def arguments(self):
        return self._args[1]

    @property
    def local_variables(self):
        return self._args[2]

    @property
    def body(self):
        return self._args[3]

    @property
    def imports(self):
        return self._args[4]

    @property
    def results(self):
        return self._args[5]

    @property
    def kind(self):
        return self._args[6]

    @property
    def domain_dtype(self):
        '''
        This property is used when we create the type of a constant for pyccel in build_pyccel_types_decorator.
        '''
        return self._domain_dtype


#==============================================================================
def expand_hdiv_hcurl(args):
    """
    This function expands vector functions of type hdiv and hculr into indexed functions
    """
    new_args         = []
    for i,a in enumerate(args):
        if isinstance(a, ScalarFunction):
            new_args += [a]
        elif isinstance(a, VectorFunction):
            if isinstance(a.space.kind, (HcurlSpaceType, HdivSpaceType)):
                new_args += [a[k] for k in  range(a.space.ldim)]
            else:
                new_args += [a]
        else:
            raise NotImplementedError("TODO")

    return tuple(new_args)

#==============================================================================
def get_multiplicity(funcs, space):
    def recursive_func(space):
        if isinstance(space, BlockVectorSpace):
            multiplicity = [recursive_func(s) for s in space.spaces]
        else:
            multiplicity = list(space.shifts)
        return multiplicity

    multiplicity = recursive_func(space)
    if not isinstance(multiplicity[0], list):
        multiplicity = [multiplicity]

    funcs = expand(funcs)
    assert len(funcs) == len(multiplicity)
    new_multiplicity = []
    for i in range(len(funcs)):
        if isinstance(funcs[i], ScalarFunction):
            new_multiplicity.append(multiplicity[i])
        elif isinstance(funcs[i].base.space.kind, (HcurlSpaceType, HdivSpaceType)):
            new_multiplicity.append(multiplicity[i])
        else:
            if i+1==len(funcs) or isinstance(funcs[i+1], ScalarFunction) or funcs[i].base != funcs[i+1].base:
                new_multiplicity.append(multiplicity[i])
    return new_multiplicity

#==============================================================================
def get_degrees(funcs, space):
    degrees = list(space.degree)
    if not isinstance(degrees[0], (list, tuple)):
        degrees = [degrees]

    funcs = expand(funcs)
    assert len(funcs) == len(degrees)
    new_degrees = []
    for i in range(len(funcs)):
        if isinstance(funcs[i], ScalarFunction):
            new_degrees.append(degrees[i])
        elif isinstance(funcs[i].base.space.kind, (HcurlSpaceType, HdivSpaceType)):
            new_degrees.append(degrees[i])
        else:
            if i+1==len(funcs) or isinstance(funcs[i+1], ScalarFunction) or funcs[i].base != funcs[i+1].base:
                new_degrees.append(degrees[i])
    return new_degrees

#==============================================================================
class AST(object):
    """
    The AST class transforms a terminal expression returned from SymPDE
    into a DefNode object, which it stores into the attribute `expr`.

    A DefNode represents a function definition, and it contains the full
    PSYDAC abstract syntax tree for the assembly function of a BilinearForm,
    LinearForm, or Functional.

    """
    def __init__(self, expr, terminal_expr, spaces, *, nquads, mapping_space=None, tag=None, mapping=None, is_rational_mapping=None,
                     num_threads=1, **kwargs):
        # ... compute terminal expr
        # TODO check that we have one single domain/interface/boundary

        is_bilinear         = False
        is_linear           = False
        is_functional       = False
        tests               = ()
        trials              = ()
        multiplicity_tests  = ()
        multiplicity_trials = ()
        multiplicity_fields = ()
        tests_degrees       = ()
        trials_degrees      = ()
        fields_degrees      = ()
        # ...
        domain              = terminal_expr.target
        dim                 = domain.dim
        constants           = expr.constants
        mask                = None
        nquads              = tuple(nquads)

        # Define mask for different domain
        if isinstance(domain, Boundary):
            mask = Mask(domain.axis, domain.ext)

        elif isinstance(domain, Interface):
            mask     = Mask(domain.axis, None)
            is_trial = {}
            if isinstance(terminal_expr.trial, PlusInterfaceOperator):
                is_trial[domain.plus]  = True
                is_trial[domain.minus] = False
            else:
                is_trial[domain.plus]  = False
                is_trial[domain.minus] = True

            kwargs["is_trial"] = is_trial

        # Define variables for different form
        if isinstance(expr, LinearForm):
            is_linear           = True
            tests               = expr.test_functions
            fields              = expr.fields
            is_broken           = spaces.symbolic_space.is_broken
            tests_degrees       = get_degrees(tests, spaces)
            multiplicity_tests  = get_multiplicity(tests, spaces.coeff_space)
            is_parallel         = spaces.coeff_space.parallel
            spaces              = spaces.symbolic_space

            # Define the type of scalar that the code should manage
            dtype           = spaces.codomain_type if hasattr(spaces, 'codomain_type') else 'real'

        elif isinstance(expr, BilinearForm):
            is_bilinear         = True
            tests               = expr.test_functions
            trials              = expr.trial_functions
            atoms               = terminal_expr.expr.atoms(ScalarFunction, VectorFunction)
            fields              = tuple(i for i in atoms if i not in tests+trials)
            is_broken           = spaces[1].symbolic_space.is_broken
            tests_degrees       = get_degrees(tests, spaces[1])
            trials_degrees      = get_degrees(trials, spaces[0])
            multiplicity_tests  = get_multiplicity(tests, spaces[1].coeff_space)
            multiplicity_trials = get_multiplicity(trials, spaces[0].coeff_space)
            is_parallel         = spaces[1].coeff_space.parallel
            spaces              = [V.symbolic_space for V in spaces]

            # Define the type of scalar that the code should manage
            if hasattr(spaces[0], 'codomain_type'):
                # TODO uncomment this line when we have a SesquilinearForm define in SymPDE
                #assert isinstance(expr, SesquilinearForm)
                dtype           = spaces[0].codomain_type
            else:
                # TODO uncomment this line when we have a SesquilinearForm define in SymPDE
                #assert not isinstance(expr, SesquilinearForm)
                dtype           = 'real'

        elif isinstance(expr, Functional):
            is_functional       = True
            fields              = tuple(expr.atoms(ScalarFunction, VectorFunction))
            is_broken           = spaces.symbolic_space.is_broken
            fields_degrees      = get_degrees(fields, spaces)
            multiplicity_fields = get_multiplicity(fields, spaces.coeff_space)
            is_parallel         = spaces.coeff_space.parallel
            spaces              = spaces.symbolic_space

            # Define the type of scalar that the code should manage
            dtype           = spaces.codomain_type if hasattr(spaces, 'codomain_type') else 'real'

        else:
            raise NotImplementedError('TODO')

        tests                = expand_hdiv_hcurl(tests)
        trials               = expand_hdiv_hcurl(trials)
        fields               = expand_hdiv_hcurl(fields)
        kwargs['nquads']     = nquads
        atoms_types          = (ScalarFunction, VectorFunction, IndexedVectorFunction)
        nderiv               = 0
        terminal_expr        = terminal_expr.expr

        if isinstance(terminal_expr, (ImmutableDenseMatrix, Matrix)):
            n_rows, n_cols    = terminal_expr.shape
            atomic_expr_field = {f:[] for f in fields}
            for i_row in range(0, n_rows):
                for i_col in range(0, n_cols):
                    atoms       = _atomic(terminal_expr[i_row, i_col], cls=atoms_types+_logical_partial_derivatives)
                    #--------------------------------------------------------------------
                    # TODO [YG, 05.02.2021]: create 'get_test_function' and use it below:
#                    field_atoms = [a for a in atoms if get_test_function(a) in fields]
                    field_atoms = []
                    #--------------------------------------------------------------------
                    for f in field_atoms:
                        a = _atomic(f, cls=atoms_types)
                        assert len(a) == 1
                        atomic_expr_field[a[0]].append(f)
                    
                    Fs = [get_atom_logical_derivatives(a) for a in atoms]
                    d = get_max_partial_derivatives(terminal_expr[i_row,i_col], logical=True, F=Fs)
                    nderiv = max(nderiv, max(d.values()))

        else:
            atoms       = _atomic(terminal_expr, cls=atoms_types+_logical_partial_derivatives)
            #--------------------------------------------------------------------
            # TODO [YG, 05.02.2021]: create 'get_test_function' and use it below:
#            field_atoms = [a for a in atoms if get_test_function(a) in fields]
            field_atoms = []
            #--------------------------------------------------------------------
            atomic_expr_field = {f:[] for f in fields}
            for f in field_atoms:
                a = _atomic(f, cls=atoms_types)
                assert len(a) == 1
                atomic_expr_field[a[0]].append(f)

            Fs = [get_atom_logical_derivatives(a) for a in atoms]
            d = get_max_partial_derivatives(terminal_expr, logical=True, F=Fs)
            nderiv = max(nderiv, max(d.values()))

            terminal_expr     = Matrix([[terminal_expr]])

        d_tests  = {v: {'global':       GlobalTensorQuadratureTestBasis(v),
                        'local':        LocalTensorQuadratureTestBasis(v),
                        'span':         GlobalSpanArray(v),
                        'local_span':   LocalSpanArray(v),
                        'multiplicity': multiplicity_tests[i],
                        'degrees':      tests_degrees[i],
                        'thread_span':  GlobalThreadSpanArray(v)} for i,v in enumerate(tests) }

        d_trials = {u: {'global':       GlobalTensorQuadratureTrialBasis(u),
                        'local':        LocalTensorQuadratureTrialBasis(u),
                        'span':         GlobalSpanArray(u),
                        'local_span':   LocalSpanArray(u),
                        'multiplicity': multiplicity_trials[i],
                        'degrees':      trials_degrees[i]} for i,u in enumerate(trials)}

        if isinstance(expr, Functional):
            d_fields = {f: {'global':       GlobalTensorQuadratureTestBasis(f),
                            'local':        LocalTensorQuadratureTestBasis(f),
                            'span':         GlobalSpanArray(f),
                            'local_span':   LocalSpanArray(f),
                            'multiplicity': multiplicity_fields[i],
                            'degrees':      fields_degrees[i]} for i,f in enumerate(fields)}

        else:
            d_fields = {f: {'global':     GlobalTensorQuadratureTestBasis (f),
                            'local' :     LocalTensorQuadratureTestBasis(f),
                            'span':       GlobalSpanArray(f),
                            'local_span': LocalSpanArray(f)} for i,f in enumerate(fields)}

        if mapping_space:
            f = (tests+trials+fields)[0]
            f = f.base if isinstance(f, IndexedVectorFunction) else f
            if isinstance(domain, Interface):
                f_m = f.duplicate('mapping_'+f.name)
                f_m = expand([f_m])[0]
                f_p = f.duplicate('mapping_plus_'+f.name)
                f_p = expand([f_p])[0]
                f   = (f_m, f_p)

                mapping_degrees_m      = get_degrees([f_m], mapping_space[0])
                mapping_degrees_p      = get_degrees([f_p], mapping_space[1])
                multiplicity_mapping_m = get_multiplicity([f_m], mapping_space[0].coeff_space)
                multiplicity_mapping_p = get_multiplicity([f_p], mapping_space[1].coeff_space)
                mapping_degrees        = (mapping_degrees_m, mapping_degrees_p)
                multiplicity_mapping   = (multiplicity_mapping_m, multiplicity_mapping_p)
            else:
                f  = f.duplicate('mapping_'+f.name)
                f  = expand([f])[0]
                f  = (f,)
                mapping_degrees      = (get_degrees(f, mapping_space),)
                multiplicity_mapping = (get_multiplicity(f, mapping_space.coeff_space),)

            d_mapping = {fi: {'global':       GlobalTensorQuadratureTestBasis (fi),
                              'local' :       LocalTensorQuadratureTestBasis(fi),
                              'span':         GlobalSpanArray(fi),
                              'local_span':   LocalSpanArray(fi),
                              'multiplicity': multiplicity_mapping_i[0],
                              'degrees':      mapping_degrees_i[0]}
                              for fi,mapping_degrees_i,multiplicity_mapping_i in zip(f,mapping_degrees,multiplicity_mapping) }
        else:
           d_mapping = {}

        if is_broken:
            if isinstance(domain, Interface):
                if mapping is None:
                    mapping_minus = IdentityMapping('M_{}'.format(domain.minus.domain.name), dim)
                    mapping_plus  = IdentityMapping('M_{}'.format(domain.plus.domain.name), dim)
                else:
                    mapping_minus = mapping.mappings[domain.minus.domain]
                    mapping_plus  = mapping.mappings[domain.plus.domain]

                mapping = InterfaceMapping(mapping_minus, mapping_plus)
            elif isinstance(domain, Boundary) and mapping:
                mapping = mapping.mappings[domain.domain]
            elif mapping:
                mapping = mapping.mappings[domain]

        if mapping is None:
            if isinstance(domain, Boundary):
                name = domain.domain.name
            else:
                name = domain.name
            mapping = IdentityMapping('M_{}'.format(name), dim)

        invert_quad_loop = False
        if mapping_space or (mapping.is_analytical and mapping.jacobian_expr.atoms(Symbol)):
            invert_quad_loop = True

        if is_linear:
            ast = _create_ast_linear_form(domain, terminal_expr, atomic_expr_field, tests, d_tests, fields, d_fields, constants,
                                          nderiv, dtype, mapping, d_mapping, is_rational_mapping, mapping_space,
                                          mask, tag, num_threads, invert_quad_loop, **kwargs)

        elif is_bilinear:
            ast = _create_ast_bilinear_form(domain, terminal_expr, atomic_expr_field, tests, d_tests, trials, d_trials,
                                            fields, d_fields, constants, nderiv, dtype, mapping,
                                            d_mapping, is_rational_mapping, mapping_space,  mask, tag, is_parallel,
                                            num_threads, invert_quad_loop, **kwargs)
        elif is_functional:
            ast = _create_ast_functional_form(domain, terminal_expr, atomic_expr_field, fields, d_fields, constants, nderiv,
                                              dtype, mapping, d_mapping, is_rational_mapping, mapping_space,
                                              mask, tag, num_threads, **kwargs)
        else:
            raise NotImplementedError('TODO')
        # ...

        self._expr        = ast
        self._nderiv      = nderiv
        self._domain      = domain
        self._mapping     = mapping
        self._num_threads = num_threads

    @property
    def expr(self):
        return self._expr

    @property
    def nderiv(self):
        return self._nderiv

    @property
    def domain(self):
        return self._domain

    @property
    def mapping(self):
        return self._mapping

    @property
    def dim(self):
        return self.domain.dim

    @property
    def num_threads(self):
        return self._num_threads

#==============================================================================
def _create_ast_bilinear_form(domain, terminal_expr, atomic_expr_field, tests,  d_tests, trials, d_trials, fields, d_fields,
                              constants, nderiv, dtype, mapping, d_mapping, is_rational_mapping, mapping_space,
                              mask, tag, is_parallel, num_threads, invert_quad_loop, **kwargs):

    """
    This function creates the assembly function of a bilinear form in the real case
    or of a sesquilinear form in complex case.

    Parameters
    ----------

    domain : <Domain>
        Sympde Domain object

    terminal_expr : <Matrix>
        atomic representation of the bilinear/sesquilinear form

    atomic_expr_field: <dict>
        dict  of atomic expressions of fields

    tests   : <list>
        list of tests functions

    d_tests : <dict>
        dictionary that contains the symbolic spans and basis values of each test function

    trials : <list>
        list of trial functions

    d_trials: <list>
        dictionary that contains the symbolic spans and basis values of each trial function

    fields : <list>
        list of fields

    constants : <list>
        list of constants

    nderiv : int
        the order of the bilinear/sesquilinear form

    dtype : str
        type of data 'complex' or 'float'

    mapping : <Mapping>
        Sympde Mapping object

    d_mapping : <dict>
        dictionary that contains the symbolic spans and basis values of the mapping

    is_rational_mapping : <bool>
        takes the value of True if the mapping is rational

    mask  : <int|None>
        the masked direction in case of boundary domain

    tag   : <str>
        tag to be added to variable names

    is_parallel   : <bool>
        True if the domain is distributed

    num_threads : <int>
        Number of threads

    invert_quad_loop : <bool>
        Invert the quadrature loop if True

    Returns
    -------
    node : DefNode
        represents the function definition node that computes the assembly

    """

    # Create flags for parallel case

    dim        = domain.dim
    backend    = kwargs.pop('backend')
    is_pyccel  = backend['name'] == 'pyccel' if backend else False
    add_openmp = is_pyccel and backend['openmp'] and num_threads>1

    pads       = variables(('pad1, pad2, pad3'), dtype='int')[:dim]
    g_quad     = [GlobalTensorQuadratureGrid(False)]
    l_quad     = [LocalTensorQuadratureGrid(False)]

    if isinstance(domain, Interface):
        g_quad.append(PlusGlobalTensorQuadratureGrid(False))
        l_quad.append(PlusLocalTensorQuadratureGrid(False))

    rank_from_coords = MatrixRankFromCoords()
    coords_from_rank = MatrixCoordsFromRank()

    nquads        = kwargs.pop('nquads', None)
    thread_span   =  dict((u,d_tests[u]['thread_span']) for u in tests)
    # ...........................................................................................
    # Define name of span keys if we used OpenMP or not
    if add_openmp:
        span  = 'local_span'
        basis = 'local'
    else:
        span  = 'span'
        basis = 'global'

    g_span = dict((u, d_tests[u][span])  for u in tests)
    f_span = dict((f, d_fields[f][span]) for f in fields)

    # Collect mapping span
    if mapping_space:
        m_span   = dict((f, d_mapping[f][span]) for f in d_mapping)
    else:
        m_span = {}

    eval_mappings       = []
    m_trials            = dict((u,d_trials[u]['multiplicity'])  for u in trials)
    m_tests             = dict((v,d_tests[v]['multiplicity'])   for v in tests)
    lengths_trials      = dict((u,LengthDofTrial(u)) for u in trials)
    lengths_tests       = dict((v,LengthDofTest(v)) for v in tests)
    lengths_fields      = dict((f,LengthDofTest(f)) for f in fields)

    # Those dictionaries were defined but never used
    # m_trials            = dict((u,d_trials[u]['multiplicity'])  for u in trials)
    # lengths_outer_tests = dict((v,LengthOuterDofTest(v)) for v in tests)
    # lengths_inner_tests = dict((v,LengthInnerDofTest(v)) for v in tests)

    # ...........................................................................................
    quad_length     = LengthQuadrature()
    el_length       = LengthElement()
    global_thread_s = GlobalThreadStarts()
    global_thread_e = GlobalThreadEnds()
    global_thread_l = GlobalThreadSizes()
    local_thread_s  = LocalThreadStarts()
    local_thread_e  = LocalThreadEnds()
    lengths         = [el_length, quad_length]

    # ...........................................................................................
    # Get the Geometry expression from the mapping
    if isinstance(domain, Interface):
        geos = [GeometryExpressions(mapping.minus, nderiv), GeometryExpressions(mapping.plus, nderiv)]
    else:
        geos = [GeometryExpressions(mapping, nderiv)]

    # Define the global and local matrices
    g_coeffs   = {f:[MatrixGlobalBasis(i, i, dtype=dtype) for i in expand([f])] for f in fields} #dtype manage the initialization at 0
    l_mats     = BlockStencilMatrixLocalBasis(trials, tests, terminal_expr, dim, tag, dtype=dtype) #dtype manage the reset at 0
    g_mats     = BlockStencilMatrixGlobalBasis(trials, tests, pads, m_tests, terminal_expr, l_mats.tag, dtype=dtype) #dtype manage the decorators type in pyccel
    # ...........................................................................................

    if nquads is not None:
        ind_quad      = index_quad.set_range(stop=Tuple(*nquads))
    else:
        ind_quad      = index_quad.set_range(stop=quad_length)

    # g_starts        = Tuple(*[ProductGenerator(global_thread_s.set_index(i), thread_coords.set_index(i)) for i in range(dim)])
    # g_ends          = Tuple(*[AddNode(ProductGenerator(global_thread_e.set_index(i), thread_coords.set_index(i)), Integer(1)) for i in range(dim)])
    l_starts        = Tuple(*[ProductGenerator(local_thread_s.set_index(i), local_index_element.set_index(i)) for i in range(dim)])
    l_ends          = Tuple(*[ProductGenerator(local_thread_e.set_index(i), local_index_element.set_index(i)) for i in range(dim)])

    #ind_element   = index_element.set_range(start=g_starts,stop=g_ends) if add_openmp else index_element.set_range(stop=el_length)
    ind_element   = index_element.set_range(start=l_starts,stop=l_ends) if add_openmp else index_element.set_range(stop=el_length)
    l_ind_element = local_index_element.set_range(stop=TensorInteger(2))

    # Create mapping loop if the user give a mapping of an interface
    if mapping_space and isinstance(domain, Interface):
        is_trial = (kwargs["is_trial"][domain.minus], kwargs["is_trial"][domain.plus])
        mappings = (mapping.minus, mapping.plus)
        ind_dof_tests = [index_dof_test.set_range(stop=Tuple(*[d+1 for d in d_mapping[f]['degrees']])) for f in d_mapping]
        # ...........................................................................................
        eval_mappings = [EvalMapping(domain, ind_quad, ind_dof_tests[i], d_mapping[fi][basis],
                                     mappings[i], geos[i], mapping_space[i], nderiv, mask,
                                     is_rational_mapping[i], trial=is_trial[i], quad_loop=(not invert_quad_loop)) for i,fi in enumerate(d_mapping)]

    # Create mapping loop if the user give a mapping of a domain
    elif mapping_space:
        ind_dof_tests  = [index_dof_test.set_range(stop=Tuple(*[d+1 for d in d_mapping[f]['degrees']])) for f in d_mapping]
        # ...........................................................................................
        eval_mappings = [EvalMapping(domain, ind_quad, ind_dof_tests[i], d_mapping[fi][basis],
                        mapping, geos[i], mapping_space, nderiv, mask, is_rational_mapping,
                        quad_loop=(not invert_quad_loop)) for i,fi in enumerate(d_mapping)]


    # Create Evaluating loop for each field
    eval_fields = []
    for f in fields:
        f_ex         = expand([f])
        coeffs       = [CoefficientBasis(i)                 for i in f_ex]
        l_coeffs     = [MatrixLocalBasis(i, dtype=dtype)    for i in f_ex] #dtype manage the initialization at 0 in the evaluating loop
        ind_dof_test = index_dof_test.set_range(stop=lengths_fields[f]+1)
        eval_field   = EvalField(domain, atomic_expr_field[f], ind_quad, ind_dof_test, d_fields[f][basis],
                                 coeffs, l_coeffs, g_coeffs[f], [f], mapping, nderiv, mask, dtype=dtype, quad_loop=(not invert_quad_loop))

        eval_fields += [eval_field]

    # Add the Mapping loop into the geometric statements if there is one
    g_stmts = []
    if mapping_space:
        g_stmts = g_stmts + eval_mappings

    # Add the Evaluating loop into the geometric statements
    g_stmts += [*eval_fields]
    g_stmts_texpr = []

    # sort tests and trials by their space type
    test_groups  = regroup(tests)
    trial_groups = regroup(trials)

    # expand every VectorFunction into IndexedVectorFunctions
    ex_tests     = expand(tests)
    ex_trials    = expand(trials)

    #=========================================================begin kernel======================================================
    for _, sub_tests in test_groups:
        for _, sub_trials in trial_groups:
            tests_indices     = [ex_tests.index(i) for i in expand(sub_tests)]
            trials_indices    = [ex_trials.index(i) for i in expand(sub_trials)]
            sub_terminal_expr = terminal_expr[tests_indices,trials_indices]

            if is_zero(sub_terminal_expr):
                continue

            q_basis_tests  = dict((v, d_tests[v][basis])            for v in sub_tests)
            q_basis_trials = dict((u, d_trials[u][basis])           for u in sub_trials)
            m_tests        = dict((v, d_tests[v]['multiplicity'])   for v in sub_tests)
            m_trials       = dict((u, d_trials[u]['multiplicity'])  for u in sub_trials)
            tests_degree   = dict((v, d_tests[v]['degrees'])        for v in sub_tests)
            trials_degrees = dict((u, d_trials[u]['degrees'])       for u in sub_trials)
            bs             = dict()
            es             = dict()
            for v in sub_tests:
                # v_str = str(SymbolicExpr(v))
#                bs[v] = variables(('b_{v}_1, b_{v}_2, b_{v}_3'.format(v=v_str)), dtype='int')[:dim] if is_parallel else [S.Zero]*dim
#                es[v] = variables(('e_{v}_1, e_{v}_2, e_{v}_3'.format(v=v_str)), dtype='int')[:dim] if is_parallel else [S.Zero]*dim
                bs[v] = [S.Zero]*dim
                es[v] = [S.Zero]*dim

#                bs and es contain the starts and the ends of the test function loops.
#                This was an optimization when we had the ghost elements and it is not needed after removing them.
#                They are not deleted because we can still use them when the communications take more time than the calculations,
#                In that case, we can disable the communications and put back the ghost elements.
#                Usualy, the communications take more time than the calculations when the degrees are small like 1 or 2,
#                and/or the number of processes is really big which makes the elements owned by a process really small and the calculations really fast.
#                This optimization can be added after improving the communications.

            if all(a==1 for a in m_tests[sub_tests[0]]+m_trials[sub_trials[0]]):
                stmts = []
                for v in sub_tests+sub_trials:
                    stmts += construct_logical_expressions(v, nderiv)

                l_sub_mats  = BlockStencilMatrixLocalBasis(sub_trials, sub_tests, sub_terminal_expr, dim, l_mats.tag,
                                                           tests_degree=tests_degree, trials_degree=trials_degrees,
                                                           tests_multiplicity=m_tests, trials_multiplicity=m_trials,
                                                           dtype=dtype)
                l_sub_scalars =  BlockScalarLocalBasis(trials = sub_trials, tests=sub_tests, expr=sub_terminal_expr,
                                                       tag=l_mats.tag, dtype=dtype)

                if invert_quad_loop:

                    # ... loop over trials
                    length = Tuple(*[d+1 for d in trials_degrees[sub_trials[0]]])
                    ind_dof_trial = index_dof_trial.set_range(stop=length)
                    stmts.append(Reduction(None,ComputeKernelExpr(sub_terminal_expr, weights=False), ElementOf(l_sub_scalars)))
                    trials_loop  = Loop((*q_basis_tests.values(), *q_basis_trials.values()), ind_dof_trial, 
                                  stmts=[*stmts, VectorAssign(ElementOf(l_sub_mats), ElementOf(l_sub_scalars),'+')])

                    # ... loop over tests
                    length = Tuple(*[d+1 for d in tests_degree[sub_tests[0]]])
                    ends   = Tuple(*[d+1-e for d,e in zip(tests_degree[sub_tests[0]], es[sub_tests[0]])])
                    starts = Tuple(*bs[sub_tests[0]])
                    ind_dof_test = index_dof_test.set_range(start=starts, stop=ends, length=length)
                    tests_loop  = Loop((), ind_dof_test, stmts=[trials_loop])
                else:
                    # Instructions needed to retrieve the precomputed values of the
                    # fields (and their derivatives) at a single quadrature point
                    stmts += flatten([eval_field.inits for eval_field in eval_fields])

                    quadrature_loop  = Loop((*l_quad, *q_basis_tests.values(), *q_basis_trials.values(), *geos), ind_quad, stmts=stmts, mask=mask)
                    reduced_quadrature_loop  = Reduce('+', ComputeKernelExpr(sub_terminal_expr, weights=False), ElementOf(l_sub_scalars), quadrature_loop)

                    # ... loop over trials
                    length = Tuple(*[d+1 for d in trials_degrees[sub_trials[0]]])
                    ind_dof_trial = index_dof_trial.set_range(stop=length)
                    trials_loop  = Loop((), ind_dof_trial, stmts=[Reset(l_sub_scalars),reduced_quadrature_loop, VectorAssign(ElementOf(l_sub_mats), ElementOf(l_sub_scalars))])

                    # ... loop over tests
                    length = Tuple(*[d+1 for d in tests_degree[sub_tests[0]]])
                    ends   = Tuple(*[d+1-e for d,e in zip(tests_degree[sub_tests[0]], es[sub_tests[0]])])
                    starts = Tuple(*bs[sub_tests[0]])
                    ind_dof_test = index_dof_test.set_range(start=starts, stop=ends, length=length)
                    tests_loop  = Loop((), ind_dof_test, stmts=[trials_loop])

                body     = (tests_loop,)
                stmts    = Block(body)
                g_stmts += [stmts]

                # This part of the code was never used and has no impact on the result of this function.
#                 if is_parallel:
#                     ln         = Tuple(*[d-1 for d in tests_degree[sub_tests[0]]])
#                     thr_s      = Tuple(*[ProductGenerator(global_thread_s.set_index(i), Tuple(thread_coords.set_index(i))) for i in range(dim)]) if add_openmp else Tuple(*[0]*dim)
#                     start_expr =  TensorMax(TensorMul(TensorAdd(TensorMul(TensorAdd(thr_s,ind_element),
#                                   Tuple(*[-1]*dim)), ln), Tuple(*[S.Zero]*dim)),Tuple(*[S.Zero]*dim))
#
#                     start_expr = TensorAssignExpr(Tuple(*bs[sub_tests[0]]), start_expr)
#                     end_expr   = TensorMax(TensorMul(TensorAdd(TensorMul(Tuple(*[-1]*dim), el_length),
#                                            TensorAdd(TensorAdd(thr_s,ind_element),
#                                            Tuple(*tests_degree[sub_tests[0]]))), Tuple(*[S.Zero]*dim)), Tuple(*[S.Zero]*dim))
#
#                     end_expr   = TensorAssignExpr(Tuple(*es[sub_tests[0]]), end_expr)
#                     g_stmts_texpr += [start_expr, end_expr]

            else:
                l_stmts    = []
                mask_inner = [[False, True] for i in range(dim)]
                for mask_inner_i in product(*mask_inner):
                    mask_inner_i = Tuple(*mask_inner_i)
                    not_mask_inner_i = Tuple(*[not i for i in mask_inner_i])
                    stmts = []
                    for v in sub_tests+sub_trials:
                        stmts += construct_logical_expressions(v, nderiv)

                    multiplicity = Tuple(*m_tests[sub_tests[0]])
                    length = Tuple(*[(d+1) % m if T else (d+1)//m for d, m, T in zip(tests_degree[sub_tests[0]], multiplicity, mask_inner_i)])
                    ind_outer_dof_test = index_outer_dof_test.set_range(stop=length)
                    outer = Tuple(*[d//m for d, m in zip(tests_degree[sub_tests[0]], multiplicity)])
                    outer = TensorAdd(TensorMul(ind_outer_dof_test, not_mask_inner_i), TensorMul(outer, mask_inner_i))

                    l_sub_mats  = BlockStencilMatrixLocalBasis(sub_trials, sub_tests, sub_terminal_expr, dim, l_mats.tag, outer=outer,
                                                               tests_degree=tests_degree, trials_degree=trials_degrees,
                                                              tests_multiplicity=m_tests, trials_multiplicity=m_trials, dtype=dtype)

                    l_sub_scalars =  BlockScalarLocalBasis(trials = sub_trials, tests=sub_tests, expr=sub_terminal_expr, tag=l_mats.tag, dtype=dtype)

                    if invert_quad_loop:

                        # ... loop over trials
                        length_t = Tuple(*[d+1 for d in trials_degrees[sub_trials[0]]])
                        ind_dof_trial = index_dof_trial.set_range(stop=length_t)
                        stmts.append(Reduction(None,ComputeKernelExpr(sub_terminal_expr, weights=False), ElementOf(l_sub_scalars)))
                        trials_loop  = Loop((*q_basis_tests.values(), *q_basis_trials.values()), ind_dof_trial,
                                stmts=[*stmts, VectorAssign(ElementOf(l_sub_mats), ElementOf(l_sub_scalars),'+')])
     
                        rem_length = Tuple(*[(d+1)-(d+1)%m for d,m in zip(tests_degree[sub_tests[0]], multiplicity)])
                        ind_inner_dof_test = index_inner_dof_test.set_range(stop=multiplicity)
                        expr1 = TensorAdd(TensorMul(ind_outer_dof_test, multiplicity),ind_inner_dof_test)
                        expr2 = TensorAdd(rem_length, ind_outer_dof_test)
                        expr  = TensorAssignExpr(index_dof_test, TensorAdd(TensorMul(expr1,not_mask_inner_i),TensorMul(expr2, mask_inner_i)))

                        # ... loop over tests
                        tests_loop  = Loop((expr,), ind_inner_dof_test, stmts=[trials_loop], mask=mask_inner_i)
                        tests_loop  = Loop((), ind_outer_dof_test, stmts=[tests_loop])
                    else:

                        # Instructions needed to retrieve the precomputed values of the
                        # fields (and their derivatives) at a single quadrature point
                        stmts += flatten([eval_field.inits for eval_field in eval_fields])

                        quadrature_loop  = Loop((*l_quad, *q_basis_tests.values(), *q_basis_trials.values(), *geos), ind_quad, stmts=stmts, mask=mask)
                        reduced_quadrature_loop  = Reduce('+', ComputeKernelExpr(sub_terminal_expr, weights=False), ElementOf(l_sub_scalars), quadrature_loop)

                        # ... loop over trials
                        length_t = Tuple(*[d+1 for d in trials_degrees[sub_trials[0]]])
                        ind_dof_trial = index_dof_trial.set_range(stop=length_t)
                        trials_loop  = Loop((), ind_dof_trial, stmts=[Reset(l_sub_scalars), reduced_quadrature_loop, VectorAssign(ElementOf(l_sub_mats), ElementOf(l_sub_scalars))])
     
                        rem_length = Tuple(*[(d+1)-(d+1)%m for d,m in zip(tests_degree[sub_tests[0]], multiplicity)])
                        ind_inner_dof_test = index_inner_dof_test.set_range(stop=multiplicity)
                        expr1 = TensorAdd(TensorMul(ind_outer_dof_test, multiplicity),ind_inner_dof_test)
                        expr2 = TensorAdd(rem_length, ind_outer_dof_test)
                        expr  = TensorAssignExpr(index_dof_test, TensorAdd(TensorMul(expr1,not_mask_inner_i),TensorMul(expr2, mask_inner_i)))

                        # ... loop over tests
                        tests_loop  = Loop((expr,), ind_inner_dof_test, stmts=[trials_loop], mask=mask_inner_i)
                        tests_loop  = Loop((), ind_outer_dof_test, stmts=[tests_loop])

                    l_stmts += [tests_loop]

                g_stmts += [*l_stmts]

    #=========================================================end kernel=========================================================
    # Create the loop over global element code for OpenMP
    if add_openmp:
#        body = [VectorAssign(Tuple(*[ProductGenerator(thread_span[u].set_index(j), num_threads) for j in range(dim)]), 
#                             Tuple(*[AddNode(2*pads[j],ProductGenerator(g_span[u].set_index(j), AddNode(el_length.set_index(j),Integer(-1)))) for j in range(dim)])) for u in thread_span]

        body          = []
        parallel_body = []
        parallel_body += [Assign(thread_id, Function("omp_get_thread_num")())]
        parallel_body += [VectorAssign(thread_coords, Tuple(*[ProductGenerator(coords_from_rank, Tuple((thread_id, i))) for i in range(dim)]))]

        for i in range(dim):
            thr_s = ProductGenerator(global_thread_s.set_index(i), Tuple(thread_coords.set_index(i)))
            thr_e = ProductGenerator(global_thread_e.set_index(i), Tuple(thread_coords.set_index(i)))
            parallel_body += [Assign(global_thread_l.set_index(i), AddNode(AddNode(thr_e, Integer(1)), MulNode(Integer(-1),thr_s)))]

        for i in range(dim):
            lhs   = local_thread_s.set_index(i)
            #thr_s = ProductGenerator(global_thread_s.set_index(i), Tuple(thread_coords.set_index(i)))
            rhs  = Array(Tuple(0, IntDivNode(global_thread_l.set_index(i), Integer(2))))
            parallel_body += [Assign(lhs, rhs)]

        for i in range(dim):
            lhs = local_thread_e.set_index(i)
            # thr_s = ProductGenerator(global_thread_s.set_index(i), Tuple(thread_coords.set_index(i)))
            # thr_e = ProductGenerator(global_thread_e.set_index(i), Tuple(thread_coords.set_index(i)))
            rhs = Array(Tuple(IntDivNode(global_thread_l.set_index(i), Integer(2)), global_thread_l.set_index(i)))
            parallel_body += [Assign(lhs, rhs)]

        get_d = lambda v:d_tests[v]['degrees'] if v in d_tests else d_tests[v.base]['degrees']
        for i in range(dim):
            parallel_body += [Allocate(d_tests[v]['local'].set_index(i),
                              (global_thread_l.set_index(i),
                              Integer(get_d(v)[i]+1),
                              Integer(nderiv+1),
                              Integer(nquads[i]))) for v in d_tests]

            thr_s = ProductGenerator(global_thread_s.set_index(i), Tuple(thread_coords.set_index(i)))
            thr_e = ProductGenerator(global_thread_e.set_index(i), Tuple(thread_coords.set_index(i)))
            args1  = tuple([Slice(None,None)]*4)
            args2  = (Slice(thr_s, AddNode(thr_e, Integer(1))),
                      Slice(None,None),
                      Slice(None,Integer(nderiv+1)),
                      Slice(None,None))

            parallel_body += [Assign(ProductGenerator(d_tests[v]['local'].set_index(i), Tuple(args1)),
                                     ProductGenerator(d_tests[v]['global'].set_index(i),Tuple(args2)))
                                     for v in d_tests]

        get_d = lambda u:d_trials[u]['degrees'] if u in d_trials else d_trials[u.base]['degrees']
        for i in range(dim):
            parallel_body += [Allocate(d_trials[u]['local'].set_index(i),
                              (global_thread_l.set_index(i),
                              Integer(get_d(u)[i]+1),
                              Integer(nderiv+1),
                              Integer(nquads[i]))) for u in d_trials]

            thr_s = ProductGenerator(global_thread_s.set_index(i), Tuple(thread_coords.set_index(i)))
            thr_e = ProductGenerator(global_thread_e.set_index(i), Tuple(thread_coords.set_index(i)))
            args1  = tuple([Slice(None,None)]*4)
            args2  = (Slice(thr_s, AddNode(thr_e, Integer(1))),
                      Slice(None,None),
                      Slice(None,Integer(nderiv+1)),
                      Slice(None,None))

            parallel_body += [Assign(ProductGenerator(d_trials[u]['local'].set_index(i), Tuple(args1)),
                                     ProductGenerator(d_trials[u]['global'].set_index(i),Tuple(args2)))
                                     for u in d_trials]

        for i in range(dim):
            parallel_body += [Allocate(d_tests[v]['local_span'].set_index(i),
                              (global_thread_l.set_index(i),)) for v in d_tests]

            thr_s = ProductGenerator(global_thread_s.set_index(i), Tuple(thread_coords.set_index(i)))
            thr_e = ProductGenerator(global_thread_e.set_index(i), Tuple(thread_coords.set_index(i)))
            args1  = (Slice(None,None),)
            args2  = (Slice(thr_s, AddNode(thr_e, Integer(1))),)

            parallel_body += [Assign(ProductGenerator(d_tests[v]['local_span'].set_index(i), Tuple(args1)),
                                     ProductGenerator(d_tests[v]['span'].set_index(i),Tuple(args2)))
                                     for v in d_tests]


        get_d = lambda f:d_fields[f].get('degrees', [lengths_fields[f].set_index(ii) for ii in range(dim)]) if f in d_fields else\
                         d_fields[f.base].get('degrees',[lengths_fields[f].set_index(ii) for ii in range(dim)])
        for i in range(dim):
            parallel_body += [Allocate(d_fields[f]['local'].set_index(i),
                              (global_thread_l.set_index(i),
                              toInteger(get_d(f)[i]+1),
                              Integer(nderiv+1),
                              Integer(nquads[i]))) for f in d_fields]

            thr_s = ProductGenerator(global_thread_s.set_index(i), Tuple(thread_coords.set_index(i)))
            thr_e = ProductGenerator(global_thread_e.set_index(i), Tuple(thread_coords.set_index(i)))
            args1  = tuple([Slice(None,None)]*4)
            args2  = (Slice(thr_s, AddNode(thr_e, Integer(1))),
                      Slice(None,None),
                      Slice(None,Integer(nderiv+1)),
                      Slice(None,None))

            parallel_body += [Assign(ProductGenerator(d_fields[f]['local'].set_index(i), Tuple(args1)),
                                     ProductGenerator(d_fields[f]['global'].set_index(i),Tuple(args2)))
                                     for f in d_fields]

        for i in range(dim):
            parallel_body += [Allocate(d_fields[f]['local_span'].set_index(i),
                              (global_thread_l.set_index(i),)) for f in d_fields]

            thr_s = ProductGenerator(global_thread_s.set_index(i), Tuple(thread_coords.set_index(i)))
            thr_e = ProductGenerator(global_thread_e.set_index(i), Tuple(thread_coords.set_index(i)))
            args1  = (Slice(None,None),)
            args2  = (Slice(thr_s, AddNode(thr_e, Integer(1))),)

            parallel_body += [Assign(ProductGenerator(d_fields[f]['local_span'].set_index(i), Tuple(args1)),
                                     ProductGenerator(d_fields[f]['span'].set_index(i),Tuple(args2)))
                                     for f in d_fields]
        if mapping_space:
            get_d = lambda f:d_mapping[f]['degrees'] if f in d_mapping else d_mapping[f.base]['degrees']
            for i in range(dim):
                parallel_body += [Allocate(d_mapping[f]['local'].set_index(i),
                                  (global_thread_l.set_index(i),
                                  Integer(get_d(f)[i]+1),
                                  Integer(nderiv+1),
                                  Integer(nquads[i]))) for f in d_mapping]

                thr_s = ProductGenerator(global_thread_s.set_index(i), Tuple(thread_coords.set_index(i)))
                thr_e = ProductGenerator(global_thread_e.set_index(i), Tuple(thread_coords.set_index(i)))
                args1  = tuple([Slice(None,None)]*4)
                args2  = (Slice(thr_s, AddNode(thr_e, Integer(1))),
                          Slice(None,None),
                          Slice(None,Integer(nderiv+1)),
                          Slice(None,None))

                parallel_body += [Assign(ProductGenerator(d_mapping[f]['local'].set_index(i), Tuple(args1)),
                                         ProductGenerator(d_mapping[f]['global'].set_index(i),Tuple(args2)))
                                         for f in d_mapping]

            for i in range(dim):
                parallel_body += [Allocate(d_mapping[f]['local_span'].set_index(i),
                                  (global_thread_l.set_index(i),)) for f in d_mapping]

                thr_s = ProductGenerator(global_thread_s.set_index(i), Tuple(thread_coords.set_index(i)))
                thr_e = ProductGenerator(global_thread_e.set_index(i), Tuple(thread_coords.set_index(i)))
                args1  = (Slice(None,None),)
                args2  = (Slice(thr_s, AddNode(thr_e, Integer(1))),)

                parallel_body += [Assign(ProductGenerator(d_mapping[f]['local_span'].set_index(i), Tuple(args1)),
                                         ProductGenerator(d_mapping[f]['span'].set_index(i),Tuple(args2)))
                                         for f in d_mapping]
    #        for i in range(dim):
#            parallel_body += [Assign(neighbour_threads.set_index(i), ProductGenerator(rank_from_coords, 
#                     Tuple(tuple(AddNode(thread_coords.set_index(j), Integer(i==j)) for j in range(dim)))))]

#        for u in thread_span:
#            expr1 = [Span(u, index=i) for i in range(dim)]
#            expr2 = [ProductGenerator(thread_span[u].set_index(i),AddNode(neighbour_threads.set_index(i),Min(Integer(0), thread_id.length))) for i in range(dim)]
#            g_stmts += [WhileLoop(NotNode(AndNode(*[StrictLessThanNode(AddNode(p, e1), e2) for p,e1,e2 in zip(pads, expr1, expr2)])), [Assign(thread_id.length, Min(Integer(100), AddNode(thread_id.length,Integer(1))))])]
#            lhs = [ProductGenerator(thread_span[u].set_index(i), Tuple(thread_id)) for i in range(dim)]
#            rhs = TensorAdd(Tuple(*expr1), Tuple(*[Integer(0)]*dim))
#            g_stmts_texpr += [TensorAssignExpr(Tuple(*lhs), rhs)]

        if invert_quad_loop:
            # ... loop over the quadrature points
            loop   = Loop((*l_quad,), ind_quad, stmts=g_stmts, mask=mask)
            g_stmts = [Reset(l_mats), *[em.inits for em in eval_mappings], *[em.inits for em in eval_fields], loop]
        else:
            g_stmts = [*[em.inits for em in eval_mappings], *g_stmts]

         #... loop over global elements
        global_loop  = Loop((*g_quad, *g_span.values(), *m_span.values(), *f_span.values(), *g_stmts_texpr),
                      ind_element, stmts=g_stmts, mask=mask)

        global_loop_reduction = Reduce('+', l_mats, g_mats, global_loop)
        global_loop           = Loop((), l_ind_element, stmts=[Comment('#$omp barrier'), global_loop_reduction], mask=mask)

        if mask is not None:
            empty_loop = Loop((), l_ind_element, stmts=[Comment('#$omp barrier'), Continue()], mask=mask)
            global_loop = IfNode((EqNode(thread_coords.set_index(mask.axis), Integer(0)),[global_loop]),(true, [empty_loop]))

        parallel_body += [global_loop]
#        parallel_body += [VectorAssign(Tuple(*[ProductGenerator(thread_span[u].set_index(j), thread_id) for j in range(dim)]), 
#                             Tuple(*[AddNode(2*pads[j],ProductGenerator(g_span[u].set_index(j), AddNode(el_length.set_index(j),Integer(-1)))) for j in range(dim)])) for u in thread_span]
    # Create the loop over global element code if we don't use OpenMP
    else:

        if invert_quad_loop:
            # ... loop over the quadrature points
            loop   = Loop((*l_quad,), ind_quad, stmts=g_stmts, mask=mask)
            g_stmts = [Reset(l_mats), *[em.inits for em in eval_fields], *[em.inits for em in eval_mappings], loop]
        else:
            g_stmts = [*[em.inits for em in eval_mappings], *g_stmts]

        # ... loop over global elements
        global_loop  = Loop((*g_quad, *g_span.values(), *m_span.values(), *f_span.values(), *g_stmts_texpr),
                      ind_element, stmts=g_stmts, mask=mask)

        body = [Reduce('+', l_mats, g_mats, global_loop)]

    # ...
    args = {}
    args['tests_basis']  = tuple(d_tests[v]['global'] for v in tests)
    args['trial_basis']  = tuple(d_trials[u]['global'] for u in trials)
    args['spans'] = tuple(d_tests[v]['span'] for v in tests)
    args['quads'] = g_quad
    args['tests_degrees']  = lengths_tests
    args['trials_degrees'] = lengths_trials
    args['quads_degree'] = lengths
    args['global_pads']  = pads
    args['local_pads']   = Pads(tests, trials)
    args['mats']  = [g_mats]

    if add_openmp:
        args['thread_args']  = (coords_from_rank, rank_from_coords, global_thread_s, global_thread_e, thread_id.length)

    # Collect fields parameters if there is one
    if mapping_space:
        args['mapping'] = flatten([eval_mapping.coeffs for eval_mapping in eval_mappings])
        args['mapping_degrees'] = [LengthDofTest(f) for f in d_mapping]
        args['mapping_basis'] = flatten([d_mapping[f]['global'] for f in d_mapping])
        args['mapping_spans'] = flatten([d_mapping[f]['span'] for f in d_mapping])


    # Collect fields parameters if there is one
    if fields:
        args['f_span']         =  tuple(d_fields[f]['span'] for f in fields)
        args['f_coeffs']       = flatten(list(g_coeffs.values()))
        args['field_basis']    = tuple(d_fields[f]['global'] for f in fields)
        args['fields_degrees'] = lengths_fields.values()
        args['f_pads']         = [f.pads for f in eval_fields]
        fields                 = tuple(f.base if isinstance(f, IndexedVectorFunction) else f for f in fields)
        args['fields']         = tuple(dict.fromkeys(fields))

    # Collect constants if there is one in the equation
    if constants:
        args['constants'] = constants

#    args['starts'] = b0s
#    args['ends']   = e0s

    # Add the allocation for the OpenMP arguments initialization
    allocations = []
    if add_openmp:
        allocations = [[Allocate(thread_span[u].set_index(i), (Integer(1+num_threads),)) for i in range(dim)] for u in thread_span]
        allocations = [Tuple(*i) for i in allocations]

    body  = allocations + body

    # Those dictionaries were defined but never used
    # m_trials      = dict((u,d_trials[u]['multiplicity'])  for u in trials)
    # m_tests       = dict((v,d_tests[v]['multiplicity'])   for v in tests)
    # trials_degree = dict((u,d_trials[u]['degrees'])       for u in trials)
    # tests_degree  = dict((v,d_tests[v]['degrees'])        for v in tests)

    local_allocations = []
    for j,u in enumerate(ex_trials):
        for i,v in enumerate(ex_tests):
            if terminal_expr[i,j] == 0:
                continue
            td    = d_tests[v]['degrees'] if v in d_tests else d_tests[v.base]['degrees']
            trd   = d_trials[u]['degrees'] if u in d_trials else d_trials[u.base]['degrees']
            tm    = d_tests[v]['multiplicity'] if v in d_tests else d_tests[v.base]['multiplicity']
            trm   = d_trials[u]['multiplicity'] if u in d_trials else d_trials[u.base]['multiplicity']
            shape = [d+1 for d in td]
            pad   = np.array([td, trd]).max(axis=0)
            diag  = compute_diag_len(pad, trm, tm)
            shape = tuple(Integer(i) for i in (shape + list(diag)))
            mat = Allocate(StencilMatrixLocalBasis(u, v, pads, l_mats.tag, dtype=dtype), shape)
            local_allocations.append(mat)

    # Collect arguments for OpenMP if used and add the parallel code
    if add_openmp:
        shared = (*thread_span.values(), coords_from_rank, rank_from_coords, global_thread_s, global_thread_e,
                  *args['tests_basis'], *args['trial_basis'], *args['spans'], *args['quads'], g_mats)
        if mapping_space:
            shared = shared + (*args['mapping'],  *args['mapping_basis'], *args['mapping_spans'])
        if fields:
            shared = shared + (*args['f_span'], *args['f_coeffs'], *args['field_basis'])

        firstprivate = (*args['tests_degrees'].values(), *args['trials_degrees'].values(), *lengths, *pads, thread_id.length)
        if mapping_space:
            firstprivate = firstprivate + (*args['mapping_degrees'], )
        if fields:
            firstprivate = firstprivate + ( *args['fields_degrees'], *args['f_pads'])
        if constants:
            firstprivate = firstprivate + (*constants,)

        body += [ParallelBlock(default='private',
                           shared=shared,
                           firstprivate=firstprivate,
                           body=local_allocations+parallel_body)]
    else:
        body = local_allocations + body

    # Add Import OpenMP if used
    local_vars = []
    imports    = []
    if add_openmp:
        imports.append(Import('pyccel.stdlib.internal.openmp',('omp_get_thread_num', )))

    # Create the tree
    node = DefNode(f'assemble_matrix_{tag}', args, local_vars, body, imports, (), 'bilinearform', domain_dtype=dtype)

    return node

#==============================================================================
def _create_ast_linear_form(domain, terminal_expr, atomic_expr_field, tests, d_tests, fields, d_fields, constants, nderiv, dtype,
                            mapping, d_mapping, is_rational_mapping, mapping_space, mask, tag, num_threads, invert_quad_loop, **kwargs):

    """
    This function creates the assembly function of a linearform

    Parameters
    ----------

    domain : <Domain>
        Sympde Domain object

    terminal_expr : <Matrix>
        atomic representation of the linear form

    atomic_expr_field : <dict>
        dict  of atomic expressions of fields

    tests   : <list>
        list of tests functions

    d_tests : <dict>
        dictionary that contains the symbolic spans and basis values of each test function

    fields  : <list>
        list of fields

    constants : <list>
        list of constants

    nderiv : int
        the order of the bilinear form

    dtype : str
        type of data 'complex' or 'float'

    mapping : <Mapping>
        Sympde Mapping object

    d_mapping : <dict>
        dictionary that contains the symbolic spans and basis values of the mapping

    is_rational_mapping : <bool>
        takes the value of True if the mapping is rational

    mask  : <int|None>
        the masked direction in case of boundary domain

    tag   : <str>
        tag to be added to variable names

    num_threads : <int>
        Number of threads

    invert_quad_loop : <bool>
        Invert the quadrature loop if True

    Returns
    -------
    node : DefNode
        represents the function definition node that computes the assembly

    """


    # Define flags
    backend   = kwargs.pop('backend', None)
    is_pyccel = backend['name'] == 'pyccel' if backend else False
    add_openmp = is_pyccel and backend['openmp'] and num_threads>1

    dim      = domain.dim
    pads     = variables(('pad1, pad2, pad3'), dtype='int')[:dim]
    g_quad   = [GlobalTensorQuadratureGrid(False)]
    l_quad   = [LocalTensorQuadratureGrid(False)]
    geo      = GeometryExpressions(mapping, nderiv)
    g_coeffs = {f:[MatrixGlobalBasis(i, i, dtype) for i in expand([f])] for f in fields}

    rank_from_coords = MatrixRankFromCoords()
    coords_from_rank = MatrixCoordsFromRank()

    nquads        = kwargs.pop('nquads', None)
    thread_span   =  dict((u,d_tests[u]['thread_span']) for u in tests)
 
    m_tests = dict((v,d_tests[v]['multiplicity'])   for v in tests)

    # Initialize BlockVector locally and globally
    l_vecs  = BlockStencilVectorLocalBasis(tests, pads, terminal_expr, tag, dtype)
    g_vecs  = BlockStencilVectorGlobalBasis(tests, pads, m_tests, terminal_expr, l_vecs.tag, dtype)

    g_span          = dict((v,d_tests[v]['span']) for v in tests)
    f_span          = dict((f,d_fields[f]['span']) for f in fields)

    # Collect mapping span when a mapping is given by the user otherwise it returns an empty dictionary
    if mapping_space:
        m_span      = dict((f,d_mapping[f]['span']) for f in d_mapping)
    else:
        m_span = {}

    eval_mappings = []
    lengths_tests   = dict((v,LengthDofTest(v)) for v in tests)
    lengths_fields  = dict((f,LengthDofTest(f)) for f in fields)

    # ...........................................................................................
    quad_length     = LengthQuadrature()
    el_length       = LengthElement()
    global_thread_s = GlobalThreadStarts()
    global_thread_e = GlobalThreadEnds()
    global_thread_l = GlobalThreadSizes()
    local_thread_s  = LocalThreadStarts()
    local_thread_e  = LocalThreadEnds()
    lengths         = [el_length,quad_length]

    # Set index of quadrature
    if nquads is not None:
        ind_quad      = index_quad.set_range(stop=Tuple(*nquads))
    else:
        ind_quad      = index_quad.set_range(stop=quad_length)

    # g_starts        = Tuple(*[ProductGenerator(global_thread_s.set_index(i), thread_coords.set_index(i)) for i in range(dim)])
    # g_ends          = Tuple(*[AddNode(ProductGenerator(global_thread_e.set_index(i), thread_coords.set_index(i)), Integer(1)) for i in range(dim)])
    l_starts        = Tuple(*[ProductGenerator(local_thread_s.set_index(i), local_index_element.set_index(i)) for i in range(dim)])
    l_ends          = Tuple(*[ProductGenerator(local_thread_e.set_index(i), local_index_element.set_index(i)) for i in range(dim)])

#    ind_element   = index_element.set_range(start=g_starts,stop=g_ends) if add_openmp else index_element.set_range(stop=el_length)
    ind_element   = index_element.set_range(start=l_starts,stop=l_ends) if add_openmp else index_element.set_range(stop=el_length)
    l_ind_element = local_index_element.set_range(stop=TensorInteger(2))

    # Create the loop for the mapping coefficient when a mapping is given by the user
    if mapping_space:
        ind_dof_test  = index_dof_test.set_range(stop=Tuple(*[d+1 for d in list(d_mapping.values())[0]['degrees']]))
        # ...........................................................................................
        eval_mapping  = EvalMapping(domain, ind_quad, ind_dof_test, list(d_mapping.values())[0]['global'],
                        mapping, geo, mapping_space, nderiv, mask, is_rational_mapping, quad_loop=(not invert_quad_loop))


    # Create the loop for the fields coefficient
    eval_fields = []
    for f in fields:
        f_ex         = expand([f])
        coeffs       = [CoefficientBasis(i)    for i in f_ex]
        l_coeffs     = [MatrixLocalBasis(i, dtype=dtype)    for i in f_ex]
        ind_dof_test = index_dof_test.set_range(stop=lengths_fields[f]+1)

        eval_field   = EvalField(domain, atomic_expr_field[f], ind_quad, ind_dof_test, d_fields[f]['global'],
                                 coeffs, l_coeffs, g_coeffs[f], [f], mapping, nderiv, mask, dtype=dtype, quad_loop=(not invert_quad_loop))
        eval_fields += [eval_field]

    g_stmts = []
    if mapping_space:
        g_stmts.append(eval_mapping)

    g_stmts += [*eval_fields]

    # sort tests by their space type
    groups = regroup(tests)
    # expand every VectorFunction into IndexedVectorFunctions
    ex_tests = expand(tests)
    # ... 
    #=========================================================begin kernel======================================================

    for _, group in groups:
        tests_indices     = [ex_tests.index(i) for i in expand(group)]
        sub_terminal_expr = terminal_expr[tests_indices, 0]
        l_sub_vecs        = BlockStencilVectorLocalBasis(group, pads, sub_terminal_expr, l_vecs.tag, dtype=dtype)
        l_sub_scalars     = BlockScalarLocalBasis(tests=group, expr=sub_terminal_expr, tag=l_vecs.tag, dtype=dtype)

        q_basis = {v: d_tests[v]['global']  for v in group}
        if is_zero(sub_terminal_expr):
            continue
        stmts = []
        for v in group:
            stmts += construct_logical_expressions(v, nderiv)

        if invert_quad_loop:
            # ... loop over tests
            length   = lengths_tests[group[0]]
            ind_dof_test = index_dof_test.set_range(stop=length+1)
            stmts.append(Reduction(None,ComputeKernelExpr(sub_terminal_expr, weights=False), ElementOf(l_sub_scalars)))
            loop  = Loop((*q_basis.values(),), ind_dof_test, stmts=[*stmts, VectorAssign(ElementOf(l_sub_vecs), ElementOf(l_sub_scalars),'+')])
        else:

            # Instructions needed to retrieve the precomputed values of the
            # fields (and their derivatives) at a single quadrature point
            stmts += flatten([eval_field.inits for eval_field in eval_fields])

            # ... loop over quadrature points
            loop  = Loop((*l_quad, *q_basis.values(), geo), ind_quad, stmts=stmts, mask=mask)
            loop = Reduce('+', ComputeKernelExpr(sub_terminal_expr, weights=False), ElementOf(l_sub_scalars), loop)

            # ... loop over tests
            length   = lengths_tests[group[0]]
            ind_dof_test = index_dof_test.set_range(stop=length+1)
            loop  = Loop((), ind_dof_test, stmts=[Reset(l_sub_scalars),loop, VectorAssign(ElementOf(l_sub_vecs), ElementOf(l_sub_scalars))])

        body  = (loop,)
        stmts = Block(body)
        g_stmts += [stmts]
    # ...

    #=========================================================end kernel=========================================================

    # Create the loop over global elements when open_mp is used with pyccel
    if add_openmp:
        body = [VectorAssign(Tuple(*[ProductGenerator(thread_span[u].set_index(j), num_threads) for j in range(dim)]), 
                             Tuple(*[AddNode(2*pads[j],ProductGenerator(g_span[u].set_index(j), AddNode(el_length.set_index(j),Integer(-1)))) for j in range(dim)])) for u in thread_span]

        parallel_body = []
        parallel_body += [Assign(thread_id, Function("omp_get_thread_num")())]
        parallel_body += [VectorAssign(thread_coords, Tuple(*[ProductGenerator(coords_from_rank, Tuple((thread_id, i))) for i in range(dim)]))]

        for i in range(dim):
            thr_s = ProductGenerator(global_thread_s.set_index(i), Tuple(thread_coords.set_index(i)))
            thr_e = ProductGenerator(global_thread_e.set_index(i), Tuple(thread_coords.set_index(i)))
            parallel_body += [Assign(global_thread_l.set_index(i), AddNode(AddNode(thr_e, Integer(1)), MulNode(Integer(-1),thr_s)))]

        for i in range(dim):
            lhs   = local_thread_s.set_index(i)
            thr_s = ProductGenerator(global_thread_s.set_index(i), Tuple(thread_coords.set_index(i)))
            rhs  = Array(Tuple(thr_s, AddNode(thr_s, IntDivNode(global_thread_l.set_index(i), Integer(2)))))
            parallel_body += [Assign(lhs, rhs)]

        for i in range(dim):
            lhs = local_thread_e.set_index(i)
            thr_s = ProductGenerator(global_thread_s.set_index(i), Tuple(thread_coords.set_index(i)))
            thr_e = ProductGenerator(global_thread_e.set_index(i), Tuple(thread_coords.set_index(i)))
            rhs = Array(Tuple(AddNode(thr_s, IntDivNode(global_thread_l.set_index(i), Integer(2))), AddNode(thr_e,Integer(1))))
            parallel_body += [Assign(lhs, rhs)]

#        for i in range(dim):
#            parallel_body += [Assign(neighbour_threads.set_index(i), ProductGenerator(rank_from_coords, 
#                     Tuple(tuple(AddNode(thread_coords.set_index(j), Integer(i==j)) for j in range(dim)))))]

#        g_stmts_texpr = []
#        for u in thread_span:
#            expr1 = [Span(u, index=i) for i in range(dim)]
#            expr2 = [ProductGenerator(thread_span[u].set_index(i),AddNode(neighbour_threads.set_index(i),Min(Integer(0), thread_id.length))) for i in range(dim)]
#            g_stmts += [WhileLoop(NotNode(AndNode(*[StrictLessThanNode(AddNode(p, e1), e2) for p,e1,e2 in zip(pads, expr1, expr2)])), [Assign(thread_id.length, Min(Integer(100), AddNode(thread_id.length,Integer(1))))])]
#            lhs = [ProductGenerator(thread_span[u].set_index(i), Tuple(thread_id)) for i in range(dim)]
#            rhs = TensorAdd(Tuple(*expr1), Tuple(*[Integer(0)]*dim))
#            g_stmts_texpr += [TensorAssignExpr(Tuple(*lhs), rhs)]

        inits = eval_mapping.inits if mapping_space else []
        if invert_quad_loop:
            # ... loop over the quadrature points
            loop   = Loop((*l_quad,), ind_quad, stmts=g_stmts, mask=mask)
            g_stmts = flatten([Reset(l_vecs), *[em.inits for em in eval_fields], inits, loop])
        else:
            g_stmts = flatten([inits, *g_stmts])

        # ... loop over global elements
        global_elements_loop  = Loop((*g_quad, *g_span.values(), *m_span.values(), *f_span.values()), ind_element, stmts=g_stmts, mask=mask)
        # ...

        global_elements_loop_reduction = Reduce('+', l_vecs, g_vecs, global_elements_loop)
        global_elements_loop           = Loop((), l_ind_element, stmts=[Comment('#$omp barrier'), global_elements_loop_reduction], mask=mask)

        # Case where the user impose a mask
        if mask is not None:
            empty_loop = Loop((), l_ind_element, stmts=[Comment('#$omp barrier'), Continue()], mask=mask)
            global_elements_loop = IfNode((EqNode(thread_coords.set_index(mask.axis), Integer(0)), [global_elements_loop]), (true, [empty_loop]))

        parallel_body += [global_elements_loop]
#        parallel_body += [VectorAssign(Tuple(*[ProductGenerator(thread_span[u].set_index(j), thread_id) for j in range(dim)]), 
#                             Tuple(*[AddNode(2*pads[j],ProductGenerator(g_span[u].set_index(j), AddNode(el_length.set_index(j),Integer(-1)))) for j in range(dim)])) for u in thread_span]

    # Create the loop over global elements when open_mp is not used with pyccel
    else:
        inits = eval_mapping.inits if mapping_space else []
        if invert_quad_loop:
            # ... loop over the quadrature points
            loop   = Loop((*l_quad,), ind_quad, stmts=g_stmts, mask=mask)
            g_stmts = flatten([Reset(l_vecs), *[em.inits for em in eval_fields], inits, loop])

        else:
            g_stmts = flatten([inits, *g_stmts])

        # ... loop over global elements
        global_element_loop  = Loop((*g_quad, *g_span.values(), *m_span.values(), *f_span.values()), ind_element, stmts=g_stmts, mask=mask)
        # ...

        body = [Reduce('+', l_vecs, g_vecs, global_element_loop)]
    # ...

    args = dict()
    args['tests_basis']  = tuple(d_tests[v]['global']  for v in tests)
    args['spans'] = g_span.values()
    args['quads'] = g_quad
    args['tests_degrees'] = lengths_tests
    args['quads_degree'] = lengths
    args['global_pads']  = pads
    args['mats']  = [g_vecs]

    # Collect the mapping data if the user give a mapping
    if mapping_space:
        args['mapping'] = eval_mapping.coeffs
        args['mapping_degrees'] = [LengthDofTest(list(d_mapping.keys())[0])]
        args['mapping_basis'] = [list(d_mapping.values())[0]['global']]
        args['mapping_spans'] = [list(d_mapping.values())[0]['span']]


    # Collect the fields data
    if fields:
        args['f_span']         = f_span.values()
        args['f_coeffs']       = flatten(list(g_coeffs.values()))
        args['field_basis']    = tuple(d_fields[f]['global'] for f in fields)
        args['fields_degrees'] = lengths_fields.values()
        args['f_pads']         = [f.pads for f in eval_fields]
        fields                 = tuple(f.base if isinstance(f, IndexedVectorFunction) else f for f in fields)
        args['fields']         = tuple(dict.fromkeys(fields))

    # Collect the constants if the expression have some
    if constants:
        args['constants'] = constants

    # Collect the thread arguments if we are in a parallel case
    if add_openmp:
        args['thread_args']  = (coords_from_rank, rank_from_coords, global_thread_s, global_thread_e, thread_id.length)

    # Allocate space for thread arguments if we are in a parallel case
    allocations = []
    if add_openmp:
        allocations = [[Allocate(thread_span[u].set_index(i), (Integer(1+num_threads),)) for i in range(dim)] for u in thread_span]
        allocations = [Tuple(*i) for i in allocations]

    # tests_degree  = dict((v,d_tests[v]['degrees']) for v in tests)

    # Allocate space for StencilVectors
    local_allocations = []
    for i,v in enumerate(ex_tests):
        if terminal_expr[i,0] == 0:
            continue
        td    = d_tests[v]['degrees'] if v in d_tests else d_tests[v.base]['degrees']
        shape = [d+1 for d in td]
        shape = tuple(Integer(i) for i in shape)
        vec = Allocate(StencilVectorLocalBasis(v, pads, l_vecs.tag, dtype=dtype), shape)
        local_allocations.append(vec)

    body  = allocations + body

    # Add the Parallel code if it's a parallel case
    if add_openmp:
        shared = (*thread_span.values(), coords_from_rank, rank_from_coords, global_thread_s, global_thread_e,
                  *args['tests_basis'], *args['spans'], *args['quads'], g_vecs)
        if mapping_space:
            shared = shared + (*eval_mapping.coeffs,  list(d_mapping.values())[0]['global'], list(d_mapping.values())[0]['span'])
        if fields:
            shared = shared + (*f_span.values(), *args['f_coeffs'], *args['field_basis'])
        
        firstprivate = (*args['tests_degrees'].values(), *lengths, *pads, thread_id.length)

        if mapping_space:
            firstprivate = firstprivate + (*args['mapping_degrees'], )
        if fields:
            firstprivate = firstprivate + ( *args['fields_degrees'], *args['f_pads'])
        if constants:
            firstprivate = firstprivate + (*constants,)
                  
        body += [ParallelBlock(default='private',
                           shared=shared,
                           firstprivate=firstprivate,
                           body=local_allocations+parallel_body)]

    else:
        body = local_allocations + body

    local_vars = []
    imports    = []
    # Imports openmp if used
    if add_openmp:
        imports.append(Import('pyccel.stdlib.internal.openmp',('omp_get_thread_num', )))

    node = DefNode(f'assemble_vector_{tag}', args, local_vars, body, imports, (), 'linearform', domain_dtype=dtype)

    return node

#==============================================================================
def _create_ast_functional_form(domain, terminal_expr, atomic_expr, fields, d_fields, constants, nderiv,
                                dtype, mapping, d_mapping, is_rational_mapping, mapping_space, mask, tag,
                                num_threads, **kwargs):
    """
    This function creates the assembly function of a Functional Form

    Parameters
    ----------

    domain : <Domain>
        Sympde Domain object

    terminal_expr : <Matrix>
        atomic representation of the Functional form

    atomic_expr   : <dict>
        atoms used in the terminal_expr

    fields   : <list>
        list of the fields

    d_fields : <dict>
        dictionary that contains the symbolic spans and basis values of each field

    constants : <list>
        list of constants

    nderiv : int
        the order of the bilinear form

    dtype : str
        type of data 'complex' or 'float'

    mapping : <Mapping>
        Sympde Mapping object

    d_mapping : <dict>
        dictionary that contains the symbolic spans and basis values of the mapping

    is_rational_mapping : <bool>
        takes the value of True if the mapping is rational

    space : <Space>
        sympde symbolic space

    mask  : <int|None>
        the masked direction in case of boundary domain

    tag   : <str>
        tag to be added to variable names

    num_threads : <int>
        Number of threads

    Returns
    -------
    node : DefNode
        represents the function definition node that computes the assembly

    """

    # Create flags for the code

    dim       = domain.dim
    backend   = kwargs.pop('backend', None)
    is_pyccel = backend['name'] == 'pyccel' if backend else False
    add_openmp = is_pyccel and backend['openmp'] and num_threads>1

    # pads   = variables(('pad1, pad2, pad3'), dtype='int')[:dim]
    g_quad = [GlobalTensorQuadratureGrid()]
    l_quad = [LocalTensorQuadratureGrid()]

    #TODO move to EvalField
    g_coeffs = {f:[MatrixGlobalBasis(i, i, dtype=dtype) for i in expand([f])] for f in fields}

    geo      = GeometryExpressions(mapping, nderiv)

    g_span   = dict((v,d_fields[v]['span']) for v in fields)

    # Collect mapping span when a mapping is given by the user otherwise it returns an empty dictionary
    if mapping_space:
        m_span  = dict((f,d_mapping[f]['span']) for f in d_mapping)
    else:
        m_span = {}

    g_basis  = dict((v,d_fields[v]['global'])  for v in fields)

    lengths_fields  = dict((f,LengthDofTest(f)) for f in fields)

    l_vec   = LocalElementBasis()
    l_vec.dtype = dtype
    g_vec   = GlobalElementBasis()
    g_vec.dtype = dtype

    # ...........................................................................................
    quad_length = LengthQuadrature()
    el_length   = LengthElement()
    lengths     = [el_length, quad_length]

    ind_quad      = index_quad.set_range(stop=quad_length)
    ind_element   = index_element.set_range(stop=el_length)

    # Create EvalMapping when a mapping is given by the user
    if mapping_space:
        ind_dof_test  = index_dof_test.set_range(stop=Tuple(*[d+1 for d in list(d_mapping.values())[0]['degrees']]))
        # ...........................................................................................
        eval_mapping  = EvalMapping(domain, ind_quad, ind_dof_test, list(d_mapping.values())[0]['global'],
                        mapping, geo, mapping_space, nderiv, mask, is_rational_mapping, quad_loop=True)

    eval_fields = []

    # Create EvalFields for each patch
    for f in fields:
        f_ex         = expand([f])
        coeffs       = [CoefficientBasis(i)    for i in f_ex]
        l_coeffs     = [MatrixLocalBasis(i, dtype=dtype)    for i in f_ex]
        ind_dof_test = index_dof_test.set_range(stop=lengths_fields[f]+1)
        eval_field   = EvalField(domain, atomic_expr[f], ind_quad, ind_dof_test, d_fields[f]['global'],
                                 coeffs, l_coeffs, g_coeffs[f], [f], mapping, nderiv, mask, dtype=dtype, quad_loop=True)
        eval_fields  += [eval_field]

    #=========================================================begin kernel======================================================
    # ... loop over tests functions to compute the value (last loop in the dependency)

    test_function_loop          = Loop((*l_quad, geo), ind_quad, stmts=flatten([eval_field.inits for eval_field in eval_fields]))
    reduced_test_function_loop  = Reduce('+', ComputeKernelExpr(terminal_expr), ElementOf(l_vec), test_function_loop)

    # ... loop over tests functions to evaluate the fields (first loop)
    kernel_stmts  = []
    if mapping_space:
        kernel_stmts += [eval_mapping.inits, eval_mapping]


    kernel_stmts += [*eval_fields, Reset(l_vec), reduced_test_function_loop]
    kernel_stmts  = Block(kernel_stmts)

    #=========================================================end kernel=========================================================
    args = {}

    args['tests_basis']  = g_basis.values()
    args['spans'] = g_span.values()
    args['quads'] = g_quad
    args['tests_degrees'] = lengths_fields
    args['quads_degree']  = lengths
    args['global_pads']   = [f.pads for f in eval_fields]
    args['mats']          = []

    # Collect mapping data when a mapping is given by the user
    if mapping_space:
        args['mapping']         = eval_mapping.coeffs
        args['mapping_degrees'] = [LengthDofTest(list(d_mapping.keys())[0])]
        args['mapping_basis']   = [list(d_mapping.values())[0]['global']]
        args['mapping_spans']   = [list(d_mapping.values())[0]['span']]

    args['f_coeffs'] = flatten(list(g_coeffs.values()))
    fields           = tuple(f.base if isinstance(f, IndexedVectorFunction) else f for f in fields)
    args['fields']   = tuple(dict.fromkeys(fields))

    # Collect constant when there is constant in the expression
    if constants:
        args['constants'] = constants

    # Case where openmp is used
    if add_openmp:
        shared = (*args['tests_basis'], *args['spans'], *args['quads'], *args['f_coeffs'], g_vec)
        if mapping_space:
            shared = shared + (*eval_mapping.coeffs,  list(d_mapping.values())[0]['global'], list(d_mapping.values())[0]['span'])

        firstprivate = (*args['tests_degrees'].values(), *lengths, *args['global_pads'])

        if mapping_space:
            firstprivate = firstprivate + (*args['mapping_degrees'], )
        if constants:
            firstprivate = firstprivate + (*constants,)
 
        global_element_loop  = Loop(iterable=(*g_quad, *g_span.values(), *m_span.values()), index=ind_element, stmts=kernel_stmts,
                      parallel=True, default='private', shared=shared, firstprivate=firstprivate)
    else:
        global_element_loop  = Loop(iterable=(*g_quad, *g_span.values(), *m_span.values()), index=ind_element, stmts=kernel_stmts)
    # ...

    body = (Reset(g_vec), Reduce('+', l_vec, g_vec, global_element_loop), Return(g_vec))

    local_vars = []
    node = DefNode(f'assemble_scalar_{tag}', args, local_vars, body, (), (g_vec,), 'functionalform', domain_dtype=dtype)

    return node
