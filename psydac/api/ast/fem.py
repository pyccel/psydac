# -*- coding: UTF-8 -*-

import numpy as np
from itertools   import groupby, product

from sympy import Basic, S, Function, Integer, StrictLessThan
from sympy import Matrix, ImmutableDenseMatrix, Float
from sympy.core.containers import Tuple

from sympde.expr                 import LinearForm
from sympde.expr                 import BilinearForm
from sympde.expr                 import Functional
from sympde.topology.basic       import Boundary, Interface
from sympde.topology             import H1SpaceType, HcurlSpaceType
from sympde.topology             import HdivSpaceType, L2SpaceType, UndefinedSpaceType
from sympde.topology             import IdentityMapping, SymbolicExpr
from sympde.topology.space       import ScalarFunction
from sympde.topology.space       import VectorFunction
from sympde.topology.space       import IndexedVectorFunction
from sympde.topology.derivatives import _logical_partial_derivatives
from sympde.topology.derivatives import get_max_logical_partial_derivatives
from sympde.topology.mapping     import InterfaceMapping
from sympde.calculus.core        import is_zero

from psydac.pyccel.ast.core import _atomic, Assign, Import, AugAssign, Return
from psydac.pyccel.ast.core import Comment

from .nodes import GlobalTensorQuadrature
from .nodes import LocalTensorQuadrature
from .nodes import GlobalTensorQuadratureTestBasis
from .nodes import GlobalTensorQuadratureTrialBasis
from .nodes import LengthElement, LengthQuadrature
from .nodes import LengthDofTrial, LengthDofTest
from .nodes import LengthOuterDofTest, LengthInnerDofTest
from .nodes import Reset, ProductGenerator
from .nodes import BlockStencilMatrixLocalBasis, StencilMatrixLocalBasis
from .nodes import BlockStencilMatrixGlobalBasis, BlockScalarLocalBasis
from .nodes import BlockStencilVectorLocalBasis, StencilVectorLocalBasis
from .nodes import BlockStencilVectorGlobalBasis
from .nodes import GlobalElementBasis
from .nodes import LocalElementBasis
from .nodes import GlobalSpan, GlobalThreadSpan, Span
from .nodes import CoefficientBasis
from .nodes import MatrixLocalBasis, MatrixGlobalBasis
from .nodes import MatrixRankFromCoords, MatrixCoordsFromRank
from .nodes import GeometryExpressions
from .nodes import Loop, VectorAssign
from .nodes import EvalMapping, EvalField
from .nodes import ComputeKernelExpr
from .nodes import ElementOf, Reduce
from .nodes import construct_logical_expressions
from .nodes import Pads, Mask
from .nodes import index_quad
from .nodes import index_element
from .nodes import index_dof_test
from .nodes import index_dof_trial
from .nodes import index_outer_dof_test
from .nodes import index_inner_dof_test, thread_id, neighbour_threads
from .nodes import thread_coords, local_index_element
from .nodes import TensorIntDiv, TensorAssignExpr, TensorInteger
from .nodes import TensorAdd, TensorMul, TensorMax
from .nodes import IntDivNode, AddNode, MulNode
from .nodes import AndNode, StrictLessThanNode, WhileLoop, NotNode
from .nodes import GlobalThreadStarts, GlobalThreadEnds, GlobalThreadSizes, NumThreads
from .nodes import LocalThreadStarts, LocalThreadEnds
from .nodes import Allocate, Min, Array

from psydac.api.ast.utilities import variables
from psydac.api.utilities     import flatten
from psydac.linalg.block      import BlockVectorSpace
from psydac.fem.vector        import ProductFemSpace

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
class Block(Basic):
    """
    This class represents a Block of statements

    """
    def __new__(cls, body):
        if not isinstance(body, (list, tuple, Tuple)):
            body = [body]

        body = Tuple(*body)
        return Basic.__new__(cls, body)

    @property
    def body(self):
        return self._args[0]

class ParallelBlock(Block):
    def __new__(cls, default='private', private=(), shared=(), firstprivate=(), lastprivate=(), body=()):
        return Basic.__new__(cls, default, private, shared, firstprivate, lastprivate, body)

    @property
    def default(self):
        return self._args[0]

    @property
    def private(self):
        return self._args[1]

    @property
    def shared(self):
        return self._args[2]

    @property
    def firstprivate(self):
        return self._args[3]

    @property
    def lastprivate(self):
        return self._args[4]

    @property
    def body(self):
        return self._args[5]
#==============================================================================
class DefNode(Basic):
    """
    DefNode represents a function definition where it contains the arguments and the body

    """
    def __new__(cls, name, arguments, local_variables, body, imports, results, kind):
        return Basic.__new__(cls, name, arguments, local_variables, body, imports, results, kind)

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
def get_quad_order(Vh):
    if isinstance(Vh, ProductFemSpace):
        return get_quad_order(Vh.spaces[0])
    return tuple([g.weights.shape[1] for g in Vh.quad_grids])

#==============================================================================
class AST(object):
    """
    The Ast class transforms a terminal expression returned from sympde
    into a DefNode

    """
    def __init__(self, expr, terminal_expr, spaces, mapping_space=None, tag=None, mapping=None, is_rational_mapping=None, num_threads=1, **kwargs):
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

        domain        = terminal_expr.target
        terminal_expr = terminal_expr.expr
        dim           = domain.dim
        constants     = expr.constants
        mask          = None

        if isinstance(domain, Boundary):
            mask = Mask(domain.axis, domain.ext)

        elif isinstance(domain, Interface):
            mask = Mask(domain.axis, None)

        if isinstance(expr, LinearForm):

            is_linear           = True
            tests               = expr.test_functions
            fields              = expr.fields
            is_broken           = spaces.symbolic_space.is_broken
            quad_order          = get_quad_order(spaces)
            tests_degrees       = get_degrees(tests, spaces)
            multiplicity_tests  = get_multiplicity(tests, spaces.vector_space)
            is_parallel         = spaces.vector_space.parallel
            spaces              = spaces.symbolic_space
        elif isinstance(expr, BilinearForm):
            is_bilinear         = True
            tests               = expr.test_functions
            trials              = expr.trial_functions
            fields              = expr.fields
            is_broken           = spaces[0].symbolic_space.is_broken
            quad_order          = get_quad_order(spaces[0])
            tests_degrees       = get_degrees(tests, spaces[0])
            trials_degrees      = get_degrees(trials, spaces[1])
            multiplicity_tests  = get_multiplicity(tests, spaces[1].vector_space)
            multiplicity_trials = get_multiplicity(trials, spaces[0].vector_space)
            is_parallel         = spaces[0].vector_space.parallel
            spaces              = [V.symbolic_space for V in spaces]

        elif isinstance(expr, Functional):
            is_functional       = True
            fields              = tuple(expr.atoms(ScalarFunction, VectorFunction))
            is_broken           = spaces.symbolic_space.is_broken
            quad_order          = get_quad_order(spaces)
            fields_degrees      = get_degrees(fields, spaces)
            multiplicity_fields = get_multiplicity(fields, spaces.vector_space)
            is_parallel         = spaces.vector_space.parallel
            spaces              = spaces.symbolic_space
        else:
            raise NotImplementedError('TODO')

        tests  = expand_hdiv_hcurl(tests)
        trials = expand_hdiv_hcurl(trials)
        fields = expand_hdiv_hcurl(fields)

        kwargs['quad_order']     = quad_order

        atoms_types = (ScalarFunction, VectorFunction, IndexedVectorFunction)

        nderiv = 1
        if isinstance(terminal_expr, (ImmutableDenseMatrix, Matrix)):
            n_rows, n_cols    = terminal_expr.shape
            atomic_expr_field = {f:[] for f in fields}
            for i_row in range(0, n_rows):
                for i_col in range(0, n_cols):
                    d           = get_max_logical_partial_derivatives(terminal_expr[i_row,i_col])
                    nderiv      = max(nderiv, max(d.values()))
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

        else:
            d           = get_max_logical_partial_derivatives(terminal_expr)
            nderiv      = max(nderiv, max(d.values()))
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

            terminal_expr     = Matrix([[terminal_expr]])

        d_tests  = {v: {'global': GlobalTensorQuadratureTestBasis (v), 
                        'span': GlobalSpan(v),
                        'multiplicity':multiplicity_tests[i],
                        'degrees': tests_degrees[i],
                        'thread_span':GlobalThreadSpan(v)} for i,v in enumerate(tests) }

        d_trials = {u: {'global': GlobalTensorQuadratureTrialBasis(u), 
                        'span': GlobalSpan(u),
                        'multiplicity':multiplicity_trials[i],
                        'degrees':trials_degrees[i]} for i,u in enumerate(trials)}

        if isinstance(expr, Functional):
            d_fields = {f: {'global': GlobalTensorQuadratureTestBasis (f), 
                            'span': GlobalSpan(f),
                            'multiplicity':multiplicity_fields[i],
                            'degrees':fields_degrees[i]} for i,f in enumerate(fields)}

        else:
            d_fields = {f: {'global': GlobalTensorQuadratureTestBasis (f), 
                            'span': GlobalSpan(f)} for i,f in enumerate(fields)}

        if mapping_space:
            f         = (tests+trials+fields)[0]
            f         = f.duplicate('mapping_'+f.name)
            f         = expand([f])[0]
            mapping_degrees      = get_degrees([f], mapping_space)
            multiplicity_mapping = get_multiplicity([f], mapping_space.vector_space)
            d_mapping = {f: {'global': GlobalTensorQuadratureTestBasis (f),
                             'span': GlobalSpan(f),
                             'multiplicity':multiplicity_mapping[0],
                             'degrees': mapping_degrees[0]}}
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

        if is_linear:
            ast = _create_ast_linear_form(terminal_expr, atomic_expr_field,
                                          tests, d_tests,
                                          fields, d_fields, constants,
                                          nderiv, domain.dim,
                                          mapping, d_mapping, is_rational_mapping, spaces, mapping_space, mask, tag,
                                          num_threads, **kwargs)

        elif is_bilinear:
            ast = _create_ast_bilinear_form(terminal_expr, atomic_expr_field,
                                            tests, d_tests,
                                            trials, d_trials,
                                            fields, d_fields, constants,
                                            nderiv, domain.dim,
                                            mapping, d_mapping, is_rational_mapping, spaces, mapping_space,  mask, tag, is_parallel,
                                            num_threads, **kwargs)

        elif is_functional:
            ast = _create_ast_functional_form(terminal_expr, atomic_expr_field,
                                              fields, d_fields, constants,
                                              nderiv, domain.dim,
                                              mapping, d_mapping, is_rational_mapping, spaces, mapping_space, mask, tag,
                                              num_threads, **kwargs)
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
#================================================================================================================================
def _create_ast_bilinear_form(terminal_expr, atomic_expr_field,
                              tests,  d_tests,
                              trials, d_trials,
                              fields, d_fields, constants,
                              nderiv, dim, mapping, d_mapping, is_rational_mapping, spaces, mapping_space, mask, tag, is_parallel,
                              num_threads, **kwargs):
    """
    This function creates the assembly function of a bilinearform

    Parameters
    ----------

    terminal_expr : <Matrix>
        atomic representation of the bilinear form

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
        the order of the bilinear form

    dim : int
        number of dimension

    mapping : <Mapping>
        Sympde Mapping object

    d_mapping : <dict>
        dictionary that contains the symbolic spans and basis values of the mapping

    is_rational_mapping : <bool>
        takes the value of True if the mapping is rational

    spaces : <list>
        list of sympde symbolic test and trial spaces

    mask  : <int|None>
        the masked direction in case of boundary domain

    tag   : <str>
        tag to be added to variable names

    is_parallel   : <bool>
        True if the domain is distributed

    num_threads : <int>
        Number of threads

    Returns
    -------
    node : DefNode
        represents the a function definition node that computes the assembly

    """

    pads      = variables(('pad1, pad2, pad3'), dtype='int')[:dim]
    b0s       = variables(('b01, b02, b03'), dtype='int')[:dim]
    e0s       = variables(('e01, e02, e03'), dtype='int')[:dim]
    g_quad    = GlobalTensorQuadrature(False)
    l_quad    = LocalTensorQuadrature(False)
    rank_from_coords = MatrixRankFromCoords()
    coords_from_rank = MatrixCoordsFromRank()

    quad_order    = kwargs.pop('quad_order', None)
    thread_span   =  dict((u,d_tests[u]['thread_span']) for u in tests)
    # ...........................................................................................
    g_span              = dict((u,d_tests[u]['span']) for u in tests)
    f_span              = dict((f,d_fields[f]['span']) for f in fields)
    if mapping_space:
        m_span   = dict((f,d_mapping[f]['span']) for f in d_mapping)
    else:
        m_span = {}

    m_trials            = dict((u,d_trials[u]['multiplicity'])  for u in trials)
    m_tests             = dict((v,d_tests[v]['multiplicity'])   for v in tests)
    lengths_trials      = dict((u,LengthDofTrial(u)) for u in trials)
    lengths_tests       = dict((v,LengthDofTest(v)) for v in tests)
    lengths_outer_tests = dict((v,LengthOuterDofTest(v)) for v in tests)
    lengths_inner_tests = dict((v,LengthInnerDofTest(v)) for v in tests)
    lengths_fields      = dict((f,LengthDofTest(f)) for f in fields)
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
    geo        = GeometryExpressions(mapping, nderiv)
    g_coeffs   = {f:[MatrixGlobalBasis(i,i) for i in expand([f])] for f in fields}
    l_mats     = BlockStencilMatrixLocalBasis(trials, tests, terminal_expr, dim, tag)
    g_mats     = BlockStencilMatrixGlobalBasis(trials, tests, pads, m_tests, terminal_expr, l_mats.tag)
    # ...........................................................................................

    if quad_order is not None:
        ind_quad      = index_quad.set_range(stop=Tuple(*quad_order))
    else:
        ind_quad      = index_quad.set_range(stop=quad_length)

    g_starts        = Tuple(*[ProductGenerator(global_thread_s.set_index(i), thread_coords.set_index(i)) for i in range(dim)])
    g_ends          = Tuple(*[AddNode(ProductGenerator(global_thread_e.set_index(i), thread_coords.set_index(i)), Integer(1)) for i in range(dim)])
    l_starts        = Tuple(*[ProductGenerator(local_thread_s.set_index(i), local_index_element.set_index(i)) for i in range(dim)])
    l_ends          = Tuple(*[ProductGenerator(local_thread_e.set_index(i), local_index_element.set_index(i)) for i in range(dim)])

    #ind_element   = index_element.set_range(start=g_starts,stop=g_ends) if num_threads>1 else index_element.set_range(stop=el_length)
    ind_element   = index_element.set_range(start=l_starts,stop=l_ends) if num_threads>1 else index_element.set_range(stop=el_length)
    l_ind_element = local_index_element.set_range(stop=TensorInteger(2))
    if mapping_space:
        ind_dof_test  = index_dof_test.set_range(stop=Tuple(*[d+1 for d in list(d_mapping.values())[0]['degrees']]))
        # ...........................................................................................
        eval_mapping = EvalMapping(ind_quad, ind_dof_test, list(d_mapping.values())[0]['global'],
                        mapping, geo, mapping_space, nderiv, mask, is_rational_mapping)

    eval_fields = []
    for f in fields:
        f_ex         = expand([f])
        coeffs       = [CoefficientBasis(i)    for i in f_ex]
        l_coeffs     = [MatrixLocalBasis(i)    for i in f_ex]
        ind_dof_test = index_dof_test.set_range(stop=lengths_fields[f]+1)
        eval_field   = EvalField(atomic_expr_field[f], ind_quad, ind_dof_test, d_fields[f]['global'], 
                                 coeffs, l_coeffs, g_coeffs[f], [f], mapping, nderiv, mask)
        eval_fields += [eval_field]

    g_stmts  = []
    if mapping_space:
        g_stmts.append(eval_mapping)

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

            q_basis_tests  = dict((v,d_tests[v]['global'])         for v in sub_tests)
            q_basis_trials = dict((u,d_trials[u]['global'])        for u in sub_trials)
            m_tests        = dict((v,d_tests[v]['multiplicity'])   for v in sub_tests)
            m_trials       = dict((u,d_trials[u]['multiplicity'])  for u in sub_trials)
            tests_degree   = dict((v,d_tests[v]['degrees'])        for v in sub_tests)
            trials_degrees = dict((u,d_trials[u]['degrees'])       for u in sub_trials)
            bs             = dict()
            es             = dict()
            for v in sub_tests:
                v_str = str(SymbolicExpr(v))
                bs[v] = variables(('b_{v}_1, b_{v}_2, b_{v}_3'.format(v=v_str)), dtype='int')[:dim] if is_parallel else [S.Zero]*dim
                es[v] = variables(('e_{v}_1, e_{v}_2, e_{v}_3'.format(v=v_str)), dtype='int')[:dim] if is_parallel else [S.Zero]*dim

            if all(a==1 for a in m_tests[sub_tests[0]]+m_trials[sub_trials[0]]):
                stmts = []
                for v in sub_tests+sub_trials:
                    stmts += construct_logical_expressions(v, nderiv)

                l_sub_mats  = BlockStencilMatrixLocalBasis(sub_trials, sub_tests, sub_terminal_expr, dim, l_mats.tag,
                                                           tests_degree=tests_degree, trials_degree=trials_degrees,
                                                           tests_multiplicity=m_tests, trials_multiplicity=m_trials)
                l_sub_scalars =  BlockScalarLocalBasis(trials = sub_trials, tests=sub_tests, expr=sub_terminal_expr, tag=l_mats.tag)

                # Instructions needed to retrieve the precomputed values of the
                # fields (and their derivatives) at a single quadrature point
                stmts += flatten([eval_field.inits for eval_field in eval_fields])
            
                loop  = Loop((l_quad, *q_basis_tests.values(), *q_basis_trials.values(), geo), ind_quad, stmts=stmts, mask=mask)
                loop  = Reduce('+', ComputeKernelExpr(sub_terminal_expr, weights=False), ElementOf(l_sub_scalars), loop)

                # ... loop over trials
                length = Tuple(*[d+1 for d in trials_degrees[sub_trials[0]]])
                ind_dof_trial = index_dof_trial.set_range(stop=length)
                loop1  = Loop((), ind_dof_trial, [Reset(l_sub_scalars),loop, VectorAssign(ElementOf(l_sub_mats), ElementOf(l_sub_scalars))])

                # ... loop over tests
                length = Tuple(*[d+1 for d in tests_degree[sub_tests[0]]])
                ends   = Tuple(*[d+1-e for d,e in zip(tests_degree[sub_tests[0]], es[sub_tests[0]])])
                starts = Tuple(*bs[sub_tests[0]])
                ind_dof_test = index_dof_test.set_range(start=starts, stop=ends, length=length)
                loop  = Loop((), ind_dof_test, [loop1])
                # ...

                body  = (loop,)
                stmts = Block(body)
                g_stmts += [stmts]

                if is_parallel:
                    ln = Tuple(*[d-1 for d in tests_degree[sub_tests[0]]])
                    start_expr =  TensorMax(TensorMul(TensorAdd(TensorMul(ind_element, Tuple(*[-1]*dim)), ln), Tuple(*b0s)),Tuple(*[S.Zero]*dim))
                    start_expr = TensorAssignExpr(Tuple(*bs[sub_tests[0]]), start_expr)
                    end_expr = TensorMax(TensorMul(TensorAdd(TensorMul(Tuple(*[-1]*dim), el_length), TensorAdd(ind_element, Tuple(*tests_degree[sub_tests[0]]))), Tuple(*e0s)), Tuple(*[S.Zero]*dim))
                    end_expr = TensorAssignExpr(Tuple(*es[sub_tests[0]]), end_expr)
                    g_stmts_texpr += [start_expr, end_expr]

            else:
                l_stmts = []
                mask_inner = [[False, True] for i in range(dim)]
                for mask_inner_i in product(*mask_inner):
                    mask_inner_i = Tuple(*mask_inner_i)
                    not_mask_inner_i = Tuple(*[not i for i in mask_inner_i])
                    stmts = []
                    for v in sub_tests+sub_trials:
                        stmts += construct_logical_expressions(v, nderiv)

                    # Instructions needed to retrieve the precomputed values of the
                    # fields (and their derivatives) at a single quadrature point
                    stmts += flatten([eval_field.inits for eval_field in eval_fields])

                    multiplicity = Tuple(*m_tests[sub_tests[0]])
                    length = Tuple(*[(d+1)%m if T else (d+1)//m for d,m,T in zip(tests_degree[sub_tests[0]], multiplicity, mask_inner_i)])
                    ind_outer_dof_test = index_outer_dof_test.set_range(stop=length)
                    outer = Tuple(*[d//m for d,m in zip(tests_degree[sub_tests[0]], multiplicity)])
                    outer = TensorAdd(TensorMul(ind_outer_dof_test, not_mask_inner_i),TensorMul(outer, mask_inner_i))

                    l_sub_mats  = BlockStencilMatrixLocalBasis(sub_trials, sub_tests, sub_terminal_expr, dim, l_mats.tag, outer=outer,
                                                               tests_degree=tests_degree, trials_degree=trials_degrees,
                                                              tests_multiplicity=m_tests, trials_multiplicity=m_trials)

                    l_sub_scalars =  BlockScalarLocalBasis(trials = sub_trials, tests=sub_tests, expr=sub_terminal_expr, tag=l_mats.tag)

                    loop  = Loop((l_quad, *q_basis_tests.values(), *q_basis_trials.values(), geo), ind_quad, stmts=stmts, mask=mask)
                    loop  = Reduce('+', ComputeKernelExpr(sub_terminal_expr, weights=False), ElementOf(l_sub_scalars), loop)

                    # ... loop over trials
                    length_t = Tuple(*[d+1 for d in trials_degrees[sub_trials[0]]])
                    ind_dof_trial = index_dof_trial.set_range(stop=length_t)
                    loop  = Loop((), ind_dof_trial, [Reset(l_sub_scalars), loop, VectorAssign(ElementOf(l_sub_mats), ElementOf(l_sub_scalars))])
 
                    rem_length = Tuple(*[(d+1)-(d+1)%m for d,m in zip(tests_degree[sub_tests[0]], multiplicity)])
                    ind_inner_dof_test = index_inner_dof_test.set_range(stop=multiplicity)
                    expr1 = TensorAdd(TensorMul(ind_outer_dof_test, multiplicity),ind_inner_dof_test)
                    expr2 = TensorAdd(rem_length, ind_outer_dof_test)
                    expr  = TensorAssignExpr(index_dof_test, TensorAdd(TensorMul(expr1,not_mask_inner_i),TensorMul(expr2, mask_inner_i)))
                    loop  = Loop((expr,), ind_inner_dof_test, [loop], mask=mask_inner_i)
                    loop  = Loop((), ind_outer_dof_test, [loop])

                    l_stmts += [loop]

                g_stmts += [*l_stmts]

    #=========================================================end kernel=========================================================
    if num_threads>1:
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

#        for u in thread_span:
#            expr1 = [Span(u, index=i) for i in range(dim)]
#            expr2 = [ProductGenerator(thread_span[u].set_index(i),AddNode(neighbour_threads.set_index(i),Min(Integer(0), thread_id.length))) for i in range(dim)]
#            g_stmts += [WhileLoop(NotNode(AndNode(*[StrictLessThanNode(AddNode(p, e1), e2) for p,e1,e2 in zip(pads, expr1, expr2)])), [Assign(thread_id.length, Min(Integer(100), AddNode(thread_id.length,Integer(1))))])]
#            lhs = [ProductGenerator(thread_span[u].set_index(i), Tuple(thread_id)) for i in range(dim)]
#            rhs = TensorAdd(Tuple(*expr1), Tuple(*[Integer(0)]*dim))
#            g_stmts_texpr += [TensorAssignExpr(Tuple(*lhs), rhs)]

         #... loop over global elements
        loop  = Loop((g_quad, *g_span.values(), *m_span.values(), *f_span.values(), *g_stmts_texpr),
                      ind_element, stmts=g_stmts, mask=mask)

        loop_reduction = Reduce('+', l_mats, g_mats, loop)
        loop           = Loop((), l_ind_element, stmts=[Comment('#$omp barrier'), loop_reduction])
        parallel_body += [loop]
#        parallel_body += [VectorAssign(Tuple(*[ProductGenerator(thread_span[u].set_index(j), thread_id) for j in range(dim)]), 
#                             Tuple(*[AddNode(2*pads[j],ProductGenerator(g_span[u].set_index(j), AddNode(el_length.set_index(j),Integer(-1)))) for j in range(dim)])) for u in thread_span]
    else:
        # ... loop over global elements
        loop  = Loop((g_quad, *g_span.values(), *m_span.values(), *f_span.values(), *g_stmts_texpr),
                      ind_element, stmts=g_stmts, mask=mask)

        body = [Reduce('+', l_mats, g_mats, loop)]

    # ...
    args = {}
    args['tests_basis']  = tuple(d_tests[v]['global'] for v in tests)
    args['trial_basis']  = tuple(d_trials[u]['global'] for u in trials)

    args['spans'] = g_span.values()
    args['quads'] = g_quad

    args['tests_degrees']  = lengths_tests
    args['trials_degrees'] = lengths_trials

    args['quads_degree'] = lengths
    args['global_pads']  = pads
    args['local_pads']   = Pads(tests, trials)

    if num_threads>1:
        args['thread_args']  = (coords_from_rank, rank_from_coords, global_thread_s, global_thread_e, thread_id.length)

    args['mats']  = [g_mats]

    if mapping_space:
        args['mapping'] = eval_mapping.coeffs
        args['mapping_degrees'] = LengthDofTest(list(d_mapping.keys())[0])
        args['mapping_basis'] = list(d_mapping.values())[0]['global']
        args['mapping_spans'] = list(d_mapping.values())[0]['span']

    if fields:
        args['f_span']         = f_span.values()
        args['f_coeffs']       = flatten(list(g_coeffs.values()))
        args['field_basis']    = tuple(d_fields[f]['global'] for f in fields)
        args['fields_degrees'] = lengths_fields.values()
        args['f_pads']         = [f.pads for f in eval_fields]
        fields                 = tuple(f.base if isinstance(f, IndexedVectorFunction) else f for f in fields)
        args['fields']         = tuple(dict.fromkeys(fields))

    if constants:
        args['constants'] = constants

    args['starts'] = b0s
    args['ends']   = e0s

    allocations = []
    if num_threads>1:
        allocations = [[Allocate(thread_span[u].set_index(i), (Integer(1+num_threads),)) for i in range(dim)] for u in thread_span]
        allocations = [Tuple(*i) for i in allocations]

    m_trials      = dict((u,d_trials[u]['multiplicity'])  for u in trials)
    m_tests       = dict((v,d_tests[v]['multiplicity'])   for v in tests)
    trials_degree = dict((u,d_trials[u]['degrees'])       for u in trials)
    tests_degree  = dict((v,d_tests[v]['degrees'])        for v in tests)

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
            mat = Allocate(StencilMatrixLocalBasis(u, v, pads, l_mats.tag), shape)
            local_allocations.append(mat)

    body  = allocations + body

    if num_threads>1:
        shared = (*thread_span.values(), coords_from_rank, rank_from_coords, global_thread_s, global_thread_e,
                  *args['tests_basis'], *args['trial_basis'], *args['spans'], args['quads'], g_mats)
        if mapping_space:
            shared = shared + (*eval_mapping.coeffs,  *list(d_mapping.values())[0]['global'], *list(d_mapping.values())[0]['span'])
        if fields:
            shared = shared + (*f_span.values(), *args['f_coeffs'], *args['field_basis'])

        firstprivate = (*args['tests_degrees'].values(), *args['trials_degrees'].values(), *lengths, *pads, *b0s, *e0s, thread_id.length)
        if mapping_space:
            firstprivate = firstprivate + (args['mapping_degrees'], )
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
    if num_threads>1:
        imports.append(Import('pyccel.stdlib.internal.openmp',('omp_get_thread_num', )))

    node = DefNode('assembly', args, local_vars, body, imports, (), 'bilinearform')

    return node

#================================================================================================================================
def _create_ast_linear_form(terminal_expr, atomic_expr_field, tests, d_tests, fields, d_fields, constants, nderiv,
                            dim, mapping, d_mapping, is_rational_mapping, space, mapping_space, mask, tag, num_threads, **kwargs):
    """
    This function creates the assembly function of a linearform

    Parameters
    ----------

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

    dim : int
        number of dimension

    mapping : <Mapping>
        Sympde Mapping object

    d_mapping : <dict>
        dictionary that contains the symbolic spans and basis values of the mapping

    is_rational_mapping : <bool>
        takes the value of True if the mapping is rational

    spaces : <Space>
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
        represents the a function definition node that computes the assembly

    """
    pads     = variables(('pad1, pad2, pad3'), dtype='int')[:dim]
    g_quad   = GlobalTensorQuadrature(False)
    l_quad   = LocalTensorQuadrature(False)
    geo      = GeometryExpressions(mapping, nderiv)
    g_coeffs = {f:[MatrixGlobalBasis(i,i) for i in expand([f])] for f in fields}

    rank_from_coords = MatrixRankFromCoords()
    coords_from_rank = MatrixCoordsFromRank()

    quad_order    = kwargs.pop('quad_order', None)
    thread_span   =  dict((u,d_tests[u]['thread_span']) for u in tests)
 
    m_tests = dict((v,d_tests[v]['multiplicity'])   for v in tests)
    l_vecs  = BlockStencilVectorLocalBasis(tests, pads, terminal_expr, tag)
    g_vecs  = BlockStencilVectorGlobalBasis(tests, pads, m_tests, terminal_expr,l_vecs.tag)

    g_span          = dict((v,d_tests[v]['span']) for v in tests)
    f_span          = dict((f,d_fields[f]['span']) for f in fields)


    if mapping_space:
        m_span      = dict((f,d_mapping[f]['span']) for f in d_mapping)
    else:
        m_span = {}

    lengths_tests   = dict((v,LengthDofTest(v)) for v in tests)
    lengths_fields  = dict((f,LengthDofTest(f)) for f in fields)

    # ...........................................................................................
    quad_length = LengthQuadrature()
    el_length   = LengthElement()
    global_thread_s = GlobalThreadStarts()
    global_thread_e = GlobalThreadEnds()
    global_thread_l = GlobalThreadSizes()
    local_thread_s  = LocalThreadStarts()
    local_thread_e  = LocalThreadEnds()
    lengths     = [el_length,quad_length]

    if quad_order is not None:
        ind_quad      = index_quad.set_range(stop=Tuple(*quad_order))
    else:
        ind_quad      = index_quad.set_range(stop=quad_length)

    g_starts        = Tuple(*[ProductGenerator(global_thread_s.set_index(i), thread_coords.set_index(i)) for i in range(dim)])
    g_ends          = Tuple(*[AddNode(ProductGenerator(global_thread_e.set_index(i), thread_coords.set_index(i)), Integer(1)) for i in range(dim)])
    l_starts        = Tuple(*[ProductGenerator(local_thread_s.set_index(i), local_index_element.set_index(i)) for i in range(dim)])
    l_ends          = Tuple(*[ProductGenerator(local_thread_e.set_index(i), local_index_element.set_index(i)) for i in range(dim)])

#    ind_element   = index_element.set_range(start=g_starts,stop=g_ends) if num_threads>1 else index_element.set_range(stop=el_length)
    ind_element   = index_element.set_range(start=l_starts,stop=l_ends) if num_threads>1 else index_element.set_range(stop=el_length)
    l_ind_element = local_index_element.set_range(stop=TensorInteger(2))
    if mapping_space:
        ind_dof_test  = index_dof_test.set_range(stop=Tuple(*[d+1 for d in list(d_mapping.values())[0]['degrees']]))
        # ...........................................................................................
        eval_mapping  = EvalMapping(ind_quad, ind_dof_test, list(d_mapping.values())[0]['global'],
                        mapping, geo, mapping_space, nderiv, mask, is_rational_mapping)

    eval_fields = []
    for f in fields:
        f_ex         = expand([f])
        coeffs       = [CoefficientBasis(i)    for i in f_ex]
        l_coeffs     = [MatrixLocalBasis(i)    for i in f_ex]
        ind_dof_test = index_dof_test.set_range(stop=lengths_fields[f]+1)
        eval_field   = EvalField(atomic_expr_field[f], ind_quad, ind_dof_test, d_fields[f]['global'], coeffs, l_coeffs, g_coeffs[f], [f], mapping, nderiv, mask)
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
        sub_terminal_expr = terminal_expr[tests_indices,0]
        l_sub_vecs        = BlockStencilVectorLocalBasis(group, pads, sub_terminal_expr, l_vecs.tag)
        l_sub_scalars     =  BlockScalarLocalBasis(tests=group, expr=sub_terminal_expr, tag=l_vecs.tag)

        q_basis = {v:d_tests[v]['global']  for v in group}
        if is_zero(sub_terminal_expr):
            continue
        stmts = []
        for v in group:
            stmts += construct_logical_expressions(v, nderiv)

        # Instructions needed to retrieve the precomputed values of the
        # fields (and their derivatives) at a single quadrature point
        stmts += flatten([eval_field.inits for eval_field in eval_fields])

        loop  = Loop((l_quad, *q_basis.values(), geo), ind_quad, stmts=stmts, mask=mask)
        loop = Reduce('+', ComputeKernelExpr(sub_terminal_expr, weights=False), ElementOf(l_sub_scalars), loop)

    # ... loop over tests
        length   = lengths_tests[group[0]]
        ind_dof_test = index_dof_test.set_range(stop=length+1)
        loop  = Loop((), ind_dof_test, [Reset(l_sub_scalars),loop, VectorAssign(ElementOf(l_sub_vecs), ElementOf(l_sub_scalars))])
        # ...
        body  = (loop,)
        stmts = Block(body)
        g_stmts += [stmts]
    # ...
    
    #=========================================================end kernel=========================================================

    if num_threads>1:
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

        # ... loop over global elements
        loop  = Loop((g_quad, *g_span.values(), *m_span.values(), *f_span.values()), ind_element, stmts=g_stmts, mask=mask)
        # ...


        loop_reduction = Reduce('+', l_vecs, g_vecs, loop)
        loop           = Loop((), l_ind_element, stmts=[Comment('#$omp barrier'),loop_reduction])
        parallel_body += [loop]
#        parallel_body += [VectorAssign(Tuple(*[ProductGenerator(thread_span[u].set_index(j), thread_id) for j in range(dim)]), 
#                             Tuple(*[AddNode(2*pads[j],ProductGenerator(g_span[u].set_index(j), AddNode(el_length.set_index(j),Integer(-1)))) for j in range(dim)])) for u in thread_span]
    else:
        # ... loop over global elements
        loop  = Loop((g_quad, *g_span.values(), *m_span.values(), *f_span.values()), ind_element, stmts=g_stmts, mask=mask)
        # ...

        body = [Reduce('+', l_vecs, g_vecs, loop)]
    # ...

    args = dict()
    args['tests_basis']  = tuple(d_tests[v]['global']  for v in tests)

    args['spans'] = g_span.values()
    args['quads'] = g_quad

    args['tests_degrees'] = lengths_tests

    args['quads_degree'] = lengths
    args['global_pads']  = pads

    args['mats']  = [g_vecs]

    if mapping_space:
        args['mapping'] = eval_mapping.coeffs
        args['mapping_degrees'] = LengthDofTest(list(d_mapping.keys())[0])
        args['mapping_basis'] = list(d_mapping.values())[0]['global']
        args['mapping_spans'] = list(d_mapping.values())[0]['span']

    if fields:
        args['f_span']         = f_span.values()
        args['f_coeffs']       = flatten(list(g_coeffs.values()))
        args['field_basis']    = tuple(d_fields[f]['global'] for f in fields)
        args['fields_degrees'] = lengths_fields.values()
        args['f_pads']         = [f.pads for f in eval_fields]
        fields                 = tuple(f.base if isinstance(f, IndexedVectorFunction) else f for f in fields)
        args['fields']         = tuple(dict.fromkeys(fields))

    if constants:
        args['constants'] = constants

    if num_threads>1:
        args['thread_args']  = (coords_from_rank, rank_from_coords, global_thread_s, global_thread_e, thread_id.length)

    allocations = []
    if num_threads>1:
        allocations = [[Allocate(thread_span[u].set_index(i), (Integer(1+num_threads),)) for i in range(dim)] for u in thread_span]
        allocations = [Tuple(*i) for i in allocations]

    tests_degree  = dict((v,d_tests[v]['degrees']) for v in tests)

    local_allocations = []
    for i,v in enumerate(ex_tests):
        if terminal_expr[i,0] == 0:
            continue
        td    = d_tests[v]['degrees'] if v in d_tests else d_tests[v.base]['degrees']
        shape = [d+1 for d in td]
        shape = tuple(Integer(i) for i in shape)
        vec = Allocate(StencilVectorLocalBasis(v, pads, l_vecs.tag), shape)
        local_allocations.append(vec)

    body  = allocations + body

    if num_threads>1:
        shared = (*thread_span.values(), coords_from_rank, rank_from_coords, global_thread_s, global_thread_e,
                  *args['tests_basis'], *args['spans'], args['quads'], g_vecs)
        if mapping_space:
            shared = shared + (*eval_mapping.coeffs,  *list(d_mapping.values())[0]['global'], *list(d_mapping.values())[0]['span'])
        if fields:
            shared = shared + (*f_span.values(), *args['f_coeffs'], *args['field_basis'])
        
        firstprivate = (*args['tests_degrees'].values(), *lengths, *pads, thread_id.length)

        if mapping_space:
            firstprivate = firstprivate + (args['mapping_degrees'], )
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
    if num_threads>1:
        imports.append(Import('pyccel.stdlib.internal.openmp',('omp_get_thread_num', )))
    node = DefNode('assembly', args, local_vars, body, imports, (), 'linearform')

    return node

#================================================================================================================================
def _create_ast_functional_form(terminal_expr, atomic_expr, fields, d_fields, constants, nderiv,
                                dim, mapping, d_mapping, is_rational_mapping, space, mapping_space, mask, tag, num_threads, **kwargs):
    """
    This function creates the assembly function of a Functional Form

    Parameters
    ----------

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

    dim : int
        number of dimension

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
        represents the a function definition node that computes the assembly

    """

    pads   = variables(('pad1, pad2, pad3'), dtype='int')[:dim]
    g_quad = GlobalTensorQuadrature()
    l_quad = LocalTensorQuadrature()

    #TODO move to EvalField
    coeffs   = [CoefficientBasis(i) for i in expand(fields)]
    l_coeffs = [MatrixLocalBasis(i) for i in expand(fields)]
    g_coeffs = {f:[MatrixGlobalBasis(i,i) for i in expand([f])] for f in fields}

    geo      = GeometryExpressions(mapping, nderiv)

    g_span   = dict((v,d_fields[v]['span']) for v in fields)
    if mapping_space:
        m_span  = dict((f,d_mapping[f]['span']) for f in d_mapping)
    else:
        m_span = {}
    g_basis  = dict((v,d_fields[v]['global'])  for v in fields)

    lengths_fields  = dict((f,LengthDofTest(f)) for f in fields)

    l_vec   = LocalElementBasis()
    g_vec   = GlobalElementBasis()

    # ...........................................................................................
    quad_length = LengthQuadrature()
    el_length   = LengthElement()
    lengths     = [el_length, quad_length]

    ind_quad      = index_quad.set_range(stop=quad_length)
    ind_element   = index_element.set_range(stop=el_length)

    if mapping_space:
        ind_dof_test  = index_dof_test.set_range(stop=Tuple(*[d+1 for d in list(d_mapping.values())[0]['degrees']]))
        # ...........................................................................................
        eval_mapping  = EvalMapping(ind_quad, ind_dof_test, list(d_mapping.values())[0]['global'],
                        mapping, geo, mapping_space, nderiv, mask, is_rational_mapping)

    eval_fields = []
    for f in fields:
        f_ex         = expand([f])
        coeffs       = [CoefficientBasis(i)    for i in f_ex]
        l_coeffs     = [MatrixLocalBasis(i)    for i in f_ex]
        ind_dof_test = index_dof_test.set_range(stop=lengths_fields[f]+1)
        eval_field   = EvalField(atomic_expr[f], ind_quad, ind_dof_test, d_fields[f]['global'], coeffs, l_coeffs, g_coeffs[f], [f], mapping, nderiv, mask)
        eval_fields  += [eval_field]

    #=========================================================begin kernel======================================================
    # ... loop over tests functions

    loop   = Loop((l_quad, geo), ind_quad, flatten([eval_field.inits for eval_field in eval_fields]))
    loop   = Reduce('+', ComputeKernelExpr(terminal_expr), ElementOf(l_vec), loop)

    # ... loop over tests functions to evaluate the fields
    stmts  = []
    if mapping_space:
        stmts.append(eval_mapping)

    stmts += [*eval_fields, Reset(l_vec), loop]
    stmts  = Block(stmts)

    #=========================================================end kernel=========================================================
    # ... loop over global elements


    args = {}

    args['tests_basis']  = g_basis.values()

    args['spans'] = g_span.values()
    args['quads'] = g_quad

    args['tests_degrees'] = lengths_fields
    args['quads_degree']  = lengths
    args['global_pads']   = [f.pads for f in eval_fields]
    args['mats']          = []

    if mapping_space:
        args['mapping']         = eval_mapping.coeffs
        args['mapping_degrees'] = LengthDofTest(list(d_mapping.keys())[0])
        args['mapping_basis']   = list(d_mapping.values())[0]['global']
        args['mapping_spans']   = list(d_mapping.values())[0]['span']

    args['f_coeffs'] = flatten(list(g_coeffs.values()))
    fields           = tuple(f.base if isinstance(f, IndexedVectorFunction) else f for f in fields)
    args['fields']   = tuple(dict.fromkeys(fields))
 
    if constants:
        args['constants'] = constants

    if num_threads>1:
        shared = (*args['tests_basis'], *args['spans'], args['quads'], *args['f_coeffs'], g_vec)
        if mapping_space:
            shared = shared + (*eval_mapping.coeffs,  *list(d_mapping.values())[0]['global'], *list(d_mapping.values())[0]['span'])

        firstprivate = (*args['tests_degrees'].values(), *lengths, *args['global_pads'])

        if mapping_space:
            firstprivate = firstprivate + (args['mapping_degrees'], )
        if constants:
            firstprivate = firstprivate + (*constants,)
 
        loop  = Loop((g_quad, *g_span.values(), *m_span.values()), ind_element, stmts, 
                      parallel=True, default='private', shared=shared, firstprivate=firstprivate)
    else:
        loop  = Loop((g_quad, *g_span.values(), *m_span.values()), ind_element, stmts)
    # ...

    body = (Assign(g_vec, Float(0.)), Reduce('+', l_vec, g_vec, loop), Return(g_vec))

    local_vars = []
    node = DefNode('assembly', args, local_vars, body, (), (g_vec,), 'functionalform')

    return node
