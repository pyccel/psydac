# -*- coding: UTF-8 -*-

from collections import OrderedDict
from itertools   import groupby

from sympy import Basic
from sympy import Matrix, ImmutableDenseMatrix
from sympy.core.containers import Tuple

from sympde.expr                 import LinearForm
from sympde.expr                 import BilinearForm
from sympde.expr                 import Functional
from sympde.topology.basic       import Boundary, Interface
from sympde.topology             import H1SpaceType, HcurlSpaceType
from sympde.topology             import HdivSpaceType, L2SpaceType, UndefinedSpaceType
from sympde.topology.space       import ScalarFunction
from sympde.topology.space       import VectorFunction
from sympde.topology.space       import IndexedVectorFunction
from sympde.topology.derivatives import _logical_partial_derivatives
from sympde.topology.derivatives import get_max_logical_partial_derivatives
from sympde.topology.mapping     import InterfaceMapping
from sympde.calculus.core        import is_zero

from pyccel.ast.core import _atomic

from .nodes import GlobalTensorQuadrature
from .nodes import LocalTensorQuadrature
from .nodes import GlobalTensorQuadratureTestBasis
from .nodes import GlobalTensorQuadratureTrialBasis
from .nodes import LengthElement, LengthQuadrature
from .nodes import LengthDofTrial, LengthDofTest
from .nodes import Reset
from .nodes import BlockStencilMatrixLocalBasis
from .nodes import BlockStencilMatrixGlobalBasis
from .nodes import BlockStencilVectorLocalBasis
from .nodes import BlockStencilVectorGlobalBasis
from .nodes import GlobalElementBasis
from .nodes import LocalElementBasis
from .nodes import GlobalSpan
from .nodes import CoefficientBasis
from .nodes import MatrixLocalBasis, MatrixGlobalBasis
from .nodes import GeometryExpressions
from .nodes import Loop
from .nodes import EvalMapping, EvalField
from .nodes import ComputeKernelExpr
from .nodes import ElementOf, Reduce
from .nodes import construct_logical_expressions
from .nodes import Pads, Mask
from .nodes import index_quad
from .nodes import index_element
from .nodes import index_dof_test
from .nodes import index_dof_trial

from psydac.api.ast.utilities import variables
from psydac.api.utilities     import flatten

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
    funcs  = OrderedDict(zip(tests, kinds))
    funcs  = sorted(funcs.items(), key=lambda x:convert(x[1]))
    grs = [OrderedDict(g) for k,g in groupby(funcs,key=lambda x:convert(x[1]))]
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
def expand_hdiv_hcurl(args):
    """
    This function expands vector functions of type hdiv and hculr into indexed functions

    """
    new_args = []
    for i in args:
        if isinstance(i, (ScalarFunction, IndexedVectorFunction)):
            new_args += [i]
        elif isinstance(i, VectorFunction):
            if isinstance(i.space.kind, (HcurlSpaceType, HdivSpaceType)):
                new_args += [i[k] for k in  range(i.space.ldim)]
            else:
                new_args += [i]
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

#==============================================================================
class DefNode(Basic):
    """
    DefNode represents a function definition where it contains the arguments and the body

    """
    def __new__(cls, name, arguments, local_variables, body):
        return Basic.__new__(cls, name, arguments, local_variables, body)

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

#==============================================================================
class AST(object):
    """
    The Ast class transforms a terminal expression returned from sympde
    into a DefNode

    """
    def __init__(self, expr, terminal_expr, spaces, tag=None, **kwargs):
        # ... compute terminal expr
        # TODO check that we have one single domain/interface/boundary

        is_bilinear   = False
        is_linear     = False
        is_functional = False
        tests         = ()
        trials        = ()
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
            mapping             = spaces.symbolic_mapping
            is_rational_mapping = spaces.is_rational_mapping
            spaces              = spaces.symbolic_space
            is_broken           = spaces.is_broken

        elif isinstance(expr, BilinearForm):
            is_bilinear         = True
            tests               = expr.test_functions
            trials              = expr.trial_functions
            fields              = expr.fields
            mapping             = spaces[0].symbolic_mapping
            is_rational_mapping = spaces[0].is_rational_mapping
            spaces              = [V.symbolic_space for V in spaces]
            is_broken           = spaces[0].is_broken

        elif isinstance(expr, Functional):
            is_functional       = True
            fields              = tuple(expr.atoms(ScalarFunction, VectorFunction))
            mapping             = spaces.symbolic_mapping
            is_rational_mapping = spaces.is_rational_mapping
            spaces              = spaces.symbolic_space
            is_broken           = spaces.is_broken

        else:
            raise NotImplementedError('TODO')

        tests  = expand_hdiv_hcurl(tests)
        trials = expand_hdiv_hcurl(trials)
        fields = expand_hdiv_hcurl(fields)

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

        d_tests  = {v: {'global': GlobalTensorQuadratureTestBasis (v), 'span': GlobalSpan(v)} for v in tests }
        d_trials = {u: {'global': GlobalTensorQuadratureTrialBasis(u), 'span': GlobalSpan(u)} for u in trials}
        d_fields = {f: {'global': GlobalTensorQuadratureTestBasis (f), 'span': GlobalSpan(f)} for f in fields}

        if is_broken:
            if is_bilinear:
                space_domain = spaces[0].domain
            else:
                space_domain = spaces.domain

            if isinstance(domain, Interface):
                mapping = InterfaceMapping(mapping[domain.minus.domain], mapping[domain.plus.domain])
            elif isinstance(domain, Boundary):
                mapping = mapping[domain.domain]
            else:
                mapping = mapping[domain]


        if is_linear:
            ast = _create_ast_linear_form(terminal_expr, atomic_expr_field, 
                                          tests, d_tests,
                                          fields, d_fields, constants,
                                          nderiv, domain.dim,
                                          mapping, is_rational_mapping, spaces, mask, tag, **kwargs)

        elif is_bilinear:
            ast = _create_ast_bilinear_form(terminal_expr, atomic_expr_field,
                                            tests, d_tests,
                                            trials, d_trials,
                                            fields, d_fields, constants,
                                            nderiv, domain.dim, 
                                            mapping, is_rational_mapping, spaces, mask, tag, **kwargs)

        elif is_functional:
            ast = _create_ast_functional_form(terminal_expr, atomic_expr_field,
                                              fields, d_fields, constants,
                                              nderiv, domain.dim, 
                                              mapping, is_rational_mapping, spaces, mask, tag, **kwargs)
        else:
            raise NotImplementedError('TODO')
        # ...

        self._expr    = ast
        self._nderiv  = nderiv
        self._domain  = domain
        self._mapping = mapping

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

#================================================================================================================================
def _create_ast_bilinear_form(terminal_expr, atomic_expr_field,
                              tests,  d_tests,
                              trials, d_trials,
                              fields, d_fields, constants,
                              nderiv, dim, mapping, is_rational_mapping, spaces, mask, tag, **kwargs):
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

    is_rational_mapping : <bool>
        takes the value of True if the mapping is rational

    spaces : <list>
        list of sympde symbolic test and trial spaces

    mask  : <int|None>
        the masked direction in case of boundary domain

    tag   : <str>
        tag to be added to variable names


    Returns
    -------
    node : DefNode
        represents the a function definition node that computes the assembly

    """

    pads       = variables(('pad1, pad2, pad3'), dtype='int')[:dim]
    g_quad     = GlobalTensorQuadrature()
    l_quad     = LocalTensorQuadrature()
    quad_order = kwargs.pop('quad_order', None)
    geo        = GeometryExpressions(mapping, nderiv)
    g_coeffs   = {f:[MatrixGlobalBasis(i,i) for i in expand([f])] for f in fields}
    l_mats     = BlockStencilMatrixLocalBasis(trials, tests, terminal_expr, dim, tag)
    g_mats     = BlockStencilMatrixGlobalBasis(trials, tests, pads, terminal_expr, l_mats.tag)

    # ...........................................................................................
    g_span          = OrderedDict((u,d_tests[u]['span']) for u in tests)
    f_span          = OrderedDict((f,d_fields[f]['span']) for f in fields)
    lengths_trials  = OrderedDict((u,LengthDofTrial(u)) for u in trials)
    lengths_tests   = OrderedDict((v,LengthDofTest(v)) for v in tests)
    lengths_fields  = OrderedDict((f,LengthDofTest(f)) for f in fields)
    # ...........................................................................................
    quad_length     = LengthQuadrature()
    el_length       = LengthElement()
    lengths         = [el_length, quad_length]

    if quad_order is not None:
        ind_quad      = index_quad.set_length(Tuple(*quad_order))
    else:
        ind_quad      = index_quad.set_length(quad_length)

    ind_element   = index_element.set_length(el_length)
    ind_dof_test  = index_dof_test.set_length(LengthDofTest(tests[0])+1)
    # ...........................................................................................
    eval_mapping = EvalMapping(ind_quad, ind_dof_test, d_tests[tests[0]]['global'], d_tests[tests[0]]['global'], mapping, geo, spaces[1], tests, nderiv, mask, is_rational_mapping)
    g_stmts      = [eval_mapping]

    for f in fields:
        f_ex         = expand([f])
        coeffs       = [CoefficientBasis(i)    for i in f_ex]
        l_coeffs     = [MatrixLocalBasis(i)    for i in f_ex]
        ind_dof_test = index_dof_test.set_length(lengths_fields[f]+1)
        eval_field   = EvalField(atomic_expr_field[f], ind_quad, ind_dof_test, d_fields[f]['global'], d_fields[f]['global'], coeffs, l_coeffs, g_coeffs[f], [f], mapping, pads, nderiv, mask)
        g_stmts     += [eval_field]

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
            l_sub_mats  = BlockStencilMatrixLocalBasis(sub_trials, sub_tests, sub_terminal_expr, dim, l_mats.tag)
            if is_zero(sub_terminal_expr):
                continue

            q_basis_tests  = OrderedDict((v,d_tests[v]['global'])  for v in sub_tests)
            q_basis_trials = OrderedDict((u,d_trials[u]['global']) for u in sub_trials)

            stmts = []
            for v in sub_tests+sub_trials:
                stmts += construct_logical_expressions(v, nderiv)

            if atomic_expr_field:
                stmts += list(eval_field.inits)

            loop  = Loop((l_quad, *q_basis_tests.values(), *q_basis_trials.values(), geo), ind_quad, stmts=stmts, mask=mask)
            loop  = Reduce('+', ComputeKernelExpr(sub_terminal_expr), ElementOf(l_sub_mats), loop)

            # ... loop over trials
            length = lengths_trials[sub_trials[0]]
            ind_dof_trial = index_dof_trial.set_length(length+1)
            loop  = Loop((), ind_dof_trial, [loop])

            # ... loop over tests
            length = lengths_tests[sub_tests[0]]
            ind_dof_test = index_dof_test.set_length(length+1)
            loop  = Loop((), ind_dof_test, [loop])
            # ...

            body  = (Reset(l_sub_mats), loop)
            stmts = Block(body)
            g_stmts += [stmts]

    #=========================================================end kernel=========================================================

    # ... loop over global elements
    loop  = Loop((g_quad, *g_span.values(), *f_span.values()),
                  ind_element, stmts=g_stmts, mask=mask)

    body = [Reduce('+', l_mats, g_mats, loop)]
    # ...
    args = OrderedDict()
    args['tests_basis']  = tuple(d_tests[v]['global'] for v in tests)
    args['trial_basis']  = tuple(d_trials[u]['global'] for u in trials)

    args['spans'] = g_span.values()
    args['quads'] = g_quad

    args['tests_degrees']  = lengths_tests
    args['trials_degrees'] = lengths_trials

    args['quads_degree'] = lengths
    args['global_pads']  = pads
    args['local_pads']   = Pads(tests, trials)

    args['mats']  = [l_mats, g_mats]

    if eval_mapping:
        args['mapping'] = eval_mapping.coeffs

    if fields:
        args['f_span']         = f_span.values()
        args['f_coeffs']       = flatten(list(g_coeffs.values()))
        args['field_basis']    = tuple(d_fields[f]['global'] for f in fields)
        args['fields_degrees'] = lengths_fields.values()
        fields                 = tuple(f.base if isinstance(f, IndexedVectorFunction) else f for f in fields)
        args['fields']         = tuple(dict.fromkeys(fields))

    if constants:
        args['constants'] = constants

    local_vars = []
    node = DefNode('assembly', args, local_vars, body)

    return node

#================================================================================================================================
def _create_ast_linear_form(terminal_expr, atomic_expr_field, tests, d_tests, fields, d_fields, constants, nderiv,
                            dim, mapping, is_rational_mapping, space, mask, tag, **kwargs):
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

    is_rational_mapping : <bool>
        takes the value of True if the mapping is rational

    spaces : <Space>
        sympde symbolic space

    mask  : <int|None>
        the masked direction in case of boundary domain

    tag   : <str>
        tag to be added to variable names


    Returns
    -------
    node : DefNode
        represents the a function definition node that computes the assembly

    """
    pads     = variables(('pad1, pad2, pad3'), dtype='int')[:dim]
    g_quad   = GlobalTensorQuadrature()
    l_quad   = LocalTensorQuadrature()
    geo      = GeometryExpressions(mapping, nderiv)
    g_coeffs = {f:[MatrixGlobalBasis(i,i) for i in expand([f])] for f in fields}

    l_vecs  = BlockStencilVectorLocalBasis(tests, pads, terminal_expr, tag)
    g_vecs  = BlockStencilVectorGlobalBasis(tests, pads, terminal_expr,l_vecs.tag)

    g_span          = OrderedDict((v,d_tests[v]['span']) for v in tests)
    f_span          = OrderedDict((f,d_fields[f]['span']) for f in fields)
    lengths_tests   = OrderedDict((v,LengthDofTest(v)) for v in tests)
    lengths_fields  = OrderedDict((f,LengthDofTest(f)) for f in fields)
    # ...........................................................................................
    quad_length = LengthQuadrature()
    el_length   = LengthElement()
    lengths     = [el_length,quad_length]

    ind_quad      = index_quad.set_length(quad_length)
    ind_element   = index_element.set_length(el_length)
    ind_dof_test = index_dof_test.set_length(LengthDofTest(tests[0])+1)
    # ...........................................................................................
    eval_mapping = EvalMapping(ind_quad, ind_dof_test, d_tests[tests[0]]['global'], d_tests[tests[0]]['global'], mapping, geo, space, tests, nderiv, mask, is_rational_mapping)
    g_stmts = [eval_mapping]

    for f in fields:
        f_ex         = expand([f])
        coeffs       = [CoefficientBasis(i)    for i in f_ex]
        l_coeffs     = [MatrixLocalBasis(i)    for i in f_ex]
        ind_dof_test = index_dof_test.set_length(lengths_fields[f]+1)
        eval_field   = EvalField(atomic_expr_field[f], ind_quad, ind_dof_test, d_fields[f]['global'], d_fields[f]['global'], coeffs, l_coeffs, g_coeffs[f], [f], mapping, pads, nderiv, mask)
        g_stmts     += [eval_field]
    # ...


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
        q_basis = {v:d_tests[v]['global']  for v in group}
        if is_zero(sub_terminal_expr):
            continue
        stmts = []
        for v in group:
            stmts += construct_logical_expressions(v, nderiv)

        if atomic_expr_field:
            stmts += list(eval_field.inits)

        loop  = Loop((l_quad, *q_basis.values(), geo), ind_quad, stmts=stmts, mask=mask)
        loop = Reduce('+', ComputeKernelExpr(sub_terminal_expr), ElementOf(l_sub_vecs), loop)

    # ... loop over tests
        length   = lengths_tests[group[0]]
        ind_dof_test = index_dof_test.set_length(length+1)
        loop  = Loop((), ind_dof_test, [loop])
        # ...
        body  = (Reset(l_sub_vecs), loop)
        stmts = Block(body)
        g_stmts += [stmts]
    # ...
    
    #=========================================================end kernel=========================================================
    # ... loop over global elements
    loop  = Loop((g_quad, *g_span.values(), *f_span.values()), ind_element, stmts=g_stmts, mask=mask)
    # ...
    body = (Reduce('+', l_vecs, g_vecs, loop),)

    args = OrderedDict()
    args['tests_basis']  = tuple(d_tests[v]['global']  for v in tests)

    args['spans'] = g_span.values()
    args['quads'] = g_quad

    args['tests_degrees'] = lengths_tests

    args['quads_degree'] = lengths
    args['global_pads']  = pads

    args['mats']  = [l_vecs, g_vecs]

    if eval_mapping:
        args['mapping'] = eval_mapping.coeffs

    if fields:
        args['f_span']         = f_span.values()
        args['f_coeffs']       = flatten(list(g_coeffs.values()))
        args['field_basis']    = tuple(d_fields[f]['global'] for f in fields)
        args['fields_degrees'] = lengths_fields.values()
        fields                 = tuple(f.base if isinstance(f, IndexedVectorFunction) else f for f in fields)
        args['fields']         = tuple(dict.fromkeys(fields))

    if constants:
        args['constants'] = constants

    local_vars = []
    node = DefNode('assembly', args, local_vars, body)

    return node

#================================================================================================================================
def _create_ast_functional_form(terminal_expr, atomic_expr, fields, d_fields, constants, nderiv,
                                dim, mapping, is_rational_mapping, space, mask, tag, **kwargs):
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

    is_rational_mapping : <bool>
        takes the value of True if the mapping is rational

    space : <Space>
        sympde symbolic space

    mask  : <int|None>
        the masked direction in case of boundary domain

    tag   : <str>
        tag to be added to variable names


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

    g_span   = OrderedDict((v,d_fields[v]['span']) for v in fields)
    g_basis  = OrderedDict((v,d_fields[v]['global'])  for v in fields)

    lengths_fields  = OrderedDict((f,LengthDofTest(f)) for f in fields)

    l_vec   = LocalElementBasis()
    g_vec   = GlobalElementBasis()

    # ...........................................................................................
    quad_length = LengthQuadrature()
    el_length   = LengthElement()
    lengths     = [el_length, quad_length]

    ind_quad      = index_quad.set_length(quad_length)
    ind_element   = index_element.set_length(el_length)

    ind_dof_test  = index_dof_test.set_length(lengths_fields[fields[0]]+1)
    # ...........................................................................................
    eval_mapping = EvalMapping(ind_quad, ind_dof_test, g_basis[fields[0]], g_basis[fields[0]], mapping, geo, space, fields, nderiv, mask, is_rational_mapping)

    eval_fields = []
    for f in fields:
        f_ex         = expand([f])
        coeffs       = [CoefficientBasis(i)    for i in f_ex]
        l_coeffs     = [MatrixLocalBasis(i)    for i in f_ex]
        ind_dof_test = index_dof_test.set_length(lengths_fields[f]+1)
        eval_field   = EvalField(atomic_expr[f], ind_quad, ind_dof_test, d_fields[f]['global'], d_fields[f]['global'], coeffs, l_coeffs, g_coeffs[f], [f], mapping, pads, nderiv, mask)
        eval_fields  += [eval_field]

    #=========================================================begin kernel======================================================
    # ... loop over tests functions

    loop   = Loop((l_quad, geo), ind_quad, flatten([eval_field.inits for eval_field in eval_fields]))
    loop   = Reduce('+', ComputeKernelExpr(terminal_expr), ElementOf(l_vec), loop)

    # ... loop over tests functions to evaluate the fields

    stmts  = Block([eval_mapping, *eval_fields, Reset(l_vec), loop])

    #=========================================================end kernel=========================================================
    # ... loop over global elements


    loop  = Loop((g_quad, *g_span.values()), ind_element, stmts)
    # ...

    body = (Reduce('+', l_vec, g_vec, loop),)

    args = OrderedDict()
    args['tests_basis']  = g_basis.values()

    args['spans'] = g_span.values()
    args['quads'] = g_quad

    args['tests_degrees'] = lengths_fields

    args['quads_degree'] = lengths
    args['global_pads']  = pads

    args['mats']  = [l_vec, g_vec]

    if eval_mapping:
        args['mapping'] = eval_mapping.coeffs

    args['f_coeffs'] = flatten(list(g_coeffs.values()))
    fields           = tuple(f.base if isinstance(f, IndexedVectorFunction) else f for f in fields)
    args['fields']   = tuple(dict.fromkeys(fields))
 
    if constants:
        args['constants'] = constants

    local_vars = []
    node = DefNode('assembly', args, local_vars, body)

    return node
