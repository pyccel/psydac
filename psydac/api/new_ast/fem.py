from .nodes import TensorQuadrature
from .nodes import TensorBasis
from .nodes import GlobalTensorQuadrature
from .nodes import LocalTensorQuadrature
from .nodes import LocalTensorQuadratureBasis
from .nodes import LocalTensorQuadratureTestBasis
from .nodes import LocalTensorQuadratureTrialBasis
from .nodes import GlobalTensorQuadratureTestBasis
from .nodes import GlobalTensorQuadratureTrialBasis
from .nodes import TensorQuadratureBasis
from .nodes import IndexElement, IndexQuadrature
from .nodes import IndexDof, IndexDofTest, IndexDofTrial
from .nodes import IndexDerivative
from .nodes import LengthElement, LengthQuadrature
from .nodes import LengthDof, LengthDofTest, LengthDofTrial
from .nodes import SplitArray
from .nodes import Reduction
from .nodes import Reset
from .nodes import LogicalValueNode
from .nodes import TensorIteration
from .nodes import TensorIterator
from .nodes import TensorGenerator
from .nodes import ProductIteration
from .nodes import ProductIterator
from .nodes import ProductGenerator
from .nodes import StencilMatrixLocalBasis
from .nodes import StencilMatrixGlobalBasis
from .nodes import BlockStencilMatrixLocalBasis
from .nodes import BlockStencilMatrixGlobalBasis
from .nodes import BlockStencilVectorLocalBasis
from .nodes import BlockStencilVectorGlobalBasis
from .nodes import BlockMatrixNode
from .nodes import StencilVectorLocalBasis
from .nodes import StencilVectorGlobalBasis
from .nodes import GlobalElementBasis
from .nodes import LocalElementBasis
from .nodes import TensorQuadratureTestBasis, TensorQuadratureTrialBasis
from .nodes import TensorTestBasis,TensorTrialBasis
from .nodes import GlobalSpan
from .nodes import CoefficientBasis
from .nodes import MatrixLocalBasis
from .nodes import GeometryExpressions
from .nodes import Loop
from .nodes import EvalMapping, EvalField
from .nodes import WeightedVolumeQuadrature
from .nodes import ComputeLogical, ComputeKernelExpr
from .nodes import ElementOf, Reduce
from .nodes import construct_logical_expressions
from .nodes import ComputePhysicalBasis
from .nodes import Pads

#==============================================================================
from sympy import Basic
from sympy import Matrix, ImmutableDenseMatrix
from sympy import symbols
from sympy.core.containers import Tuple

from sympde.expr import TerminalExpr
from sympde.expr import LinearForm
from sympde.expr import BilinearForm
from sympde.expr import Functional

from sympde.topology             import H1SpaceType, HcurlSpaceType
from sympde.topology             import HdivSpaceType, L2SpaceType, UndefinedSpaceType
from sympde.topology             import element_of
from sympde.topology             import ScalarField
from sympde.topology             import VectorField, IndexedVectorField
from sympde.topology.space       import ScalarTestFunction
from sympde.topology.space       import VectorTestFunction
from sympde.topology.space       import IndexedTestTrial
from sympde.topology.derivatives import _partial_derivatives
from sympde.topology.derivatives import _logical_partial_derivatives
from sympde.topology.derivatives import get_max_partial_derivatives



from pyccel.ast import EmptyLine
from pyccel.ast.core      import _atomic

from .nodes import index_quad
from .nodes import index_dof, index_dof_test, index_dof_trial
from .nodes import index_element
from .nodes import index_deriv

from collections import OrderedDict
from itertools   import groupby



#==============================================================================
def convert(dtype):
    if isinstance(dtype, (H1SpaceType, UndefinedSpaceType)):
        return 0
    elif isinstance(dtype, HcurlSpaceType):
        return 1
    elif isinstance(dtype, HdivSpaceType):
        return 2
    elif isinstance(dtype, L2SpaceType):
        return 3
#==============================================================================
def regroupe(tests):
    tests  = [i.base if isinstance(i, IndexedTestTrial) else i for i in tests]
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
        if isinstance(d, (HcurlSpaceType, HdivSpaceType)):
            dim = g[0].space.ldim
            for i in range(dim):
                s = [u[i] for u in g]
                groups += [(d,tuple(s))]
        else:
            groups += [(d,g)]
    return groups
#==============================================================================
def expand(args):
    new_args = []
    for i in args:
        if isinstance(i, (ScalarTestFunction, IndexedTestTrial)):
            new_args += [i]
        elif isinstance(i, VectorTestFunction):
            new_args += [i[k] for k in  range(i.space.ldim)]
        else:
            raise NotImplementedError("TODO")
    return new_args
#==============================================================================
def expand_hdiv_hcurl(args):
    new_args = []
    for i in args:
        if isinstance(i, (ScalarTestFunction, IndexedTestTrial)):
            new_args += [i]
        elif isinstance(i, VectorTestFunction):
            if isinstance(i.space.kind, (HcurlSpaceType, HdivSpaceType)):
                new_args += [i[k] for k in  range(i.space.ldim)]
            else:
                new_args += [i]
        else:
            raise NotImplementedError("TODO")
    return new_args
#==============================================================================
def get_length(args):
    n = 0
    for i in args:
        if isinstance(i, ScalarTestFunction):
            n += 1
        elif isinstance(i, VectorTestFunction):
            n += i.space.ldim
        else:
            raise NotImplementedError("TODO")
    return n

#==============================================================================
def is_scalar_field(expr):

    if isinstance(expr, _partial_derivatives):
        return is_scalar_field(expr.args[0])

    elif isinstance(expr, _logical_partial_derivatives):
        return is_scalar_field(expr.args[0])

    elif isinstance(expr, ScalarField):
        return True

    return False

#==============================================================================
def is_vector_field(expr):

    if isinstance(expr, _partial_derivatives):
        return is_vector_field(expr.args[0])

    elif isinstance(expr, _logical_partial_derivatives):
        return is_vector_field(expr.args[0])

    elif isinstance(expr, (VectorField, IndexedVectorField)):
        return True

    return False
#==============================================================================
class Block(Basic):
    """."""
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
    """."""
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
    """
    def __init__(self, expr, spaces, Mapping):
        # ... compute terminal expr
        # TODO check that we have one single domain/interface/boundary

        is_bilinear   = False
        is_linear     = False
        is_functional = False
        tests         = []
        trials        = []
        # ...

        terminal_expr = TerminalExpr(expr)
        domain        = terminal_expr[0].target
        terminal_expr = terminal_expr[0].expr
        dim           = domain.dim

        if isinstance(expr, LinearForm):
            is_linear = True
            tests     = expr.test_functions

        elif isinstance(expr, BilinearForm):
            is_bilinear = True
            tests       = expr.test_functions
            trials      = expr.trial_functions

        elif isinstance(expr, Functional):
            is_functional = True
            fields = tuple(expr.atoms(ScalarTestFunction, VectorTestFunction))
            assert len(fields) == 1
            tests = fields[0]
            tests = Tuple(tests)
        else:
            raise NotImplementedError('TODO')

        atoms_types = (_partial_derivatives,
                       VectorTestFunction,
                       ScalarTestFunction,
                       IndexedTestTrial,
                       ScalarField,
                       VectorField, IndexedVectorField)

        nderiv = 0
        if isinstance(terminal_expr, Matrix):
            n_rows, n_cols = terminal_expr.shape
            atomic_expr = terminal_expr.zeros(n_rows, n_cols)
            atomic_expr_field = []
            for i_row in range(0, n_rows):
                for i_col in range(0, n_cols):
                    d = get_max_partial_derivatives(terminal_expr[i_row,i_col])
                    nderiv = max(nderiv, max(d.values()))
                    atoms  = _atomic(terminal_expr[i_row, i_col], cls=atoms_types)
                    fields = [atom for atom in atoms if is_scalar_field(atom) or is_vector_field(atom)]
                    atoms  = [atom for atom in atoms if atom not in fields ]
                    atomic_expr[i_row, i_col] = Tuple(*atoms)
                    atomic_expr_field += [*fields]
        else:
            d = get_max_partial_derivatives(terminal_expr)
            nderiv = max(nderiv, max(d.values()))
            atoms  = _atomic(terminal_expr, cls=atoms_types)
            atomic_expr_field = [atom for atom in atoms if is_scalar_field(atom) or is_vector_field(atom)]
            atomic_expr       = [atom for atom in atoms if atom not in atomic_expr_field ]
            atomic_expr       = Matrix([[Tuple(*atomic_expr)]])
            terminal_expr     = Matrix([[terminal_expr]])

        # ...

        d_tests = {}
        tests  = expand_hdiv_hcurl(tests)
        trials = expand_hdiv_hcurl(trials)
        for v in tests:
            d = {}
            d['global'] = GlobalTensorQuadratureTestBasis(v)
            d['local']  = LocalTensorQuadratureTestBasis(v)
            d['array']  = TensorQuadratureTestBasis(v)
            d['basis']  = TensorTestBasis(v)
            d['span']   = GlobalSpan(v)

            d_tests[v] = d
        # ...

        # ...
        d_trials = {}
        for v in trials:
            d = {}
            d['global'] = GlobalTensorQuadratureTrialBasis(v)
            d['local']  = LocalTensorQuadratureTrialBasis(v)
            d['array']  = TensorQuadratureTrialBasis(v)
            d['basis']  = TensorTrialBasis(v)
            d['span']   = GlobalSpan(v)

            d_trials[v] = d
        # ...

        shapes_tests  = {}
        shapes_trials = {}

        start = 0
        for v in tests:
            ln = 1 if isinstance(v, ScalarTestFunction) else dim
            end = start + ln
            shapes_tests[v] = (start, end)
            start = end

        start = 0
        for u in trials:
            ln = 1 if isinstance(u, ScalarTestFunction) else dim
            end = start + ln
            shapes_trials[u] = (start, end)
            start = end

        # ...
        if is_linear:
            ast = _create_ast_linear_form(terminal_expr, atomic_expr, atomic_expr_field, 
                                          tests, d_tests, shapes_tests,
                                          nderiv, domain.dim, 
                                          Mapping, spaces)

        elif is_bilinear:
            ast = _create_ast_bilinear_form(terminal_expr, atomic_expr, atomic_expr_field,
                                            tests, d_tests, shapes_tests,
                                            trials, d_trials, shapes_trials,
                                            nderiv, domain.dim, 
                                            Mapping, spaces)

        elif is_functional:
            ast = _create_ast_functional_form(terminal_expr, atomic_expr_field,
                                              tests, d_tests, shapes_tests,
                                              nderiv, domain.dim, 
                                              Mapping, spaces)
        else:
            raise NotImplementedError('TODO')
        # ...

        self._expr   = ast
        self._nderiv = nderiv
        self._domain = domain

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
    def dim(self):
        return self.domain.dim
#==============================================================================
#TODO add the parallel case

#================================================================================================================================
def _create_ast_bilinear_form(terminal_expr, atomic_expr, atomic_expr_field,
                              tests, d_tests, shapes_tests,
                              trials, d_trials, shapes_trials,
                              nderiv, dim, Mapping, spaces):
    """
    """

    pads   = symbols('pad1, pad2, pad3')[:dim]
    g_quad = GlobalTensorQuadrature()
    l_quad = LocalTensorQuadrature()
    #TODO should we use tests or trials for fields
    coeffs   = [CoefficientBasis(i) for i in tests]
    l_coeffs = [MatrixLocalBasis(i) for i in tests]
    geo      = GeometryExpressions(Mapping, nderiv)

    l_mats  = BlockStencilMatrixLocalBasis(trials, tests, terminal_expr, dim)
    g_mats  = BlockStencilMatrixGlobalBasis(trials, tests, pads, terminal_expr, l_mats.tag)

    q_basis_tests  = {v:d_tests[v]['array']  for v in tests}
    q_basis_trials = {u:d_trials[u]['array'] for u in trials}

    l_basis_tests  = {v:d_tests[v]['local']  for v in tests}
    l_basis_trials = {u:d_trials[u]['local'] for u in trials}

    g_basis_tests  = {v:d_tests[v]['global']  for v in tests}
    g_basis_trials = {u:d_trials[u]['global'] for u in trials}

    g_span          = {u:d_tests[u]['span'] for u in tests}

    lengths_trials  = {u:LengthDofTrial(u) for u in trials}
    lengths_tests   = {v:LengthDofTest(v) for v in tests}
    # ...........................................................................................
    quad_length = LengthQuadrature()
    el_length   = LengthElement()
    lengths     = [el_length,quad_length]

    ind_quad      = index_quad.set_length(quad_length)
    ind_element   = index_element.set_length(el_length)
    ind_dof_trial = index_dof_trial.set_length(LengthDofTrial(trials[0]))
    # ...........................................................................................
    eval_mapping = EvalMapping(ind_quad, ind_dof_trial, q_basis_trials[trials[0]], l_basis_trials[trials[0]], Mapping, geo, spaces[1], nderiv)

    if atomic_expr_field:
        eval_field   = EvalField(atomic_expr_field, ind_quad, q_basis_trials[trials[0]], coeff, trials, nderiv)

    # ... loop over tests to evaluate fields
    fields = EmptyLine()
    if atomic_expr_field:
        fields = Loop((*l_basis_trials, l_coeff), index_dof, [eval_field])

    g_stmts = [eval_mapping, fields]
    test_groups  = regroupe(tests)
    trial_groups = regroupe(trials)
    ex_tests     = expand(tests)
    ex_trials    = expand(trials)


    #=========================================================begin kernel======================================================

    for te_dtype,sub_tests in test_groups:
        for tr_dtype, sub_trials in trial_groups:

            tests_indices = [ex_tests.index(i) for i in expand(sub_tests)]
            trials_indices = [ex_trials.index(i) for i in expand(sub_trials)]
            sub_terminal_expr = terminal_expr[tests_indices,trials_indices]
            sub_atomic_expr   = atomic_expr[tests_indices,trials_indices]
            l_sub_mats  = BlockStencilMatrixLocalBasis(sub_trials,sub_tests, pads, sub_terminal_expr, l_mats.tag)
            if sub_terminal_expr.is_zero:
                continue
            q_basis_tests  = {v:d_tests[v]['array']  for v in sub_tests}
            q_basis_trials = {u:d_trials[u]['array'] for u in sub_trials}

            l_basis_tests  = {v:d_tests[v]['local']  for v in sub_tests}
            l_basis_trials = {u:d_trials[u]['local'] for u in sub_trials}

            stmts = []
            for v in sub_tests+sub_trials:
                stmts += construct_logical_expressions(v, nderiv)
            
            for expr in sub_atomic_expr[:]:
                stmts += [ComputePhysicalBasis(i) for i in expr]
            # ...

            if atomic_expr_field:
                stmts += list(eval_fiel.inits)

            loop  = Loop((l_quad, *q_basis_tests.values(), *q_basis_trials.values(), geo), ind_quad, stmts)
            loop  = Reduce('+', ComputeKernelExpr(sub_terminal_expr), ElementOf(l_sub_mats), loop)

            # ... loop over trials
            for u in l_basis_trials:
                stmts = [loop]
                length = lengths_trials[u]
                ind_dof_trial = index_dof_trial.set_length(length)
                loop  = Loop(l_basis_trials[u], ind_dof_trial, stmts)

            # ... loop over tests
            for v in l_basis_tests:
                stmts = [loop]
                length = lengths_tests[v]
                ind_dof_test = index_dof_test.set_length(length)
                loop  = Loop(l_basis_tests[v], ind_dof_test, stmts)
            # ...

            body  = (Reset(l_sub_mats), loop)
            stmts = Block(body)
            g_stmts += [stmts]
    #=========================================================end kernel=========================================================

    # ... loop over global elements
    loop  = Loop((g_quad, *g_basis_tests.values(), *g_basis_trials.values(), *g_span.values()),
                  ind_element, g_stmts)

    body = [Reduce('+', l_mats, g_mats, loop)]
    # ...
    args = {}
    args['tests_basis']  = g_basis_tests.values()
    args['trial_basis']  = g_basis_trials.values()

    args['spans'] = g_span.values()
    args['quads'] = g_quad

    args['tests_degrees'] = lengths_tests
    args['trials_degrees'] = lengths_trials

    args['quads_degree'] = lengths
    args['global_pads']  = pads
    args['local_pads']   = Pads(tests, trials)

    if atomic_expr_field:
        args['coeffs'] = l_coeffs

    if eval_mapping:
        args['mapping'] = eval_mapping.coeffs

    args['mats']  = [l_mats, g_mats]

    local_vars = [*q_basis_tests, *q_basis_trials]
    stmt = DefNode('assembly', args, local_vars, body)

    return stmt

#================================================================================================================================
def _create_ast_linear_form(terminal_expr, atomic_expr, atomic_expr_field, tests, d_tests, shapes_tests, nderiv, dim, Mapping, space):
    """
    """
    pads   = symbols('pad1, pad2, pad3')[:dim]
    g_quad = GlobalTensorQuadrature()
    l_quad = LocalTensorQuadrature()

    coeffs   = [CoefficientBasis(i) for i in tests]
    l_coeffs = [MatrixLocalBasis(i) for i in tests]
    geo      = GeometryExpressions(Mapping, nderiv)

    l_vecs  = BlockStencilVectorLocalBasis(tests, pads, terminal_expr)
    g_vecs  = BlockStencilVectorGlobalBasis(tests, pads, terminal_expr,l_vecs.tag)
    q_basis = {v:d_tests[v]['array']  for v in tests}
    l_basis = {v:d_tests[v]['local']  for v in tests}

    g_basis = {v:d_tests[v]['global']  for v in tests}
    g_span  = {u:d_tests[u]['span'] for u in tests}

    lengths         = []
    # ...........................................................................................
    quad_length = LengthQuadrature()
    el_length   = LengthElement()
    lengths    += [el_length,quad_length]

    ind_quad      = index_quad.set_length(quad_length)
    ind_element   = index_element.set_length(el_length)
    ind_dof_test = index_dof_test.set_length(LengthDofTest(tests[0]))
    # ...........................................................................................
    eval_mapping = EvalMapping(ind_quad, ind_dof_test, q_basis[tests[0]], l_basis[tests[0]], Mapping, geo, space, nderiv)

    if atomic_expr_field:
        eval_field   = EvalField(atomic_expr_field, ind_quad, q_basis, coeff, tests, nderiv)

    # ...
    g_stmts = [eval_mapping]
    groups = regroupe(tests)
    ex_tests = expand(tests)
    # ... 
    #=========================================================begin kernel======================================================

    for dtype, group in groups:
        tests_indices = [ex_tests.index(i) for i in expand(group)]
        sub_terminal_expr = terminal_expr[tests_indices,0]
        sub_atomic_expr   = atomic_expr[tests_indices,0]
        l_sub_vecs  = BlockStencilVectorLocalBasis(group, pads, sub_terminal_expr, l_vecs.tag)
        q_basis = {v:d_tests[v]['array']  for v in group}
        l_basis = {v:d_tests[v]['local']  for v in group}
        if sub_terminal_expr.is_zero:
            continue
        stmts = []
        for v in group:
            stmts += construct_logical_expressions(v, nderiv)

        for expr in sub_atomic_expr[:]:
            stmts += [ComputePhysicalBasis(i) for i in expr]

        if atomic_expr_field:
            stmts += list(eval_fiel.inits)

        loop  = Loop((l_quad, *q_basis.values(), geo), ind_quad, stmts)
        loop = Reduce('+', ComputeKernelExpr(sub_terminal_expr), ElementOf(l_sub_vecs), loop)
    # ...

    # ... loop over tests to evaluate fields
        fields = EmptyLine()
        if atomic_expr_field:
            fields = Loop((*l_basis, l_coeff), ind_dof, [eval_field])

    # ... loop over tests
        for v in l_basis:
            stmts    = [loop]
            length   = LengthDofTest(v)
            lengths += [length]

            ind_dof_test = index_dof_test.set_length(length)
            loop  = Loop(l_basis[v], ind_dof_test, stmts)
        # ...
        body  = (EmptyLine(), fields, Reset(l_sub_vecs), loop)
        stmts = Block(body)
        g_stmts += [stmts]
    # ...
    
    #=========================================================end kernel=========================================================
    # ... loop over global elements
    loop  = Loop((g_quad, *g_basis.values(), *g_span.values()), ind_element, g_stmts)
    # ...
    body = (Reduce('+', l_vecs, g_vecs, loop),)

    args  = [*g_basis, *g_span, g_quad]
    args += [*lengths, *pads]    

    if atomic_expr_field:
        args += [*l_coeffs]
    if eval_mapping:
        args+= [*eval_mapping.coeffs]

    mats = [l_vecs, g_vecs]

    local_vars = [*q_basis]
    stmt = DefNode('assembly', args, mats, local_vars, body)
    # ...

    return stmt

#================================================================================================================================
def _create_ast_functional_form(terminal_expr, atomic_expr, tests, d_tests, shapes_tests, nderiv, dim, Mapping, space):
    """
    """

    pads   = symbols('pad1, pad2, pad3')[:dim]
    g_quad = GlobalTensorQuadrature()
    l_quad = LocalTensorQuadrature()

    #TODO should we use tests or trials for fields
    coeffs   = [CoefficientBasis(i) for i in tests]
    l_coeffs = [MatrixLocalBasis(i) for i in tests]
    geo      = GeometryExpressions(Mapping, nderiv)

    l_vecs  = BlockStencilVectorLocalBasis(tests, pads, terminal_expr)
    g_vecs  = BlockStencilVectorGlobalBasis(tests, pads, terminal_expr)

    q_basis = {v:d_tests[v]['array']  for v in tests}
    l_basis = {v:d_tests[v]['local']  for v in tests}
    g_basis = {v:d_tests[v]['global']  for v in tests}
    g_span  = {u:d_tests[u]['span'] for u in tests}

    l_vec   = LocalElementBasis()
    g_vec   = GlobalElementBasis()

    # ...........................................................................................
    quad_length = LengthQuadrature()
    el_length   = LengthElement()
    test_length = LengthDofTest(tests[0])
    lengths     = [el_length, quad_length, test_length]

    ind_quad      = index_quad.set_length(quad_length)
    ind_element   = index_element.set_length(el_length)
    ind_dof_test  = index_dof_test.set_length(test_length)
    # ...........................................................................................
    eval_mapping = EvalMapping(ind_quad, ind_dof_test, q_basis[tests[0]], l_basis[tests[0]], Mapping, geo, space, nderiv)
    eval_field   = EvalField(atomic_expr, ind_quad, q_basis[tests[0]], coeffs, tests, nderiv)

    #=========================================================begin kernel======================================================
    # ... loop over tests functions

    loop   = Loop((l_quad, *q_basis.values(), geo), ind_quad, eval_field.inits)
    loop   = Reduce('+', ComputeKernelExpr(terminal_expr), ElementOf(l_vec), loop)

    # ...
    # ... loop over tests functions to evaluate the fields
    fields = Loop((*l_basis.values(), *l_coeffs), ind_dof_test, [eval_field])
    stmts  = Block([EmptyLine(), eval_mapping, fields, Reset(l_vec), loop])

    #=========================================================end kernel=========================================================
    # ... loop over global elements


    loop  = Loop((g_quad, *g_basis.values(), *g_span.values()), ind_element, stmts)
    # ...

    body = (Reduce('+', l_vec, g_vec, loop),)

    args  = [*g_basis, *g_span, g_quad] 
    args += [*lengths, *pads, *coeffs]

    if eval_mapping:
        args+= [*eval_mapping.coeffs]

    mats = [l_vec, g_vec]

    local_vars = [*q_basis]
    stmt = DefNode('assembly', args, mats, local_vars, body)
    # ...

    return stmt
