#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np

from sympy import S
from sympy import IndexedBase, Indexed
from sympy import Mul, Matrix
from sympy import Add, And, StrictLessThan, Eq
from sympy import Not
from sympy import Symbol
from sympy import Basic
from sympy import MutableDenseNDimArray as MArray
from sympy.simplify import cse_main
from sympy.core.containers import Tuple

from sympde.topology import (dx1, dx2, dx3)
from sympde.topology import SymbolicExpr
from sympde.topology import LogicalExpr
from sympde.expr.evaluation import _split_test_function
from sympde.topology import SymbolicWeightedVolume
from sympde.topology import Boundary, NormalVector, Interface
from sympde.topology.basic import BasicDomain
from sympde.topology.mapping import Mapping

from sympde.topology.derivatives import get_index_logical_derivatives

from psydac.pyccel.ast.core      import Assign, AugAssign, For
from psydac.pyccel.ast.core      import Variable, IndexedVariable, IndexedElement
from psydac.pyccel.ast.core      import Slice
from psydac.pyccel.ast.core      import EmptyNode, Import, While, Return, If
from psydac.pyccel.ast.core      import CodeBlock, FunctionDef, Comment
from psydac.pyccel.ast.builtins  import Range

from psydac.api.utilities     import flatten, random_string
from psydac.api.ast.utilities import variables, math_atoms_as_str, get_name
from psydac.api.ast.utilities import build_pythran_types_header
from psydac.api.ast.utilities import build_pyccel_type_annotations

from .nodes import AtomicNode
from .nodes import BasisAtom
from .nodes import LogicalBasisValue
from .nodes import TensorQuadrature
from .nodes import LocalTensorQuadratureBasis
from .nodes import LocalTensorQuadratureTestBasis
from .nodes import LocalTensorQuadratureTrialBasis
from .nodes import GlobalTensorQuadratureTestBasis
from .nodes import GlobalTensorQuadratureTrialBasis
from .nodes import GlobalTensorQuadratureBasis
from .nodes import SplitArray
from .nodes import Reduction
from .nodes import LogicalValueNode
from .nodes import TensorIteration
from .nodes import TensorIterator
from .nodes import TensorGenerator
from .nodes import ProductIteration
from .nodes import ProductIterator
from .nodes import ProductGenerator
from .nodes import StencilMatrixLocalBasis
from .nodes import StencilMatrixGlobalBasis, ScalarLocalBasis
from .nodes import BlockStencilMatrixLocalBasis
from .nodes import BlockStencilMatrixGlobalBasis
from .nodes import BlockStencilVectorLocalBasis, BlockScalarLocalBasis
from .nodes import BlockStencilVectorGlobalBasis
from .nodes import StencilVectorLocalBasis
from .nodes import StencilVectorGlobalBasis
from .nodes import GlobalElementBasis
from .nodes import LocalElementBasis
from .nodes import TensorQuadratureTestBasis, TensorQuadratureTrialBasis
from .nodes import Span
from .nodes import Loop
from .nodes import WeightedVolumeQuadrature
from .nodes import LengthDofTest

from .nodes import index_dof_test, index_dof_trial
from .nodes import index_deriv, Max, Min

from .nodes import Zeros, ZerosLike, Array
from .fem import expand

#==============================================================================
class Shape(Basic):
    @property
    def arg(self):
        return self._args[0]

def is_scalar_array(var):
    indices = var.indices
    for ind in indices:
        if isinstance(ind, Slice):
            return False
    return True

#==============================================================================
def parse(expr, settings, backend=None):
    """
    A function which takes a PSYDAC AST and transforms it to a Pyccel AST.

    This function takes a PSYDAC abstract syntax tree (AST) and returns a
    Pyccel AST. In turn, this can be translated to Python code through a call
    to the function `pycode` from `psydac.pyccel.codegen.printing.pycode`.

    Parameters
    ----------
    expr : Any
        PSYDAC AST, of any type supported by the Parser class.

    settings : dict
        Dictionary that contains number of dimension, mappings and target if provided

    Returns
    -------
    ast : psydac.pyccel.ast.basic.PyccelAstNode | psydac.pyccel.ast.core.FunctionDef
        Pyccel abstract syntax tree that can be translated into a Python code.
    """
    psy_parser = Parser(settings, backend)
    ast = psy_parser.doit(expr)
    return ast

#==============================================================================
class Parser(object):
    """
    A Parser which takes a PSYDAC AST and transforms it to a Pyccel AST.

    This class takes a PSYDAC AST and transforms it to the AST of an old and
    reduced version of Pyccel, which is shipped as `psydac.pyccel`. This
    "mini-Pyccel" is then used for printing the Python code.

    The parsing is performed by passing any object of a "supported type" to the
    method `Parser.doit`. This in turn calls `Parser._visit` which starts a
    recursive tree traversal through specialized `Parser._visit_<NODE_TYPE>`
    methods. If successful, `Parser.doit` generally returns a `PyccelAstNode`
    object (from `psydac.pyccel.ast.basic`). If the input object is a `DefNode`
    (representing a function definition) it returns a `FunctionDef` object
    (from `psydac.pyccel.ast.core`). The resulting Pyccel AST can be printed to
    Python code using the function `pycode` from
    `psydac.pyccel.codegen.printing.pycode`.

    By "supported types" we mean any `<NODE_TYPE>` type for which a method
    `Parser._visit_<NODE_TYPE>` is provided. The matching is done by name, and
    it also checks any superclasses listed in `<NODE_TYPE>.__mro__` in the given
    order.

    Parameters
    ----------
    settings : dict[str, Any]
        A dictionary with required integer arguments `dim` (number of dimensions)
        and `nderiv` (maximum number of derivatives), required argument `target`
        (symbolic domain of expression, of type `BasicDomain` from
        `sympde.topology.basic`), and optional argument `mapping` (domain
        `Mapping` from `sympde.topology.mapping`).

    backend : dict[str, Any]
        The backend dictionary as defined in `psydac.api.settings`.
    """
    def __init__(self, settings, backend=None):

        # Copy settings, hence input dictionary is not modified
        settings = settings.copy()

        # ... Pop values from settings and perform sanity checks
        dim = settings.pop('dim', None)
        if dim is None:
            raise ValueError('dim not provided')
        else:
            assert isinstance(dim, int)

            assert dim > 0
        nderiv = settings.pop('nderiv', None)
        if nderiv is None:
            raise ValueError('nderiv not provided')
        else:
            assert isinstance(nderiv, int)
            assert nderiv >= 0

        target = settings.pop('target', None)
        if target is None:
            raise ValueError('target not provided')
        else:
            assert isinstance(target, BasicDomain)

        mapping = settings.pop('mapping', None)
        if mapping is not None:
            assert isinstance(mapping, Mapping)
        # ...

        # Store extracted values and other settings
        self._dim      = dim
        self._nderiv   = nderiv
        self._target   = target
        self._mapping  = mapping
        self._settings = settings

        # Store backend dictionary
        if backend is not None:
            assert isinstance(backend, dict)
            assert 'name' in backend.keys()
        self.backend = backend

        # TODO improve
        self.indices          = {}
        self.shapes           = {}
        self.functions        = {}
        self.variables        = {}
        self.arguments        = {}
        self.allocated        = {}
        self._math_functions  = ()

    @property
    def settings(self):
        return self._settings

    @property
    def dim(self):
        return self._dim

    @property
    def nderiv(self):
        return self._nderiv

    @property
    def mapping(self):
        return self._mapping

    @property
    def target(self):
        return self._target

    def doit(self, expr, **settings):
        return self._visit(expr, **settings)

    def insert_variables(self, *args):
        args = flatten(args)
        for arg in args:
            self.variables[str(arg)] = arg

    def get_shape(self, expr):
        lhs = expr.lhs
        rhs = expr.rhs

        rhs_indices = []
        if isinstance(rhs, (Indexed, IndexedElement)):
            rhs_indices = rhs.indices
        lhs_indices = lhs.indices

        #TODO fix probleme of indices we should have a unique way of getting indices
        lhs_indices = [None if isinstance(i, Slice) and i.start is None else i for i in lhs_indices]
        rhs_indices = [None if isinstance(i, Slice) and i.start is None else i for i in rhs_indices]
        shape_lhs   = None
        shape       = []

        if all(i is None for i in lhs_indices):
            for i in rhs_indices:
                if i is None:
                    shape.append(None)
                elif str(i) in self.indices:
                    shape.append(self.indices[str(i)]-1)
                elif isinstance(i, Slice) and i.start and i.end:
                    shape.append(i.end-i.start)
            if len(shape) == len(rhs_indices):
                if any(s is None for s in shape):
                    shape     = tuple(Slice(None,None) if i is None else 0 for i in shape)
                    rhs       = rhs.base
                    shape_lhs = Shape(rhs[shape])
                else:
                    shape_lhs = tuple(shape)

        elif all(i is not None for i in lhs_indices):
            for i in lhs_indices:
                if str(i) in self.indices:
                    shape.append(self.indices[str(i)])
            if len(shape) == len(lhs_indices):
                shape_lhs = tuple(shape)

        return shape_lhs

    def _visit(self, expr, **settings):
        classes = type(expr).__mro__
        for cls in classes:
            annotation_method = '_visit_' + cls.__name__
            if hasattr(self, annotation_method):
                return getattr(self, annotation_method)(expr, **settings)
        # Unknown object, we raise an error.
        raise NotImplementedError('{}'.format(type(expr)))

    # ....................................................
    def _visit_VectorAssign(self, expr, **kwargs):
        lhs = self._visit(expr.lhs)
        rhs = self._visit(expr.rhs)
        if expr.op is None:
            return [Assign(l,r) for l,r in zip(lhs, rhs) if l is not None and r is not None and l is not S.Zero]
        else:
            return [AugAssign(l,expr.op, r) for l,r in zip(lhs, rhs) if l is not None and r is not None and l is not S.Zero]
    # ....................................................
    def _visit_Assign(self, expr, **kwargs):

        lhs = self._visit(expr.lhs)
        rhs = self._visit(expr.rhs)

        # ... extract slices from rhs
        slices = []
        if isinstance(rhs, IndexedElement):
            slices = [i for i in rhs.indices if isinstance(i, Slice)]
        # ...

        # ... update lhs with slices
        if len(slices) > 0:
            # TODO add assert on type lhs
            if isinstance(lhs, (IndexedBase, IndexedVariable)):
                lhs = lhs[slices]

            elif isinstance(lhs, Symbol):
                lhs = IndexedBase(lhs.name)[slices]

        expr = Assign(lhs, rhs)
        # ..

        if isinstance(lhs, (IndexedElement, Indexed)):
            name = str(lhs.base)

            shape = self.get_shape(expr)
            if shape:
                self.shapes[name] = shape

        return expr

    # ....................................................
    def _visit_AugAssign(self, expr, **kwargs):

        lhs = self._visit(expr.lhs)
        rhs = self._visit(expr.rhs)
        op  = expr.op

        # ... extract slices from rhs
        slices = []
        if isinstance(rhs, IndexedElement):
            slices = [i for i in indices if isinstance(i, Slice)]
        # ...

        # ... update lhs with slices
        if len(slices) > 0:
            # TODO add assert on type lhs
            if isinstance(lhs, (IndexedBase, IndexedVariable)):
                lhs = lhs[slices]
            else:
                raise NotImplementedError('{}'.format(type(lhs)))

        expr = AugAssign(lhs,op,rhs)
        # ...
        if isinstance(lhs, (IndexedElement,Indexed)):
            name = str(lhs.base)

            shape = self.get_shape(expr)
            if shape:
                self.shapes[name] = shape

        return expr

    def _visit_Allocate(self, expr, **kwargs):
        arr   = self._visit(expr.array)
        shape = [self._visit(i) for i in expr.shape]
        self.allocated[arr.name] = arr
        return Assign(arr, Zeros(tuple(shape), arr.dtype))

    # ....................................................
    def _visit_AddNode(self, expr, **kwargs):
        return self._visit_Add(expr)

    def _visit_MulNode(self, expr, **kwargs):
        return self._visit_Mul(expr)

    # ....................................................
    def _visit_IntDivNode(self, expr, **kwargs):
        args = [self._visit(a) for a in expr.args]
        return args[0]//args[1]

    # ....................................................
    def _visit_AndNode(self, expr, **kwargs):
        args = [self._visit(a) for a in expr.args]
        return And(*args)

    def _visit_NotNode(self, expr, **kwargs):
        return Not(self._visit(expr.args[0]))

    def _visit_EqNode(self, expr, **kwargs):
        return Eq(self._visit(expr.args[0]), self._visit(expr.args[1]))

    # ....................................................
    def _visit_StrictLessThanNode(self, expr, **kwargs):
        a = self._visit(expr.args[0])
        b = self._visit(expr.args[1])
        return StrictLessThan(a,b)

    # ....................................................
    def _visit_Add(self, expr, **kwargs):
        args   = [self._visit(i) for i in expr.args]
        tuples = [e for e in args if isinstance(e, tuple)]
        args   = [e for e in args if not e in tuples]
        expr   =  Add(*args)
        if tuples:
            args  = list(tuples[0])
            for e in tuples[1:]:
                args = [args[i]+e[i] for i in range(len(args))]
            tuples = tuple(Add(expr,e) for e in args)
            return tuples
        return expr

    # ....................................................
    def _visit_Mul(self, expr, **kwargs):
        args = [self._visit(i) for i in expr.args]
        return Mul(*args)

    # ....................................................
    def _visit_Symbol(self, expr, **kwargs):
        return expr

    # ....................................................
    def _visit_Variable(self, expr, **kwargs):
        return expr

    # ....................................................
    def _visit_IndexedVariable(self, expr, **kwargs):
        return expr

    # ....................................................
    def _visit_Tuple(self, expr, **kwargs):
        args = [self._visit(i) for i in expr]
        return Tuple(*args)

    def _visit_Array(self, expr, **kwargs):
        data  = self._visit(expr.data)
        dtype = expr.dtype
        return Array(data, dtype=dtype)

    # ....................................................
    def _visit_Block(self, expr, **kwargs):
        body = [self._visit(i) for i in expr.body]
        body = flatten(body)
        if len(body) == 1:
            return body[0]

        else:
            return CodeBlock(body)

    # ....................................................
    def _visit_ParallelBlock(self, expr, **kwargs):
        body         = [self._visit(i) for i in expr.body]
        body         = list(flatten(body))
        default      = expr.default
        shared       = [self._visit(i) for i in expr.shared]
        private      = [self._visit(i) for i in expr.private]
        firstprivate = [self._visit(i) for i in expr.firstprivate]
        lastprivate  = [self._visit(i) for i in expr.lastprivate]
        shared       = flatten([list(i.values())[0] if isinstance(i, dict) else i for i in shared])
        private      = flatten([list(i.values())[0] if isinstance(i, dict) else i for i in private])
        firstprivate = flatten([list(i.values())[0] if isinstance(i, dict)else i for i in firstprivate])
        lastprivate  = flatten([list(i.values())[0] if isinstance(i, dict) else i for i in lastprivate])
        txt          = '#$ omp parallel default({}) &\n'.format(default)
        txt         += '#$ shared({}) &\n'.format(','.join(str(i) for i in shared if i)) if shared else ''
        txt         += '#$ private({}) &\n'.format(','.join(str(i) for i in private if i)) if private else ''
        txt         += '#$ firstprivate({}) &\n'.format(','.join(str(i) for i in firstprivate if i)) if firstprivate else ''
        txt         += '#$ lastprivate({})'.format(','.join(str(i) for i in lastprivate if i)) if lastprivate else ''
        cmt          = [Comment(txt.rstrip().rstrip('&'))]
        endcmt       = [Comment('#$ omp end parallel')]
        return CodeBlock(cmt + body + endcmt)

    # ....................................................
    def _visit_DefNode(self, expr, **kwargs):
        """ Convert the DefNode object to a FunctionDef and sort its arguments in the following order:
            - 1D tests basis functions
            - 1D trial basis functions (if present)
            - 1D mapping basis functions (if present)
            - Span of tests basis functions
            - Span of mapping basis functions (if present)
            - 1D quadrature points and weights
            - Degrees of tests basis functions
            - Degrees of trial basis functions (if present)
            - Degrees of mapping basis functions (if present)
            - Quadrature degrees in each dimension
            - Length of ghost regions for global matrices/vectors (if present)
            - Coefficient of mapping (if present)
            - Global matrices/vectors
            - 1D basis function of field space (if present)
            - Span of field basis functions (if present)
            - Degrees of field basis functions (if present)
            - Length of ghost regions for field vectors (if present)
            - Coefficient of field (if present)
            - Constants (if present)
            """

        args   = expr.arguments.copy()
        f_args = ()

        tests_basis = args.pop('tests_basis')
        trial_basis = args.pop('trial_basis',[])

        g_span = args.pop('spans')
        g_quad = args.pop('quads')

        lengths_tests  = args.pop('tests_degrees')
        lengths_trials = args.pop('trials_degrees', {})

        lengths = args.pop('quads_degree')
        g_pads  = args.pop('global_pads')
        l_pads  = args.pop('local_pads', None)

        mats = args.pop('mats')
        
        map_coeffs  = args.pop('mapping', None)
        map_degrees = args.pop('mapping_degrees', None)
        map_basis   = args.pop('mapping_basis', None)
        map_span    = args.pop('mapping_spans', None)
        thread_args = args.pop('thread_args', None)

        if not map_coeffs:
            map_coeffs  = []
            map_degrees = []
            map_basis   = []
            map_span    = []

        constants = args.pop('constants', None)
        f_coeffs  = args.pop('f_coeffs' , None)

        starts = args.pop('starts', [])
        ends   = args.pop('ends'  , [])
        if f_coeffs:
            f_span    = args.pop('f_span',      [])
            f_basis   = args.pop('field_basis', [])
            f_degrees = args.pop('fields_degrees', [])
            f_pads    = args.pop('f_pads', [])
            f_args    = (*f_basis, *f_span, *f_degrees, *f_pads, *f_coeffs)

        args = [*tests_basis, *trial_basis, *map_basis,\
                *g_span, *map_span, *g_quad,\
                *lengths_tests.values(), *lengths_trials.values(),\
                *map_degrees, *lengths, *g_pads, *map_coeffs]

        if mats:
            exprs = [mat.expr for mat in mats]
            mats  = [self._visit(mat) for mat in mats]
            mats  = [[a for a,e in zip(mat[:],expr[:]) if e] for mat,expr in zip(mats, exprs)]
            mats  = flatten(mats)

        args      = [self._visit(i, **kwargs) for i in args]
        args      = [tuple(arg.values())[0] if isinstance(arg, dict) else arg for arg in args]
        arguments = flatten(args) + mats

        if f_args:
            f_args     = [self._visit(i, **kwargs) for i in f_args]
            f_args     = [tuple(arg.values())[0] if isinstance(arg, dict) else arg for arg in f_args]
            arguments += flatten(f_args)

        if constants:
            arguments += [self._visit(i, **kwargs) for i in constants]

        arguments += starts + ends

        if thread_args:
            arguments += flatten([self._visit(i, **kwargs) for i in thread_args])

        body = flatten(tuple(self._visit(i, **kwargs) for i in expr.body))

        inits = []
        for k,i in self.shapes.items():
            if not k in self.variables: continue
            var = self.variables[k]
            if var in arguments or var.name in self.allocated:
                continue
            if isinstance(i, Shape):
                inits.append(Assign(var, ZerosLike(i.arg)))
            else:
                inits.append(Assign(var, Zeros(i, dtype=var.dtype)))

        inits.append(EmptyNode())
        body = tuple(inits) + body
        name = expr.name

        math_library  = 'cmath' if expr.domain_dtype=='complex' else 'math' # Function names are the same
        math_imports  = (*self._math_functions,)
        numpy_imports = ('array', 'zeros', 'zeros_like', 'floor')
        imports       = [Import('numpy', numpy_imports)] + \
                        ([Import(math_library, math_imports)] if math_imports else []) + \
                        [*expr.imports]

        results = [self._visit(a) for a in expr.results]

        if self.backend['name'] == 'pyccel':
            arguments = build_pyccel_type_annotations(arguments)
            decorators = {}
        elif self.backend['name'] == 'pythran':
            header = build_pythran_types_header(name, arguments) # Could this work??
        else:
            decorators = {}

        func = FunctionDef(name, arguments, results, body, imports=imports, decorators=decorators)
        stmts = func

        self.functions[name] = func
        return stmts

    def _visit_EvalField(self, expr, **kwargs):
        body = self._visit(expr.body, **kwargs)
        return body

    def _visit_EvalMapping(self, expr, **kwargs):
        if self._mapping.is_analytical:
            return EmptyNode()
        stmts = self._visit(expr.stmts)
        return stmts
    # ....................................................
    def _visit_Grid(self, expr, **kwargs):
        raise NotImplementedError('TODO')

    # ....................................................
    def _visit_Element(self, expr, **kwargs):
        raise NotImplementedError('TODO')

    # ....................................................
    def _visit_GlobalTensorQuadratureGrid(self, expr, **kwargs):
        dim   = self.dim
        rank  = expr.rank

        names = 'global_x1:%s'%(dim+1)
        points   = variables(names, dtype='real', rank=rank, cls=IndexedVariable)

        if expr.weights:
            names = 'global_w1:%s'%(dim+1)
            weights  = variables(names, dtype='real', rank=rank, cls=IndexedVariable)

            # gather by axis
            targets = tuple(zip(points, weights))
        else:
            weights = []
            targets = tuple(zip(points))

        self.insert_variables(*points, *weights)

        return {0: targets}

    # ....................................................
    def _visit_LocalTensorQuadratureGrid(self, expr, **kwargs):
        dim    = self.dim
        rank   = expr.rank
        names  = 'local_x1:%s'%(dim+1)
        points = variables(names, dtype='real', rank=rank, cls=IndexedVariable)

        if expr.weights:
            names   = 'local_w1:%s'%(dim+1)
            weights = variables(names, dtype='real', rank=rank, cls=IndexedVariable)

            # gather by axis
            targets = tuple(zip(points, weights))
        else:
            weights = []
            targets = tuple(zip(points))

        self.insert_variables(*points, *weights)

        return {0: targets}

    # ....................................................
    def _visit_PlusGlobalTensorQuadratureGrid(self, expr, **kwargs):
        dim    = self.dim
        rank   = expr.rank
        names  = 'global_x1:%s_plus'%(dim+1)
        points = variables(names, dtype='real', rank=rank, cls=IndexedVariable)

        # gather by axis
        self.insert_variables(*points)

        points = tuple(zip(points))
        return dict([(0, points)])

    # ....................................................
    def _visit_PlusLocalTensorQuadratureGrid(self, expr, **kwargs):
        dim    = self.dim
        rank   = expr.rank
        names  = 'local_x1:%s_plus'%(dim+1)
        points = variables(names, dtype='real', rank=rank, cls=IndexedVariable)

        self.insert_variables(*points)

        points = tuple(zip(points))
        return dict([(0, points)])

    # ....................................................
    def _visit_TensorQuadrature(self, expr, **kwargs):
        dim = self.dim
        names   = 'x1:%s' % (dim+1)
        points  = variables(names, dtype='real', cls=Variable)

        if expr.weights:
            names   = 'w1:%s' % (dim+1)
            weights = variables(names, dtype='real', cls=Variable)

            # gather by axis
            targets = tuple(zip(points, weights))
        else:
            weights  = []
            targets  = tuple(zip(points))

        self.insert_variables(*points, *weights)

        return {0: targets}

    # ....................................................
    def _visit_PlusTensorQuadrature(self, expr, **kwargs):
        dim = self.dim
        names   = 'x1:%s_plus' % (dim+1)
        points  = variables(names, dtype='real', cls=Variable)

        targets = tuple(zip(points))

        self.insert_variables(*points)

        return dict([(0, targets)])

    # ....................................................
    def _visit_GlobalThreadSpanArray(self, expr, **kwargs):
        dim    = self.dim
        rank   = expr.rank
        target = SymbolicExpr(expr.target)
        name   = 'thread_spans_{}_'.format(target)
        targets = variables('{}1:{}'.format(name, dim+1), dtype='int', rank=1, cls=IndexedVariable)
        if expr.index is not None:
            return targets[expr.index]
        return targets

    # ....................................................
    def _visit_GlobalThreadStarts(self, expr, **kwargs):
        dim     = self.dim
        targets = variables('global_thread_starts_1:{}'.format(dim+1), dtype='int', rank=1, cls=IndexedVariable)
        if expr.index is not None:
            return targets[expr.index]
        return targets

    # ....................................................
    def _visit_GlobalThreadEnds(self, expr, **kwargs):
        dim     = self.dim
        targets = variables('global_thread_ends_1:{}'.format(dim+1), dtype='int', rank=1, cls=IndexedVariable)
        if expr.index is not None:
            return targets[expr.index]
        return targets

    # ....................................................
    def _visit_GlobalThreadSizes(self, expr, **kwargs):
        dim     = self.dim
        targets = variables('global_thread_size_1:{}'.format(dim+1), dtype='int')
        if expr.index is not None:
            return targets[expr.index]
        return targets

    # ....................................................
    def _visit_LocalThreadStarts(self, expr, **kwargs):
        dim     = self.dim
        targets = variables('local_thread_starts_1:{}'.format(dim+1), dtype='int', rank=1, cls=IndexedVariable)
        if expr.index is not None:
            return targets[expr.index]
        return targets

    # ....................................................
    def _visit_LocalThreadEnds(self, expr, **kwargs):
        dim     = self.dim
        targets = variables('local_thread_ends_1:{}'.format(dim+1), dtype='int', rank=1, cls=IndexedVariable)
        if expr.index is not None:
            return targets[expr.index]
        return targets

    # ....................................................
    def _visit_MatrixQuadrature(self, expr, **kwargs):
        rank   = self._visit(expr.rank)
        dtype  = expr.dtype
        target = SymbolicExpr(expr.target)
        name   = 'arr_{}'.format(target.name)
        var    =  IndexedVariable(name, dtype=dtype, rank=rank)
        self.insert_variables(var)
        return var
    # ....................................................
    def _visit_GlobalTensorQuadratureBasis(self, expr, **kwargs):
        # TODO add label
        dim = self.dim
        rank = expr.rank
        unique_scalar_space = expr.unique_scalar_space
        is_scalar           = expr.is_scalar
        target              = expr.target
        label               = str(SymbolicExpr(target))
        if isinstance(expr, GlobalTensorQuadratureTestBasis):
            if not unique_scalar_space:
                names = 'global_test_basis_{label}(1:{j})_1:{i}'.format(label=label,i=dim+1,j=dim+1)
            else:
                names = 'global_test_basis_{label}_1:{i}'.format(label=label,i=dim+1)

        elif isinstance(expr, GlobalTensorQuadratureTrialBasis):
            if not unique_scalar_space:
                names = 'global_trial_basis_{label}(1:{j})_1:{i}'.format(label=label,i=dim+1,j=dim+1)
            else:
                names = 'global_trial_basis_{label}_1:{i}'.format(label=label,i=dim+1)

        else:
            if not unique_scalar_space:
                names = 'global_basis_{label}(1:{j})_1:{i}'.format(label=label,i=dim+1,j=dim+1)
            else:
                names = 'global_basis_{label}_1:{i}'.format(label=label,i=dim+1)

        targets = variables(names, dtype='real', rank=rank, cls=IndexedVariable)
        if expr.index is not None:
            return targets[expr.index]

        self.insert_variables(*targets)

        arrays = {}
        if unique_scalar_space and not is_scalar:
            for i in range(dim):
                arrays[target[i]] = tuple(zip(targets))
        elif not unique_scalar_space:
            for i in range(dim):
                arrays[target[i]] = tuple(zip(targets[i::dim]))
        else:
            arrays[target] = tuple(zip(targets))
        return arrays
    # ....................................................
    def _visit_LocalTensorQuadratureBasis(self, expr, **kwargs):
        dim  = self.dim
        rank = expr.rank
        unique_scalar_space = expr.unique_scalar_space
        is_scalar           = expr.is_scalar
        target              = expr.target
        label               = str(SymbolicExpr(target))
        if isinstance(expr, LocalTensorQuadratureTestBasis):
            if not unique_scalar_space:
                names = 'local_test_basis_{label}(1:{j})_1:{i}'.format(label=label,i=dim+1,j=dim+1)
            else:
                names = 'local_test_basis_{label}_1:{i}'.format(label=label,i=dim+1)

        elif isinstance(expr, LocalTensorQuadratureTrialBasis):
            if not unique_scalar_space:
                names = 'local_trial_basis_{label}(1:{j})_1:{i}'.format(label=label,i=dim+1,j=dim+1)
            else:
                names = 'local_trial_basis_{label}_1:{i}'.format(label=label,i=dim+1)

        else:
            if not unique_scalar_space:
                names = 'local_basis_{label}(1:{j})_1:{i}'.format(label=label,i=dim+1,j=dim+1)
            else:
                names = 'local_basis_{label}_1:{i}'.format(label=label,i=dim+1)

        targets = variables(names, dtype='real', rank=rank, cls=IndexedVariable)
        if expr.index is not None:
            return targets[expr.index]

        self.insert_variables(*targets)

        arrays = {}
        if unique_scalar_space and not is_scalar:
            for i in range(dim):
                arrays[target[i]] = tuple(zip(targets))
        elif not unique_scalar_space:
            for i in range(dim):
                arrays[target[i]] = tuple(zip(targets[i::dim]))
        else:
            arrays[target] = tuple(zip(targets))
        return arrays

    # ....................................................
    def _visit_TensorQuadratureBasis(self, expr, **kwargs):
        dim  = self.dim
        rank = expr.rank
        unique_scalar_space = expr.unique_scalar_space
        is_scalar           = expr.is_scalar
        target              = expr.target
        label               = str(SymbolicExpr(target))

        if isinstance(expr, TensorQuadratureTestBasis):
            if not unique_scalar_space:
                names = 'test_basis_{label}(1:{j})_1:{i}'.format(label=label,i=dim+1,j=dim+1)
            else:
                names = 'test_basis_{label}_1:{i}'.format(label=label,i=dim+1)

        elif isinstance(expr, TensorQuadratureTrialBasis):
            if not unique_scalar_space:
                names = 'trial_basis_{label}(1:{j})_1:{i}'.format(label=label,i=dim+1,j=dim+1)
            else:
                names = 'trial_basis_{label}_1:{i}'.format(label=label,i=dim+1)
        else:
            if not unique_scalar_space:
                names = 'array_basis_{label}(1:{j})_1:{i}'.format(label=label,i=dim+1,j=dim+1)
            else:
                names = 'array_basis_{label}_1:{i}'.format(label=label,i=dim+1)
        # ...

        targets = variables(names, dtype='real', rank=rank, cls=IndexedVariable)

        self.insert_variables(*targets)
        arrays = {}
        if unique_scalar_space and not is_scalar:
            for i in range(dim):
                arrays[target[i]] = tuple(zip(targets))
        elif not unique_scalar_space:
            for i in range(dim):
                arrays[target[i]] = tuple(zip(targets[i::dim]))
        else:
            arrays[target] = tuple(zip(targets))
        return arrays

    # ....................................................
    def _visit_GlobalSpanArray(self, expr, **kwargs):
        dim    = self.dim
        rank   = expr.rank
        target = expr.target
        label  = SymbolicExpr(target).name

        names   = 'global_span_{}_1:{}'.format(label, str(dim+1))
        targets = variables(names, dtype='int', rank=rank, cls=IndexedVariable)
        if expr.index is not None:
            return targets[expr.index]

        self.insert_variables(*targets)
        if not isinstance(targets[0], (tuple, list, Tuple)):
            targets = [targets]
        target = {target: tuple(zip(*targets))}
        return target

    # ....................................................
    def _visit_LocalSpanArray(self, expr, **kwargs):
        dim    = self.dim
        rank   = expr.rank
        target = expr.target
        label  = SymbolicExpr(target).name

        names   = 'local_span_{}_1:{}'.format(label, str(dim+1))
        targets = variables(names, dtype='int', rank=rank, cls=IndexedVariable)
        if expr.index is not None:
            return targets[expr.index]

        self.insert_variables(*targets)
        if not isinstance(targets[0], (tuple, list, Tuple)):
            targets = [targets]
        target = {target: tuple(zip(*targets))}
        return target

    # ....................................................
    def _visit_Span(self, expr, **kwargs):
        dim     = self.dim
        target  = expr.target
        label   = SymbolicExpr(target).name
        names   = 'span_{}_1:{}'.format(label,str(dim+1))
        targets = variables(names, dtype='int')

        if expr.index is not None:
            return targets[expr.index]

        self.insert_variables(*targets)
        if not isinstance(targets[0], (tuple, list, Tuple)):
            targets = [targets]

        target = {target: tuple(zip(*targets))}
        return target

    # ....................................................
    def _visit_Pads(self, expr, **kwargs):
        dim           = self.dim
        tests         = expand(expr.tests)
        tests_degree  = expr.tests_degree
        trials_degree = expr.trials_degree
        m_tests       = expr.tests_multiplicity
        m_trials      = expr.trials_multiplicity

        if expr.trials is not None:
            trials = expand(expr.trials)
            pads = MArray.zeros(len(tests), len(trials), dim)
            for i in range(pads.shape[0]):
                for j in range(pads.shape[1]):
                    label1  = SymbolicExpr(tests[i]).name
                    label2  = SymbolicExpr(trials[j]).name
                    names   = f'pad_{label2}_{label1}_1:{dim+1}'
                    targets = variables(names, dtype='int')
                    pads[i, j, :] = targets
                    self.insert_variables(*targets)
            if expr.test_index is not None and expr.trial_index is not None:
                if expr.dim_index is not None:
                    return pads[expr.test_index, expr.trial_index, expr.dim_index]
                return pads[expr.test_index, expr.trial_index]
                
        else:
            pads = MArray.zeros(len(tests), 1, dim)
            for i in range(pads.shape[0]):
                label1  = SymbolicExpr(tests[i]).name
                names   = f'pad_{label1}_1:{dim+1}'
                targets = variables(names, dtype='int')
                pads[i, 0, :] = targets
                self.insert_variables(*targets)

            if expr.test_index is not None:
                if expr.dim_index is not None:
                    return pads[expr.test_index, 0, expr.dim_index]
                return pads[expr.test_index, 0]
            #...

        return pads

    # ....................................................
    def _visit_TensorBasis(self, expr, **kwargs):
        # TODO label
        dim    = self.dim
        nderiv = self.nderiv
        target = expr.target

        ops   = [dx1, dx2, dx3][:dim]
        atoms =  _split_test_function(target)
        args  = {}
        for atom in atoms:
            sub_args = [None]*dim
            for i in range(dim):
                d  = ops[i]
                a  = atoms[atom][i]
                ls = [a]
                for _ in range(1, nderiv+1):
                    a = d(a)
                    ls.append(a)
                sub_args[i] = tuple(ls)
            args[atom] = tuple(sub_args)
        return args

    # ....................................................
    def _visit_CoefficientBasis(self, expr, **kwargs):
        target = SymbolicExpr(expr.target)
        name   = 'coeff_{}'.format(target.name)
        var    = IndexedVariable(name, dtype='real', rank=self.dim)
        self.insert_variables(var)
        return var

    def _visit_MatrixCoordsFromRank(self, expr, **kwargs):
        var  = IndexedVariable('coords_from_rank', dtype='int', rank=2)
        return var

    def _visit_MatrixRankFromCoords(self, expr, **kwargs):
        var  = IndexedVariable('rank_from_coords', dtype='int', rank=self.dim)
        return var

    # ....................................................
    def _visit_MatrixLocalBasis(self, expr, **kwargs):
        rank   = self._visit(expr.rank)
        target = SymbolicExpr(expr.target)
        dtype  = expr.dtype
        name   = 'arr_coeffs_{}'.format(target.name)
        var    = IndexedVariable(name, dtype=dtype, rank=rank)
        self.insert_variables(var)
        return var

    # ....................................................
    def _visit_MatrixGlobalBasis(self, expr, **kwargs):
        rank   = self._visit(expr.rank)
        target = SymbolicExpr(expr.target)
        dtype  = expr.dtype
        name   = 'global_arr_coeffs_{}'.format(target.name)
        var    = IndexedVariable(name, dtype=dtype, rank=rank)
        self.insert_variables(var)
        return var

    # ....................................................
    def _visit_Reset(self, expr, **kwargs):
        var = expr.var
        lhs  = self._visit(var, **kwargs)
        if hasattr(var, 'dtype'):
            dtype = var.dtype
        else:
            dtype = 'real'
        # Define data type of the 0
        zero = 0.0j if dtype == 'complex' else 0.0
        if isinstance(var, (LocalElementBasis, GlobalElementBasis)):
            return Assign(lhs, zero)

        elif isinstance(var, BlockScalarLocalBasis):
            expr = var.expr
            return tuple(Assign(a, zero) for a,b in zip(lhs[:], expr[:]) if b)

        expr = var.expr
        if not any(lhs[:]):
            return ()
        rank = [l.rank for l in lhs[:] if l][0]
        args  = [Slice(None, None)]*rank
        return tuple(Assign(a[args], zero) for a,b in zip(lhs[:], expr[:]) if b)

    # ....................................................
    def _visit_Reduce(self, expr, **kwargs):
        op   = expr.op
        lhs  = expr.lhs
        rhs  = expr.rhs
        loop = expr.loop
        parallel     = loop.parallel
        default      = loop.default
        shared       = loop.shared
        private      = loop.private
        firstprivate = loop.firstprivate
        lastprivate  = loop.lastprivate
        reduction    = None
        if parallel:
            reduction = 'reduction({}:{})'.format(expr.op, self._visit(lhs).name)

        stmts = list(loop.stmts) + [Reduction(op, rhs, lhs)]
        loop  = Loop(loop.iterable, loop.index, stmts=stmts, mask=loop.mask, parallel=parallel,
                    default=default, shared=shared, private=private,
                    firstprivate=firstprivate, lastprivate=lastprivate, reduction=reduction)
        return self._visit(loop, **kwargs)

    # ....................................................
    def _visit_Reduction(self, expr, **kwargs):
        op   = expr.op
        lhs  = expr.lhs
        expr = expr.expr

        if isinstance(lhs, (GlobalElementBasis, LocalElementBasis)):
            lhs = self._visit(lhs, **kwargs)
            rhs = self._visit(expr, **kwargs)
            return (AugAssign(lhs, op, rhs),)

        elif isinstance(lhs, BlockStencilMatrixLocalBasis):
            lhs  = self._visit_BlockStencilMatrixLocalBasis(lhs)
            expr = self._visit(expr, op=op, lhs=lhs)
            return expr
        elif isinstance(lhs, BlockStencilMatrixGlobalBasis):

            dim          = self.dim
            rank         = lhs.rank
            pads         = lhs.pads
            multiplicity = lhs.multiplicity
            tests        = expand(lhs._tests)

            tests_2 = lhs._tests
            lhs     = self._visit_BlockStencilMatrixGlobalBasis(lhs)
            rhs     = self._visit(expr)

            pads       = self._visit(pads)
            rhs_slices = [Slice(None, None)]*rank
            for k1 in range(lhs.shape[0]):
                test    = tests[k1]
                test    = test if test in tests_2 else test.base
                spans   = self._visit_Span(Span(test))
                degrees = self._visit_LengthDofTest(LengthDofTest(test))
                spans   = flatten(*spans.values())
                m       = multiplicity[test] if test in multiplicity else multiplicity[test.base]

                lhs_starts = [spans[i]+m[i]*pads[i]-degrees[i] for i in range(dim)]
                lhs_ends   = [spans[i]+m[i]*pads[i]+1          for i in range(dim)]

                if isinstance(self._target, Interface):
                    axis = self._target.axis
                    lhs_starts[axis] = m[axis]*pads[axis]
                    lhs_ends[axis]   = m[axis]*pads[axis] + degrees[axis] + 1

                for k2 in range(lhs.shape[1]):
                    if expr.expr[k1,k2]:
                        lhs_slices  = [Slice(s, e) for s,e in zip(lhs_starts, lhs_ends)]
                        lhs_slices += [Slice(None, None)]*dim
                        lhs[k1,k2] = [lhs[k1,k2][lhs_slices]]
                        rhs[k1,k2] = [rhs[k1,k2][rhs_slices]]

            if op is None:
                return tuple( Assign(a, b) for a,b,e in zip(lhs[:], rhs[:], expr.expr[:]) if e)
            else:
                return tuple( AugAssign(a, op, b) for a,b,e in zip(lhs[:], rhs[:], expr.expr[:]) if e)

        elif isinstance(lhs, BlockStencilVectorLocalBasis):
            lhs = self._visit_BlockStencilVectorLocalBasis(lhs)
            expr = self._visit(expr, op=op, lhs=lhs)
            return expr
        elif isinstance(lhs, BlockStencilVectorGlobalBasis):
            dim   = self.dim
            rank  = lhs.rank
            pads  = lhs.pads
            multiplicity = lhs.multiplicity
            tests = expand(lhs._tests)
            tests_2 = lhs._tests
            lhs = self._visit_BlockStencilVectorGlobalBasis(lhs)
            rhs = self._visit(expr)
            pads    = self._visit(pads)
            rhs_slices = [Slice(None, None)]*rank

            for k in range(lhs.shape[0]):
                if expr.expr[k,0]:
                    test = tests[k]
                    m    = multiplicity[test] if test in multiplicity else multiplicity[test.base]
                    test = test if test in tests_2 else test.base
                    spans   = self._visit_Span(Span(test))
                    spans   = flatten(*spans.values())
                    degrees = self._visit_LengthDofTest(LengthDofTest(test))
                    lhs_starts = [spans[i]+m[i]*pads[i]-degrees[i] for i in range(dim)]
                    lhs_ends   = [spans[i]+m[i]*pads[i]+1          for i in range(dim)]
                    lhs_slices = [Slice(s, e) for s,e in zip(lhs_starts, lhs_ends)]
                    lhs[k,0] = lhs[k,0][lhs_slices]
                    rhs[k,0] = rhs[k,0][rhs_slices]

            return tuple( AugAssign(a, op, b) for a,b,e in zip(lhs[:], rhs[:], expr.expr[:]) if e)
        else:
            if not( lhs is None ):
                lhs = self._visit(lhs)

            return self._visit(expr, op=op, lhs=lhs)

    # ....................................................
    def _visit_ComputeLogical(self, expr, op=None, lhs=None, **kwargs):
        expr = expr.expr
        if lhs is None:
            if not isinstance(expr, (Add, Mul)):
                lhs = self._visit_AtomicNode(AtomicNode(expr), **kwargs)
            else:
                lhs = random_string( 6 )
                lhs = Symbol('tmp_{}'.format(lhs))

        node = LogicalValueNode(expr)
        rhs = self._visit_LogicalValueNode(node, **kwargs)

        if op is None:
            stmt = Assign(lhs, rhs)
        else:
            stmt = AugAssign(lhs, op, rhs)

        return self._visit(stmt, **kwargs)

    # ....................................................
    def _visit_ComputeLogicalBasis(self, expr, op=None, lhs=None, **kwargs):
        lhs  = lhs or expr.lhs
        expr = expr.expr
        if lhs is None:
            atom = BasisAtom(expr)
            lhs  = self._visit_BasisAtom(atom, **kwargs)

        rhs = self._visit_LogicalBasisValue(LogicalBasisValue(expr), **kwargs)

        if op is None:
            stmt = Assign(lhs, rhs)
        else:
            stmt = AugAssign(lhs, op, rhs)

        return self._visit(stmt, **kwargs)

    # ....................................................
    def _visit_ComputeKernelExpr(self, expr, op=None, lhs=None, **kwargs):
        """
        Compute Symbolic expression given by the user

        """
        if lhs is None:
            if not isinstance(expr, (Add, Mul)):
                lhs = self._visit_BasisAtom(BasisAtom(expr), **kwargs)
            else:
                lhs = random_string( 6 )
                lhs = Symbol('tmp_{}'.format(lhs))

        exprs   = expr.expr
        mapping = self.mapping

        if expr.weights:
            weight  = SymbolicWeightedVolume(mapping)
            weight  = SymbolicExpr(weight)
        else:
            weight  = 1

        rhs = [weight*self._visit(expr, **kwargs) for expr in exprs[:]]
        lhs = lhs[:]

        # Create a new name for the temporaries used in each patch
        name = get_name(lhs)
        temps, rhs = cse_main.cse(rhs, symbols=cse_main.numbered_symbols(prefix=f'temp{name}'))

        normal_vec_stmts = []
        normal_vectors = expr.expr.atoms(NormalVector)
        target         = self._target
        dim            = self._dim

        if normal_vectors:
            axis    = target.axis
            ext     = target.ext if isinstance(target, Boundary) else 1

        vars_plus = []
        if isinstance(target, Interface):
            mapping = mapping.minus
            target  = target.minus
            axis    = target.axis
            ext     = target.ext
        elif isinstance(target, Boundary):
            ext  = target.ext
            axis = target.axis


        for vec in normal_vectors:

            J_inv   = LogicalExpr(mapping.jacobian_inv_expr, mapping(target))
            J_inv   = SymbolicExpr(J_inv)
            values  = ext * J_inv[axis, :]
            normalization = values.dot(values)**0.5
            values  = [v for v in values]
            values  = [v1/normalization for v1 in values]
            normal_vec_stmts += [Assign(SymbolicExpr(vec[i]), values[i]) for i in range(dim)]

        if op is None:
            stmts = [Assign(i, j) for i,j in zip(lhs,rhs) if j]
        else:
            stmts = [AugAssign(i, op, j) for i,j in zip(lhs,rhs) if j]

        temps = tuple(Assign(a,b) for a,b in temps)
        stmts = tuple(self._visit(stmt, **kwargs) for stmt in stmts)
        stmts = tuple(vars_plus) + tuple(normal_vec_stmts) + temps + stmts

        math_functions = math_atoms_as_str(list(exprs)+normal_vec_stmts, 'math')
        math_functions = tuple(m for m in math_functions if m not in self._math_functions)
        self._math_functions = math_functions + self._math_functions
        return stmts

    # ....................................................
    def _visit_BasisAtom(self, expr, **kwargs):
        """
        Transform derivatives of the ScalarFunction into the correspondant symbol.
        """
        symbol = SymbolicExpr(expr.expr)
        self.variables[str(symbol.name)] = symbol
        return symbol

    # ....................................................
    def _visit_AtomicNode(self, expr, **kwargs):
        if isinstance(expr.expr, WeightedVolumeQuadrature):
            expr = SymbolicWeightedVolume(self.mapping)
            return SymbolicExpr(expr)

        else:
            return SymbolicExpr(expr.expr)

    # ....................................................
    def _visit_LogicalBasisValue(self, expr, **kwargs):
        """
        Split the derivatives of the ScalarFunction along the dimensions, transform it into the correspondant symbol.
        """
        # ...
        dim = self.dim
        coords = ['x1', 'x2', 'x3'][:dim]

        expr   = expr.expr
        atom   = BasisAtom(expr).atom

        # Split the ScalarFunction along each dimension
        atoms  = _split_test_function(atom)
        ops = [dx1, dx2, dx3][:dim]
        d_atoms = dict(zip(coords, atoms[atom]))
        d_ops   = dict(zip(coords, ops))
        d_indices = get_index_logical_derivatives(expr)

        # Create the symbol of the derivative for each splitted ScalarFunction
        args = []
        for k, u in d_atoms.items():
            d = d_ops[k]
            n = d_indices[k]
            for _ in range(n):
                u = d(u)
            u = SymbolicExpr(u)
            args.append(u)

        # ...
        # Do the multiplication needed
        expr = Mul(*args)

        return expr

    # ....................................................
    def _visit_LogicalValueNode(self, expr, **kwargs):
        """
        This Node seems to return the multiplication of weight.
        """

        #TODO Should we clear this function and replace it by a _visit_WeigthedVolumeQuadrature
        expr = expr.expr
        target = self.target

        if isinstance(expr, WeightedVolumeQuadrature):
            #TODO improve l_quad should not be used like this
            l_quad = self._visit_TensorQuadrature(TensorQuadrature(), **kwargs)
            _, weights = list(zip(*list(l_quad.values())[0]))
            if isinstance(target, Boundary):
                weights = list(weights)
                weights.pop(target.axis)
            wvol = Mul(*weights)
            return wvol
        else:
            raise TypeError('{} not available'.format(type(expr)))

    # ....................................................
    def _visit_PhysicalGeometryValue(self, expr, **kwargs):
        target  = self._target
        expr = LogicalExpr(expr.expr, mapping(target))

        return SymbolicExpr(expr)

    # ....................................................
    def _visit_ElementOf(self, expr, **kwargs):
        """
        Create an MutableDenseMatrix containing either a variable for a scalar or an IndexedElement to index an element of the matrix/vector
        """
        dim    = self.dim
        target = expr.target


        # Case where we need to create an element of the matrix indented
        if isinstance(target, BlockStencilMatrixLocalBasis):
            # improve we shouldn't use index_dof_test
            rows = self._visit(index_dof_test)
            outer = self._visit(target.outer) if target.outer else rows
            cols = self._visit(index_dof_trial)
            pads = target.pads
            tests  = expand(target._tests)
            trials = expand(target._trials)

            targets = self._visit_BlockStencilMatrixLocalBasis(target)
            for i in range(targets.shape[0]):
                for j in range(targets.shape[1]):
                    if targets[i,j] is S.Zero:
                        continue
                    if trials[j] in pads.trials_multiplicity:
                        trials_m  = pads.trials_multiplicity[trials[j]]
                        trials_d  = pads.trials_degree[trials[j]]
                    else:
                        trials_m = pads.trials_multiplicity[trials[j].base]
                        trials_d = pads.trials_degree[trials[j].base]

                    if tests[i] in pads.tests_multiplicity:
                        tests_m  = pads.tests_multiplicity[tests[i]]
                        tests_d  = pads.tests_degree[tests[i]]
                    else:
                        tests_m = pads.tests_multiplicity[tests[i].base]
                        tests_d = pads.tests_degree[tests[i].base]

                    pp1     = [max(tests_d[k], trials_d[k]) for k in range(dim)]
                    pp2     = [int((np.ceil((pp1[k]+1)/tests_m[k])-1)*trials_m[k]) for k in range(dim)]
                    padding = [p2-min(0,p2-p1) for p1,p2 in zip(pp1, pp2)]
                    indices = tuple(rows) + tuple(cols[k]+padding[k]-outer[k]*trials_m[k] for k in range(dim))
                    targets[i,j] = targets[i,j][indices]
            return targets

        # Case where we need to create an element of the vector indented
        elif isinstance(target, BlockStencilVectorLocalBasis):
            targets = self._visit_BlockStencilVectorLocalBasis(target, **kwargs)

            rows = self._visit(index_dof_test)
            indices = list(rows)
            for i in range(targets.shape[0]):
                for j in range(targets.shape[1]):
                    if targets[i,j] is S.Zero:
                        continue
                    targets[i,j] = targets[i,j][indices]
            return targets

        # Case where we need to create a scalar for the kernel loop (l_el_{tag})
        elif isinstance(target, LocalElementBasis):
            target = self._visit(target, **kwargs)
            return (target,)

        # Case where we need to create a scalar for the kernel loop (c_v_u_{tag})/(c_v_{tag})
        elif isinstance(target, BlockScalarLocalBasis):
            targets = self._visit(target)
            return targets

        else:
            raise NotImplementedError('TODO')

    # .............................................................................
    def _visit_BlockStencilMatrixLocalBasis(self, expr, **kwargs):
        pads    = self._visit_Pads(expr.pads)
        tests   = expr._tests
        trials  = expr._trials
        tag     = expr.tag
        dtype   = expr.dtype
        tests   = expand(tests)
        trials  = expand(trials)
        targets = Matrix.zeros(len(tests), len(trials))
        for i, v in enumerate(tests):
            for j, u in enumerate(trials):
                if expr.expr[i, j] == 0:
                    continue
                mat = StencilMatrixLocalBasis(u=u, v=v, pads=pads[i, j], tag=tag, dtype=dtype)
                mat = self._visit_StencilMatrixLocalBasis(mat, **kwargs)
                targets[i, j] = mat
        return targets

    def _visit_BlockStencilMatrixGlobalBasis(self, expr, **kwargs):
        pads    = expr.pads
        tests   = expr._tests
        trials  = expr._trials
        tag     = expr.tag
        dtype   = expr.dtype
        tests   = expand(tests)
        trials  = expand(trials)
        targets = Matrix.zeros(len(tests), len(trials))
        for i, v in enumerate(tests):
            for j, u in enumerate(trials):
                if expr.expr[i, j] == 0:
                    continue
                mat = StencilMatrixGlobalBasis(u=u, v=v, pads=pads, tag=tag, dtype=dtype)
                mat = self._visit_StencilMatrixGlobalBasis(mat, **kwargs)
                targets[i, j] = mat
        return targets

    def _visit_BlockStencilVectorLocalBasis(self, expr, **kwargs):
        pads    = expr.pads
        tests   = expr._tests
        tag     = expr.tag
        dtype   = expr.dtype
        tests   = expand(tests)
        targets = Matrix.zeros(len(tests), 1)
        for i, v in enumerate(tests):
            if expr.expr[i, 0] == 0:
                continue
            mat = StencilVectorLocalBasis(v, pads, tag, dtype)
            mat = self._visit_StencilVectorLocalBasis(mat, **kwargs)
            targets[i, 0] = mat
        return targets

    def _visit_BlockStencilVectorGlobalBasis(self, expr, **kwargs):
        pads    = expr.pads
        tests   = expr._tests
        tag     = expr.tag
        dtype   = expr.dtype
        tests   = expand(tests)
        targets = Matrix.zeros(len(tests), 1)
        for i,v in enumerate(tests):
            if expr.expr[i, 0] == 0:
                continue
            mat = StencilVectorGlobalBasis(v, pads, tag, dtype)
            mat = self._visit_StencilVectorGlobalBasis(mat, **kwargs)
            targets[i, 0] = mat
        return targets

    # .............................................................................
    def _visit_BlockScalarLocalBasis(self, expr, **kwargs):
        tag     = expr.tag
        dtype   = expr.dtype
        tests   = expand(expr._tests)
        trials  = expand(expr._trials) if expr._trials else (None,)
        targets = Matrix.zeros(len(tests), len(trials))
        for i,v in enumerate(tests):
            for j,u in enumerate(trials):
                if expr.expr[i, j] == 0:
                    continue
                var = ScalarLocalBasis(u, v, tag, dtype)
                var = self._visit_ScalarLocalBasis(var, **kwargs)
                targets[i, j] = var
        return targets

    # .............................................................................
    def _visit_StencilMatrixLocalBasis(self, expr, **kwargs):
        rank   = expr.rank
        tag    = expr.tag
        dtype  = expr.dtype
        name   = '_'.join(str(SymbolicExpr(e)) for e in expr.name)

        name = 'l_mat_{}_{}'.format(name, tag)
        var  = IndexedVariable(name, dtype=dtype, rank=rank)
        self.insert_variables(var)
        return var

    # ....................................................
    def _visit_StencilVectorLocalBasis(self, expr, **kwargs):
        rank  = expr.rank
        tag   = expr.tag
        dtype = expr.dtype
        name  = str(SymbolicExpr(expr.name))
        name  = 'l_vec_{}_{}'.format(name, tag)
        var   = IndexedVariable(name, dtype=dtype, rank=rank)
        self.insert_variables(var)
        return var

    # ....................................................
    def _visit_StencilMatrixGlobalBasis(self, expr, **kwargs):
        rank  = expr.rank
        tag   = expr.tag
        dtype = expr.dtype
        name  = '_'.join(str(SymbolicExpr(e)) for e in expr.name)
        name  = 'g_mat_{}_{}'.format(name, tag)
        var   = IndexedVariable(name, dtype=dtype, rank=rank)
        self.insert_variables(var)
        return var

    # ....................................................
    def _visit_StencilVectorGlobalBasis(self, expr, **kwargs):
        rank  = expr.rank
        tag   = expr.tag
        dtype = expr.dtype
        name  = str(SymbolicExpr(expr.name))
        name  = 'g_vec_{}_{}'.format(name, tag)
        var   = IndexedVariable(name, dtype=dtype, rank=rank)
        self.insert_variables(var)
        return var

    def _visit_GlobalElementBasis(self, expr, **kwargs):
        tag   = expr.tag
        dtype = expr.dtype
        name  = 'g_el_{}'.format(tag)
        var   = variables(name, dtype=dtype)
        self.insert_variables(var)
        return var

    def _visit_LocalElementBasis(self, expr, **kwargs):
        tag   = expr.tag
        dtype = expr.dtype
        name  = 'l_el_{}'.format(tag)
        var   = variables(name, dtype=dtype)
        self.insert_variables(var)
        return var

    def _visit_ScalarLocalBasis(self, expr, **kwargs):
        tag   = expr.tag
        dtype = expr.dtype
        basis = (expr._test,)
        if expr._trial:
            basis = (expr._test, expr._trial)
        name = '_'.join(str(SymbolicExpr(e)) for e in basis)
        name = 'contribution_{}_{}'.format(name, tag)
        var  = variables(name, dtype=dtype)
        self.insert_variables(var)
        return var

    # ....................................................
    def _visit_Pattern(self, expr, **kwargs):
        # this is for multi-indices for the moment
        dim = self.dim
        args = []
        for a in expr:
            if a is None:
                args.append([Slice(None, None)]*dim)

            elif isinstance(a, int):
                args.append([a]*dim)

            else:
                v = self._visit(a)
                args.append(v)
        args = list(zip(*args))
        return args

    def _visit_Slice(self, expr, **kwargs):
        args = [self._visit(a) if a is not None else None for a in expr.args]
        return Slice(args[0], args[1])

    def _visit_TensorIntDiv(self, expr, **kwargs):
        args = [self._visit(a, **kwargs) for a in expr.args]
        arg1 = args[0]
        arg2 = args[1]
        newargs = []
        for i,j in zip(arg1, arg2):
            newargs.append(i//j)
            
        return tuple(newargs)

    def _visit_TensorAdd(self, expr, **kwargs):
        args = [self._visit(a, **kwargs) for a in expr.args]
        arg1 = args[0]
        arg2 = args[1]
        newargs = []
        for i,j in zip(arg1, arg2):
            newargs.append(i+j)
            
        return tuple(newargs)

    def _visit_TensorMul(self, expr, **kwargs):
        args = [self._visit(a, **kwargs) for a in expr.args]
        arg1 = args[0]
        arg2 = args[1]
        newargs = []
        for i,j in zip(arg1, arg2):
            newargs.append(i*j)
            
        return tuple(newargs)

    def _visit_TensorMax(self, expr, **kwargs):
        args = [self._visit(a, **kwargs) for a in expr.args]
        arg1 = args[0]
        arg2 = args[1]
        newargs = []
        for i,j in zip(arg1, arg2):
            newargs.append(Max(i,j))

        return tuple(newargs)

    def _visit_TensorInteger(self, expr, **kwargs):
        return (expr.args[0],)*self.dim
    # ....................................................

    def _visit_Max(self, expr, **kwargs):
        args = [self._visit(i) for i in expr.args]
        return Max(*args)

    def _visit_Min(self, expr, **kwargs):
        args = [self._visit(i) for i in expr.args]
        return Min(*args)

    def _visit_Expr(self, expr, **kwargs):
        return SymbolicExpr(expr)

    def _visit_Return( self, expr, **kwargs):
        return Return(self._visit(expr.expr))

    def _visit_NumThreads(self, expr, **kwargs):
        target =  variables('num_threads', dtype='int')
        self.insert_variables(target)
        return target

    def _visit_BooleanTrue(self, expr, **kwargs):
        return int(True)

    def _visit_BooleanFalse(self, expr, **kwargs):
        return int(False)
    # ....................................................
    def _visit_ThreadId(self, expr, **kwargs):
        return variables('thread_id', dtype='int')
    # ...................................................
    def _visit_NeighbourThreadCoordinates(self, expr, **kwargs):
        dim    = self.dim
        target =  variables('next_thread_coords_1:%d'%(dim+1), dtype='int')
        if expr.index is not None:
            return target[expr.index]
        self.insert_variables(*target)
        return target
    # ....................................................
    def _visit_ThreadCoordinates(self, expr, **kwargs):
        dim    = self.dim
        target =  variables('thread_coords_1:%d'%(dim+1), dtype='int')
        if expr.index is not None:
            return target[expr.index]
        return target
    # ....................................................
    def _visit_IndexElement(self, expr, **kwargs):
        dim    = self.dim
        target =  variables('i_element_1:%d'%(dim+1), dtype='int')
        self.insert_variables(*target)
        return target
    # ....................................................
    def _visit_LocalIndexElement(self, expr, **kwargs):
        dim    = self.dim
        target =  variables('local_i_element_1:%d'%(dim+1), dtype='int')
        if expr.index is not None:
            return target[expr.index]
        return target
    # ....................................................
    def _visit_IndexQuadrature(self, expr, **kwargs):
        dim = self.dim
        target = variables('i_quad_1:%d'%(dim+1), dtype='int')
        self.insert_variables(*target)
        return target
    # ....................................................
    def _visit_IndexDof(self, expr, **kwargs):
        dim = self.dim
        target = variables('i_basis_1:%d'%(dim+1), dtype='int')
        self.insert_variables(*target)
        return target
    # ....................................................
    def _visit_IndexDofTrial(self, expr, **kwargs):
        dim = self.dim
        target = variables('j_basis_1:%d'%(dim+1), dtype='int')
        self.insert_variables(*target)
        return target
    # ....................................................
    def _visit_IndexDofTest(self, expr, **kwargs):
        dim = self.dim
        target = variables('i_basis_1:%d'%(dim+1), dtype='int')
        self.insert_variables(*target)
        return target
    # ....................................................
    def _visit_IndexOuterDofTest(self, expr, **kwargs):
        dim = self.dim
        target = variables('outer_i_basis_1:%d'%(dim+1), dtype='int')
        self.insert_variables(*target)
        return target
    # ....................................................
    def _visit_IndexInnerDofTest(self, expr, **kwargs):
        dim = self.dim
        target = variables('inner_i_basis_1:%d'%(dim+1), dtype='int')
        self.insert_variables(*target)
        return target
    # ....................................................
    def _visit_IndexDerivative(self, expr, **kwargs):
        raise NotImplementedError('TODO')

    # ....................................................
    def _visit_LengthElement(self, expr, **kwargs):
        dim = self.dim
        names = 'n_element_1:%d'%(dim+1)
        target = variables(names, dtype='int', cls=Variable)
        if expr.index is not None:
            return target[expr.index]
        self.insert_variables(*target)
        return target
    # ....................................................
    def _visit_LengthQuadrature(self, expr, **kwargs):
        dim = self.dim
        names = 'k1:%d'%(dim+1)
        target = variables(names, dtype='int', cls=Variable)
        self.insert_variables(*target)
        return target
    # ....................................................
    def _visit_LengthDof(self, expr, **kwargs):
        dim = self.dim
        names = 'p1:%d'%(dim+1)
        target = variables(names, dtype='int')
        self.insert_variables(*target)
        return target
    # ....................................................
    def _visit_LengthDofTest(self, expr, **kwargs):
        dim = self.dim
        target = expr.target
        if target:
            target = '_' + str(SymbolicExpr(target))
        else:
            target = ''

        names = 'test{}_p1:{}'.format(target, dim+1)
        target = variables(names, dtype='int')
        if expr.index is not None:
            return target[expr.index]
        self.insert_variables(*target)
        return target
    # ....................................................
    def _visit_LengthOuterDofTest(self, expr, **kwargs):
        dim = self.dim
        target = expr.target
        if target:
            target = '_' + str(SymbolicExpr(target))
        else:
            target = ''

        names = 'test_outer{}_p1:{}'.format(target, dim+1)
        target = variables(names, dtype='int')
        self.insert_variables(*target)
        return target
    # ....................................................
    def _visit_LengthInnerDofTest(self, expr, **kwargs):
        dim = self.dim
        target = expr.target
        if target:
            target = '_' + str(SymbolicExpr(target))
        else:
            target = ''

        names = 'test_inner{}_p1:{}'.format(target, dim+1)
        target = variables(names, dtype='int')
        self.insert_variables(*target)
        return target

    # ....................................................
    def _visit_LengthDofTrial(self, expr, **kwargs):
        dim = self.dim
        target = expr.target
        if target:
            target = '_' + str(SymbolicExpr(target))
        else:
            target = ''

        names = 'trial{}_p1:{}'.format(target, dim+1)
        target = variables(names, dtype='int')
        self.insert_variables(*target)
        return target
    # ....................................................
    def _visit_RankDimension(self, expr, **kwargs):
        return self.dim

    # ....................................................
    def _visit_TensorIterator(self, expr, **kwargs):
        target = self._visit(expr.target)
        return target

    # ....................................................
    def _visit_ProductIterator(self, expr, **kwargs):
        target = self._visit(expr.target)
        return target

    # ....................................................
    def _visit_TensorGenerator(self, expr, **kwargs):

        targets = self._visit(expr.target)
        if expr.dummies is None:
            #TODO check if we never pass this condition
            return expr.target

        if not hasattr(expr.target, 'pattern'):
            return targets

        patterns = expr.target.pattern()
        patterns = self._visit_Pattern(patterns)
        args = {}
        for i,target in targets.items():
            args[i] = []
            for p, xs in zip(patterns, target):
                ls = []
                for x in xs:
                    ls.append(x[p])
                args[i].append(tuple(ls))
            args[i] = tuple(args[i])

        return args
    # ....................................................
    def _visit_ProductGenerator(self, expr, **kwargs):
        target = self._visit(expr.target)

        # treat dummies and put them in the namespace
        dummies = self._visit(expr.dummies)
        dummies = dummies[0] # TODO add comment
        return target[dummies]

    # ....................................................
    def _visit_TensorIteration(self, expr, **kwargs):
        """
        Initialize index in loop
        Parameters
        ----------
        expr
        kwargs

        Returns
        -------
        inits : list
            Initialization instructions

        """
        dim       = self.dim
        iterator  = self._visit(expr.iterator)
        generator = self._visit(expr.generator)

        # Case of a simple iterable
        if isinstance(iterator, (tuple, Tuple, list)):
            return [[Assign(i, g)] for i, g in zip(iterator, generator)]

        # Case of a dictionary

        inits = [()]*dim

        for l_xs, g_xs in zip(iterator.values(), generator.values()):
            if isinstance(expr.generator.target, (LocalTensorQuadratureBasis, GlobalTensorQuadratureBasis)):
                positions = [expr.generator.target.positions[index_deriv]]
                g_xs = [SplitArray(xs[0], positions, [self.nderiv+1]) for xs in g_xs]
                g_xs = [tuple(self._visit(xs, **kwargs)) for xs in g_xs]

            for i in  range(dim):
                ls = []
                for l_x,g_x in zip(l_xs[i], g_xs[i]):
                    if isinstance(expr.generator.target, (LocalTensorQuadratureBasis, GlobalTensorQuadratureBasis)):
                        lhs = self._visit_BasisAtom(BasisAtom(l_x))
                    else:
                        lhs = l_x
                    ls += [self._visit(Assign(lhs, g_x))]
                inits[i] += tuple(ls)
        inits = [flatten(init) for init in inits]
        return  inits

    # ....................................................
    def _visit_ProductIteration(self, expr, **kwargs):
        # TODO for the moment, we do not return indices and lengths
        iterator  = self._visit(expr.iterator)
        generator = self._visit(expr.generator)
    
        return Assign(iterator, generator)

    def _visit_RAT(self, expr):
        return str(expr)

    def _visit_WhileLoop(self, expr, **kwargs):
        cond = self._visit(expr.condition)
        body = [self._visit(a) for a in expr.body]
        return While(cond, body)

    def _visit_IfNode(self, expr, **kwargs):
        args = []
        for a in expr.args:
            cond = self._visit(a[0])
            body = [self._visit(i) for i in a[1]]
            args += [(cond, body)]
        return If(*args)
    # ....................................................
    def _visit_Loop(self, expr, **kwargs):
        """
        Create
        """
        # we first create iteration statements
        # these iterations are splitted between what is tensor or not

        # ... treate tensor iterations

        t_iterator   = [i for i in expr.iterator  if isinstance(i, TensorIterator)]
        t_generator  = [i for i in expr.generator if isinstance(i, TensorGenerator)]
        t_iterations = [TensorIteration(i, j)
                        for i,j in zip(t_iterator, t_generator)]

        indices = list(self._visit(expr.index))
        starts, stops, lengths = list(self._visit(expr.index.start)), list(self._visit(expr.index.stop)), list(self._visit(expr.index.length))

        for i,j in zip(flatten(indices), flatten(lengths)):
            self.indices[str(i)] = j

        inits = [()]*self._dim
        if t_iterations:
            t_iterations = [self._visit_TensorIteration(i) for i in t_iterations]

            # indices and lengths are supposed to be repeated here
            # we only take the first occurence
            for init in t_iterations:
                for i in range(self._dim):
                    inits[i] += tuple(init[i])

        # ...
        # ... treate product iterations
        p_iterator   = [i for i in expr.iterator  if isinstance(i, ProductIterator)]
        p_generator  = [i for i in expr.generator if isinstance(i, ProductGenerator)]
        p_iterations = [ProductIteration(i,j)
                        for i,j in zip(p_iterator, p_generator)]

        p_inits = []
        if p_iterations:
            p_inits = [self._visit_ProductIteration(i) for i in p_iterations]
        # ...

        # ... add weighted volume if local quadrature loop
        mapping = self.mapping
        geo_stmts = expr.get_geometry_stmts(mapping)
        geo_stmts = self._visit(geo_stmts, **kwargs)
        # ...

        # ...
        # visit loop statements

        stmts = self._visit(expr.stmts, **kwargs)
        stmts = flatten(stmts)

        # update with product statements if available
        body = list(p_inits) + list(geo_stmts) + list(stmts)
        mask = expr.mask

        if isinstance(mask,(tuple,Tuple,list)):
            mask_init = []         
            for axis,T in enumerate(mask):
                if T:
                    indices[axis] = None
                    starts [axis] = None
                    stops  [axis] = None
                    mask_init    += list(inits[axis])
                    inits[axis]   = None
            indices = [i for i in indices if i is not None]
            starts  = [i for i in starts  if i is not None]
            stops   = [i for i in stops   if i is not None]
            inits   = [i for i in inits   if i is not None]

        elif mask:
            axis      = mask.axis
            index     = indices.pop(axis)
            start     = starts.pop(axis)
            stop      = stops.pop(axis)
            init      = inits.pop(axis)
            mask_init = [Assign(index, 0), *init]

        if expr.parallel:
            body = list(flatten(inits)) + body
            for index, s, e in zip(indices[::-1], starts[::-1], stops[::-1]):
                body = [For(index, Range(s, e), body)]
        else:
            for index, s, e, init in zip(indices[::-1], starts[::-1], stops[::-1], inits[::-1]):

                body = list(init) + body
                body = [For(index, Range(s, e), body)]
        # ...
        # remove the list and return the For Node only

        if mask:
            body = [*mask_init, *body]

        if expr.parallel:
            default      = expr.default
            shared       = [self._visit(i) for i in expr.shared] if expr.shared  else []
            private      = [self._visit(i) for i in expr.private] if expr.private  else []
            firstprivate = [self._visit(i) for i in expr.firstprivate] if expr.firstprivate  else []
            lastprivate  = [self._visit(i) for i in expr.lastprivate] if expr.lastprivate  else []
            shared       = flatten([list(i.values())[0] if isinstance(i, dict) else i for i in shared])
            private      = flatten([list(i.values())[0] if isinstance(i, dict) else i for i in private])
            firstprivate = flatten([list(i.values())[0] if isinstance(i, dict)else i for i in firstprivate])
            lastprivate  = flatten([list(i.values())[0] if isinstance(i, dict) else i for i in lastprivate])
            txt          = '#$ omp parallel default({}) &\n'.format(default)
            txt         += '#$ shared({}) &\n'.format(','.join(str(i) for i in shared if i)) if shared else ''
            txt         += '#$ private({}) &\n'.format(','.join(str(i) for i in private if i)) if private else ''
            txt         += '#$ firstprivate({}) &\n'.format(','.join(str(i) for i in firstprivate if i)) if firstprivate else ''
            txt         += '#$ lastprivate({})'.format(','.join(str(i) for i in lastprivate if i)) if lastprivate else ''
            for_pragmas  = '#$ omp for schedule(static) collapse({})'.format(self._dim)
            if expr.reduction:
                for_pragmas = for_pragmas + expr.reduction 

            cmt          = [Comment(txt.rstrip().rstrip('&')), Comment(for_pragmas)]
            endcmt       = [Comment('#$ omp end parallel')]
            body         = [*cmt, *body, *endcmt]

        if len(body) > 1:
            body = CodeBlock(body)
        elif len(body) == 1:
            body = body[0]

        return body

    # ....................................................
    def _visit_SplitArray(self, expr, **kwargs):
        target    = expr.target
        positions = expr.positions
        lengths   = expr.lengths
        base      = target.base

        args = []
        for p,n in zip(positions, lengths):
            indices = target.indices # sympy is return a tuple of tuples
            indices = [i for i in indices] # make a copy
            for i in range(n):
                indices[p] = i
                x = base[tuple(indices)]
                args.append(x)
        return args

    def _visit_Comment(self, expr, **kwargs):
        return expr

    # ....................................................
    def _visit_IndexedElement(self, expr, **kwargs):
        return expr

    # ....................................................
    # TODO to be removed. usefull for testing
    def _visit_Pass(self, expr, **kwargs):
        return expr

    def _visit_Continue(self, expr, **kwargs):
        return expr

    def _visit_EmptyNode(self ,expr, **kwargs):
        return expr

    def _visit_NoneType(self, expr, **kwargs):
        return expr

    def _visit_int(self, expr, **kwargs):
        return expr

    def _visit_float(self, expr, **kwargs):
        return expr

    def _visit_complex(self, expr, **kwargs):
        return expr

