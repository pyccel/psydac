from collections import OrderedDict

from sympy import IndexedBase, Indexed
from sympy import Mul, Matrix
from sympy import Add
from sympy import Abs
from sympy import Symbol, Idx
from sympy import Max, Range
from sympy import Basic, Function
from sympy.simplify import cse_main
from sympy.core.containers import Tuple


from pyccel.ast.core      import Assign, Product, AugAssign, For
from pyccel.ast.core      import Variable, IndexedVariable, IndexedElement
from pyccel.ast.core      import Slice, String, ValuedArgument
from pyccel.ast.core      import EmptyNode, Import
from pyccel.ast.core      import CodeBlock, FunctionDef


from sympde.topology import (dx1, dx2, dx3)
from sympde.topology import SymbolicExpr
from sympde.topology import LogicalExpr, Jacobian
from sympde.expr.evaluation import _split_test_function
from sympde.calculus.matrices import SymbolicDeterminant
from sympde.topology import SymbolicWeightedVolume, InterfaceMapping
from sympde.topology import Boundary, NormalVector, Interface

from sympde.topology.derivatives import get_index_logical_derivatives

from .nodes import AtomicNode
from .nodes import BasisAtom
from .nodes import PhysicalBasisValue
from .nodes import LogicalBasisValue
from .nodes import TensorQuadrature
from .nodes import LocalTensorQuadratureBasis
from .nodes import LocalTensorQuadratureTestBasis
from .nodes import LocalTensorQuadratureTrialBasis
from .nodes import GlobalTensorQuadratureTestBasis
from .nodes import GlobalTensorQuadratureTrialBasis
from .nodes import GlobalTensorQuadratureBasis
from .nodes import TensorQuadratureBasis
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
from .nodes import StencilMatrixGlobalBasis
from .nodes import BlockStencilMatrixLocalBasis
from .nodes import BlockStencilMatrixGlobalBasis
from .nodes import BlockStencilVectorLocalBasis
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
from .nodes import index_deriv

from .nodes import   Zeros, ZerosLike
from .fem import expand, expand_hdiv_hcurl
from psydac.api.ast.utilities import variables, math_atoms_as_str
from psydac.api.utilities     import flatten
from psydac.api.ast.utilities import build_pythran_types_header
from psydac.api.ast.utilities import build_pyccel_types_decorator

#==============================================================================
# TODO move it
import string
import random
def random_string( n ):
    chars    = string.ascii_lowercase + string.digits
    selector = random.SystemRandom()
    return ''.join( selector.choice( chars ) for _ in range( n ) )

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
    This function takes a Psydac Ast and returns a Pyccel Ast

    Parameters
    ----------

    expr: <Psydac Ast>
        psydac ast node

    settings : <dict>
        dictionary that continas number of dimension, mappings and target if provided

    Returns
    -------

    ast : Pyccel Ast
        pyccel abstract syntax tree that can be translated into a Python code

    """
    psy_parser = Parser(settings, backend)
    ast = psy_parser.doit(expr)
    return ast

#==============================================================================
class Parser(object):
    """
    This class takes a Psyadac Ast and transforms it to a Pyccel Ast
    by calling the Parser.doit method

    """
    def __init__(self, settings, backend=None):

        settings = settings.copy()

        dim = settings.pop('dim', None)
        if dim is None:
            raise ValueError('dim not provided')

        self._dim = dim
        # ...

        nderiv = settings.pop('nderiv', None)
        if nderiv is None:
            raise ValueError('nderiv not provided')

        self._nderiv = nderiv

        target = settings.pop('target', None)
        if target is None:
            raise ValueError('target not provided')

        self._target = target

        self._mapping = settings.pop('mapping', None)

        self._settings = settings
        self.backend   = backend

        # TODO improve
        self.indices          = OrderedDict()
        self.shapes           = OrderedDict()
        self.functions        = OrderedDict()
        self.variables        = OrderedDict()
        self.arguments        = OrderedDict()
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
        shape_lhs = None
        shape = []

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
                    shape = tuple(Slice(None,None) if i is None else 0 for i in shape)
                    rhs = rhs.base
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
    # ....................................................
    def _visit_Add(self, expr, **kwargs):
        args = [self._visit(i) for i in expr.args]
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

    # ....................................................
    def _visit_Block(self, expr, **kwargs):
        body = [self._visit(i) for i in expr.body]
        body = flatten(body)
        if len(body) == 1:
            return body[0]

        else:
            return CodeBlock(body)

    def _visit_DefNode(self, expr, **kwargs):

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
        
        map_coeffs = args.pop('mapping', None)
        constants  = args.pop('constants', None)

        f_coeffs   = args.pop('f_coeffs',    None)

        if f_coeffs:
            f_span     = args.pop('f_span',      [])
            f_basis    = args.pop('field_basis', [])
            f_degrees  = args.pop('fields_degrees', [])
            f_args     = (*f_basis, *f_span, *f_degrees, *f_coeffs)

        inits = []
        if l_pads:
            tests = l_pads.tests
            trials = l_pads.trials
            l_pads = self._visit(l_pads , **kwargs)
            for i,v in enumerate(expand(tests)):
                for j,u in enumerate(expand(trials)):
                    test_ln = lengths_tests[v] if v in lengths_tests else lengths_tests[v.base]
                    trial_ln = lengths_trials[u] if u in lengths_trials else lengths_trials[u.base]
                    test_ln = self._visit(test_ln, **kwargs)
                    trial_ln = self._visit(trial_ln, **kwargs)
                    for stmt in zip(l_pads[i,j], test_ln, trial_ln):
                        inits += [Assign(stmt[0], Function('max')(tuple(stmt[1:])))]

        args = [*tests_basis, *trial_basis, *g_span, g_quad, *lengths_tests.values(), *lengths_trials.values(), *lengths, *g_pads]

        if isinstance(mats[0], (LocalElementBasis, GlobalElementBasis)):
            mats = [self._visit(mat) for mat in mats]
        else:
            exprs     = [mat.expr for mat in mats]

            mats      = [self._visit(mat) for mat in mats]
            mats      = [[a for a,e in zip(mat[:],expr[:]) if e] for mat,expr in zip(mats, exprs)]
            mats      = flatten(mats)

        args = [self._visit(i, **kwargs) for i in args]
        args = [tuple(arg.values())[0] if isinstance(arg, dict) else arg for arg in args]
        arguments = flatten(args) + mats

        if map_coeffs:
            arguments += [self._visit(i, **kwargs) for i in map_coeffs]

        if constants:
            arguments += [self._visit(i, **kwargs) for i in constants]

        if f_args:
            f_args     = [self._visit(i, **kwargs) for i in f_args]
            f_args     = [tuple(arg.values())[0] if isinstance(arg, dict) else arg for arg in f_args]
            arguments += flatten(f_args)

        body = flatten(tuple(self._visit(i, **kwargs) for i in expr.body))

        for k,i in self.shapes.items():
            var = self.variables[k]
            if var in arguments:
                continue
            if isinstance(i, Shape):
                inits.append(Assign(var, ZerosLike(i.arg)))
            else:
                inits.append(Assign(var, Zeros(i)))

        inits.append(EmptyNode())
        body =  tuple(inits) + body
        name = expr.name
        imports = ('zeros', 'zeros_like') + tuple(self._math_functions)
        imports = [Import('numpy', imports)]

        if self.backend['name'] == 'pyccel':
            a = [String(str(i)) for i in build_pyccel_types_decorator(arguments)]
            decorators = {'types': Function('types')(*a)}
        elif self.backend['name'] == 'numba':
            decorators = {'njit': Function('njit')(ValuedArgument(Symbol('fastmath'), self.backend['fastmath']))}
        elif self.backend['name'] == 'pythran':
            header = build_pythran_types_header(name, arguments)
        else:
            decorators = {}

        if self.backend['name'] == 'numba':
            func = FunctionDef(name, arguments, [], body, decorators=decorators)
            stmts = CodeBlock([*imports , func])
        else:
            func = FunctionDef(name, arguments, [], body, imports=imports, decorators=decorators)
            stmts = func

        self.functions[name] = func

        return stmts

    def _visit_EvalField(self, expr, **kwargs):
        g_coeffs   = expr.g_coeffs
        l_coeffs   = expr.l_coeffs
        tests      = expand_hdiv_hcurl(expr._tests)
        mats       = expr.atoms
        dim        = self._dim
        lhs_slices = [Slice(None,None)]*dim
        mats       = [self._visit(mat, **kwargs) for mat in mats]
        inits      = {mat:Assign(mat[lhs_slices], 0.) for mat in mats}
        body       = self._visit(expr.body, **kwargs)
        stmts      = OrderedDict()
        pads       = expr.pads

        for l_coeff,g_coeff in zip(l_coeffs, g_coeffs):
            basis        = g_coeff.test
            basis        = basis if basis in tests else basis.base
            degrees      = self._visit_LengthDofTest(LengthDofTest(basis))
            spans        = flatten(self._visit_Span(Span(basis))[basis])
            rhs_starts   = [spans[i]-degrees[i] + pads[i] for i in range(dim)]
            rhs_ends     = [spans[i]+pads[i]+1          for i in range(dim)]
            rhs_slices   = [Slice(s, e) for s,e in zip(rhs_starts, rhs_ends)]
            l_coeff      = self._visit(l_coeff, **kwargs)
            g_coeff      = self._visit(g_coeff, **kwargs)
            stmt         = self._visit_Assign(Assign(l_coeff[lhs_slices], g_coeff[rhs_slices]), **kwargs)
            stmts[stmt.lhs.base] = stmt
        return CodeBlock([*inits.values() , *stmts.values() , body])

    def _visit_EvalMapping(self, expr, **kwargs):
        if self._mapping.is_analytical:
            return EmptyNode()
        values  = expr.values
        coeffs  = expr.coeffs
        l_coeffs = expr.local_coeffs
        stmts   = []
        dim = self._dim
        tests = expr._tests
        test = coeffs[0].test
        test = test if test in tests else test.base
        lhs_slices = [Slice(None,None)]*dim
        for coeff, l_coeff in zip(coeffs, l_coeffs):
            spans   = flatten(self._visit_Span(Span(test))[test])
            degrees = self._visit_LengthDofTest(LengthDofTest(test))
            coeff   = self._visit(coeff)
            l_coeff = self._visit(l_coeff)
            rhs_starts = [spans[i] for i in range(dim)]
            rhs_ends   = [spans[i]+degrees[i]+1          for i in range(dim)]
            rhs_slices = [Slice(s, e) for s,e in zip(rhs_starts, rhs_ends)]
            stmt       = self._visit_Assign(Assign(l_coeff[lhs_slices], coeff[rhs_slices]), **kwargs)
            stmts.append(stmt)

        inits = []
        for val in values:
            val = self._visit(val, **kwargs)
            inits.append(Assign(val[lhs_slices], 0.))
        loop = self._visit(expr.loop, **kwargs)
        stmts.append(loop)
        return CodeBlock(inits+stmts)

    # ....................................................
    def _visit_Grid(self, expr, **kwargs):
        raise NotImplementedError('TODO')

    # ....................................................
    def _visit_Element(self, expr, **kwargs):
        raise NotImplementedError('TODO')

    # ....................................................
    def _visit_GlobalTensorQuadrature(self, expr, **kwargs):
        dim  = self.dim
        rank = expr.rank

        names = 'global_x1:%s'%(dim+1)
        points   = variables(names, dtype='real', rank=rank, cls=IndexedVariable)

        names = 'global_w1:%s'%(dim+1)
        weights  = variables(names, dtype='real', rank=rank, cls=IndexedVariable)

        # gather by axis
        
        targets = tuple(zip(points, weights))

        self.insert_variables(*points, *weights)

        return OrderedDict([(0,targets)])


    # ....................................................
    def _visit_LocalTensorQuadrature(self, expr, **kwargs):
        dim  = self.dim
        rank = expr.rank

        names = 'local_x1:%s'%(dim+1)
        points   = variables(names, dtype='real', rank=rank, cls=IndexedVariable)

        names = 'local_w1:%s'%(dim+1)
        weights  = variables(names, dtype='real', rank=rank, cls=IndexedVariable)

        # gather by axis
        
        targets = tuple(zip(points, weights))

        self.insert_variables(*points, *weights)

        return OrderedDict([(0,targets)])

    # ....................................................
    def _visit_TensorQuadrature(self, expr, **kwargs):
        dim = self.dim
        names   = 'x1:%s'%(dim+1)
        points  = variables(names, dtype='real', cls=Variable)

        names   = 'w1:%s'%(dim+1)
        weights = variables(names, dtype='real', cls=Variable)

        # gather by axis
        
        targets = tuple(zip(points, weights))

        self.insert_variables(*points, *weights)

        return OrderedDict([(0,targets)])


    # ....................................................
    def _visit_MatrixQuadrature(self, expr, **kwargs):
        rank   = self._visit(expr.rank)
        target = SymbolicExpr(expr.target)

        name = 'arr_{}'.format(target.name)
        var  =  IndexedVariable(name, dtype='real', rank=rank)
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

        self.insert_variables(*targets)

        arrays = OrderedDict()
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
        dim = self.dim
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
        self.insert_variables(*targets)

        arrays = OrderedDict()
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
        arrays = OrderedDict()
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
    def _visit_GlobalSpan(self, expr, **kwargs):
        dim    = self.dim
        rank   = expr.rank
        target = expr.target
        label = SymbolicExpr(target).name

        names  = 'global_span_{}_1:{}'.format(label, str(dim+1))
        targets = variables(names, dtype='int', rank=rank, cls=IndexedVariable)

        self.insert_variables(*targets)
        if not isinstance(targets[0], (tuple, list, Tuple)):
            targets = [targets]
        target = OrderedDict([(target ,tuple(zip(*targets)))])
        return target
    # ....................................................
    def _visit_Span(self, expr, **kwargs):
        dim = self.dim
        target = expr.target
        label  = SymbolicExpr(target).name
        names  = 'span_{}_1:{}'.format(label,str(dim+1))
        targets = variables(names, dtype='int')

        self.insert_variables(*targets)
        if not isinstance(targets[0], (tuple, list, Tuple)):
            targets = [targets]

        target = OrderedDict([(target ,tuple(zip(*targets)))])
        return target

    def _visit_Pads(self, expr, **kwargs):
        dim = self.dim
        tests  = expand(expr.tests)
        trials = expand(expr.trials)
        pads = Matrix.zeros(len(tests),len(trials))
        for i in range(pads.shape[0]):
            for j in range(pads.shape[1]):
                label1 = SymbolicExpr(tests[i]).name
                label2 = SymbolicExpr(trials[j]).name
                names  = 'pad_{}_{}_1:{}'.format(label2, label1, str(dim+1))
                targets = variables(names, dtype='int')
                pads[i,j] = Tuple(*targets)
                self.insert_variables(*targets)
        return pads
    # ....................................................
    def _visit_TensorBasis(self, expr, **kwargs):
        # TODO label
        dim = self.dim
        nderiv = self.nderiv
        target = expr.target

        ops = [dx1, dx2, dx3][:dim]
        atoms =  _split_test_function(target)
        args = OrderedDict()
        for atom in atoms:
            sub_args = [None]*dim
            for i in range(dim):
                d = ops[i]
                a = atoms[atom][i]
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
        name = 'coeff_{}'.format(target.name)
        var  = IndexedVariable(name, dtype='real', rank=self.dim)
        self.insert_variables(var)
        return var

    # ....................................................
    def _visit_MatrixLocalBasis(self, expr, **kwargs):
        rank   = self._visit(expr.rank)
        target = SymbolicExpr(expr.target)
        name = 'arr_coeffs_{}'.format(target.name)
        var  = IndexedVariable(name, dtype='real', rank=rank)
        self.insert_variables(var)
        return var
    # ....................................................

    def _visit_MatrixGlobalBasis(self, expr, **kwargs):
        rank   = self._visit(expr.rank)
        target = SymbolicExpr(expr.target)

        name = 'global_arr_coeffs_{}'.format(target.name)
        var  = IndexedVariable(name, dtype='real', rank=rank)
        self.insert_variables(var)
        return var
    # ....................................................
    def _visit_Reset(self, expr, **kwargs):
        var = expr.var
        lhs  = self._visit(var, **kwargs)
        if isinstance(var, (GlobalElementBasis,LocalElementBasis)):
            args = 0
        else:
            expr = var.expr
            rank = lhs[0,0].rank
            args  = [Slice(None, None)]*rank
            return tuple(Assign(a[args], 0.) for a,b in zip(lhs[:], expr[:]) if b)
        
        return Assign(lhs[args], 0.)

    # ....................................................
    def _visit_Reduce(self, expr, **kwargs):
        op   = expr.op
        lhs  = expr.lhs
        rhs  = expr.rhs
        loop = expr.loop

        stmts = list(loop.stmts) + [Reduction(op, rhs, lhs)]
        loop  = Loop(loop.iterable, loop.index, stmts=stmts, mask=loop.mask)
        return self._visit(loop, **kwargs)

    # ....................................................
    def _visit_Reduction(self, expr, **kwargs):
        op   = expr.op
        lhs  = expr.lhs
        expr = expr.expr

        if isinstance(lhs, GlobalElementBasis):
            lhs = self._visit(lhs, **kwargs)
            rhs = self._visit(expr, **kwargs)
            return (AugAssign(lhs[0], op, rhs[0]),)

        elif isinstance(lhs, LocalElementBasis):
            lhs = self._visit(lhs, **kwargs)
            rhs = self._visit(expr, **kwargs)
            return (AugAssign(lhs[0], op, rhs[0]),)

        elif isinstance(lhs, BlockStencilMatrixLocalBasis):
            lhs = self._visit_BlockStencilMatrixLocalBasis(lhs)
            expr = self._visit(expr, op=op, lhs=lhs)
            return expr
        elif isinstance(lhs, BlockStencilMatrixGlobalBasis):

            dim  = self.dim
            rank = lhs.rank
            pads = lhs.pads
            tests = expand(lhs._tests)
            tests_2 = expand_hdiv_hcurl(lhs._tests)
            lhs = self._visit_BlockStencilMatrixGlobalBasis(lhs)
            rhs = self._visit(expr)

            pads    = self._visit(pads)
            rhs_slices = [Slice(None, None)]*rank
            for k1 in range(lhs.shape[0]):
                test = tests[k1]
                test = test if test in tests_2 else test.base
                spans   = self._visit_Span(Span(test))
                degrees = self._visit_LengthDofTest(LengthDofTest(test))
                spans   = flatten(*spans.values())

                lhs_starts = [spans[i]+pads[i]-degrees[i] for i in range(dim)]
                lhs_ends   = [spans[i]+pads[i]+1          for i in range(dim)]
                if isinstance(self._target, Interface):
                    axis = self._target.axis
                    lhs_starts[axis] = pads[axis]
                    lhs_ends[axis]   = pads[axis] + degrees[axis] + 1

                for k2 in range(lhs.shape[1]):
                    if expr.expr[k1,k2]:
                        lhs_slices  = [Slice(s, e) for s,e in zip(lhs_starts, lhs_ends)]
                        lhs_slices += [Slice(None, None)]*dim
                        lhs[k1,k2] = [lhs[k1,k2][lhs_slices]]
                        rhs[k1,k2] = [rhs[k1,k2][rhs_slices]]

            return tuple( AugAssign(a, op, b) for a,b,e in zip(lhs[:], rhs[:], expr.expr[:]) if e)

        elif isinstance(lhs, BlockStencilVectorLocalBasis):
            lhs = self._visit_BlockStencilVectorLocalBasis(lhs)
            expr = self._visit(expr, op=op, lhs=lhs)
            return expr
        elif isinstance(lhs, BlockStencilVectorGlobalBasis):
            dim   = self.dim
            rank  = lhs.rank
            pads  = lhs.pads
            tests = expand(lhs._tests)
            tests_2 = expand_hdiv_hcurl(lhs._tests)
            lhs = self._visit_BlockStencilVectorGlobalBasis(lhs)
            rhs = self._visit(expr)
            pads    = self._visit(pads)
            rhs_slices = [Slice(None, None)]*rank

            for k in range(lhs.shape[0]):
                if expr.expr[k,0]:
                    test = tests[k]
                    test = test if test in tests_2 else test.base
                    spans   = self._visit_Span(Span(test))
                    spans   = flatten(*spans.values())
                    degrees = self._visit_LengthDofTest(LengthDofTest(test))
                    lhs_starts = [spans[i]+pads[i]-degrees[i] for i in range(dim)]
                    lhs_ends   = [spans[i]+pads[i]+1          for i in range(dim)]
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
        expr = expr.expr
        if lhs is None:
            if not isinstance(expr, (Add, Mul)):
                atom = BasisAtom(expr)
                lhs  = self._visit_BasisAtom(atom, **kwargs)
            else:
                lhs = random_string( 6 )
                lhs = Symbol('tmp_{}'.format(lhs))

        expr = LogicalBasisValue(expr)
        rhs = self._visit_LogicalBasisValue(expr, **kwargs)

        if op is None:
            stmt = Assign(lhs, rhs)
        else:
            stmt = AugAssign(lhs, op, rhs)

        return self._visit(stmt, **kwargs)

    # ....................................................
    def _visit_ComputeKernelExpr(self, expr, op=None, lhs=None, **kwargs):
        if lhs is None:
            if not isinstance(expr, (Add, Mul)):
                lhs = self._visit_BasisAtom(BasisAtom(expr), **kwargs)
            else:
                lhs = random_string( 6 )
                lhs = Symbol('tmp_{}'.format(lhs))

        exprs   = expr.expr
        mapping = self.mapping

        weight  = SymbolicWeightedVolume(mapping)
        weight  = SymbolicExpr(weight)

        rhs = [weight*self._visit(expr, **kwargs) for expr in exprs[:]]
        lhs = lhs[:]

        temps = []
        temps, rhs = cse_main.cse(rhs, symbols=cse_main.numbered_symbols(prefix='temp'))

        normal_vec_stmts = []
        normal_vectors = expr.expr.atoms(NormalVector)
        target         = self._target
        dim            = self._dim

        if isinstance(target, Interface):
            mapping = mapping.minus
            target  = target.minus
            axis    = target.axis
            ext     = 1
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
        stmts = tuple(normal_vec_stmts) + temps + stmts
        math_functions = math_atoms_as_str(list(exprs)+normal_vec_stmts, 'numpy')
        math_functions = tuple(m for m in math_functions if m not in self._math_functions)
        self._math_functions = math_functions + self._math_functions
        return stmts

    # ....................................................
    def _visit_BasisAtom(self, expr, **kwargs):
        symbol = SymbolicExpr(expr.expr)
        self.variables[str(symbol.name)] = symbol
        return symbol

    # ....................................................
    def _visit_AtomicNode(self, expr, **kwargs):
        if isinstance(expr.expr, WeightedVolumeQuadrature):
            expr = SymbolicWeightedVolume(self.mapping)
            return self._visit(expr, **kwargs )

        else:
            return SymbolicExpr(expr.expr)

    # ....................................................
    def _visit_LogicalBasisValue(self, expr, **kwargs):
        # ...
        dim = self.dim
        coords = ['x1', 'x2', 'x3'][:dim]

        expr   = expr.expr
        atom   = BasisAtom(expr).atom
        atoms  = _split_test_function(atom)
        ops = [dx1, dx2, dx3][:dim]
        d_atoms = dict(zip(coords, atoms[atom]))
        d_ops   = dict(zip(coords, ops))
        d_indices = get_index_logical_derivatives(expr)
        args = []
        for k,u in d_atoms.items():
            d = d_ops[k]
            n = d_indices[k]
            for _ in range(n):
                u = d(u)
            args.append(u)
        # ...

        expr = Mul(*args)
        expr =  SymbolicExpr(expr)
        return expr

    # ....................................................
    def _visit_LogicalValueNode(self, expr, **kwargs):

        expr = expr.expr
        target = self.target

        if isinstance(expr, WeightedVolumeQuadrature):
            #TODO improve l_quad should not be used like this
            l_quad = TensorQuadrature()
            l_quad = self._visit_TensorQuadrature(l_quad, **kwargs)
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
        mapping = self.mapping
        expr = LogicalExpr(mapping, expr.expr)

        return SymbolicExpr(expr)

    # ....................................................
    def _visit_ElementOf(self, expr, **kwargs):
        dim    = self.dim
        target = expr.target
        #improve we shouldn't use index_dof_test
        if isinstance(target, BlockStencilMatrixLocalBasis):
            rows = self._visit(index_dof_test)
            cols = self._visit(index_dof_trial)
            pads = self._visit_Pads(target.pads)

            targets = self._visit_BlockStencilMatrixLocalBasis(target)
            for i in range(targets.shape[0]):
                for j in range(targets.shape[1]):
                    padding  = pads[i,j]
                    indices = tuple(rows) + tuple(cols[k]+padding[k]-rows[k] for k in range(dim))
                    targets[i,j] = targets[i,j][indices]
            return targets

        elif isinstance(target, BlockStencilVectorLocalBasis):
            targets = self._visit_BlockStencilVectorLocalBasis(target, **kwargs)

            rows = self._visit(index_dof_test)
            indices = list(rows)
            for i in range(targets.shape[0]):
                for j in range(targets.shape[1]):
                    targets[i,j] = targets[i,j][indices]
            return targets
        elif isinstance(target, LocalElementBasis):
            target = self._visit(target, **kwargs)
            return (target[0],)

        else:
            raise NotImplementedError('TODO')

    # .............................................................................
    def _visit_BlockStencilMatrixLocalBasis(self, expr, **kwargs):
        pads   = self._visit_Pads(expr.pads)
        tests  = expr._tests
        trials = expr._trials
        tag    = expr.tag
        tests   = expand(tests)
        trials  = expand(trials)
        targets = Matrix.zeros(len(tests), len(trials))
        for i,v in enumerate(tests):
            for j,u in enumerate(trials):
                mat = StencilMatrixLocalBasis(u, v, pads[i,j], tag)
                mat = self._visit_StencilMatrixLocalBasis(mat, **kwargs)
                targets[i,j] = mat
        return targets

    def _visit_BlockStencilMatrixGlobalBasis(self, expr, **kwargs):
        pads   = expr.pads
        tests  = expr._tests
        trials = expr._trials
        tag    = expr.tag
        tests   = expand(tests)
        trials  = expand(trials)
        targets = Matrix.zeros(len(tests), len(trials))
        for i,v in enumerate(tests):
            for j,u in enumerate(trials):
                mat = StencilMatrixGlobalBasis(u, v, pads, tag)
                mat = self._visit_StencilMatrixGlobalBasis(mat, **kwargs)
                targets[i,j] = mat
        return targets

    def _visit_BlockStencilVectorLocalBasis(self, expr, **kwargs):
        pads   = expr.pads
        tests  = expr._tests
        tag    = expr.tag
        tests   = expand(tests)
        targets = Matrix.zeros(len(tests), 1)
        for i,v in enumerate(tests):
            mat = StencilVectorLocalBasis(v, pads, tag)
            mat = self._visit_StencilVectorLocalBasis(mat, **kwargs)
            targets[i,0] = mat
        return targets

    def _visit_BlockStencilVectorGlobalBasis(self, expr, **kwargs):
        pads   = expr.pads
        tests  = expr._tests
        tag    = expr.tag
        tests   = expand(tests)
        targets = Matrix.zeros(len(tests), 1)
        for i,v in enumerate(tests):
            mat = StencilVectorGlobalBasis(v, pads, tag)
            mat = self._visit_StencilVectorGlobalBasis(mat, **kwargs)
            targets[i,0] = mat
        return targets


    # .............................................................................
    def _visit_StencilMatrixLocalBasis(self, expr, **kwargs):
        rank = expr.rank
        tag  = expr.tag
        name = '_'.join(str(SymbolicExpr(e)) for e in expr.name)

        name = 'l_mat_{}_{}'.format(name, tag)
        var  = IndexedVariable(name, dtype='real', rank=rank)
        self.insert_variables(var)
        return var

    # ....................................................
    def _visit_StencilVectorLocalBasis(self, expr, **kwargs):
        rank = expr.rank
        tag  = expr.tag
        name = str(SymbolicExpr(expr.name))
        name = 'l_vec_{}_{}'.format(name, tag)
        var  = IndexedVariable(name, dtype='real', rank=rank) 
        self.insert_variables(var)
        return var

    # ....................................................
    def _visit_StencilMatrixGlobalBasis(self, expr, **kwargs):
        rank = expr.rank
        tag  = expr.tag
        name = '_'.join(str(SymbolicExpr(e)) for e in expr.name)
        name = 'g_mat_{}_{}'.format(name, tag)
        var  = IndexedVariable(name, dtype='real', rank=rank)
        self.insert_variables(var)
        return var

    # ....................................................
    def _visit_StencilVectorGlobalBasis(self, expr, **kwargs):
        rank = expr.rank
        tag  = expr.tag
        name = str(SymbolicExpr(expr.name))
        name = 'g_vec_{}_{}'.format(name, tag)        
        var  = IndexedVariable(name, dtype='real', rank=rank)
        self.insert_variables(var)
        return var

    def _visit_GlobalElementBasis(self, expr, **kwargs):
        tag  = expr.tag
        name = 'g_el_{}'.format(tag)
        var  = IndexedVariable(name, dtype='real', rank=1) 
        self.insert_variables(var)
        return var

    def _visit_LocalElementBasis(self, expr, **kwargs):
        tag  = expr.tag
        name = 'l_el_{}'.format(tag)
        var  = IndexedVariable(name, dtype='real', rank=1) 
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

    # ....................................................
    def _visit_Expr(self, expr, **kwargs):
        return SymbolicExpr(expr)

    # ....................................................
    def _visit_IndexElement(self, expr, **kwargs):
        dim    = self.dim
        target =  variables('i_element_1:%d'%(dim+1), dtype='int')
        self.insert_variables(*target)
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
    def _visit_IndexDerivative(self, expr, **kwargs):
        raise NotImplementedError('TODO')

    # ....................................................
    def _visit_LengthElement(self, expr, **kwargs):
        dim = self.dim
        names = 'n_element_1:%d'%(dim+1)
        target = variables(names, dtype='int', cls=Variable)
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

        patterns = expr.target.pattern()
        patterns = self._visit_Pattern(patterns)
        args = OrderedDict()
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
        dim       = self.dim
        iterator  = self._visit(expr.iterator)
        generator = self._visit(expr.generator)

        inits = [()]*dim
        for (i, l_xs),(j, g_xs) in zip(iterator.items(), generator.items()):
            if isinstance(expr.generator.target, GlobalTensorQuadratureBasis):
                positions = [expr.generator.target.positions[index_deriv]]
                g_xs = [SplitArray(xs[0], positions, [self.nderiv+1]) for xs in g_xs]
                g_xs = [tuple(self._visit(xs, **kwargs)) for xs in g_xs]

            for i in  range(dim):
                ls = []
                for l_x,g_x in zip(l_xs[i], g_xs[i]):
                    if isinstance(expr.generator.target, GlobalTensorQuadratureBasis):
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
    # ....................................................
    def _visit_Loop(self, expr, **kwargs):
        # we first create iteration statements
        # these iterations are splitted between what is tensor or not

        # ... treate tensor iterations

        t_iterator   = [i for i in expr.iterator  if isinstance(i, TensorIterator)]
        t_generator  = [i for i in expr.generator if isinstance(i, TensorGenerator)]
        t_iterations = [TensorIteration(i,j)
                        for i,j in zip(t_iterator, t_generator)]

        indices, lengths = list(self._visit(expr.index)), list(self._visit(expr.index.length))
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

        if mask:
            axis   = mask.axis
            index  = indices.pop(axis)
            length = lengths.pop(axis)
            init   = inits.pop(axis)
            mask_init = [Assign(index, 0), *init]
        for index, length, init in zip(indices[::-1], lengths[::-1], inits[::-1]):

            body = list(init) + body
            body = [For(index, Range(length), body)]
        # ...
        # remove the list and return the For Node only

        if mask:
            body = CodeBlock([*mask_init, *body])
        elif len(body) > 1:
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

    # ....................................................
    def _visit_IndexedElement(self, expr, **kwargs):
        return expr

    # ....................................................
    # TODO to be removed. usefull for testing
    def _visit_Pass(self, expr, **kwargs):
        return expr

    def _visit_EmptyNode(self ,expr, **kwargs):
        return expr

    def _visit_NoneType(self, expr, **kwargs):
        return expr


