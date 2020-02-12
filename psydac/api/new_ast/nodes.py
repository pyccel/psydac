from collections import OrderedDict
from itertools import product

from sympy import Basic
from sympy.core.singleton import Singleton
from sympy.core.compatibility import with_metaclass
from sympy.core.containers import Tuple
from sympy import AtomicExpr
from sympy import Symbol, Mul

from sympde.topology import ScalarTestFunction, VectorTestFunction
from sympde.topology import (dx1, dx2, dx3)
from sympde.topology import Mapping
from sympde.topology import SymbolicDeterminant
from sympde.topology import SymbolicInverseDeterminant
from sympde.topology import SymbolicWeightedVolume
from sympde.topology import IdentityMapping
from sympde.topology import element_of, VectorFunctionSpace, ScalarFunctionSpace
from sympde.topology import H1SpaceType, HcurlSpaceType, HdivSpaceType, L2SpaceType, UndefinedSpaceType

from .utilities import physical2logical
from pyccel.ast import AugAssign, Assign
from pyccel.ast import EmptyLine
from pyccel.ast.core      import _atomic

from sympde.topology.derivatives import get_index_logical_derivatives, get_atom_logical_derivatives

def expand(args):
    new_args = []
    for i in args:
        if isinstance(i, ScalarTestFunction):
            new_args += [i]
        elif isinstance(i, VectorTestFunction):
            new_args += [i[k] for k in  range(i.space.ldim)]
        else:
            raise NotImplementedError("TODO")
    return new_args

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
# TODO move it
import string
import random
def random_string( n ):
    chars    = string.ascii_lowercase + string.digits
    selector = random.SystemRandom()
    return ''.join( selector.choice( chars ) for _ in range( n ) )


#==============================================================================
class ArityType(with_metaclass(Singleton, Basic)):
    """Base class representing a form type: bilinear/linear/functional"""
    pass

class BilinearArity(ArityType):
    pass

class LinearArity(ArityType):
    pass

class FunctionalArity(ArityType):
    pass

#==============================================================================
class LengthNode(with_metaclass(Singleton, Basic)):
    """Base class representing one length of an iterator"""
    pass

class LengthElement(LengthNode):
    pass

class LengthQuadrature(LengthNode):
    pass

class LengthDof(LengthNode):
    pass

class LengthDofTrial(LengthNode):
    pass

class LengthDofTest(LengthNode):
    pass
#==============================================================================
class IndexNode(with_metaclass(Singleton, Basic)):
    """Base class representing one index of an iterator"""
    def __new__(cls, index_length):
        assert isinstance(index_length, LengthNode)
        return Basic.__new__(cls, index_length)

    @property
    def length(self):
        return self._args[0]

class IndexElement(IndexNode):
    pass

class IndexQuadrature(IndexNode):
    pass

class IndexDof(IndexNode):
    pass

class IndexDofTrial(IndexNode):
    pass

class IndexDofTest(IndexNode):
    pass

class IndexDerivative(IndexNode):
    def __new__(cls):
        return Basic.__new__(cls)

index_element   = IndexElement(LengthElement())
index_quad      = IndexQuadrature(LengthQuadrature())
index_dof       = IndexDof(LengthDof())
index_dof_trial = IndexDofTrial(LengthDofTrial())
index_dof_test  = IndexDofTest(LengthDofTest())
index_deriv     = IndexDerivative()



#==============================================================================
class RankNode(with_metaclass(Singleton, Basic)):
    """Base class representing a rank of an iterator"""
    pass

class RankDimension(RankNode):
    pass

rank_dim = RankDimension()

#==============================================================================
class BaseNode(Basic):
    """
    """
    pass

#==============================================================================
class Element(BaseNode):
    """
    """
    pass

#==============================================================================
class Pattern(Tuple):
    """
    """
    pass

class EvalField(BaseNode):
    def __new__(cls, atoms, index, basis, coeffs, tests, nderiv):

        stmts  = []
        inits  = []
        for v in tests:
            stmts += construct_logical_expressions(v, nderiv)
        atoms   = [physical2logical(i) for i in atoms]

        for coeff in coeffs:
            for a in atoms:
                node    = AtomicNode(a)
                val     = ProductGenerator(MatrixQuadrature(a), index)
                rhs     = Mul(coeff,node)
                stmts  += [AugAssign(val, '+', rhs)]

                inits += [Assign(node,val)]
                inits += [ComputePhysicalBasis(a)]

        inits = Tuple(*inits)
        body  = Loop( basis, index, stmts)

        return Basic.__new__(cls, atoms, inits, body)

    @property
    def atoms(self):
        return self._args[0]

    @property
    def inits(self):
        return self._args[1]

    @property
    def body(self):
        return self._args[2]

class EvalMapping(BaseNode):
    """."""
    def __new__(cls, quads, indices_basis, q_basis, l_basis, mapping, components, space, nderiv):
        atoms  = components.arguments
        basis  = q_basis
        target = basis.target
        if isinstance(target, VectorTestFunction):
            target = target[0]
            #TODO improve how should we do the eval mapping when it's a VectorTestFunction
        new_atoms  = []
        nodes      = []
        l_coeffs   = set()
        for a in atoms:
            atom   = get_atom_logical_derivatives(a)
            node   = a.subs(atom, target)
            new_atoms.append(atom)
            nodes.append(node)
#            if isinstance(node, Matrix):
#                nodes += [*node[:]]
#            else:
#                nodes.append(node)

        stmts = [ComputeLogicalBasis(v,) for v in set(nodes)]
        for i in range(len(atoms)):
            l_coeffs.add(MatrixLocalBasis(new_atoms[i]))
            node    = AtomicNode(nodes[i])
            val     = ProductGenerator(MatrixQuadrature(atoms[i]), quads)
            rhs     = Mul(CoefficientBasis(new_atoms[i]),node)
            stmts  += [AugAssign(val, '+', rhs)]

        loop   = Loop((q_basis, *l_coeffs), quads, stmts)
        loop   = Loop(l_basis, indices_basis, [loop])

        return Basic.__new__(cls, loop, l_coeffs)

    @property
    def loop(self):
        return self._args[0]
    @property
    def coeffs(self):
        return self._args[1]
#==============================================================================
class IteratorBase(BaseNode):
    """
    """
    def __new__(cls, target, dummies=None):
        if not dummies is None:
            if not isinstance(dummies, (list, tuple, Tuple)):
                dummies = [dummies]
            dummies = Tuple(*dummies)

        return Basic.__new__(cls, target, dummies)

    @property
    def target(self):
        return self._args[0]

    @property
    def dummies(self):
        return self._args[1]

#==============================================================================
class TensorIterator(IteratorBase):
    pass

#==============================================================================
class ProductIterator(IteratorBase):
    pass

#==============================================================================
# TODO dummies should not be None
class GeneratorBase(BaseNode):
    """
    """
    def __new__(cls, target, dummies):
        if not isinstance(dummies, (list, tuple, Tuple)):
            dummies = [dummies]
        dummies = Tuple(*dummies)

        if not isinstance(target, (ArrayNode, MatrixNode)):
            raise TypeError('expecting an ArrayNode')

        return Basic.__new__(cls, target, dummies)

    @property
    def target(self):
        return self._args[0]

    @property
    def dummies(self):
        return self._args[1]

#==============================================================================
class TensorGenerator(GeneratorBase):
    pass

#==============================================================================
class ProductGenerator(GeneratorBase):
    pass

#==============================================================================
class Grid(BaseNode):
    """
    """
    pass

#==============================================================================
class ScalarNode(BaseNode, AtomicExpr):
    """
    """
    pass

#==============================================================================
class ArrayNode(BaseNode, AtomicExpr):
    """
    """
    _rank = None
    _positions = None
    _free_indices = None

    @property
    def rank(self):
        return self._rank

    @property
    def positions(self):
        return self._positions

    @property
    def free_indices(self):
        if self._free_indices is None:
            return list(self.positions.keys())

        else:
            return self._free_indices

    def pattern(self):
        positions = {}
        for a in self.free_indices:
            positions[a] = self.positions[a]

        args = [None]*self.rank
        for k,v in positions.items():
            args[v] = k

        return Pattern(*args)

#==============================================================================
class MatrixNode(ArrayNode):
    pass

class BlockMatrixNode(MatrixNode):
    pass
#==============================================================================
class GlobalTensorQuadrature(ArrayNode):
    """
    """
    _rank = 2
    _positions = {index_element: 0, index_quad: 1}
    _free_indices = [index_element]

#==============================================================================
class LocalTensorQuadrature(ArrayNode):
    # TODO add set_positions
    """
    """
    _rank = 1
    _positions = {index_quad: 0}

#==============================================================================
class TensorQuadrature(ScalarNode):
    """
    """
    pass

#==============================================================================
class MatrixQuadrature(MatrixNode):
    """
    """
    _rank = rank_dim

    def __new__(cls, target):
        # TODO check target
        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class WeightedVolumeQuadrature(ScalarNode):
    """
    """
    pass

#==============================================================================
class GlobalTensorQuadratureBasis(ArrayNode):
    """
    """
    _rank = 4
    _positions = {index_quad: 3, index_deriv: 2, index_dof: 1, index_element: 0}
    _free_indices = [index_element]

    def __new__(cls, target):
        if not isinstance(target, (ScalarTestFunction, VectorTestFunction)):
            raise TypeError('Expecting a scalar/vector test function')
        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

    @property
    def unique_scalar_space(self):
        unique_scalar_space = True
        space = self.target.space
        if isinstance(space, VectorFunctionSpace):
            unique_scalar_space = isinstance(space.kind, UndefinedSpaceType)
        return unique_scalar_space

    @property
    def is_scalar(self):
        return isinstance(self.target, ScalarTestFunction)

#==============================================================================
class LocalTensorQuadratureBasis(ArrayNode):
    """
    """
    _rank = 3
    _positions = {index_quad: 2, index_deriv: 1, index_dof: 0}
    _free_indices = [index_dof]

    def __new__(cls, target):
        if not isinstance(target, (ScalarTestFunction, VectorTestFunction)):
            raise TypeError('Expecting a scalar/vector test function')
        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

    @property
    def unique_scalar_space(self):
        unique_scalar_space = True
        space = self.target.space
        if isinstance(space, VectorFunctionSpace):
            unique_scalar_space = isinstance(space.kind, UndefinedSpaceType)
        return unique_scalar_space

    @property
    def is_scalar(self):
        return isinstance(self.target, ScalarTestFunction)
#==============================================================================
class TensorQuadratureBasis(ArrayNode):
    """
    """
    _rank = 2
    _positions = {index_quad: 1, index_deriv: 0}
    _free_indices = [index_quad]

    def __new__(cls, target):
        if not isinstance(target, (ScalarTestFunction, VectorTestFunction)):
            raise TypeError('Expecting a scalar/vector test function')

        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

    @property
    def unique_scalar_space(self):
        unique_scalar_space = True
        space = self.target.space
        if isinstance(space, VectorFunctionSpace):
            unique_scalar_space = isinstance(space.kind, UndefinedSpaceType)
        return unique_scalar_space

    @property
    def is_scalar(self):
        return isinstance(self.target, ScalarTestFunction)
#==============================================================================
class CoefficientBasis(ScalarNode):
    """
    """
    def __new__(cls, target):
        ls = target.atoms(ScalarTestFunction, VectorTestFunction, Mapping)
        if not len(ls) == 1:
            raise TypeError('Expecting a scalar/vector test function or a Mapping')
        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class TensorBasis(CoefficientBasis):
    pass

#==============================================================================
class GlobalTensorQuadratureTestBasis(GlobalTensorQuadratureBasis):
    _positions = {index_quad: 3, index_deriv: 2, index_dof_test: 1, index_element: 0}

#==============================================================================
class LocalTensorQuadratureTestBasis(LocalTensorQuadratureBasis):
    _positions = {index_quad: 2, index_deriv: 1, index_dof_test: 0}
    _free_indices = [index_dof_test]

#==============================================================================
class TensorQuadratureTestBasis(TensorQuadratureBasis):
    pass

#==============================================================================
class TensorTestBasis(TensorBasis):
    pass

#==============================================================================
class GlobalTensorQuadratureTrialBasis(GlobalTensorQuadratureBasis):
    _positions = {index_quad: 3, index_deriv: 2, index_dof_trial: 1, index_element: 0}

#==============================================================================
class LocalTensorQuadratureTrialBasis(LocalTensorQuadratureBasis):
    _positions = {index_quad: 2, index_deriv: 1, index_dof_trial: 0}
    _free_indices = [index_dof_trial]

#==============================================================================
class TensorQuadratureTrialBasis(TensorQuadratureBasis):
    pass

#==============================================================================
class TensorTrialBasis(TensorBasis):
    pass

#==============================================================================
class MatrixLocalBasis(MatrixNode):
    """
    used to describe local dof over an element
    """
    _rank = rank_dim

    def __new__(cls, target):
        # TODO check target
        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class StencilMatrixLocalBasis(MatrixNode):
    """
    used to describe local dof over an element as a stencil matrix
    """
    def __new__(cls, u, v, pads, tag=None):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = 2*len(pads)
        tag  = tag or random_string( 6 )
        name = (u, v)
        return Basic.__new__(cls, pads, rank, name, tag)

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def name(self):
        return self._args[2]

    @property
    def tag(self):
        return self._args[3]
#==============================================================================
class StencilMatrixGlobalBasis(MatrixNode):
    """
    used to describe local dof over an element as a stencil matrix
    """
    def __new__(cls, u, v, pads, tag=None):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = 2*len(pads)
        tag  = tag or random_string( 6 )
        name = (u, v)
        return Basic.__new__(cls, pads, rank, name, tag)

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def name(self):
        return self._args[2]

    @property
    def tag(self):
        return self._args[3]

#==============================================================================
class StencilVectorLocalBasis(MatrixNode):
    """
    used to describe local dof over an element as a stencil vector
    """
    def __new__(cls, v, pads, tag=None):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = len(pads)
        tag  = tag or random_string( 6 )
        name = v
        return Basic.__new__(cls, pads, rank, name, tag)

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def name(self):
        return self._args[2]

    @property
    def tag(self):
        return self._args[3]



#==============================================================================
class StencilVectorGlobalBasis(MatrixNode):
    """
    used to describe local dof over an element as a stencil vector
    """
    def __new__(cls, v, pads, tag=None):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = len(pads)
        tag  = tag or random_string( 6 )
        name = v
        return Basic.__new__(cls, pads, rank, name, tag)

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def name(self):
        return self._args[2]

    @property
    def tag(self):
        return self._args[3]



class LocalElementBasis(MatrixNode):
    tag  = random_string( 6 )

class GlobalElementBasis(MatrixNode):
    tag  = random_string( 6 )

class BlockStencilMatrixLocalBasis(BlockMatrixNode):
    """
    used to describe local dof over an element as a block stencil matrix
    """
    def __new__(cls, trials, tests, pads, expr):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = 2*len(pads)
        tag  = random_string( 6 )
        obj  = Basic.__new__(cls, pads, rank, tag, expr)
        obj._trials = trials
        obj._tests  = tests
        return obj

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def tag(self):
        return self._args[2]

    @property
    def expr(self):
        return self._args[3]

    @property
    def unique_scalar_space(self):
        types = (H1SpaceType, L2SpaceType, UndefinedSpaceType)
        spaces = self.trials.space
        cond = False
        for cls in types:
            cond = cond or all(isinstance(space.kind, cls) for space in spaces)
        return cond

#==============================================================================
class BlockStencilMatrixGlobalBasis(BlockMatrixNode):
    """
    used to describe local dof over an element as a block stencil matrix
    """
    def __new__(cls, trials, tests, pads, expr):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = 2*len(pads)
        tag  = random_string( 6 )
        obj  = Basic.__new__(cls, pads, rank, tag, expr)
        obj._trials = trials
        obj._tests  = tests
        return obj

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def tag(self):
        return self._args[2]

    @property
    def expr(self):
        return self._args[3]

    @property
    def unique_scalar_space(self):
        types = (H1SpaceType, L2SpaceType, UndefinedSpaceType)
        spaces = self.trials.space
        cond = False
        for cls in types:
            cond = cond or all(isinstance(space.kind, cls) for space in spaces)
        return cond


class BlockStencilVectorLocalBasis(BlockMatrixNode):
    """
    used to describe local dof over an element as a block stencil matrix
    """
    def __new__(cls,tests, pads, expr):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = 2*len(pads)
        tag  = random_string( 6 )
        obj  = Basic.__new__(cls, pads, rank, tag, expr)
        obj._tests  = tests
        return obj

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def tag(self):
        return self._args[2]

    @property
    def expr(self):
        return self._args[3]

    @property
    def unique_scalar_space(self):
        types = (H1SpaceType, L2SpaceType, UndefinedSpaceType)
        spaces = self._tests.space
        cond = False
        for cls in types:
            cond = cond or all(isinstance(space.kind, cls) for space in spaces)
        return cond

#==============================================================================
class BlockStencilVectorGlobalBasis(BlockMatrixNode):
    """
    used to describe local dof over an element as a block stencil matrix
    """
    def __new__(cls, tests, pads, expr):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = 2*len(pads)
        tag  = random_string( 6 )
        obj  = Basic.__new__(cls, pads, rank, tag, expr)
        obj._tests  = tests
        return obj

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def tag(self):
        return self._args[2]

    @property
    def expr(self):
        return self._args[3]

    @property
    def unique_scalar_space(self):
        types = (H1SpaceType, L2SpaceType, UndefinedSpaceType)
        spaces = self._tests.space
        cond = False
        for cls in types:
            cond = cond or all(isinstance(space.kind, cls) for space in spaces)
        return cond
#==============================================================================
class GlobalSpan(ArrayNode):
    """
    """
    _rank = 1
    _positions = {index_element: 0}

    def __new__(cls, target):
        if not isinstance(target, (ScalarTestFunction, VectorTestFunction)):
            raise TypeError('Expecting a scalar/vector test function')

        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class Span(ScalarNode):
    """
    """
    def __new__(cls, target=None):
        if not( target is None ):
            if not isinstance(target, (ScalarTestFunction, VectorTestFunction)):
                raise TypeError('Expecting a scalar/vector test function')

        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class Evaluation(BaseNode):
    """
    """
    pass

#==============================================================================
class FieldEvaluation(Evaluation):
    """
    """
    pass

#==============================================================================
class MappingEvaluation(Evaluation):
    """
    """
    pass

#==============================================================================
class ComputeNode(Basic):
    """
    """
    def __new__(cls, expr):
        return Basic.__new__(cls, expr)

    @property
    def expr(self):
        return self._args[0]

#==============================================================================
class ComputePhysical(ComputeNode):
    """
    """
    pass

#==============================================================================
class ComputePhysicalBasis(ComputePhysical):
    """
    """
    pass

#==============================================================================
class ComputeKernelExpr(ComputeNode):
    """
    """
    def __new__(cls, expr):
        return Basic.__new__(cls, expr)

    @property
    def expr(self):
        return self._args[0]

#==============================================================================
class ComputeLogical(ComputeNode):
    """
    """
    pass

#==============================================================================
class ComputeLogicalBasis(ComputeLogical):
    """
    """
    pass

#==============================================================================
class Reduction(Basic):
    """
    """
    def __new__(cls, op, expr, lhs=None):
        # TODO add verification on op = '-', '+', '*', '/'
        return Basic.__new__(cls, op, expr, lhs)

    @property
    def op(self):
        return self._args[0]

    @property
    def expr(self):
        return self._args[1]

    @property
    def lhs(self):
        return self._args[2]

#==============================================================================
class Reduce(Basic):
    """
    """
    def __new__(cls, op, rhs, lhs, loop):
        # TODO add verification on op = '-', '+', '*', '/'
        if not isinstance(loop, Loop):
            raise TypeError('Expecting a Loop')

        return Basic.__new__(cls, op, rhs, lhs, loop)

    @property
    def op(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]

    @property
    def lhs(self):
        return self._args[2]

    @property
    def loop(self):
        return self._args[3]

#==============================================================================
class Reset(Basic):
    """
    """
    def __new__(cls, var, expr=None):
        return Basic.__new__(cls, var, expr)

    @property
    def var(self):
        return self._args[0]

    @property
    def expr(self):
        return self._args[1]

#==============================================================================
class ElementOf(Basic):
    """
    """
    def __new__(cls, target):
        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class ExprNode(Basic):
    """
    """
    pass

#==============================================================================
class AtomicNode(ExprNode, AtomicExpr):
    """
    """

    @property
    def expr(self):
        return self._args[0]

#==============================================================================
class ValueNode(ExprNode):
    """
    """
    def __new__(cls, expr):
        return Basic.__new__(cls, expr)

    @property
    def expr(self):
        return self._args[0]

#==============================================================================
class PhysicalValueNode(ValueNode):
    """
    """
    pass

#==============================================================================
class LogicalValueNode(ValueNode):
    """
    """
    pass

#==============================================================================
class PhysicalBasisValue(PhysicalValueNode):
    """
    """
    pass

#==============================================================================
class LogicalBasisValue(LogicalValueNode):
    """
    """
    pass

#==============================================================================
class PhysicalGeometryValue(PhysicalValueNode):
    """
    """
    pass

#==============================================================================
class LogicalGeometryValue(LogicalValueNode):
    """
    """
    pass

#==============================================================================
class BasisAtom(AtomicNode):
    """
    """
    def __new__(cls, expr):
        types = (IndexedTestTrial, VectorTestFunction,
                 ScalarTestFunction, IndexedVectorField,
                 ScalarField, VectorField)

        ls = _atomic(expr, cls=types)
        if not(len(ls) == 1):
            raise ValueError('Expecting an expression with one test function')

        u = ls[0]

        obj = Basic.__new__(cls, expr)
        obj._atom = u
        return obj

    @property
    def expr(self):
        return self._args[0]

    @property
    def atom(self):
        return self._atom

#==============================================================================
class GeometryAtom(AtomicNode):
    """
    """
    def __new__(cls, expr):
        ls = list(expr.atoms(Mapping))
        if not(len(ls) == 1):
            raise ValueError('Expecting an expression with one mapping')

        # TODO
        u = ls[0]

        obj = Basic.__new__(cls, expr)
        obj._atom = u
        return obj

    @property
    def expr(self):
        return self._args[0]

    @property
    def atom(self):
        return self._atom

#==============================================================================
class GeometryExpr(Basic):
    """
    """
    def __new__(cls, expr):
        # TODO assert on expr
        atom = GeometryAtom(expr)
        expr = MatrixQuadrature(expr)

        return Basic.__new__(cls, atom, expr)

    @property
    def atom(self):
        return self._args[0]

    @property
    def expr(self):
        return self._args[1]

#==============================================================================
class Loop(BaseNode):
    """
    class to describe a loop of an iterator over a generator.
    """

    def __new__(cls, iterable, index, stmts=None):
        # ...
        if not( isinstance(iterable, (list, tuple, Tuple)) ):
            iterable = [iterable]

        iterable = Tuple(*iterable)
        # ...

        # ... replace GeometryExpressions by a list of expressions
        others = [i for i in iterable if not isinstance(i, GeometryExpressions)]
        geos   = [i.expressions for i in iterable if isinstance(i, GeometryExpressions)]

        if len(geos) == 1:
            geos = list(geos[0])

        elif len(geos) > 1:
            raise NotImplementedError('TODO')

        iterable = others + geos
        iterable = Tuple(*iterable)
        # ...

        # ...
        if not( isinstance(index, IndexNode) ):
            raise TypeError('Expecting an index node')
        # ...

        # ... TODO - add assert w.r.t index type
        #          - this should be splitted/moved somewhere
        iterator = []
        generator = []
        for a in iterable:
            i,g = construct_itergener(a, index)
            iterator.append(i)
            generator.append(g)
        # ...
        # ...
        iterator = Tuple(*iterator)
        generator = Tuple(*generator)
        # ...

        # ...
        if stmts is None:
            stmts = []

        elif not isinstance(stmts, (tuple, list, Tuple)):
            stmts = [stmts]

        stmts = Tuple(*stmts)
        # ...

        obj = Basic.__new__(cls, iterable, index, stmts)
        obj._iterator  = iterator
        obj._generator = generator

        return obj

    @property
    def iterable(self):
        return self._args[0]

    @property
    def index(self):
        return self._args[1]

    @property
    def stmts(self):
        return self._args[2]

    @property
    def iterator(self):
        return self._iterator

    @property
    def generator(self):
        return self._generator

    def get_geometry_stmts(self, mapping):
        args = []

        l_quad = list(self.generator.atoms(LocalTensorQuadrature))
        if len(l_quad) == 0:
            return Tuple(*args)

        assert(len(l_quad) == 1)
        l_quad = l_quad[0]

        if isinstance(mapping, IdentityMapping):
            args += [ComputeLogical(WeightedVolumeQuadrature(l_quad))]
            return Tuple(*args)

        args += [ComputeLogical(WeightedVolumeQuadrature(l_quad))]

        # add stmts related to the geometry
        # TODO add other expressions
        args += [ComputeLogical(SymbolicDeterminant(mapping))]
        args += [ComputeLogical(SymbolicInverseDeterminant(mapping))]
        args += [ComputeLogical(SymbolicWeightedVolume(mapping))]

        return Tuple(*args)

#==============================================================================
class TensorIteration(BaseNode):
    """
    """

    def __new__(cls, iterator, generator):
        # ...
        if not( isinstance(iterator, TensorIterator) ):
            raise TypeError('Expecting an TensorIterator')

        if not( isinstance(generator, TensorGenerator) ):
            raise TypeError('Expecting a TensorGenerator')
        # ...

        return Basic.__new__(cls, iterator, generator)

    @property
    def iterator(self):
        return self._args[0]

    @property
    def generator(self):
        return self._args[1]

#==============================================================================
class ProductIteration(BaseNode):
    """
    """

    def __new__(cls, iterator, generator):
        # ...
        if not( isinstance(iterator, ProductIterator) ):
            raise TypeError('Expecting an ProductIterator')

        if not( isinstance(generator, ProductGenerator) ):
            raise TypeError('Expecting a ProductGenerator')
        # ...

        return Basic.__new__(cls, iterator, generator)

    @property
    def iterator(self):
        return self._args[0]

    @property
    def generator(self):
        return self._args[1]

#==============================================================================
class SplitArray(BaseNode):
    """
    """
    def __new__(cls, target, positions, lengths):
        if not isinstance(positions, (list, tuple, Tuple)):
            positions = [positions]
        positions = Tuple(*positions)

        if not isinstance(lengths, (list, tuple, Tuple)):
            lengths = [lengths]
        lengths = Tuple(*lengths)

        return Basic.__new__(cls, target, positions, lengths)

    @property
    def target(self):
        return self._args[0]

    @property
    def positions(self):
        return self._args[1]

    @property
    def lengths(self):
        return self._args[2]

#==============================================================================
def construct_logical_expressions(u, nderiv):
    dim = u.space.ldim

    ops = [dx1, dx2, dx3][:dim]
    r = range(nderiv+1)
    ranges = [r]*dim
    indices = product(*ranges)

    indices = list(indices)
    indices = [ijk for ijk in indices if sum(ijk) <= nderiv]

    args = []
    u = [u] if isinstance(u, ScalarTestFunction) else [u[i] for i in range(dim)]
    for ijk in indices:
        for atom in u:
            for n,op in zip(ijk, ops):
                for i in range(1, n+1):
                    atom = op(atom)
            args.append(atom)
    return [ComputeLogicalBasis(i) for i in args]


#==============================================================================
class GeometryExpressions(Basic):
    """
    """
    def __new__(cls, M, nderiv):
        dim = M.rdim

        ops = [dx1, dx2, dx3][:dim]
        r = range(nderiv+1)
        ranges = [r]*dim
        indices = product(*ranges)

        indices = list(indices)
        indices = [ijk for ijk in indices if sum(ijk) <= nderiv]

        args = []
        for d in range(dim):
            for ijk in indices:
                atom = M[d]
                for n,op in zip(ijk, ops):
                    for i in range(1, n+1):
                        atom = op(atom)
                args.append(atom)

        expressions = [GeometryExpr(i) for i in args]

        args        = Tuple(*args)
        expressions = Tuple(*expressions)
        return Basic.__new__(cls, args, expressions)

    @property
    def arguments(self):
        return self._args[0]

    @property
    def expressions(self):
        return self._args[1]
#==============================================================================
def construct_itergener(a, index):
    """
    """
    # ... create generator
    if isinstance(a, GlobalTensorQuadrature):
        generator = TensorGenerator(a, index)
        element   = LocalTensorQuadrature()

    elif isinstance(a, LocalTensorQuadrature):
        generator = TensorGenerator(a, index)
        element   = TensorQuadrature()

    elif isinstance(a, GlobalTensorQuadratureTrialBasis):
        generator = TensorGenerator(a, index)
        element   = LocalTensorQuadratureTrialBasis(a.target)

    elif isinstance(a, LocalTensorQuadratureTrialBasis):
        generator = TensorGenerator(a, index)
        element   = TensorQuadratureTrialBasis(a.target)

    elif isinstance(a, TensorQuadratureTrialBasis):
        generator = TensorGenerator(a, index)
        element   = TensorTrialBasis(a.target)

    elif isinstance(a, GlobalTensorQuadratureTestBasis):
        generator = TensorGenerator(a, index)
        element   = LocalTensorQuadratureTestBasis(a.target)

    elif isinstance(a, LocalTensorQuadratureTestBasis):
        generator = TensorGenerator(a, index)
        element   = TensorQuadratureTestBasis(a.target)

    elif isinstance(a, TensorQuadratureTestBasis):
        generator = TensorGenerator(a, index)
        element   = TensorTestBasis(a.target)

    elif isinstance(a, GlobalTensorQuadratureBasis):
        generator = TensorGenerator(a, index)
        element   = LocalTensorQuadratureBasis(a.target)

    elif isinstance(a, LocalTensorQuadratureBasis):
        generator = TensorGenerator(a, index)
        element   = TensorQuadratureBasis(a.target)

    elif isinstance(a, TensorQuadratureBasis):
        generator = TensorGenerator(a, index)
        element   = TensorBasis(a.target)

    elif isinstance(a, GlobalSpan):
        generator = TensorGenerator(a, index)
        element   = Span(a.target)

    elif isinstance(a, MatrixLocalBasis):
        generator = ProductGenerator(a, index)
        element   = CoefficientBasis(a.target)

    elif isinstance(a, GeometryExpr):
        generator = ProductGenerator(a.expr, index)
        element   = a.atom

    else:
        raise TypeError('{} not available'.format(type(a)))
    # ...

    # ... create iterator
    if isinstance(element, LocalTensorQuadrature):
        iterator = TensorIterator(element)

    elif isinstance(element, TensorQuadrature):
        iterator = TensorIterator(element)

    elif isinstance(element, LocalTensorQuadratureTrialBasis):
        iterator = TensorIterator(element)

    elif isinstance(element, TensorQuadratureTrialBasis):
        iterator = TensorIterator(element)

    elif isinstance(element, TensorTrialBasis):
        iterator = TensorIterator(element)

    elif isinstance(element, LocalTensorQuadratureTestBasis):
        iterator = TensorIterator(element)

    elif isinstance(element, TensorQuadratureTestBasis):
        iterator = TensorIterator(element)

    elif isinstance(element, TensorTestBasis):
        iterator = TensorIterator(element)

    elif isinstance(element, LocalTensorQuadratureBasis):
        iterator = TensorIterator(element)

    elif isinstance(element, TensorQuadratureBasis):
        iterator = TensorIterator(element)

    elif isinstance(element, TensorBasis):
        iterator = TensorIterator(element)

    elif isinstance(element, Span):
        iterator = TensorIterator(element)

    elif isinstance(element, CoefficientBasis):
        iterator = ProductIterator(element)

    elif isinstance(element, GeometryAtom):
        iterator = ProductIterator(element)

    else:
        raise TypeError('{} not available'.format(type(element)))
    # ...

    return iterator, generator

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
from sympy import Matrix, ImmutableDenseMatrix
from sympy import symbols

from sympde.expr import TerminalExpr
from sympde.expr import LinearForm
from sympde.expr import BilinearForm
from sympde.expr import Functional
from sympde.topology             import element_of
from sympde.topology             import ScalarField
from sympde.topology             import VectorField, IndexedVectorField
from sympde.topology.space       import ScalarTestFunction
from sympde.topology.space       import VectorTestFunction
from sympde.topology.space       import IndexedTestTrial
from sympde.topology.derivatives import _partial_derivatives
from sympde.topology.derivatives import _logical_partial_derivatives
from sympde.topology.derivatives import get_max_partial_derivatives

class DefNode(Basic):
    """."""
    def __new__(cls, name, arguments, mats, local_variables, body):
        return Basic.__new__(cls, name, arguments, mats, local_variables, body)

    @property
    def name(self):
        return self._args[0]

    @property
    def arguments(self):
        return self._args[1]

    @property
    def mats(self):
        return self._args[2]

    @property
    def local_variables(self):
        return self._args[3]

    @property
    def body(self):
        return self._args[4]

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

    l_mats  = BlockStencilMatrixLocalBasis(trials, tests, pads, terminal_expr)
    g_mats  = BlockStencilMatrixGlobalBasis(trials, tests, pads, terminal_expr)

    q_basis_tests  = {v:d_tests[v]['array']  for v in tests}
    q_basis_trials = {u:d_trials[u]['array'] for u in trials}

    l_basis_tests  = {v:d_tests[v]['local']  for v in tests}
    l_basis_trials = {u:d_trials[u]['local'] for u in trials}

    g_basis_tests  = {v:d_tests[v]['global']  for v in tests}
    g_basis_trials = {u:d_trials[u]['global'] for u in trials}

    # TODO d_trials or d_tests here?
    g_span          = {u:d_trials[u]['span'] for u in trials}

    # ...........................................................................................

    eval_mapping = EvalMapping(index_quad, index_dof_trial, q_basis_trials[trials[0]], l_basis_trials[trials[0]], Mapping, geo, spaces[1], nderiv)

    if atomic_expr_field:
        eval_field   = EvalField(atomic_expr_field, index_quad, q_basis_trials[trials[0]], coeff, trials, nderiv)

    #=========================================================begin kernel======================================================
    stmts = []
    for v in tests:
        stmts += construct_logical_expressions(v, nderiv)
    
    for expr in atomic_expr[:]:
        stmts += [ComputePhysicalBasis(i) for i in expr]
    # ...

    if atomic_expr_field:
        stmts += list(eval_fiel.inits)

    loop  = Loop((l_quad, *q_basis_tests.values(), *q_basis_trials.values(), geo), index_quad, stmts)
    loop  = Reduce('+', ComputeKernelExpr(terminal_expr), ElementOf(l_mats), loop)
    # ... loop over tests to evaluate fields

    fields = EmptyLine()
    if atomic_expr_field:
        fields = Loop((*l_basis_trials, l_coeff), index_dof, [eval_field])

    # ... loop over trials

    stmts = [loop]
    for u in l_basis_trials:
        loop  = Loop(l_basis_trials[u], index_dof_trial, stmts)

    # ... loop over tests

    stmts = [loop]
    for v in l_basis_tests:
        loop  = Loop(l_basis_tests[v], index_dof_test, stmts)
    # ...

    body  = (EmptyLine(), eval_mapping, fields, Reset(l_mats), loop)
    stmts = Block(body)

    #=========================================================end kernel=========================================================

    # ... loop over global elements
    loop  = Loop((g_quad, *g_basis_tests.values(), *g_basis_trials.values(), *g_span.values()),
                  index_element, stmts)


    body = [Reduce('+', l_mats, g_mats, loop)]
    # ...

    indices = [index_element, index_quad, index_dof_test, index_dof_trial]
    lengths = [i.length for i in indices]

    args = [*g_basis_tests.values(), *g_basis_trials.values(), *g_span.values(), g_quad, *lengths, *pads]

    if atomic_expr_field:
        args += [*l_coeffs]
    if eval_mapping:
        args+= [*eval_mapping.coeffs]

    mats  = [l_mats, g_mats]

    local_vars = [*q_basis_tests, *q_basis_trials]
    stmt = DefNode('assembly', args, mats, local_vars, body)

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
    g_vecs  = BlockStencilVectorGlobalBasis(tests, pads, terminal_expr)

    q_basis = {v:d_tests[v]['array']  for v in tests}
    l_basis = {v:d_tests[v]['local']  for v in tests}
    g_basis = {v:d_tests[v]['global']  for v in tests}
    g_span  = {u:d_tests[u]['span'] for u in tests}


    # ...........................................................................................

    eval_mapping = EvalMapping(index_quad, index_dof_test, q_basis[tests[0]], l_basis[tests[0]], Mapping, geo, space, nderiv)

    if atomic_expr_field:
        eval_field   = EvalField(atomic_expr_field, index_quad, q_basis, coeff, tests, nderiv)

    # ...
    #=========================================================begin kernel======================================================
    stmts = []
    for v in tests:
        stmts += construct_logical_expressions(v, nderiv)

    for expr in atomic_expr[:]:
        stmts += [ComputePhysicalBasis(i) for i in expr]

    if atomic_expr_field:
        stmts += list(eval_fiel.inits)

    loop  = Loop((l_quad, *q_basis.values(), geo), index_quad, stmts)
    loop = Reduce('+', ComputeKernelExpr(terminal_expr), ElementOf(l_vecs), loop)
    # ...

    # ... loop over tests to evaluate fields
    fields = EmptyLine()
    if atomic_expr_field:
        fields = Loop((*l_basis, l_coeff), index_dof, [eval_field])

    # ... loop over tests

    stmts = [loop]
    for v in l_basis:
        loop  = Loop(l_basis[v], index_dof_test, stmts)
    # ...

    body  = (EmptyLine(), eval_mapping, fields, Reset(l_vecs), loop)
    stmts = Block(body)
    # ...
    #=========================================================end kernel=========================================================
    # ... loop over global elements
    loop  = Loop((g_quad, *g_basis.values(), *g_span.values()), index_element, stmts)
    # ...

    body = (Reduce('+', l_vecs, g_vecs, loop),)

    indices = [index_element, index_quad, index_dof_test]
    lengths = [i.length for i in indices]

    args       = [*g_basis, *g_span, g_quad, *lengths, *pads]

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

    eval_mapping = EvalMapping(index_quad, index_dof_test, q_basis[tests[0]], l_basis[tests[0]], Mapping, geo, space, nderiv)
    eval_field   = EvalField(atomic_expr, index_quad, q_basis[tests[0]], coeffs, tests, nderiv)

    #=========================================================begin kernel======================================================
    # ... loop over tests functions

    loop   = Loop((l_quad, *q_basis.values(), geo), index_quad, eval_field.inits)
    loop   = Reduce('+', ComputeKernelExpr(terminal_expr), ElementOf(l_vec), loop)

    # ...
    # ... loop over tests functions to evaluate the fields
    fields = Loop((*l_basis.values(), *l_coeffs), index_dof_test, [eval_field])
    stmts  = Block([EmptyLine(), eval_mapping, fields, Reset(l_vec), loop])

    #=========================================================end kernel=========================================================
    # ... loop over global elements


    loop  = Loop((g_quad, *g_basis.values(), *g_span.values()), index_element, stmts)
    # ...

    body = (Reduce('+', l_vec, g_vec, loop),)

    indices = [index_element, index_quad, index_dof_test]
    lengths = [i.length for i in indices]
    args    = [*g_basis, *g_span, g_quad, *lengths, *pads, *coeffs]

    if eval_mapping:
        args+= [*eval_mapping.coeffs]

    mats = [l_vec, g_vec]

    local_vars = [*q_basis]
    stmt = DefNode('assembly', args, mats, local_vars, body)
    # ...

    return stmt

