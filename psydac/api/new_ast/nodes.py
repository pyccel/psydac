from collections import OrderedDict
from itertools import product

from sympy import Basic
from sympy.core.singleton import Singleton
from sympy.core.compatibility import with_metaclass
from sympy.core.containers import Tuple
from sympy import AtomicExpr
from sympy import Symbol

from sympde.topology import ScalarTestFunction, VectorTestFunction
from sympde.topology import (dx1, dx2, dx3)
from sympde.topology import Mapping
from sympde.topology import SymbolicDeterminant
from sympde.topology import SymbolicInverseDeterminant
from sympde.topology import SymbolicWeightedVolume
from sympde.topology import IdentityMapping

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
class IndexNode(with_metaclass(Singleton, Basic)):
    """Base class representing one index of an iterator"""
    pass

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
    pass

index_element   = IndexElement()
index_quad      = IndexQuadrature()
index_dof       = IndexDof()
index_dof_trial = IndexDofTrial()
index_dof_test  = IndexDofTest()
index_deriv     = IndexDerivative()

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

length_element   = LengthElement()
length_quad      = LengthQuadrature()
length_dof       = LengthDof()
length_dof_trial = LengthDofTrial()
length_dof_test  = LengthDofTest()

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
    _free_positions = None

    @property
    def rank(self):
        return self._rank

    @property
    def positions(self):
        return self._positions

    @property
    def free_positions(self):
        if self._free_positions is None:
            return list(self.positions.keys())

        else:
            return self._free_positions

    def pattern(self, args=None):
        if args is None:
            args = self.free_positions

        positions = {}
        for a in args:
            positions[a] = self.positions[a]

        args = [None]*self.rank
        for k,v in positions.items():
            args[v] = k

        return Pattern(*args)

#==============================================================================
class MatrixNode(BaseNode, AtomicExpr):
    """
    """
    _rank = None

    @property
    def rank(self):
        return self._rank

    def pattern(self, positions):
        raise NotImplementedError('TODO')

#==============================================================================
class GlobalTensorQuadrature(ArrayNode):
    """
    """
    _rank = 2
    _positions = {index_element: 0, index_quad: 1}
    _free_positions = [index_element]

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
    _free_positions = [index_element]

    def __new__(cls, target):
        if not isinstance(target, (ScalarTestFunction, VectorTestFunction)):
            raise TypeError('Expecting a scalar/vector test function')

        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class LocalTensorQuadratureBasis(ArrayNode):
    """
    """
    _rank = 3
    _positions = {index_quad: 2, index_deriv: 1, index_dof: 0}
    _free_positions = [index_dof]

    def __new__(cls, target):
        if not isinstance(target, (ScalarTestFunction, VectorTestFunction)):
            raise TypeError('Expecting a scalar/vector test function')

        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class TensorQuadratureBasis(ArrayNode):
    """
    """
    _rank = 2
    _positions = {index_quad: 1, index_deriv: 0}
    _free_positions = [index_quad]

    def __new__(cls, target):
        if not isinstance(target, (ScalarTestFunction, VectorTestFunction)):
            raise TypeError('Expecting a scalar/vector test function')

        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class CoefficientBasis(ScalarNode):
    """
    """
    def __new__(cls, target):
        if not isinstance(target, (ScalarTestFunction, VectorTestFunction)):
            raise TypeError('Expecting a scalar/vector test function')

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
    _free_positions = [index_dof_test]

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
    _free_positions = [index_dof_trial]

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
    def __new__(cls, pads):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = 2*len(pads)
        tag  = random_string( 6 )
        return Basic.__new__(cls, pads, rank, tag)

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def tag(self):
        return self._args[2]

#==============================================================================
class StencilVectorLocalBasis(MatrixNode):
    """
    used to describe local dof over an element as a stencil vector
    """
    def __new__(cls, pads):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = len(pads)
        tag  = random_string( 6 )
        return Basic.__new__(cls, pads, rank, tag)

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def tag(self):
        return self._args[2]

#==============================================================================
class StencilMatrixGlobalBasis(MatrixNode):
    """
    used to describe local dof over an element as a stencil matrix
    """
    def __new__(cls, pads):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = 2*len(pads)
        tag  = random_string( 6 )
        return Basic.__new__(cls, pads, rank, tag)

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def tag(self):
        return self._args[2]

#==============================================================================
class StencilVectorGlobalBasis(MatrixNode):
    """
    used to describe local dof over an element as a stencil vector
    """
    def __new__(cls, pads):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = len(pads)
        tag  = random_string( 6 )
        return Basic.__new__(cls, pads, rank, tag)

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def tag(self):
        return self._args[2]

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
    pass

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
    def __new__(cls, expr):
        return Basic.__new__(cls, expr)

    @property
    def expr(self):
        return self._args[0]

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
        ls  = list(expr.atoms(ScalarTestFunction))
        ls += list(expr.atoms(VectorTestFunction))
        if not(len(ls) == 1):
            print(expr, type(expr))
            print(ls)
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
            print(expr, type(expr))
            print(ls)
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
        geos   = [i.arguments for i in iterable if isinstance(i, GeometryExpressions)]
        with_geo = False # TODO remove
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
    for ijk in indices:
        atom = u
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

        args = [GeometryExpr(i) for i in args]

        args = Tuple(*args)
        return Basic.__new__(cls, args)

    @property
    def arguments(self):
        return self._args[0]

#==============================================================================
def construct_itergener(a, index):
    """
    """
    # ... create generator
    if isinstance(a, GlobalTensorQuadrature):
        generator = TensorGenerator(a, index)
        element = LocalTensorQuadrature()

    elif isinstance(a, LocalTensorQuadrature):
        generator = TensorGenerator(a, index)
        element = TensorQuadrature()

    elif isinstance(a, GlobalTensorQuadratureTrialBasis):
        generator = TensorGenerator(a, index)
        element = LocalTensorQuadratureTrialBasis(a.target)

    elif isinstance(a, LocalTensorQuadratureTrialBasis):
        generator = TensorGenerator(a, index)
        element = TensorQuadratureTrialBasis(a.target)

    elif isinstance(a, TensorQuadratureTrialBasis):
        generator = TensorGenerator(a, index)
        element = TensorTrialBasis(a.target)

    elif isinstance(a, GlobalTensorQuadratureTestBasis):
        generator = TensorGenerator(a, index)
        element = LocalTensorQuadratureTestBasis(a.target)

    elif isinstance(a, LocalTensorQuadratureTestBasis):
        generator = TensorGenerator(a, index)
        element = TensorQuadratureTestBasis(a.target)

    elif isinstance(a, TensorQuadratureTestBasis):
        generator = TensorGenerator(a, index)
        element = TensorTestBasis(a.target)

    elif isinstance(a, GlobalTensorQuadratureBasis):
        generator = TensorGenerator(a, index)
        element = LocalTensorQuadratureBasis(a.target)

    elif isinstance(a, LocalTensorQuadratureBasis):
        generator = TensorGenerator(a, index)
        element = TensorQuadratureBasis(a.target)

    elif isinstance(a, TensorQuadratureBasis):
        generator = TensorGenerator(a, index)
        element = TensorBasis(a.target)

    elif isinstance(a, GlobalSpan):
        generator = TensorGenerator(a, index)
        element = Span(a.target)

    elif isinstance(a, MatrixLocalBasis):
        generator = ProductGenerator(a, index)
        element = CoefficientBasis(a.target)

    elif isinstance(a, GeometryExpr):
        generator = ProductGenerator(a.expr, index)
        element = a.atom

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
    """
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
from pyccel.ast.core      import _atomic
from sympde.expr import TerminalExpr
from sympde.expr import LinearForm
from sympde.expr import BilinearForm
from sympde.topology             import element_of
from sympde.topology             import ScalarField
from sympde.topology             import VectorField, IndexedVectorField
from sympde.topology.space       import ScalarTestFunction
from sympde.topology.space       import VectorTestFunction
from sympde.topology.space       import IndexedTestTrial
from sympde.topology.derivatives import _partial_derivatives
from sympde.topology.derivatives import _logical_partial_derivatives
from sympde.topology.derivatives import get_max_partial_derivatives
class AST(object):
    """
    """
    def __init__(self, expr):
        # ... compute terminal expr
        # TODO check that we have one single domain/interface/boundary
        terminal_expr = TerminalExpr(expr)
        domain        = terminal_expr[0].target
        terminal_expr = terminal_expr[0].expr

#        print('> terminal expr = ', terminal_expr)
        # ...

        # ... compute max deriv
        nderiv = 0
        if isinstance(terminal_expr, Matrix):
            n_rows, n_cols = terminal_expr.shape
            for i_row in range(0, n_rows):
                for i_col in range(0, n_cols):
                    d = get_max_partial_derivatives(terminal_expr[i_row,i_col])
                    nderiv = max(nderiv, max(d.values()))
        else:
            d = get_max_partial_derivatives(terminal_expr)
            nderiv = max(nderiv, max(d.values()))

#        print('> nderiv = ', nderiv)
        # ...

        # ...
        is_bilinear   = False
        is_linear     = False
        is_functional = False
        tests         = []
        trials        = []

        if isinstance(expr, LinearForm):
            is_linear = True
            tests     = expr.test_functions

        elif isinstance(expr, BilinearForm):
            is_bilinear = True
            tests       = expr.test_functions
            trials      = expr.trial_functions

        else:
            raise NotImplementedError('TODO')
        # ...

        # ...
        atoms_types = (_partial_derivatives,
                       VectorTestFunction,
                       ScalarTestFunction,
                       IndexedTestTrial,
                       ScalarField,
                       VectorField, IndexedVectorField)

        atoms  = _atomic(terminal_expr, cls=atoms_types)
        # ...

        # ...
        atomic_expr_field        = [atom for atom in atoms if is_scalar_field(atom)]
        atomic_expr_vector_field = [atom for atom in atoms if is_vector_field(atom)]

        atomic_expr = [atom for atom in atoms if not( atom in atomic_expr_field ) and
                                                 not( atom in atomic_expr_vector_field)]
        # ...

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

        # ...
        if is_linear:
            ast = _create_ast_linear_form(terminal_expr, atomic_expr, tests, d_tests,
                                          nderiv, domain.dim)

        elif is_bilinear:
            ast = _create_ast_bilinear_form(terminal_expr, atomic_expr,
                                            tests, d_tests,
                                            trials, d_trials,
                                            nderiv, domain.dim)


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
def _create_ast_linear_form(terminal_expr, atomic_expr, tests, d_tests, nderiv, dim):
    """
    """
    pads   = symbols('p1, p2, p3')[:dim]
    g_quad = GlobalTensorQuadrature()
    l_quad = LocalTensorQuadrature()

    # ...
    stmts = []
    for v in tests:
        stmts += construct_logical_expressions(v, nderiv)

    stmts += [ComputePhysicalBasis(i) for i in atomic_expr]
    # ...

    # ...
    a_basis = tuple([d['array'] for v,d in d_tests.items()])

    loop  = Loop((l_quad, *a_basis), index_quad, stmts)
    # ...

    # ... TODO
    l_vec = StencilVectorLocalBasis(pads)
    # ...

    # ...
    loop = Reduce('+', ComputeKernelExpr(terminal_expr), ElementOf(l_vec), loop)
    # ...

    # ... loop over tests
    l_basis = tuple([d['local'] for v,d in d_tests.items()])
    stmts = [loop]
    loop  = Loop(l_basis, index_dof_test, stmts)
    # ...

    # ... TODO
    body  = (Reset(l_vec), loop)
    stmts = Block(body)
    # ...

    # ...
    g_basis = tuple([d['global'] for v,d in d_tests.items()])
    g_span  = tuple([d['span']   for v,d in d_tests.items()])

    loop  = Loop((g_quad, *g_basis, *g_span), index_element, stmts)
    # ...

    # ... TODO
    g_vec = StencilVectorGlobalBasis(pads)
    # ...

    # ... TODO
    body = (Reset(g_vec), Reduce('+', l_vec, g_vec, loop))
    stmt = Block(body)
    # ...

    return stmt

#==============================================================================
def _create_ast_bilinear_form(terminal_expr, atomic_expr,
                              tests, d_tests,
                              trials, d_trials,
                              nderiv, dim):
    """
    """
    pads   = symbols('p1, p2, p3')[:dim]
    g_quad = GlobalTensorQuadrature()
    l_quad = LocalTensorQuadrature()

    # ...
    stmts = []
    for v in tests:
        stmts += construct_logical_expressions(v, nderiv)

    stmts += [ComputePhysicalBasis(i) for i in atomic_expr]
    # ...

    # ...
    a_basis_tests  = tuple([d['array'] for v,d in d_tests.items()])
    a_basis_trials = tuple([d['array'] for v,d in d_trials.items()])

    loop  = Loop((l_quad, *a_basis_tests, *a_basis_trials), index_quad, stmts)
    # ...

    # ... TODO
    l_mat = StencilMatrixLocalBasis(pads)
    # ...

    # ...
    loop = Reduce('+', ComputeKernelExpr(terminal_expr), ElementOf(l_mat), loop)
    # ...

    # ... loop over trials
    l_basis = tuple([d['local'] for v,d in d_trials.items()])
    stmts = [loop]
    loop  = Loop(l_basis, index_dof_trial, stmts)
    # ...

    # ... loop over tests
    l_basis = tuple([d['local'] for v,d in d_tests.items()])
    stmts = [loop]
    loop  = Loop(l_basis, index_dof_test, stmts)
    # ...

    # ... TODO
    body  = (Reset(l_mat), loop)
    stmts = Block(body)
    # ...

    # ...
    g_basis_tests  = tuple([d['global'] for v,d in d_tests.items()])
    g_basis_trials = tuple([d['global'] for v,d in d_trials.items()])
    # TODO d_trials or d_tests here?
    g_span         = tuple([d['span']   for v,d in d_trials.items()])

    loop  = Loop((g_quad, *g_basis_tests, *g_basis_trials, *g_span),
                 index_element, stmts)
    # ...

    # ... TODO
    g_mat = StencilMatrixGlobalBasis(pads)
    # ...

    # ... TODO
    body = (Reset(g_mat), Reduce('+', l_mat, g_mat, loop))
    stmt = Block(body)
    # ...

    return stmt
