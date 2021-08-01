# -*- coding: UTF-8 -*-

from collections import OrderedDict
from itertools import product

from sympy import Basic, Expr
from sympy import AtomicExpr
from sympy import Function
from sympy import Mul
from sympy.core.singleton     import Singleton
from sympy.core.containers    import Tuple
from sympy.core.compatibility import with_metaclass

from sympde.topology import elements_of
from sympde.topology import ScalarFunction, VectorFunction
from sympde.topology import VectorFunctionSpace
from sympde.topology import IndexedVectorFunction
from sympde.topology import H1SpaceType, L2SpaceType, UndefinedSpaceType
from sympde.topology import Mapping
from sympde.topology import dx1, dx2, dx3
from sympde.topology import get_atom_logical_derivatives

from pyccel.ast.core import AugAssign, Assign
from pyccel.ast.core import _atomic

#==============================================================================
# TODO move it
import string
import random
def random_string( n ):
    chars    = string.ascii_lowercase + string.digits
    selector = random.SystemRandom()
    return ''.join( selector.choice( chars ) for _ in range( n ) )

class ZerosLike(Function):
    @property
    def rhs(self):
        return self._args[0]

class Zeros(Function):
    @property
    def shape(self):
        return self._args[0]

class FloorDiv(Function):
    def __new__(cls, arg1, arg2):
        if arg2 == 1:
            return arg1
        else:
            return Basic.__new__(cls, arg1, arg2)

    @property
    def arg1(self):
        return self._args[0]

    @property
    def arg2(self):
        return self._args[1]
#==============================================================================
class ArityType(with_metaclass(Singleton, Basic)):
    """Base class representing a form type: bilinear/linear/functional"""

class BilinearArity(ArityType):
    pass

class LinearArity(ArityType):
    pass

class FunctionalArity(ArityType):
    pass

#==============================================================================
class LengthNode(Expr):
    """Base class representing one length of an iterator"""
    def __new__(cls, target=None):
        obj = Basic.__new__(cls, target)
        return obj

    @property
    def target(self):
        return self._args[0]

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

class LengthOuterDofTest(LengthNode):
    pass
    
class LengthInnerDofTest(LengthNode):
    pass
    
class TensorExpression(Expr):
    def __new__(cls, *args):
        return Expr.__new__(cls, *args)

class TensorIntDiv(TensorExpression):
    pass

class TensorAdd(TensorExpression):
    pass

class TensorMul(TensorExpression):
    pass
#==============================================================================
class TensorAssignExpr(Basic):
    def __new__(cls, lhs, rhs):
        assert isinstance(lhs, Expr)
        assert isinstance(rhs, Expr)
        return Basic.__new__(cls, lhs, rhs)

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]

#==============================================================================
class IndexNode(Expr):
    """Base class representing one index of an iterator"""
    def __new__(cls, length=None):
        obj = Basic.__new__(cls)
        obj._length = length
        return obj

    @property
    def length(self):
        return self._length

    def set_length(self, length):
        obj = type(self)(length)
        return obj

class IndexElement(IndexNode):
    pass

class IndexQuadrature(IndexNode):
    pass

class IndexDof(IndexNode):
    pass

class IndexDofTrial(IndexDof):
    pass

class IndexDofTest(IndexDof):
    pass

class IndexOuterDofTest(IndexDof):
    pass

class IndexInnerDofTest(IndexDof):
    pass

class IndexDerivative(IndexNode):
    def __new__(cls, length=None):
        return Basic.__new__(cls)

index_element        = IndexElement()
index_quad           = IndexQuadrature()
index_dof            = IndexDof()
index_dof_test       = IndexDofTest()
index_dof_trial      = IndexDofTrial()
index_deriv          = IndexDerivative()
index_outer_dof_test = IndexOuterDofTest()
index_inner_dof_test = IndexInnerDofTest()

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
class Mask(Basic):
    def __new__(cls, axis, ext):
        return Basic.__new__(cls, axis, ext)

    @property
    def axis(self):
        return self._args[0]

    @property
    def ext(self):
        return self._args[1]
#==============================================================================
class EvalField(BaseNode):
    """
       This function computes atomic expressions needed
       to evaluate  EvaluteField/VectorField final expression

        Parameters:
        ----------
        atoms: tuple_like (Expr)
            The atomic expression to be evaluated

        q_index: <IndexQuadrature>
            Indices used for the quadrature loops

        l_index : <IndexDofTest>
            Indices used for the basis loops

        q_basis : <GlobalTensorQuadratureTestBasis>
            The 1d basis function of the tensor-product space

        coeffs  : tuple_like (CoefficientBasis)
            Coefficient of the basis function

        l_coeffs : tuple_like (MatrixLocalBasis)
            Local coefficient of the basis functions

        g_coeffs : tuple_like (MatrixGlobalBasis)
            Global coefficient of the basis functions

        tests   : tuple_like (Variable)
            The field to be evaluated

        mapping : <Mapping>
            Sympde Mapping object

        nderiv  : int
            Maximum number of derivatives

        mask    : int,optional
            The fixed direction in case of a boundary integral
    """
    def __new__(cls, atoms, q_index, l_index, q_basis, coeffs, l_coeffs, g_coeffs, tests, mapping, nderiv, mask=None):

        stmts_1  = []
        stmts_2  = OrderedDict()
        inits    = []
        mats     = []

        for v in tests:
            stmts_1 += construct_logical_expressions(v, nderiv)

        logical_atoms   = [a.expr for a in stmts_1]
        new_atoms = {}
        for a in logical_atoms:
            atom = str(get_atom_logical_derivatives(a))
            if atom  in new_atoms:
                new_atoms[atom] += [a]
            else:
                new_atoms[atom]  = [a]

        logical_atoms = new_atoms
        for coeff, l_coeff in zip(coeffs,l_coeffs):
            for a in logical_atoms[str(coeff.target)]:
                node    = AtomicNode(a)
                mat     = MatrixQuadrature(a)
                val     = ProductGenerator(mat, q_index)
                rhs     = Mul(coeff,node)
                stmts_1 += [AugAssign(val, '+', rhs)]
                mats    += [mat]
                inits += [Assign(node,val)]
                stmts_2[coeff] = Assign(coeff, ProductGenerator(l_coeff, l_index))

        body  = Loop( q_basis, q_index, stmts=stmts_1, mask=mask)
        stmts_2 = [*stmts_2.values(), body]
        body  = Loop((), l_index, stmts_2)
        obj = Basic.__new__(cls, Tuple(*mats), inits, body)
        obj._l_coeffs = l_coeffs
        obj._g_coeffs = g_coeffs
        obj._tests    = tests
        obj._pads     = Pads(tests)
        return obj

    @property
    def atoms(self):
        return self._args[0]

    @property
    def inits(self):
        return self._args[1]

    @property
    def body(self):
        return self._args[2]

    @property
    def pads(self):
        return self._pads

    @property
    def l_coeffs(self):
        return self._l_coeffs

    @property
    def g_coeffs(self):
        return self._g_coeffs

class RAT(Basic):
    pass
#==============================================================================
class EvalMapping(BaseNode):
    """
        This function computes atomic expressions needed
        to evaluate  EvalMapping final expression.

        Parameters:
        ----------
        quads: <IndexQuadrature>
            Indices used for the quadrature loops

        indices_basis : <IndexDofTest>
            Indices used for the basis loops

        q_basis : <GlobalTensorQuadratureTestBasis>
            The 1d basis function of the tensor-product space

        mapping : <Mapping>
            Sympde Mapping object

        components  : <GeometryExpressions>
            The 1d coefficients of the mapping

        mapping_space : <VectorSpace>
            The vector space of the mapping

        tests   : tuple_like (Variable)
            The dummy variable for the test functions

        nderiv  : <int>
            Maximum number of derivatives

        mask    : int,optional
            The fixed direction in case of a boundary integral

        is_irational: bool,optional
            True if the mapping is irational
    """
    def __new__(cls, quads, indices_basis, q_basis, mapping, components, mapping_space, tests, nderiv, mask=None, is_rational=None):
        mapping_atoms  = components.arguments
        basis          = q_basis
        target         = basis.target
        multiplicity   = tuple(mapping_space.vector_space.shifts) if mapping_space else ()
        pads           = tuple(mapping_space.vector_space.pads) if mapping_space else ()
        
        if isinstance(target, IndexedVectorFunction):
            space          = target.base.space
        else:
            space      = target.space
        weight,        = elements_of(space, names='weight')
        if isinstance(target, VectorFunction):
            target = target[0]

        l_coeffs    = []
        g_coeffs    = []
        values      = set()

        components  = [get_atom_logical_derivatives(a) for a in mapping_atoms]
        test_atoms  = [a.subs(comp, target) for a,comp in zip(mapping_atoms, components)]
        weights     = [a.subs(comp, weight) for a,comp in zip(mapping_atoms, components)]

        stmts       = [ComputeLogicalBasis(v,) for v in set(test_atoms)]
        declarations = []
        rationalization = []
        for test,mapp,comp in zip(test_atoms, mapping_atoms, components):
            test    = AtomicNode(test)
            val     = ProductGenerator(MatrixQuadrature(mapp), quads)
            if is_rational:
                rhs     = Mul(CoefficientBasis(comp),CoefficientBasis(weight),test)
            else:
                rhs     = Mul(CoefficientBasis(comp),test)
            stmts  += [AugAssign(val, '+', rhs)]

            l_coeff = MatrixLocalBasis(comp)
            g_coeff = MatrixGlobalBasis(comp, target)
            declarations += [Assign(mapp, val)]
            if l_coeff not in l_coeffs:
                l_coeffs.append(l_coeff)
                g_coeffs.append(g_coeff)

            values.add(val.target)

        if is_rational:
            l_coeffs.append(MatrixLocalBasis(weight))
            g_coeffs.append(MatrixGlobalBasis(weight, target))

            for node, w in set(zip(test_atoms, weights)):
                node    = AtomicNode(node)
                val     = ProductGenerator(MatrixQuadrature(w), quads)
                rhs     = Mul(CoefficientBasis(weight),node)
                stmts  += [AugAssign(val, '+', rhs)]
                values.add(val.target)
                declarations += [Assign(w,val)]

            for test,mapp,w in zip(test_atoms, mapping_atoms, weights):
                comp = get_atom_logical_derivatives(mapp)
                lhs  = ProductGenerator(MatrixQuadrature(mapp), quads)
                rhs  = mapp.subs(comp,comp/weight)
                rationalization += [Assign(lhs, rhs)]

            rationalization  = [*declarations, *rationalization]

        loop   = Loop((q_basis,*l_coeffs), indices_basis, stmts)
        loop   = Loop((), quads, stmts=[loop, *rationalization], mask=mask)

        obj    = Basic.__new__(cls, loop, l_coeffs, g_coeffs, values, multiplicity, pads)
        obj._tests = tests
        return obj

    @property
    def loop(self):
        return self._args[0]

    @property
    def local_coeffs(self):
        return self._args[1]

    @property
    def coeffs(self):
        return self._args[2]

    @property
    def values(self):
        return self._args[3]

    @property
    def multiplicity(self):
        return self._args[4]

    @property
    def pads(self):
        return self._args[5]
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

        if not isinstance(target, (ArrayNode, MatrixNode, Expr)):
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
    _free_indices = [index_element, index_quad, index_dof]

    def __new__(cls, target):
        if not isinstance(target, (ScalarFunction, VectorFunction, IndexedVectorFunction)):
            raise TypeError('Expecting a scalar/vector test function')
        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

    @property
    def unique_scalar_space(self):
        unique_scalar_space = True
        if isinstance(self.target, IndexedVectorFunction):
            return True
        space = self.target.space
        if isinstance(space, VectorFunctionSpace):
            unique_scalar_space = isinstance(space.kind, (UndefinedSpaceType, H1SpaceType, L2SpaceType))
        return unique_scalar_space

    @property
    def is_scalar(self):
        return isinstance(self.target, (ScalarFunction, IndexedVectorFunction))

#==============================================================================
class LocalTensorQuadratureBasis(ArrayNode):
    """
    """
    _rank = 3
    _positions = {index_quad: 2, index_deriv: 1, index_dof: 0}
    _free_indices = [index_dof]

    def __new__(cls, target):
        if not isinstance(target, (ScalarFunction, VectorFunction, IndexedVectorFunction)):
            raise TypeError('Expecting a scalar/vector test function')
        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

    @property
    def unique_scalar_space(self):
        unique_scalar_space = True
        if isinstance(self.target, IndexedVectorFunction):
            return True
        space = self.target.space
        if isinstance(space, VectorFunctionSpace):
            unique_scalar_space = isinstance(space.kind, (UndefinedSpaceType, H1SpaceType, L2SpaceType))
        return unique_scalar_space

    @property
    def is_scalar(self):
        return isinstance(self.target, (ScalarFunction, IndexedVectorFunction))
#==============================================================================
class TensorQuadratureBasis(ArrayNode):
    """
    """
    _rank = 2
    _positions = {index_quad: 1, index_deriv: 0}
    _free_indices = [index_quad]

    def __new__(cls, target):
        if not isinstance(target, (ScalarFunction, VectorFunction, IndexedVectorFunction)):
            raise TypeError('Expecting a scalar/vector test function')

        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

    @property
    def unique_scalar_space(self):
        unique_scalar_space = True
        if isinstance(self.target, IndexedVectorFunction):
            return True
        space = self.target.space
        if isinstance(space, VectorFunctionSpace):
            unique_scalar_space = isinstance(space.kind, (UndefinedSpaceType, H1SpaceType, L2SpaceType))
        return unique_scalar_space

    @property
    def is_scalar(self):
        return isinstance(self.target, (ScalarFunction, IndexedVectorFunction))
#==============================================================================
class CoefficientBasis(ScalarNode):
    """
    """
    def __new__(cls, target):
        ls = target.atoms(ScalarFunction, VectorFunction, Mapping)
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
    _free_indices = [index_element, index_quad, index_dof_test]

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
    _free_indices = [index_element, index_quad, index_dof_trial]

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

class MatrixGlobalBasis(MatrixNode):
    """
    used to describe global dof
    """
    _rank = rank_dim

    def __new__(cls, target, test):
        # TODO check target
        return Basic.__new__(cls, target, test)

    @property
    def target(self):
        return self._args[0]

    @property
    def test(self):
        return self._args[1]
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
    def __new__(cls, trials, tests, expr, dim, tag=None, outer=None,
                     tests_degree=None, trials_degree=None,
                     tests_multiplicity=None, trials_multiplicity=None):

        pads = Pads(tests, trials, tests_degree, trials_degree,
                    tests_multiplicity, trials_multiplicity)

        rank = 2*dim
        tag  = tag or random_string( 6 )
        obj  = Basic.__new__(cls, pads, rank, trials_multiplicity, tag, expr)
        obj._trials = trials
        obj._tests  = tests
        obj._outer  = outer
        return obj

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def trials_multiplicity(self):
        return self._args[2]

    @property
    def tag(self):
        return self._args[3]

    @property
    def expr(self):
        return self._args[4]

    @property
    def outer(self):
        return self._outer

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
    def __new__(cls, trials, tests, pads, multiplicity, expr, tag=None):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = 2*len(pads)
        tag  = tag or random_string( 6 )
        obj  = Basic.__new__(cls, pads, multiplicity, rank, tag, expr)
        obj._trials = trials
        obj._tests  = tests
        return obj

    @property
    def pads(self):
        return self._args[0]

    @property
    def multiplicity(self):
        return self._args[1]

    @property
    def rank(self):
        return self._args[2]

    @property
    def tag(self):
        return self._args[3]

    @property
    def expr(self):
        return self._args[4]

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
    def __new__(cls,tests, pads, expr, tag=None):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = len(pads)
        tag  = tag or random_string( 6 )
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
    def __new__(cls, tests, pads, multiplicity, expr, tag=None):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = len(pads)
        tag  = tag or random_string( 6 )
        obj  = Basic.__new__(cls, pads, multiplicity, rank, tag, expr)
        obj._tests  = tests
        return obj

    @property
    def pads(self):
        return self._args[0]

    @property
    def multiplicity(self):
        return self._args[1]

    @property
    def rank(self):
        return self._args[2]

    @property
    def tag(self):
        return self._args[3]

    @property
    def expr(self):
        return self._args[4]

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
        if not isinstance(target, (ScalarFunction, VectorFunction, IndexedVectorFunction)):
            raise TypeError('Expecting a scalar/vector test function')

        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class Span(ScalarNode):
    """
    """
    def __new__(cls, target):
        if not isinstance(target, (ScalarFunction, VectorFunction, IndexedVectorFunction)):
            raise TypeError('Expecting a scalar/vector test function')

        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

class Pads(ScalarNode):
    """
    """
    def __new__(cls, tests, trials=None, tests_degree=None, trials_degree=None,
                    tests_multiplicity=None, trials_multiplicity=None):
        for target in tests:
            if not isinstance(target, (ScalarFunction, VectorFunction, IndexedVectorFunction)):
                raise TypeError('Expecting a scalar/vector test function')
        if trials:
            for target in trials:
                if not isinstance(target, (ScalarFunction, VectorFunction, IndexedVectorFunction)):
                    raise TypeError('Expecting a scalar/vector test function')
        obj = Basic.__new__(cls, tests, trials)
        obj._tests_degree = tests_degree
        obj._trials_degree = trials_degree
        obj._tests_multiplicity = tests_multiplicity
        obj._trials_multiplicity = trials_multiplicity
        return obj

    @property
    def tests(self):
        return self._args[0]

    @property
    def trials(self):
        return self._args[1]

    @property
    def tests_degree(self):
        return self._tests_degree

    @property
    def trials_degree(self):
        return self._trials_degree

    @property
    def tests_multiplicity(self):
        return self._tests_multiplicity

    @property
    def trials_multiplicity(self):
        return self._trials_multiplicity

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
    pass

#==============================================================================
class LogicalValueNode(ValueNode):
    pass

#==============================================================================
class PhysicalBasisValue(PhysicalValueNode):
    pass

#==============================================================================
class LogicalBasisValue(LogicalValueNode):
    pass

#==============================================================================
class PhysicalGeometryValue(PhysicalValueNode):
    pass

#==============================================================================
class LogicalGeometryValue(LogicalValueNode):
    pass

#==============================================================================
class BasisAtom(AtomicNode):
    """
    """
    def __new__(cls, expr):
        types = (IndexedVectorFunction, VectorFunction, ScalarFunction)

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
    class to describe a dimensionless loop of an iterator over a generator.

    Parameters
    ----------
    iterable : <list|iterator>
        list of iterator object

    index    : <IndexNode>
        represent the dimensionless index used in the for loop

    stmts    : <list|None>
        list of body statements

    mask     : <int|None>
        the masked dimension where we fix the index in that dimension

    """

    def __new__(cls, iterable, index, stmts=None, mask=None):
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

        obj = Basic.__new__(cls, iterable, index, stmts, mask)
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
    def mask(self):
        return self._args[3]

    @property
    def iterator(self):
        return self._iterator

    @property
    def generator(self):
        return self._generator

    def get_geometry_stmts(self, mapping):

        l_quad = list(self.generator.atoms(LocalTensorQuadrature))
        if len(l_quad) == 0:
            return Tuple()

        assert(len(l_quad) == 1)
        l_quad = l_quad[0]

        args = [ComputeLogical(WeightedVolumeQuadrature(l_quad))]
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
    if isinstance(u, IndexedVectorFunction):
        dim = u.base.space.ldim
    else:
        dim = u.space.ldim

    ops = [dx1, dx2, dx3][:dim]
    r = range(nderiv+1)
    ranges = [r]*dim
    indices = product(*ranges)

    indices = list(indices)
    indices = [ijk for ijk in indices if sum(ijk) <= nderiv]

    args = []
    u = [u] if isinstance(u, (ScalarFunction, IndexedVectorFunction)) else [u[i] for i in range(dim)]
    for ijk in indices:
        for atom in u:
            for n,op in zip(ijk, ops):
                for _ in range(1, n+1):
                    atom = op(atom)
            args.append(atom)
    return [ComputeLogicalBasis(i) for i in args]

#==============================================================================
class GeometryExpressions(Basic):
    """
    """
    def __new__(cls, M, nderiv):
        expressions = []
        args        = []
        if not M.is_analytical:

            dim = M.ldim
            nderiv = 1 if nderiv == 0 else nderiv
            ops = [dx1, dx2, dx3][:dim]
            r = range(nderiv+1)
            ranges = [r]*dim
            indices = product(*ranges)

            indices = list(indices)
            indices = [ijk for ijk in indices if sum(ijk) <= nderiv]

            for d in range(dim):
                for ijk in indices:
                    atom = M[d]
                    for n,op in zip(ijk, ops):
                        for _ in range(1, n+1):
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
        element   = TensorTrialBasis(a.target)

    elif isinstance(a, GlobalTensorQuadratureTestBasis):
        generator = TensorGenerator(a, index)
        element   = TensorTestBasis(a.target)

    elif isinstance(a, GlobalTensorQuadratureBasis):
        generator = TensorGenerator(a, index)
        element   = TensorBasis(a.target)

    elif isinstance(a, TensorQuadratureBasis):
        generator = TensorGenerator(a, index)
        element   = TensorBasis(a.target)

    elif isinstance(a, GlobalSpan):
        generator = TensorGenerator(a, index)
        element   = Span(a.target)

    elif isinstance(a, MatrixLocalBasis):
        generator = ProductGenerator(a, index)
        element   = CoefficientBasis(a.target)

    elif isinstance(a, MatrixGlobalBasis):
        generator = ProductGenerator(a, index)
        element   = MatrixLocalBasis(a.target)

    elif isinstance(a, GeometryExpr):
        generator = ProductGenerator(a.expr, index)
        element   = a.atom
    elif isinstance(a, TensorAssignExpr):
        generator = TensorGenerator(a.lhs, index)
        element   = a.rhs
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

    elif isinstance(element, MatrixLocalBasis):
        iterator = ProductIterator(element)

    elif isinstance(element, Expr):
        iterator = TensorIterator(element)
    else:
        raise TypeError('{} not available'.format(type(element)))
    # ...

    return iterator, generator

