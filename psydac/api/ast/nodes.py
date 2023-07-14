# -*- coding: UTF-8 -*-

from itertools import product

from sympy import Basic, Expr
from sympy import AtomicExpr, S
from sympy import Function
from sympy import Mul
from sympy.core.singleton     import Singleton
from sympy.core.containers    import Tuple
from sympde.old_sympy_utilities import with_metaclass

from sympde.topology import element_of
from sympde.topology import ScalarFunction, VectorFunction
from sympde.topology import VectorFunctionSpace
from sympde.topology import IndexedVectorFunction
from sympde.topology import H1SpaceType, L2SpaceType, UndefinedSpaceType
from sympde.topology import Mapping
from sympde.topology import dx1, dx2, dx3
from sympde.topology import get_atom_logical_derivatives

from psydac.pyccel.ast.core import AugAssign, Assign
from psydac.pyccel.ast.core import _atomic
from psydac.api.utilities   import flatten

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
    def __new__(cls, shape, dtype='float64'):
        return Basic.__new__(cls, shape, dtype)

    @property
    def shape(self):
        return self._args[0]

    @property
    def dtype(self):
        return self._args[1]

class Array(Function):
    def __new__(cls, data, dtype=None):
        return Basic.__new__(cls, data, dtype)

    @property
    def data(self):
        return self._args[0]

    @property
    def dtype(self):
        return self._args[1]

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

class Max(Expr):
    pass

class Min(Expr):
    pass

class Allocate(Basic):

    def __new__(cls, arr, shape):
        return Basic.__new__(cls, arr, shape)

    @property
    def array(self):
        return self._args[0]

    @property
    def shape(self):
        return self._args[1]
#==============================================================================
class VectorAssign(Basic):

    def __new__(cls, lhs, rhs, op=None):
        return Basic.__new__(cls, lhs, rhs, op)

    @property
    def lhs(self):
        return self._args[0]

    @property
    def rhs(self):
        return self._args[1]

    @property
    def op(self):
        return self._args[2]
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
    def __new__(cls, target=None, index=None):
        obj = Basic.__new__(cls, target, index)
        return obj

    @property
    def target(self):
        return self._args[0]

    @property
    def index(self):
        return self._args[1]

    def set_index(self, index):
        obj = type(self)(target=self.target, index=index)
        return obj

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

class NumThreads(LengthNode):
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

class TensorMax(TensorExpression):
    pass

class TensorInteger(TensorExpression):
    pass

#==============================================================================
class TensorAssignExpr(Basic):
    def __new__(cls, lhs, rhs):
        assert isinstance(lhs, (Expr, Tuple))
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
    def __new__(cls, start=0, stop=None, length=None, index=None):
        obj = Basic.__new__(cls)
        obj._start  = start
        obj._stop   = stop
        obj._length = length
        obj._index  = index
        return obj

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop

    @property
    def length(self):
        return self._length

    @property
    def index(self):
        return self._index

    def set_range(self, start=TensorInteger(0), stop=None, length=None):
        if length is None:
            length = stop
        obj = type(self)(start=start, stop=stop, length=length, index=self.index)
        return obj

    def set_index(self, index):
        obj = type(self)(start=self.start, stop=self.stop, length=self.length, index=index)
        return obj

    def _hashable_content(self):
        args = (self.start, self.stop, self.length, self.index)
        return tuple([a for a in args if a is not None])
 
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

class ThreadId(IndexDof):
    pass

class ThreadCoordinates(IndexDof):
    pass

class NeighbourThreadCoordinates(IndexDof):
    pass

class LocalIndexElement(IndexDof):
    pass

class IndexDerivative(IndexNode):
    def __new__(cls, length=None):
        return Basic.__new__(cls)

    def _hashable_content(self):
        return type(self).__mro__

index_element        = IndexElement()
thread_id            = ThreadId(length=NumThreads())
thread_coords        = ThreadCoordinates()
neighbour_threads    = NeighbourThreadCoordinates()
index_quad           = IndexQuadrature()
index_dof            = IndexDof()
index_dof_test       = IndexDofTest()
index_dof_trial      = IndexDofTrial()
index_deriv          = IndexDerivative()
index_outer_dof_test = IndexOuterDofTest()
index_inner_dof_test = IndexInnerDofTest()
local_index_element  = LocalIndexElement()

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
    def __new__(cls, atoms, q_index, l_index, q_basis, coeffs, l_coeffs, g_coeffs, tests, mapping, nderiv, mask=None, dtype='real'):

        stmts_1  = []
        stmts_2  = {}
        inits    = []
        mats     = []

        # TODO check why the rhs f is considered as a test function
        for v in tests:
            stmts_1 += construct_logical_expressions(v, nderiv, prefix='basis')

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
                # We add value in the array
                mat     = MatrixQuadrature(a, dtype)
                val     = ProductGenerator(mat, q_index)
                rhs     = Mul(coeff, BasisAtom(a, prefix='basis'))
                stmts_1 += [AugAssign(val, '+', rhs)]
                mats    += [mat]

                # We initialize array for the kernel loop
                inits += [Assign(BasisAtom(a), val)]

                # We extract data from arr_coeffs_{ScalarFunction.name}
                stmts_2[coeff] = Assign(coeff, ProductGenerator(l_coeff, l_index))

        quad_loop  = Loop( q_basis, q_index, stmts=stmts_1, mask=mask, prefix='basis')
        stmts_2 = [*stmts_2.values(), quad_loop]
        basis_loop  = Loop((), l_index, stmts=stmts_2)
        obj = Basic.__new__(cls, Tuple(*mats), inits, basis_loop, dtype)
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
    def dtype(self):
        return self._args[3]

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

        nderiv  : <int>
            Maximum number of derivatives

        mask    : int,optional
            The fixed direction in case of a boundary integral

        is_rational: bool,optional
            True if the mapping is rational
    """
    def __new__(cls, quads, indices_basis, q_basis, mapping, components, mapping_space, nderiv, mask=None, is_rational=None, trial=None):
        mapping_atoms  = components.arguments
        basis          = q_basis
        target         = basis.target
        multiplicity   = tuple(mapping_space.vector_space.shifts) if mapping_space else ()
        pads           = tuple(mapping_space.vector_space.pads) if mapping_space else ()
        
        if isinstance(target, IndexedVectorFunction):
            space          = target.base.space
        else:
            space      = target.space

        if mapping.is_plus:
            weight = element_of(space, name='weight_plus')
        else:
            weight = element_of(space, name='weight')

        if isinstance(target, VectorFunction):
            target = target[0]

        if isinstance(weight, VectorFunction):
            weight = weight[0]

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

        basis_loop  = Loop((q_basis,*l_coeffs), indices_basis, stmts=stmts)
        quad_loop   = Loop((), quads, stmts=[basis_loop , *rationalization], mask=mask)

        obj    = Basic.__new__(cls, quad_loop, l_coeffs, g_coeffs, values, multiplicity, pads)
        obj._mapping = mapping
        obj._trial   = trial
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

    @property
    def mapping(self):
        return self._mapping

    @property
    def trial(self):
        return self._trial

#==============================================================================
class IteratorBase(BaseNode):
    """
    """
    def __new__(cls, target, dummies=None, *, prefix=None):
        if not dummies is None:
            if not isinstance(dummies, (list, tuple, Tuple)):
                dummies = [dummies]
            dummies = Tuple(*dummies)

        if not isinstance(prefix, (str, type(None))):
            raise TypeError('expecting a prefix as a string')

        return Basic.__new__(cls, target, dummies, prefix)

    @property
    def target(self):
        return self._args[0]

    @property
    def dummies(self):
        return self._args[1]

    @property
    def prefix(self):
        return self._args[2]

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
            raise TypeError('expecting ArrayNode, MatrixNode or Expr')

        return Basic.__new__(cls, target, dummies)

    @property
    def target(self):
        return self._args[0]

    @property
    def dummies(self):
        return self._args[1]

#==============================================================================
class TensorGenerator(GeneratorBase):
    """
    This class represent an array list of array elements with arbitrary number of dimensions.
    the length of the list is given by the rank of target.

    Parameters
    ----------

    target : <ArrayNode|MatrixNode>
        the array object

    dummies : <Tuple|tuple|list>
        multidimensional index

    Examples
    --------
    >>> T = TensorGenerator(GlobalTensorQuadrature(), index_quad)
    >>> T
    TensorGenerator(GlobalTensorQuadrature(), (IndexQuadrature(),))
    >>> ast = parse(T, settings={'dim':2,'nderiv':2,'target':Square()})
    >>> ast[0]
    ((IndexedElement(local_x1, i_quad_1), IndexedElement(local_w1, i_quad_1)),
     (IndexedElement(local_x2, i_quad_2), IndexedElement(local_w2, i_quad_2)))
    """

#==============================================================================
class ProductGenerator(GeneratorBase):
    """
    This class represent an element of an array with arbitrary number of dimensions.

    Parameters
    ----------

    target : <ArrayNode|MatrixNode>
        the array object

    dummies : <Tuple|tuple|list>
        multidimensional index

    Examples
    --------
    >>> P = ProductGenerator(MatrixRankFromCoords(), thread_coords)
    >>> P
    ProductGenerator(MatrixRankFromCoords(), (ThreadCoordinates(),))
    >>> ast = parse(P, settings={'dim':2,'nderiv':2,'target':Square()})
    >>> ast
    IndexedElement(rank_from_coords, thread_coords_1, thread_coords_2)
    """

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

class BlockLinearOperatorNode(MatrixNode):
    pass

#==============================================================================
class GlobalTensorQuadratureGrid(ArrayNode):
    """This class represents the quadrature points and weights in a domain.
    """
    _rank = 2
    _positions = {index_element: 0, index_quad: 1}
    _free_indices = [index_element]

    def __init__(self, weights=True):
        self._weights = weights

    @property
    def weights( self ):
        return self._weights

#==============================================================================
class PlusGlobalTensorQuadratureGrid(GlobalTensorQuadratureGrid):
    """This class represents the quadrature points and weights in the plus side of an interface.
    """

#==============================================================================
class LocalTensorQuadratureGrid(ArrayNode):
    """ This class represents the element wise quadrature points and weights in a domain.
    """
    _rank = 1
    _positions = {index_quad: 0}

    def __init__(self, weights=True):
        self._weights = weights

    @property
    def weights( self ):
        return self._weights

#==============================================================================
class PlusLocalTensorQuadratureGrid(LocalTensorQuadratureGrid):
    """This class represents the element wise quadrature points and weights in the plus side of an interface.
    """

#==============================================================================
class TensorQuadrature(ScalarNode):
    """This class represents the quadrature point and weight in a domain."""

    def __init__(self, weights=True):
        self._weights = weights

    @property
    def weights( self ):
        return self._weights

#==============================================================================
class PlusTensorQuadrature(TensorQuadrature):
    """This class represents the quadrature point and weight in the plus side of an interface.
    """

#==============================================================================
class MatrixQuadrature(MatrixNode):
    """
    """
    _rank = rank_dim

    def __new__(cls, target, dtype='real'):
        # TODO check target
        return Basic.__new__(cls, target, dtype)

    @property
    def target(self):
        return self._args[0]

    @property
    def dtype(self):
        return self._args[1]

#==============================================================================
class MatrixRankFromCoords(MatrixNode):
    pass
#==============================================================================
class MatrixCoordsFromRank(MatrixNode):
    pass
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

    def __new__(cls, target, index=None):
        if not isinstance(target, (ScalarFunction, VectorFunction, IndexedVectorFunction)):
            raise TypeError('Expecting a scalar/vector test function')
        return Basic.__new__(cls, target, index)

    @property
    def target(self):
        return self._args[0]

    @property
    def index(self):
        return self._args[1]

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

    def set_index(self, index):
        return type(self)(self.target, index)
#==============================================================================
class LocalTensorQuadratureBasis(ArrayNode):
    """
    """
    _rank = 4
    _positions = {index_quad: 3, index_deriv: 2, index_dof: 1, index_element: 0}
    _free_indices = [index_element, index_quad, index_dof]

    def __new__(cls, target, index=None):
        if not isinstance(target, (ScalarFunction, VectorFunction, IndexedVectorFunction)):
            raise TypeError('Expecting a scalar/vector test function')
        return Basic.__new__(cls, target, index)

    @property
    def target(self):
        return self._args[0]

    @property
    def index(self):
        return self._args[1]

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

    def set_index(self, index):
        return type(self)(self.target, index)
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
    _positions = {index_quad: 3, index_deriv: 2, index_dof_test: 1, index_element: 0}
    _free_indices = [index_element, index_quad, index_dof_test]

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
    _positions = {index_quad: 3, index_deriv: 2, index_dof_trial: 1, index_element: 0}
    _free_indices = [index_element, index_quad, index_dof_trial]

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

    def __new__(cls, target, test, dtype='real'):
        # TODO check target
        return Basic.__new__(cls, target, test, dtype)

    @property
    def target(self):
        return self._args[0]

    @property
    def test(self):
        return self._args[1]

    @property
    def dtype(self):
        return self._args[2]
#==============================================================================
class MatrixLocalBasis(MatrixNode):
    """
    used to describe local dof over an element
    """
    _rank = rank_dim

    def __new__(cls, target, dtype='real'):
        # TODO check target
        return Basic.__new__(cls, target, dtype)

    @property
    def target(self):
        return self._args[0]

    @property
    def dtype(self):
        return self._args[1]

#==============================================================================
class StencilMatrixLocalBasis(MatrixNode):
    """
    used to describe local dof over an element as a stencil matrix
    """
    def __new__(cls, u, v, pads, tag=None, dtype='real'):

        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = 2*len(pads)
        tag  = tag or random_string( 6 )
        name = (u, v)
        return Basic.__new__(cls, pads, rank, name, tag, dtype)

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

    @property
    def dtype(self):
        return self._args[4]

#==============================================================================
class StencilMatrixGlobalBasis(MatrixNode):
    """
    used to describe local dof over an element as a stencil matrix
    """
    def __new__(cls, u, v, pads, tag=None, dtype='real'):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = 2*len(pads)
        tag  = tag or random_string( 6 )
        name = (u, v)
        return Basic.__new__(cls, pads, rank, name, tag, dtype)

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

    @property
    def dtype(self):
        return self._args[4]

#==============================================================================
class StencilVectorLocalBasis(MatrixNode):
    """
    used to describe local dof over an element as a stencil vector
    """
    def __new__(cls, v, pads, tag=None, dtype='real'):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = len(pads)
        tag  = tag or random_string( 6 )
        name = v
        return Basic.__new__(cls, pads, rank, name, tag, dtype)

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

    @property
    def dtype(self):
        return self._args[4]

#==============================================================================
class StencilVectorGlobalBasis(MatrixNode):
    """
    used to describe local dof over an element as a stencil vector
    """
    def __new__(cls, v, pads, tag=None, dtype='real'):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = len(pads)
        tag  = tag or random_string( 6 )
        name = v
        return Basic.__new__(cls, pads, rank, name, tag, dtype)

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

    @property
    def dtype(self):
        return self._args[4]


#==============================================================================
class LocalElementBasis(MatrixNode):
    tag  = random_string( 6 )

class GlobalElementBasis(MatrixNode):
    tag  = random_string( 6 )

#==============================================================================
class BlockStencilMatrixLocalBasis(BlockLinearOperatorNode):
    """
    used to describe local dof over an element as a block stencil matrix
    """
    def __new__(cls, trials, tests, expr, dim, tag=None, outer=None,
                     tests_degree=None, trials_degree=None,
                     tests_multiplicity=None, trials_multiplicity=None, dtype='real'):

        pads = Pads(tests, trials, tests_degree, trials_degree,
                    tests_multiplicity, trials_multiplicity)

        rank = 2*dim
        tag  = tag or random_string( 6 )
        obj  = Basic.__new__(cls, pads, rank, trials_multiplicity, tag, expr, dtype)
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
    def dtype(self):
        return self._args[5]

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
class BlockStencilMatrixGlobalBasis(BlockLinearOperatorNode):
    """
    used to describe local dof over an element as a block stencil matrix
    """
    def __new__(cls, trials, tests, pads, multiplicity, expr, tag=None, dtype='real'):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = 2*len(pads)
        tag  = tag or random_string( 6 )
        obj  = Basic.__new__(cls, pads, multiplicity, rank, tag, expr, dtype)
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
    def dtype(self):
        return self._args[5]

    @property
    def unique_scalar_space(self):
        types = (H1SpaceType, L2SpaceType, UndefinedSpaceType)
        spaces = self.trials.space
        cond = False
        for cls in types:
            cond = cond or all(isinstance(space.kind, cls) for space in spaces)
        return cond

#==============================================================================
class BlockStencilVectorLocalBasis(BlockLinearOperatorNode):
    """
    used to describe local dof over an element as a block stencil matrix
    """
    def __new__(cls,tests, pads, expr, tag=None, dtype='real'):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = len(pads)
        tag  = tag or random_string( 6 )
        obj  = Basic.__new__(cls, pads, rank, tag, expr, dtype)
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
    def dtype(self):
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
class BlockStencilVectorGlobalBasis(BlockLinearOperatorNode):
    """
    used to describe local dof over an element as a block stencil matrix
    """
    def __new__(cls, tests, pads, multiplicity, expr, tag=None, dtype='real'):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = len(pads)
        tag  = tag or random_string( 6 )
        obj  = Basic.__new__(cls, pads, multiplicity, rank, tag, expr, dtype)
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
    def dtype(self):
        return self._args[5]

    @property
    def unique_scalar_space(self):
        types = (H1SpaceType, L2SpaceType, UndefinedSpaceType)
        spaces = self._tests.space
        cond = False
        for cls in types:
            cond = cond or all(isinstance(space.kind, cls) for space in spaces)
        return cond

#==============================================================================
class ScalarLocalBasis(ScalarNode):
    """
     This is used to describe scalar dof over an element
    """
    def __new__(cls, u=None, v=None, tag=None, dtype='real'):
        tag  = tag or random_string( 6 )
        obj  = Basic.__new__(cls, tag, dtype)
        obj._test  = v
        obj._trial = u
        return obj

    @property
    def tag(self):
        return self._args[0]

    @property
    def dtype(self):
        return self._args[1]

    @property
    def trial(self):
        return self._trial

    @property
    def test(self):
        return self._test
#==============================================================================
class BlockScalarLocalBasis(ScalarNode):
    """
       This is used to describe a block of scalar dofs over an element
    """
    def __new__(cls, trials=None, tests=None, expr=None, tag=None, dtype='real'):

        tag = tag or random_string( 6 )
        obj = Basic.__new__(cls, tag, dtype)
        obj._tests  = tests
        obj._trials = trials
        obj._expr   = expr
        return obj

    @property
    def tag(self):
        return self._args[0]

    @property
    def dtype(self):
        return self._args[1]

    @property
    def tests(self):
        return self._tests

    @property
    def trials(self):
        return self._trials

    @property
    def expr(self):
        return self._expr

#==============================================================================
class SpanArray(ArrayNode):
    """
     This represents the global span array
    """

    def __new__(cls, target, index=None):
        if not isinstance(target, (ScalarFunction, VectorFunction, IndexedVectorFunction)):
            raise TypeError('Expecting a scalar/vector test function')

        return Basic.__new__(cls, target, index)

    @property
    def target(self):
        return self._args[0]

    @property
    def index(self):
        return self._args[1]

    def set_index(self, index):
        return type(self)(self.target, index)

#==============================================================================
class GlobalSpanArray(SpanArray):
    """
     This represents the global span array
    """
    _rank = 1
    _positions = {index_element: 0}

#==============================================================================
class LocalSpanArray(SpanArray):
    """
     This represents the local span array
    """
    _rank = 1
    _positions = {index_element: 0}
#==============================================================================
class GlobalThreadSpanArray(SpanArray):
    """
     This represents the global span array of each thread
    """
    _rank = 1
#==============================================================================
class GlobalThreadStarts(ArrayNode):
    """
     This represents the threads starts over the decomposed domain
    """
    _rank = 1
    def __new__(cls, index=None):
        # TODO check target
        return Basic.__new__(cls, index)

    @property
    def index(self):
        return self._args[0]

    def set_index(self, index):
        return GlobalThreadStarts(index)
 
#==============================================================================
class GlobalThreadEnds(ArrayNode):
    """
     This represents the threads ends over the decomposed domain
    """
    _rank = 1
    def __new__(cls, index=None):
        # TODO check target
        return Basic.__new__(cls, index)

    @property
    def index(self):
        return self._args[0]

    def set_index(self, index):
        return GlobalThreadEnds(index)

#==============================================================================
class GlobalThreadSizes(ArrayNode):
    """
     This represents the number of elements owned by a thread
    """
    _rank = 1
    def __new__(cls, index=None):
        # TODO check target
        return Basic.__new__(cls, index)

    @property
    def index(self):
        return self._args[0]

    def set_index(self, index):
        return GlobalThreadSizes(index)

#==============================================================================
class LocalThreadStarts(ArrayNode):
    """
     This represents the local threads starts over the decomposed domain
    """
    _rank = 1
    def __new__(cls, index=None):
        # TODO check target
        return Basic.__new__(cls, index)

    @property
    def index(self):
        return self._args[0]

    def set_index(self, index):
        return LocalThreadStarts(index)

#==============================================================================
class LocalThreadEnds(ArrayNode):
    """
     This represents the local threads ends over the decomposed domain
    """
    _rank = 1
    def __new__(cls, index=None):
        # TODO check target
        return Basic.__new__(cls, index)

    @property
    def index(self):
        return self._args[0]

    def set_index(self, index):
        return LocalThreadEnds(index)

#==============================================================================
class Span(ScalarNode):
    """
     This represents the span of a basis in an element
    """
    def __new__(cls, target, index=None):
        if not isinstance(target, (ScalarFunction, VectorFunction, IndexedVectorFunction)):
            raise TypeError('Expecting a scalar/vector test function')

        return Basic.__new__(cls, target, index)

    @property
    def target(self):
        return self._args[0]

    @property
    def index(self):
        return self._args[1]

    def set_index(self, index):
        return Span(self.target, index)

class Pads(ScalarNode):
    """
     This represents the global pads
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
    def __new__(cls, expr, weights=True):
        return Basic.__new__(cls, expr, weights)

    @property
    def expr(self):
        return self._args[0]

    @property
    def weights(self):
        return self._args[1]
#==============================================================================
class ComputeLogical(ComputeNode):
    """
    """
    pass

#==============================================================================
class ComputeLogicalBasis(ComputeLogical):
    """
    """
    def __new__(cls, expr, *, prefix=None):

        if not isinstance(prefix, (str, type(None))):
            raise TypeError('expecting a prefix as a string')
        return Basic.__new__(cls, expr, prefix)

    @property
    def expr(self):
        return self._args[0]

    @property
    def prefix(self):
        return self._args[1]

#==============================================================================
class Reduction(Basic):
    """
    """
    def __new__(cls, op, expr, lhs=None):
        if not op in ['-', '+', '*', '/']:
            raise TypeError("Expecting an operation type in : '-', '+', '*', '/'")
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
        if not op in ['-', '+', '*', '/']:
            raise TypeError("Expecting an operation type in : '-', '+', '*', '/'")
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
    """
    """
    def __new__(cls, expr, *, prefix=None):
        if not isinstance(prefix, (str, type(None))):
            raise TypeError('expecting a prefix as a string')
        return Basic.__new__(cls, expr, prefix)

    @property
    def expr(self):
        return self._args[0]

    @property
    def prefix(self):
        return self._args[1]

#==============================================================================
class PhysicalGeometryValue(PhysicalValueNode):
    pass

#==============================================================================
class LogicalGeometryValue(LogicalValueNode):
    pass

#==============================================================================
class BasisAtom(AtomicNode):
    """
    Used to describe a temporary for the basis coefficient or in the kernel.
    """
    def __new__(cls, expr, *, prefix=None):
        types = (IndexedVectorFunction, VectorFunction, ScalarFunction)

        ls = _atomic(expr, cls=types)
        if not(len(ls) == 1):
            raise ValueError('Expecting an expression with one test function')

        u = ls[0]

        if not isinstance(prefix, (str, type(None))):
            raise TypeError('expecting a prefix as a string')

        obj = Basic.__new__(cls, expr, prefix)
        obj._atom = u
        return obj

    @property
    def expr(self):
        return self._args[0]

    @property
    def prefix(self):
        return self._args[1]

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
    def __new__(cls, expr, dtype='real'):
        # TODO assert on expr
        atom = GeometryAtom(expr)
        expr = MatrixQuadrature(expr, dtype)

        return Basic.__new__(cls, atom, expr)

    @property
    def atom(self):
        return self._args[0]

    @property
    def expr(self):
        return self._args[1]

#==============================================================================
class IfNode(BaseNode):
    def __new__(cls, *args):
        args = tuple(args)
        return Basic.__new__(cls, args)

    @property
    def args(self):
        return self._args[0]

#==============================================================================
class WhileLoop(BaseNode):
    def __new__(cls, condition, body):
        body = tuple(body)
        return Basic.__new__(cls, condition, body)

    @property
    def condition(self):
        return self._args[0]

    @property
    def body(self):
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

    parallel : <bool|None>
        specifies whether the loop should be executed in parallel or in serial

    default: <str|None>
        specifies the default behavior of the variables in a parallel region

    shared : <list|tuple|None>
        specifies the shared variables in the parallel region

    private: <list|tuple|None>
        specifies the private variables in the parallel region

    firstprivate: <list|tuple|None>
        specifies the first private variables in the parallel region

    lastprivate: <list|tuple|None>
        specifies the last private variables in the parallel region
    """

    def __new__(cls, iterable, index, *, stmts=None, mask=None,
                parallel=None, default=None, shared=None,
                private=None, firstprivate=None, lastprivate=None,
                reduction=None, prefix=None):
        # ...
        if not( isinstance(iterable, (list, tuple, Tuple)) ):
            iterable = [iterable]

        iterable = Tuple(*iterable)
        # ...

        # ... replace GeometryExpressions by a list of expressions
        others = [i for i in iterable if not isinstance(i, GeometryExpressions)]
        geos   = [i.expressions for i in iterable if isinstance(i, GeometryExpressions)]

        iterable = others + flatten(geos)
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
            i,g = construct_itergener(a, index, prefix=prefix)
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

        obj._iterator     = iterator
        obj._generator    = generator
        obj._parallel     = parallel
        obj._default      = default
        obj._shared       = shared
        obj._private      = private
        obj._firstprivate = firstprivate
        obj._lastprivate  = lastprivate
        obj._reduction    = reduction

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

    @property
    def parallel(self):
        return self._parallel

    @property
    def default(self):
        return self._default

    @property
    def shared(self):
        return self._shared

    @property
    def private(self):
        return self._private

    @property
    def firstprivate(self):
        return self._firstprivate

    @property
    def lastprivate(self):
        return self._lastprivate

    @property
    def reduction(self):
        return self._reduction

    def get_geometry_stmts(self, mapping):

        l_quad = list(self.generator.atoms(LocalTensorQuadratureGrid))
        if len(l_quad) == 0:
            return Tuple()

        l_quad = l_quad[0]
        args   = []
        if l_quad.weights:
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
def construct_logical_expressions(u, nderiv, *, prefix=None):
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
    return [ComputeLogicalBasis(i, prefix=prefix) for i in args]

#==============================================================================
class GeometryExpressions(Basic):
    """
    """
    def __new__(cls, M, nderiv, dtype='real'):
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

            expressions = [GeometryExpr(i, dtype) for i in args]

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
def construct_itergener(a, index, *, prefix=None):
    """
    Create the generator and the iterator based on a and the index
    """
    # ... create generator
    if isinstance(a, PlusGlobalTensorQuadratureGrid):
        generator = TensorGenerator(a, index)
        element   = PlusLocalTensorQuadratureGrid()

    elif isinstance(a, PlusLocalTensorQuadratureGrid):
        generator = TensorGenerator(a, index)
        element   = PlusTensorQuadrature()

    elif isinstance(a, GlobalTensorQuadratureGrid):
        generator = TensorGenerator(a, index)
        element   = LocalTensorQuadratureGrid()

    elif isinstance(a, LocalTensorQuadratureGrid):
        generator = TensorGenerator(a, index)
        element   = TensorQuadrature(a.weights)

    elif isinstance(a, (LocalTensorQuadratureTrialBasis, GlobalTensorQuadratureTrialBasis)):
        generator = TensorGenerator(a, index)
        element   = TensorTrialBasis(a.target)

    elif isinstance(a, (LocalTensorQuadratureTestBasis, GlobalTensorQuadratureTestBasis)):
        generator = TensorGenerator(a, index)
        element   = TensorTestBasis(a.target)

    elif isinstance(a, (GlobalTensorQuadratureBasis, TensorQuadratureBasis)):
        generator = TensorGenerator(a, index)
        element   = TensorBasis(a.target)

    elif isinstance(a, (LocalSpanArray, GlobalSpanArray)):
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
        generator = TensorGenerator(a.rhs, index)
        element   = a.lhs

    else:
        raise TypeError('{} not available'.format(type(a)))
    # ...

    # ... create iterator
    tensor_classes = (LocalTensorQuadratureGrid,
                      TensorQuadrature,
                      LocalTensorQuadratureBasis,
                      TensorQuadratureBasis,
                      TensorBasis,
                      Span)

    product_classes = (CoefficientBasis,
                       GeometryAtom,
                       MatrixLocalBasis)

    if isinstance(element, tensor_classes):
        iterator = TensorIterator(element, prefix=prefix)

    elif isinstance(element, product_classes):
        iterator = ProductIterator(element, prefix=prefix)

    elif isinstance(element, (Expr, Tuple)):
        iterator = TensorIterator(element, prefix=prefix)

    else:
        raise TypeError('{} not available'.format(type(element)))
    # ...

    return iterator, generator

#=============================================================================================
# the Expression class works with fixed dimension expressions instead of vectorized one,
# where in some cases we need to treat each dimesion diffrently

class Expression(Expr):
    """
    The Expression class gives us the possibility to create specific instructions for some dimension,
    where the generated code is not in a vectorized form.
    For example, the class Loop generates 2 for loops in 2D and 3 in 3D,
    the expressions that are generated are the same for 2D and 3D,
    because they are written in a way that allows them to be applied in any dimension,
    with the fixed dimension expression we can specify the generated code for a specific dimension,
    so the generated code in the second dimension of the 2D loop is diffrent from the one in the first dimension of the 2D loop
    """
    def __new__(cls, *args):
        return Expr.__new__(cls, *args)

class AddNode(Expression):
    pass

class MulNode(Expression):
    pass

class IntDivNode(Expression):
    pass

class AndNode(Expression):
    pass

class NotNode(Expression):
    pass

class EqNode(Expression):
    pass

class StrictLessThanNode(Expression):
    pass
