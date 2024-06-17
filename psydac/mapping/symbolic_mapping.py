# coding: utf-8
import numpy as np 

from sympy                 import Indexed, IndexedBase, Idx
from sympy                 import Matrix, ImmutableDenseMatrix
from sympy                 import Function, Expr
from sympy                 import sympify
from sympy                 import cacheit
from sympy.core            import Basic
from sympy.core            import Symbol,Integer
from sympy.core            import Add, Mul, Pow
from sympy.core.numbers    import ImaginaryUnit
from sympy.core.containers import Tuple
from sympy                 import S
from sympy                 import sqrt, symbols
from sympy.core.exprtools  import factor_terms
from sympy.polys.polytools import parallel_poly_from_expr

from sympde.core              import Constant
from sympde.core.basic        import BasicMapping
from sympde.core.basic        import CalculusFunction
from sympde.core.basic        import _coeffs_registery
from sympde.calculus.core     import PlusInterfaceOperator, MinusInterfaceOperator
from sympde.calculus.core     import grad, div, curl, laplace #, hessian
from sympde.calculus.core     import dot, inner, outer, _diff_ops
from sympde.calculus.core     import has, DiffOperator
from sympde.calculus.matrices import MatrixSymbolicExpr, MatrixElement, SymbolicTrace, Inverse
from sympde.calculus.matrices import SymbolicDeterminant, Transpose

from sympde.topology.basic       import BasicDomain, Union, InteriorDomain
from sympde.topology.basic       import Boundary, Connectivity, Interface
from sympde.topology.domain      import Domain, NCubeInterior
from sympde.topology.domain      import NormalVector
from sympde.topology.space       import ScalarFunction, VectorFunction, IndexedVectorFunction
from sympde.topology.space       import Trace
from sympde.topology.datatype    import HcurlSpaceType, H1SpaceType, L2SpaceType, HdivSpaceType, UndefinedSpaceType
from sympde.topology.derivatives import dx, dy, dz, DifferentialOperator
from sympde.topology.derivatives import _partial_derivatives
from sympde.topology.derivatives import get_atom_derivatives, get_index_derivatives_atom
from sympde.topology.derivatives import _logical_partial_derivatives
from sympde.topology.derivatives import get_atom_logical_derivatives, get_index_logical_derivatives_atom
from sympde.topology.derivatives import LogicalGrad_1d, LogicalGrad_2d, LogicalGrad_3d
from sympde.utilities.utils import lambdify_sympde

from abstract_mapping import AbstractMapping

# TODO fix circular dependency between sympde.topology.domain and sympde.topology.mapping
# TODO fix circular dependency between sympde.expr.evaluation and sympde.topology.mapping

__all__ = (
    'AnalyticalMapping',
    'Contravariant',
    'Covariant',
    'InterfaceMapping',
    'InverseMapping',
    'Jacobian',
    'JacobianInverseSymbol',
    'JacobianSymbol',
    'LogicalExpr',
    'MappedDomain',
    'MappingApplication',
    'MultiPatchMapping',
    'PullBack',
    'SymbolicExpr',
    'SymbolicWeightedVolume',
    'get_logical_test_function',
)

#==============================================================================
@cacheit
def cancel(f):
    try:
        f           = factor_terms(f, radical=True)
        p, q        = f.as_numer_denom()
        # TODO accelerate parallel_poly_from_expr
        (p, q), opt = parallel_poly_from_expr((p,q))
        c, P, Q     = p.cancel(q)
        return c*(P.as_expr()/Q.as_expr())
    except:
        return f

def get_logical_test_function(u):
    space           = u.space
    kind            = space.kind
    dim             = space.ldim
    logical_domain  = space.domain.logical_domain
    l_space         = type(space)(space.name, logical_domain, kind=kind)
    el              = l_space.element(u.name)
    return el


#==============================================================================
class AnalyticMapping(BasicMapping,AbstractMapping):
    """
    Represents a AnalyticMapping object.

    Examples

    """
    _expressions  = None # used for analytical mapping
    _jac          = None
    _inv_jac      = None
    _constants    = None
    _callable_map = None
    _ldim         = None
    _pdim         = None

    def __new__(cls, name, dim=None, **kwargs):

        ldim        = kwargs.pop('ldim', cls._ldim)
        pdim        = kwargs.pop('pdim', cls._pdim)
        coordinates = kwargs.pop('coordinates', None)
        evaluate    = kwargs.pop('evaluate', True)

        dims = [dim, ldim, pdim]
        for i,d in enumerate(dims):
            if isinstance(d, (tuple, list, Tuple, Matrix, ImmutableDenseMatrix)):
                if not len(d) == 1:
                    raise ValueError('> Expecting a tuple, list, Tuple of length 1')
                dims[i] = d[0]

        dim, ldim, pdim = dims

        if dim is None:
            assert ldim is not None
            assert pdim is not None
            assert pdim >= ldim
        else:
            ldim = dim
            pdim = dim


        obj = IndexedBase.__new__(cls, name, shape=pdim)

        if not evaluate:
            return obj

        if coordinates is None:
            _coordinates = [Symbol(name) for name in ['x', 'y', 'z'][:pdim]]
        else:
            if not isinstance(coordinates, (list, tuple, Tuple)):
                raise TypeError('> Expecting list, tuple, Tuple')

            for a in coordinates:
                if not isinstance(a, (str, Symbol)):
                    raise TypeError('> Expecting str or Symbol')

            _coordinates = [Symbol(u) for u in coordinates]

        obj._name                = name
        obj._ldim                = ldim
        obj._pdim                = pdim
        obj._coordinates         = tuple(_coordinates)
        obj._jacobian            = kwargs.pop('jacobian', JacobianSymbol(obj))
        obj._is_minus            = None
        obj._is_plus             = None

        lcoords = ['x1', 'x2', 'x3'][:ldim]
        lcoords = [Symbol(i) for i in lcoords]
        obj._logical_coordinates = Tuple(*lcoords)
        # ...
        if not( obj._expressions is None ):
            coords = ['x', 'y', 'z'][:pdim]

            # ...
            args = []
            for i in coords:
                x = obj._expressions[i]
                x = sympify(x)
                args.append(x)

            args = Tuple(*args)
            # ...
            zero_coords = ['x1', 'x2', 'x3'][ldim:]

            for i in zero_coords:
                x = sympify(i)
                args = args.subs(x,0)
            # ...

            constants        = list(set(args.free_symbols) - set(lcoords))
            constants_values = {a.name:Constant(a.name) for a in constants}
            # subs constants as Constant objects instead of Symbol
            constants_values.update( kwargs )
            d = {a:constants_values[a.name] for a in constants}
            args = args.subs(d)

            obj._expressions = args
            obj._constants   = tuple(a for a in constants if isinstance(constants_values[a.name], Symbol))

            args  = [obj[i] for i in range(pdim)]
            exprs = obj._expressions
            subs  = list(zip(_coordinates, exprs))

            if obj._jac is None and obj._inv_jac is None:
                obj._jac     = Jacobian(obj).subs(list(zip(args, exprs)))
                obj._inv_jac = obj._jac.inv() if pdim == ldim else None
            elif obj._inv_jac is None:
                obj._jac     = ImmutableDenseMatrix(sympify(obj._jac)).subs(subs)
                obj._inv_jac = obj._jac.inv() if pdim == ldim else None

            elif obj._jac is None:
                obj._inv_jac = ImmutableDenseMatrix(sympify(obj._inv_jac)).subs(subs)
                obj._jac     = obj._inv_jac.inv()
            else:
                obj._jac     = ImmutableDenseMatrix(sympify(obj._jac)).subs(subs)
                obj._inv_jac = ImmutableDenseMatrix(sympify(obj._inv_jac)).subs(subs)

        else:
            obj._jac     = Jacobian(obj)

        obj._metric     = obj._jac.T*obj._jac
        obj._metric_det = obj._metric.det()

        return obj

    
    #--------------------------------------------------------------------------
    #Abstract Interface : 
    
    @property
    def name( self ):
        return self._name

    @property
    def ldim( self ):
        return self._ldim

    @property
    def pdim( self ):
        return self._pdim
    
    def _evaluate_domain( self, domain ):
        assert(isinstance(domain, BasicDomain))
        return MappedDomain(self, domain)
    
    def _evaluate_point( self, *eta ):
        variables = self._logical_coordinates
        expressions = self._expressions
        func_eval = tuple(lambdify_sympde( variables, expr) for expr in expressions)
        return tuple( f( *eta ) for f in func_eval)
    
    def _evaluate_1d_arrays(self, X, Y):
        if X.shape != Y.shape:
            raise ValueError("Shape mismatch between 1D arrays")
        
        result_X = np.zeros_like(X, dtype=np.float64)
        result_Y = np.zeros_like(Y, dtype=np.float64)
        
        for i in range(X.shape[0]):
            result_X[i], result_Y[i] = self._evaluate_point(X[i], Y[i])
       
        return result_X, result_Y
    
    def _evaluate_meshgrid(self, *args):
        if len(args) != 2:
            raise ValueError("Expected two arrays for meshgrid evaluation")
        
        X, Y = args
        if X.shape != Y.shape:
            raise ValueError("Shape mismatch between meshgrid arrays")
        
        # Create empty arrays to store results
        result_X = np.zeros_like(X, dtype=np.float64)
        result_Y = np.zeros_like(Y, dtype=np.float64)
        
        # Iterate over the meshgrid points and evaluate the mapping
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                result_X[i, j], result_Y[i, j] = self._evaluate_point(X[i, j], Y[i, j])
        
        return result_X, result_Y
    
    def __call__( self, *args ):
        if len(args) == 1 and isinstance(args[0], BasicDomain):
            return self._evaluate_domain(args[0])
        elif all(isinstance(arg, (int, float, Symbol)) for arg in args):
            return self._evaluate_point(*args)
        elif all(isinstance(arg, np.ndarray) for arg in args):
            if (arg.shape==1 for arg in args):
                return self._evaluate_1d_arrays(*args)
            elif (arg.shape==2 for arg in args):
                return self._evaluate_meshgrid(*args)
            else :
                raise TypeError("Invalid dimension for called object")
        else:
            raise TypeError("Invalid arguments for __call__")

    def jacobian_eval( self, *eta ):
        variables = self._logical_coordinates
        jac = self._jac 
        jac_eval = lambdify_sympde( variables, jac)
        return jac_eval( *eta )
        
    def jacobian_inv_eval( self, *eta ):
        variables = self._logical_coordinates
        inv_jac = self._inv_jac
        inv_jac_eval = lambdify_sympde( variables, inv_jac)
        return inv_jac_eval( *eta )
    
    def metric_eval( self, *eta ):
        variables = self._logical_coordinates
        metric = self._metric 
        metric_eval = lambdify_sympde( variables, metric)
        return metric_eval( *eta )
    
    def metric_det_eval( self, *eta ):
        variables = self._logical_coordinates
        metric_det = self._metric_det 
        metric_det_eval = lambdify_sympde( variables, metric_det)
        return metric_det_eval( *eta )   

#--------------------------------------------------------------------------

    @property
    def coordinates( self ):
        if self.pdim == 1:
            return self._coordinates[0]
        else:
            return self._coordinates

    @property
    def logical_coordinates( self ):
        if self.ldim == 1:
            return self._logical_coordinates[0]
        else:
            return self._logical_coordinates

    @property
    def jacobian( self ):
        return self._jacobian

    @property
    def det_jacobian( self ):
        return self.jacobian.det()

    @property
    def is_analytical( self ):
        return not( self._expressions is None )

    @property
    def expressions( self ):
        return self._expressions

    @property
    def jacobian_expr( self ):
        return self._jac

    @property
    def jacobian_inv_expr( self ):
        if not self.is_analytical and self._inv_jac is None:
            self._inv_jac = self.jacobian_expr.inv()
        return self._inv_jac

    @property
    def metric_expr( self ):
        return self._metric

    @property
    def metric_det_expr( self ):
        return self._metric_det

    @property
    def constants( self ):
        return self._constants

    @property
    def is_minus( self ):
        return self._is_minus

    @property
    def is_plus( self ):
        return self._is_plus

    def set_plus_minus( self, **kwargs):
        minus = kwargs.pop('minus', False)
        plus  = kwargs.pop('plus', False)
        assert plus is not minus

        self._is_plus  = plus
        self._is_minus = minus

    def copy(self):
        obj = AnalyticMapping(self.name,
                     ldim=self.ldim,
                     pdim=self.pdim,
                     evaluate=False)

        obj._name                = self.name
        obj._ldim                = self.ldim
        obj._pdim                = self.pdim
        obj._coordinates         = self.coordinates
        obj._jacobian            = JacobianSymbol(obj)
        obj._logical_coordinates = self.logical_coordinates
        obj._expressions         = self._expressions
        obj._constants           = self._constants
        obj._jac                 = self._jac
        obj._inv_jac             = self._inv_jac
        obj._metric              = self._metric
        obj._metric_det          = self._metric_det
        obj.__callable_map       = self._callable_map
        obj._is_plus             = self._is_plus
        obj._is_minus            = self._is_minus
        return obj

    def _hashable_content(self):
        args = (self.name, self.ldim, self.pdim, self._coordinates, self._logical_coordinates,
                self._expressions, self._constants, self._is_plus, self._is_minus)
        return tuple([a for a in args if a is not None])

    def _eval_subs(self, old, new):
        return self

    def _sympystr(self, printer):
        sstr = printer.doprint
        return sstr(self.name)
    
    
#==============================================================================
class InverseMapping(AnalyticMapping):
    def __new__(cls, mapping):
        assert isinstance(mapping, AnalyticMapping)
        name     = mapping.name
        ldim     = mapping.ldim
        pdim     = mapping.pdim
        coords   = mapping.logical_coordinates
        jacobian = mapping.jacobian.inv()
        return AnalyticMapping.__new__(cls, name, ldim=ldim, pdim=pdim, coordinates=coords, jacobian=jacobian)

#==============================================================================
class JacobianSymbol(MatrixSymbolicExpr):
    _axis = None
    def __new__(cls, mapping, axis=None):
        assert isinstance(mapping, AnalyticMapping)
        if axis is not None:
            assert isinstance(axis, (int, Integer))
        obj = MatrixSymbolicExpr.__new__(cls, mapping)
        obj._axis = axis
        return obj

    @property
    def mapping(self):
        return self._args[0]

    @property
    def axis(self):
        return self._axis

    def inv(self):
        return JacobianInverseSymbol(self.mapping, self.axis)

    def _hashable_content(self):
        if self.axis is not None:
            return (type(self).__name__, self.mapping, self.axis)
        else:
            return (type(self).__name__, self.mapping)

    def __hash__(self):
        return hash(self._hashable_content())

    def _eval_subs(self, old, new):
        if isinstance(new, AnalyticMapping):
            if self.axis is not None:
                obj = JacobianSymbol(new, self.axis)
            else:
                obj = JacobianSymbol(new)
            return obj
        return self
    def _sympystr(self, printer):
        sstr = printer.doprint
        if self.axis:
            return 'Jacobian({},{})'.format(sstr(self.mapping.name), self.axis)
        else:
            return 'Jacobian({})'.format(sstr(self.mapping.name))

#==============================================================================
class JacobianInverseSymbol(MatrixSymbolicExpr):
    _axis = None
    is_Matrix     = False
    def __new__(cls, mapping, axis=None):
        assert isinstance(mapping, AnalyticMapping)
        if axis is not None:
            assert isinstance(axis, int)
        obj = MatrixSymbolicExpr.__new__(cls, mapping)
        obj._axis = axis
        return obj

    @property
    def mapping(self):
        return self._args[0]

    @property
    def axis(self):
        return self._axis

    def _hashable_content(self):
        if self.axis is not None:
            return (type(self).__name__, self.mapping, self.axis)
        else:
            return (type(self).__name__, self.mapping)

    def __hash__(self):
        return hash(self._hashable_content())

    def _sympystr(self, printer):
        sstr = printer.doprint
        if self.axis:
            return 'Jacobian({},{})**(-1)'.format(sstr(self.mapping.name), self.axis)
        else:
            return 'Jacobian({})**(-1)'.format(sstr(self.mapping.name))

#==============================================================================
class InterfaceMapping(AnalyticMapping):
    """
    InterfaceMapping is used to represent a mapping in the interface.

    Attributes
    ----------
    minus : AnalyticMapping
        the mapping on the negative direction of the interface
    plus  : AnalyticMapping
        the mapping on the positive direction of the interface
    """

    def __new__(cls, minus, plus):
        assert isinstance(minus, AnalyticMapping)
        assert isinstance(plus,  AnalyticMapping)
        minus = minus.copy()
        plus  = plus.copy()

        minus.set_plus_minus(minus=True)
        plus.set_plus_minus(plus=True)

        name = '{}|{}'.format(str(minus.name), str(plus.name))
        obj  = AnalyticMapping.__new__(cls, name, ldim=minus.ldim, pdim=minus.pdim)
        obj._minus = minus
        obj._plus  = plus
        return obj

    @property
    def minus(self):
        return self._minus

    @property
    def plus(self):
        return self._plus

    @property
    def is_analytical(self):
        return self.minus.is_analytical and self.plus.is_analytical

    def _eval_subs(self, old, new):
        minus = self.minus.subs(old, new)
        plus  = self.plus.subs(old, new)
        return InterfaceMapping(minus, plus)

    def _eval_simplify(self, **kwargs):
        return self

#==============================================================================
class MultiPatchMapping(AnalyticMapping):

    def __new__(cls, dic):
        assert isinstance( dic, dict)
        return Basic.__new__(cls, dic)

    @property
    def mappings(self):
        return self.args[0]

    @property
    def is_analytical(self):
        return all(a.is_analytical for a in self.mappings.values())

    @property
    def ldim(self):
        return list(self.mappings.values())[0].ldim

    @property
    def pdim(self):
        return list(self.mappings.values())[0].pdim

    @property
    def is_analytical(self):
        return all(e.is_analytical for e in self.mappings.values())

    def _eval_subs(self, old, new):
        return self

    def _eval_simplify(self, **kwargs):
        return self

    def __hash__(self):
        return hash((*self.mappings.values(), *self.mappings.keys()))

    def _sympystr(self, printer):
        sstr = printer.doprint
        mappings = (sstr(i) for i in self.mappings.values())
        return 'MultiPatchMapping({})'.format(', '.join(mappings))

#==============================================================================
class MappedDomain(BasicDomain):
    """."""

    #@cacheit
    def __new__(cls, mapping, logical_domain):
        assert(isinstance(mapping,AbstractMapping))
        assert(isinstance(logical_domain, BasicDomain))
        if isinstance(logical_domain, Domain):
            kwargs = dict(
            dim            = logical_domain._dim,
            mapping        = mapping,
            logical_domain = logical_domain)
            boundaries     = logical_domain.boundary
            interiors      = logical_domain.interior

            if isinstance(interiors, Union):
                kwargs['interiors'] = Union(*[mapping(a) for a in interiors.args])
            else:
                kwargs['interiors'] = mapping(interiors)

            if isinstance(boundaries, Union):
                kwargs['boundaries'] = [mapping(a) for a in boundaries.args]
            elif boundaries:
                kwargs['boundaries'] = mapping(boundaries)

            interfaces =  logical_domain.connectivity.interfaces
            if interfaces:
                if isinstance(interfaces, Union):
                    interfaces = interfaces.args
                else:
                    interfaces = [interfaces]
                connectivity = {}
                for e in interfaces:
                    connectivity[e.name] = Interface(e.name, mapping(e.minus), mapping(e.plus))
                kwargs['connectivity'] = Connectivity(connectivity)

            name = '{}({})'.format(str(mapping.name), str(logical_domain.name))
            return Domain(name, **kwargs)

        elif isinstance(logical_domain, NCubeInterior):
            name  = logical_domain.name
            dim   = logical_domain.dim
            dtype = logical_domain.dtype
            min_coords = logical_domain.min_coords
            max_coords = logical_domain.max_coords
            name = '{}({})'.format(str(mapping.name), str(name))
            return NCubeInterior(name, dim, dtype, min_coords, max_coords, mapping, logical_domain)
        elif isinstance(logical_domain, InteriorDomain):
            name  = logical_domain.name
            dim   = logical_domain.dim
            dtype = logical_domain.dtype
            name = '{}({})'.format(str(mapping.name), str(name))
            return InteriorDomain(name, dim, dtype, mapping, logical_domain)
        elif isinstance(logical_domain, Boundary):
            name   = logical_domain.name
            axis   = logical_domain.axis
            ext    = logical_domain.ext
            domain = mapping(logical_domain.domain)
            return Boundary(name, domain, axis, ext, mapping, logical_domain)
        else:
            raise NotImplementedError('TODO')
#==============================================================================
class SymbolicWeightedVolume(Expr):
    """
    This class represents the symbolic weighted volume of a quadrature rule
    """
#TODO move this somewhere else
#==============================================================================
class MappingApplication(Function):
    nargs = None

    def __new__(cls, *args, **options):

        if options.pop('evaluate', True):
            r = cls.eval(*args)
        else:
            r = None

        if r is None:
            return Basic.__new__(cls, *args, **options)
        else:
            return r

class PullBack(Expr):
    is_commutative = False

    def __new__(cls, u, mapping=None):
        if not isinstance(u, (VectorFunction, ScalarFunction)):
            raise TypeError('{} must be of type ScalarFunction or VectorFunction'.format(str(u)))

        if u.space.domain.mapping is None:
            raise ValueError('The pull-back can be performed only to mapped domains')

        space = u.space
        kind  = space.kind
        dim   = space.ldim
        el    = get_logical_test_function(u)

        if space.is_broken:
            assert mapping is not None
        else:
            mapping = space.domain.mapping

        J = mapping.jacobian
        if isinstance(kind, (UndefinedSpaceType, H1SpaceType)):
            expr = el

        elif isinstance(kind, HcurlSpaceType):
            expr = J.inv().T * el

        elif isinstance(kind, HdivSpaceType):
            expr = (J/J.det()) * el

        elif isinstance(kind, L2SpaceType):
            expr = el/J.det()

#        elif isinstance(kind, UndefinedSpaceType):
#            raise ValueError('kind must be specified in order to perform the pull-back transformation')
        else:
            raise ValueError("Unrecognized kind '{}' of space {}".format(kind, str(u.space)))

        obj       = Expr.__new__(cls, u)
        obj._expr = expr
        obj._kind = kind
        obj._test = el
        return obj

    @property
    def expr(self):
        return self._expr

    @property
    def kind(self):
        return self._kind

    @property
    def test(self):
        return self._test

#==============================================================================
class Jacobian(MappingApplication):
    r"""
    This class calculates the Jacobian of a mapping F
    where [J_{F}]_{i,j} =  \frac{\partial F_{i}}{\partial x_{j}}
    or simply J_{F} =  (\nabla F)^T

    """

    @classmethod
    def eval(cls, F):
        """
        this class methods computes the jacobian of a mapping

        Parameters:
        ----------
         F: AnalyticMapping
            mapping object

        Returns:
        ----------
         expr : ImmutableDenseMatrix
            the jacobian matrix
        """

        if not isinstance(F, AnalyticMapping):
            raise TypeError('> Expecting a AnalyticMapping object')

        if F.jacobian_expr is not None:
            return F.jacobian_expr

        pdim = F.pdim
        ldim = F.ldim

        F = [F[i] for i in range(0, F.pdim)]
        F = Tuple(*F)

        if ldim == 1:
            expr = LogicalGrad_1d(F)

        elif ldim == 2:
            expr = LogicalGrad_2d(F)

        elif ldim == 3:
            expr = LogicalGrad_3d(F)

        return expr.T

#==============================================================================
class Covariant(MappingApplication):
    """

    Examples

    """

    @classmethod
    def eval(cls, F, v):

        """
        This class methods computes the covariant transformation

        Parameters:
        ----------
         F: AnalyticMapping
            mapping object

         v: <tuple|list|Tuple|ImmutableDenseMatrix|Matrix>
            the basis function

        Returns:
        ----------
         expr : Tuple
            the covariant transformation
        """

        if not isinstance(v, (tuple, list, Tuple, ImmutableDenseMatrix, Matrix)):
            raise TypeError('> Expecting a tuple, list, Tuple, Matrix')

        assert F.pdim == F.ldim

        M   = Jacobian(F).inv().T
        dim = F.pdim

        if dim == 1:
            b = M[0,0] * v[0]
            return Tuple(b)
        else:
            n,m = M.shape
            w   = []
            for i in range(0, n):
                w.append(S.Zero)

            for i in range(0, n):
                for j in range(0, m):
                    w[i] += M[i,j] * v[j]
            return Tuple(*w)

#==============================================================================
class Contravariant(MappingApplication):
    """

    Examples

    """

    @classmethod
    def eval(cls, F, v):
        """
        This class methods computes the contravariant transformation

        Parameters:
        ----------
         F: AnalyticMapping
            mapping object

         v: <tuple|list|Tuple|ImmutableDenseMatrix|Matrix>
            the basis function

        Returns:
        ----------
         expr : Tuple
            the contravariant transformation
        """

        if not isinstance(F, AnalyticMapping):
            raise TypeError('> Expecting a AnalyticMapping')

        if not isinstance(v, (tuple, list, Tuple, ImmutableDenseMatrix, Matrix)):
            raise TypeError('> Expecting a tuple, list, Tuple, Matrix')

        M = Jacobian(F)
        M = M/M.det()
        v = Matrix(v)
        v = M*v
        return Tuple(*v)

#==============================================================================
class LogicalExpr(CalculusFunction):

    def __new__(cls, expr, domain, **options):
        # (Try to) sympify args first

        if options.pop('evaluate', True):
            r = cls.eval(expr, domain, **options)
        else:
            r = None

        if r is None:
            obj = Basic.__new__(cls, expr, domain)
            return obj
        else:
            return r

    @property
    def expr(self):
        return self._args[0]

    @property
    def domain(self):
        return self._args[1]

    def __getitem__(self, indices, **kw_args):
        if is_sequence(indices):
            # Special case needed because M[*my_tuple] is a syntax error.
            return Indexed(self, *indices, **kw_args)
        else:
            return Indexed(self, indices, **kw_args)

    @classmethod
    def eval(cls, expr, domain, **options):
        """."""

        from sympde.expr.evaluation import TerminalExpr, DomainExpression
        from sympde.expr.expr import BilinearForm, LinearForm, BasicForm, Norm
        from sympde.expr.expr import Integral

        types = (ScalarFunction, VectorFunction, DifferentialOperator, Trace, Integral)

        mapping   = domain.mapping
        dim       = domain.dim
        assert mapping

        # TODO this is not the dim of the domain
        l_coords  = ['x1', 'x2', 'x3'][:dim]
        ph_coords = ['x', 'y', 'z']

        if not has(expr, types):
            if has(expr, DiffOperator):
                return cls( expr, domain, evaluate=False)
            else:
                syms = symbols(ph_coords[:dim])
                if isinstance(mapping, InterfaceMapping):
                    mapping = mapping.minus
                    # here we assume that the two mapped domains
                    # are identical in the interface so we choose one of them
                Ms   = [mapping[i] for i in range(dim)]
                expr = expr.subs(list(zip(syms, Ms)))

                if mapping.is_analytical:
                    expr = expr.subs(list(zip(Ms, mapping.expressions)))
                return expr

        if isinstance(expr, Symbol) and expr.name in l_coords:
            return expr

        if isinstance(expr, Symbol) and expr.name in ph_coords:
            return mapping[ph_coords.index(expr.name)]

        elif isinstance(expr, Add):
            args = [cls.eval(a, domain) for a in expr.args]
            v    =  S.Zero
            for i in args:
                v += i
            n,d = v.as_numer_denom()
            return n/d

        elif isinstance(expr, Mul):
            args = [cls.eval(a, domain) for a in expr.args]
            v    =  S.One
            for i in args:
                v *= i
            return v

        elif isinstance(expr, _logical_partial_derivatives):
            if mapping.is_analytical:
                Ms   = [mapping[i] for i in range(dim)]
                expr = expr.subs(list(zip(Ms, mapping.expressions)))
            return expr

        elif isinstance(expr, IndexedVectorFunction):
            el = cls.eval(expr.base, domain)
            el = TerminalExpr(el, domain=domain.logical_domain)
            return el[expr.indices[0]]

        elif isinstance(expr, MinusInterfaceOperator):
            mapping = mapping.minus
            newexpr = PullBack(expr.args[0], mapping)
            test    = newexpr.test
            newexpr = newexpr.expr.subs(test, MinusInterfaceOperator(test))
            return newexpr

        elif isinstance(expr, PlusInterfaceOperator):
            mapping = mapping.plus
            newexpr = PullBack(expr.args[0], mapping)
            test    = newexpr.test
            newexpr = newexpr.expr.subs(test, PlusInterfaceOperator(test))
            return newexpr

        elif isinstance(expr, (VectorFunction, ScalarFunction)):
            return PullBack(expr, mapping).expr

        elif isinstance(expr, Transpose):
            arg = cls(expr.arg, domain)
            return Transpose(arg)
            
        elif isinstance(expr, grad):
            arg = expr.args[0]
            if isinstance(mapping, InterfaceMapping):
                if isinstance(arg, MinusInterfaceOperator):
                    a     = arg.args[0]
                    mapping = mapping.minus
                elif isinstance(arg, PlusInterfaceOperator):
                    a = arg.args[0]
                    mapping = mapping.plus
                else:
                    raise TypeError(arg)

                arg = type(arg)(cls.eval(a, domain))
            else:
                arg = cls.eval(arg, domain)

            return mapping.jacobian.inv().T*grad(arg)

        elif isinstance(expr, curl):
            arg = expr.args[0]
            if isinstance(mapping, InterfaceMapping):
                if isinstance(arg, MinusInterfaceOperator):
                    arg     = arg.args[0]
                    mapping = mapping.minus
                elif isinstance(arg, PlusInterfaceOperator):
                    arg = arg.args[0]
                    mapping = mapping.plus
                else:
                    raise TypeError(arg)

            if isinstance(arg, VectorFunction):
                arg = PullBack(arg, mapping)
            else:
                arg = cls.eval(arg, domain)

            if isinstance(arg, PullBack) and isinstance(arg.kind, HcurlSpaceType):
                J   = mapping.jacobian
                arg = arg.test
                if isinstance(expr.args[0], (MinusInterfaceOperator, PlusInterfaceOperator)):
                    arg = type(expr.args[0])(arg)
                if expr.is_scalar:
                    return (1/J.det())*curl(arg)

                return (J/J.det())*curl(arg)
            else:
                raise NotImplementedError('TODO')

        elif isinstance(expr, div):
            arg = expr.args[0]
            if isinstance(mapping, InterfaceMapping):
                if isinstance(arg, MinusInterfaceOperator):
                    arg     = arg.args[0]
                    mapping = mapping.minus
                elif isinstance(arg, PlusInterfaceOperator):
                    arg = arg.args[0]
                    mapping = mapping.plus
                else:
                    raise TypeError(arg)

            if isinstance(arg, (ScalarFunction, VectorFunction)):
                arg = PullBack(arg, mapping)
            else:

                arg = cls.eval(arg, domain)

            if isinstance(arg, PullBack) and isinstance(arg.kind, HdivSpaceType):
                J   = mapping.jacobian
                arg = arg.test
                if isinstance(expr.args[0], (MinusInterfaceOperator, PlusInterfaceOperator)):
                    arg = type(expr.args[0])(arg)
                return (1/J.det())*div(arg)
            elif isinstance(arg, PullBack):
                return SymbolicTrace(mapping.jacobian.inv().T*grad(arg.test))
            else:
                raise NotImplementedError('TODO')

        elif isinstance(expr, laplace):
            arg = expr.args[0]
            v   = cls.eval(grad(arg), domain)
            v   = mapping.jacobian.inv().T*grad(v)
            return SymbolicTrace(v)

#        elif isinstance(expr, hessian):
#           arg = expr.args[0]
#            if isinstance(mapping, InterfaceMapping):
#                if isinstance(arg, MinusInterfaceOperator):
#                    arg     = arg.args[0]
#                    mapping = mapping.minus
#                elif isinstance(arg, PlusInterfaceOperator):
#                    arg = arg.args[0]
#                    mapping = mapping.plus
#                else:
#                    raise TypeError(arg)
#            v   = cls.eval(grad(expr.args[0]), domain)
#            v   = mapping.jacobian.inv().T*grad(v)
#            return v

        elif isinstance(expr, (dot, inner, outer)):
            args = [cls.eval(arg, domain) for arg in expr.args]
            return type(expr)(*args)

        elif isinstance(expr, _diff_ops):
            raise NotImplementedError('TODO')

        # TODO MUST BE MOVED AFTER TREATING THE CASES OF GRAD, CURL, DIV IN FEEC
        elif isinstance(expr, (Matrix, ImmutableDenseMatrix)):
            n_rows, n_cols = expr.shape
            lines          = []
            for i_row in range(0, n_rows):
                line = []
                for i_col in range(0, n_cols):
                    line.append(cls.eval(expr[i_row,i_col], domain))
                lines.append(line)
            return type(expr)(lines)

        elif isinstance(expr, dx):
            if expr.atoms(PlusInterfaceOperator):
                mapping = mapping.plus
            elif expr.atoms(MinusInterfaceOperator):
                mapping = mapping.minus

            arg = expr.args[0]
            arg = cls(arg, domain, evaluate=True)

            if isinstance(arg, PullBack):
                arg = TerminalExpr(arg, domain=domain.logical_domain)
            elif isinstance(arg, MatrixElement):
                arg = TerminalExpr(arg, domain=domain.logical_domain)
            # ...
            if dim == 1:
                lgrad_arg = LogicalGrad_1d(arg)

                if not isinstance(lgrad_arg, (list, tuple, Tuple, Matrix)):
                    lgrad_arg = Tuple(lgrad_arg)

            elif dim == 2:
                lgrad_arg = LogicalGrad_2d(arg)

            elif dim == 3:
                lgrad_arg = LogicalGrad_3d(arg)
            
            grad_arg = Covariant(mapping, lgrad_arg)
            expr = grad_arg[0]
            return expr

        elif isinstance(expr, dy):
            if expr.atoms(PlusInterfaceOperator):
                mapping = mapping.plus
            elif expr.atoms(MinusInterfaceOperator):
                mapping = mapping.minus

            arg = expr.args[0]
            arg = cls(arg, domain, evaluate=True)
            if isinstance(arg, PullBack):
                arg = TerminalExpr(arg, domain=domain.logical_domain)
            elif isinstance(arg, MatrixElement):
                arg = TerminalExpr(arg, domain=domain.logical_domain)

            # ..p
            if dim == 1:
                lgrad_arg = LogicalGrad_1d(arg)

            elif dim == 2:
                lgrad_arg = LogicalGrad_2d(arg)

            elif dim == 3:
                lgrad_arg = LogicalGrad_3d(arg)

            grad_arg = Covariant(mapping, lgrad_arg)

            expr = grad_arg[1]
            return expr

        elif isinstance(expr, dz):
            if expr.atoms(PlusInterfaceOperator):
                mapping = mapping.plus
            elif expr.atoms(MinusInterfaceOperator):
                mapping = mapping.minus

            arg = expr.args[0]
            arg = cls(arg, domain, evaluate=True)
            if isinstance(arg, PullBack):
                arg = TerminalExpr(arg, domain=domain.logical_domain)
            elif isinstance(arg, MatrixElement):
                arg = TerminalExpr(arg, domain=domain.logical_domain)
            # ...
            if dim == 1:
                lgrad_arg = LogicalGrad_1d(arg)

            elif dim == 2:
                lgrad_arg = LogicalGrad_2d(arg)

            elif dim == 3:
                lgrad_arg = LogicalGrad_3d(arg)

            grad_arg = Covariant(mapping, lgrad_arg)

            expr = grad_arg[2]

            return expr

        elif isinstance(expr, (Symbol, Indexed)):
            return expr

        elif isinstance(expr, NormalVector):
            return expr

        elif isinstance(expr, Pow):
            b = expr.base
            e = expr.exp
            expr =  Pow(cls(b, domain), cls(e, domain))
            return expr

        elif isinstance(expr, Trace):
            e = cls.eval(expr.expr, domain)
            bd = expr.boundary.logical_domain
            order = expr.order
            return Trace(e, bd, order)

        elif isinstance(expr, Integral):
            domain  = expr.domain
            mapping = domain.mapping


            assert domain is not None

            if expr.is_domain_integral:
                J   = mapping.jacobian
                det = sqrt((J.T*J).det())
            else:
                axis = domain.axis
                J    = JacobianSymbol(mapping, axis=axis)
                det  = sqrt((J.T*J).det())

            body   = cls.eval(expr.expr, domain)*det
            domain  = domain.logical_domain
            return Integral(body, domain)

        elif isinstance(expr, BilinearForm):
            tests   = [get_logical_test_function(a) for a in expr.test_functions]
            trials  = [get_logical_test_function(a) for a in expr.trial_functions]
            body    = cls.eval(expr.expr, domain)
            return BilinearForm((trials, tests), body)

        elif isinstance(expr, LinearForm):
            tests   = [get_logical_test_function(a) for a in expr.test_functions]
            body    = cls.eval(expr.expr, domain)
            return LinearForm(tests, body)

        elif isinstance(expr, Norm):
            kind           = expr.kind
            exponent       = expr.exponent
            e              = cls.eval(expr.expr, domain)
            domain         = domain.logical_domain
            norm           = Norm(e, domain, kind, evaluate=False)
            norm._exponent = exponent
            return norm

        elif isinstance(expr, DomainExpression):
            domain  = expr.target
            J       = domain.mapping.jacobian
            newexpr = cls.eval(expr.expr, domain)
            newexpr = TerminalExpr(newexpr, domain=domain)
            domain  = domain.logical_domain
            det     = TerminalExpr(sqrt((J.T*J).det()), domain=domain)
            return DomainExpression(domain, ImmutableDenseMatrix([[newexpr*det]]))
            
        elif isinstance(expr, Function):
            args = [cls.eval(a, domain) for a in expr.args]
            return type(expr)(*args)

        return cls(expr, domain, evaluate=False)

#==============================================================================
class SymbolicExpr(CalculusFunction):
    """returns a sympy expression where partial derivatives are converted into
    sympy Symbols."""

    @cacheit
    def __new__(cls, *args, **options):
        # (Try to) sympify args first

        if options.pop('evaluate', True):
            r = cls.eval(*args)
        else:
            r = None

        if r is None:
            return Basic.__new__(cls, *args, **options)
        else:
            return r

    def __getitem__(self, indices, **kw_args):
        if is_sequence(indices):
            # Special case needed because M[*my_tuple] is a syntax error.
            return Indexed(self, *indices, **kw_args)
        else:
            return Indexed(self, indices, **kw_args)

    @classmethod
    @cacheit
    def eval(cls, *_args, **kwargs):
        """."""

        if not _args:
            return

        if not len(_args) == 1:
            raise ValueError('Expecting one argument')

        expr = _args[0]
        code = kwargs.pop('code', None)

        if isinstance(expr, Add):
            args = [cls.eval(a, code=code) for a in expr.args]
            v = Add(*args)
            return v

        elif isinstance(expr, Mul):
            args = [cls.eval(a, code=code) for a in expr.args]
            v    = Mul(*args)
            return v

        elif isinstance(expr, Pow):
            b = expr.base
            e = expr.exp
            v = Pow(cls.eval(b, code=code), e)
            return v

        elif isinstance(expr, _coeffs_registery):
            return expr

        elif isinstance(expr, (list, tuple, Tuple)):
            expr = [cls.eval(a, code=code) for a in expr]
            return Tuple(*expr)

        elif isinstance(expr, (Matrix, ImmutableDenseMatrix)):

            lines = []
            n_row,n_col = expr.shape
            for i_row in range(0,n_row):
                line = []
                for i_col in range(0,n_col):
                    line.append(cls.eval(expr[i_row, i_col], code=code))

                lines.append(line)

            return type(expr)(lines)

        elif isinstance(expr, (ScalarFunction, VectorFunction)):
            if code:
                name = '{name}_{code}'.format(name=expr.name, code=code)
            else:
                name = str(expr.name)

            return Symbol(name)

        elif isinstance(expr, ( PlusInterfaceOperator, MinusInterfaceOperator)):
            return cls.eval(expr.args[0], code=code)

        elif isinstance(expr, Indexed):
            base = expr.base
            if isinstance(base, AnalyticMapping):
                if expr.indices[0] == 0:
                    name = 'x'
                elif expr.indices[0] == 1:
                    name = 'y'
                elif expr.indices[0] == 2:
                    name = 'z'
                else:
                    raise ValueError('Wrong index')

                if base.is_plus:
                    name = name + '_plus'
            else:
                name =  '{base}_{i}'.format(base=base.name, i=expr.indices[0])

            if code:
                name = '{name}_{code}'.format(name=name, code=code)

            return Symbol(name)

        elif isinstance(expr, _partial_derivatives):
            atom = get_atom_derivatives(expr)
            indices = get_index_derivatives_atom(expr, atom)
            code = None
            if indices:
                index = indices[0]
                code = ''
                index =dict(sorted(index.items()))

                for k,n in list(index.items()):
                    code += k*n
            return cls.eval(atom, code=code)

        elif isinstance(expr, _logical_partial_derivatives):
            atom = get_atom_logical_derivatives(expr)
            indices = get_index_logical_derivatives_atom(expr, atom)
            code = None
            if indices:
                index = indices[0]
                code = ''
                index = dict(sorted(index.items()))
                for k,n in list(index.items()):
                    code += k*n
            return cls.eval(atom, code=code)

        elif isinstance(expr, AnalyticMapping):
            return Symbol(expr.name)

        # ... this must be done here, otherwise codegen for FEM will not work
        elif isinstance(expr, Symbol):
            return expr

        elif isinstance(expr, IndexedBase):
            return expr

        elif isinstance(expr, Indexed):
            return expr

        elif isinstance(expr, Idx):
            return expr

        elif isinstance(expr, Function):
            args = [cls.eval(a, code=code) for a in expr.args]
            return type(expr)(*args)

        elif isinstance(expr, ImaginaryUnit):
            return expr


        elif isinstance(expr, SymbolicWeightedVolume):
            mapping = expr.args[0]
            if isinstance(mapping, InterfaceMapping):
                mapping = mapping.minus
            name = 'wvol_{mapping}'.format(mapping=mapping)

            return Symbol(name)

        elif isinstance(expr, SymbolicDeterminant):
            name = 'det_{}'.format(str(expr.args[0]))
            return Symbol(name)

        elif isinstance(expr, PullBack):
            return cls.eval(expr.expr, code=code)

        # Expression must always be translated to Sympy!
        # TODO: check if we should use 'sympy.sympify(expr)' instead
        else:
            raise NotImplementedError('Cannot translate to Sympy: {}'.format(expr))
