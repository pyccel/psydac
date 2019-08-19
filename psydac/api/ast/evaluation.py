# TODO: - replace call to print_expression by SymbolicExpr (may need to use LogicalExpr)

from collections import OrderedDict
from itertools import groupby
import numpy as np

from sympy import Basic
from sympy import symbols, Symbol, IndexedBase, Indexed, Function
from sympy import Mul, Add, Tuple, Min, Max, Pow
from sympy import Matrix, ImmutableDenseMatrix
from sympy import sqrt as sympy_sqrt
from sympy import S as sympy_S
from sympy import Integer, Float
from sympy.core.relational    import Le, Ge
from sympy.logic.boolalg      import And
from sympy import Mod, Abs
from sympy.core.function import AppliedUndef

from pyccel.ast.core import Variable, IndexedVariable
from pyccel.ast.core import For
from pyccel.ast.core import Assign
from pyccel.ast.core import AugAssign
from pyccel.ast.core import Slice
from pyccel.ast.core import Range, Product
from pyccel.ast.core import FunctionDef
from pyccel.ast.core import FunctionCall
from pyccel.ast.core import Import
from pyccel.ast import Zeros
from pyccel.ast import Import
from pyccel.ast import DottedName
from pyccel.ast import Nil
from pyccel.ast import Len
from pyccel.ast import If, Is, Return
from pyccel.ast import String, Print, Shape
from pyccel.ast import Comment, NewLine
from pyccel.ast.core      import _atomic
from pyccel.ast.utilities import build_types_decorator


from sympde.core import Cross_3d
from sympde.core import Constant
from sympde.calculus import grad
from sympde.topology import Mapping
from sympde.topology import ScalarField
from sympde.topology import VectorField, IndexedVectorField
from sympde.topology import Boundary, BoundaryVector, NormalVector, TangentVector
from sympde.topology import Covariant, Contravariant
from sympde.topology import ElementArea
from sympde.topology import LogicalExpr
from sympde.topology import SymbolicExpr
from sympde.topology.derivatives import _partial_derivatives
from sympde.topology.derivatives import _logical_partial_derivatives
from sympde.topology.derivatives import get_max_partial_derivatives
from sympde.topology.space import ScalarFunctionSpace
from sympde.topology.space import ScalarTestFunction
from sympde.topology.space import VectorTestFunction
from sympde.topology.space import IndexedTestTrial
from sympde.topology.space import Trace
from sympde.topology.derivatives import print_expression
from sympde.topology.derivatives import get_atom_derivatives
from sympde.topology.derivatives import get_index_derivatives
from sympde.expr import BilinearForm, LinearForm, Functional, BasicForm

from psydac.fem.splines import SplineSpace
from psydac.fem.tensor  import TensorFemSpace
from psydac.fem.vector  import ProductFemSpace

from .basic import SplBasic
from .utilities import build_pythran_types_header, variables
from .utilities import filter_loops, filter_product
from .utilities import rationalize_eval_mapping
from .utilities import compute_atoms_expr_mapping
from .utilities import compute_atoms_expr_field


#==============================================================================
class EvalQuadratureMapping(SplBasic):

    def __new__(cls, space, mapping, discrete_boundary=None, name=None,
                boundary_basis=None, nderiv=1, is_rational_mapping=None,
                area=None, backend=None):

        if not isinstance(mapping, Mapping):
            raise TypeError('> Expecting a Mapping object')

        obj = SplBasic.__new__(cls, mapping, name=name,
                               prefix='eval_mapping', mapping=mapping,
                               is_rational_mapping=is_rational_mapping)

        obj._space = space
        obj._discrete_boundary = discrete_boundary
        obj._boundary_basis = boundary_basis
        obj._backend = backend

        dim = mapping.rdim

        # ...
        lcoords = ['x1', 'x2', 'x3'][:dim]
        obj._lcoords = symbols(lcoords)
        # ...

        # ...
        ops = _partial_derivatives[:dim]
        M = mapping

        components = [M[i] for i in range(0, dim)]

        d_elements = {}
        d_elements[0] = list(components)

        if nderiv > 0:
            ls = [d(M[i]) for d in ops for i in range(0, dim)]

            d_elements[1] = ls

        if nderiv > 1:
            ls = [d1(d2(M[i])) for e,d1 in enumerate(ops)
                               for d2 in ops[:e+1]
                               for i in range(0, dim)]

            d_elements[2] = ls

        if nderiv > 2:
            raise NotImplementedError('TODO')

        elements = [i for l in d_elements.values() for i in l]
        obj._elements = tuple(elements)
        obj._d_elements = d_elements

        obj._components = tuple(components)
        obj._nderiv = nderiv
        obj._area = area
        # ...

        obj._func = obj._initialize()

        return obj

    @property
    def space(self):
        return self._space

    @property
    def boundary_basis(self):
        return self._boundary_basis

    @property
    def nderiv(self):
        return self._nderiv

    @property
    def lcoords(self):
        return self._lcoords

    @property
    def elements(self):
        return self._elements

    @property
    def d_elements(self):
        return self._d_elements

    @property
    def components(self):
        return self._components

    @property
    def mapping_coeffs(self):
        return self._mapping_coeffs

    @property
    def mapping_values(self):
        return self._mapping_values

    @property
    def backend(self):
        return self._backend

    @property
    def area(self):
        return self._area

    @property
    def weights(self):
        return self._weights

    def build_arguments(self, data):

        other = data
        if self.area:
            other = other + self.weights + (self.area, )

        return self.basic_args + other

    def _initialize(self):
        space = self.space
        dim = space.ldim

        _print = lambda i: print_expression(i, mapping_name=False)
        mapping_atoms = [_print(f) for f in self.components]
        mapping_str = [_print(f) for f in self.elements]
#        mapping_str = sorted([_print(f) for f in self.elements])

        # ... declarations
        degrees        = variables( 'p1:%s'%(dim+1), 'int')
        orders         = variables( 'k1:%s'%(dim+1), 'int')
        indices_basis  = variables( 'jl1:%s'%(dim+1), 'int')
        indices_quad   = variables( 'g1:%s'%(dim+1), 'int')
        basis          = variables('basis1:%s'%(dim+1),
                                  dtype='real', rank=3, cls=IndexedVariable)
        mapping_coeffs = variables(['coeff_{}'.format(f) for f in mapping_atoms],
                                  dtype='real', rank=dim, cls=IndexedVariable)
        mapping_values = variables(['{}_values'.format(f) for f in mapping_str],
                                  dtype='real', rank=dim, cls=IndexedVariable)

        # ... needed for area
        weights = variables('quad_w1:%s'%(dim+1),
                            dtype='real', rank=1, cls=IndexedVariable)

        self._weights = weights
        # ...

        weights_elements = []
        if self.is_rational_mapping:
            # TODO check if 'w' exist already
            weights_pts = ScalarField(self.space, name='w')

            weights_elements = [weights_pts]

            # ...
            nderiv = self.nderiv
            ops = _partial_derivatives[:dim]

            if nderiv > 0:
                weights_elements += [d(weights_pts) for d in ops]

            if nderiv > 1:
                weights_elements += [d1(d2(weights_pts)) for e,d1 in enumerate(ops)
                                                     for d2 in ops[:e+1]]

            if nderiv > 2:
                raise NotImplementedError('TODO')
            # ...

            mapping_weights_str = [_print(f) for f in weights_elements]
            mapping_wvalues = variables(['{}_values'.format(f) for f in mapping_weights_str],
                                                dtype='real', rank=dim, cls=IndexedVariable)

            mapping_coeffs  = mapping_coeffs + (IndexedVariable('coeff_w', dtype='real', rank=dim),)
            mapping_values  = mapping_values + tuple(mapping_wvalues)

        weights_elements = tuple(weights_elements)
        # ...

        # ... ranges
        ranges_basis = [Range(degrees[i]+1) for i in range(dim)]
        ranges_quad  = [Range(orders[i]) for i in range(dim)]
        # ...

        # ... basic arguments
        self._basic_args = (orders)
        # ...

        # ...
        self._mapping_coeffs  = mapping_coeffs
        self._mapping_values  = mapping_values
        # ...

        # ...
        Nj = ScalarTestFunction(space, name='Nj')
        body = []
        init_basis = OrderedDict()
        updates = []
        atomic_exprs = self.elements + weights_elements
        inits, updates = compute_atoms_expr_mapping(atomic_exprs, indices_quad,
                                                    indices_basis, basis, Nj)

        for init in inits:
            basis_name = str(init.lhs)
            init_basis[basis_name] = init

        init_basis = OrderedDict(sorted(init_basis.items()))
        body += list(init_basis.values())
        body += updates
        # ...

        # put the body in tests for loops
        body = filter_loops(indices_basis, ranges_basis, body,
                            self.discrete_boundary,
                            boundary_basis=self.boundary_basis)

        if self.is_rational_mapping:
            stmts = rationalize_eval_mapping(self.mapping, self.nderiv,
                                             self.space, indices_quad)

            body += stmts

        # ...
        if self.area:
            weight = filter_product(indices_quad, weights, self.discrete_boundary)

            stmts = area_eval_mapping(self.mapping, self.area, dim, indices_quad, weight)

            body += stmts
        # ...

        # put the body in for loops of quadrature points
        body = filter_loops(indices_quad, ranges_quad, body,
                            self.discrete_boundary,
                            boundary_basis=self.boundary_basis)

        # initialization of the matrix
        init_vals = [f[[Slice(None,None)]*dim] for f in mapping_values]
        init_vals = [Assign(e, 0.0) for e in init_vals]
        body =  init_vals + body

        if self.area:
            # add init to 0 at the begining
            body = [Assign(self.area, 0.0)] + body

            # add power to 1/dim
            body += [Assign(self.area, Pow(self.area, 1./dim))]

        func_args = self.build_arguments(degrees + basis + mapping_coeffs + mapping_values)

        decorators = {}
        header = None
        if self.backend['name'] == 'pyccel':
            decorators = {'types': build_types_decorator(func_args)}
        elif self.backend['name'] == 'numba':
            decorators = {'jit':[]}
        elif self.backend['name'] == 'pythran':
            header = build_pythran_types_header(self.name, func_args)

        return FunctionDef(self.name, list(func_args), [], body,
                           decorators=decorators,header=header)


#==============================================================================
class EvalQuadratureField(SplBasic):

    def __new__(cls, space, fields, discrete_boundary=None, name=None,
                boundary_basis=None, mapping=None, is_rational_mapping=None,backend=None):

        if not isinstance(fields, (tuple, list, Tuple)):
            raise TypeError('> Expecting an iterable')

        obj = SplBasic.__new__(cls, space, name=name,
                               prefix='eval_field', mapping=mapping,
                               is_rational_mapping=is_rational_mapping)

        obj._space = space
        obj._fields = Tuple(*fields)
        obj._discrete_boundary = discrete_boundary
        obj._boundary_basis = boundary_basis
        obj._backend = backend
        obj._func = obj._initialize()


        return obj

    @property
    def space(self):
        return self._space

    @property
    def fields(self):
        return self._fields

    @property
    def map_stmts(self):
        return self._map_stmts

    @property
    def boundary_basis(self):
        return self._boundary_basis

    @property
    def backend(self):
        return self._backend

    def build_arguments(self, data):

        other = data

        return self.basic_args + other

    def _initialize(self):
        space = self.space
        dim = space.ldim
        mapping = self.mapping

        field_atoms = self.fields.atoms(ScalarField)

        # ... declarations
        degrees       = variables( 'p1:%s'%(dim+1), 'int')
        orders        = variables( 'k1:%s'%(dim+1), 'int')
        indices_basis = variables( 'jl1:%s'%(dim+1), 'int')
        indices_quad  = variables( 'g1:%s'%(dim+1), 'int')
        basis         = variables('basis1:%s'%(dim+1),
                                  dtype='real', rank=3, cls=IndexedVariable)
        fields_coeffs = variables(['coeff_{}'.format(f) for f in field_atoms],
                                  dtype='real', rank=dim, cls=IndexedVariable)

        # ...

        # ... ranges
        ranges_basis = [Range(degrees[i]+1) for i in range(dim)]
        ranges_quad  = [Range(orders[i]) for i in range(dim)]
        # ...

        # ... basic arguments
        self._basic_args = (orders)
        # ...

        # ...
        Nj = ScalarTestFunction(space, name='Nj')
        body = []
        init_basis = OrderedDict()
        init_map   = OrderedDict()

        inits, updates, map_stmts, fields = compute_atoms_expr_field(self.fields, indices_quad, indices_basis,
                                                               basis, Nj,
                                                               mapping=mapping)

        for init in inits:
            basis_name = str(init.lhs)
            init_basis[basis_name] = init
        for stmt in map_stmts:
            init_map[str(stmt.lhs)] = stmt

        self._fields = Tuple(*fields)

        fields_str    = [print_expression(f) for f in self.fields]
        fields_val    = variables(['{}_values'.format(f) for f in set(fields_str)],
                                  dtype='real', rank=dim, cls=IndexedVariable)


        init_basis = OrderedDict(sorted(init_basis.items()))
        body += list(init_basis.values())
        body += updates
        self._map_stmts = init_map
        # ...

        # put the body in tests for loops
        body = filter_loops(indices_basis, ranges_basis, body,
                            self.discrete_boundary,
                            boundary_basis=self.boundary_basis)


        # put the body in for loops of quadrature points
        body = filter_loops(indices_quad, ranges_quad, body,
                            self.discrete_boundary,
                            boundary_basis=self.boundary_basis)


        # initialization of the matrix
        init_vals = [f[[Slice(None,None)]*dim] for f in fields_val]
        init_vals = [Assign(e, 0.0) for e in init_vals]
        body =  init_vals + body

        func_args = self.build_arguments(degrees + basis + fields_coeffs + fields_val)

        decorators = {}
        header = None
        if self.backend['name'] == 'pyccel':
            decorators = {'types': build_types_decorator(func_args)}
        elif self.backend['name'] == 'numba':
            decorators = {'jit':[]}
        elif self.backend['name'] == 'pythran':
            header = build_pythran_types_header(self.name, func_args)

        return FunctionDef(self.name, list(func_args), [], body,
                           decorators=decorators,header=header)



#==============================================================================
class EvalQuadratureVectorField(SplBasic):

    def __new__(cls, space, vector_fields, discrete_boundary=None, name=None,
                boundary_basis=None, mapping=None, is_rational_mapping=None, backend = None):

        if not isinstance(vector_fields, (tuple, list, Tuple)):
            raise TypeError('> Expecting an iterable')

        obj = SplBasic.__new__(cls, space, name=name,
                               prefix='eval_vector_field', mapping=mapping,
                               is_rational_mapping=is_rational_mapping)

        obj._space = space
        obj._vector_fields = Tuple(*vector_fields)
        obj._discrete_boundary = discrete_boundary
        obj._boundary_basis = boundary_basis
        obj._backend = backend
        obj._func = obj._initialize()

        return obj

    @property
    def space(self):
        return self._space

    @property
    def vector_fields(self):
        return self._vector_fields

    @property
    def map_stmts(self):
        return self._map_stmts

    @property
    def boundary_basis(self):
        return self._boundary_basis

    @property
    def backend(self):
        return self._backend

    def build_arguments(self, data):

        other = data

        return self.basic_args + other

    def _initialize(self):
        space = self.space
        dim = space.ldim
        mapping = self.mapping

        vector_field_atoms = self.vector_fields.atoms(VectorField)
        vector_field_atoms = [f[i] for f in vector_field_atoms for i in range(0, dim)]


        # ... declarations
        degrees       = variables('p1:%s'%(dim+1),  'int')
        orders        = variables('k1:%s'%(dim+1),  'int')
        indices_basis = variables('jl1:%s'%(dim+1), 'int')
        indices_quad  = variables('g1:%s'%(dim+1),  'int')
        basis         = variables('basis1:%s'%(dim+1),
                                  dtype='real', rank=3, cls=IndexedVariable)
        coeffs = ['coeff_{}'.format(print_expression(f)) for f in vector_field_atoms]
        vector_fields_coeffs = variables(coeffs, dtype='real', rank=dim, cls=IndexedVariable)
        # ...

        # ... ranges
        ranges_basis = [Range(degrees[i]+1) for i in range(dim)]
        ranges_quad  = [Range(orders[i]) for i in range(dim)]
        # ...

        # ... basic arguments
        self._basic_args = (orders)
        # ...

        # ...
        Nj = VectorField(space, name='Nj')
        body = []
        init_basis = OrderedDict()
        init_map   = OrderedDict()
        
        inits, updates, map_stmts, vector_fields = compute_atoms_expr_field(self.vector_fields, indices_quad, indices_basis,
                                                               basis, Nj, mapping=mapping)

        for init in inits:
            basis_name = str(init.lhs)
            init_basis[basis_name] = init
        for stmt in map_stmts:
            init_map[str(stmt.lhs)] = stmt

        self._vector_fields = Tuple(*vector_fields)
        vector_fields_str   =  [print_expression(f) for f in self.vector_fields]
        vector_fields_val   = variables(['{}_values'.format(f) for f in vector_fields_str],
                                          dtype='real', rank=dim, cls=IndexedVariable)

        init_basis = OrderedDict(sorted(init_basis.items()))
        body += list(init_basis.values())
        body += updates
        self._map_stmts = init_map
        # ...

        # put the body in tests for loops
        body = filter_loops(indices_basis, ranges_basis, body,
                            self.discrete_boundary,
                            boundary_basis=self.boundary_basis)

        # put the body in for loops of quadrature points
        body = filter_loops(indices_quad, ranges_quad, body,
                            self.discrete_boundary,
                            boundary_basis=self.boundary_basis)

        # initialization of the matrix
        init_vals = [f[[Slice(None,None)]*dim] for f in vector_fields_val]
        init_vals = [Assign(e, 0.0) for e in init_vals]
        body =  init_vals + body

        func_args = self.build_arguments(degrees + basis + vector_fields_coeffs + vector_fields_val)

        decorators = {}
        header = None
        if self.backend['name'] == 'pyccel':
            decorators = {'types': build_types_decorator(func_args)}
        elif self.backend['name'] == 'numba':
            decorators = {'jit':[]}
        elif self.backend['name'] == 'pythran':
            header = build_pythran_types_header(self.name, func_args)

        return FunctionDef(self.name, list(func_args), [], body,
                           decorators=decorators,header=header)


#==============================================================================
# TODO move it
def _create_loop(indices, ranges, body):

    dim = len(indices)
    for i in range(dim-1,-1,-1):
        rx = ranges[i]
        x = indices[i]

        start = rx.start
        end   = rx.stop

        rx = Range(start, end)
        body = [For(x, rx, body)]

    return body

#==============================================================================
class EvalArrayField(SplBasic):

    def __new__(cls, space, fields, discrete_boundary=None, name=None,
                boundary_basis=None, mapping=None, is_rational_mapping=None,backend=None):

        if not isinstance(fields, (tuple, list, Tuple)):
            raise TypeError('> Expecting an iterable')

        obj = SplBasic.__new__(cls, space, name=name,
                               prefix='eval_field', mapping=mapping,
                               is_rational_mapping=is_rational_mapping)

        obj._space = space
        obj._fields = Tuple(*fields)
        obj._discrete_boundary = discrete_boundary
        obj._boundary_basis = boundary_basis
        obj._backend = backend
        obj._func = obj._initialize()


        return obj

    @property
    def space(self):
        return self._space

    @property
    def fields(self):
        return self._fields

    @property
    def map_stmts(self):
        return self._map_stmts

    @property
    def boundary_basis(self):
        return self._boundary_basis

    @property
    def backend(self):
        return self._backend

    def build_arguments(self, data):

        other = data

        return self.basic_args + other

    def _initialize(self):
        space = self.space
        dim = space.ldim
        mapping = self.mapping

        field_atoms = self.fields.atoms(ScalarField)
        fields_str = sorted([print_expression(f) for f in self.fields])

        # ... declarations
        degrees        = variables( 'p1:%s'%(dim+1), 'int')
        orders         = variables( 'k1:%s'%(dim+1), 'int')
        indices_basis  = variables( 'jl1:%s'%(dim+1), 'int')
        indices_quad   = variables( 'g1:%s'%(dim+1), 'int')

        basis          = variables('basis1:%s'%(dim+1),
                                  dtype='real', rank=3, cls=IndexedVariable)
        fields_coeffs  = variables(['coeff_{}'.format(f) for f in field_atoms],
                                  dtype='real', rank=dim, cls=IndexedVariable)
        fields_val     = variables(['{}_values'.format(f) for f in fields_str],
                                  dtype='real', rank=dim, cls=IndexedVariable)

        spans  = variables( 'spans1:%s'%(dim+1),
                            dtype = 'int', rank = 1, cls = IndexedVariable )
        i_spans = variables( 'i_span1:%s'%(dim+1), 'int')
        # ...

        # ... ranges
        # we add the degree because of the padding
        ranges_basis = [Range(i_spans[i], i_spans[i]+degrees[i]+1) for i in range(dim)]
        ranges_quad  = [Range(orders[i]) for i in range(dim)]
        # ...

        # ... basic arguments
        self._basic_args = (orders)
        # ...

        # ...
        body = []
        updates = []
        # ...

        # ...
        Nj = ScalarTestFunction(space, name='Nj')
        init_basis = OrderedDict()
        init_map   = OrderedDict()

        inits, updates, map_stmts, _ = compute_atoms_expr_field(self.fields, indices_quad, indices_basis,
                                                               basis, Nj, mapping=mapping)


        for init in inits:
            basis_name = str(init.lhs)
            init_basis[basis_name] = init
        for stmt in map_stmts:
            init_map[str(stmt.lhs)] = stmt

        init_basis = OrderedDict(sorted(init_basis.items()))
        body += list(init_basis.values())
        body += updates
        self._map_stmts = init_map
        # ...

        # put the body in tests for loops
        body = _create_loop(indices_basis, ranges_basis, body)


        # put the body in for loops of quadrature points
        assign_spans = []
        for x, i_span, span in zip(indices_quad, i_spans, spans):
            assign_spans += [Assign(i_span, span[x])]

        body = assign_spans + body

        body = _create_loop(indices_quad, ranges_quad, body)


        # initialization of the matrix
        init_vals = [f[[Slice(None,None)]*dim] for f in fields_val]
        init_vals = [Assign(e, 0.0) for e in init_vals]
        body =  init_vals + body

        func_args = self.build_arguments(degrees + spans + basis + fields_coeffs + fields_val)

        decorators = {}
        header = None
        if self.backend['name'] == 'pyccel':
            decorators = {'types': build_types_decorator(func_args)}
        elif self.backend['name'] == 'numba':
            decorators = {'jit':[]}
        elif self.backend['name'] == 'pythran':
            header = build_pythran_types_header(self.name, func_args)

        return FunctionDef(self.name, list(func_args), [], body,
                           decorators=decorators,header=header)


#==============================================================================
class EvalArrayMapping(SplBasic):

    def __new__(cls, space, mapping, name=None,
                nderiv=1, is_rational_mapping=None,
                backend=None):

        if not isinstance(mapping, Mapping):
            raise TypeError('> Expecting a Mapping object')

        obj = SplBasic.__new__(cls, mapping, name=name,
                               prefix='eval_mapping', mapping=mapping,
                               is_rational_mapping=is_rational_mapping)

        obj._space = space
        obj._backend = backend

        dim = mapping.rdim

        # ...
        lcoords = ['x1', 'x2', 'x3'][:dim]
        obj._lcoords = symbols(lcoords)
        # ...

        # ...
        ops = _partial_derivatives[:dim]
        M = mapping

        components = [M[i] for i in range(0, dim)]

        d_elements = {}
        d_elements[0] = list(components)

        if nderiv > 0:
            ls = [d(M[i]) for d in ops for i in range(0, dim)]

            d_elements[1] = ls

        if nderiv > 1:
            ls = [d1(d2(M[i])) for e,d1 in enumerate(ops)
                               for d2 in ops[:e+1]
                               for i in range(0, dim)]

            d_elements[2] = ls

        if nderiv > 2:
            raise NotImplementedError('TODO')

        elements = [i for l in d_elements.values() for i in l]
        obj._elements = tuple(elements)
        obj._d_elements = d_elements

        obj._components = tuple(components)
        obj._nderiv = nderiv
        # ...

        obj._func = obj._initialize()

        return obj

    @property
    def space(self):
        return self._space

    @property
    def nderiv(self):
        return self._nderiv

    @property
    def lcoords(self):
        return self._lcoords

    @property
    def elements(self):
        return self._elements

    @property
    def d_elements(self):
        return self._d_elements

    @property
    def components(self):
        return self._components

    @property
    def mapping_coeffs(self):
        return self._mapping_coeffs

    @property
    def mapping_values(self):
        return self._mapping_values

    @property
    def backend(self):
        return self._backend

    @property
    def weights(self):
        return self._weights

    def build_arguments(self, data):

        other = data

        return self.basic_args + other

    def _initialize(self):
        space = self.space
        dim = space.ldim

        _print = lambda i: print_expression(i, mapping_name=False)
        mapping_atoms = [_print(f) for f in self.components]
        mapping_str = [_print(f) for f in self.elements]
#        mapping_str = sorted([_print(f) for f in self.elements])

        # ... declarations
        degrees        = variables( 'p1:%s'%(dim+1), 'int')
        orders         = variables( 'k1:%s'%(dim+1), 'int')
        indices_basis  = variables( 'jl1:%s'%(dim+1), 'int')
        indices_quad   = variables( 'g1:%s'%(dim+1), 'int')
        basis          = variables('basis1:%s'%(dim+1),
                                  dtype='real', rank=3, cls=IndexedVariable)
        mapping_coeffs = variables(['coeff_{}'.format(f) for f in mapping_atoms],
                                  dtype='real', rank=dim, cls=IndexedVariable)
        mapping_values = variables(['{}_values'.format(f) for f in mapping_str],
                                  dtype='real', rank=dim, cls=IndexedVariable)

        spans  = variables( 'spans1:%s'%(dim+1),
                            dtype = 'int', rank = 1, cls = IndexedVariable )
        i_spans = variables( 'i_span1:%s'%(dim+1), 'int')

        # ... needed for area
        weights = variables('quad_w1:%s'%(dim+1),
                            dtype='real', rank=1, cls=IndexedVariable)

        self._weights = weights
        # ...

        weights_elements = []
        if self.is_rational_mapping:
            # TODO check if 'w' exist already
            weights_pts = ScalarField(self.space, name='w')

            weights_elements = [weights_pts]

            # ...
            nderiv = self.nderiv
            ops = _partial_derivatives[:dim]

            if nderiv > 0:
                weights_elements += [d(weights_pts) for d in ops]

            if nderiv > 1:
                weights_elements += [d1(d2(weights_pts)) for e,d1 in enumerate(ops)
                                                     for d2 in ops[:e+1]]

            if nderiv > 2:
                raise NotImplementedError('TODO')
            # ...

            mapping_weights_str = [_print(f) for f in weights_elements]
            mapping_wvalues = variables(['{}_values'.format(f) for f in mapping_weights_str],
                                                dtype='real', rank=dim, cls=IndexedVariable)

            mapping_coeffs  = mapping_coeffs + (IndexedVariable('coeff_w', dtype='real', rank=dim),)
            mapping_values  = mapping_values + tuple(mapping_wvalues)

        weights_elements = tuple(weights_elements)
        # ...

        # ... ranges
        # we add the degree because of the padding
        ranges_basis = [Range(i_spans[i], i_spans[i]+degrees[i]+1) for i in range(dim)]
        ranges_quad  = [Range(orders[i]) for i in range(dim)]
        # ...

        # ... basic arguments
        self._basic_args = (orders)
        # ...

        # ...
        self._mapping_coeffs  = mapping_coeffs
        self._mapping_values  = mapping_values
        # ...

        # ...
        Nj             = ScalarTestFunction(space, name='Nj')
        body           = []
        init_basis     = OrderedDict()
        atomic_exprs   = self.elements + weights_elements
        inits, updates = compute_atoms_expr_mapping(atomic_exprs, indices_quad,
                                                      indices_basis, basis, Nj)
        for init in inits:
            basis_name = str(init.lhs)
            init_basis[basis_name] = init

        init_basis = OrderedDict(sorted(init_basis.items()))
        body      += list(init_basis.values())
        body      += updates
        # ...

        # put the body in tests for loops
        body = _create_loop(indices_basis, ranges_basis, body)

        if self.is_rational_mapping:
            stmts = rationalize_eval_mapping(self.mapping, self.nderiv,
                                             self.space, indices_quad)

            body += stmts

        assign_spans = []
        for x, i_span, span in zip(indices_quad, i_spans, spans):
            assign_spans += [Assign(i_span, span[x])]

        body = assign_spans + body
        # put the body in for loops of quadrature points
        body = _create_loop(indices_quad, ranges_quad, body)

        # initialization of the matrix
        init_vals = [f[[Slice(None,None)]*dim] for f in mapping_values]
        init_vals = [Assign(e, 0.0) for e in init_vals]
        body =  init_vals + body

        func_args = self.build_arguments(degrees + spans + basis + mapping_coeffs + mapping_values)

        decorators = {}
        header = None
        if self.backend['name'] == 'pyccel':
            decorators = {'types': build_types_decorator(func_args)}
        elif self.backend['name'] == 'numba':
            decorators = {'jit':[]}
        elif self.backend['name'] == 'pythran':
            header = build_pythran_types_header(self.name, func_args)

        return FunctionDef(self.name, list(func_args), [], body,
                           decorators=decorators,header=header)

