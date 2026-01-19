#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from sympy import symbols, Range
from sympy import Tuple

from sympde.topology             import Mapping
from sympde.topology             import ScalarFunction
from sympde.topology             import SymbolicExpr
from sympde.topology.space       import element_of
from sympde.topology.derivatives import _logical_partial_derivatives

from psydac.pyccel.ast.core      import IndexedVariable
from psydac.pyccel.ast.core      import For
from psydac.pyccel.ast.core      import Assign
from psydac.pyccel.ast.core      import Slice
from psydac.pyccel.ast.core      import FunctionDef

from .basic     import SplBasic
from .utilities import build_pythran_types_header, variables
from .utilities import build_pyccel_type_annotations
from .utilities import rationalize_eval_mapping
from .utilities import compute_atoms_expr_mapping
from .utilities import compute_atoms_expr_field

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
# NOTE: this is used in module 'psydac.api.ast.glt'
class EvalArrayField(SplBasic):

    def __new__(cls, space, fields, boundary=None, name=None,
                boundary_basis=None, mapping=None, is_rational_mapping=None,backend=None):

        if not isinstance(fields, (tuple, list, Tuple)):
            raise TypeError('> Expecting an iterable')

        obj = SplBasic.__new__(cls, space, name=name,
                               prefix='eval_field', mapping=mapping,
                               is_rational_mapping=is_rational_mapping)

        obj._space = space
        obj._fields = Tuple(*fields)
        obj._boundary = boundary
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

        field_atoms = self.fields.atoms(ScalarFunction)
        fields_str = sorted([SymbolicExpr(f).name for f in self.fields])

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
        Nj = element_of(space, name='Nj')
        init_basis = {}
        init_map   = {}

        inits, updates, map_stmts, fields = compute_atoms_expr_field(self.fields, indices_quad, indices_basis,
                                                               basis, Nj, mapping=mapping)

        self._fields = fields
        for init in inits:
            basis_name = str(init.lhs)
            init_basis[basis_name] = init
        for stmt in map_stmts:
            init_map[str(stmt.lhs)] = stmt

        init_basis = dict(sorted(init_basis.items()))
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
            func_args = build_pyccel_type_annotations(func_args)
        elif self.backend['name'] == 'pythran':
            header = build_pythran_types_header(self.name, func_args)

        return FunctionDef(self.name, list(func_args), [], body,
                           decorators=decorators, header=header)


#==============================================================================
# NOTE: this is used in module 'psydac.api.ast.glt'
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

        dim = mapping.ldim

        # ...
        lcoords = ['x1', 'x2', 'x3'][:dim]
        obj._lcoords = symbols(lcoords)
        # ...

        # ...
        ops = _logical_partial_derivatives[:dim]
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

        mapping_atoms = [SymbolicExpr(f).name for f in self.components]
        mapping_str   = [SymbolicExpr(f).name for f in self.elements  ]

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
            weights_pts = element_of(self.space, name='w')

            weights_elements = [weights_pts]

            # ...
            nderiv = self.nderiv
            ops = _logical_partial_derivatives[:dim]

            if nderiv > 0:
                weights_elements += [d(weights_pts) for d in ops]

            if nderiv > 1:
                weights_elements += [d1(d2(weights_pts)) for e,d1 in enumerate(ops)
                                                     for d2 in ops[:e+1]]

            if nderiv > 2:
                raise NotImplementedError('TODO')
            # ...

            mapping_weights_str = [SymbolicExpr(f).name for f in weights_elements]
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
        Nj             = element_of(space, name='Nj')
        body           = []
        init_basis     = {}
        atomic_exprs   = self.elements + weights_elements
        inits, updates = compute_atoms_expr_mapping(atomic_exprs, indices_quad,
                                                      indices_basis, basis, Nj)
        for init in inits:
            basis_name = str(init.lhs)
            init_basis[basis_name] = init

        init_basis = dict(sorted(init_basis.items()))
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
            func_args = build_pyccel_type_annotations(func_args)
        elif self.backend['name'] == 'pythran':
            header = build_pythran_types_header(self.name, func_args)

        return FunctionDef(self.name, list(func_args), [], body,
                           decorators=decorators,header=header)
