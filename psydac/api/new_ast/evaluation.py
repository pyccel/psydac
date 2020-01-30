from collections import OrderedDict

from sympy import symbols
from sympy import Tuple, Pow

from sympde.topology             import Mapping
from sympde.topology             import ScalarField
from sympde.topology             import VectorField
from sympde.topology             import SymbolicExpr
from sympde.topology.space       import element_of
from sympde.topology.space       import ScalarFunctionSpace
from sympde.topology.space       import VectorFunctionSpace
from sympde.topology.datatype    import dtype_space_registry
from sympde.topology.derivatives import _logical_partial_derivatives

from pyccel.ast.core      import IndexedVariable
from pyccel.ast.core      import For
from pyccel.ast.core      import Assign
from pyccel.ast.core      import Slice
from pyccel.ast.core      import Range
from pyccel.ast.core      import FunctionDef
from pyccel.ast.utilities import build_types_decorator

from .basic     import SplBasic
from .utilities import build_pythran_types_header, variables
from .utilities import filter_loops, filter_product, select_loops
from .utilities import rationalize_eval_mapping
from .utilities import compute_atoms_expr_mapping
from .utilities import compute_atoms_expr_field

#==============================================================================
class EvalQuadratureMapping(SplBasic):

    def __new__(cls, space, mapping, boundary=None, name=None,
                boundary_basis=None, nderiv=1, is_rational_mapping=None,
                area=None, backend=None):

        if not isinstance(mapping, Mapping):
            raise TypeError('> Expecting a Mapping object')

        #.....................................................................
        # If vector space is of undefined type, we assume that each component
        # lives in H1; otherwise we raise an error. TODO: improve!
        if isinstance(space, VectorFunctionSpace):
            if space.kind == dtype_space_registry['undefined']:
                space = ScalarFunctionSpace(
                    name   = space.name,
                    domain = space.domain,
                    kind   = 'H1'
                )
            else:
                msg = 'Cannot evaluate vector spaces of kind {}'.format(space.kind)
                raise NotImplementedError(msg)
        #.....................................................................

        obj = SplBasic.__new__(cls, mapping, name=name,
                               prefix='eval_mapping', mapping=mapping,
                               is_rational_mapping=is_rational_mapping)

        obj._space = space
        obj._boundary = boundary
        obj._boundary_basis = boundary_basis
        obj._backend = backend

        dim = mapping.rdim

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
        Nj           = element_of(space, name='Nj')
        body         = []
        init_basis   = OrderedDict()
        updates      = []
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
        body = select_loops(indices_basis, ranges_basis, body, boundary=None)

        if self.is_rational_mapping:
            stmts = rationalize_eval_mapping(self.mapping, self.nderiv,
                                             self.space, indices_quad)

            body += stmts

        # ...
        if self.area:
            weight = filter_product(indices_quad, weights, self.boundary)

            stmts = area_eval_mapping(self.mapping, self.area, dim, indices_quad, weight)

            body += stmts
        # ...

        # put the body in for loops of quadrature points
        body = filter_loops(indices_quad, ranges_quad, body,
                            self.boundary,
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


