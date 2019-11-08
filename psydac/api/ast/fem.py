from collections import OrderedDict
from itertools import groupby
import numpy as np

from sympy import symbols, Symbol, IndexedBase
from sympy import Mul, Tuple
from sympy import Matrix, ImmutableDenseMatrix
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
from pyccel.ast import Zeros
from pyccel.ast import Import
from pyccel.ast import DottedName
from pyccel.ast import Nil
from pyccel.ast import Len
from pyccel.ast import If, Is, Return
from pyccel.ast import String, Print, Shape
from pyccel.ast import Comment
from pyccel.ast.core      import _atomic
from pyccel.ast.utilities import build_types_decorator

from sympde.core                 import Constant
from sympde.topology             import ScalarField
from sympde.topology             import VectorField, IndexedVectorField
from sympde.topology             import Boundary, BoundaryVector, NormalVector, TangentVector
from sympde.topology             import ElementArea
from sympde.topology             import LogicalExpr
from sympde.topology             import SymbolicExpr
from sympde.topology             import UndefinedSpaceType
from sympde.topology.space       import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology.space       import ProductSpace
from sympde.topology.space       import ScalarTestFunction
from sympde.topology.space       import VectorTestFunction
from sympde.topology.space       import element_of
from sympde.topology.space       import IndexedTestTrial
from sympde.topology.derivatives import _partial_derivatives
from sympde.topology.derivatives import _logical_partial_derivatives
from sympde.topology.derivatives import get_max_partial_derivatives
from sympde.expr                 import BilinearForm, LinearForm, Functional

from psydac.fem.splines import SplineSpace
from psydac.fem.tensor  import TensorFemSpace
from psydac.fem.vector  import ProductFemSpace

from .basic      import SplBasic
from .evaluation import EvalQuadratureMapping, EvalQuadratureField, EvalQuadratureVectorField
from .utilities  import random_string
from .utilities  import build_pythran_types_header, variables
from .utilities  import compute_normal_vector, compute_tangent_vector
from .utilities  import select_loops, filter_product
from .utilities  import compute_atoms_expr
from .utilities  import is_scalar_field, is_vector_field
from .utilities  import math_atoms_as_str


FunctionalForms = (BilinearForm, LinearForm, Functional)

#==============================================================================
def init_loop_quadrature(indices, ranges, discrete_boundary):
    stmts = []
    if not discrete_boundary:
        return stmts

    # TODO improve using namedtuple or a specific class ? to avoid the 0 index
    #      => make it easier to understand
    quad_mask = [i[0] for i in discrete_boundary]
    quad_ext  = [i[1] for i in discrete_boundary]

    dim = len(indices)
    for i in range(dim-1,-1,-1):
        rx = ranges[i]
        x = indices[i]

        if i in quad_mask:
            i_index = quad_mask.index(i)
            ext = quad_ext[i_index]

            stmts += [Assign(x, 0)]

    return stmts

#==============================================================================
def init_loop_basis(indices, ranges, discrete_boundary):
    stmts = []
    if not discrete_boundary:
        return stmts

    # TODO improve using namedtuple or a specific class ? to avoid the 0 index
    #      => make it easier to understand
    quad_mask = [i[0] for i in discrete_boundary]
    quad_ext  = [i[1] for i in discrete_boundary]

    dim = len(indices)
    for i in range(dim-1,-1,-1):
        rx = ranges[i]
        x = indices[i]

        if i in quad_mask:
            i_index = quad_mask.index(i)
            ext = quad_ext[i_index]

            if ext == -1:
                value = rx.start

            elif ext == 1:
                value = rx.stop - 1

            stmts += [Assign(x, value)]

    return stmts

#==============================================================================
def init_loop_support(indices_elm, n_elements,
                      indices_span, spans, ranges,
                      points_in_elm, points,
                      weights_in_elm, weights,
                      test_basis_in_elm, test_basis,
                      trial_basis_in_elm, trial_basis,
                      is_bilinear, discrete_boundary):
    stmts = []
    if not discrete_boundary:
        return stmts

    # TODO improve using namedtuple or a specific class ? to avoid the 0 index
    #      => make it easier to understand
    quad_mask = [i[0] for i in discrete_boundary]
    quad_ext  = [i[1] for i in discrete_boundary]

    dim = len(indices_elm)
    for i in range(dim-1,-1,-1):
        rx = ranges[i]
        x = indices_elm[i]

        if i in quad_mask:
            i_index = quad_mask.index(i)
            ext = quad_ext[i_index]

            if ext == -1:
                value = rx.start

            elif ext == 1:
                value = rx.stop - 1

            stmts += [Assign(x, value)]

    axis = quad_mask[0]

    # ... assign element index
    ncells = n_elements[axis]
    ie = indices_elm[axis]
    # ...

    # ... assign span index
    i_span = indices_span[axis]
    stmts += [Assign(i_span, spans[axis][ie])]
    # ...

    # ... assign points, weights and basis
    # ie is substitute by 0
    # sympy does not like ':'
    _slice = Slice(None,None)

    stmts += [Assign(points_in_elm[axis], points[axis][0,_slice])]
    stmts += [Assign(weights_in_elm[axis], weights[axis][0,_slice])]
    stmts += [Assign(test_basis_in_elm[axis], test_basis[axis][0,_slice,_slice,_slice])]

    if is_bilinear:
        stmts += [Assign(trial_basis_in_elm[axis], trial_basis[axis][0,_slice,_slice,_slice])]
    # ...

    return stmts

#==============================================================================
# TODO take exponent to 1/dim
def area_eval_mapping(mapping, area, dim, indices_quad, weight):

    stmts = []

    # mapping components and their derivatives
    ops      = _logical_partial_derivatives[:dim]
    elements = [d(mapping[i]) for d in ops for i in range(0, dim)]

    # declarations
    stmts += [Comment('declarations')]
    for e in elements:
        lhs      = SymbolicExpr(e)
        rhs_name = lhs.name + '_values'
        rhs      = IndexedBase(rhs_name)[indices_quad]
        stmts   += [Assign(lhs, rhs)]

    # jacobian determinant
    jac    = SymbolicExpr(mapping.det_jacobian)
    stmts += [AugAssign(area, '+', Abs(jac) * weight)]

    return stmts

#==============================================================================
# target is used when there are multiple expression (domain/boundaries)
class Kernel(SplBasic):

    def __new__(cls, weak_form, kernel_expr, target=None,
                discrete_boundary=None, name=None, boundary_basis=None,
                mapping=None, is_rational_mapping=None,symbolic_space=None, backend=None):

        if not isinstance(weak_form, FunctionalForms):
            raise TypeError('> Expecting a weak formulation')

        if symbolic_space:
            symbolic_space= symbolic_space[0]
            
        unique_scalar_space = True
        if isinstance(symbolic_space, ProductSpace):
            spaces = symbolic_space.spaces
            space = spaces[0]
            unique_scalar_space = all(sp.kind==space.kind for sp in spaces)
        elif isinstance(symbolic_space, VectorFunctionSpace):
            unique_scalar_space = isinstance(symbolic_space.kind, UndefinedSpaceType)

        # ...
        # get the target expr if there are multiple expressions (domain/boundary)
        on_boundary = False
        if target is None:
            if len(kernel_expr) > 1:
                msg = '> weak form has multiple expression, but no target was given'
                raise ValueError(msg)

            e = kernel_expr[0]
            on_boundary = isinstance(e.target, Boundary)
            kernel_expr = e.expr

        else:
            ls = [i for i in kernel_expr if i.target is target]
            e = ls[0]
            on_boundary = isinstance(e.target, Boundary)
            kernel_expr = e.expr
        # ...

        # ...
        if discrete_boundary:
            if not isinstance(discrete_boundary, (tuple, list)):
                raise TypeError('> Expecting a tuple or list for discrete_boundary')

            discrete_boundary = list(discrete_boundary)
            if not isinstance(discrete_boundary[0], (tuple, list)):
                discrete_boundary = [discrete_boundary]
            # discrete_boundary is now a list of lists
        # ...

        # ... discrete_boundary must be given if there are Trace nodes
        if on_boundary and not discrete_boundary:
            raise ValueError('> discrete_bounary must be provided for a boundary Kernel')
        # ...

        # ... default value for boundary_basis is True if on boundary
        if on_boundary and (boundary_basis is None):
            boundary_basis = True
        # ...

        tag = random_string( 8 )
        obj = SplBasic.__new__(cls, tag, name=name,
                               prefix='kernel', mapping=mapping,
                               is_rational_mapping=is_rational_mapping)

        obj._weak_form           = weak_form
        obj._kernel_expr         = kernel_expr
        obj._target              = target
        obj._discrete_boundary   = discrete_boundary
        obj._boundary_basis      = boundary_basis
        obj._area                = None
        obj._user_functions      = []
        obj._backend             = backend
        obj._symbolic_space      = symbolic_space
        obj._unique_scalar_space = unique_scalar_space

        obj._func = obj._initialize()
        return obj

    @property
    def weak_form(self):
        return self._weak_form

    @property
    def kernel_expr(self):
        return self._kernel_expr

    @property
    def target(self):
        return self._target

    @property
    def boundary_basis(self):
        return self._boundary_basis

    @property
    def n_rows(self):
        return self._n_rows

    @property
    def n_cols(self):
        return self._n_cols

    @property
    def max_nderiv(self):
        return self._max_nderiv

    @property
    def zero_terms(self):
        return self._zero_terms

    @property
    def constants(self):
        return self._constants

    @property
    def fields(self):
        return self._fields

    @property
    def fields_coeffs(self):
        return self._fields_coeffs

    @property
    def vector_fields(self):
        return self._vector_fields

    @property
    def vector_fields_coeffs(self):
        return self._vector_fields_coeffs

    @property
    def fields_val(self):
        return self._fields_val

    @property
    def vector_fields_val(self):
        return self._vector_fields_val

    @property
    def mapping_coeffs(self):
        if not self.eval_mapping:
            return ()

        return self.eval_mapping.mapping_coeffs

    @property
    def mapping_values(self):
        if not self.eval_mapping:
            return ()

        return self.eval_mapping.mapping_values

    @property
    def eval_fields(self):
        return self._eval_fields

    @property
    def eval_vector_fields(self):
        return self._eval_vector_fields

    @property
    def eval_mapping(self):
        return self._eval_mapping

    @property
    def area(self):
        return self._area

    @property
    def user_functions(self):
        return self._user_functions

    @property
    def unique_scalar_space(self):
        return self._unique_scalar_space
        
    @property
    def symbolic_space(self):
        return self._symbolic_space
        
    @property
    def backend(self):
        return self._backend

    def build_arguments(self, data):

        other = data

        if self.mapping_values:
            other = self.mapping_values + other

        if self.constants:
            other = other + self.constants

        return self.basic_args + other

    def _initialize(self):
        is_linear   = isinstance(self.weak_form, LinearForm)
        is_bilinear = isinstance(self.weak_form, BilinearForm)
        is_function = isinstance(self.weak_form, Functional)
        unique_scalar_space = self.unique_scalar_space

        expr = self.kernel_expr
        mapping = self.mapping

        # ... area of an element
        area = list(expr.atoms(ElementArea))
        if area:
            assert(len(area) == 1)
            area = area[0]

            self._area = Variable('real', 'area')

            # update exp
            expr = expr.subs(area, self.area)
        # ...

        # ... undefined functions
        funcs = expr.atoms(AppliedUndef)
        if funcs:
            self._user_functions = [f.func for f in list(funcs)]
        # ...

        # ...
        n_rows = 1 ; n_cols = 1
        if is_bilinear:
            if isinstance(expr, (Matrix, ImmutableDenseMatrix)):
                n_rows = expr.shape[0]
                n_cols = expr.shape[1]

        if is_linear:
            if isinstance(expr, (Matrix, ImmutableDenseMatrix)):
                n_rows = expr.shape[0]

        self._n_rows = n_rows
        self._n_cols = n_cols
        # ...

        dim      = self.weak_form.ldim
        dim_test = dim

        if is_bilinear:
            dim_trial = dim
        else:
            dim_trial = 0

        # ... coordinates
        coordinates = self.weak_form.coordinates
        if dim == 1:
            coordinates = [coordinates]
        # ...

        # ...
        constants = tuple(expr.atoms(Constant))
        self._constants = []
        # we need this, since Constant is an extension of Symbol and the type is
        # given as for sympy Symbol
        for c in constants:
            dtype = 'real'
            if c.is_integer:
                dtype = 'int'

            elif c.is_real:
                dtype = 'real'

            elif c.is_complex:
                dtype = 'complex'

            self._constants.append(Variable(dtype, str(c.name)))

        self._constants = tuple(self._constants)
        # ...

        # ...
        atoms_types = (_partial_derivatives,
                       VectorTestFunction,
                       ScalarTestFunction,
                       IndexedTestTrial,
                       ScalarField,
                       VectorField, IndexedVectorField)

        atoms  = _atomic(expr, cls=atoms_types)
        # ...

        # ...
        atomic_expr_field        = [atom for atom in atoms if is_scalar_field(atom)]
        atomic_expr_vector_field = [atom for atom in atoms if is_vector_field(atom)]

        atomic_expr       = [atom for atom in atoms if not( atom in atomic_expr_field ) and
                                                       not( atom in atomic_expr_vector_field)]
        # ...


        field_atoms   = tuple(expr.atoms(ScalarField))
        fields        = []

        # ... create EvalQuadratureField
        self._eval_fields = []
        self._map_stmts_fields = OrderedDict()
        if atomic_expr_field:
            keyfunc = lambda F: F.space.name
            data = sorted(field_atoms, key=keyfunc)
            for space_str, group in groupby(data, keyfunc):
                g_names = set([f.name for f in group])
                fields_expressions = []
                for e in atomic_expr_field:
                    fs = e.atoms(ScalarField)
                    f_names = set([f.name for f in fs])
                    if f_names & g_names:
                        fields_expressions += [e]
                        space = list(fs)[0].space

                eval_field = EvalQuadratureField(space, fields_expressions,
                                       discrete_boundary=self.discrete_boundary,
                                       boundary_basis=self.boundary_basis,
                                       mapping=mapping,backend=self.backend)
                fields += list(eval_field.fields)
                self._eval_fields.append(eval_field)
                for k,v in eval_field.map_stmts.items():
                    self._map_stmts_fields[k] = v

        # update dependencies
        self._dependencies += self.eval_fields

        # TODO: remove these?
        d_subs = dict(zip(_partial_derivatives, _logical_partial_derivatives))
        fields_logical     = tuple(f.subs(d_subs) for f in fields)
        fields_str         = tuple(SymbolicExpr(f).name for f in fields)
        fields_logical_str = tuple(SymbolicExpr(f).name for f in fields_logical)
        # ...

        vector_field_atoms   = tuple(expr.atoms(VectorField))
        vector_fields        = []

        # ... create EvalQuadratureVectorField
        self._eval_vector_fields = []
        if atomic_expr_vector_field:
            keyfunc = lambda F: F.space.name
            data = sorted(vector_field_atoms, key=keyfunc)
            for space_str, group in groupby(data, keyfunc):
                g_names = set([f.name for f in group])
                vector_fields_expressions = []
                for e in atomic_expr_vector_field:
                    fs = e.atoms(VectorField)
                    f_names = set([f.name for f in fs])
                    if f_names & g_names:
                        vector_fields_expressions += [e]
                        space = list(fs)[0].space

                eval_vector_field = EvalQuadratureVectorField(space, vector_fields_expressions,
                                                    discrete_boundary=self.discrete_boundary,
                                                    boundary_basis=self.boundary_basis,
                                                    mapping=mapping,backend=self.backend)
                vector_fields += list(eval_vector_field.vector_fields)
                self._eval_vector_fields.append(eval_vector_field)
                for k,v in eval_vector_field.map_stmts.items():
                    self._map_stmts_fields[k] = v

        # update dependencies
        self._dependencies  += self.eval_vector_fields

        # TODO: remove these?
        vector_fields_logical     = tuple(f.subs(d_subs) for f in vector_fields)
        vector_fields_str         = tuple(SymbolicExpr(f).name for f in vector_fields)
        vector_fields_logical_str = tuple(SymbolicExpr(f).name for f in vector_fields_logical)
        # ...

        # ... TODO add it as a method to basic class
        nderiv = 1
        if isinstance(self.kernel_expr, Matrix):
            n_rows, n_cols = self.kernel_expr.shape
            for i_row in range(0, n_rows):
                for i_col in range(0, n_cols):
                    d = get_max_partial_derivatives(self.kernel_expr[i_row,i_col])
                    nderiv = max(nderiv, max(d.values()))
        else:
            d = get_max_partial_derivatives(self.kernel_expr)
            nderiv = max(nderiv, max(d.values()))

        self._max_nderiv = nderiv
        # ...

        # ... mapping
        mapping = self.mapping
        self._eval_mapping = None
        if mapping:

            if is_bilinear or is_linear:
                space = self.weak_form.test_spaces[0]

            elif is_function:
                space = self.weak_form.space

            eval_mapping = EvalQuadratureMapping(space, mapping,
                                       discrete_boundary=self.discrete_boundary,
                                       boundary_basis=self.boundary_basis,
                                       nderiv=nderiv,
                                       is_rational_mapping=self.is_rational_mapping,
                                       area=self.area,
                                       backend=self.backend)
            self._eval_mapping = eval_mapping

            # update dependencies
            self._dependencies += [self.eval_mapping]
        # ...

        if is_bilinear or is_linear:
            test_function = self.weak_form.test_functions
            if not isinstance(test_function, (tuple, Tuple)):
                test_function = Tuple(test_function)

        elif is_function:
            test_function = element_of(self.weak_form.space, name='Nj')
            test_function = Tuple(test_function)

        # creation of symbolic vars
        if is_bilinear:
            rank = 2*dim

        elif is_linear:
            rank = dim

        elif is_function:
            rank = 1

        if isinstance(expr, Matrix):
            sh   = expr.shape

            # ...
            mats = []
            for i_row in range(0, sh[0]):
                for i_col in range(0, sh[1]):
                    mats.append('mat_{}{}'.format(i_row, i_col))

            mats = variables(mats, dtype='real', rank=rank, cls=IndexedVariable)
            # ...

            # ...
            v = []
            for i_row in range(0, sh[0]):
                for i_col in range(0, sh[1]):
                    v.append('v_{}{}'.format(i_row, i_col))

            v = variables(v, 'real')
            # ...
            expr = expr[:]
            ln   = len(expr)

        else:
            mats = (IndexedVariable('mat_00', dtype='real', rank=rank),)

            v    = (Variable('real', 'v_00'),)
            ln   = 1

            expr = [expr]
            
        # ... looking for 0 terms
        zero_terms = [i for i,e in enumerate(expr) if e == 0]
        self._zero_terms = zero_terms
        
        # ...

        # ... declarations
        fields        = symbols(fields_str)
        fields_logical = symbols(fields_logical_str)

        fields_coeffs = variables(['coeff_{}'.format(f) for f in field_atoms],
                                          dtype='real', rank=dim, cls=IndexedVariable)
        fields_val    = variables(['{}_values'.format(f) for f in fields_logical_str],
                                          dtype='real', rank=dim, cls=IndexedVariable)

        vector_fields        = symbols(vector_fields_str)
        vector_fields_logical = symbols(vector_fields_logical_str)

        vector_field_atoms = [f[i] for f in vector_field_atoms for i in range(0, dim)]
        coeffs = ['coeff_{}'.format(SymbolicExpr(f).name) for f in vector_field_atoms]
        vector_fields_coeffs = variables(coeffs, dtype='real', rank=dim, cls=IndexedVariable)

        vector_fields_val    = variables(['{}_values'.format(f) for f in vector_fields_str],
                                          dtype='real', rank=dim, cls=IndexedVariable)

        test_degrees  = variables('test_d1:%s'%(dim+1),  'int')
        trial_degrees = variables('trial_d1:%s'%(dim+1), 'int')
        test_pads     = variables('test_p1:%s'%(dim+1),  'int')
        trial_pads    = variables('trial_p1:%s'%(dim+1), 'int')

        indices_quad  = variables('g1:%s'%(dim+1),  'int')
        qds_dim       = variables('k1:%s'%(dim+1),  'int')
        indices_test  = variables('il1:%s'%(dim+1), 'int')
        indices_trial = variables('jl1:%s'%(dim+1), 'int')
        wvol          = Variable('real', 'wvol')

        basis_trial   = variables('trial_bs1:%s'%(dim+1),
                                  dtype='real', rank=3, cls=IndexedVariable)
        basis_test    = variables('test_bs1:%s'%(dim+1),
                                  dtype='real', rank=3, cls=IndexedVariable)
        weighted_vols = variables('quad_w1:%s'%(dim+1),
                                  dtype='real', rank=1, cls=IndexedVariable)
        positions     = variables('quad_u1:%s'%(dim+1),
                                  dtype='real', rank=1, cls=IndexedVariable)

        # ...

        # ...
        if is_bilinear:
            self._basic_args = (test_degrees + trial_degrees + trial_pads +
                                basis_test + basis_trial +
                                positions + weighted_vols)
                                
            if self.eval_fields:
                self._basic_args = self._basic_args + fields_val

        if is_linear or is_function:
            self._basic_args = (test_degrees +
                                basis_test +
                                positions + weighted_vols+
                                fields_val + vector_fields_val)
        # ...

        # ...
        if mapping:
            mapping_elements = [SymbolicExpr(i) for i in self.eval_mapping.elements]
            mapping_coeffs   = self.eval_mapping.mapping_coeffs
            mapping_values   = self.eval_mapping.mapping_values
        else:
            mapping_elements = ()
            mapping_coeffs   = ()
            mapping_values   = ()
        # ...

        # ...
        self._fields_val = fields_val
        self._vector_fields_val = vector_fields_val
        self._fields = fields
        self._fields_logical = fields_logical
        self._fields_coeffs = fields_coeffs
        self._vector_fields = vector_fields
        self._vector_fields_logical = vector_fields_logical
        self._vector_fields_coeffs = vector_fields_coeffs
        self._mapping_coeffs = mapping_coeffs
        # ...

        # ranges
        ranges_test  = [Range(test_degrees[i]+1) for i in range(dim_test)]
        ranges_trial = [Range(trial_degrees[i]+1) for i in range(dim_trial)]
        ranges_quad  = [Range(qds_dim[i]) for i in range(dim)]
        # ...

        # body of kernel

        init_basis = OrderedDict()
        init_map   = OrderedDict()

        init_stmts, map_stmts = compute_atoms_expr(atomic_expr,
                                                 indices_quad,
                                                 indices_test,
                                                 indices_trial,
                                                 basis_trial,
                                                 basis_test,
                                                 coordinates,
                                                 test_function,
                                                 is_linear,
                                                 mapping)

        for stmt in init_stmts:
            init_basis[str(stmt.lhs)] = stmt

        for stmt in map_stmts:
            init_map[str(stmt.lhs)] = stmt
         
        if unique_scalar_space:
            ln   = 1
            funcs = [[None]]

        else:
            funcs = [[None]*self._n_cols for i in range(self._n_rows)]

        for indx in range(ln):

            if not unique_scalar_space and indx in zero_terms:
                continue
                
            elif not unique_scalar_space:
                start = indx
                end   = indx + 1
                i_row = indx//self._n_cols
                i_col = indx -i_row*self._n_cols
                
            else:
                i_row = 0
                i_col = 0
                start = 0
                end   = len(expr)
                
            body = []
            init_basis = OrderedDict(sorted(init_basis.items()))
            body += list(init_basis.values())

            if mapping:
                body += [Assign(lhs, rhs[indices_quad]) for lhs, rhs in zip(mapping_elements,
                                                              mapping_values)]

            # ... normal/tangent vectors
            init_map_bnd   = OrderedDict()
            if isinstance(self.target, Boundary):
                vectors = self.kernel_expr.atoms(BoundaryVector)
                normal_vec = symbols('normal_1:%d'%(dim+1))
                tangent_vec = symbols('tangent_1:%d'%(dim+1))

                for vector in vectors:
                    if isinstance(vector, NormalVector):
                        # replace n[i] by its scalar components
                        for i in range(0, dim):
                            expr = [e.subs(vector[i], normal_vec[i]) for e in expr]

                        map_stmts, stmts = compute_normal_vector(normal_vec,
                                                      self.discrete_boundary,
                                                      mapping)

                    elif isinstance(vector, TangentVector):
                        # replace t[i] by its scalar components
                        for i in range(0, dim):
                            expr = [e.subs(vector[i], tangent_vec[i]) for e in expr]

                        map_stmts, stmts = compute_tangent_vector(tangent_vec,
                                                       self.discrete_boundary,
                                                       mapping)

                    for stmt in map_stmts:
                        init_map_bnd[str(stmt.lhs)] = stmt

                    init_map_bnd = OrderedDict(sorted(init_map_bnd.items()))
                    for stmt in list(init_map_bnd.values()):
                        body += [stmt]

                    body += stmts
            # ...

            if mapping:
                inv_jac = Symbol('inv_jac')
                det_jac = Symbol('det_jac')

                if not  isinstance(self.target, Boundary):

                    # ... inv jacobian
                    jac = mapping.det_jacobian
                    jac = SymbolicExpr(jac)
                    # ...

                    body += [Assign(det_jac, jac)]
                    body += [Assign(inv_jac, 1./jac)]

                    # TODO do we use the same inv_jac?
        #            if not isinstance(self.target, Boundary):
        #                body += [Assign(inv_jac, 1/jac)]

                    init_map = OrderedDict(sorted(init_map.items()))
                    for stmt in list(init_map.values()):
                        body += [stmt.subs(1/jac, inv_jac)]

            else:
                body += [Assign(coordinates[i],positions[i][indices_quad[i]])
                         for i in range(dim)]
            # ...

            # ...
            weighted_vol = filter_product(indices_quad, weighted_vols, self.discrete_boundary)
            # ...

            # ...
            # add fields and vector fields
            if not mapping:
                # ... fields
                for i in range(len(fields_val)):
                    body.append(Assign(fields[i],fields_val[i][indices_quad]))
                # ...

                # ... vector_fields
                for i in range(len(vector_fields_val)):
                    body.append(Assign(vector_fields[i],vector_fields_val[i][indices_quad]))
                # ...

            else:
                # ... fields
                for i in range(len(fields_val)):
                    body.append(Assign(fields_logical[i],fields_val[i][indices_quad]))
                # ...

                # ... vector_fields
    #            if vector_fields_val:
    #                print(vector_fields_logical)
    #                print(vector_fields_val)
    #                import sys; sys.exit(0)
                for i in range(len(vector_fields_val)):
                    body.append(Assign(vector_fields_logical[i],vector_fields_val[i][indices_quad]))
                # ...

                # ... substitute expression of inv_jac
                for k,stmt in self._map_stmts_fields.items():
                    body += [stmt.subs(1/jac, inv_jac)]
                # ...

            # TODO use positive mapping all the time? Abs?
            if mapping:
                weighted_vol = weighted_vol * Abs(det_jac)

            body.append(Assign(wvol,weighted_vol))

            for i in range(start, end):
                if not( i in zero_terms ):
                    e = SymbolicExpr(Mul(expr[i],wvol))

                    body.append(AugAssign(v[i],'+', e))
            # ...

            # ... stmts for initializtion: only when boundary is present
            init_stmts = []
            # ...

            # ...
            # put the body in for loops of quadrature points
            init_stmts += init_loop_quadrature( indices_quad, ranges_quad,
                                                self.discrete_boundary )

            body = select_loops( indices_quad, ranges_quad, body,
                                 self.discrete_boundary,
                                 boundary_basis=self.boundary_basis)

            # initialization of intermediate vars
            init_vars = [Assign(v[i],0.0) for i in range(start, end) if not( i in zero_terms )]
            body = init_vars + body
            # ...

            if dim_trial:
                trial_idxs = tuple([indices_trial[i]+trial_pads[i]-indices_test[i] for i in range(dim)])
                idxs = indices_test + trial_idxs
            else:
                idxs = indices_test

            if is_bilinear or is_linear:
                for i in range(start, end):
                    if not( i in zero_terms ):
                        body.append(Assign(mats[i][idxs],v[i]))

            elif is_function:
                for i in range(start, end):
                    if not( i in zero_terms ):
                        body.append(Assign(mats[i][0],v[i]))

            # ...
            # put the body in tests and trials for loops
            if is_bilinear:
                init_stmts += init_loop_basis( indices_test,  ranges_test,  self.discrete_boundary )
                init_stmts += init_loop_basis( indices_trial, ranges_trial, self.discrete_boundary )

                body = select_loops(indices_test, ranges_test, body,
                                    self.discrete_boundary,
                                    boundary_basis=self.boundary_basis)

                body = select_loops(indices_trial, ranges_trial, body,
                                    self.discrete_boundary,
                                    boundary_basis=self.boundary_basis)

            if is_linear:
                init_stmts += init_loop_basis( indices_test, ranges_test, self.discrete_boundary )

                body = select_loops(indices_test, ranges_test, body,
                                    self.discrete_boundary,
                                    boundary_basis=self.boundary_basis)

            # ...

            # ... add init stmts
            body = init_stmts + body
            # ...

            # ...
            # initialization of the matrix
            if is_bilinear or is_linear:
                init_mats = [mats[i][[Slice(None,None)]*(dim_test+dim_trial)] for i in range(start, end) if not( i in zero_terms )]

                init_mats = [Assign(e, 0.0) for e in init_mats]
                body =  init_mats + body

            # call eval field
            for eval_field in self.eval_fields:
                args = test_degrees + basis_test + fields_coeffs + fields_val

                args = eval_field.build_arguments(args)

                body = [FunctionCall(eval_field.func, args)] + body

            imports = []

            # call eval vector_field
            for eval_vector_field in self.eval_vector_fields:
                args = test_degrees + basis_test + vector_fields_coeffs + vector_fields_val
                args = eval_vector_field.build_arguments(args)
                body = [FunctionCall(eval_vector_field.func, args)] + body

            # call eval mapping
            if self.eval_mapping:
                args = (test_degrees + basis_test + mapping_coeffs + mapping_values)
                args = eval_mapping.build_arguments(args)
                body = [FunctionCall(eval_mapping.func, args)] + body

            # init/eval area
            if self.area:
                # evaluation of the area if the mapping is not used
                if not mapping:
                    stmts = [AugAssign(self.area, '+', weighted_vol)]
                    stmts = select_loops( indices_quad, ranges_quad, stmts,
                                          self.discrete_boundary,
                                          boundary_basis=self.boundary_basis)

                    body = stmts + body

                # init area
                body = [Assign(self.area, 0.0)] + body

            # compute length of logical points
            len_quads = [Assign(k, Len(u)) for k,u in zip(qds_dim, positions)]
            body = len_quads + body

            # get math functions and constants
            math_elements = math_atoms_as_str(self.kernel_expr, 'numpy')
            math_imports  = [Import(e, 'numpy') for e in math_elements]

            imports += math_imports
            self._imports = imports
            # function args
            mats_args = tuple([mats[i] for i in range(start, end) if not( i in zero_terms )])
            func_args = fields_coeffs + vector_fields_coeffs + mapping_coeffs + mats_args
                
            func_args = self.build_arguments(func_args)

            decorators = {}
            header = None
            if self.backend['name'] == 'pyccel':
                decorators = {'types': build_types_decorator(func_args)}
            elif self.backend['name'] == 'numba':
                decorators = {'jit':[]}
            elif self.backend['name'] == 'pythran':
                header = build_pythran_types_header(self.name, func_args)
            
            funcs[i_row][i_col] = FunctionDef(self.name+'_'+str(i_row)+str(i_col), list(func_args), [], body,
                                    decorators=decorators,header=header)
   
        return funcs

#==============================================================================
class Assembly(SplBasic):

    def __new__(cls, kernel, name=None, discrete_space=None, comm=None,
                mapping=None, is_rational_mapping=None, backend=None):

        # ... Check arguments
        if not isinstance(kernel, Kernel):
            raise TypeError('> Expecting a kernel')

        if isinstance(discrete_space, (tuple, list)):
            space = discrete_space[0]
        else:
            space = discrete_space

        if not isinstance(space, (SplineSpace, TensorFemSpace, ProductFemSpace)):
            raise NotImplementedError('Only Spline, Tensor and Product spaces are available')
        # ...

        obj = SplBasic.__new__(cls, kernel.tag, name=name,
                               prefix='assembly', mapping=mapping,
                               is_rational_mapping=is_rational_mapping)

        obj._kernel = kernel
        obj._discrete_space = discrete_space
        obj._comm = comm
        obj._discrete_boundary = kernel.discrete_boundary
        obj._backend = backend

        # update dependencies
        obj._dependencies += [kernel]

        obj._func = obj._initialize()
        return obj

    @property
    def weak_form(self):
        return self.kernel.weak_form

    @property
    def kernel(self):
        return self._kernel

    @property
    def discrete_space(self):
        return self._discrete_space

    @property
    def comm(self):
        return self._comm

    @property
    def global_matrices(self):
        return self._global_matrices

    @property
    def backend(self):
        return self._backend

    def build_arguments(self, data):

        other = data

        if self.kernel.constants:
            other = other + self.kernel.constants

        if self.kernel.mapping_coeffs:
            other = self.kernel.mapping_coeffs + other

        return self.basic_args + other

    def _initialize(self):
        kernel = self.kernel
        form   = self.weak_form
        fields = kernel.fields
        fields_coeffs = kernel.fields_coeffs
        vector_fields = kernel.vector_fields
        vector_fields_coeffs = kernel.vector_fields_coeffs
        zero_terms = kernel.zero_terms

        is_linear   = isinstance(self.weak_form, LinearForm)
        is_bilinear = isinstance(self.weak_form, BilinearForm)
        is_function = isinstance(self.weak_form, Functional)
        
        if is_bilinear:
        
            Wh = self.discrete_space[0]
            Vh = self.discrete_space[1]
            is_product_space = isinstance(Wh, ProductFemSpace)
            ln = 1
            if is_product_space:
                ln = len(Wh.spaces)
        else:
        
            Wh = self.discrete_space
            ln = 1
            is_product_space = isinstance(self.discrete_space, ProductFemSpace)
            if is_product_space:
                ln = len(self.discrete_space.spaces)
            
        unique_scalar_space = kernel.unique_scalar_space

        dim    = form.ldim

        n_rows = kernel.n_rows
        n_cols = kernel.n_cols

        axis_bnd = []
        if self.discrete_boundary:
            axis_bnd = [i[0] for i in self.discrete_boundary]


        # ... declarations

        starts         = variables('s1:%s'%(dim+1), 'int')
        ends           = variables('e1:%s'%(dim+1), 'int')

        n_elements     = variables('n_elements_1:%s'%(dim+1), 'int')
        element_starts = variables('element_s1:%s'%(dim+1),   'int')
        element_ends   = variables('element_e1:%s'%(dim+1),   'int')

        indices_elm   = variables('ie1:%s'%(dim+1), 'int')
        indices_span  = variables('is1:%s(1:%s)'%(dim+1, ln+1), 'int')

        test_pads     = variables('test_p1:%s(1:%s)'%(dim+1,ln+1), 'int')
        trial_pads    = variables('trial_p1:%s(1:%s)'%(dim+1,ln+1), 'int')
        
        test_degrees  = variables('test_d1:%s(1:%s)'%(dim+1,ln+1), 'int')
        trial_degrees = variables('trial_d1:%s(1:%s)'%(dim+1,ln+1), 'int')

        indices_il    = variables('il1:%s'%(dim+1), 'int')
        indices_i     = variables('i1:%s'%(dim+1),  'int')
        npts          = variables('n1:%s'%(dim+1),  'int')
        
        trial_basis    = variables('trial_basis_1:%s(1:%s)'%(dim+1,ln+1), dtype='real', rank=4, cls=IndexedVariable)
        test_basis     = variables('test_basis_1:%s(1:%s)'%(dim+1,ln+1), dtype='real', rank=4, cls=IndexedVariable)

        spans          = variables('test_spans_1:%s(1:%s)'%(dim+1,ln+1), dtype='int', rank=1, cls=IndexedVariable)
        quad_orders    = variables( 'k1:%s'%(dim+1), dtype='int')

        trial_basis_in_elm = variables('trial_bs1:%s(1:%s)'%(dim+1,ln+1), dtype='real', rank=3, cls=IndexedVariable)
        test_basis_in_elm  = variables('test_bs1:%s(1:%s)'%(dim+1,ln+1), dtype='real', rank=3, cls=IndexedVariable)

        points_in_elm  = variables('quad_u1:%s'%(dim+1), dtype='real', rank=1, cls=IndexedVariable)
        weights_in_elm = variables('quad_w1:%s'%(dim+1), dtype='real', rank=1, cls=IndexedVariable)

        points   = variables('points_1:%s'%(dim+1), dtype='real', rank=2, cls=IndexedVariable)
        weights  = variables('weights_1:%s'%(dim+1), dtype='real', rank=2, cls=IndexedVariable)
        # ...

        # TODO improve: select args parallel/serial
        if is_bilinear:
            self._basic_args = (n_elements +
                                element_starts + element_ends +
                                starts + ends +
                                npts +
                                quad_orders +
                                test_degrees + trial_degrees +
                                test_pads  + trial_pads +
                                spans +
                                points + weights +
                                test_basis + trial_basis)

        if is_linear or is_function:
            self._basic_args = (n_elements +
                                element_starts + element_ends +
                                starts + ends +
                                npts +
                                quad_orders +
                                test_degrees +
                                test_pads +
                                spans +
                                points + weights +
                                test_basis)
        # ...

        # ...
        if is_bilinear:
            rank = 2*dim

        elif is_linear:
            rank = dim

        elif is_function:
            rank = 1
        # ...

        # ... element matrices
        element_matrices = OrderedDict()
        ind = 0
        for i in range(0, n_rows):
            for j in range(0, n_cols):
                if not( ind in zero_terms ):
                    mat = 'mat_{i}{j}'.format(i=i,j=j)

                    mat = IndexedVariable(mat, dtype='real', rank=rank)

                    element_matrices[i,j] = mat

                ind += 1
        # ...

        # ... global matrices
        ind = 0
        global_matrices = OrderedDict()
        for i in range(0, n_rows):
            for j in range(0, n_cols):
                if not( ind in zero_terms ):
                    mat = 'M_{i}{j}'.format(i=i,j=j)

                    mat = IndexedVariable(mat, dtype='real', rank=rank)

                    global_matrices[i,j] = mat

                ind += 1
        # ...

        # sympy does not like ':'
        _slice = Slice(None,None)

        # assignments
        body  = [Assign(indices_span[i*ln+j], spans[i*ln+j][indices_elm[i]])
                 for i,j in np.ndindex(dim, ln) if not(i in axis_bnd)]
                 
        if self.debug and self.detailed:
            msg = lambda x: (String('> span {} = '.format(x)), x)
            body += [Print(msg(indices_span[i])) for i in range(dim*ln)]

        body += [Assign(points_in_elm[i], points[i][indices_elm[i],_slice])
                 for i in range(dim) if not(i in axis_bnd) ]

        body += [Assign(weights_in_elm[i], weights[i][indices_elm[i],_slice])
                 for i in range(dim) if not(i in axis_bnd) ]

        body += [Assign(test_basis_in_elm[i*ln+j], test_basis[i*ln+j][indices_elm[i],_slice,_slice,_slice])
                 for i,j in np.ndindex(dim,ln) if not(i in axis_bnd) ]

        if is_bilinear:
            body += [Assign(trial_basis_in_elm[i*ln+j], trial_basis[i*ln+j][indices_elm[i],_slice,_slice,_slice])
                     for i,j in np.ndindex(dim,ln) if not(i in axis_bnd) ]

        # ... kernel call
        mats = tuple(element_matrices.values())

        gslices = [Slice(sp-s-d+p, sp+p+1-s) for sp,d,p,s in
                   zip(indices_span[::ln], test_degrees[::ln], test_pads[::ln], starts)]
        f_coeffs  = tuple([f[gslices] for f in fields_coeffs])
        vf_coeffs = tuple([f[gslices] for f in vector_fields_coeffs])
        m_coeffs  = tuple([f[gslices] for f in kernel.mapping_coeffs])

        if is_bilinear:
            if not unique_scalar_space:
                for (i,j), M in element_matrices.items():
                    args = kernel.build_arguments(f_coeffs + vf_coeffs + m_coeffs + (M,))
                    args = list(args)
                    args[:dim] = test_degrees[i::ln]
                    args[dim:2*dim] = trial_degrees[j::ln]
                    args[2*dim:3*dim] = [max(pi,pj) for pi,pj in zip(Vh.spaces[i].degree,Wh.spaces[j].degree)]
                    args[3*dim:4*dim] = test_basis_in_elm[i::ln]
                    args[4*dim:5*dim] = trial_basis_in_elm[j::ln]

                    body += [FunctionCall(kernel.func[i][j], args)]
                    
            else:  
                args = kernel.build_arguments(f_coeffs + vf_coeffs + m_coeffs + mats)
                args = list(args)

                args[:dim] = test_degrees[::ln]
                args[dim:2*dim] = trial_degrees[::ln]
                args[2*dim:3*dim] = trial_pads[j::ln]
                args[3*dim:4*dim] = test_basis_in_elm[::ln]
                args[4*dim:5*dim] = trial_basis_in_elm[::ln]

                body += [FunctionCall(kernel.func[0][0], args)]
                
        else:
            if not unique_scalar_space:
                for (i,j), M in element_matrices.items():
                    
                    args = kernel.build_arguments(f_coeffs + vf_coeffs + m_coeffs + (M,))
                    args = list(args)
                    args[:dim] = test_degrees[i::ln]
                    args[dim:2*dim] = test_basis_in_elm[i::ln]
                    body += [FunctionCall(kernel.func[i][j], args)]
                    
            else:
                args = kernel.build_arguments(f_coeffs + vf_coeffs + m_coeffs + mats)
                args = list(args)
                args[:dim] = test_degrees[::ln]
                args[dim:2*dim] = test_basis_in_elm[::ln]
                body += [FunctionCall(kernel.func[0][0], args)]
            
        # ...

        # ... update global matrices
        lslices = [Slice(None,None)]*dim
        if is_bilinear:
            lslices += [Slice(None,None)]*dim # for assignement


        if is_function:
            lslices = 0
            gslices = 0

        for (i,j), M in global_matrices.items():
           
            mat = element_matrices[i,j]
            local_test_degrees = test_degrees[i::ln]
            local_indices_span = indices_span[i::ln]
            local_test_pads = test_pads[i::ln]
                

            if is_bilinear:

                if ( self.comm is None ):
                    gslices = [Slice(sp-d+p,sp+p+1) for sp,d,p in zip(local_indices_span, local_test_degrees,local_test_pads)]
    
                else:
                    gslices = [Slice(sp-s-d+p,sp+p+1-s) for sp,d,p,s in zip(local_indices_span,
                                                                   local_test_degrees,
                                                                   local_test_pads,
                                                                       starts)]

                gslices += [Slice(None,None)]*dim # for assignement

            if is_linear:
                if ( self.comm is None ):
                    gslices = [Slice(sp-d+p,sp+p+1) for sp,d,p in zip(local_indices_span, local_test_degrees,local_test_pads)]

                else:
                    gslices = [Slice(sp-s-d+p,sp+p+1-s) for sp,d,p,s in zip(local_indices_span,
                                                                   local_test_degrees,
                                                                   local_test_pads,
                                                                   starts)]
            
            stmt = AugAssign(M[gslices], '+', mat[lslices])

            body += [stmt]
        # ...

        # ... loop over elements
        if is_function:
            ranges_elm  = [Range(s, e+1) for s,e in zip(element_starts,
                                                      element_ends)]

        else:
            ranges_elm  = [Range(0, n_elements[i]) for i in range(dim)]

        # TODO call init_loops
        init_stmts = init_loop_support( indices_elm, n_elements,
                                       indices_span, spans, ranges_elm,
                                       points_in_elm, points,
                                       weights_in_elm, weights,
                                       test_basis_in_elm, test_basis,
                                       trial_basis_in_elm, trial_basis,
                                       is_bilinear, self.discrete_boundary )

        body = select_loops(indices_elm, ranges_elm, body,
                            self.kernel.discrete_boundary, boundary_basis=False)

        body = init_stmts + body

        # ...

        # ... prelude
        imports = []

        # import zeros from numpy
        stmt = Import('zeros', 'numpy')
        imports += [stmt]

        # import product from itertools
        stmt = Import('product', 'itertools')
        imports += [stmt]

        prelude = []
        # allocate element matrices

        for (i,j),mat in element_matrices.items():

            orders  = [p+1 for p in test_degrees[i::ln]]
            spads   = [2*p+1 for p in test_pads[j::ln]]
            
            if is_bilinear:
                if not unique_scalar_space:
                    spads = [2*max(pi,pj)+1 for pi,pj in zip(Vh.spaces[i].degree,Wh.spaces[j].degree)]
                    
                args  = tuple(orders + spads)

            if is_linear:
                args = tuple(orders)

            if is_function:
                args = tuple([1])

            stmt = Assign(mat, Zeros(args))
            prelude += [stmt]



        # allocate mapping values
        if self.kernel.mapping_values:
            for v in self.kernel.mapping_values:
                stmt = Assign(v, Zeros(quad_orders))
                prelude += [stmt]

        # TODO allocate field values
        if self.kernel.fields:
            fields_shape = tuple(FunctionCall('len',[p[0,Slice(None,None)]]) for p in points)
            for F_value in self.kernel.fields_val:
                prelude += [Assign(F_value, Zeros(fields_shape))]

        if self.kernel.vector_fields_val:
            fields_shape = tuple(FunctionCall('len',[p[0,Slice(None,None)]]) for p in points)
            for F_value in self.kernel.vector_fields_val:
                prelude += [Assign(F_value, Zeros(fields_shape))]

        # ...
        if self.debug:
            for ij, M in global_matrices.items():
                i,j = ij
                prelude += [Print((String('> shape {} = '.format(M)), Shape(M)))]
        # ...

        # ...
        body = prelude + body
        # ...

        # ...
        mats = tuple(global_matrices.values())
        self._global_matrices = global_matrices
        # ...
        self._imports = imports
        # function args
        func_args = self.build_arguments(fields_coeffs + vector_fields_coeffs + mats)

        decorators = {}
        header = None
        if self.backend['name'] == 'pyccel':
            decorators = {'types': build_types_decorator(func_args),'external_call':[]}
        elif self.backend['name'] == 'numba':
            decorators = {'jit':[]}
        elif self.backend['name'] == 'pythran':
            header = build_pythran_types_header(self.name, func_args)

        return FunctionDef(self.name, list(func_args), [], body,
                           decorators=decorators,header=header)


#==============================================================================
class Interface(SplBasic):

    def __new__(cls, assembly, name=None, backend=None,
                discrete_space=None, comm=None, mapping=None, is_rational_mapping=None):

        if not isinstance(assembly, Assembly):
            raise TypeError('> Expecting an Assembly')

        obj = SplBasic.__new__(cls, assembly.tag, name=name,
                               prefix='interface', mapping=mapping,
                               is_rational_mapping=is_rational_mapping)

        obj._assembly = assembly
        obj._backend = backend
        obj._discrete_space = discrete_space
        obj._comm = comm

        dim = assembly.weak_form.ldim


        # update dependencies
	# TODO uncomment later
        #lo_dot = LinearOperatorDot(dim, backend)
        #v_dot  = VectorDot(dim, backend)

        #obj._dots = [lo_dot, v_dot]
        #obj._dependencies += [assembly, lo_dot, v_dot]

        obj._dependencies += [assembly]

        obj._func = obj._initialize()
        return obj

    @property
    def weak_form(self):
        return self._assembly.weak_form
        
    @property
    def space(self):
        return self._assembly.kernel.symbolic_space

    @property
    def assembly(self):
        return self._assembly

    @property
    def backend(self):
        return self._backend

    @property
    def discrete_space(self):
        return self._discrete_space

    @property
    def comm(self):
        return self._comm

    @property
    def max_nderiv(self):
        return self.assembly.kernel.max_nderiv

    def build_arguments(self, data):
        # data must be at the end, since they are optional
        return self.basic_args + data

    @property
    def in_arguments(self):
        return self._in_arguments

    @property
    def inout_arguments(self):
        return self._inout_arguments

    @property
    def user_functions(self):
        return self.assembly.kernel.user_functions

# TODO uncomment later
    #@property
    #def dots(self):
    #    return self._dots


    def _initialize(self):
        form = self.weak_form
        assembly = self.assembly
        global_matrices = assembly.global_matrices
        fields = tuple(form.expr.atoms(ScalarField))
        fields = sorted(fields, key=lambda x: str(x.name))
        fields = tuple(fields)
        zero_terms = assembly.kernel.zero_terms

        vector_fields = tuple(form.expr.atoms(VectorField))
        vector_fields = sorted(vector_fields, key=lambda x: str(x.name))
        vector_fields = tuple(vector_fields)

        is_linear   = isinstance(self.weak_form, LinearForm)
        is_bilinear = isinstance(self.weak_form, BilinearForm)
        is_function = isinstance(self.weak_form, Functional)

        dim = form.ldim
        
        if is_bilinear:
            Wh = self.discrete_space[0]
            Vh = self.discrete_space[1]
            
        else:
            Wh = self.discrete_space
            
        is_product_fem_space = isinstance(Wh, ProductFemSpace)
        unique_scalar_space = assembly.kernel.unique_scalar_space

        # ... declarations

        test_space = Symbol('W')
        trial_space = Symbol('V')
        grid = Symbol('grid')
        test_basis_values = Symbol('test_basis_values')
        trial_basis_values = Symbol('trial_basis_values')

        if is_bilinear:
            basis_values = (test_basis_values, trial_basis_values)

        else:
            basis_values = (test_basis_values,)

        if is_bilinear:
            spaces = (test_space, trial_space)
            test_vector_space = DottedName(test_space, 'vector_space')
            trial_vector_space = DottedName(trial_space, 'vector_space')

            ln = 1
            if is_product_fem_space: 
                ln = len(Wh.spaces)         
                test_vector_space = DottedName(test_vector_space, 'spaces')
                trial_vector_space = DottedName(trial_vector_space, 'spaces')
            # ...

        if is_linear or is_function:
            test_vector_space = DottedName(test_space, 'vector_space')
            spaces = (test_space,)
            ln = 1
            if is_product_fem_space:
                ln = len(Wh.spaces)
                test_vector_space = DottedName(test_vector_space, 'spaces')
            # ...

        n_elements     = variables('n_elements_1:%s'%(dim+1), 'int')
        starts         = variables('s1:%s'%(dim+1), 'int')
        ends           = variables('e1:%s'%(dim+1), 'int')
        npts           = variables('n1:%s'%(dim+1), 'int')
        element_starts = variables('element_s1:%s'%(dim+1), 'int')
        element_ends   = variables('element_e1:%s'%(dim+1), 'int')

        test_degrees   = variables('test_d1:%s(1:%s)'%(dim+1,ln+1), 'int')
        trial_degrees  = variables('trial_d1:%s(1:%s)'%(dim+1,ln+1), 'int')
        
        test_pads      = variables('test_p1:%s(1:%s)'%(dim+1,ln+1), 'int')
        trial_pads     = variables('trial_p1:%s(1:%s)'%(dim+1,ln+1), 'int')
        
        trial_basis    = variables('trial_basis_1:%s(1:%s)'%(dim+1,ln+1), dtype='real', rank=4, cls=IndexedVariable)
        test_basis     = variables('test_basis_1:%s(1:%s)'%(dim+1,ln+1), dtype='real', rank=4, cls=IndexedVariable)

        spans          = variables('test_spans_1:%s(1:%s)'%(dim+1,ln+1), dtype='int', rank=1, cls=IndexedVariable)
        quad_orders    = variables( 'k1:%s'%(dim+1), dtype='int')

        points         = variables('points_1:%s'%(dim+1),  dtype='real', rank=2, cls=IndexedVariable)
        weights        = variables('weights_1:%s'%(dim+1), dtype='real', rank=2, cls=IndexedVariable)


        test_spaces, trial_spaces = symbols('test_spaces, trial_spaces', cls=IndexedBase)
        spans_attr , basis_attr   = symbols('spans, basis', cls=IndexedBase)
        pads                      = symbols('pads')
        
	# TODO uncomment later
        #dots           = symbols('lo_dot v_dot')
        #dot            = Symbol('dot')

        mapping = ()
        if self.mapping:
            mapping = Symbol('mapping')
        # ...

        # ...
        self._basic_args = spaces + (grid,) + basis_values
        # ...

        spaces = IndexedBase('spaces')
        
        # ... interface body
        body = []
        body += [Assign(test_spaces, test_vector_space)]
        
        if is_bilinear:
            body += [Assign(trial_spaces, trial_vector_space)]
            
        # ... grid data
        body += [Assign(n_elements,     DottedName(grid, 'n_elements'))]
        body += [Assign(points,         DottedName(grid, 'points'))]
        body += [Assign(weights,        DottedName(grid, 'weights'))]
        body += [Assign(quad_orders,    DottedName(grid, 'quad_order'))]
        body += [Assign(element_starts, DottedName(grid, 'local_element_start'))]
        body += [Assign(element_ends,   DottedName(grid, 'local_element_end'))]
        # ...

        # ... basis values
        if is_product_fem_space:
            for i in range(ln):
                body += [Assign(spans[i::ln],      DottedName(test_basis_values, spans_attr[i]))]
                body += [Assign(test_basis[i::ln], DottedName(test_basis_values, basis_attr[i]))]   
       
        else:
            body += [Assign(spans,      DottedName(test_basis_values, spans_attr))]
            body += [Assign(test_basis, DottedName(test_basis_values, basis_attr))]

        if is_bilinear:
            if is_product_fem_space:
                for i in range(ln):
                    body += [Assign(trial_basis[i::ln], DottedName(trial_basis_values, basis_attr[i]))]   
            else:
                body += [Assign(trial_basis, DottedName(trial_basis_values, basis_attr))]
        # ...

        # ... getting data from fem space
        if is_product_fem_space:
            for i in range(ln):
                body += [Assign(test_degrees[i::ln], DottedName(test_space,spaces[i], 'degree'))]
                body += [Assign(test_pads   [i::ln], DottedName(test_spaces[i], 'pads'))]
            
        else:
            body += [Assign(test_degrees, DottedName(test_space, 'degree'))]
            body += [Assign(test_pads   , DottedName(test_spaces, 'pads'))]
            
        if is_bilinear:
        
            if is_product_fem_space:
                for i in range(ln):
                    body += [Assign(trial_degrees[i::ln], DottedName(trial_space,spaces[i], 'degree'))]
                    body += [Assign(trial_pads   [i::ln], DottedName(trial_spaces[i], 'pads'))]
            else:
                body += [Assign(trial_degrees, DottedName(trial_space, 'degree'))]
                body += [Assign(trial_pads   , DottedName(trial_spaces, 'pads'))]

        if is_product_fem_space:
            body += [Assign(starts, DottedName(test_spaces[0], 'starts'))]
            body += [Assign(ends,   DottedName(test_spaces[0], 'ends'))]
            body += [Assign(npts,   DottedName(test_spaces[0], 'npts'))]
        
        else:
            body += [Assign(starts, DottedName(test_spaces, 'starts'))]
            body += [Assign(ends,   DottedName(test_spaces, 'ends'))]
            body += [Assign(npts,   DottedName(test_spaces, 'npts'))]
        # ...
        if mapping:
            # we limit the range to dim, since the last element can be the
            # weights when using NURBS
            for i, coeff in enumerate(assembly.kernel.mapping_coeffs[:dim]):
                component = IndexedBase(DottedName(mapping, '_fields'))[i]
                c_var = DottedName(component, '_coeffs', '_data')
                body += [Assign(coeff, c_var)]

            # NURBS case
            if self.is_rational_mapping:
                coeff = assembly.kernel.mapping_coeffs[-1]

                component = DottedName(mapping, '_weights_field')
                c_var = DottedName(component, '_coeffs', '_data')
                body += [Assign(coeff, c_var)]
        # ...

        # ...
        imports = []
        if not is_function:
            if is_bilinear:
                imports += [Import('StencilMatrix', 'psydac.linalg.stencil')]

            if is_linear:
                imports += [Import('StencilVector', 'psydac.linalg.stencil')]


            for ij,M in global_matrices.items():
                (i,j) = ij
                if_cond = Is(M, Nil())
                if is_bilinear:
                    if is_product_fem_space:
                        spj = Vh.spaces[j]
                        spi = Wh.spaces[i]
                        pads_args = tuple(max(pi,pj) for pi,pj in zip(spj.degree,spi.degree))
                        args = [trial_spaces[j], test_spaces[i], Assign(pads,pads_args)]
                    else:
                        args = [trial_spaces , test_spaces]
                        
                    if_body = [Assign(M, FunctionCall('StencilMatrix', args))]
                    # TODO uncomment later
                    #if_body.append(Assign(DottedName(M,'_dot'),dots[0]))

                if is_linear:
                    if is_product_fem_space:
                        args = [test_spaces[i]]
                    else:
                        args = [test_spaces]
                        
                    if_body = [Assign(M, FunctionCall('StencilVector', args))]
                    # TODO uncomment later
                    #if_body.append(Assign(DottedName(M,'_dot'),dots[1]))

                stmt = If((if_cond, if_body))
                body += [stmt]

        else:
            imports += [Import('zeros', 'numpy')]
            for M in global_matrices.values():
                body += [Assign(M, Zeros(1))]
        # ...

        # ...
        self._inout_arguments = list(global_matrices.values())
        self._in_arguments = list(self.assembly.kernel.constants) + list(fields) + list(vector_fields)
        # ...

        # ... call to assembly
        if is_bilinear or is_linear:
            mat_data = [DottedName(M, '_data') for M in global_matrices.values()]

        elif is_function:
            mat_data = [M for M in global_matrices.values()]

        mat_data       = tuple(mat_data)

        field_data     = [DottedName(F, '_coeffs', '_data') for F in fields]
        field_data     = tuple(field_data)

        vector_field_data     = [DottedName(F, '_coeffs[{}]'.format(i),
                                            '_data') for F in
                                 vector_fields for i in range(0, dim)]
        vector_field_data     = tuple(vector_field_data)

        args = assembly.build_arguments(field_data + vector_field_data + mat_data)

        body += [FunctionCall(assembly.func, args)]
        # ...

        # ... IMPORTANT: ghost regions must be up-to-date
        if not( self.comm is None ):
            if is_linear:
                for M in global_matrices.values():
                    f_name = '{}.update_ghost_regions'.format(str(M.name))
                    stmt = FunctionCall(f_name, [])
                    body += [stmt]
        # ...

        # ... results
        if is_bilinear or is_linear:
            n_rows = self.assembly.kernel.n_rows
            n_cols = self.assembly.kernel.n_cols

            if n_rows * n_cols > 1:
                if is_bilinear:
                    L = IndexedBase('L')

                    imports += [Import('BlockMatrix', 'psydac.linalg.block')]

                    # ... TODO this is a duplicated code => use a function to define
                    # global_matrices
                    ind = 0
                    d = {}
                    for i in range(0, n_rows):
                        for j in range(0, n_cols):
                            if not( ind in zero_terms ):
                                mat = IndexedBase('M_{i}{j}'.format(i=i,j=j))
                                d[(i,j)] = mat

                            ind += 1
                    # ...

                    # ... create product space
                    test_vector_space  = DottedName(test_space , 'vector_space')
                    trial_vector_space = DottedName(trial_space, 'vector_space')
                    body += [Assign(L, FunctionCall('BlockMatrix', [test_vector_space, trial_vector_space]))]
                    d = OrderedDict(sorted(d.items()))
                    for k,v in d.items():
                        body += [Assign(L[k], v)]


                elif is_linear:
                    L = IndexedBase('L')

                    # ... TODO this is a duplicated code => use a function to define
                    # global_matrices
                    # n_cols is equal to 1

                    ind = 0
                    d = {}
                    j = 0
                    for i in range(0, n_rows):
                        if not( ind in zero_terms ):
                            mat = IndexedBase('M_{i}{j}'.format(i=i,j=j))
                            d[i] = mat

                        ind += 1
                    # ...

                    imports += [Import('BlockVector', 'psydac.linalg.block')]

                    # ... create product space
                    test_vector_space = DottedName(test_space, 'vector_space')
                    # ...

                    body += [Assign(L, FunctionCall('BlockVector', [test_vector_space]))]
                    d = OrderedDict(sorted(d.items()))
                    for k,v in d.items():
                        body += [Assign(L[k], v)]

                body += [Return(L)]

            else:
                M = list(global_matrices.values())[0]
                body += [Return(M)]

        elif is_function:
            if len(global_matrices) == 1:
                M = list(global_matrices.values())[0]
                body += [Return(M[0])]

            else:
                body += [Return(M[0]) for M in global_matrices.values()]
        # ...

        # ... arguments
        if is_bilinear or is_linear:
            mats = [Assign(M, Nil()) for M in global_matrices.values()]
            mats = tuple(mats)

        elif is_function:
            mats = ()

        if mapping:
            mapping = (mapping,)

        if self.assembly.kernel.constants:
            constants = self.assembly.kernel.constants
            args = mapping + constants + fields + vector_fields + mats

        else:
            args = mapping + fields + vector_fields + mats

        func_args = self.build_arguments(args)
        # ...

        self._imports = imports
        
        return FunctionDef(self.name, list(func_args), [], body)


# TODO uncomment later
class LinearOperatorDot(SplBasic):

    def __new__(cls, ndim, backend=None):


        obj = SplBasic.__new__(cls, 'dot',name='lo_dot',prefix='lo_dot')
        obj._ndim = ndim
        obj._backend = backend
        obj._func = obj._initilize()
        return obj

    @property
    def ndim(self):
        return self._ndim

    @property
    def func(self):
        return self._func

    @property
    def backend(self):
        return self._backend


    def _initilize(self):

        ndim = self.ndim
        nrows           = variables('n1:%s'%(ndim+1),  'int')
        pads            = variables('p1:%s'%(ndim+1),  'int')
        indices1        = variables('ind1:%s'%(ndim+1),'int')
        indices2        = variables('i1:%s'%(ndim+1),  'int')
        extra_rows      = variables('extra_rows','int',rank=1,cls=IndexedVariable)

        ex,v            = variables('ex','int'), variables('v','real')
        x, out          = variables('x, out','real',cls=IndexedVariable, rank=ndim)
        mat             = variables('mat','real',cls=IndexedVariable, rank=2*ndim)

        body = []
        ranges = [Range(2*p+1) for p in pads]
        target = Product(*ranges)


        v1 = x[tuple(i+j for i,j in zip(indices1,indices2))]
        v2 = mat[tuple(i+j for i,j in zip(indices1,pads))+tuple(indices2)]
        v3 = out[tuple(i+j for i,j in zip(indices1,pads))]

        body = [AugAssign(v,'+' ,Mul(v1,v2))]
        body = [For(indices2, target, body)]
        body.insert(0,Assign(v, 0.0))
        body.append(Assign(v3,v))
        ranges = [Range(i) for i in nrows]
        target = Product(*ranges)
        body = [For(indices1,target,body)]

        for dim in range(ndim):
            body.append(Assign(ex,extra_rows[dim]))


            v1 = [i+j for i,j in zip(indices1, indices2)]
            v2 = [i+j for i,j in zip(indices1, pads)]
            v1[dim] += nrows[dim]
            v2[dim] += nrows[dim]
            v3 = v2
            v1 = x[tuple(v1)]
            v2 = mat[tuple(v2)+ indices2]
            v3 = out[tuple(v3)]

            rows = list(nrows)
            rows[dim] = ex
            ranges = [2*p+1 for p in pads]
            ranges[dim] -= indices1[dim] + 1
            ranges =[Range(i) for i in ranges]
            target = Product(*ranges)

            for_body = [AugAssign(v, '+',Mul(v1,v2))]
            for_body = [For(indices2, target, for_body)]
            for_body.insert(0,Assign(v, 0.0))
            for_body.append(Assign(v3,v))

            ranges = [Range(i) for i in rows]
            target = Product(*ranges)
            body += [For(indices1, target, for_body)]


        func_args =  (extra_rows, mat, x, out) + nrows + pads

        self._imports = [Import('product','itertools')]

        decorators = {}
        header = None

        if self.backend['name'] == 'pyccel':
            decorators = {'types': build_types_decorator(func_args), 'external_call':[]}
        elif self.backend['name'] == 'numba':
            decorators = {'jit':[]}
        elif self.backend['name'] == 'pythran':
            header = build_pythran_types_header(self.name, func_args)

        return FunctionDef(self.name, list(func_args), [], body,
                           decorators=decorators,header=header)


class VectorDot(SplBasic):

    def __new__(cls, ndim, backend=None):


        obj = SplBasic.__new__(cls, 'dot', name='v_dot', prefix='v_dot')
        obj._ndim = ndim
        obj._backend = backend
        obj._func = obj._initilize()
        return obj

    @property
    def ndim(self):
        return self._ndim

    @property
    def func(self):
        return self._func

    @property
    def backend(self):
        return self._backend

    def _initilize(self):

        ndim = self.ndim

        indices = variables('i1:%s'%(ndim+1),'int')
        dims    = variables('n1:%s'%(ndim+1),'int')
        pads    = variables('p1:%s'%(ndim+1),'int')
        out     = variables('out','real')
        x1,x2   = variables('x1, x2','real',rank=ndim,cls=IndexedVariable)

        body = []
        ranges = [Range(p,n-p) for n,p in zip(dims,pads)]
        target = Product(*ranges)


        v1 = x1[indices]
        v2 = x2[indices]

        body = [AugAssign(out,'+' ,Mul(v1,v2))]
        body = [For(indices, target, body)]
        body.insert(0,Assign(out, 0.0))
        body.append(Return(out))

        func_args =  (x1, x2) + pads + dims

        self._imports = [Import('product','itertools')]

        decorators = {}
        header = None

        if self.backend['name'] == 'pyccel':
            decorators = {'types': build_types_decorator(func_args), 'external':[]}
        elif self.backend['name'] == 'numba':
            decorators = {'jit':[]}
        elif self.backend['name'] == 'pythran':
            header = build_pythran_types_header(self.name, func_args)

        return FunctionDef(self.name, list(func_args), [], body,
                           decorators=decorators,header=header)
