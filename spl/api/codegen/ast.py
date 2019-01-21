# TODO - in pycode, when printing a For loop, we should check if end == start + 1
#        in which case, we shall replace the For statement by its body and subs
#        the iteration index by its value (start)

from collections import OrderedDict
from itertools import groupby
import string
import random
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

from .utilities import build_pythran_types_header, variables


from sympde.core import Cross_3d
from sympde.core import Constant
from sympde.core.math import math_atoms_as_str
from sympde.calculus import grad
from sympde.topology import Mapping
from sympde.topology import Field
from sympde.topology import VectorField, IndexedVectorField
from sympde.topology import Boundary, BoundaryVector, NormalVector, TangentVector
from sympde.topology import Covariant, Contravariant
from sympde.topology import ElementArea
from sympde.topology.derivatives import _partial_derivatives
from sympde.topology.derivatives import get_max_partial_derivatives
from sympde.topology.space import FunctionSpace
from sympde.topology.space import TestFunction
from sympde.topology.space import VectorTestFunction
from sympde.topology.space import IndexedTestTrial
from sympde.topology.space import Trace
from sympde.topology.derivatives import print_expression
from sympde.topology.derivatives import get_atom_derivatives
from sympde.topology.derivatives import get_index_derivatives
from sympde.expr import BilinearForm, LinearForm, Integral, BasicForm
from sympde.printing.pycode import pycode  # TODO remove from here

from spl.fem.splines import SplineSpace
from spl.fem.tensor  import TensorFemSpace
from spl.fem.vector  import ProductFemSpace


FunctionalForms = (BilinearForm, LinearForm, Integral)

#==============================================================================
def random_string( n ):
    chars    = string.ascii_lowercase + string.digits
    selector = random.SystemRandom()
    return ''.join( selector.choice( chars ) for _ in range( n ) )


#==============================================================================
def compute_normal_vector(vector, discrete_boundary, mapping):
    dim = len(vector)
    pdim = dim - len(discrete_boundary)
    if len(discrete_boundary) > 1: raise NotImplementedError('TODO')

    face = discrete_boundary[0]
    axis = face[0] ; ext = face[1]

    map_stmts = []
    body = []

    if not mapping:

        values = np.zeros(dim)
        values[axis] = ext

    else:
        M = mapping
        inv_jac = Symbol('inv_jac')
        det_jac = Symbol('det_jac')

        # ... construct jacobian on manifold
        lines = []
        n_row,n_col = M.jacobian.shape
        range_row = [i for i in range(0,n_row) if not(i == axis)]
        range_col = range(0,n_col)
        for i_row in range_row:
            line = []
            for i_col in range_col:
                line.append(M.jacobian[i_col, i_row])

            lines.append(line)

        J = Matrix(lines)
        # ...

        # ...
        ops = _partial_derivatives[:dim]
        elements = [d(M[i]) for d in ops for i in range(0, dim)]
        for e in elements:
            new = print_expression(e, mapping_name=False)
            new = Symbol(new)
            J = J.subs(e, new)
        # ...

        if dim == 1:
            raise NotImplementedError('TODO')

        elif dim == 2:
            J = J[0,:]
            # TODO shall we use sympy_sqrt here? is there any difference in
            # Fortran between sqrt and Pow(, 1/2)?
            j = (sum(J[i]**2 for i in range(0, dim)))**(1/2)

            values = [inv_jac*J[1], -inv_jac*J[0]]

        elif dim == 3:

            x_s = J[0,:]
            x_t = J[1,:]

            values = Cross_3d(x_s, x_t)
            j = (sum(J[i]**2 for i in range(0, dim)))**(1/2)
            values = [inv_jac*v for v in values]


        # change the orientation
        values = [ext*i for i in values]

        map_stmts += [Assign(det_jac, j)]
        map_stmts += [Assign(inv_jac, 1./j)]

    for i in range(0, dim):
        body += [Assign(vector[i], values[i])]

#    print(map_stmts)
#    print(body)
#    import sys; sys.exit(0)

    return map_stmts, body

#==============================================================================
def compute_tangent_vector(vector, discrete_boundary, mapping):
    raise NotImplementedError('TODO')


#==============================================================================
# TODO remove it later
def filter_loops(indices, ranges, body, discrete_boundary, boundary_basis=False):

    quad_mask = []
    quad_ext = []
    if discrete_boundary:
        # TODO improve using namedtuple or a specific class ? to avoid the 0 index
        #      => make it easier to understand
        quad_mask = [i[0] for i in discrete_boundary]
        quad_ext  = [i[1] for i in discrete_boundary]

        # discrete_boundary gives the perpendicular indices, then we need to
        # remove them from directions

    dim = len(indices)
    for i in range(dim-1,-1,-1):
        rx = ranges[i]
        x = indices[i]
        start = rx.start
        end   = rx.stop

        if i in quad_mask:
            i_index = quad_mask.index(i)
            ext = quad_ext[i_index]
            if ext == -1:
                end = start + 1

            elif ext == 1:
                start = end - 1
            else:
                raise ValueError('> Wrong value for ext. It should be -1 or 1')

        rx = Range(start, end)
        body = [For(x, rx, body)]
    body = fusion_loops(body)
    return body


#==============================================================================
def select_loops(indices, ranges, body, discrete_boundary, boundary_basis=False):

    quad_mask = []
    quad_ext = []
    if discrete_boundary:
        # TODO improve using namedtuple or a specific class ? to avoid the 0 index
        #      => make it easier to understand
        quad_mask = [i[0] for i in discrete_boundary]
        quad_ext  = [i[1] for i in discrete_boundary]

        # discrete_boundary gives the perpendicular indices, then we need to
        # remove them from directions

    dim = len(indices)
    dims = [i for i in range(dim-1,-1,-1) if not( i in quad_mask )]

    for i in dims:
        rx = ranges[i]
        x = indices[i]
        start = rx.start
        end   = rx.stop

        rx = Range(start, end)
        body = [For(x, rx, body)]

    body = fusion_loops(body)
    return body

def fusion_loops(loops):
    ranges = []
    indices = []
    loops_cp = loops

    while len(loops) == 1 and isinstance(loops[0], For):

        loops = loops[0]
        target = loops.target
        iterable = loops.iterable

        if isinstance(iterable, Product):
            ranges  += list(iterable.elements)
            indices += list(target)
            if not isinstance(target,(tuple,list,Tuple)):
                raise ValueError('target must be a list or a tuple of indices')

        elif isinstance(iterable, Range):
            ranges.append(iterable)
            indices.append(target)
        else:
            raise TypeError('only range an product are supported')

        loops = loops.body

    if len(ranges)>1:
        return [For(indices, Product(*ranges), loops)]
    else:
        return loops_cp

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
def filter_product(indices, args, discrete_boundary):

    mask = []
    ext = []
    if discrete_boundary:
        # TODO improve using namedtuple or a specific class ? to avoid the 0 index
        #      => make it easier to understand
        mask = [i[0] for i in discrete_boundary]
        ext  = [i[1] for i in discrete_boundary]

        # discrete_boundary gives the perpendicular indices, then we need to
        # remove them from directions

    dim = len(indices)
    args = [args[i][indices[i]] for i in range(dim) if not(i in mask)]

    return Mul(*args)


#==============================================================================
def compute_atoms_expr(atom, indices_quad, indices_test,
                       indices_trial, basis_trial,
                       basis_test, cords, test_function,
                       is_linear,
                       mapping):

    cls = (_partial_derivatives,
           VectorTestFunction,
           TestFunction,
           IndexedTestTrial)

    dim  = len(indices_test)

    if not isinstance(atom, cls):
#        print(atom, type(atom))
#        import sys; sys.exit(0)
        raise TypeError('atom must be of type {}'.format(str(cls)))

    orders = [0 for i in range(0, dim)]
    p_indices = get_index_derivatives(atom)

    # ...
    def _get_name(atom):
        atom_name = None
        if isinstance( atom, TestFunction ):
            atom_name = str(atom.name)

        elif isinstance( atom, VectorTestFunction ):
            atom_name = str(atom.name)

        elif isinstance( atom, IndexedTestTrial ):
            atom_name = str(atom.base.name)

        else:
            raise TypeError('> Wrong type')

        return atom_name
    # ...

    if isinstance(atom, _partial_derivatives):
        a = get_atom_derivatives(atom)
        atom_name = _get_name(a)

        orders[atom.grad_index] = p_indices[atom.coordinate]

    else:
        atom_name = _get_name(atom)

    test_names = [_get_name(i) for i in test_function]
    test = atom_name in test_names

    if test or is_linear:
        basis  = basis_test
        idxs   = indices_test
    else:
        basis  = basis_trial
        idxs   = indices_trial

    args = []
    for i in range(dim):
        args.append(basis[i][idxs[i],orders[i],indices_quad[i]])

    # ... assign basis on quad point
    logical = not( mapping is None )
    name = print_expression(atom, logical=logical)
    assign = Assign(Symbol(name), Mul(*args))
    # ...

    # ... map basis function
    map_stmts = []
    if mapping and  isinstance(atom, _partial_derivatives):
        name = print_expression(atom)

        a = get_atom_derivatives(atom)

        M = mapping
        dim = M.rdim
        ops = _partial_derivatives[:dim]

        # ... gradient
        lgrad_B = [d(a) for d in ops]
        grad_B = Covariant(mapping, lgrad_B)
        rhs = grad_B[atom.grad_index]

        # update expression
        elements = [d(M[i]) for d in ops for i in range(0, dim)]
        for e in elements:
            new = print_expression(e, mapping_name=False)
            new = Symbol(new)
            rhs = rhs.subs(e, new)

        for e in lgrad_B:
            new = print_expression(e, logical=True)
            new = Symbol(new)
            rhs = rhs.subs(e, new)
        # ...

        map_stmts += [Assign(Symbol(name), rhs)]
        # ...
    # ...

    return assign, map_stmts


#==============================================================================
def compute_atoms_expr_field(atom, indices_quad,
                            idxs, basis,
                            test_function, mapping):

    if not is_field(atom):
        raise TypeError('atom must be a field expr')

    field = list(atom.atoms(Field))[0]
    field_name = 'coeff_'+str(field.name)

    # ...
    if isinstance(atom, _partial_derivatives):
        direction = atom.grad_index + 1

    else:
        direction = 0
    # ...

    # ...
    test_function = atom.subs(field, test_function)
    name = print_expression(test_function)
    test_function = Symbol(name)
    # ...

    # ...
    args = []
    dim  = len(idxs)
    for i in range(dim):
        if direction == i+1:
            args.append(basis[i][idxs[i],1,indices_quad[i]])

        else:
            args.append(basis[i][idxs[i],0,indices_quad[i]])

    init = Assign(test_function, Mul(*args))
    # ...

    # ...
    args = [IndexedBase(field_name)[idxs], test_function]

    val_name = print_expression(atom) + '_values'
    val  = IndexedBase(val_name)[indices_quad]
    update = AugAssign(val,'+',Mul(*args))
    # ...

    # ... map basis function
    map_stmts = []
    if mapping and  isinstance(atom, _partial_derivatives):
        name = print_expression(atom)

        a = get_atom_derivatives(atom)

        M = mapping
        dim = M.rdim
        ops = _partial_derivatives[:dim]

        # ... gradient
        lgrad_B = [d(a) for d in ops]
        grad_B = Covariant(mapping, lgrad_B)
        rhs = grad_B[atom.grad_index]

        # update expression
        elements = [d(M[i]) for d in ops for i in range(0, dim)]
        for e in elements:
            new = print_expression(e, mapping_name=False)
            new = Symbol(new)
            rhs = rhs.subs(e, new)

        for e in lgrad_B:
            new = print_expression(e, logical=True)
            new = Symbol(new)
            rhs = rhs.subs(e, new)
        # ...

        map_stmts += [Assign(Symbol(name), rhs)]
        # ...
    # ...

    return init, update, map_stmts


#==============================================================================
def compute_atoms_expr_vector_field(atom, indices_quad,
                            idxs, basis,
                            test_function, mapping):

    if not is_vector_field(atom):
        raise TypeError('atom must be a vector field expr')


    vector_field = atom
    vector_field_name = 'coeff_' + print_expression(get_atom_derivatives(atom))

    # ...
    if isinstance(atom, _partial_derivatives):
        direction = atom.grad_index + 1

    else:
        direction = 0
    # ...

    # ...
    base = list(atom.atoms(VectorField))[0]
    test_function = atom.subs(base, test_function)
    name = print_expression(test_function)
    test_function = Symbol(name)
    # ...

    # ...
    args = []
    dim  = len(idxs)
    for i in range(dim):
        if direction == i+1:
            args.append(basis[i][idxs[i],1,indices_quad[i]])

        else:
            args.append(basis[i][idxs[i],0,indices_quad[i]])

    init = Assign(test_function, Mul(*args))
    # ...

    # ...
    args = [IndexedBase(vector_field_name)[idxs], test_function]
    val_name = print_expression(atom) + '_values'
    val  = IndexedBase(val_name)[indices_quad]
    update = AugAssign(val,'+',Mul(*args))
    # ...

    # ... map basis function
    map_stmts = []
    if mapping and  isinstance(atom, _partial_derivatives):
        name = print_expression(atom)

        a = get_atom_derivatives(atom)

        M = mapping
        dim = M.rdim
        ops = _partial_derivatives[:dim]

        # ... gradient
        lgrad_B = [d(a) for d in ops]
        grad_B = Covariant(mapping, lgrad_B)
        rhs = grad_B[atom.grad_index]

        # update expression
        elements = [d(M[i]) for d in ops for i in range(0, dim)]
        for e in elements:
            new = print_expression(e, mapping_name=False)
            new = Symbol(new)
            rhs = rhs.subs(e, new)

        for e in lgrad_B:
            new = print_expression(e, logical=True)
            new = Symbol(new)
            rhs = rhs.subs(e, new)
        # ...

        map_stmts += [Assign(Symbol(name), rhs)]
        # ...
    # ...

    return init, update, map_stmts


#==============================================================================
def compute_atoms_expr_mapping(atom, indices_quad,
                               idxs, basis,
                               test_function):

    _print = lambda i: print_expression(i, mapping_name=False)

    element = get_atom_derivatives(atom)
    element_name = 'coeff_' + _print(element)

    # ...
    if isinstance(atom, _partial_derivatives):
        direction = atom.grad_index + 1

    else:
        direction = 0
    # ...

    # ...
    test_function = atom.subs(element, test_function)
    name = print_expression(test_function, logical=True)
    test_function = Symbol(name)
    # ...

    # ...
    args = []
    dim  = len(idxs)
    for i in range(dim):
        if direction == i+1:
            args.append(basis[i][idxs[i],1,indices_quad[i]])

        else:
            args.append(basis[i][idxs[i],0,indices_quad[i]])

    init = Assign(test_function, Mul(*args))
    # ...

    # ...
    args = [IndexedBase(element_name)[idxs], test_function]
    val_name = _print(atom) + '_values'
    val  = IndexedBase(val_name)[indices_quad]
    update = AugAssign(val,'+',Mul(*args))
    # ...

    return init, update

#==============================================================================
def rationalize_eval_mapping(mapping, nderiv, space, indices_quad):

    _print = lambda i: print_expression(i, mapping_name=False)

    M = mapping
    dim = space.ldim
    ops = _partial_derivatives[:dim]

    # ... mapping components and their derivatives
    components = [M[i] for i in range(0, dim)]
    elements = list(components)

    if nderiv > 0:
        elements += [d(M[i]) for d in ops for i in range(0, dim)]

    if nderiv > 1:
        elements += [d1(d2(M[i])) for e,d1 in enumerate(ops)
                                  for d2 in ops[:e+1]
                                  for i in range(0, dim)]

    if nderiv > 2:
        raise NotImplementedError('TODO')
    # ...

    # ... weights and their derivatives
    weights = Field('w', space)

    weights_elements = [weights]
    if nderiv > 0:
        weights_elements += [d(weights) for d in ops]

    if nderiv > 1:
        weights_elements += [d1(d2(weights)) for e,d1 in enumerate(ops)
                                             for d2 in ops[:e+1]]

    if nderiv > 2:
        raise NotImplementedError('TODO')
    # ...

    stmts = []
    # declarations
    stmts += [Comment('declarations')]
    for atom in elements + weights_elements:
        atom_name = _print(atom)
        val_name = atom_name + '_values'
        val  = IndexedBase(val_name)[indices_quad]

        stmt = Assign(atom_name, val)
        stmts += [stmt]

    # assignements
    stmts += [Comment('rationalize')]

    # 0 order terms
    w  = Symbol( _print( weights ) )
    for i in range(dim):
        u  = Symbol( _print( M[i] ) )

        val_name = u.name + '_values'
        val  = IndexedBase(val_name)[indices_quad]
        stmt = Assign(val, u / w )

        stmts += [stmt]

    # 1 order terms
    if nderiv >= 1:
        for d in ops:
            w  = Symbol( _print( weights    ) )
            dw = Symbol( _print( d(weights) ) )

            for i in range(dim):
                u  = Symbol( _print( M[i]    ) )
                du = Symbol( _print( d(M[i]) ) )

                val_name = du.name + '_values'
                val  = IndexedBase(val_name)[indices_quad]
                stmt = Assign(val, du / w - u * dw / w**2 )

                stmts += [stmt]

    # 2 order terms
    if nderiv >= 2:
        raise NotImplementedError("")

    return stmts

#==============================================================================
# TODO take exponent to 1/dim
def area_eval_mapping(mapping, area, dim, indices_quad, weight):

    _print = lambda i: print_expression(i, mapping_name=False)

    M = mapping
    ops = _partial_derivatives[:dim]

    # ... mapping components and their derivatives
    elements = [d(M[i]) for d in ops for i in range(0, dim)]
    # ...

    stmts = []
    # declarations
    stmts += [Comment('declarations')]
    for atom in elements:
        atom_name = _print(atom)
        val_name = atom_name + '_values'
        val  = IndexedBase(val_name)[indices_quad]

        stmt = Assign(atom_name, val)
        stmts += [stmt]

    # ... inv jacobian
    jac = mapping.det_jacobian
    rdim = mapping.rdim
    ops = _partial_derivatives[:rdim]
    elements = [d(mapping[i]) for d in ops for i in range(0, rdim)]
    for e in elements:
        new = print_expression(e, mapping_name=False)
        new = Symbol(new)
        jac = jac.subs(e, new)
    # ...

    # ...
    stmts += [AugAssign(area, '+', Abs(jac) * weight)]
    # ...

    return stmts


#==============================================================================
def is_field(expr):

    if isinstance(expr, _partial_derivatives):
        return is_field(expr.args[0])

    elif isinstance(expr, Field):
        return True

    return False

#==============================================================================
def is_vector_field(expr):

    if isinstance(expr, _partial_derivatives):
        return is_vector_field(expr.args[0])

    elif isinstance(expr, (VectorField, IndexedVectorField)):
        return True

    return False


#==============================================================================
class SplBasic(Basic):
    _discrete_boundary = None

    def __new__(cls, tag, name=None, prefix=None, debug=False, detailed=False,
                mapping=None, is_rational_mapping=None):

        if name is None:
            if prefix is None:
                raise ValueError('prefix must be given')

            name = '{prefix}_{tag}'.format(tag=tag, prefix=prefix)

        obj = Basic.__new__(cls)

        obj._name                = name
        obj._tag                 = tag
        obj._dependencies        = []
        obj._debug               = debug
        obj._detailed            = detailed
        obj._mapping             = mapping
        obj._is_rational_mapping = is_rational_mapping
        obj._imports = []

        return obj

    @property
    def name(self):
        return self._name

    @property
    def tag(self):
        return self._tag

    @property
    def func(self):
        return self._func

    @property
    def basic_args(self):
        return self._basic_args

    @property
    def dependencies(self):
        return self._dependencies

    @property
    def debug(self):
        return self._debug

    @property
    def detailed(self):
        return self._detailed

    @property
    def mapping(self):
        return self._mapping

    @property
    def is_rational_mapping(self):
        return self._is_rational_mapping

    @property
    def discrete_boundary(self):
        return self._discrete_boundary

    @property
    def imports(self):
        return self._imports


#==============================================================================
class EvalMapping(SplBasic):

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
            weights_pts = Field('w', self.space)

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
        Nj = TestFunction(space, name='Nj')
        body = []
        init_basis = OrderedDict()
        updates = []
        for atom in self.elements + weights_elements:
            init, update = compute_atoms_expr_mapping(atom, indices_quad,
                                                      indices_basis, basis, Nj)

            updates.append(update)

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
class EvalField(SplBasic):

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

        field_atoms = self.fields.atoms(Field)
        fields_str = sorted([print_expression(f) for f in self.fields])

        # ... declarations
        degrees       = variables( 'p1:%s'%(dim+1), 'int')
        orders        = variables( 'k1:%s'%(dim+1), 'int')
        indices_basis = variables( 'jl1:%s'%(dim+1), 'int')
        indices_quad  = variables( 'g1:%s'%(dim+1), 'int')
        basis         = variables('basis1:%s'%(dim+1),
                                  dtype='real', rank=3, cls=IndexedVariable)
        fields_coeffs = variables(['coeff_{}'.format(f) for f in field_atoms],
                                  dtype='real', rank=dim, cls=IndexedVariable)
        fields_val    = variables(['{}_values'.format(f) for f in fields_str],
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
        Nj = TestFunction(space, name='Nj')
        body = []
        init_basis = OrderedDict()
        init_map   = OrderedDict()
        updates = []
        for atom in self.fields:
            init, update, map_stmts = compute_atoms_expr_field(atom, indices_quad, indices_basis,
                                                               basis, Nj,
                                                               mapping=mapping)

            updates.append(update)

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
class EvalVectorField(SplBasic):

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
        vector_fields_str = sorted([print_expression(f) for f in self.vector_fields])

        # ... declarations
        degrees       = variables('p1:%s'%(dim+1),  'int')
        orders        = variables('k1:%s'%(dim+1),  'int')
        indices_basis = variables('jl1:%s'%(dim+1), 'int')
        indices_quad  = variables('g1:%s'%(dim+1),  'int')
        basis         = variables('basis1:%s'%(dim+1),
                                  dtype='real', rank=3, cls=IndexedVariable)
        coeffs = ['coeff_{}'.format(print_expression(f)) for f in vector_field_atoms]
        vector_fields_coeffs = variables(coeffs, dtype='real', rank=dim, cls=IndexedVariable)
        vector_fields_val    = variables(['{}_values'.format(f) for f in vector_fields_str],
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
        Nj = VectorField(space, name='Nj')
        body = []
        init_basis = OrderedDict()
        init_map   = OrderedDict()
        updates = []
        for atom in self.vector_fields:
            init, update, map_stmts = compute_atoms_expr_vector_field(atom, indices_quad, indices_basis,
                                                                      basis, Nj,
                                                                      mapping=mapping)

            updates.append(update)

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
# target is used when there are multiple expression (domain/boundaries)
class Kernel(SplBasic):

    def __new__(cls, weak_form, kernel_expr, target=None,
                discrete_boundary=None, name=None, boundary_basis=None,
                mapping=None, is_rational_mapping=None,backend=None):

        if not isinstance(weak_form, FunctionalForms):
            raise TypeError('> Expecting a weak formulation')

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

        obj._weak_form         = weak_form
        obj._kernel_expr       = kernel_expr
        obj._target            = target
        obj._discrete_boundary = discrete_boundary
        obj._boundary_basis    = boundary_basis
        obj._area              = None
        obj._backend           = backend

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
    def fields_tmp_coeffs(self):
        return self._fields_tmp_coeffs

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
        is_function = isinstance(self.weak_form, Integral)

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
                       TestFunction,
                       IndexedTestTrial,
                       Field,
                       VectorField, IndexedVectorField)

        atoms  = _atomic(expr, cls=atoms_types)
        # ...

        # ...
        atomic_expr_field        = [atom for atom in atoms if is_field(atom)]
        atomic_expr_vector_field = [atom for atom in atoms if is_vector_field(atom)]

        atomic_expr       = [atom for atom in atoms if not( atom in atomic_expr_field ) and
                                                       not( atom in atomic_expr_vector_field)]
        # ...

        # TODO use print_expression
        fields_str    = sorted(tuple(map(pycode, atomic_expr_field)))
        fields_logical_str = sorted([print_expression(f, logical=True) for f in
                                     atomic_expr_field])
        field_atoms   = tuple(expr.atoms(Field))

        # ... create EvalField
        self._eval_fields = []
        self._map_stmts_fields = OrderedDict()
        if atomic_expr_field:
            keyfunc = lambda F: F.space.name
            data = sorted(field_atoms, key=keyfunc)
            for space_str, group in groupby(data, keyfunc):
                g_names = set([f.name for f in group])
                fields_expressions = []
                for e in atomic_expr_field:
                    fs = e.atoms(Field)
                    f_names = set([f.name for f in fs])
                    if f_names & g_names:
                        fields_expressions += [e]
                        space = list(fs)[0].space

                eval_field = EvalField(space, fields_expressions,
                                       discrete_boundary=self.discrete_boundary,
                                       boundary_basis=self.boundary_basis,
                                       mapping=mapping,backend=self.backend)

                self._eval_fields.append(eval_field)
                for k,v in eval_field.map_stmts.items():
                    self._map_stmts_fields[k] = v

        # update dependencies
        self._dependencies += self.eval_fields
        # ...

        # ...
        vector_fields_str    = sorted(tuple(print_expression(i) for i in  atomic_expr_vector_field))
        vector_fields_logical_str = sorted([print_expression(f, logical=True) for f in
                                     atomic_expr_vector_field])
        vector_field_atoms   = tuple(expr.atoms(VectorField))

        # ... create EvalVectorField
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

                eval_vector_field = EvalVectorField(space, vector_fields_expressions,
                                                    discrete_boundary=self.discrete_boundary,
                                                    boundary_basis=self.boundary_basis,
                                                    mapping=mapping,backend=self.backend)
                self._eval_vector_fields.append(eval_vector_field)
                for k,v in eval_vector_field.map_stmts.items():
                    self._map_stmts_fields[k] = v

        # update dependencies
        self._dependencies += self.eval_vector_fields
        # ...

        # ...
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

            eval_mapping = EvalMapping(space, mapping,
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
                test_function = [test_function]
                test_function = Tuple(*test_function)

        elif is_function:
            test_function = TestFunction(self.weak_form.space, name='Nj')
            test_function = [test_function]
            test_function = Tuple(*test_function)

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
        fields_val    = variables(['{}_values'.format(f) for f in fields_str],
                                          dtype='real', rank=dim, cls=IndexedVariable)

        fields_tmp_coeffs = variables(['tmp_coeff_{}'.format(f) for f in field_atoms],
                                              dtype='real', rank=dim, cls=IndexedVariable)

        vector_fields        = symbols(vector_fields_str)
        vector_fields_logical = symbols(vector_fields_logical_str)

        vector_field_atoms = [f[i] for f in vector_field_atoms for i in range(0, dim)]
        coeffs = ['coeff_{}'.format(print_expression(f)) for f in vector_field_atoms]
        vector_fields_coeffs = variables(coeffs, dtype='real', rank=dim, cls=IndexedVariable)

        vector_fields_val    = variables(['{}_values'.format(f) for f in vector_fields_str],
                                          dtype='real', rank=dim, cls=IndexedVariable)

        test_degrees  = variables('test_p1:%s'%(dim+1),  'int')
        trial_degrees = variables('trial_p1:%s'%(dim+1), 'int')
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
            self._basic_args = (test_pads + trial_pads +
                                basis_test + basis_trial +
                                positions + weighted_vols)

        if is_linear or is_function:
            self._basic_args = (test_pads +
                                basis_test +
                                positions + weighted_vols+
                                fields_val + vector_fields_val)
        # ...

        # ...
        mapping_elements = ()
        mapping_coeffs = ()
        mapping_values = ()
        if mapping:
            _eval = self.eval_mapping
            _print = lambda i: print_expression(i, mapping_name=False)

            mapping_elements = [_print(i) for i in _eval.elements]
            mapping_elements = symbols(tuple(mapping_elements))

            mapping_coeffs = [_print(i) for i in _eval.mapping_coeffs]
            mapping_coeffs = variables(mapping_coeffs, dtype='real', rank=dim, cls=IndexedVariable)

            mapping_values = [_print(i) for i in _eval.mapping_values]
            mapping_values = variables(mapping_values, dtype='real', rank=dim, cls=IndexedVariable)
        # ...

        # ...
        self._fields_val = fields_val
        self._vector_fields_val = vector_fields_val
        self._fields = fields
        self._fields_logical = fields_logical
        self._fields_coeffs = fields_coeffs
        self._fields_tmp_coeffs = fields_tmp_coeffs
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
        body = []

        init_basis = OrderedDict()
        init_map   = OrderedDict()
        for atom in atomic_expr:
            init, map_stmts = compute_atoms_expr(atom,
                                                 indices_quad,
                                                 indices_test,
                                                 indices_trial,
                                                 basis_trial,
                                                 basis_test,
                                                 coordinates,
                                                 test_function,
                                                 is_linear,
                                                 mapping)

            init_basis[str(init.lhs)] = init
            for stmt in map_stmts:
                init_map[str(stmt.lhs)] = stmt

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
                rdim = mapping.rdim
                ops = _partial_derivatives[:rdim]
                elements = [d(mapping[i]) for d in ops for i in range(0, rdim)]
                for e in elements:
                    new = print_expression(e, mapping_name=False)
                    new = Symbol(new)
                    jac = jac.subs(e, new)
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

        for i in range(ln):
            if not( i in zero_terms ):
                e = Mul(expr[i],wvol)

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
        init_vars = [Assign(v[i],0.0) for i in range(ln) if not( i in zero_terms )]
        body = init_vars + body
        # ...

        if dim_trial:
            trial_idxs = tuple([indices_trial[i]+trial_pads[i]-indices_test[i] for i in range(dim)])
            idxs = indices_test + trial_idxs
        else:
            idxs = indices_test

        if is_bilinear or is_linear:
            for i in range(ln):
                if not( i in zero_terms ):
                    body.append(Assign(mats[i][idxs],v[i]))

        elif is_function:
            for i in range(ln):
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
            init_mats = [mats[i][[Slice(None,None)]*(dim_test+dim_trial)] for i in range(ln) if not( i in zero_terms )]

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
        math_elements = math_atoms_as_str(self.kernel_expr)
        math_imports = []
        for e in math_elements:
            math_imports += [Import(e, 'numpy')]

        imports += math_imports
        self._imports = imports
        # function args
        mats = tuple([M for i,M in enumerate(mats) if not( i in zero_terms )])
        func_args = self.build_arguments(fields_coeffs + vector_fields_coeffs + mapping_coeffs + mats)

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
class Assembly(SplBasic):

    def __new__(cls, kernel, name=None, discrete_space=None, periodic=None,
                comm=None, mapping=None, is_rational_mapping=None, backend=None):

        if not isinstance(kernel, Kernel):
            raise TypeError('> Expecting a kernel')

        obj = SplBasic.__new__(cls, kernel.tag, name=name,
                               prefix='assembly', mapping=mapping,
                               is_rational_mapping=is_rational_mapping)

        # ... get periodicity of the space
        dim    = kernel.weak_form.ldim
        periodic = [False for i in range(0, dim)]
        if not( discrete_space is None ):
            if isinstance(discrete_space, (tuple, list)):
                space = discrete_space[0]

            else:
                space = discrete_space

            if isinstance(space, SplineSpace):
                periodic = [space.periodic]

            elif isinstance(space, TensorFemSpace):
                periodic = space.periodic

            elif isinstance(space, ProductFemSpace):
                periodic = space.spaces[0].periodic

            else:
                raise NotImplementedError('Only Spline, Tensor and Product spaces are available')
        # ...

        obj._kernel = kernel
        obj._discrete_space = discrete_space
        obj._periodic = periodic
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
    def periodic(self):
        return self._periodic

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
        fields_tmp_coeffs = kernel.fields_tmp_coeffs
        vector_fields = kernel.vector_fields
        vector_fields_coeffs = kernel.vector_fields_coeffs
        zero_terms = kernel.zero_terms

        is_linear   = isinstance(self.weak_form, LinearForm)
        is_bilinear = isinstance(self.weak_form, BilinearForm)
        is_function = isinstance(self.weak_form, Integral)

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
        indices_span  = variables('is1:%s'%(dim+1), 'int')

        test_pads     = variables('test_p1:%s'%(dim+1),  'int')
        trial_pads    = variables('trial_p1:%s'%(dim+1), 'int')
        test_degrees  = variables('test_p1:%s'%(dim+1),  'int')
        trial_degrees = variables('trial_p1:%s'%(dim+1), 'int')

        quad_orders   = variables('k1:%s'%(dim+1),  'int')
        indices_il    = variables('il1:%s'%(dim+1), 'int')
        indices_i     = variables('i1:%s'%(dim+1),  'int')
        npts          = variables('n1:%s'%(dim+1),  'int')


        trial_basis   = variables('trial_basis_1:%s'%(dim+1), dtype='real', rank=4, cls=IndexedVariable)
        test_basis    = variables('test_basis_1:%s'%(dim+1), dtype='real', rank=4, cls=IndexedVariable)

        trial_basis_in_elm = variables('trial_bs1:%s'%(dim+1), dtype='real', rank=3, cls=IndexedVariable)
        test_basis_in_elm  = variables('test_bs1:%s'%(dim+1), dtype='real', rank=3, cls=IndexedVariable)

        points_in_elm  = variables('quad_u1:%s'%(dim+1), dtype='real', rank=1, cls=IndexedVariable)
        weights_in_elm = variables('quad_w1:%s'%(dim+1), dtype='real', rank=1, cls=IndexedVariable)


        points   = variables('points_1:%s'%(dim+1), dtype='real', rank=2, cls=IndexedVariable)
        weights  = variables('weights_1:%s'%(dim+1), dtype='real', rank=2, cls=IndexedVariable)
        spans    = variables('test_spans_1:%s'%(dim+1), dtype='int', rank=1, cls=IndexedVariable)
        # ...

        # ...
        # TODO improve: select args parallel/serial
        if is_bilinear:
            self._basic_args = (n_elements +
                                element_starts + element_ends +
                                starts + ends +
                                npts +
                                quad_orders +
                                test_degrees + trial_degrees +
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
        element_matrices = {}
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
        global_matrices = {}
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
        body  = [Assign(indices_span[i], spans[i][indices_elm[i]])
                 for i in range(dim) if not(i in axis_bnd)]
        if self.debug and self.detailed:
            msg = lambda x: (String('> span {} = '.format(x)), x)
            body += [Print(msg(indices_span[i])) for i in range(dim)]

        body += [Assign(points_in_elm[i], points[i][indices_elm[i],_slice])
                 for i in range(dim) if not(i in axis_bnd) ]

        body += [Assign(weights_in_elm[i], weights[i][indices_elm[i],_slice])
                 for i in range(dim) if not(i in axis_bnd) ]

        body += [Assign(test_basis_in_elm[i], test_basis[i][indices_elm[i],_slice,_slice,_slice])
                 for i in range(dim) if not(i in axis_bnd) ]

        if is_bilinear:
            body += [Assign(trial_basis_in_elm[i], trial_basis[i][indices_elm[i],_slice,_slice,_slice])
                     for i in range(dim) if not(i in axis_bnd) ]

        # ... kernel call
        ind = 0
        mats = []
        for i in range(0, n_rows):
            for j in range(0, n_cols):
                if not( ind in zero_terms ):
                    mats.append(element_matrices[i,j])

                ind += 1

        mats = tuple(mats)

        if not( self.comm is None ) and any(self.periodic) :
            # ...
            stmts = []
            for i,il,p,span,n,per in zip( indices_i,
                                          indices_il,
                                          test_degrees,
                                          indices_span,
                                          npts,
                                          self.periodic ):

                if not per:
                    stmts += [Assign(i, span-p+il)]

                else:
                    stmts += [Assign(i, Mod(span-p+il, n))]

#            _indices_i = [i for i,s,p in zip(indices_i, starts, test_degrees)]
            _indices_i = [i-s+p for i,s,p in zip(indices_i, starts, test_degrees)]
            for x,tmp_x in zip(fields_coeffs, fields_tmp_coeffs):
#                stmts += [Print([_indices_i, '    ', indices_i, starts])]
                stmts += [Assign(tmp_x[indices_il], x[_indices_i])]

            ranges = [Range(0, test_degrees[i]+1) for i in range(dim)]
            for x,rx in list(zip(indices_il, ranges))[::-1]:
                stmts = [For(x, rx, stmts)]

            body += stmts
            # ...

            # ...
            f_coeffs = tuple(fields_tmp_coeffs)
            # ...

            # ... TODO add tmp for vector fields and mapping
            gslices = [Slice(i-s,i+p+1-s) for i,p,s in zip(indices_span,
                                                           test_degrees,
                                                           starts)]
            vf_coeffs = tuple([f[gslices] for f in vector_fields_coeffs])
            m_coeffs = tuple([f[gslices] for f in kernel.mapping_coeffs])
            # ...

        else:
            gslices = [Slice(i-s,i+p+1-s) for i,p,s in zip(indices_span,
                                                           test_degrees,
                                                           starts)]
            f_coeffs = tuple([f[gslices] for f in fields_coeffs])
            vf_coeffs = tuple([f[gslices] for f in vector_fields_coeffs])
            m_coeffs = tuple([f[gslices] for f in kernel.mapping_coeffs])

        args = kernel.build_arguments(f_coeffs + vf_coeffs + m_coeffs + mats)
        body += [FunctionCall(kernel.func, args)]
        # ...

        # ... update global matrices
        lslices = [Slice(None,None)]*dim
        if is_bilinear:
            lslices += [Slice(None,None)]*dim # for assignement

        if is_bilinear:

            if ( self.comm is None ):
                gslices = [Slice(i,i+p+1) for i,p in zip(indices_span, test_degrees)]

            else:
                gslices = [Slice(i-s,i+p+1-s) for i,p,s in zip(indices_span,
                                                               test_degrees,
                                                               starts)]

            gslices += [Slice(None,None)]*dim # for assignement

        if is_linear:
            if ( self.comm is None ):
                gslices = [Slice(i,i+p+1) for i,p in zip(indices_span, test_degrees)]

            else:
                gslices = [Slice(i-s,i+p+1-s) for i,p,s in zip(indices_span,
                                                               test_degrees,
                                                               starts)]

        if is_function:
            lslices = 0
            gslices = 0

        for ij, M in global_matrices.items():
            i,j = ij
            mat = element_matrices[i,j]

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
        orders  = [p+1 for p in test_degrees]
        spads   = [2*p+1 for p in test_pads]
        ind = 0
        for i in range(0, n_rows):
            for j in range(0, n_cols):
                if not( ind in zero_terms ):
                    mat = element_matrices[i,j]

                    if is_bilinear:
                        args = tuple(orders + spads)

                    if is_linear:
                        args = tuple(orders)

                    if is_function:
                        args = tuple([1])

                    stmt = Assign(mat, Zeros(args))
                    prelude += [stmt]

                ind += 1

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
        if not( self.comm is None ) and any(self.periodic) :
            for v in self.kernel.fields_tmp_coeffs:
                stmt = Assign(v, Zeros(orders))
                prelude += [stmt]
        # ...

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
        mats = []
        for ij, M in global_matrices.items():
            i,j = ij
            mats.append(M)

        mats = tuple(mats)
        self._global_matrices = mats
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
        return self.assembly.weak_form

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

# TODO uncomment later
    #@property
    #def dots(self):
    #    return self._dots


    def _initialize(self):
        form = self.weak_form
        assembly = self.assembly
        global_matrices = assembly.global_matrices
        fields = tuple(form.expr.atoms(Field))
        fields = sorted(fields, key=lambda x: str(x.name))
        fields = tuple(fields)
        zero_terms = assembly.kernel.zero_terms

        vector_fields = tuple(form.expr.atoms(VectorField))
        vector_fields = sorted(vector_fields, key=lambda x: str(x.name))
        vector_fields = tuple(vector_fields)

        is_linear   = isinstance(self.weak_form, LinearForm)
        is_bilinear = isinstance(self.weak_form, BilinearForm)
        is_function = isinstance(self.weak_form, Integral)

        dim = form.ldim

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
            Wh = self.discrete_space[0]
            Vh = self.discrete_space[1]

            # ... TODO improve
            if isinstance(Wh, ProductFemSpace):
                v = Wh.spaces[0]
                if not all([w.vector_space is v.vector_space for w in Wh.spaces[1:]]):
                    raise NotImplementedError('vector spaces must be the same')

                test_vector_space = DottedName(test_vector_space, 'spaces[0]')
            # ...

            # ... TODO improve
            if isinstance(Vh, ProductFemSpace):
                v = Vh.spaces[0]
                if not all([w.vector_space is v.vector_space for w in Vh.spaces[1:]]):
                    raise NotImplementedError('vector spaces must be the same')

                trial_vector_space = DottedName(trial_vector_space, 'spaces[0]')
            # ...

        if is_linear or is_function:
            test_vector_space = DottedName(test_space, 'vector_space')
            spaces = (test_space,)
            Wh = self.discrete_space

            # ... TODO improve
            if isinstance(Wh, ProductFemSpace):
                v = Wh.spaces[0]
                if not all([w.vector_space is v.vector_space for w in Wh.spaces[1:]]):
                    raise NotImplementedError('vector spaces must be the same')

                test_vector_space = DottedName(test_vector_space, 'spaces[0]')
            # ...

        n_elements     = variables('n_elements_1:%s'%(dim+1), 'int')
        starts         = variables('s1:%s'%(dim+1), 'int')
        ends           = variables('e1:%s'%(dim+1), 'int')
        npts           = variables('n1:%s'%(dim+1), 'int')
        element_starts = variables('element_s1:%s'%(dim+1), 'int')
        element_ends   = variables('element_e1:%s'%(dim+1), 'int')

        test_degrees   = variables('test_p1:%s'%(dim+1), 'int')
        trial_degrees  = variables('trial_p1:%s'%(dim+1), 'int')


        points         = variables('points_1:%s'%(dim+1),  dtype='real', rank=2, cls=IndexedVariable)
        weights        = variables('weights_1:%s'%(dim+1), dtype='real', rank=2, cls=IndexedVariable)

        trial_basis    = variables('trial_basis_1:%s'%(dim+1), dtype='real', rank=4, cls=IndexedVariable)
        test_basis     = variables('test_basis_1:%s'%(dim+1), dtype='real', rank=4, cls=IndexedVariable)

        spans          = variables('test_spans_1:%s'%(dim+1), dtype='int', rank=1, cls=IndexedVariable)
        quad_orders    = variables( 'k1:%s'%(dim+1), 'int')

	# TODO uncomment later
        #dots           = symbols('lo_dot v_dot')
        #dot            = Symbol('dot')

        mapping = ()
        if self.mapping:
            mapping = Symbol('mapping')
        # ...

        # ...
        if dim == 1:
            points        = points[0]
            weights       = weights[0]
            trial_basis   = trial_basis[0]
            test_basis    = test_basis[0]
            spans         = spans[0]
            quad_orders   = quad_orders[0]
        # ...

        # ...
        self._basic_args = spaces + (grid,) + basis_values
        # ...

        # ... interface body
        body = []
        # ...

        # ... grid data
        body += [Assign(n_elements,     DottedName(grid, 'n_elements'))]
        body += [Assign(points,         DottedName(grid, 'points'))]
        body += [Assign(weights,        DottedName(grid, 'weights'))]
        body += [Assign(quad_orders,    DottedName(grid, 'quad_order'))]
        body += [Assign(element_starts, DottedName(grid, 'local_element_start'))]
        body += [Assign(element_ends,   DottedName(grid, 'local_element_end'))]
        # ...

        # ... basis values
        body += [Assign(spans,      DottedName(test_basis_values, 'spans'))]
        body += [Assign(test_basis, DottedName(test_basis_values, 'basis'))]

        if is_bilinear:
            body += [Assign(trial_basis, DottedName(trial_basis_values, 'basis'))]
        # ...

        # ... getting data from fem space
        body += [Assign(test_degrees, DottedName(test_vector_space, 'pads'))]
        if is_bilinear:
            body += [Assign(trial_degrees, DottedName(trial_vector_space, 'pads'))]

        body += [Assign(starts, DottedName(test_vector_space, 'starts'))]
        body += [Assign(ends,   DottedName(test_vector_space, 'ends'))]
        body += [Assign(npts,   DottedName(test_vector_space, 'npts'))]

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
                imports += [Import('StencilMatrix', 'spl.linalg.stencil')]

            if is_linear:
                imports += [Import('StencilVector', 'spl.linalg.stencil')]

            for M in global_matrices:
                if_cond = Is(M, Nil())
                if is_bilinear:
                    args = [test_vector_space, trial_vector_space]
                    if_body = [Assign(M, FunctionCall('StencilMatrix', args))]
# TODO uncomment later
                    #if_body.append(Assign(DottedName(M,'_dot'),dots[0]))


                if is_linear:
                    args = [test_vector_space]
                    if_body = [Assign(M, FunctionCall('StencilVector', args))]
# TODO uncomment later
                    #if_body.append(Assign(DottedName(M,'_dot'),dots[1]))

                stmt = If((if_cond, if_body))
                body += [stmt]

        else:
            imports += [Import('zeros', 'numpy')]
            for M in global_matrices:
                body += [Assign(M, Zeros(1))]
        # ...

        # ...
        self._inout_arguments = [M for M in global_matrices]
        self._in_arguments = list(self.assembly.kernel.constants) + list(fields) + list(vector_fields)
        # ...

        # ... call to assembly
        if is_bilinear or is_linear:
            mat_data = [DottedName(M, '_data') for M in global_matrices]

        elif is_function:
            mat_data = [M for M in global_matrices]

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
                for M in global_matrices:
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

                    imports += [Import('BlockMatrix', 'spl.linalg.block')]

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
                    test_vector_space = DottedName(test_space, 'vector_space')
                    trial_vector_space = DottedName(trial_space, 'vector_space')
                    # ...

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

                    imports += [Import('BlockVector', 'spl.linalg.block')]

                    # ... create product space
                    test_vector_space = DottedName(test_space, 'vector_space')
                    # ...

                    body += [Assign(L, FunctionCall('BlockVector', [test_vector_space]))]
                    d = OrderedDict(sorted(d.items()))
                    for k,v in d.items():
                        body += [Assign(L[k], v)]

                body += [Return(L)]

            else:
                M = global_matrices[0]
                body += [Return(M)]

        elif is_function:
            if len(global_matrices) == 1:
                M = global_matrices[0]
                body += [Return(M[0])]

            else:
                body += [Return(M[0]) for M in global_matrices]
        # ...

        # ... arguments
        if is_bilinear or is_linear:
            mats = [Assign(M, Nil()) for M in global_matrices]
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


