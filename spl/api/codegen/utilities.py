import re
import string
import random
import numpy as np

from sympy import Symbol, IndexedBase
from sympy import Mul, Add, Tuple, Min, Max, Pow
from sympy import Matrix
from sympy import sqrt as sympy_sqrt
from sympy.utilities.iterables import cartes

from pyccel.ast.core import Variable, IndexedVariable
from pyccel.ast.core import For
from pyccel.ast.core import Assign
from pyccel.ast.core import AugAssign
from pyccel.ast.core import Range, Product
from pyccel.ast import Comment, NewLine

from sympde.topology.derivatives import _partial_derivatives
from sympde.topology.space import ScalarTestFunction
from sympde.topology.space import VectorTestFunction
from sympde.topology.space import IndexedTestTrial
from sympde.topology import ScalarField
from sympde.topology import VectorField, IndexedVectorField
from sympde.topology.derivatives import print_expression
from sympde.topology.derivatives import get_atom_derivatives
from sympde.topology.derivatives import get_index_derivatives
from sympde.topology import LogicalExpr
from sympde.topology import SymbolicExpr
from sympde.core import Cross_3d



#==============================================================================
def random_string( n ):
    chars    = string.ascii_lowercase + string.digits
    selector = random.SystemRandom()
    return ''.join( selector.choice( chars ) for _ in range( n ) )


#==============================================================================
def is_field(expr):

    if isinstance(expr, _partial_derivatives):
        return is_field(expr.args[0])

    elif isinstance(expr, ScalarField):
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
def compute_atoms_expr(atom, indices_quad, indices_test,
                       indices_trial, basis_trial,
                       basis_test, cords, test_function,
                       is_linear,
                       mapping):

    cls = (_partial_derivatives,
           VectorTestFunction,
           ScalarTestFunction,
           IndexedTestTrial)

    dim  = len(indices_test)

    if not isinstance(atom, cls):
        raise TypeError('atom must be of type {}'.format(str(cls)))

    orders = [0 for i in range(0, dim)]
    p_indices = get_index_derivatives(atom)

    # ...
    def _get_name(atom):
        atom_name = None
        if isinstance( atom, ScalarTestFunction ):
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
        rhs = LogicalExpr(mapping, atom)
        rhs = SymbolicExpr(rhs)
        map_stmts = [Assign(Symbol(name), rhs)]
    # ...

    return assign, map_stmts


#==============================================================================
def compute_atoms_expr_field(atom, indices_quad,
                            idxs, basis,
                            test_function, mapping):

    if not is_field(atom):
        raise TypeError('atom must be a field expr')

    field = list(atom.atoms(ScalarField))[0]
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
        rhs = LogicalExpr(mapping, atom)
        rhs = SymbolicExpr(rhs)
        map_stmts = [Assign(Symbol(name), rhs)]
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
        rhs = LogicalExpr(mapping, atom)
        rhs = SymbolicExpr(rhs)
        map_stmts = [Assign(Symbol(name), rhs)]
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
    # TODO check if 'w' exist already
    weights = ScalarField(space, name='w')

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

#==============================================================================
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

        J = SymbolicExpr(J)

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
_range = re.compile('([0-9]*:[0-9]+|[a-zA-Z]?:[a-zA-Z])')

def variables(names, dtype, **args):

    def contruct_variable(cls, name, dtype, rank, **args):
        if issubclass(cls, Variable):
            return Variable(dtype,  name, rank=rank, **args)
        elif issubclass(cls, IndexedVariable):
            return IndexedVariable(name, dtype=dtype, rank=rank, **args)
        else:
            raise TypeError('only Variables and IndexedVariables are supported')

    result = []
    cls = args.pop('cls', Variable)

    rank = args.pop('rank', 0)

    if isinstance(names, str):
        marker = 0
        literals = [r'\,', r'\:', r'\ ']
        for i in range(len(literals)):
            lit = literals.pop(0)
            if lit in names:
                while chr(marker) in names:
                    marker += 1
                lit_char = chr(marker)
                marker += 1
                names = names.replace(lit, lit_char)
                literals.append((lit_char, lit[1:]))
        def literal(s):
            if literals:
                for c, l in literals:
                    s = s.replace(c, l)
            return s

        names = names.strip()
        as_seq = names.endswith(',')
        if as_seq:
            names = names[:-1].rstrip()
        if not names:
            raise ValueError('no symbols given')

        # split on commas
        names = [n.strip() for n in names.split(',')]
        if not all(n for n in names):
            raise ValueError('missing symbol between commas')
        # split on spaces
        for i in range(len(names) - 1, -1, -1):
            names[i: i + 1] = names[i].split()

        seq = args.pop('seq', as_seq)

        for name in names:
            if not name:
                raise ValueError('missing variable')

            if ':' not in name:
                var = contruct_variable(cls, literal(name), dtype, rank, **args)
                result.append(var)
                continue

            split = _range.split(name)
            # remove 1 layer of bounding parentheses around ranges
            for i in range(len(split) - 1):
                if i and ':' in split[i] and split[i] != ':' and \
                        split[i - 1].endswith('(') and \
                        split[i + 1].startswith(')'):
                    split[i - 1] = split[i - 1][:-1]
                    split[i + 1] = split[i + 1][1:]
            for i, s in enumerate(split):
                if ':' in s:
                    if s[-1].endswith(':'):
                        raise ValueError('missing end range')
                    a, b = s.split(':')
                    if b[-1] in string.digits:
                        a = 0 if not a else int(a)
                        b = int(b)
                        split[i] = [str(c) for c in range(a, b)]
                    else:
                        a = a or 'a'
                        split[i] = [string.ascii_letters[c] for c in range(
                            string.ascii_letters.index(a),
                            string.ascii_letters.index(b) + 1)]  # inclusive
                    if not split[i]:
                        break
                else:
                    split[i] = [s]
            else:
                seq = True
                if len(split) == 1:
                    names = split[0]
                else:
                    names = [''.join(s) for s in cartes(*split)]
                if literals:
                    result.extend([contruct_variable(cls, literal(s), dtype, rank, **args) for s in names])
                else:
                    result.extend([contruct_variable(cls, s, dtype, rank, **args) for s in names])

        if not seq and len(result) <= 1:
            if not result:
                return ()
            return result[0]

        return tuple(result)
    elif isinstance(names,(tuple,list)):
        return tuple(variables(i, dtype, cls=cls,rank=rank,**args) for i in names)
    else:
        raise TypeError('Expecting a string')






def build_pythran_types_header(name, args, order=None):
    """
    builds a types decorator from a list of arguments (of FunctionDef)
    """
    types = []
    for a in args:
        if isinstance(a, Variable):
            dtype = pythran_dtypes[a.dtype.name.lower()]

        elif isinstance(a, IndexedVariable):
            dtype = pythran_dtypes[a.dtype.name.lower()]

        else:
            raise TypeError('unepected type for {}'.format(a))

        if a.rank > 0:
            shape = ['[]' for i in range(0, a.rank)]
            shape = ''.join(i for i in shape)
            dtype = '{dtype}{shape}'.format(dtype=dtype, shape=shape)
            if order and a.rank > 1:
                dtype = "{dtype}".format(dtype=dtype, ordering=order)

        types.append(dtype)
    types = ', '.join(_type for _type in types)
    header = '#pythran export {name}({types})'.format(name=name, types=types)
    return header

pythran_dtypes = {'real':'float','int':'int'}
