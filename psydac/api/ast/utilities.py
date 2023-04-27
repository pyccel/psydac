import re
import string
import random

from sympy import Symbol, IndexedBase, Indexed, Idx
from sympy import Mul, Function
from sympy.utilities.iterables import cartes

from sympde.topology.space       import ScalarFunction, VectorFunction, IndexedVectorFunction, element_of
from sympde.topology             import Mapping, LogicalExpr, SymbolicExpr
from sympde.topology.derivatives import _partial_derivatives, _logical_partial_derivatives, get_atom_derivatives, \
                                        get_index_derivatives, get_atom_logical_derivatives, get_index_logical_derivatives
from sympde.core                 import Constant

from psydac.pyccel.ast.core import Variable, IndexedVariable, Assign, AugAssign, _atomic, Comment, String

#==============================================================================
def random_string( n ):
    chars    = string.ascii_lowercase + string.digits
    selector = random.SystemRandom()
    return ''.join( selector.choice( chars ) for _ in range( n ) )

#==============================================================================
def is_mapping(expr):

    if isinstance(expr, _logical_partial_derivatives):
        return is_mapping(expr.args[0])

    elif isinstance(expr, Indexed) and isinstance(expr.base, Mapping):
        return True

    elif isinstance(expr, Mapping):
        return True

    return False

#==============================================================================
def compute_atoms_expr_field(atomic_exprs, indices_quad, idxs, basis, test_function, mapping):

    """
    This function computes atomic expressions needed
    to evaluate  EvaluteField/VectorField final expression

    Parameters
    ----------
    atomic_exprs : <list>
        list of atoms

    indices_quad : <list>
        list of quadrature indices used in the quadrature loops

    idxs : <list>
        list of basis functions indices used in the for loops of the basis functions


    basis : <list>
        list of basis functions in each dimesion

    test_function : <Symbol>
        test_function Symbol

    mapping : <Mapping>
        Mapping object

    Returns
    -------
    inits : <list>
       list of assignments of the atomic expression evaluated in the quadrature points

    updates : <list>
        list of augmented assignments which are updated in each loop iteration

    map_stmts : <list>
        list of assignments of atomic expression in case of mapping

    new_atoms: <list>
        updated list of atomic expressions (some were introduced in case of a mapping)
    """

    inits     = []
    updates   = []
    map_stmts = []

    cls = (_partial_derivatives,
           ScalarFunction,
           IndexedVectorFunction,
           VectorFunction)

    # If there is a mapping, compute [dx(u), dy(u), dz(u)] as functions
    # of [dx1(u), dx2(u), dx3(u)], and store results into intermediate
    # variables [u_x, u_y, u_z]. (Same thing is done for higher derivatives.)
    #
    # Accordingly, we create a new list of atoms where all partial derivatives
    # are taken with respect to the logical coordinates.
    if mapping:

        new_atoms = set()
        map_stmts = []
        get_index = get_index_logical_derivatives
        get_atom  = get_atom_logical_derivatives

        for atom in atomic_exprs:

            if isinstance(atom, _partial_derivatives):
                lhs   = SymbolicExpr(atom)
                rhs_p = LogicalExpr(mapping, atom)

                # we look for new_atoms that must be added to atomic_exprs
                # because we need them in the maps stmts
                logical_atoms = _atomic(rhs_p, cls=_logical_partial_derivatives)
                for a in logical_atoms:
                    ls = _atomic(a, Symbol)
                    assert len(ls) == 1
                    if isinstance(ls[0], cls):
                        new_atoms.add(a)

                rhs = SymbolicExpr(rhs_p)
                map_stmts += [Assign(lhs, rhs)]

            else:
                new_atoms.add(atom)

    else:
        new_atoms = atomic_exprs
        map_stmts = []
        get_index = get_index_derivatives
        get_atom  = get_atom_derivatives

    # Make sure that we only pick one between 'dx1(dx2(u))' and 'dx2(dx1(u))'
    new_atoms = {SymbolicExpr(a).name : a for a in new_atoms}
    new_atoms = tuple(new_atoms.values())

    # Create a list of statements for initialization of the point values,
    # for each of the atoms in our (possibly new) list.
    inits = []
    for atom in new_atoms:

        # Extract field, compute name of coefficient variable, and get base
        if atom.atoms(ScalarFunction):
            field      = atom.atoms(ScalarFunction).pop()
            field_name = 'coeff_' + SymbolicExpr(field).name
            base       = field
        elif atom.atoms(VectorFunction):
            field      = atom.atoms(IndexedVectorFunction).pop()
            field_name = 'coeff_' + SymbolicExpr(field).name
            base       = field.base
        else:
            raise TypeError('atom must be either scalar or vector field')

        # Obtain variable for storing point values of test function
        test_fun = SymbolicExpr(atom.subs(base, test_function))

        # ...
        orders = [*get_index(atom).values()]
        args   = [b[i, d, q] for b, i, d, q in zip(basis, idxs, orders, indices_quad)]
        inits += [Assign(test_fun, Mul(*args))]
        # ...

        # ...
        args     = [IndexedBase(field_name)[idxs], test_fun]
        val_name = SymbolicExpr(atom).name + '_values'
        val      = IndexedBase(val_name)[indices_quad]
        updates += [AugAssign(val,'+',Mul(*args))]
        # ...

    return inits, updates, map_stmts, new_atoms

#==============================================================================
# TODO: merge into 'compute_atoms_expr_field'
def compute_atoms_expr_mapping(atomic_exprs, indices_quad, idxs, basis, test_function):

    """
    This function computes atomic expressions needed
    to evaluate  EvalMapping final expression

    Parameters
    ----------

    atomic_exprs : <list>
        list of atoms

    indices_quad : <list>
        list of quadrature indices used in the quadrature loops

    idxs : <list>
        list of basis functions indices used in the for loops of the basis functions

    basis : <list>
        list of basis functions in each dimesion

    test_function : <Symbol>
        test_function Symbol

    Returns
    -------
    inits : <list>
       list of assignments of the atomic expression evaluated in the quadrature points

    updates : <list>
        list of augmented assignments which are updated in each loop iteration
    """

    inits   = []
    updates = []
    for atom in atomic_exprs:

        element = get_atom_logical_derivatives(atom)
        element_name = 'coeff_' + SymbolicExpr(element).name

        # ...
        test_fun = atom.subs(element, test_function)
        test_fun = SymbolicExpr(test_fun)
        # ...

        # ...
        orders = [*get_index_logical_derivatives(atom).values()]
        args   = [b[i, d, q] for b, i, d, q in zip(basis, idxs, orders, indices_quad)]
        inits += [Assign(test_fun, Mul(*args))]
        # ...

        # ...
        val_name = SymbolicExpr(atom).name + '_values'
        val      = IndexedBase(val_name)[indices_quad]
        expr     = IndexedBase(element_name)[idxs] * test_fun
        updates += [AugAssign(val, '+', expr)]
        # ...

    return inits, updates

#==============================================================================
def rationalize_eval_mapping(mapping, nderiv, space, indices_quad):

    M = mapping
    dim = space.ldim
    ops = _logical_partial_derivatives[:dim]

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
    weights = element_of(space, name='w')

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
        atom_name = SymbolicExpr(atom).name
        val_name = atom_name + '_values'
        val  = IndexedBase(val_name)[indices_quad]

        stmt = Assign(atom_name, val)
        stmts += [stmt]

    # assignements
    stmts += [Comment('rationalize')]

    # 0 order terms
    for i in range(dim):
        w = SymbolicExpr(weights)
        u = SymbolicExpr(M[i])

        val_name = u.name + '_values'
        val  = IndexedBase(val_name)[indices_quad]
        stmt = Assign(val, u / w )

        stmts += [stmt]

    # 1 order terms
    if nderiv >= 1:
        for d in ops:
            w  = SymbolicExpr(  weights )
            dw = SymbolicExpr(d(weights))

            for i in range(dim):
                u  = SymbolicExpr(  M[i] )
                du = SymbolicExpr(d(M[i]))

                val_name = du.name + '_values'
                val  = IndexedBase(val_name)[indices_quad]
                stmt = Assign(val, du / w - u * dw / w**2 )

                stmts += [stmt]

    # 2 order terms
    if nderiv >= 2:
        for e, d1 in enumerate(ops):
            for d2 in ops[:e+1]:
                w     = SymbolicExpr(      weights  )
                d1w   = SymbolicExpr(   d1(weights) )
                d2w   = SymbolicExpr(   d2(weights) )
                d1d2w = SymbolicExpr(d1(d2(weights)))

                for i in range(dim):
                    u     = SymbolicExpr(      M[i]  )
                    d1u   = SymbolicExpr(   d1(M[i]) )
                    d2u   = SymbolicExpr(   d2(M[i]) )
                    d1d2u = SymbolicExpr(d1(d2(M[i])))

                    val_name = d1d2u.name + '_values'
                    val  = IndexedBase(val_name)[indices_quad]
                    stmt = Assign(val,
                            d1d2u / w - u * d1d2w / w**2
                            - d1w * d2u / w**2 - d2w * d1u / w**2
                            + 2 * u * d1w * d2w / w**3)

                    stmts += [stmt]

    return stmts

#==============================================================================
_range = re.compile('([0-9]*:[0-9]+|[a-zA-Z]?:[a-zA-Z])')

def variables(names, dtype, **args):

    def contruct_variable(cls, name, dtype, rank, **args):
        if issubclass(cls, Variable):
            return Variable(dtype,  name, rank=rank, **args)
        elif issubclass(cls, IndexedVariable):
            return IndexedVariable(name, dtype=dtype, rank=rank, **args)
        elif cls==Idx:
            assert dtype == "int"
            rank = args.pop('rank', 0)
            assert rank == 0
            return Idx(name)
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

#==============================================================================
def build_pyccel_types_decorator(args, order=None):
    """
    builds a types decorator from a list of arguments (of FunctionDef)
    """
    types = []
    for a in args:
        if isinstance(a, Variable):
            rank  = a.rank
            dtype = a.dtype.name.lower()

        elif isinstance(a, IndexedVariable):
            rank  = a.rank
            dtype = a.dtype.name.lower()

        elif isinstance(a, Constant):
            rank = 0
            if a.is_integer:
                dtype = 'int'
            elif a.is_real:
                dtype = 'float'
            elif a.is_complex:
                dtype = 'complex'
            else:
                dtype = 'float' # default value

        else:
            raise TypeError('unexpected type for {}'.format(a))

        if rank > 0:
            shape = ','.join(':' * rank)
            dtype = '{dtype}[{shape}]'.format(dtype=dtype, shape=shape)
            if order and rank > 1:
                dtype = "{dtype}(order={ordering})".format(dtype=dtype, ordering=order)

        dtype = String(dtype)
        types.append(dtype)

    return types

#==============================================================================
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

#==============================================================================

from sympy import preorder_traversal, NumberSymbol, Pow, S
from sympy.printing.pycode import _known_functions_math, _known_constants_math, _known_functions_mpmath, _known_constants_mpmath, _known_functions_numpy


def math_atoms_as_str(expr, lib='math'):
    """
    Given a Sympy expression, find all known mathematical atoms (functions and
    constants) that need to be imported from a math library (e.g. Numpy) when
    generating Python code.

    Parameters
    ----------
    expr : sympy.core.expr.Expr
        Symbolic expression for which Python code is to be generated.

    lib : str
        Library used to translate symbolic functions/constants into standard
        Python ones. Options: ['math', 'mpmath', 'numpy']. Default: 'math'.

    Returns
    -------
    imports : set of str
        Set of all names (strings) to be imported.

    """

    # Choose translation dictionaries
    if lib == 'math':
        known_functions = _known_functions_math
        known_constants = _known_constants_math
    elif lib == 'mpmath':
        known_functions = _known_functions_mpmath
        known_constants = _known_constants_mpmath
    elif lib == 'numpy':
        known_functions = _known_functions_numpy
        known_constants = _known_constants_math   # numpy version missing
    else:
        raise ValueError("Library {} not supported.".format(mod))

    # Initialize variables
    math_functions = set()
    math_constants = set()
    sqrt = False

    # Walk expression tree
    for i in preorder_traversal(expr):

        # Search for math functions (e.g. cos, sin, exp, ...)
        if isinstance(i, Function):
            s = str(type(i))
            if s in known_functions:
                p = known_functions[s]
                math_functions.add(p)

        # Search for math constants (e.g. pi, e, ...)
        elif isinstance(i, NumberSymbol):
            s = type(i).__name__
            if s in known_constants:
                p = known_constants[s]
                math_constants.add(p)

        # Search for square roots
        elif (not sqrt):
            if isinstance(i, Pow) and ((i.exp is S.Half) or (i.exp == -S.Half)):
                math_functions.add('sqrt')
                sqrt = True

    return set.union(math_functions, math_constants)
