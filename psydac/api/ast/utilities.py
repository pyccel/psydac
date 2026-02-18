#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import re
import string
import random
from itertools import chain

from sympy import Symbol, IndexedBase, Indexed, Idx
from sympy import Mul, Pow, Function, Tuple
from sympy import sqrt as sympy_sqrt, Range
from sympy.utilities.iterables import cartes

from sympde.topology.space       import ScalarFunction
from sympde.topology.space       import VectorFunction
from sympde.topology.space       import IndexedVectorFunction
from sympde.topology.space       import element_of
from sympde.topology             import Mapping
from sympde.topology             import Boundary
from sympde.topology.derivatives import _partial_derivatives
from sympde.topology.derivatives import _logical_partial_derivatives
from sympde.topology.derivatives import get_atom_derivatives
from sympde.topology.derivatives import get_index_derivatives
from sympde.topology.derivatives import get_atom_logical_derivatives
from sympde.topology.derivatives import get_index_logical_derivatives
from sympde.topology.derivatives import get_index_derivatives_atom, get_index_logical_derivatives_atom
from sympde.topology             import LogicalExpr
from sympde.topology             import SymbolicExpr
from sympde.core                 import Constant

from psydac.pyccel.ast.core import Variable, IndexedVariable
from psydac.pyccel.ast.core import For
from psydac.pyccel.ast.core import Assign
from psydac.pyccel.ast.core import AugAssign
from psydac.pyccel.ast.core import Product
from psydac.pyccel.ast.core import _atomic
from psydac.pyccel.ast.core import Comment
from psydac.pyccel.ast.core import String
from psydac.pyccel.ast.core import AnnotatedArgument

__all__ = (
    'build_pyccel_type_annotations',
    'build_pythran_types_header',
    'compute_atoms_expr',
    'compute_atoms_expr_field',
    'compute_atoms_expr_mapping',
    'compute_boundary_jacobian',
    'compute_normal_vector',
    'compute_tangent_vector',
    'filter_loops',
    'filter_product',
    'fusion_loops',
    'get_name',
    'is_mapping',
    'logical2physical',
    'math_atoms_as_str',
    'rationalize_eval_mapping',
    'select_loops',
    'variables',
)

#==============================================================================
def get_max_partial_derivatives(expr, logical=False, F=None):
    """
    Compute the maximum order of partial derivatives for each coordinate in an expression.

    Parameters
    ----------
    expr : sympy.Expr
        The SymPDE expression to analyze for partial derivatives.

    logical : bool, optional
        If True, it considers logical coordinates (x1, x2, x3); otherwise, it considers physical coordinates (x, y, z).
    
    F : sympy.Atom | list[sympy.Atom], optional
        If provided, it restricts the analysis to the specified atom(s). Otherwise,
        it uses all atoms of default types that are contained in `expr`. The default
        types represent elements of function spaces in SymPDE: `ScalarFunction`,
        `VectorFunction`, and `IndexedVectorFunction`.

    Returns
    -------
    d : dict[str, int]
        A dictionary with keys ('x1', 'x2', 'x3') for logical or ('x', 'y', 'z') for physical coordinates and their corresponding maximum order of partial derivatives.

    Notes
    -----
    [TODO] Move to SymPDE and combine the `get_index(_logical)_derivatives_atom` functions there.
    """
    if logical:
        d = {'x1': 0, 'x2': 0, 'x3': 0}
        get_index = get_index_logical_derivatives_atom
    else:
        d = {'x': 0, 'y': 0, 'z': 0}
        get_index = get_index_derivatives_atom

    if F is None:
        F = (list(expr.atoms(ScalarFunction)) +
             list(expr.atoms(VectorFunction)) +
             list(expr.atoms(IndexedVectorFunction)))
    elif not hasattr(F, '__iter__'):
        F = [F]

    indices = chain.from_iterable(get_index(expr, Fi) for Fi in F)

    for dd in indices:
        for k, v in dd.items():
            if v > d[k]:
                d[k] = v

    return d

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
def logical2physical(expr):

    partial_der = dict(zip(_logical_partial_derivatives,_partial_derivatives))
    
    if isinstance(expr, _logical_partial_derivatives):
        argument = logical2physical(expr.args[0])
        new_expr = partial_der[type(expr)](argument)
        return new_expr
    else:
        return expr
#==============================================================================
def _get_name(atom):
    atom_name = None
    if isinstance( atom, ScalarFunction ):
        atom_name = str(atom.name)

    elif isinstance( atom, VectorFunction ):
        atom_name = str(atom.name)

    elif isinstance( atom, IndexedVectorFunction ):
        atom_name = str(atom.base.name)

    else:
        raise TypeError('> Wrong type')

    return atom_name
        
#==============================================================================
def compute_atoms_expr(atomic_exprs, indices_quad, indices_test,
                       indices_trial, basis_trial,
                       basis_test, cords, test_function,
                       is_linear,
                       mapping):

    """
    This function computes atomic expressions needed
    to evaluate  the Kernel final expression

    Parameters
    ----------
    atomic_exprs : <list>
        list of atoms

    indices_quad : <list>
        list of quadrature indices used in the quadrature loops

    indices_test : <list>
        list of  test_functions indices used in the for loops of the basis functions

    indices_trial : <list>
        list of  trial_functions indices used in the for loops of the basis functions

    basis_test : <list>
        list of basis functions in each dimesion

    cords : <list>
        list of coordinates Symbols

    test_function : <Symbol>
        test_function Symbol

    is_linear : <boolean>
        variable to determine if we are in the linear case

    mapping : <Mapping>
        Mapping object

    Returns
    -------
    inits : <list>
       list of assignments of the atomic expression evaluated in the quadrature points

    map_stmts : <list>
        list of assigments of atomic expression in case of mapping

    """

    cls = (_partial_derivatives,
           VectorFunction,
           ScalarFunction,
           IndexedVectorFunction)
    
    dim  = len(indices_test)

    if not isinstance(atomic_exprs, (list, tuple, Tuple)):
        raise TypeError('Expecting a list of atoms')
    
    for atom in atomic_exprs:   
        if not isinstance(atom, cls):
            raise TypeError('atom must be of type {}'.format(str(cls)))
    
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

    # Create a list of statements for initialization of the point values,
    # for each of the atoms in our (possibly new) list.
    inits = []
    for atom in new_atoms:
        orders = [*get_index(atom).values()]
        a      = get_atom(atom)
        test   = _get_name(a) in [_get_name(f) for f in test_function]

        if test or is_linear:
            basis  = basis_test
            idxs   = indices_test
        else:
            basis  = basis_trial
            idxs   = indices_trial

        args   = [b[i, d, q] for b, i, d, q in zip(basis, idxs, orders, indices_quad)]
        lhs    = SymbolicExpr(atom)
        rhs    = Mul(*args)
        inits += [Assign(lhs, rhs)]

    # Return the initialization statements, and the additional initialization
    # of intermediate variables in case of mapping
    return inits, map_stmts

#==============================================================================
def compute_atoms_expr_field(atomic_exprs, indices_quad,
                            idxs, basis,
                            test_function, mapping):

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
        # ...

        # ...
        args     = [IndexedBase(field_name)[idxs], test_fun]
        val_name = SymbolicExpr(atom).name + '_values'
        val      = IndexedBase(val_name)[indices_quad]
        updates += [AugAssign(val,'+',Mul(*args))]
        # ...

    return inits, updates, map_stmts, new_atoms

#==============================================================================
# TODO: merge into 'compute_atoms_expr_field'
def compute_atoms_expr_mapping(atomic_exprs, indices_quad,
                               idxs, basis,
                               test_function):

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

        # ...
        test_fun = atom.subs(element, test_function)
        test_fun = SymbolicExpr(test_fun)
        # ...

        # ...
        orders = [*get_index_logical_derivatives(atom).values()]
        args   = [b[i, d, q] for b, i, d, q in zip(basis, idxs, orders, indices_quad)]
        inits += [Assign(test_fun, Mul(*args))]
        # ...

        # ...
        val_name = SymbolicExpr(atom).name + '_values'
        val      = IndexedBase(val_name)[indices_quad]
        expr     = IndexedBase(element_name)[idxs] * test_fun
        updates += [AugAssign(val, '+', expr)]
        # ...

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
def filter_product(indices, args, boundary):

    mask = []
    ext = []
    if boundary:

        if isinstance(boundary, Boundary):
            mask = [boundary.axis]
            ext  = [boundary.ext]
        else:
            raise TypeError

        # discrete_boundary gives the perpendicular indices, then we need to
        # remove them from directions

    dim = len(indices)
    args = [args[i][indices[i]] for i in range(dim) if not(i in mask)]

    return Mul(*args)

#==============================================================================
# TODO remove it later
def filter_loops(indices, ranges, body, boundary, boundary_basis=False):

    quad_mask = []
    quad_ext = []
    if boundary:

        if isinstance(boundary, Boundary):
            quad_mask = [boundary.axis]
            quad_ext  = [boundary.ext]
        else:
            raise TypeError

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
def select_loops(indices, ranges, body, boundary, boundary_basis=False):

    quad_mask = []
    quad_ext = []
    if boundary:

        if isinstance(boundary, Boundary):
            quad_mask = [boundary.axis]
            quad_ext  = [boundary.ext]
        else:
            raise TypeError

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
def compute_boundary_jacobian(parent_namespace, boundary, mapping=None):

    # Sanity check on arguments
    if not isinstance(boundary, Boundary):
        raise TypeError(boundary)

    if mapping is None:
        stmts = []

    else:
        # Compute metric determinant g on manifold
        J  = SymbolicExpr(mapping.jacobian)
        Jm = J[:, [i for i in range(J.shape[1]) if i != boundary.axis]]
        g  = (Jm.T * Jm).det()

        # Create statements for computing sqrt(g)
        det_jac_bnd = parent_namespace['det_jac_bnd']
        stmts       = [Assign(det_jac_bnd, sympy_sqrt(g))]

    return stmts

#==============================================================================
def compute_normal_vector(parent_namespace, vector, boundary, mapping=None):

    # Sanity check on arguments
    if isinstance(boundary, Boundary):
        axis = boundary.axis
        ext  = boundary.ext
    else:
        raise TypeError(boundary)

    # If there is no mapping, normal vector has only one non-zero component,
    # which is +1 or -1 according to the orientation of the boundary.
    if mapping is None:
        return [Assign(v, ext if i==axis else 0) for i, v in enumerate(vector)]

    # Given the Jacobian matrix J, we need to extract the (i=axis) row of
    # J^(-1) and then normalize it. We recall that J^(-1)[i, j] is equal to
    # the cofactor of J[i, j] divided by det(J). For efficiency we only
    # compute the cofactors C[i=0:dim] of the (j=axis) column of J, and we
    # do not divide them by det(J) because the normal vector will need to
    # be normalized anyway.
    #
    # NOTE: we also change the vector orientation according to 'ext'
    J = SymbolicExpr(mapping.jacobian)
    values = [ext * J.cofactor(i, j=axis) for i in range(J.shape[0])]

    # Create statements for computing normal vector components
    stmts = [Assign(lhs, rhs) for lhs, rhs in zip(vector, values)]

    # Normalize vector
    inv_norm_variable = Symbol('inv_norm')
    inv_norm_value    = 1 / sympy_sqrt(sum(v**2 for v in values))
    stmts += [Assign(inv_norm_variable, inv_norm_value)]
    stmts += [AugAssign(v, '*', inv_norm_variable) for v in vector]

    return stmts

#==============================================================================
def compute_tangent_vector(parent_namespace, vector, boundary, mapping):
    raise NotImplementedError('TODO')

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
def build_pyccel_type_annotations(args, order=None):

    new_args = []

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
                raise TypeError(f"The Constant {a} don't have any information about the type of the variable.\n"
                                f"Please create the Constant like this Constant('{a}', real=True), Constant('{a}', complex=True) or Constant('{a}', integer=True).")

        else:
            raise TypeError('unexpected type for {}'.format(a))

        if rank > 0:
            shape = ','.join(':' * rank)
            dtype = '{dtype}[{shape}]'.format(dtype=dtype, shape=shape)
            if order and rank > 1:
                dtype = "{dtype}(order={ordering})".format(dtype=dtype, ordering=order)

        dtype = String(dtype)
        new_a = AnnotatedArgument(a, dtype)
        new_args.append(new_a)

    return new_args

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

from sympy import preorder_traversal
from sympy import NumberSymbol
from sympy import Pow, S

_known_functions_math = {
    'acos': 'acos',
    'acosh': 'acosh',
    'asin': 'asin',
    'asinh': 'asinh',
    'atan': 'atan',
    'atan2': 'atan2',
    'atanh': 'atanh',
    'ceiling': 'ceil',
    'cos': 'cos',
    'cosh': 'cosh',
    'erf': 'erf',
    'erfc': 'erfc',
    'exp': 'exp',
    'expm1': 'expm1',
    'factorial': 'factorial',
    'floor': 'floor',
    'gamma': 'gamma',
    'hypot': 'hypot',
    'loggamma': 'lgamma',
    'log': 'log',
    'ln': 'log',
    'log10': 'log10',
    'log1p': 'log1p',
    'log2': 'log2',
    'sin': 'sin',
    'sinh': 'sinh',
    'Sqrt': 'sqrt',
    'tan': 'tan',
    'tanh': 'tanh'

}  # Not used from ``math``: [copysign isclose isfinite isinf isnan ldexp frexp pow modf
# radians trunc fmod fsum gcd degrees fabs]
_known_constants_math = {
    'Exp1': 'e',
    'Pi': 'pi',
    'E': 'e'
    # Only in python >= 3.5:
    # 'Infinity': 'inf',
    # 'NaN': 'nan'
}

_not_in_mpmath = 'log1p log2'.split()
_in_mpmath = [(k, v) for k, v in _known_functions_math.items() if k not in _not_in_mpmath]
_known_functions_mpmath = dict(_in_mpmath, **{
    'beta': 'beta',
    'fresnelc': 'fresnelc',
    'fresnels': 'fresnels',
    'sign': 'sign',
})
_known_constants_mpmath = {
    'Exp1': 'e',
    'Pi': 'pi',
    'GoldenRatio': 'phi',
    'EulerGamma': 'euler',
    'Catalan': 'catalan',
    'NaN': 'nan',
    'Infinity': 'inf',
    'NegativeInfinity': 'ninf'
}

_not_in_numpy = 'erf erfc factorial gamma loggamma'.split()
_in_numpy = [(k, v) for k, v in _known_functions_math.items() if k not in _not_in_numpy]
_known_functions_numpy = dict(_in_numpy, **{
    'acos': 'arccos',
    'acosh': 'arccosh',
    'asin': 'arcsin',
    'asinh': 'arcsinh',
    'atan': 'arctan',
    'atan2': 'arctan2',
    'atanh': 'arctanh',
    'exp2': 'exp2',
    'sign': 'sign',
})
_known_constants_numpy = {
    'Exp1': 'e',
    'Pi': 'pi',
    'EulerGamma': 'euler_gamma',
    'NaN': 'nan',
    'Infinity': 'PINF',
    'NegativeInfinity': 'NINF'
}


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
        known_constants = _known_constants_numpy   # numpy version missing
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

def get_name(lhs):
    """
    Given a list of variable return the meaningful part of the name of the
    first variable that has a _name attribute.

    Was added to solve issue #327 caused by trying to access the name of a 
    variable that has not such attribute.

    Parameters
    ----------
    lhs : list
        list from whom we need to extract a name.

    Returns
    -------
    str
        meaningful part of the name of the variable or "zero term" if no 
        variable has a name.

    """
    for term in lhs:
        if hasattr(term, '_name'):
            return term._name[12:-8]
    return "zero_term"
