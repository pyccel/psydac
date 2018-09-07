from collections import OrderedDict

from sympy import Basic
from sympy import symbols, Symbol, IndexedBase, Indexed, Matrix, Function
from sympy import Mul, Add, Tuple

from pyccel.ast.core import For
from pyccel.ast.core import Assign
from pyccel.ast.core import AugAssign
from pyccel.ast.core import Slice
from pyccel.ast.core import Range
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
from pyccel.parser.parser import _atomic

from sympde.core import Constant
from sympde.core import Field
from sympde.core import atomize
from sympde.core import BilinearForm, LinearForm, FunctionForm, BasicForm
from sympde.core.derivatives import _partial_derivatives
from sympde.core.space import TestFunction
from sympde.core.space import VectorTestFunction
from sympde.core import BilinearForm, LinearForm, FunctionForm
from sympde.printing.pycode import pycode  # TODO remove from here
from sympde.core.derivatives import print_expression

FunctionalForms = (BilinearForm, LinearForm, FunctionForm)

def compute_atoms_expr(atom,indices_qds,indices_test,
                      indices_trial, basis_trial,
                      basis_test,cords,test_function):

    cls = (_partial_derivatives,
           VectorTestFunction,
           TestFunction)

    if not isinstance(atom, cls):
        raise TypeError('atom must be of type {}'.format(str(cls)))

    if isinstance(atom, _partial_derivatives):
        direction = atom.grad_index + 1
        test      = test_function in atom.atoms(TestFunction)
    else:
        direction = 0
        test      = atom == test_function

    if test:
        basis  = basis_test
        idxs   = indices_test
    else:
        basis  = basis_trial
        idxs   = indices_trial

    args = []
    dim  = len(indices_test)
    for i in range(dim):
        if direction == i+1:
            args.append(basis[i][idxs[i],1,indices_qds[i]])
        else:
            args.append(basis[i][idxs[i],0,indices_qds[i]])
    return Assign(atom, Mul(*args))

def compute_atoms_expr_field(atom, indices_qds,
                            idxs, basis,
                            test_function):

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
            args.append(basis[i][idxs[i],1,indices_qds[i]])

        else:
            args.append(basis[i][idxs[i],0,indices_qds[i]])

    init = Assign(test_function, Mul(*args))
    # ...

    # ...
    args = [IndexedBase(field_name)[idxs], test_function]
    val_name = print_expression(atom) + '_values'
    val  = IndexedBase(val_name)[indices_qds]
    update = AugAssign(val,'+',Mul(*args))
    # ...

    return init, update

def is_field(expr):

    if isinstance(expr, _partial_derivatives):
        return is_field(expr.args[0])

    elif isinstance(expr, Field):
        return True

    return False

class SplBasic(Basic):

    def __new__(cls, arg, name=None, prefix=None, debug=False, detailed=False):

        if name is None:
            if prefix is None:
                raise ValueError('prefix must be given')

            ID = abs(hash(arg))
            name = '{prefix}_{ID}'.format(ID=ID, prefix=prefix)

        obj = Basic.__new__(cls)
        obj._name = name
        obj._dependencies = []
        obj._debug = debug
        obj._detailed = detailed

        return obj

    @property
    def name(self):
        return self._name

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

class EvalField(SplBasic):

    def __new__(cls, space, fields, name=None):

        if not isinstance(fields, (tuple, list, Tuple)):
            raise TypeError('> Expecting an iterable')

        obj = SplBasic.__new__(cls, space, name=name, prefix='eval_field')

        obj._space = space
        obj._fields = Tuple(*fields)
        obj._func = obj._initialize()

        return obj

    @property
    def space(self):
        return self._space

    @property
    def fields(self):
        return self._fields

    def build_arguments(self, data):

        other = data

        return self.basic_args + other

    def _initialize(self):
        space = self.space
        dim = space.ldim

        field_atoms = self.fields.atoms(Field)
        fields_str = [print_expression(f) for f in self.fields]

        # ... declarations
        degrees       = symbols('p1:%d'%(dim+1))
        orders        = symbols('k1:%d'%(dim+1))
        basis         = symbols('basis1:%d'%(dim+1), cls=IndexedBase)
        indices_basis = symbols('jl1:%d'%(dim+1))
        indices_quad  = symbols('g1:%d'%(dim+1))
        fields_val    = symbols(tuple(f+'_values' for f in fields_str),cls=IndexedBase)
        fields_coeffs = symbols(tuple('coeff_'+str(f) for f in field_atoms),cls=IndexedBase)
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
        updates = []
        for atom in self.fields:
            init, update = compute_atoms_expr_field(atom, indices_quad, indices_basis,
                                                    basis, Nj)

            updates.append(update)

            basis_name = str(init.lhs)
            init_basis[basis_name] = init

        body += list(init_basis.values())
        body += updates
        # ...

        # put the body in tests for loops
        for i in range(dim-1,-1,-1):
            body = [For(indices_basis[i], ranges_basis[i],body)]

        # put the body in for loops of quadrature points
        for i in range(dim-1,-1,-1):
            body = [For(indices_quad[i], ranges_quad[i],body)]

        # initialization of the matrix
        init_vals = [f[[Slice(None,None)]*dim] for f in fields_val]
        init_vals = [Assign(e, 0.0) for e in init_vals]
        body =  init_vals + body

        func_args = self.build_arguments(degrees + basis + fields_coeffs + fields_val)

        return FunctionDef(self.name, list(func_args), [], body)

class Kernel(SplBasic):

    def __new__(cls, weak_form, name=None):

        if not isinstance(weak_form, FunctionalForms):
            raise TypeError('> Expecting a weak formulation')

        obj = SplBasic.__new__(cls, weak_form, name=name, prefix='kernel')

        obj._weak_form = weak_form
        obj._n_rows = 1
        obj._n_cols = 1
        obj._func = obj._initialize()

        return obj

    @property
    def weak_form(self):
        return self._weak_form

    @property
    def n_rows(self):
        return self._n_rows

    @property
    def n_cols(self):
        return self._n_cols

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
    def mapping_coeffs(self):
        return self._mapping_coeffs

    @property
    def eval_fields(self):
        return self._eval_fields

    def build_arguments(self, data):

        other = data

        # fields are placed before data
        if self.fields_coeffs:
            other = self.fields_coeffs + other

        if self.constants:
            other = other + self.constants

        if self.mapping_coeffs:
            other = other + (self.mapping_coeffs,)

        return self.basic_args + other

    def _initialize(self):

        is_linear   = isinstance(self.weak_form, LinearForm)
        is_bilinear = isinstance(self.weak_form, BilinearForm)
        is_function = isinstance(self.weak_form, FunctionForm)

        weak_form = self.weak_form.expr
        expr = atomize(weak_form)

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
        self._constants = constants
        # ...

        atoms_types = (_partial_derivatives, VectorTestFunction, TestFunction,
                       Field)
        atoms  = _atomic(expr, cls=atoms_types)

        atomic_expr_field = [atom for atom in atoms if is_field(atom)]
        atomic_expr       = [atom for atom in atoms if atom not in atomic_expr_field ]

        fields_str    = tuple(map(pycode, atomic_expr_field))
        field_atoms   = tuple(expr.atoms(Field))

        # ... create EvalField
        # TODO must use groupby on space and create different EvalField
        self._eval_fields = []
        if atomic_expr_field:
            space = self.weak_form.test_spaces[0]
            eval_field = EvalField(space, atomic_expr_field)
            self._eval_fields.append(eval_field)

        # update dependencies
        self._dependencies += self.eval_fields
        # ...

        test_function = self.weak_form.test_functions[0]

        # creation of symbolic vars
        if isinstance(expr, Matrix):
            sh   = expr.shape
            mats = symbols('mat0:{}(0:{})'.format(sh[0], sh[1]),cls=IndexedBase)
            v    = symbols('v0:{}(0:{})'.format(sh[0], sh[1]),cls=IndexedBase)
            expr = expr[:]
            ln   = len(expr)

        else:
            mats = (IndexedBase('mat_00'),)
            v    = [symbols('v_00')]
            expr = [expr]
            ln   = 1

        # ... declarations
        wvol          = symbols('wvol')
        basis_trial   = symbols('trial_bs1:%d'%(dim+1), cls=IndexedBase)
        basis_test    = symbols('test_bs1:%d'%(dim+1), cls=IndexedBase)
        weighted_vols = symbols('w1:%d'%(dim+1), cls=IndexedBase)
        positions     = symbols('u1:%d'%(dim+1), cls=IndexedBase)
        test_pads     = symbols('test_p1:%d'%(dim+1))
        trial_pads    = symbols('trial_p1:%d'%(dim+1))
        test_degrees  = symbols('test_p1:%d'%(dim+1))
        trial_degrees = symbols('trial_p1:%d'%(dim+1))
        indices_qds   = symbols('g1:%d'%(dim+1))
        qds_dim       = symbols('k1:%d'%(dim+1))
        indices_test  = symbols('il1:%d'%(dim+1))
        indices_trial = symbols('jl1:%d'%(dim+1))
        fields        = symbols(fields_str)
        fields_val    = symbols(tuple(f+'_values' for f in fields_str),cls=IndexedBase)
        fields_coeffs = symbols(tuple('coeff_'+str(f) for f in field_atoms),cls=IndexedBase)

        # TODO
        mapping_coeffs = None
        # ...

        # ...
        self._basic_args = (test_pads + trial_pads +
                            basis_test + basis_trial +
                            positions + weighted_vols)
        # ...

        # ...
        self._fields = fields
        self._fields_coeffs = fields_coeffs
        self._mapping_coeffs = mapping_coeffs
        # ...

        # ranges
        ranges_qdr   = [Range(qds_dim[i]) for i in range(dim)]
        ranges_test  = [Range(test_degrees[i]+1) for i in range(dim_test)]
        ranges_trial = [Range(trial_degrees[i]+1) for i in range(dim_trial)]
        # ...

        # body of kernel
        body   = [Assign(coordinates[i],positions[i][indices_qds[i]])\
                 for i in range(dim)]
        body  += [compute_atoms_expr(atom,indices_qds,
                                      indices_test,
                                      indices_trial,
                                      basis_trial,
                                      basis_test,
                                      coordinates,
                                      test_function) for atom in atomic_expr]

        weighted_vol = [weighted_vols[i][indices_qds[i]] for i in range(dim)]
        weighted_vol = Mul(*weighted_vol)
        # ...
        # add fields
        for i in range(len(fields_val)):
            body.append(Assign(fields[i],fields_val[i][indices_qds]))

        body.append(Assign(wvol,weighted_vol))

        for i in range(ln):
            body.append(AugAssign(v[i],'+',Mul(expr[i],wvol)))
        # ...

        # ...
        # put the body in for loops of quadrature points
        for i in range(dim-1,-1,-1):
            body = [For(indices_qds[i],ranges_qdr[i],body)]

        # initialization of intermediate vars
        init_vars = [Assign(v[i],0.0) for i in range(ln)]
        body = init_vars + body
        # ...

        if dim_trial:
            trial_idxs = tuple([indices_trial[i]+trial_pads[i]-indices_test[i] for i in range(dim)])
            idxs = indices_test + trial_idxs
        else:
            idxs = indices_test

        for i in range(ln):
            body.append(Assign(mats[i][idxs],v[i]))

        # ...
        # put the body in tests and trials for loops
        for i in range(dim_trial-1,-1,-1):
            body = [For(indices_trial[i],ranges_trial[i],body)]

        for i in range(dim_test-1,-1,-1):
            body = [For(indices_test[i],ranges_test[i],body)]
        # ...

        # ...
        # initialization of the matrix
        init_mats = [mats[i][[Slice(None,None)]*(dim_test+dim_trial)] for i in range(ln)]
        init_mats = [Assign(e, 0.0) for e in init_mats]
        body =  init_mats + body

        # call eval field
        for eval_field in self.eval_fields:
            args = test_degrees + basis_test + fields_coeffs + fields_val
            args = eval_field.build_arguments(args)
            body = [FunctionCall(eval_field.func, args)] + body

        # calculate field values
        if fields_val:
            prelude  = [Import('zeros', 'numpy')]
            allocate = [Assign(f, Zeros(qds_dim)) for f in fields_val]
            body = prelude + allocate + body

        # compute length of logical points
        len_quads = [Assign(k, Len(u)) for k,u in zip(qds_dim, positions)]
        body = len_quads + body

        # function args
        func_args = self.build_arguments(mats)

        return FunctionDef(self.name, list(func_args), [], body)

class Assembly(SplBasic):

    def __new__(cls, weak_form, name=None):

        if not isinstance(weak_form, FunctionalForms):
            raise TypeError('> Expecting a weak formulation')

        obj = SplBasic.__new__(cls, weak_form, name=name, prefix='assembly')

        obj._weak_form = weak_form

        kernel = Kernel(weak_form)
        obj._kernel = kernel

        # update dependencies
        obj._dependencies += [kernel]

        obj._func = obj._initialize()
        return obj

    @property
    def weak_form(self):
        return self._weak_form

    @property
    def kernel(self):
        return self._kernel

    @property
    def global_matrices(self):
        return self._global_matrices

    def build_arguments(self, data):

        other = data

        if self.kernel.constants:
            other = other + self.kernel.constants

        if self.kernel.mapping_coeffs:
            other = other + (self.kernel.mapping_coeffs,)

        return self.basic_args + other

    def _initialize(self):
        kernel = self.kernel
        form   = self.weak_form
        fields = kernel.fields
        fields_coeffs = kernel.fields_coeffs

        dim    = form.ldim

        n_rows = kernel.n_rows
        n_cols = kernel.n_cols

        # ... declarations
        starts             = symbols('s1:%d'%(dim+1))
        ends               = symbols('e1:%d'%(dim+1))
        test_pads          = symbols('test_p1:%d'%(dim+1))
        trial_pads         = symbols('trial_p1:%d'%(dim+1))
        test_degrees       = symbols('test_p1:%d'%(dim+1))
        trial_degrees      = symbols('trial_p1:%d'%(dim+1))
        points             = symbols('points_1:%d'%(dim+1), cls=IndexedBase)
        weights            = symbols('weights_1:%d'%(dim+1), cls=IndexedBase)
        trial_basis        = symbols('trial_basis_1:%d'%(dim+1), cls=IndexedBase)
        test_basis         = symbols('test_basis_1:%d'%(dim+1), cls=IndexedBase)
        indices_elm        = symbols('ie1:%d'%(dim+1))
        indices_span       = symbols('is1:%d'%(dim+1))
        points_in_elm      = symbols('u1:%d'%(dim+1), cls=IndexedBase)
        weights_in_elm     = symbols('w1:%d'%(dim+1), cls=IndexedBase)
        spans              = symbols('test_spans_1:%d'%(dim+1), cls=IndexedBase)
        trial_basis_in_elm = symbols('trial_bs1:%d'%(dim+1), cls=IndexedBase)
        test_basis_in_elm  = symbols('test_bs1:%d'%(dim+1), cls=IndexedBase)

        # TODO remove later and replace by Len inside Kernel
        quad_orders    = symbols('k1:%d'%(dim+1))
        # ...

        # ...
        self._basic_args = (starts + ends +
                            test_degrees + trial_degrees +
                            spans +
                            points + weights +
                            test_basis + trial_basis)
        # ...

        # ... element matrices
        element_matrices = {}
        for i in range(0, n_rows):
            for j in range(0, n_cols):
                mat = IndexedBase('mat_{i}{j}'.format(i=i,j=j))
                element_matrices[i,j] = mat
        # ...

        # ... global matrices
        global_matrices = {}
        for i in range(0, n_rows):
            for j in range(0, n_cols):
                mat = IndexedBase('M_{i}{j}'.format(i=i,j=j))
                global_matrices[i,j] = mat
        # ...

        # sympy does not like ':'
        _slice = Slice(None,None)

        # ranges
        ranges_elm  = [Range(starts[i], ends[i]-test_pads[i]+1) for i in range(dim)]

        # assignments
        body  = [Assign(indices_span[i], spans[i][indices_elm[i]]) for i in range(dim)]
        if self.debug and self.detailed:
            msg = lambda x: (String('> span {} = '.format(x)), x)
            body += [Print(msg(indices_span[i])) for i in range(dim)]

        body += [Assign(points_in_elm[i], points[i][indices_elm[i],_slice]) for i in range(dim)]
        body += [Assign(weights_in_elm[i], weights[i][indices_elm[i],_slice]) for i in range(dim)]
        body += [Assign(trial_basis_in_elm[i], trial_basis[i][indices_elm[i],_slice,_slice,_slice]) for i in range(dim)]
        body += [Assign(test_basis_in_elm[i], test_basis[i][indices_elm[i],_slice,_slice,_slice]) for i in range(dim)]

        # kernel call
        args = kernel.func.arguments
        body += [FunctionCall(kernel.func, args)]

        # ... update global matrices
        lslices = [Slice(None,None)]*2*dim
        gslices = [Slice(i-p,i+1) for i,p in zip(indices_span, test_degrees)]
        gslices += [Slice(None,None)]*dim # for assignement

        for i in range(0, n_rows):
            for j in range(0, n_cols):
                M = global_matrices[i,j]
                mat = element_matrices[i,j]

                stmt = AugAssign(M[gslices], '+', mat[lslices])
                body += [stmt]
        # ...

        # ... loop over elements
        for i in range(dim-1,-1,-1):
            body = [For(indices_elm[i], ranges_elm[i], body)]
        # ...

        # ... prelude
        prelude = []

        # import zeros from numpy
        stmt = Import('zeros', 'numpy')
        prelude += [stmt]

        orders  = [p+1 for p in test_degrees]
        spads   = [2*p+1 for p in test_pads]
        for i in range(0, n_rows):
            for j in range(0, n_cols):
                mat = element_matrices[i,j]

                stmt = Assign(mat, Zeros((*orders, *spads)))
                prelude += [stmt]

                if self.debug:
                    prelude += [Print((String('> shape {} = '.format(mat)), *orders, *spads))]
        # ...

        # ...
        if self.debug:
            for i in range(0, n_rows):
                for j in range(0, n_cols):
                    M = global_matrices[i,j]
                    prelude += [Print((String('> shape {} = '.format(M)), Shape(M)))]
        # ...

        # ...
        body = prelude + body
        # ...

        # ...
        mats = []
        for i in range(0, n_rows):
            for j in range(0, n_cols):
                M = global_matrices[i,j]
                mats.append(M)
        mats = tuple(mats)
        self._global_matrices = mats
        # ...

        # function args
        func_args = self.build_arguments(fields_coeffs + mats)

        return FunctionDef(self.name, list(func_args), [], body)

class Interface(SplBasic):

    def __new__(cls, weak_form, name=None):

        if not isinstance(weak_form, FunctionalForms):
            raise TypeError('> Expecting a weak formulation')

        obj = SplBasic.__new__(cls, weak_form, name=name, prefix='interface')

        obj._weak_form = weak_form

        assembly = Assembly(weak_form)
        obj._assembly = assembly

        # update dependencies
        obj._dependencies += [assembly]

        obj._func = obj._initialize()
        return obj

    @property
    def weak_form(self):
        return self._weak_form

    @property
    def assembly(self):
        return self._assembly

    def build_arguments(self, data):
        # data must be at the end, since they are optional

        other = ()

        if self.assembly.kernel.constants:
            other = other + self.assembly.kernel.constants

        other = other + data

        return self.basic_args + other

    def _initialize(self):
        form = self.weak_form
        assembly = self.assembly
        global_matrices = assembly.global_matrices
        fields = assembly.kernel.fields

        dim = form.ldim

        # ... declarations
        test_space = Symbol('W')
        trial_space = Symbol('V')
        spaces = (test_space, trial_space)

        starts         = symbols('s1:%d'%(dim+1))
        ends           = symbols('e1:%d'%(dim+1))
        test_degrees   = symbols('test_p1:%d'%(dim+1))
        trial_degrees  = symbols('trial_p1:%d'%(dim+1))
        points         = symbols('points_1:%d'%(dim+1), cls=IndexedBase)
        weights        = symbols('weights_1:%d'%(dim+1), cls=IndexedBase)
        trial_basis    = symbols('trial_basis_1:%d'%(dim+1), cls=IndexedBase)
        test_basis     = symbols('test_basis_1:%d'%(dim+1), cls=IndexedBase)
        spans          = symbols('test_spans_1:%d'%(dim+1), cls=IndexedBase)
        quad_orders    = symbols('k1:%d'%(dim+1))
        # ...

        # ...
        self._basic_args = spaces
        # ...

        # ... getting data from fem space
        body = []

        body += [Assign(starts, DottedName(test_space, 'vector_space', 'starts'))]
        body += [Assign(ends, DottedName(test_space, 'vector_space', 'ends'))]
        body += [Assign(test_degrees, DottedName(test_space, 'vector_space', 'pads'))]
        body += [Assign(trial_degrees, DottedName(trial_space, 'vector_space', 'pads'))]

        body += [Assign(spans, DottedName(test_space, 'spans'))]
        body += [Assign(quad_orders, DottedName(test_space, 'quad_order'))]
        body += [Assign(points, DottedName(test_space, 'quad_points'))]
        body += [Assign(weights, DottedName(test_space, 'quad_weights'))]

        body += [Assign(test_basis, DottedName(test_space, 'quad_basis'))]
        body += [Assign(trial_basis, DottedName(trial_space, 'quad_basis'))]
        # ...

        # ...
#        body += [NewLine()]
#        body += [Comment('Create stencil matrices if not given')]
        body += [Import('StencilMatrix', 'spl.linalg.stencil')]
        for M in global_matrices:
            if_cond = Is(M, Nil())
            args = [DottedName(test_space, 'vector_space'),
                    DottedName(trial_space, 'vector_space')]
            if_body = [Assign(M, FunctionCall('StencilMatrix', args))]

            stmt = If((if_cond, if_body))
            body += [stmt]
        # ...

        # ... call to assembly
        mat_data       = [DottedName(M, '_data') for M in global_matrices]
        mat_data       = tuple(mat_data)

        field_data     = [DottedName(F, '_coeffs', '_data') for F in fields]
        field_data     = tuple(field_data)

        args = assembly.build_arguments(field_data + mat_data)

        body += [FunctionCall(assembly.func, args)]
        # ...

        # ... results
        if len(global_matrices) == 1:
            body += [Return(global_matrices[0])]

        else:
            body += [Return(global_matrices)]
        # ...

        # ... arguments
        mats = [Assign(M, Nil()) for M in global_matrices]
        mats = tuple(mats)

        func_args = self.build_arguments(fields + mats)
        # ...

        return FunctionDef(self.name, list(func_args), [], body)
