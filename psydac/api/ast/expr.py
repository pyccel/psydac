#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from sympy import symbols, Symbol, IndexedBase
from sympy import Mul, Tuple, Range
from sympy import Matrix, ImmutableDenseMatrix
from sympy.core.numbers import ImaginaryUnit

from sympde.core                 import Constant
from sympde.topology.space       import ScalarFunction
from sympde.topology.space       import VectorFunction
from sympde.topology.space       import IndexedVectorFunction
from sympde.topology.derivatives import _partial_derivatives
from sympde.topology.derivatives import _logical_partial_derivatives
from sympde.topology.derivatives import get_atom_derivatives
from sympde.topology.derivatives import get_index_derivatives
from sympde.topology             import LogicalExpr
from sympde.topology             import SymbolicExpr
from sympde.calculus.matrices    import SymbolicDeterminant

from psydac.api.utilities   import random_string
from psydac.fem.vector      import MultipatchFemSpace

from psydac.pyccel.ast.core import IndexedVariable
from psydac.pyccel.ast.core import For
from psydac.pyccel.ast.core import Assign
from psydac.pyccel.ast.core import AugAssign
from psydac.pyccel.ast.core import Slice
from psydac.pyccel.ast.core import FunctionDef
from psydac.pyccel.ast.core import FunctionCall
from psydac.pyccel.ast.core import Import
from psydac.pyccel.ast.core import Nil
from psydac.pyccel.ast.core import Len
from psydac.pyccel.ast.core import If, Is, Return
from psydac.pyccel.ast.core import _atomic

from .basic      import SplBasic
from .utilities  import build_pythran_types_header, variables
from .utilities  import build_pyccel_type_annotations
from .utilities  import math_atoms_as_str
from .utilities  import get_max_partial_derivatives
from .nodes      import Zeros

#==============================================================================
def is_scalar_field(expr):

    if isinstance(expr, _partial_derivatives):
        return is_scalar_field(expr.args[0])

    elif isinstance(expr, _logical_partial_derivatives):
        return is_scalar_field(expr.args[0])

    elif isinstance(expr, ScalarFunction):
        return True

    return False

#==============================================================================
def is_vector_field(expr):

    if isinstance(expr, _partial_derivatives):
        return is_vector_field(expr.args[0])

    elif isinstance(expr, _logical_partial_derivatives):
        return is_vector_field(expr.args[0])

    elif isinstance(expr, (VectorFunction, IndexedVectorFunction)):
        return True

    return False

#==============================================================================
def compute_atoms_expr(atom, basis, indices, loc_indices, dim):

    cls = (_partial_derivatives,
           ScalarFunction,
           VectorFunction,
           IndexedVectorFunction)

    if not isinstance(atom, cls):
        raise TypeError('atom must be of type {}'.format(str(cls)))

    p_indices = get_index_derivatives(atom)
    orders = [0 for i in range(0, dim)]
    ind    = 0
    a = atom
    if isinstance(atom, _partial_derivatives):
        a = get_atom_derivatives(atom)
        orders[atom.grad_index] = p_indices[atom.coordinate]
        

    if isinstance(a, IndexedVectorFunction):
        ind = a.indices[0]
    args = []
    for i in range(dim):
        if isinstance(a, IndexedVectorFunction):
            args.append(basis[ind+i*dim][loc_indices[i],orders[i],indices[i]])
        elif isinstance(a, ScalarFunction):
            args.append(basis[i][loc_indices[i],orders[i],indices[i]])
        else:
            raise NotImplementedError('TODO')

    #
    return tuple(args), ind


class ExprKernel(SplBasic):

    def __new__(cls, expr, space, name=None, mapping=None, is_rational_mapping=None, backend=None):

        tag = random_string( 8 )
        obj = SplBasic.__new__(cls, tag, name=name,
                               prefix='kernel', mapping=mapping,
                               is_rational_mapping=is_rational_mapping)

        obj._expr = expr
        obj._space = space
        obj._user_functions = []
        obj._backend = backend

        obj._func = obj._initialize()

        return obj

    @property
    def expr(self):
        return self._expr

    @property
    def dim(self):
        return self._dim

    @property
    def space(self):
        return self._space

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
    def coordinates(self):
        return self._coordinates

    @property
    def fields(self):
        return self._fields
        
    @property
    def vector_fields(self):
        return self._vector_fields
        
    @property
    def fields_coeff(self):
        return self._fields_coeff
        
    @property
    def vector_fields_coeff(self):
        return self._vector_fields_coeff
        
    @property
    def constants(self):
        return self._constants

    @property
    def global_mats(self):
        return self._global_mats

    @property
    def global_mats_types(self):
        return self._global_mats_types

    @property
    def user_functions(self):
        return self._user_functions

    @property
    def backend(self):
        return self._backend

    def build_arguments(self, data):

        other = data

        if self.constants:
            other = other + self.constants

        return self.basic_args + other

    def _initialize(self):
        Vh   = self.space
        expr = self.expr
        dim  = Vh.ldim
        if isinstance(Vh, MultipatchFemSpace):
            size = len(Vh.spaces)
        else:
            size = 1
        
        self._dim = dim
        # ... discrete values
        

        n_elements = Vh.ncells
        degrees    = Vh.degree
        # TODO improve
        if isinstance(Vh, MultipatchFemSpace):
            degrees = degrees[0]
        # ...


        n_rows = 1 ; n_cols = 1
        if isinstance(expr, (Matrix, ImmutableDenseMatrix)):
            n_rows = expr.shape[0]
            n_cols = expr.shape[1]

        self._n_rows = n_rows
        self._n_cols = n_cols
        # ...

        # ...
        prelude = []
        body = []
        imports = []
        # ...

        # ...
        degrees     = variables('p1:%s(1:%s)'%(dim+1,size+1), 'int')
        n_elements  = variables('n1:%s'%(dim+1), 'int')
        xis         = variables('x1:%s'%(dim+1), 'real')
        arr_xis     = variables('arr_x1:%s'%(dim+1), dtype='real', rank=1, cls=IndexedVariable)
        indices     = variables('i1:%s'%(dim+1), 'int')
        loc_indices = variables('j1:%s'%(dim+1), 'int')
        
        lengths     = variables('k1:%s'%(dim+1), 'int')

        ranges      = [Range(lengths[i]) for i in range(dim)]
        # ...

        d_vals = {}
        for i in range(0, n_rows):
            for j in range(0, n_cols):
                is_complex = False
                mat = IndexedBase('val_{i}{j}'.format(i=i,j=j))
                d_vals[i, j] = mat
        # ...

        xs = [Symbol(x) for x in ['x', 'y', 'z'][:dim]]
        for xi, x in zip(xis, xs):
            expr = expr.subs(x, xi)
        # ...

        # ...
        atoms_types = (_partial_derivatives,
                       _logical_partial_derivatives,
                       ScalarFunction,
                       VectorFunction, IndexedVectorFunction,
                       SymbolicDeterminant,
                       Symbol)

        atoms  = _atomic(expr, cls=atoms_types)
        self._constants = _atomic(expr, cls=Constant)
        self._coordinates = tuple(xis)
        # ...
        
        atomic_expr_field        = [atom for atom in atoms if is_scalar_field(atom)]
        atomic_expr_vector_field = [atom for atom in atoms if is_vector_field(atom)]

        self._fields = tuple(expr.atoms(ScalarFunction))
        self._vector_fields = tuple(expr.atoms(VectorFunction))
        # ...
        fields_str        = tuple(SymbolicExpr(f).name for f in atomic_expr_field)
        vector_fields_str = tuple(SymbolicExpr(f).name for f in atomic_expr_vector_field)
        
        fields = symbols(fields_str)
        vector_fields = symbols(vector_fields_str)
        
        if fields:
            fields_coeff = variables(['F_coeff',],
                    dtype='real', rank=dim, cls=IndexedVariable)
        else:
            field_coeff = ()

        if vector_fields:                              
            vector_fields_coeff = variables(['F_{}_coeff'.format(str(i)) for i in range(size)],
                    dtype='real', rank=dim, cls=IndexedVariable)
        else:
            vector_fields_coeff = ()
                                          
        self._fields_coeff        = fields_coeff
        self._vector_fields_coeff = vector_fields_coeff 
                                          
        if fields or vector_fields_str:
            basis  = variables( 'basis1:%s(1:%s)'%(dim+1,size+1),
                                dtype = 'real',
                                rank = 3,
                                cls = IndexedVariable )

            spans  = variables( 'spans1:%s(1:%s)'%(dim+1,size+1),
                                dtype = 'int',
                                rank = 1,
                                cls = IndexedVariable )

        # ... TODO add it as a method to basic class
        nderiv = 0

        if isinstance(expr, Matrix):
            n_rows, n_cols = expr.shape
            for i_row in range(0, n_rows):
                for i_col in range(0, n_cols):
                    d_atoms  = _atomic(expr[i_row,i_col], cls=atoms_types)
                    Fs = [get_atom_derivatives(a) for a in d_atoms]
                    d = get_max_partial_derivatives(expr[i_row,i_col], logical=False, F=Fs)
                    nderiv = max(nderiv, max(d.values()))
        else:
            Fs = [get_atom_derivatives(a) for a in atoms]
            d = get_max_partial_derivatives(expr, logical=False, F=Fs)
            nderiv = max(nderiv, max(d.values()))

        self._max_nderiv = nderiv



        # ...
        for l,arr_xi in zip(lengths, arr_xis):
            prelude += [Assign(l, Len(arr_xi))]
        # ...

        # ...
        slices = [Slice(None,None)]*dim
        for i_row in range(0, n_rows):
            for i_col in range(0, n_cols):
                symbol = d_vals[i_row,i_col]
                prelude += [Assign(symbol[slices], 0.)]
        # ...

        # ... fields
        for i in range(len(fields)):
            body.append(Assign(fields[i],0))
            atom      = atomic_expr_field[i]
            atoms,ind = compute_atoms_expr(atom, basis, indices, loc_indices, dim)
            slices = tuple(sp[id]-p+j for sp,p,j,id in zip(spans, degrees, loc_indices, indices))
            args   = args + (fields_coeff[0][slices],)
            for_body = [AugAssign(fields[i],'+',Mul(*args))]
            loc_ranges = [Range(j) for j in degrees]
            for j in range(dim):
                for_body = [For(loc_indices[dim-1-j], loc_ranges[dim-1-j],for_body)]
            
            body += for_body
            
        for i in range(len(vector_fields)):
            body.append(Assign(vector_fields[i],0))
            atom = atomic_expr_vector_field[i]
            atoms, ind = compute_atoms_expr(atom, basis, indices, loc_indices, dim)
            slices = tuple(sp[id]-p+j for sp,p,j,id in zip(spans[ind::size], degrees[ind::size], loc_indices,indices))
            atoms   = atoms + (vector_fields_coeff[ind][slices],)
            for_body = [AugAssign(vector_fields[i],'+',Mul(*atoms))]
            loc_ranges = [Range(j) for j in degrees[ind::size]]
            for j in range(dim):
                for_body = [For(loc_indices[dim-1-j], loc_ranges[dim-1-j],for_body)]
            
            body += for_body
        
        # ...

        # ...
        for i_row in range(0, n_rows):
            for i_col in range(0, n_cols):
                val = d_vals[i_row,i_col]
                val = val[indices]

                if isinstance(expr, (Matrix, ImmutableDenseMatrix)):
                    rhs   = SymbolicExpr(expr[i_row, i_col])
                    body += [Assign(val, rhs)]

                else:
                    rhs   = SymbolicExpr(expr)
                    body += [Assign(val, rhs)]

        for i in range(dim-1, -1, -1):
            x = indices[i]
            rx = ranges[i]

            xi = xis[i]
            arr_xi = arr_xis[i]
            body = [Assign(xi, arr_xi[x])] + body

            body = [For(x, rx, body)]

        # ...
        body = prelude + body
        # ...

        # ... get math functions and constants
        math_elements = math_atoms_as_str(expr, 'math')
        math_imports  = [Import('math', e) for e in math_elements]

        imports += math_imports
        # ...

        # ...
        self._basic_args = arr_xis + fields_coeff + vector_fields_coeff + degrees + basis + spans
        # ...

        # ...
        mats = []
        for i in range(0, n_rows):
            for j in range(0, n_cols):
                mats.append(d_vals[i, j])
        mats = tuple(mats)
        self._global_mats = mats
        # ...
        mats_types = []
        if isinstance(expr, (Matrix, ImmutableDenseMatrix)):
            for i in range(0, n_rows):
                for j in range(0, n_cols):
                    dtype = 'float'
                    if expr[i, j].atoms(ImaginaryUnit):
                        dtype = 'complex'
                    mats_types.append(dtype)

        else:
            dtype = 'float'
            if expr.atoms(ImaginaryUnit):
                dtype = 'complex'
            mats_types.append(dtype)

        mats_types = tuple(mats_types)
        self._global_mats_types = mats_types

        self._imports = imports

        # function args

        func_args = self.build_arguments(mats)
        decorators = {}
        header = None

        if self.backend['name'] == 'pyccel':
            func_args = build_pyccel_type_annotations(func_args)
        elif self.backend['name'] == 'pythran':
            header = build_pythran_types_header(self.name, func_args)

        return FunctionDef(self.name, list(func_args), [], body,
                           decorators=decorators, header=header)


class ExprInterface(SplBasic):

    def __new__(cls, kernel, name=None, mapping=None, is_rational_mapping=None, backend=None):

        if not isinstance(kernel, ExprKernel):
            raise TypeError('> Expecting an ExprKernel')

        obj = SplBasic.__new__(cls, kernel.tag, name=name,
                               prefix='interface', mapping=mapping,
                               is_rational_mapping=is_rational_mapping)

        obj._kernel = kernel
        obj._backend = backend

        # update dependencies
        obj._dependencies += [kernel]

        obj._func = obj._initialize()
        return obj

    @property
    def kernel(self):
        return self._kernel

    @property
    def backend(self):
        return self._backend

    @property
    def max_nderiv(self):
        return self.kernel.max_nderiv

    @property
    def n_rows(self):
        return self.kernel.n_rows

    @property
    def n_cols(self):
        return self.kernel.n_cols

    @property
    def global_mats_types(self):
        return self.kernel.global_mats_types

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
        return self.kernel.user_functions

    def _initialize(self):

        kernel = self.kernel
        global_mats = kernel.global_mats
        global_mats_types = kernel.global_mats_types
        fields = kernel.fields
        vector_fields = kernel.vector_fields
        dim = kernel.dim


        # ... declarations
        space = Symbol('W')

        arr_xis    = symbols('arr_x1:%d'%(dim+1), cls=IndexedBase)
        lengths    = symbols('k1:%d'%(dim+1))
        # ...

        self._basic_args = (space, ) + kernel.basic_args
        # ...
        imports = []
        prelude = []
        body = []

        # ...
        imports += [Import('numpy',('zeros',))]
        # ...

        # ...
        for l,arr_xi in zip(lengths, arr_xis):
            prelude += [Assign(l, Len(arr_xi))]
        # ...

        # ...
        if dim > 1:
            lengths = Tuple(*lengths)
            lengths = [lengths]

        for M,dtype in zip(global_mats, global_mats_types):
            if_cond = Is(M, Nil())

            _args = list(lengths) + ['{}'.format(dtype)]
            if_body = [Assign(M, Zeros(*_args))]

            stmt = If((if_cond, if_body))
            body += [Import('numpy',('zeros',)), stmt]
        # ...

        # ...
        body = prelude + body
        # ...

        # ...
        self._inout_arguments = list(global_mats)
        self._in_arguments = list(self.kernel.coordinates) + list(self.kernel.constants) + list(fields) + list(vector_fields)
        # ...

        # ... call to kernel
        # TODO add fields
        mat_data       = tuple(global_mats)

        args =  mat_data
        args = kernel.build_arguments(args)

        body += [FunctionCall(kernel.func, args)]
        # ...

        # ... results
        if len(global_mats) == 1:
            M = global_mats[0]
            body += [Return(M)]

        else:
            body += [Return(global_mats)]
        # ...

        # ... arguments
        mats = [Assign(M, Nil()) for M in global_mats]
        mats = tuple(mats)


        # TODO improve using in_arguments
        if self.kernel.constants:
            constants = self.kernel.constants
            args =  constants  + mats
        else:
            args =   mats

        func_args = self.build_arguments(args)
        # ...

        self._imports = imports
        return FunctionDef(self.name, list(func_args), [], body)
