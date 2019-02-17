
from collections import OrderedDict
from itertools import groupby
import string
import random
import numpy as np

from sympy import Basic
from sympy import symbols, Symbol, IndexedBase, Indexed, Function
from sympy import Mul, Add, Tuple
from sympy import Matrix, ImmutableDenseMatrix
from sympy import sqrt as sympy_sqrt
from sympy import S as sympy_S
from sympy import simplify, expand
from sympy.core.numbers import ImaginaryUnit

from pyccel.ast.core import Variable, IndexedVariable
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

from sympde.core import Constant
from sympde.topology import ScalarField, VectorField
from sympde.topology import ScalarTestFunction, VectorTestFunction
from sympde.expr import BilinearForm
from sympde.core.math import math_atoms_as_str

#from gelato.core import gelatize

from gelato.glt import BasicGlt

from .basic import SplBasic
from .utilities import random_string
from .utilities import build_pythran_types_header, variables


class GltKernel(SplBasic):

    def __new__(cls, expr, spaces, name=None, mapping=None, is_rational_mapping=None, backend=None):

        tag = random_string( 8 )
        obj = SplBasic.__new__(cls, tag, name=name,
                               prefix='kernel', mapping=mapping,
                               is_rational_mapping=is_rational_mapping)

        obj._expr = expr
        obj._spaces = spaces
        obj._eval_fields = None
        obj._eval_mapping = None
        obj._user_functions = []
        obj._backend = backend

        obj._func = obj._initialize()

        return obj

    @property
    def expr(self):
        return self._expr

    @property
    def form(self):
        return self.expr.form

    @property
    def spaces(self):
        return self._spaces

    @property
    def n_rows(self):
        return self._n_rows

    @property
    def n_cols(self):
        return self._n_cols

    # needed for MPI comm => TODO improve BasicCodeGen
    @property
    def max_nderiv(self):
        return None

    @property
    def with_coordinates(self):
        return self._with_coordinates

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def constants(self):
        return self.expr.constants

    @property
    def fields(self):
        return self.expr.fields

    @property
    def fields_coeffs(self):
        return self._fields_coeffs

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
    def eval_mapping(self):
        return self._eval_mapping

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

        if self.mapping_values:
            other = self.mapping_values + other

        if self.constants:
            other = other + self.constants

        return self.basic_args + other

    def _initialize(self):
        weak_form = self.form
        dim       = self.expr.ldim

        # ... discrete values
        Vh, Wh = self.spaces

        n_elements = Vh.ncells
        degrees    = Vh.degree
        # ...

        # ...
        kernel_expr = self.expr.expr

        #*************************************
        # TODO must be moved to __call__ of GltExpr
        #*************************************
        # ... TODO must be moved to __call__ of GltExpr
        ns = ['nx', 'ny', 'nz'][:dim]
        ns = [Symbol(i, integer=True) for i in ns]

        if isinstance(n_elements, int):
            n_elements = [n_elements]*dim

        if not( len(ns) == len(n_elements) ):
            raise ValueError('Wrong size for n_elements')

        for n,v in zip(ns, n_elements):
            kernel_expr = kernel_expr.subs(n, v)
        # ...

        # ...
        ps = ['px', 'py', 'pz'][:dim]
        ps = [Symbol(i, integer=True) for i in ps]

        if isinstance(degrees, int):
            degrees = [degrees]*dim

        if not( len(ps) == len(degrees) ):
            raise ValueError('Wrong size for degrees')

        d_degrees = {}
        for p,v in zip(ps, degrees):
            d_degrees[p] = v

        atoms = list(self.expr.expr.atoms(BasicGlt))
        for a in atoms:
            func = a.func
            p,t = a._args[:]
            newargs = (d_degrees[p], t)
            newa = func(*newargs)
            kernel_expr = kernel_expr.subs(a, newa)
        # ...
        #*************************************

#        kernel_expr = expand(kernel_expr)
#        kernel_expr = simplify(kernel_expr)
#        kernel_expr = kernel_expr.evalf()
        # ...

        # ...
        n_rows = 1 ; n_cols = 1
        if isinstance(kernel_expr, (Matrix, ImmutableDenseMatrix)):
            n_rows = kernel_expr.shape[0]
            n_cols = kernel_expr.shape[1]

        self._n_rows = n_rows
        self._n_cols = n_cols
        # ...


        # ...
        degrees    = variables('p1:%d'%(dim+1), 'int')
        n_elements = variables('n1:%d'%(dim+1), 'int')
        tis        = variables('t1:%d'%(dim+1), 'real')
        arr_tis    = variables('arr_t1:%d'%(dim+1), dtype='real', rank=1, cls=IndexedVariable)
        xis        = variables('x1:%d'%(dim+1), 'real')
        arr_xis    = variables('arr_x1:%d'%(dim+1), dtype='real', rank=1, cls=IndexedVariable)
        indices    = variables('i1:%d'%(dim+1), 'int')
        lengths    = variables('nt1:%d'%(dim+1), 'int')

        ranges     = [Range(lengths[i]) for i in range(dim)]
        # ...

        # ...
        self._coordinates = tuple(self.expr.space_variables)
        self._with_coordinates = (len(self._coordinates) > 0)
        # ...

        # ...
        d_symbols = {}
        for i in range(0, n_rows):
            for j in range(0, n_cols):
                is_complex = False
                mat = IndexedBase('symbol_{i}{j}'.format(i=i,j=j))
                d_symbols[i,j] = mat
        # ...

        # ... replace tx/ty/tz by t1/t2/t3
        txs = [Symbol(tx) for tx in ['tx', 'ty', 'tz'][:dim]]
        for ti, tx in zip(tis, txs):
            kernel_expr = kernel_expr.subs(tx, ti)

        xs = [Symbol(x) for x in ['x', 'y', 'z'][:dim]]
        for xi, x in zip(xis, xs):
            kernel_expr = kernel_expr.subs(x, xi)
        # ...

        # ...
        prelude = []
        for l,arr_ti in zip(lengths, arr_tis):
            prelude += [Assign(l, Len(arr_ti))]
        # ...

        # ...
        slices = [Slice(None,None)]*dim
        for i_row in range(0, n_rows):
            for i_col in range(0, n_cols):
                symbol = d_symbols[i_row,i_col]
                prelude += [Assign(symbol[slices], 0.)]
        # ...

        # ...
        body = []

        for i_row in range(0, n_rows):
            for i_col in range(0, n_cols):
                symbol = d_symbols[i_row,i_col]
                symbol = symbol[indices]

                if isinstance(kernel_expr, (Matrix, ImmutableDenseMatrix)):
                    body += [Assign(symbol, kernel_expr[i_row,i_col])]

                else:
                    body += [Assign(symbol, kernel_expr)]

        for i in range(dim-1,-1,-1):
            x = indices[i]
            rx = ranges[i]

            ti = tis[i]
            arr_ti = arr_tis[i]
            body = [Assign(ti, arr_ti[x])] + body
            if self.with_coordinates:
                xi = xis[i]
                arr_xi = arr_xis[i]
                body = [Assign(xi, arr_xi[x])] + body

            body = [For(x, rx, body)]
        # ...

        # ...
        body = prelude + body
        # ...

        # get math functions and constants
        math_elements = math_atoms_as_str(kernel_expr)
        math_imports = []
        for e in math_elements:
            math_imports += [Import(e, 'numpy')]
        body = math_imports + body

        # ...
        self._basic_args = [*arr_tis]
        self._basic_args = tuple(self._basic_args)
        # ...

        # ...
        mats = []
        for i in range(0, n_rows):
            for j in range(0, n_cols):
                mats.append(d_symbols[i,j])
        mats = tuple(mats)
        self._global_mats = mats
        # ...

        # ...
        mats_types = []
        if isinstance(kernel_expr, (Matrix, ImmutableDenseMatrix)):
            for i in range(0, n_rows):
                for j in range(0, n_cols):
                    dtype = 'float'
                    if kernel_expr[i,j].atoms(ImaginaryUnit):
                        dtype = 'complex'
                    mats_types.append(dtype)

        else:
            dtype = 'float'
            if kernel_expr.atoms(ImaginaryUnit):
                dtype = 'complex'
            mats_types.append(dtype)

        mats_types = tuple(mats_types)
        self._global_mats_types = mats_types
        # ...

        # function args
        func_args = self.build_arguments(self.coordinates + mats)

#        return FunctionDef(self.name, list(func_args), [], body)

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


class GltInterface(SplBasic):

    def __new__(cls, kernel, name=None, mapping=None, is_rational_mapping=None, backend=None):

        if not isinstance(kernel, GltKernel):
            raise TypeError('> Expecting an GltKernel')

        obj = SplBasic.__new__(cls, kernel.tag, name=name,
                               prefix='interface', mapping=mapping,
                               is_rational_mapping=is_rational_mapping)

        obj._kernel = kernel
        obj._mapping = mapping
        obj._backend = backend

        # update dependencies
        obj._dependencies += [kernel]

        obj._func = obj._initialize()
        return obj

    @property
    def kernel(self):
        return self._kernel

    @property
    def mapping(self):
        return self._mapping

    @property
    def backend(self):
        return self._backend

    # needed for MPI comm => TODO improve BasicCodeGen
    @property
    def max_nderiv(self):
        return None

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
    def coordinates(self):
        return self.kernel.coordinates

    @property
    def user_functions(self):
        return self.kernel.user_functions

    def _initialize(self):
        form = self.kernel.form
        kernel = self.kernel
        global_mats = kernel.global_mats
        global_mats_types = kernel.global_mats_types
        fields = form.fields
        fields = sorted(fields, key=lambda x: str(x.name))
        fields = tuple(fields)

        dim = form.ldim

        # ... declarations
        test_space = Symbol('W')
        trial_space = Symbol('V')

        arr_tis    = symbols('arr_t1:%d'%(dim+1), cls=IndexedBase)
        lengths    = symbols('nt1:%d'%(dim+1))

        mapping = ()
        if self.mapping:
            mapping = Symbol('mapping')
        # ...

        # ...
        self._basic_args = kernel.basic_args
        # ...

        # ...
        body = []

        # ...
        if mapping:
            for i, coeff in enumerate(kernel.kernel.mapping_coeffs):
                component = IndexedBase(DottedName(mapping, '_fields'))[i]
                body += [Assign(coeff, DottedName(component, '_coeffs', '_data'))]
        # ...

        # ...
        prelude = [Import('zeros', 'numpy')]
        for l,arr_ti in zip(lengths, arr_tis):
            prelude += [Assign(l, Len(arr_ti))]
        # ...

        # ...
        if dim > 1:
            lengths = Tuple(*lengths)
            lengths = [lengths]

        body = []
        for M,dtype in zip(global_mats, global_mats_types):
            if_cond = Is(M, Nil())

            _args = list(lengths) + ['dtype={}'.format(dtype)]
            if_body = [Assign(M, FunctionCall('zeros', _args))]

            stmt = If((if_cond, if_body))
            body += [stmt]
        # ...

        # ...
        body = prelude + body
        # ...

        # ...
        self._inout_arguments = list(global_mats)
        self._in_arguments = list(self.coordinates) + list(self.kernel.constants) + list(fields)
        # ...

        # ... call to kernel
        # TODO add fields
        mat_data       = tuple(global_mats)

        field_data     = [DottedName(F, '_coeffs', '_data') for F in fields]
        field_data     = tuple(field_data)

        args = kernel.build_arguments(self.coordinates + field_data + mat_data)

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

        if mapping:
            mapping = (mapping,)

        # TODO improve using in_arguments
        if self.kernel.constants:
            constants = self.kernel.constants

            if self.coordinates:
                coordinates = self.coordinates
                args = mapping + constants + coordinates + fields + mats

            else:
                args = mapping + constants + fields + mats

        else:
            if self.coordinates:
                coordinates = self.coordinates
                args = mapping + coordinates + fields + mats

            else:
                args = mapping + fields + mats

        func_args = self.build_arguments(args)
        # ...

        return FunctionDef(self.name, list(func_args), [], body)
