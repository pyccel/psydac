# coding: utf-8

import sys
import os
import importlib

from sympy import Mul, Tuple
from sympy import Mod, Abs, Range, Symbol
from sympy import Function

from pyccel.ast.core import Variable, IndexedVariable
from pyccel.ast.core import For
from pyccel.ast.core import Slice, String

from pyccel.ast.datatypes import NativeInteger
from pyccel.ast.core import Assign
from pyccel.ast.core import AugAssign
from pyccel.ast.core import Product
from pyccel.ast.core import FunctionDef
from pyccel.ast.core import FunctionCall
from pyccel.ast.core import Import
from pyccel.ast.utilities import build_types_decorator

from functools import lru_cache

from psydac.api.ast.utilities import variables, math_atoms_as_str
from psydac.fem.splines import SplineSpace
from psydac.fem.tensor  import TensorFemSpace
from psydac.fem.vector  import ProductFemSpace

from psydac.api.ast.basic import SplBasic
from psydac.api.printing import pycode

from psydac.api.settings        import PSYDAC_BACKENDS, PSYDAC_DEFAULT_FOLDER
from psydac.api.utilities       import mkdir_p, touch_init_file, random_string, write_code

from mpi4py import MPI

def variable_to_sympy(x):
    if isinstance(x, Variable) and isinstance(x.dtype, NativeInteger):
        x = Symbol(x.name, integer=True)
    return x

@lru_cache(maxsize=32)
class LinearOperatorDot(SplBasic):

    def __new__(cls, ndim, **kwargs):
        return SplBasic.__new__(cls, 'dot', name='lo_dot', prefix='lo_dot')

    def __init__(self, ndim, **kwargs):

        backend = dict(kwargs.pop('backend'))
        code             = self._initialize(ndim, backend=backend, **kwargs)
        self._arguments = dict((str(a.name),a) for a in code.arguments)
        self._code      = code
        self._folder    = self._initialize_folder()

        self._generate_code(backend=backend)
        self._compile(backend=backend)

    @property
    def func(self):
        return self._func

    @property
    def arguments(self):
        return self._arguments

    @property
    def code(self):
        return self._code

    @property
    def folder(self):
        return self._folder

    def _initialize(self, ndim, **kwargs):

        nrows           = kwargs.pop('nrows', variables('n1:%s'%(ndim+1),  'int'))
        nrows_extra     = kwargs.pop('nrows_extra', variables('ne1:%s'%(ndim+1),  'int'))
        pads            = kwargs.pop('pads', variables('p1:%s'%(ndim+1),  'int'))
        gpads           = kwargs.pop('gpads', variables('gp1:%s'%(ndim+1), 'int'))
        indices1        = variables('i1:%s'%(ndim+1),  'int')
        indices2        = variables('k1:%s'%(ndim+1),  'int')

        v               = variables('v','real')
        x, out          = variables('x, out','real',cls=IndexedVariable, rank=ndim)
        mat             = variables('mat','real',cls=IndexedVariable, rank=2*ndim)

        backend         = kwargs.pop('backend', None)

        body = []
        ranges = [Range(2*variable_to_sympy(p)+1) for p in pads]
        target = Product(*ranges)

        diff = [variable_to_sympy(gp)-variable_to_sympy(p) for gp,p in zip(gpads, pads)]

        v1 = x[tuple(i+j+d for i,j,d in zip(indices1,indices2, diff))]
        v2 = mat[tuple(i+j for i,j in zip(indices1,gpads))+ tuple(indices2)]
        v3 = out[tuple(i+j for i,j in zip(indices1,gpads))]

        body = [AugAssign(v,'+' ,Mul(v2, v1))]

        if ndim>1 and backend and backend['name'] == 'numba':
            for i,j in zip(indices2[::-1], target.args[::-1]):
                body = [For(i,j, body)]
        else:
            body = [For(indices2, target, body)]

        body.insert(0,Assign(v, 0.0))
        body.append(Assign(v3,v))
        ranges = [Range(variable_to_sympy(i)) for i in nrows]
        target = Product(*ranges)

        if ndim>1 and backend and backend['name'] == 'numba':
            for i,j in zip(indices1[::-1], target.args[::-1]):
                body = [For(i,j, body)]
        else:
            body = [For(indices1,target,body)]

        for dim in range(ndim):

            if nrows_extra[dim] == 0:continue

            v1 = [i+j+d for i,j,d in zip(indices1, indices2, diff)]
            v2 = [i+j for i,j in zip(indices1, gpads)]
            v1[dim] += nrows[dim]
            v2[dim] += nrows[dim]
            v3 = v2
            v1 = x[tuple(v1)]
            v2 = mat[tuple(v2)+ indices2]
            v3 = out[tuple(v3)]

            rows = list(nrows)
            rows[dim] = nrows_extra[dim]
            ranges = [2*variable_to_sympy(p)+1 for p in pads]
            ranges[dim] -= variable_to_sympy(indices1[dim]) + 1
            ranges =[Range(i) for i in ranges]
            target = Product(*ranges)

            for_body = [AugAssign(v, '+',Mul(v1,v2))]

            if ndim>1 and backend and backend['name'] == 'numba':
                for i,j in zip(indices2[::-1], target.args[::-1]):
                    for_body = [For(i,j, for_body)]
            else:
                for_body = [For(indices2, target, for_body)]

            for_body.insert(0,Assign(v, 0.0))
            for_body.append(Assign(v3,v))

            ranges = [Range(variable_to_sympy(i)) for i in rows]
            target = Product(*ranges)

            if ndim>1 and backend and backend['name'] == 'numba':
                for i,j in zip(indices1[::-1], target.args[::-1]):
                    for_body = [For(i,j, for_body)]
                body += for_body
            else:
                body  += [For(indices1, target, for_body)]

        func_args =  (mat, x, out)

        if isinstance(nrows[0], Variable):
            func_args = func_args + tuple(nrows)

        if isinstance(nrows_extra[0], Variable):
            func_args = func_args + tuple(nrows_extra)

        if isinstance(gpads[0], Variable):
            func_args = func_args + tuple(gpads)

        if isinstance(pads[0],  Variable):
            func_args = func_args + tuple(pads)

        decorators = {}
        header     = None

        if backend and backend['name'] == 'numba':
            imports = []
        else:
            imports = [Import('itertools', 'product')]

        if backend:
            if backend['name'] == 'pyccel':
                a = [String(str(i)) for i in build_types_decorator(func_args)]
                decorators = {'types': Function('types')(*a)}
            elif backend['name'] == 'numba':
                decorators = {'njit': Symbol('njit')}

            elif backend['name'] == 'pythran':
                header = build_pythran_types_header(name, func_args)

        func = FunctionDef(self.name, list(func_args), [], body, imports=imports, decorators=decorators)
        return func

    def _initialize_folder(self, folder=None):
        # ...
        if folder is None:
            basedir = os.getcwd()
            folder = PSYDAC_DEFAULT_FOLDER
            folder = os.path.join( basedir, folder )

            # ... add __init__ to all directories to be able to
            touch_init_file('__pycache__')
            for root, dirs, files in os.walk(folder):
                touch_init_file(root)
            # ...

        else:
            raise NotImplementedError('user output folder not yet available')

        folder = os.path.abspath( folder )
        mkdir_p(folder)
        # ...

        return folder

    def _generate_code(self, backend=None):
        code = ''
        tag = random_string( 8 )
        if backend and backend['name'] == 'pyccel':
            imports = 'from pyccel.decorators import types'
        elif backend and backend['name'] == 'numba':
            imports = 'from numba import njit'

        if MPI.COMM_WORLD.rank == 0:
            modname = 'dependencies_{}'.format(tag)
            code = '{imports}\n{code}'.format(imports=imports, code=pycode.pycode(self.code))
            write_code(modname+ '.py', code, folder = self.folder)
        else:
            modname = None

        self._modname =  MPI.COMM_WORLD.bcast( modname, root=0 )

    def _compile(self, backend=None):

        module_name = self._modname
        sys.path.append(self.folder)
        package = importlib.import_module( module_name )
        sys.path.remove(self.folder)

        if backend and backend['name'] == 'pyccel':
            package = self._compile_pyccel(package, backend)

        self._func = getattr(package, 'lo_dot')

    def _compile_pyccel(self, mod, backend, verbose=False):

        # ... convert python to fortran using pyccel
        compiler       = backend['compiler']
        fflags         = backend['flags']
        accelerator    = backend['accelerator']
        _PYCCEL_FOLDER = backend['folder']

        from pyccel.epyccel import epyccel

        fmod = epyccel(mod,
                       compiler    = compiler,
                       fflags      = fflags,
                       accelerator = accelerator,
                       comm        = MPI.COMM_WORLD,
                       bcast       = True,
                       folder      = _PYCCEL_FOLDER,
                       verbose     = verbose)

        return fmod

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

    def _initialize(self):

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

        self._imports = [Import('itertools', 'product')]

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
