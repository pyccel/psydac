# coding: utf-8

import sys
import os
import importlib
import inspect

import numpy as np

from functools import lru_cache
from mpi4py    import MPI

from sympy import Mul, Tuple
from sympy import Mod as sy_Mod, Abs, Range, Symbol, Max
from sympy import Function, Integer

from psydac.pyccel.ast.core import Variable, IndexedVariable
from psydac.pyccel.ast.core import For
from psydac.pyccel.ast.core import Slice, String
from psydac.pyccel.ast.datatypes import NativeInteger
from psydac.pyccel.ast.core import ValuedArgument
from psydac.pyccel.ast.core import Assign
from psydac.pyccel.ast.core import AugAssign
from psydac.pyccel.ast.core import Product
from psydac.pyccel.ast.core import FunctionDef
from psydac.pyccel.ast.core import FunctionCall
from psydac.pyccel.ast.core import Import

from psydac.api.ast.nodes     import FloorDiv
from psydac.api.ast.utilities import variables, math_atoms_as_str
from psydac.api.ast.utilities import build_pyccel_types_decorator
from psydac.fem.splines import SplineSpace
from psydac.fem.tensor  import TensorFemSpace
from psydac.fem.vector  import ProductFemSpace
from psydac.api.ast.basic import SplBasic
from psydac.api.printing import pycode
from psydac.api.settings        import PSYDAC_BACKENDS, PSYDAC_DEFAULT_FOLDER
from psydac.api.utilities       import mkdir_p, touch_init_file, random_string, write_code

#==============================================================================
def variable_to_sympy(x):
    if isinstance(x, Variable) and isinstance(x.dtype, NativeInteger):
        x = Symbol(x.name, integer=True)
    return x

#==============================================================================
def compute_diag_len(p, md, mc, return_padding=False):
    n = ((np.ceil((p+1)/mc)-1)*md).astype('int')
    ep = np.minimum(0, n-p)
    n = n-ep + p+1
    if return_padding:
        return n.astype('int'), (-ep).astype('int')
    else:
        return n.astype('int')

def Mod(a,b):
    if b == 1:
        return Integer(0)
    else:
        return sy_Mod(a,b)

@lru_cache(maxsize=32)
class LinearOperatorDot(SplBasic):

    def __new__(cls, ndim, **kwargs):
        return SplBasic.__new__(cls, 'dot', name='lo_dot', prefix='lo_dot')

    def __init__(self, ndim, **kwargs):

        backend = dict(kwargs.pop('backend'))
        code            = self._initialize(ndim, backend=backend, **kwargs)
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
        starts          = kwargs.pop('starts', variables('s1:%s'%(ndim+1),  'int'))
        indices1        = variables('i1:%s'%(ndim+1),  'int')
        bb              = variables('b1:%s'%(ndim+1),  'int')
        indices2        = variables('k1:%s'%(ndim+1),  'int')

        v               = variables('v','real')
        x, out          = variables('x, out','real',cls=IndexedVariable, rank=ndim)
        mat             = variables('mat','real',cls=IndexedVariable, rank=2*ndim)

        backend         = kwargs.pop('backend', None)

        pads            = kwargs.pop('pads')
        gpads           = kwargs.pop('gpads')
        cm              = kwargs.pop('cm')
        dm              = kwargs.pop('dm')

        ndiags, _ = list(zip(*[compute_diag_len(p,mj,mi, return_padding=True) for p,mi,mj in zip(pads,cm,dm)]))

        inits = [Assign(b,p*m+p+1-n-Mod(s,m)) for b,p,m,n,s in zip(bb, gpads, dm, ndiags, starts) if not isinstance(p*m+p+1-n-Mod(s,m),(int,np.int64, Integer))]
        bb    = [b if not isinstance(p*m+p+1-n-Mod(s,m),(int,np.int64, Integer)) else p*m+p+1-n-Mod(s,m) for b,p,m,n,s in zip(bb, gpads, dm, ndiags, starts)]
        body  = []
        ranges = [Range(variable_to_sympy(n)) for n in ndiags]
        target = Product(*ranges)

        diff = [variable_to_sympy(gp-p) for gp,p in zip(gpads, pads)]

        v1 = x[tuple(b-d+FloorDiv((i1+Mod(s,mj)),mi)*mj + i2 for i1,mi,mj,b,s,d,i2 in zip(indices1,cm,dm,bb,starts,diff,indices2))]
        v2 = mat[tuple(i+m*j for i,j,m in zip(indices1,gpads,cm))+ tuple(indices2)]
        v3 = out[tuple(i+m*j for i,j,m in zip(indices1,gpads,cm))]

        body = [AugAssign(v,'+' ,Mul(v2, v1))]

        # Decompose fused loop over Cartesian product of multiple ranges
        # into nested loops, each over a single range
        if ndim > 1:
            for i,j in zip(indices2[::-1], target.args[::-1]):
                body = [For(i,j, body)]
        else:
            body = [For(indices2, target, body)]

        body.insert(0,Assign(v, 0.0))
        body.append(Assign(v3,v))
        ranges = [Range(variable_to_sympy(i)) for i in nrows]
        target = Product(*ranges)

        # Decompose fused loop over Cartesian product of multiple ranges
        # into nested loops, each over a single range
        if ndim > 1:
            for i,j in zip(indices1[::-1], target.args[::-1]):
                body = [For(i,j, body)]
        else:
            body = [For(indices1, target, body)]

        nrowscopy = list(nrows).copy()
        nrows     = list(nrows)
        for dim in range(ndim):

            if nrows_extra[dim] == 0:continue

            v1 = [b-d+FloorDiv((i1+(nrows[dim] if dim==x else 0)+Mod(s,mj)),mi)*mj + i2 for x,i1,mi,mj,b,s,d,i2 in zip(range(ndim), indices1,cm,dm,bb,starts,diff,indices2)]
            v2 = [i+m*j for i,j,m in zip(indices1,gpads,cm)]

            v2[dim] += nrows[dim]

            v3 = v2
            v1 = x[tuple(v1)]
            v2 = mat[tuple(v2)+ indices2]
            v3 = out[tuple(v3)]

            rows = list(nrows)
            rows[dim] = nrows_extra[dim]

            
            ranges       = [variable_to_sympy(n) for n in ndiags]
            ranges[dim] -= variable_to_sympy(indices1[dim]) + 1
            ranges       = [ind if i>=dim else ind - Max(0, variable_to_sympy(d1)+1-variable_to_sympy(r)) for i,(ind,d1,r) in enumerate(zip(ranges, indices1, nrowscopy)) ]
            ranges       = [Range(i) for i in ranges]

            target = Product(*ranges)

            for_body = [AugAssign(v, '+',Mul(v1,v2))]

            # Decompose fused loop over Cartesian product of multiple ranges
            # into nested loops, each over a single range
            if ndim > 1:
                for i,j in zip(indices2[::-1], target.args[::-1]):
                    for_body = [For(i,j, for_body)]
            else:
                for_body = [For(indices2, target, for_body)]

            for_body.insert(0,Assign(v, 0.0))
            for_body.append(Assign(v3,v))

            ranges = [Range(variable_to_sympy(i)) for i in rows]
            target = Product(*ranges)

            # Decompose fused loop over Cartesian product of multiple ranges
            # into nested loops, each over a single range
            if ndim > 1:
                for i,j in zip(indices1[::-1], target.args[::-1]):
                    for_body = [For(i,j, for_body)]
                body += for_body
            else:
                body += [For(indices1, target, for_body)]


            nrows[dim] += nrows_extra[dim]

        body      = inits + body
        func_args =  (mat, x, out)

        if isinstance(starts[0], Variable):
            func_args = func_args + tuple(starts)

        if isinstance(nrowscopy[0], Variable):
            func_args = func_args + tuple(nrowscopy)

        if isinstance(nrows_extra[0], Variable):
            func_args = func_args + tuple(nrows_extra)

        decorators = {}
        header     = None
        imports    = []

        if backend:
            if backend['name'] == 'pyccel':
                a = [String(str(i)) for i in build_pyccel_types_decorator(func_args)]
                decorators = {'types': Function('types')(*a)}
            elif backend['name'] == 'numba':
                decorators = {'njit': Function('njit')(ValuedArgument(Symbol('fastmath'), backend['fastmath']))}
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
        else:
            imports = ''

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
        _PYCCEL_FOLDER = backend['folder']

        from pyccel.epyccel import epyccel

        fmod = epyccel(mod,
                       compiler    = compiler,
                       fflags      = fflags,
                       comm        = MPI.COMM_WORLD,
                       bcast       = True,
                       folder      = _PYCCEL_FOLDER,
                       verbose     = verbose)

        return fmod

@lru_cache(maxsize=32)
class TransposeOperator(SplBasic):

    def __new__(cls, ndim, **kwargs):
        return SplBasic.__new__(cls, 'transpose', name='lo_transpose', prefix='lo_transpose')

    def __init__(self, ndim, **kwargs):

        self.ndim        = ndim
        backend          = dict(kwargs.pop('backend'))
        self._code       = inspect.getsource(_transpose[ndim])
        self._args_dtype = _args_dtype[ndim]
        self._folder     = self._initialize_folder()

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
        dec  = ''
        code = self._code
        tag = random_string( 8 )
        if backend and backend['name'] == 'pyccel':
            imports = 'from pyccel.decorators import types'
            # Add @types decorator due the  minimum required Pyccel version 0.10.1
            dec     = '@types({})'.format(','.join(self._args_dtype))
        elif backend and backend['name'] == 'numba':
            imports = 'from numba import njit'
            dec     = '@njit(fastmath={})'.format(backend['fastmath'])
        else:
            imports = ''

        if MPI.COMM_WORLD.rank == 0:
            modname = 'dependencies_{}'.format(tag)
            code = '{imports}\n{dec}\n{code}'.format(imports=imports, dec=dec, code=code)
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

        self._func = getattr(package, 'transpose_{}d'.format(self.ndim))

    def _compile_pyccel(self, mod, backend, verbose=False):

        # ... convert python to fortran using pyccel
        compiler       = backend['compiler']
        fflags         = backend['flags']
        _PYCCEL_FOLDER = backend['folder']

        from pyccel.epyccel import epyccel

        fmod = epyccel(mod,
                       compiler    = compiler,
                       fflags      = fflags,
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
            decorators = {'types': build_pyccel_types_decorator(func_args), 'external':[]}
        elif self.backend['name'] == 'numba':
            decorators = {'jit':[]}
        elif self.backend['name'] == 'pythran':
            header = build_pythran_types_header(self.name, func_args)

        return FunctionDef(self.name, list(func_args), [], body,
                           decorators=decorators,header=header)

#========================================================================================================
def transpose_1d( M:'float[:,:]', Mt:'float[:,:]', n1:int, nc1:int, gp1:int, p1:int,
                  dm1:int, cm1:int, nd1:int, ndT1:int, si1:int, sk1:int, sl1:int):

    d1 = gp1-p1

    for x1 in range(n1):
        j1 = dm1*gp1 + x1
        for l1 in range(nd1):

            i1 = si1 + cm1*(x1//dm1) + l1 + d1
            k1 = sk1 + x1%dm1-dm1*(l1//cm1)

            if k1<ndT1 and k1>-1 and l1<nd1 and i1<nc1:
                Mt[j1, l1+sl1] = M[i1, k1]

# ...
def transpose_2d( M:'float[:,:,:,:]', Mt:'float[:,:,:,:]', n1:int, n2:int, nc1:int, nc2:int,
                   gp1:int, gp2:int, p1:int, p2:int, dm1:int, dm2:int,
                   cm1:int, cm2:int, nd1:int, nd2:int, ndT1:int, ndT2:int,
                   si1:int, si2:int, sk1:int, sk2:int, sl1:int, sl2:int):

    d1 = gp1-p1
    d2 = gp2-p2

    for x1 in range(n1):
        for x2 in range(n2):

            j1 = dm1*gp1 + x1
            j2 = dm2*gp2 + x2
     
            for l1 in range(nd1):
                for l2 in range(nd2):

                    i1 = si1 + cm1*(x1//dm1) + l1 + d1
                    i2 = si2 + cm2*(x2//dm2) + l2 + d2

                    k1 = sk1 + x1%dm1-dm1*(l1//cm1)
                    k2 = sk2 + x2%dm2-dm2*(l2//cm2)

                    if k1<ndT1 and k1>-1 and k2<ndT2 and k2>-1\
                        and l1<nd1 and l2<nd2 and i1<nc1 and i2<nc2:
                        Mt[j1,j2, l1+sl1,l2+sl2] = M[i1,i2, k1,k2]

# ...
def transpose_3d( M:'float[:,:,:,:,:,:]', Mt:'float[:,:,:,:,:,:]', n1:int, n2:int, n3:int,
                  nc1:int, nc2:int, nc3:int, gp1:int, gp2:int, gp3:int, p1:int, p2:int, p3:int,
                  dm1:int, dm2:int, dm3:int, cm1:int, cm2:int, cm3:int, nd1:int, nd2:int, nd3:int,
                  ndT1:int, ndT2:int, ndT3:int, si1:int, si2:int, si3:int, sk1:int, sk2:int, sk3:int,
                  sl1:int, sl2:int, sl3:int):

    d1 = gp1-p1
    d2 = gp2-p2
    d3 = gp3-p3

    for x1 in range(n1):
        for x2 in range(n2):
            for x3 in range(n3):

                j1 = dm1*gp1 + x1
                j2 = dm2*gp2 + x2
                j3 = dm3*gp3 + x3

                for l1 in range(nd1):
                    for l2 in range(nd2):
                        for l3 in range(nd3):

                            i1 = si1 + cm1*(x1//dm1) + l1 + d1
                            i2 = si2 + cm2*(x2//dm2) + l2 + d2
                            i3 = si3 + cm3*(x3//dm3) + l3 + d3

                            k1 = sk1 + x1%dm1-dm1*(l1//cm1)
                            k2 = sk2 + x2%dm2-dm2*(l2//cm2)
                            k3 = sk3 + x3%dm3-dm3*(l3//cm3)

                            if k1<ndT1 and k1>-1 and k2<ndT2 and k2>-1 and k3<ndT3 and k3>-1\
                                and l1<nd1 and l2<nd2 and l3<nd3 and i1<nc1 and i2<nc2 and i3<nc3:
                                Mt[j1,j2,j3, l1 + sl1,l2 + sl2,l3 + sl3] = M[i1,i2,i3, k1,k2,k3]

_transpose  = {1:transpose_1d,2:transpose_2d, 3:transpose_3d}
_args_dtype = {1:[repr('float[:,:]')]*2 + ['int']*11,2:[repr('float[:,:,:,:]')]*2 + ['int']*22, 3:[repr('float[:,:,:,:,:,:]')]*2 + ['int']*33}
