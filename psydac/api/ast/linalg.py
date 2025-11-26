#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import sys
import os
import importlib
import inspect

import numpy as np

from functools import lru_cache
from mpi4py    import MPI

from sympy import Mul
from sympy import Mod as sy_Mod, Range, Symbol, Max
from sympy import Function, Integer

from psydac.pyccel.ast.core import Variable, IndexedVariable
from psydac.pyccel.ast.core import For, Comment
from psydac.pyccel.ast.core import String
from psydac.pyccel.ast.core import ValuedArgument
from psydac.pyccel.ast.core import Assign
from psydac.pyccel.ast.core import AugAssign
from psydac.pyccel.ast.core import Product
from psydac.pyccel.ast.core import FunctionDef
from psydac.pyccel.ast.core import Import

from psydac.pyccel.ast.datatypes import NativeInteger

from psydac.api.ast.nodes     import FloorDiv
from psydac.api.ast.utilities import variables
from psydac.api.ast.utilities import build_pyccel_type_annotations
from psydac.api.utilities     import flatten

from psydac.api.ast.basic     import SplBasic
from psydac.api.printing      import pycode
from psydac.api.settings      import PSYDAC_DEFAULT_FOLDER
from psydac.api.utilities     import mkdir_p, touch_init_file, random_string, write_code

#==============================================================================
def variable_to_sympy(x):
    if isinstance(x, Variable) and isinstance(x.dtype, NativeInteger):
        x = Symbol(x.name, integer=True)
    return x

#==============================================================================
def compute_diag_len(p, md, mc, return_padding=False):
    p, md, mc = np.int64(p), np.int64(md), np.int64(mc)
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

def toInteger(a):
    if isinstance(a, np.int64):
        return Integer(int(a))
    return a
#==============================================================================

class LinearOperatorDot(SplBasic):
    """
    Generate the Matrix Vector Product function for a BlockLinearOperator,StencilMatrix or StencilInterfaceMatrix.
    In case of a BlockLinearOperator we give the number of blocks along the rows and columns specified with the block_shape.
    In case of StencilMatrix or StencilInterfaceMatrix the block_shape = (1,1).

    Parameters
    ----------
    ndim : int
        Number of dimensions.

    block_shape: tuple of ints
        The number of blocks along the rows and columns.

    comm: MPI.Comm
        MPI intra-communicator.
    """
    def __new__(cls, ndim, block_shape, comm=None, **kwargs):
        if comm is not None:
            assert isinstance(comm, MPI.Comm)
            comm_id = comm.py2f()
        else:
            comm_id = None
        return cls.__hashable_new__(ndim, block_shape, comm_id, **kwargs)

    @classmethod
    @lru_cache(maxsize=32)
    def __hashable_new__(cls, ndim, block_shape, comm_id=None, **kwargs):

        # If integer communicator is provided, convert it to mpi4py object
        comm = None if comm_id is None else MPI.COMM_WORLD.f2py(comm_id)

        # Generate random tag, unique for all processes in MPI communicator
        tag = random_string(8)
        if comm is not None and comm.size>1:
            tag = comm.bcast(tag, root=0)

        # Create new instance of this class
        obj = SplBasic.__new__(cls, tag, prefix='lo_dot', comm=comm)

        # Initialize instance (code generation happens here)
        backend        = dict(kwargs.pop('backend'))
        code           = obj._initialize(ndim, block_shape, backend=backend, **kwargs)
        obj._arguments = dict((str(a.name), a) for a in code.arguments)
        obj._code      = code
        obj._folder    = obj._initialize_folder()
        obj._generate_code(backend=backend)
        obj._compile(backend=backend)

        # Return instance
        return obj

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

    def _initialize(self, ndim, block_shape, **kwargs):

        keys         = kwargs.pop('keys')
        backend      = kwargs.pop('backend', None)
        nrows        = list(kwargs.pop('nrows', [None]*len(keys)))
        nrows_extra  = list(kwargs.pop('nrows_extra', [None]*len(keys)))
        starts       = list(kwargs.pop('starts', [None]*len(keys)))
        pads         = kwargs.pop('pads')
        gpads        = kwargs.pop('gpads')
        cm           = kwargs.pop('cm')
        dm           = kwargs.pop('dm')
        interface    = kwargs.pop('interface', False)
        flip_axis       = kwargs.pop('flip_axis',[1]*ndim)
        interface_axis  = kwargs.pop('interface_axis', None)
        d_start         = kwargs.pop('d_start', None)
        c_start         = kwargs.pop('c_start', None)
        dtype           = kwargs.pop('dtype', float)

        # Adapt the type of data treated in our dot function
        if dtype==complex:
            dtype_string='complex'
        else:
            dtype_string='real'

        mats            = [variables('mat{}'.format(''.join(str(i) for i in key)),dtype_string, cls=IndexedVariable, rank=2*ndim) for key in keys]
        xs              = [variables('x{}'.format(i),dtype_string, cls=IndexedVariable, rank=ndim) for i in range(block_shape[1])]
        outs            = [variables('out{}'.format(i),dtype_string, cls=IndexedVariable, rank=ndim) for i in range(block_shape[0])]

        func_args    = (*mats, *xs, *outs)
        shared       = (*mats, *xs, *outs)
        firstprivate = ()
        openmp       = False if backend is None else backend["openmp"]
        gbody        = []

        for it in range(2):
            diag_keys = True if it==0 else False
            for k,key in enumerate(keys):
                if diag_keys and key[0] != key[1]:continue
                if not diag_keys and key[0] == key[1]:continue
                key_str         = ''.join(str(i) for i in key)
                nrows_k         = nrows[k] if nrows[k] else variables('n{}_1:%s'.format(key_str)%(ndim+1),  'int')
                nrows_extra_k   = nrows_extra[k] if nrows_extra[k] else variables('ne{}_1:%s'.format(key_str)%(ndim+1),  'int')
                starts_k        = starts[k] if starts[k] else variables('s{}_1:%s'.format(key_str)%(ndim+1),  'int')
                nrows_k         = tuple(map(toInteger,nrows_k))
                nrows_extra_k   = tuple(map(toInteger,nrows_extra_k))
                starts_k        = tuple(map(toInteger,starts_k))
                indices1        = variables('i1:%s'%(ndim+1),  'int')
                bb              = variables('b1:%s'%(ndim+1),  'int')
                indices2        = variables('k1:%s'%(ndim+1),  'int')
                v               = variables('v{}'.format(key_str),dtype_string)
                xshape          = variables('xn1:%s'%(ndim+1),  'int')

                pads_k          = tuple(map(toInteger, pads[k]))
                gpads_k         = tuple(map(toInteger,gpads[k]))
                cm_k            = tuple(map(toInteger,cm[k]))
                dm_k            = tuple(map(toInteger,dm[k]))
                d_start_k       = toInteger(d_start[k] if d_start else None)
                c_start_k       = toInteger(c_start[k] if c_start else None)

                nrows[k]       = tuple(nrows_k)
                nrows_extra[k] = tuple(nrows_extra_k)
                starts[k]      = tuple(starts_k)

                mat = mats[k]
                x   = xs[key[1]]
                out = outs[key[0]]

                ndiags, _ = list(zip(*[compute_diag_len(p,mj,mi, return_padding=True) for p,mi,mj in zip(pads_k,cm_k,dm_k)]))

                inits  = [Assign(b,p*m+p+1-n-Mod(s,m)) for b,p,m,n,s in zip(bb, gpads_k, dm_k, ndiags, starts_k) if not isinstance(p*m+p+1-n-Mod(s,m),(int,np.int64, Integer))]

                if any(f==-1 for f in flip_axis):
                    inits.append(Assign(xshape, Function('shape')(x)))

                bb     = [b if not isinstance(p*m+p+1-n-Mod(s,m),(int,np.int64, Integer)) else p*m+p+1-n-Mod(s,m) for b,p,m,n,s in zip(bb, gpads_k, dm_k, ndiags, starts_k)]

                ranges = [Range(variable_to_sympy(n)) for n in ndiags]
                diff   = [variable_to_sympy(gp-p) for gp,p in zip(gpads_k, pads_k)]

    #            if d_start_k:bb[interface_axis] += d_start_k

                x_indices = []
                for i1,mi,mj,b,s,d,i2,f,xl in zip(indices1,cm_k,dm_k,bb,starts_k,diff,indices2,flip_axis,xshape):
                    index =  b-d+FloorDiv((i1+Mod(s,mj)),mi)*mj + i2
                    if f==-1:
                        index = xl-1-index
                    x_indices.append(index)

                out_indices = [i+m*j for i,j,m in zip(indices1,gpads_k,cm_k)]
                if c_start_k:out_indices[interface_axis] += c_start_k

                v1 = x[tuple(x_indices)]
                v2 = mat[tuple(i+m*j for i,j,m in zip(indices1,gpads_k,cm_k))+ tuple(indices2)]
                v3 = out[tuple(out_indices)]

                body = [AugAssign(v,'+' ,Mul(v2, v1))]

                # Decompose fused loop over Cartesian product of multiple ranges
                # into nested loops, each over a single range
                for i,j in zip(indices2[::-1], ranges[::-1]):
                    body = [For(i,j, body)]

                # Adapt data type of the variable v=0
                if dtype==complex:
                    body.insert(0,Assign(v, 0.0+0j))
                else:
                    body.insert(0,Assign(v, 0.0))

                if diag_keys:
                    body.append(Assign(v3,v))
                else:
                    body.append(AugAssign(v3,'+',v))

                ranges = [Range(variable_to_sympy(i)) for i in nrows_k]

                # Decompose fused loop over Cartesian product of multiple ranges
                # into nested loops, each over a single range
                for i,j in zip(indices1[::-1], ranges[::-1]):
                    body = [For(i,j, body)]

                if openmp:
                    pragma = "#$omp for schedule(static) collapse({}) nowait".format(str(ndim))
                    body   = [Comment(pragma)] + body

                nrowscopy_k = list(nrows_k).copy()
                nrows_k     = list(nrows_k)
                for dim in range(ndim):

                    if nrows_extra_k[dim] == 0:continue

                    v1 = [b-d+FloorDiv((i1+(nrows_k[dim] if dim==x else 0)+Mod(s,mj)),mi)*mj + i2 for x,i1,mi,mj,b,s,d,i2 in zip(range(ndim), indices1,cm_k,dm_k,bb,starts_k,diff,indices2)]
                    v2 = [i+m*j for i,j,m in zip(indices1,gpads_k,cm_k)]

                    v2[dim] += nrows_k[dim]

                    v3 = v2

                    for i,v1i in enumerate(v1):
                        if flip_axis[i] == -1:
                            v1[i] = xshape[i]-1-v1[i]

                    v1 = x[tuple(v1)]
                    v2 = mat[tuple(v2)+ indices2]

                    if c_start_k:v3[interface_axis] += c_start_k
                    v3 = out[tuple(v3)]

                    rows = list(nrows_k)
                    rows[dim] = nrows_extra_k[dim]

                    ranges       = [variable_to_sympy(n) for n in ndiags]
                    ranges[dim] -= variable_to_sympy(indices1[dim]) + 1
                    ranges       = [ind if i>=dim else ind - Max(0, variable_to_sympy(d1)+1-variable_to_sympy(r)) for i,(ind,d1,r) in enumerate(zip(ranges, indices1, nrowscopy_k)) ]
                    ranges       = [Range(i) for i in ranges]

                    for_body = [AugAssign(v, '+',Mul(v1,v2))]

                    # Decompose fused loop over Cartesian product of multiple ranges
                    # into nested loops, each over a single range
                    for i,j in zip(indices2[::-1], ranges[::-1]):
                        for_body = [For(i,j, for_body)]

                    # Adapt data type of the variable v=0
                    if dtype == complex:
                        for_body.insert(0, Assign(v, 0.0 + 0j))
                    else:
                        for_body.insert(0, Assign(v, 0.0))

                    if diag_keys:
                        for_body.append(Assign(v3,v))
                    else:
                        for_body.append(AugAssign(v3,'+',v))

                    ranges = [Range(variable_to_sympy(i)) for i in rows]

                    # Decompose fused loop over Cartesian product of multiple ranges
                    # into nested loops, each over a single range
                    for i,j in zip(indices1[::-1], ranges[::-1]):
                        for_body = [For(i,j, for_body)]

                    if openmp:
                        pragma = "#$omp for schedule(static) collapse({}) nowait".format(str(ndim))
                        for_body = [Comment(pragma)] + for_body

                    body += for_body

                    nrows_k[dim] += nrows_extra_k[dim]

                body      = inits + body
                gbody    += body

        body = gbody

        if isinstance(starts[0][0], Variable):
            func_args    = func_args    + tuple(flatten(starts))
            firstprivate = firstprivate + tuple(flatten(starts))

        if isinstance(nrows[0][0], Variable):
            func_args    = func_args    + tuple(flatten(nrows))
            firstprivate = firstprivate + tuple(flatten(nrows))

        if isinstance(nrows_extra[0][0], Variable):
            func_args    = func_args    + tuple(flatten(nrows_extra))
            firstprivate = firstprivate + tuple(flatten(nrows_extra))

        decorators = {}
        header     = None
        imports    = []
        if backend:
            if backend['name'] == 'pyccel':
                func_args = build_pyccel_type_annotations(func_args)
            elif backend['name'] == 'pythran':
                header = build_pythran_types_header(name, func_args)

        if openmp:
            shared = ','.join(str(a) for a in shared)
            firstprivate = "firstprivate({})".format(','.join(str(a) for a in firstprivate)) if firstprivate else ""
            pragma1 = "#$omp parallel default(private) shared({}) {}\n".format(shared, firstprivate)
            pragma2 = "#$omp end parallel"
            body    = [Comment(pragma1)] + body + [Comment(pragma2)]

        return FunctionDef(self.name, list(func_args), [], body, imports=imports, decorators=decorators)

    def _initialize_folder(self, folder=None):
        # ...
        if folder is None:
            basedir = os.getcwd()
            folder = PSYDAC_DEFAULT_FOLDER['name']
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

        modname = 'dependencies_{}'.format(self.tag)

        if self.comm is None or self.comm.rank == 0:
            python_code = pycode.pycode(self.code)
            write_code(modname + '.py', python_code, folder=self.folder)

        self._modname = modname

    def _compile(self, backend=None):

        # Make sure that code generated by process 0 is available to all others
        # Cheapest solution is a broadcast from process 0
        comm = self.comm
        if comm is not None and comm.size > 1:
            comm.bcast(0, root=0)

        module_name = self._modname
        sys.path.append(self.folder)
        importlib.invalidate_caches()
        package = importlib.import_module(module_name)
        sys.path.remove(self.folder)

        if backend and backend['name'] == 'pyccel':
            package = self._compile_pyccel(package, backend)

        self._func = getattr(package, self.name)

    def _compile_pyccel(self, mod, backend, verbose=False):

        # ... convert python to fortran using pyccel
        compiler_family = backend['compiler_family']
        flags           = backend['flags']
        _PYCCEL_FOLDER  = backend['folder']
        openmp          = backend["openmp"]

        from pyccel import epyccel

        fmod = epyccel(mod,
                       openmp  = openmp,
                       compiler_family = compiler_family,
                       flags   = flags,
                       comm    = self.comm,
                       bcast   = True,
                       folder  = _PYCCEL_FOLDER,
                       verbose = verbose)
        return fmod

#==============================================================================
class VectorInner(SplBasic):

    def __new__(cls, ndim, backend=None):
        tag = random_string(8)
        obj = SplBasic.__new__(cls, tag, prefix='v_dot')
        obj._ndim = ndim
        obj._backend = backend
        obj._func = obj._initialize()
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

        indices = variables('i1:%s'%(ndim+1), 'int')
        dims    = variables('n1:%s'%(ndim+1), 'int')
        pads    = variables('p1:%s'%(ndim+1), 'int')
        out     = variables('out','real')
        x1,x2   = variables('x1, x2','real', rank=ndim, cls=IndexedVariable)

        body = []
        ranges = [Range(p, n-p) for n, p in zip(dims, pads)]
        target = Product(*ranges)

        v1 = x1[indices]
        v2 = x2[indices]

        body = [AugAssign(out, '+' , Mul(v1, v2))]
        body = [For(indices, target, body)]
        body.insert(0, Assign(out, 0.0))
        body.append(Return(out))

        func_args = (x1, x2) + pads + dims

        self._imports = [Import('itertools', 'product')]

        decorators = {}
        header = None

        if self.backend['name'] == 'pyccel':
            func_args = build_pyccel_type_annotations(func_args)
        elif self.backend['name'] == 'pythran':
            header = build_pythran_types_header(self.name, func_args)

        return FunctionDef(self.name, list(func_args), [], body,
                           decorators=decorators, header=header)
