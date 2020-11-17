# coding: utf-8

from sympy import Mul, Tuple
from sympy import Mod, Abs

from pyccel.ast.core import Variable, IndexedVariable
from pyccel.ast.core import For
from pyccel.ast.core import Assign
from pyccel.ast.core import AugAssign
from pyccel.ast.core import Range, Product
from pyccel.ast.core import FunctionDef
from pyccel.ast.core import FunctionCall
from pyccel.ast.core import Import
from pyccel.ast.utilities import build_types_decorator
from psydac.api.ast.utilities import variables, math_atoms_as_str

from psydac.fem.splines import SplineSpace
from psydac.fem.tensor  import TensorFemSpace
from psydac.fem.vector  import ProductFemSpace

from psydac.api.ast.basic import SplBasic

class LinearOperatorDot(SplBasic):

    def __new__(cls, ndim, backend=None):


        obj = SplBasic.__new__(cls, 'dot',name='lo_dot',prefix='lo_dot')
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

#        if self.backend['name'] == 'pyccel':
#            decorators = {'types': build_types_decorator(func_args), 'external_call':[]}
#        elif self.backend['name'] == 'numba':
#            decorators = {'jit':[]}
#        elif self.backend['name'] == 'pythran':
#            header = build_pythran_types_header(self.name, func_args)

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
