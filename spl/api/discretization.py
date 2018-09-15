# coding: utf-8

from collections import OrderedDict

from pyccel.ast import Nil

from sympde.core import BasicForm as sym_BasicForm
from sympde.core import BilinearForm as sym_BilinearForm
from sympde.core import LinearForm as sym_LinearForm
from sympde.core import Integral as sym_Integral
from sympde.core import Equation as sym_Equation
from sympde.core import Model as sym_Model
from sympde.core import Boundary as sym_Boundary
from sympde.core import evaluate

from spl.api.codegen.ast import Kernel
from spl.api.codegen.ast import Assembly
from spl.api.codegen.ast import Interface
from spl.api.codegen.printing import pycode
from spl.api.codegen.utils import write_code
from spl.linalg.stencil import StencilVector, StencilMatrix

import os
import importlib
import string
import random

SPL_DEFAULT_FOLDER = '__pycache__/spl'

def random_string( n ):
    chars    = string.ascii_uppercase + string.ascii_lowercase + string.digits
    selector = random.SystemRandom()
    return ''.join( selector.choice( chars ) for _ in range( n ) )


class DiscreteBoundary(object):
    def __init__(self, expr, axis=None, ext=None):
        if not isinstance(expr, sym_Boundary):
            raise TypeError('> Expecting a Boundary object')

        if not(axis) and not(ext):
            msg = '> for the moment, both axis and ext must be given'
            raise NotImplementedError(msg)

        self._expr = expr
        self._axis = axis
        self._ext = ext

    @property
    def expr(self):
        return self._expr

    @property
    def axis(self):
        return self._axis

    @property
    def ext(self):
        return self._ext

class BasicDiscrete(object):

    def __init__(self, a, kernel_expr, namespace=globals(), to_compile=True,
                 module_name=None, boundary=None, target=None):

        # ...
        if not target:
            if len(kernel_expr) > 1:
                raise ValueError('> Expecting only one kernel')

            target = kernel_expr[0].target
        # ...

        # ...
        if boundary:
            if not isinstance(boundary, (tuple, list, DiscreteBoundary)):
                raise TypeError('> Expecting a tuple, list or DiscreteBoundary')

            if isinstance(boundary, DiscreteBoundary):
                if not( boundary.expr is target ):
                    raise ValueError('> Unconsistent boundary with symbolic model')

                boundary = [boundary.axis, boundary.ext]
                boundary = [boundary]

            # boundary is now a list of boundaries
            # TODO shall we keep it this way? since this is the simplest
            # interface to be able to compute Integral on a curve in 3d
        # ...

        # ...
        kernel = Kernel(a, kernel_expr, target=target,
                        discrete_boundary=boundary)
        assembly = Assembly(kernel)
        interface = Interface(assembly)
        # ...

        # ...
        self._expr = a
        self._kernel_expr = kernel_expr
        self._target = target
        self._tag = kernel.tag
        self._mapping = None
        self._interface = interface
        self._dependencies = self.interface.dependencies
        # ...

        # generate python code as strings for dependencies
        self._dependencies_code = self._generate_code()

        self._dependencies_fname = None
        self._interface_code = None
        self._func = None
        if to_compile:
            # save dependencies code
            self._save_code(module_name=module_name)

            # generate code for Python interface
            self._generate_interface_code()

            # compile code
            self._compile(namespace)

    @property
    def expr(self):
        return self._expr

    @property
    def kernel_expr(self):
        return self._kernel_expr

    @property
    def target(self):
        return self._target

    @property
    def tag(self):
        return self._tag

    @property
    def mapping(self):
        return self._mapping

    @property
    def interface(self):
        return self._interface

    @property
    def dependencies(self):
        return self._dependencies

    @property
    def interface_code(self):
        return self._interface_code

    @property
    def dependencies_code(self):
        return self._dependencies_code

    @property
    def dependencies_fname(self):
        return self._dependencies_fname

    @property
    def dependencies_modname(self):
        module_name = os.path.splitext(self.dependencies_fname)[0]
        module_name = module_name.replace('/', '.')
        return module_name

    @property
    def func(self):
        return self._func

    def _generate_code(self):
        # ... generate code that can be pyccelized
        code = ''
        for dep in self.dependencies:
            code = '{code}\n{dep}'.format(code=code, dep=pycode(dep))
        # ...
        return code

    def _save_code(self, module_name=None):
        folder = SPL_DEFAULT_FOLDER

        code = self.dependencies_code
        if module_name is None:
            module_name = 'dependencies_{}'.format(self.tag)
        self._dependencies_fname = write_code(module_name, code, ext='py', folder=folder)

    def _generate_interface_code(self, module_name=None):
        imports = []

        # ... generate imports from dependencies module
        pattern = 'from {module} import {dep}'

        if module_name is None:
            module_name = self.dependencies_modname

        for dep in self.dependencies:
            txt = pattern.format(module=module_name, dep=dep.name)
            imports.append(txt)
        # ...

        # ...
        imports = '\n'.join(imports)
        # ...

        code = pycode(self.interface)

        self._interface_code = '{imports}\n{code}'.format(imports=imports, code=code)

    def _compile(self, namespace, module_name=None):

        if module_name is None:
            module_name = self.dependencies_modname

        # ...
        dependencies_module = importlib.import_module(module_name)
        # ...

        # ...
        code = self.interface_code
        name = self.interface.name

        exec(code, namespace)
        interface = namespace[name]
        # ...

        self._func = interface

    def _check_arguments(self, **kwargs):

        _kwargs = {}

        # ... mandatory arguments
        sym_args = self.interface.in_arguments
        keys = [str(a) for a in sym_args]
        for key in keys:
            try:
                _kwargs[key] = kwargs[key]
            except:
                raise KeyError('Unconsistent argument with interface')
        # ...

        # ... optional (inout) arguments
        sym_args = self.interface.inout_arguments
        keys = [str(a) for a in sym_args]
        for key in keys:
            try:
                _kwargs[key] = kwargs[key]
            except:
                pass
        # ...

        return _kwargs


class DiscreteBilinearForm(BasicDiscrete):

    def __init__(self, expr, kernel_expr, *args, **kwargs):
        if not isinstance(expr, sym_BilinearForm):
            raise TypeError('> Expecting a symbolic BilinearForm')

        BasicDiscrete.__init__(self, expr, kernel_expr, **kwargs)

        if not args:
            raise ValueError('> fem spaces must be given as a list/tuple')

        self._spaces = args[0]

        if len(args) > 1:
            self._mapping = args[1]

    @property
    def spaces(self):
        return self._spaces

    def assemble(self, **kwargs):
        newargs = tuple(self.spaces)

        if self.mapping:
            newargs = newargs + (self.mapping,)

        kwargs = self._check_arguments(**kwargs)

        return self.func(*newargs, **kwargs)

class DiscreteLinearForm(BasicDiscrete):

    def __init__(self, expr, kernel_expr, *args, **kwargs):
        if not isinstance(expr, sym_LinearForm):
            raise TypeError('> Expecting a symbolic LinearForm')

        BasicDiscrete.__init__(self, expr, kernel_expr, **kwargs)

        self._space = args[0]

        if len(args) > 1:
            self._mapping = args[1]

    @property
    def space(self):
        return self._space

    def assemble(self, **kwargs):
        newargs = (self.space,)

        if self.mapping:
            newargs = newargs + (self.mapping,)

        kwargs = self._check_arguments(**kwargs)

        return self.func(*newargs, **kwargs)

class DiscreteIntegral(BasicDiscrete):

    def __init__(self, expr, kernel_expr, *args, **kwargs):
        if not isinstance(expr, sym_Integral):
            raise TypeError('> Expecting a symbolic Integral')

        BasicDiscrete.__init__(self, expr, kernel_expr, **kwargs)

        self._space = args[0]

        if len(args) > 1:
            self._mapping = args[1]

    @property
    def space(self):
        return self._space

    def assemble(self, **kwargs):
        newargs = (self.space,)

        if self.mapping:
            newargs = newargs + (self.mapping,)

        kwargs = self._check_arguments(**kwargs)

        return self.func(*newargs, **kwargs)


class DiscreteSumForm(BasicDiscrete):

    def __init__(self, a, kernel_expr, *args, **kwargs):
        if not isinstance(a, (sym_BilinearForm, sym_LinearForm, sym_Integral)):
            raise TypeError('> Expecting a symbolic BilinearForm, LinearFormn Integral')

        self._expr = a

        # create a module name if not given
        tag = random_string( 8 )
        module_name = kwargs.pop('module_name', 'dependencies_{}'.format(tag))

        # ...
        forms = []
        boundaries = kwargs.pop('boundary', [])
        if isinstance(boundaries, DiscreteBoundary): boundaries = [boundaries]

        kwargs['to_compile'] = False
        kwargs['module_name'] = module_name
        for e in kernel_expr:
            kwargs['target'] = e.target
            if isinstance(e.target, sym_Boundary):
                boundary = [i for i in boundaries if i.expr is e.target]
                if boundary: kwargs['boundary'] = boundary[0]

            if isinstance(a, sym_BilinearForm):
                ah = DiscreteBilinearForm(a, kernel_expr, *args, **kwargs)

            elif isinstance(a, sym_LinearForm):
                ah = DiscreteLinearForm(a, kernel_expr, *args, **kwargs)

            elif isinstance(a, sym_Integral):
                ah = DiscreteIntegral(a, kernel_expr, *args, **kwargs)

            forms.append(ah)
            kwargs['boundary'] = None

        self._forms = forms
        # ...

        # ... save all dependencies codes in one single string
        code = ''
        for ah in self.forms:
            code = '{code}\n{ah}'.format(code=code, ah=ah.dependencies_code)
        self._dependencies_code = code
        # ...

        # ...
        # save dependencies code
        self._save_code(module_name=module_name)
        # ...

        # ...
        namespace = kwargs.pop('namespace', globals())
        module_name = self.dependencies_modname
        code = ''
        for ah in self.forms:
            # generate code for Python interface
            ah._generate_interface_code(module_name=module_name)

            # compile code
            ah._compile(namespace, module_name=module_name)
        # ...

    @property
    def forms(self):
        return self._forms

    def assemble(self, **kwargs):
        form = self.forms[0]
        M = form.assemble(**kwargs)
        if isinstance(M, (StencilVector, StencilMatrix)):
            M = [M]

        for f in self.forms[1:]:
            n = len(f.interface.inout_arguments)
            # add arguments
            for i in range(0, n):
                key = str(f.interface.inout_arguments[i])
                kwargs[key] = M[i]

#            print('> before = ', M[0]._data.flatten())
            M = form.assemble(**kwargs)
            if isinstance(M, (StencilVector, StencilMatrix)):
                M = [M]
#            print('> after  = ', M[0]._data.flatten())

            # remove arguments
            for i in range(0, n):
                key = str(f.interface.inout_arguments[i])
                kwargs.pop(key)

        if len(M) == 1: M = M[0]

        return M

class DiscreteEquation(BasicDiscrete):

    def __init__(self, expr, *args, **kwargs):
        if not isinstance(expr, sym_Equation):
            raise TypeError('> Expecting a symbolic Equation')

        self._expr = expr
        self._lhs = kwargs.pop('lhs', None)
        self._rhs = kwargs.pop('rhs', None)

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs

    def solve(self, *args, **kwargs):
        raise NotImplementedError('TODO')

class Model(BasicDiscrete):

    def __init__(self, expr, *args, **kwargs):
        if not isinstance(expr, sym_Model):
            raise TypeError('> Expecting a symbolic Model')

        if not args:
            raise ValueError('> fem spaces must be given as a list/tuple')

        self._expr = expr
        self._spaces = args[0]

        if len(args) > 1:
            self._mapping = args[1]

        # create a module name if not given
        tag = random_string( 8 )
        module_name = kwargs.pop('module_name', 'dependencies_{}'.format(tag))

        # ... create discrete forms
        test_space = self.spaces[0]
        trial_space = self.spaces[1]
        d_forms = {}
        # TODO treat equation forms
        for name, a in list(expr.forms.items()):
            kernel_expr = evaluate(a)
            if isinstance(a, sym_BilinearForm):
                spaces = (test_space, trial_space)
                ah = DiscreteBilinearForm(a, kernel_expr, spaces,
                                          to_compile=False,
                                          module_name=module_name)

            elif isinstance(a, sym_LinearForm):
                ah = DiscreteLinearForm(a, kernel_expr, test_space,
                                        to_compile=False,
                                        module_name=module_name)

            elif isinstance(a, sym_Integral):
                ah = DiscreteIntegral(a, kernel_expr, test_space,
                                      to_compile=False,
                                      module_name=module_name)

            d_forms[name] = ah

        d_forms = OrderedDict(sorted(d_forms.items()))
        self._forms = d_forms
        # ...

        # ...
        if expr.equation:
            # ...
            lhs_h = None
            lhs = expr.equation.lhs
            if not isinstance(lhs, Nil):
                if lhs.name in list(d_forms.keys()):
                    lhs_h = d_forms[lhs.name]
            # ...

            # ...
            rhs_h = None
            rhs = expr.equation.rhs
            if not isinstance(rhs, Nil):
                if rhs.name in list(d_forms.keys()):
                    rhs_h = d_forms[rhs.name]
            # ...

            equation = DiscreteEquation(expr.equation, lhs=lhs_h, rhs=rhs_h)
            self._equation = equation
        # ...

        # ... save all dependencies codes in one single string
        code = ''
        for name, ah in list(self.forms.items()):
            code = '{code}\n{ah}'.format(code=code, ah=ah.dependencies_code)
        self._dependencies_code = code
        # ...

        # ...
        # save dependencies code
        self._save_code(module_name=module_name)
        # ...

        # ...
        namespace = kwargs.pop('namespace', globals())
        module_name = self.dependencies_modname
        code = ''
        for name, ah in list(self.forms.items()):
            # generate code for Python interface
            ah._generate_interface_code(module_name=module_name)

            # compile code
            ah._compile(namespace, module_name=module_name)
        # ...

    @property
    def forms(self):
        return self._forms

    @property
    def equation(self):
        return self._equation

    @property
    def spaces(self):
        return self._spaces

    def assemble(self, **kwargs):
        lhs = self.equation.lhs
        if lhs:
            lhs.assemble(**kwargs)

        rhs = self.equation.rhs
        if rhs:
            raise NotImplementedError('TODO')



def discretize(a, *args, **kwargs):

    if isinstance(a, sym_BasicForm):
        kernel_expr = evaluate(a)
        if len(kernel_expr) > 1:
            return DiscreteSumForm(a, kernel_expr, *args, **kwargs)

    if isinstance(a, sym_BilinearForm):
        return DiscreteBilinearForm(a, kernel_expr, *args, **kwargs)

    elif isinstance(a, sym_LinearForm):
        return DiscreteLinearForm(a, kernel_expr, *args, **kwargs)

    elif isinstance(a, sym_Integral):
        return DiscreteIntegral(a, kernel_expr, *args, **kwargs)

    elif isinstance(a, sym_Model):
        return Model(a, *args, **kwargs)
