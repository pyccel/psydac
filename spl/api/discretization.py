# coding: utf-8
from sympde.core import BilinearForm as sym_BilinearForm
from sympde.core import LinearForm as sym_LinearForm
from sympde.core import FunctionForm as sym_FunctionForm
from sympde.core import Model as sym_Model

from spl.api.codegen.ast import Interface
from spl.api.codegen.printing import pycode
from spl.api.codegen.utils import write_code

import os
import importlib

class BasicForm(object):

    def __init__(self, expr, namespace=globals(), to_compile=True, module_name=None):
        self._expr = expr
        self._mapping = None
        self._interface = Interface(expr)
        self._dependencies = self.interface.dependencies

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
        folder = 'tmp'

        code = self.dependencies_code
        if module_name is None:
            ID = abs(hash(self))
            module_name = 'dependencies_{}'.format(ID)
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

class BilinearForm(BasicForm):

    def __init__(self, expr, *args, **kwargs):
        if not isinstance(expr, sym_BilinearForm):
            raise TypeError('> Expecting a symbolic BilinearForm')

        BasicForm.__init__(self, expr, **kwargs)

        if not args:
            raise ValueError('> fem spaces must be given as a list/tuple')

        self._spaces = args[0]

        if len(args) > 1:
            self._mapping = args[1]

    @property
    def spaces(self):
        return self._spaces

    def assemble(self, *args, **kwargs):
        newargs = tuple(self.spaces)

        if self.mapping:
            newargs = newargs + (self.mapping,)

        newargs = newargs + tuple(args)

        return self.func(*newargs, **kwargs)

class LinearForm(BasicForm):

    def __init__(self, expr, *args, **kwargs):
        if not isinstance(expr, sym_LinearForm):
            raise TypeError('> Expecting a symbolic LinearForm')

        BasicForm.__init__(self, expr, **kwargs)

        self._space = args[0]

        if len(args) > 1:
            self._mapping = args[1]

    @property
    def space(self):
        return self._space

    def assemble(self, *args, **kwargs):
        newargs = (self.space,)

        if self.mapping:
            newargs = newargs + (self.mapping,)

        newargs = newargs + tuple(args)

        return self.func(*newargs, **kwargs)

class FunctionForm(BasicForm):

    def __init__(self, expr, *args, **kwargs):
        if not isinstance(expr, sym_FunctionForm):
            raise TypeError('> Expecting a symbolic FunctionForm')

        BasicForm.__init__(self, expr, **kwargs)

        self._space = args[0]

        if len(args) > 1:
            self._mapping = args[1]

    @property
    def space(self):
        return self._space

    def assemble(self, *args, **kwargs):
        newargs = (self.space,)

        if self.mapping:
            newargs = newargs + (self.mapping,)

        newargs = newargs + tuple(args)

        return self.func(*newargs, **kwargs)

class Model(BasicForm):

    def __init__(self, expr, *args, **kwargs):
        if not isinstance(expr, sym_Model):
            raise TypeError('> Expecting a symbolic Model')

        if not args:
            raise ValueError('> fem spaces must be given as a list/tuple')

        self._spaces = args[0]

        if len(args) > 1:
            self._mapping = args[1]

        # create a module name if not given
        module_name = kwargs.pop('module_name', 'dependencies_{}'.format(abs(hash(self))))

        # ... create discrete forms
        test_space = self.spaces[0]
        trial_space = self.spaces[1]
        forms = []
        for a in expr.forms:
            if isinstance(a, sym_BilinearForm):
                spaces = (test_space, trial_space)
                ah = BilinearForm(a, spaces, to_compile=False,
                                  module_name=module_name)

            elif isinstance(a, sym_LinearForm):
                ah = LinearForm(a, test_space, to_compile=False,
                                module_name=module_name)

            elif isinstance(a, sym_FunctionForm):
                ah = FunctionForm(a, test_space, to_compile=False,
                                  module_name=module_name)

            forms.append(ah)
        # ...

        # ... save all dependencies codes in one single string
        code = ''
        for ah in forms:
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
        for ah in forms:
            # generate code for Python interface
            ah._generate_interface_code(module_name=module_name)

            # compile code
            ah._compile(namespace, module_name=module_name)
        # ...

    @property
    def spaces(self):
        return self._spaces

    def assemble(self, *args, **kwargs):
        raise NotImplementedError('TODO')


def discretize(a, *args, **kwargs):

    if isinstance(a, sym_BilinearForm):
        return BilinearForm(a, *args, **kwargs)

    elif isinstance(a, sym_LinearForm):
        return LinearForm(a, *args, **kwargs)

    elif isinstance(a, sym_FunctionForm):
        return FunctionForm(a, *args, **kwargs)

    elif isinstance(a, sym_Model):
        return Model(a, *args, **kwargs)

