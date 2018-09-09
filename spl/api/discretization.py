# coding: utf-8
from sympde.core import BilinearForm as sym_BilinearForm
from sympde.core import LinearForm as sym_LinearForm
from sympde.core import FunctionForm as sym_FunctionForm

from spl.api.codegen.ast import Interface
from spl.api.codegen.printing import pycode
from spl.api.codegen.utils import write_code

import os
import importlib

class BasicForm(object):

    def __init__(self, expr, namespace=globals()):
        self._expr = expr
        self._mapping = None
        self._interface = Interface(expr)
        self._dependencies = self.interface.dependencies

        # generate python code as strings for dependencies
        self._dependencies_code = self._generate_code()

        # save dependencies code
        self._dependencies_fname = self._save_code()

        # generate code for Python interface
        self._interface_code = self._generate_interface_code()

        # compile code
        self._func = self._compile(namespace)

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

    def _save_code(self):
        folder = 'tmp'

        code = self.dependencies_code
        ID = abs(hash(self))
        name = 'dependencies_{}'.format(ID)
        fname = write_code(name, code, ext='py', folder=folder)
        return fname

    def _generate_interface_code(self):
        imports = []

        # ... generate imports from dependencies module
        pattern = 'from {module} import {dep}'

        module_name = self.dependencies_modname
        for dep in self.dependencies:
            txt = pattern.format(module=module_name, dep=dep.name)
            imports.append(txt)
        # ...

        # ...
        imports = '\n'.join(imports)
        # ...

        code = pycode(self.interface)

        return  '{imports}\n{code}'.format(imports=imports, code=code)

    def _compile(self, namespace):
        # ...
        module_name = self.dependencies_modname
        dependencies_module = importlib.import_module(module_name)
        # ...

        # ...
        code = self.interface_code
        name = self.interface.name

        exec(code, namespace)
        interface = namespace[name]
        # ...

        return interface

class BilinearForm(BasicForm):

    def __init__(self, expr, *args, **kwargs):
        if not isinstance(expr, sym_BilinearForm):
            raise TypeError('> Expecting a symbolic BilinearForm')

        BasicForm.__init__(self, expr)

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

        BasicForm.__init__(self, expr)

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

        BasicForm.__init__(self, expr)

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

def discretize_BilinearForm(expr, *args, **kwargs):
    form = BilinearForm(expr, *args, **kwargs)
    return form

def discretize_LinearForm(expr, *args, **kwargs):
    form = LinearForm(expr, *args, **kwargs)
    return form

def discretize_FunctionForm(expr, *args, **kwargs):
    form = FunctionForm(expr, *args, **kwargs)
    return form

def discretize(expr, *args, **kwargs):
    name = expr.__class__.__name__
    func = eval('discretize_{}'.format(name))
    return func(expr, *args, **kwargs)
