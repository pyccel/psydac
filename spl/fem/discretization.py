# coding: utf-8
from sympde.core import BilinearForm as sym_BilinearForm
from sympde.core import LinearForm as sym_LinearForm
from sympde.core import FunctionForm as sym_FunctionForm

from spl.codegen.ast import Interface
from spl.codegen.printing import pycode
from spl.codegen.utils import write_code

import os
import importlib

class BasicForm(object):

    def __init__(self, expr, namespace=globals()):
        self._expr = expr
        self._interface = Interface(expr)
        self._dependencies = self.interface.dependencies

        self._interface_code = None
        self._dependencies_code = None

        # generate python code as strings for dependencies
        self._generate_code()

        # save dependencies code
        self._save_code()

        # ... generate code for Python interface
        self._generate_interface_code()
        # ...

        # compile code
        self._compile(namespace)

    @property
    def expr(self):
        return self._expr

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

    def _generate_code(self):
        # ... generate code that can be pyccelized
        code = ''
        for dep in self.dependencies:
            code = '{code}\n{dep}'.format(code=code, dep=pycode(dep))
        self._dependencies_code = code
        # ...

    def _save_code(self):
        folder = 'tmp'

        # ... save dependencies
        code = self.dependencies_code
        ID = abs(hash(self))
        name = 'dependencies_{}'.format(ID)
        fname = write_code(name, code, ext='py', folder=folder)
        self._dependencies_fname = fname
        # ...

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

        self._interface_code = '{imports}\n{code}'.format(imports=imports,
                                                          code=code)

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
        print(interface)
        # ...

class BilinearForm(BasicForm):

    def __init__(self, expr):
        if not isinstance(expr, sym_BilinearForm):
            raise TypeError('> Expecting a symbolic BilinearForm')

        BasicForm.__init__(self, expr)

class LinearForm(BasicForm):

    def __init__(self, expr):
        if not isinstance(expr, sym_LinearForm):
            raise TypeError('> Expecting a symbolic LinearForm')

        BasicForm.__init__(self, expr)

class FunctionForm(BasicForm):

    def __init__(self, expr):
        if not isinstance(expr, sym_FunctionForm):
            raise TypeError('> Expecting a symbolic FunctionForm')

        BasicForm.__init__(self, expr)

def discretize_BilinearForm(expr, *args, **kwargs):
#    print('> Enter discretize_BilinearForm')
    form = BilinearForm(expr)
    return form

def discretize_LinearForm(expr, *args, **kwargs):
#    print('> Enter discretize_LinearForm')
    form = LinearForm(expr)
    return form

def discretize_FunctionForm(expr, *args, **kwargs):
#    print('> Enter discretize_FunctionForm')
    form = FunctionForm(expr)
    return form

def discretize(expr, *args, **kwargs):
    name = expr.__class__.__name__
    func = eval('discretize_{}'.format(name))
    return func(expr, *args, **kwargs)
