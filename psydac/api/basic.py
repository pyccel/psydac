# coding: utf-8

# TODO: - init_fem is called whenever we call discretize. we should check that
#         nderiv has not been changed. shall we add quad_order too?

# TODO: avoid using os.system and use subprocess.call

from collections import OrderedDict
from collections import namedtuple

from pyccel.ast import Nil
from pyccel.epyccel import get_source_function

from sympde.topology import Domain, Boundary


from psydac.api.ast.fem              import Kernel
from psydac.api.ast.fem              import Assembly
from psydac.api.ast.fem              import Interface
from psydac.api.ast.glt              import GltKernel
from psydac.api.ast.glt              import GltInterface

from psydac.api.new_ast.fem  import AST
from psydac.api.new_ast.parser import parse

from psydac.api.printing.pycode      import pycode
from psydac.api.essential_bc         import apply_essential_bc
from psydac.api.settings             import PSYDAC_BACKEND_PYTHON, PSYDAC_DEFAULT_FOLDER
from psydac.linalg.stencil           import StencilVector, StencilMatrix
from psydac.linalg.iterative_solvers import cg
from psydac.fem.splines              import SplineSpace
from psydac.fem.tensor               import TensorFemSpace
from psydac.fem.vector               import ProductFemSpace
from psydac.cad.geometry             import Geometry
from psydac.mapping.discrete         import SplineMapping, NurbsMapping

from sympde.expr.basic import BasicForm
from sympde.expr       import Functional
from sympde.topology.space import ScalarField, VectorField, IndexedVectorField
from gelato.expr       import GltExpr
from sympy import Add, Mul

import inspect
import sys
import os
import importlib
import string
import random
import numpy as np
from mpi4py import MPI


#==============================================================================
def mkdir_p(folder):
    if os.path.isdir(folder):
        return
    os.makedirs(folder, exist_ok=True)

#==============================================================================
def touch_init_file(path):
    mkdir_p(path)
    path = os.path.join(path, '__init__.py')
    with open(path, 'a'):
        os.utime(path, None)

#==============================================================================
def random_string( n ):
    # we remove uppercase letters because of f2py
    chars    = string.ascii_lowercase + string.digits
    selector = random.SystemRandom()
    return ''.join( selector.choice( chars ) for _ in range( n ) )

#==============================================================================
def write_code(filename, code, folder=None):
    if not folder:
        folder = os.getcwd()

    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise ValueError('{} folder does not exist'.format(folder))

    filename = os.path.basename( filename )
    filename = os.path.join(folder, filename)

    # TODO check if init exists
    # add __init__.py for imports
    touch_init_file(folder)

    f = open(filename, 'w')
    for line in code:
        f.write(line)
    f.close()

    return filename

#==============================================================================
# TODO have it as abstract class
class BasicCodeGen(object):
    """ Basic class for any discrete concept that needs code generation """

    def __init__(self, expr, **kwargs):

        namespace = kwargs.pop('namespace', globals())
        backend   = kwargs.pop('backend', PSYDAC_BACKEND_PYTHON)
        folder    = kwargs.pop('folder', None)
        comm      = kwargs.pop('comm', None)
        root      = kwargs.pop('root', None)
        # ...
        if not( comm is None):
            if root is None:
                root = 0

            assert isinstance( comm, MPI.Comm )
            assert isinstance( root, int      )

            if comm.rank == root:
                tag = random_string( 8 )
                ast = self._create_ast( expr, tag, comm=comm, backend=backend, **kwargs )
                max_nderiv = ast.nderiv

            else:
                ast = None

            comm.Barrier()
            tag = comm.bcast( tag, root=root )
            max_nderiv = comm.bcast( max_nderiv, root=root )
            #user_functions = comm.bcast( user_functions, root=root )
        else:
            tag = random_string( 8 )
            ast = self._create_ast( expr, tag, backend=backend, **kwargs )
            max_nderiv = ast.nderiv
        # ...
        user_functions = None
        self._expr = expr
        self._tag = tag
        self._ast = ast
        self._user_functions = user_functions
        self._backend = backend
        self._folder = self._initialize_folder(folder)
        self._comm = comm
        self._root = root
        self._max_nderiv = max_nderiv

        self._code = None
        self._func = None
        self._dependencies_fname   = None
        self._dependencies_modname = None
        # ...

        # ... when using user defined functions, there must be passed as
        #     arguments of discretize. here we create a dictionary where the key
        #     is the function name, and the value is a valid implementation.
        # if user_functions:
        #     for f in user_functions:
        #         if not hasattr(f, '_imp_'):
        #             # TODO raise appropriate error message
        #             raise ValueError('can not find {} implementation'.format(f))

        if ast:
            self._code = self._generate_code()
            self._save_code()

            if self.backend['name'] == 'pyccel':
                self._compile_pyccel(namespace)
            elif self.backend['name'] == 'pythran':
                self._compile_pythran(namespace)

            # compile code
            self._compile(namespace)

        if not( comm is None):
            comm.Barrier()
            if comm.rank != root:
                if self.backend['name'] == 'pyccel':
                    _folder = os.path.join(self.folder, self.backend['folder'])
                    sys.path.append(_folder)

                interface_module_name = interface_name
                self._set_func(interface_module_name, interface_name)

                if self.backend['name'] == 'pyccel':
                    _folder = os.path.join(self.folder, self.backend['folder'])
                    sys.path.remove(_folder)

            comm.Barrier()


    @property
    def expr(self):
        return self._expr

    @property
    def tag(self):
        return self._tag

    @property
    def user_functions(self):
        return self._user_functions

    @property
    def ast(self):
        return self._ast

    @property
    def code(self):
        return self._interface_code

    @property
    def func(self):
        return self._func

    @property
    def backend(self):
        return self._backend

    @property
    def comm(self):
        return self._comm

    @property
    def root(self):
        return self._root

    @property
    def folder(self):
        return self._folder

    @property
    def dependencies_fname(self):
        return self._dependencies_fname

    @property
    def dependencies_modname(self):
        return self._dependencies_modname

    def _create_ast(self, **kwargs):
        raise NotImplementedError('Must be implemented')

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

    def _generate_code(self):
        # ... generate code that can be pyccelized
        code = ''

        if self.backend['name'] == 'pyccel':

            code += '\nfrom pyccel.decorators import types'
            code += '\nfrom pyccel.decorators import external, external_call'

        elif self.backend['name'] == 'numba':
            code = 'from numba import jit'

        ast = self.ast
        expr = parse(ast.expr, settings={'dim': ast.dim, 'nderiv': ast.nderiv, 'mapping':ast.mapping, 'target':ast.domain})

        code = '{code}\n{dep}'.format(code=code, dep=pycode(expr))

        return code

    def _save_code(self):
        # ...
        code = self._code
        module_name = 'dependencies_{}'.format(self.tag)

        #module_name = 'dependencies_gv1ocbgr'
        self._dependencies_fname = '{}.py'.format(module_name)
        write_code(self._dependencies_fname, code, folder = self.folder)
        # ...

        # TODO check this? since we are using relative paths now
        self._dependencies_modname = module_name.replace('/', '.')

    def _compile_pythran(self, namespace):

        module_name = self.dependencies_modname

        basedir = os.getcwd()
        os.chdir(self.folder)
        curdir = os.getcwd()
        sys.path.append(self.folder)
        os.system('pythran {}.py -O3'.format(module_name))
        sys.path.remove(self.folder)

        # ...
    def _compile_pyccel(self, namespace, verbose=False):

        module_name = self.dependencies_modname

        # ...
        from pyccel.epyccel import epyccel

        # ... convert python to fortran using pyccel
        compiler       = self.backend['compiler']
        fflags         = self.backend['flags']
        accelerator    = self.backend['accelerator']
        _PYCCEL_FOLDER = self.backend['folder']
        # ...

        basedir = os.getcwd()
        os.chdir(self.folder)
        curdir = os.getcwd()

        # ...
        sys.path.append(self.folder)
        package = importlib.import_module( module_name )
        f2py_module = epyccel( package,
                               compiler    = compiler,
                               fflags      = fflags,
                               accelerator = accelerator,
                               comm        = self.comm,
                               bcast       = False,
                               folder      = _PYCCEL_FOLDER )
        sys.path.remove(self.folder)
        # ...

        # ... get list of all functions inside the f2py module
        functions = []
        for name, obj in inspect.getmembers(f2py_module):
            if callable(obj) and not( name.startswith( 'f2py_' ) ):
                functions.append(name)
        # ...

        # update module name for dependencies
        # needed for interface when importing assembly
        name = os.path.join(_PYCCEL_FOLDER, f2py_module.__name__)
        name = name.replace('/', '.')
        imports = []
        for f in functions:
            pattern = 'from {name} import {f}'
            stmt = pattern.format( name = name, f = f )
            imports.append(stmt)
        imports = '\n'.join(i for i in imports)

        self._interface_base_import_code = imports
        # ...

        os.chdir(basedir)

    def _compile(self, namespace):

        module_name = self.dependencies_modname
        self._set_func(module_name, self.ast.expr.name)

    def _set_func(self, module_name, name):
        # ...
        sys.path.append(self.folder)
        package = importlib.import_module( module_name )
        sys.path.remove(self.folder)
        # ...

        self._func = getattr(package, name)

#==============================================================================
class BasicDiscrete(BasicCodeGen):
    """ mapping is the symbolic mapping here.
    kwargs is used to pass user defined functions for the moment.
    """

    def __init__(self, expr, kernel_expr, **kwargs):

        if isinstance(kernel_expr, list):
            if len(kernel_expr) == 1:
                kernel_expr = kernel_expr[0]
            else:
                raise ValueError('> Expecting only one kernel')

        kwargs['kernel_expr'] = kernel_expr

        BasicCodeGen.__init__(self, expr, **kwargs)
        # ...
        self._kernel_expr = kernel_expr
        # ...

    @property
    def kernel_expr(self):
        return self._kernel_expr

    @property
    def target(self):
        return self._target

    @property
    def mapping(self):
        return self._mapping

    @property
    def is_rational_mapping(self):
        return self._is_rational_mapping

    @property
    def max_nderiv(self):
        return self._max_nderiv

    def _create_ast(self, expr,tag, **kwargs):
        is_rational_mapping = kwargs.pop('is_rational_mapping', None)
        discrete_space      = kwargs.pop('discrete_space', None)
        comm                = kwargs.pop('comm', None)
        kernel_expr         = kwargs['kernel_expr']

        return AST(expr, kernel_expr, discrete_space, is_rational_mapping, tag)
