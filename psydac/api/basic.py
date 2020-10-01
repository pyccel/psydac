# coding: utf-8

# TODO: - init_fem is called whenever we call discretize. we should check that
#         nderiv has not been changed. shall we add quad_order too?

# TODO: avoid using os.system and use subprocess.call

from collections import OrderedDict
from collections import namedtuple

from pyccel.ast.core import Nil

from sympde.topology import Domain

from psydac.api.ast.fem  import AST
from psydac.api.ast.parser import parse

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
from psydac.api.utilities            import mkdir_p, touch_init_file, random_string, write_code

from sympde.topology.space import ScalarField, VectorField, IndexedVectorField
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
                func_name = ast.expr.name
                arguments = ast.expr.arguments.copy()
                free_args = arguments.pop('fields', ()) +  arguments.pop('constants', ())
                free_args = tuple(str(i) for i in free_args)

            else:
                tag = None
                ast = None
                max_nderiv = None
                func_name  = None
                free_args  = None

            comm.Barrier()
            tag = comm.bcast( tag, root=root )
            max_nderiv = comm.bcast( max_nderiv, root=root )
            func_name  = comm.bcast(func_name, root=root)
            free_args  = comm.bcast(free_args, root=root)
            #user_functions = comm.bcast( user_functions, root=root )
        else:
            tag = random_string( 8 )
            ast = self._create_ast( expr, tag, backend=backend, **kwargs )
            max_nderiv = ast.nderiv
            func_name = ast.expr.name
            arguments = ast.expr.arguments.copy()
            free_args = arguments.pop('fields', ()) +  arguments.pop('constants', ())
            free_args = tuple(str(i) for i in free_args)
        # ...
        user_functions = None
        self._expr = expr
        self._tag = tag
        self._ast = ast
        self._func_name = func_name
        self._free_args = free_args
        self._user_functions = user_functions
        self._backend = backend
        self._folder = self._initialize_folder(folder)
        self._comm = comm
        self._root = root
        self._max_nderiv = max_nderiv
        self._code = None
        self._func = None
        self._dependencies_modname = 'dependencies_{}'.format(self.tag)
        self._dependencies_fname   = '{}.py'.format(self._dependencies_modname)
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
            self._save_code(self._generate_code(), backend=self.backend['name'])
            # compile code
            self._compile(namespace)

            if self.backend['name'] == 'pyccel':
                self._compile_pyccel(namespace)
            elif self.backend['name'] == 'pythran':
                self._compile_pythran(namespace)

        if not( comm is None):
            comm.Barrier()
            if comm.rank != root:
                if self.backend['name'] == 'pyccel':
                    _folder = os.path.join(self.folder, self.backend['folder'])
                    sys.path.append(_folder)

                self._set_func(self._dependencies_modname, self._func_name)

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
    def free_args(self):
        return self._free_args

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
            code = 'from pyccel.decorators import types'
        elif self.backend['name'] == 'numba':
            code = 'from numba import jit'

        ast = self.ast
        expr = parse(ast.expr, settings={'dim': ast.dim, 'nderiv': ast.nderiv, 'mapping':ast.mapping, 'target':ast.domain}, backend=self.backend)

        code = '{code}\n{dep}'.format(code=code, dep=pycode(expr))

        return code

    def _save_code(self, code, backend=None):
        # ...
        write_code(self._dependencies_fname, code, folder = self.folder)

    def _compile_pythran(self, namespace):
        raise NotImplementedError('Pythran is not available')

    def _compile_pyccel(self, namespace, verbose=False):

        module_name = self.dependencies_modname
        # ... convert python to fortran using pyccel
        compiler       = self.backend['compiler']
        fflags         = self.backend['flags']
        accelerator    = self.backend['accelerator']
        _PYCCEL_FOLDER = self.backend['folder']

        from pyccel.epyccel import epyccel

        func = epyccel(self._func,
                       compiler    = compiler,
                       fflags      = fflags,
                       accelerator = accelerator,
                       comm        = self.comm,
                       bcast       = False)
        self._func = func

        # ...

    def _compile(self, namespace):

        module_name = self.dependencies_modname

        sys.path.append(self.folder)
        package = importlib.import_module( module_name )
        sys.path.remove(self.folder)

        self._func = getattr(package, self._func_name)
#==============================================================================
class BasicDiscrete(BasicCodeGen):
    """ mapping is the symbolic mapping here.
    kwargs is used to pass user defined functions for the moment.
    """

    def __init__(self, expr, kernel_expr, **kwargs):

        if isinstance(kernel_expr, (tuple, list)):
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
        discrete_space      = kwargs.pop('discrete_space', None)
        kernel_expr         = kwargs['kernel_expr']

        return AST(expr, kernel_expr, discrete_space, tag)
