# coding: utf-8

# TODO: - init_fem is called whenever we call discretize. we should check that
#         nderiv has not been changed. shall we add quad_order too?

# TODO: avoid using os.system and use subprocess.call

import sys
import os
import importlib
import numpy as np
from mpi4py import MPI

from psydac.api.ast.fem         import AST
from psydac.api.ast.parser      import parse
from psydac.api.printing.pycode import pycode
from psydac.api.settings        import PSYDAC_BACKENDS, PSYDAC_DEFAULT_FOLDER
from psydac.api.utilities       import mkdir_p, touch_init_file, random_string, write_code

__all__ = ('BasicCodeGen', 'BasicDiscrete')

#==============================================================================
# TODO have it as abstract class
class BasicCodeGen:
    """ Basic class for any discrete concept that needs code generation """

    def __init__(self, expr, **kwargs):

        # Get default backend from environment, or use 'python'.
        default_backend = PSYDAC_BACKENDS.get(os.environ.get('PSYDAC_BACKEND'))\
                       or PSYDAC_BACKENDS['python']

        namespace = kwargs.pop('namespace', globals())
        backend   = kwargs.pop('backend', None) or default_backend
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
                max_nderiv    = np.array(ast.nderiv)
                func_name     = ast.expr.name.encode()
                arguments     = ast.expr.arguments.copy()
                free_args     = arguments.pop('fields', ()) +  arguments.pop('constants', ())
                free_args     = np.char.array(tuple(str(i).encode() for i in free_args))
                tag           = tag.encode()
                num_free_args = np.array([len(free_args), int(free_args.itemsize)])
            else:
                tag           = bytearray(256)
                max_nderiv    = np.array(0)
                func_name     = bytearray(256)
                num_free_args = np.array([0,0])
                ast           = None

            req1 = comm.Ibcast((num_free_args, MPI.INT), root=root)
            req2 = comm.Ibcast((tag, MPI.CHAR) , root=root )
            req3 = comm.Ibcast((func_name, MPI.CHAR), root=root)
            req4 = comm.Ibcast((max_nderiv, MPI.INT), root=root )
            MPI.Request.Wait(req1)

            if comm.rank != root:
                free_args = np.chararray((num_free_args[0],), itemsize=num_free_args[1])

            req1 = comm.Ibcast((free_args, num_free_args[0], MPI.CHAR),  root=root)
            MPI.Request.Waitall([req1, req2, req3, req4])

            tag        =  str(np.array(tag.decode(), dtype=np.str))
            max_nderiv = int(max_nderiv)
            func_name  = str(np.array(func_name.decode(), dtype=np.str))
            free_args  = tuple(free_args.astype(str).tolist())
            #user_functions = comm.bcast( user_functions, root=root )
        else:
            tag = random_string( 8 )
            ast = self._create_ast( expr, tag, backend=backend, **kwargs )
            max_nderiv = ast.nderiv
            func_name = ast.expr.name
            arguments = ast.expr.arguments.copy()
            free_args = arguments.pop('fields', ()) +  arguments.pop('constants', ())
            free_args = tuple(str(i) for i in free_args)

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

        if comm is not None:comm.Barrier()
        # compile code
        self._compile(namespace)

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
            code = 'from numba import njit'

        ast = self.ast
        expr = parse(ast.expr, settings={'dim': ast.dim, 'nderiv': ast.nderiv, 'mapping':ast.mapping, 'target':ast.domain}, backend=self.backend)

        code = '{code}\n{dep}'.format(code=code, dep=pycode(expr))

        return code

    def _save_code(self, code, backend=None):
        # ...
        write_code(self._dependencies_fname, code, folder = self.folder)

    def _compile_pythran(self, namespace, mod):
        raise NotImplementedError('Pythran is not available')

    def _compile_pyccel(self, namespace, mod, verbose=False):

        # ... convert python to fortran using pyccel
        compiler       = self.backend['compiler']
        fflags         = self.backend['flags']
        accelerators   = ["openmp"] if self.backend["openmp"] else []
        _PYCCEL_FOLDER = self.backend['folder']

        from pyccel.epyccel import epyccel
        fmod = epyccel(mod,
                       accelerators = accelerators,
                       compiler    = compiler,
                       fflags      = fflags,
                       comm        = self.comm,
                       bcast       = True,
                       folder      = _PYCCEL_FOLDER,
                       verbose     = verbose)

        return fmod

    def _compile(self, namespace):

        module_name = self.dependencies_modname
        sys.path.append(self.folder)
        package = importlib.import_module( module_name )
        sys.path.remove(self.folder)

        if self.backend['name'] == 'pyccel':
            package = self._compile_pyccel(namespace, package)
        elif self.backend['name'] == 'pythran':
            package = self._compile_pythran(namespace, package)

        self._func = getattr(package, self._func_name)

#==============================================================================
class BasicDiscrete(BasicCodeGen):
    """ mapping is the symbolic mapping here.
    kwargs is used to pass user defined functions for the moment.
    """

    def __init__(self, expr, kernel_expr, **kwargs):

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
        quad_order          = kwargs.pop('quad_order', None)
        is_rational_mapping = kwargs.pop('is_rational_mapping', None)
        mapping             = kwargs.pop('mapping', None)
        mapping_space       = kwargs.pop('mapping_space', None)
        num_threads         = kwargs.pop('num_threads', 1)
        backend             = kwargs.pop('backend', None)

        return AST(expr, kernel_expr, discrete_space, mapping_space=mapping_space, tag=tag, quad_order=quad_order,
                    mapping=mapping, is_rational_mapping=is_rational_mapping, backend=backend, num_threads=num_threads)

