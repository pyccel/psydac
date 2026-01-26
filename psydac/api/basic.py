#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#

# TODO: - init_fem is called whenever we call discretize. we should check that
#         nderiv has not been changed. shall we add nquads too?

# TODO: avoid using os.system and use subprocess.call

import sys
import os
import importlib
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
    """ Basic class for any discrete concept that needs code generation.

    Parameters
    ----------

    folder: str
        The output folder where we generate the code.

    comm: MPI.Comm
        The mpi communicator used in the parallel case.

    root: int
        The process that is responsible of generating the code.

    discrete_space: FemSpace | list of FemSpace
        The discrete fem spaces.

    kernel_expr : sympde.expr.evaluation.KernelExpression
        The atomic representation of the bi-linear form.

    nquads: list of tuple
        The number of quadrature points used in the assembly method.

    is_rational_mapping : bool
        takes the value of True if the mapping is rational.

    mapping: Sympde.topology.Mapping
        The symbolic mapping of the bi-linear form domain.

    mapping_space: FemSpace
       The discete space of the mapping.

    num_threads: int
       Number of threads used in the computing kernels.
 
    backend: dict
        The backend used to accelerate the computing kernels.
        The content of the dictionary can be found in psydac/api/settings.py.

    """
    def __init__(self, expr, *, folder=None, comm=None, root=None, discrete_space=None,
                       kernel_expr=None, nquads=None, is_rational_mapping=None, mapping=None,
                       mapping_space=None, num_threads=None, backend=None):

        # Get default backend from environment, or use 'python'.
        default_backend = PSYDAC_BACKENDS.get(os.environ.get('PSYDAC_BACKEND'))\
                       or PSYDAC_BACKENDS['python']

        backend   = backend or default_backend
        # ...
        if not( comm is None) and comm.size>1:
            if root is None:
                root = 0

            assert isinstance( comm, MPI.Comm )
            assert isinstance( root, int      )

            if comm.rank == root:
                tag = random_string( 8 )
                ast = self._create_ast( expr=expr, tag=tag, comm=comm, discrete_space=discrete_space,
                           kernel_expr=kernel_expr, nquads=nquads, is_rational_mapping=is_rational_mapping,
                           mapping=mapping, mapping_space=mapping_space, num_threads=num_threads, backend=backend )

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

            tag        = comm.bcast(tag, root=root )
            func_name  = comm.bcast(func_name, root=root)
            max_nderiv = comm.bcast(max_nderiv, root=root )
            free_args  = comm.bcast(free_args, root=root)
            #user_functions = comm.bcast( user_functions, root=root )
        else:
            tag = random_string( 8 )
            ast = self._create_ast( expr=expr, tag=tag, discrete_space=discrete_space,
                       kernel_expr=kernel_expr, nquads=nquads, is_rational_mapping=is_rational_mapping,
                       mapping=mapping, mapping_space=mapping_space, num_threads=num_threads, backend=backend )

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

        # ... when using user defined functions, there must be passed as
        #     arguments of discretize. here we create a dictionary where the key
        #     is the function name, and the value is a valid implementation.
        # if user_functions:
        #     for f in user_functions:
        #         if not hasattr(f, '_imp_'):
        #             # TODO raise appropriate error message
        #             raise ValueError('can not find {} implementation'.format(f))

        if ast:
            python_code = self._generate_code()
            self._save_code(python_code, backend=self.backend['name'])

        if comm is not None and comm.size > 1:
            comm.Barrier()

        # compile code
        self._compile()

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

    def _generate_code(self):
        """
        Generate Python code which can be pyccelized.
        """
        psydac_ast = self.ast

        parser_settings = {
            'dim'    : psydac_ast.dim,
            'nderiv' : psydac_ast.nderiv,
            'mapping': psydac_ast.mapping,
            'target' : psydac_ast.domain
        }

        pyccel_ast  = parse(psydac_ast.expr, settings=parser_settings, backend=self.backend)
        python_code = pycode(pyccel_ast)

        return python_code

    def _save_code(self, code, backend=None):
        # ...
        write_code(self._dependencies_fname, code, folder = self.folder)

    def _compile_pythran(self, mod):
        raise NotImplementedError('Pythran is not available')

    def _compile_pyccel(self, mod, verbose=False):

        # ... convert python to fortran using pyccel
        compiler_family = self.backend['compiler_family']
        flags           = self.backend['flags']
        openmp          = self.backend["openmp"]
        _PYCCEL_FOLDER  = self.backend['folder']

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

    def _compile(self):

        module_name = self.dependencies_modname
        sys.path.append(self.folder)
        package = importlib.import_module( module_name )
        sys.path.remove(self.folder)

        if self.backend['name'] == 'pyccel':
            package = self._compile_pyccel(package)
        elif self.backend['name'] == 'pythran':
            package = self._compile_pythran(package)

        self._func = getattr(package, self._func_name)

#==============================================================================
class BasicDiscrete(BasicCodeGen):
    """ mapping is the symbolic mapping here.
    kwargs is used to pass user defined functions for the moment.
    """

    def __init__(self, expr, kernel_expr, *, folder=None, comm=None, root=None, discrete_space=None,
                       nquads=None, is_rational_mapping=None, mapping=None,
                       mapping_space=None, num_threads=None, backend=None):

        BasicCodeGen.__init__(self, expr, folder=folder, comm=comm, root=root, discrete_space=discrete_space,
                       kernel_expr=kernel_expr, nquads=nquads, is_rational_mapping=is_rational_mapping,
                       mapping=mapping, mapping_space=mapping_space, num_threads=num_threads, backend=backend)
        # ...
        self._kernel_expr = kernel_expr
        # ...

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

    def _create_ast(self, **kwargs):

        expr           = kwargs.pop('expr')
        kernel_expr    = kwargs.pop('kernel_expr')
        discrete_space = kwargs.pop('discrete_space')

        mapping_space  = kwargs.pop('mapping_space', None)
        tag            = kwargs.pop('tag', None)
        nquads         = kwargs.pop('nquads', None)
        mapping        = kwargs.pop('mapping', None)
        num_threads    = kwargs.pop('num_threads', None)
        backend        = kwargs.pop('backend', None)
        is_rational_mapping = kwargs.pop('is_rational_mapping', None)

        return AST(expr, kernel_expr, discrete_space, mapping_space=mapping_space,
                   tag=tag, nquads=nquads, mapping=mapping, is_rational_mapping=is_rational_mapping,
                   backend=backend, num_threads=num_threads)
