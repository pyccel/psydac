# coding: utf-8

# TODO for the moment we assume Product of same space

import numpy as np
from itertools import product
from scipy.linalg import eig as eig_solver

from gelato.expr     import GltExpr as sym_GltExpr

from psydac.api.ast.glt       import GltKernel
from psydac.api.ast.glt       import GltInterface
from psydac.api.settings      import PSYDAC_BACKEND_PYTHON, PSYDAC_DEFAULT_FOLDER
from psydac.api.grid          import CollocationBasisValues

from psydac.api.utilities     import mkdir_p, touch_init_file, random_string, write_code
from psydac.cad.geometry      import Geometry
from psydac.mapping.discrete  import SplineMapping, NurbsMapping

from psydac.fem.splines import SplineSpace
from psydac.fem.tensor  import TensorFemSpace
from psydac.fem.vector  import ProductFemSpace

from sympde.expr.basic            import BasicForm
from psydac.api.printing.pycode   import pycode

import inspect
import sys
import os
import importlib
import string
import random
from mpi4py import MPI

class GltBasicCodeGen(object):
    """ Basic class for any discrete concept that needs code generation """

    def __init__(self, expr, **kwargs):

        namespace = kwargs.pop('namespace', globals())
        backend   = kwargs.pop('backend', PSYDAC_BACKEND_PYTHON)
        folder    = kwargs.pop('folder', None)
        comm      = kwargs.pop('comm', None)
        root      = kwargs.pop('root', None)
        # ...
        if not( comm is None):
            if root is None:
                root = 0

            assert isinstance( comm, MPI.Comm )
            assert isinstance( root, int      )

            if comm.rank == root:
                tag = random_string( 8 )
                ast = self._create_ast( expr, tag, comm=comm, backend=backend, **kwargs )
                interface = ast['interface']
                max_nderiv = interface.max_nderiv
                in_arguments = [str(a) for a in interface.in_arguments]
                inout_arguments = [str(a) for a in interface.inout_arguments]
                user_functions = interface.user_functions

            else:
                interface = None
                tag = None
                max_nderiv = None
                in_arguments = None
                inout_arguments = None
                user_functions = None

            comm.Barrier()
            tag = comm.bcast( tag, root=root )
            max_nderiv = comm.bcast( max_nderiv, root=root )
            in_arguments = comm.bcast( in_arguments, root=root )
            inout_arguments = comm.bcast( inout_arguments, root=root )
            user_functions = comm.bcast( user_functions, root=root )

        else:
            tag = random_string( 8 )
            ast = self._create_ast( expr, tag, backend=backend, **kwargs )
            interface = ast['interface']
            max_nderiv = interface.max_nderiv
            interface_name = interface.name
            in_arguments = [str(a) for a in interface.in_arguments]
            inout_arguments = [str(a) for a in interface.inout_arguments]
            user_functions = interface.user_functions
        # ...

        # ...
        self._expr = expr
        self._tag = tag
        self._interface = interface
        self._in_arguments = in_arguments
        self._inout_arguments = inout_arguments
        self._user_functions = user_functions
        self._backend = backend
        self._folder = self._initialize_folder(folder)
        self._comm = comm
        self._root = root
        self._max_nderiv = max_nderiv

        self._dependencies = None
        self._dependencies_code = None
        self._dependencies_fname = None
        self._dependencies_modname = None

        interface_name = 'interface_{}'.format(tag)
        self._interface_name = interface_name
        self._interface_code = None
        self._interface_base_import_code = None
        self._func = None
        # ...

        # ... when using user defined functions, there must be passed as
        #     arguments of discretize. here we create a dictionary where the key
        #     is the function name, and the value is a valid implementation.
        if user_functions:
            for f in user_functions:
                if not hasattr(f, '_imp_'):
                    # TODO raise appropriate error message
                    raise ValueError('can not find {} implementation'.format(f))
        # ...

        # generate python code as strings for dependencies
        if not( interface is None ):
            self._dependencies = interface.dependencies
            self._dependencies_code = self._generate_code()

        if not( interface is None ):
            # save dependencies code
            self._save_code()

            if self.backend['name'] == 'pyccel':
                self._compile_pyccel(namespace)
            elif self.backend['name'] == 'pythran':
                self._compile_pythran(namespace)

            # generate code for Python interface
            self._generate_interface_code()

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
    def in_arguments(self):
        return self._in_arguments

    @property
    def inout_arguments(self):
        return self._inout_arguments

    @property
    def user_functions(self):
        return self._user_functions

    @property
    def interface(self):
        return self._interface

    @property
    def dependencies(self):
        return self._dependencies

    @property
    def interface_name(self):
        return self._interface_name

    @property
    def interface_code(self):
        return self._interface_code

    @property
    def interface_base_import_code(self):
        return self._interface_base_import_code

    @property
    def dependencies_code(self):
        return self._dependencies_code

    @property
    def dependencies_fname(self):
        return self._dependencies_fname

    @property
    def dependencies_modname(self):
        return self._dependencies_modname

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
            for root, _, _ in os.walk(folder):
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

        imports = '\n'.join(pycode(imp) for dep in self.dependencies for imp in dep.imports )

        code = '{code}\n{imports}'.format(code=code, imports=imports)

        # ... add user defined functions
        if self.user_functions:
            for func in self.user_functions:
                func_code = get_source_function(func._imp_)
                code = '{code}\n{func_code}'.format(code=code, func_code=func_code)
        # ...

        for dep in self.dependencies:
            code = '{code}\n{dep}'.format(code=code, dep=pycode(dep))
        # ...
        return code

    def _save_code(self):
        # ...
        code = self.dependencies_code
        module_name = 'dependencies_{}'.format(self.tag)

        self._dependencies_fname = '{}.py'.format(module_name)
        write_code(self.dependencies_fname, code, folder = self.folder)
        # ...

        # TODO check this? since we are using relative paths now
        self._dependencies_modname = module_name.replace('/', '.')

    def _generate_interface_code(self):
        imports = []

        module_name = self.dependencies_modname

        # ...
        if self.backend['name'] == 'pyccel':
            imports += [self.interface_base_import_code]

        else:
            # ... generate imports from dependencies module
            pattern = 'from {module} import {dep}'

            for dep in self.dependencies:
                txt = pattern.format(module=module_name, dep=dep.name)
                imports.append(txt)
            # ...
        # ...

        imports = '\n'.join(imports)

        code = pycode(self.interface)

        self._interface_code = '{imports}\n{code}'.format(imports=imports, code=code)

    def _compile_pythran(self, namespace):

        module_name = self.dependencies_modname

        os.chdir(self.folder)
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

        # ...
        basedir = os.getcwd()
        os.chdir(self.folder)
        # ...

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

        # ... TODO move to save
        code = self.interface_code
        interface_module_name = 'interface_{}'.format(self.tag)
        fname = '{}.py'.format(interface_module_name)
        fname = write_code(fname, code, folder = self.folder)
        # ...

        self._set_func(interface_module_name, self.interface_name)

    def _set_func(self, interface_module_name, interface_name):
        # ...
        sys.path.append(self.folder)
        package = importlib.import_module( interface_module_name )
        sys.path.remove(self.folder)
        # ...

        self._func = getattr(package, interface_name)

    def _check_arguments(self, **kwargs):

        # TODO do we need a method from Interface to map the dictionary of arguments
        # that are passed for the call (in the same spirit of build_arguments)
        # the idea is to be sure of their order, since they can be passed to
        # Fortran

        _kwargs = {}

        # ... mandatory arguments
        for key in self.in_arguments:
            try:
                _kwargs[key] = kwargs[key]
            except:
                raise KeyError('Unconsistent argument with interface')
        # ...

        # ... optional (inout) arguments
        for key in self.inout_arguments:
            try:
                _kwargs[key] = kwargs[key]
            except:
                pass
        # ...

        return _kwargs

#==============================================================================
class DiscreteGltExpr(GltBasicCodeGen):

    def __init__(self, expr, *args, **kwargs):
        if not isinstance(expr, sym_GltExpr):
            raise TypeError('> Expecting a symbolic Glt expression')

        if not args:
            raise ValueError('> fem spaces must be given as a list/tuple')

        assert( len(args) == 2 )

        # ...
        domain_h = args[0]
        assert( isinstance(domain_h, Geometry) )

        mapping = list(domain_h.mappings.values())[0]
        self._mapping = mapping

        is_rational_mapping = False
        if not( mapping is None ):
            is_rational_mapping = isinstance( mapping, NurbsMapping )

        self._is_rational_mapping = is_rational_mapping
        # ...

        # ...
        self._spaces = args[1]
        # ...

        # ...
        kwargs['domain']  = domain_h.domain
        kwargs['mapping'] = self.spaces[0].symbolic_mapping
        kwargs['is_rational_mapping'] = is_rational_mapping

        GltBasicCodeGen.__init__(self, expr, **kwargs)
        # ...

#        print('====================')
#        print(self.dependencies_code)
#        print('====================')
#        print(self.interface_code)
#        print('====================')
#        import sys; sys.exit(0)

    @property
    def mapping(self):
        return self._mapping

    @property
    def spaces(self):
        return self._spaces

    # TODO add comm and treate parallel case
    def _create_ast(self, expr, tag, **kwargs):

        domain              = kwargs.pop('domain', None)
        backend             = kwargs.pop('backend', PSYDAC_BACKEND_PYTHON)
        is_rational_mapping = kwargs.pop('is_rational_mapping', None)
        # ...
        kernel = GltKernel( expr, self.spaces,
                            name = 'kernel_{}'.format(tag),
                            domain = domain,
                            is_rational_mapping = is_rational_mapping,
                            backend = backend, **kwargs )

        interface = GltInterface( kernel,
                                  name = 'interface_{}'.format(tag),
                                  domain = domain,
                                  is_rational_mapping = is_rational_mapping,
                                  backend = backend , **kwargs)
        # ...

        ast = {'kernel': kernel, 'interface': interface}
        return ast


    def _check_arguments(self, **kwargs):

        # TODO do we need a method from Interface to map the dictionary of arguments
        # that are passed for the call (in the same spirit of build_arguments)
        # the idea is to be sure of their order, since they can be passed to
        # Fortran

        _kwargs = {}

        # ... mandatory arguments
        sym_args = self.interface.in_arguments
        keys = [str(a) for a in sym_args]
        for key in keys:
            try:
                # we use x1 for the call rather than arr_x1, to keep x1 inside
                # the loop
                if key == 'x1':
                    _kwargs['arr_x1'] = kwargs[key]

                elif key == 'x2':
                    _kwargs['arr_x2'] = kwargs[key]

                elif key == 'x3':
                    _kwargs['arr_x3'] = kwargs[key]

                else:
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

    def evaluate(self, *args, **kwargs):

        kwargs = self._check_arguments(**kwargs)

        Vh = self.spaces[0]
        is_block = False
        if isinstance(Vh, ProductFemSpace):
            Vh = Vh.spaces[0]
            is_block = True

        if not isinstance(Vh, TensorFemSpace):
            raise NotImplementedError('Only TensorFemSpace is available for the moment')

        args = args + (Vh,)

        dim = Vh.ldim

        if self.expr.form.fields or self.mapping:
            nderiv = self.interface.max_nderiv
            xis = [kwargs['arr_x{}'.format(i)] for i in range(1,dim+1)]
            grid = tuple(xis)
            # TODO assert that xis are inside the space domain
            basis_values = CollocationBasisValues(grid, Vh, nderiv=nderiv)
            args = args + (basis_values,)
        # ...

        if self.mapping:
            args = args + (self.mapping,)

        values = self.func(*args, **kwargs)

        if is_block:
            # n_rows = n_cols here
            n_rows = self.interface.n_rows
            n_cols = self.interface.n_cols
            nbasis = [V.nbasis for V in Vh.spaces]

            d = {}
            i = 0
            for i_row in range(0, n_rows):
                for i_col in range(0, n_cols):
                    d[i_row, i_col] = values[i]
                    i += 1

            eig_mat = np.zeros((n_rows,*nbasis))

            # ... compute dtype of the matrix
            dtype = 'float'
            are_complex = [i == 'complex' for i in self.interface.global_mats_types]
            if any(are_complex):
                dtype = 'complex'
            # ...

            mat = np.zeros((n_rows,n_cols), dtype=dtype)

            if dim == 2:
                for i1 in range(0, nbasis[0]):
                    for i2 in range(0, nbasis[1]):
                        mat[...] = 0.
                        for i_row in range(0,n_rows):
                            for i_col in range(0,n_cols):
                                mat[i_row,i_col] = d[i_row,i_col][i1, i2]
                        w,v = eig_solver(mat)
                        wr = w.real
                        eig_mat[:,i1,i2] = wr[:]

            elif dim == 3:
                for i1 in range(0, nbasis[0]):
                    for i2 in range(0, nbasis[1]):
                        for i3 in range(0, nbasis[2]):
                            mat[...] = 0.
                            for i_row in range(0,n_rows):
                                for i_col in range(0,n_cols):
                                    mat[i_row,i_col] = d[i_row,i_col][i1, i2, i3]
                            w,v = eig_solver(mat)
                            wr = w.real
                            eig_mat[:,i1,i2,i3] = wr[:]

            else:
                raise NotImplementedError('')

            values = eig_mat

        return values

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def eig(self, **kwargs):
        """
        Approximates the eigenvalues of the matrix associated to the given
        bilinear form.
        the current algorithm is based on a uniform sampling of the glt symbol.
        """
        Vh = self.spaces[0]
        if isinstance(Vh, ProductFemSpace):
            Vh = Vh.spaces[0]

        if not isinstance(Vh, TensorFemSpace):
            raise NotImplementedError('Only TensorFemSpace is available for the moment')

        nbasis = [V.nbasis for V in Vh.spaces]
        bounds = [V.domain for V in Vh.spaces]
        dim    = Vh.ldim

        # ... fourier variables (as arguments)
        ts = [np.linspace(-np.pi, np.pi, n) for n in nbasis]
        args = tuple(ts)
        # ...

        # ... space variables (as key words)
        if self.interface.with_coordinates:
            xs = [np.linspace(bound[0], bound[1], n) for n, bound in zip(nbasis, bounds)]
            for n,x in zip(['x1', 'x2', 'x3'][:dim], xs):
                kwargs[n] = x
        # ...

        values = self(*args, **kwargs)

        return values
