# coding: utf-8

# TODO: - init_fem is called whenever we call discretize. we should check that
#         nderiv has not been changed. shall we add quad_order too?

# TODO: avoid using os.system and use subprocess.call

from collections import OrderedDict
from collections import namedtuple

from pyccel.ast import Nil

from sympde.expr     import BasicForm as sym_BasicForm
from sympde.expr     import BilinearForm as sym_BilinearForm
from sympde.expr     import LinearForm as sym_LinearForm
from sympde.expr     import Integral as sym_Integral
from sympde.expr     import Equation as sym_Equation
from sympde.expr     import Model as sym_Model
from sympde.expr     import Boundary as sym_Boundary
from sympde.expr     import Norm as sym_Norm
from sympde.expr     import DirichletBC
from sympde.expr     import evaluate
from sympde.topology import Domain, Boundary
from sympde.topology import Line, Square, Cube
from sympde.topology import BasicFunctionSpace
from sympde.topology import FunctionSpace, VectorFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import Mapping

from spl.api.grid                 import QuadratureGrid, BoundaryQuadratureGrid
from spl.api.grid                 import BasisValues
from spl.api.codegen.ast          import Kernel
from spl.api.codegen.ast          import Assembly
from spl.api.codegen.ast          import Interface
from spl.api.codegen.printing     import pycode
from spl.api.boundary_condition   import DiscreteBoundary
from spl.api.boundary_condition   import DiscreteComplementBoundary
from spl.api.boundary_condition   import DiscreteBoundaryCondition, DiscreteDirichletBC
from spl.api.boundary_condition   import apply_homogeneous_dirichlet_bc
from spl.api.settings             import SPL_BACKEND_PYTHON, SPL_DEFAULT_FOLDER
from spl.linalg.stencil           import StencilVector, StencilMatrix
from spl.linalg.iterative_solvers import cg
from spl.fem.splines              import SplineSpace
from spl.fem.tensor               import TensorFemSpace
from spl.fem.vector               import ProductFemSpace
from spl.cad.geometry             import Geometry
from spl.mapping.discrete         import SplineMapping, NurbsMapping

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
    os.makedirs(folder)

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
LinearSystem = namedtuple('LinearSystem', ['lhs', 'rhs'])

#==============================================================================
_default_solver = {'solver':'cg', 'tol':1e-9, 'maxiter':1000, 'verbose':False}

def driver_solve(L, **kwargs):
    if not isinstance(L, LinearSystem):
        raise TypeError('> Expecting a LinearSystem object')

    M = L.lhs
    rhs = L.rhs

    name = kwargs.pop('solver')
    return_info = kwargs.pop('info', False)

    if name == 'cg':
        x, info = cg( M, rhs, **kwargs )
        if return_info:
            return x, info
        else:
            return x
    else:
        raise NotImplementedError('Only cg solver is available')

#==============================================================================
class BasicDiscrete(object):
    """ mapping is the symbolic mapping here."""

    def __init__(self, a, kernel_expr, namespace=globals(),
                 boundary=None, target=None,
                 boundary_basis=None, backend=SPL_BACKEND_PYTHON, folder=None,
                 discrete_space=None, comm=None, root=None, mapping=None,
                 is_rational_mapping=None):

        # ...
        if not target:
            if len(kernel_expr) > 1:
                raise ValueError('> Expecting only one kernel')

            target = kernel_expr[0].target
        # ...

        # ...
        if boundary:
            if not isinstance(boundary, (tuple, list, Boundary)):
                raise TypeError('> Expecting a tuple, list or Boundary')

            if isinstance(boundary, Boundary):
                if not( boundary is target ):
                    raise ValueError('> Unconsistent boundary with symbolic model')

                boundary = [boundary.axis, boundary.ext]
                boundary = [boundary]
                boundary_basis = True # TODO set it to False for Nitch method

            # boundary is now a list of boundaries
            # TODO shall we keep it this way? since this is the simplest
            # interface to be able to compute Integral on a curve in 3d
        # ...

        # ...
        def _create_ast(tag):
            kernel = Kernel( a, kernel_expr,
                             name                = 'kernel_{}'.format(tag),
                             target              = target,
                             mapping             = mapping,
                             is_rational_mapping = is_rational_mapping,
                             discrete_boundary   = boundary,
                             boundary_basis      = boundary_basis,
                             backend = backend )

            assembly = Assembly( kernel,
                                 name           = 'assembly_{}'.format(tag),
                                 mapping        = mapping,
                                 is_rational_mapping = is_rational_mapping,
                                 discrete_space = discrete_space,
                                 comm           = comm,
                                 backend = backend )

            interface = Interface( assembly,
                                   name                = 'interface_{}'.format(tag),
                                   mapping             = mapping,
                                   is_rational_mapping = is_rational_mapping,
                                   backend             = backend,
                                   discrete_space      = discrete_space,
                                   comm                = comm)

            return kernel, assembly, interface
        # ...

        # ...
        if not( comm is None):
            if root is None:
                root = 0

            assert isinstance( comm, MPI.Comm )
            assert isinstance( root, int      )

            if comm.rank == root:
                tag = random_string( 8 )
                kernel, assembly, interface = _create_ast( tag )
                max_nderiv = interface.max_nderiv
                in_arguments = [str(a) for a in interface.in_arguments]
                inout_arguments = [str(a) for a in interface.inout_arguments]

            else:
                kernel = None
                assembly = None
                interface = None
                tag = None
                max_nderiv = None
                in_arguments = None
                inout_arguments = None

            comm.Barrier()
            tag = comm.bcast( tag, root=root )
            max_nderiv = comm.bcast( max_nderiv, root=root )
            in_arguments = comm.bcast( in_arguments, root=root )
            inout_arguments = comm.bcast( inout_arguments, root=root )

        else:
            tag = random_string( 8 )
            kernel, assembly, interface = _create_ast( tag )
            max_nderiv = interface.max_nderiv
            interface_name = interface.name
            in_arguments = [str(a) for a in interface.in_arguments]
            inout_arguments = [str(a) for a in interface.inout_arguments]
        # ...

        # ...
        self._expr = a
        self._kernel_expr = kernel_expr
        self._target = target
        self._tag = tag
        self._interface = interface
        self._in_arguments = in_arguments
        self._inout_arguments = inout_arguments
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
        # ...

        # generate python code as strings for dependencies
        if not( interface is None ):
            self._dependencies = interface.dependencies
            self._dependencies_code = self._generate_code()
            

        if not( interface is None ):
            # save dependencies code
            self._save_code()

            if self.backend['name'] == 'pyccel':
                self._compile_pyccel(namespace)

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
    def kernel_expr(self):
        return self._kernel_expr

    @property
    def target(self):
        return self._target

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
    def mapping(self):
        return self._mapping

    @property
    def is_rational_mapping(self):
        return self._is_rational_mapping

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

    @property
    def max_nderiv(self):
        return self._max_nderiv

    def _initialize_folder(self, folder=None):
        # ...
        if folder is None:
            basedir = os.getcwd()
            folder = SPL_DEFAULT_FOLDER
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
        
        if self.backend['name'] == 'pyccel':
            code = 'from pyccel.decorators import types'
            code = '{code}\nfrom pyccel.decorators import external, external_call'.format( code = code )
            
        elif self.backend['name'] == 'numba':
            code = 'from numba import jit'
        else:
            code = ''

        imports = '\n'.join(pycode(imp) for dep in self.dependencies for imp in dep.imports )
        
        code = '{code}\n{imports}'.format(code=code, imports=imports)

        for dep in self.dependencies:
            code = '{code}\n{dep}'.format(code=code, dep=pycode(dep))
        # ...
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
        prefix = ''

        module_name = self.dependencies_modname

        # ...
        if self.backend['name'] == 'pyccel':
            imports += [self.interface_base_import_code]

        else:
            # ... generate imports from dependencies module
            pattern = 'from {module} import {dep}'

            for dep in self.dependencies:
                txt = pattern.format(module=module_name, dep=dep.name)
                imports.append(txt)
            # ...
        # ...

        imports = '\n'.join(imports)

        code = pycode(self.interface)

        self._interface_code = '{imports}\n{code}'.format(imports=imports, code=code)

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
        curdir = os.getcwd()
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

        module_name = self.dependencies_modname

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
class DiscreteBilinearForm(BasicDiscrete):

    def __init__(self, expr, kernel_expr, *args, **kwargs):
        if not isinstance(expr, sym_BilinearForm):
            raise TypeError('> Expecting a symbolic BilinearForm')

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

        kwargs['discrete_space']      = self.spaces
        kwargs['mapping']             = self.spaces[0].symbolic_mapping
        kwargs['is_rational_mapping'] = is_rational_mapping
        kwargs['comm']                = domain_h.comm

        BasicDiscrete.__init__(self, expr, kernel_expr, **kwargs)

        # ...
        test_space  = self.spaces[0]
        trial_space = self.spaces[1]
        # ...

        # ...
        quad_order = kwargs.pop('quad_order', None)
        boundary   = kwargs.pop('boundary',   None)
        # ...

        # ...
        # TODO must check that spaces lead to the same QuadratureGrid
        if boundary is None:
            self._grid = QuadratureGrid( test_space, quad_order = quad_order )

        else:

            self._grid = BoundaryQuadratureGrid( test_space,
                                                 boundary.axis,
                                                 boundary.ext,
                                                 quad_order = quad_order )
        # ...

        # ...
        self._test_basis = BasisValues( test_space, self.grid,
                                        nderiv = self.max_nderiv )

        self._trial_basis = BasisValues( trial_space, self.grid,
                                         nderiv = self.max_nderiv )
        # ...


    @property
    def spaces(self):
        return self._spaces

    @property
    def grid(self):
        return self._grid

    @property
    def test_basis(self):
        return self._test_basis

    @property
    def trial_basis(self):
        return self._trial_basis

    def assemble(self, **kwargs):
        newargs = tuple(self.spaces) + (self.grid, self.test_basis, self.trial_basis)

        if self.mapping:
            newargs = newargs + (self.mapping,)

        kwargs = self._check_arguments(**kwargs)

        return self.func(*newargs, **kwargs)

#        # TODO remove => this is for debug only
#        import sys
#        sys.path.append(self.folder)
#        from interface_9entwkkx import  interface_9entwkkx
#        sys.path.remove(self.folder)
#        return  interface_9entwkkx(*newargs, **kwargs)

#==============================================================================
class DiscreteLinearForm(BasicDiscrete):

    def __init__(self, expr, kernel_expr, *args, **kwargs):
        if not isinstance(expr, sym_LinearForm):
            raise TypeError('> Expecting a symbolic LinearForm')

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
        self._space = args[1]
        # ...

        kwargs['discrete_space']      = self.space
        kwargs['mapping']             = self.space.symbolic_mapping
        kwargs['is_rational_mapping'] = is_rational_mapping
        kwargs['comm']                = domain_h.comm

        BasicDiscrete.__init__(self, expr, kernel_expr, **kwargs)

        # ...
        quad_order = kwargs.pop('quad_order', None)
        boundary   = kwargs.pop('boundary',   None)
        # ...

        # ...
        if boundary is None:
            self._grid = QuadratureGrid( self.space, quad_order = quad_order )

        else:

            self._grid = BoundaryQuadratureGrid( self.space,
                                                 boundary.axis,
                                                 boundary.ext,
                                                 quad_order = quad_order )
        # ...

        # ...
        self._test_basis = BasisValues( self.space, self.grid,
                                        nderiv = self.max_nderiv )
        # ...

    @property
    def space(self):
        return self._space

    @property
    def grid(self):
        return self._grid

    @property
    def test_basis(self):
        return self._test_basis

    def assemble(self, **kwargs):
        newargs = (self.space, self.grid, self.test_basis)

        if self.mapping:
            newargs = newargs + (self.mapping,)

        kwargs = self._check_arguments(**kwargs)

        return self.func(*newargs, **kwargs)


#==============================================================================
class DiscreteIntegral(BasicDiscrete):

    def __init__(self, expr, kernel_expr, *args, **kwargs):
        if not isinstance(expr, sym_Integral):
            raise TypeError('> Expecting a symbolic Integral')

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
        self._space = args[1]
        # ...

        kwargs['discrete_space']      = self.space
        kwargs['mapping']             = self.space.symbolic_mapping
        kwargs['is_rational_mapping'] = is_rational_mapping
        kwargs['comm']                = domain_h.comm

        BasicDiscrete.__init__(self, expr, kernel_expr, **kwargs)

        # ...
        quad_order = kwargs.pop('quad_order', None)
        boundary   = kwargs.pop('boundary',   None)
        # ...

        # ...
        if boundary is None:
            self._grid = QuadratureGrid( self.space, quad_order = quad_order )

        else:

            self._grid = BoundaryQuadratureGrid( self.space,
                                                 boundary.axis,
                                                 boundary.ext,
                                                 quad_order = quad_order )
        # ...

        # ...
        self._test_basis = BasisValues( self.space, self.grid,
                                        nderiv = self.max_nderiv )
        # ...

    @property
    def space(self):
        return self._space

    @property
    def grid(self):
        return self._grid

    @property
    def test_basis(self):
        return self._test_basis

    def assemble(self, **kwargs):
        newargs = (self.space, self.grid, self.test_basis)

        if self.mapping:
            newargs = newargs + (self.mapping,)

        kwargs = self._check_arguments(**kwargs)

        v = self.func(*newargs, **kwargs)

#        # ... TODO remove => this is for debug only
#        import sys
#        sys.path.append(self.folder)
#        from interface_pt3xujb5 import  interface_pt3xujb5
#        sys.path.remove(self.folder)
#        v = interface_pt3xujb5(*newargs, **kwargs)
#        # ...

        # case of a norm
        if isinstance(self.expr, sym_Norm):
            if not( self.comm is None ):
                v = self.comm.allreduce(sendobj=v)

            if self.expr.exponent == 2:
                # add abs because of 0 machine
                v = np.sqrt(np.abs(v))

            else:
                raise NotImplementedError('TODO')

        return v


#==============================================================================
class DiscreteSumForm(BasicDiscrete):

    def __init__(self, a, kernel_expr, *args, **kwargs):
        if not isinstance(a, (sym_BilinearForm, sym_LinearForm, sym_Integral)):
            raise TypeError('> Expecting a symbolic BilinearForm, LinearFormn Integral')

        self._expr = a

        backend = kwargs.get('backend', None)
        self._backend = backend

        folder = kwargs.get('folder', None)
        self._folder = self._initialize_folder(folder)

        # create a module name if not given
        tag = random_string( 8 )

        # ...
        forms = []
        boundaries = kwargs.pop('boundary', [])
        if isinstance(boundaries, DiscreteBoundary):
            boundaries = [boundaries]

        for e in kernel_expr:
            kwargs['target'] = e.target
            if isinstance(e.target, sym_Boundary):
                boundary = [i for i in boundaries if i is e.target]
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

    @property
    def forms(self):
        return self._forms

    def assemble(self, **kwargs):
        form = self.forms[0]
        M = form.assemble(**kwargs)
        if isinstance(M, (StencilVector, StencilMatrix)):
            M = [M]

        for form in self.forms[1:]:
            n = len(form.interface.inout_arguments)
            # add arguments
            for i in range(0, n):
                key = str(form.interface.inout_arguments[i])
                kwargs[key] = M[i]

            M = form.assemble(**kwargs)
            if isinstance(M, (StencilVector, StencilMatrix)):
                M = [M]

            # remove arguments
            for i in range(0, n):
                key = str(form.interface.inout_arguments[i])
                kwargs.pop(key)

        if len(M) == 1: M = M[0]

        return M

#==============================================================================
class DiscreteEquation(BasicDiscrete):

    def __init__(self, expr, *args, **kwargs):
        if not isinstance(expr, sym_Equation):
            raise TypeError('> Expecting a symbolic Equation')

        # ...
        bc = expr.bc

        if bc:
            bc = [DiscreteDirichletBC(i.boundary) for i in bc]
        # ...

        self._expr = expr
        # since lhs and rhs are calls, we need to take their expr

        # ...
        test_trial = args[1]
        test_space = test_trial[0]
        trial_space = test_trial[1]
        # ...

        # ...
        boundaries_lhs = expr.lhs.expr.atoms(sym_Boundary)
        boundaries_lhs = list(boundaries_lhs)

        boundaries_rhs = expr.rhs.expr.atoms(sym_Boundary)
        boundaries_rhs = list(boundaries_rhs)
        # ...

        # ...
        kwargs['boundary'] = None
        if boundaries_lhs:
            kwargs['boundary'] = boundaries_lhs

        newargs = list(args)
        newargs[1] = test_trial

        self._lhs = discretize(expr.lhs.expr, *newargs, **kwargs)
        # ...

        # ...
        kwargs['boundary'] = None
        if boundaries_rhs:
            kwargs['boundary'] = boundaries_rhs

        newargs = list(args)
        newargs[1] = test_space
        self._rhs = discretize(expr.rhs.expr, *newargs, **kwargs)
        # ...

        self._bc = bc
        self._linear_system = None
        self._trial_space = trial_space
        self._test_space = test_space

    @property
    def expr(self):
        return self._expr

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs

    @property
    def test_space(self):
        return self._test_space

    @property
    def trial_space(self):
        return self._trial_space

    @property
    def bc(self):
        return self._bc

    @property
    def linear_system(self):
        return self._linear_system

    def assemble(self, **kwargs):
        assemble_lhs = kwargs.pop('assemble_lhs', True)
        assemble_rhs = kwargs.pop('assemble_rhs', True)

        if assemble_lhs:
            M = self.lhs.assemble(**kwargs)
            if self.bc:
                # TODO change it: now apply_bc can be called on a list/tuple
                for bc in self.bc:
                    apply_homogeneous_dirichlet_bc(self.test_space, bc, M)
        else:
            M = self.linear_system.lhs

        if assemble_rhs:
            rhs = self.rhs.assemble(**kwargs)
            if self.bc:
                # TODO change it: now apply_bc can be called on a list/tuple
                for bc in self.bc:
                    apply_homogeneous_dirichlet_bc(self.test_space, bc, rhs)

        else:
            rhs = self.linear_system.rhs

        self._linear_system = LinearSystem(M, rhs)

    def solve(self, **kwargs):
        settings = kwargs.pop('settings', _default_solver)

        rhs = kwargs.pop('rhs', None)
        if rhs:
            kwargs['assemble_rhs'] = False

        self.assemble(**kwargs)

        if rhs:
            L = self.linear_system
            L = LinearSystem(L.lhs, rhs)
            self._linear_system = L

        return driver_solve(self.linear_system, **settings)

#==============================================================================
# TODO not used for the moment
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

        # ... create discrete forms
        test_space = self.spaces[0]
        trial_space = self.spaces[1]
        d_forms = {}
        # TODO treat equation forms
        for name, a in list(expr.forms.items()):
            kernel_expr = evaluate(a)
            if isinstance(a, sym_BilinearForm):
                spaces = (test_space, trial_space)
                ah = DiscreteBilinearForm(a, kernel_expr, spaces)

            elif isinstance(a, sym_LinearForm):
                ah = DiscreteLinearForm(a, kernel_expr, test_space)

            elif isinstance(a, sym_Integral):
                ah = DiscreteIntegral(a, kernel_expr, test_space)

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
        self._save_code()
        # ...

        # ...
        namespace = kwargs.pop('namespace', globals())
        code = ''
        for name, ah in list(self.forms.items()):
            # generate code for Python interface
            ah._generate_interface_code()

            # compile code
            ah._compile(namespace)
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


#==============================================================================
# TODO multi patch
# TODO bounds and knots
def discretize_space(V, domain_h, *args, **kwargs):
    degree           = kwargs.pop('degree', None)
    comm             = domain_h.comm
    symbolic_mapping = None

    # from a discrete geoemtry
    # TODO improve condition on mappings
    if isinstance(domain_h, Geometry) and all(domain_h.mappings.values()):
        if len(domain_h.mappings.values()) > 1:
            raise NotImplementedError('Multipatch not yet available')

        mapping = list(domain_h.mappings.values())[0]
        Vh = mapping.space

        # TODO how to give a name to the mapping?
        symbolic_mapping = Mapping('M', domain_h.pdim)

        if not( comm is None ) and domain_h.ldim == 1:
            raise NotImplementedError('must create a TensorFemSpace in 1d')

    elif not( degree is None ):
        assert(hasattr(domain_h, 'ncells'))

        ncells = domain_h.ncells

        # 1d case
        if V.ldim == 1:
            raise NotImplementedError('TODO')

        # 2d case
        elif V.ldim in [2,3]:
            assert(isinstance( degree, (list, tuple) ))
            assert( len(degree) == V.ldim )

            # Create uniform grid
            grids = [np.linspace( 0., 1., num=ne+1 ) for ne in ncells]

            # Create 1D finite element spaces and precompute quadrature data
            spaces = [SplineSpace( p, grid=grid ) for p,grid in zip(degree, grids)]

            Vh = TensorFemSpace( *spaces, comm=comm )

    # Product and Vector spaces are constructed here using H1 subspaces
    if V.shape > 1:
        spaces = [Vh for i in range(V.shape)]
        Vh = ProductFemSpace(*spaces)

    # add symbolic_mapping as a member to the space object
    setattr(Vh, 'symbolic_mapping', symbolic_mapping)

    return Vh

#==============================================================================
def discretize_domain(domain, *args, **kwargs):
    filename = kwargs.pop('filename', None)
    ncells   = kwargs.pop('ncells',   None)
    comm     = kwargs.pop('comm',     None)

    if not( ncells is None ):
        dtype = domain.dtype

        if dtype['type'].lower() == 'line' :
            return Geometry.as_line(ncells, comm=comm)

        elif dtype['type'].lower() == 'square' :
            return Geometry.as_square(ncells, comm=comm)

        elif dtype['type'].lower() == 'cube' :
            return Geometry.as_cube(ncells, comm=comm)

        else:
            msg = 'no corresponding discrete geometry is available, given {}'
            msg = msg.format(dtype['type'])

            raise ValueError(msg)

    elif not( filename is None ):
        geometry = Geometry(filename=filename, comm=comm)

    return geometry


#==============================================================================
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

    elif isinstance(a, sym_Equation):
        return DiscreteEquation(a, *args, **kwargs)

    elif isinstance(a, sym_Model):
        return Model(a, *args, **kwargs)

    elif isinstance(a, BasicFunctionSpace):
        return discretize_space(a, *args, **kwargs)

    elif isinstance(a, Domain):
        return discretize_domain(a, *args, **kwargs)
