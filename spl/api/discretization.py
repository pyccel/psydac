# coding: utf-8

# TODO: - init_fem is called whenever we call discretize. we should check that
#         nderiv has not been changed. shall we add quad_order too?

# TODO: avoid using os.system and use subprocess.call

from collections import OrderedDict
from collections import namedtuple

from pyccel.ast import Nil

from sympde.core import BasicForm as sym_BasicForm
from sympde.core import BilinearForm as sym_BilinearForm
from sympde.core import LinearForm as sym_LinearForm
from sympde.core import Integral as sym_Integral
from sympde.core import Equation as sym_Equation
from sympde.core import Model as sym_Model
from sympde.core import Boundary as sym_Boundary
from sympde.core import Norm as sym_Norm
from sympde.core import evaluate

from spl.api.codegen.ast import Kernel
from spl.api.codegen.ast import Assembly
from spl.api.codegen.ast import Interface
from spl.api.codegen.printing import pycode
from spl.api.boundary_condition import DiscreteBoundary
from spl.api.boundary_condition import DiscreteComplementBoundary
from spl.api.boundary_condition import DiscreteBoundaryCondition, DiscreteDirichletBC
from spl.api.boundary_condition import apply_homogeneous_dirichlet_bc
from spl.api.settings import SPL_BACKEND_PYTHON, SPL_DEFAULT_FOLDER
from spl.linalg.stencil import StencilVector, StencilMatrix
from spl.linalg.iterative_solvers import cg

import sys
import os
import importlib
import string
import random
import numpy as np


#==============================================================================
def mkdir_p(folder):
    if os.path.isdir(folder):
        return
    os.makedirs(folder)

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
    cmd = 'touch {}/__init__.py'.format(folder)
    os.system(cmd)

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

    def __init__(self, a, kernel_expr, namespace=globals(), to_compile=True,
                 module_name=None, boundary=None, target=None,
                 boundary_basis=None, backend=SPL_BACKEND_PYTHON, folder=None):

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
#                    print(boundary.expr)
#                    print(target)
#                    import sys; sys.exit(0)
                    raise ValueError('> Unconsistent boundary with symbolic model')

                boundary = [boundary.axis, boundary.ext]
                boundary = [boundary]
                boundary_basis = True # TODO set it to False for Nitch method

            # boundary is now a list of boundaries
            # TODO shall we keep it this way? since this is the simplest
            # interface to be able to compute Integral on a curve in 3d
        # ...

        # ...
        kernel = Kernel(a, kernel_expr, target=target,
                        discrete_boundary=boundary,
                        boundary_basis=boundary_basis)
        assembly = Assembly(kernel)
        interface = Interface(assembly, backend=backend)
        # ...

        # ...
        self._expr = a
        self._kernel_expr = kernel_expr
        self._target = target
        self._tag = kernel.tag
        self._mapping = None
        self._interface = interface
        self._dependencies = self.interface.dependencies
        self._backend = backend
        self._folder = self._initiliaze_folder(folder)
        # ...

        # generate python code as strings for dependencies
        self._dependencies_code = self._generate_code()

        self._dependencies_fname = None
        self._dependencies_modname = None
        self._interface_code = None
        self._interface_base_import_code = None
        self._func = None
        if to_compile:
            # save dependencies code
            self._save_code(module_name=module_name)

            if self.backend['name'] == 'pyccel':
                self._compile_pyccel(namespace)

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
    def folder(self):
        return self._folder

    def _initiliaze_folder(self, folder=None):
        # ...
        if folder is None:
            basedir = os.getcwd()
            folder = SPL_DEFAULT_FOLDER
            folder = os.path.join( basedir, folder )

        folder = os.path.abspath( folder )
        mkdir_p(folder)
        # ...

        return folder

    def _generate_code(self):
        # ... generate code that can be pyccelized
        code = 'from pyccel.decorators import types'
        for dep in self.dependencies:
            code = '{code}\n{dep}'.format(code=code, dep=pycode(dep))
        # ...
        return code

    def _save_code(self, module_name=None):
        # ...
        code = self.dependencies_code
        if module_name is None:
            module_name = 'dependencies_{}'.format(self.tag)

        self._dependencies_fname = '{}.py'.format(module_name)
        write_code(self.dependencies_fname, code, folder = self.folder)
        # ...

        # TODO check this? since we are using relative paths now
        self._dependencies_modname = module_name.replace('/', '.')

    def _generate_interface_code(self, module_name=None):
        imports = []
        prefix = ''

        if module_name is None:
            module_name = self.dependencies_modname

        # ...
        if self.backend['name'] == 'pyccel':
            imports += [self.interface_base_import_code]

            pattern = '{dep} = {module}.f2py_{dep}'

            for dep in self.dependencies:
                txt = pattern.format(module=module_name, dep=dep.name)
                imports.append(txt)

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
        from pyccel.epyccel import epyccel, compile_fortran
        from pyccel.codegen.utilities import execute_pyccel

        folder = SPL_DEFAULT_FOLDER
        mkdir_p(folder)

        basedir = os.getcwd()
        os.chdir(folder)
        curdir = os.getcwd()

        # ... convert python to fortran using pyccel
        openmp = False
        compiler = 'gfortran'

        fname = os.path.basename(self.dependencies_fname)
        package = epyccel(fname, compiler = compiler)
        # ...
        import sys; sys.exit(0)

        # ...
        fname = os.path.basename(fname).split('.')[0]
        fname = '{}.o'.format(fname)
        libname = '{}'.format(self.tag).lower() # because of f2py

        cmd = 'ar -r lib{libname}.a {fname} '.format(fname=fname, libname=libname)
        os.system(cmd)

        if verbose:
            print(cmd)
        # ...

        # ... construct a f2py interface for the assembly
        # be careful: because of f2py we must use lower case
        from pyccel.ast.utilities import build_types_decorator
        from pyccel.ast.core import FunctionDef
        from pyccel.ast.core import FunctionCall

        tag = self.tag

        assembly = self.interface.assembly
        module_name = module_name.split('.')[-1]

        func = assembly.func
        args = func.arguments

        body = [FunctionCall(func, args)]

        # ...
        def construct_header(func_name, args):
            args = build_types_decorator(args, order='F')
            args = ','.join("{}".format(i) for i in args)
            pattern = '#$ header procedure static {name}({args})'
            return pattern.format(name=func_name, args=args)
        # ...

        func_name = 'f2py_{}'.format(assembly.name)
        header = construct_header(func_name, args)
        func = FunctionDef(func_name, list(args), [], body)
        code = pycode(func)
        imports = 'from {mod} import {func}'.format(mod=module_name,
                                                    func=assembly.name)

        code = '{imports}\n{header}\n{code}'.format(imports=imports,
                                                    header=header,
                                                    code=code)

        module_name = 'f2py_dependencies_{}'.format(tag)
        fname = write_code(module_name, code, folder = self.folder)

        fname = execute_pyccel(fname, output='', convert_only=True)

        f = open(fname)
        code = f.readlines()
        f.close()

        name = 'f2py_dependencies_{}'.format(self.tag)

        extra_args = ' -L{} '.format(curdir)

        output, cmd = compile_fortran( code, name,
                                       extra_args= extra_args,
                                       libs      = [libname],
                                       compiler  = compiler,
                                       mpi       = False,
                                       openmp    = openmp)
        # ...

        if verbose:
            print(cmd)

        os.chdir(basedir)

        # ...
        # update module name for dependencies
        # needed for interface when importing assembly
        # name.name is needed for f2py
        name = os.path.join(folder, name)
        name = name.replace('/', '.')

        import_mod = 'from {name} import {module_name}'.format(name=name,
                                                               module_name=module_name)
        self._interface_base_import_code = import_mod
        self._dependencies_modname = module_name
        # ...

    def _compile(self, namespace, module_name=None):

        if module_name is None:
            module_name = self.dependencies_modname

        # ... TODO move to save
        code = self.interface_code
        interface_module_name = 'interface_{}'.format(self.tag)
        fname = '{}.py'.format(interface_module_name)
        fname = write_code(fname, code, folder = self.folder)
        # ...

        # ...
        sys.path.append(self.folder)
        package = importlib.import_module( interface_module_name )
        sys.path.remove(self.folder)
        # ...

        self._func = getattr(package, self.interface.name)

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

        # initialize fem space basis/quad
        for V in self.spaces:
            V.init_fem(nderiv=self.interface.max_nderiv)

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

        # initialize fem space basis/quad
        self.space.init_fem(nderiv=self.interface.max_nderiv)

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

        # initialize fem space basis/quad
        self.space.init_fem(nderiv=self.interface.max_nderiv)

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

        v = self.func(*newargs, **kwargs)

        # case of a norm
        if isinstance(self.expr, sym_Norm):
            if self.expr.exponent == 2:
                v = np.sqrt(v)

            else:
                raise NotImplementedError('TODO')

        return v


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
        if isinstance(boundaries, DiscreteBoundary):
            boundaries = [boundaries]

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

class DiscreteEquation(BasicDiscrete):

    def __init__(self, expr, *args, **kwargs):
        if not isinstance(expr, sym_Equation):
            raise TypeError('> Expecting a symbolic Equation')

        # ...
        bc = kwargs.pop('bc', None)

        if bc:
            if isinstance(bc, DiscreteBoundaryCondition):
                bc = [bc]

            elif isinstance(bc, (list, tuple)):
                for i in bc:
                    if not isinstance(i, DiscreteBoundaryCondition):
                        msg = '> Expecting a list of DiscreteBoundaryCondition'
                        raise TypeError(msg)

            else:
                raise TypeError('> Wrong type for bc')

            newbc = []
            for b in bc:
                bnd = b.boundary
                if isinstance(bnd, DiscreteComplementBoundary):
                    domain = bnd.boundaries[0].expr.domain
                    for axis, ext in zip(bnd.axis, bnd.ext):
                        name = random_string( 3 )
                        B = sym_Boundary(name, domain)
                        B = DiscreteBoundary(B, axis=axis, ext=ext)
                        other = b.duplicate(B)
                        newbc.append(other)

                else:
                    newbc.append(b)

            bc = newbc
        # ...

        self._expr = expr
        # since lhs and rhs are calls, we need to take their expr

        # ...
        test_trial = args[0]
        test_space = test_trial[0]
        trial_space = test_trial[1]
        # ...

        # ...
        boundaries = kwargs.pop('boundary', [])
        if isinstance(boundaries, DiscreteBoundary):
            boundaries = [boundaries]

        boundaries_lhs = expr.lhs.expr.atoms(sym_Boundary)
        boundaries_lhs = [i for i in boundaries if i.expr in boundaries_lhs]

        boundaries_rhs = expr.rhs.expr.atoms(sym_Boundary)
        boundaries_rhs = [i for i in boundaries if i.expr in boundaries_rhs]
        # ...

        # ...
        kwargs['boundary'] = None
        if boundaries_lhs:
            kwargs['boundary'] = boundaries_lhs

        self._lhs = discretize(expr.lhs.expr, test_trial, *args[1:], **kwargs)
        # ...

        # ...
        kwargs['boundary'] = None
        if boundaries_rhs:
            kwargs['boundary'] = boundaries_rhs

        self._rhs = discretize(expr.rhs.expr, test_space, *args[1:], **kwargs)
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
            for bc in self.bc:
                apply_homogeneous_dirichlet_bc(self.test_space, bc, M)
        else:
            M = self.linear_system.lhs

        if assemble_rhs:
            rhs = self.rhs.assemble(**kwargs)
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

    elif isinstance(a, sym_Equation):
        return DiscreteEquation(a, *args, **kwargs)

    elif isinstance(a, sym_Model):
        return Model(a, *args, **kwargs)
