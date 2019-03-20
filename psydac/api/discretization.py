# coding: utf-8

# TODO: - init_fem is called whenever we call discretize. we should check that
#         nderiv has not been changed. shall we add quad_order too?

# TODO: avoid using os.system and use subprocess.call

from collections import OrderedDict
from collections import namedtuple

from pyccel.ast import Nil
from pyccel.epyccel import get_source_function

from sympde.expr     import BasicForm as sym_BasicForm
from sympde.expr     import BilinearForm as sym_BilinearForm
from sympde.expr     import LinearForm as sym_LinearForm
from sympde.expr     import Functional as sym_Functional
from sympde.expr     import Equation as sym_Equation
from sympde.expr     import Boundary as sym_Boundary
from sympde.expr     import Norm as sym_Norm
from sympde.expr     import TerminalExpr
from sympde.topology import Domain, Boundary
from sympde.topology import Line, Square, Cube
from sympde.topology import BasicFunctionSpace
from sympde.topology import FunctionSpace, VectorFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import Mapping
from sympde.topology import H1SpaceType, HcurlSpaceType, HdivSpaceType, L2SpaceType

from gelato.expr     import GltExpr as sym_GltExpr


from psydac.api.basic                import BasicDiscrete
from psydac.api.fem                  import DiscreteBilinearForm
from psydac.api.fem                  import DiscreteLinearForm
from psydac.api.fem                  import DiscreteFunctional
from psydac.api.fem                  import DiscreteSumForm
from psydac.api.glt                  import DiscreteGltExpr

from psydac.api.essential_bc         import apply_essential_bc
from psydac.linalg.iterative_solvers import cg
from psydac.fem.splines              import SplineSpace
from psydac.fem.tensor               import TensorFemSpace
from psydac.fem.vector               import ProductFemSpace
from psydac.cad.geometry             import Geometry
from psydac.mapping.discrete         import SplineMapping, NurbsMapping

import inspect
import sys
import os
import importlib
import string
import random
import numpy as np
from mpi4py import MPI



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
class DiscreteEquation(BasicDiscrete):

    def __init__(self, expr, *args, **kwargs):
        if not isinstance(expr, sym_Equation):
            raise TypeError('> Expecting a symbolic Equation')

        # ...
        bc = expr.bc
        # ...

        self._expr = expr
        # since lhs and rhs are calls, we need to take their expr

        # ...
        test_trial = args[1]
        test_space = test_trial[0]
        trial_space = test_trial[1]
        # ...

        # ...
        boundaries_lhs = expr.lhs.atoms(sym_Boundary)
        boundaries_lhs = list(boundaries_lhs)

        boundaries_rhs = expr.rhs.atoms(sym_Boundary)
        boundaries_rhs = list(boundaries_rhs)
        # ...

        # ...
        kwargs['boundary'] = None
        if boundaries_lhs:
            kwargs['boundary'] = boundaries_lhs

        newargs = list(args)
        newargs[1] = test_trial

        self._lhs = discretize(expr.lhs, *newargs, **kwargs)
        # ...

        # ...
        kwargs['boundary'] = None
        if boundaries_rhs:
            kwargs['boundary'] = boundaries_rhs

        newargs = list(args)
        newargs[1] = test_space
        self._rhs = discretize(expr.rhs, *newargs, **kwargs)
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
                    apply_essential_bc(self.test_space, bc, M)
        else:
            M = self.linear_system.lhs

        if assemble_rhs:
            rhs = self.rhs.assemble(**kwargs)
            if self.bc:
                # TODO change it: now apply_bc can be called on a list/tuple
                for bc in self.bc:
                    apply_essential_bc(self.test_space, bc, rhs)

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
# TODO multi patch
# TODO bounds and knots
def discretize_space(V, domain_h, *args, **kwargs):
    degree           = kwargs.pop('degree', None)
    comm             = domain_h.comm
    symbolic_mapping = None
    kind             = V.kind
    ldim             = V.ldim

    # from a discrete geoemtry
    # TODO improve condition on mappings
    if isinstance(domain_h, Geometry) and all(domain_h.mappings.values()):
        if len(domain_h.mappings.values()) > 1:
            raise NotImplementedError('Multipatch not yet available')

        mapping = list(domain_h.mappings.values())[0]
        Vh = mapping.space

        # TODO how to give a name to the mapping?
        symbolic_mapping = Mapping('M', domain_h.pdim)

        if not( comm is None ) and ldim == 1:
            raise NotImplementedError('must create a TensorFemSpace in 1d')

    elif not( degree is None ):
        assert(hasattr(domain_h, 'ncells'))

        ncells = domain_h.ncells

        assert(isinstance( degree, (list, tuple) ))
        assert( len(degree) == ldim )

        # Create uniform grid
        grids = [np.linspace( 0., 1., num=ne+1 ) for ne in ncells]

        # Create 1D finite element spaces and precompute quadrature data
        spaces = [SplineSpace( p, grid=grid ) for p,grid in zip(degree, grids)]
        
        Vh = TensorFemSpace( *spaces, comm=comm )

        if isinstance(kind, HcurlSpaceType):
        
            if ldim == 2:
                Vh = [Vh.reduce_degree(axes=[0]),Vh.reduce_degree(axes=[1])]
            elif ldim == 3:
                Vh = [Vh.reduce_degree(axes=[0]),Vh.reduce_degree(axes=[1]),Vh.reduce_degree(axes=[2])]
            
        elif isinstance(kind, HdivSpaceType):
        
            if ldim == 2:
                Vh = [Vh.reduce_degree(axes=[1]), Vh.reduce_degree(axes=[0])]
            elif ldim == 3:
                Vh = [Vh.reduce_degree(axes=[1,2]),Vh.reduce_degree(axes=[0,2]),Vh.reduce_degree(axes=[0,1])]
            
        elif isinstance(kind, L2SpaceType):
            
            if ldim == 1:
                Vh = Vh.reduce_degree(axes=[0])
            elif ldim == 2:
                Vh = Vh.reduce_degree(axes=[0,1])
            elif ldim == 3:
                Vh = Vh.reduce_degree(axes=[0,1,2])

    # Product and Vector spaces are constructed here using H1 subspaces

    if V.shape > 1:
    
        if isinstance(V, VectorFunctionSpace):
            spaces = [Vh for i in range(V.shape)]
            
        elif isinstance(V, ProductSpace):
            spaces = []
            
            for Vi in V.spaces:
            
                if isinstance(Vi, VectorFunctionSpace):
                    spaces += [Vh for i in range(Vi.shape)]
                    
                else:
                    spaces += [discretize_space(Vi, domain_h, *args, degree=degree,**kwargs)]
        
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
        kernel_expr = TerminalExpr(a)
#        print('=================')
#        print(kernel_expr)
#        print('=================')
#        sys.exit(0)
        if len(kernel_expr) > 1:
            return DiscreteSumForm(a, kernel_expr, *args, **kwargs)

    if isinstance(a, sym_BilinearForm):
        return DiscreteBilinearForm(a, kernel_expr, *args, **kwargs)

    elif isinstance(a, sym_LinearForm):
        return DiscreteLinearForm(a, kernel_expr, *args, **kwargs)

    elif isinstance(a, sym_Functional):
        return DiscreteFunctional(a, kernel_expr, *args, **kwargs)

    elif isinstance(a, sym_Equation):
        return DiscreteEquation(a, *args, **kwargs)

    elif isinstance(a, BasicFunctionSpace):
        return discretize_space(a, *args, **kwargs)

    elif isinstance(a, Domain):
        return discretize_domain(a, *args, **kwargs)

    elif isinstance(a, sym_GltExpr):
        return DiscreteGltExpr(a, *args, **kwargs)

    else:
        raise NotImplementedError('given {}'.format(type(a)))
