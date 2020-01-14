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
from sympde.expr     import BoundaryIntegral as sym_BoundaryIntegral
from sympde.expr     import Equation as sym_Equation
from sympde.expr     import Boundary as sym_Boundary
from sympde.expr     import Norm as sym_Norm
from sympde.expr     import TerminalExpr
from sympde.topology import Domain, Boundary
from sympde.topology import Line, Square, Cube
from sympde.topology import BasicFunctionSpace
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace, Derham
from sympde.topology import ProductSpace
from sympde.topology import Mapping
from sympde.topology import H1SpaceType, HcurlSpaceType, HdivSpaceType, L2SpaceType, UndefinedSpaceType

from gelato.expr     import GltExpr as sym_GltExpr
from sympy           import Expr    as sym_Expr

from psydac.api.basic                import BasicDiscrete
from psydac.api.fem                  import DiscreteBilinearForm
from psydac.api.fem                  import DiscreteLinearForm
from psydac.api.fem                  import DiscreteFunctional
from psydac.api.fem                  import DiscreteSumForm
from psydac.api.glt                  import DiscreteGltExpr
from psydac.api.expr                 import DiscreteExpr

from psydac.api.essential_bc         import apply_essential_bc
from psydac.linalg.iterative_solvers import cg
from psydac.fem.splines              import SplineSpace
from psydac.fem.tensor               import TensorFemSpace
from psydac.fem.vector               import ProductFemSpace
from psydac.cad.geometry             import Geometry
from psydac.mapping.discrete         import SplineMapping, NurbsMapping
from psydac.feec.derivatives         import Grad, Curl, Div
from psydac.feec.utilities           import Interpolation, interpolation_matrices

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
        domain      = args[0]
        trial_test  = args[1]
        trial_space = trial_test[0]
        test_space  = trial_test[1]
        # ...

        # ...
        boundaries_lhs = expr.lhs.atoms(sym_BoundaryIntegral)
        boundaries_lhs = [a.domain for a in boundaries_lhs]

        boundaries_rhs = expr.rhs.atoms(sym_BoundaryIntegral)
        boundaries_rhs = [a.domain for a in boundaries_rhs]
        # ...

        # ...

        kwargs['boundary'] = None
        if boundaries_lhs:
            kwargs['boundary'] = boundaries_lhs

        newargs = list(args)
        newargs[1] = trial_test

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
        self._domain      = domain
        self._trial_space = trial_space
        self._test_space  = test_space

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
    def domain(self):
        return self._domain

    @property
    def trial_space(self):
        return self._trial_space

    @property
    def test_space(self):
        return self._test_space

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

        settings = _default_solver.copy()
        settings.update(kwargs.pop('settings', {}))

        rhs = kwargs.pop('rhs', None)
        if rhs:
            kwargs['assemble_rhs'] = False

        self.assemble(**kwargs)

        if rhs:
            L = self.linear_system
            L = LinearSystem(L.lhs, rhs)
            self._linear_system = L

        #----------------------------------------------------------------------
        # [YG, 18/11/2019]
        #
        # Impose inhomogeneous Dirichlet boundary conditions through
        # L2 projection on the boundary. This requires setting up a
        # new variational formulation and solving the resulting linear
        # system to obtain a solution that does not live in the space
        # of homogeneous solutions. Such a solution is then used as
        # initial guess when the model equation is to be solved by an
        # iterative method. Our current method of solution does not
        # modify the initial guess at the boundary.
        #
        if self.bc:

            # Inhomogeneous Dirichlet boundary conditions
            idbcs = [i for i in self.bc if i.rhs != 0]

            if idbcs:

                from sympde.expr import integral
                from sympde.expr import find
                from sympde.topology import element_of #, ScalarTestFunction

                # Extract trial functions from model equation
                u = self.expr.trial_functions

                # Create test functions in same space of trial functions
                # TODO: check if we should generate random names
                V = ProductSpace(*[ui.space for ui in u])
                v = element_of(V, name='v:{}'.format(len(u)))

                # In a system, each essential boundary condition is applied to
                # only one component (bc.variable) of the state vector. Hence
                # we will select the correct test function using a dictionary.
                test_dict = dict(zip(u, v))

                # TODO: use dot product for vector quantities
#                product  = lambda f, g: (f * g if isinstance(g, ScalarTestFunction) else dot(f, g))
                product  = lambda f, g: f * g

                # Construct variational formulation that performs L2 projection
                # of boundary conditions onto the correct space
                factor   = lambda bc : bc.lhs.xreplace(test_dict)
                lhs_expr = sum(integral(i.boundary, product(i.lhs, factor(i))) for i in idbcs)
                rhs_expr = sum(integral(i.boundary, product(i.rhs, factor(i))) for i in idbcs)
                equation = find(u, forall=v, lhs=lhs_expr, rhs=rhs_expr)

                # Discretize weak form
                domain_h   = self.domain
                Vh         = self.trial_space
                equation_h = discretize(equation, domain_h, [Vh, Vh])

                # Find inhomogeneous solution (use CG as system is symmetric)
                loc_settings = settings.copy()
                loc_settings['solver'] = 'cg'
                X = equation_h.solve(**loc_settings)

                # Use inhomogeneous solution as initial guess to solver
                settings['x0'] = X

        #----------------------------------------------------------------------

        return driver_solve(self.linear_system, **settings)

#==============================================================================
class DiscreteDerham(BasicDiscrete):
    def __init__(self, *spaces):
    
        dim       = len(spaces) - 1
        self._V0  = None
        self._V1  = None
        self._V2  = None
        self._V3  = None
        self._dim = dim
        
        if dim == 1:
            self._V0 = spaces[0]
            self._V1 = spaces[1]
        elif dim == 2:
            self._V0 = spaces[0]
            self._V1 = spaces[1]
            self._V2 = spaces[2]
        elif dim == 3:
            self._V0 = spaces[0]
            self._V1 = spaces[1]
            self._V2 = spaces[2]
            self._V3 = spaces[3]

    # ...
    @property
    def dim(self):
        return self._dim

    # ...  
    @property
    def V0(self):
        return self._V0

    @property
    def V1(self):
        return self._V1        

    @property
    def V2(self):
        return self._V2
        
    @property
    def V3(self):
        return self._V3  

#==============================================================================           
def discretize_derham(V, domain_h, *args, **kwargs):

    spaces = [discretize_space(Vi, domain_h, *args, **kwargs) for Vi in V.spaces]
    ldim   = V.shape
    
    if ldim == 1:
        D0 = Grad(spaces[0], spaces[1].vector_space)
        setattr(spaces[0], 'grad', D0)
        
        interpolation0 = Interpolation(H1=spaces[0])
        interpolation1 = Interpolation(H1=spaces[0], L2=spaces[1])
        
        setattr(spaces[0], 'interpolate', interpolation0)
        setattr(spaces[1], 'interpolate', interpolation1)
        
    elif ldim == 2:
        
        if isinstance(V.spaces[1].kind, HcurlSpaceType):
            D0 = Grad(spaces[0], spaces[1].vector_space)
            D1 = Curl(spaces[0], spaces[1].vector_space, spaces[2].vector_space)
            setattr(spaces[0], 'grad', D0)
            setattr(spaces[1], 'curl', D1)
            
            interpolation0 = Interpolation(H1=spaces[0])
            interpolation1 = Interpolation(H1=spaces[0], Hcurl=spaces[1])
            interpolation2 = Interpolation(H1=spaces[0], L2=spaces[2])
            
        else:
            D0 = Grad(spaces[0], spaces[1].vector_space)
            D1 = Div(spaces[0], spaces[1].vector_space, spaces[2].vector_space)
            setattr(spaces[0], 'grad', D0)
            setattr(spaces[1], 'div', D1)
            
            interpolation0 = Interpolation(H1=spaces[0])
            interpolation1 = Interpolation(H1=spaces[0], Hdiv=spaces[1])
            interpolation2 = Interpolation(H1=spaces[0], L2=spaces[2])

        setattr(spaces[0], 'interpolate', interpolation0)
        setattr(spaces[1], 'interpolate', interpolation1)
        setattr(spaces[2], 'interpolate', interpolation2)
            
    elif ldim == 3:
        D0 = Grad(spaces[0], spaces[1].vector_space)
        D1 = Curl(spaces[0], spaces[1].vector_space, spaces[2].vector_space)  
        D2 = Div(spaces[0], spaces[2].vector_space, spaces[3].vector_space)
        
        setattr(spaces[1], 'grad', D0)
        setattr(spaces[2], 'curl', D1)
        setattr(spaces[3], 'div' , D2)
        
        interpolation0 = Interpolation(H1=spaces[0])
        interpolation1 = Interpolation(H1=spaces[0], Hcurl=spaces[1])
        interpolation2 = Interpolation(H1=spaces[0], Hdiv=spaces[2])
        interpolation3 = Interpolation(H1=spaces[0], L2=spaces[3])
        
        setattr(spaces[0], 'interpolate', interpolation0)
        setattr(spaces[1], 'interpolate', interpolation1)
        setattr(spaces[2], 'interpolate', interpolation2)
        setattr(spaces[3], 'interpolate', interpolation3)
        

        
    return DiscreteDerham(*spaces)
#==============================================================================
# TODO multi patch
# TODO knots
def discretize_space(V, domain_h, *args, **kwargs):
    degree           = kwargs.pop('degree', None)
    normalize        = kwargs.pop('normalize', True)
    comm             = domain_h.comm
    symbolic_mapping = None
    kind             = V.kind
    ldim             = V.ldim
    
    if isinstance(V, ProductSpace):
        kwargs['normalize'] = normalize
        normalize = False
    else:
        normalize = normalize

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

        ncells     = domain_h.ncells
        min_coords = domain_h.domain.min_coords
        max_coords = domain_h.domain.max_coords

        assert(isinstance( degree, (list, tuple) ))
        assert( len(degree) == ldim )

        # Create uniform grid
        grids = [np.linspace(xmin, xmax, num=ne + 1)
                 for xmin, xmax, ne in zip(min_coords, max_coords, ncells)]

        # Create 1D finite element spaces and precompute quadrature data
        spaces = [SplineSpace( p, grid=grid ) for p,grid in zip(degree, grids)]
        Vh = TensorFemSpace( *spaces, comm=comm )
        
        if isinstance(kind, L2SpaceType):
  
            if ldim == 1:
                Vh = Vh.reduce_degree(axes=[0], normalize=normalize)
            elif ldim == 2:
                Vh = Vh.reduce_degree(axes=[0,1], normalize=normalize)
            elif ldim == 3:
                Vh = Vh.reduce_degree(axes=[0,1,2], normalize=normalize)

    # Product and Vector spaces are constructed here
    if V.shape > 1:
        spaces = []
        if isinstance(V, VectorFunctionSpace):
        
            if isinstance(kind, (H1SpaceType, L2SpaceType,  UndefinedSpaceType)):
                spaces = [Vh for i in range(V.shape)]
                
            elif isinstance(kind, HcurlSpaceType):
                if ldim == 2:
                    spaces = [Vh.reduce_degree(axes=[0], normalize=normalize), 
                              Vh.reduce_degree(axes=[1], normalize=normalize)]
                elif ldim == 3:
                    spaces = [Vh.reduce_degree(axes=[0], normalize=normalize), 
                              Vh.reduce_degree(axes=[1], normalize=normalize), 
                              Vh.reduce_degree(axes=[2], normalize=normalize)]
                else:
                    raise NotImplementedError('TODO')
                
            elif isinstance(kind, HdivSpaceType):
            
                if ldim == 2:
                    spaces = [Vh.reduce_degree(axes=[1], normalize=normalize), 
                              Vh.reduce_degree(axes=[0], normalize=normalize)]
                elif ldim == 3:
                    spaces = [Vh.reduce_degree(axes=[1,2], normalize=normalize), 
                              Vh.reduce_degree(axes=[0,2], normalize=normalize), 
                              Vh.reduce_degree(axes=[0,1], normalize=normalize)]
                else:
                    raise NotImplementedError('TODO')
                    
        elif isinstance(V, ProductSpace):
            
            for Vi in V.spaces:
                space = discretize_space(Vi, domain_h, *args, degree=degree,**kwargs)
                if isinstance(space, ProductFemSpace):
                    spaces += list(space.spaces)
                else:
                    spaces += [space]
        
        Vh = ProductFemSpace(*spaces)
        setattr(Vh, 'shape', V.shape)

    # add symbolic_mapping as a member to the space object
    setattr(Vh, 'symbolic_mapping', symbolic_mapping)
    

    return Vh

#==============================================================================
def discretize_domain(domain, *, filename=None, ncells=None, comm=None):

    if not (filename or ncells):
        raise ValueError("Must provide either 'filename' or 'ncells'")

    elif filename and ncells:
        raise ValueError("Cannot provide both 'filename' and 'ncells'")

    elif filename:
        return Geometry(filename=filename, comm=comm)

    elif ncells:
        return Geometry.from_topological_domain(domain, ncells, comm)

#==============================================================================
def discretize(a, *args, **kwargs):

    if isinstance(a, sym_BasicForm):
        if not a.is_annotated:
            a = a.annotate()
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
        
    elif isinstance(a, Derham):
        return discretize_derham(a, *args, **kwargs)

    elif isinstance(a, Domain):
        return discretize_domain(a, *args, **kwargs)

    elif isinstance(a, sym_GltExpr):
        return DiscreteGltExpr(a, *args, **kwargs)
        
    elif isinstance(a, sym_Expr):
        return DiscreteExpr(a, *args, **kwargs)

    else:
        raise NotImplementedError('given {}'.format(type(a)))
