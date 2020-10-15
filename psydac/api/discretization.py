# coding: utf-8

# TODO: - init_fem is called whenever we call discretize. we should check that
#         nderiv has not been changed. shall we add quad_order too?

# TODO: avoid using os.system and use subprocess.call

from collections import OrderedDict
from collections import namedtuple

from pyccel.ast.core import Nil
from pyccel.epyccel  import get_source_function

from sympde.expr     import BasicForm as sym_BasicForm
from sympde.expr     import BilinearForm as sym_BilinearForm
from sympde.expr     import LinearForm as sym_LinearForm
from sympde.expr     import Functional as sym_Functional
from sympde.expr     import Integral
from sympde.expr     import Equation as sym_Equation
from sympde.expr     import Boundary as sym_Boundary, Interface as sym_Interface
from sympde.expr     import Norm as sym_Norm
from sympde.expr     import TerminalExpr
from sympde.topology import Domain, Boundary
from sympde.topology import Line, Square, Cube
from sympde.topology import BasicFunctionSpace
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace, Derham
from sympde.topology import ProductSpace
from sympde.topology import Mapping, IdentityMapping, LogicalExpr
from sympde.topology import H1SpaceType, HcurlSpaceType, HdivSpaceType, L2SpaceType, UndefinedSpaceType
from sympde.topology.basic import Union
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
from psydac.feec.global_projectors   import Projector_H1, Projector_Hcurl, Projector_Hdiv, Projector_L2

from psydac.feec.derivatives import Derivative_1D, Gradient_2D, Gradient_3D
from psydac.feec.derivatives import ScalarCurl_2D, VectorCurl_2D, Curl_3D
from psydac.feec.derivatives import Divergence_2D, Divergence_3D

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

    name        = kwargs.pop('solver')
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
        boundaries_lhs = expr.lhs.atoms(Integral)
        boundaries_lhs = [a.domain for a in boundaries_lhs if a.is_boundary_integral]

        boundaries_rhs = expr.rhs.atoms(Integral)
        boundaries_rhs = [a.domain for a in boundaries_rhs if a.is_boundary_integral]
        # ...

        # ...

        kwargs['boundary'] = []
        if boundaries_lhs:
            kwargs['boundary'] = boundaries_lhs

        newargs = list(args)
        newargs[1] = trial_test

        self._lhs = discretize(expr.lhs, *newargs, **kwargs)
        # ...

        # ...
        kwargs['boundary'] = []
        if boundaries_rhs:
            kwargs['boundary'] = boundaries_rhs
        
        newargs = list(args)
        newargs[1] = test_space
        self._rhs = discretize(expr.rhs, *newargs, **kwargs)
        # ...

        self._bc = bc
        self._linear_system = None
        self._domain        = domain
        self._trial_space   = trial_space
        self._test_space    = test_space

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
        settings = {k:kwargs[k] if k in kwargs else it for k,it in _default_solver.items()}
        settings.update({it[0]:it[1] for it in kwargs.items() if it[0] not in settings})

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
                loc_settings.pop('info',False)
                X = equation_h.solve(**loc_settings)

                # Use inhomogeneous solution as initial guess to solver
                settings['x0'] = X

        #----------------------------------------------------------------------

        return driver_solve(self.linear_system, **settings)

#==============================================================================
class DiscreteDerham(BasicDiscrete):
    """
    Rerpresent the discrete De Rham sequence
    
    """
    def __init__(self, *spaces):

        dim          = len(spaces) - 1
        self._dim    = dim
        self._spaces = spaces

        if dim not in [1,2,3]:
            raise ValueError('dimension {} is not available'.format(dim))

    @property
    def dim(self):
        return self._dim

    @property
    def V0(self):
        return self._spaces[0]

    @property
    def V1(self):
        return self._spaces[1]

    @property
    def V2(self):
        return self._spaces[2]

    @property
    def V3(self):
        return self._spaces[3]

    @property
    def spaces(self):
        return self._spaces

    @property
    def derivatives_as_matrices(self):
        return tuple(V.diff.matrix for V in self.spaces[:-1])

    @property
    def derivatives_as_operators(self):
        return tuple(V.diff for V in self.spaces[:-1])

    def projectors(self, *, kind='global', nquads=None):

        if not (kind == 'global'):
            raise NotImplementedError('only global projectors are available')

        if self.dim == 1:
            P0 = Projector_H1(self.V0)
            P1 = Projector_L2(self.V1, nquads)
            return P0, P1

        elif self.dim == 2:
            raise NotImplementedError('TODO')

        elif self.dim == 3:
            P0 = Projector_H1(self.V0)
            P1 = Projector_Hcurl(self.V1, nquads)
            P2 = Projector_Hdiv(self.V2, nquads)
            P3 = Projector_L2(self.V3, nquads)
            return P0, P1, P2, P3

#==============================================================================           
def discretize_derham(Complex, domain_h, *args, **kwargs):

    ldim     = Complex.shape
    spaces   = Complex.spaces
    d_spaces = [None]*(ldim+1)

    if ldim == 1:

        d_spaces[0] = discretize_space(spaces[0], domain_h, *args, basis='B', **kwargs)
        d_spaces[1] = discretize_space(spaces[1], domain_h, *args, basis='M', **kwargs)

        D0 = Derivative_1D(d_spaces[0], d_spaces[1])

        d_spaces[0].diff = d_spaces[0].grad = D0
        
    elif ldim == 2:

        d_spaces[0] = discretize_space(spaces[0], domain_h, *args, basis='B', **kwargs)
        d_spaces[1] = discretize_space(spaces[1], domain_h, *args, basis='M', **kwargs)
        d_spaces[2] = discretize_space(spaces[2], domain_h, *args, basis='M', **kwargs)

        if isinstance(spaces[1].kind, HcurlSpaceType):
            D0 =   Gradient_2D(d_spaces[0], d_spaces[1])
            D1 = ScalarCurl_2D(d_spaces[1], d_spaces[2])

            d_spaces[0].diff = d_spaces[0].grad = D0
            d_spaces[1].diff = d_spaces[1].curl = D1
            
        else:
            D0 = VectorCurl_2D(d_spaces[0], d_spaces[1])
            D1 = Divergence_2D(d_spaces[1], d_spaces[2])

            d_spaces[0].diff = d_spaces[0].rot = D0
            d_spaces[1].diff = d_spaces[1].div = D1

    elif ldim == 3:

        d_spaces[0] = discretize_space(spaces[0], domain_h, *args, basis='B', **kwargs)
        d_spaces[1] = discretize_space(spaces[1], domain_h, *args, basis='M', **kwargs)
        d_spaces[2] = discretize_space(spaces[2], domain_h, *args, basis='M', **kwargs)
        d_spaces[3] = discretize_space(spaces[3], domain_h, *args, basis='M', **kwargs)

        D0 =   Gradient_3D(d_spaces[0], d_spaces[1])
        D1 =       Curl_3D(d_spaces[1], d_spaces[2])  
        D2 = Divergence_3D(d_spaces[2], d_spaces[3])

        d_spaces[0].diff = d_spaces[0].grad = D0
        d_spaces[1].diff = d_spaces[1].curl = D1
        d_spaces[2].diff = d_spaces[2].div  = D2

    return DiscreteDerham(*d_spaces)

#==============================================================================
# TODO multi patch
# TODO knots
def discretize_space(V, domain_h, *args, **kwargs):

    degree              = kwargs.pop('degree', None)
    basis               = kwargs.pop('basis', 'B')
    comm                = domain_h.comm
    kind                = V.kind
    ldim                = V.ldim
    periodic            = kwargs.pop('periodic', [False]*ldim)

    is_rational_mapping = False
    
    if isinstance(V, ProductSpace):
        kwargs['basis'] = basis
        basis = 'B'
    else:
        basis = basis

    # from a discrete geoemtry
    # TODO improve condition on mappings
    # TODO how to give a name to the mapping?
    if isinstance(domain_h, Geometry) and all(domain_h.mappings.values()):
        if len(domain_h.mappings.values()) > 1:
            raise NotImplementedError('Multipatch not yet available')

        mapping = list(domain_h.mappings.values())[0]

        g_spaces = [mapping.space]
        is_rational_mapping = isinstance( mapping, NurbsMapping )

        symbolic_mapping = Mapping('M', domain_h.pdim)

        if not( comm is None ) and ldim == 1:
            raise NotImplementedError('must create a TensorFemSpace in 1d')

    elif not( degree is None ):

        assert(hasattr(domain_h, 'ncells'))
        interiors = domain_h.domain.interior
        if isinstance(interiors, Union):
            interiors = interiors.args
            interfaces = domain_h.domain.interfaces

            if isinstance(interfaces, sym_Interface):
                interfaces = [interfaces]
            elif isinstance(interfaces, Union):
                interfaces = interfaces.args
            else:
                interfaces = []
        else:
            interiors = [interiors]

        if domain_h.domain.mapping is None:
            if len(interiors) == 1:
                symbolic_mapping = IdentityMapping('M_{}'.format(interiors[0].name), ldim)
            else:
                symbolic_mapping = {D:IdentityMapping('M_{}'.format(D.name), ldim) for D in interiors}
        else:
            if len(interiors) == 1:
                symbolic_mapping = domain_h.domain.mapping
            else:
                symbolic_mapping = domain_h.domain.mapping.mappings

        g_spaces = []
        for i,interior in enumerate(interiors):
            ncells     = domain_h.ncells
            min_coords = interior.min_coords
            max_coords = interior.max_coords

            assert(isinstance( degree, (list, tuple) ))
            assert( len(degree) == ldim )

            # Create uniform grid
            grids = [np.linspace(xmin, xmax, num=ne + 1)
                     for xmin, xmax, ne in zip(min_coords, max_coords, ncells)]

            # Create 1D finite element spaces and precompute quadrature data

            spaces = [SplineSpace( p, grid=grid , periodic=P) for p,grid, P in zip(degree, grids, periodic)]
            Vh     = None
            if i>0:
                for e in interfaces:
                    plus = e.plus.domain
                    minus = e.minus.domain
                    if plus == interior:
                        index = interiors.index(minus)
                    elif minus == interior:
                        index = interiors.index(plus)
                    else:
                        continue
                    if index<i:
                        nprocs = None
                        if comm is not None:
                            nprocs = g_spaces[index].vector_space.cart.nprocs
                        Vh = TensorFemSpace( *spaces, comm=comm, nprocs=nprocs, reverse_axis=e.axis)
                        break
                else:
                    Vh = TensorFemSpace( *spaces, comm=comm)
            else:
                Vh = TensorFemSpace( *spaces, comm=comm)

            if Vh is None:
                raise ValueError('Unable to discretize the space')

            if isinstance(kind, L2SpaceType):

                if ldim == 1:
                    Vh = Vh.reduce_degree(axes=[0], basis=basis)
                elif ldim == 2:
                    Vh = Vh.reduce_degree(axes=[0,1], basis=basis)
                elif ldim == 3:
                    Vh = Vh.reduce_degree(axes=[0,1,2], basis=basis)

            g_spaces.append(Vh)
    # Product and Vector spaces are constructed here

    if V.shape > 1:
        new_spaces = []
        for Vh in g_spaces:
            if isinstance(V, VectorFunctionSpace):

                if isinstance(kind, (H1SpaceType, L2SpaceType,  UndefinedSpaceType)):
                    spaces = [Vh for i in range(V.shape)]

                elif isinstance(kind, HcurlSpaceType):
                    if ldim == 2:
                        spaces = [Vh.reduce_degree(axes=[0], basis=basis),
                                  Vh.reduce_degree(axes=[1], basis=basis)]
                    elif ldim == 3:
                        spaces = [Vh.reduce_degree(axes=[0], basis=basis),
                                  Vh.reduce_degree(axes=[1], basis=basis),
                                  Vh.reduce_degree(axes=[2], basis=basis)]
                    else:
                        raise NotImplementedError('TODO')

                elif isinstance(kind, HdivSpaceType):
 
                    if ldim == 2:
                        spaces = [Vh.reduce_degree(axes=[1], basis=basis),
                                  Vh.reduce_degree(axes=[0], basis=basis)]
                    elif ldim == 3:
                        spaces = [Vh.reduce_degree(axes=[1,2], basis=basis),
                                  Vh.reduce_degree(axes=[0,2], basis=basis),
                                  Vh.reduce_degree(axes=[0,1], basis=basis)]
                    else:
                        raise NotImplementedError('TODO')

            elif isinstance(V, ProductSpace):
                spaces = []
                for Vi in V.spaces:
                    space = discretize_space(Vi, domain_h, *args, degree=degree,**kwargs)

                    if isinstance(space, ProductFemSpace):
                        spaces += list(space.spaces)
                    else:
                        spaces += [space]
            else:
                raise TypeError('space must be of type VectorSpace or ProductSpace got {}'.format(V))
            new_spaces += spaces

    else:
        new_spaces = g_spaces

    if len(new_spaces) == 1:
        Vh = new_spaces[0]
        setattr(Vh, 'shape', 1)
    else:
        Vh = ProductFemSpace(*new_spaces)
        setattr(Vh, 'shape', len(new_spaces))

    # add symbolic_mapping as a member to the space object
    setattr(Vh, 'symbolic_mapping', symbolic_mapping)
    setattr(Vh, 'is_rational_mapping', is_rational_mapping)
    setattr(Vh, 'symbolic_space', V)

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
        domain_h = args[0]
        assert( isinstance(domain_h, Geometry) )
        mapping     = domain_h.domain.mapping

        if isinstance(a, sym_Norm):
            kernel_expr = TerminalExpr(a)
            if not mapping is None:
                kernel_expr = tuple(LogicalExpr(i) for i in kernel_expr)
        else:
            if not mapping is None:
                a       = LogicalExpr (a)
            kernel_expr = TerminalExpr(a)
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
