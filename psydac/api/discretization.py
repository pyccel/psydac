# coding: utf-8

# TODO: - init_fem is called whenever we call discretize. we should check that
#         nderiv has not been changed. shall we add quad_order too?

from collections import namedtuple
from collections import OrderedDict

from sympy import Expr as sym_Expr
import numpy as np

from sympde.expr     import BasicForm as sym_BasicForm
from sympde.expr     import BilinearForm as sym_BilinearForm
from sympde.expr     import LinearForm as sym_LinearForm
from sympde.expr     import Functional as sym_Functional
from sympde.expr     import Integral
from sympde.expr     import Equation as sym_Equation
from sympde.expr     import Norm as sym_Norm
from sympde.expr     import TerminalExpr
from sympde.calculus import dot
from sympde.topology import Domain, Interface
from sympde.topology import Line, Square, Cube
from sympde.topology import BasicFunctionSpace
from sympde.topology import ScalarFunctionSpace, ScalarFunction
from sympde.topology import VectorFunctionSpace, VectorFunction
from sympde.topology import ProductSpace
from sympde.topology import Derham
from sympde.topology import Mapping, IdentityMapping, LogicalExpr
from sympde.topology import H1SpaceType, HcurlSpaceType, HdivSpaceType, L2SpaceType, UndefinedSpaceType
from sympde.topology.basic import Union

from gelato.expr import GltExpr as sym_GltExpr

from psydac.api.basic                import BasicDiscrete
from psydac.api.fem                  import DiscreteBilinearForm
from psydac.api.fem                  import DiscreteLinearForm
from psydac.api.fem                  import DiscreteFunctional
from psydac.api.fem                  import DiscreteSumForm
from psydac.api.feec                 import DiscreteDerham
from psydac.api.glt                  import DiscreteGltExpr
from psydac.api.expr                 import DiscreteExpr
from psydac.api.essential_bc         import apply_essential_bc
from psydac.api.utilities            import flatten
from psydac.linalg.iterative_solvers import cg, pcg, bicg
from psydac.fem.basic                import FemField
from psydac.fem.splines              import SplineSpace
from psydac.fem.tensor               import TensorFemSpace
from psydac.fem.vector               import ProductFemSpace
from psydac.cad.geometry             import Geometry
from psydac.mapping.discrete         import NurbsMapping

__all__ = ('discretize',)

#==============================================================================
LinearSystem = namedtuple('LinearSystem', ['lhs', 'rhs'])

#==============================================================================
_default_solver = {'solver':'cg', 'tol':1e-9, 'maxiter':3000, 'verbose':False}

def driver_solve(L, **kwargs):
    if not isinstance(L, LinearSystem):
        raise TypeError('> Expecting a LinearSystem object')

    M = L.lhs
    rhs = L.rhs

    name        = kwargs.pop('solver')
    return_info = kwargs.pop('info', False)

    if name == 'cg':
        x, info = cg( M, rhs, **kwargs )
    elif name == 'pcg':
        x, info = pcg( M, rhs, **kwargs )
    elif name == 'bicg':
        x, info = bicg( M, M.T, rhs, **kwargs )
    else:
        raise NotImplementedError("Solver '{}' is not available".format(name))

    return (x, info) if return_info else x

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
                apply_essential_bc(M, *self.bc)
        else:
            M = self.linear_system.lhs

        if assemble_rhs:
            rhs = self.rhs.assemble(**kwargs)
            if self.bc:
                apply_essential_bc(rhs, *self.bc)

        else:
            rhs = self.linear_system.rhs

        self._linear_system = LinearSystem(M, rhs)

    def solve(self, **kwargs):

        rhs = kwargs.pop('rhs', None)
        if rhs:
            kwargs['assemble_rhs'] = False

        self.assemble(**kwargs)

        free_args = set(self.lhs.free_args + self.rhs.free_args)
        for e in free_args: kwargs.pop(e, None)

        if rhs:
            L = self.linear_system
            L = LinearSystem(L.lhs, rhs)
            self._linear_system = L

        settings = {k:kwargs[k] if k in kwargs else it for k,it in _default_solver.items()}
        settings.update({it[0]:it[1] for it in kwargs.items() if it[0] not in settings})

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

        if self.bc:

            # Inhomogeneous Dirichlet boundary conditions
            idbcs = [i for i in self.bc if i.rhs != 0]

            if idbcs:

                from sympde.expr import integral
                from sympde.expr import find
                from sympde.topology import element_of #, ScalarFunction

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

                # Compute product of (u, v) using dot product for vector quantities
                product  = lambda f, g: (f * g if isinstance(g, ScalarFunction) else dot(f, g))

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
                loc_settings.pop('info', False)
                loc_settings.pop('pc'  , False)
                uh = equation_h.solve(**loc_settings)

                # Use inhomogeneous solution as initial guess to solver
                settings['x0'] = uh.coeffs

        #----------------------------------------------------------------------


        if settings.get('info', False):
            X, info = driver_solve(self.linear_system, **settings)
            uh = FemField(self.trial_space, coeffs=X)
            return uh, info

        else:
            X  = driver_solve(self.linear_system, **settings)
            uh = FemField(self.trial_space, coeffs=X)
            return uh

#==============================================================================           
def discretize_derham(derham, domain_h, *args, **kwargs):

    ldim     = derham.shape
    mapping  = derham.spaces[0].domain.mapping

    bases  = ['B'] + ldim * ['M']
    spaces = [discretize_space(V, domain_h, *args, basis=basis, **kwargs) \
            for V, basis in zip(derham.spaces, bases)]

    return DiscreteDerham(mapping, *spaces)

#==============================================================================
def reduce_space_degrees(V, Vh, basis='B'):

    if isinstance(V.kind, HcurlSpaceType):
        if V.ldim == 2:
            spaces = [Vh.reduce_degree(axes=[0], basis=basis),
                      Vh.reduce_degree(axes=[1], basis=basis)]
        elif V.ldim == 3:
            spaces = [Vh.reduce_degree(axes=[0], basis=basis),
                      Vh.reduce_degree(axes=[1], basis=basis),
                      Vh.reduce_degree(axes=[2], basis=basis)]
        else:
            raise NotImplementedError('TODO')

        Vh = ProductFemSpace(*spaces)
    elif isinstance(V.kind, HdivSpaceType):

        if V.ldim == 2:
            spaces = [Vh.reduce_degree(axes=[1], basis=basis),
                      Vh.reduce_degree(axes=[0], basis=basis)]
        elif V.ldim == 3:
            spaces = [Vh.reduce_degree(axes=[1,2], basis=basis),
                      Vh.reduce_degree(axes=[0,2], basis=basis),
                      Vh.reduce_degree(axes=[0,1], basis=basis)]
        else:
            raise NotImplementedError('TODO')

        Vh = ProductFemSpace(*spaces)

    elif isinstance(V.kind, L2SpaceType):
        if V.ldim == 1:
            Vh = Vh.reduce_degree(axes=[0], basis=basis)
        elif V.ldim == 2:
            Vh = Vh.reduce_degree(axes=[0,1], basis=basis)
        elif V.ldim == 3:
            Vh = Vh.reduce_degree(axes=[0,1,2], basis=basis)

    elif not isinstance(V.kind,  (H1SpaceType, UndefinedSpaceType)):
        raise NotImplementedError('TODO')

    if isinstance(V, VectorFunctionSpace):
        if isinstance(V.kind, (H1SpaceType, L2SpaceType, UndefinedSpaceType)):
            Vh = ProductFemSpace(*[Vh]*V.ldim)

    return Vh

#==============================================================================
# TODO knots
def discretize_space(V, domain_h, *args, **kwargs):
    """
    This function creates the discretized space starting from the symbolic space.

    Parameters
    ----------

    V : <FunctionSpace>
        the symbolic space

    domain_h   : <Geometry>
        the discretized domain

    Returns
    -------
    Vh : <FemSpace>
        represents the discrete fem space

    """

#    we have two two cases, the case where we have a geometry file,
#    and the case where we have either an analytical mapping or without the mapping.
#    We build the dictionary g_spaces for each interior domain, where it conatians the interiors as keys and the spaces as values,
#    we then create the compatible spaces if needed with the suitable basis functions.

    degree              = kwargs.pop('degree', None)
    comm                = domain_h.comm
    ldim                = V.ldim
    periodic            = kwargs.pop('periodic', [False]*ldim)
    basis               = kwargs.pop('basis', 'B')
    is_rational_mapping = False

    # from a discrete geoemtry
    # TODO improve condition on mappings
    # TODO how to give a name to the mapping?

    g_spaces = OrderedDict()
    if isinstance(domain_h, Geometry) and all(domain_h.mappings.values()):
        if len(domain_h.mappings.values()) > 1:
            raise NotImplementedError('Multipatch not yet available')

        interiors = [domain_h.domain.interior]
        mappings  = [domain_h.mappings[inter.logical_domain.name] for inter in interiors]

        spaces    = [m.space for m in mappings]
        g_spaces  = OrderedDict(zip(interiors, spaces))

        is_rational_mapping = all(isinstance( mapping, NurbsMapping ) for mapping in mappings)
        symbolic_mapping    = Mapping('M', domain_h.pdim)

        if not( comm is None ) and ldim == 1:
            raise NotImplementedError('must create a TensorFemSpace in 1d')

    elif not( degree is None ):

        assert(hasattr(domain_h, 'ncells'))
        interiors = domain_h.domain.interior
        if isinstance(interiors, Union):
            interiors = interiors.args
            interfaces = domain_h.domain.interfaces

            if isinstance(interfaces, Interface):
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
                            nprocs = g_spaces[interiors[index]].vector_space.cart.nprocs
                        Vh = TensorFemSpace( *spaces, comm=comm, nprocs=nprocs, reverse_axis=e.axis)
                        break
                else:
                    Vh = TensorFemSpace( *spaces, comm=comm)
            else:
                Vh = TensorFemSpace( *spaces, comm=comm)

            if Vh is None:
                raise ValueError('Unable to discretize the space')

            g_spaces[interior] = Vh

    for inter in g_spaces:
        Vh = g_spaces[inter]
        if isinstance(V, ProductSpace):
            spaces = [reduce_space_degrees(Vi, Vh, basis=basis) for Vi in V.spaces]
            spaces = [Vh.spaces if isinstance(Vh, ProductFemSpace) else Vh for Vh in spaces]
            spaces = flatten(spaces)
            Vh     = ProductFemSpace(*spaces)
        else:
            Vh = reduce_space_degrees(V, Vh, basis=basis)

        setattr(Vh, 'symbolic_domain', inter)
        setattr(Vh, 'symbolic_space', V)
        g_spaces[inter] = Vh

    Vh = ProductFemSpace(*g_spaces.values())
    # add symbolic_mapping as a member to the space object
    setattr(Vh, 'symbolic_mapping', symbolic_mapping)
    setattr(Vh, 'is_rational_mapping', is_rational_mapping)
    setattr(Vh, 'symbolic_space', V)
    setattr(Vh, 'symbolic_domain', domain_h.domain)

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
        domain  = domain_h.domain

        if isinstance(a, sym_Norm):
            kernel_expr = TerminalExpr(a, domain)
            if not domain.mapping is None:
                kernel_expr = tuple(LogicalExpr(i, domain) for i in kernel_expr)
        else:
            if not domain.mapping is None:
                a       = LogicalExpr (a, domain)
            kernel_expr = TerminalExpr(a, domain)

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
