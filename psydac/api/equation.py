#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from collections import namedtuple

from sympde.topology import ScalarFunction
from sympde.topology import ProductSpace
from sympde.topology import element_of
from sympde.calculus import dot
from sympde.expr     import integral
from sympde.expr     import Integral
from sympde.expr     import find
from sympde.expr     import Equation

from psydac.api.basic                import BasicDiscrete
from psydac.api.essential_bc         import apply_essential_bc
from psydac.fem.basic                import FemField
from psydac.linalg.solvers           import inverse

__all__ = ('l2_boundary_projection', 'DiscreteEquation')

#==============================================================================
LinearSystem = namedtuple('LinearSystem', ['lhs', 'rhs'])

#==============================================================================
_default_solver = {'solver':'cg', 'tol':1e-9, 'maxiter':3000, 'verbose':False}

def l2_boundary_projection(equation):
    """
    Create an auxiliary equation (weak formulation) that solves for the
    boundary trace of the solution by performing an L2 projection of
    inhomogeneous essential boundary conditions.

    Return None if no inhomogeneous essential BCs are imposed on the solution.

    Parameters
    ----------
    equation : sympde.expr.Equation
        Weak formulation of PDE of interest, which may have essential BCs.

    Returns
    -------
    eqn_bc : sympde.expr.Equation
        Weak formulation that performs L2 projection of inhomogeneous essential
        boundary conditions onto the trial space. None if not needed.

    """
    if not isinstance(equation, Equation):
        raise TypeError('> Expecting a symbolic Equation')

    if not equation.bc:
        return None

    # Inhomogeneous Dirichlet boundary conditions
    idbcs = [i for i in equation.bc if i.rhs != 0]

    if not idbcs:
        return None

    # Extract trial functions from model equation
    u = equation.trial_functions

    # Create test functions in same space of trial functions
    # TODO: check if we should generate random names
    V = ProductSpace(*[ui.space for ui in u])
    v = element_of(V, name='v:{}'.format(len(u)))

    # In a system, each essential boundary condition is applied to
    # only one component (bc.variable) of the state vector. Hence
    # we will select the correct test function using a dictionary.
    test_dict = dict(zip(u, v))

    # Compute product of (u, v) using dot product for vector quantities
    product  = lambda f, g: (f * g if g.atoms(ScalarFunction) else dot(f, g))

    # Construct variational formulation that performs L2 projection
    # of boundary conditions onto the correct space
    factor   = lambda bc : bc.lhs.xreplace(test_dict)
    lhs_expr = sum(integral(i.boundary, product(i.lhs, factor(i))) for i in idbcs)
    rhs_expr = sum(integral(i.boundary, product(i.rhs, factor(i))) for i in idbcs)
    eqn_bc   = find(u, forall=v, lhs=lhs_expr, rhs=rhs_expr)

    return eqn_bc

#==============================================================================
class DiscreteEquation(BasicDiscrete):

    def __init__(self, expr, *args, **kwargs):
        if not isinstance(expr, Equation):
            raise TypeError('> Expecting a symbolic Equation')

        # Warning: circular dependency
        from psydac.api.discretization import discretize

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

        self._lhs = discretize(expr.lhs, domain, trial_test, **kwargs)
        # ...
        # sum_factorization (sum factorization algorithm) is currently not supported for LinearForms (the rhs)
        kwargs_rhs = kwargs.copy()
        if 'sum_factorization' in kwargs:
            kwargs_rhs.pop('sum_factorization')
        self._rhs = discretize(expr.rhs, domain, test_space, **kwargs_rhs)
        # ...

        # ...
        # Create boundary equation (None if not needed)
        eqn_bc   = l2_boundary_projection(expr)
        eqn_bc_h = DiscreteEquation(eqn_bc, domain, [trial_space, trial_space], **kwargs) \
                if eqn_bc else None
        # ...

        self._bc                = bc
        self._linear_system     = None
        self._domain            = domain
        self._trial_space       = trial_space
        self._test_space        = test_space
        self._boundary_equation = eqn_bc_h
        self._solver_parameters = _default_solver.copy()

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

    @property
    def boundary_equation(self):
        return self._boundary_equation

    def set_solver(self, solver, **kwargs):
        self._solver_parameters.update(solver=solver, **kwargs)

    def get_solver(self):
        return self._solver_parameters

    #--------------------------------------------------------------------------
    def assemble(self, **kwargs):

        # Decide if we should assemble
        assemble_lhs = not self.linear_system or self.lhs.free_args
        assemble_rhs = not self.linear_system or self.rhs.free_args

        # Matrix (left-hand side)
        if assemble_lhs:
            A = self.lhs.assemble(reset=True, **kwargs)
            if self.bc:
                apply_essential_bc(A, *self.bc)
        else:
            A = self.linear_system.lhs

        # Vector (right-hand side)
        if assemble_rhs:
            b = self.rhs.assemble(reset=True, **kwargs)
            if self.bc:
                apply_essential_bc(b, *self.bc)
        else:
            b = self.linear_system.rhs

        # Store linear system
        self._linear_system = LinearSystem(A, b)

    #--------------------------------------------------------------------------
    def solve(self, **kwargs):

        self.assemble(**kwargs)

        # Free arguments of current equation
        free_args = set(self.lhs.free_args + self.rhs.free_args)

        # Free arguments of boundary equation
        if self.boundary_equation:
            bc_eq = self.boundary_equation
            free_args_bc = set(bc_eq.lhs.free_args + bc_eq.rhs.free_args)
        else:
            free_args_bc = set()

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

        settings = self.get_solver()
        if self.boundary_equation:

            # Find inhomogeneous solution (use CG as system is symmetric)
            self.boundary_equation.set_solver('cg', info=False)
            uh = self.boundary_equation.solve(**kwargs)

            # Use inhomogeneous solution as initial guess to solver
            settings['x0'] = uh.coeffs
        #----------------------------------------------------------------------
        L = self.linear_system
        M = L.lhs
        rhs = L.rhs

        solver = settings.get('solver')
        solver_settings = settings.copy()
        solver_settings.pop('solver')

        if 'info' in settings:
            inf = settings.get('info')
            solver_settings.pop('info')
        else:
            inf = False

        M_inv = inverse(M, solver, **solver_settings)
        if inf == True:
            X = M_inv @ rhs
            uh = FemField(self.trial_space, coeffs=X)
            info = M_inv.get_info()
            return uh, info
        else:
            X = M_inv @ rhs
            uh = FemField(self.trial_space, coeffs=X)
            return uh
