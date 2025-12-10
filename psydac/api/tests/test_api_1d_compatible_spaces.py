# -*- coding: UTF-8 -*-

from sympy               import pi, sin
from scipy.sparse.linalg import spsolve

from sympde.calculus import dot, div
from sympde.topology import ScalarFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import element_of
from sympde.topology import Line
from sympde.expr     import BilinearForm, LinearForm
from sympde.expr     import integral
from sympde.expr     import Norm
from sympde.expr     import find

from psydac.fem.basic          import FemField
from psydac.api.discretization import discretize

#==============================================================================

def run_system_1_1d_dir(f0, sol, ncells, degree):
    # ... abstract model
    domain = Line()

    V1 = ScalarFunctionSpace('V1', domain, kind='Hdiv')
    V2 = ScalarFunctionSpace('V2', domain, kind='L2')
    X  = ProductSpace(V1, V2)

    x = domain.coordinates

    F = element_of(V2, name='F')

    p,q = [element_of(V1, name=i) for i in ['p', 'q']]
    u,v = [element_of(V2, name=i) for i in ['u', 'v']]
    
    int_0 = lambda expr: integral(domain , expr)

    a  = BilinearForm(((p,u),(q,v)), int_0(dot(p,q) + dot(div(q),u) + dot(div(p),v)) )
    l  = LinearForm((q,v), int_0(dot(f0, v)))

    error = F-sol
    l2norm_F = Norm(error, domain, kind='l2')


    equation = find([p,u], forall=[q,v], lhs=a((p,u),(q,v)), rhs=l(q,v))
    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells)
    # ...

    # ... discrete spaces
    V1h = discretize(V1, domain_h, degree=degree)
    V2h = discretize(V2, domain_h, degree=degree)
    Xh  = discretize(X , domain_h, degree=degree)
    # ... dsicretize the equation using Dirichlet bc
    ah = discretize(equation, domain_h, [Xh, Xh], symbolic_space=[X, X])

    # ... discretize norms
    l2norm_F_h = discretize(l2norm_F, domain_h, V2h)


    ah.assemble()

    M   = ah.linear_system.lhs.tosparse().tocsc()
    rhs = ah.linear_system.rhs.toarray()
    sol = spsolve(M, rhs)

    phi2 = FemField(V2h)    
    phi2.coeffs[0:V2h.nbasis] = sol[V1h.nbasis:]
    
    l2_error = l2norm_F_h.assemble(F=phi2)

    return l2_error

###############################################################################
#            SERIAL TESTS
###############################################################################

#==============================================================================

def test_api_system_1_1d_dir_1():

    from sympy.abc import x

    f0 = -(2*pi)**2*sin(2*pi*x)
    u  = sin(2*pi*x)
    x  = run_system_1_1d_dir(f0, u,ncells=[10], degree=[2])
