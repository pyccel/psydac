# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin, Tuple, Matrix

from sympde.core import Constant
from sympde.calculus import grad, dot, inner, cross, rot, curl, div
from sympde.calculus import laplace, hessian
from sympde.topology import (dx, dy, dz)
from sympde.topology import FunctionSpace, VectorFunctionSpace
from sympde.topology import ScalarField, VectorField
from sympde.topology import ProductSpace
from sympde.topology import ScalarTestFunction
from sympde.topology import VectorTestFunction
from sympde.topology import Boundary, NormalVector, TangentVector
from sympde.topology import Domain, Line, Square, Cube
from sympde.topology import Trace, trace_0, trace_1
from sympde.topology import Union
from sympde.expr import BilinearForm, LinearForm
from sympde.expr import Norm
from sympde.expr import find, EssentialBC

from psydac.fem.basic   import FemField
from psydac.fem.vector   import VectorFemField
from psydac.api.discretization import discretize

from numpy import linspace, zeros, allclose
import numpy as np
from mpi4py import MPI
import pytest

#==============================================================================

def run_system_1_1d_dir(f0, ncells, degree):

    # ... abstract model
    domain = Line()

    V1 = FunctionSpace('V1', domain, kind='Hdiv')
    V2 = FunctionSpace('V2', domain, kind='L2')
    X  = ProductSpace(V1, V2)

    x = domain.coordinates


    p,q = [ScalarTestFunction(V1, name=i) for i in ['p', 'q']]
    u,v = [ScalarTestFunction(V2, name=i) for i in ['u', 'v']]



    a  = BilinearForm(((p,u),(q,v)),dot(p,q) + dot(div(q),u) + dot(div(p),v) )

    l  = LinearForm((q,v), dot(f0, v))

 
    bc = EssentialBC(p, 0, domain.boundary)
    equation = find([p,u], forall=[q,v], lhs=a((p,u),(q,v)), rhs=l(q,v), bc=bc)
    # ...


    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells)
    # ...

    # ... discrete spaces
    V1h = discretize(V1, domain_h, degree=degree)
    V2h = discretize(V2, domain_h, degree=degree)
    Xh  = discretize(X , domain_h, degree=degree)
   
    # ... dsicretize the equation using Dirichlet bc
    ah = discretize(a, domain_h, [Xh, Xh], symbolic_space=[X, X])
    # ...
    
    a=ah.assemble()
    print(a)

    return x

###############################################################################
#            SERIAL TESTS
###############################################################################

#==============================================================================
@pytest.mark.skip
def test_api_system_1_1d_dir_1():

    from sympy.abc import x

    f0 =  sin(2*pi*x)


    x = run_system_1_1d_dir(f0, ncells=[10], degree=[2])


