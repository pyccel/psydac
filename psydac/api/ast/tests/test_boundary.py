#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os
from pathlib import Path

from sympy import symbols
from sympy import Tuple

from sympde.calculus import grad, dot
from sympde.topology import ScalarFunctionSpace
from sympde.topology import elements_of
from sympde.topology import Square
from sympde.topology import IdentityMapping #,Mapping, PolarMapping
from sympde.expr     import integral
from sympde.expr     import LinearForm
from sympde.expr     import BilinearForm
from sympde.topology import NormalVector
from sympde.expr.evaluation import TerminalExpr

from psydac.api.ast.fem          import AST
from psydac.api.ast.parser       import parse
from psydac.api.discretization   import discretize
from psydac.api.printing.pycode  import pycode
from psydac.api.settings         import PSYDAC_BACKENDS

#==============================================================================
# Get the mesh directory
import psydac.cad.mesh as mesh_mod
mesh_dir = Path(mesh_mod.__file__).parent
filename = os.path.join(mesh_dir, 'identity_2d.h5')

# Choose backend
backend = PSYDAC_BACKENDS['python']

#==============================================================================
def test_codegen():
    kappa = 10**15
    domain  = Square()
    mapping = IdentityMapping('M',2)
    nn      = NormalVector('nn')
    V       = ScalarFunctionSpace('V', domain)
    u,v     = elements_of(V, names='u,v')

    x,y     = symbols('x, y')
    B       = domain.get_boundary(axis=0,ext=1)
    int_1   = lambda expr: integral(B, expr)
    int_0   = lambda expr: integral(domain, expr)
    g       = Tuple(x**2, y**2)

    b       = BilinearForm((u,v), int_1(v*dot(grad(u), nn)))
    l       = LinearForm(v, int_1(-x*y*(1-y)*dot(grad(v),nn)))

    # Create computational domain from topological domain
    domain_h = discretize(domain, ncells=[2**2,2**2])

    # Discrete spaces
    Vh = discretize(V, domain_h, degree=[1,1])
    print(TerminalExpr(b, domain)[0])

    ast_b = AST(b, TerminalExpr(b, domain)[0], [Vh, Vh], nquads=(2, 2), backend=backend)
    ast_b = parse(ast_b.expr, settings={'dim':2,'nderiv':1,'mapping':mapping, 'target':B}, backend=backend)
    print(pycode(ast_b))

    print('==============================================================================================================')
    ast_l = AST(l, TerminalExpr(l, domain)[0], Vh, nquads=(2, 2), backend=backend)
    ast_l = parse(ast_l.expr, settings={'dim':2,'nderiv':1,'mapping':mapping,'target':B}, backend=backend)
    print(pycode(ast_l))

#==============================================================================
if __name__ == '__main__':
    test_codegen()
