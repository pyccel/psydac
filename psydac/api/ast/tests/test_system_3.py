#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os
from pathlib import Path

from sympy import pi, sin, Tuple

from sympde.calculus import div
from sympde.topology import VectorFunctionSpace
from sympde.topology import element_of
from sympde.topology import Square
from sympde.topology import Mapping#, IdentityMapping, PolarMapping
from sympde.expr     import integral
from sympde.expr     import BilinearForm
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
    domain = Square()
    M      = Mapping('M', domain.dim)
    V = VectorFunctionSpace('W', domain,kind='hdiv')

    x,y = domain.coordinates

    f = Tuple(2*pi**2*sin(pi*x)*sin(pi*y),
              2*pi**2*sin(pi*x)*sin(pi*y))

    Fe = Tuple(sin(pi*x)*sin(pi*y), sin(pi*x)*sin(pi*y))

    F = element_of(V, name='F')

    u,v = [element_of(V, name=i) for i in ['u', 'v']]

    int_0 = lambda expr: integral(domain , expr)

    b = BilinearForm((u,v), int_0(div(v)*div(u)))

    # Create computational domain from topological domain
    domain_h = discretize(domain, filename=filename)

    # Discrete spaces
    Vh = discretize(V, domain_h)

    print('============================================BilinearForm=========================================')
    ast_b  = AST(b, TerminalExpr(b, domain)[0], [Vh, Vh], nquads=(3, 3), backend=backend)
    stmt_b = parse(ast_b.expr, settings={'dim':2, 'nderiv':1, 'mapping':M, 'target':domain}, backend=backend)
    print(pycode(stmt_b))

#==============================================================================
if __name__ == '__main__':
    test_codegen()
