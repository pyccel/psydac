#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os
from pathlib import Path

from sympy import cos, sin

from sympde.calculus import dot, div
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import element_of
from sympde.topology import Square
from sympde.topology import Mapping
from sympde.expr     import integral
from sympde.expr     import LinearForm
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

    V1 = VectorFunctionSpace('V1', domain, kind='Hdiv')
    V2 = ScalarFunctionSpace('V2', domain, kind='L2')

    x,y = domain.coordinates

    F = element_of(V2, name='F')


    p,q = [element_of(V1, name=i) for i in ['p', 'q']]
    u,v = [element_of(V2, name=i) for i in ['u', 'v']]

    int_0 = lambda expr: integral(domain , expr)

    f1     = cos(x)*sin(y)
    f2     = sin(2*x)*sin(2*y)

    b  = BilinearForm(((p,u),(q,v)), int_0(dot(p,q) + div(q)*u + div(p)*v) )
    l  = LinearForm((q,v), int_0(f1*q[0]+f2*q[1]+v))

    # Create computational domain from topological domain
    domain_h = discretize(domain, filename=filename)

    # Discrete spaces
    Vh = discretize(V1*V2, domain_h)

    print('============================================BilinearForm=========================================')
    ast_b    = AST(b, TerminalExpr(b, domain)[0], [Vh, Vh], nquads=(3, 3), backend=backend)
    stmt_b = parse(ast_b.expr, settings={'dim':2, 'nderiv':1, 'mapping':M, 'target':domain}, backend=backend)
    print(pycode(stmt_b))

    print('============================================LinearForm===========================================')
    ast_l    = AST(l, TerminalExpr(l, domain)[0], Vh, nquads=(3, 3), backend=backend)
    stmt_l = parse(ast_l.expr, settings={'dim':2, 'nderiv':1, 'mapping':M, 'target':domain}, backend=backend)
    print(pycode(stmt_l))

#==============================================================================
if __name__ == '__main__':
    test_codegen()
