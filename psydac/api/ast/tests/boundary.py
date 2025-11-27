#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
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

import os
# ...

try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']

except KeyError:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..','..')
    mesh_dir = os.path.join(base_dir, 'mesh')
    filename = os.path.join(mesh_dir, 'identity_2d.h5')

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
    Vh = discretize(V, domain_h, degree=[1,1], mapping=mapping)
    print(TerminalExpr(b)[0])
    ast_b = AST(b, TerminalExpr(b)[0], [Vh, Vh])
    ast_b = parse(ast_b.expr, settings={'dim':2,'nderiv':1,'mapping':Vh.symbolic_mapping, 'target':B})
    print(pycode(ast_b))

    print('==============================================================================================================')
    ast_l = AST(l, TerminalExpr(l)[0], Vh)
    ast_l = parse(ast_l.expr, settings={'dim':2,'nderiv':1,'mapping':Vh.symbolic_mapping,'target':B})
    print(pycode(ast_l))

