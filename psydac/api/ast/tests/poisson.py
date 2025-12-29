#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os

from sympy import symbols
from sympy import sin, pi

from sympde.calculus import grad, dot
from sympde.topology import ScalarFunctionSpace
from sympde.topology import elements_of, LogicalExpr
from sympde.topology import Square
from sympde.topology import Mapping, IdentityMapping, PolarMapping
from sympde.expr     import integral
from sympde.expr     import LinearForm
from sympde.expr     import BilinearForm
from sympde.expr     import Norm, SemiNorm
from sympde.expr.evaluation import TerminalExpr

from psydac.api.ast.fem          import AST
from psydac.api.ast.parser       import parse
from psydac.api.discretization   import discretize
from psydac.api.printing.pycode  import pycode

# ...
try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']

except KeyError:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..','..')
    mesh_dir = os.path.join(base_dir, 'mesh')
    filename = os.path.join(mesh_dir, 'identity_2d.h5')


def test_codegen():
    domain  = Square()
    M       = Mapping('M',2)
    domain  = M(domain)
    V       = ScalarFunctionSpace('V', domain)
    u,v     = elements_of(V, names='u,v')

    x,y      = symbols('x, y')

    b        = LogicalExpr(M, BilinearForm((u,v), integral(domain, dot(grad(u), grad(v)))))
    l        = LogicalExpr(M, LinearForm(v, integral(domain, v*2*pi**2*sin(pi*x)*sin(pi*y))))

    # Create computational domain from topological domain
    domain_h = discretize(domain, filename=filename)

    # Discrete spaces
    Vh = discretize(V, domain_h)

    error  = u - sin(pi*x)*sin(pi*y)
    l2norm = LogicalExpr(M,     Norm(error, domain, kind='l2'))
    h1norm = LogicalExpr(M, SemiNorm(error, domain, kind='h1'))

    ast_b = AST(b, TerminalExpr(b)[0],[Vh, Vh])
    ast_b = parse(ast_b.expr, settings={'dim':2, 'nderiv':1, 'mapping':M, 'target':domain.logical_domain})
    print(pycode(ast_b))

    print('==============================================================================================================')
    ast_l = AST(l, TerminalExpr(l)[0], Vh)
    ast_l = parse(ast_l.expr, settings={'dim':2, 'nderiv':1, 'mapping':M, 'target':domain.logical_domain})
    print(pycode(ast_l))


    print('==============================================================================================================')
    ast_l2norm = AST(l2norm, TerminalExpr(l2norm)[0], Vh)
    ast_l2norm = parse(ast_l2norm.expr, settings={'dim':2, 'nderiv':1, 'mapping':M, 'target':domain.logical_domain})
    print(pycode(ast_l2norm))

test_codegen()
