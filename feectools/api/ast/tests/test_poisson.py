#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os
from pathlib import Path

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
    domain  = Square()
    M       = Mapping('M',2)
    domain  = M(domain)
    V       = ScalarFunctionSpace('V', domain)
    u,v     = elements_of(V, names='u,v')

    x,y     = symbols('x, y')

    b       = LogicalExpr(BilinearForm((u,v), integral(domain, dot(grad(u), grad(v)))), domain)
    l       = LogicalExpr(LinearForm(v, integral(domain, v*2*pi**2*sin(pi*x)*sin(pi*y))), domain)

    # Create computational domain from topological domain
    domain_h = discretize(domain, filename=filename)

    # Discrete spaces
    Vh = discretize(V, domain_h)

    error  = u - sin(pi*x)*sin(pi*y)
    l2norm = LogicalExpr(    Norm(error, domain, kind='l2'), domain)
    h1norm = LogicalExpr(SemiNorm(error, domain, kind='h1'), domain)

    ast_b = AST(b, TerminalExpr(b, domain)[0], [Vh, Vh], nquads=(2, 2), backend=backend)
    ast_b = parse(ast_b.expr, settings={'dim':2, 'nderiv':1, 'mapping':M, 'target':domain.logical_domain}, backend=backend)
    print(pycode(ast_b))

    print('==============================================================================================================')
    ast_l = AST(l, TerminalExpr(l, domain)[0], Vh, nquads=(2, 2), backend=backend)
    ast_l = parse(ast_l.expr, settings={'dim':2, 'nderiv':1, 'mapping':M, 'target':domain.logical_domain}, backend=backend)
    print(pycode(ast_l))

    print('==============================================================================================================')
    ast_l2norm = AST(l2norm, TerminalExpr(l2norm, domain)[0], Vh, nquads=(2, 2), backend=backend)
    ast_l2norm = parse(ast_l2norm.expr, settings={'dim':2, 'nderiv':1, 'mapping':M, 'target':domain.logical_domain}, backend=backend)
    print(pycode(ast_l2norm))

#==============================================================================
if __name__ == '__main__':
    test_codegen()
