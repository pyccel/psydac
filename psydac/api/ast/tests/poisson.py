# -*- coding: UTF-8 -*-


from sympy import symbols
from sympy import sin,pi

from sympde.calculus import grad, dot

from sympde.topology import ScalarFunctionSpace
from sympde.topology import elements_of
from sympde.topology import Square
from sympde.topology import IdentityMapping#,Mapping PolarMapping
from sympde.expr     import integral
from sympde.expr     import LinearForm
from sympde.expr     import BilinearForm
from sympde.expr     import Norm
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
    domain  = Square()
    #mapping = IdentityMapping('M',2, c1= 1., c2= 3., rmin = 1., rmax=2.)
    V       = ScalarFunctionSpace('V', domain)
    u,v     = elements_of(V, names='u,v')

    x,y      = symbols('x, y')

    b        = BilinearForm((u,v), integral(domain, dot(grad(u), grad(v))))
    l        = LinearForm(v, integral(domain, v*2*pi**2*sin(pi*x)*sin(pi*y)))

    # Create computational domain from topological domain
    domain_h = discretize(domain, filename=filename)

    # Discrete spaces
    Vh = discretize(V, domain_h)

    error  = u - sin(pi*x)*sin(pi*y)
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    ast_b = AST(b, TerminalExpr(b)[0],[Vh, Vh])
    ast_b = parse(ast_b.expr, settings={'dim':2,'nderiv':1,'mapping':Vh.symbolic_mapping})
    print(pycode(ast_b))

    print('==============================================================================================================')
    ast_l = AST(l, TerminalExpr(l)[0], Vh)
    ast_l = parse(ast_l.expr, settings={'dim':2,'nderiv':1,'mapping':Vh.symbolic_mapping})
    print(pycode(ast_l))


    print('==============================================================================================================')
    ast_l2norm = AST(l2norm, TerminalExpr(l2norm)[0], Vh)
    ast_l2norm = parse(ast_l2norm.expr, settings={'dim':2,'nderiv':1,'mapping':Vh.symbolic_mapping})
    print(pycode(ast_l2norm))


