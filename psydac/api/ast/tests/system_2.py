# -*- coding: UTF-8 -*-

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
    ast_b    = AST(b, TerminalExpr(b)[0], [Vh, Vh])
    stmt_b = parse(ast_b.expr, settings={'dim':2,'nderiv':1, 'mapping':Vh.symbolic_mapping})
    print(pycode(stmt_b))

    print('============================================LinearForm===========================================')
    ast_l    = AST(l, TerminalExpr(l)[0], Vh)
    stmt_l = parse(ast_l.expr, settings={'dim':2,'nderiv':1, 'mapping':Vh.symbolic_mapping})
    print(pycode(stmt_l))

