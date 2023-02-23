from sympde.topology     import Square, Derham, element_of
from sympde.expr.expr    import BilinearForm, integral, LinearForm
from sympde.calculus     import Dot 
from psydac.feec.multipatch.api import discretize
from psydac.api.settings        import PSYDAC_BACKENDS
from psydac.fem.basic                 import FemField
from sympy               import Tuple


def test_axis_projection():
    domain=Square('OmegaLog', bounds1=(0,1), bounds2 = (0,1))
    derham  = Derham(domain, ["H1", "Hdiv", "L2"])
    domain_h = discretize(domain, ncells=[4,4], periodic=[True,True])
    derham_h = discretize(derham, domain_h, degree=(2,2))

    V1h = derham_h.V1
    V2h = derham_h.V2

    u   = element_of(V1h.symbolic_space, name='u')
    f   = element_of(V2h.symbolic_space, name='f')
    expr = u[0]*f
    Pei = BilinearForm((u,f), integral(domain, expr))
    pei = discretize(Pei, domain_h, (V1h,V2h), backend=PSYDAC_BACKENDS['python'])
    Peih = pei.assemble()
    uh = V1h.vector_space.zeros()
    test = Peih.dot(uh)

def test_assembly(nc=4, deg=4):
    """
    Solve the incompressible Navier-Stokes equations with structure preserving conga method
    """
    ncells = [nc, nc]
    degree = [deg, deg]
    backend_language='python'

    domain = Square("Omega", bounds1 = (0,1), bounds2 = (0,1))
    domain_h = discretize(domain, ncells=ncells, periodic=[True,True])
    derham  = Derham(domain, ["H1", "Hdiv", "L2"])
    derham_h = discretize(derham, domain_h, degree=degree)

    # multi-patch (broken) spaces
    V1h = derham_h.V1

    v   = element_of(V1h.symbolic_space, name='v')
    u   = element_of(V1h.symbolic_space, name='u')

    expr = Dot(u,v)
    Al = LinearForm(u, integral(domain, expr))
    alh = discretize(Al, domain_h, V1h, backend=PSYDAC_BACKENDS[backend_language])

    #initial solution
    f1=Tuple(1,1)
    f2=Tuple(2,2)

    expr = Dot(f1, v)
    l = LinearForm(v, integral(domain, expr))
    lh = discretize(l, domain_h, V1h, backend=PSYDAC_BACKENDS[backend_language])
    uh1_c  = lh.assemble()

    expr = Dot(f2, v)
    l = LinearForm(v, integral(domain, expr))
    lh = discretize(l, domain_h, V1h, backend=PSYDAC_BACKENDS[backend_language])
    uh2_c  = lh.assemble()

    # compute approximate vector field u
    uh1 = FemField(V1h, uh1_c)
    uh2 = FemField(V1h, uh2_c)
    #Femfields associated, projections of the velocity on components and associated gradient           
    test1 = alh.assemble(v=uh1)
    test2 = alh.assemble(v=uh2)
    assert not (test1==test2)

if __name__ == '__main__':
    test_axis_projection()
    test_assembly()
