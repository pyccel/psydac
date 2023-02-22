from sympde.topology     import Square, Derham, element_of
from sympde.expr.expr    import BilinearForm, integral
from psydac.feec.multipatch.api import discretize
from psydac.api.settings        import PSYDAC_BACKENDS
from sympde.calculus            import Dot
from psydac.linalg.solvers      import inverse
from psydac.linalg.basic import IdentityOperator



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

def test_grad_div():
    domain=Square('OmegaLog', bounds1=(0,1), bounds2 = (0,1))
    derham  = Derham(domain, ["H1", "Hdiv", "L2"])
    domain_h = discretize(domain, ncells=[4,4], periodic=[True,True])
    derham_h = discretize(derham, domain_h, degree=(2,2))

    V1h = derham_h.V1
    V2h = derham_h.V2

    _, div = derham_h.derivatives_as_matrices

    I1_b = IdentityOperator(V1h.vector_space)
    I2_b = IdentityOperator(V2h.vector_space)

    Op1 = div @I1_b

    Op2 = 2*I2_b

    Op = Op2@Op1

    uh1 = V1h.vector_space.zeros()
    uh2 = V2h.vector_space.zeros()

    test = Op.dot(uh1, out=uh2)

if __name__ == '__main__':
    test_axis_projection()
    test_grad_div()