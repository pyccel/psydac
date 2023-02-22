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

    curl, div = derham_h.derivatives_as_matrices

    I1_b = IdentityOperator(V1h.vector_space)

    a   = element_of(V1h.symbolic_space, name='a')
    b   = element_of(V1h.symbolic_space, name='b')

    expr = Dot(a,b)

    A = BilinearForm((a,b), integral(domain, expr))
    Ah = discretize(A, domain_h, (V1h,V1h), backend=PSYDAC_BACKENDS['python'])

    dH1_b = Ah.assemble()
    H1_b  = inverse(dH1_b, 'cg')

    a   = element_of(V2h.symbolic_space, name='a')
    b   = element_of(V2h.symbolic_space, name='b')

    expr = a*b

    A = BilinearForm((a,b), integral(domain, expr))
    Ah = discretize(A, domain_h, (V2h,V2h), backend=PSYDAC_BACKENDS['python'])

    div2 = div @I1_b

    dH2_b = Ah.assemble()

    grad = -H1_b@div.T@dH2_b

    graddiv = grad@div2

    uh = V1h.vector_space.zeros()

    test = graddiv.dot(uh)

if __name__ == '__main__':
    test_axis_projection()
    test_grad_div()