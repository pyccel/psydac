#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from sympde.topology            import Square, Derham, element_of
from sympde.expr.expr           import BilinearForm, integral
from psydac.api.discretization  import discretize
from psydac.api.settings        import PSYDAC_BACKENDS

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
    uh = V1h.coeff_space.zeros()
    test = Peih.dot(uh)

if __name__ == '__main__':
    test_axis_projection()
