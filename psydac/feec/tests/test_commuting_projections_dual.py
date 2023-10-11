from psydac.feec.derivatives import Gradient_3D
from psydac.feec.derivatives import Curl_3D
from psydac.feec.derivatives import Divergence_3D

from sympde.expr                    import LinearForm, integral      
from sympde.topology                import Derham, element_of, Cube  
from psydac.api.discretization import discretize
from psydac.api.settings import PSYDAC_BACKENDS

import numpy as np
from sympy import sin, cos
import pytest


@pytest.mark.parametrize('Nel', [8, 12])
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('bc', [True, False])
def test_3d_commuting_pro_dual_1(Nel, p, bc):
    # Test transpose div

    fun1    = lambda xi1, xi2, xi3 : sin(xi1)*sin(xi2)*sin(xi3)
    D1fun1  = lambda xi1, xi2, xi3 : cos(xi1)*sin(xi2)*sin(xi3)
    D2fun1  = lambda xi1, xi2, xi3 : sin(xi1)*cos(xi2)*sin(xi3)
    D3fun1  = lambda xi1, xi2, xi3 : sin(xi1)*sin(xi2)*cos(xi3)

    Nel = [Nel]*3
    p   = [p]*3
    bc  = [bc]*3

    # Side lengths of logical cube [0, L]^3
    L = [2*np.pi, 2*np.pi , 2*np.pi]

    domain = Cube('domain', bounds1=(0, L[0]), bounds2=(0,L[1]), bounds3=(0,L[2]))
    derham = Derham(domain)
    domain_h = discretize(domain, ncells=Nel, periodic=bc)
    derham_h = discretize(derham, domain_h, degree=p)

    v2 = element_of(derham.V2, name='v2')
    v3 = element_of(derham.V3, name='v3')
    div = Divergence_3D(derham_h.V2, derham_h.V3)

    f2 = LinearForm(v2, integral(domain, D1fun1(*domain.coordinates)*v2[0] + D2fun1(*domain.coordinates)*v2[1] + D3fun1(*domain.coordinates)*v2[2]))
    f3 = LinearForm(v3, integral(domain, fun1(*domain.coordinates) * v3))

    u2 = discretize(f2, domain_h, derham_h.V2, backend=PSYDAC_BACKENDS['pyccel-gcc']).assemble()
    u3 = discretize(f3, domain_h, derham_h.V3, backend=PSYDAC_BACKENDS['pyccel-gcc']).assemble()

    divT_u3 = - div.matrix.T.dot(u3)

    error = abs((u2-divT_u3).toarray()).max()
    assert error < 9e-05

@pytest.mark.parametrize('Nel', [8, 12])
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('bc', [True, False])
def test_3d_commuting_pro_dual_2(Nel, p, bc):
    # Test transpose curl

    fun1    = lambda xi1, xi2, xi3 : sin(xi1)*sin(xi2)*sin(xi3)
    D1fun1  = lambda xi1, xi2, xi3 : cos(xi1)*sin(xi2)*sin(xi3)
    D2fun1  = lambda xi1, xi2, xi3 : sin(xi1)*cos(xi2)*sin(xi3)
    D3fun1  = lambda xi1, xi2, xi3 : sin(xi1)*sin(xi2)*cos(xi3)

    fun2    = lambda xi1, xi2, xi3 :   sin(2*xi1)*sin(2*xi2)*sin(2*xi3)
    D1fun2  = lambda xi1, xi2, xi3 : 2*cos(2*xi1)*sin(2*xi2)*sin(2*xi3)
    D2fun2  = lambda xi1, xi2, xi3 : 2*sin(2*xi1)*cos(2*xi2)*sin(2*xi3)
    D3fun2  = lambda xi1, xi2, xi3 : 2*sin(2*xi1)*sin(2*xi2)*cos(2*xi3)

    fun3    = lambda xi1, xi2, xi3 :   sin(3*xi1)*sin(3*xi2)*sin(3*xi3)
    D1fun3  = lambda xi1, xi2, xi3 : 3*cos(3*xi1)*sin(3*xi2)*sin(3*xi3)
    D2fun3  = lambda xi1, xi2, xi3 : 3*sin(3*xi1)*cos(3*xi2)*sin(3*xi3)
    D3fun3  = lambda xi1, xi2, xi3 : 3*sin(3*xi1)*sin(3*xi2)*cos(3*xi3)

    #curl
    cf1 = lambda xi1, xi2, xi3 : D2fun3(xi1, xi2, xi3) - D3fun2(xi1, xi2, xi3)
    cf2 = lambda xi1, xi2, xi3 : D3fun1(xi1, xi2, xi3) - D1fun3(xi1, xi2, xi3)
    cf3 = lambda xi1, xi2, xi3 : D1fun2(xi1, xi2, xi3) - D2fun1(xi1, xi2, xi3)

    Nel = [Nel]*3
    p   = [p]*3
    bc  = [bc]*3

    # Side lengths of logical cube [0, L]^3
    L = [2*np.pi, 2*np.pi , 2*np.pi]

    domain = Cube('domain', bounds1=(0, L[0]), bounds2=(0,L[1]), bounds3=(0,L[2]))
    derham = Derham(domain)
    domain_h = discretize(domain, ncells=Nel, periodic=bc)
    derham_h = discretize(derham, domain_h, degree=p)

    v1 = element_of(derham.V1, name='v1')
    v2 = element_of(derham.V2, name='v2')
    curl = Curl_3D(derham_h.V1, derham_h.V2)

    f1 = LinearForm(v1, integral(domain, cf1(*domain.coordinates)*v1[0] + cf2(*domain.coordinates)*v1[1] + cf3(*domain.coordinates)*v1[2]))
    f2 = LinearForm(v2, integral(domain, fun1(*domain.coordinates)*v2[0] + fun2(*domain.coordinates)*v2[1] + fun3(*domain.coordinates)*v2[2]))

    u1 = discretize(f1, domain_h, derham_h.V1, backend=PSYDAC_BACKENDS['pyccel-gcc']).assemble()
    u2 = discretize(f2, domain_h, derham_h.V2, backend=PSYDAC_BACKENDS['pyccel-gcc']).assemble()

    curlT_u2 = curl.matrix.T.dot(u2)

    error = abs((u1-curlT_u2).toarray()).max()
    assert error < 9e-3


@pytest.mark.parametrize('Nel', [8, 12])
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('bc', [True, False])
def test_3d_commuting_pro_dual_3(Nel, p, bc):
    # Test transpose grad
        
    fun1    = lambda xi1, xi2, xi3 : sin(xi1)*sin(xi2)*sin(xi3)
    D1fun1  = lambda xi1, xi2, xi3 : cos(xi1)*sin(xi2)*sin(xi3)

    fun2    = lambda xi1, xi2, xi3 :   sin(2*xi1)*sin(2*xi2)*sin(2*xi3)
    D2fun2  = lambda xi1, xi2, xi3 : 2*sin(2*xi1)*cos(2*xi2)*sin(2*xi3)

    fun3    = lambda xi1, xi2, xi3 :   sin(3*xi1)*sin(3*xi2)*sin(3*xi3)
    D3fun3  = lambda xi1, xi2, xi3 : 3*sin(3*xi1)*sin(3*xi2)*cos(3*xi3)

    Nel = [Nel]*3
    p   = [p]*3
    bc  = [bc]*3

    # Side lengths of logical cube [0, L]^3
    L = [2*np.pi, 2*np.pi , 2*np.pi]

    domain = Cube('domain', bounds1=(0, L[0]), bounds2=(0,L[1]), bounds3=(0,L[2]))
    derham = Derham(domain)
    domain_h = discretize(domain, ncells=Nel, periodic=bc)
    derham_h = discretize(derham, domain_h, degree=p)

    v0 = element_of(derham.V0, name='v0')
    v1 = element_of(derham.V1, name='v1')
    grad = Gradient_3D(derham_h.V0, derham_h.V1)    

    f0 = LinearForm(v0, integral(domain, (D1fun1(*domain.coordinates) + D2fun2(*domain.coordinates) + D3fun3(*domain.coordinates))*v0))
    f1 = LinearForm(v1, integral(domain, fun1(*domain.coordinates)*v1[0] + fun2(*domain.coordinates)*v1[1] + fun3(*domain.coordinates)*v1[2]))

    u0 = discretize(f0, domain_h, derham_h.V0, backend=PSYDAC_BACKENDS['pyccel-gcc']).assemble()
    u1 = discretize(f1, domain_h, derham_h.V1, backend=PSYDAC_BACKENDS['pyccel-gcc']).assemble()

    gradT_u1 = -grad.matrix.T.dot(u1)

    error = abs((u0-gradT_u1).toarray()).max()
    assert error < 4e-3 


