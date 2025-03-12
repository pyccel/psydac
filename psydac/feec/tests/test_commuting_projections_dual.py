from psydac.feec.derivatives        import Gradient_3D
from psydac.feec.derivatives        import Curl_3D
from psydac.feec.derivatives        import Divergence_3D
from sympde.expr                    import LinearForm, integral      
from sympde.topology                import Derham, element_of, Cube  
from psydac.api.discretization      import discretize
from psydac.api.settings            import PSYDAC_BACKENDS
from sympy                          import sin, cos
import numpy as np
import pytest


@pytest.mark.parametrize('Nel', [8, 12])
@pytest.mark.parametrize('Nq', [5])
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('bc', [True, False])
@pytest.mark.parametrize('m', [1,2])
def test_weak_gradient_3d(Nel, Nq, p, bc, m):
    """Test weak gradient on a 3D domain, where the dual sequence is homogeneous.

    This test checks that for $f \in V_3^* = H_0^1$ it holds:
    $$ \int_{\Omega} \nabla f \cdot \boldsymbol{v}_2 dV = \int_{\Omega} f \widetilde{\mathrm{grad}_h} \boldsymbol{v}_2 dV $$
    for all $\boldsymbol{v}_2 \in V_2^* = H_0^{\mathrm{curl}}$. In particular, $f = 0$ on $\partial \Omega$.
    In such case the weak gradient corresponds to the operator $-\mathbb{D^T}$ where $\mathbb{D}$ is the divergence matrix.
    
    Parameters
    ----------
    Nel : int
        Number of cells in each direction.

    Nq : int
        Number of quadrature points in each direction.

    p : int
        B-Spline degree in each direction.

    bc : bool
        If True, periodic boundary conditions are applied in the domain boundary.

    m : int
        Knot multiplicity in each direction.

    """

    fun1    = lambda xi1, xi2, xi3 : sin(xi1)*sin(xi2)*sin(xi3)
    D1fun1  = lambda xi1, xi2, xi3 : cos(xi1)*sin(xi2)*sin(xi3)
    D2fun1  = lambda xi1, xi2, xi3 : sin(xi1)*cos(xi2)*sin(xi3)
    D3fun1  = lambda xi1, xi2, xi3 : sin(xi1)*sin(xi2)*cos(xi3)

    Nel = [Nel]*3
    Nq  = [Nq]*3
    p   = [p]*3
    bc  = [bc]*3
    m   = [m]*3

    # Side lengths of logical cube [0, L]^3
    L = [2*np.pi, 2*np.pi , 2*np.pi]

    domain = Cube('domain', bounds1=(0, L[0]), bounds2=(0,L[1]), bounds3=(0,L[2]))
    derham = Derham(domain)
    domain_h = discretize(domain, ncells=Nel, periodic=bc)
    derham_h = discretize(derham, domain_h, degree=p, multiplicity=m)

    v2 = element_of(derham.V2, name='v2')
    v3 = element_of(derham.V3, name='v3')
    div = Divergence_3D(derham_h.V2, derham_h.V3)

    f2 = LinearForm(v2, integral(domain, D1fun1(*domain.coordinates)*v2[0] + D2fun1(*domain.coordinates)*v2[1] + D3fun1(*domain.coordinates)*v2[2]))
    f3 = LinearForm(v3, integral(domain, fun1(*domain.coordinates) * v3))

    u2 = discretize(f2, domain_h, derham_h.V2, nquads=Nq, backend=PSYDAC_BACKENDS['pyccel-gcc']).assemble()
    u3 = discretize(f3, domain_h, derham_h.V3, nquads=Nq, backend=PSYDAC_BACKENDS['pyccel-gcc']).assemble()

    divT_u3 = - div.matrix.T.dot(u3)

    error = abs((u2-divT_u3).toarray()).max()
    assert error < 2e-10


@pytest.mark.parametrize('Nel', [8, 12])
@pytest.mark.parametrize('Nq', [6])
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('bc', [True, False])
@pytest.mark.parametrize('m', [1,2])
def test_weak_curl_3d(Nel, Nq, p, bc, m):
    """Test weak curl on a 3D domain, where the dual sequence is homogeneous. 
    This test checks that for $\boldsymbol{f} \in V_2^* = H_0^{\mathrm{curl}}$ it holds:
    $$ \int_{\Omega} (\nabla \times \boldsymbol{f}) \cdot \boldsymbol{v}_1 dV = \int_{\Omega} \boldsymbol{f} \cdot \widetilde{\mathrm{curl}_h} \boldsymbol{v}_1 dV $$
    for all $\boldsymbol{v}_1 \in V_1^* = H_0^{\mathrm{div}}$. In particular,  $\boldsymbol{n} \times \boldsymbol{f} = 0$ on $\partial \Omega$ where $\boldsymbol{n}$ is the outward unit normal vector. 
    # In such case the weak curl corresponds to the operator $\mathbb{C^T}$ where $\mathbb{C}$ is the curl matrix.
    
    Parameters
    ----------
    Nel : int
        Number of cells in each direction.

    Nq : int
        Number of quadrature points in each direction.

    p : int
        B-Spline degree in each direction.

    bc : bool
        If True, periodic boundary conditions are applied in the domain boundary.

    m : int
        Knot multiplicity in each direction.

    """

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
    Nq  = [Nq]*3
    p   = [p]*3
    bc  = [bc]*3
    m   = [m]*3

    # Side lengths of logical cube [0, L]^3
    L = [2*np.pi, 2*np.pi , 2*np.pi]

    domain = Cube('domain', bounds1=(0, L[0]), bounds2=(0,L[1]), bounds3=(0,L[2]))
    derham = Derham(domain)
    domain_h = discretize(domain, ncells=Nel, periodic=bc)
    derham_h = discretize(derham, domain_h, degree=p, multiplicity=m)

    v1 = element_of(derham.V1, name='v1')
    v2 = element_of(derham.V2, name='v2')
    curl = Curl_3D(derham_h.V1, derham_h.V2)

    f1 = LinearForm(v1, integral(domain, cf1(*domain.coordinates)*v1[0] + cf2(*domain.coordinates)*v1[1] + cf3(*domain.coordinates)*v1[2]))
    f2 = LinearForm(v2, integral(domain, fun1(*domain.coordinates)*v2[0] + fun2(*domain.coordinates)*v2[1] + fun3(*domain.coordinates)*v2[2]))

    u1 = discretize(f1, domain_h, derham_h.V1, nquads=Nq, backend=PSYDAC_BACKENDS['pyccel-gcc']).assemble()
    u2 = discretize(f2, domain_h, derham_h.V2, nquads=Nq, backend=PSYDAC_BACKENDS['pyccel-gcc']).assemble()

    curlT_u2 = curl.matrix.T.dot(u2)

    error = abs((u1-curlT_u2).toarray()).max()
    assert error < 2e-9


@pytest.mark.parametrize('Nel', [8, 12])
@pytest.mark.parametrize('Nq', [6])
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('bc', [True, False])
@pytest.mark.parametrize('m', [1,2])
def test_weak_divergence_3d(Nel, Nq, p, bc, m):
    """Test weak divergence on a 3D domain, where the dual sequence is homogeneous.
    This test checks that for $\boldsymbol{f} \in V_1^* = H_0^{\mathrm{div}}$ it holds:
    $$ \int_{\Omega} (\nabla \cdot \boldsymbol{f}) v_0 dV = \int_{\Omega} \boldsymbol{f} \cdot \widetilde{\mathrm{div}_h} v_0 dV $$
    for all $v_0 \in V_0^* = H_0^1$. In particular,  $\boldsymbol{n} \cdot \boldsymbol{f} = 0$ on $\partial \Omega$ where $\boldsymbol{n}$ is the outward unit normal vector. 
    In such case the weak divergence corresponds to the operator $-\mathbb{G^T}$ where $\mathbb{G}$ is the gradient matrix.
    
    Parameters
    ----------
    Nel : int
        Number of cells in each direction.

    Nq : int
        Number of quadrature points in each direction.

    p : int
        B-Spline degree in each direction.

    bc : bool
        If True, periodic boundary conditions are applied in the domain boundary.

    m : int
        Knot multiplicity in each direction.

    """

    fun1    = lambda xi1, xi2, xi3 : sin(xi1)*sin(xi2)*sin(xi3)
    D1fun1  = lambda xi1, xi2, xi3 : cos(xi1)*sin(xi2)*sin(xi3)

    fun2    = lambda xi1, xi2, xi3 :   sin(2*xi1)*sin(2*xi2)*sin(2*xi3)
    D2fun2  = lambda xi1, xi2, xi3 : 2*sin(2*xi1)*cos(2*xi2)*sin(2*xi3)

    fun3    = lambda xi1, xi2, xi3 :   sin(3*xi1)*sin(3*xi2)*sin(3*xi3)
    D3fun3  = lambda xi1, xi2, xi3 : 3*sin(3*xi1)*sin(3*xi2)*cos(3*xi3)

    Nel = [Nel]*3
    Nq  = [Nq]*3
    p   = [p]*3
    bc  = [bc]*3
    m   = [m]*3

    # Side lengths of logical cube [0, L]^3
    L = [2*np.pi, 2*np.pi , 2*np.pi]

    domain = Cube('domain', bounds1=(0, L[0]), bounds2=(0,L[1]), bounds3=(0,L[2]))
    derham = Derham(domain)
    domain_h = discretize(domain, ncells=Nel, periodic=bc)
    derham_h = discretize(derham, domain_h, degree=p, multiplicity=m)

    v0 = element_of(derham.V0, name='v0')
    v1 = element_of(derham.V1, name='v1')
    grad = Gradient_3D(derham_h.V0, derham_h.V1)    

    f0 = LinearForm(v0, integral(domain, (D1fun1(*domain.coordinates) + D2fun2(*domain.coordinates) + D3fun3(*domain.coordinates))*v0))
    f1 = LinearForm(v1, integral(domain, fun1(*domain.coordinates)*v1[0] + fun2(*domain.coordinates)*v1[1] + fun3(*domain.coordinates)*v1[2]))

    u0 = discretize(f0, domain_h, derham_h.V0, nquads=Nq, backend=PSYDAC_BACKENDS['pyccel-gcc']).assemble()
    u1 = discretize(f1, domain_h, derham_h.V1, nquads=Nq, backend=PSYDAC_BACKENDS['pyccel-gcc']).assemble()

    gradT_u1 = -grad.matrix.T.dot(u1)

    error = abs((u0-gradT_u1).toarray()).max()
    assert error < 5e-10


#==============================================================================
if __name__ == '__main__':

    Nel = 8
    Nq  = 8
    p   = 2
    bc  = True
    m   = 2

    test_weak_gradient_3d (Nel, Nq, p, bc, m)
    test_weak_curl_3d(Nel, Nq, p, bc, m)
    test_weak_divergence_3d(Nel, Nq, p, bc, m)
