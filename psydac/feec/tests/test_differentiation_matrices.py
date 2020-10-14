import numpy as np
import pytest

from psydac.core.bsplines    import make_knots
from psydac.fem.basic        import FemField
from psydac.fem.tensor       import SplineSpace, TensorFemSpace
from psydac.fem.vector       import ProductFemSpace, VectorFemField
from psydac.feec.derivatives import Grad, Curl, Div

#==============================================================================
@pytest.mark.parametrize('domain', [(0, 1), (-2, 3)])
@pytest.mark.parametrize('ncells', [11, 37])
@pytest.mark.parametrize('degree', [2, 3, 4, 5])
@pytest.mark.parametrize('periodic', [True, False])

def test_Grad_1d(domain, ncells, degree, periodic):

    breaks = np.linspace(*domain, num=ncells+1)
    knots  = make_knots(breaks, degree, periodic)

    # H1 space (0-forms)
    N  = SplineSpace(degree=degree, knots=knots, periodic=periodic, basis='B')
    V0 = TensorFemSpace(N)

    # L2 space (1-forms)
    V1 = V0.reduce_degree(axes=[0], basis='M')

    # Linear operator: gradient
    grad = Grad(V0, V1)

    # Create random field in V0
    u0 = FemField(V0)

    s, = V0.vector_space.starts
    e, = V0.vector_space.ends

    u0.coeffs[s:e+1] = np.random.random(e-s+1)

    # Compute gradient (=derivative) of u0
    u1 = grad(u0)

    # Create evaluation grid, and check if ∂/∂x u0(x) == u1(x)
    xgrid = np.linspace(*N.domain, num=11)
    y_du0_dx = np.array([u0.gradient(x)[0] for x in xgrid])
    y_u1 = np.array([u1(x) for x in xgrid])

    assert np.allclose(y_du0_dx, y_u1, rtol=1e-14, atol=1e-14)

#==============================================================================
@pytest.mark.parametrize('domain', [([-2, 3], [6, 8])])              # 1 case
@pytest.mark.parametrize('ncells', [(10, 9), (27, 15)])              # 2 cases
@pytest.mark.parametrize('degree', [(3, 2), (4, 5)])                 # 2 cases
@pytest.mark.parametrize('periodic', [(True, False), (False, True)]) # 2 cases

def test_Grad_2d(domain, ncells, degree, periodic):

    # Compute breakpoints along each direction
    breaks = [np.linspace(*lims, num=n+1) for lims, n in zip(domain, ncells)]

    # H1 space (0-forms)
    Nx, Ny = [SplineSpace(degree=d, grid=g, periodic=p, basis='B') \
                                  for d, g, p in zip(degree, breaks, periodic)]
    V0 = TensorFemSpace(Nx, Ny)

    # H-curl space (1-forms)
    DxNy = V0.reduce_degree(axes=[0], basis='M')
    NxDy = V0.reduce_degree(axes=[1], basis='M')
    V1 = ProductFemSpace(DxNy, NxDy)

    # Linear operator: gradient
    grad = Grad(V0, V1)

    # Create random field in V0
    u0 = FemField(V0)

    s1, s2 = V0.vector_space.starts
    e1, e2 = V0.vector_space.ends

    u0.coeffs[s1:e1+1, s2:e2+1] = np.random.random((e1-s1+1, e2-s2+1))

    # Compute gradient (=derivative) of u0
    u1 = grad(u0)

    # x and y components of u1 vector field
    u1x = u1.fields[0]
    u1y = u1.fields[1]

    # Create evaluation grid, and check if
    # ∂/∂x u0(x, y) == u1x(x, y)
    # ∂/∂y u0(x, y) == u1y(x, y)

    xgrid = np.linspace(*domain[0], num=11)
    ygrid = np.linspace(*domain[1], num=11)

    vals_grad_u0 = np.array([[u0.gradient(x, y) for x in xgrid] for y in ygrid])
    vals_u1 = np.array([[[u1x(x, y), u1y(x, y)] for x in xgrid] for y in ygrid])

    assert np.allclose(vals_grad_u0, vals_u1, rtol=1e-14, atol=1e-14)

#==============================================================================
@pytest.mark.parametrize('domain', [([-2, 3], [6, 8], [-0.5, 0.5])])  # 1 case
@pytest.mark.parametrize('ncells', [(4, 5, 7)])                       # 1 case
@pytest.mark.parametrize('degree', [(3, 2, 5), (2, 4, 7)])            # 2 cases
@pytest.mark.parametrize('periodic', [( True, False, False),          # 3 cases
                                      (False,  True, False),
                                      (False, False,  True)])

def test_Grad_3d(domain, ncells, degree, periodic):

    # Compute breakpoints along each direction
    breaks = [np.linspace(*lims, num=n+1) for lims, n in zip(domain, ncells)]

    # H1 space (0-forms)
    Nx, Ny, Nz = [SplineSpace(degree=d, grid=g, periodic=p, basis='B') \
                                  for d, g, p in zip(degree, breaks, periodic)]
    V0 = TensorFemSpace(Nx, Ny, Nz)

    # H-curl space (1-forms)
    DxNyNz = V0.reduce_degree(axes=[0], basis='M')
    NxDyNz = V0.reduce_degree(axes=[1], basis='M')
    NxNyDz = V0.reduce_degree(axes=[2], basis='M')
    V1 = ProductFemSpace(DxNyNz, NxDyNz, NxNyDz)

    # Linear operator: gradient
    grad = Grad(V0, V1)

    # Create random field in V0
    u0 = FemField(V0)
    s1, s2, s3 = V0.vector_space.starts
    e1, e2, e3 = V0.vector_space.ends
    u0.coeffs[s1:e1+1, s2:e2+1, s3:e3+1] = np.random.random((e1-s1+1, e2-s2+1, e3-s3+1))

    # Compute gradient (=derivative) of u0
    u1 = grad(u0)

    # Components of u1 vector field
    u1x, u1y, u1z = u1.fields

    # Create random evaluation points (x, y, z) for evaluating fields
    npts = 1000
    xyz_pts = [[lims[0]+s*(lims[1]-lims[0]) for s, lims in zip(np.random.random(3), domain)]
        for i in range(npts)]

    # Check if
    #   ∂/∂x u0(x, y, z) == u1x(x, y, z)
    #   ∂/∂y u0(x, y, z) == u1y(x, y, z)
    #   ∂/∂z u0(x, y, z) == u1z(x, y, z)

    vals_grad_u0 = np.array([u0.gradient(*xyz) for xyz in xyz_pts])
    vals_u1 = np.array([[u1x(*xyz), u1y(*xyz), u1z(*xyz)] for xyz in xyz_pts])

    assert np.allclose(vals_grad_u0, vals_u1, rtol=1e-14, atol=1e-14)

#==============================================================================
@pytest.mark.parametrize('domain', [([-2, 3], [6, 8], [-0.5, 0.5])])  # 1 case
@pytest.mark.parametrize('ncells', [(4, 5, 7)])                       # 1 case
@pytest.mark.parametrize('degree', [(3, 2, 5), (2, 4, 7)])            # 2 cases
@pytest.mark.parametrize('periodic', [( True, False, False),          # 3 cases
                                      (False,  True, False),
                                      (False, False,  True)])

def test_Curl_3d(domain, ncells, degree, periodic):

    # Compute breakpoints along each direction
    breaks = [np.linspace(*lims, num=n+1) for lims, n in zip(domain, ncells)]

    # H1 space (0-forms)
    Nx, Ny, Nz = [SplineSpace(degree=d, grid=g, periodic=p, basis='B') \
                                  for d, g, p in zip(degree, breaks, periodic)]
    V0 = TensorFemSpace(Nx, Ny, Nz)

    # H-curl space (1-forms)
    DxNyNz = V0.reduce_degree(axes=[0], basis='M')
    NxDyNz = V0.reduce_degree(axes=[1], basis='M')
    NxNyDz = V0.reduce_degree(axes=[2], basis='M')
    V1 = ProductFemSpace(DxNyNz, NxDyNz, NxNyDz)

    # H-div space (2-forms)
    NxDyDz = V0.reduce_degree(axes=[1, 2], basis='M')
    DxNyDz = V0.reduce_degree(axes=[2, 0], basis='M')
    DxDyNz = V0.reduce_degree(axes=[0, 1], basis='M')
    V2 = ProductFemSpace(NxDyDz, DxNyDz, DxDyNz)

    # Linear operator: curl
    curl = Curl(V1, V2)

    # ...
    # Create random field in V1
    u1 = VectorFemField(V1)

    s1, s2, s3 = V1.vector_space[0].starts
    e1, e2, e3 = V1.vector_space[0].ends
    u1.coeffs[0][s1:e1+1, s2:e2+1, s3:e3+1] = np.random.random((e1-s1+1, e2-s2+1, e3-s3+1))

    s1, s2, s3 = V1.vector_space[1].starts
    e1, e2, e3 = V1.vector_space[1].ends
    u1.coeffs[1][s1:e1+1, s2:e2+1, s3:e3+1] = np.random.random((e1-s1+1, e2-s2+1, e3-s3+1))

    s1, s2, s3 = V1.vector_space[2].starts
    e1, e2, e3 = V1.vector_space[2].ends
    u1.coeffs[2][s1:e1+1, s2:e2+1, s3:e3+1] = np.random.random((e1-s1+1, e2-s2+1, e3-s3+1))
    # ...

    # Compute curl of u1
    u2 = curl(u1)

    # Components of vector fields u1 and u2
    u1x, u1y, u1z = u1.fields
    u2x, u2y, u2z = u2.fields

    # Create random evaluation points (x, y, z) for evaluating fields
    npts = 1000
    xyz_pts = [[lims[0]+s*(lims[1]-lims[0]) for s, lims in zip(np.random.random(3), domain)]
        for i in range(npts)]

    # Check if
    #   ∂/∂y u1z(x, y, z) - ∂/∂z u1y(x, y, z) == u2x(x, y, z)
    #   ∂/∂z u1x(x, y, z) - ∂/∂x u1z(x, y, z) == u2y(x, y, z)
    #   ∂/∂x u1y(x, y, z) - ∂/∂y u1x(x, y, z) == u2z(x, y, z)

    def eval_curl(fx, fy, fz, *eta):
        dfx_dx, dfx_dy, dfx_dz = fx.gradient(*eta)
        dfy_dx, dfy_dy, dfy_dz = fy.gradient(*eta)
        dfz_dx, dfz_dy, dfz_dz = fz.gradient(*eta)
        return [dfz_dy - dfy_dz,
                dfx_dz - dfz_dx,
                dfy_dx - dfx_dy]

    vals_curl_u1 = np.array([eval_curl(u1x, u1y, u1z, *xyz) for xyz in xyz_pts])
    vals_u2 = np.array([[u2x(*xyz), u2y(*xyz), u2z(*xyz)] for xyz in xyz_pts])

    assert np.allclose(vals_curl_u1, vals_u2, rtol=1e-14, atol=1e-14)

#==============================================================================
if __name__ == '__main__':

    test_Grad_1d(domain=[0, 1], ncells=12, degree=3, periodic=False)
    test_Grad_1d(domain=[0, 1], ncells=12, degree=3, periodic=True)

    test_Grad_2d(domain=([0, 1], [0, 1]), ncells=(10, 15), degree=(3, 2), periodic=(False, True))

    test_Grad_3d(
        domain   = ([0, 1], [0, 1], [0, 1]),
        ncells   = (5, 8, 4),
        degree   = (3, 2, 3),
        periodic = (False, True, True)
    )

    test_Curl_3d(
        domain   = ([0, 1], [0, 1], [0, 1]),
        ncells   = (5, 8, 4),
        degree   = (3, 2, 3),
        periodic = (False, True, True)
    )
