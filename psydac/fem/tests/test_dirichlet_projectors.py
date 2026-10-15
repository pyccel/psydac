#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import  numpy as np
import  pytest
from    mpi4py import MPI
from    sympy import sin, pi, sqrt, Tuple

from    sympde.calculus               import inner, cross
from    sympde.expr                   import integral, LinearForm, BilinearForm
from    sympde.topology               import elements_of, Derham, Mapping, Line, Square, Cube, Union, NormalVector, ScalarFunctionSpace, VectorFunctionSpace
from    sympde.topology.datatype      import H1Space, HcurlSpace

from    psydac.api.discretization     import discretize
from    psydac.api.settings           import PSYDAC_BACKEND_GPYCCEL
from    psydac.fem.projectors         import DirichletProjector
from    psydac.linalg.basic           import LinearOperator, IdentityOperator
from    psydac.linalg.block           import BlockVectorSpace
from    psydac.linalg.solvers         import inverse
from    psydac.linalg.tests.utilities import check_linop_equality_using_rng, SinMapping1D, Annulus, SquareTorus

#===============================================================================
@pytest.mark.parametrize('dim', [1, 2])

def test_function_space_boundary_projector(dim):

    tol = 1e-15

    ncells_3d   = [8, 8, 8]
    degree_3d   = [2, 2, 2]
    periodic_3d = [False, True, False]

    comm     = None
    backend  = PSYDAC_BACKEND_GPYCCEL

    logical_domain_1d = Line  ('L', bounds= (0,   1))
    logical_domain_2d = Square('S', bounds1=(0.5, 1), bounds2=(0, 2*np.pi))
    logical_domain_3d = Cube  ('C', bounds1=(0.5, 1), bounds2=(0, 2*np.pi), bounds3=(0, 1))
    logical_domains   = [logical_domain_1d, logical_domain_2d, logical_domain_3d]

    mapping_1d = SinMapping1D('LM')
    mapping_2d = Annulus     ('A' )
    mapping_3d = SquareTorus ('ST')
    mappings   = [mapping_1d, mapping_2d, mapping_3d]

    rng  = np.random.default_rng(42)

    print()
    print(f' ----- Test projectors in dimension {dim} -----')
    print()

    domain        = mappings[dim-1](logical_domains[dim-1])
    from sympde.utilities.utils import plot_domain
    #plot_domain(domain, draw=True, isolines=True)

    # Obtain "true" boundary, i.e., remove periodic y-direction boundary
    if dim == 1:
        boundary  = domain.boundary
    elif dim == 2:
        boundary  = Union(domain.get_boundary(axis=0, ext=-1), domain.get_boundary(axis=0, ext=1))
    else:
        boundary  = Union(domain.get_boundary(axis=0, ext=-1), domain.get_boundary(axis=0, ext=1),
                            domain.get_boundary(axis=2, ext=-1), domain.get_boundary(axis=2, ext=1))
        
    ncells    = [ncells_3d[0], ]   if dim == 1 else ncells_3d  [0:dim]
    degree    = [degree_3d[0], ]   if dim == 1 else degree_3d  [0:dim]
    periodic  = [periodic_3d[0], ] if dim == 1 else periodic_3d[0:dim]

    domain_h = discretize(domain, ncells=ncells, periodic=periodic, comm=comm)

    nn            = NormalVector('nn')

    for i in range(dim):
        print(f'      - Test DP{i}')

        # The function defined here satisfy the corresponding homogeneous Dirichlet BCs
        if dim == 1:
            x = domain.coordinates
            V = ScalarFunctionSpace('V', domain, kind='H1') # testing various kind arguments
            f = sin(2*pi*x)
        if dim == 2:
            x, y = domain.coordinates
            if i == 0:
                V  = ScalarFunctionSpace('V', domain, kind=H1Space) # testing various kind arguments
                f  = (sqrt(x**2 + y**2)-0.5) * (sqrt(x**2 + y**2)-1)
            else:
                V  = VectorFunctionSpace('V', domain, kind='hCuRl') # testing various kind arguments
                f1 = x
                f2 = y
                f  = Tuple(f1, f2)
        if dim == 3:
            x, y, z = domain.coordinates
            if i == 0:
                V  = ScalarFunctionSpace('V', domain, kind='h1') # testing various kind arguments
                f  = (sqrt(x**2 + y**2)-0.5) * (sqrt(x**2 + y**2)-1) * z * (z-1)
            elif i == 1:
                V  = VectorFunctionSpace('V', domain, kind=HcurlSpace) # testing various kind arguments
                f1 = z * (z - 1) * x
                f2 = z * (z - 1) * y
                f3 = (sqrt(x**2 + y**2)-0.5) * (sqrt(x**2 + y**2)-1)
                f  = Tuple(f1, f2, f3)
            else:
                V  = VectorFunctionSpace('V', domain, kind='Hdiv') # testing various kind arguments
                f1 = (sqrt(x**2 + y**2)-0.5) * (sqrt(x**2 + y**2)-1)
                f2 = (sqrt(x**2 + y**2)-0.5) * (sqrt(x**2 + y**2)-1)
                f3 = z * (z-1) * sin(x*y)
                f  = Tuple(f1, f2, f3)

        u, v = elements_of(V, names='u, v')
        if i == 0:
            boundary_expr = u*v
        if (i == 1) and (dim == 2):
            boundary_expr = cross(nn, u) * cross(nn, v)
        if (i == 1) and (dim == 3):
            boundary_expr = inner(cross(nn, u), cross(nn, v))
        if i == 2:
            boundary_expr = inner(nn, u) * inner(nn, v)

        Vh   = discretize(V, domain_h, degree=degree)
        expr = inner(u, v) if isinstance(Vh.coeff_space, BlockVectorSpace) else u*v

        a   = BilinearForm((u, v), integral(domain,            expr))            
        ab  = BilinearForm((u, v), integral(boundary, boundary_expr))

        ah  = discretize(a,  domain_h, (Vh, Vh), backend=backend)
        abh = discretize(ab, domain_h, (Vh, Vh), backend=backend, sum_factorization=False)

        I   = IdentityOperator(Vh.coeff_space)
        DP = DirichletProjector(Vh)

        M   = ah.assemble()
        M_0 = DP @ M @ DP + (I - DP)
        Mb  = abh.assemble()

        # We project f into the conforming discrete space using a penalization method. It's coefficients are stored in fc
        lexpr = inner(v, f) if isinstance(Vh.coeff_space, BlockVectorSpace) else v*f
        l = LinearForm(v, integral(domain, lexpr))
        lh = discretize(l, domain_h, Vh, backend=backend)
        rhs = lh.assemble()
        A = M + 1e30*Mb
        A_inv = inverse(A, 'cg', maxiter=1000, tol=1e-10)
        fc = A_inv @ rhs

        # 1.
        # In 1D, 2D, 3D, the coefficients of functions satisfying homogeneous Dirichlet 
        # boundary conditions should not change under application of the corresponding projector
        fc2     = DP @ fc
        diff    = fc - fc2
        err_sqr = diff.inner(diff)
        print(f' || f - P @ f ||^2      = {err_sqr}')
        assert err_sqr < tol**2

        # 2.1
        # After applying a projector to a random vector, we want to verify that the 
        # corresponding boundary integral vanishes
        rdm_coeffs = Vh.coeff_space.zeros()
        print(' Random boundary integrals:')
        for _ in range(3):
            if isinstance(rdm_coeffs.space, BlockVectorSpace):
                for block in rdm_coeffs.blocks:
                    rng.random(size=block._data.shape, dtype="float64", out=block._data)
            else:
                rng.random(size=rdm_coeffs._data.shape, dtype="float64", out=rdm_coeffs._data)
            rdm_coeffs2 = DP @ rdm_coeffs
            scaled_boundary_int_rdm_sqr      = Mb.dot_inner(rdm_coeffs, rdm_coeffs) / rdm_coeffs.space.dimension**2
            scaled_boundary_int_proj_rdm_sqr = Mb.dot_inner(rdm_coeffs2, rdm_coeffs2) / rdm_coeffs.space.dimension**2
            print(f'  rdm: {scaled_boundary_int_rdm_sqr}    proj. rdm: {scaled_boundary_int_proj_rdm_sqr}')
            assert scaled_boundary_int_proj_rdm_sqr < tol**2

        # 2.2
        # Test toarray(): (DP @ rdm_coeffs).toarray() should be equal to DP.toarray().dot(rdm_coeffs.toarray())
        DP_arr          = DP.toarray()
        rdm_coeffs_arr  = rdm_coeffs.toarray()
        diff_arr        = DP_arr.dot(rdm_coeffs_arr) - rdm_coeffs2.toarray()
        err_sqr         = diff_arr.dot(diff_arr)
        assert err_sqr < tol**2

        # 3.
        # We want to verify that applying a projector twice does not change the vector twice
        fc3         = DP @ fc2
        diff        = fc2 - fc3
        err_sqr     = diff.inner(diff)
        print(f' || P @ f - P @ P @ f ||^2 = {err_sqr}')
        assert err_sqr < tol**2

        # 4.
        # Finally, the modified mass matrix should still compute inner products correctly
        l2_norm_sqr     = M.dot_inner  (fc, fc)
        l2_norm2_sqr    = M_0.dot_inner(fc, fc)
        err_sqr         = abs(l2_norm_sqr - l2_norm2_sqr)
        print(f' || P @ f ||^2          = {l2_norm_sqr} should be equal to')
        print(f' || P @ f ||^2  (alt)   = {l2_norm2_sqr}')
        # M.dot_inner(fc, fc) and M_0.dot_inner(fc, fc) are the same only up to order 1e-15.
        # Hence, we can't expect err_sqr to be less than tol**2, but only less than tol.
        assert err_sqr < tol

        print()

#===============================================================================
@pytest.mark.parametrize('dim', [1, 3])
@pytest.mark.mpi

def test_discrete_derham_boundary_projector(dim):

    tol = 1e-15

    ncells   = [8, 8, 8]
    degree   = [2, 2, 2]
    periodic = [False, True, False]

    comm     = MPI.COMM_WORLD
    backend  = PSYDAC_BACKEND_GPYCCEL

    logical_domain_1d = Line  ('L', bounds= (0,   1))
    logical_domain_2d = Square('S', bounds1=(0.5, 1), bounds2=(0, 2*np.pi))
    logical_domain_3d = Cube  ('C', bounds1=(0.5, 1), bounds2=(0, 2*np.pi), bounds3=(0, 1))
    logical_domains   = [logical_domain_1d, logical_domain_2d, logical_domain_3d]

    mapping_1d = SinMapping1D('LM')
    mapping_2d = Annulus     ('A' )
    mapping_3d = SquareTorus ('ST')
    mappings   = [mapping_1d, mapping_2d, mapping_3d]

    rng  = np.random.default_rng(42)

    # The following are functions (1D, 2D & 3D) satisfying homogeneous Dirichlet BCs

    f11     = lambda x : np.sin(2*np.pi*x)

    r2      = lambda x, y : np.sqrt(x**2 + y**2)
    f21     = lambda x, y : (r2(x, y) - 0.5) * (r2(x, y) - 1)
    f22_1   = lambda x, y : x
    f22_2   = lambda x, y : y
    f22     = (f22_1, f22_2)

    f31     = lambda x, y, z : (r2(x, y) - 0.5) * (r2(x, y) - 1) * z * (z - 1)
    f32_1   = lambda x, y, z : z * (z - 1) * x
    f32_2   = lambda x, y, z : z * (z - 1) * y
    f32_3   = lambda x, y, z : (r2(x, y) - 0.5) * (r2(x, y) - 1)
    f32     = (f32_1, f32_2, f32_3)
    f33_1   = lambda x, y, z : (r2(x, y) - 0.5) * (r2(x, y) - 1)
    f33_2   = lambda x, y, z : (r2(x, y) - 0.5) * (r2(x, y) - 1)
    f33_3   = lambda x, y, z : z * (z - 1) * np.sin(x*y)
    f33     = (f33_1, f33_2, f33_3)

    funs    = [[f11], [f21, f22], [f31, f32, f33]]

    print()
    print(f' ----- Test projectors in dimension {dim} -----')
    print()

    domain        = mappings[dim-1](logical_domains[dim-1])
    from sympde.utilities.utils import plot_domain
    #plot_domain(domain, draw=True, isolines=True)

    # Obtain "true" boundary, i.e., remove periodic y-direction boundary
    if dim == 1:
        boundary  = domain.boundary
    elif dim == 2:
        boundary  = Union(domain.get_boundary(axis=0, ext=-1), domain.get_boundary(axis=0, ext=1))
    else:
        boundary  = Union(domain.get_boundary(axis=0, ext=-1), domain.get_boundary(axis=0, ext=1),
                            domain.get_boundary(axis=2, ext=-1), domain.get_boundary(axis=2, ext=1))

    derham        = Derham(domain) if dim in (1, 3) else Derham(domain, sequence=['h1', 'hcurl', 'l2'])

    ncells_dim    = [ncells[0], ] if dim == 1 else ncells[0:dim]
    degree_dim    = [degree[0], ] if dim == 1 else degree[0:dim]
    periodic_dim  = [periodic[0], ] if dim == 1 else periodic[0:dim]

    domain_h      = discretize(domain, ncells=ncells_dim, periodic=periodic_dim, comm=comm)
    derham_h      = discretize(derham, domain_h, degree=degree_dim)

    d_projectors = derham_h.dirichlet_projectors(kind='linop')

    if dim == 2: 
        conf_projectors = derham_h.conforming_projectors(kind='linop', hom_bc=True)

    nn            = NormalVector('nn')

    for i in range(dim):
        print(f'      - Test DP{i}')

        u, v = elements_of(derham.spaces[i], names='u, v')

        if i == 0:
            boundary_expr = u*v
        if (i == 1) and (dim == 2):
            boundary_expr = cross(nn, u) * cross(nn, v)
        if (i == 1) and (dim == 3):
            boundary_expr = inner(cross(nn, u), cross(nn, v))
        if i == 2:
            boundary_expr = inner(nn, u) * inner(nn, v)

        expr = inner(u, v) if isinstance(derham_h.spaces[i].coeff_space, BlockVectorSpace) else u*v

        a   = BilinearForm((u, v), integral(domain,            expr))            
        ab  = BilinearForm((u, v), integral(boundary, boundary_expr))

        ah  = discretize(a,  domain_h, (derham_h.spaces[i], derham_h.spaces[i]), backend=backend)
        abh = discretize(ab, domain_h, (derham_h.spaces[i], derham_h.spaces[i]), backend=backend, sum_factorization=False)

        I   = IdentityOperator(derham_h.spaces[i].coeff_space)
        DP = d_projectors[i]

        if dim == 2: 
            CP = conf_projectors[i]
            check_linop_equality_using_rng(DP, CP)

        M   = ah.assemble()
        M_0 = DP @ M @ DP + (I - DP)
        Mb  = abh.assemble()

        f   = funs[dim-1][i]
        fc  = derham_h.projectors()[i](f).coeffs

        # 1.
        # In 1D, 2D, 3D, the coefficients of functions satisfying homogeneous Dirichlet 
        # boundary conditions should not change under application of the corresponding projector
        fc2     = DP @ fc
        diff    = fc - fc2
        err_sqr = diff.inner(diff)
        print(f' || f - P @ f ||^2      = {err_sqr}')
        assert err_sqr < tol**2

        # 2.1
        # After applying a projector to a random vector, we want to verify that the 
        # corresponding boundary integral vanishes
        rdm_coeffs = derham_h.spaces[i].coeff_space.zeros()
        print(' Random boundary integrals:')
        for _ in range(3):
            if isinstance(rdm_coeffs.space, BlockVectorSpace):
                for block in rdm_coeffs.blocks:
                    rng.random(size=block._data.shape, dtype="float64", out=block._data)
            else:
                rng.random(size=rdm_coeffs._data.shape, dtype="float64", out=rdm_coeffs._data)
            rdm_coeffs2 = DP @ rdm_coeffs
            scaled_boundary_int_rdm_sqr      = Mb.dot_inner(rdm_coeffs, rdm_coeffs) / rdm_coeffs.space.dimension**2
            scaled_boundary_int_proj_rdm_sqr = Mb.dot_inner(rdm_coeffs2, rdm_coeffs2) / rdm_coeffs.space.dimension**2
            print(f'  rdm: {scaled_boundary_int_rdm_sqr}    proj. rdm: {scaled_boundary_int_proj_rdm_sqr}')
            assert scaled_boundary_int_proj_rdm_sqr < tol**2

        # 2.2
        # Test tosparse(): (DP @ rdm_coeffs).toarray() should be equal to DP.tosparse().dot(rdm_coeffs.toarray())
        DP_spr          = DP.tosparse()
        rdm_coeffs_arr  = rdm_coeffs.toarray()
        diff_arr        = DP_spr.dot(rdm_coeffs_arr) - rdm_coeffs2.toarray()
        err_sqr         = diff_arr.dot(diff_arr)
        assert err_sqr < tol**2


        # 3.
        # We want to verify that applying a projector twice does not change the vector twice
        fc3     = DP @ fc2
        diff    = fc2 - fc3
        err_sqr = diff.inner(diff)
        print(f' || P @ f - P @ P @ f ||^2 = {err_sqr}')
        assert err_sqr < tol**2

        # 4.
        # Finally, the modified mass matrix should still compute inner products correctly
        l2_norm_sqr     = M.dot_inner  (fc, fc)
        l2_norm2_sqr    = M_0.dot_inner(fc, fc)
        err_sqr         = abs(l2_norm_sqr - l2_norm2_sqr)
        print(f' || P @ f ||^2          = {l2_norm_sqr} should be equal to')
        print(f' || P @ f ||^2  (alt)   = {l2_norm2_sqr}')
        # M.dot_inner(fc, fc) and M_0.dot_inner(fc, fc) are the same only up to order 1e-15.
        # Hence, we can't expect err_sqr to be less than tol**2, but only less than tol.
        assert err_sqr < tol

        print()

#===============================================================================
def test_discrete_derham_boundary_projector_multipatch():

    tol = 1e-15

    ncells   = [8, 8]
    degree   = [2, 2]

    comm     = None
    backend  = PSYDAC_BACKEND_GPYCCEL

    from psydac.feec.multipatch_domain_utilities import build_multipatch_domain
    domain = build_multipatch_domain(domain_name='annulus_3')

    rng = np.random.default_rng(42)

    # The following are functions satisfying homogeneous Dirichlet BCs
    r      = lambda x, y : np.sqrt(x**2 + y**2)
    f1     = lambda x, y : (r(x, y) - 0.5) * (r(x, y) - 1)
    f2_1   = lambda x, y : x
    f2_2   = lambda x, y : y
    f2     = (f2_1, f2_2)
    funs   = [f1, f2]
    print()

    boundary = domain.boundary

    derham = Derham(domain, sequence=['h1', 'hcurl', 'l2'])
    
    ncells_h = {}
    for D in domain.interior:
        ncells_h[D.name] = ncells

    domain_h = discretize(domain, ncells=ncells_h, comm=comm)
    derham_h = discretize(derham, domain_h, degree=degree)

    projectors = derham_h.projectors(nquads=[(d + 1) for d in degree])

    d_projectors = derham_h.dirichlet_projectors(kind='linop')

    nn = NormalVector('nn')

    for i in range(2):
        print(f'      - Test DP{i}')

        u, v = elements_of(derham.spaces[i], names='u, v')

        if i == 0:
            boundary_expr = u*v
            expr = u*v
        if (i == 1):
            boundary_expr = cross(nn, u) * cross(nn, v)
            expr = inner(u,v)

        a   = BilinearForm((u, v), integral(domain,            expr))            
        ab  = BilinearForm((u, v), integral(boundary, boundary_expr))

        ah  = discretize(a,  domain_h, (derham_h.spaces[i], derham_h.spaces[i]), backend=backend)
        abh = discretize(ab, domain_h, (derham_h.spaces[i], derham_h.spaces[i]), backend=backend, sum_factorization=False)

        I   = IdentityOperator(derham_h.spaces[i].coeff_space)
        DP = d_projectors[i]

        M   = ah.assemble()
        M_0 = DP @ M @ DP + (I - DP)
        Mb  = abh.assemble()

        f   = funs[i]
        fc  = projectors[i](f).coeffs

        # 1.
        # The coefficients of functions satisfying homogeneous Dirichlet 
        # boundary conditions should not change under application of the corresponding projector
        fc2     = DP @ fc
        diff    = fc - fc2
        err_sqr = diff.inner(diff)
        print(f' || f - P @ f ||^2      = {err_sqr}')
        assert err_sqr < tol**2

        # 2.1
        # After applying a projector to a random vector, we want to verify that the 
        # corresponding boundary integral vanishes
        rdm_coeffs = derham_h.spaces[i].coeff_space.zeros()
        print(' Random boundary integrals:')
        for _ in range(3):
            for patch in rdm_coeffs.blocks:

                if isinstance(patch.space, BlockVectorSpace):
                    for block in patch.blocks:
                        rng.random(size=block._data.shape, dtype="float64", out=block._data)
                else:
                    rng.random(size=patch._data.shape, dtype="float64", out=patch._data)

            rdm_coeffs2 = DP @ rdm_coeffs
            scaled_boundary_int_rdm_sqr      = Mb.dot_inner(rdm_coeffs, rdm_coeffs) / rdm_coeffs.space.dimension**2
            scaled_boundary_int_proj_rdm_sqr = Mb.dot_inner(rdm_coeffs2, rdm_coeffs2) / rdm_coeffs.space.dimension**2
            print(f'  rdm: {scaled_boundary_int_rdm_sqr}    proj. rdm: {scaled_boundary_int_proj_rdm_sqr}')
            assert scaled_boundary_int_proj_rdm_sqr < tol**2

        # 2.2
        # Test toarray(): (DP @ rdm_coeffs).toarray() should be equal to DP.toarray().dot(rdm_coeffs.toarray())
        DP_arr          = DP.toarray()
        rdm_coeffs_arr  = rdm_coeffs.toarray()
        diff_arr        = DP_arr.dot(rdm_coeffs_arr) - rdm_coeffs2.toarray()
        err_sqr         = diff_arr.dot(diff_arr)
        assert err_sqr < tol**2

        # 3.
        # We want to verify that applying a projector twice does not change the vector twice
        fc3     = DP @ fc2
        diff    = fc2 - fc3
        err_sqr = diff.inner(diff)
        print(f' || P @ f - P @ P @ f ||^2 = {err_sqr}')
        assert err_sqr < tol**2

        # 4.
        # Finally, the modified mass matrix should still compute inner products correctly
        l2_norm_sqr     = M.dot_inner  (fc, fc)
        l2_norm2_sqr    = M_0.dot_inner(fc, fc)
        err_sqr         = abs(l2_norm_sqr - l2_norm2_sqr)
        print(f' || P @ f ||^2          = {l2_norm_sqr} should be equal to')
        print(f' || P @ f ||^2  (alt)   = {l2_norm2_sqr}')
        # M.dot_inner(fc, fc) and M_0.dot_inner(fc, fc) are the same only up to order 1e-15.
        # Hence, we can't expect err_sqr to be less than tol**2, but only less than tol.
        assert err_sqr < tol

        print()


# ===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================

if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
